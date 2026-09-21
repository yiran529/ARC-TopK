import copy
import argparse
import os
import tempfile
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from optimizers.muon import Muon, polar_express
from optimizers.utils import add_muon_args, build_muon_optimizer


def _identity_orthogonalization(update, epsilon):
    return update


def test_muon_uses_nesterov_momentum_and_decoupled_weight_decay():
    weight = nn.Parameter(torch.ones(2, 2))
    optimizer = Muon(
        [weight], lr=0.1, mu=0.5, weight_decay=0.1,
        nesterov=True, adjust_lr=None,
        orthogonalize=_identity_orthogonalization,
    )

    weight.grad = torch.full_like(weight, 0.5)
    optimizer.step()
    assert torch.allclose(weight, torch.full_like(weight, 0.915))
    assert torch.allclose(optimizer.state[weight]["momentum"], torch.full_like(weight, 0.5))

    weight.grad = torch.full_like(weight, 0.25)
    optimizer.step()
    assert torch.allclose(weight, torch.full_like(weight, 0.85585))
    assert torch.allclose(optimizer.state[weight]["momentum"], torch.full_like(weight, 0.5))


def test_muon_spectral_scaling_and_convolution_flattening():
    weight = nn.Parameter(torch.ones(2, 1, 2, 2))
    seen = []

    def ortho(update, epsilon):
        seen.append(tuple(update.shape))
        return torch.ones_like(update)

    optimizer = Muon([weight], lr=0.1, weight_decay=0, orthogonalize=ortho)
    weight.grad = torch.ones_like(weight)
    optimizer.step()

    assert seen == [(1, 2, 4)]
    assert torch.allclose(weight, torch.full_like(weight, 1 - 0.1 * (2 / 4) ** 0.5))


def test_muon_batches_same_shape_matrices():
    left = nn.Parameter(torch.ones(2, 2))
    right = nn.Parameter(torch.ones(2, 2))
    calls = []

    def ortho(update, epsilon):
        calls.append(tuple(update.shape))
        return torch.ones_like(update)

    optimizer = Muon([left, right], lr=0.1, weight_decay=0, orthogonalize=ortho)
    left.grad = torch.ones_like(left)
    right.grad = torch.ones_like(right)
    optimizer.step()

    assert calls == [(2, 2, 2)]
    assert torch.allclose(left, torch.full_like(left, 0.9))
    assert torch.allclose(right, torch.full_like(right, 0.9))


def test_adamw_fallback_updates_scalar_parameters():
    bias = nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = Muon(
        [{"params": [bias], "algorithm": "adamw", "lr": 0.1, "weight_decay": 0.1}],
        betas=(0.9, 0.999), epsilon=1e-8,
    )
    bias.grad = torch.tensor([0.5, -0.25])
    optimizer.step()

    assert torch.allclose(bias, torch.tensor([0.89, 2.08]), atol=1e-6)
    assert optimizer.state[bias]["step"] == 1


def test_state_dict_resume_matches_uninterrupted_steps():
    weight = nn.Parameter(torch.ones(2, 2))
    optimizer = Muon([weight], lr=0.1, mu=0.5, orthogonalize=_identity_orthogonalization)
    weight.grad = torch.full_like(weight, 0.5)
    optimizer.step()

    restored_weight = nn.Parameter(weight.detach().clone())
    restored = Muon([restored_weight], lr=0.1, mu=0.5, orthogonalize=_identity_orthogonalization)
    restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))

    weight.grad = torch.full_like(weight, 0.25)
    restored_weight.grad = torch.full_like(restored_weight, 0.25)
    optimizer.step()
    restored.step()
    assert torch.equal(weight, restored_weight)
    assert torch.equal(optimizer.state[weight]["momentum"], restored.state[restored_weight]["momentum"])


def test_bfloat16_adamw_checkpoint_restores_fp32_moments():
    bias = nn.Parameter(torch.ones(100, dtype=torch.bfloat16))
    optimizer = Muon([{"params": [bias], "algorithm": "adamw"}], lr=0.01)
    bias.grad = torch.linspace(-1, 1, 100, dtype=torch.bfloat16)
    optimizer.step()
    expected_exp_avg = optimizer.state[bias]["exp_avg"].clone()
    expected_exp_avg_sq = optimizer.state[bias]["exp_avg_sq"].clone()

    restored_bias = nn.Parameter(bias.detach().clone())
    restored = Muon([{"params": [restored_bias], "algorithm": "adamw"}], lr=0.01)
    restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))

    assert restored.state[restored_bias]["exp_avg"].dtype == torch.float32
    assert restored.state[restored_bias]["exp_avg_sq"].dtype == torch.float32
    assert torch.equal(restored.state[restored_bias]["exp_avg"], expected_exp_avg)
    assert torch.equal(restored.state[restored_bias]["exp_avg_sq"], expected_exp_avg_sq)
    next_grad = torch.linspace(1, -0.5, 100, dtype=torch.bfloat16)
    bias.grad = next_grad.clone()
    restored_bias.grad = next_grad.clone()
    optimizer.step()
    restored.step()
    assert torch.equal(bias, restored_bias)


def test_polar_express_preserves_identity_direction():
    result = polar_express(torch.eye(2))
    assert result.shape == (2, 2)
    assert torch.isfinite(result).all()
    assert 0.5 < result[0, 0] < 1.5
    assert torch.allclose(result, torch.eye(2, dtype=result.dtype) * result[0, 0])


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(5, 3)
        self.proj_a = nn.Linear(3, 3)
        self.proj_b = nn.Linear(3, 3)
        self.conv = nn.Conv2d(1, 2, 2)
        self.norm = nn.BatchNorm2d(2)
        self.classifier = nn.Linear(3, 2)
        self.lm_head = nn.Linear(3, 5, bias=False)
        self.lm_head.weight = self.embed_tokens.weight


class SharedOutputHeadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 3, bias=False)
        self.lm_head = nn.Linear(3, 3, bias=False)
        self.lm_head.weight = self.proj.weight


def test_builder_selects_matrices_and_assigns_each_parameter_once():
    model = TinyModel()
    optimizer = build_muon_optimizer(model, lr=0.1, scalar_lr=0.01)
    matrix_ids = {
        id(p) for group in optimizer.param_groups if group["algorithm"] == "muon"
        for p in group["params"]
    }
    assert matrix_ids == {
        id(model.proj_a.weight), id(model.proj_b.weight), id(model.conv.weight)
    }
    all_params = [p for group in optimizer.param_groups for p in group["params"]]
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(all_params) == len(trainable)
    assert {id(p) for p in all_params} == {id(p) for p in trainable}
    assert all(group["lr"] == 0.01 for group in optimizer.param_groups if group["algorithm"] == "adamw")


def test_builder_rejects_model_without_muon_eligible_weights():
    model = nn.Sequential(nn.Embedding(5, 3), nn.LayerNorm(3))
    with pytest.raises(ValueError, match="Muon-eligible"):
        build_muon_optimizer(model, lr=0.1)


def test_output_head_exclusion_wins_for_shared_parameter():
    model = SharedOutputHeadModel()
    with pytest.raises(ValueError, match="Muon-eligible"):
        build_muon_optimizer(model, lr=0.1)


def test_muon_cli_flags_keep_optional_scalar_settings():
    parser = argparse.ArgumentParser()
    add_muon_args(parser)
    defaults = parser.parse_args([])
    assert defaults.muon_mu == 0.95
    assert defaults.muon_scalar_lr is None
    assert defaults.muon_scalar_beta2 == 0.999
    assert defaults.muon_epsilon == 1e-8
    assert defaults.muon_adjust_lr == "spectral_norm"
    assert not defaults.muon_local_orthogonalization
    selected = parser.parse_args([
        "--muon_mu", "0.8", "--muon_scalar_lr", "0.01",
        "--muon_scalar_beta2", "0.999",
        "--muon_epsilon", "1e-6", "--muon_scalar_eps", "1e-9",
        "--muon_adjust_lr", "none", "--muon_compile",
    ])
    assert selected.muon_mu == 0.8
    assert selected.muon_scalar_lr == 0.01
    assert selected.muon_scalar_beta2 == 0.999
    assert selected.muon_epsilon == 1e-6
    assert selected.muon_scalar_eps == 1e-9
    assert selected.muon_adjust_lr == "none"
    assert selected.muon_compile


def test_builder_param_groups_accept_scheduler_changes():
    model = TinyModel()
    optimizer = build_muon_optimizer(model, lr=0.1, scalar_lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 0.5)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
    assert all(group["lr"] == pytest.approx(0.005) for group in optimizer.param_groups[1:])


def _distributed_muon_worker(rank, init_file):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        params = [nn.Parameter(torch.ones(2, 2)) for _ in range(3)]

        def rank_tagged_ortho(updates, epsilon):
            # Rank zero owns the first two matrices. Rank one owns the third
            # plus padding. Each rank must receive both ranks' outputs.
            assert updates.shape == (2, 2, 2)
            return torch.full_like(updates, rank + 1)

        optimizer = Muon(
            params, lr=0.1, mu=0.0, weight_decay=0.0,
            adjust_lr=None, orthogonalize=rank_tagged_ortho,
            distributed_orthogonalization=True,
        )
        for p in params:
            p.grad = torch.ones_like(p)
        optimizer.step()
        for p, expected in zip(params, (0.9, 0.9, 0.8)):
            assert torch.allclose(p, torch.full_like(p, expected))
        stats = optimizer.communication_bits_stats()
        assert stats["this_step"] == {
            "gradient_presence": 48,
            "orthogonalization_results": 512,
        }
        assert stats["total"] == stats["this_step"]
    finally:
        dist.destroy_process_group()


def test_distributed_muon_assigns_matrices_to_ranks_and_gathers_updates():
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_distributed_muon_worker, args=(os.path.join(directory, "init"),), nprocs=2)


def _mismatched_gradient_worker(rank, init_file):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2,
        timeout=timedelta(seconds=5),
    )
    try:
        params = [nn.Parameter(torch.ones(2, 2)) for _ in range(2)]
        optimizer = Muon(
            params, lr=0.1, orthogonalize=_identity_orthogonalization,
            distributed_orthogonalization=True,
        )
        params[0].grad = torch.ones_like(params[0])
        if rank == 0:
            params[1].grad = torch.ones_like(params[1])
        with pytest.raises(RuntimeError, match="gradient presence differs across ranks"):
            optimizer.step()
    finally:
        dist.destroy_process_group()


def test_distributed_muon_rejects_mismatched_gradient_layout_before_collectives():
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_mismatched_gradient_worker, args=(os.path.join(directory, "init"),), nprocs=2)
