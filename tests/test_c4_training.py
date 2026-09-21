import math

from c4.run_llama_pretraining import (
    build_update_metrics,
    collect_communication_metrics,
    elapsed_seconds,
    evaluation_loss_totals_after_batch,
    has_reached_training_limit,
    loss_to_perplexity,
    mean_update_loss,
    should_sync_gradients,
    should_stop_evaluation,
)


def test_training_limit_stops_before_the_next_update():
    assert not has_reached_training_limit(update_step=0, num_training_steps=1)
    assert has_reached_training_limit(update_step=1, num_training_steps=1)


def test_gradient_sync_happens_only_on_the_last_microbatch_of_each_update():
    assert [should_sync_gradients(step, 4) for step in range(1, 9)] == [
        False, False, False, True, False, False, False, True
    ]


def test_update_loss_is_the_arithmetic_mean_of_microbatch_losses():
    assert mean_update_loss(2.0 + 4.0 + 8.0, 3) == 14.0 / 3.0


def test_evaluation_stops_when_effective_token_target_is_reached():
    assert not should_stop_evaluation(9_999_999, 10_000_000)
    assert should_stop_evaluation(10_000_000, 10_000_000)


def test_evaluation_loss_totals_weight_batch_mean_by_prediction_tokens():
    loss_numerator, evaluated_tokens = evaluation_loss_totals_after_batch(
        0.0, 0, 2.0, 3
    )
    loss_numerator, evaluated_tokens = evaluation_loss_totals_after_batch(
        loss_numerator, evaluated_tokens, 4.0, 1
    )

    assert loss_numerator == 10.0
    assert evaluated_tokens == 4
    assert loss_numerator / evaluated_tokens == 2.5


def test_elapsed_seconds_uses_end_minus_start():
    assert elapsed_seconds(10.0, 12.5) == 2.5


def test_loss_to_perplexity_is_exponential():
    assert loss_to_perplexity(2.0) == math.exp(2.0)


def test_update_metrics_use_explicit_timing_and_memory_names():
    metrics = build_update_metrics(
        loss=2.0,
        lr=0.001,
        update_step=3,
        tokens_seen=1_024,
        tokens_in_update=512,
        step_time_s=2.0,
        total_batch_size=512,
        batches_in_update=4,
        peak_memory_allocated_mb=123.5,
        peak_memory_reserved_mb=256.0,
    )

    assert metrics["step_time_s"] == 2.0
    assert metrics["throughput_tokens_per_s"] == 256.0
    assert metrics["peak_memory_allocated_mb"] == 123.5
    assert metrics["peak_memory_reserved_mb"] == 256.0


class _HookState:
    comm_bits_this_round = 80

    def compression_bits_stats(self):
        return 2.0, 100, 50


class _MuonOptimizer:
    def communication_bits_stats(self):
        return {
            "this_step": {
                "gradient_presence": 4,
                "orthogonalization_results": 6,
            },
            "total": {
                "gradient_presence": 40,
                "orthogonalization_results": 60,
            },
        }


def test_communication_metrics_include_ddp_and_muon_step_and_total_bits():
    metrics, next_ddp_total = collect_communication_metrics(
        _HookState(), _MuonOptimizer(), ddp_comm_bits_total=120
    )

    assert next_ddp_total == 200
    assert metrics["ddp_comm_bits_step"] == 80
    assert metrics["ddp_comm_bits_total"] == 200
    assert metrics["ddp_compression_bits_before_total"] == 100
    assert metrics["ddp_compression_bits_after_total"] == 50
    assert metrics["muon_gradient_presence_bits_step"] == 4
    assert metrics["muon_gradient_presence_bits_total"] == 40
    assert metrics["muon_orthogonalization_results_bits_step"] == 6
    assert metrics["muon_orthogonalization_results_bits_total"] == 60
    assert metrics["muon_comm_bits_step"] == 10
    assert metrics["muon_comm_bits_total"] == 100
