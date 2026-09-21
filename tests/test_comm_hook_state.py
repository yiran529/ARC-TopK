import torch

from comm_hooks.group_topk_hook_no_reshape import GroupTopKState


def test_comm_hook_state_roundtrip_preserves_error_feedback_and_rng():
    state = GroupTopKState(
        process_group=None,
        r=2,
        compress_ratio=0.25,
        start_compress_iter=3,
        use_error_feedback="ef21",
        seed=17,
    )
    state.iter = 9
    state.total_bit_before_compression = 1000
    state.total_bit_after_compression = 250
    state.comm_bits_this_round = 64
    state.error_dict[0] = torch.tensor([1.25, -2.5])
    state.global_error_dict[0] = torch.tensor([0.5, 3.0])

    checkpoint = state.state_dict()
    expected_next_random = torch.rand(4, generator=state.rng)

    restored = GroupTopKState(
        process_group=None,
        r=2,
        compress_ratio=0.25,
        start_compress_iter=3,
        use_error_feedback="ef21",
        seed=999,
    )
    restored.load_state_dict(checkpoint, device=torch.device("cpu"))

    assert restored.iter == 9
    assert restored.total_bit_before_compression == 1000
    assert restored.total_bit_after_compression == 250
    assert restored.comm_bits_this_round == 64
    assert torch.equal(restored.error_dict[0], state.error_dict[0])
    assert torch.equal(restored.global_error_dict[0], state.global_error_dict[0])
    assert torch.equal(torch.rand(4, generator=restored.rng), expected_next_random)
