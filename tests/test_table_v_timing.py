import math

import pytest

from c4.scripts.run_table_v_muon import cells, result_is_accepted
from c4.table_v_timing import accumulate_hook_communication_bits, summarize_timing


def test_table_v_matrix_matches_paper_models_and_four_muon_arms():
    matrix = list(cells())

    assert len(matrix) == 16
    assert [cell["model"] for cell in matrix] == [
        model
        for model in ("60m", "130m", "350m", "1b")
        for _ in range(4)
    ]
    assert [cell["arm"] for cell in matrix[:4]] == [
        "dense",
        "topk",
        "randk",
        "arctopk",
    ]
    assert [cell["run_id"].split("-", 1)[0] for cell in matrix] == [
        f"CM{number:03d}" for number in range(20, 36)
    ]
    assert all("-rerun1-" in cell["run_id"] for cell in matrix)
    assert all("-fp32-" in cell["run_id"] for cell in matrix)


def test_table_v_commands_use_short_warmup_and_paper_timing_batch():
    for cell in cells():
        command = cell["command"]
        assert command[command.index("--batch_size") + 1] == "1"
        assert command[command.index("--warmup_iterations") + 1] == "100"
        assert command[command.index("--measured_iterations") + 1] == "50"
        assert command[command.index("--max_length") + 1] == "256"
        assert command[command.index("--dtype") + 1] == "float32"
        assert command[command.index("--optimizer") + 1] == "muon"
        assert "--muon_compile" not in command


def test_compressed_commands_enable_normal_training_compression_immediately_after_warmup():
    for cell in cells():
        command = cell["command"]
        compressor = command[command.index("--compressor") + 1]
        expected = {
            "dense": "none",
            "topk": "topk_sync",
            "randk": "randk_sync",
            "arctopk": "group_topk_no_reshape",
        }[cell["arm"]]
        assert compressor == expected
        assert command[command.index("--start_compress_iter") + 1] == (
            "100" if cell["arm"] != "dense" else "0"
        )
        assert command[command.index("--compress_ratio") + 1] == "0.2"
        assert "--disable_compression_warmup" in command
        assert "--compress_only_2d" not in command


@pytest.mark.parametrize(
    ("exit_code", "result", "accepted"),
    [
        (0, {"status": "completed", "measured_steps": 50}, True),
        (0, {"status": "completed", "measured_steps": 49}, False),
        (0, {"status": "running", "measured_steps": 50}, False),
        (1, {"status": "completed", "measured_steps": 50}, False),
        (0, None, False),
    ],
)
def test_launcher_accepts_only_complete_successful_cells(exit_code, result, accepted):
    assert result_is_accepted(exit_code, result) is accepted


def test_summarize_timing_uses_slowest_rank_continuous_window():
    result = summarize_timing(
        rank_elapsed=[5.0, 5.5],
        rank_steps=[[0.1] * 50, [0.11] * 50],
        expected_steps=50,
    )

    assert result["measured_steps"] == 50
    assert result["slowest_rank"] == 1
    assert result["mean_iteration_seconds"] == pytest.approx(0.11)


def test_hook_communication_bits_are_accumulated_and_round_counter_is_reset():
    class State:
        comm_bits_this_round = 123

    state = State()

    total = accumulate_hook_communication_bits(state, 400)

    assert total == 523
    assert state.comm_bits_this_round == 0


@pytest.mark.parametrize(
    ("rank_elapsed", "rank_steps"),
    [
        ([5.0], [[0.1] * 49]),
        ([math.nan], [[0.1] * 50]),
        ([5.0], [[0.1] * 49 + [math.inf]]),
    ],
)
def test_summarize_timing_rejects_invalid_windows(rank_elapsed, rank_steps):
    with pytest.raises(ValueError):
        summarize_timing(rank_elapsed, rank_steps, expected_steps=50)
