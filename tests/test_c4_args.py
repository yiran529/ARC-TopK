from types import SimpleNamespace

from c4.pept_utils.args_utils import check_args_torchrun_main


def test_max_train_tokens_converts_to_update_steps_using_sequence_length():
    args = SimpleNamespace(
        save_dir=None,
        model_config="c4/configs/llama_130m.json",
        tags=None,
        total_batch_size=256,
        batch_size=32,
        gradient_accumulation=2,
        max_train_tokens=1_000_000,
        max_length=256,
        num_training_steps=10_000,
        continue_from=None,
        dtype="float32",
    )

    result = check_args_torchrun_main(args)

    assert result.num_training_steps == 16
