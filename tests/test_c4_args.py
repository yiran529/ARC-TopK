from types import SimpleNamespace

from c4.pept_utils.args_utils import check_args_torchrun_main


def test_max_train_tokens_converts_to_update_steps_using_sequence_length():
    args = SimpleNamespace(
        save_dir=None,
        model_config="c4/configs/llama_130m.json",
        tags=None,
        save_every=0,
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


def test_checkpoint_directory_is_not_created_when_checkpointing_is_disabled():
    args = SimpleNamespace(
        save_dir=None,
        model_config="c4/configs/llama_130m.json",
        tags=None,
        save_every=0,
        total_batch_size=256,
        batch_size=32,
        gradient_accumulation=2,
        max_train_tokens=None,
        max_length=256,
        num_training_steps=10_000,
        continue_from=None,
        dtype="float32",
    )

    result = check_args_torchrun_main(args)

    assert result.save_dir is None


def test_checkpoint_directory_is_generated_when_periodic_saving_is_enabled():
    args = SimpleNamespace(
        save_dir=None,
        model_config="c4/configs/llama_130m.json",
        tags=None,
        save_every=100,
        total_batch_size=256,
        batch_size=32,
        gradient_accumulation=2,
        max_train_tokens=None,
        max_length=256,
        num_training_steps=10_000,
        continue_from=None,
        dtype="float32",
    )

    result = check_args_torchrun_main(args)

    assert result.save_dir.startswith("checkpoints/llama_130m-")
