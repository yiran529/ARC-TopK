import gzip
import json
from itertools import islice
from pathlib import Path

import datasets
import datasets.distributed
import pytest
import torch

from c4.pept_utils.c4_data import load_c4_split


def _write_shard(path: Path, *texts: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for index, text in enumerate(texts):
            stream.write(
                json.dumps(
                    {
                        "text": text,
                        "timestamp": f"2026-09-21T00:00:0{index}Z",
                        "url": f"https://example.test/{text}",
                    }
                )
                + "\n"
            )


def test_local_train_shards_are_sorted_and_read_without_network(tmp_path):
    _write_shard(
        tmp_path / "en" / "c4-train.00001-of-01024.json.gz",
        "train-one",
    )
    _write_shard(
        tmp_path / "en" / "c4-train.00000-of-01024.json.gz",
        "train-zero-a",
        "train-zero-b",
    )

    dataset = load_c4_split(str(tmp_path), "train")

    assert [row["text"] for row in dataset] == [
        "train-zero-a",
        "train-zero-b",
        "train-one",
    ]


def test_local_validation_uses_only_validation_shards(tmp_path):
    _write_shard(
        tmp_path / "en" / "c4-train.00000-of-01024.json.gz",
        "train",
    )
    _write_shard(
        tmp_path / "en" / "c4-validation.00000-of-00008.json.gz",
        "validation",
    )

    dataset = load_c4_split(str(tmp_path), "validation")

    assert [row["text"] for row in dataset] == ["validation"]


def test_missing_local_split_names_expected_glob(tmp_path):
    with pytest.raises(
        FileNotFoundError,
        match=r"c4-train\.\[0-9\].*-of-01024\.json\.gz",
    ):
        load_c4_split(str(tmp_path), "train")


def test_repeat_local_cycles_deterministically(tmp_path):
    _write_shard(
        tmp_path / "en" / "c4-train.00000-of-01024.json.gz",
        "first",
        "second",
    )

    dataset = load_c4_split(str(tmp_path), "train", repeat_local=True)

    assert [row["text"] for row in dataset.take(5)] == [
        "first",
        "second",
        "first",
        "second",
        "first",
    ]


def test_repeat_shuffle_then_rank_split_is_worker_compatible(tmp_path):
    for index in range(4):
        _write_shard(
            tmp_path / "en" / f"c4-train.{index:05d}-of-01024.json.gz",
            f"shard-{index}",
        )

    dataset = load_c4_split(str(tmp_path), "train", repeat_local=True).shuffle(
        seed=42,
        buffer_size=2,
    )
    rank_dataset = datasets.distributed.split_dataset_by_node(
        dataset,
        rank=1,
        world_size=2,
    )

    rows = list(rank_dataset.take(4))

    assert len(rows) == 4
    assert all(row["text"].startswith("shard-") for row in rows)


def test_repeat_dataset_can_be_iterated_by_dataloader_workers(tmp_path):
    for index in range(4):
        _write_shard(
            tmp_path / "en" / f"c4-train.{index:05d}-of-01024.json.gz",
            f"worker-{index}",
        )

    dataset = load_c4_split(str(tmp_path), "train", repeat_local=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=2)

    rows = list(islice(loader, 8))

    assert len(rows) == 8
    assert all(row["text"].startswith("worker-") for row in rows)


def test_manifest_limits_loader_to_listed_files(tmp_path):
    listed = []
    for index in range(10):
        filename = f"en/c4-train.{index:05d}-of-01024.json.gz"
        path = tmp_path / filename
        _write_shard(path, f"listed-{index}")
        listed.append(filename)

    _write_shard(
        tmp_path / "en" / "c4-train.00010-of-01024.json.gz",
        "old-extra-shard",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "repository": "allenai/c4",
                "revision": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
                "config": "en",
                "train_files": listed,
                "validation_files": [],
                "file_sizes": {
                    filename: (tmp_path / filename).stat().st_size for filename in listed
                },
                "complete": True,
            }
        ),
        encoding="utf-8",
    )

    dataset = load_c4_split(str(tmp_path), "train")

    assert [row["text"] for row in dataset] == [f"listed-{index}" for index in range(10)]


def test_manifest_must_be_complete(tmp_path):
    filename = "en/c4-train.00000-of-01024.json.gz"
    _write_shard(tmp_path / filename, "text")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "train_files": [filename],
                "file_sizes": {filename: (tmp_path / filename).stat().st_size},
                "complete": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="complete"):
        load_c4_split(str(tmp_path), "train")


@pytest.mark.parametrize(
    ("manifest_filename", "expected_error"),
    [
        ("en/c4-train.00001-of-01024.json.gz", FileNotFoundError),
        ("en/c4-train.00000-of-01024.json.gz", ValueError),
    ],
)
def test_manifest_validates_file_existence_and_recorded_size(
    tmp_path,
    manifest_filename,
    expected_error,
):
    existing_filename = "en/c4-train.00000-of-01024.json.gz"
    _write_shard(tmp_path / existing_filename, "text")
    recorded_size = (tmp_path / existing_filename).stat().st_size
    if expected_error is ValueError:
        recorded_size += 1
    else:
        existing_filename = manifest_filename
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "train_files": [existing_filename],
                "file_sizes": {existing_filename: recorded_size},
                "complete": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(expected_error):
        load_c4_split(str(tmp_path), "train")


def test_local_glob_requires_five_digits_and_valid_index_range(tmp_path):
    _write_shard(
        tmp_path / "en" / "c4-train.0000-of-01024.json.gz",
        "short-index",
    )
    _write_shard(
        tmp_path / "en" / "c4-train.01024-of-01024.json.gz",
        "out-of-range",
    )
    _write_shard(
        tmp_path / "en" / "c4-train.00002-of-01024.json.gz",
        "valid",
    )

    dataset = load_c4_split(str(tmp_path), "train")

    assert [row["text"] for row in dataset] == ["valid"]
