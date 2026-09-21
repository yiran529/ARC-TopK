import json
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from c4.scripts.download_c4_raw_shards import (
    DEFAULT_REVISION,
    InsufficientDiskSpaceError,
    download_c4_shards,
    select_shard_filenames,
    shard_filename,
)


def _metadata_for(filenames, size=4):
    return {filename: size for filename in filenames}


def _metadata_with_sha(filenames, contents):
    return {
        filename: {
            "size": len(contents[filename]),
            "sha256": hashlib.sha256(contents[filename]).hexdigest(),
        }
        for filename in filenames
    }


def test_shard_filename_generation_and_bounds():
    assert shard_filename("train", 0) == "en/c4-train.00000-of-01024.json.gz"
    assert shard_filename("validation", 7) == "en/c4-validation.00007-of-00008.json.gz"

    with pytest.raises(ValueError):
        shard_filename("train", -1)
    with pytest.raises(ValueError):
        shard_filename("train", 1024)
    with pytest.raises(ValueError):
        shard_filename("validation", 8)

    train, validation = select_shard_filenames(1, 0)
    assert train == ["en/c4-train.00000-of-01024.json.gz"]
    assert validation == []

    with pytest.raises(ValueError):
        select_shard_filenames(0, 0)
    with pytest.raises(ValueError):
        select_shard_filenames(1025, 0)
    with pytest.raises(ValueError):
        select_shard_filenames(1, 9)


def test_disk_space_is_checked_before_download(tmp_path):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)

    def metadata(revision, filenames):
        return _metadata_for(filenames, size=100)

    with pytest.raises(InsufficientDiskSpaceError, match="free space"):
        download_c4_shards(
            tmp_path,
            num_train_shards=1,
            num_validation_shards=0,
            minimum_free_gib=1,
            metadata_fn=metadata,
            download_fn=download,
            disk_usage_fn=lambda path: SimpleNamespace(free=100),
        )

    assert calls == []


def test_expected_size_file_is_reused_without_downloading(tmp_path):
    train_filename = "en/c4-train.00000-of-01024.json.gz"
    destination = tmp_path / train_filename
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"done")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "revision": DEFAULT_REVISION,
                "train_files": [train_filename],
                "validation_files": [],
                "file_sizes": {train_filename: 4},
                "complete": True,
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        raise AssertionError("complete shard should be reused")

    manifest = download_c4_shards(
        tmp_path,
        num_train_shards=1,
        num_validation_shards=0,
        minimum_free_gib=0,
        metadata_fn=lambda revision, filenames: _metadata_for(filenames, size=4),
        download_fn=download,
        disk_usage_fn=lambda path: SimpleNamespace(free=0),
    )

    assert calls == []
    assert manifest["complete"] is True
    assert manifest["file_sizes"] == {train_filename: 4}
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest


def test_equal_size_file_without_manifest_is_not_trusted(tmp_path):
    train_filename = "en/c4-train.00000-of-01024.json.gz"
    destination = tmp_path / train_filename
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"old-")

    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        destination.write_bytes(b"new!")
        return str(destination)

    contents = {train_filename: b"new!"}

    download_c4_shards(
        tmp_path,
        num_train_shards=1,
        num_validation_shards=0,
        minimum_free_gib=0,
        metadata_fn=lambda revision, filenames: _metadata_with_sha(filenames, contents),
        download_fn=download,
        disk_usage_fn=lambda path: SimpleNamespace(free=4),
    )

    assert len(calls) == 1


def test_interrupted_download_reuses_each_hash_verified_shard_for_space_preflight(tmp_path):
    first_filename = "en/c4-train.00000-of-01024.json.gz"
    second_filename = "en/c4-train.00001-of-01024.json.gz"
    first_destination = tmp_path / first_filename
    first_destination.parent.mkdir(parents=True)
    first_destination.write_bytes(b"done")
    contents = {first_filename: b"done", second_filename: b"next"}
    calls = []

    def download(**kwargs):
        calls.append(kwargs["filename"])
        destination = tmp_path / kwargs["filename"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents[kwargs["filename"]])
        return str(destination)

    manifest = download_c4_shards(
        tmp_path,
        num_train_shards=2,
        num_validation_shards=0,
        minimum_free_gib=0,
        metadata_fn=lambda revision, filenames: _metadata_with_sha(filenames, contents),
        download_fn=download,
        disk_usage_fn=lambda path: SimpleNamespace(free=4),
    )

    assert calls == [second_filename]
    assert manifest["complete"] is True


def test_untrusted_same_size_fallback_with_wrong_sha_does_not_publish_manifest(tmp_path):
    train_filename = "en/c4-train.00000-of-01024.json.gz"
    destination = tmp_path / train_filename
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"old-")
    metadata = {
        train_filename: {
            "size": 4,
            "sha256": hashlib.sha256(b"new!").hexdigest(),
        }
    }

    def fallback_download(**kwargs):
        return str(destination)

    with pytest.raises(IOError, match="SHA256"):
        download_c4_shards(
            tmp_path,
            num_train_shards=1,
            num_validation_shards=0,
            minimum_free_gib=0,
            metadata_fn=lambda revision, filenames: metadata,
            download_fn=fallback_download,
            disk_usage_fn=lambda path: SimpleNamespace(free=4),
        )

    assert not (tmp_path / "manifest.json").exists()


def test_failed_download_keeps_completed_shards_and_no_complete_manifest(tmp_path):
    downloaded = []
    filenames = [
        "en/c4-train.00000-of-01024.json.gz",
        "en/c4-train.00001-of-01024.json.gz",
    ]
    contents = {filename: b"done" for filename in filenames}

    def download(**kwargs):
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if downloaded:
            raise RuntimeError("network broke")
        path.write_bytes(b"done")
        downloaded.append(path)
        return str(path)

    with pytest.raises(RuntimeError, match="network broke"):
        download_c4_shards(
            tmp_path,
            num_train_shards=2,
            num_validation_shards=0,
            revision=DEFAULT_REVISION,
            minimum_free_gib=0,
                metadata_fn=lambda revision, filenames: _metadata_with_sha(filenames, contents),
            download_fn=download,
            disk_usage_fn=lambda path: SimpleNamespace(free=8),
        )

    assert downloaded[0].exists()
    assert not (tmp_path / "manifest.json").exists()


def test_changed_rerun_archives_complete_manifest_before_failed_download(tmp_path):
    def successful_download(**kwargs):
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"done")
        return str(path)

    contents = {"en/c4-train.00000-of-01024.json.gz": b"done"}
    metadata = lambda revision, filenames: _metadata_with_sha(filenames, contents)
    download_c4_shards(
        tmp_path,
        num_train_shards=1,
        num_validation_shards=0,
        minimum_free_gib=0,
        metadata_fn=metadata,
        download_fn=successful_download,
        disk_usage_fn=lambda path: SimpleNamespace(free=4),
    )
    (tmp_path / "en" / "c4-train.00000-of-01024.json.gz").unlink()

    with pytest.raises(RuntimeError, match="network broke"):
        download_c4_shards(
            tmp_path,
            num_train_shards=1,
            num_validation_shards=0,
            minimum_free_gib=0,
            metadata_fn=metadata,
            download_fn=lambda **kwargs: (_ for _ in ()).throw(
                RuntimeError("network broke")
            ),
            disk_usage_fn=lambda path: SimpleNamespace(free=4),
        )

    assert not (tmp_path / "manifest.json").exists()
    archived = list(tmp_path.glob("manifest.json.stale-*"))
    assert len(archived) == 1
    assert json.loads(archived[0].read_text())["complete"] is True


@pytest.mark.parametrize("minimum_free_gib", [-1, math.nan, math.inf])
def test_minimum_free_gib_must_be_finite_and_nonnegative(tmp_path, minimum_free_gib):
    with pytest.raises(ValueError, match="minimum_free_gib"):
        download_c4_shards(
            tmp_path,
            num_train_shards=1,
            num_validation_shards=0,
            minimum_free_gib=minimum_free_gib,
            metadata_fn=lambda revision, filenames: _metadata_for(filenames),
            download_fn=lambda **kwargs: None,
            disk_usage_fn=lambda path: SimpleNamespace(free=10),
        )


def test_revision_must_be_a_full_commit_sha_before_network(tmp_path):
    with pytest.raises(ValueError, match="40-character"):
        download_c4_shards(
            tmp_path,
            num_train_shards=1,
            num_validation_shards=0,
            revision="main",
            metadata_fn=lambda revision, filenames: pytest.fail("metadata should not run"),
            download_fn=lambda **kwargs: pytest.fail("download should not run"),
            disk_usage_fn=lambda path: SimpleNamespace(free=10),
        )
