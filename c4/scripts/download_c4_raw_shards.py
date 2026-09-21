#!/usr/bin/env python3
"""Download a revision-pinned subset of the raw English C4 shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

from huggingface_hub import HfApi, hf_hub_download


REPOSITORY = "allenai/c4"
CONFIG = "en"
DEFAULT_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
NUM_TRAIN_SHARDS = 1024
NUM_VALIDATION_SHARDS = 8
DEFAULT_NUM_TRAIN_SHARDS = 30
DEFAULT_NUM_VALIDATION_SHARDS = 1
DEFAULT_MINIMUM_FREE_GIB = 4.0
_FULL_SHA_RE = re.compile(r"[0-9a-fA-F]{40}\Z")


class InsufficientDiskSpaceError(RuntimeError):
    """Raised when the planned download would violate the free-space reserve."""


def _validate_revision(revision: str) -> str:
    if not isinstance(revision, str) or _FULL_SHA_RE.fullmatch(revision) is None:
        raise ValueError("revision must be a 40-character commit SHA")
    return revision.lower()


def _validate_minimum_free_gib(minimum_free_gib: float) -> float:
    if (
        isinstance(minimum_free_gib, bool)
        or not isinstance(minimum_free_gib, (int, float))
        or not math.isfinite(float(minimum_free_gib))
        or minimum_free_gib < 0
    ):
        raise ValueError("minimum_free_gib must be finite and nonnegative")
    return float(minimum_free_gib)


def shard_filename(split: str, index: int) -> str:
    """Return the raw C4 filename for a zero-based shard index."""
    if split == "train":
        shard_count = NUM_TRAIN_SHARDS
    elif split == "validation":
        shard_count = NUM_VALIDATION_SHARDS
    else:
        raise ValueError(f"unsupported C4 split: {split!r}")

    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < shard_count:
        raise ValueError(
            f"{split} shard index must be in [0, {shard_count - 1}], got {index!r}"
        )

    return f"en/c4-{split}.{index:05d}-of-{shard_count:05d}.json.gz"


def select_shard_filenames(
    num_train_shards: int,
    num_validation_shards: int,
) -> tuple[list[str], list[str]]:
    """Select the first requested train and validation shards in stable order."""
    if (
        not isinstance(num_train_shards, int)
        or isinstance(num_train_shards, bool)
        or not 1 <= num_train_shards <= NUM_TRAIN_SHARDS
    ):
        raise ValueError(
            f"num_train_shards must be between 1 and {NUM_TRAIN_SHARDS}"
        )
    if (
        not isinstance(num_validation_shards, int)
        or isinstance(num_validation_shards, bool)
        or not 0 <= num_validation_shards <= NUM_VALIDATION_SHARDS
    ):
        raise ValueError(
            f"num_validation_shards must be between 0 and {NUM_VALIDATION_SHARDS}"
        )

    train_filenames = [shard_filename("train", index) for index in range(num_train_shards)]
    validation_filenames = [
        shard_filename("validation", index) for index in range(num_validation_shards)
    ]
    return train_filenames, validation_filenames


def get_remote_file_sizes(
    revision: str,
    filenames: Sequence[str],
) -> dict[str, int]:
    """Query Hub metadata and return expected byte sizes for selected files."""
    return {
        filename: int(metadata["size"])
        for filename, metadata in get_remote_file_metadata(revision, filenames).items()
    }


def get_remote_file_metadata(
    revision: str,
    filenames: Sequence[str],
) -> dict[str, dict[str, str | int | None]]:
    """Query Hub metadata, including LFS SHA256 values when available."""
    info = HfApi().dataset_info(
        REPOSITORY,
        revision=revision,
        files_metadata=True,
    )
    metadata = {
        sibling.rfilename: {
            "size": sibling.size,
            "sha256": getattr(getattr(sibling, "lfs", None), "sha256", None),
        }
        for sibling in info.siblings
        if sibling.size is not None
    }
    missing = [filename for filename in filenames if filename not in metadata]
    if missing:
        raise RuntimeError(
            f"Hub metadata has no sizes for selected C4 files: {', '.join(missing)}"
        )
    return {filename: metadata[filename] for filename in filenames}


def _normalize_remote_metadata(
    metadata: Mapping[str, object],
) -> dict[str, dict[str, str | int | None]]:
    normalized = {}
    for filename, value in metadata.items():
        if isinstance(value, Mapping):
            size = value.get("size")
            sha256 = value.get("sha256", value.get("lfs_sha256"))
        else:
            size = getattr(value, "size", value if isinstance(value, int) else None)
            sha256 = getattr(value, "sha256", None)
            lfs = getattr(value, "lfs", None)
            if sha256 is None:
                sha256 = getattr(lfs, "sha256", None)
        if size is None:
            continue
        normalized[filename] = {
            "size": int(size),
            "sha256": str(sha256).lower() if sha256 is not None else None,
        }
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *,
    revision: str,
    train_filenames: Sequence[str],
    validation_filenames: Sequence[str],
    file_sizes: Mapping[str, int],
    file_hashes: Mapping[str, str] | None = None,
    complete: bool,
) -> dict:
    """Build the auditable manifest payload written after a successful run."""
    sizes = {filename: int(file_sizes[filename]) for filename in file_sizes}
    requested_counts = {
        "train": len(train_filenames),
        "validation": len(validation_filenames),
    }
    manifest = {
        "repository": REPOSITORY,
        "revision": revision,
        "config": CONFIG,
        "train_files": list(train_filenames),
        "validation_files": list(validation_filenames),
        "selected_train_filenames": list(train_filenames),
        "selected_validation_filenames": list(validation_filenames),
        "requested_counts": requested_counts,
        "requested_train_shards": requested_counts["train"],
        "requested_validation_shards": requested_counts["validation"],
        "file_sizes": sizes,
        "total_bytes": sum(sizes.values()),
        "complete": bool(complete),
        "status": "complete" if complete else "incomplete",
    }
    if file_hashes:
        manifest["file_sha256"] = {
            filename: str(digest).lower() for filename, digest in file_hashes.items()
        }
    return manifest


def write_manifest_atomically(output_dir: str | Path, manifest: Mapping) -> Path:
    """Serialize a manifest through a same-directory temporary file and replace."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_dir,
            prefix=".manifest-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = stream.name
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, manifest_path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return manifest_path


def _read_manifest_if_present(manifest_path: Path) -> dict | None:
    if not manifest_path.is_file():
        return None
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _manifest_filenames(manifest: Mapping, split: str) -> list:
    filenames = manifest.get(f"{split}_files")
    if filenames is None:
        filenames = manifest.get(f"selected_{split}_filenames")
    return filenames if isinstance(filenames, list) else []


def _manifest_matches_request(
    manifest: Mapping,
    *,
    output_dir: Path,
    revision: str,
    train_filenames: Sequence[str],
    validation_filenames: Sequence[str],
    expected_sizes: Mapping[str, int] | None,
    expected_hashes: Mapping[str, str | None] | None = None,
) -> bool:
    if manifest.get("complete") is not True or manifest.get("revision") != revision:
        return False
    if _manifest_filenames(manifest, "train") != list(train_filenames):
        return False
    if _manifest_filenames(manifest, "validation") != list(validation_filenames):
        return False
    file_sizes = manifest.get("file_sizes")
    if not isinstance(file_sizes, dict):
        return False

    for filename in list(train_filenames) + list(validation_filenames):
        recorded_size = file_sizes.get(filename)
        if not isinstance(recorded_size, int) or isinstance(recorded_size, bool):
            return False
        if expected_sizes is not None and recorded_size != expected_sizes[filename]:
            return False
        destination = output_dir / filename
        if not destination.is_file() or destination.stat().st_size != recorded_size:
            return False
        expected_hash = expected_hashes.get(filename) if expected_hashes is not None else None
        if expected_hash is not None and _sha256_file(destination) != expected_hash:
            return False
    return True


def _archive_manifest(output_dir: Path) -> Path | None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    archive_path = output_dir / f"manifest.json.stale-{time.time_ns()}.json"
    while archive_path.exists():
        archive_path = output_dir / f"manifest.json.stale-{time.time_ns()}.json"
    os.replace(manifest_path, archive_path)
    return archive_path


def download_c4_shards(
    output_dir: str | Path,
    *,
    num_train_shards: int = DEFAULT_NUM_TRAIN_SHARDS,
    num_validation_shards: int = DEFAULT_NUM_VALIDATION_SHARDS,
    revision: str = DEFAULT_REVISION,
    minimum_free_gib: float = DEFAULT_MINIMUM_FREE_GIB,
    metadata_fn: Callable[[str, Sequence[str]], Mapping[str, int]] | None = None,
    download_fn: Callable[..., str] | None = None,
    disk_usage_fn: Callable[[str | Path], object] = shutil.disk_usage,
) -> dict:
    """Download selected raw shards, reusing complete files and writing a manifest."""
    train_filenames, validation_filenames = select_shard_filenames(
        num_train_shards,
        num_validation_shards,
    )
    filenames = train_filenames + validation_filenames
    revision = _validate_revision(revision)
    minimum_free_gib = _validate_minimum_free_gib(minimum_free_gib)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    existing_manifest = _read_manifest_if_present(manifest_path)
    manifest_archived = False
    if existing_manifest is not None and not _manifest_matches_request(
        existing_manifest,
        output_dir=output_dir,
        revision=revision,
        train_filenames=train_filenames,
        validation_filenames=validation_filenames,
        expected_sizes=None,
    ):
        _archive_manifest(output_dir)
        manifest_archived = True

    if metadata_fn is None:
        metadata_fn = get_remote_file_metadata
    remote_metadata = _normalize_remote_metadata(metadata_fn(revision, filenames))
    expected_sizes = {
        filename: int(remote_metadata[filename]["size"])
        for filename in remote_metadata
    }
    expected_hashes = {
        filename: metadata["sha256"]
        for filename, metadata in remote_metadata.items()
    }
    missing_sizes = [filename for filename in filenames if filename not in expected_sizes]
    if missing_sizes:
        raise RuntimeError(
            f"Metadata did not provide sizes for selected files: {', '.join(missing_sizes)}"
        )

    if existing_manifest is not None and not manifest_archived and not _manifest_matches_request(
        existing_manifest,
        output_dir=output_dir,
        revision=revision,
        train_filenames=train_filenames,
        validation_filenames=validation_filenames,
        expected_sizes=expected_sizes,
        expected_hashes=expected_hashes,
    ):
        _archive_manifest(output_dir)
        manifest_archived = True

    trusted_manifest = (
        existing_manifest is not None
        and not manifest_archived
        and _manifest_matches_request(
            existing_manifest,
            output_dir=output_dir,
            revision=revision,
            train_filenames=train_filenames,
            validation_filenames=validation_filenames,
            expected_sizes=expected_sizes,
            expected_hashes=expected_hashes,
        )
    )

    trusted_existing_files = set()
    for filename in filenames:
        destination = output_dir / filename
        if trusted_manifest:
            trusted_existing_files.add(filename)
        elif (
            destination.is_file()
            and destination.stat().st_size == expected_sizes[filename]
            and expected_hashes.get(filename) is not None
            and _sha256_file(destination) == expected_hashes[filename]
        ):
            trusted_existing_files.add(filename)

    pending_filenames = []
    for filename in filenames:
        if filename not in trusted_existing_files:
            pending_filenames.append(filename)

    planned_bytes = sum(expected_sizes[filename] for filename in pending_filenames)
    free_bytes = int(disk_usage_fn(output_dir).free)
    required_free_bytes = int(minimum_free_gib * (1024**3))
    if free_bytes < planned_bytes + required_free_bytes:
        raise InsufficientDiskSpaceError(
            "Insufficient free space: "
            f"{free_bytes} bytes free, {planned_bytes} bytes planned, "
            f"and {required_free_bytes} bytes reserved"
        )

    download_callable = hf_hub_download if download_fn is None else download_fn
    for filename in pending_filenames:
        destination = output_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected_hash = expected_hashes.get(filename)
        if expected_hash is None:
            raise IOError(f"Remote SHA256 unavailable for {filename}")
        download_callable(
            repo_id=REPOSITORY,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            local_dir=output_dir,
        )
        if not destination.is_file():
            raise FileNotFoundError(
                f"Hub download did not create expected destination: {destination}"
            )
        actual_size = destination.stat().st_size
        if actual_size != expected_sizes[filename]:
            raise IOError(
                f"Downloaded {filename} has size {actual_size}, "
                f"expected {expected_sizes[filename]}"
            )
        actual_hash = _sha256_file(destination)
        if actual_hash != expected_hash:
            raise IOError(
                f"Downloaded {filename} has SHA256 {actual_hash}, "
                f"expected {expected_hash}"
            )

    successful_sizes = {
        filename: (output_dir / filename).stat().st_size for filename in filenames
    }
    manifest = build_manifest(
        revision=revision,
        train_filenames=train_filenames,
        validation_filenames=validation_filenames,
        file_sizes=successful_sizes,
        file_hashes={
            filename: expected_hashes[filename]
            for filename in filenames
            if expected_hashes.get(filename) is not None
        },
        complete=True,
    )
    write_manifest_atomically(output_dir, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-train-shards", type=int, default=DEFAULT_NUM_TRAIN_SHARDS)
    parser.add_argument(
        "--num-validation-shards",
        type=int,
        default=DEFAULT_NUM_VALIDATION_SHARDS,
    )
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--minimum-free-gib",
        type=float,
        default=DEFAULT_MINIMUM_FREE_GIB,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    return download_c4_shards(
        args.output_dir,
        num_train_shards=args.num_train_shards,
        num_validation_shards=args.num_validation_shards,
        revision=args.revision,
        minimum_free_gib=args.minimum_free_gib,
    )


if __name__ == "__main__":
    main()
