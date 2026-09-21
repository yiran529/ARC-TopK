"""C4 data-source resolution for Hub and local raw shards."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import datasets
from datasets.iterable_dataset import _BaseExamplesIterable


_LOCAL_SPLIT_SPECS = {
    "train": {
        "pattern": "en/c4-train.[0-9][0-9][0-9][0-9][0-9]-of-01024.json.gz",
        "regex": re.compile(r"c4-train\.(\d{5})-of-01024\.json\.gz\Z"),
        "shard_count": 1024,
    },
    "validation": {
        "pattern": "en/c4-validation.[0-9][0-9][0-9][0-9][0-9]-of-00008.json.gz",
        "regex": re.compile(r"c4-validation\.(\d{5})-of-00008\.json\.gz\Z"),
        "shard_count": 8,
    },
}


class _InfiniteRepeatExamplesIterable(_BaseExamplesIterable):
    """Repeat an examples iterable while preserving the datasets 3.6 API."""

    def __init__(self, ex_iterable: _BaseExamplesIterable):
        super().__init__()
        self.ex_iterable = ex_iterable

    def _init_state_dict(self) -> dict:
        self._state_dict = {
            "repeat_index": 0,
            "examples_iterable": self.ex_iterable._init_state_dict(),
            "type": self.__class__.__name__,
        }
        return self._state_dict

    def __iter__(self):
        repeat_index = self._state_dict["repeat_index"] if self._state_dict else 0
        while True:
            yield from self.ex_iterable
            repeat_index += 1
            if self._state_dict:
                self._state_dict["repeat_index"] = repeat_index
                self._state_dict["examples_iterable"] = self.ex_iterable._init_state_dict()

    def shuffle_data_sources(self, generator):
        return self.__class__(self.ex_iterable.shuffle_data_sources(generator))

    def shard_data_sources(self, num_shards: int, index: int, contiguous=True):
        return self.__class__(
            self.ex_iterable.shard_data_sources(
                num_shards=num_shards,
                index=index,
                contiguous=contiguous,
            )
        )

    def split_shard_indices_by_worker(self, num_shards: int, index: int, contiguous=True):
        return self.ex_iterable.split_shard_indices_by_worker(
            num_shards=num_shards,
            index=index,
            contiguous=contiguous,
        )

    @property
    def is_typed(self):
        return self.ex_iterable.is_typed

    @property
    def features(self):
        return self.ex_iterable.features

    @property
    def num_shards(self) -> int:
        return self.ex_iterable.num_shards


def _validate_local_path(path: Path, split: str) -> bool:
    spec = _LOCAL_SPLIT_SPECS[split]
    match = spec["regex"].fullmatch(path.name)
    return match is not None and int(match.group(1)) < spec["shard_count"]


def _manifest_files(root: Path, split: str) -> list[Path] | None:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid local C4 manifest: {manifest_path}") from exc

    if manifest.get("complete") is not True:
        raise ValueError(f"local C4 manifest is not complete: {manifest_path}")

    filenames = manifest.get(f"{split}_files")
    if filenames is None:
        filenames = manifest.get(f"selected_{split}_filenames")
    if not isinstance(filenames, list) or not filenames:
        raise ValueError(f"manifest has no files for split {split!r}: {manifest_path}")

    file_sizes = manifest.get("file_sizes")
    if not isinstance(file_sizes, dict):
        raise ValueError(f"manifest has no file sizes: {manifest_path}")

    root_resolved = root.resolve()
    paths = []
    seen = set()
    for filename in filenames:
        if not isinstance(filename, str) or filename in seen:
            raise ValueError(f"invalid or duplicate manifest filename: {filename!r}")
        seen.add(filename)
        path = root / filename
        try:
            path.resolve().relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(f"manifest filename escapes dataset directory: {filename!r}") from exc
        if not _validate_local_path(path, split):
            raise ValueError(f"invalid {split} shard filename in manifest: {filename!r}")
        if not path.is_file():
            raise FileNotFoundError(f"manifest shard is missing: {path}")
        recorded_size = file_sizes.get(filename)
        if not isinstance(recorded_size, int) or isinstance(recorded_size, bool):
            raise ValueError(f"manifest has no valid size for {filename!r}")
        actual_size = path.stat().st_size
        if actual_size != recorded_size:
            raise ValueError(
                f"manifest size mismatch for {filename!r}: "
                f"recorded {recorded_size}, found {actual_size}"
            )
        paths.append(path)
    return paths


def _local_split_files(dataset_path: str, split: str) -> list[Path]:
    root = Path(dataset_path)
    if split not in _LOCAL_SPLIT_SPECS:
        raise ValueError(f"unsupported local C4 split: {split!r}")

    manifest_files = _manifest_files(root, split)
    if manifest_files is not None:
        return manifest_files

    spec = _LOCAL_SPLIT_SPECS[split]
    candidates = sorted(root.glob(spec["pattern"]))
    files = [path for path in candidates if path.is_file() and _validate_local_path(path, split)]
    if not files:
        raise FileNotFoundError(
            f"No local C4 {split} shards found; expected glob: {root / spec['pattern']}"
        )
    return files


def _repeat_local_dataset(dataset: datasets.IterableDataset) -> datasets.IterableDataset:
    return datasets.IterableDataset(
        ex_iterable=_InfiniteRepeatExamplesIterable(dataset._ex_iterable),
        info=dataset._info.copy(),
        split=dataset._split,
        formatting=dataset._formatting,
        shuffling=copy.deepcopy(dataset._shuffling),
        distributed=copy.deepcopy(dataset._distributed),
        token_per_repo_id=dataset._token_per_repo_id,
    )


def load_c4_split(
    dataset_path: str,
    split: str,
    *,
    repeat_local: bool = False,
) -> datasets.IterableDataset:
    """Load one C4 split from the Hub or a directory of raw JSONL shards."""
    if dataset_path == "allenai/c4":
        return datasets.load_dataset(
            dataset_path,
            "en",
            split=split,
            streaming=True,
        )

    if Path(dataset_path).is_dir():
        files = _local_split_files(dataset_path, split)
        dataset = datasets.load_dataset(
            "json",
            data_files={split: [str(path) for path in files]},
            split=split,
            streaming=True,
        )
        if repeat_local:
            dataset = _repeat_local_dataset(dataset)
        return dataset

    return datasets.load_dataset(dataset_path, split=split, streaming=True)
