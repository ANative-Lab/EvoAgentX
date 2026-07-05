import copy
import random
from typing import Any, Iterator, List, Optional, Set, Tuple

from ..benchmark.benchmark import Benchmark

Batch = Tuple[List[Any], List[Any], List[Any]]
VALID_SPLITS = ("train", "dev", "test")


class DataLoader:
    """Batch iterator for benchmark examples.

    Args:
        benchmark: Benchmark instance that owns the source data.
        split: Dataset split to load. Valid values are ``"train"``, ``"dev"``,
            and ``"test"``.
        batch_size: Number of examples yielded per batch.
        shuffle: Whether to shuffle examples before batching.
        seed: Random seed used by benchmark sampling and loader shuffling.
        indices: Optional explicit example indices to select from the split.
        sample_k: Optional number of examples to randomly sample from the
            selected indices. If ``indices`` is also set, sampling happens within
            those indices.
        skip_ids: Optional set of benchmark example ids to exclude, typically
            used to resume an evaluation and skip already processed examples.

    Each call to ``iter(dataloader)`` reloads the split, applies filtering and
    sampling, and yields a new pass over the selected batches. Returned examples
    are deep-copied from the benchmark data so mutations made by the caller do
    not modify the benchmark's underlying data.
    """

    def __init__(
        self,
        benchmark: Benchmark,
        split: str = "test",
        batch_size: int = 1,
        shuffle: bool = False,
        seed: Optional[int] = None,
        indices: Optional[List[int]] = None,
        sample_k: Optional[int] = None,
        skip_ids: Optional[Set[Any]] = None,
    ):
        assert split in VALID_SPLITS, f"Invalid split: {split}. Valid values are: {VALID_SPLITS}"
        self.benchmark = benchmark
        self.split = split
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.indices = indices
        self.sample_k = sample_k
        self.skip_ids = set(skip_ids) if skip_ids else set()

    def _load_split(self) -> List[Any]:
        getter = {
            "train": self.benchmark.get_train_data,
            "dev": self.benchmark.get_dev_data,
            "test": self.benchmark.get_test_data,
        }[self.split]
        return copy.deepcopy(getter(indices=self.indices, sample_k=self.sample_k, seed=self.seed))

    def _examples(self) -> List[Any]:
        data = self._load_split()
        # Drop already-processed examples (resume).
        if self.skip_ids:
            data = [ex for ex in data if self.benchmark.get_id(ex) not in self.skip_ids]
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(data)
        return data

    def __len__(self) -> int:
        n = len(self._examples())
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Batch]:
        examples = self._examples()
        for start in range(0, len(examples), self.batch_size):
            batch = examples[start:start + self.batch_size]
            ids = self.benchmark.get_ids(batch)
            labels = self.benchmark.get_labels(batch)
            yield ids, batch, labels
