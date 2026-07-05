import asyncio
import random
from abc import ABC, abstractmethod
from typing import Optional, List, Any

from ..core.logging import logger


class Benchmark(ABC):

    """
    Base class for benchmarks.

    A ``Benchmark`` is responsible for loading / storing / providing data across
    train/dev/test splits and defining how a prediction is evaluated.

    Subclasses must implement:
        - ``_load_data``: populate ``_train_data`` / ``_dev_data`` / ``_test_data``
        - ``_get_id``: return a stable unique id for an example
        - ``_get_label``: return the ground-truth label for an example
        - ``evaluate``: compute metrics for a prediction / label pair
    """

    def __init__(self, name: str, path: str, mode: str = "all", **kwargs):
        """Initialize the benchmark.

        Args:
            name (str): The name of the benchmark.
            path (str): The path to the dataset.
            mode (str): which split(s) to load, one of ["all", "train", "dev", "test"].
            **kwargs: Additional parameters for customization.
        """
        valid_mode = ["all", "train", "dev", "test"]
        assert mode in valid_mode, f"Invalid value for mode: {mode}. Available choices: {valid_mode}"

        self.name = name
        self.path = path
        self.mode = mode
        self.kwargs = kwargs

        self._train_data: Optional[List[Any]] = None
        self._dev_data: Optional[List[Any]] = None
        self._test_data: Optional[List[Any]] = None

        self._load_data()
        self._validate_unique_ids()

    # ------------------------------------------------------------------ #
    # Abstract hooks
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _load_data(self):
        """Load data from ``self.path`` into ``_train_data`` / ``_dev_data`` / ``_test_data``."""
        pass

    @abstractmethod
    def _get_id(self, example: Any) -> Any:
        """Return the stable unique id for a given example."""
        pass

    @abstractmethod
    def _get_label(self, example: Any) -> Any:
        """Return the ground-truth label for a given example."""
        pass

    def evaluate(self, prediction: Any, label: Any) -> dict:
        """Evaluate a single prediction against the ground-truth label."""
        raise NotImplementedError(f"{type(self).__name__} does not implement evaluate.")

    async def async_evaluate(self, prediction: Any, label: Any) -> dict:
        """Async variant. Defaults to running ``evaluate`` in a worker thread."""
        return await asyncio.to_thread(self.evaluate, prediction, label)

    def _validate_unique_ids(self) -> None:
        """Warn when loaded examples have duplicate ids within the same split."""
        for split_name, data in (
            ("train", self._train_data),
            ("dev", self._dev_data),
            ("test", self._test_data),
        ):
            if data is None:
                continue

            seen_ids = []
            duplicate_ids = []
            for example in data:
                example_id = self._get_id(example=example)
                if any(example_id == seen_id for seen_id in seen_ids):
                    if not any(example_id == duplicate_id for duplicate_id in duplicate_ids):
                        duplicate_ids.append(example_id)
                else:
                    seen_ids.append(example_id)

            if duplicate_ids:
                logger.warning(
                    f"{split_name} data for benchmark {type(self).__name__} contains non-unique example ids: "
                    f"{sorted(duplicate_ids, key=str)}. Methods such as get_example_by_id may return only the "
                    "first matching example, making id-based lookup ambiguous."
                )

    # ------------------------------------------------------------------ #
    # Id / label accessors
    # ------------------------------------------------------------------ #
    def get_id(self, example: Any) -> Any:
        return self._get_id(example=example)

    def get_ids(self, examples: List[Any]) -> List[Any]:
        return [self._get_id(example=example) for example in examples]

    def get_label(self, example: Any) -> Any:
        return self._get_label(example=example)

    def get_labels(self, examples: List[Any]) -> List[Any]:
        return [self._get_label(example=example) for example in examples]

    # ------------------------------------------------------------------ #
    # Data accessors
    # ------------------------------------------------------------------ #
    def get_data_by_mode(self, mode: str = "test") -> List[Any]:
        assert mode in ["train", "dev", "test"], \
            f"Invalid value for mode: {mode}. Available choices: ['train', 'dev', 'test']"
        data = {"train": self._train_data, "dev": self._dev_data, "test": self._test_data}[mode]
        if data is None:
            logger.warning(f"{mode} data for benchmark {type(self).__name__} is not loaded or None. Return an empty list.")
            return []
        return data

    def get_example_by_id(self, example_id: Any, mode: str = None) -> Optional[Any]:
        if mode is not None and mode not in ["train", "dev", "test", "all"]:
            raise ValueError(f"Invalid value for mode: {mode}. Available choices: ['train', 'dev', 'test', 'all']")
        if mode is None or mode == "all":
            data = []
            for split in (self._train_data, self._dev_data, self._test_data):
                if split is not None:
                    data.extend(split)
        else:
            data = self.get_data_by_mode(mode=mode)
        for example in data:
            if self._get_id(example=example) == example_id:
                return example
        return None

    def get_example_by_index(self, index: int, mode: str = "test") -> Optional[Any]:
        data = self.get_data_by_mode(mode=mode)
        return data[index] if index < len(data) else None

    def _get_data(
        self,
        data: List[Any],
        indices: Optional[List[int]] = None,
        sample_k: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[Any]:
        """Select a subset of ``data`` by explicit indices and/or random sampling."""
        if data is None:
            return []
        if indices is None:
            indices = list(range(len(data)))
        if sample_k is not None:
            if seed is not None:
                random.seed(seed)
            indices = random.sample(indices, k=min(sample_k, len(indices)))
        return [data[idx] for idx in indices]

    def get_train_data(self, indices: Optional[List[int]] = None, sample_k: Optional[int] = None, seed: Optional[int] = None) -> List[Any]:
        return self._get_data(self.get_data_by_mode("train"), indices=indices, sample_k=sample_k, seed=seed)

    def get_dev_data(self, indices: Optional[List[int]] = None, sample_k: Optional[int] = None, seed: Optional[int] = None) -> List[Any]:
        return self._get_data(self.get_data_by_mode("dev"), indices=indices, sample_k=sample_k, seed=seed)

    def get_test_data(self, indices: Optional[List[int]] = None, sample_k: Optional[int] = None, seed: Optional[int] = None) -> List[Any]:
        return self._get_data(self.get_data_by_mode("test"), indices=indices, sample_k=sample_k, seed=seed)
