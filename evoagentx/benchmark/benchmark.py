import asyncio
import random
from abc import ABC, abstractmethod
from typing import Optional, List, Any

from ..core.logging import logger
from ..optimizers.core.data import ExampleId

SPLITS = ("train", "dev", "test")

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

    Every ``Benchmark`` structurally satisfies the optimization engine's
    ``DataSource`` protocol through ``get_split`` / ``get_id`` /
    ``get_reference``; no inheritance or registration is required.
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
        self._index_examples()

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

    def _split_items(self):
        """Yield ``(split_name, examples)`` for every loaded split."""
        data_by_split = {"train": self._train_data, "dev": self._dev_data, "test": self._test_data}
        for split_name in SPLITS:
            data = data_by_split[split_name]
            if data is not None:
                yield split_name, data

    def _index_examples(self) -> None:
        """Build the id index, validating ids along the way.

        An example id is the primary key of every observation the optimization
        engine records: it keys ``ObservationKey``, is written to report.jsonl,
        and is what a caller passes back as ``example_ids`` to re-evaluate a
        subset. It must therefore be a non-empty string, which is enforced here.
        Duplicates are only warned about, since they degrade id-based lookup
        rather than breaking it outright.

        Called once from ``__init__``. A subclass that mutates the loaded data
        afterwards must call it again to keep the index in step.

        Raises:
            ValueError: If ``_get_id`` returns anything but a non-empty string.
        """
        self._id_index: dict = {}
        for split_name, data in self._split_items():
            index: dict = {}
            duplicate_ids = set()
            for example in data:
                example_id = self._get_id(example=example)
                if not isinstance(example_id, str) or not example_id:
                    raise ValueError(
                        f"{type(self).__name__}._get_id returned {example_id!r} for an example in the "
                        f"{split_name} split. Example ids must be non-empty strings: they are the "
                        "primary key of every observation and are persisted as JSON. Wrap the raw "
                        "value in str() inside _get_id, or derive an id when the data has none."
                    )
                if example_id in index:
                    duplicate_ids.add(example_id)
                else:
                    index[example_id] = example  # first occurrence wins
            self._id_index[split_name] = index

            if duplicate_ids:
                logger.warning(
                    f"{split_name} data for benchmark {type(self).__name__} contains non-unique example ids: "
                    f"{sorted(duplicate_ids)}. Methods such as get_example_by_id may return only the "
                    "first matching example, making id-based lookup ambiguous."
                )

        shared_ids = set()
        for i, split_name in enumerate(SPLITS):
            for other in SPLITS[i + 1:]:
                shared_ids |= set(self._id_index.get(split_name, {})) & set(self._id_index.get(other, {}))
        if shared_ids:
            logger.warning(
                f"Splits of benchmark {type(self).__name__} share example ids: {sorted(shared_ids)[:10]}. "
                "get_example_by_id without a mode resolves them to the train/dev/test order, so an id "
                "no longer names one example."
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
        """Return the example carrying ``example_id``, or ``None``.

        Served from the id index built at construction time, so this is a dict
        lookup rather than a scan of the split.

        Args:
            example_id (Any): The id, as returned by ``get_id``.
            mode (str): Which split to search; ``None`` / ``"all"`` searches
                train, then dev, then test.

        Returns:
            Optional[Any]: The matching example, or ``None`` if there is none.
        """
        if mode is not None and mode not in ["train", "dev", "test", "all"]:
            raise ValueError(f"Invalid value for mode: {mode}. Available choices: ['train', 'dev', 'test', 'all']")
        splits = SPLITS if mode is None or mode == "all" else (mode,)
        for split_name in splits:
            example = self._id_index.get(split_name, {}).get(example_id)
            if example is not None:
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

    # ------------------------------------------------------------------ #
    # DataSource protocol
    # ------------------------------------------------------------------ #
    def _select_by_ids(self, split: str, example_ids: List[ExampleId]) -> List[Any]:
        """Select the examples of ``split`` named by ``example_ids``, in that order."""
        duplicates = sorted({eid for eid in example_ids if example_ids.count(eid) > 1}, key=str)
        if duplicates:
            raise ValueError(
                f"example_ids contains duplicate ids: {duplicates}. Repeated observations of the "
                "same example are expressed by EvalSpec.repeat, not by repeating an id."
            )
        selected = [self.get_example_by_id(eid, mode=split) for eid in example_ids]
        missing = [eid for eid, example in zip(example_ids, selected) if example is None]
        if missing:
            raise ValueError(
                f"example_ids not found in the {split} split of {type(self).__name__}: "
                f"{sorted(missing, key=str)[:10]}{' ...' if len(missing) > 10 else ''}"
            )
        return selected

    def get_split(
        self,
        split: str = "test",
        *,
        example_ids: Optional[List[ExampleId]] = None,
        sample_k: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[Any]:
        """Return the examples of ``split``, optionally narrowed to a subset.

        ``example_ids`` filters first (keeping the caller's order), then
        ``sample_k`` draws a random subset from what is left. The draw uses a
        local RNG seeded with ``seed``: the same seed yields the same examples in
        the same order, and the global ``random`` state is left untouched.

        Args:
            split (str): One of ``"train"`` / ``"dev"`` / ``"test"``.
            example_ids (Optional[List[ExampleId]]): Explicit subset by id.
            sample_k (Optional[int]): Random subset size; ``None`` = keep all.
            seed (Optional[int]): Seed for that random draw.

        Returns:
            List[Any]: The selected examples, in the order they are to be observed.
        """
        data = self.get_data_by_mode(mode=split)
        if example_ids is not None:
            data = self._select_by_ids(split, list(example_ids))
        if sample_k is not None:
            data = random.Random(seed).sample(data, k=min(sample_k, len(data)))
        return data

    def get_reference(self, example_id: ExampleId) -> Any:
        """Return what a prediction for ``example_id`` is scored against.

        Defaults to the example's label. Benchmarks scored against a rubric or a
        checklist override this to return that instead.

        Args:
            example_id (ExampleId): Id of the example, from ``get_id``.

        Returns:
            Any: The reference passed to the scorer.

        Raises:
            KeyError: If no loaded example carries that id.
        """
        example = self.get_example_by_id(example_id)
        if example is None:
            raise KeyError(f"{type(self).__name__} has no example with id {example_id!r}.")
        return self._get_label(example=example)
