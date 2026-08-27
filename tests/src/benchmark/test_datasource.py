import importlib
import importlib.util
import inspect
import os
import pkgutil
import shutil
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

import evoagentx.benchmark as benchmark_package
from evoagentx.benchmark.benchmark import Benchmark
from evoagentx.benchmark.humaneval import HumanEval
from evoagentx.benchmark.mbpp import MBPP
from evoagentx.core.module_utils import load_json, save_json
from evoagentx.optimizers.core.data import DataSource


class TestHumanEvalAsDataSource(unittest.TestCase):
    """Step 1 acceptance: HumanEval satisfies the DataSource protocol."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data = load_json(path="tests/data/benchmark/humaneval_samples.jsonl", type="jsonl")
        save_json(self.sample_data, os.path.join(self.temp_dir, "HumanEval.jsonl"), type="jsonl")
        with patch("evoagentx.benchmark.humaneval.download_raw_humaneval_data"):
            self.benchmark = HumanEval(path=self.temp_dir, mode="test")
        self.all_ids = [ex["task_id"] for ex in self.sample_data]

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    # -- protocol conformance ------------------------------------------- #
    def test_satisfies_data_source_protocol(self):
        self.assertIsInstance(self.benchmark, DataSource)
        self.assertTrue(issubclass(HumanEval, DataSource))

    # -- get_split ------------------------------------------------------- #
    def test_get_split_returns_whole_split(self):
        data = self.benchmark.get_split("test")
        self.assertEqual([self.benchmark.get_id(ex) for ex in data], self.all_ids)
        self.assertEqual(self.benchmark.get_split("train"), [])

    def test_get_split_by_example_ids_follows_caller_order(self):
        wanted = [self.all_ids[3], self.all_ids[0], self.all_ids[7]]
        data = self.benchmark.get_split("test", example_ids=wanted)
        self.assertEqual([self.benchmark.get_id(ex) for ex in data], wanted)

    def test_get_split_rejects_unknown_and_duplicate_ids(self):
        with self.assertRaises(ValueError):
            self.benchmark.get_split("test", example_ids=["HumanEval/does-not-exist"])
        with self.assertRaises(ValueError):
            self.benchmark.get_split("test", example_ids=[self.all_ids[0], self.all_ids[0]])

    def test_same_seed_yields_identical_subset(self):
        first = self.benchmark.get_split("test", sample_k=4, seed=42)
        second = self.benchmark.get_split("test", sample_k=4, seed=42)
        self.assertEqual(len(first), 4)
        self.assertEqual(
            [self.benchmark.get_id(ex) for ex in first],
            [self.benchmark.get_id(ex) for ex in second],
        )

    def test_sampling_does_not_disturb_global_random_state(self):
        import random

        random.seed(7)
        expected = [random.random() for _ in range(3)]
        random.seed(7)
        self.benchmark.get_split("test", sample_k=4, seed=42)
        self.assertEqual([random.random() for _ in range(3)], expected)

    def test_example_ids_and_sample_k_compose(self):
        wanted = self.all_ids[:5]
        data = self.benchmark.get_split("test", example_ids=wanted, sample_k=2, seed=1)
        self.assertEqual(len(data), 2)
        self.assertTrue(set(self.benchmark.get_id(ex) for ex in data) <= set(wanted))

    def test_sample_k_larger_than_split_is_clamped(self):
        data = self.benchmark.get_split("test", sample_k=999, seed=1)
        self.assertEqual(len(data), len(self.all_ids))

    # -- get_reference --------------------------------------------------- #
    def test_get_reference_defaults_to_label(self):
        example = self.benchmark.get_split("test")[0]
        reference = self.benchmark.get_reference(self.benchmark.get_id(example))
        self.assertEqual(reference, self.benchmark.get_label(example))
        self.assertEqual(reference["entry_point"], example["entry_point"])

    def test_get_reference_raises_on_unknown_id(self):
        with self.assertRaises(KeyError):
            self.benchmark.get_reference("HumanEval/does-not-exist")


class _FakeBenchmark(Benchmark):
    """Minimal benchmark whose per-split ids come straight from the constructor."""

    def __init__(self, ids, train_ids=None, dev_ids=None):
        self._ids = {"test": ids, "train": train_ids, "dev": dev_ids}
        super().__init__(name="Fake", path="unused", mode="all")

    @staticmethod
    def _examples(split, ids):
        return None if ids is None else [{"id": i, "answer": f"{split}-answer-{i}"} for i in ids]

    def _load_data(self):
        self._train_data = self._examples("train", self._ids["train"])
        self._dev_data = self._examples("dev", self._ids["dev"])
        self._test_data = self._examples("test", self._ids["test"])

    def _get_id(self, example: Any) -> Any:
        return example["id"]

    def _get_label(self, example: Any) -> Any:
        return example["answer"]


class TestExampleIdContract(unittest.TestCase):
    """Ids are the primary key of an observation, so they are checked on load."""

    def test_string_ids_are_accepted(self):
        self.assertEqual(len(_FakeBenchmark(["a", "b"]).get_split("test")), 2)

    def test_non_string_ids_are_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            _FakeBenchmark([1, 2])
        self.assertIn("non-empty strings", str(ctx.exception))

    def test_empty_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            _FakeBenchmark(["a", ""])

    def test_duplicate_ids_only_warn(self):
        with patch("evoagentx.benchmark.benchmark.logger.warning") as mock_warning:
            _FakeBenchmark(["a", "a"])
        self.assertEqual(mock_warning.call_count, 1)
        self.assertIn("non-unique example ids", mock_warning.call_args[0][0])


class TestIdIndex(unittest.TestCase):
    """get_example_by_id is served from an index built at construction time."""

    def test_lookup_finds_examples_in_every_split(self):
        benchmark = _FakeBenchmark(["t1", "t2"], train_ids=["r1"], dev_ids=["d1"])
        for example_id, split in (("r1", "train"), ("d1", "dev"), ("t1", "test")):
            self.assertEqual(benchmark.get_example_by_id(example_id)["answer"], f"{split}-answer-{example_id}")
        self.assertIsNone(benchmark.get_example_by_id("nope"))

    def test_lookup_respects_mode(self):
        benchmark = _FakeBenchmark(["t1"], train_ids=["r1"])
        self.assertIsNone(benchmark.get_example_by_id("r1", mode="test"))
        self.assertEqual(benchmark.get_example_by_id("r1", mode="train")["answer"], "train-answer-r1")
        with self.assertRaises(ValueError):
            benchmark.get_example_by_id("r1", mode="validation")

    def test_ids_shared_across_splits_warn(self):
        with patch("evoagentx.benchmark.benchmark.logger.warning") as mock_warning:
            benchmark = _FakeBenchmark(["x"], train_ids=["x"])
        self.assertIn("share example ids", " ".join(call[0][0] for call in mock_warning.call_args_list))
        # ambiguous, but resolved in train/dev/test order — the documented behaviour
        self.assertEqual(benchmark.get_example_by_id("x")["answer"], "train-answer-x")

    def test_index_can_be_rebuilt_after_mutation(self):
        benchmark = _FakeBenchmark(["t1"])
        benchmark._test_data.append({"id": "t2", "answer": "test-answer-t2"})
        self.assertIsNone(benchmark.get_example_by_id("t2"))
        benchmark._index_examples()
        self.assertEqual(benchmark.get_example_by_id("t2")["answer"], "test-answer-t2")


class TestAllBenchmarksSatisfyProtocol(unittest.TestCase):
    """Every Benchmark in the package is a DataSource, with no per-class work."""

    def _benchmark_classes(self):
        found = {}
        for module_info in pkgutil.iter_modules(benchmark_package.__path__):
            try:
                module = importlib.import_module(f"evoagentx.benchmark.{module_info.name}")
            except ImportError:  # optional heavy dependency missing in this env
                continue
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if issubclass(cls, Benchmark) and cls is not Benchmark:
                    found[f"{cls.__module__}.{cls.__name__}"] = cls
        return found

    def test_every_benchmark_class_is_a_data_source(self):
        classes = self._benchmark_classes()
        self.assertGreaterEqual(len(classes), 10)
        for name, cls in classes.items():
            with self.subTest(benchmark=name):
                self.assertTrue(issubclass(cls, DataSource))


class TestMBPPIds(unittest.TestCase):
    """MBPP's raw task_id is an int; the benchmark exposes it as a string."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        sample_data = load_json(path="tests/data/benchmark/mbpp_samples.json", type="json")
        save_json(sample_data, os.path.join(self.temp_dir, "sanitized-mbpp.json"), type="json")
        self.benchmark = MBPP(path=self.temp_dir, mode="test")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_ids_are_strings_and_round_trip(self):
        example = self.benchmark.get_split("test")[0]
        example_id = self.benchmark.get_id(example)
        self.assertIsInstance(example_id, str)
        self.assertEqual(example_id, str(example["task_id"]))
        self.assertEqual(self.benchmark.get_reference(example_id), self.benchmark.get_label(example))

    def test_scorer_prompt_lookup_still_works(self):
        example = self.benchmark.get_split("test")[0]
        # the scorer resolves prompts from the raw int task_id carried by the label
        self.assertEqual(
            self.benchmark.get_example_by_id(str(example["task_id"]))["prompt"],
            example["prompt"],
        )


if __name__ == "__main__":
    unittest.main()