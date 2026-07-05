"""Tests for the refactored evaluation pipeline: DataLoader + EvaluationPipeline +
EvaluationReport + the to_result() bridge to the optimizer-facing EvaluationResult."""

import asyncio
import os
import tempfile
from unittest.mock import patch

import pytest

from evoagentx.benchmark.benchmark import BaseBenchmark
from evoagentx.evaluators import DataLoader, EvaluationPipeline
from evoagentx.evaluators.result import EvaluationReport, EvaluationResult, ItemResult


class FakeBenchmark(BaseBenchmark):
    """Tiny benchmark: example {id, x, y=2x}; the "correct" prediction is y."""

    def __init__(self, n: int = 6):
        self._n = n
        super().__init__(name="fake", path=".", mode="all")

    def _load_data(self):
        self._test_data = [{"id": i, "x": i, "y": 2 * i} for i in range(self._n)]
        self._train_data = []
        self._dev_data = []

    def _get_id(self, example):
        return example["id"]

    def _get_label(self, example):
        return example["y"]


class DuplicateIdBenchmark(FakeBenchmark):
    def _load_data(self):
        self._test_data = [{"id": "dup", "x": 1, "y": 2}, {"id": "dup", "x": 2, "y": 4}]
        self._train_data = None
        self._dev_data = []


def _scorer(prediction, label):
    return {"acc": 1.0 if prediction == label else 0.0}


def _double(example):
    """Correct program: example dict -> 2x."""
    return example["x"] * 2


def _ids_from_loader(loader):
    ids = []
    for batch_ids, _, _ in loader:
        ids.extend(batch_ids)
    return ids


def test_base_benchmark_warns_on_duplicate_ids_in_loaded_split():
    with patch("evoagentx.benchmark.benchmark.logger.warning") as warning:
        DuplicateIdBenchmark()

    warning.assert_called_once()
    message = warning.call_args.args[0]
    assert "test data" in message
    assert "non-unique example ids" in message
    assert "get_example_by_id" in message


# --------------------------------------------------------------------------- #
# DataLoader
# --------------------------------------------------------------------------- #
def test_dataloader_invalid_split_message_lists_valid_values():
    with pytest.raises(AssertionError) as exc:
        DataLoader(FakeBenchmark(), split="validation")

    assert "Invalid split: validation" in str(exc.value)
    assert "train" in str(exc.value)
    assert "dev" in str(exc.value)
    assert "test" in str(exc.value)


def test_dataloader_examples_are_independent_from_benchmark_data():
    bench = FakeBenchmark(n=1)
    bench._test_data[0]["nested"] = {"value": "original"}

    _, batch, _ = next(iter(DataLoader(bench, split="test")))
    batch[0]["x"] = 999
    batch[0]["nested"]["value"] = "changed"

    assert bench._test_data[0]["x"] == 0
    assert bench._test_data[0]["nested"]["value"] == "original"


def test_dataloader_reloads_split_for_each_iteration():
    bench = FakeBenchmark(n=6)
    original_get_test_data = bench.get_test_data
    calls = []

    def tracking_get_test_data(*args, **kwargs):
        calls.append(kwargs)
        return original_get_test_data(*args, **kwargs)

    bench.get_test_data = tracking_get_test_data
    loader = DataLoader(bench, split="test", batch_size=2, sample_k=3, seed=7)

    first_pass = _ids_from_loader(loader)
    second_pass = _ids_from_loader(loader)

    assert len(calls) == 2
    assert first_pass == second_pass
    assert len(first_pass) == 3


# --------------------------------------------------------------------------- #
# Report / aggregate / to_result
# --------------------------------------------------------------------------- #
def test_run_all_correct_aggregate_and_to_result():
    bench = FakeBenchmark(n=6)
    pipeline = EvaluationPipeline(max_concurrency=4, verbose=False)
    report = pipeline.run(_double, DataLoader(bench, split="test"), _scorer)

    assert isinstance(report, EvaluationReport)
    assert report.total_items == 6
    assert report.total_errors == 0
    assert report.aggregate() == {"acc": 1.0}

    result = report.to_result()
    assert isinstance(result, EvaluationResult)
    assert result.metrics == {"acc": 1.0}
    assert result.metadata["total_items"] == 6
    assert result.metadata["total_errors"] == 0
    assert result.metadata["error_ids"] == []


def test_on_error_policies_count_failures():
    bench = FakeBenchmark(n=4)

    def flaky(example):
        if example["x"] == 1:  # 1 of 4 fails
            raise RuntimeError("boom")
        return example["x"] * 2

    report = EvaluationPipeline(verbose=False).run(flaky, DataLoader(bench, split="test"), _scorer)
    assert report.total_items == 4
    assert report.total_errors == 1

    # zero (default): failure penalized -> 3/4 correct
    assert report.aggregate(on_error="zero")["acc"] == pytest.approx(0.75)
    # ignore: average over successes only -> 3/3
    assert report.aggregate(on_error="ignore")["acc"] == pytest.approx(1.0)
    # raise: blow up when any failure
    with pytest.raises(RuntimeError):
        report.aggregate(on_error="raise")

    # to_result default uses on_error="zero"
    assert report.to_result().metrics["acc"] == pytest.approx(0.75)
    assert report.to_result().metadata["error_ids"] == [1]


def test_all_failed_trial_scores_zero_via_declared_metric_keys():
    """When every item crashes, a scorer-declared key set makes on_error='zero' report
    0.0 per key instead of collapsing to an empty dict (which would leave the optimizer's
    objective undefined)."""
    from evoagentx.benchmark.scorers.base import Scorer

    class AccScorer(Scorer):
        def score(self, prediction, label):
            return {"acc": 1.0 if prediction == label else 0.0}

        @property
        def metric_keys(self):
            return ("acc",)

    bench = FakeBenchmark(n=4)

    def always_boom(example):
        raise RuntimeError("boom")

    report = EvaluationPipeline(verbose=False).run(
        always_boom, DataLoader(bench, split="test"), AccScorer())
    assert report.total_items == 4
    assert report.total_errors == 4
    assert report.metric_keys == ("acc",)  # captured from the scorer

    # zero: all-failed trial is the worst score, not an empty/undefined objective
    assert report.aggregate(on_error="zero") == {"acc": 0.0}
    assert report.to_result().metrics == {"acc": 0.0}
    # ignore: no successes to average -> genuinely no data
    assert report.aggregate(on_error="ignore") == {}


def test_all_failed_trial_without_declared_keys_returns_empty():
    """Back-compat: a bare-callable scorer declares no keys, so an all-failed trial can't
    know which keys to zero-fill and still returns {}."""
    bench = FakeBenchmark(n=3)

    def always_boom(example):
        raise RuntimeError("boom")

    report = EvaluationPipeline(verbose=False).run(
        always_boom, DataLoader(bench, split="test"), _scorer)
    assert report.metric_keys == ()
    assert report.aggregate(on_error="zero") == {}
    # ...unless the caller supplies keys explicitly (the escape hatch)
    assert report.aggregate(on_error="zero", metric_keys=["acc"]) == {"acc": 0.0}
    assert report.to_result(metric_keys=["acc"]).metrics == {"acc": 0.0}


def test_metric_keys_not_persisted_but_restorable_on_load():
    """metric_keys is a scorer contract, not per-item data, so save/load doesn't carry it.
    A standalone all-failed report re-aggregates to {} unless keys are re-supplied."""
    report = EvaluationReport(
        items=[ItemResult(id=i, error="boom") for i in range(3)],
        metric_keys=("acc",),
    )
    assert report.aggregate(on_error="zero") == {"acc": 0.0}

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "report.jsonl")
        report.save(path)

        # metric_keys is gone after a bare load -> back to {}
        loaded = EvaluationReport.load(path)
        assert loaded.metric_keys == ()
        assert loaded.aggregate(on_error="zero") == {}

        # ...but re-supplying the scorer's keys restores the zero-fill
        restored = EvaluationReport.load(path, metric_keys=("acc",))
        assert restored.metric_keys == ("acc",)
        assert restored.aggregate(on_error="zero") == {"acc": 0.0}


def test_input_fn_and_output_fn():
    bench = FakeBenchmark(n=3)
    # program now takes a raw int and returns a dict; hooks adapt both ends.
    report = EvaluationPipeline(verbose=False).run(
        program=lambda x: {"answer": x * 2},
        dataloader=DataLoader(bench, split="test"),
        scorer=_scorer,
        input_fn=lambda ex: ex["x"],
        output_fn=lambda raw: raw["answer"],
    )
    assert report.aggregate() == {"acc": 1.0}


def test_trace_fn_and_to_result_include_traces():
    bench = FakeBenchmark(n=3)
    report = EvaluationPipeline(verbose=False).run(
        _double,
        DataLoader(bench, split="test"),
        _scorer,
        trace_fn=lambda ex, raw: {"x": ex["x"], "raw": raw},
    )
    assert all(it.trace is not None for it in report.items)
    result = report.to_result(include_traces=True)
    assert len(result.traces) == 3
    # callable mapper form
    result2 = report.to_result(include_traces=lambda it: it.trace["x"])
    assert sorted(result2.traces) == [0, 1, 2]


def test_to_result_builder_override():
    bench = FakeBenchmark(n=2)
    report = EvaluationPipeline(verbose=False).run(_double, DataLoader(bench, split="test"), _scorer)
    result = report.to_result(builder=lambda rep: EvaluationResult(
        metrics={"n": rep.total_items}, artifacts={"custom": True}))
    assert result.metrics == {"n": 2}
    assert result.artifacts == {"custom": True}


# --------------------------------------------------------------------------- #
# Batch mode
# --------------------------------------------------------------------------- #
def test_batch_mode():
    bench = FakeBenchmark(n=6)
    report = EvaluationPipeline(max_concurrency=2, verbose=False).run(
        program=lambda xs: [x * 2 for x in xs],   # called once per batch
        dataloader=DataLoader(bench, split="test", batch_size=3),
        scorer=_scorer,
        input_fn=lambda ex: ex["x"],
        program_mode="batch",
    )
    assert report.total_items == 6
    assert report.aggregate() == {"acc": 1.0}


def test_batch_mode_wrong_length_fails_items():
    bench = FakeBenchmark(n=4)
    report = EvaluationPipeline(verbose=False).run(
        program=lambda xs: [1],  # wrong length on purpose
        dataloader=DataLoader(bench, split="test", batch_size=2),
        scorer=_scorer,
        input_fn=lambda ex: ex["x"],
        program_mode="batch",
    )
    assert report.total_errors == 4
    assert all("expected" in err for _, err in report.errors)


# --------------------------------------------------------------------------- #
# Async path
# --------------------------------------------------------------------------- #
def test_arun_async_program():
    bench = FakeBenchmark(n=5)

    async def aprogram(example):
        await asyncio.sleep(0)
        return example["x"] * 2

    async def go():
        return await EvaluationPipeline(max_concurrency=3, verbose=False).arun(
            aprogram, DataLoader(bench, split="test"), _scorer)

    report = asyncio.run(go())
    assert report.total_items == 5
    assert report.aggregate() == {"acc": 1.0}


def test_arun_sync_program():
    bench = FakeBenchmark(n=4)

    async def go():
        return await EvaluationPipeline(verbose=False).arun(
            _double, DataLoader(bench, split="test"), _scorer)

    report = asyncio.run(go())
    assert report.aggregate() == {"acc": 1.0}


# --------------------------------------------------------------------------- #
# output_dir: JSONL item log, resume, traceback dumping
# --------------------------------------------------------------------------- #
def test_output_dir_writes_jsonl_item_log():
    bench = FakeBenchmark(n=3)
    with tempfile.TemporaryDirectory() as tmp:
        report = EvaluationPipeline(verbose=False).run(
            _double, DataLoader(bench, split="test"), _scorer,
            output_dir=tmp, flush_every=1)

        log_path = os.path.join(tmp, "report.jsonl")
        assert os.path.exists(log_path)
        loaded = EvaluationReport.load(log_path)
        assert {it.id for it in loaded.items} == {0, 1, 2}
        assert loaded.aggregate() == report.aggregate() == {"acc": 1.0}
        # Full items survive the round-trip, predictions included.
        assert loaded.predictions == report.predictions


def test_resume_is_lossless_and_skips_processed_items():
    bench = FakeBenchmark(n=6)
    calls = []

    def tracked(example):
        calls.append(example["x"])
        return example["x"] * 2

    with tempfile.TemporaryDirectory() as tmp:
        # First pass over 3 examples (indices 0,1,2), streaming the item log.
        loader1 = DataLoader(bench, split="test", indices=[0, 1, 2])
        EvaluationPipeline(verbose=False).run(
            tracked, loader1, _scorer, output_dir=tmp, flush_every=1)
        assert sorted(calls) == [0, 1, 2]

        # Resume over all 6: the first 3 are rebuilt from the log (with predictions),
        # only 3,4,5 are freshly evaluated, and the report covers all 6.
        loader2 = DataLoader(bench, split="test")
        report = EvaluationPipeline(verbose=False).run(
            tracked, loader2, _scorer, output_dir=tmp, flush_every=1, resume=True)
        assert sorted(calls) == [0, 1, 2, 3, 4, 5]
        assert {it.id for it in report.items} == {0, 1, 2, 3, 4, 5}
        assert report.predictions == {i: 2 * i for i in range(6)}  # lossless resume
        assert report.aggregate() == {"acc": 1.0}


def test_fresh_run_replaces_stale_item_log():
    bench = FakeBenchmark(n=2)
    with tempfile.TemporaryDirectory() as tmp:
        EvaluationPipeline(verbose=False).run(
            _double, DataLoader(bench, split="test"), _scorer, output_dir=tmp, resume=False)
        # A second run with resume=False must overwrite, not append to, the previous log.
        EvaluationPipeline(verbose=False).run(
            _double, DataLoader(bench, split="test"), _scorer, output_dir=tmp, resume=False)
        loaded = EvaluationReport.load(os.path.join(tmp, "report.jsonl"))
        assert loaded.total_items == 2


def test_failed_item_traceback_captured_on_report():
    bench = FakeBenchmark(n=2)

    def flaky(example):
        if example["x"] == 1:
            raise RuntimeError("boom")
        return example["x"] * 2

    with tempfile.TemporaryDirectory() as tmp:
        report = EvaluationPipeline(verbose=False).run(
            flaky,
            DataLoader(bench, split="test"),
            _scorer,
            output_dir=tmp,
            flush_every=1,
        )

        assert report.total_errors == 1
        # The full traceback rides on the failed item; no separate errors/ dir is written.
        assert not os.path.exists(os.path.join(tmp, "errors"))
        assert "RuntimeError: boom" in report.failures[0].traceback


def test_failed_item_traceback_survives_jsonl_round_trip():
    bench = FakeBenchmark(n=2)

    def flaky(example):
        if example["x"] == 1:
            raise RuntimeError("boom")
        return example["x"] * 2

    with tempfile.TemporaryDirectory() as tmp:
        EvaluationPipeline(verbose=False).run(
            flaky, DataLoader(bench, split="test"), _scorer, output_dir=tmp)
        loaded = EvaluationReport.load(os.path.join(tmp, "report.jsonl"))
        failed = [it for it in loaded.items if not it.ok]
        assert len(failed) == 1
        assert "RuntimeError: boom" in failed[0].traceback


def test_shared_pipeline_concurrent_aruns_keep_output_dirs_separate():
    """One pipeline instance shared by concurrent arun calls (the optimizer's
    concurrent-trials pattern) must not cross-write logs between runs."""
    bench = FakeBenchmark(n=4)
    pipeline = EvaluationPipeline(max_concurrency=2, verbose=False)

    def make_flaky(tag):
        async def flaky(example):
            await asyncio.sleep(0.01)
            if example["x"] == 1:
                raise RuntimeError(f"boom-{tag}")
            return example["x"] * 2
        return flaky

    with tempfile.TemporaryDirectory() as tmp:
        dir_a, dir_b = os.path.join(tmp, "a"), os.path.join(tmp, "b")

        async def go():
            return await asyncio.gather(
                pipeline.arun(make_flaky("a"), DataLoader(bench, split="test"), _scorer, output_dir=dir_a),
                pipeline.arun(make_flaky("b"), DataLoader(bench, split="test"), _scorer, output_dir=dir_b),
            )

        report_a, report_b = asyncio.run(go())
        assert report_a.total_errors == report_b.total_errors == 1

        for run_dir, tag in ((dir_a, "a"), (dir_b, "b")):
            loaded = EvaluationReport.load(os.path.join(run_dir, "report.jsonl"))
            assert loaded.total_items == 4
            assert f"boom-{tag}" in loaded.failures[0].traceback


def test_item_log_degrades_non_json_prediction_to_repr():
    bench = FakeBenchmark(n=1)

    class Opaque:
        def __repr__(self):
            return "<opaque>"

    with tempfile.TemporaryDirectory() as tmp:
        EvaluationPipeline(verbose=False).run(
            lambda ex: Opaque(),
            DataLoader(bench, split="test"),
            lambda pred, label: {"acc": 1.0},
            output_dir=tmp,
        )
        loaded = EvaluationReport.load(os.path.join(tmp, "report.jsonl"))
        assert loaded.items[0].prediction == "<opaque>"
        assert loaded.items[0].metrics == {"acc": 1.0}


def test_scorer_subclass_default_ascore_offloads_sync_score():
    """A Scorer subclass implementing only score() must work on the async path."""
    from evoagentx.benchmark.scorers.base import Scorer

    class AccScorer(Scorer):
        def score(self, prediction, label):
            return {"acc": 1.0 if prediction == label else 0.0}

    bench = FakeBenchmark(n=3)

    async def go():
        return await EvaluationPipeline(verbose=False).arun(
            _double, DataLoader(bench, split="test"), AccScorer())

    report = asyncio.run(go())
    assert report.total_errors == 0
    assert report.aggregate() == {"acc": 1.0}


def test_item_result_ok_flag():
    ok = ItemResult(id=1, metrics={"acc": 1.0})
    bad = ItemResult(id=2, error="[program] boom")
    assert ok.ok is True
    assert bad.ok is False


# --------------------------------------------------------------------------- #
# Bounded concurrency (codex #3) + runner dispatch / to_thread (codex #4)
# --------------------------------------------------------------------------- #
def test_sync_concurrency_is_bounded():
    import threading
    import time

    bench = FakeBenchmark(n=12)
    lock = threading.Lock()
    state = {"cur": 0, "peak": 0}

    def slow(example):
        with lock:
            state["cur"] += 1
            state["peak"] = max(state["peak"], state["cur"])
        time.sleep(0.02)
        with lock:
            state["cur"] -= 1
        return example["x"] * 2

    report = EvaluationPipeline(max_concurrency=3, verbose=False).run(
        slow, DataLoader(bench, split="test"), _scorer)
    assert report.total_items == 12
    assert state["peak"] <= 3  # never more than max_concurrency in flight


def test_async_concurrency_is_bounded():
    bench = FakeBenchmark(n=12)
    state = {"cur": 0, "peak": 0}

    async def slow(example):
        state["cur"] += 1                       # no await between inc and record -> atomic
        state["peak"] = max(state["peak"], state["cur"])
        await asyncio.sleep(0.01)
        state["cur"] -= 1
        return example["x"] * 2

    async def go():
        return await EvaluationPipeline(max_concurrency=4, verbose=False).arun(
            slow, DataLoader(bench, split="test"), _scorer)

    report = asyncio.run(go())
    assert report.total_items == 12
    assert state["peak"] <= 4


def test_async_sync_program_runs_in_threads_not_blocking_loop():
    """A genuinely-sync program on the async path must be offloaded (asyncio.to_thread),
    so calls overlap instead of serializing on the event loop."""
    import threading
    import time

    bench = FakeBenchmark(n=4)
    lock = threading.Lock()
    state = {"cur": 0, "peak": 0}

    def blocking(example):
        with lock:
            state["cur"] += 1
            state["peak"] = max(state["peak"], state["cur"])
        time.sleep(0.05)
        with lock:
            state["cur"] -= 1
        return example["x"] * 2

    async def go():
        return await EvaluationPipeline(max_concurrency=4, verbose=False).arun(
            blocking, DataLoader(bench, split="test"), _scorer)

    report = asyncio.run(go())
    assert report.aggregate() == {"acc": 1.0}
    assert state["peak"] >= 2  # overlapped -> not serialized on the loop


def test_runner_protocol_dispatch():
    """An object exposing execute/async_execute is dispatched to the right method per path."""
    bench = FakeBenchmark(n=3)

    class Runner:
        def __init__(self):
            self.sync_calls = 0
            self.async_calls = 0

        def execute(self, inputs):
            self.sync_calls += 1
            return inputs["x"] * 2

        async def async_execute(self, inputs):
            self.async_calls += 1
            return inputs["x"] * 2

    def input_fn(ex):
        return {"x": ex["x"]}

    sync_runner = Runner()
    rep = EvaluationPipeline(verbose=False).run(sync_runner, DataLoader(bench, split="test"), _scorer, input_fn=input_fn)
    assert rep.aggregate() == {"acc": 1.0}
    assert sync_runner.sync_calls == 3 and sync_runner.async_calls == 0

    async_runner = Runner()

    async def go():
        return await EvaluationPipeline(verbose=False).arun(
            async_runner, DataLoader(bench, split="test"), _scorer, input_fn=input_fn)

    rep2 = asyncio.run(go())
    assert rep2.aggregate() == {"acc": 1.0}
    assert async_runner.async_calls == 3 and async_runner.sync_calls == 0


def test_run_dispatches_execute_on_non_callable_object():
    bench = FakeBenchmark(n=3)

    class Program:
        def __init__(self):
            self.calls = 0

        def execute(self, example):
            self.calls += 1
            return example["x"] * 2

    program = Program()
    report = EvaluationPipeline(verbose=False).run(
        program, DataLoader(bench, split="test"), _scorer)

    assert report.aggregate() == {"acc": 1.0}
    assert program.calls == 3


def test_run_rejects_non_callable_without_execute():
    class Program:
        pass

    with pytest.raises(TypeError, match="program must be callable"):
        EvaluationPipeline(verbose=False).run(
            Program(), DataLoader(FakeBenchmark(n=1), split="test"), _scorer)


def test_arun_dispatches_async_call_object():
    bench = FakeBenchmark(n=3)

    class Program:
        async def __call__(self, example):
            await asyncio.sleep(0)
            return example["x"] * 2

    async def go():
        return await EvaluationPipeline(verbose=False).arun(
            Program(), DataLoader(bench, split="test"), _scorer)

    report = asyncio.run(go())
    assert report.aggregate() == {"acc": 1.0}


def test_arun_prefers_explicit_async_method_over_dynamic_call():
    bench = FakeBenchmark(n=3)

    class Program:
        def __init__(self):
            self.sync_calls = 0
            self.async_calls = 0

        def __call__(self, example):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return self.execute(example)
            return self.async_execute(example)

        def execute(self, example):
            self.sync_calls += 1
            return -1

        async def async_execute(self, example):
            self.async_calls += 1
            return example["x"] * 2

    program = Program()

    async def go():
        return await EvaluationPipeline(verbose=False).arun(
            program, DataLoader(bench, split="test"), _scorer)

    report = asyncio.run(go())
    assert report.aggregate() == {"acc": 1.0}
    assert program.async_calls == 3
    assert program.sync_calls == 0


def test_arun_ignores_sync_method_named_async_execute_and_uses_callable():
    bench = FakeBenchmark(n=3)

    class Program:
        def __init__(self):
            self.async_execute_calls = 0
            self.call_calls = 0

        def async_execute(self, example):
            self.async_execute_calls += 1
            return -1

        def __call__(self, example):
            self.call_calls += 1
            return example["x"] * 2

    program = Program()

    async def go():
        return await EvaluationPipeline(verbose=False).arun(
            program, DataLoader(bench, split="test"), _scorer)

    report = asyncio.run(go())
    assert report.aggregate() == {"acc": 1.0}
    assert program.async_execute_calls == 0
    assert program.call_calls == 3


def test_arun_rejects_non_callable_without_async_method():
    class Program:
        def execute(self, example):
            return example["x"] * 2

    async def go():
        return await EvaluationPipeline(verbose=False).arun(
            Program(), DataLoader(FakeBenchmark(n=1), split="test"), _scorer)

    with pytest.raises(TypeError, match="program must be callable"):
        asyncio.run(go())
