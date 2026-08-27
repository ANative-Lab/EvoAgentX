"""Step 2 acceptance: ``ScoreRequest`` / ``ScoreResult`` protocol upstreamed to ``optimizers/core``."""

import ast
import asyncio
import os
import unittest

from evoagentx.optimizers.core.scoring import (
    CompositeScorer,
    ScoreRequest,
    ScoreResult,
    Scorer,
    as_scorer,
    ObservationKey,
)


def _req(prediction="pred", reference="ref", **kwargs) -> ScoreRequest:
    return ScoreRequest(
        key=("ex-1", 0),
        inputs={"question": "q"},
        prediction=prediction,
        reference=reference,
        trace=None,
        **kwargs,
    )


class AccScorer(Scorer):
    """Fixed metric-key contract: always reports ``acc``, even on a mismatch."""

    METRIC_KEYS = ("acc",)

    @property
    def metric_keys(self):
        return self.METRIC_KEYS

    async def score(self, req: ScoreRequest) -> ScoreResult:
        return ScoreResult(metrics={"acc": 1.0 if req.prediction == req.reference else 0.0})


class BoomScorer(Scorer):
    """A scorer that always fails — the caller is responsible for the ``{key: 0.0}`` fallback."""

    @property
    def metric_keys(self):
        return ("acc",)

    async def score(self, req: ScoreRequest) -> ScoreResult:
        raise RuntimeError("boom")


class FakeJudgeScorer(Scorer):
    """Fake async LLM-judge: genuinely awaits, and produces all three signal channels."""

    @property
    def metric_keys(self):
        return ("judge_score",)

    async def score(self, req: ScoreRequest) -> ScoreResult:
        await asyncio.sleep(0)
        correct = req.prediction == req.reference
        return ScoreResult(
            metrics={"judge_score": 1.0 if correct else 0.0},
            feedback="matches the rubric" if correct else "does not match the rubric",
            details={"rubric_hits": ["exact_match"] if correct else []},
        )


class TestScoreRequestResult(unittest.TestCase):
    def test_score_request_carries_observation_key(self):
        req = _req()
        self.assertEqual(req.key, ("ex-1", 0))
        self.assertIsInstance(req.key, tuple)

    def test_score_result_defaults(self):
        result = ScoreResult(metrics={"acc": 1.0})
        self.assertIsNone(result.feedback)
        self.assertEqual(result.details, {})


class TestScorerContract(unittest.TestCase):
    def test_scorer_is_abstract_async_only(self):
        with self.assertRaises(TypeError):
            Scorer()  # abstract: score() has no implementation

    def test_metric_keys_default_empty_unless_declared(self):
        class UndeclaredScorer(Scorer):
            async def score(self, req: ScoreRequest) -> ScoreResult:
                return ScoreResult(metrics={})

        self.assertEqual(UndeclaredScorer().metric_keys, ())
        self.assertEqual(AccScorer().metric_keys, ("acc",))

    def test_declared_metric_keys_survive_a_failed_score(self):
        """metric_keys is a scorer-level contract independent of any particular call
        outcome — the caller (not the scorer) is what turns a raised exception into the
        declared keys defaulting to 0.0; this test only pins that the contract is still
        readable after ``score`` raises."""
        scorer = BoomScorer()
        with self.assertRaises(RuntimeError):
            asyncio.run(scorer.score(_req()))
        self.assertEqual(scorer.metric_keys, ("acc",))


class TestAsScorer(unittest.TestCase):
    def test_wraps_sync_callable(self):
        def fn(prediction, reference):
            return {"acc": 1.0 if prediction == reference else 0.0}

        scorer = as_scorer(fn, metric_keys=("acc",))
        self.assertIsInstance(scorer, Scorer)
        self.assertEqual(scorer.metric_keys, ("acc",))
        result = asyncio.run(scorer.score(_req("x", "x")))
        self.assertEqual(result.metrics, {"acc": 1.0})

    def test_wraps_async_callable(self):
        async def fn(prediction, reference):
            await asyncio.sleep(0)
            return {"acc": 1.0 if prediction == reference else 0.0}

        scorer = as_scorer(fn)
        result = asyncio.run(scorer.score(_req("x", "y")))
        self.assertEqual(result.metrics, {"acc": 0.0})

    def test_passthrough_for_existing_scorer(self):
        scorer = AccScorer()
        self.assertIs(as_scorer(scorer), scorer)

    def test_rejects_non_callable(self):
        with self.assertRaises(TypeError):
            as_scorer(object())


class TestFakeJudgeScorer(unittest.TestCase):
    """Async judge-style scorer exercising all three signal channels (§0.5)."""

    def test_three_channels_populated(self):
        result = asyncio.run(FakeJudgeScorer().score(_req("42", "42")))
        self.assertEqual(result.metrics, {"judge_score": 1.0})
        self.assertEqual(result.feedback, "matches the rubric")
        self.assertEqual(result.details, {"rubric_hits": ["exact_match"]})

    def test_mismatch(self):
        result = asyncio.run(FakeJudgeScorer().score(_req("42", "7")))
        self.assertEqual(result.metrics, {"judge_score": 0.0})
        self.assertEqual(result.details, {"rubric_hits": []})


class TestCompositeScorer(unittest.TestCase):
    def test_merges_metrics_with_prefixes(self):
        composite = CompositeScorer([AccScorer(), FakeJudgeScorer()], prefixes=["qa", "judge"])
        self.assertEqual(composite.metric_keys, ("qa/acc", "judge/judge_score"))
        result = asyncio.run(composite.score(_req("42", "42")))
        self.assertEqual(result.metrics, {"qa/acc": 1.0, "judge/judge_score": 1.0})

    def test_merges_metrics_without_prefixes_last_writer_wins(self):
        composite = CompositeScorer([AccScorer(), AccScorer()])
        result = asyncio.run(composite.score(_req("42", "42")))
        self.assertEqual(result.metrics, {"acc": 1.0})

    def test_metric_keys_unknown_if_any_child_is_unknown(self):
        plain = as_scorer(lambda p, r: {"x": 1.0})  # no declared metric_keys
        composite = CompositeScorer([AccScorer(), plain])
        self.assertEqual(composite.metric_keys, ())

    def test_feedback_concatenated_and_details_kept_side_by_side(self):
        composite = CompositeScorer([AccScorer(), FakeJudgeScorer()], prefixes=["qa", "judge"])
        result = asyncio.run(composite.score(_req("42", "42")))
        self.assertIn("matches the rubric", result.feedback)
        self.assertIn("judge", result.feedback)
        self.assertEqual(result.details, {"judge": {"rubric_hits": ["exact_match"]}})

    def test_wraps_plain_callables(self):
        composite = CompositeScorer([lambda p, r: {"acc": 1.0 if p == r else 0.0}])
        result = asyncio.run(composite.score(_req("x", "x")))
        self.assertEqual(result.metrics, {"acc": 1.0})


class TestDependencyDirection(unittest.TestCase):
    """``optimizers/`` must stay importable without the host application: the new
    architecture modules (``core`` and onward) never import ``evoagentx.benchmark``.

    ``legacy/`` and ``engine/`` are the pre-existing implementations already marked for
    wholesale deletion at cleanup (§7 step 6) and are exempt until then.
    """

    EXEMPT_DIRS = {"legacy", "engine"}

    def test_no_new_architecture_module_imports_benchmark(self):
        optimizers_root = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "evoagentx", "optimizers"
        )
        optimizers_root = os.path.normpath(optimizers_root)
        offenders = []
        for dirpath, dirnames, filenames in os.walk(optimizers_root):
            rel = os.path.relpath(dirpath, optimizers_root)
            top = rel.split(os.sep)[0] if rel != "." else ""
            if top in self.EXEMPT_DIRS:
                dirnames[:] = []
                continue
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                path = os.path.join(dirpath, filename)
                with open(path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=path)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name == "evoagentx.benchmark" or alias.name.startswith("evoagentx.benchmark."):
                                offenders.append(f"{path}: import {alias.name}")
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        is_absolute_benchmark = module == "evoagentx.benchmark" or module.startswith("evoagentx.benchmark.")
                        is_relative_benchmark = node.level > 0 and (module == "benchmark" or module.startswith("benchmark."))
                        if is_absolute_benchmark or is_relative_benchmark:
                            offenders.append(f"{path}: from {'.' * node.level}{module} import ...")
        self.assertEqual(offenders, [], f"optimizers/ (excluding {self.EXEMPT_DIRS}) must not import evoagentx.benchmark:\n" + "\n".join(offenders))


if __name__ == "__main__":
    unittest.main()
