"""EvaluationPipeline: task-agnostic orchestration.

Wires together a few small pieces and nothing more:
    - a ``DataLoader``        -> yields (ids, batch_data, batch_labels)
    - a ``program``           -> a (sync or async) callable under evaluation
    - a ``Scorer`` / callable -> (prediction, label) -> metrics dict

Adapting the black-box ``program`` to the benchmark's I/O is done with three optional
hooks, so the user is never forced to wrap the program in a bespoke lambda:
    - ``input_fn(example)            -> program input``     (a.k.a. collate)
    - ``output_fn(raw_output)        -> prediction``        (a.k.a. postprocess)
    - ``trace_fn(example, raw_output)-> trace | (trace, artifacts)``  (opt-in, scenario 3)

The pipeline produces an :class:`EvaluationReport` — the per-item run log. Aggregation and
conversion to the optimizer-facing :class:`EvaluationResult` are explicit and live on the
report (``aggregate`` / ``to_result``), so the same pipeline serves both a plain evaluation
and an optimizer ``evaluate_fn``::

    report = pipeline.run(program, dataloader, scorer)        # plain eval
    print(report.aggregate())

    evaluate_fn = lambda program: pipeline.arun(program, dataloader, scorer).to_result()

Persistence: pass ``output_dir`` to stream results to disk as they complete:
    - ``<output_dir>/report.jsonl``  : append-only JSONL, one serialized ItemResult per
      line. It is both the run log (reload with ``EvaluationReport.load``) and the
      resume checkpoint (``resume=True`` skips already-logged ids and rebuilds their
      items losslessly). Each failed item carries its full ``traceback`` on this line.

Responsibilities: bounded concurrency, per-item error capture (skip & record), progress,
optional streaming persistence/resume, and report assembly. It does NOT know about
WorkFlow/ActionGraph or about QA/code/math — those live in the program and in scorers
respectively.

Entry points:
    - ``run``  : synchronous (bounded ThreadPoolExecutor window; assumes a sync program)
    - ``arun`` : asynchronous (bounded worker pool; program may be a runner / async / sync)
"""

import asyncio
import contextvars
import inspect
import json
import os
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Callable, List, Optional, Tuple

from ..core.logging import logger
from ..benchmark.scorers.base import Scorer, as_scorer
from .dataloader import DataLoader
from .result import EvaluationReport, ItemResult

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


# A single unit of work handed to a worker: parallel lists of (ids, datas, labels).
# Length 1 in "single" mode; the whole batch in "batch" mode.
Unit = Tuple[List[Any], List[Any], List[Any]]


class _ItemLog:
    """Append-only JSONL log of ItemResults; doubles as the resume checkpoint.

    ``add`` buffers items and appends them to the file every ``every`` completions, so
    a long run never rewrites earlier records. ``load`` rebuilds the full ItemResults
    of a previous run, which makes resume lossless (predictions/traces included, modulo
    the repr degradation of non-JSON payloads).
    """

    def __init__(self, path: Optional[str], every: int):
        self.path = path
        self.every = max(1, every)
        self.completed = 0
        self.pending: List[ItemResult] = []

    def load(self) -> List[ItemResult]:
        if not self.path or not os.path.exists(self.path):
            return []
        items: List[ItemResult] = []
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(ItemResult.from_json_dict(json.loads(line)))
        return items

    def add(self, item: ItemResult) -> None:
        if not self.path:
            return
        self.pending.append(item)
        self.completed += 1
        if self.completed % self.every == 0:
            self.flush()

    def flush(self) -> None:
        if not self.path or not self.pending:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a") as f:
            for item in self.pending:
                f.write(json.dumps(item.to_json_dict()) + "\n")
        self.pending = []


class EvaluationPipeline:

    def __init__(
        self,
        max_concurrency: int = 1,
        verbose: bool = True,
        save_predictions: bool = True,
    ):
        # NOTE: instances hold no per-run mutable state, so one pipeline can safely be
        # shared across concurrent run()/arun() calls (e.g. concurrent optimizer trials).
        self.max_concurrency = max(1, max_concurrency)
        self.verbose = verbose
        self.save_predictions = save_predictions

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #
    def _setup(
        self,
        scorer,
        dataloader: DataLoader,
        output_dir: Optional[str],
        flush_every: int,
        resume: bool,
    ) -> Tuple[Scorer, _ItemLog, EvaluationReport]:
        scorer = as_scorer(scorer)
        log_path = os.path.join(output_dir, "report.jsonl") if output_dir else None
        log = _ItemLog(log_path, flush_every)
        # Carry the scorer's declared metric keys onto the report so aggregation knows the
        # objective vector even if every item fails (see EvaluationReport.aggregate).
        report = EvaluationReport(metric_keys=tuple(scorer.metric_keys))
        if resume:
            prior = log.load()
            report.items.extend(prior)
            dataloader.skip_ids = set(dataloader.skip_ids) | {item.id for item in prior}
            if prior:
                logger.info(f"Resuming: skipping {len(prior)} already-processed items.")
        elif log_path and os.path.exists(log_path):
            # Fresh run (resume=False): drop the previous run's item log in the same
            # output_dir so the log reflects exactly this run. Warn first, since this
            # discards data — pass resume=True to continue from it instead.
            logger.warning(
                f"Overwriting existing item log at {log_path} (resume=False). "
                f"Pass resume=True to continue from it instead."
            )
            os.remove(log_path)
        return scorer, log, report

    def _units(self, dataloader: DataLoader, program_mode: str) -> List[Unit]:
        if program_mode == "batch":
            return [(ids, data, labels) for ids, data, labels in dataloader]
        if program_mode == "single":
            return [
                ([example_id], [data], [label])
                for ids, batch_data, batch_labels in dataloader
                for example_id, data, label in zip(ids, batch_data, batch_labels)
            ]
        raise ValueError(f"program_mode must be 'single' or 'batch', got {program_mode!r}.")

    def _progress(self, total: Optional[int]):
        if self.verbose and tqdm is not None:
            return tqdm(total=total, desc="Evaluating")
        return None

    @staticmethod
    def _apply(fn: Optional[Callable], value: Any) -> Any:
        return fn(value) if fn else value
    
    def _sync_caller(self, program: Any) -> Callable[[Any], Any]:
        """Resolve the sync call: ``program_input -> output``."""
        method = getattr(program, "execute", None)
        if callable(method):
            return method
        if not callable(program):
            raise TypeError("program must be callable or expose a method named execute.")
        return program

    def _async_caller(self, program: Any) -> Callable[[Any], Any]:
        """Resolve an async call ``program_input -> awaitable output`` that never blocks the loop.

        - explicit async method -> await it
        - ``async def``         -> await it
        - plain sync callable   -> run in a worker thread
        """
        # try to find an explicit async execute method first 
        method = getattr(program, "async_execute", None)
        if callable(method) and inspect.iscoroutinefunction(method):
            async def call(value, method=method):
                return await method(value)
            return call

        if inspect.iscoroutinefunction(program) or inspect.iscoroutinefunction(getattr(program, "__call__", None)):
            async def call(value):
                return await program(value)
            return call

        if not callable(program):
            raise TypeError("program must be callable or expose an async method named async_execute.")

        async def call(value):
            output = await asyncio.to_thread(program, value)
            if inspect.isawaitable(output):  # a sync callable that returned a coroutine
                output = await output
            return output
        return call

    def _extract_trace(self, trace_fn, data, raw) -> Tuple[Any, Any]:
        if not trace_fn:
            return None, None
        try:
            result = trace_fn(data, raw)
        except Exception as exc:
            logger.warning(f"trace_fn failed; dropping trace. ({exc})")
            return None, None
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, None

    def _error_item(self, example_id, stage: str, tb: str, t0: float, prediction: Any = None) -> ItemResult:
        # The full traceback rides on the item and is persisted with it in the JSONL log.
        message = f"[{stage}] {tb.strip().splitlines()[-1] if tb else ''}"
        logger.warning(f"Item {example_id} failed in {stage}; skipped. ({message})")
        return ItemResult(id=example_id, prediction=prediction, error=message, traceback=tb, elapsed_time=time.time() - t0)

    def _collect(self, report: EvaluationReport, log: _ItemLog, item: ItemResult) -> None:
        report.items.append(item)
        log.add(item)

    # ------------------------------------------------------------------ #
    # Synchronous path
    # ------------------------------------------------------------------ #
    def run(
        self,
        program: Callable[[Any], Any],
        dataloader: DataLoader,
        scorer,
        input_fn: Optional[Callable[[Any], Any]] = None,
        output_fn: Optional[Callable[[Any], Any]] = None,
        trace_fn: Optional[Callable[[Any, Any], Any]] = None,
        program_mode: str = "single",
        output_dir: Optional[str] = None,
        flush_every: int = 50,
        resume: bool = True,
    ) -> EvaluationReport:
        scorer, log, report = self._setup(scorer, dataloader, output_dir, flush_every, resume)
        units = self._units(dataloader, program_mode)
        call = self._sync_caller(program)
        start = time.time()
        bar = self._progress(len(units))
        units_iter = iter(units)

        def submit_next(executor, futures) -> bool:
            unit = next(units_iter, None)
            if unit is None:
                return False
            ids, datas, labels = unit
            ctx = contextvars.copy_context()
            fut = executor.submit(
                ctx.run,
                self._process_unit,
                call, scorer, input_fn, output_fn, trace_fn, program_mode, ids, datas, labels,
            )
            futures.add(fut)
            return True

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            # Keep at most ``max_concurrency`` futures outstanding: prime the window, then
            # refill one for each completed future instead of submitting everything up front.
            futures: set = set()
            for _ in range(self.max_concurrency):
                if not submit_next(executor, futures):
                    break
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    for item in fut.result():
                        self._collect(report, log, item)
                    if bar:
                        bar.set_postfix(errors=report.total_errors)
                        bar.update(1)
                    submit_next(executor, futures)

        log.flush()
        if bar:
            bar.close()
        report.elapsed_time = time.time() - start
        return report

    def _process_unit(self, call, scorer, input_fn, output_fn, trace_fn, program_mode, ids, datas, labels) -> List[ItemResult]:
        if program_mode == "batch":
            return self._process_batch(call, scorer, input_fn, output_fn, trace_fn, ids, datas, labels)
        return [self._process_one(call, scorer, input_fn, output_fn, trace_fn, ids[0], datas[0], labels[0])]

    def _process_one(self, call, scorer: Scorer, input_fn, output_fn, trace_fn, example_id, data, label) -> ItemResult:
        t0 = time.time()
        try:
            raw = call(self._apply(input_fn, data))
        except Exception:
            return self._error_item(example_id, "program", traceback.format_exc(), t0)
        return self._score_one(scorer, output_fn, trace_fn, example_id, data, raw, label, t0)

    def _process_batch(self, call, scorer: Scorer, input_fn, output_fn, trace_fn, ids, datas, labels) -> List[ItemResult]:
        t0 = time.time()
        try:
            outputs = call([self._apply(input_fn, d) for d in datas])
        except Exception:
            tb = traceback.format_exc()
            return [self._error_item(i, "program(batch)", tb, t0) for i in ids]
        invalid = self._validate_batch_outputs(outputs, len(ids))
        if invalid is not None:
            return [self._error_item(i, "program(batch)", invalid, t0) for i in ids]
        return [
            self._score_one(scorer, output_fn, trace_fn, i, d, raw, label, t0)
            for i, d, raw, label in zip(ids, datas, outputs, labels)
        ]

    def _score_one(self, scorer: Scorer, output_fn, trace_fn, example_id, data, raw, label, t0) -> ItemResult:
        try:
            prediction = self._apply(output_fn, raw)
        except Exception:
            return self._error_item(example_id, "output_fn", traceback.format_exc(), t0)
        try:
            metrics = scorer.score(prediction, label)
        except Exception:
            return self._error_item(
                example_id, "scorer", traceback.format_exc(), t0,
                prediction=prediction if self.save_predictions else None,
            )
        trace, artifacts = self._extract_trace(trace_fn, data, raw)
        return ItemResult(
            id=example_id,
            prediction=prediction if self.save_predictions else None,
            metrics=metrics,
            trace=trace,
            artifacts=artifacts,
            elapsed_time=time.time() - t0,
        )

    @staticmethod
    def _validate_batch_outputs(outputs, n: int) -> Optional[str]:
        """Return an error message if a batch program's outputs are not a length-n sequence."""
        if not isinstance(outputs, (list, tuple)):
            return f"batch program must return a list/tuple, got {type(outputs).__name__}."
        if len(outputs) != n:
            return f"batch program returned {len(outputs)} outputs; expected {n} (one per example)."
        return None

    # ------------------------------------------------------------------ #
    # Asynchronous path
    # ------------------------------------------------------------------ #
    async def arun(
        self,
        program: Callable[[Any], Any],
        dataloader: DataLoader,
        scorer,
        input_fn: Optional[Callable[[Any], Any]] = None,
        output_fn: Optional[Callable[[Any], Any]] = None,
        trace_fn: Optional[Callable[[Any, Any], Any]] = None,
        program_mode: str = "single",
        output_dir: Optional[str] = None,
        flush_every: int = 50,
        resume: bool = True,
    ) -> EvaluationReport:
        scorer, log, report = self._setup(scorer, dataloader, output_dir, flush_every, resume)
        units = self._units(dataloader, program_mode)
        acall = self._async_caller(program)
        start = time.time()
        bar = self._progress(len(units))
        lock = asyncio.Lock()
        units_iter = iter(units)

        async def worker():
            # A fixed pool of these runs; each pulls the next unit until the iterator is
            # drained. ``next`` has no await, so it is atomic across cooperatively-scheduled
            # coroutines — no extra lock needed to hand out work.
            while True:
                unit = next(units_iter, None)
                if unit is None:
                    return
                ids, datas, labels = unit
                items = await self._aprocess_unit(
                    acall, scorer, input_fn, output_fn, trace_fn, program_mode, ids, datas, labels,
                )
                async with lock:
                    for item in items:
                        self._collect(report, log, item)
                    if bar:
                        bar.set_postfix(errors=report.total_errors)
                        bar.update(1)

        num_workers = min(self.max_concurrency, len(units)) or 0
        await asyncio.gather(*(worker() for _ in range(num_workers)))

        log.flush()
        if bar:
            bar.close()
        report.elapsed_time = time.time() - start
        return report

    async def _aprocess_unit(self, acall, scorer, input_fn, output_fn, trace_fn, program_mode, ids, datas, labels) -> List[ItemResult]:
        if program_mode == "batch":
            return await self._aprocess_batch(acall, scorer, input_fn, output_fn, trace_fn, ids, datas, labels)
        return [await self._aprocess_one(acall, scorer, input_fn, output_fn, trace_fn, ids[0], datas[0], labels[0])]

    async def _aprocess_one(self, acall, scorer: Scorer, input_fn, output_fn, trace_fn, example_id, data, label) -> ItemResult:
        t0 = time.time()
        try:
            raw = await acall(self._apply(input_fn, data))
        except Exception:
            return self._error_item(example_id, "program", traceback.format_exc(), t0)
        return await self._ascore_one(scorer, output_fn, trace_fn, example_id, data, raw, label, t0)

    async def _aprocess_batch(self, acall, scorer: Scorer, input_fn, output_fn, trace_fn, ids, datas, labels) -> List[ItemResult]:
        t0 = time.time()
        try:
            outputs = await acall([self._apply(input_fn, d) for d in datas])
        except Exception:
            tb = traceback.format_exc()
            return [self._error_item(i, "program(batch)", tb, t0) for i in ids]
        invalid = self._validate_batch_outputs(outputs, len(ids))
        if invalid is not None:
            return [self._error_item(i, "program(batch)", invalid, t0) for i in ids]
        return [
            await self._ascore_one(scorer, output_fn, trace_fn, i, d, raw, label, t0)
            for i, d, raw, label in zip(ids, datas, outputs, labels)
        ]

    async def _ascore_one(self, scorer: Scorer, output_fn, trace_fn, example_id, data, raw, label, t0) -> ItemResult:
        try:
            prediction = self._apply(output_fn, raw)
        except Exception:
            return self._error_item(example_id, "output_fn", traceback.format_exc(), t0)
        try:
            metrics = await scorer.ascore(prediction, label)
        except Exception:
            return self._error_item(
                example_id, "scorer", traceback.format_exc(), t0,
                prediction=prediction if self.save_predictions else None,
            )
        trace, artifacts = self._extract_trace(trace_fn, data, raw)
        return ItemResult(
            id=example_id,
            prediction=prediction if self.save_predictions else None,
            metrics=metrics,
            trace=trace,
            artifacts=artifacts,
            elapsed_time=time.time() - t0,
        )
