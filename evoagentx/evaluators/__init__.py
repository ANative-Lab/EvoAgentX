"""Evaluation package.

Submodules are imported lazily so that importing a light leaf (e.g.
``evoagentx.evaluators.result``, used by the optimizer engine) does NOT drag in the
heavy legacy ``Evaluator`` and its ``workflow`` / ``agents`` dependencies.
"""

import importlib
from typing import TYPE_CHECKING

_LAZY = {
    "Evaluator": ".evaluator",
    "EvaluationPipeline": ".pipeline",
    "EvaluationReport": ".result",
    "EvaluationResult": ".result",
    "ItemResult": ".result",
    "DataLoader": ".dataloader",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_path, __name__)
    return getattr(module, name)


def __dir__():
    return sorted(__all__)


if TYPE_CHECKING:  # for static analyzers / IDEs only
    from .evaluator import Evaluator  # noqa: F401
    from .pipeline import EvaluationPipeline  # noqa: F401
    from .result import EvaluationReport, EvaluationResult, ItemResult  # noqa: F401
    from .dataloader import DataLoader  # noqa: F401
