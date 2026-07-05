import math
import os
from enum import Enum
from uuid import uuid4
from pydantic import Field, field_validator, model_validator
from jsonschema import validate, ValidationError
from typing import Any, Dict, Iterable, List, Optional, Literal, Union

from ...core.module import BaseModule
# EvaluationResult is defined in evaluators/ (the evaluation layer the optimizer consumes)
# and re-exported here for backward compatibility. The dependency direction is
# optimizers -> evaluators; evaluators never imports optimizers.
from ...evaluators.result import EvaluationResult  # noqa: F401  (re-exported)

class OptimizationUnitType(str, Enum):
    # Generic structured field or scalar parameter that has no more specific type.
    FIELD = "field"

    # Prompt text or prompt template content used by an agent or workflow.
    PROMPT = "prompt"

    # Model selection or model configuration exposed as an optimizable unit.
    MODEL = "model"

    # Memory contents, retrieval policy, or memory-store configuration.
    MEMORY = "memory"

    # File-backed artifact whose semantics are adapter-defined.
    FILE = "file"

    # Source-code artifact; use metadata to mark domain roles such as skills/tools.
    CODE = "code"


class ChangeOperation(str, Enum):
    """Common operation labels for adapter-defined merge semantics.

    The optimizer engine validates that each ``UnitChange.operation`` is allow-listed by
    the target unit, but the adapter's ``merge_changes`` method owns the concrete
    semantics. Adapters are NOT limited to these labels: a ``UnitChange.operation`` may
    also be an arbitrary string (e.g. ``"diff"``, ``"consolidate"``, ``"forget"``,
    ``"refine"``) as long as the target unit allow-lists it and ``merge_changes`` knows
    how to apply it.
    """

    # Replace the current unit value with the provided payload.
    REPLACE = "replace"

    # Recursively update a dict/object-like value with fields from the payload.
    PATCH = "patch"

    # Append one payload item to a list, or append a text block to a string.
    APPEND = "append"

    # Extend a list/string with multiple payload values, or shallow-extend a dict.
    EXTEND = "extend"

    # Delete an item/key/substring identified by the payload.
    DELETE = "delete"

    # Leave the current value unchanged while still recording a proposed operation.
    NOOP = "noop"


STANDARD_CHANGE_OPERATIONS = frozenset(operation.value for operation in ChangeOperation)


def assert_pure_json(value: Any, path: str = "<root>") -> None:
    """
    Validate that ``value`` is a pure, round-trip-safe JSON data tree.

    This is intentionally stricter than ``json.dumps``: it rejects values that *serialize*
    but do not survive a save/load round-trip unchanged, which would silently corrupt a
    resumed run. Only ``str`` / ``int`` / ``float`` / ``bool`` / ``None`` / ``list`` /
    ``dict`` (with ``str`` keys) are allowed; specifically rejected are:

    * non-``str`` dict keys — JSON coerces them to strings, so ``{1: "a"}`` reloads as
      ``{"1": "a"}``;
    * tuples and other non-``list`` sequences — they reload as lists;
    * ``NaN`` / ``Infinity`` floats — not valid JSON, and reload depends on the parser;
    * arbitrary objects — ``BaseModule.save_module`` writes them as ``null``; and
    * any dict carrying a ``"class_name"`` key — ``BaseModule`` revives it into a live
      instance on load (see ``core/module.py``) instead of returning the plain dict, so
      the payload changes type (or fails) on resume.

    Args:
        value: The payload value to validate.
        path: Human-readable location used in error messages.

    Raises:
        ValueError: If ``value`` is not a pure, round-trip-safe JSON data tree.
    """
    # NB: bool is a subclass of int; both are valid JSON scalars.
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(
                f"{path}: NaN/Infinity floats are not valid JSON and may not round-trip on resume"
            )
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"{path}: dict key {key!r} is not a str; JSON coerces non-str keys to "
                    f"strings, so the value would change on resume"
                )
            if key == "class_name":
                raise ValueError(
                    f"{path}.{key}: payload contains a 'class_name' key, which BaseModule "
                    f"revives into a live instance on resume instead of a plain dict"
                )
            assert_pure_json(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for idx, item in enumerate(value):
            assert_pure_json(item, f"{path}[{idx}]")
        return
    raise ValueError(
        f"{path}: value of type {type(value).__name__!r} cannot be persisted; only "
        f"str/int/float/bool/None/list/dict are allowed (tuples reload as lists, and "
        f"non-serializable objects are written as null)"
    )


class OptimizationUnit(BaseModule):
    """
    Declare an optimization unit that can be modified by optimizers.

    Example 1: Register a model-name unit whose value must be one of a fixed set
    ```
    unit = OptimizationUnit(
        name="assistant_model",
        unit_type=OptimizationUnitType.MODEL,
        json_schema={
            "type": "string",
            "enum": ["gpt-3.5", "gpt-4"]
        }
    )
    ```

    Example 2: Register a prompt unit that supports replacing the whole prompt or appending an additional instruction block:
    ```
    unit = OptimizationUnit(
        name="system_prompt",
        unit_type=OptimizationUnitType.PROMPT,
        json_schema={
            "type": "string",
            "description": "The full prompt text"
        },
        allowed_operations=[ChangeOperation.REPLACE, ChangeOperation.APPEND],
        operation_schemas={
            "append": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Prompt text to append."},
                    "position_hint": {"type": "string", "enum": ["start", "end"], "description": "Optional hint for where to append the text."}
                },
            "required": ["text"],
            "additionalProperties": False
        }
      }
    )
    ```
    """
    name: str = Field(description="Name of the optimization unit; also serves as the default uid")
    unit_type: OptimizationUnitType = Field(description="Type of the optimization unit")

    uid: Optional[str] = Field(default="", description="Stable unique ID for this unit across adapter reconstructions; defaults to name if not set explicitly")
    json_schema: Optional[dict] = Field(default=None, description="Optional schema for the optimization unit, used to validate the parameters associated with this unit")
    allowed_operations: Optional[List[Union[ChangeOperation, str]]] = Field(default_factory=lambda: [ChangeOperation.REPLACE], description="Operations this unit accepts in UnitChange. May contain standard ChangeOperation members and/or adapter-defined operation strings (e.g. 'diff', 'consolidate'). Defaults to replacement updates for backward compatibility.")
    operation_schemas: Optional[Dict[str, dict]] = Field(default_factory=dict, description="Optional per-operation JSON schemas for UnitChange.value payloads, keyed by operation name (standard or adapter-defined). For replace operations, json_schema is used when no operation-specific schema is provided.")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Adapter-defined unit metadata used by optimizers to understand semantics such as file path, prompt role, retrieval policy, constraints, or dependencies.")

    @model_validator(mode="before")
    @classmethod
    def _default_uid_to_name(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("uid"):
            data = {**data, "uid": data.get("name", "")}
        return data

    def validate_value(self, value: Any, context: str = "value") -> None:
        """
        Validate a complete optimization-unit value against this unit's json_schema.

        Args:
            value: The full unit value to validate.
            context: Human-readable validation context for error messages.

        Raises:
            ValueError: If the value does not conform to this unit's json_schema.
        """
        if self.json_schema is None:
            return
        try:
            validate(instance=value, schema=self.json_schema)
        except ValidationError as e:
            raise ValueError(
                f"Value for unit '{self.name}' (uid={self.uid}) in {context} "
                f"does not conform to json_schema: {e.message}"
            )


class UnitChange(BaseModule):
    uid: str = Field(description="Unique ID of the optimization unit to change")
    value: Optional[Any] = Field(default=None, description="Payload for this change. For operation='replace', this is the new unit value. Other operations may interpret it as a patch, appended item, deletion selector, etc. Defaults to None for payload-less operations such as NOOP or adapter-defined operations that carry no value (e.g. 'consolidate'); a missing value round-trips through save/resume instead of failing validation.")
    
    old_value: Optional[Any] = Field(default=None, description="Previous value of the optimization unit before change (optional)")
    operation: Optional[Union[ChangeOperation, str]] = Field(default=ChangeOperation.REPLACE, description="Operation to apply to the target unit. May be a standard ChangeOperation or an adapter-defined operation string. Adapters define the concrete merge semantics for non-replace and custom operations in merge_changes.")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optimizer-defined metadata for provenance, rationale, confidence, or algorithm-specific bookkeeping.")

    @staticmethod
    def validate_value(value: Any, unit: OptimizationUnit, operation: Union[ChangeOperation, str] = ChangeOperation.REPLACE) -> None:
        """
        Validate a value against the json_schema of the given OptimizationUnit.

        Args:
            value: The candidate value to validate.
            unit: The OptimizationUnit whose json_schema defines the validation rules.
            operation: The change operation whose payload is being validated.

        Raises:
            ValueError: If the value does not conform to the unit's json_schema.
        """
        if operation not in unit.allowed_operations:
            raise ValueError(
                f"Operation '{operation}' is not allowed for unit '{unit.name}' "
                f"(uid={unit.uid}). Allowed operations: {unit.allowed_operations}"
            )

        operation_key = operation.value if isinstance(operation, ChangeOperation) else str(operation)
        schema = unit.operation_schemas.get(operation_key)
        if schema is None and operation_key == ChangeOperation.REPLACE.value:
            unit.validate_value(value, context=f"operation '{operation_key}'")
            return

        if schema is None:
            return # No schema means no validation needed
        
        try:
            validate(instance=value, schema=schema)
        except ValidationError as e:
            raise ValueError(
                f"Value for unit '{unit.name} (uid={unit.uid}) "
                f"does not conform to the schema for operation '{operation}': {e.message}"
            )
            
    @classmethod
    def create(
        cls,
        unit: OptimizationUnit,
        new_value: Any = None,
        old_value: Optional[Any] = None,
        operation: Union[ChangeOperation, str] = ChangeOperation.REPLACE,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "UnitChange":
        """
        Preferred constructor for optimizer-side change construction.
        Validates the value against the unit's json_schema before constructing the instance, ensuring changes applied to a unit are always valid. 

        Args:
            unit: The OptimizationUnit being changed.
            new_value: The new value to apply to the unit.
            old_value: The previous value of the unit before change (optional).
            operation: The operation to apply to the unit.
            metadata: Optional optimizer-defined metadata for this change.
        
        Returns:
            A UnitChange instance with the provided values if validation passes.
        """
        cls.validate_value(new_value, unit, operation=operation)
        return cls(uid=unit.uid, value=new_value, old_value=old_value, operation=operation, metadata=metadata or {})


class OptimizationProposal(BaseModule):
    source_snapshot_id: str = Field(description="The snapshot_id of the base snapshot that the proposed changes will be applied on")
    changes: List[UnitChange] = Field(description="List of proposed changes to apply on the base snapshot for the next trial")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optimizer-defined metadata for the proposal, such as parent IDs, generation number, sampler params, or rationale.")


ValidationStatus = Literal["passed", "failed", "skipped"]


class ValidationResult(BaseModule):
    """Result from one adapter-defined validation step before a trial is evaluated."""
    validator: str = Field(description="Name of the validation step")
    status: ValidationStatus = Field(description="Validation status")
    message: Optional[str] = Field(default=None, description="Human-readable validation message")
    details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Structured validation details")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Validator-owned metadata")

    @property
    def ok(self) -> bool:
        return self.status in ("passed", "skipped")


class TrialRecord(BaseModule):
    """Data model for recording the results of a single optimization trial."""
    trial_id: int = Field(description="Unique identifier for the trial")
    changes: List[UnitChange] = Field(description="List of changes applied in this trial")
    source_snapshot_id: str = Field(description="The snapshot_id of the base snapshot that the proposed changes were applied on in this trial")
    status: Literal["completed", "failed"] = Field(description="Status of the trial")

    snapshot_id: Optional[str] = Field(default=None, description="The snapshot_id of the program state after applying the changes in this trial")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="Evaluation metrics collected for this trial")
    traces: Optional[List[Any]] = Field(default_factory=list, description="Execution traces or trajectory data collected during evaluation.")
    artifacts: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Artifacts produced or consumed during evaluation.")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Trial-level metadata, including structured evaluator metadata or optimizer bookkeeping.")
    validation_results: List[ValidationResult] = Field(default_factory=list, description="Validation results collected before evaluation.")
    trial_dir: Optional[str] = Field(default=None, description="Persistent per-trial directory under <save_dir>/trials where evaluation results are archived, if the run has a save_dir.")
    workspace_dir: Optional[str] = Field(default=None, description="Trial workspace directory used to isolate file-backed artifacts, if any.")
    error: Optional[str] = Field(default=None, description="Error message if the trial failed")


class SnapShot(BaseModule):
    snapshot_id: str = Field(default_factory=lambda: uuid4().hex[:8], description="Unique identifier for the snapshot")
    unit_values: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Mapping from unit uid to its current value; covers only the optimizable units declared by the adapter. All values must be JSON-serializable (str, int, float, bool, None, list, dict) so snapshots can be persisted and resumed without data loss.")
    program_config: Optional[Dict[str, Any]] = Field(default=None, description="Complete adapter-defined configuration required to fully reconstruct the underlying program (e.g. model settings, pipeline paths, non-optimizable params). Opaque to the framework; populated and consumed exclusively by the adapter's take_snapshot / from_snapshot. Set to None if unit_values alone is sufficient for reconstruction. Must be JSON-serializable (str, int, float, bool, None, list, dict) — non-serializable objects will be rejected at construction time.")

    @field_validator("unit_values", mode="before")
    @classmethod
    def _require_unit_values_json_safe(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            return v  # let pydantic's type check handle non-dict
        assert_pure_json(v, "unit_values")
        return v

    @field_validator("program_config", mode="before")
    @classmethod
    def _require_program_config_json_safe(cls, v: Any) -> Any:
        if v is None:
            return v
        assert_pure_json(v, "program_config")
        return v


BASELINE_TRIAL_ID = 0


def _assert_payload_fields_pure_json(obj: Any, path: str) -> None:
    """
    Walk a run-state object graph and require every opaque payload field to be pure JSON.

    Structural composition in this engine always flows through ``BaseModule`` instances and
    typed lists of them (snapshots, trial records, changes, validation results); the only
    *raw* dicts/scalars reachable are the ``Any``-typed user payloads (optimizer_state,
    metrics, traces, artifacts, metadata, unit_values, program_config, change values,
    validation details, adapter fingerprint, ...). So recursing through ``BaseModule``
    fields and lists, and handing everything else to :func:`assert_pure_json`, validates
    exactly those payloads — and only those — without enumerating each field by hand.

    The framework-level ``"class_name"`` keys that drive revival live on serialized
    ``BaseModule`` dicts, which this walk reaches as model *fields* (never as raw dicts),
    so they are not mistaken for payload ``class_name`` keys.
    """
    if isinstance(obj, BaseModule):
        for field_name in type(obj).model_fields:
            _assert_payload_fields_pure_json(getattr(obj, field_name, None), f"{path}.{field_name}")
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            _assert_payload_fields_pure_json(item, f"{path}[{idx}]")
    else:
        assert_pure_json(obj, path)


class OptimizationRunState(BaseModule):

    snapshots: List[SnapShot] = Field(default_factory=list, description="List of snapshots for all completed trials in the optimization run")
    trial_records: List[TrialRecord] = Field(default_factory=list, description="List of records for all completed trials in the optimization run")
    best_snapshot_id: Optional[str] = Field(default=None, description="The snapshot_id of the best snapshot found so far in the optimization run, according to the defined objective and evaluation metrics")
    best_metrics: Optional[Dict[str, Any]] = Field(default=None, description="The evaluation metrics associated with best_snapshot_id; None until the baseline has been evaluated")
    optimizer_state: Dict[str, Any] = Field(default_factory=dict, description="Optimizer-owned persistent state, such as sampler state, populations, archives, generated candidate pools, or minibatch cursors.")
    adapter_fingerprint: Optional[Dict[str, Any]] = Field(default=None, description="Adapter compatibility fingerprint captured at run initialization.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Run-level metadata.")

    current_step: int = Field(default=0, description="Number of proposals attempted so far (not successful evaluations). Incremented by len(proposals) each batch, including failed trials.")
    save_dir: Optional[str] = Field(default="./", description="Directory to save optimization state and results")

    def get_snapshot_by_id(self, snapshot_id: str) -> Optional[SnapShot]:
        """
        Look up a snapshot by its snapshot_id.

        Args:
            snapshot_id: The unique identifier of the target snapshot.

        Returns:
            The matching SnapShot, or None if no snapshot with that ID exists.
        """
        for snapshot in self.snapshots:
            if snapshot.snapshot_id == snapshot_id:
                return snapshot
        return None

    def prune_snapshots(self, keep_ids: Iterable[str]) -> int:
        """
        Drop stored snapshots whose snapshot_id is not in `keep_ids`.

        Bounds the memory/disk footprint of long online-accumulation runs, where every
        trial would otherwise retain a full copy of the (growing) program state. Trial
        records are left untouched; a record whose snapshot was pruned simply has no
        materializable snapshot (`get_snapshot_by_id` returns None for it). Callers must
        include every snapshot they still need (best, baseline, branch heads, the sources
        of any future proposals) in `keep_ids`.

        Args:
            keep_ids: snapshot_ids to retain; all others are discarded.

        Returns:
            The number of snapshots removed.
        """
        keep = set(keep_ids)
        before = len(self.snapshots)
        self.snapshots = [snapshot for snapshot in self.snapshots if snapshot.snapshot_id in keep]
        return before - len(self.snapshots)

    def get_trial_record_by_id(self, trial_id: int) -> Optional[TrialRecord]:
        """
        Look up a TrialRecord by its trial_id.

        Args:
            trial_id: The unique identifier of the target trial.

        Returns:
            The matching TrialRecord, or None if no record with that ID exists.
        """
        for record in self.trial_records:
            if record.trial_id == trial_id:
                return record
        return None

    def get_baseline_record(self) -> Optional[TrialRecord]:
        """
        Return the baseline TrialRecord (trial_id == BASELINE_TRIAL_ID), or None if not present.
        """
        for record in self.trial_records:
            if record.trial_id == BASELINE_TRIAL_ID:
                return record
        return None

    @staticmethod
    def load_state(path: str) -> "OptimizationRunState":
        """
        Deserialize and return an OptimizationRunState from a previously saved file.

        Used by `Optimizer.optimize` when `resume_from` is provided, so that
        already-completed trials are skipped and the run continues from where it
        left off.

        Args:
            path: File path (or directory) written by a prior `save_state` call.

        Returns:
            The restored OptimizationRunState, including all past snapshots,
            trial records, best_snapshot_id, and current_step.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exist, cannot load optimization state")

        if os.path.isdir(path):
            path = os.path.join(path, "optimization_state.json")
        return OptimizationRunState.from_file(path=path)

    def _assert_persistable(self) -> None:
        """
        Reject a run state that would not survive a save/load round-trip, before it is written.

        ``SnapShot.unit_values`` / ``program_config`` are validated at construction, but the
        remaining opaque ``Any`` payloads — ``optimizer_state`` and each ``TrialRecord``'s
        ``metrics`` / ``traces`` / ``artifacts`` / ``metadata`` (plus change values,
        validation details, and the adapter fingerprint) — are mutated in place after
        construction, so they need a checkpoint-time guard. ``BaseModule.save_module`` writes
        non-serializable objects as ``null`` instead of failing, ``BaseModule`` revives any
        dict carrying a ``"class_name"`` key into a live instance on load, and plain
        ``json.dumps`` would silently turn tuples into lists and non-``str`` keys into
        strings — each a way for a resumed run to drift from the saved one. This validates
        every payload as a pure JSON data tree and raises with the offending path otherwise.

        Raises:
            ValueError: If any payload is not a pure, round-trip-safe JSON data tree.
        """
        try:
            _assert_payload_fields_pure_json(self, "OptimizationRunState")
        except ValueError as exc:
            raise ValueError(
                f"OptimizationRunState cannot be checkpointed without data loss: {exc}. "
                f"Keep optimizer_state (reduce live objects via serialize_optimizer_state()) "
                f"and each TrialRecord's metrics/traces/artifacts/metadata as pure JSON data: "
                f"str/int/float (no NaN/Infinity)/bool/None/list/dict with str keys, and no "
                f"'class_name' keys."
            ) from exc

    def save_state(self) -> str:
        """
        Serialize the current OptimizationRunState to disk under `self.save_dir`.

        Called after every completed trial so the run can be resumed if interrupted.

        Returns:
            The file path where the state was written.
        """
        self._assert_persistable()
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, "optimization_state.json")
        self.save_module(path=path)
        return path
