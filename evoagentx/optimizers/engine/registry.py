import re
from typing import Any, Callable, Dict, List
from ...prompts.template import PromptTemplate
import copy
import warnings
_INDEX_RE = re.compile(r'^(.*?)\[(.*?)\]$')

# _PATH_RE = re.compile(r"""
#     ([a-zA-Z_]\w*)       |  # attr name
#     \[(\d+)\]            |  # list index
#     \[['"](.+?)['"]\]       # dict key
# """, re.VERBOSE)

_PATH_RE = re.compile(r"""
    ([a-zA-Z_]\w*)               |  # attr name
    \[\s*(-?\d+)\s*\]            |  # list index (allow negative)
    \[\s*['"]([^'\"]+)['"]\s*\]     # dict key (inside quotes)
""", re.VERBOSE)

class OptimizableField:
    """
    Represents a parameter that can be optimized.

    This class encapsulates a runtime attribute using dynamic getter and setter
    functions. It allows the parameter to be exposed and manipulated by an external
    optimizer. An initial snapshot of the field can be stored and later used to reset
    the field to its original value.
    """

    def __init__(
        self,
        name: str,
        getter: Callable[[], Any],
        setter: Callable[[Any], None]
    ):
        """
        Initialize an OptimizableField instance.

        Parameters
        ----------
        name : str
            The alias used to register the field in the registry.
        getter : Callable[[], Any]
            A function that returns the current value of the field.
        setter : Callable[[Any], None]
            A function that sets a new value to the field.
        """
        self.name = name
        self._get = getter
        self._set = setter
        self._initial_value = None

    def get(self) -> Any:
        """
        Retrieve the current value of the field.

        Returns
        -------
        Any
            The current value of the field.
        """
        return self._get()

    def set(self, value: Any) -> None:
        """
        Update the field with a new value.

        Parameters
        ----------
        value : Any
            The new value to assign to the field.
        """
        self._set(value)
    
    def init_snapshot(self) -> None:
        """
        Capture a snapshot of the current field value.

        This method stores a deep copy of the current field value so that it
        can be restored later using `reset()`.
        """
        current = self.get()
        self._initial_value = safe_deepcopy(current)
    
    def reset(self) -> None:
        """
        Reset the field to its initial value.

        If the current value object defines a `__reset__()` method, it will be
        called to perform the reset. Otherwise, the field is reset to the deep-copied
        initial value stored by `init_snapshot()`.

        Raises
        ------
        ValueError
            If `init_snapshot()` has not been called before `reset()`.
        """
        current = self.get()

        if self._initial_value is None:
            raise ValueError(f"Field '{self.name}' has no snapshot. Call init_snapshot() first.")

        if hasattr(current, "__reset__") and callable(current.__reset__):
            current.__reset__()
        else:
            self.set(safe_deepcopy(self._initial_value))


class ParamRegistry:
    """
    Central registry for all parameters that can be exposed to optimization.

    Allows dynamic binding and tracking of runtime attributes via dot-paths,
    dictionary keys, or list indices. Provides getter/setter access to all
    registered parameters for optimizers.
    """
    def __init__(self) -> None:
        """Initialize an empty registry of optimizable fields."""
        self.fields: Dict[str, OptimizableField] = {}

    def register_field(self, field: OptimizableField):
        """Manually register an OptimizableField with its alias name."""
        field.init_snapshot()
        self.fields[field.name] = field

    def get(self, name: str) -> Any:
        """Retrieve the current value of a registered field by name."""
        return self.fields[name].get()
    
    def get_field(self, name: str) -> OptimizableField:
        """Retrieve the OptimizableField object by name."""
        if name not in self.fields:
            raise ValueError(f"Field '{name}' is not registered.")
        else:
            return self.fields[name]

    def set(self, name: str, value: Any):
        """Set the value of a registered field by name."""
        self.fields[name].set(value)

    def names(self) -> List[str]:
        """Return a list of all registered field names (aliases)."""
        return list(self.fields.keys())
    
    def reset(self):
        """Roll back all registered fields to their initial values."""
        for field in self.fields.values():
            field.reset()
    
    def reset_field(self, name: str):
        """Roll back a registered field to its initial value."""
        self.fields[name].reset()

    def track(
        self,
        root_or_obj: Any,
        path_or_attr: str | None = None,
        *,
        name: str | None = None,
    ):
        """
        Register a parameter to be optimized. Supports both nested paths and direct attributes.

        Parameters:
        - root_or_obj (Any): the base object or container
        - path_or_attr (str): a path like 'prompt.template' or a direct attribute like 'template'
        - name (str | None): optional alias for this parameter

        Supported formats:
        - registry.track(program, "prompt.template")              # nested attribute
        - registry.track(program, "metadata['style']")           # dictionary key
        - registry.track(program, "components[2].prefix")        # list index
        - registry.track(program.prompt, "template")             # direct object + attribute
        - registry.track([
            (program, "prompt.template"),
            (program, "metadata['style']", "style"),
            (program.prompt, "prefix", "prompt_prefix")
          ])                                                    # batch registration
        - registry.track(program, "prompt.template").track(program, "prompt.prefix")  # chained calls
        
        - registry.track(program, "prompt_template_obj")  # register a prompt_template instance

        Returns:
        - self (PromptRegistry): for chaining
        """
        if isinstance(root_or_obj, (list, tuple)):
            # batch mode: track([(obj, path), (obj, path, alias), ...])
            # Example:
            # registry.track([
            #     (program, "prompt.template"),
            #     (program, "metadata['style']", "style"),
            #     (program.prompt, "prefix", "prompt_prefix")
            # ])
            for item in root_or_obj:
                if not isinstance(item, (list, tuple)) or len(item) not in (2, 3):
                    raise ValueError(
                        "Each batch item must be (obj, path) or (obj, path, alias)."
                    )
                if len(item) == 2:
                    self.track(item[0], item[1])
                else:
                    self.track(item[0], item[1], name=item[2])
            return self

        if not isinstance(path_or_attr, str):
            raise TypeError(
                "path_or_attr must be a string for a single track() call."
            )

        if "." in path_or_attr or "[" in path_or_attr:
            return self._track_path(root_or_obj, path_or_attr, name)
        else:
            key = name or path_or_attr

            def getter():
                return getattr(root_or_obj, path_or_attr)

            def setter(v):
                setattr(root_or_obj, path_or_attr, v)

            field = OptimizableField(key, getter, setter)
            if key in self.fields:
                import warnings
                warnings.warn(f"Field '{key}' is already registered. Overwriting.")
            self.register_field(field)
            return self

    def _track_path(self, root: Any, path: str, name: str | None = None):
        """
        Internal helper that registers a nested field (via dot path, index, or key)
        as an OptimizableField by dynamically creating getter and setter functions.

        Parameters:
        - root (Any): the root object to start walking from
        - path (str): dot-separated path supporting list/dict access
        - name (Optional[str]): alias for the parameter (defaults to last path segment)

        Returns:
        - self
        """
        key = name if name is not None else path
        parent, leaf = self._walk(root, path)

        def getter():
            return parent[leaf] if isinstance(parent, (list, dict)) else getattr(parent, leaf)

        def setter(v):
            if isinstance(parent, (list, dict)):
                parent[leaf] = v
            else:
                setattr(parent, leaf, v)

        field = OptimizableField(key, getter, setter)
        self.register_field(field)
        return self
    

    def _walk(self, root, path: str):
        """
        Internal helper to resolve a dot-separated path string into its parent container
        and the leaf attribute/key/index for assignment or retrieval.

        Supports:
        - Nested attributes: e.g. "a.b.c"
        - Dict key access: e.g. "config['key']"
        - List index access: e.g. "layers[0]"

        Parameters:
        - root (Any): root object to walk from
        - path (str): path string to resolve
        - create_missing (bool): unused placeholder for future extensions

        Returns:
        - (parent, leaf): where parent[leaf] or getattr(parent, leaf) is the target
        """
        cur = root
        parts = []
        for match in _PATH_RE.finditer(path):
            attr, idx, key = match.groups()
            if attr:
                parts.append(attr)
            elif idx:
                parts.append(int(idx))
            elif key:
                parts.append(key)

        for part in parts[:-1]:
            if isinstance(part, int):
                cur = cur[part]
            else:
                cur = getattr(cur, part) if hasattr(cur, part) else cur[part]

        leaf = parts[-1]
        parent = cur
        return parent, leaf

    def _walk_old(self, root, path: str):
        """
        Unused Function
        Internal helper to resolve a dot-separated path string into its parent container
        and the leaf attribute/key/index for assignment or retrieval.

        Supports:
        - Nested attributes: e.g. "a.b.c"
        - Dict key access: e.g. "config['key']"
        - List index access: e.g. "layers[0]"

        Parameters:
        - root (Any): root object to walk from
        - path (str): path string to resolve
        - create_missing (bool): unused placeholder for future extensions

        Returns:
        - (parent, leaf): where parent[leaf] or getattr(parent, leaf) is the target
        """
        cur = root
        parts = path.split(".")
        for part in parts[:-1]:
            m = _INDEX_RE.match(part)
            if m:
                attr, idx = m.groups()
                cur = getattr(cur, attr) if attr else cur
                idx = idx.strip()
                if (idx.startswith("'") and idx.endswith("'")) or (idx.startswith('"') and idx.endswith('"')):
                    idx = idx[1:-1]
                elif idx.isdigit():
                    idx = int(idx)
                cur = cur[idx]
            else:
                cur = getattr(cur, part)

        leaf = parts[-1]
        m = _INDEX_RE.match(leaf)
        if m:
            attr, idx = m.groups()
            parent = getattr(cur, attr) if attr else cur
            idx = idx.strip()
            if (idx.startswith("'") and idx.endswith("'")) or (idx.startswith('"') and idx.endswith('"')):
                idx = idx[1:-1]
            elif idx.isdigit():
                idx = int(idx)
            return parent, idx
        return cur, leaf


def safe_deepcopy(obj):
    """
    Safely attempt to deep copy any Python object, with graceful fallback behavior.

    This function performs a standard `copy.deepcopy` when possible. If that fails
    (e.g., due to the presence of uncopyable components such as file handles, threads,
    or custom classes that don't support deep copying), it falls back to a more resilient strategy:

    1. Attempts to create a blank instance of the object's class using `__new__`.
    2. Recursively copies all attributes found in the object's `__dict__`, using:
    - `safe_deepcopy` for deep recursive copy,
    - `copy.copy` as a shallow fallback,
    - or the original reference as a last resort.
    3. If the object has no `__dict__` or cannot be instantiated, returns the original object.

    Parameters:
        obj (Any): The object to be deep copied.

    Returns:
        Any: A deep copy of the input object if possible, or a best-effort fallback copy.
    
    Warnings:
        Issues a `warnings.warn()` message whenever:
        - The deep copy fails and fallback mechanisms are used.
        - An attribute copy fails and falls back to a shallower or direct reference.
        - The class cannot be re-instantiated and the original reference is returned.

    Notes:
        - This function is intended for robust copying in systems where user-defined objects,
        templates, or agents may not support strict deep copying.
        - It is not guaranteed to preserve identity semantics or copy objects with `__slots__`.
        - For critical correctness or mutation isolation, ensure your objects are deepcopy-compatible.

    Example:
        >>> obj = CustomObject()
        >>> obj_copy = safe_deepcopy(obj)
    """
    try:
        return copy.deepcopy(obj)
    except Exception:
        warnings.warn(f"Failed to deepcopy {obj.__class__.__name__}. Falling back to advanced handling.")
        pass  # fallback to custom handling

    try:
        # Try to create a blank instance of the same class
        new_instance = obj.__class__.__new__(obj.__class__)
    except Exception:
        warnings.warn(f"Failed to create a blank instance of {obj.__class__.__name__}. Falling back to reference.")
        return obj  # can't even make a blank instance, fallback to reference

    for attr, value in getattr(obj, "__dict__", {}).items():
        try:
            setattr(new_instance, attr, safe_deepcopy(value))  # recursive copy
        except Exception:
            try:
                warnings.warn(f"Failed to copy {attr} of {obj.__class__.__name__}. Falling back to shallow copy.")
                setattr(new_instance, attr, copy.copy(value))  # shallow copy
            except Exception:
                warnings.warn(f"Failed to copy {attr} of {obj.__class__.__name__}. Falling back to reference.")
                setattr(new_instance, attr, value)  # fallback to reference

    return new_instance
    

class PromptTemplateRegister(ParamRegistry):
    """
    Unused Class
    Enhanced parameter registry that supports directly registering PromptTemplate instances
    or prompt strings as a single optimizable object.
    """

    def track(
        self,
        root_or_obj: Any,
        path_or_attr: str | None = None,
        *,
        name: str | None = None,
    ):
        if isinstance(root_or_obj, (list, tuple)):
            for item in root_or_obj:
                if not isinstance(item, (list, tuple)) or len(item) not in (2, 3):
                    raise ValueError(
                        "Each batch item must be (obj, path) or (obj, path, alias)."
                    )
                if len(item) == 2:
                    self.track(item[0], item[1])
                else:
                    self.track(item[0], item[1], name=item[2])
            return self

        if not isinstance(path_or_attr, str):
            raise TypeError(
                "path_or_attr must be a string for a single track() call."
            )
        
        if '.' in path_or_attr or '[' in path_or_attr:
            return self._track_path(root_or_obj, path_or_attr, name)
        else:
            key = name or path_or_attr

        try:
            value = getattr(root_or_obj, path_or_attr)
        except AttributeError:
            return super().track(root_or_obj, path_or_attr, name=name)

        if isinstance(value, (str, PromptTemplate)):
            # Register the entire prompt or PromptTemplate object
            field = OptimizableField(
                key,
                getter=lambda: getattr(root_or_obj, path_or_attr),
                setter=lambda v: setattr(root_or_obj, path_or_attr, v)
            )
            self.register_field(field)
            return self

        # Fall back to original path-based tracking if not str/template
        return super().track(root_or_obj, path_or_attr, name=name)
