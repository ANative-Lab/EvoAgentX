import ast
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[3] / "evoagentx" / "tools"


def _inputs_literals(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.Assign):
            targets = node.targets
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == "inputs" for t in targets) and isinstance(node.value, ast.Dict):
            yield node.value


def _arrays_without_items(schema, path):
    if not isinstance(schema, dict):
        return
    if schema.get("type") == "array" and "items" not in schema:
        yield path
    for key, value in schema.items():
        if isinstance(value, dict):
            yield from _arrays_without_items(value, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                yield from _arrays_without_items(item, f"{path}.{key}[{index}]")


def test_tool_array_inputs_declare_items():
    missing = []
    for source in sorted(TOOLS_DIR.rglob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for literal in _inputs_literals(tree):
            try:
                inputs = ast.literal_eval(literal)
            except ValueError:
                continue
            for name, schema in inputs.items():
                for path in _arrays_without_items(schema, name):
                    missing.append(f"{source.relative_to(TOOLS_DIR)}:{literal.lineno} {path}")
    assert not missing, "array parameters without 'items' (OpenAI rejects these):\n" + "\n".join(missing)
