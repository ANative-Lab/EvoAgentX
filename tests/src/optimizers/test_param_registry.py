from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from evoagentx.optimizers.engine.registry import ParamRegistry, PromptTemplateRegister


@dataclass
class DummyProgram:
    prompt: str = "initial prompt"
    prefix: str = "initial prefix"
    metadata: dict[str, str] = field(default_factory=lambda: {"style": "initial style"})


def test_track_accepts_documented_list_batch():
    program = DummyProgram()
    registry = ParamRegistry()

    assert (
        registry.track(
            [
                (program, "prompt"),
                (program, "prefix", "prefix_alias"),
            ]
        )
        is registry
    )

    assert registry.names() == ["prompt", "prefix_alias"]
    assert registry.get("prefix_alias") == "initial prefix"

    registry.set("prefix_alias", "updated prefix")
    assert program.prefix == "updated prefix"

    registry.reset()
    assert program.prompt == "initial prompt"
    assert program.prefix == "initial prefix"


def test_track_accepts_tuple_batch_with_paths_and_aliases():
    program = DummyProgram()
    registry = ParamRegistry()

    registry.track(
        (
            (program, "metadata['style']", "style"),
            (program, "prompt"),
        )
    )

    assert registry.names() == ["style", "prompt"]
    assert registry.get("style") == "initial style"

    registry.set("style", "updated style")
    assert program.metadata["style"] == "updated style"


def test_track_keeps_single_field_chaining_behavior():
    program = DummyProgram()
    registry = ParamRegistry()

    returned = registry.track(program, "prompt").track(program, "prefix", name="alias")

    assert returned is registry
    assert registry.names() == ["prompt", "alias"]


@pytest.mark.parametrize("registry_type", [ParamRegistry, PromptTemplateRegister])
def test_track_rejects_single_call_without_path(registry_type):
    registry = registry_type()

    with pytest.raises(TypeError, match="path_or_attr must be a string"):
        registry.track(DummyProgram())
