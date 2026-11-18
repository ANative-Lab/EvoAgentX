from evoagentx.models.base_model import LLMOutputParser


class RepeatedHeadingOutput(LLMOutputParser):
    count: int
    enabled: bool
    label: str


def test_title_parser_uses_latest_value_for_repeated_headings():
    content = """## count
1
## count
2
## enabled
false
## enabled
true
## label
alpha
## label
beta"""

    result = RepeatedHeadingOutput.parse(content, parse_mode="title")

    assert result.count == 2
    assert result.enabled is True
    assert result.label == "beta"
