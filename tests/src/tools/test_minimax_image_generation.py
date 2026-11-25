"""Exercise MiniMax image generation through the public toolkit."""

from unittest.mock import Mock, patch

import pytest
import requests

from evoagentx.tools import MiniMaxImageToolkit


@pytest.mark.parametrize(
    "region,tld,model",
    [
        ("global", "io", "MiniMax-H3-Image"),
        ("cn", "cn", "MiniMax-H3-Image-Highspeed"),
    ],
)
def test_image_generation_preserves_partial_success(region, tld, model):
    """Check the regional request and expose successful and failed items."""
    tool = MiniMaxImageToolkit(api_key="example", region=region).get_tool(
        "minimax_image_generation"
    )
    error = {"code": "1027", "message": "Image failed the content safety check"}
    data = [
        {"url": "https://example.com/image.jpg", "size": "1024x1024"},
        {"error": error},
    ]
    response = Mock(status_code=200)
    response.json.return_value = {"data": data, "request_id": "request-id"}
    with patch("requests.post", return_value=response) as post:
        result = tool(prompt="A snowy field", model=model, size="1024x1024", n=2)
    post.assert_called_once_with(
        f"https://api.minimax.{tld}/v1/images/generations",
        headers={"Authorization": "Bearer example"},
        json={
            "model": model,
            "prompt": "A snowy field",
            "size": "1024x1024",
            "n": 2,
            "response_format": "url",
        },
        timeout=600,
        allow_redirects=False,
    )
    assert result == {
        "results": [data[0]["url"]],
        "count": 1,
        "errors": [{"index": 1, **error}],
    }


def test_generation_reports_request_error_without_retry(monkeypatch):
    """Surface the provider message and leave non-idempotent retries to the caller."""
    monkeypatch.setenv("MINIMAX_API_KEY", "example")
    tool = MiniMaxImageToolkit().get_tool("minimax_image_generation")
    response = Mock(status_code=429)
    response.json.return_value = {
        "error": {"type": "rate_limit_error", "message": "Rate limited (1002)"}
    }
    with patch("requests.post", return_value=response) as post:
        assert tool(prompt="A fox") == {
            "error": "MiniMax returned HTTP 429: Rate limited (1002)"
        }
    assert post.call_count == 1
    with patch("requests.post", side_effect=requests.Timeout("read timed out")) as post:
        assert "read timed out" in tool(prompt="A fox")["error"]
    assert post.call_count == 1


def test_generation_returns_default_model_image():
    """A prompt alone returns an image URL using the standard H3 model."""
    toolkit = MiniMaxImageToolkit(api_key="example")
    tool = toolkit.get_tool("minimax_image_generation")
    assert toolkit.get_tool_schemas()[0]["function"]["parameters"]["required"] == [
        "prompt"
    ]
    image_url = "https://example.com/generated.jpg"
    response = Mock(status_code=200)
    response.json.return_value = {
        "data": [{"url": image_url, "revised_prompt": "A fox in sunlight"}]
    }
    with patch("requests.post", return_value=response) as post:
        assert tool(prompt="A fox") == {
            "results": [image_url],
            "count": 1,
            "errors": [],
        }
    assert post.call_args.kwargs["json"]["model"] == "MiniMax-H3-Image"
