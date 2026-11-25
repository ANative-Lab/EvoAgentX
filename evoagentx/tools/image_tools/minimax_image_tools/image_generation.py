"""MiniMax H3 text-to-image generation."""

import os

import requests
from pydantic import PrivateAttr

from ...tool import Tool

_ENDPOINTS = {
    "global": "https://api.minimax.io/v1/images/generations",
    "cn": "https://api.minimax.cn/v1/images/generations",
}
_MODELS = ("MiniMax-H3-Image", "MiniMax-H3-Image-Highspeed")


class MiniMaxImageGenerationTool(Tool):
    """Generate images with H3 and return their temporary URLs."""

    name: str = "minimax_image_generation"
    description: str = "Generate images from a text prompt using MiniMax H3."
    inputs: dict[str, dict[str, str]] = {  # noqa: RUF012 - Tool validates the schema at class creation.
        "prompt": {"type": "string", "description": "Text describing the image."},
        "model": {
            "type": "string",
            "description": "MiniMax-H3-Image or MiniMax-H3-Image-Highspeed.",
        },
        "size": {"type": "string", "description": "Image size: auto or WIDTHxHEIGHT."},
        "n": {"type": "integer", "description": "Number of images, from 1 to 10."},
    }
    required: list[str] = ["prompt"]  # noqa: RUF012 - Tool validates this field at class creation.
    _api_key: str = PrivateAttr()
    _endpoint: str = PrivateAttr()

    def __init__(self, api_key: str | None = None, region: str = "global"):
        """Configure the regional endpoint and read the key from MINIMAX_API_KEY if omitted."""
        super().__init__()
        if region not in _ENDPOINTS:
            raise ValueError("MiniMax region must be 'global' or 'cn'.")
        self._endpoint = _ENDPOINTS[region]
        self._api_key = api_key or os.getenv("MINIMAX_API_KEY")
        if not self._api_key:
            raise ValueError("Provide a MiniMax API key or set MINIMAX_API_KEY.")

    def __call__(
        self,
        prompt: str,
        model: str = "MiniMax-H3-Image",
        size: str = "auto",
        n: int = 1,
    ):
        """Return successful image URLs and any per-image errors, or a request error."""
        if model not in _MODELS or not 1 <= n <= 10 or not prompt.strip():
            return {
                "error": "Provide a nonempty prompt, a supported H3 model and n between 1 and 10."
            }
        try:
            # A second POST would generate and charge for another batch of images.
            response = requests.post(
                self._endpoint,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": model,
                    "prompt": prompt,
                    "n": n,
                    "size": size,
                    "response_format": "url",
                },
                timeout=600,
                allow_redirects=False,
            )
            if not 200 <= response.status_code < 300:
                try:
                    message = response.json()["error"]["message"]
                except (KeyError, TypeError, ValueError):
                    message = response.reason
                return {
                    "error": f"MiniMax returned HTTP {response.status_code}: {message}"
                }
            items = response.json()["data"]
            if not isinstance(items, list) or not items:
                return {"error": "MiniMax returned no image results."}
            urls, errors = [], []
            for index, item in enumerate(items):
                if not isinstance(item, dict):
                    errors.append({"index": index, "message": "Invalid image result."})
                elif isinstance(item.get("error"), dict):
                    errors.append({"index": index, **item["error"]})
                elif isinstance(item.get("url"), str) and item["url"]:
                    urls.append(item["url"])
                else:
                    errors.append({"index": index, "message": "Missing image URL."})
            return {"results": urls, "count": len(urls), "errors": errors}
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            return {"error": f"MiniMax image generation failed: {exc}"}
