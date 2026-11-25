"""Expose H3 image generation through the EvoAgentX toolkit interface."""

from ...tool import Toolkit
from .image_generation import MiniMaxImageGenerationTool


class MiniMaxImageToolkit(Toolkit):
    """Provide the MiniMax text-to-image tool."""

    def __init__(
        self,
        api_key: str | None = None,
        region: str = "global",
        name: str = "MiniMaxImageToolkit",
    ):
        """Select the credential and its region for image generation."""
        super().__init__(
            name=name,
            tools=[MiniMaxImageGenerationTool(api_key=api_key, region=region)],
        )
