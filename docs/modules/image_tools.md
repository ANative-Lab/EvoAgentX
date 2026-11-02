---
title: Image Tools Module (OpenAI / OpenRouter / Flux)
---

This document describes the image tools in EvoAgentX: module layout, naming, public API, inputs/outputs, and behavior.

## Naming

- Tool classes end with `Tool`; toolkits end with `Toolkit`.
- Casual names without version suffixes.
- Examples: `OpenAIImageEditTool`, `OpenRouterImageGenerationEditTool`, `FluxImageGenerationEditTool`.

## Directory Layout

`evoagentx/tools/image_tools/`

- `openai_image_tools/`
  - `image_generation.py` → `OpenAIImageGenerationTool`
  - `image_edit.py` → `OpenAIImageEditTool`
  - `image_analysis_openai.py` → `OpenAIImageAnalysisTool`
  - `toolkit.py` → `OpenAIImageToolkit`
- `openrouter_image_tools/`
  - `image_generation.py` → `OpenRouterImageGenerationEditTool`
  - `image_analysis.py` → `ImageAnalysisTool`
  - `toolkit.py` → `OpenRouterImageToolkit`, `ImageAnalysisToolkit`
  - `utils.py` → minimal; add only when functions are shared across modules
- `flux_image_tools/`
  - `image_generation_edit.py` → `FluxImageGenerationEditTool`
  - `toolkit.py` → `FluxImageGenerationToolkit`

## Public API (via `evoagentx.tools`)

```python
from evoagentx.tools import (
  OpenAIImageToolkit,
  OpenRouterImageToolkit,
  FluxImageGenerationToolkit,
)
```

## Toolkits and Tools

### OpenAIImageToolkit

- Tools included:
  - `openai_image_generation` (text-to-image: gpt-image-1 / dall-e-3 / dall-e-2)
  - `openai_image_edit` (gpt-image-1 image editing)
  - `openai_image_analysis` (Responses API image understanding; supports `image_url` / `image_path`)
- Constructor params: `api_key`, `organization_id`, `generation_model`, `save_path`, `storage_handler`

Details:
- `OpenAIImageGenerationTool`
  - Inputs (common): `prompt` (str, required), `model`, `size`, `quality`, `n`, `style`, `response_format`
  - Some models support: `background`, `moderation`, `output_compression`, `output_format`, `partial_images`, `stream`
  - Returns: `{ "results": ["<local_path>", ...], "count": int }`
- `OpenAIImageEditTool`
  - Inputs: `prompt` (str, required), `images` (str|List[str], required), `size`, `n`, `background`, `input_fidelity`, `output_compression`, `output_format`, `partial_images`, `quality`, `stream`, `image_name`
  - Returns: `{ "results": ["<local_path>", ...], "count": int }`
- `OpenAIImageAnalysisTool`
  - Inputs: `prompt` (str, required), `image_url` (str) or `image_path` (str), `model` (str)
  - Returns: `{ "content": str }`

Environment:
- Requires `OPENAI_API_KEY` (and optional `OPENAI_ORGANIZATION_ID` if used).

### OpenRouterImageToolkit

- Tools included:
  - `openrouter_image_generation_edit` (text-to-image; if images provided → edit). Supports `image_urls` and `image_paths`.
  - `image_analysis` (multimodal analysis; supports `image_url` / `image_path` / `pdf_path`).
- Constructor params: `api_key`, `storage_handler`

Details:
- `OpenRouterImageGenerationEditTool`
  - Inputs: `prompt` (str, required), `image_urls` (List[str]), `image_paths` (List[str]), `model` (str), `api_key` (str), `save_path` (str), `output_basename` (str)
  - Behavior: no images → generate; with images → edit/compose. Saves any returned data URLs to `save_path`.
  - Returns: `{ "saved_paths": ["<local_path>", ...] }` or `{ "warning": str, "raw": object }`
- `ImageAnalysisTool`
  - Inputs: `prompt` (str, required), `image_url` (str), `image_path` (str), `pdf_path` (str)
  - Returns: `{ "content": str, "usage": object }` (usage may be absent)

Environment:
- Requires `OPENROUTER_API_KEY`.

### ImageAnalysisToolkit (OpenRouter)

- Convenience wrapper that only exposes `image_analysis`.
- Constructor params: `api_key`, `model`, `storage_handler`

### FluxImageGenerationToolkit

- Tools included:
  - `flux_image_generation_edit` (no `input_image` → generate; with `input_image` (base64) → edit)
- Constructor params: `api_key`, `save_path`, `storage_handler`

Details:
- `FluxImageGenerationEditTool`
  - Inputs: `prompt` (str, required), `input_image` (base64 str), `seed` (int), `aspect_ratio` (str), `output_format` (str), `prompt_upsampling` (bool), `safety_tolerance` (int)
  - Returns: `{ "file_path": "<local_path>" }`

Environment:
- Requires `BFL_API_KEY`.

## I/O and Return Contracts

- All tools use the storage handler architecture:
  - **Storage Handler**: All toolkits accept `storage_handler` parameter (defaults to `LocalStorageHandler`)
  - **File Operations**: Automatic path translation and format detection
  - **Supported Storage**: Local filesystem, Supabase cloud storage, and Neon cloud storage
- Typical returns:
  - OpenAI generation/edit: `{ "results": ["<local_path>", ...] }`
  - OpenRouter generation/edit: `{ "saved_paths": ["<local_path>", ...] }`
  - Flux generation/edit: `{ "file_path": "<local_path>" }`
  - Analysis: `{ "content": "...", "usage": { ... } }`

## Error Handling

- Tools return `{ "error": str }` on recoverable errors; callers should guard on `'error' in result`.
- Some tools may return `{ "warning": str, "raw": object }` when the upstream API returns no images but a valid response.

## Extensibility

- Add new vendor tools under `image_tools/<vendor>_image_tools/` with the same naming and return conventions.
- Keep `utils.py` minimal; inline helpers inside tool classes unless functions are used by multiple modules.


