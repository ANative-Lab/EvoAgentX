---
title: Image Tools Tutorial (OpenAI / OpenRouter / Flux)
---

This tutorial shows how to use image tools in EvoAgentX, including:

- OpenAI: generation, editing, analysis (Responses API)
- OpenRouter: generation/editing (supports local file paths and remote URLs), analysis
- Flux (bfl.ai): generation/editing (Kontext Max)

Note: Naming is casual; tool classes end with Tool, toolkits end with Toolkit; no version suffix.

## Quick Start

Before running examples, set these environment variables as needed:

- OPENAI_API_KEY (optional OPENAI_ORGANIZATION_ID)
- OPENROUTER_API_KEY (used by OpenRouter and analysis in Flux section)
- BFL_API_KEY (Flux)

```python
from evoagentx.tools import OpenAIImageToolkit, OpenRouterImageToolkit, FluxImageGenerationToolkit

# Basic usage with default local storage
tk = OpenAIImageToolkit(api_key=OPENAI_API_KEY)

# Custom storage path
from evoagentx.tools import LocalStorageHandler
storage_handler = LocalStorageHandler(base_path="./my_images")
tk = OpenAIImageToolkit(api_key=OPENAI_API_KEY, storage_handler=storage_handler)
```

## OpenAI Image Tools

`OpenAIImageToolkit` groups three tools:

- openai_image_generation (text-to-image, supports gpt-image-1 / dall-e-3)
- openai_image_edit (gpt-image-1 image editing)
- openai_image_analysis (Responses API image understanding, supports image_url or image_path)

Example: generate → edit → analyze

```python
tk = OpenAIImageToolkit(api_key=OPENAI_API_KEY, organization_id=OPENAI_ORG_ID,
                        generation_model="gpt-image-1", save_path="./generated_images",
                        storage_handler=None)  # Uses default LocalStorageHandler

gen = tk.get_tool("openai_image_generation")
edit = tk.get_tool("openai_image_edit")
analyze = tk.get_tool("openai_image_analysis")

res_gen = gen(prompt="A cute baby owl...", size="1024x1024")
src_path = res_gen["results"][0]

res_edit = edit(prompt="Add a red scarf", images=src_path, size="1024x1024")
edited_path = res_edit["results"][0]

res_ana = analyze(prompt="Describe this image", image_path=edited_path, model="gpt-4o-mini")
print(res_ana.get("content", ""))
```

### Standalone: Generation (openai_image_generation)

```python
from evoagentx.tools import OpenAIImageToolkit

tk = OpenAIImageToolkit(api_key=OPENAI_API_KEY, generation_model="gpt-image-1", save_path="./generated_images")
gen = tk.get_tool("openai_image_generation")

result = gen(
  prompt="A watercolor landscape with mountains and lake",
  model="gpt-image-1",            # gpt-image-1 | dall-e-3 | dall-e-2
  size="1024x1024",               # model-specific size
  quality="high",                  # gpt-image-1/dall-e-3
  n=1,                              # dalle-3 fixed to 1
  style="vivid",                   # dall-e-3 vivid|natural
  response_format="url",           # dalle-2/3: url|b64_json
)
print(result["results"])  # list of local paths
```

Common optional params (model-dependent): `background`, `moderation`, `output_compression`, `output_format`, `partial_images`, `stream`.

### Standalone: Edit (openai_image_edit)

```python
from evoagentx.tools import OpenAIImageToolkit

tk = OpenAIImageToolkit(api_key=OPENAI_API_KEY, save_path="./edited_images")
edit = tk.get_tool("openai_image_edit")

result = edit(
  prompt="Add a small red boat on the lake",
  images="./generated_images/sample.png",  # string or array
  size="1024x1024",
  background="opaque",
  quality="high",
  n=1,
  image_name="edited_boat"
)
print(result["results"])  # list of local paths
```

Optional: `mask_path` (same size PNG), `input_fidelity`, `output_compression`, `output_format`, `partial_images`, `stream`.

### Standalone: Analysis (openai_image_analysis)

```python
from evoagentx.tools import OpenAIImageToolkit

tk = OpenAIImageToolkit(api_key=OPENAI_API_KEY)
analyze = tk.get_tool("openai_image_analysis")

# Option 1: remote URL
print(analyze(prompt="Describe this image", image_url="https://example.com/img.png"))

# Option 2: local path (auto-converted to data URL)
print(analyze(prompt="Summarize the image", image_path="./edited_images/edited_boat_1.png", model="gpt-4o-mini"))
```

## OpenRouter Image Tools

`OpenRouterImageToolkit` groups:

- openrouter_image_generation_edit (text-to-image and editing; supports image_urls and image_paths)
- image_analysis (OpenRouter multimodal analysis; supports image_url / image_path / pdf_path)

Example: edit with a local path

```python
from evoagentx.tools import OpenRouterImageToolkit

tk = OpenRouterImageToolkit(api_key=OPENROUTER_API_KEY, storage_handler=None)  # Uses default LocalStorageHandler
gen = tk.get_tool("openrouter_image_generation_edit")

res = gen(
  prompt="Add a bold title at the top",
  image_paths=["./imgs/base.png"],
  model="google/gemini-2.5-flash-image-preview",
  save_path="./openrouter_images",
  output_basename="edited"
)
print(res.get("saved_paths", []))
```

### Standalone: Generation (openrouter_image_generation_edit, no image → generate)

```python
from evoagentx.tools import OpenRouterImageToolkit

tk = OpenRouterImageToolkit(api_key=OPENROUTER_API_KEY)
gen = tk.get_tool("openrouter_image_generation_edit")

res = gen(
  prompt="A minimalist poster of a mountain at sunrise",
  model="google/gemini-2.5-flash-image-preview",
  save_path="./openrouter_images",
  output_basename="mountain"
)
print(res.get("saved_paths", []))
```

### Standalone: Edit (openrouter_image_generation_edit, with images → edit)

Accepts remote URLs or local paths: `image_urls` / `image_paths`.

```python
from evoagentx.tools import OpenRouterImageToolkit

tk = OpenRouterImageToolkit(api_key=OPENROUTER_API_KEY)
gen = tk.get_tool("openrouter_image_generation_edit")

# Using local path
res = gen(
  prompt="Add a bold 'GEMINI' title",
  image_paths=["./openrouter_images/mountain.png"],
  model="google/gemini-2.5-flash-image-preview",
  save_path="./openrouter_images",
  output_basename="edited"
)
print(res.get("saved_paths", []))
```

### Standalone: Analysis (image_analysis)

```python
from evoagentx.tools import OpenRouterImageToolkit

tk = OpenRouterImageToolkit(api_key=OPENROUTER_API_KEY)
analyze = tk.get_tool("image_analysis")

# Either URL or local path
print(analyze(prompt="Describe", image_url="https://example.com/a.png"))
print(analyze(prompt="Describe", image_path="./openrouter_images/edited.png"))
```

## Flux Image Tools

`FluxImageGenerationToolkit` includes the generation/edit tool by default:

- flux_image_generation_edit (no input_image → generate; with input_image(base64) → edit)

The tool returns a local file path under `save_path`.

```python
from evoagentx.tools import FluxImageGenerationToolkit

tk = FluxImageGenerationToolkit(api_key=BFL_API_KEY, save_path="./flux_generated_images")
gen = tk.get_tool("flux_image_generation_edit")

res = gen(prompt="A cyberpunk city", output_format="jpeg", seed=42)
print(res["file_path"])  # local file path
```

### Standalone: Generation (flux_image_generation_edit, without input_image)

```python
from evoagentx.tools import FluxImageGenerationToolkit

tk = FluxImageGenerationToolkit(api_key=BFL_API_KEY, save_path="./flux_generated_images")
gen = tk.get_tool("flux_image_generation_edit")

res = gen(
  prompt="A futuristic cyberpunk city",
  seed=42,
  output_format="jpeg",
  prompt_upsampling=False,
  safety_tolerance=2,
)
print(res["file_path"])
```

### Standalone: Edit (flux_image_generation_edit, with input_image base64)

```python
from evoagentx.tools import FluxImageGenerationToolkit
import base64

tk = FluxImageGenerationToolkit(api_key=BFL_API_KEY, save_path="./flux_generated_images")
gen = tk.get_tool("flux_image_generation_edit")

with open("./flux_generated_images/base.jpg", "rb") as f:
  b64 = base64.b64encode(f.read()).decode("utf-8")

res = gen(
  prompt="Add a glowing red umbrella",
  input_image=b64,
  output_format="jpeg",
)
print(res["file_path"])
```

## Storage & File IO Notes

- **Storage Support**: All image tools support flexible storage options:
  - **Local Storage**: Default option for saving images to local filesystem
  - **Remote Storage**: Optional Supabase or Neon cloud storage for distributed access
- **File Operations**: All file I/O is handled automatically through the storage system
- **Toolkit Parameters**: All toolkits accept `storage_handler` parameter for custom storage
- Both OpenRouter and OpenAI analysis tools support `image_path` (auto-converted to data URL); `image_url` also works.

## FAQ

- **No image returned?** Check API key, model, and quota.
- **OpenAI Responses error?** Ensure the model supports image input (e.g., gpt-4o-mini).
- **Flux save failed?** Ensure `save_path` is writable and there is disk space.
- **Storage errors?** 
  - Local storage: Check directory permissions and disk space
  - Supabase storage: Verify environment variables and bucket access
  - Neon storage: Verify environment variables and bucket access
- **File not found errors?** Check if the storage path is correct and accessible


