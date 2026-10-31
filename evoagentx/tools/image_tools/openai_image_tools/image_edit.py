from typing import Dict, Optional, List
from ...tool import Tool
from ...storage_handler import FileStorageHandler, LocalStorageHandler
from .openai_utils import create_openai_client


class OpenAIImageEditTool(Tool):
    name: str = "openai_image_edit"
    description: str = "Edit images using OpenAI gpt-image-1 (direct, minimal validation)."

    inputs: Dict[str, Dict[str, str]] = {
        "prompt": {"type": "string", "description": "Edit instruction. Required."},
        "images": {"type": "array", "description": "Image path(s) png/webp/jpg <50MB. Required. Single string accepted and normalized to array.", "items": {"type": "string"}},
        "mask_path": {"type": "string", "description": "Optional PNG mask path (same size as first image)."},
        "size": {"type": "string", "description": "1024x1024 | 1536x1024 | 1024x1536 | auto"},
        "n": {"type": "integer", "description": "1-10"},
        "background": {"type": "string", "description": "transparent | opaque | auto"},
        "input_fidelity": {"type": "string", "description": "high | low"},
        "output_compression": {"type": "integer", "description": "0-100 for jpeg/webp"},
        "output_format": {"type": "string", "description": "png | jpeg | webp (default png)"},
        "partial_images": {"type": "integer", "description": "0-3 partial streaming"},
        "quality": {"type": "string", "description": "auto | high | medium | low"},
        "stream": {"type": "boolean", "description": "streaming mode"},
        "image_name": {"type": "string", "description": "Optional output base name"},
    }
    required: Optional[List[str]] = ["prompt", "images"]

    def __init__(self, api_key: str, organization_id: str = None, save_path: str = "./edited_images", 
                 storage_handler: Optional[FileStorageHandler] = None):
        super().__init__()
        self.api_key = api_key
        self.organization_id = organization_id
        self.save_path = save_path
        self.storage_handler = storage_handler or LocalStorageHandler(base_path=save_path)

    def __call__(
        self,
        prompt: str,
        images: list,
        mask_path: str = None,
        size: str = None,
        n: int = None,
        background: str = None,
        input_fidelity: str = None,
        output_compression: int = None,
        output_format: str = None,
        partial_images: int = None,
        quality: str = None,
        stream: bool = None,
        image_name: str = None,
    ):
        try:
            client = create_openai_client(self.api_key, self.organization_id)

            # Accept either list[str] or a single string at runtime
            if isinstance(images, str):
                image_paths = [images]
            else:
                image_paths = list(images)

            opened_images = []
            temp_paths = []
            mask_fh = None
            try:
            # ensure compatibility and open files using storage handler
                for p in image_paths:
                    use_path, tmp = self._ensure_image_edit_compatible(p)
                    if tmp:
                        temp_paths.append(tmp)
                    opened_images.append(open(use_path, "rb"))

                api_kwargs = {
                    "model": "gpt-image-1",
                    "prompt": prompt,
                    "image": opened_images if len(opened_images) > 1 else opened_images[0],
                }
                if size is not None:
                    api_kwargs["size"] = size
                if n is not None:
                    api_kwargs["n"] = n
                if background is not None:
                    api_kwargs["background"] = background
                if input_fidelity is not None:
                    api_kwargs["input_fidelity"] = input_fidelity
                if output_compression is not None:
                    api_kwargs["output_compression"] = output_compression
                if output_format is not None:
                    api_kwargs["output_format"] = output_format
                if partial_images is not None:
                    api_kwargs["partial_images"] = partial_images
                if quality is not None:
                    api_kwargs["quality"] = quality
                if stream is not None:
                    api_kwargs["stream"] = stream

                if mask_path:
                    mask_fh = open(mask_path, "rb")
                    api_kwargs["mask"] = mask_fh

                response = client.images.edit(**api_kwargs)
            finally:
                for fh in opened_images:
                    try:
                        fh.close()
                    except Exception:
                        pass
                if mask_fh:
                    try:
                        mask_fh.close()
                    except Exception:
                        pass
                # cleanup temps
                import os
                for tp in temp_paths:
                    try:
                        if tp and os.path.exists(tp):
                            os.remove(tp)
                    except Exception:
                        pass

            # Save base64 images using storage handler
            import base64
            import time
            results = []
            for i, img in enumerate(response.data):
                try:
                    img_bytes = base64.b64decode(img.b64_json)
                    ts = int(time.time())
                    if image_name:
                        filename = f"{image_name.rsplit('.', 1)[0]}_{i+1}.png"
                    else:
                        filename = f"image_edit_{ts}_{i+1}.png"
                    
                    # Save using storage handler
                    result = self.storage_handler.save(filename, img_bytes)
                    
                    if result["success"]:
                        # Return the translated path that was actually used for saving
                        translated_path = self.storage_handler.translate_in(filename)
                        results.append(translated_path)
                    else:
                        results.append(f"Error saving image {i+1}: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    results.append(f"Error saving image {i+1}: {e}")

            return {"results": results, "count": len(results)}
        except Exception as e:
            return {"error": f"gpt-image-1 editing failed: {e}"}
    
    def _ensure_image_edit_compatible(self, image_path: str) -> tuple[str, str | None]:
        """
        Ensure the image matches OpenAI edit requirements using storage handler.
        If not, convert to RGBA and save to a temporary path. Return (usable_path, temp_path).
        Caller may delete temp_path after the request completes.
        """
        try:
            from PIL import Image
            from io import BytesIO
            import os
            
            # Use storage handler to read the image
            result = self.storage_handler.read(image_path)
            if not result["success"]:
                raise FileNotFoundError(f"Could not read image {image_path}: {result.get('error', 'Unknown error')}")
            
            # Get image content as bytes
            if isinstance(result["content"], bytes):
                content = result["content"]
            else:
                # If content is not bytes, convert to bytes
                content = str(result["content"]).encode('utf-8')
            
            # Open image from bytes
            with Image.open(BytesIO(content)) as img:
                if img.mode in ("RGBA", "LA", "L"):
                    # Image is already compatible, return the translated path
                    translated_path = self.storage_handler.translate_in(image_path)
                    return translated_path, None
                
                # Convert to RGBA
                rgba_img = img.convert("RGBA")
                
                # Save to temporary file using storage handler
                temp_filename = f"temp_rgba_{hash(image_path) % 10000}.png"
                buffer = BytesIO()
                rgba_img.save(buffer, format='PNG')
                temp_content = buffer.getvalue()
                
                # Save using storage handler
                result = self.storage_handler.save(temp_filename, temp_content)
                if result["success"]:
                    temp_path = self.storage_handler.translate_in(temp_filename)
                    return temp_path, temp_path
                else:
                    # Fallback to direct file I/O if storage handler fails
                    temp_path = os.path.join("workplace", "images", "temp_rgba_image.png")
                    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                    rgba_img.save(temp_path)
                    return temp_path, temp_path
                
        except Exception:
            # On error, return the translated path and let the caller decide
            translated_path = self.storage_handler.translate_in(image_path)
            return translated_path, None


