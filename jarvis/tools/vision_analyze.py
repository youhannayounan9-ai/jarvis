"""
jarvis/tools/vision_analyze.py
──────────────────────────────
Tool: vision_analyze

Analyzes an image using a local multimodal model (llava).
"""

import base64
from pathlib import Path

from litellm import completion

from jarvis.config import settings
from jarvis.tools.base import BaseTool
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


class VisionAnalyzeTool(BaseTool):
    name = "vision_analyze"
    description = (
        "CRITICAL: Use this to analyze an image, screenshot, or photo. "
        "You MUST provide the absolute file path to the image."
    )
    parameters = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "The absolute path to the image file.",
            },
        },
        "required": ["image_path"],
    }
    risk_level = "FILE_READ"
    timeout_seconds = 60.0

    def run(self, image_path: str, **kwargs) -> str:
        try:
            target = Path(image_path).resolve()
            allowed_dir = settings.file_reader_allowed_path

            if not target.is_relative_to(allowed_dir):
                log.warning("vision_path_traversal_blocked", path=str(target))
                return f"ERROR: Access denied. You can only read files inside '{allowed_dir}'."

            if not target.exists() or not target.is_file():
                return f"ERROR: Image file not found at {target}."
            
            ext = target.suffix.lower()
            if ext in (".jpg", ".jpeg"):
                mime = "image/jpeg"
            elif ext == ".png":
                mime = "image/png"
            elif ext == ".webp":
                mime = "image/webp"
            else:
                mime = "image/jpeg"

            with open(target, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")

            log.info("vision_analyze_start", path=str(target))
            response = completion(
                model="ollama_chat/llava",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in detail."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{encoded}"}
                            }
                        ]
                    }
                ],
            )
            return response.choices[0].message.content or "No description returned."

        except Exception as e:
            log.error("vision_analyze_error", error=str(e), path=image_path)
            return f"ERROR: Failed to analyze image. {e}"
