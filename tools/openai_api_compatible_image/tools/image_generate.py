import base64
import re
from typing import Any
from collections.abc import Generator
import requests
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

class ImageGenerateTool(Tool):
    def _invoke(
        self, tool_parameters: dict[str, Any]
    ) -> Generator[ToolInvokeMessage, None, None]:
        prompt = tool_parameters.get("prompt", "")
        if not prompt:
            yield self.create_text_message("Please provide a prompt.")
            return

        api_key = self.runtime.credentials.get("api_key", "")
        base_url = self.runtime.credentials.get("endpoint_url", "")
        if not base_url.endswith("/"):
            base_url += "/"

        model = tool_parameters.get("model") or self.runtime.credentials.get("model_name", "gemini-3.1-flash-image")
        endpoint_type = self.runtime.credentials.get("endpoint_type", "chat_completions")
        size_key = tool_parameters.get("size", "square")

        size_mapping_dalle = {
            "square": "1024x1024",
            "vertical": "1024x1792",
            "horizontal": "1792x1024",
        }
        aspect_ratio_mapping = {
            "square": "1:1",
            "vertical": "9:16",
            "horizontal": "16:9",
        }

        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        images_data = []

        if endpoint_type == "chat_completions":
            url = base_url + "chat/completions"
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "modalities": ["text", "image"],
                "response_modalities": ["TEXT", "IMAGE"],
                "stream": False,
                "image_config": {
                    "aspect_ratio": aspect_ratio_mapping.get(size_key, "1:1")
                }
            }
            try:
                res = requests.post(url, headers=headers, json=payload, timeout=180)
                if res.status_code != 200:
                    yield self.create_text_message(f"API request failed ({res.status_code}): {res.text}")
                    return
                res_json = res.json()
                choices = res_json.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    imgs = msg.get("images") or choices[0].get("images") or res_json.get("images")
                    if imgs and isinstance(imgs, list):
                        for img in imgs:
                            if isinstance(img, str):
                                images_data.append(img)
                            elif isinstance(img, dict):
                                url_val = img.get("url") or img.get("b64_json") or img.get("base64")
                                if isinstance(url_val, dict):
                                    url_val = url_val.get("url")
                                if url_val:
                                    images_data.append(url_val)
                    content = msg.get("content")
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                                img_obj = part.get("image_url") or part.get("image")
                                url_val = img_obj.get("url") if isinstance(img_obj, dict) else (img_obj or part.get("url"))
                                if url_val:
                                    images_data.append(url_val)
                    elif isinstance(content, str) and ("http://" in content or "https://" in content or "data:image" in content):
                        urls = re.findall(r'https?://[^\s\)]+|data:image/[^;]+;base64,[A-Za-z0-9+/=]+', content)
                        images_data.extend(urls)
            except Exception as e:
                yield self.create_text_message(f"Invocation error: {str(e)}")
                return
        else:
            url = base_url + "images/generations"
            payload = {
                "model": model,
                "prompt": prompt,
                "size": size_mapping_dalle.get(size_key, "1024x1024"),
                "response_format": "b64_json",
            }
            try:
                res = requests.post(url, headers=headers, json=payload, timeout=180)
                if res.status_code != 200:
                    yield self.create_text_message(f"API request failed ({res.status_code}): {res.text}")
                    return
                res_json = res.json()
                for item in res_json.get("data", []):
                    b64 = item.get("b64_json") or item.get("url")
                    if b64:
                        images_data.append(b64)
            except Exception as e:
                yield self.create_text_message(f"Invocation error: {str(e)}")
                return

        if not images_data:
            yield self.create_text_message("No image returned by API.")
            return

        for img_val in images_data:
            if not isinstance(img_val, str):
                continue
            if img_val.startswith("http://") or img_val.startswith("https://"):
                try:
                    img_res = requests.get(img_val, timeout=30)
                    blob_bytes = img_res.content
                    mime_type = img_res.headers.get("Content-Type", "image/png")
                except Exception as e:
                    yield self.create_text_message(f"Failed to fetch image URL: {str(e)}")
                    continue
            else:
                b64_str = img_val
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                try:
                    blob_bytes = base64.b64decode(b64_str)
                    mime_type = "image/png"
                except Exception as e:
                    yield self.create_text_message(f"Failed to decode base64 image: {str(e)}")
                    continue

            yield self.create_blob_message(
                blob=blob_bytes,
                meta={"mime_type": mime_type}
            )
