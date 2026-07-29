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
        image_resolution = tool_parameters.get("image_resolution", "1K")
        output_mime_type = tool_parameters.get("output_mime_type", "image/png")
        temperature = tool_parameters.get("temperature", 1.0)
        top_p = tool_parameters.get("top_p", 0.95)
        thinking_level = tool_parameters.get("thinking_level", "MINIMAL")
        safety_settings_val = tool_parameters.get("safety_settings", "OFF")
        quality = tool_parameters.get("quality", "standard")
        style = tool_parameters.get("style", "vivid")
        seed_id = tool_parameters.get("seed_id")

        size_mapping_dalle = {
            "square": "1024x1024",
            "vertical": "1024x1792",
            "horizontal": "1792x1024",
            "4:3": "1024x768",
            "3:4": "768x1024",
        }
        aspect_ratio_mapping = {
            "square": "1:1",
            "vertical": "9:16",
            "horizontal": "16:9",
            "4:3": "4:3",
            "3:4": "3:4",
        }

        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        images_data = []

        if endpoint_type == "chat_completions":
            url = base_url + "chat/completions"
            
            image_config = {}
            if size_key in aspect_ratio_mapping:
                image_config["aspect_ratio"] = aspect_ratio_mapping[size_key]
            if image_resolution:
                image_config["image_size"] = image_resolution
            if output_mime_type:
                image_config["output_mime_type"] = output_mime_type

            thinking_config = {}
            if thinking_level and thinking_level != "OFF":
                thinking_config["thinking_level"] = thinking_level

            safety_settings_list = []
            if safety_settings_val:
                for cat in [
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "HARM_CATEGORY_HARASSMENT",
                ]:
                    safety_settings_list.append({
                        "category": cat,
                        "threshold": safety_settings_val,
                    })

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "modalities": ["text", "image"],
                "response_modalities": ["TEXT", "IMAGE"],
                "stream": False,
                "temperature": float(temperature),
                "top_p": float(top_p),
            }

            if image_config:
                payload["image_config"] = image_config
            if thinking_config:
                payload["thinking_config"] = thinking_config
            if safety_settings_list:
                payload["safety_settings"] = safety_settings_list

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
                "quality": quality,
                "style": style,
                "response_format": "b64_json",
            }
            if seed_id:
                payload["extra_body"] = {"seed": seed_id}

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
                    mime_type = img_res.headers.get("Content-Type", output_mime_type or "image/png")
                except Exception as e:
                    yield self.create_text_message(f"Failed to fetch image URL: {str(e)}")
                    continue
            else:
                b64_str = img_val
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                try:
                    blob_bytes = base64.b64decode(b64_str)
                    mime_type = output_mime_type or "image/png"
                except Exception as e:
                    yield self.create_text_message(f"Failed to decode base64 image: {str(e)}")
                    continue

            yield self.create_blob_message(
                blob=blob_bytes,
                meta={"mime_type": mime_type}
            )
