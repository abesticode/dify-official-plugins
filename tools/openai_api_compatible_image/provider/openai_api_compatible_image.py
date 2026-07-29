from typing import Any
import requests
from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError

class OpenAICompatibleImageProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        endpoint_url = credentials.get("endpoint_url", "").strip()
        model_name = credentials.get("model_name", "").strip()
        if not endpoint_url:
            raise ToolProviderCredentialValidationError("API Endpoint URL is required.")
        if not model_name:
            raise ToolProviderCredentialValidationError("Model Name is required.")
