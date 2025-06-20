from typing import Optional, Dict, Any
from langchain_community.chat_models import ChatOllama

class ChatOllamaThinking(ChatOllama):
    """Extension of ChatOllama that adds support for the `think` parameter."""

    think: Optional[bool] = None

    @property
    def _default_params(self) -> Dict[str, Any]:
        params = super()._default_params
        if self.think is not None:
            params = dict(params)
            params["think"] = self.think
        return params
