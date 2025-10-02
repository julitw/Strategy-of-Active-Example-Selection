from typing import Any, Dict, Optional
from langchain.llms.base import LLM
import requests
from pydantic import Field


    
class BielikLLM(LLM):
    token: str = Field(..., description="Token for API authentication")
    logprobs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Stores the log probabilities from the last call")
    temperature: float = Field(default=0.0, description="Sampling temperature")

    @property
    def _llm_type(self) -> str:
        return "bielik"

    def _call(self, prompt: str, stop: list = None, **kwargs) -> str:
        payload = {
            "model": "bielik",
            "messages": [{"role": "user", "content": prompt}],
            "logprobs": True,  
            "temperature": self.temperature 
        }

        headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }

        response = requests.post(
            "https://services.clarin-pl.eu/api/v1/oapi/chat/completions",
            json=payload,
            headers=headers
        )

        if response.status_code != 200:
            raise ValueError(f"Request failed with status {response.status_code}: {response.text}")
        result = response.json()

        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        self.logprobs = result.get("choices", [{}])[0].get("logprobs", {})

        return content