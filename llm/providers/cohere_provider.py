import os
import requests
from typing import Generator
import json
from ..base import BaseLLM

class CohereLLM(BaseLLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = os.getenv("CO_API_KEY")
        self.url = "https://api.cohere.com/v2/chat"
        if not self.api_key:
            raise ValueError("Cohere API key missing in environment variables.")

    def _get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _get_payload(self, input_text: str, stream: bool = False):
        return {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": {"type": "text", "text": input_text}
            }],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "p": self.top_p,
            "stream": stream,
            **self.options
        }

    def generate(self, input_text: str) -> str:
        try:
            response = requests.post(
                self.url,
                headers=self._get_headers(),
                json=self._get_payload(input_text)
            )
            response.raise_for_status()
            return response.json()["message"]["content"][0]["text"].strip()
        except requests.exceptions.RequestException as e:
            return f"Error during Cohere request: {e}"
        except KeyError:
            return "Error parsing Cohere API response."

    def generate_stream(self, input_text: str) -> Generator[str, None, None]:
        try:
            response = requests.post(
                self.url,
                headers=self._get_headers(),
                json=self._get_payload(input_text, stream=True),
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = json.loads(line[6:])
                        if 'text' in data:
                            yield data['text']
        except Exception as e:
            yield f"Error during streaming: {e}"
