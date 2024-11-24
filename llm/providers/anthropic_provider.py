import os
import requests
from typing import Generator
import json
from ..base import BaseLLM

class AnthropicLLM(BaseLLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.url = "https://api.anthropic.com/v1/messages"
        if not self.api_key:
            raise ValueError("Anthropic API key missing in environment variables.")

    def _get_headers(self):
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }

    def _get_payload(self, input_text: str, stream: bool = False):
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": input_text}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
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
            return response.json()["content"][0]["text"].strip()
        except requests.exceptions.RequestException as e:
            return f"Error during Anthropic request: {e}"
        except KeyError:
            return "Error parsing Anthropic API response."

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
                        if line == 'data: [DONE]':
                            break
                        data = json.loads(line[6:])
                        if 'delta' in data and 'text' in data['delta']:
                            yield data['delta']['text']
        except Exception as e:
            yield f"Error during streaming: {e}"
