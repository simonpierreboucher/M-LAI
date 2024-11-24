import os
import requests
import json
from typing import Generator
from ..base import BaseLLM

class OpenAILLM(BaseLLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.url = "https://api.openai.com/v1/chat/completions"
        if not self.api_key:
            raise ValueError("OpenAI API key missing in environment variables.")

    def _get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _get_payload(self, input_text: str, stream: bool = False):
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": input_text}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
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
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            return f"Error during OpenAI request: {e}"
        except KeyError:
            return "Error parsing OpenAI API response."

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
                        if content := data['choices'][0].get('delta', {}).get('content'):
                            yield content
        except Exception as e:
            yield f"Error during streaming: {e}"
