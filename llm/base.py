from abc import ABC, abstractmethod
import os
from typing import Generator, Optional
from dotenv import load_dotenv

load_dotenv()

class BaseLLM(ABC):
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1500,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        **kwargs
    ):
        if not 0 <= temperature <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        if not 0 <= top_p <= 1:
            raise ValueError("Top_p must be between 0 and 1")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.options = kwargs

    @abstractmethod
    def generate(self, input_text: str) -> str:
        pass

    @abstractmethod
    def generate_stream(self, input_text: str) -> Generator[str, None, None]:
        pass
