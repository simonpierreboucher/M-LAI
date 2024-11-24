# llm/__init__.py
from .providers.openai_provider import OpenAILLM
from .providers.anthropic_provider import AnthropicLLM
from .providers.mistral_provider import MistralLLM
from .providers.cohere_provider import CohereLLM
from .chatbot import Chatbot  # Add this line

class LLM:
    @staticmethod
    def create(provider='openai', **kwargs):
        provider = provider.lower()
        if provider == 'openai':
            return OpenAILLM(**kwargs)
        elif provider == 'anthropic':
            return AnthropicLLM(**kwargs)
        elif provider == 'mistral':
            return MistralLLM(**kwargs)
        elif provider == 'cohere':
            return CohereLLM(**kwargs)
        else:
            raise ValueError(
                f"Invalid provider: {provider}. "
                "Choose from 'openai', 'anthropic', 'mistral', or 'cohere'."
            )
