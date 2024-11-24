from .providers.openai_provider import OpenAILLM
from .providers.anthropic_provider import AnthropicLLM
from .providers.mistral_provider import MistralLLM
from .providers.cohere_provider import CohereLLM

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
