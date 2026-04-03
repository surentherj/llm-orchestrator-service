from app.services.llm.model_registry import MODEL_REGISTRY

from app.services.llm.providers.openai_provider import (
    OpenAIProvider
)

from app.services.llm.providers.gemini_provider import (
    GeminiProvider
)


class ProviderRouter:

    def __init__(self):

        self.providers = {

            "openai": OpenAIProvider(),
            "gemini": GeminiProvider()
        }

    async def generate(
        self,
        model_key,
        prompt
    ):

        config = MODEL_REGISTRY[model_key]

        provider_name = config["provider"]

        provider = self.providers[
            provider_name
        ]

        return await provider.generate(
            model_key,
            prompt
        )

    async def embed(
        self,
        model_key,
        text
    ):

        config = MODEL_REGISTRY[model_key]

        provider = self.providers[
            config["provider"]
        ]

        return await provider.embed(
            model_key,
            text
        )


provider_router = ProviderRouter()