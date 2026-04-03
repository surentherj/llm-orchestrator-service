import os
import asyncio

from google import genai

from app.services.llm.base_provider import BaseProvider


class GeminiProvider(BaseProvider):

    def __init__(self):

        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )

    # -------------------------
    # TEXT GENERATION
    # -------------------------

    async def generate(
        self,
        model_key: str,
        prompt: str
    ) -> str:

        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=model_key,
            contents=prompt
        )
        return response.text

    # -------------------------
    # EMBEDDING
    # -------------------------

    async def embed(
        self,
        model_key: str,
        text: str
    ):

        response = await asyncio.to_thread(
            self.client.models.embed_content,
            model=model_key,
            contents=text
        )

        return response.embeddings[0].values