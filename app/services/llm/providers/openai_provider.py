import os
from openai import AsyncOpenAI
from app.services.llm.base_provider import BaseProvider


class OpenAIProvider(BaseProvider):

    def __init__(self):

        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_KEY")
        )

    async def generate(
        self,
        model,
        prompt
    ):

        response = await self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content

    async def embed(
        self,
        model,
        text
    ):

        response = await self.client.embeddings.create(
            model=model,
            input=text
        )

        return response.data[0].embedding