from abc import ABC, abstractmethod


class BaseProvider(ABC):

    @abstractmethod
    async def generate(
        self,
        model,
        prompt
    ):
        pass

    @abstractmethod
    async def embed(
        self,
        model,
        text
    ):
        pass