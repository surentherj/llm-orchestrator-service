import json
import redis.asyncio as redis
from app.core.config import settings


class SessionStore:

    def __init__(self):

        self.client = redis.from_url(
            settings.REDIS_URL
        )

    async def get(self, conversation_id):

        data = await self.client.get(
            conversation_id
        )

        if not data:
            return {}

        return json.loads(data)

    async def set(
        self,
        conversation_id,
        data
    ):

        await self.client.set(
            conversation_id,
            json.dumps(data),
            ex=3600
        )


session_store = SessionStore()