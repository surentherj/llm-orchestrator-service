import os
from dotenv import load_dotenv

load_dotenv()


class Settings:

    POSTGRES_URL = os.getenv("POSTGRES_URL")

    REDIS_URL = os.getenv("REDIS_URL")

    DEFAULT_MODEL = "gpt-4o-mini"

    DEFAULT_EMBEDDING = "text-embedding-3-small"

    SELLER_CONTACT_HINT = os.getenv(
        "SELLER_CONTACT_HINT",
        "through this chat",
    )
    SELLER_HOURS_TEXT = os.getenv(
        "SELLER_HOURS_TEXT",
        "Varies by seller — ask for today’s window",
    )
    SHIPPING_DISPATCH_TEXT = os.getenv(
        "SHIPPING_DISPATCH_TEXT",
        "1–3 business days after confirmation",
    )
    ESCALATION_SLA_TEXT = os.getenv(
        "ESCALATION_SLA_TEXT",
        "within a few hours in this chat",
    )


settings = Settings()