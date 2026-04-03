"""Deterministic seller-facing replies — zero LLM cost. Override via env in config."""

from app.core.config import settings


def shipping_text() -> str:
    return (
        f"Shipping: we pack after payment confirmation. "
        f"Typical dispatch: {settings.SHIPPING_DISPATCH_TEXT}. "
        f"Tracking is shared by the seller once shipped."
    )


def returns_text() -> str:
    return (
        "Returns: eligible within the window stated on your order/pack slip. "
        "Open-box/custom items may be excluded. The seller confirms each case."
    )


def payment_text() -> str:
    return (
        "Payment: methods depend on this seller (card, wallet, link, cash on delivery, etc.). "
        "They send exact steps when you confirm an order."
    )


def contact_text() -> str:
    h = settings.SELLER_CONTACT_HINT
    return f"Contact: message this chat or reach the seller {h}."


def hours_text() -> str:
    return (
        f"Hours: {settings.SELLER_HOURS_TEXT}. "
        "Replies outside hours may be next business window."
    )


STORE_TOPIC_REPLIES = {
    "shipping": shipping_text,
    "returns": returns_text,
    "payment": payment_text,
    "contact": contact_text,
    "hours": hours_text,
}


def escalation_message(ticket_id: str) -> str:
    return (
        f"Escalation #{ticket_id}: a person will review this chat. "
        f"Expect a reply {settings.ESCALATION_SLA_TEXT}. "
        f"You can also message: {settings.SELLER_CONTACT_HINT}"
    )
