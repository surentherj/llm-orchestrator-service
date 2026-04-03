"""
Session field names and intent disambiguation — single source of truth
(no scattered magic strings for Redis/session shape).
"""

from __future__ import annotations

# --- Session keys (persisted with the chat session) ---
AWAITING_ORDER_CONFIRM = "awaiting_order_confirm"
HUMAN_FOLLOWUP_PENDING = "human_followup_pending"
UNRESOLVED_STREAK = "unresolved_streak"

# Must match generator instruction in orchestrator (exact reply when facts lack answer).
LLM_REFUSAL_NOT_IN_FACTS = "Not in provided catalog data."

# User text signals (language-agnostic tokens; extend per locale in one place).
_HUMAN_INTENT_SUBSTRINGS = (
    "connect",
    "human",
    "agent",
    "seller",
    "talk to",
    "someone",
    "person",
    "help me",
    "reach you",
    "live chat",
)

_CHECKOUT_AFFIRM_SUBSTRINGS = (
    "yes",
    "yeah",
    "yep",
    "ok",
    "okay",
    "confirm",
    "confirmed",
    "proceed",
    "go ahead",
    "place it",
    "place order",
    "do it",
    "sure",
)


def user_message_prefers_human(text: str | None) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    return any(s in t for s in _HUMAN_INTENT_SUBSTRINGS)


def user_message_confirms_checkout(text: str | None) -> bool:
    """True only if this looks like confirming an order, not accepting a human offer."""
    t = (text or "").strip().lower()
    if not t or user_message_prefers_human(t):
        return False
    if len(t) > 40:
        return any(s in t for s in _CHECKOUT_AFFIRM_SUBSTRINGS)
    return any(s == t or s in t for s in _CHECKOUT_AFFIRM_SUBSTRINGS)


def user_message_rejects_checkout(text: str | None) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    return any(
        x in t
        for x in (
            "no ",
            "nope",
            "nah",
            "cancel",
            "stop",
            "wait",
            "not now",
            "don't",
            "dont",
        )
    ) or t in ("no", "n")


def remap_intents_after_detection(
    intents: list[str],
    session: dict,
) -> list[str]:
    """
    If the model mislabels a reply to 'talk to human' as confirm_checkout,
    map to human_handoff when we previously invited a human and checkout is not pending.
    """
    if not intents:
        return intents
    if session.get(AWAITING_ORDER_CONFIRM):
        return intents
    if not session.get(HUMAN_FOLLOWUP_PENDING):
        return intents
    if "confirm_checkout" not in intents:
        return intents

    out: list[str] = []
    added_handoff = False
    for i in intents:
        if i == "confirm_checkout":
            if not added_handoff:
                out.append("human_handoff")
                added_handoff = True
        elif i == "human_handoff":
            if not added_handoff:
                out.append(i)
                added_handoff = True
        else:
            out.append(i)
    return out if out else ["human_handoff"]


def mark_human_followup_invited(session: dict) -> None:
    session[HUMAN_FOLLOWUP_PENDING] = True


def clear_human_followup(session: dict) -> None:
    session[HUMAN_FOLLOWUP_PENDING] = False


def reply_invites_human_followup(
    reply: str | None,
    *,
    catalog_miss_reply: str,
    human_hint_reply: str,
) -> bool:
    """True when the bot just told the user automation is insufficient / talk to a person."""
    r = (reply or "").lower()
    if not r:
        return False
    if human_hint_reply and human_hint_reply.lower() in r:
        return True
    if catalog_miss_reply and catalog_miss_reply.lower() in r:
        return True
    if LLM_REFUSAL_NOT_IN_FACTS.lower() in r:
        return True
    if "human" in r and any(
        k in r for k in ("connect", "agent", "seller", "reach", "reply", "escalat")
    ):
        return True
    return False
