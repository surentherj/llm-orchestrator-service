import json

from app.services.llm.provider_router import provider_router

VALID_INTENTS = [
    "product_query",
    "add_to_cart",
    "remove_from_cart",
    "clear_cart",
    "view_cart",
    "place_order",
    "confirm_checkout",
    "reject_checkout",
    "order_status",
    "order_history",
    "order_issue",
    "store_info",
    "human_handoff",
    "thanks",
    "greeting",
    "explain",
    "fallback",
]

# State: lp,li,lr,wc,hf = last listing, intent, reply trim, awaiting checkout confirm, invited human last turn.
INTENT_PROMPT = """Social/chat commerce router. JSON only.

Intents:
product_query|add_to_cart|remove_from_cart|clear_cart|view_cart|place_order|confirm_checkout|reject_checkout|order_status|order_history|order_issue|store_info|human_handoff|thanks|greeting|explain|fallback
confirm_checkout ONLY if State.wc=true (pending checkout). Never use confirm_checkout for ok/yes after unrelated replies.
If State.hf=true: connect/agent/seller/human/ok to reach person→human_handoff (not confirm_checkout).
If State.wc=true: yes/ok/confirm/proceed→confirm_checkout; no/cancel/stop→reject_checkout.
order_issue=cancel/complaint/damaged/refund. store_info=shipping/returns/payment/hours/contact.

entities:
product_name,tags[],price_threshold,price_condition(below|above|exact),require_in_stock(bool)
quantity(number|null) // for add_to_cart, default 1
store_topic(shipping|returns|payment|contact|hours|null)
issue_kind(cancel|complaint|null)

Rules: State.lp for it/this/that/same. New listing clears carryover. budget→price_threshold+price_condition.

Output:{"intents":[],"entities":{"product_name":null,"tags":[],"price_threshold":null,"price_condition":null,"require_in_stock":false,"quantity":null,"store_topic":null,"issue_kind":null},"normalized_query":"","requires_llm_answer":false}"""


class IntentRouter:
    async def detect(self, message, context_state=None, model_key="gemini-3.1-flash-lite-preview"):
        parts = [INTENT_PROMPT]
        if context_state:
            parts.append(f"State:{context_state}")
        parts.append(f"Msg:{message}")
        prompt = "\n".join(parts)

        result = await provider_router.generate(model_key, prompt)

        parsed_json = None
        try:
            start = result.find("{")
            end = result.rfind("}") + 1
            if start != -1 and end != 0:
                parsed_json = json.loads(result[start:end])
        except Exception:
            pass

        if parsed_json:
            extracted = [i for i in parsed_json.get("intents", []) if i in VALID_INTENTS]
            if not extracted:
                extracted = ["fallback"]
            parsed_json["intents"] = extracted
            ent = parsed_json.get("entities") or {}
            st = ent.get("store_topic")
            if st not in (None, "shipping", "returns", "payment", "contact", "hours"):
                st = None
            ik = ent.get("issue_kind")
            if ik not in (None, "cancel", "complaint"):
                ik = None
            qty = ent.get("quantity")
            try:
                qty = int(qty) if qty is not None else None
                if qty is not None and qty < 1:
                    qty = 1
            except (TypeError, ValueError):
                qty = None
            parsed_json["entities"] = {
                "product_name": ent.get("product_name"),
                "tags": ent.get("tags") if isinstance(ent.get("tags"), list) else [],
                "price_threshold": ent.get("price_threshold"),
                "price_condition": ent.get("price_condition"),
                "require_in_stock": bool(ent.get("require_in_stock")),
                "quantity": qty,
                "store_topic": st,
                "issue_kind": ik,
            }
            if "normalized_query" not in parsed_json:
                parsed_json["normalized_query"] = message
            parsed_json["requires_llm_answer"] = bool(parsed_json.get("requires_llm_answer"))
            return parsed_json

        return {
            "intents": ["fallback"],
            "entities": {
                "product_name": None,
                "tags": [],
                "price_threshold": None,
                "price_condition": None,
                "require_in_stock": False,
                "quantity": None,
                "store_topic": None,
                "issue_kind": None,
            },
            "normalized_query": message,
            "requires_llm_answer": False,
        }


intent_router = IntentRouter()
