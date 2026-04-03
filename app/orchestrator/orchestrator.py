import json
import uuid

from app.core.config import settings
from app.orchestrator import conversation as conv
from app.orchestrator.intent_router import intent_router
from app.services.actions.cart_actions import (
    add_to_cart,
    clear_cart,
    format_cart_reply,
    normalize_cart_items,
    remove_from_cart,
)
from app.services.knowledge import store_replies
from app.services.llm.provider_router import provider_router
from app.services.memory.session_store import session_store
from app.services.retrieval.hybrid_search import hybrid_search

_EXPLAIN_TEMPLATE_SCORE_MIN = 0.35
_UNRESOLVED_STREAK_MAX = 2

_CONTEXT_LLM_PROMPT = (
    "Use ONLY the Facts JSON. If it does not contain the answer, output exactly: "
    "Not in provided catalog data.\n"
    "Max 2 short sentences. No new products, prices, or policies.\n"
    "Facts:{facts}\nQ:{q}\n"
)

_FALLBACK_REPLY = (
    "I can help with products, cart, orders, shipping, returns, or payment basics. "
    "What do you need?"
)
_NO_CATALOG_REPLY = "No matching listing in this seller’s catalog."
_HUMAN_HINT = "Reply HUMAN anytime to reach the seller in person."


def _compact_state_for_prompt(session: dict) -> str:
    raw = session.get("context_state", {})
    lp = raw.get("lp") or raw.get("last_product_mentioned") or session.get("last_product", "")
    li = raw.get("li") or raw.get("last_intent", "")
    lr = (raw.get("lr") or raw.get("last_ai_reply", ""))[:180]
    d = {k: v for k, v in {"lp": lp, "li": li, "lr": lr}.items() if v}
    if session.get(conv.AWAITING_ORDER_CONFIRM):
        d["wc"] = True
    if session.get(conv.HUMAN_FOLLOWUP_PENDING):
        d["hf"] = True
    return json.dumps(d, separators=(",", ":"))


def _money(x: float) -> str:
    return f"₹{int(x)}" if float(x) == int(float(x)) else f"₹{x:.2f}"


def _order_detail_block(order: dict) -> str:
    lines = order.get("lines")
    if lines:
        parts = []
        for L in lines:
            lt = L.get("line_total", L.get("price", 0) * L.get("qty", 1))
            parts.append(
                f"• {L.get('title', '')} × {L.get('qty', 1)} @ {_money(L.get('price', 0))} = {_money(lt)}"
            )
        total = order.get("total", 0)
        cnt = order.get("item_count", sum(L.get("qty", 1) for L in lines))
        return "\n".join(parts) + f"\nTotal ({cnt} pcs): {_money(total)}"
    legacy = order.get("items") or []
    return ", ".join(legacy) if legacy else "(no line detail)"


def _facts_for_llm(results: list | None) -> str:
    if not results:
        return "[]"
    slim = []
    for r in results[:3]:
        slim.append(
            {
                "title": r.get("title"),
                "price": r.get("price"),
                "stock": r.get("stock"),
                "tags": r.get("tags"),
                "quality_note": r.get("quality_note"),
            }
        )
    return json.dumps(slim, separators=(",", ":"))


def _substantive_intents(intents: list) -> bool:
    return bool(
        set(intents)
        & {
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
            "explain",
        }
    )


def _product_line(greeting_prefix: str, item: dict) -> str:
    st = item.get("stock", 0)
    stock_txt = f" In stock: {st}." if st > 0 else " Currently out of stock."
    return f"{greeting_prefix}{item['title']} — ₹{item['price']}.{stock_txt}"


def _reply_unresolved(reply: str | None) -> bool:
    if not reply:
        return True
    if _NO_CATALOG_REPLY in reply or _FALLBACK_REPLY in reply:
        return True
    if "Not in provided catalog data" in reply:
        return True
    return False


class Orchestrator:
    async def handle(
        self,
        conversation_id,
        message,
        model_key="gemini-3.1-flash-lite-preview",
    ):
        greeting_prefix = ""
        session = await session_store.get(conversation_id)
        context_state_str = _compact_state_for_prompt(session)

        intent_data = await intent_router.detect(
            message,
            context_state=context_state_str,
            model_key=model_key,
        )

        intents = intent_data.get("intents", [])
        entities = intent_data.get("entities", {})
        query = intent_data.get("normalized_query", message)

        ticket_id_turn: str | None = None

        def mint_ticket(reason: str) -> str:
            nonlocal ticket_id_turn
            if ticket_id_turn is None:
                ticket_id_turn = str(uuid.uuid4())[:8].upper()
                session.setdefault("escalation_tickets", []).append(
                    {
                        "id": ticket_id_turn,
                        "reason": reason,
                        "snippet": (message or "")[:280],
                    }
                )
            return ticket_id_turn

        if "add_to_cart" in intents and not entities.get("product_name"):
            last_product = session.get("last_product")
            if last_product:
                entities["product_name"] = last_product

        if "remove_from_cart" in intents and not entities.get("product_name"):
            lp = session.get("last_product")
            if lp:
                entities["product_name"] = lp

        intents = conv.remap_intents_after_detection(intents, session)

        if "human_handoff" in intents:
            session[conv.AWAITING_ORDER_CONFIRM] = False

        response_payload = {"intents": intents}
        results = None

        if "greeting" in intents and _substantive_intents(intents):
            greeting_prefix = "Hi! "

        _clears_pending_flows = (
            "product_query",
            "add_to_cart",
            "remove_from_cart",
            "clear_cart",
            "explain",
            "place_order",
            "store_info",
            "order_status",
            "order_history",
            "order_issue",
        )
        if session.get(conv.AWAITING_ORDER_CONFIRM) and any(
            i in intents for i in _clears_pending_flows
        ):
            session[conv.AWAITING_ORDER_CONFIRM] = False
        if any(i in intents for i in _clears_pending_flows):
            conv.clear_human_followup(session)

        if session.get(conv.AWAITING_ORDER_CONFIRM):
            conf = "confirm_checkout" in intents
            rej = "reject_checkout" in intents
            if not conf and not rej:
                other = set(intents) - {
                    "confirm_checkout",
                    "reject_checkout",
                    "greeting",
                    "thanks",
                    "fallback",
                }
                if not other:
                    if conv.user_message_confirms_checkout(message):
                        conf = True
                    elif conv.user_message_rejects_checkout(message):
                        rej = True
            if conf:
                cart_c = normalize_cart_items(session.get("cart"))
                if not cart_c:
                    session[conv.AWAITING_ORDER_CONFIRM] = False
                    response_payload["reply"] = (
                        f"{greeting_prefix}Cart is empty — nothing to confirm."
                    )
                else:
                    order_id = str(uuid.uuid4())[:8]
                    lines_out = []
                    subtotal = 0.0
                    n = 0
                    for it in cart_c:
                        lt = it["price"] * it["qty"]
                        subtotal += lt
                        n += it["qty"]
                        lines_out.append(
                            {
                                "title": it["title"],
                                "price": it["price"],
                                "qty": it["qty"],
                                "line_total": lt,
                            }
                        )
                    new_order = {
                        "order_id": order_id,
                        "lines": lines_out,
                        "total": subtotal,
                        "item_count": n,
                        "status": "pending shipment",
                    }
                    orders = session.get("orders", [])
                    orders.append(new_order)
                    session["orders"] = orders
                    session["cart"] = []
                    session[conv.AWAITING_ORDER_CONFIRM] = False
                    detail = _order_detail_block(new_order)
                    response_payload["order"] = new_order
                    response_payload["cart"] = []
                    response_payload["reply"] = (
                        f"{greeting_prefix}Order {order_id} confirmed.\n{detail}\n"
                        f"Status: {new_order['status']} — we’ll notify you when it updates."
                    )
            elif rej:
                session[conv.AWAITING_ORDER_CONFIRM] = False
                response_payload["reply"] = (
                    f"{greeting_prefix}Order not placed — your cart is unchanged.\n"
                    f"{format_cart_reply('', normalize_cart_items(session.get('cart'))).strip()}"
                )
        elif ("confirm_checkout" in intents or "reject_checkout" in intents) and (
            "reply" not in response_payload
        ):
            if "confirm_checkout" in intents:
                response_payload["reply"] = (
                    f"{greeting_prefix}No checkout waiting. Add items, then say PLACE ORDER."
                )
            else:
                response_payload["reply"] = (
                    f"{greeting_prefix}No checkout in progress."
                )

        needs_hybrid = (
            "product_query" in intents
            or "explain" in intents
            or (
                "add_to_cart" in intents
                and (
                    entities.get("price_threshold") is not None
                    or entities.get("product_name")
                    or entities.get("tags")
                )
            )
        )
        if needs_hybrid:
            strict_filter = not (
                "add_to_cart" in intents and entities.get("price_threshold") is not None
            )
            search_query = query
            pn = entities.get("product_name")
            if pn and isinstance(pn, str) and pn.lower() not in search_query.lower():
                search_query = f"{search_query} {pn}".strip()
            results = await hybrid_search.search(
                search_query, entities, strict_price_filter=strict_filter
            )
            response_payload["results"] = results

            if results:
                session["last_product"] = results[0]["title"]

        if "remove_from_cart" in intents and "reply" not in response_payload:
            session[conv.AWAITING_ORDER_CONFIRM] = False
            rm = await remove_from_cart(session, entities.get("product_name"))
            response_payload["cart"] = rm["cart"]
            if rm["removed"]:
                rest = format_cart_reply("", rm["cart"]).replace("Your cart is empty.", "").strip()
                if rest.startswith("Your cart"):
                    response_payload["reply"] = f"{greeting_prefix}Updated cart:\n{rest}"
                else:
                    response_payload["reply"] = (
                        f"{greeting_prefix}Removed one unit. {rest or 'Cart is now empty.'}"
                    )
            else:
                response_payload["reply"] = f"{greeting_prefix}{rm['message']}."

        if "clear_cart" in intents and "reply" not in response_payload:
            session[conv.AWAITING_ORDER_CONFIRM] = False
            cl = await clear_cart(session)
            response_payload["cart"] = cl["cart"]
            response_payload["reply"] = f"{greeting_prefix}Cart cleared."

        if "add_to_cart" in intents:
            can_add = True

            if results is not None:
                if len(results) == 0:
                    can_add = False
                    response_payload["reply"] = (
                        f"{greeting_prefix}Could not find a product matching those conditions."
                    )
                else:
                    top_item = results[0]
                    if top_item.get("stock", 0) <= 0:
                        can_add = False
                        response_payload["reply"] = (
                            f"{greeting_prefix}{top_item['title']} is out of stock."
                        )
                    else:
                        price_threshold = entities.get("price_threshold")
                        price_condition = entities.get("price_condition")

                        if price_threshold is not None:
                            price = top_item["price"]
                            failed = (
                                (price_condition == "below" and price >= price_threshold)
                                or (price_condition == "above" and price <= price_threshold)
                                or (price_condition == "exact" and price != price_threshold)
                            )

                            if failed:
                                can_add = False
                                response_payload["reply"] = (
                                    f"{greeting_prefix}Found {top_item['title']} for ₹{price}; "
                                    "does not meet your price condition."
                                )

                        if can_add:
                            entities["product_name"] = top_item["title"]

            if can_add:
                if not entities.get("product_name"):
                    can_add = False
                    if "reply" not in response_payload:
                        response_payload["reply"] = (
                            f"{greeting_prefix}Say which product to add, or browse one first."
                        )

                if can_add:
                    session[conv.AWAITING_ORDER_CONFIRM] = False
                    qty_add = entities.get("quantity")
                    try:
                        qty_add = max(1, int(qty_add if qty_add is not None else 1))
                    except (TypeError, ValueError):
                        qty_add = 1
                    if results and len(results) > 0:
                        avail = int(results[0].get("stock") or 0)
                        ri = results[0]
                        already = 0
                        for c in normalize_cart_items(session.get("cart")):
                            if (
                                c["title"].lower() == ri["title"].lower()
                                and float(c["price"]) == float(ri["price"])
                            ):
                                already += c["qty"]
                        if already + qty_add > avail:
                            can_add = False
                            response_payload["reply"] = (
                                f"{greeting_prefix}Only {avail} in stock for {ri['title']}; "
                                f"you have {already} in cart. Lower the quantity or remove some."
                            )
                    price_add = 0.0
                    pid = None
                    if can_add and results and len(results) > 0:
                        price_add = float(results[0]["price"])
                        pid = results[0].get("id")
                    if can_add:
                        action_resp = await add_to_cart(
                            session,
                            {
                                "product_name": entities.get("product_name"),
                                "price": price_add,
                                "quantity": qty_add,
                                "product_id": pid,
                            },
                        )
                        response_payload["cart"] = action_resp["cart"]
                        if "reply" not in response_payload:
                            qnote = f" (×{qty_add})" if qty_add != 1 else ""
                            response_payload["reply"] = (
                                f"{greeting_prefix}Added {entities.get('product_name')}{qnote} "
                                f"at {_money(price_add)}."
                            )

        if "greeting" in intents and not _substantive_intents(intents):
            response_payload["reply"] = (
                "Hello! How can I help you today?"
            )

        if "thanks" in intents and "reply" not in response_payload:
            response_payload["reply"] = f"{greeting_prefix}You're welcome — happy to help."

        if "store_info" in intents and "reply" not in response_payload:
            topic = entities.get("store_topic") or "contact"
            fn = store_replies.STORE_TOPIC_REPLIES.get(topic) or store_replies.contact_text
            response_payload["reply"] = f"{greeting_prefix}{fn()}"

        if "order_issue" in intents and "reply" not in response_payload:
            kind = entities.get("issue_kind") or "complaint"
            orders = session.get("orders", [])
            if kind == "cancel":
                if not orders:
                    response_payload["reply"] = (
                        f"{greeting_prefix}No order found to cancel. "
                        "If you paid elsewhere, say HUMAN to reach the seller."
                    )
                else:
                    tid = mint_ticket("cancel")
                    orders[-1]["cancel_requested"] = True
                    session["orders"] = orders
                    response_payload["escalation"] = {
                        "ticket_id": tid,
                        "status": "queued",
                        "type": "cancel",
                    }
                    response_payload["reply"] = (
                        f"{greeting_prefix}Cancel request #{tid} logged for your latest order. "
                        f"The seller will confirm via {settings.SELLER_CONTACT_HINT}."
                    )
            else:
                tid = mint_ticket("complaint")
                response_payload["escalation"] = {
                    "ticket_id": tid,
                    "status": "queued",
                    "type": "complaint",
                }
                response_payload["reply"] = (
                    f"{greeting_prefix}Sorry about the trouble. "
                    f"{store_replies.escalation_message(tid)}"
                )

        if "view_cart" in intents and "reply" not in response_payload:
            response_payload["reply"] = format_cart_reply(
                greeting_prefix, session.get("cart", [])
            )

        if "place_order" in intents and "reply" not in response_payload:
            cart_po = normalize_cart_items(session.get("cart", []))
            if not cart_po:
                response_payload["reply"] = (
                    f"{greeting_prefix}Cart is empty — add items before checkout."
                )
            else:
                session[conv.AWAITING_ORDER_CONFIRM] = True
                preview = format_cart_reply("", cart_po).strip()
                added_pref = (
                    f"Added {entities.get('product_name')}. "
                    if "add_to_cart" in intents and entities.get("product_name")
                    else ""
                )
                response_payload["reply"] = (
                    f"{greeting_prefix}{added_pref}Ready to place your order?\n{preview}\n\n"
                    "Reply **CONFIRM** to place it, or **NO** to keep editing your cart."
                )

        if "order_status" in intents and "reply" not in response_payload:
            orders = session.get("orders", [])
            if orders:
                recent = orders[-1]
                extra = ""
                if recent.get("cancel_requested"):
                    extra = "\nNote: cancel requested — seller pending."
                body = _order_detail_block(recent)
                response_payload["reply"] = (
                    f"{greeting_prefix}Latest order **{recent['order_id']}** — {recent['status']}.\n"
                    f"{body}{extra}"
                )
            else:
                response_payload["reply"] = f"{greeting_prefix}No orders yet."

        if "order_history" in intents and "reply" not in response_payload:
            orders = session.get("orders", [])
            if orders:
                blocks = []
                for o in orders:
                    ob = _order_detail_block(o)
                    blocks.append(f"**{o['order_id']}** ({o['status']})\n{ob}")
                response_payload["reply"] = (
                    f"{greeting_prefix}Your orders:\n\n" + "\n\n".join(blocks)
                )
            else:
                response_payload["reply"] = f"{greeting_prefix}No past orders."

        explain_from_kb = False
        if "explain" in intents and results and "reply" not in response_payload:
            top = results[0]
            if top.get("score", 0) >= _EXPLAIN_TEMPLATE_SCORE_MIN:
                qn = top.get("quality_note") or "See title, price, tags."
                explain_from_kb = True
                response_payload["reply"] = (
                    f"{greeting_prefix}{top['title']} (₹{top['price']}, stock {top.get('stock', 0)}): {qn}"
                )

        kb_sufficient = bool(
            results and results[0].get("score", 0) >= _EXPLAIN_TEMPLATE_SCORE_MIN
        )

        has_facts = bool(results)
        wants_llm = (
            "reply" not in response_payload
            and has_facts
            and (
                ("explain" in intents and not explain_from_kb)
                or (intent_data.get("requires_llm_answer") and not kb_sufficient)
            )
        )

        if "reply" not in response_payload and "fallback" in intents:
            response_payload["reply"] = _FALLBACK_REPLY

        if wants_llm:
            facts = _facts_for_llm(results)
            prompt = _CONTEXT_LLM_PROMPT.format(facts=facts, q=message)
            llm_reply = await provider_router.generate(model_key, prompt)
            response_payload["reply"] = f"{greeting_prefix}{llm_reply.strip()}"

        elif "reply" not in response_payload:
            if results:
                response_payload["reply"] = _product_line(greeting_prefix, results[0])
            else:
                response_payload["reply"] = f"{greeting_prefix}{_NO_CATALOG_REPLY}"

        if "human_handoff" in intents:
            tid = mint_ticket("human")
            response_payload["escalation"] = {
                "ticket_id": tid,
                "status": "queued",
                "type": "human",
            }
            esc = store_replies.escalation_message(tid)
            cur = response_payload.get("reply", "")
            if cur and esc not in cur:
                response_payload["reply"] = f"{cur}\n\n{esc}"
            else:
                response_payload["reply"] = esc
            conv.clear_human_followup(session)

        fr = response_payload.get("reply", "")
        if _reply_unresolved(fr):
            session[conv.UNRESOLVED_STREAK] = session.get(conv.UNRESOLVED_STREAK, 0) + 1
        else:
            session[conv.UNRESOLVED_STREAK] = 0

        if session.get(conv.UNRESOLVED_STREAK, 0) >= _UNRESOLVED_STREAK_MAX:
            response_payload["suggest_human"] = True
            if _HUMAN_HINT not in fr:
                response_payload["reply"] = f"{response_payload.get('reply', '')}\n\n{_HUMAN_HINT}"

        fr_out = response_payload.get("reply", "")
        if "human_handoff" not in intents and conv.reply_invites_human_followup(
            fr_out,
            catalog_miss_reply=_NO_CATALOG_REPLY,
            human_hint_reply=_HUMAN_HINT,
        ):
            conv.mark_human_followup_invited(session)

        session["context_state"] = {
            "lp": (entities.get("product_name") or session.get("last_product") or "")[
                :120
            ],
            "li": intents[0] if intents else "fallback",
            "lr": (response_payload.get("reply") or "")[:200],
            "lt": ticket_id_turn,
        }

        await session_store.set(conversation_id, session)
        return response_payload


orchestrator = Orchestrator()
