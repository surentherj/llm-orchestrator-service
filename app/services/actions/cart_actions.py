def normalize_cart_items(cart) -> list[dict]:
    out: list[dict] = []
    for x in cart or []:
        if isinstance(x, dict) and x.get("title"):
            out.append(
                {
                    "title": str(x["title"]),
                    "price": float(x.get("price") or 0),
                    "qty": max(1, int(x.get("qty") or 1)),
                    "id": x.get("id"),
                }
            )
        elif isinstance(x, str) and x.strip():
            out.append(
                {"title": x.strip(), "price": 0.0, "qty": 1, "id": None}
            )
    return out


def _same_line(a: dict, b: dict) -> bool:
    return a["title"].lower() == b["title"].lower() and float(a["price"]) == float(
        b["price"]
    )


def cart_line_strings(cart) -> list[str]:
    lines, _, _ = cart_totals(cart)
    return lines


def cart_totals(cart) -> tuple[list[str], float, int]:
    """Returns (human lines, subtotal, total_qty)."""
    items = normalize_cart_items(cart)
    out_lines: list[str] = []
    subtotal = 0.0
    total_qty = 0
    for it in items:
        lt = it["price"] * it["qty"]
        subtotal += lt
        total_qty += it["qty"]
        p = it["price"]
        p_txt = f"₹{int(p)}" if p == int(p) else f"₹{p:.2f}"
        lt_txt = f"₹{int(lt)}" if lt == int(lt) else f"₹{lt:.2f}"
        out_lines.append(f"{it['title']} × {it['qty']} @ {p_txt} = {lt_txt}")
    return out_lines, subtotal, total_qty


def format_cart_reply(greeting_prefix: str, cart) -> str:
    lines, subtotal, total_qty = cart_totals(cart)
    if not lines:
        return f"{greeting_prefix}Your cart is empty."
    body = "\n".join(lines)
    st = f"₹{int(subtotal)}" if subtotal == int(subtotal) else f"₹{subtotal:.2f}"
    return (
        f"{greeting_prefix}Your cart ({total_qty} items):\n{body}\n"
        f"Subtotal: {st}"
    )


def _matches_title(title: str, needle: str) -> bool:
    a, b = title.lower(), needle.lower()
    return a in b or b in a


async def add_to_cart(session, intent_data: dict):
    cart = normalize_cart_items(session.get("cart"))
    title = intent_data.get("product_name")
    price = float(intent_data.get("price") if intent_data.get("price") is not None else 0)
    qty = intent_data.get("quantity")
    try:
        qty = max(1, int(qty if qty is not None else 1))
    except (TypeError, ValueError):
        qty = 1

    if not title:
        session["cart"] = cart
        return {"message": "No product", "cart": cart, "added": False}

    new_item = {
        "title": title,
        "price": price,
        "qty": qty,
        "id": intent_data.get("product_id"),
    }
    merged = False
    for i, it in enumerate(cart):
        if _same_line(it, new_item):
            cart[i] = {**it, "qty": it["qty"] + qty}
            merged = True
            break
    if not merged:
        cart.append(new_item)
    session["cart"] = cart
    return {"message": "Added to cart", "cart": cart, "added": True, "merged": merged}


async def remove_from_cart(session, product_name: str | None):
    cart = normalize_cart_items(session.get("cart"))
    if not cart:
        return {"removed": False, "cart": cart, "message": "Cart is empty"}
    if not product_name:
        return {"removed": False, "cart": cart, "message": "Say which item to remove"}

    removed = False
    for i, it in enumerate(cart):
        if _matches_title(it["title"], product_name):
            if it["qty"] > 1:
                cart[i] = {**it, "qty": it["qty"] - 1}
            else:
                cart.pop(i)
            removed = True
            break
    session["cart"] = cart
    return {
        "removed": removed,
        "cart": cart,
        "message": "Removed" if removed else "No matching item in cart",
    }


async def clear_cart(session):
    session["cart"] = []
    return {"cart": [], "message": "Cart cleared"}
