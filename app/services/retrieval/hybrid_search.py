import hashlib
import json
from collections import OrderedDict

# Replace with your catalog API/DB. Shape: id, title, price, stock, tags[], quality_note (optional).
# Demo rows are intentionally generic — not a real assortment.
STUB_CATALOG = [
    {
        "id": 1,
        "title": "Demo Listing A",
        "price": 100,
        "stock": 10,
        "tags": ["sample", "demo"],
        "quality_note": "Placeholder detail for integration tests.",
    },
    {
        "id": 2,
        "title": "Demo Listing B",
        "price": 250,
        "stock": 0,
        "tags": ["sample", "demo"],
        "quality_note": "Placeholder detail for integration tests.",
    },
    {
        "id": 3,
        "title": "Demo Listing C",
        "price": 500,
        "stock": 5,
        "tags": ["sample", "demo"],
        "quality_note": "Placeholder detail for integration tests.",
    },
]


def _normalize_query_text(query: str) -> str:
    return " ".join(
        raw.strip(".,?!'\"").lower()
        for raw in query.split()
        if raw.strip(".,?!'\"")
    )


def _item_lexicon(item: dict) -> set:
    words = set(item["title"].lower().split())
    for t in item.get("tags") or []:
        words.add(t.lower())
    return words


class HybridSearch:
    """Rank by title/tags/query overlap + price + stock filters. Category-agnostic."""

    def __init__(self):
        self._result_cache: OrderedDict[str, list] = OrderedDict()
        self._cache_max = 256

    def _cache_key(self, query: str, entities: dict, strict_price_filter: bool) -> str:
        payload = json.dumps(
            {"q": query, "e": entities, "s": strict_price_filter},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    async def search(
        self,
        query: str,
        entities: dict | None = None,
        strict_price_filter: bool = True,
    ):
        if entities is None:
            entities = {}

        query = _normalize_query_text(query)
        ck = self._cache_key(query, entities, strict_price_filter)
        if ck in self._result_cache:
            self._result_cache.move_to_end(ck)
            return [dict(r) for r in self._result_cache[ck]]

        price_threshold = entities.get("price_threshold")
        price_condition = entities.get("price_condition")
        require_in_stock = bool(entities.get("require_in_stock"))
        entity_tags = [t.lower() for t in (entities.get("tags") or []) if t]

        stop = {"", "a", "an", "the", "for", "me", "show", "get", "need", "want", "any"}
        query_parts = set(query.split()) - stop

        results = []
        for item in STUB_CATALOG:
            if require_in_stock and item.get("stock", 0) <= 0:
                continue

            if price_threshold and strict_price_filter:
                if price_condition == "below" and item["price"] >= price_threshold:
                    continue
                if price_condition == "above" and item["price"] <= price_threshold:
                    continue
                if price_condition == "exact" and item["price"] != price_threshold:
                    continue

            lex = _item_lexicon(item)
            tag_set = [t.lower() for t in item.get("tags") or []]

            if query_parts:
                overlap = len(query_parts.intersection(lex))
                sem = min(1.0, overlap / max(1, len(query_parts)))
                extra = query_parts - lex
                fuzzy = 0.0
                for qtok in extra:
                    if len(qtok) < 3:
                        continue
                    if any(qtok in tw or tw in qtok for tw in lex):
                        fuzzy += 0.2
                fuzzy = min(1.0, fuzzy)
            else:
                sem, fuzzy = 0.4, 0.0

            if entity_tags:
                tag_hits = sum(
                    1 for t in entity_tags if t in tag_set or t in lex
                )
                tag_score = min(1.0, tag_hits / max(1, len(entity_tags)))
            else:
                tag_score = 0.45

            score = 0.45 * sem + 0.35 * tag_score + 0.2 * fuzzy

            if score > 0 or not entities:
                results.append({**item, "score": round(score, 4)})

        results.sort(key=lambda x: x["score"], reverse=True)
        self._result_cache[ck] = [dict(r) for r in results]
        self._result_cache.move_to_end(ck)
        while len(self._result_cache) > self._cache_max:
            self._result_cache.popitem(last=False)

        return results


hybrid_search = HybridSearch()
