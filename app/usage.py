"""Spending on model calls: each call is recorded with its tokens and priced by the model that ran it.

OpenRouter bills each call and returns its exact cost. Other providers' calls are priced from the
public OpenRouter model catalog, which lists Anthropic, OpenAI and Google models at their vendors'
token prices (cache reads and writes, and long-prompt tiers, included). A call whose model isn't in
the catalog stays unpriced and is retried on the next summary.
"""

import logging
import re
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from . import db
from .config import BUDGETS, HTTP_HEADERS

log = logging.getLogger(__name__)

CATALOG_URL = "https://openrouter.ai/api/v1/models"
KEY_URL = "https://openrouter.ai/api/v1/key"
CATALOG_TTL = 24 * 3600
KEY_TTL = 60
# The catalog's vendor prefix for each direct provider's model ids.
VENDORS = {"anthropic": "anthropic", "openai": "openai", "gemini": "google"}
# How many periods the history chart shows.
HISTORY = {"day": 14, "week": 12, "month": 12}

_catalog: dict[str, dict] = {}  # model id, exact or normalized (see _norm) -> pricing
_catalog_at = 0.0
_key_budgets: dict[str, tuple[float, list]] = {}  # OpenRouter key -> (when read, its budgets)


def _norm(model_id: str) -> str:
    """Matches a provider's id to the catalog's: "anthropic/claude-opus-4-8-20260528" and
    "anthropic/claude-opus-4.8" both become "anthropic/claude-opus-4-8"."""
    vendor, _, name = model_id.lower().rpartition("/")
    name = re.sub(r"-(\d{8}|\d{4}-\d{2}-\d{2})$", "", name.removeprefix("models/"))
    return f"{vendor}/{name.replace('.', '-')}"


async def _load_catalog():
    global _catalog, _catalog_at
    if _catalog and time.time() - _catalog_at < CATALOG_TTL:
        return
    async with httpx.AsyncClient(timeout=20, headers=HTTP_HEADERS) as client:
        r = await client.get(CATALOG_URL)
        r.raise_for_status()
    catalog = {}
    for m in r.json()["data"]:
        catalog[m["id"]] = m["pricing"]
        if ":" not in m["id"]:  # :free, :batch, ... are priced differently from the base model
            for alias in (m["id"], m.get("canonical_slug") or ""):
                catalog.setdefault(_norm(alias), m["pricing"])
    _catalog, _catalog_at = catalog, time.time()


def _pricing(provider: str, model: str) -> dict | None:
    if provider == "openrouter":
        return _catalog.get(model) or _catalog.get(_norm(model))
    full = f"{VENDORS[provider]}/{model}"
    # Gemini previews ("gemini-3.8-flash-preview-09-2026") fall back to the released model's price.
    return (_catalog.get(full) or _catalog.get(_norm(full))
            or _catalog.get(_norm(re.sub(r"-(preview|exp|latest)\b.*$", "", full))))


def _cost(pricing: dict, u: dict) -> float | None:
    """USD for one call. Token counts follow providers/base.py: input includes cached reads and
    cache writes; cache_write_1h is the part of cache_write written with the 1-hour TTL."""
    # Long-prompt tiers apply to the whole call. (Time-of-day discounts aren't modelled.)
    tiers = [o for o in pricing.get("overrides") or []
             if "utc_days" not in o and u["input"] >= o.get("min_prompt_tokens", 0)]
    if tiers:
        pricing = {**pricing, **max(tiers, key=lambda o: o.get("min_prompt_tokens", 0))}
    prompt = float(pricing.get("prompt") or 0)
    completion = float(pricing.get("completion") or 0)
    if prompt < 0 or completion < 0:  # e.g. a router whose price depends on the model it picks
        return None

    def rate(key, default):
        return float(pricing[key]) if pricing.get(key) else default

    write_5m = rate("input_cache_write", prompt)
    write_1h = u.get("cache_write_1h", 0)
    uncached = max(0, u["input"] - u["cached_input"] - u["cache_write"])
    return (uncached * prompt + u["cached_input"] * rate("input_cache_read", prompt)
            + (u["cache_write"] - write_1h) * write_5m + write_1h * rate("input_cache_write_1h", write_5m)
            + u["output"] * completion)


async def price(provider: str, model: str, u: dict) -> float | None:
    try:
        await _load_catalog()
    except Exception as e:  # offline: priced on a later summary
        log.warning("Could not load model prices: %s", e)
    pricing = _pricing(provider, model)
    return _cost(pricing, u) if pricing else None


async def record(user_id: str, provider: str, model: str, u: dict, *, kind: str, paper_id: str | None = None,
                 conversation_id: str | None = None):
    """Never raises: a failure here must not break the answer or analysis being recorded."""
    try:
        cost = u.get("cost")  # billed by the provider (OpenRouter)
        if cost is None:
            cost = await price(provider, model, u)
        db.add_usage({
            "created_at": time.time(), "provider": provider, "model": model, "kind": kind,
            "paper_id": paper_id, "conversation_id": conversation_id, "input": u.get("input", 0),
            "cached_input": u.get("cached_input", 0), "cache_write": u.get("cache_write", 0),
            "output": u.get("output", 0), "cost": cost, "user_id": user_id,
        })
    except Exception:
        log.exception("Could not record usage for %s %s", provider, model)


async def _price_missing():
    rows = db.list_usage(None, unpriced_only=True)
    if not rows:
        return
    costs = {}
    for r in rows:
        c = await price(r["provider"], r["model"], r)
        if c is not None:
            costs[r["id"]] = c
    db.set_usage_costs(costs)


def _period_start(d: date, period: str) -> date:
    if period == "week":
        return d - timedelta(days=d.weekday())
    if period == "month":
        return d.replace(day=1)
    return d


def _previous(start: date, period: str) -> date:
    if period == "month":
        return (start - timedelta(days=1)).replace(day=1)
    return start - timedelta(days=7 if period == "week" else 1)


async def _openrouter_budgets(key: str | None) -> list[dict]:
    """The OpenRouter key's credit limits: spending there counts every app using the key."""
    if not key:
        return []
    read_at, cached = _key_budgets.get(key, (0.0, []))
    if time.time() - read_at < KEY_TTL:
        return cached
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(KEY_URL, headers={"Authorization": f"Bearer {key}"})
            r.raise_for_status()
        data = r.json()["data"]
    except Exception as e:
        log.warning("Could not read the OpenRouter key's limits: %s", e)
        return cached
    budgets = [
        {"source": "OpenRouter key", "limit": b["limit_usd"], "spent": b["spend_usd"],
         "reset": b.get("reset_interval"), "resets_at": b.get("resets_at")}
        for b in data.get("budgets") or [] if b.get("limit_usd") is not None
    ]
    if not budgets and data.get("limit") is not None:  # keys from before per-interval budgets
        budgets = [{"source": "OpenRouter key", "limit": data["limit"], "spent": data.get("usage") or 0,
                    "reset": data.get("limit_reset"), "resets_at": None}]
    _key_budgets[key] = (time.time(), budgets)
    return budgets


async def summary(user: dict, tz_name: str | None) -> dict:
    """The user's spending per day, week and month in their time zone: the current period, recent
    periods for a chart, and the current period by model, with any budgets (BUDGETS are the
    server keys' budgets, so only admins see them)."""
    try:
        tz = ZoneInfo(tz_name) if tz_name else None
    except (ZoneInfoNotFoundError, ValueError):
        tz = None
    await _price_missing()
    today = datetime.now(tz).date()
    starts = {}  # period -> period starts, newest first
    for period, n in HISTORY.items():
        s = [_period_start(today, period)]
        while len(s) < n:
            s.append(_previous(s[-1], period))
        starts[period] = s
    since = min(s[-1] for s in starts.values())
    rows = db.list_usage(user["id"], since=datetime.combine(since, datetime.min.time(), tz).timestamp())
    budgets = BUDGETS if user["is_admin"] else {}

    out = {"currency": "USD", "total": db.usage_total(user["id"]), "periods": {},
           "provider_budgets": await _openrouter_budgets(user["keys"].get("openrouter"))}
    for period, period_starts in starts.items():
        current = period_starts[0]
        history = {s: 0.0 for s in period_starts}
        models: dict[tuple, dict] = {}
        calls = unpriced = 0
        for r in rows:
            start = _period_start(datetime.fromtimestamp(r["created_at"], tz).date(), period)
            if start in history:
                history[start] += r["cost"] or 0
            if start != current:
                continue
            calls += 1
            unpriced += r["cost"] is None
            m = models.setdefault((r["provider"], r["model"]), {
                "provider": r["provider"], "model": r["model"], "cost": 0.0, "calls": 0, "unpriced": 0,
                "input": 0, "output": 0})
            m["cost"] += r["cost"] or 0
            m["calls"] += 1
            m["unpriced"] += r["cost"] is None
            m["input"] += r["input"]
            m["output"] += r["output"]
        out["periods"][period] = {
            "start": current.isoformat(), "cost": history[current], "calls": calls, "unpriced": unpriced,
            "history": [{"start": s.isoformat(), "cost": history[s]} for s in reversed(period_starts)],
            "models": sorted(models.values(), key=lambda m: -m["cost"]),
            "budget": budgets.get(period),
        }
    return out
