"""
pipeline/tools.py
JLL LLM tool definitions and handlers.

Tool names and parameter shapes taken from:
  - bot (1).py: IntegrationToolHandler, lines 13906-14100
  - integration (1).js: endpoint config block, lines 13920-13940

The 5 tools the LLM can call:
  1. search_properties
  2. get_property_details
  3. areas_by_budget
  4. submit_callback
  5. schedule_site_visit
"""

import asyncio
import json
import time
from typing import Any

from logger import get_logger
from pipeline import jll_client

# Full property-details cache TTL — 30 minutes per session
_DETAILS_CACHE_TTL = 1800

log = get_logger("tools")


# ── Tool schemas (passed to Groq as function definitions) ─────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_properties",
            "description": "Search JLL listings after collecting city, type, budget, location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Chennai, Bengaluru, or Hyderabad."},
                    "property_type": {"type": "string", "description": "Apartment, Villa, or Plot."},
                    "location": {"type": "string", "description": "Area in the city."},
                    "min_price": {"type": "string", "description": "Min price in rupees. Use '0' or omit if no lower bound. 50L=5000000, 1cr=10000000, 2cr=20000000."},
                    "max_price": {"type": "string", "description": "Max price in rupees. 50L=5000000, 80L=8000000, 1cr=10000000, 1.5cr=15000000, 2cr=20000000, 3cr=30000000. Single budget → use as max_price only, never double it."},
                    "bedrooms": {"type": "string", "description": "BHK count, e.g. '2'."},
                    "page": {"type": "string", "description": "Page number, start '1'.", "default": "1"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_property_details",
            "description": "Get full details for a specific named property. Call immediately when the caller names a project — no other information needed.",  # CALLBACK-TAG-v1
            "parameters": {
                "type": "object",
                "properties": {
                    "property_id": {"type": "string", "description": "Property slug/ID."},
                    "city": {"type": "string", "description": "City."},
                },
                "required": ["property_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "areas_by_budget",
            "description": "Find areas within budget when search returns 0 results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City."},
                    "property_type": {"type": "string", "description": "Type."},
                    "min_price": {"type": "string", "description": "Min rupees."},
                    "max_price": {"type": "string", "description": "Max rupees."},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_callback",
            "description": "Request a callback. Collect name and phone first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Full name."},
                    "phone": {"type": "string", "description": "Phone number."},
                    "city": {"type": "string", "description": "City."},
                    "property_id": {"type": "string", "description": "Property ID if known."},
                },
                "required": ["name", "phone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_site_visit",
            "description": "Schedule a site visit. Collect name, phone, date first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Full name."},
                    "phone": {"type": "string", "description": "Phone."},
                    "property_id": {"type": "string", "description": "Property ID."},
                    "preferred_date": {"type": "string", "description": "Visit date."},
                },
                "required": ["name", "phone", "property_id"],
            },
        },
    },
]


# ── Tool handler ──────────────────────────────────────────────────────────────

class JLLToolHandler:
    """
    Executes LLM tool calls and formats results for voice delivery.
    Mirrors IntegrationToolHandler in bot (1).py, simplified for terminal use.
    """

    def __init__(self):
        # Session memory — mirrors _gathered_search in bot (1).py
        self.gathered: dict = {}
        # Shown property slugs for pagination dedup — mirrors _shown_slugs in bot (1).py
        self.shown_slugs: set = set()
        # Current page number
        self.current_page: int = 1
        # Last search key for detecting "show more" vs fresh search
        self._last_search_key: str = ""
        # Shallow search summaries keyed by slug (from search results)
        self._property_details_map: dict = {}
        # Full details cache: slug → {"data": {...}, "ts": float}
        # Populated by background prefetch right after search returns.
        self._full_details_cache: dict = {}
        # Ordered list of properties from the last search, preserving position
        # so "first property / second one / that project" resolve correctly.
        self._search_results_ordered: list[dict] = []

    @staticmethod
    def _to_int(val: Any) -> int | None:
        """Safely coerce a string or int to int; return None if empty/invalid."""
        if val is None or val == "":
            return None
        try:
            return int(float(str(val)))
        except (ValueError, TypeError):
            return None

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _get_cached_details(self, slug: str) -> dict | None:
        """Return cached full details if present and not expired."""
        entry = self._full_details_cache.get(slug)
        if not entry:
            return None
        if time.monotonic() - entry["ts"] > _DETAILS_CACHE_TTL:
            del self._full_details_cache[slug]
            log.debug("[CACHE] Expired details for %s", slug)
            return None
        return entry["data"]

    def _set_cached_details(self, slug: str, data: dict) -> None:
        """Store full details in cache, evicting oldest if cache is large."""
        if len(self._full_details_cache) >= 50:
            oldest = min(self._full_details_cache, key=lambda k: self._full_details_cache[k]["ts"])
            del self._full_details_cache[oldest]
        self._full_details_cache[slug] = {"data": data, "ts": time.monotonic()}

    async def _prefetch_details(self, slug: str, city: str) -> None:
        """Background task: fetch full property details and warm the cache."""
        try:
            result = await jll_client.get_property_details(property_id=slug, city=city)
            data = result.get("data") or {}
            if data:
                self._set_cached_details(slug, data)
                log.debug("[PREFETCH] Details cached for %s", slug)
            else:
                log.debug("[PREFETCH] Empty response for %s — not cached", slug)
        except Exception as exc:
            log.warning("[PREFETCH] Failed for %s: %s", slug, exc)

    def _search_key(self, args: dict) -> str:
        return "|".join([
            str(args.get("city") or "").lower(),
            str(args.get("location") or "").lower(),
            str(args.get("property_type") or "").lower(),
            str(args.get("min_price") or ""),
            str(args.get("max_price") or ""),
        ])

    async def handle(self, tool_name: str, args: dict) -> str:
        """
        Dispatch a tool call and return a voice-ready result string.
        Called by the pipeline after the LLM emits a function_call frame.
        """
        # Logging is handled by agent.py tool handler wrapper

        if tool_name == "search_properties":
            return await self._handle_search(args)
        elif tool_name == "get_property_details":
            return await self._handle_details(args)
        elif tool_name == "areas_by_budget":
            return await self._handle_areas(args)
        elif tool_name == "submit_callback":
            return await self._handle_callback(args)
        elif tool_name == "schedule_site_visit":
            return await self._handle_site_visit(args)
        else:
            return f"Unknown tool: {tool_name}"

    async def _handle_search(self, args: dict) -> str:
        # Coerce numeric fields (LLM may return them as strings)
        min_price = self._to_int(args.get("min_price"))
        max_price = self._to_int(args.get("max_price"))
        page_raw = self._to_int(args.get("page")) or 1

        # Update gathered search memory with coerced values
        for field in ("city", "property_type", "location", "bedrooms"):
            if args.get(field):
                self.gathered[field] = args[field]
        if min_price is not None:
            self.gathered["min_price"] = min_price
        if max_price is not None:
            self.gathered["max_price"] = max_price

        # Detect "show more" vs fresh search
        new_key = self._search_key(args)
        if new_key == self._last_search_key and self.shown_slugs:
            self.current_page += 1
        else:
            self.current_page = page_raw
            self.shown_slugs.clear()
            self._last_search_key = new_key

        result = await jll_client.search_properties(
            city=args.get("city", ""),
            property_type=args.get("property_type", ""),
            location=args.get("location", ""),
            min_price=min_price,
            max_price=max_price,
            bedrooms=str(args.get("bedrooms") or ""),
            page=self.current_page,
            limit=3,
            exclude_slugs=list(self.shown_slugs),
        )

        items = result.get("data") or []
        has_more = result.get("has_more", False)
        # API may return total count under various keys
        total_count = (
            result.get("total")
            or result.get("total_count")
            or result.get("count")
            or result.get("totalCount")
        )

        if not items:
            self.gathered["result_count"] = 0
            return (
                "SEARCH_RESULT: zero_results\n"
                f"city={args.get('city')} location={args.get('location')} "
                f"type={args.get('property_type')} budget={args.get('min_price')}-{args.get('max_price')}\n"
                "No properties found matching these criteria."
            )

        # Build total string: use API total if available, else estimate from has_more
        if total_count:
            total_str = str(total_count)
        elif has_more:
            total_str = f"more than {len(items)}"
        else:
            total_str = str(len(items))

        lines = [f"SEARCH_RESULT: {total_str} properties match this search (showing top {len(items)})"]
        over_budget_count = 0
        for i, p in enumerate(items, 1):
            slug = p.get("Project_Slug") or p.get("project_slug") or p.get("property_id") or ""
            name = p.get("Project_Name_Original") or p.get("Project_Name") or p.get("name") or "?"
            location = p.get("Location") or p.get("location") or p.get("Micro_Market") or ""
            price = p.get("starting_price") or p.get("price") or ""
            configs = p.get("configurations") or p.get("configs") or []
            bhk_types = list({
                c.get("Config_Type") or c.get("type") or ""
                for c in configs
                if c.get("Config_Type") or c.get("type")
            })
            bhk_summary = "/".join(sorted(bhk_types)[:3]) or ""
            price_int = self._to_int(price)
            over_budget = (
                max_price is not None
                and price_int is not None
                and price_int > max_price
            )
            if over_budget:
                over_budget_count += 1
            budget_tag = " [OVER BUDGET]" if over_budget else ""
            if slug:
                self.shown_slugs.add(slug)
                self._property_details_map[slug] = p
            lines.append(
                f"{i}. name={name} | location={location} | bhk={bhk_summary} | "
                f"price={price} | property_id={slug}{budget_tag}"
            )

        if max_price is not None and over_budget_count == len(items):
            lines.append(
                f"NOTE: ALL {len(items)} results exceed stated max_price of {max_price}. "
                "No in-budget options found — inform the caller there are no properties within their budget."
            )

        self.gathered["result_count"] = len(items)

        # Rebuild ordered list on fresh search; extend it on "show more"
        city_val = args.get("city", self.gathered.get("city", ""))
        if new_key != self._last_search_key:
            self._search_results_ordered = []
        for p in items:
            slug = p.get("Project_Slug") or p.get("project_slug") or p.get("property_id") or ""
            name = p.get("Project_Name_Original") or p.get("Project_Name") or p.get("name") or ""
            if slug:
                self._search_results_ordered.append({"slug": slug, "name": name, "city": city_val})

        # Background-prefetch full details for every result so follow-up
        # questions ("tell me about the first one") are answered instantly.
        for p in items:
            slug = p.get("Project_Slug") or p.get("project_slug") or p.get("property_id") or ""
            if slug and not self._get_cached_details(slug):
                asyncio.create_task(self._prefetch_details(slug, city_val))

        return "\n".join(lines)

    async def _handle_details(self, args: dict) -> str:
        property_id = str(args.get("property_id") or "").strip()
        city = str(args.get("city") or self.gathered.get("city") or "").strip()

        # If the LLM passed a truncated slug (missing the -sqft / -bhk suffix),
        # resolve it to the full slug from the known property map or details cache.
        if property_id not in self._property_details_map and property_id not in self._full_details_cache:
            for known_slug in list(self._property_details_map.keys()):
                if known_slug.startswith(property_id + "-") or known_slug == property_id:
                    log.info("[TOOL] Partial slug '%s' resolved to '%s'", property_id, known_slug)
                    property_id = known_slug
                    break

        # Serve from pre-fetched cache if available (zero API latency)
        cached = self._get_cached_details(property_id)
        if cached:
            log.info("[TOOL] get_property_details cache HIT  slug=%s", property_id)
            data = cached
        else:
            log.info("[TOOL] get_property_details cache MISS slug=%s — fetching now", property_id)
            result = await jll_client.get_property_details(property_id=property_id, city=city)
            data = result.get("data") or self._property_details_map.get(property_id) or {}
            if data:
                self._set_cached_details(property_id, data)

        if not data:
            return f"PROPERTY_DETAILS: not_found property_id={property_id}"

        name = data.get("name") or data.get("Project_Name") or property_id
        location = data.get("location") or data.get("Location") or ""
        price = data.get("starting_price") or data.get("price") or "Price on request"
        possession = data.get("possession") or data.get("Possession") or ""
        developer = data.get("developer") or ""
        if isinstance(developer, dict):
            developer = developer.get("name") or ""

        configs = data.get("configs") or data.get("configurations") or []
        bhk_lines = []
        for c in configs[:4]:
            cfg_type = c.get("Config_Type") or c.get("type") or ""
            cfg_price = c.get("FinalPrice") or c.get("All_Price") or c.get("price") or ""
            cfg_area = c.get("super_builtup") or c.get("carpet_area") or c.get("area") or ""
            if cfg_type:
                bhk_lines.append(f"  {cfg_type}: {cfg_area} sqft @ {cfg_price}")

        lines = [
            f"PROPERTY_DETAILS: {name}",
            f"location={location}",
            f"starting_price={price}",
            f"developer={developer}",
            f"possession={possession}",
        ]
        if bhk_lines:
            lines.append("configurations:\n" + "\n".join(bhk_lines))

        return "\n".join(lines)

    async def _handle_areas(self, args: dict) -> str:
        result = await jll_client.areas_by_budget(
            city=args.get("city", self.gathered.get("city", "")),
            property_type=args.get("property_type", self.gathered.get("property_type", "")),
            min_price=self._to_int(args.get("min_price")) or self._to_int(self.gathered.get("min_price")),
            max_price=self._to_int(args.get("max_price")) or self._to_int(self.gathered.get("max_price")),
        )
        areas = result.get("areas") or []
        if not areas:
            return "AREAS_RESULT: No areas found within this budget."
        area_names = [a.get("name") or a.get("area") or "" for a in areas[:5] if a.get("name") or a.get("area")]
        return f"AREAS_RESULT: Areas with properties in your budget: {', '.join(area_names)}"

    async def _handle_callback(self, args: dict) -> str:
        result = await jll_client.submit_callback(
            name=args.get("name", ""),
            phone=args.get("phone", ""),
            city=args.get("city", self.gathered.get("city", "")),
            property_id=args.get("property_id", ""),
        )
        if result.get("success"):
            return "CALLBACK_SUBMITTED: success The JLL team will contact you shortly."
        return f"CALLBACK_SUBMITTED: error {result.get('error', 'Unknown error')}"

    async def _handle_site_visit(self, args: dict) -> str:
        result = await jll_client.schedule_site_visit(
            name=args.get("name", ""),
            phone=args.get("phone", ""),
            property_id=args.get("property_id", ""),
            preferred_date=args.get("preferred_date", ""),
        )
        if result.get("success"):
            return "SITE_VISIT_SCHEDULED: success Your site visit has been booked."
        return f"SITE_VISIT_SCHEDULED: error {result.get('error', 'Unknown error')}"
