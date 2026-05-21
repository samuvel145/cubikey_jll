import os

_PROMPT_FILE = os.path.join(os.path.dirname(__file__), "..", "system_prompt.txt")


def _load_system_prompt() -> str:
    """Load system prompt from system_prompt.txt next to the project root."""
    try:
        with open(_PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        # Minimal fallback if file is missing
        return (
            "You are {assistant_name}, a voice sales assistant for JLL Homes. "
            "Help users find properties in Chennai, Bengaluru, or Hyderabad."
        )


JLL_SYSTEM_PROMPT = _load_system_prompt()



GATHER_PHASE_PROMPT = """CURRENT SESSION STATE
You are in the requirement-gathering phase. You have collected:
{gathered_summary}

NEXT STEP: {next_question}
Ask only this one question. Do not search yet.
"""


def build_system_prompt(assistant_name: str) -> str:
    """Return the fully formatted JLL system prompt.

    Safe for prompts that do NOT contain {assistant_name} (e.g. the current
    system_prompt.txt which hardcodes 'Riya').
    """
    try:
        prompt = JLL_SYSTEM_PROMPT.format(assistant_name=assistant_name)
    except KeyError:
        prompt = JLL_SYSTEM_PROMPT  # no placeholder — use as-is

    # Tell the LLM how to handle the hidden startup trigger
    prompt += (
        "\n\nSTARTUP: When the user message is exactly '[BEGIN]', "
        "deliver the OPENING message defined above. Do not mention '[BEGIN]' to the caller."
    )
    return prompt


def build_gather_hint(gathered: dict) -> str:
    """
    Return a short hint injected into the system prompt before each LLM call
    showing what has been collected and what single question to ask next.

    Question order matches the system prompt's mandatory sequence:
      STEP 1 → property_type
      STEP 2 → budget
      STEP 3 → location (always Chennai; city is implicit)

    City is omitted from next_question — this is a Chennai-only agent.
    """
    city = gathered.get("city", "Chennai")
    prop_type = gathered.get("property_type", "")
    location = gathered.get("location", "")
    min_price = gathered.get("min_price")
    max_price = gathered.get("max_price")
    bedrooms = gathered.get("bedrooms", "")

    parts = []
    if prop_type:
        parts.append(f"type={prop_type}")
    if location:
        parts.append(f"area={location}")
    if min_price or max_price:
        lo = f"{int(min_price):,}" if min_price else "0"
        hi = f"{int(max_price):,}" if max_price else "∞"
        parts.append(f"budget=₹{lo}–₹{hi}")
    if bedrooms:
        parts.append(f"bedrooms={bedrooms}")

    gathered_summary = ", ".join(parts) if parts else "nothing yet"

    # STEP 1 — property type first, always
    if not prop_type:
        next_q = "What type of property are you looking at in Chennai — apartment, villa, or plot?"
    # STEP 2 — budget second
    elif not (min_price or max_price):
        next_q = "What's your budget range?"
    # STEP 3 — location last, right before search
    elif not location:
        next_q = f"Which area in {city} are you looking at?"
    else:
        next_q = "All required fields are collected. Call search_properties immediately with no text."

    return GATHER_PHASE_PROMPT.format(
        gathered_summary=gathered_summary,
        next_question=next_q,
    )
