"""
pronunciation_normalizer.py  — PRON-NORM-v1
Converts area names, abbreviations, and real-estate shorthand into
TTS-friendly phonetic text before synthesis.

Revert keyword: PRON-NORM-v1
  - Delete this file
  - Remove `from pronunciation_normalizer import normalize as tts_normalize`
    from processors.py
  - Remove the `text = tts_normalize(text)` line from
    TextNormalizerProcessor._normalise()

Zero external dependencies. Sub-millisecond latency.
"""

import re

# ── Pronunciation dictionary ──────────────────────────────────────────────────
# Longer phrases first (multi-word before single-word) — enforced by sort below.
PRONUNCIATION_DICT: dict[str, str] = {
    # Chennai area names that TTS commonly mispronounces
    "OMR":           "O M R",
    "ECR":           "E C R",
    "GST Road":      "G S T Road",
    "T Nagar":       "Tee Nagar",
    "T.Nagar":       "Tee Nagar",
    "TNagar":        "Tee Nagar",
    "KK Nagar":      "K K Nagar",
    "K.K.Nagar":     "K K Nagar",
    "KK Nagar":      "K K Nagar",
    # Real estate abbreviations
    "4BHK":          "4 B H K",
    "3BHK":          "3 B H K",
    "2BHK":          "2 B H K",
    "1BHK":          "1 B H K",
    "BHK":           "B H K",
    "SBA":           "S B A",
    "RERA":          "RERA",          # pronounced as a word — keep as-is
    "EMI":           "E M I",
    "OC":            "O C",
    "CC":            "C C",
    "sq.ft":         "square feet",
    "sq ft":         "square feet",
    "sqft":          "square feet",
    "lacs":          "lakhs",
    "lac":           "lakh",
    # Common abbreviations in real-estate speech
    "approx.":       "approximately",
    "approx":        "approximately",
    "govt":          "government",
    "Govt":          "government",
    "kms":           "kilometres",
    "km":            "kilometres",
    "No.":           "number",
    "no.":           "number",
    "vs.":           "versus",
    "vs":            "versus",
    "etc.":          "etcetera",
    "etc":           "etcetera",
    "i.e.":          "that is",
    "i.e":           "that is",
    "e.g.":          "for example",
    "e.g":           "for example",
}

# Currency context rules — only when directly attached to a number
_CURRENCY_L_RE  = re.compile(r'(\d+)\s*[Ll]\b')
_CURRENCY_CR_RE = re.compile(r'(\d+(?:\.\d+)?)\s*[Cc]r\b')

# Fallback: space out unknown ALL-CAPS words (2–5 letters)
_ACRONYM_RE = re.compile(r'\b([A-Z]{2,5})\b')

# Words that are all-caps but should be spoken as words, not letter-by-letter
_SKIP_ACRONYMS = {"RERA", "JLL"}


def _apply_currency(text: str) -> str:
    text = _CURRENCY_L_RE.sub(r'\1 lakh', text)
    text = _CURRENCY_CR_RE.sub(r'\1 crore', text)
    return text


def _apply_dict(text: str) -> str:
    for key in sorted(PRONUNCIATION_DICT, key=len, reverse=True):
        replacement = PRONUNCIATION_DICT[key]
        pattern = re.compile(
            r'(?<![A-Za-z])' + re.escape(key) + r'(?![A-Za-z])',
            re.IGNORECASE,
        )
        text = pattern.sub(replacement, text)
    return text


def _apply_acronym_fallback(text: str) -> str:
    def _space(m: re.Match) -> str:
        w = m.group(1)
        if w in _SKIP_ACRONYMS:
            return w
        return ' '.join(list(w))
    return _ACRONYM_RE.sub(_space, text)


def normalize(text: str) -> str:
    """
    Main entry point — call on any text string before sending to TTS.
    Pipeline:
      1. Currency context rules  (50L → 50 lakh, 2Cr → 2 crore)
      2. Dictionary replacement  (OMR → O M R, T Nagar → Tee Nagar)
      3. Fallback acronym rule   (unknown ALL-CAPS word → spaced letters)
      4. Whitespace cleanup
    """
    if not text or not text.strip():
        return text
    text = _apply_currency(text)
    text = _apply_dict(text)
    text = _apply_acronym_fallback(text)
    text = re.sub(r'  +', ' ', text).strip()
    return text
