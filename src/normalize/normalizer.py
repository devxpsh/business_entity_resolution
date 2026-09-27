"""
Non-destructive Normalization Module for Business Entity Resolution.

Provides canonical normalization for business names and addresses,
numeric token extraction, phonetic key generation, and non-destructive
synonym/abbreviation mappings.
"""

import re
import unicodedata
from typing import List, Set, Tuple, Optional
import unidecode
import jellyfish

# Regex patterns compiled for maximum speed
RE_PUNCT_EXCLUDE = re.compile(r"[^\w\s&+]")
RE_MULTI_SPACE = re.compile(r"\s+")
RE_NUMBERS = re.compile(r"\b\d+\b")
RE_ALPHANUM_CODE = re.compile(r"\b[A-Z0-9-]{3,}\b", re.IGNORECASE)

# Standard legal & business noise suffixes / prefixes that appear across sources
COMMON_LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "llc", "l l c",
    "ltd", "limited", "pvt", "private", "co", "company", "plc",
    "llp", "gmbh", "sarl", "sa", "dba", "d b a", "pc", "p c",
    "partners", "services", "center", "solutions", "group", "holdings"
}

def clean_unicode(text: Optional[str]) -> str:
    """Normalize unicode, transliterate diacritics/accents to ASCII."""
    if not text or not isinstance(text, str) or text.lower() == "nan":
        return ""
    # NFKD decomposition + unidecode for accents (e.g., É -> E)
    text = unicodedata.normalize("NFKD", text)
    text = unidecode.unidecode(text)
    return text.strip()

def normalize_text_canonical(text: Optional[str]) -> str:
    """Canonical string normalization:
    1. Transliterate diacritics / accents.
    2. Lowercase.
    3. Remove punctuation (preserve &, +).
    4. Collapse whitespace.
    """
    cleaned = clean_unicode(text)
    if not cleaned:
        return ""
    cleaned = cleaned.lower()
    cleaned = RE_PUNCT_EXCLUDE.sub(" ", cleaned)
    cleaned = RE_MULTI_SPACE.sub(" ", cleaned).strip()
    return cleaned

def extract_numeric_tokens(text: Optional[str]) -> List[str]:
    """Extract numeric tokens (e.g., street numbers, PIN codes, plot numbers)."""
    if not text or not isinstance(text, str):
        return []
    cleaned = clean_unicode(text)
    return RE_NUMBERS.findall(cleaned)

def extract_alphanumeric_codes(text: Optional[str]) -> List[str]:
    """Extract alphanumeric identifiers (e.g. D-88, WZ-187C, G-1)."""
    if not text or not isinstance(text, str):
        return []
    cleaned = clean_unicode(text)
    matches = RE_ALPHANUM_CODE.findall(cleaned)
    # Filter to codes that contain at least one digit and one letter
    codes = [m.lower().replace("-", "") for m in matches if any(c.isdigit() for c in m) and any(c.isalpha() for c in m)]
    return list(dict.fromkeys(codes))

def get_phonetic_code(name_token: str) -> str:
    """Return Metaphone code for a single token."""
    if not name_token:
        return ""
    try:
        return jellyfish.metaphone(name_token)
    except Exception:
        return ""

def get_name_tokens(canonical_name: str) -> List[str]:
    """Tokenize canonical name, excluding empty tokens."""
    if not canonical_name:
        return []
    return [t for t in canonical_name.split() if t]

def get_core_name_tokens(tokens: List[str]) -> List[str]:
    """Filter out common generic legal/business tokens to get the entity core name."""
    core = [t for t in tokens if t not in COMMON_LEGAL_SUFFIXES and len(t) > 1]
    return core if core else tokens

def get_acronym(tokens: List[str]) -> str:
    """Compute initials/acronym from name tokens."""
    core = get_core_name_tokens(tokens)
    return "".join(t[0] for t in core if t)

def load_synonyms_dict(json_path: Optional[str] = None) -> dict:
    """Load mined synonyms dictionary mapping variant tokens to canonical forms."""
    import json
    import os
    if json_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(base_dir, "mined_synonyms.json")
    if not os.path.isfile(json_path):
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            pairs = json.load(f)
        mapping = {}
        for pair_key in pairs:
            if ":" in pair_key:
                w1, w2 = pair_key.split(":", 1)
                # Map both directions to a unified standard (e.g. shorter or standard word)
                mapping[w2] = w1
        return mapping
    except Exception:
        return {}

# Global cache for synonyms
_SYNONYMS_MAP = None

def normalize_with_synonyms(text: Optional[str]) -> str:
    """Non-destructive parallel feature: maps mined noisy variants to standard tokens."""
    global _SYNONYMS_MAP
    if _SYNONYMS_MAP is None:
        _SYNONYMS_MAP = load_synonyms_dict()
    canonical = normalize_text_canonical(text)
    if not canonical or not _SYNONYMS_MAP:
        return canonical
    tokens = canonical.split()
    replaced = [_SYNONYMS_MAP.get(t, t) for t in tokens]
    return " ".join(replaced)

