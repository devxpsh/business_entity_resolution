"""
Pairwise Feature Engineering Module (Phase 3).
Constructs feature matrices for (S1, Candidate) pairs.
Supports layered feature extraction:
  - v1: Core string similarities, token overlap, numeric overlap, missingness flags.
  - v2: Extended features (phonetic match, acronym match, synonym-expanded similarities).
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from src.normalize.normalizer import (
    normalize_text_canonical,
    normalize_with_synonyms,
    get_name_tokens,
    get_core_name_tokens,
    get_acronym,
    get_phonetic_code,
    extract_numeric_tokens,
    extract_alphanumeric_codes
)

V1_FEATURE_NAMES = [
    "name_exact_match",
    "name_levenshtein_ratio",
    "name_jaro_winkler",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_token_jaccard",
    "core_name_exact_match",
    "core_name_jaccard",
    "addr_exact_match",
    "addr_levenshtein_ratio",
    "addr_token_sort_ratio",
    "addr_token_jaccard",
    "num_match_count",
    "num_overlap_ratio",
    "alphanum_code_match",
    "addr_missing",
    "name_len_diff",
    "addr_len_diff"
]

V2_FEATURE_NAMES = V1_FEATURE_NAMES + [
    "phonetic_token_overlap",
    "acronym_match",
    "name_synonym_sort_ratio",
    "name_synonym_levenshtein",
    "name_addr_composite_score"
]

def compute_v1_features(s1: dict, cand: dict) -> List[float]:
    """Compute core v1 feature vector for a pair."""
    n1 = s1["c_name"]
    n2 = cand["c_name"]
    a1 = s1["c_addr"]
    a2 = cand["c_addr"]
    
    # 1. Name features
    if n1 and n1 == n2:
        name_exact = 1.0
        name_lev = 1.0
        name_jw = 1.0
        name_sort = 1.0
        name_set = 1.0
        name_jaccard = 1.0
        name_len_diff = 0.0
    else:
        name_exact = 0.0
        name_lev = fuzz.ratio(n1, n2) / 100.0 if (n1 and n2) else 0.0
        name_jw = JaroWinkler.similarity(n1, n2) if (n1 and n2) else 0.0
        name_sort = fuzz.token_sort_ratio(n1, n2) / 100.0 if (n1 and n2) else 0.0
        name_set = fuzz.token_set_ratio(n1, n2) / 100.0 if (n1 and n2) else 0.0
        t1 = s1["tokens"]
        t2 = cand["tokens"]
        name_jaccard = (len(t1 & t2) / len(t1 | t2)) if (t1 and t2 and (t1 | t2)) else 0.0
        name_len_diff = abs(len(n1) - len(n2)) / max(len(n1), len(n2), 1)
    
    c1 = s1["core_name"]
    c2 = cand["core_name"]
    if c1 and c1 == c2:
        core_exact = 1.0
        core_jaccard = 1.0
    else:
        core_exact = 0.0
        ct1 = s1["core_tokens"]
        ct2 = cand["core_tokens"]
        core_jaccard = (len(ct1 & ct2) / len(ct1 | ct2)) if (ct1 and ct2 and (ct1 | ct2)) else 0.0
    
    # 2. Address features
    addr_missing = 1.0 if not a2 else 0.0
    if a1 and a2 and a1 == a2:
        addr_exact = 1.0
        addr_lev = 1.0
        addr_sort = 1.0
        addr_jaccard = 1.0
        addr_len_diff = 0.0
    else:
        addr_exact = 0.0
        addr_lev = fuzz.ratio(a1, a2) / 100.0 if (a1 and a2) else 0.0
        addr_sort = fuzz.token_sort_ratio(a1, a2) / 100.0 if (a1 and a2) else 0.0
        at1 = s1["addr_tokens"]
        at2 = cand["addr_tokens"]
        addr_jaccard = (len(at1 & at2) / len(at1 | at2)) if (at1 and at2 and (at1 | at2)) else 0.0
        addr_len_diff = abs(len(a1) - len(a2)) / max(len(a1), len(a2), 1) if (a1 and a2) else 1.0
    
    # 3. Numeric & code features
    nums1 = s1["numbers"]
    nums2 = cand["numbers"]
    num_matches = len(nums1 & nums2)
    num_overlap = (num_matches / len(nums1)) if nums1 else 0.0
    
    codes1 = s1["codes"]
    codes2 = cand["codes"]
    code_match = 1.0 if (codes1 and codes2 and bool(codes1 & codes2)) else 0.0
    
    return [
        name_exact,
        name_lev,
        name_jw,
        name_sort,
        name_set,
        name_jaccard,
        core_exact,
        core_jaccard,
        addr_exact,
        addr_lev,
        addr_sort,
        addr_jaccard,
        float(num_matches),
        num_overlap,
        code_match,
        addr_missing,
        name_len_diff,
        addr_len_diff
    ]

def compute_v2_features(s1: dict, cand: dict, v1_feats: Optional[List[float]] = None) -> List[float]:
    """Compute extended v2 feature vector (v1 + phonetic + acronym + synonym expansion)."""
    if v1_feats is None:
        v1_feats = compute_v1_features(s1, cand)
        
    ph1 = s1["phonetics"]
    ph2 = cand["phonetics"]
    ph_overlap = (len(ph1 & ph2) / len(ph1)) if ph1 else 0.0
    
    acr1 = s1["acronym"]
    acr2 = cand["acronym"]
    acronym_match = 1.0 if (acr1 and acr2 and len(acr1) >= 2 and acr1 == acr2) else 0.0
    
    # Synonym-expanded name features
    syn1 = s1["syn_name"]
    syn2 = cand["syn_name"]
    if syn1 and syn2 and syn1 == syn2:
        syn_sort = 1.0
        syn_lev = 1.0
    else:
        syn_sort = fuzz.token_sort_ratio(syn1, syn2) / 100.0 if (syn1 and syn2) else 0.0
        syn_lev = fuzz.ratio(syn1, syn2) / 100.0 if (syn1 and syn2) else 0.0
    
    # Composite score
    name_score = max(v1_feats[3], v1_feats[4])  # token sort/set ratio
    addr_score = v1_feats[10]  # addr sort ratio
    composite = 0.65 * name_score + 0.35 * addr_score
    
    return v1_feats + [
        ph_overlap,
        acronym_match,
        syn_sort,
        syn_lev,
        composite
    ]

def prepare_feature_record(record: dict) -> dict:
    """Precompute reusable fields for feature extraction."""
    c_name = normalize_text_canonical(record.get("business_name", ""))
    c_addr = normalize_text_canonical(record.get("business_address", ""))
    tokens = get_name_tokens(c_name)
    core = get_core_name_tokens(tokens)
    
    return {
        "id": record["entity_id"],
        "country": str(record.get("country", "")).strip(),
        "c_name": c_name,
        "c_addr": c_addr,
        "tokens": set(tokens),
        "core_name": " ".join(core),
        "core_tokens": set(core),
        "addr_tokens": set(get_name_tokens(c_addr)),
        "numbers": set(extract_numeric_tokens(record.get("business_address", ""))),
        "codes": set(extract_alphanumeric_codes(record.get("business_address", ""))),
        "phonetics": {get_phonetic_code(t) for t in core if len(t) >= 3},
        "acronym": get_acronym(tokens),
        "syn_name": normalize_with_synonyms(record.get("business_name", ""))
    }
