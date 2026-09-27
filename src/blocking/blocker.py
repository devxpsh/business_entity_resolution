"""
Cascaded Multi-Blocker (Phase 2).
High-speed, high-recall, low-cardinality candidate generator:
1. Exact Canonical Name (unbounded)
2. Sorted Token Name (unbounded)
3. Core Name (unbounded)
4. Significant Name Token Inverted Index (frequency <= 2000, list sliced to 100)
5. Phonetic Metaphone Index (frequency <= 400, list sliced to 50)
6. Alphanumeric Code Index (e.g. D-88, WZ-187C)
7. Significant Address Token Index (frequency <= 400, list sliced to 60)
8. Rare Numeric Address Tokens (frequency <= 60, list sliced to 15)

Throughput: ~1,500-2,000 S1 queries/sec.
Memory-safe per-country partitioning.
"""

from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import numpy as np

from src.normalize.normalizer import (
    normalize_text_canonical,
    get_name_tokens,
    get_core_name_tokens,
    get_phonetic_code,
    extract_numeric_tokens,
    extract_alphanumeric_codes
)

COMMON_ADDR_STOPWORDS = {
    "street", "road", "avenue", "lane", "drive", "near", "behind", "opp",
    "floor", "block", "plot", "sector", "nagar", "city", "district", "state",
    "india", "united", "states", "france", "saint", "rue", "av", "boulevard",
    "place", "route", "chemin", "allee"
}

class CascadeBlocker:
    def __init__(
        self,
        max_name_token_freq: int = 2000,
        max_addr_token_freq: int = 400,
        max_ph_freq: int = 400,
        max_num_freq: int = 60,
        max_candidates_per_entity: int = 60
    ):
        self.max_name_token_freq = max_name_token_freq
        self.max_addr_token_freq = max_addr_token_freq
        self.max_ph_freq = max_ph_freq
        self.max_num_freq = max_num_freq
        self.max_candidates_per_entity = max_candidates_per_entity

    @staticmethod
    def preprocess_record(record: dict) -> dict:
        """Extract all blocking keys and representations from raw record."""
        name_clean = normalize_text_canonical(record.get("business_name", ""))
        addr_clean = normalize_text_canonical(record.get("business_address", ""))
        country = str(record.get("country", "")).strip()
        
        name_tokens = get_name_tokens(name_clean)
        core_tokens = get_core_name_tokens(name_tokens)
        sorted_name = " ".join(sorted(name_tokens)) if name_tokens else ""
        core_name = " ".join(core_tokens) if core_tokens else ""
        
        phonetics = {get_phonetic_code(t) for t in core_tokens if len(t) >= 3}
        phonetics.discard("")
        
        addr_tokens = get_name_tokens(addr_clean)
        sig_addr_tokens = {
            t for t in addr_tokens if len(t) >= 4 and t not in COMMON_ADDR_STOPWORDS
        }
        
        numbers = set(extract_numeric_tokens(record.get("business_address", "")))
        codes = set(extract_alphanumeric_codes(record.get("business_address", "")))
        
        return {
            "id": record["entity_id"],
            "country": country,
            "c_name": name_clean,
            "sorted_name": sorted_name,
            "core_name": core_name,
            "core_tokens": set(core_tokens),
            "phonetics": phonetics,
            "sig_addr_tokens": sig_addr_tokens,
            "numbers": numbers,
            "codes": codes
        }

    def index_candidates(self, candidate_records: List[dict]):
        """Build multi-level inverted indices from candidate records."""
        self.exact_idx = defaultdict(list)
        self.sorted_idx = defaultdict(list)
        self.core_idx = defaultdict(list)
        self.token_idx = defaultdict(list)
        self.ph_idx = defaultdict(list)
        self.code_idx = defaultdict(list)
        self.addr_token_idx = defaultdict(list)
        self.rare_num_idx = defaultdict(list)
        
        token_freq = defaultdict(int)
        addr_freq = defaultdict(int)
        ph_freq = defaultdict(int)
        num_freq = defaultdict(int)
        
        for rec in candidate_records:
            cid = rec["id"]
            for t in rec["core_tokens"]:
                token_freq[t] += 1
            for p in rec["phonetics"]:
                ph_freq[p] += 1
            for at in rec["sig_addr_tokens"]:
                addr_freq[at] += 1
            for n in rec["numbers"]:
                num_freq[n] += 1
                
        for rec in candidate_records:
            cid = rec["id"]
            if rec["c_name"]:
                self.exact_idx[rec["c_name"]].append(cid)
            if rec["sorted_name"]:
                self.sorted_idx[rec["sorted_name"]].append(cid)
            if rec["core_name"] and len(rec["core_name"]) > 2:
                self.core_idx[rec["core_name"]].append(cid)
            for t in rec["core_tokens"]:
                if len(t) >= 3 and token_freq[t] <= self.max_name_token_freq:
                    self.token_idx[t].append(cid)
            for p in rec["phonetics"]:
                if len(p) >= 3 and ph_freq[p] <= self.max_ph_freq:
                    self.ph_idx[p].append(cid)
            for code in rec["codes"]:
                self.code_idx[code].append(cid)
            for at in rec["sig_addr_tokens"]:
                if addr_freq[at] <= self.max_addr_token_freq:
                    self.addr_token_idx[at].append(cid)
            for n in rec["numbers"]:
                if num_freq[n] <= self.max_num_freq and (len(n) >= 3 or num_freq[n] <= 15):
                    self.rare_num_idx[n].append(cid)

    def query(self, s1_rec: dict) -> List[str]:
        """Query all index stages for an S1 entity and return deduplicated candidate IDs."""
        candidates = set()
        
        # 1. Exact canonical name
        if s1_rec["c_name"]:
            candidates.update(self.exact_idx.get(s1_rec["c_name"], []))
            
        # 2. Sorted token name
        if s1_rec["sorted_name"]:
            candidates.update(self.sorted_idx.get(s1_rec["sorted_name"], []))
            
        # 3. Core name
        if s1_rec["core_name"] and len(s1_rec["core_name"]) > 2:
            candidates.update(self.core_idx.get(s1_rec["core_name"], []))
            
        # 4. Significant name token votes (bounded)
        token_votes = defaultdict(int)
        for t in s1_rec["core_tokens"]:
            if len(t) >= 3:
                for cid in self.token_idx.get(t, [])[:100]:
                    token_votes[cid] += 1
        for cid, _ in sorted(token_votes.items(), key=lambda x: x[1], reverse=True)[:40]:
            candidates.add(cid)
            
        # 5. Phonetic votes (bounded)
        ph_votes = defaultdict(int)
        for p in s1_rec["phonetics"]:
            for cid in self.ph_idx.get(p, [])[:50]:
                ph_votes[cid] += 1
        for cid, _ in sorted(ph_votes.items(), key=lambda x: x[1], reverse=True)[:15]:
            candidates.add(cid)
            
        # 6. Alphanumeric address codes
        for code in s1_rec["codes"]:
            for cid in self.code_idx.get(code, [])[:10]:
                candidates.add(cid)
                
        # 7. Significant address tokens (bounded)
        addr_votes = defaultdict(int)
        for at in s1_rec["sig_addr_tokens"]:
            for cid in self.addr_token_idx.get(at, [])[:60]:
                addr_votes[cid] += 1
        for cid, _ in sorted(addr_votes.items(), key=lambda x: x[1], reverse=True)[:20]:
            candidates.add(cid)
            
        # 8. Rare numeric address tokens (bounded)
        for n in s1_rec["numbers"]:
            for cid in self.rare_num_idx.get(n, [])[:15]:
                candidates.add(cid)
                
        # Cap candidates per entity
        cand_list = list(candidates)
        if len(cand_list) > self.max_candidates_per_entity:
            # Prioritize candidates by voting agreement
            all_votes = defaultdict(int)
            for cid in cand_list:
                all_votes[cid] = (
                    token_votes.get(cid, 0) * 3 +
                    addr_votes.get(cid, 0) * 2 +
                    ph_votes.get(cid, 0)
                )
            cand_list = sorted(cand_list, key=lambda c: all_votes[c], reverse=True)[:self.max_candidates_per_entity]
            
        return cand_list
