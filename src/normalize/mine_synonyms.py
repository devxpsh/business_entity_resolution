"""
Mines frequent abbreviations, legal variations, and token synonyms
from Ground Truth match pairs.
Outputs mined_synonyms.json for parallel feature extraction (Phase 3 v2).
"""

import os
import sys
import json
from collections import defaultdict
import pandas as pd
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.normalize.normalizer import normalize_text_canonical, get_name_tokens

TRAIN_GT_PATH = "dataset/train/train_ground_truth.tsv"
TRAIN_S1_PATH = "dataset/train/train_source1.tsv"
TRAIN_S2_PATH = "dataset/train/train_source2.tsv"
TRAIN_S3_PATH = "dataset/train/train_source3.tsv"
OUTPUT_PATH = "code/business_entity_resolution/src/normalize/mined_synonyms.json"

def mine_synonyms(max_s1_samples: int = 50000, min_support: int = 25):
    print(f"Reading first {max_s1_samples} rows from ground truth...")
    gt = pd.read_csv(TRAIN_GT_PATH, sep="\t", nrows=max_s1_samples)
    
    # Collect all needed IDs
    s1_needed = set(gt["source1_entity_id"])
    s2_needed = set()
    s3_needed = set()
    
    pairs = []
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        mids = str(row["matched_entity_ids"]).split(",") if pd.notna(row["matched_entity_ids"]) else []
        for mid in mids:
            mid = mid.strip()
            if not mid:
                continue
            if mid.startswith("S2-"):
                s2_needed.add(mid)
            elif mid.startswith("S3-"):
                s3_needed.add(mid)
            pairs.append((s1_id, mid))
            
    print(f"Sampling {len(pairs)} true match pairs ({len(s1_needed)} S1, {len(s2_needed)} S2, {len(s3_needed)} S3)...")
    
    # Load records in chunks
    s1_names = {}
    for chunk in pd.read_csv(TRAIN_S1_PATH, sep="\t", usecols=["entity_id", "business_name"], chunksize=200000):
        sub = chunk[chunk["entity_id"].isin(s1_needed)]
        for _, r in sub.iterrows():
            s1_names[r["entity_id"]] = normalize_text_canonical(r["business_name"])
        if len(s1_names) == len(s1_needed):
            break
            
    cand_names = {}
    for chunk in pd.read_csv(TRAIN_S2_PATH, sep="\t", usecols=["entity_id", "business_name"], chunksize=200000):
        sub = chunk[chunk["entity_id"].isin(s2_needed)]
        for _, r in sub.iterrows():
            cand_names[r["entity_id"]] = normalize_text_canonical(r["business_name"])
        if len(cand_names) >= len(s2_needed):
            break
            
    for chunk in pd.read_csv(TRAIN_S3_PATH, sep="\t", usecols=["entity_id", "business_name"], chunksize=200000):
        sub = chunk[chunk["entity_id"].isin(s3_needed)]
        for _, r in sub.iterrows():
            cand_names[r["entity_id"]] = normalize_text_canonical(r["business_name"])
        if len(cand_names) >= len(s2_needed) + len(s3_needed):
            break
            
    print("Mining token substitution pairs...")
    pair_counts = defaultdict(int)
    
    for s1_id, cand_id in pairs:
        n1 = s1_names.get(s1_id, "")
        n2 = cand_names.get(cand_id, "")
        if not n1 or not n2:
            continue
            
        t1_set = set(get_name_tokens(n1))
        t2_set = set(get_name_tokens(n2))
        
        diff1 = t1_set - t2_set
        diff2 = t2_set - t1_set
        
        # Look for aligned token differences
        for w1 in diff1:
            for w2 in diff2:
                if len(w1) < 2 or len(w2) < 2:
                    continue
                # Canonical ordering: shorter word -> longer word
                short, long = (w1, w2) if len(w1) <= len(w2) else (w2, w1)
                
                # Check prefix/abbreviation or small edit distance
                if long.startswith(short) and len(short) >= 2:
                    pair_counts[(short, long)] += 1
                elif len(short) >= 4 and len(long) >= 4 and Levenshtein.distance(short, long) <= 2:
                    pair_counts[(short, long)] += 1

    # Filter by minimum support
    filtered_synonyms = {}
    for (short, long), count in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True):
        if count >= min_support:
            filtered_synonyms[f"{short}:{long}"] = count
            
    print(f"Mined {len(filtered_synonyms)} synonym/abbreviation pairs with support >= {min_support}.")
    print("Top 30 mined pairs:")
    for k, v in list(filtered_synonyms.items())[:30]:
        print(f"  {k} (count={v})")
        
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(filtered_synonyms, f, indent=2)
    print(f"Saved mined synonyms to {OUTPUT_PATH}")

if __name__ == "__main__":
    mine_synonyms()
