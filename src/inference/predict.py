"""
Full Test Inference Pipeline (Phase 9).
Processes test data partitioned by country (open-set: handles France, US, India).
Generates:
  1. output/candidate_pairs.tsv (blocking candidates)
  2. output/matching_results.tsv (final entity resolution matches)

Guarantees:
  - Exact TSV format (tab-separated, no quotes)
  - Every S1 entity in test_source1.tsv appears on exactly one row
  - matched_entity_ids is a strict subset of candidate_entity_ids
  - No intra-list duplicate IDs
  - Memory-safe chunked processing with lazy feature evaluation
"""

import os
import sys
import time
import argparse
import pickle
import gc
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.blocking.blocker import CascadeBlocker
from src.features.features import (
    prepare_feature_record,
    compute_v1_features,
    compute_v2_features
)

TEST_S1_PATH = "dataset/test/test_source1.tsv"
TEST_S2_PATH = "dataset/test/test_source2.tsv"
TEST_S3_PATH = "dataset/test/test_source3.tsv"
MODEL_PATH = "code/business_entity_resolution/src/model/artifacts/lgbm_model_v2.pkl"
OUTPUT_DIR = "output"

def load_country_candidates(country: str) -> Dict[str, Tuple[str, str]]:
    """Load lightweight (name, address) tuples for S2 and S3 candidates for a specific country."""
    print(f"[{country}] Loading S2 and S3 candidate records via itertuples...")
    t0 = time.time()
    cands_raw = {}
    cols = ["entity_id", "business_name", "business_address", "country"]
    
    # Read S2
    for chunk in pd.read_csv(TEST_S2_PATH, sep="\t", chunksize=500000, usecols=cols):
        sub = chunk[chunk["country"] == country]
        for row in sub.itertuples(index=False):
            cands_raw[row.entity_id] = (
                str(row.business_name) if pd.notna(row.business_name) else "",
                str(row.business_address) if pd.notna(row.business_address) else ""
            )
            
    # Read S3
    for chunk in pd.read_csv(TEST_S3_PATH, sep="\t", chunksize=500000, usecols=cols):
        sub = chunk[chunk["country"] == country]
        for row in sub.itertuples(index=False):
            cands_raw[row.entity_id] = (
                str(row.business_name) if pd.notna(row.business_name) else "",
                str(row.business_address) if pd.notna(row.business_address) else ""
            )
            
    print(f"[{country}] Loaded {len(cands_raw):,} candidate records in {time.time() - t0:.1f}s.")
    return cands_raw

def process_country_partition(
    country: str,
    model,
    threshold: float,
    match_file,
    cand_file,
    chunk_size: int = 30000
):
    """Run candidate generation and matching for one country partition."""
    print(f"\n{'='*70}\nSTARTING COUNTRY PARTITION: {country}\n{'='*70}")
    
    # 1. Load candidate raw strings
    cands_raw = load_country_candidates(country)
    
    # 2. Build multi-index blocker
    print(f"[{country}] Preprocessing candidate keys and building multi-index...")
    t0 = time.time()
    blocker = CascadeBlocker(
        max_name_token_freq=2000,
        max_addr_token_freq=400,
        max_ph_freq=400,
        max_num_freq=60,
        max_candidates_per_entity=50
    )
    preprocessed_cands = [
        blocker.preprocess_record({
            "entity_id": cid,
            "business_name": name,
            "business_address": addr,
            "country": country
        })
        for cid, (name, addr) in cands_raw.items()
    ]
    blocker.index_candidates(preprocessed_cands)
    print(f"[{country}] Blocker indexing ready in {time.time() - t0:.1f}s.")
    
    # Free preprocessed candidate structures
    del preprocessed_cands
    gc.collect()
    
    # 3. Stream S1 entities for this country in chunks
    print(f"[{country}] Streaming S1 entities from {TEST_S1_PATH}...")
    s1_count = 0
    match_count = 0
    t_start = time.time()
    s1_cols = ["entity_id", "business_name", "business_address", "country"]
    
    for chunk in pd.read_csv(TEST_S1_PATH, sep="\t", chunksize=chunk_size, usecols=s1_cols):
        sub_s1 = chunk[chunk["country"] == country]
        if sub_s1.empty:
            continue
            
        s1_batch_records = {
            row.entity_id: {
                "entity_id": row.entity_id,
                "business_name": str(row.business_name) if pd.notna(row.business_name) else "",
                "business_address": str(row.business_address) if pd.notna(row.business_address) else "",
                "country": country
            }
            for row in sub_s1.itertuples(index=False)
        }
        s1_block_pre = {sid: blocker.preprocess_record(r) for sid, r in s1_batch_records.items()}
        s1_feats = {sid: prepare_feature_record(r) for sid, r in s1_batch_records.items()}
        
        # Candidate generation
        s1_to_cands = {}
        for sid, rec in s1_block_pre.items():
            s1_to_cands[sid] = blocker.query(rec)
            
        # Collect active unique candidate IDs needed for this batch
        active_cids = {cid for cands in s1_to_cands.values() for cid in cands if cid in cands_raw}
        
        # Lazy feature prep for active candidates only
        active_cand_feats = {
            cid: prepare_feature_record({
                "entity_id": cid,
                "business_name": cands_raw[cid][0],
                "business_address": cands_raw[cid][1],
                "country": country
            })
            for cid in active_cids
        }
        
        # Form batch pairs
        batch_pairs = []
        for sid, cands in s1_to_cands.items():
            for cid in cands:
                if cid in active_cand_feats:
                    batch_pairs.append((sid, cid))
                    
        # Feature extraction & scoring
        pair_to_prob = {}
        if batch_pairs:
            X_batch = []
            for sid, cid in batch_pairs:
                s1 = s1_feats[sid]
                c = active_cand_feats[cid]
                v1 = compute_v1_features(s1, c)
                v2 = compute_v2_features(s1, c, v1)
                X_batch.append(v2)
                
            X_batch = np.array(X_batch, dtype=np.float32)
            probs = model.predict_proba(X_batch)[:, 1]
            
            for (sid, cid), p in zip(batch_pairs, probs):
                pair_to_prob[(sid, cid)] = float(p)
                
        # Write outputs for batch
        for sid, cands in s1_to_cands.items():
            cand_str = ",".join(cands)
            cand_file.write(f"{sid}\t{cand_str}\n")
            
            # Matched list: strictly from cands where p >= threshold
            matched = [cid for cid in cands if pair_to_prob.get((sid, cid), 0.0) >= threshold]
            match_str = ",".join(matched)
            match_file.write(f"{sid}\t{match_str}\n")
            
            if matched:
                match_count += 1
            s1_count += 1
            
        cand_file.flush()
        match_file.flush()
        rate = s1_count / max(time.time() - t_start, 0.1)
        print(f"[{country}] Processed {s1_count:,} S1 entities... ({match_count:,} matched, speed: {rate:.0f} ent/s)")
        
    print(f"[{country}] Finished all {s1_count:,} entities in {time.time() - t_start:.1f}s.")
    del cands_raw, blocker
    gc.collect()
    return s1_count

def run_inference(
    countries: Optional[List[str]] = None,
    threshold: float = 0.85
):
    print("="*70)
    print("STARTING TEST INFERENCE PIPELINE")
    print(f"Decision Threshold: {threshold}")
    print("="*70)
    
    print(f"Loading model artifact from {MODEL_PATH}...")
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Scan for countries and exact entity counts
    print("Scanning test_source1.tsv for all distinct countries and row counts...")
    country_s1_counts = {}
    for chunk in pd.read_csv(TEST_S1_PATH, sep="\t", usecols=["country"], chunksize=300000):
        for c, cnt in chunk["country"].value_counts().items():
            country_s1_counts[c] = country_s1_counts.get(c, 0) + cnt
            
    all_detected_countries = sorted(list(country_s1_counts.keys()))
    print(f"Detected countries in test set (open set): {country_s1_counts}")
    
    run_countries = countries if countries else all_detected_countries
    t0 = time.time()
    
    for c in run_countries:
        match_c_path = os.path.join(OUTPUT_DIR, f"matching_results_{c}.tsv")
        cand_c_path = os.path.join(OUTPUT_DIR, f"candidate_pairs_{c}.tsv")
        req_count = country_s1_counts.get(c, 0)
        
        # Check if already complete
        if os.path.exists(match_c_path) and os.path.exists(cand_c_path):
            with open(match_c_path, "r", encoding="utf-8") as f_chk:
                lines = sum(1 for _ in f_chk)
            if lines == req_count + 1:
                print(f"[{c}] Partition file {match_c_path} already complete ({lines - 1:,} rows). Skipping computation.")
                continue
                
        with open(match_c_path, "w", encoding="utf-8") as f_match, open(cand_c_path, "w", encoding="utf-8") as f_cand:
            f_match.write("source1_entity_id\tmatched_entity_ids\n")
            f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
            process_country_partition(
                country=c,
                model=model,
                threshold=threshold,
                match_file=f_match,
                cand_file=f_cand,
                chunk_size=30000
            )
            
    # Check if all detected countries have completed files to merge
    all_done = all(
        os.path.exists(os.path.join(OUTPUT_DIR, f"matching_results_{c}.tsv")) and
        os.path.exists(os.path.join(OUTPUT_DIR, f"candidate_pairs_{c}.tsv"))
        for c in all_detected_countries
    )
    
    if all_done:
        print("\n" + "="*70)
        print("Merging all country partitions into final submission files...")
        final_match = os.path.join(OUTPUT_DIR, "matching_results.tsv")
        final_cand = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
        
        total_rows = 0
        with open(final_match, "w", encoding="utf-8") as fm_out:
            fm_out.write("source1_entity_id\tmatched_entity_ids\n")
            for c in all_detected_countries:
                c_path = os.path.join(OUTPUT_DIR, f"matching_results_{c}.tsv")
                with open(c_path, "r", encoding="utf-8") as f_in:
                    next(f_in) # skip header
                    for line in f_in:
                        fm_out.write(line)
                        total_rows += 1
                        
        with open(final_cand, "w", encoding="utf-8") as fc_out:
            fc_out.write("source1_entity_id\tcandidate_entity_ids\n")
            for c in all_detected_countries:
                c_path = os.path.join(OUTPUT_DIR, f"candidate_pairs_{c}.tsv")
                with open(c_path, "r", encoding="utf-8") as f_in:
                    next(f_in) # skip header
                    for line in f_in:
                        fc_out.write(line)
                        
        print(f"INFERENCE COMPLETE! Total S1 entities merged: {total_rows:,}")
        print(f"Total pipeline time: {time.time() - t0:.1f}s")
        print(f"Saved: {final_match}")
        print(f"Saved: {final_cand}")
        print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", type=str, default=None, help="Specific country to run (e.g. France, US, India). Default: all.")
    parser.add_argument("--threshold", type=float, default=0.85, help="Decision probability threshold.")
    args = parser.parse_args()
    
    target_countries = [args.country] if args.country else None
    run_inference(countries=target_countries, threshold=args.threshold)
