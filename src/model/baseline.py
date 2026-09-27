"""
Simple Manually-Tuned Baseline (Phase 4).
Evaluates a deterministic rule-based similarity baseline on validation data.
Serves as the non-negotiable anchor reference point before ML modeling.

Score function:
  score = 0.65 * max(name_token_sort_ratio, name_token_set_ratio) + 0.35 * addr_similarity
Decision rule:
  predict match if score >= threshold
"""

import os
import sys
import time
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np
from rapidfuzz import fuzz

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.normalize.normalizer import normalize_text_canonical, get_name_tokens
from src.blocking.blocker import CascadeBlocker
from src.utils.metrics import compute_macro_f05, compute_entity_f05

TRAIN_S1_PATH = "dataset/train/train_source1.tsv"
TRAIN_S2_PATH = "dataset/train/train_source2.tsv"
TRAIN_S3_PATH = "dataset/train/train_source3.tsv"
TRAIN_GT_PATH = "dataset/train/train_ground_truth.tsv"
VAL_IDS_PATH = "code/business_entity_resolution/src/data/splits/val_s1_ids_50000.txt"

def compute_pair_similarity(s1_name: str, s1_addr: str, cand_name: str, cand_addr: str) -> float:
    """Deterministic score function combining name and address similarities."""
    name_sort = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0
    name_set = fuzz.token_set_ratio(s1_name, cand_name) / 100.0
    name_sim = max(name_sort, name_set)
    
    if cand_addr and s1_addr:
        addr_sim = fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0
        score = 0.65 * name_sim + 0.35 * addr_sim
    else:
        # If candidate address is missing, rely slightly more conservatively on name similarity
        score = name_sim * 0.90
        
    return score

def load_eval_data(val_sample_size: int = 3000):
    print(f"Loading {val_sample_size} validation entities from split file...")
    with open(VAL_IDS_PATH, "r") as f:
        all_val_ids = [line.strip() for line in f if line.strip()]
    eval_ids = set(all_val_ids[:val_sample_size])
    
    # Load Ground Truth
    gt_df = pd.read_csv(TRAIN_GT_PATH, sep="\t")
    sub_gt = gt_df[gt_df["source1_entity_id"].isin(eval_ids)]
    
    gt_dict = {}
    needed_cand_ids = set()
    for _, r in sub_gt.iterrows():
        sid = r["source1_entity_id"]
        mids = str(r["matched_entity_ids"]).split(",") if pd.notna(r["matched_entity_ids"]) else []
        clean_mids = set(m.strip() for m in mids if m.strip())
        gt_dict[sid] = clean_mids
        needed_cand_ids.update(clean_mids)
        
    print(f"Ground truth loaded for {len(gt_dict)} entities. True match count: {len(needed_cand_ids)}")
    
    # Load S1 records
    s1_records = {}
    for chunk in pd.read_csv(TRAIN_S1_PATH, sep="\t", chunksize=200000):
        sub = chunk[chunk["entity_id"].isin(eval_ids)]
        for _, r in sub.iterrows():
            s1_records[r["entity_id"]] = {
                "entity_id": r["entity_id"],
                "business_name": r["business_name"],
                "business_address": r["business_address"],
                "country": r["country"]
            }
        if len(s1_records) >= len(eval_ids):
            break
            
    # Load candidate pool: all true targets + 150k distractors
    print("Loading candidate pool from S2 and S3...")
    cand_records = {}
    for path in [TRAIN_S2_PATH, TRAIN_S3_PATH]:
        for chunk in pd.read_csv(path, sep="\t", chunksize=300000):
            # True match targets
            target_sub = chunk[chunk["entity_id"].isin(needed_cand_ids)]
            for _, r in target_sub.iterrows():
                cand_records[r["entity_id"]] = {
                    "entity_id": r["entity_id"],
                    "business_name": r["business_name"],
                    "business_address": r["business_address"],
                    "country": r["country"]
                }
            # Distractors
            if len(cand_records) < 150000:
                dist_sample = chunk.sample(n=min(10000, len(chunk)), random_state=42)
                for _, r in dist_sample.iterrows():
                    cand_records[r["entity_id"]] = {
                        "entity_id": r["entity_id"],
                        "business_name": r["business_name"],
                        "business_address": r["business_address"],
                        "country": r["country"]
                    }
            if needed_cand_ids.issubset(set(cand_records.keys())) and len(cand_records) >= 150000:
                break
                
    print(f"Candidate pool size: {len(cand_records)} records. True target coverage: {len(needed_cand_ids.intersection(cand_records))}/{len(needed_cand_ids)}")
    return s1_records, cand_records, gt_dict

def run_baseline_evaluation():
    s1_records, cand_records, gt_dict = load_eval_data(val_sample_size=3000)
    
    print("\nInitializing CascadeBlocker...")
    blocker = CascadeBlocker(max_candidates_per_entity=60)
    
    print("Preprocessing candidate records...")
    preprocessed_cands = [blocker.preprocess_record(r) for r in cand_records.values()]
    blocker.index_candidates(preprocessed_cands)
    
    print("Preprocessing S1 records & generating candidate pairs...")
    s1_pre = {sid: blocker.preprocess_record(r) for sid, r in s1_records.items()}
    
    # Generate candidates & evaluate candidate recall
    start_time = time.time()
    all_candidate_pairs = {}
    total_true = sum(len(m) for m in gt_dict.values())
    found_true = 0
    total_cands = 0
    
    # Store preprocessed text for rapid scoring
    raw_s1_text = {sid: (s1_pre[sid]["c_name"], s1_pre[sid]["sig_addr_tokens"]) for sid in s1_pre}
    raw_cand_text = {cid: (blocker.cand_data[cid]["c_name"], blocker.cand_data[cid]["sig_addr_tokens"]) for cid in blocker.cand_data}
    
    # Pre-score candidate pairs
    scored_candidates = {}
    for sid, s1_rec in s1_pre.items():
        cands = blocker.query(s1_rec)
        all_candidate_pairs[sid] = cands
        total_cands += len(cands)
        true_mids = gt_dict.get(sid, set())
        found_true += len(true_mids.intersection(set(cands)))
        
        # Compute baseline score for each candidate
        sid_scores = []
        n1 = s1_records[sid]["business_name"]
        a1 = s1_records[sid]["business_address"]
        for cid in cands:
            c_rec = cand_records[cid]
            n2 = c_rec["business_name"]
            a2 = c_rec["business_address"]
            s = compute_pair_similarity(n1, a1, n2, a2)
            sid_scores.append((cid, s))
        scored_candidates[sid] = sid_scores
        
    blocker_recall = (found_true / total_true) * 100 if total_true > 0 else 100.0
    avg_cands = total_cands / len(s1_pre)
    print(f"Blocking complete in {time.time() - start_time:.2f}s.")
    print(f"Candidate Recall Ceiling: {blocker_recall:.2f}% | Avg Candidates/Entity: {avg_cands:.2f}")
    
    print("\n" + "="*75)
    print(f"| {'Threshold':<10} | {'Macro F0.5':<12} | {'Macro Prec':<12} | {'Macro Rec':<12} | {'Singleton Acc':<14} |")
    print("="*75)
    
    thresholds = [0.60, 0.65, 0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90]
    best_thresh = 0.80
    best_f05 = 0.0
    best_metrics = None
    
    for thresh in thresholds:
        preds = {}
        for sid, scored_list in scored_candidates.items():
            matches = {cid for cid, s in scored_list if s >= thresh}
            preds[sid] = matches
            
        m = compute_macro_f05(gt_dict, preds)
        print(f"| {thresh:<10.2f} | {m['macro_f05']:<12.4f} | {m['macro_precision']:<12.4f} | {m['macro_recall']:<12.4f} | {m['singleton_acc']:<14.4f} |")
        if m["macro_f05"] > best_f05:
            best_f05 = m["macro_f05"]
            best_thresh = thresh
            best_metrics = m
            
    print("="*75)
    print(f"\n[PHASE 4 BASELINE CHECKPOINT]")
    print(f"Optimal Rule-based Threshold: {best_thresh:.2f}")
    print(f"Baseline Macro F0.5:         {best_metrics['macro_f05']:.4f}")
    print(f"Baseline Macro Precision:    {best_metrics['macro_precision']:.4f}")
    print(f"Baseline Macro Recall:       {best_metrics['macro_recall']:.4f}")
    print(f"Baseline Singleton Accuracy: {best_metrics['singleton_acc']:.4f}")
    print(f"Blocking Candidate Recall:   {blocker_recall:.2f}%\n")
    return best_metrics

if __name__ == "__main__":
    run_baseline_evaluation()
