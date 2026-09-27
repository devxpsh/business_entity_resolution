"""
Entity-Level Decision Layer (Phase 7).
Compares Strategy A (Calibrated Flat Threshold), Strategy B (Top-1 with Margin),
and Strategy C (Soft Multi-Match with Confidence Degradation).
Directly optimizes for Macro F0.5 on validation split.
"""

import os
import sys
import pickle
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.utils.metrics import compute_macro_f05
from src.model.train_lgbm import load_data_split, MODEL_SAVE_DIR
from src.blocking.blocker import CascadeBlocker
from src.features.features import prepare_feature_record, compute_v1_features, compute_v2_features

def run_decision_layer_ablation():
    print("Loading saved model and validation data...")
    artifact_path = os.path.join(MODEL_SAVE_DIR, "lgbm_model_v2.pkl")
    with open(artifact_path, "rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    
    # Load 3000 validation entities
    s1_records, cand_records, _, val_gt, _, val_s1_set = load_data_split(
        n_train_s1=10,
        n_val_s1=3000
    )
    
    # Generate candidate pairs & feature matrix for validation
    print("Precomputing validation candidate predictions...")
    blocker = CascadeBlocker(max_candidates_per_entity=60)
    preprocessed_cands = [blocker.preprocess_record(r) for r in cand_records.values()]
    blocker.index_candidates(preprocessed_cands)
    s1_block_pre = {sid: blocker.preprocess_record(r) for sid, r in s1_records.items() if sid in val_s1_set}
    
    s1_feats = {sid: prepare_feature_record(r) for sid, r in s1_records.items() if sid in val_s1_set}
    cand_feats = {cid: prepare_feature_record(r) for cid, r in cand_records.items()}
    
    val_pairs = []
    val_pair_map = {sid: [] for sid in val_s1_set}
    for sid in val_s1_set:
        cands = blocker.query(s1_block_pre[sid])
        for cid in cands:
            if cid in cand_feats:
                val_pairs.append((sid, cid))
                val_pair_map[sid].append(cid)
                
    X_val = []
    for sid, cid in val_pairs:
        s1 = s1_feats[sid]
        c = cand_feats[cid]
        v1 = compute_v1_features(s1, c)
        v2 = compute_v2_features(s1, c, v1)
        X_val.append(v2)
    X_val = np.array(X_val, dtype=np.float32)
    
    probs = model.predict_proba(X_val)[:, 1]
    
    # Store predictions per S1 entity as list of (candidate_id, probability) sorted descending
    entity_cand_probs = {sid: [] for sid in val_s1_set}
    for (sid, cid), p in zip(val_pairs, probs):
        entity_cand_probs[sid].append((cid, float(p)))
        
    for sid in entity_cand_probs:
        entity_cand_probs[sid].sort(key=lambda x: x[1], reverse=True)
        
    print(f"Validation candidates scored for {len(entity_cand_probs)} S1 entities.")
    
    # -------------------------------------------------------------
    # 1. STRATEGY A: Calibrated Flat Threshold
    # -------------------------------------------------------------
    print("\nEvaluating Strategy A (Calibrated Flat Threshold)...")
    best_tau_A = 0.5
    best_mA = {"macro_f05": 0.0}
    for tau in [0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        preds = {}
        for sid, c_list in entity_cand_probs.items():
            preds[sid] = {cid for cid, p in c_list if p >= tau}
        m = compute_macro_f05(val_gt, preds)
        if m["macro_f05"] > best_mA["macro_f05"]:
            best_mA = m
            best_tau_A = tau
            
    # -------------------------------------------------------------
    # 2. STRATEGY B: Top-1 with Margin
    # -------------------------------------------------------------
    print("Evaluating Strategy B (Top-1 with Margin)...")
    best_params_B = (0.5, 0.1)
    best_mB = {"macro_f05": 0.0}
    for tau1 in [0.50, 0.60, 0.70, 0.80]:
        for delta in [0.05, 0.10, 0.20]:
            preds = {}
            for sid, c_list in entity_cand_probs.items():
                if not c_list:
                    preds[sid] = set()
                    continue
                p1 = c_list[0][1]
                p2 = c_list[1][1] if len(c_list) > 1 else 0.0
                if p1 >= tau1 and (p1 - p2) >= delta:
                    preds[sid] = {c_list[0][0]}
                else:
                    preds[sid] = set()
            m = compute_macro_f05(val_gt, preds)
            if m["macro_f05"] > best_mB["macro_f05"]:
                best_mB = m
                best_params_B = (tau1, delta)
                
    # -------------------------------------------------------------
    # 3. STRATEGY C: Soft Multi-Match with Confidence Criteria
    # -------------------------------------------------------------
    print("Evaluating Strategy C (Soft Multi-Match with Dynamic Gap)...")
    best_params_C = (0.70, 0.50, 0.35)
    best_mC = {"macro_f05": 0.0}
    for tau_primary in [0.65, 0.70, 0.75]:
        for tau_multi in [0.50, 0.55, 0.60]:
            for max_gap in [0.25, 0.35, 0.45]:
                preds = {}
                for sid, c_list in entity_cand_probs.items():
                    if not c_list or c_list[0][1] < tau_primary:
                        preds[sid] = set()
                        continue
                    p1 = c_list[0][1]
                    matches = {c_list[0][0]}
                    for cid, p_k in c_list[1:]:
                        if p_k >= tau_multi and (p1 - p_k) <= max_gap:
                            matches.add(cid)
                    preds[sid] = matches
                m = compute_macro_f05(val_gt, preds)
                if m["macro_f05"] > best_mC["macro_f05"]:
                    best_mC = m
                    best_params_C = (tau_primary, tau_multi, max_gap)
                    
    # Side-by-side comparison table
    print("\n" + "="*80)
    print(f"| {'Strategy':<30} | {'Optimal Parameters':<22} | {'Macro F0.5':<10} | {'Prec':<7} | {'Rec':<7} |")
    print("="*80)
    print(f"| {'A: Calibrated Flat Threshold':<30} | {f'tau={best_tau_A}':<22} | {best_mA['macro_f05']:<10.4f} | {best_mA['macro_precision']:<7.4f} | {best_mA['macro_recall']:<7.4f} |")
    print(f"| {'B: Top-1 with Margin':<30} | {f'tau1={best_params_B[0]}, d={best_params_B[1]}':<22} | {best_mB['macro_f05']:<10.4f} | {best_mB['macro_precision']:<7.4f} | {best_mB['macro_recall']:<7.4f} |")
    print(f"| {'C: Soft Multi-Match':<30} | {f'p={best_params_C[0]}, m={best_params_C[1]}, g={best_params_C[2]}':<22} | {best_mC['macro_f05']:<10.4f} | {best_mC['macro_precision']:<7.4f} | {best_mC['macro_recall']:<7.4f} |")
    print("="*80)
    
    winner = "Strategy A" if best_mA["macro_f05"] >= best_mC["macro_f05"] else "Strategy C"
    print(f"\nWinning Decision Strategy: {winner}")
    return {
        "strategy_A": (best_tau_A, best_mA),
        "strategy_B": (best_params_B, best_mB),
        "strategy_C": (best_params_C, best_mC),
        "winner": winner
    }

if __name__ == "__main__":
    run_decision_layer_ablation()
