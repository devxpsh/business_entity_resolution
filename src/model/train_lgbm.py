"""
Training & Validation Pipeline for Primary LightGBM Entity Matching Model (Phases 5, 6, 7).

1. Group-based train/val split strictly by source1_entity_id (zero leak).
2. Labeled pair construction with hard-negative mining (1:3 to 1:5 ratio).
3. Layered feature extraction (v1 vs v2 ablation).
4. LightGBM binary classification with calibration check.
5. Entity-level decision thresholding optimized for Macro F0.5.
"""

import os
import sys
import time
import json
import pickle
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.calibration import calibration_curve

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.normalize.normalizer import normalize_text_canonical
from src.blocking.blocker import CascadeBlocker
from src.features.features import (
    prepare_feature_record,
    compute_v1_features,
    compute_v2_features,
    V1_FEATURE_NAMES,
    V2_FEATURE_NAMES
)
from src.utils.metrics import compute_macro_f05

TRAIN_S1_PATH = "dataset/train/train_source1.tsv"
TRAIN_S2_PATH = "dataset/train/train_source2.tsv"
TRAIN_S3_PATH = "dataset/train/train_source3.tsv"
TRAIN_GT_PATH = "dataset/train/train_ground_truth.tsv"
VAL_IDS_PATH = "code/business_entity_resolution/src/data/splits/val_s1_ids_50000.txt"
MODEL_SAVE_DIR = "code/business_entity_resolution/src/model/artifacts"

def load_data_split(n_train_s1: int = 8000, n_val_s1: int = 3000):
    print(f"Loading data splits: {n_train_s1} train S1 entities, {n_val_s1} validation S1 entities...")
    with open(VAL_IDS_PATH, "r") as f:
        val_id_pool = [line.strip() for line in f if line.strip()]
    val_s1_set = set(val_id_pool[:n_val_s1])
    
    gt_df = pd.read_csv(TRAIN_GT_PATH, sep="\t")
    
    # Train S1 entities (strict zero overlap with validation)
    all_gt_s1 = gt_df["source1_entity_id"].tolist()
    train_s1_list = []
    for sid in all_gt_s1:
        if sid not in val_s1_set:
            train_s1_list.append(sid)
            if len(train_s1_list) >= n_train_s1:
                break
    train_s1_set = set(train_s1_list)
    
    assert len(train_s1_set.intersection(val_s1_set)) == 0, "ERROR: Overlap detected between train and val S1 IDs!"
    print(f"Verified strict zero overlap: {len(train_s1_set)} train S1, {len(val_s1_set)} val S1.")
    
    # Extract Ground Truth for train and val
    train_gt = {}
    val_gt = {}
    needed_cand_ids = set()
    
    for _, r in gt_df.iterrows():
        sid = r["source1_entity_id"]
        if sid in train_s1_set or sid in val_s1_set:
            mids = str(r["matched_entity_ids"]).split(",") if pd.notna(r["matched_entity_ids"]) else []
            clean_mids = set(m.strip() for m in mids if m.strip())
            if sid in train_s1_set:
                train_gt[sid] = clean_mids
            else:
                val_gt[sid] = clean_mids
            needed_cand_ids.update(clean_mids)
            
    print(f"True match targets in ground truth: {len(needed_cand_ids)} candidate IDs.")
    
    # Load S1 records
    all_needed_s1 = train_s1_set.union(val_s1_set)
    s1_records = {}
    for chunk in pd.read_csv(TRAIN_S1_PATH, sep="\t", chunksize=200000):
        sub = chunk[chunk["entity_id"].isin(all_needed_s1)]
        for _, r in sub.iterrows():
            s1_records[r["entity_id"]] = r.to_dict()
        if len(s1_records) >= len(all_needed_s1):
            break
            
    # Load candidate records: all true targets + ~180k distractors
    print("Loading candidate records from S2 and S3...")
    cand_records = {}
    for path in [TRAIN_S2_PATH, TRAIN_S3_PATH]:
        for chunk in pd.read_csv(path, sep="\t", chunksize=300000):
            target_sub = chunk[chunk["entity_id"].isin(needed_cand_ids)]
            for _, r in target_sub.iterrows():
                cand_records[r["entity_id"]] = r.to_dict()
            if len(cand_records) < 180000:
                dist_sample = chunk.sample(n=min(12000, len(chunk)), random_state=42)
                for _, r in dist_sample.iterrows():
                    cand_records[r["entity_id"]] = r.to_dict()
            if needed_cand_ids.issubset(set(cand_records.keys())) and len(cand_records) >= 180000:
                break
                
    print(f"Loaded {len(s1_records)} S1 entities and {len(cand_records)} candidate records.")
    return s1_records, cand_records, train_gt, val_gt, train_s1_set, val_s1_set

def build_training_pairs(
    s1_records: dict,
    cand_records: dict,
    train_gt: dict,
    train_s1_set: set,
    blocker: CascadeBlocker,
    s1_pre: dict,
    max_negatives_per_entity: int = 5
) -> Tuple[List[dict], List[int]]:
    """Construct labeled training pairs with hard-negative mining."""
    print(f"Building training pairs for {len(train_s1_set)} S1 entities (max {max_negatives_per_entity} hard negatives/entity)...")
    pairs = []
    labels = []
    
    pos_count = 0
    neg_count = 0
    
    for sid in train_s1_set:
        s1_rec = s1_pre[sid]
        true_mids = train_gt.get(sid, set())
        
        # 1. Add all positive pairs
        for mid in true_mids:
            if mid in blocker.cand_data:
                pairs.append((sid, mid))
                labels.append(1)
                pos_count += 1
                
        # 2. Query blocker for candidates
        cand_ids = blocker.query(s1_rec)
        
        # 3. Hard negative mining: filter candidates not in ground truth
        neg_cands = [cid for cid in cand_ids if cid not in true_mids and cid in blocker.cand_data]
        
        # Subsample top hard negatives
        for cid in neg_cands[:max_negatives_per_entity]:
            pairs.append((sid, cid))
            labels.append(0)
            neg_count += 1
            
    print(f"Constructed {len(pairs)} training pairs: {pos_count} positives (1), {neg_count} hard negatives (0). Ratio ~1:{neg_count/max(pos_count, 1):.1f}")
    return pairs, labels

def run_experiment():
    s1_records, cand_records, train_gt, val_gt, train_s1_set, val_s1_set = load_data_split(
        n_train_s1=8000,
        n_val_s1=3000
    )
    
    print("\nPreparing feature records & initializing blocker...")
    start_time = time.time()
    s1_feats = {sid: prepare_feature_record(r) for sid, r in s1_records.items()}
    cand_feats = {cid: prepare_feature_record(r) for cid, r in cand_records.items()}
    
    blocker = CascadeBlocker(max_candidates_per_entity=60)
    preprocessed_cands = [blocker.preprocess_record(r) for r in cand_records.values()]
    blocker.index_candidates(preprocessed_cands)
    s1_block_pre = {sid: blocker.preprocess_record(r) for sid, r in s1_records.items()}
    print(f"Preprocessing completed in {time.time() - start_time:.2f}s.")
    
    # 1. Construct labeled pairs for training
    train_pairs, train_labels = build_training_pairs(
        s1_records, cand_records, train_gt, train_s1_set, blocker, s1_block_pre, max_negatives_per_entity=5
    )
    
    # 2. Extract v1 and v2 feature matrices for training
    print("Extracting feature matrices for training pairs...")
    X_train_v1 = []
    X_train_v2 = []
    for sid, cid in train_pairs:
        s1 = s1_feats[sid]
        c = cand_feats[cid]
        v1 = compute_v1_features(s1, c)
        v2 = compute_v2_features(s1, c, v1)
        X_train_v1.append(v1)
        X_train_v2.append(v2)
        
    X_train_v1 = np.array(X_train_v1, dtype=np.float32)
    X_train_v2 = np.array(X_train_v2, dtype=np.float32)
    y_train = np.array(train_labels, dtype=np.int32)
    
    # 3. Generate validation candidate pairs & features
    print(f"Generating validation candidate pairs for {len(val_s1_set)} S1 entities...")
    val_candidates_map = {}
    val_pairs = []
    for sid in val_s1_set:
        cands = blocker.query(s1_block_pre[sid])
        val_candidates_map[sid] = cands
        for cid in cands:
            if cid in cand_feats:
                val_pairs.append((sid, cid))
                
    print(f"Extracted {len(val_pairs)} validation candidate pairs.")
    X_val_v1 = []
    X_val_v2 = []
    for sid, cid in val_pairs:
        s1 = s1_feats[sid]
        c = cand_feats[cid]
        v1 = compute_v1_features(s1, c)
        v2 = compute_v2_features(s1, c, v1)
        X_val_v1.append(v1)
        X_val_v2.append(v2)
        
    X_val_v1 = np.array(X_val_v1, dtype=np.float32)
    X_val_v2 = np.array(X_val_v2, dtype=np.float32)
    
    # Common LightGBM params (MIT licensed)
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 45,
        "min_child_samples": 40,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 250,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": 4
    }
    
    # -------------------------------------------------------------
    # MODEL A: LightGBM trained with v1 Core Features
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("TRAINING MODEL 1: LightGBM with v1 Core Features")
    print("="*70)
    model_v1 = lgb.LGBMClassifier(**lgb_params)
    model_v1.fit(X_train_v1, y_train)
    val_preds_proba_v1 = model_v1.predict_proba(X_val_v1)[:, 1]
    
    # Evaluate across decision thresholds
    best_thresh_v1, best_m_v1 = evaluate_threshold_grid(val_pairs, val_preds_proba_v1, val_gt, val_s1_set)
    print(f"Model v1 Optimal Threshold: {best_thresh_v1:.2f} | Validation Macro F0.5: {best_m_v1['macro_f05']:.4f}")
    
    # -------------------------------------------------------------
    # MODEL B: LightGBM trained with v2 Extended Features
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("TRAINING MODEL 2: LightGBM with v2 Extended Features (Ablation)")
    print("="*70)
    model_v2 = lgb.LGBMClassifier(**lgb_params)
    model_v2.fit(X_train_v2, y_train)
    val_preds_proba_v2 = model_v2.predict_proba(X_val_v2)[:, 1]
    
    best_thresh_v2, best_m_v2 = evaluate_threshold_grid(val_pairs, val_preds_proba_v2, val_gt, val_s1_set)
    print(f"Model v2 Optimal Threshold: {best_thresh_v2:.2f} | Validation Macro F0.5: {best_m_v2['macro_f05']:.4f}")
    
    # Incremental lift
    lift = best_m_v2["macro_f05"] - best_m_v1["macro_f05"]
    baseline_lift = best_m_v2["macro_f05"] - 0.7494
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON & VALIDATION LIFT SUMMARY")
    print("="*70)
    print(f"Phase 4 Rule Baseline F0.5:       0.7494")
    print(f"LightGBM v1 Core F0.5:            {best_m_v1['macro_f05']:.4f} (+{best_m_v1['macro_f05'] - 0.7494:.4f} vs Baseline)")
    print(f"LightGBM v2 Extended F0.5:        {best_m_v2['macro_f05']:.4f} (+{lift:.4f} vs v1, +{baseline_lift:.4f} vs Baseline)")
    print(f"Macro Precision (v2):             {best_m_v2['macro_precision']:.4f}")
    print(f"Macro Recall (v2):                {best_m_v2['macro_recall']:.4f}")
    print(f"Singleton Accuracy (v2):          {best_m_v2['singleton_acc']:.4f}")
    print("="*70)
    
    # Feature Importance Analysis
    print("\nTop 15 Feature Importances (v2):")
    importances = model_v2.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for rank, idx in enumerate(sorted_idx[:15], 1):
        print(f"  {rank:2d}. {V2_FEATURE_NAMES[idx]:<30} (split={importances[idx]})")
        
    # Check calibration
    prob_true, prob_pred = calibration_curve(
        [1 if cid in val_gt.get(sid, set()) else 0 for sid, cid in val_pairs],
        val_preds_proba_v2,
        n_bins=10
    )
    print("\nModel Calibration Curve (Predicted Prob vs Empirical True Match Prob):")
    for p_pred, p_true in zip(prob_pred, prob_true):
        print(f"  Pred: {p_pred:.3f} -> True: {p_true:.3f}")
        
    # Save the best model artifact
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model_artifact = {
        "model": model_v2,
        "feature_names": V2_FEATURE_NAMES,
        "optimal_threshold": best_thresh_v2,
        "metrics": best_m_v2
    }
    artifact_path = os.path.join(MODEL_SAVE_DIR, "lgbm_model_v2.pkl")
    with open(artifact_path, "wb") as f:
        pickle.dump(model_artifact, f)
    print(f"\nSaved production model artifact to {artifact_path}")
    return model_artifact

def evaluate_threshold_grid(
    val_pairs: List[Tuple[str, str]],
    probabilities: np.ndarray,
    val_gt: dict,
    val_s1_set: set
) -> Tuple[float, dict]:
    """Search optimal decision threshold for macro F0.5."""
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    best_thresh = 0.50
    best_f05 = 0.0
    best_m = None
    
    # Group pairs by S1
    s1_to_pair_indices = {sid: [] for sid in val_s1_set}
    for idx, (sid, cid) in enumerate(val_pairs):
        s1_to_pair_indices[sid].append((cid, idx))
        
    for thresh in thresholds:
        preds = {}
        for sid, c_list in s1_to_pair_indices.items():
            matched = {cid for cid, p_idx in c_list if probabilities[p_idx] >= thresh}
            preds[sid] = matched
            
        m = compute_macro_f05(val_gt, preds)
        if m["macro_f05"] > best_f05:
            best_f05 = m["macro_f05"]
            best_thresh = thresh
            best_m = m
            
    return best_thresh, best_m

if __name__ == "__main__":
    run_experiment()
