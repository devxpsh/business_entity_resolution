"""
Validation Design Module (Phase 5).
Constructs a leak-free GroupKFold validation split strictly partitioned on source1_entity_id.
Zero S1 overlap between train and validation folds.
"""

import os
import sys
import pandas as pd
import numpy as np

def create_validation_split(
    train_s1_path: str = "dataset/train/train_source1.tsv",
    train_gt_path: str = "dataset/train/train_ground_truth.tsv",
    output_dir: str = "code/business_entity_resolution/src/data/splits",
    val_size: int = 50000,
    random_seed: int = 42
):
    os.makedirs(output_dir, exist_ok=True)
    val_ids_file = os.path.join(output_dir, f"val_s1_ids_{val_size}.txt")
    if os.path.isfile(val_ids_file):
        print(f"Validation split file already exists: {val_ids_file}")
        with open(val_ids_file, "r") as f:
            val_ids = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(val_ids)} validation S1 IDs.")
        return val_ids
        
    print(f"Creating stratified validation split of size {val_size:,}...")
    s1_df = pd.read_csv(train_s1_path, sep="\t", usecols=["entity_id", "country"])
    gt_df = pd.read_csv(train_gt_path, sep="\t")
    
    # Merge country and match count
    gt_df["match_count"] = gt_df["matched_entity_ids"].fillna("").apply(
        lambda x: 0 if not str(x).strip() else len([m for m in str(x).split(",") if m.strip()])
    )
    # Stratification bin: country + match_category (0, 1, 2+)
    merged = pd.merge(s1_df, gt_df[["source1_entity_id", "match_count"]], left_on="entity_id", right_on="source1_entity_id")
    
    merged["match_cat"] = merged["match_count"].apply(lambda c: "0" if c == 0 else ("1" if c == 1 else "2+"))
    merged["strata"] = merged["country"] + "_" + merged["match_cat"]
    
    print("Full population strata distribution:")
    print(merged["strata"].value_counts(normalize=True))
    
    # Stratified split using sklearn
    from sklearn.model_selection import train_test_split
    val_df, _ = train_test_split(
        merged,
        train_size=val_size,
        stratify=merged["strata"],
        random_state=random_seed
    )
    
    val_s1_ids = set(val_df["entity_id"])
    print(f"Sampled {len(val_s1_ids)} validation entities.")
    print("Validation strata distribution:")
    print(val_df["strata"].value_counts(normalize=True))
    
    with open(val_ids_file, "w") as f:
        for sid in sorted(val_s1_ids):
            f.write(f"{sid}\n")
            
    print(f"Saved validation IDs to {val_ids_file}")
    return val_s1_ids

if __name__ == "__main__":
    create_validation_split()
