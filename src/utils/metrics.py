"""
Metric computation strictly following the problem statement:
Macro-averaged F0.5 across all Source 1 entities, including singletons.
"""

from typing import Dict, List, Set, Union
import numpy as np

def compute_entity_f05(true_mids: Set[str], pred_mids: Set[str]) -> float:
    """Compute F0.5 score for a single Source 1 entity.
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    Singletons:
      - true empty, pred empty: 1.0
      - true empty, pred non-empty: 0.0
      - true non-empty, pred empty: 0.0
    """
    n_true = len(true_mids)
    n_pred = len(pred_mids)
    
    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0
    if n_pred == 0:
        return 0.0
        
    tp = len(true_mids & pred_mids)
    if tp == 0:
        return 0.0
        
    precision = tp / n_pred
    recall = tp / n_true
    
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return (1.25 * precision * recall) / denom

def compute_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]]
) -> Dict[str, float]:
    """Compute macro-averaged F0.5 across all Source 1 entities in ground truth.
    Returns dictionary with overall macro F0.5, precision, recall, and singleton accuracy.
    """
    scores = []
    precisions = []
    recalls = []
    singleton_scores = []
    matched_scores = []
    
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        n_true = len(true_set)
        n_pred = len(pred_set)
        
        f05 = compute_entity_f05(true_set, pred_set)
        scores.append(f05)
        
        if n_true == 0:
            singleton_scores.append(f05)
            precisions.append(1.0 if n_pred == 0 else 0.0)
            recalls.append(1.0)
        else:
            matched_scores.append(f05)
            tp = len(true_set & pred_set)
            precisions.append(tp / n_pred if n_pred > 0 else 0.0)
            recalls.append(tp / n_true)
            
    return {
        "macro_f05": float(np.mean(scores)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "singleton_acc": float(np.mean(singleton_scores)) if singleton_scores else 1.0,
        "matched_entity_f05": float(np.mean(matched_scores)) if matched_scores else 0.0,
        "n_entities": len(scores)
    }

if __name__ == "__main__":
    # Test on the example from problem statement README
    true_example = {"S2-00047", "S3-00812"}
    pred_example = {"S2-00047", "S2-00193", "S3-00812"}
    score = compute_entity_f05(true_example, pred_example)
    print(f"Worked example score: {score:.3f} (Expected: 0.714)")
    assert abs(score - 0.714) < 0.001, f"Expected 0.714, got {score}"
    
    # Test singleton: both empty
    assert compute_entity_f05(set(), set()) == 1.0
    # Test singleton: false positive
    assert compute_entity_f05(set(), {"S2-00001"}) == 0.0
    # Test true match: false negative (empty pred)
    assert compute_entity_f05({"S2-00001"}, set()) == 0.0
    print("All metric unit tests passed.")
