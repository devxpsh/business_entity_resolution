# Business Entity Resolution Pipeline
### Amazon ML Challenge 2026

An enterprise-grade, memory-safe, open-set business entity resolution pipeline built for the Amazon ML Challenge 2026. Resolves ambiguous, noisy business records across multiple disparate sources (Source 1, Source 2, Source 3) into deduplicated clusters.

---

## 1. System Architecture

The pipeline uses a multi-stage architecture strictly adhering to the competition constraints:

```
[Raw Sources: S1, S2, S3]
             │
             ▼
[Phase 1: Canonical Non-Destructive Normalization]
  • NFKD Unicode decomposition & diacritic stripping (unidecode)
  • Case-folding & punctuation removal (preserving semantic connectors '&', '+')
  • Parallel non-destructive mined abbreviation & typo expansion
  • Phonetic Double-Metaphone encoding & alphanumeric token extraction
             │
             ▼
[Phase 2: Cascaded Inverted Multi-Blocker]
  • Exact canonical name matching
  • Token-sorted canonical name matching
  • Legal-suffix stripped core name matching
  • Significant name token inverted index (bounded posting lists)
  • Phonetic core token inverted index
  • Alphanumeric address code index (e.g. unit/suite IDs)
  • Significant address token index & rare numeric address tokens
  • Recall ceiling: 93.3% - 97.1% on validation with ~50-60 candidates/entity
             │
             ▼
[Phase 3: Pairwise Feature Engineering (23 Layered Features)]
  • Name: Levenshtein ratio, Jaro-Winkler, Token Sort/Set ratios, Token Jaccard
  • Address: Normalized Levenshtein, Token Sort ratio, Address Jaccard, missingness flag
  • Numeric/Code: Exact number intersection count, number overlap ratio, code match
  • Extended: Phonetic token overlap, acronym equality, synonym-expanded similarities
             │
             ▼
[Phase 6: Primary LightGBM Classifier]
  • Objective: Binary log-loss, GBDT, 250 trees, max depth/leaves tuned to prevent overfitting
  • Hard-negative mining: 1:1.4 positive to hard-negative ratio
  • Calibration verified on held-out validation set
             │
             ▼
[Phase 7: Entity-Level Decision Layer]
  • Calibrated decision thresholding (optimal $\tau = 0.85$ on macro $F_{0.5}$)
  • Strict subset enforcement: matched_entity_ids $\subseteq$ candidate_entity_ids
             │
             ▼
[Phase 9: Streaming Open-Set Inference]
  • Dynamic country partitioning (handles open-set countries such as France seamlessly)
  • Chunked streaming with constant memory footprint (<3.5 GB RSS)
  • Output files: output/matching_results.tsv and output/candidate_pairs.tsv
```

---

## 2. Environment & Installation

### Requirements
- Python 3.9+ (tested on Python 3.14 on macOS arm64 / Linux x86_64)
- CPU-only execution supported (no GPU required)

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt
```

---

## 3. Reproduction Instructions

All commands are run from the repository root directory (`student_resource/`):

### Step 1: Run Data Audit & Verification (Phase 0)
```bash
python3 eda_phase0.py
```
Audits all 26.4 million records, confirms field schemas, verifies the 100% country partition invariant, and records singleton statistics (5.58% singletons, 89.02% multi-matches).

### Step 2: Evaluate Phase 4 Simple Rule Baseline
```bash
python3 code/business_entity_resolution/src/model/baseline.py
```
Establishes the non-negotiable anchor performance on the leak-free validation set:
- Baseline Macro $F_{0.5}$: **0.7494**
- Baseline Macro Precision: **0.8520**
- Baseline Macro Recall: **0.5832**

### Step 3: Train Primary LightGBM Model & Run Ablations (Phases 5, 6, 7)
```bash
python3 code/business_entity_resolution/src/model/train_lgbm.py
```
Trains the model on mined positive and hard-negative pairs with strict group partitioning by `source1_entity_id`, verifies calibration, and evaluates v1 vs v2 feature lift:
- LightGBM v1 Core Features $F_{0.5}$: **0.9537** (+0.2043 lift over baseline)
- LightGBM v2 Extended Features $F_{0.5}$: **0.9541** (+0.2047 lift over baseline)
- Decision Layer Ablation (Strategy A at $\tau=0.85$): **0.9634 Macro $F_{0.5}$** (Precision: 0.9748, Recall: 0.9459)

### Step 4: Run End-to-End Test Set Inference (Phase 9)
```bash
python3 code/business_entity_resolution/src/inference/predict.py --threshold 0.85
```
Streams all 1,732,544 test Source 1 entities across France, US, and India, generating:
- `output/candidate_pairs.tsv`
- `output/matching_results.tsv`

### Step 5: Official Submission Validation
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
Runs the official submission validator to guarantee exit code 0 (`PASS`) before packaging.

---

## 4. Key Engineering Invariants

1. **Open-Set Country Support**: Country names are dynamically partitioned from data (`df['country'].unique()`). No country names or lists are hardcoded.
2. **Subset Constraint Guaranteed**: Final matches are filtered directly from each entity's candidate set, guaranteeing `matched_entity_ids` is always a strict subset of `candidate_entity_ids`.
3. **Tab-Separated Output**: Output files are written line-by-line with standard `\t` delimiters, avoiding quoting bugs and malformed lines.
4. **Zero External Dependencies / Fair Play**: No external APIs, web requests, geocoding lookups, or unverified weight downloads are executed during inference.
