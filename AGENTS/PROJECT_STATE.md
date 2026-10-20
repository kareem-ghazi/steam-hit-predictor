# Project State: Steam Hit Predictor

## 1. Completed Components
- **Environment Setup:** Virtual environment `.venv` configured using `--system-site-packages` from master Conda environment (`C:\Users\kareemghazi\anaconda3\envs\master`). `xgboost` installed.
- **Dataset Ingestion:** Kaggle dataset loaded from `raw_data/steam.csv`.
- **Data Cleaning & Preprocessing (`ml-pipeline/data_prep.py`):**
  - Deduplicated dataset and verified uniqueness on `appid`.
  - Canonicalized text formatting: trimmed extraneous whitespace and standardized semicolon delimiters with alphabetical token sorting (e.g. `action;indie`).
  - Imputed missing metadata (`developer`, `publisher`).
  - Clamped numeric columns (`price`, `playtime`, `ratings`) to valid non-negative ranges.
  - Engineered `total_reviews` and `approval_rate` (`positive_ratings / total_reviews`).
  - Engineered `rating_text` reflecting official Steam review tiers (Overwhelmingly Positive, Very Positive, Positive, Mostly Positive, Mixed, Mostly Negative, Negative, Very Negative, Overwhelmingly Negative, Not Enough Reviews), factoring in review volume.
  - Engineered `is_hit` target variable (`1` if `approval_rate >= 0.80` and `total_reviews >= 50`, else `0`).
  - Added mock LLM vibe scores (`hype_score`, `story_score`, `nostalgia_score`) scaled from 1.0 to 10.0.
  - Dropped target-leakage columns (`positive_ratings`, `negative_ratings`, `total_reviews`, `approval_rate`).
  - Stratified 80% / 10% / 10% train/valid/test split based on `is_hit`.
  - **Ordinal Encoding:** Fitted `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)` strictly on `genres` and `platforms` from the training split, encoded all splits, and persisted the encoder artifact to `saved_models/ml_models/ordinal_encoder.pkl`.
  - Saved clean splits to `cleaned_data/train.csv`, `cleaned_data/valid.csv`, and `cleaned_data/test.csv`.
- **Model Training & Evaluation (`ml-pipeline/xgboost_train.py`):**
  - Updated `SteamFeatureTransformer` to format features using ordinal encoded categories:
    `['price', 'genres_encoded', 'platforms_encoded', 'hype_score', 'story_score', 'nostalgia_score']`.
  - Trained `XGBClassifier` with `scale_pos_weight` and monitored validation loss against `cleaned_data/valid.csv`.
  - Evaluated on holdout `cleaned_data/test.csv`:
    - **Accuracy:** `94.02%`
    - **Precision:** `80.68%`
    - **Recall:** `92.27%`
    - **F1-Score:** `0.8608`
  - Saved model artifact with transformer and encoder to `saved_models/ml_models/xgboost_hit_predictor.pkl`.

## 2. Newly Installed Dependencies
- `xgboost` (installed in `.venv`)

## 3. Current Data Flow
```text
raw_data/steam.csv
       │
       ▼
[ml-pipeline/data_prep.py]
  - Cleaning & canonical formatting
  - Feature Engineering (approval_rate, rating_text, is_hit, mock vibe scores)
  - Leakage column dropping
  - Stratified 80 / 10 / 10 Split
  - Ordinal Encoding for genres & platforms
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
cleaned_data/                               saved_models/ml_models/ordinal_encoder.pkl
  ├── train.csv (80% - 21,660 rows)              │
  ├── valid.csv (10% - 2,707 rows)               │
  └── test.csv  (10% - 2,708 rows)               │
       │                                         │
       ▼                                         │
[ml-pipeline/xgboost_train.py] ◄─────────────────┘
  - SteamFeatureTransformer (6 features: price, genres_encoded, platforms_encoded, hype, story, nostalgia)
  - XGBClassifier trained with valid.csv early-stopping monitoring
  - Holdout test evaluation (94.02% Acc, 0.8608 F1)
       │
       ▼
saved_models/ml_models/xgboost_hit_predictor.pkl
```

## 4. Next Step
- **Target File:** `llm-pipeline/inference_baseline.py`
- **Goal:** Phase 2: LLM Baselines - Run chosen local model (e.g. Llama-3-8B / Mistral) end-to-end on Steam game descriptions to establish qualitative baseline outputs.
