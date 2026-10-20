# Project State: Steam Hit Predictor

## 1. Completed Components
- **Phase 1: Data Engineering & Tabular ML:**
  - Configured project `.venv` using master Conda environment packages with `--system-site-packages`.
  - Processed `raw_data/steam.csv` with deduplication, null imputation, non-negative numeric validation, and canonical formatting.
  - Engineered target `is_hit` (`approval_rate >= 0.80` & `total_reviews >= 50`) and official Steam review tiers (`rating_text`).
  - Stratified 80% / 10% / 10% train/valid/test split saved to `cleaned_data/train.csv` (21,660 rows), `cleaned_data/valid.csv` (2,707 rows), and `cleaned_data/test.csv` (2,708 rows).
  - Applied and saved `OrdinalEncoder` for `genres` and `platforms` (`saved_models/ml_models/ordinal_encoder.pkl`).
  - Trained `XGBClassifier` with validation tracking; achieved **94.02% Accuracy** and **0.8608 F1-Score** on holdout test set (`saved_models/ml_models/xgboost_hit_predictor.pkl`).
- **Phase 2: LLM Baselines:**
  - Created `.env` file for `HF_TOKEN` configuration.
  - Implemented `llm-pipeline/inference_baseline.py`:
    - Loaded local instruct LLM onto NVIDIA RTX 4060 GPU (`cuda:0`).
    - Configured system prompts and JSON extraction parser with regex fallback.
    - Verified qualitative score generation (`hype_score`, `story_score`, `nostalgia_score`) on real Steam games (Counter-Strike, Portal, Dota 2) with 100% valid JSON parsing.
  - Implemented `llm-pipeline/prompting_baseline.py`:
    - Benchmarked **Zero-Shot**, **Few-Shot**, and **Chain-of-Thought (CoT)** prompting across representative Steam games.
    - Recorded metrics to `llm-pipeline/prompting_baseline_results.json`:
      - **Few-Shot:** 100.0% JSON validity rate, 2.284s avg latency (fastest and most consistent).
      - **Zero-Shot:** 100.0% JSON validity rate, 2.993s avg latency.
      - **Chain-of-Thought:** 40.0% JSON validity rate, 12.586s avg latency (identifying verbosity and token cutoff challenges, strongly justifying Phase 3 fine-tuning).

## 2. Newly Installed / Configured Dependencies
- `xgboost` (installed in `.venv`)
- `python-dotenv` (installed in `.venv`)
- `.env` configured for `HF_TOKEN`
- (Inherited from master environment: `torch` 2.14+cu130, `transformers` 5.18.0, `accelerate`, `scikit-learn`, `pandas`)

## 3. Current Data Flow
```text
raw_data/steam.csv ───► [ml-pipeline/data_prep.py]
                             │
                             ├─► cleaned_data/ (train.csv, valid.csv, test.csv)
                             ├─► saved_models/ml_models/ordinal_encoder.pkl
                             ▼
                     [ml-pipeline/xgboost_train.py]
                             │
                             ▼
                     saved_models/ml_models/xgboost_hit_predictor.pkl
                                     ▲
                                     │ (Tabular + Vibe Features)
                                     │
raw_data/steam_description_data.csv ─┴─► [llm-pipeline/inference_baseline.py]
                                         [llm-pipeline/prompting_baseline.py]
                                           - Extracts Hype, Story, Nostalgia
                                           - Benchmarked Zero vs Few vs CoT
```

## 4. Next Step
- **Target File:** `llm-pipeline/qlora_training.py`
- **Goal:** Phase 3: LLM Fine-Tuning & Main Build - Construct instruction dataset converting Steam descriptions to target JSON vibe scores, configure 4-bit QLoRA with `bitsandbytes` and `peft`/`SFTTrainer`, train the adapter, save to `saved_models/llm_adapters/`, and perform before/after evaluation.
