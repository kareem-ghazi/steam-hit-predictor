# Project State: Steam Hit Predictor

## 1. Completed Components
- **Phase 1: Data Engineering & Tabular ML:**
  - Configured project `.venv` using master Conda environment packages with `--system-site-packages`.
  - Cleaned tabular Steam store dataset (`raw_data/steam.csv`): deduplication, null imputation, non-negative numeric clamping.
  - Cleaned and prepared Steam descriptions (`raw_data/steam_description_data.csv`): stripped HTML tags, unescaped HTML entities, normalized whitespace, and exported 27,334 records to `cleaned_data/steam_descriptions_cleaned.csv`.
  - Merged cleaned descriptions with the tabular dataset and engineered target `is_hit` (`approval_rate >= 0.80` & `total_reviews >= 50`) and official Steam review tiers (`rating_text`).
  - Added `english` language flag and cleaned `publisher` column.
  - Stratified 80% / 10% / 10% train/valid/test split saved to `cleaned_data/train.csv` (21,660 rows), `cleaned_data/valid.csv` (2,707 rows), and `cleaned_data/test.csv` (2,708 rows).
  - Applied and saved `OrdinalEncoder` across `genres`, `platforms`, and `publisher` (`saved_models/ml_models/ordinal_encoder.pkl`).
  - Retrained `XGBClassifier` on 8 features (`price`, `english`, `genres_encoded`, `platforms_encoded`, `publisher_encoded`, `hype_score`, `story_score`, `nostalgia_score`) with validation tracking.
  - Achieved **94.28% Accuracy**, **82.44% Precision**, **90.79% Recall**, and **0.8642 F1-Score** on the holdout test set (`saved_models/ml_models/xgboost_hit_predictor.pkl`).
- **Phase 2: LLM Baselines & Containerization:**
  - Created `.env` file for Hugging Face authentication (`HF_TOKEN`).
  - Implemented `llm-pipeline/inference_baseline.py` using **Qwen3.5-0.8B** on GPU (`cuda:0`):
    - System prompt and JSON payload parser with markdown and trailing comma sanitization.
    - Added real-time RAM & VRAM tracking after every generation (monitoring process RSS, system RAM used/total/%, and GPU allocated/reserved memory).
  - Implemented `llm-pipeline/prompting_baseline.py`:
    - Benchmarked **Zero-Shot**, **Few-Shot**, and **Chain-of-Thought (CoT)** prompting strategies.
    - Evaluated both **JSON Validity Rate** and **Accuracy Rate (MAE and % within ±2.0 of reference ground truth)**.
    - Saved benchmark metrics to `llm-pipeline/prompting_baseline_results.json`:
      - **Few-Shot:** 100.0% JSON validity, 75.0% accuracy, 1.38 MAE, 4.475s avg latency.
      - **Zero-Shot:** 100.0% JSON validity, 66.7% accuracy, 1.62 MAE, 6.709s avg latency.
      - **Chain-of-Thought:** 0.0% validity, excessive reasoning length causes format drift and token cutoff, providing empirical motivation for Phase 3 fine-tuning.
  - Created root [requirements.txt](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/requirements.txt) with pinned dependencies across data, ML, LLM, FastAPI, and test libraries.
  - Created root [Dockerfile](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/Dockerfile) containerizing the environment and FastAPI backend.

## 2. Dependencies & Runtime
- `xgboost`
- `psutil`
- `python-dotenv`
- `torch` (CUDA 12.1+ compatible, running on RTX 4060 Laptop GPU)
- `transformers` 5.18.0
- `accelerate`
- `scikit-learn`
- `pandas`

## 3. Current Data Flow
```text
raw_data/steam.csv ────────────────┐
raw_data/steam_description_data.csv─┴─► [ml-pipeline/data_prep.py]
                                              │
                                              ├─► cleaned_data/steam_descriptions_cleaned.csv
                                              ├─► cleaned_data/ (train.csv, valid.csv, test.csv)
                                              ├─► saved_models/ml_models/ordinal_encoder.pkl
                                              │   (genres, platforms, publisher)
                                              ▼
                                      [ml-pipeline/xgboost_train.py]
                                              │
                                              ▼
                                      saved_models/ml_models/xgboost_hit_predictor.pkl
                                      (94.28% Acc, 0.8642 F1)
                                              ▲
                                              │
                                      [llm-pipeline/inference_baseline.py]
                                      [llm-pipeline/prompting_baseline.py]
                                        - Powered by Qwen3.5-0.8B on CUDA
                                        - Real-time RAM & VRAM tracking
                                        - Accuracy & JSON validity metrics
```

## 4. Next Step
- **Target File:** `llm-pipeline/qlora_training.py`
- **Goal:** Phase 3: LLM Fine-Tuning & Main Build - Construct an instruction dataset pairing cleaned Steam descriptions with target JSON vibe scores, configure 4-bit QLoRA using `bitsandbytes` and `peft`/`SFTTrainer`, execute training on `cuda:0`, save the adapter to `saved_models/llm_adapters/`, and run before/after evaluation.
