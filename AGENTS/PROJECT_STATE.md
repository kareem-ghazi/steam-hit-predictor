# Project State: Steam Hit Predictor

## 1. Completed Components
- **Multi-Dataset Ingestion & Harmonization:**
  - Downloaded and organized 3 additional Steam datasets into dedicated folders in `raw_data/`:
    - `raw_data/artermiloff_steam_games_dataset/`: March 2025 Steam games dataset (89,618 records).
    - `raw_data/fronkongames_steam_games_dataset/`: Complete Steam games archive (125,855 records).
    - `raw_data/crainbramp_steam_dataset_2025/`: Multi-modal gaming analytics dataset — **CSV package only** (180MB `applications.csv`, `reviews.csv`, metadata CSVs; excluding the 11.6GB image/embedding archive).
    - Existing reference dataset: `raw_data/steam_games_cleaned/` (original 27,075 records).
  - Built [raw_data/merge_datasets.py](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/raw_data/merge_datasets.py):
    - Conformed strictly to the 18 columns of `raw_data/steam.csv` and 4 columns of `raw_data/steam_description_data.csv`.
    - Fully harmonized platforms, languages, owners brackets, developers, tags, and pricing.
    - Dropped duplicates by `appid`, filtered out missing/NA records, and aligned descriptions 1-to-1.
    - Stored unified merged files directly in `raw_data/` (`raw_data/steam.csv` and `raw_data/steam_description_data.csv`) without separate isolated folders: **85,249 total 100% clean, non-null games**.
- **Phase 1: Data Engineering & Tabular ML:**
  - Configured project `.venv` using master Conda environment packages with `--system-site-packages`.
  - Embedded descriptions directly into splits: `short_description`, `about_the_game`, `detailed_description`, and `clean_description` are direct columns inside `cleaned_data/train.csv` (68,199 rows), `cleaned_data/valid.csv` (8,525 rows), and `cleaned_data/test.csv` (8,525 rows).
  - **Dedicated Heuristic Vibe Score Generator ([generate_vibe_scores.py](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/generate_vibe_scores.py)):**
    - Evaluates genuine description-correlated heuristic scoring across all 85,249 catalog games.
    - **Hype Score (1.0 - 10.0):** (Mean: 5.23, Std: 1.19)
    - **Story Score (1.0 - 10.0):** (Mean: 5.85, Std: 2.24)
    - **Nostalgia Score (1.0 - 10.0):** (Mean: 4.79, Std: 1.48)
    - Fully integrated into [ml-pipeline/data_prep.py](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/ml-pipeline/data_prep.py).
  - **Organized column hierarchy with target at end**:
    1. Identifiers & Metadata: `appid`, `name`, `release_date`, `english`, `developer`, `publisher`, `platforms`, `required_age`, `categories`, `genres`, `steamspy_tags`, `achievements`, `average_playtime`, `median_playtime`, `owners`, `price`
    2. Steam Descriptions: `short_description`, `about_the_game`, `detailed_description`, `clean_description`
    3. Ordinal Encoded Features: `genres_encoded`, `platforms_encoded`, `publisher_encoded`
    4. LLM Vibe Scores: `hype_score`, `story_score`, `nostalgia_score`
    5. Steam Rating Tier: `rating_text`
    6. **Target Variable**: `is_hit` (**placed at the very end**)
  - Applied and saved `OrdinalEncoder` across `genres`, `platforms`, and `publisher` (`saved_models/ml_models/ordinal_encoder.pkl`).
  - Retrained `XGBClassifier` in [ml-pipeline/xgboost_train.py](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/ml-pipeline/xgboost_train.py) on expanded 85,249-game dataset:
    - Holdout test evaluation: **69.34% Accuracy**, **67.86% Recall**, **35.80% Precision**, **0.4687 F1-Score**.
    - Saved model artifact to `saved_models/ml_models/xgboost_hit_predictor.pkl`.
- **Phase 2: LLM Baselines & Containerization:**
  - Created `.env` file for Hugging Face authentication (`HF_TOKEN`).
  - Implemented `llm-pipeline/inference_baseline.py` using **Qwen3.5-0.8B** on GPU (`cuda:0`):
    - System prompt and JSON payload parser with regex fallback.
    - Real-time RAM & VRAM tracking after every generation.
    - Exports metrics to `llm-pipeline/inference_baseline_results.json` (100.0% validity rate, 5.328s avg latency, ~1.97 GB Process RAM, ~1.41 GB GPU VRAM).
  - Implemented `llm-pipeline/prompting_baseline.py`:
    - Benchmarked **Zero-Shot**, **Few-Shot**, and **Chain-of-Thought (CoT)** prompting strategies.
    - Structured CoT prompt with concise reasoning dimensions, achieving **100.0% validity** across all three strategies:
      - **Zero-Shot:** 100.0% validity, 75.0% accuracy (±2.0 tolerance), 1.58 MAE, 5.694s latency.
      - **Few-Shot:** 100.0% validity, 75.0% accuracy (±2.0 tolerance), 1.67 MAE, 4.093s latency.
      - **Chain-of-Thought:** 100.0% validity, 66.7% accuracy (±2.0 tolerance), 1.83 MAE, 13.735s latency.
    - Saved benchmark metrics to `llm-pipeline/prompting_baseline_results.json`.
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
- `numpy`

## 3. Current Data Flow
```text
raw_data/steam.csv ────────────────┐
raw_data/steam_description_data.csv─┴─► [ml-pipeline/data_prep.py]
                                              ▲
                                              │ Uses [generate_vibe_scores.py]
                                              │ (Calculates real description & metadata vibes)
                                              │
                                              ├─► cleaned_data/ (train.csv, valid.csv, test.csv)
                                              │   (short, about, detailed, clean_description; target is_hit at end)
                                              ├─► saved_models/ml_models/ordinal_encoder.pkl
                                              │   (genres, platforms, publisher)
                                              ▼
                                      [ml-pipeline/xgboost_train.py]
                                              │
                                              ▼
                                      saved_models/ml_models/xgboost_hit_predictor.pkl
                                      (72.60% Acc, 67.22% Recall, 35% Vibe Importance)
                                              ▲
                                              │
                                      [llm-pipeline/inference_baseline.py]
                                        -> llm-pipeline/inference_baseline_results.json (100% validity)
                                      [llm-pipeline/prompting_baseline.py]
                                        -> llm-pipeline/prompting_baseline_results.json (100% validity)
```

## 4. Next Step
- **Target File:** `llm-pipeline/qlora_training.py`
- **Goal:** Phase 3: LLM Fine-Tuning & Main Build - Construct an instruction dataset pairing cleaned Steam descriptions with target JSON vibe scores, configure 4-bit QLoRA using `bitsandbytes` and `peft`/`SFTTrainer`, execute training on `cuda:0`, save the adapter to `saved_models/llm_adapters/`, and run before/after evaluation.
