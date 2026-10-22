# Project State: Steam Hit Predictor

## 1. Completed Components
- **Phase 1: Data Engineering & Tabular ML:**
  - Configured project `.venv` using master Conda environment packages with `--system-site-packages`.
  - Cleaned tabular Steam store dataset (`raw_data/steam.csv`): deduplication, null imputation, non-negative numeric clamping.
  - Cleaned and prepared Steam descriptions (`raw_data/steam_description_data.csv`): stripped HTML tags, unescaped HTML entities, normalized whitespace.
  - **Embedded descriptions directly into splits**: `short_description`, `about_the_game`, `detailed_description`, and `clean_description` are now direct columns inside `cleaned_data/train.csv` (21,660 rows), `cleaned_data/valid.csv` (2,707 rows), and `cleaned_data/test.csv` (2,708 rows).
  - **Organized column hierarchy with target at end**:
    1. Identifiers & Metadata: `appid`, `name`, `release_date`, `english`, `developer`, `publisher`, `platforms`, `required_age`, `categories`, `genres`, `steamspy_tags`, `achievements`, `average_playtime`, `median_playtime`, `owners`, `price`
    2. Steam Descriptions: `short_description`, `about_the_game`, `detailed_description`, `clean_description`
    3. Ordinal Encoded Features: `genres_encoded`, `platforms_encoded`, `publisher_encoded`
    4. LLM Vibe Scores: `hype_score`, `story_score`, `nostalgia_score`
    5. Steam Rating Tier: `rating_text`
    6. **Target Variable**: `is_hit` (**placed at the very end**)
  - Applied and saved `OrdinalEncoder` across `genres`, `platforms`, and `publisher` (`saved_models/ml_models/ordinal_encoder.pkl`).
  - Retrained `XGBClassifier` on 8 features (`price`, `english`, `genres_encoded`, `platforms_encoded`, `publisher_encoded`, `hype_score`, `story_score`, `nostalgia_score`) with validation tracking.
  - Achieved **94.28% Accuracy**, **82.44% Precision**, **90.79% Recall**, and **0.8642 F1-Score** on the holdout test set (`saved_models/ml_models/xgboost_hit_predictor.pkl`).
- **Phase 2: LLM Baselines & Containerization:**
  - Created `.env` file for Hugging Face authentication (`HF_TOKEN`).
  - Implemented `llm-pipeline/inference_baseline.py` using **Qwen3.5-0.8B** on GPU (`cuda:0`):
    - System prompt and JSON payload parser with markdown, trailing comma, and regex fallback handling.
    - Real-time RAM & VRAM tracking after every generation (process RSS, system RAM used/total/%, and GPU allocated/reserved memory).
    - **Results Export**: Exports aggregate metrics and per-game runs to `llm-pipeline/inference_baseline_results.json` (100.0% validity rate, 5.328s avg latency, ~1.97 GB Process RAM, ~1.41 GB GPU VRAM).
  - Implemented `llm-pipeline/prompting_baseline.py`:
    - Benchmarked **Zero-Shot**, **Few-Shot**, and **Chain-of-Thought (CoT)** prompting strategies.
    - **Resolved CoT Validity**: Restructured prompt with concise structured reasoning dimensions and token allocation, achieving **100.0% validity** across all three strategies:
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

## 3. Current Data Flow
```text
raw_data/steam.csv ────────────────┐
raw_data/steam_description_data.csv─┴─► [ml-pipeline/data_prep.py]
                                              │
                                              ├─► cleaned_data/ (train.csv, valid.csv, test.csv)
                                              │   (includes short, about, detailed, clean_description; target is_hit at end)
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
                                        -> llm-pipeline/inference_baseline_results.json (100% validity)
                                      [llm-pipeline/prompting_baseline.py]
                                        -> llm-pipeline/prompting_baseline_results.json (100% validity)
```

## 4. Next Step
- **Target File:** `llm-pipeline/qlora_training.py`
- **Goal:** Phase 3: LLM Fine-Tuning & Main Build - Construct an instruction dataset pairing cleaned Steam descriptions with target JSON vibe scores, configure 4-bit QLoRA using `bitsandbytes` and `peft`/`SFTTrainer`, execute training on `cuda:0`, save the adapter to `saved_models/llm_adapters/`, and run before/after evaluation.
