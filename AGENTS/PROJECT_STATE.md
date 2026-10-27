# Project State: Steam Hit Predictor

## 1. Completed Components
- **Multi-Dataset Ingestion & Harmonization:**
  - Downloaded and organized 3 additional Steam datasets into dedicated folders in `raw_data/`:
    - `raw_data/crainbramp_steam_dataset_2025/`: Multi-modal gaming analytics dataset — **CSV package only** (180MB `applications.csv`, metadata CSVs, relational category/developer/genre/publisher mappings).
    - `raw_data/artermiloff_steam_games_dataset/`: March 2025 Steam games dataset (89,618 records).
    - `raw_data/fronkongames_steam_games_dataset/`: Complete Steam games archive (125,855 records).
    - Existing reference dataset: `raw_data/steam_games_cleaned/` (original 27,075 records).
  - Built and executed [raw_data/merge_datasets.py](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/raw_data/merge_datasets.py) (and mirrored in [raw_data/_scripts/merge_datasets.py](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/raw_data/_scripts/merge_datasets.py)):
    - **Crainbramp Game Integration:** Integrated all games from `crainbramp_steam_dataset_2025` (`type == 'game'`, 150,275 games) and unioned with reference games from `steam_games_cleaned` for **153,163 total clean, non-null games**.
    - Relational metadata (`developer`, `publisher`, `genres`, `categories`) mapped using fast dictionary indexing across `crainbramp`'s auxiliary relation tables.
    - Fully cross-filled ratings (`positive_ratings`, `negative_ratings`), `steamspy_tags`, `owners`, playtime (`average_playtime`, `median_playtime`), descriptions (`detailed_description`, `about_the_game`, `short_description`), and media metadata across `artermiloff`, `fronkongames`, and `steam_games_cleaned`.
    - **Modular Columnar Split (`raw_data/dataset_attributes/`):** Instead of monolithic `steam.csv` and `steam_description_data.csv`, the unified dataset is partitioned column-by-column into individual CSV files inside `raw_data/dataset_attributes/`. Each file contains `['appid', <attribute>]` (51 attribute files).
    - Data card generated at [raw_data/dataset_attributes/DATA_CARD.md](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/raw_data/dataset_attributes/DATA_CARD.md) documenting all attributes.
- **Phase 1: Data Engineering & Tabular ML ([ml-pipeline/data_prep.py](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/ml-pipeline/data_prep.py)):**
  - **Modular Source Ingestion:** Loads required attributes directly from `raw_data/dataset_attributes/`.
  - **Strict Entity Filtering:** Dropped rows missing `developer` or `publisher` entirely (no imputation), dropping 7,743 invalid records.
  - **Review Volume Filtering First:** Filtered out games with fewer than 50 total reviews first (`total_reviews >= 50`), yielding **27,971 robust, well-reviewed games**.
  - **Target Variable `is_hit`:** Formed target label `is_hit = 1 if approval_rate >= 0.80 else 0`, yielding a well-balanced distribution: **15,614 Hits (55.8%)** and **12,357 Non-Hits (44.2%)**.
  - **Leakage Columns Retained:** Retained `positive_ratings`, `negative_ratings`, `total_reviews`, and `approval_rate` in the dataset for safety measures and downstream telemetry.
  - **USD Price Unification & StandardScaler:** Converted non-USD currencies to USD using exchange rates; fitted `StandardScaler` strictly on training set prices to generate `price_scaled` (saved artifact: `saved_models/ml_models/price_scaler.pkl`).
  - **Categorical Columns Preserved:** Removed `OrdinalEncoder` from data prep, preserving raw categorical string formats for `genres`, `platforms`, `publisher`, `developer`, `categories`, and `steamspy_tags`.
  - **Dedicated Heuristic Vibe Score Generator ([ml-pipeline/generate_vibe_scores.py](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/ml-pipeline/generate_vibe_scores.py)):**
    - **Description-Leakage-Free Engine:** Calculates calibrated vibe scores **purely from non-description metadata** (genres, categories, steamspy tags, publisher prestige, ownership footprint, playtime, release year, pricing signals).
    - **Zero Description Text Used:** Completely excludes `short_description`, `about_the_game`, `detailed_description`, and `clean_description` to prevent target data leakage during Phase 3 LLM fine-tuning.
    - **Hype Score (1.0 - 10.0):** (Mean: 5.60, Std: 1.18, Min: 4.0, Max: 10.0)
    - **Story Score (1.0 - 10.0):** (Mean: 5.82, Std: 2.39, Min: 1.0, Max: 10.0)
    - **Nostalgia Score (1.0 - 10.0):** (Mean: 4.33, Std: 1.08, Min: 3.5, Max: 10.0)
  - **Three Split Folders Architecture (`cleaned_data/`):**
    - `cleaned_data/train/`: Contains `games.csv` (22,376 rows, 27 columns with target `is_hit` at the end) and `games_descriptions.csv` (22,376 rows, 5 columns: `appid`, `short_description`, `about_the_game`, `detailed_description`, `clean_description`).
    - `cleaned_data/valid/`: Contains `games.csv` (2,797 rows, 27 columns) and `games_descriptions.csv` (2,797 rows, 5 columns).
    - `cleaned_data/test/`: Contains `games.csv` (2,798 rows, 27 columns) and `games_descriptions.csv` (2,798 rows, 5 columns).
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
raw_data/dataset_attributes/*.csv (51 attribute files)
                       │
                       ▼
           [ml-pipeline/data_prep.py]
           - Drops missing dev/pub entirely
           - Unifies price in USD & StandardScaler
           - Filters total_reviews >= 50 first
           - Computes approval_rate, rating_text, is_hit
           - Retains review leakage columns for safety
           - Generates metadata-only vibe scores via generate_vibe_scores.py (no text leakage)
                       │
                       ▼
          cleaned_data/
          ├── train/
          │   ├── games.csv              (22,376 rows, 27 cols, target is_hit at end)
          │   └── games_descriptions.csv (22,376 rows, 5 cols: appid, short, about, detailed, clean)
          ├── valid/
          │   ├── games.csv              (2,797 rows, 27 cols)
          │   └── games_descriptions.csv (2,797 rows, 5 cols)
          └── test/
              ├── games.csv              (2,798 rows, 27 cols)
              └── games_descriptions.csv (2,798 rows, 5 cols)
                       │
                       ▼
          [ml-pipeline/xgboost_train.py]
                       │
                       ▼
          saved_models/ml_models/xgboost_hit_predictor.pkl
                       ▲
                       │
          [llm-pipeline/inference_baseline.py] -> 100% validity
          [llm-pipeline/prompting_baseline.py] -> 100% validity
```

## 4. Next Step
- **Target File:** `llm-pipeline/qlora_training.py`
- **Goal:** Phase 3: LLM Fine-Tuning & Main Build - Construct an instruction dataset pairing cleaned Steam descriptions with target JSON vibe scores (free of textual description data leakage), configure 4-bit QLoRA using `bitsandbytes` and `peft`/`SFTTrainer`, execute training on `cuda:0`, save the adapter to `saved_models/llm_adapters/`, and run before/after evaluation.
