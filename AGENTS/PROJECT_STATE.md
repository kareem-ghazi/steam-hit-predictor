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
    - **Modular Columnar Split (`raw_data/dataset_attributes/`):** Partitioned column-by-column into individual CSV files inside `raw_data/dataset_attributes/`. Each file contains `['appid', <attribute>]` (51 attribute files).
    - Data card generated at [raw_data/dataset_attributes/DATA_CARD.md](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/raw_data/dataset_attributes/DATA_CARD.md).
- **Phase 1: Full Data Engineering Pipeline in `data_prep.py` ([ml-pipeline/data_prep.py](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/ml-pipeline/data_prep.py)):**
  - **Modular Source Ingestion:** Loads required attributes directly from `raw_data/dataset_attributes/`.
  - **Strict Entity Filtering:** Dropped rows missing `developer` or `publisher` entirely (no imputation), and dropped invalid/missing name, platforms, or genres.
  - **Review Volume Filtering First:** Filtered out games with fewer than 50 total reviews first (`total_reviews >= 50`), yielding **27,810 robust, well-reviewed games**.
  - **Target Variable `is_hit`:** Formed target label `is_hit = 1 if approval_rate >= 0.80 else 0`, yielding a well-balanced distribution: **15,552 Hits (55.9%)** and **12,258 Non-Hits (44.1%)**.
  - **Leakage Columns Retained:** Retained `positive_ratings`, `negative_ratings`, `total_reviews`, and `approval_rate` in the dataset for safety measures and downstream telemetry.
  - **USD Price Unification & StandardScaler:** Converted non-USD currencies to USD using exchange rates; fitted `StandardScaler` strictly on training set prices to generate `price_scaled` (saved artifact: `saved_models/ml_models/price_scaler.pkl`).
  - **Dedicated Heuristic Vibe Score Generator ([ml-pipeline/generate_vibe_scores.py](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/ml-pipeline/generate_vibe_scores.py)):**
    - Calculated calibrated vibe scores **purely from non-description metadata** (genres, categories, steamspy tags, publisher prestige, ownership footprint, playtime, release year, pricing signals, and player approval rating).
    - **Zero Description Text Used:** Completely excludes description text to prevent data leakage during Phase 3 LLM fine-tuning.
    - **Hype Score (1.0 - 10.0):** (Mean: 5.86, Std: 1.21, Min: 3.0, Max: 10.0)
    - **Story Score (1.0 - 10.0):** (Mean: 6.00, Std: 2.40, Min: 1.0, Max: 10.0)
    - **Nostalgia Score (1.0 - 10.0):** (Mean: 4.45, Std: 1.18, Min: 2.7, Max: 10.0)
  - **Complete Feature Preparation & Noise Pruning in `data_prep.py`:**
    - Transferred all categorical encoding and feature transformation logic (`SteamFeatureTransformer`) directly into `data_prep.py`.
    - **Pre-Release Leakage Prevention:** Removed `owners` completely from the feature matrix so the model relies purely on pre-release signals.
    - **Noise & Garbled Feature Pruning:**
      - Sanitized multi-label tokens via `parse_delimited_tokens`: eliminated empty bracket strings (`[]`, `['']`), foreign-language one-offs, and non-alphanumeric noise.
      - Enforced `min_genre_count = 20` for multi-hot genres, removing all 23 noisy one-off columns (e.g. `genre_____`, `genre___`, `genre___________`, `genre___casual____indie____rpg____strategy__`, `genre_a__o`, `genre_aventura`, etc.).
      - Eliminated `cat___` (caused by literal `[]` in categories) and standardized all column names with single underscores (e.g. `tag_point_click`, `genre_design_illustration`, `cat_single_player`), avoiding triple or consecutive underscores.
      - Explicit check in `fit()` and `transform()` to prune all-zero features across the training set.
      - Verified **0 all-zero columns** across all three splits (train, valid, test).
    - **Multi-Label Decomposition (`MultiLabelBinarizer`):**
      - `genres`: 26 clean standard Steam genres (`genre_<name>`) + `num_genres`.
      - `categories`: 53 clean categories with frequency >= 30 (`cat_<name>`) + `num_categories`.
      - `steamspy_tags`: Top 100 community tags (`tag_<name>`) + `num_tags`.
      - `platforms`: Platform indicators (`platform_windows`, `platform_mac`, `platform_linux`) + `num_platforms`.
    - **Native Categoricals & Entity Frequency Signals:**
      - `developer`: High-cardinality control with rare levels mapped to `other_developer` + `dev_game_count`.
      - `publisher`: Frequent publishers preserved, rare mapped to `other_publisher` + `pub_game_count`.
      - `platforms` & `primary_genre`: Encoded as native `pd.Categorical`.
    - **Total Clean Feature Space:** Exactly **200 model features** (215 total columns in `games.csv` including metadata, telemetry, and target `is_hit` at the end).
    - **Persisted Feature Transformer:** Saved transformer bundle to `saved_models/ml_models/feature_transformer.pkl`.
    - **Provided `load_prepared_data()`:** Helper function to reload splits with matching `pd.Categorical` definitions.
  - **Strictly Sorted Three Split Folders Architecture (`cleaned_data/`):**
    - Both `games.csv` and `games_descriptions.csv` across all splits are strictly sorted by `appid` ascending (`is_monotonic_increasing == True`).
    - Exact 1-to-1 row alignment verified by `appid` between `games.csv` and `games_descriptions.csv`.
    - `cleaned_data/train/`: `games.csv` (22,248 rows, 215 cols, 200 model features, target `is_hit` at end) and `games_descriptions.csv` (22,248 rows, 5 cols).
    - `cleaned_data/valid/`: `games.csv` (2,781 rows, 215 cols, 200 model features, target `is_hit` at end) and `games_descriptions.csv` (2,781 rows, 5 cols).
    - `cleaned_data/test/`: `games.csv` (2,781 rows, 215 cols, 200 model features, target `is_hit` at end) and `games_descriptions.csv` (2,781 rows, 5 cols).
- **Streamlined XGBoost Classifier ([ml-pipeline/xgboost_train.py](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/ml-pipeline/xgboost_train.py)):**
  - **Pure Model Training & Evaluation:** Stripped of all data preparation code. Loads already-prepared feature matrices directly from `cleaned_data/` via `load_prepared_data()`.
  - **Native Categorical Support:** Trained with `enable_categorical=True` and `tree_method='hist'`.
  - **Holdout Test Set Evaluation (2,781 games, 200 features):**
    - **Accuracy:** **96.40%** (2,681 / 2,781 correct predictions)
    - **Precision:** **95.55%**
    - **Recall:** **98.14%** (1,526 / 1,555 hits detected)
    - **F1-Score:** **96.83%**
    - **Validation Logloss:** **0.1062** (early stopping best iteration: 299)
    - **Top Predictors:** `nostalgia_score` (0.1545), `cat_multi_player` (0.0807), `genre_casual` (0.0705), `genre_action` (0.0693), `genre_massively_multiplayer` (0.0590), `story_score` (0.0552), `hype_score` (0.0468), `genre_adventure` (0.0300), `genre_free_to_play` (0.0286), `primary_genre` (0.0269), `cat_mmo` (0.0268), `publisher` (0.0250), `genre_rpg` (0.0229), `price` (0.0222), `cat_single_player` (0.0207).
  - Saved updated model artifact to `saved_models/ml_models/xgboost_hit_predictor.pkl`.
- **Phase 2: LLM Baselines & Containerization:**
  - Created `.env` file for Hugging Face authentication (`HF_TOKEN`).
  - Implemented `llm-pipeline/inference_baseline.py` using **Qwen3.5-0.8B** on GPU (`cuda:0`):
    - System prompt and JSON payload parser with regex fallback.
    - Real-time RAM & VRAM tracking after every generation.
    - Exports metrics to `llm-pipeline/baseline_results/inference_baseline_results.json` (100.0% validity rate, 5.328s avg latency, ~1.97 GB Process RAM, ~1.41 GB GPU VRAM).
  - Implemented `llm-pipeline/prompting_baseline.py`:
    - Benchmarked **Zero-Shot**, **Few-Shot**, and **Chain-of-Thought (CoT)** prompting strategies.
    - Structured CoT prompt with concise reasoning dimensions, achieving **100.0% validity** across all three strategies:
      - **Zero-Shot:** 100.0% validity, 75.0% accuracy (±2.0 tolerance), 1.58 MAE, 5.694s latency.
      - **Few-Shot:** 100.0% validity, 75.0% accuracy (±2.0 tolerance), 1.67 MAE, 4.093s latency.
      - **Chain-of-Thought:** 100.0% validity, 66.7% accuracy (±2.0 tolerance), 1.83 MAE, 13.735s latency.
    - Saved benchmark metrics to `llm-pipeline/baseline_results/prompting_baseline_results.json`.
  - **Adapted `llm-pipeline` to Three-Split `cleaned_data/` Dataset Architecture:**
    - Updated `inference_baseline.py` and `prompting_baseline.py` to seamlessly load and merge game attributes and descriptions directly from `cleaned_data/` (`train/`, `valid/`, `test/` pairing `games.csv` with `games_descriptions.csv` on `appid`).
    - Implemented shared dataset utilities (`load_cleaned_data`, `resolve_data_dir`, `resolve_output_path`) providing execution path flexibility.
    - Added configurable `--data_dir` CLI parameter to both scripts.
    - Dynamically bound benchmark evaluation ground truth to the metadata-derived non-leakage vibe scores (`hype_score`, `story_score`, `nostalgia_score`) computed by `generate_vibe_scores.py` and saved in `cleaned_data/`.
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
           - Drops missing dev/pub/name/genres entirely
           - Unifies price in USD & StandardScaler
           - Filters total_reviews >= 50 first
           - Computes approval_rate, rating_text, is_hit
           - Retains review leakage columns for safety
           - Generates metadata-only vibe scores via generate_vibe_scores.py (no text leakage)
           - Prunes low-frequency noise (min_genre_count=20, min_cat_count=30, top_n_tags=100)
           - Prunes all-zero features and malformed brackets/multi-underscores
           - Sorts train, valid, test splits strictly by appid ascending
           - Fits SteamFeatureTransformer (multi-hot binarization, native categoricals, frequency counts)
           - Persists price_scaler.pkl and feature_transformer.pkl
                       │
                       ▼
          cleaned_data/ (Pre-prepared with 200 features, sorted by appid)
          ├── train/
          │   ├── games.csv              (22,248 rows, 215 cols, 200 model features, target is_hit at end)
          │   └── games_descriptions.csv (22,248 rows, 5 cols: appid, short, about, detailed, clean)
          ├── valid/
          │   ├── games.csv              (2,781 rows, 215 cols, 200 model features)
          │   └── games_descriptions.csv (2,781 rows, 5 cols)
          └── test/
              ├── games.csv              (2,781 rows, 215 cols, 200 model features)
              └── games_descriptions.csv (2,781 rows, 5 cols)
                       │
         ┌─────────────┴────────────────────────┐
         │                                      │
         ▼                                      ▼
[ml-pipeline/xgboost_train.py]        [llm-pipeline/load_cleaned_data()]
- Loads pre-prepared features directly│
  via load_prepared_data()            ├──> [llm-pipeline/inference_baseline.py]
- Zero feature extraction in train    │    (RAM tracking + JSON schema validation)
- Trains XGBoost with early stopping  │
         │                            └──> [llm-pipeline/prompting_baseline.py]
         ▼                                 (Zero-shot, Few-shot, CoT benchmarks)
saved_models/ml_models/
xgboost_hit_predictor.pkl (96.40% test accuracy, 96.83% F1)
```

## 4. Next Step
- **Target File:** `llm-pipeline/qlora_training.py`
- **Goal:** Phase 3: LLM Fine-Tuning & Main Build - Construct an instruction dataset pairing cleaned Steam descriptions (`cleaned_data/train/games_descriptions.csv`) with target JSON vibe scores (`cleaned_data/train/games.csv`, free of textual description data leakage), configure 4-bit QLoRA using `bitsandbytes` and `peft`/`SFTTrainer`, execute training on `cuda:0`, save the adapter to `saved_models/llm_adapters/`, and run before/after evaluation on `cleaned_data/valid/` and `cleaned_data/test/`.
