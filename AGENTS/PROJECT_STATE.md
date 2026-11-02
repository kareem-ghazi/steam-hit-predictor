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
- **Phase 1: Modular Data Engineering & Refactored `ml-pipeline/` Architecture:**
  - **Modular Pipeline Split:** Refactored monolithic preparation code into focused, maintainable, single-responsibility modules:
    - [ml-pipeline/config.py](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/ml-pipeline/config.py): Constants, FX exchange rates, Steam rating tiers logic, and token delimiter parsing.
    - [ml-pipeline/cleaner.py](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/ml-pipeline/cleaner.py): Attribute loading, HTML description unescaping and cleaning, price normalization, and strict data filtering.
    - [ml-pipeline/feature_transformer.py](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/ml-pipeline/feature_transformer.py): `SteamFeatureTransformer` class for multi-label binarization, native categorical handling, and automated all-zero feature pruning.
    - [ml-pipeline/data_prep.py](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/ml-pipeline/data_prep.py): High-level pipeline coordinator orchestrating feature engineering, ordered DataFrame construction, stratified 80/10/10 splitting, MinMaxScaler price scaling, saving, and loading.
    - [ml-pipeline/generate_vibe_scores.py](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/ml-pipeline/generate_vibe_scores.py): Heuristic vibe score engine derived purely from metadata (zero textual description leakage).
  - **Strict Entity & Volume Filtering:** Dropped rows missing `developer` or `publisher` entirely (no imputation), and filtered out games with fewer than 50 total reviews first (`total_reviews >= 50`), yielding **27,810 robust games**.
  - **Target Variable `is_hit`:** Formed target label `is_hit = 1 if approval_rate >= 0.80 else 0`, yielding **15,552 Hits (55.9%)** and **12,258 Non-Hits (44.1%)**.
  - **MinMax Scaled Price:** Replaced StandardScaler with `MinMaxScaler(feature_range=(0, 1))` fitted strictly on the train split, generating bounded `price_scaled` [0.0, 1.0] and serialized to [saved_models/ml_models/price_scaler.pkl](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/saved_models/ml_models/price_scaler.pkl).
  - **Removal of Entity & Multi-label Counts:** Completely removed the 6 count features (`dev_game_count`, `pub_game_count`, `num_platforms`, `num_genres`, `num_categories`, `num_tags`) from the feature transformer, dataframe builder, and exported datasets, reducing the model feature space from 200 to 194.
  - **Integer Casting for Discrete Counts & Ratings:**
    - Converted float reviews and counts to regular integers: `positive_ratings`, `negative_ratings`, `total_reviews`, `achievements`, `required_age`, `english`, and all binary indicators (`platform_*`, `genre_*`, `cat_*`, `tag_*`, `is_hit`).
    - True continuous columns preserved as floats: `price`, `price_scaled`, `approval_rate`, `hype_score`, `story_score`, `nostalgia_score`.
  - **Structured Logical Column Ordering:**
    - Ordered all columns in `cleaned_data/` splits into intuitive, cohesive groups:
      1. Core Metadata & Identifiers (`appid`, `name`, `release_date`, `developer`, `publisher`, `platforms`, `primary_genre`)
      2. Pricing (`currency`, `price`, `price_scaled`)
      3. Attributes (`english`, `required_age`, `achievements`, `owners`)
      4. Reference Strings (`genres`, `categories`, `steamspy_tags`)
      5. Continuous Vibe Scores (`hype_score`, `story_score`, `nostalgia_score`)
      6. Multi-hot Platform Indicators (`platform_windows`, `platform_mac`, `platform_linux`)
      7. Multi-hot Genre Indicators (`genre_*`)
      8. Multi-hot Category Indicators (`cat_*`)
      9. Multi-hot Tag Indicators (`tag_*`)
      10. Review Telemetry (`positive_ratings`, `negative_ratings`, `total_reviews`, `approval_rate`, `rating_text`)
      11. Target (`is_hit` as the final column).
  - **Strict Ascending Sort by `appid`:** Verified `appid.is_monotonic_increasing == True` across `train`, `valid`, and `test` splits, with exact 1-to-1 row alignment between `games.csv` and `games_descriptions.csv`.
  - **Dataset Split Dimensions (`cleaned_data/`):**
    - `cleaned_data/train/`: `games.csv` (22,248 rows, 209 cols, 194 model features, target `is_hit` at end) and `games_descriptions.csv` (22,248 rows, 5 cols).
    - `cleaned_data/valid/`: `games.csv` (2,781 rows, 209 cols, 194 model features) and `games_descriptions.csv` (2,781 rows, 5 cols).
    - `cleaned_data/test/`: `games.csv` (2,781 rows, 209 cols, 194 model features) and `games_descriptions.csv` (2,781 rows, 5 cols).
- **Streamlined XGBoost Classifier with Feature Exemption ([ml-pipeline/xgboost_train.py](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/ml-pipeline/xgboost_train.py)):**
  - **Feature Exemption Control:** Added `exempt_features` argument and `--exempt-features` CLI parameter (supporting exact column names and wildcard patterns like `tag_*` or `price_scaled`).
  - **Native Categorical Support:** Trained with `enable_categorical=True` and `tree_method='hist'` on 4 native categorical columns (`developer`, `publisher`, `platforms`, `primary_genre`).
  - **Holdout Test Set Evaluation (2,781 games, 194 features):**
    - **Accuracy:** **96.26%** (2,677 / 2,781 correct predictions)
    - **Precision:** **95.43%**
    - **Recall:** **98.01%** (1,524 / 1,555 hits detected)
    - **F1-Score:** **96.70%**
    - **Validation Logloss:** **0.1096** (early stopping best iteration: 299)
    - **Top Predictors:** `nostalgia_score` (0.1729), `genre_massively_multiplayer` (0.0959), `cat_multi_player` (0.0716), `story_score` (0.0612), `hype_score` (0.0520), `primary_genre` (0.0367), `cat_mmo` (0.0341), `genre_casual` (0.0339), `genre_free_to_play` (0.0300), `publisher` (0.0267).
  - Saved updated model artifact to [saved_models/ml_models/xgboost_hit_predictor.pkl](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/saved_models/ml_models/xgboost_hit_predictor.pkl).
- **Phase 2: LLM Baselines & Containerization:**
  - Created `.env` file for Hugging Face authentication (`HF_TOKEN`).
  - Implemented `llm-pipeline/inference_baseline.py` using **Qwen3.5-0.8B** on GPU (`cuda:0`):
    - Real-time RAM & VRAM tracking after every generation.
    - Exports metrics to `llm-pipeline/baseline_results/inference_baseline_results.json` (100.0% validity rate, 5.328s avg latency, ~1.97 GB Process RAM, ~1.41 GB GPU VRAM).
  - Implemented `llm-pipeline/prompting_baseline.py`:
    - Benchmarked **Zero-Shot**, **Few-Shot**, and **Chain-of-Thought (CoT)** prompting strategies.
    - Saved benchmark metrics to `llm-pipeline/baseline_results/prompting_baseline_results.json`.
  - **Adapted `llm-pipeline` to Three-Split `cleaned_data/` Dataset Architecture:**
    - Seamlessly loads and merges game attributes and descriptions directly from `cleaned_data/` (`train/`, `valid/`, `test/` pairing `games.csv` with `games_descriptions.csv` on `appid`).
  - Created root [requirements.txt](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/requirements.txt) and root [Dockerfile](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/Dockerfile).
- **Phase 3: 8-Bit QLoRA Fine-Tuning & Main Build ([llm-pipeline/qlora_training.py](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/llm-pipeline/qlora_training.py)):**
  - **BitsAndBytes 8-Bit Quantization:** Loaded base model `Qwen/Qwen3.5-0.8B` in 8-bit mode (`BitsAndBytesConfig(load_in_8bit=True)`), verified running on NVIDIA GeForce RTX 4060 Laptop GPU.
  - **Modified Nucleus Sampling (`top_p`):** Configured and modified the `top_p=0.9` parameter (with `temperature=0.7`, `do_sample=True`) across model `generation_config` and generation sampling.
  - **LoRA Configuration:** Attached Low-Rank Adaptation using PEFT with hyperparameters: $r=16, \alpha=32$, dropout=0.05, target modules `["q_proj", "k_proj", "v_proj", "o_proj"]`, configuring 1,081,344 trainable parameters (0.14% of the base model).
  - **Dataset Pairing & Instruction Formatting:** Formatted training instruction pairs pairing cleaned descriptions from `cleaned_data/train/games_descriptions.csv` with non-leakage ground-truth vibe scores from `cleaned_data/train/games.csv`.
  - **Training Loop with SFTTrainer:** Executed fine-tuning on GPU with `SFTTrainer` and `SFTConfig` (batch size 2, gradient accumulation 4, learning rate 2e-4). Final training loss dropped to 1.2086, with token accuracy reaching 84.3%.
  - **Adapter Persistence:** Saved fine-tuned LoRA adapter weights and tokenizer to [saved_models/llm_adapters/](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/saved_models/llm_adapters/) and [saved_models/llm_adapters/qwen_steam_vibe_8bit/](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/saved_models/llm_adapters/qwen_steam_vibe_8bit/).
  - **Before vs After Holdout Evaluation Benchmark:**
    - **JSON Schema Validity:** 100.0% -> 100.0%
    - **Mean Absolute Error (MAE):** 1.93 -> **1.42** (0.51 MAE reduction vs consensus)
    - **Accuracy (+/- 2.0 tolerance):** 60.0% -> **75.6%** (+15.6% improvement)
    - Metrics serialized to [llm-pipeline/baseline_results/qlora_training_results.json](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/llm-pipeline/baseline_results/qlora_training_results.json).

## 2. Dependencies & Runtime
- `bitsandbytes` 0.50.2
- `peft` 0.21.2
- `trl` 1.17.1
- `datasets` 5.1.0
- `torch` (CUDA 12.1+ compatible, running on RTX 4060 Laptop GPU)
- `transformers` 5.18.0
- `accelerate` 1.15.0
- `xgboost`
- `psutil`
- `python-dotenv`
- `scikit-learn`
- `pandas`
- `numpy`

## 3. Current Data Flow
```text
raw_data/dataset_attributes/*.csv (51 attribute files)
                       │
                       ▼
           [ml-pipeline/cleaner.py]
           - Standardizes delimited strings & unifies prices to USD
           - Drops missing dev/pub/name/genres/platforms entirely
           - Casts integer attributes (reviews, achievements, age, english)
                       │
                       ▼
          [ml-pipeline/data_prep.py]
           - Filters total_reviews >= 50 first
           - Computes approval_rate, rating_text, is_hit
           - Retains review leakage columns for safety
           - Generates vibe scores via generate_vibe_scores.py (no text leakage)
           - MinMax scales price to [0, 1] on train split -> price_scaler.pkl
           - Fits SteamFeatureTransformer (194 features, counts removed)
           - Sorts splits strictly by appid ascending
           - Assembles ordered games.csv & games_descriptions.csv
           - Persists price_scaler.pkl and feature_transformer.pkl
                       │
                       ▼
          cleaned_data/ (Pre-prepared with 194 features, sorted by appid)
          ├── train/
          │   ├── games.csv              (22,248 rows, 209 cols, 194 model features, target is_hit at end)
          │   └── games_descriptions.csv (22,248 rows, 5 cols: appid, short, about, detailed, clean)
          ├── valid/
          │   ├── games.csv              (2,781 rows, 209 cols, 194 model features)
          │   └── games_descriptions.csv (2,781 rows, 5 cols)
          └── test/
              ├── games.csv              (2,781 rows, 209 cols, 194 model features)
              └── games_descriptions.csv (2,781 rows, 5 cols)
                       │
         ┌─────────────┼────────────────────────┐
         │             │                        │
         ▼             ▼                        ▼
[ml-pipeline/    [llm-pipeline/           [llm-pipeline/
 xgboost_train.py] inference_baseline.py]  qlora_training.py]
- Consumes 194   - RAM & VRAM tracking    - 8-bit BitsAndBytes
  features       - JSON validation          quantization
- Categoricals                             - Modified top_p=0.9
- Early stopping                           - SFTTrainer fine-tuning
         │                                 - Before/After eval
         ▼                                      │
saved_models/ml_models/                         ▼
xgboost_hit_predictor.pkl                saved_models/llm_adapters/
(96.26% test acc, 96.70% F1)             adapter_model.safetensors (MAE: 1.42)
```

## 4. Next Step
- **Target File:** `backend/main.py`
- **Goal:** Phase 4: The FastAPI Backend - Build `backend/main.py` that loads `cleaned_data/` into memory and initializes both the XGBoost classifier (`xgboost_hit_predictor.pkl`) and the fine-tuned QLoRA adapter (`saved_models/llm_adapters/`), creating `/predict` (accepting new game concepts, predicting vibe scores with LLM, and predicting hit/flop with XGBoost) and `/similar` (returning 3 historical matching games).
