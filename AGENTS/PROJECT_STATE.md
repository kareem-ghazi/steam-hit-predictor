# Project State: Steam Hit Predictor

## 1. Completed Components
- **Multi-Dataset Ingestion & Harmonization:**
  - Downloaded and organized 3 additional Steam datasets into dedicated folders in `raw_data/`:
    - `raw_data/crainbramp_steam_dataset_2025/`: Multi-modal gaming analytics dataset — **CSV package only** (180MB `applications.csv`, metadata CSVs, relational category/developer/genre/publisher mappings).
    - `raw_data/artermiloff_steam_games_dataset/`: March 2025 Steam games dataset (89,618 records).
    - `raw_data/fronkongames_steam_games_dataset/`: Complete Steam games archive (125,855 records).
    - Existing reference dataset: `raw_data/steam_games_cleaned/` (original 27,075 records).
  - Built and executed [raw_data/merge_datasets.py](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/raw_data/merge_datasets.py):
    - **Crainbramp Game Integration:** Integrated all games from `crainbramp_steam_dataset_2025` (`type == 'game'`, 150,275 games) and unioned with reference games from `steam_games_cleaned` for **153,163 total clean, non-null games**.
    - Relational metadata (`developer`, `publisher`, `genres`, `categories`) mapped using fast dictionary indexing across `crainbramp`'s auxiliary relation tables.
    - Fully cross-filled ratings, tags, owners, playtime, descriptions, and media metadata.
    - **Modular Columnar Split (`raw_data/dataset_attributes/`):** Partitioned into 51 individual CSV files inside `raw_data/dataset_attributes/`.
    - Data card generated at [raw_data/dataset_attributes/DATA_CARD.md](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/raw_data/dataset_attributes/DATA_CARD.md).
- **Phase 1: Modular Data Engineering & Refactored `ml-pipeline/` Architecture:**
  - **Modular Pipeline Split:** Refactored into [ml-pipeline/config.py](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/ml-pipeline/config.py), [ml-pipeline/cleaner.py](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/ml-pipeline/cleaner.py), [ml-pipeline/feature_transformer.py](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/ml-pipeline/feature_transformer.py), [ml-pipeline/data_prep.py](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/ml-pipeline/data_prep.py), and [ml-pipeline/generate_vibe_scores.py](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/ml-pipeline/generate_vibe_scores.py).
  - **Strict Entity & Volume Filtering:** Dropped rows missing developer/publisher, filtered `total_reviews >= 50` yielding **27,810 robust games**.
  - **Target Variable `is_hit`:** Formed target label `is_hit = 1 if approval_rate >= 0.80 else 0`, yielding **15,552 Hits (55.9%)** and **12,258 Non-Hits (44.1%)**.
  - **MinMax Scaled Price:** Replaced StandardScaler with `MinMaxScaler(feature_range=(0, 1))` fitted strictly on train split, serialized to [saved_models/ml_models/price_scaler.pkl](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/saved_models/ml_models/price_scaler.pkl).
  - **Feature Space:** 194 model features (removed 6 redundant count columns).
  - **Integer Casting for Discrete Counts & Ratings:** Converted float reviews and counts to regular integers.
  - **Dataset Split Dimensions (`cleaned_data/`):**
    - `cleaned_data/train/`: `games.csv` (22,248 rows, 209 cols, 194 model features, target `is_hit` at end) and `games_descriptions.csv` (22,248 rows, 5 cols).
    - `cleaned_data/valid/`: `games.csv` (2,781 rows, 209 cols, 194 model features) and `games_descriptions.csv` (2,781 rows, 5 cols).
    - `cleaned_data/test/`: `games.csv` (2,781 rows, 209 cols, 194 model features) and `games_descriptions.csv` (2,781 rows, 5 cols).
  - **Streamlined XGBoost Classifier ([ml-pipeline/xgboost_train.py](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/ml-pipeline/xgboost_train.py)):**
    - **Holdout Test Set Evaluation (2,781 games, 194 features):**
      - **Accuracy:** **96.26%** (2,677 / 2,781 correct predictions)
      - **Precision:** **95.43%**
      - **Recall:** **98.01%** (1,524 / 1,555 hits detected)
      - **F1-Score:** **96.70%**
      - **Validation Logloss:** **0.1096**
    - Saved model artifact to [saved_models/ml_models/xgboost_hit_predictor.pkl](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/saved_models/ml_models/xgboost_hit_predictor.pkl).
- **Phase 2: LLM Baselines & Containerization:**
  - Implemented `llm-pipeline/inference_baseline.py` using **Qwen3.5-0.8B** on GPU (`cuda:0`).
  - Implemented `llm-pipeline/prompting_baseline.py` (Zero-Shot, Few-Shot, Chain-of-Thought).
  - Created root [requirements.txt](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/requirements.txt) and root [Dockerfile](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/Dockerfile).
- **Phase 3: Optimized 8-Bit QLoRA Fine-Tuning & Standalone Kaggle Notebook Execution ([llm-pipeline/qlora_training.py](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/llm-pipeline/qlora_training.py), [kaggle_upload_dataset.py](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/kaggle_upload_dataset.py)):**
  - Fine-tuned `Qwen/Qwen3.5-0.8B` in 8-bit mode with LoRA ($r=16, \alpha=32$, dropout=0.05) and `bf16=True`.
  - Vectorized dataset pre-formatting with `Dataset.map(batched=True)` and batched eval with `torch.compile()` (4.20s/game).
  - Uploaded dataset to Kaggle: `https://www.kaggle.com/datasets/kareemghazi/steam-hit-predictor-cleaned-data`.
  - Saved adapter weights to [saved_models/llm_adapters/qwen_steam_vibe_8bit/](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/saved_models/llm_adapters/qwen_steam_vibe_8bit/).
  - Benchmark improvement: MAE reduced from 1.98 to **1.44**, accuracy raised from 58.3% to **73.3%**.
- **Phase 4: The FastAPI Backend ([backend/main.py](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/backend/main.py), [backend/requirements.txt](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/backend/requirements.txt)):**
  - Built production FastAPI server loading all 27,810 games in memory, XGBoost model, price scaler, and QLoRA LLM adapter.
  - Endpoints: `POST /predict`, `GET /similar`, `GET /games`, `GET /metadata`, `GET /health`.
  - Verified live inference with FastAPI TestClient.
- **Phase 5: The React Frontend Dashboard ([frontend/](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/frontend/)):**
  - Initialized React 18 + Vite 5 frontend project with `lucide-react` icons.
  - Implemented custom Steam dark theme styling in [frontend/src/index.css](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/frontend/src/index.css) (`#0a0e17`, `#131b29`, `#182335`, Steam blue `#66c0f4`, green hits `#10b981`, red flops `#ef4444`).
  - Built [frontend/src/components/DatasetExplorer.jsx](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/frontend/src/components/DatasetExplorer.jsx):
    - Browses, searches, filters, and paginates through all 27,810 games in `cleaned_data/`.
    - Highlights `rating_text` Steam rating tiers with color-coded badges (Overwhelmingly Positive, Very Positive, Mixed, etc.).
    - Filter controls for genre, hit/flop status, and text search.
    - One-click "Use in Developer Studio" action to transfer game data into the prediction form.
  - Built [frontend/src/components/PredictionStudio.jsx](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/frontend/src/components/PredictionStudio.jsx):
    - Inputs for title, store description pitch, developer, publisher, genre, tags, categories, platforms, price, and achievements.
    - Quick Archetype Presets (Metroidvania Masterpiece, Retro Boomer Shooter, Generic Zombie Survival, Cozy Farming Sim).
    - Autonomous AI Vibe Mode (QLoRA LLM auto-extraction) vs Manual Vibe Score Override sliders.
    - Submits to `POST /predict` with live loading indicators.
  - Built [frontend/src/components/ResultsPane.jsx](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/frontend/src/components/ResultsPane.jsx):
    - Glowing Commercial Hit (emerald) vs Market Flop (red) hero banner with hit probability percentage and confidence score.
    - Projected Steam rating tier badge and inference duration telemetry.
    - Visual progress meters for LLM vibe scores (Hype, Story, Nostalgia) with source attribution badge.
    - Renders 3 closest historical comparables from the backend with review metrics, price, and descriptions.
  - Built [frontend/src/App.jsx](file:///c:/Users/kareemghazi/Desktop/steam-hit-predictor/frontend/src/App.jsx):
    - Steam-styled header with real-time backend connection status badge (`GET /health`).
    - Tab switching between Developer Studio and Dataset Explorer.
    - Cohesive state management and responsive layout.
  - Verified production build: `npm run build` completed successfully in 5.30s (0 errors).

## 2. Dependencies & Runtime
- **Backend:** `fastapi` 0.165.0, `uvicorn` 0.54.0, `pydantic` 2.13.5, `httpx` 0.28.1, `pytest` 9.2.0, `torch` (CUDA 12.1+), `transformers` 5.18.0, `peft` 0.21.2, `bitsandbytes` 0.50.2, `xgboost` 2.1.4, `scikit-learn` 1.6.1, `pandas` 2.2.3, `numpy` 2.2.3.
- **Frontend:** Node.js v24.11.0, npm 11.6.4, `react` ^18.3.1, `react-dom` ^18.3.1, `lucide-react` ^1.16.0, `vite` ^5.4.21.

## 3. Current Data Flow
```text
React Frontend (Vite :5173)
   │
   ├──► [DatasetExplorer.jsx] ─── GET /games, GET /metadata ───► FastAPI Backend (:8000)
   │         ▲                                                        │
   │         └── Click "Use in Developer Studio" ─────────────────────┘
   │
   └──► [PredictionStudio.jsx] ── POST /predict (Payload) ─────► FastAPI Backend (:8000)
             │                                                        │
             │                                                        ├──► QLoRA LLM (Qwen3.5-0.8B)
             │                                                        ├──► Transformer (194 feats)
             │                                                        ├──► XGBoost (96.3% Acc)
             │                                                        └──► Historical Memory (27.8k)
             ▼                                                        │
        [ResultsPane.jsx] ◄──── JSON Prediction Result ───────────────┘
          - Hit/Flop Banner + Probability %
          - Projected Rating Tier Badge
          - LLM Vibe Scores Breakdown
          - 3 Historical Similar Games
```

## 4. Next Step
- **Target Component:** Phase 6: Testing & Detailed Documentation (`tests/` and `docs/`)
- **Goal:**
  - Create `tests/test_backend.py` for comprehensive FastAPI unit tests.
  - Create `tests/App.test.jsx` for React component unit tests.
  - Create root `README.md` with complete setup and execution guide.
  - Create markdown documentation files in `docs/`:
    - `docs/architecture.md`
    - `docs/ml_pipeline.md`
    - `docs/llm_pipeline.md`
    - `docs/backend.md`
    - `docs/frontend.md`
