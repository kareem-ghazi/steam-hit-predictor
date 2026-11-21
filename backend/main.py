#!/usr/bin/env python3
"""
Steam Hit Predictor - Phase 4: FastAPI Backend Server
Orchestrates hybrid inference combining fine-tuned 8-bit QLoRA LLM vibe scoring
and an XGBoost classifier, backed by the 27,810 cleaned Steam games dataset.
"""

import os
import sys
import re
import json
import time
import pickle
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
import torch
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import database
from database import (
    init_db, list_private_games, get_private_game,
    create_private_game, update_private_game,
    save_game_prediction, delete_private_game
)

# Ensure root, ml-pipeline, and llm-pipeline are in sys.path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
ML_PIPELINE_DIR = os.path.join(PROJECT_ROOT, "ml-pipeline")
LLM_PIPELINE_DIR = os.path.join(PROJECT_ROOT, "llm-pipeline")

for p in [PROJECT_ROOT, ML_PIPELINE_DIR, LLM_PIPELINE_DIR, BACKEND_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Load environment variables
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ---------------------------------------------------------------------------
# Global Application State
# ---------------------------------------------------------------------------
class AppState:
    dataset_df: Optional[pd.DataFrame] = None
    xgb_model: Optional[Any] = None
    feature_transformer: Optional[Any] = None
    training_features: List[str] = []
    price_scaler: Optional[Any] = None
    
    # LLM State
    llm_tokenizer: Optional[Any] = None
    llm_model: Optional[Any] = None
    llm_status: str = "uninitialized"  # "ready", "fallback", "error"
    device: str = "cpu"


state = AppState()


# ---------------------------------------------------------------------------
# Helper: Vibe Score Prompt & Parser
# ---------------------------------------------------------------------------
def build_vibe_prompt(game_name: str, genres: str, description: str) -> str:
    """Builds chat-formatted prompt for Qwen3.5-0.8B QLoRA adapter."""
    system_inst = (
        "You are an expert video game industry analyst. Your task is to evaluate a game's concept "
        "and description, then output three vibe scores on a scale from 1.0 to 10.0:\n"
        "1. hype_score: anticipated community excitement, viral potential, or multiplayer buzz.\n"
        "2. story_score: narrative depth, lore, worldbuilding, or emotional engagement.\n"
        "3. nostalgia_score: retro aesthetic, classic genre homage, or legacy franchise appeal.\n\n"
        "You MUST respond ONLY with a valid JSON object matching this exact schema:\n"
        '{\n  "hype_score": float,\n  "story_score": float,\n  "nostalgia_score": float\n}\n'
        "Do NOT include markdown formatting, code fences, or any explanation outside the JSON."
    )
    user_content = (
        f"Game: {game_name}\n"
        f"Genres: {genres}\n"
        f"Description:\n{description.strip()}"
    )
    return (
        f"<|im_start|>system\n{system_inst}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def parse_vibe_json(raw_text: str) -> Optional[Dict[str, float]]:
    """Extracts valid JSON scores from raw model generation."""
    clean = re.sub(r"<\|im_end\|>.*", "", raw_text, flags=re.DOTALL).strip()
    match = re.search(r"\{[^{}]*\}", clean)
    if not match:
        return None

    try:
        data = json.loads(match.group(0))
        required = ["hype_score", "story_score", "nostalgia_score"]
        if not all(k in data for k in required):
            return None
        return {
            k: round(max(1.0, min(10.0, float(data[k]))), 2)
            for k in required
        }
    except Exception:
        return None


def calculate_fallback_vibe_scores(genres: str, description: str) -> Dict[str, float]:
    """Robust non-leakage heuristic vibe scores when LLM is in fallback mode."""
    g_lower = genres.lower()
    desc_lower = description.lower()

    # Hype heuristic: multiplayer, competitive, action, sandbox buzz
    hype = 5.0
    if any(w in g_lower or w in desc_lower for w in ["multiplayer", "co-op", "mmo", "pvp", "battle"]):
        hype += 1.8
    if any(w in desc_lower for w in ["open world", "survival", "sandbox", "crafting"]):
        hype += 1.2
    if any(w in desc_lower for w in ["fast-paced", "intense", "tactical"]):
        hype += 0.8

    # Story heuristic: narrative, rpg, adventure, choices
    story = 5.0
    if any(w in g_lower or w in desc_lower for w in ["rpg", "adventure", "visual novel"]):
        story += 1.8
    if any(w in desc_lower for w in ["lore", "story", "narrative", "characters", "mystery", "emotional"]):
        story += 1.4

    # Nostalgia heuristic: retro, pixel, 8-bit, classic, arcade
    nostalgia = 5.0
    if any(w in g_lower or w in desc_lower for w in ["retro", "pixel", "arcade", "classic", "metroidvania"]):
        nostalgia += 2.2
    if any(w in desc_lower for w in ["90s", "8-bit", "16-bit", "nostalgic", "tribute", "homage"]):
        nostalgia += 1.5

    return {
        "hype_score": round(max(1.0, min(10.0, hype)), 2),
        "story_score": round(max(1.0, min(10.0, story)), 2),
        "nostalgia_score": round(max(1.0, min(10.0, nostalgia)), 2)
    }


def extract_vibe_scores(game_name: str, genres: str, description: str) -> Tuple[Dict[str, float], str]:
    """Generates vibe scores using the fine-tuned QLoRA adapter or falls back safely."""
    if state.llm_status == "ready" and state.llm_model is not None and state.llm_tokenizer is not None:
        try:
            prompt = build_vibe_prompt(game_name, genres, description)
            inputs = state.llm_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(state.device)

            with torch.no_grad():
                outputs = state.llm_model.generate(
                    **inputs,
                    max_new_tokens=80,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=state.llm_tokenizer.pad_token_id
                )

            gen_tokens = outputs[0][len(inputs.input_ids[0]):]
            gen_text = state.llm_tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            parsed = parse_vibe_json(gen_text)
            if parsed is not None:
                return parsed, "qlora_adapter"
        except Exception as e:
            print(f"Notice: LLM generation error ({e}), utilizing heuristic fallback.")

    return calculate_fallback_vibe_scores(genres, description), "heuristic_fallback"


# ---------------------------------------------------------------------------
# Data Loading & Initialization
# ---------------------------------------------------------------------------
def load_all_cleaned_splits() -> pd.DataFrame:
    """Loads and concatenates train, valid, and test splits into an optimized master dataset DataFrame."""
    splits_data = []
    data_root = os.path.join(PROJECT_ROOT, "cleaned_data")

    # Keep only essential metadata columns needed for dataset browsing, filtering, and similarity search
    essential_cols = [
        "appid", "name", "release_date", "developer", "publisher", "platforms",
        "primary_genre", "genres", "categories", "steamspy_tags", "price",
        "price_scaled", "rating_text", "approval_rate", "positive_ratings",
        "negative_ratings", "total_reviews", "is_hit", "hype_score", "story_score", "nostalgia_score"
    ]

    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(data_root, split)
        games_csv = os.path.join(split_dir, "games.csv")
        desc_csv = os.path.join(split_dir, "games_descriptions.csv")

        if os.path.exists(games_csv):
            df_g = pd.read_csv(games_csv)
            cols_to_use = [c for c in essential_cols if c in df_g.columns]
            df_g_filtered = df_g[cols_to_use].copy()
            df_g_filtered["split"] = split

            if os.path.exists(desc_csv):
                df_d = pd.read_csv(desc_csv, usecols=["appid", "short_description", "clean_description"])
                desc_subset = df_d.drop_duplicates(subset=["appid"])
                df_merged = pd.merge(df_g_filtered, desc_subset, on="appid", how="left")
            else:
                df_merged = df_g_filtered
                df_merged["short_description"] = ""
                df_merged["clean_description"] = ""

            splits_data.append(df_merged)

    if not splits_data:
        raise FileNotFoundError(f"Could not find cleaned dataset splits in '{data_root}'.")

    master_df = pd.concat(splits_data, ignore_index=True)
    master_df = master_df.drop_duplicates(subset=["appid"]).sort_values(by="appid").reset_index(drop=True)

    # Ensure required columns are present and clean
    master_df["name"] = master_df["name"].fillna("Unknown Game").astype(str)
    master_df["developer"] = master_df["developer"].fillna("Unknown").astype(str)
    master_df["publisher"] = master_df["publisher"].fillna("Unknown").astype(str)
    master_df["primary_genre"] = master_df["primary_genre"].fillna("Indie").astype(str)
    master_df["genres"] = master_df["genres"].fillna("Indie").astype(str)
    master_df["rating_text"] = master_df["rating_text"].fillna("Positive").astype(str)
    master_df["price"] = pd.to_numeric(master_df["price"], errors="coerce").fillna(0.0)
    master_df["approval_rate"] = pd.to_numeric(master_df["approval_rate"], errors="coerce").fillna(0.8)
    master_df["total_reviews"] = pd.to_numeric(master_df["total_reviews"], errors="coerce").fillna(0).astype(int)
    master_df["is_hit"] = pd.to_numeric(master_df["is_hit"], errors="coerce").fillna(0).astype(int)

    for v in ["hype_score", "story_score", "nostalgia_score"]:
        master_df[v] = pd.to_numeric(master_df[v], errors="coerce").fillna(5.0)

    print(f"Loaded master Steam dataset: {len(master_df):,} total historical games.")
    return master_df

def initialize_ml_models():
    """Loads XGBoost classifier, feature transformer, and price scaler."""
    models_dir = os.path.join(PROJECT_ROOT, "saved_models", "ml_models")
    xgb_path = os.path.join(models_dir, "xgboost_hit_predictor.pkl")
    scaler_path = os.path.join(models_dir, "price_scaler.pkl")

    if not os.path.exists(xgb_path):
        raise FileNotFoundError(f"XGBoost artifact not found at '{xgb_path}'.")

    with open(xgb_path, "rb") as f:
        artifact = pickle.load(f)

    state.xgb_model = artifact["model"]
    state.feature_transformer = artifact["transformer"]
    state.training_features = artifact["training_features"]

    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            state.price_scaler = pickle.load(f)

    print(f"Loaded XGBoost classifier ({len(state.training_features)} features, early stop best iter: {artifact['metrics'].get('best_iteration')}).")


def initialize_llm():
    """Initializes Qwen3.5-0.8B tokenizer and 8-bit QLoRA adapter."""
    adapter_candidates = [
        os.path.join(PROJECT_ROOT, "saved_models", "llm_adapters", "qwen_steam_vibe_8bit"),
        os.path.join(PROJECT_ROOT, "saved_models", "llm_adapters"),
    ]
    adapter_dir = next((c for c in adapter_candidates if os.path.exists(os.path.join(c, "adapter_config.json"))), None)

    if not adapter_dir:
        print("Notice: QLoRA adapter weights not found; running with heuristic vibe scorer.")
        state.llm_status = "fallback"
        return

    hf_token = os.getenv("HF_TOKEN") or None
    state.device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        print(f"Loading QLoRA tokenizer from '{adapter_dir}'...")
        state.llm_tokenizer = AutoTokenizer.from_pretrained(adapter_dir, token=hf_token)
        if state.llm_tokenizer.pad_token is None:
            state.llm_tokenizer.pad_token = state.llm_tokenizer.eos_token
        state.llm_tokenizer.padding_side = "left"

        base_model_id = "Qwen/Qwen3.5-0.8B"
        print(f"Loading base LLM '{base_model_id}' on {state.device}...")

        if state.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map="auto",
                token=hf_token
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.float32,
                device_map=None,
                token=hf_token
            )

        print(f"Attaching fine-tuned LoRA adapter from '{adapter_dir}'...")
        state.llm_model = PeftModel.from_pretrained(base_model, adapter_dir)
        state.llm_model.eval()
        state.llm_status = "ready"
        print("QLoRA 8-bit LLM vibe evaluator ready!")
    except Exception as e:
        print(f"Warning: Could not initialize QLoRA model: {e}. Activating heuristic fallback.")
        state.llm_status = "fallback"


# ---------------------------------------------------------------------------
# FastAPI Lifespan Handler
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("==================================================================")
    print("      STARTING STEAM HIT PREDICTOR FASTAPI BACKEND SERVER         ")
    print("==================================================================")
    start_t = time.time()
    
    # 0. Initialize SQLite Database
    init_db()
    
    # 1. Load Cleaned Dataset into Memory
    state.dataset_df = load_all_cleaned_splits()
    
    # 2. Load XGBoost Model & Transformer
    initialize_ml_models()
    
    # 3. Initialize QLoRA LLM Pipeline
    initialize_llm()
    
    print(f"Backend initialized successfully in {time.time() - start_t:.2f}s!")
    print("==================================================================\n")
    yield
    print("Shutting down Steam Hit Predictor Backend Server...")


# ---------------------------------------------------------------------------
# FastAPI Application & CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Steam Hit Predictor API",
    description="Hybrid AI prediction backend uniting QLoRA LLM vibe scoring and an XGBoost tabular classifier.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Request & Response Schemas
# ---------------------------------------------------------------------------
class VibeScores(BaseModel):
    hype_score: float = Field(..., ge=1.0, le=10.0, description="1.0 to 10.0 anticipated viral/multiplayer buzz")
    story_score: float = Field(..., ge=1.0, le=10.0, description="1.0 to 10.0 narrative and emotional depth")
    nostalgia_score: float = Field(..., ge=1.0, le=10.0, description="1.0 to 10.0 retro/classic franchise appeal")


class GamePredictionRequest(BaseModel):
    name: str = Field(..., example="Chrono Odyssey", description="Title of the video game concept")
    description: str = Field(..., example="A tactical sci-fi roguelike where you control time to survive procedural space fleets.", description="Detailed game concept description")
    developer: str = Field(default="Independent Studio", example="Cosmic Byte", description="Developer studio name")
    publisher: str = Field(default="Independent Studio", example="Cosmic Byte", description="Publisher name")
    primary_genre: Optional[str] = Field(default=None, example="Action", description="Primary genre (auto-derived if omitted)")
    genres: str = Field(default="Action;Indie;RPG", example="Action;Indie;RPG", description="Semicolon or comma-separated genres")
    categories: Optional[str] = Field(default="Single-player;Steam Achievements", example="Single-player;Steam Achievements", description="Delimited category features")
    steamspy_tags: Optional[str] = Field(default="Action;Roguelike;Sci-fi;Indie", example="Action;Roguelike;Sci-fi;Indie", description="Delimited player community tags")
    platforms: Optional[str] = Field(default="windows", example="windows;mac", description="Supported platforms")
    price: float = Field(default=19.99, ge=0.0, example=19.99, description="Base game retail price in USD")
    english: int = Field(default=1, ge=0, le=1, example=1, description="English language support (1 or 0)")
    required_age: int = Field(default=0, ge=0, example=0, description="Required minimum age")
    achievements: int = Field(default=0, ge=0, example=15, description="Number of Steam achievements")
    vibe_scores: Optional[VibeScores] = Field(default=None, description="Optional manual override of vibe scores")


class PrivateGameCreate(BaseModel):
    name: str = Field(..., description="Title of the video game concept")
    description: str = Field(..., description="Detailed game concept description")
    developer: Optional[str] = Field(default="Independent Studio")
    publisher: Optional[str] = Field(default="Independent Studio")
    primary_genre: Optional[str] = Field(default="Action")
    genres: Optional[str] = Field(default="Action;Indie")
    categories: Optional[str] = Field(default="Single-player;Steam Achievements")
    steamspy_tags: Optional[str] = Field(default="Action;Indie;Difficult")
    platforms: Optional[str] = Field(default="windows")
    price: Optional[float] = Field(default=19.99, ge=0.0)
    english: Optional[int] = Field(default=1)
    required_age: Optional[int] = Field(default=0)
    achievements: Optional[int] = Field(default=0)
    username: Optional[str] = Field(default="guest")


class PrivateGameUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    developer: Optional[str] = None
    publisher: Optional[str] = None
    primary_genre: Optional[str] = None
    genres: Optional[str] = None
    categories: Optional[str] = None
    steamspy_tags: Optional[str] = None
    platforms: Optional[str] = None
    price: Optional[float] = None
    english: Optional[int] = None
    required_age: Optional[int] = None
    achievements: Optional[int] = None


class SimilarGame(BaseModel):
    appid: int
    name: str
    developer: str
    publisher: str
    primary_genre: str
    genres: str
    price: float
    rating_text: str
    approval_rate: float
    total_reviews: int
    is_hit: int
    hype_score: float
    story_score: float
    nostalgia_score: float
    short_description: str


class PredictionResponse(BaseModel):
    game_name: str
    is_hit: int = Field(..., description="1 = Hit (approval >= 80%), 0 = Flop")
    prediction_label: str = Field(..., description="'Hit' or 'Flop'")
    hit_probability: float = Field(..., description="Model probability [0.0, 1.0] of being a Hit")
    confidence_score: float = Field(..., description="Prediction confidence [0.5, 1.0]")
    projected_rating_tier: str = Field(..., description="Projected Steam review rating tier")
    vibe_scores: VibeScores
    vibe_scores_source: str = Field(..., description="'qlora_adapter' or 'heuristic_fallback'")
    features_used_count: int
    similar_games: List[SimilarGame]
    inference_duration_ms: float


# ---------------------------------------------------------------------------
# Similarity Search Helper Function
# ---------------------------------------------------------------------------
def find_similar_games(
    genre: str,
    price: float,
    rating_text: Optional[str] = None,
    n: int = 3,
    exclude_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Finds n historical games matching the specified genre, price tier, and rating tier.
    Price brackets:
      - Free: $0.00
      - Budget: $0.01 - $9.99
      - Mid-Tier: $10.00 - $29.99
      - Premium: $30.00+
    """
    if state.dataset_df is None or state.dataset_df.empty:
        return []

    df = state.dataset_df.copy()

    if exclude_name:
        df = df[df["name"].str.lower() != exclude_name.lower().strip()]

    # 1. Price bracket filtering
    if price == 0:
        price_mask = df["price"] == 0
    elif price < 10.0:
        price_mask = (df["price"] > 0) & (df["price"] < 10.0)
    elif price < 30.0:
        price_mask = (df["price"] >= 10.0) & (df["price"] < 30.0)
    else:
        price_mask = df["price"] >= 30.0

    subset = df[price_mask]
    if len(subset) < n:
        # Fallback: expand price bracket within ±$10
        subset = df[abs(df["price"] - price) <= 12.0]
    if len(subset) < n:
        subset = df

    # 2. Genre filtering
    genre_token = genre.split(";")[0].split(",")[0].strip().lower()
    genre_mask = (
        subset["primary_genre"].str.lower().str.contains(genre_token, regex=False, na=False)
        | subset["genres"].str.lower().str.contains(genre_token, regex=False, na=False)
    )
    genre_subset = subset[genre_mask]
    if len(genre_subset) >= n:
        subset = genre_subset

    # 3. Rating Tier filtering
    if rating_text:
        r_mask = subset["rating_text"].str.lower() == rating_text.lower().strip()
        r_subset = subset[r_mask]
        if len(r_subset) >= n:
            subset = r_subset

    # 4. Rank by review volume & relevance
    ranked = subset.sort_values(by=["total_reviews", "approval_rate"], ascending=[False, False]).head(n)

    results = []
    for _, row in ranked.iterrows():
        desc = str(row.get("clean_description", "") or row.get("short_description", "") or "")
        snippet = desc[:180] + "..." if len(desc) > 180 else desc
        results.append({
            "appid": int(row.get("appid", 0)),
            "name": str(row.get("name", "Unknown")),
            "developer": str(row.get("developer", "Unknown")),
            "publisher": str(row.get("publisher", "Unknown")),
            "primary_genre": str(row.get("primary_genre", "Indie")),
            "genres": str(row.get("genres", "")),
            "price": round(float(row.get("price", 0.0)), 2),
            "rating_text": str(row.get("rating_text", "Positive")),
            "approval_rate": round(float(row.get("approval_rate", 0.0)), 3),
            "total_reviews": int(row.get("total_reviews", 0)),
            "is_hit": int(row.get("is_hit", 0)),
            "hype_score": round(float(row.get("hype_score", 5.0)), 2),
            "story_score": round(float(row.get("story_score", 5.0)), 2),
            "nostalgia_score": round(float(row.get("nostalgia_score", 5.0)), 2),
            "short_description": snippet
        })

    return results


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Status"])
def root_status():
    """Health check and backend telemetry status."""
    return {
        "status": "healthy",
        "app_name": "Steam Hit Predictor API",
        "dataset_records_loaded": len(state.dataset_df) if state.dataset_df is not None else 0,
        "xgboost_features": len(state.training_features),
        "llm_evaluator_status": state.llm_status,
        "device": state.device,
        "endpoints": {
            "predict": "POST /predict",
            "similar": "GET /similar",
            "games": "GET /games",
            "metadata": "GET /metadata"
        }
    }


@app.get("/health", tags=["Status"])
def health_check():
    if state.xgb_model is None or state.dataset_df is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Models or dataset not initialized")
    return {"status": "ok", "llm_status": state.llm_status}


def execute_game_prediction(game_data: Dict[str, Any], vibe_scores_override: Optional[Dict[str, float]] = None) -> PredictionResponse:
    """
    Executes the complete dual ML+LLM inference pipeline on game data dictionary.
    """
    start_t = time.time()

    if state.xgb_model is None or state.feature_transformer is None:
        raise HTTPException(status_code=500, detail="XGBoost model is not initialized.")

    name = game_data.get("name", "Untitled Project")
    description = game_data.get("description", "")
    developer = game_data.get("developer", "Independent Studio")
    publisher = game_data.get("publisher", "Independent Studio")
    genres = game_data.get("genres", "Action;Indie")
    categories = game_data.get("categories", "Single-player;Steam Achievements")
    steamspy_tags = game_data.get("steamspy_tags", "")
    platforms = game_data.get("platforms", "windows")
    price = float(game_data.get("price", 19.99))
    english = int(game_data.get("english", 1))
    required_age = int(game_data.get("required_age", 0))
    achievements = int(game_data.get("achievements", 0))

    # 1. Resolve Primary Genre
    primary_genre = game_data.get("primary_genre")
    if not primary_genre or not str(primary_genre).strip():
        primary_genre = genres.split(";")[0].split(",")[0].strip() or "Indie"

    # 2. Extract or apply vibe scores
    if vibe_scores_override is not None:
        vibe_dict = {
            "hype_score": float(vibe_scores_override.get("hype_score", 7.0)),
            "story_score": float(vibe_scores_override.get("story_score", 7.0)),
            "nostalgia_score": float(vibe_scores_override.get("nostalgia_score", 7.0))
        }
        vibe_source = "user_override"
    else:
        vibe_dict, vibe_source = extract_vibe_scores(
            game_name=name,
            genres=genres,
            description=description
        )

    # 3. Assemble single-sample DataFrame for transformer
    sample_dict = {
        "name": name,
        "developer": developer,
        "publisher": publisher,
        "primary_genre": primary_genre,
        "genres": genres,
        "categories": categories or "",
        "steamspy_tags": steamspy_tags or "",
        "platforms": platforms or "windows",
        "price": price,
        "english": english,
        "required_age": required_age,
        "achievements": achievements,
        "hype_score": vibe_dict["hype_score"],
        "story_score": vibe_dict["story_score"],
        "nostalgia_score": vibe_dict["nostalgia_score"]
    }
    sample_df = pd.DataFrame([sample_dict])

    # 4. Transform features into the exact 194 training features
    try:
        transformed_df = state.feature_transformer.transform(sample_df)
        X = transformed_df[state.training_features]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature transformation error: {str(e)}")

    # 5. Execute XGBoost Prediction
    try:
        pred_val = int(state.xgb_model.predict(X)[0])
        proba_val = float(state.xgb_model.predict_proba(X)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {str(e)}")

    label = "Hit" if pred_val == 1 else "Flop"
    confidence = proba_val if pred_val == 1 else (1.0 - proba_val)

    # 6. Project Steam Rating Tier
    if pred_val == 1:
        if proba_val >= 0.90:
            rating_tier = "Overwhelmingly Positive"
        elif proba_val >= 0.80:
            rating_tier = "Very Positive"
        else:
            rating_tier = "Positive"
    else:
        if proba_val >= 0.40:
            rating_tier = "Mixed"
        elif proba_val >= 0.20:
            rating_tier = "Mostly Negative"
        else:
            rating_tier = "Overwhelmingly Negative"

    # 7. Find 3 Similar Historical Games
    similar = find_similar_games(
        genre=primary_genre,
        price=price,
        rating_text=rating_tier,
        n=3,
        exclude_name=name
    )

    duration_ms = (time.time() - start_t) * 1000.0

    return PredictionResponse(
        game_name=name,
        is_hit=pred_val,
        prediction_label=label,
        hit_probability=round(proba_val, 4),
        confidence_score=round(confidence, 4),
        projected_rating_tier=rating_tier,
        vibe_scores=VibeScores(**vibe_dict),
        vibe_scores_source=vibe_source,
        features_used_count=len(state.training_features),
        similar_games=[SimilarGame(**g) for g in similar],
        inference_duration_ms=round(duration_ms, 2)
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_game_hit(req: GamePredictionRequest):
    """
    Accepts game concept data, extracts vibe scores via QLoRA LLM adapter,
    transforms tabular attributes into 194 model features, predicts Hit/Flop with XGBoost,
    and returns 3 similar historical games.
    """
    vibe_override = req.vibe_scores.model_dump() if req.vibe_scores else None
    return execute_game_prediction(req.model_dump(), vibe_scores_override=vibe_override)


# ---------------------------------------------------------------------------
# Private Games SQLite Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/private-games", tags=["Private Games"])
def get_user_private_games(username: Optional[str] = Query(None, description="Username to filter games")):
    """Returns private studio projects saved in the SQLite database."""
    return list_private_games(username=username)


@app.post("/api/private-games", tags=["Private Games"])
def create_new_private_game(req: PrivateGameCreate):
    """Creates a new private studio project in SQLite."""
    return create_private_game(req.model_dump(), username=req.username or "guest")


@app.get("/api/private-games/{game_id}", tags=["Private Games"])
def get_single_private_game(game_id: int):
    """Retrieves a single private project by its SQLite ID."""
    game = get_private_game(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Private game not found.")
    return game


@app.put("/api/private-games/{game_id}", tags=["Private Games"])
def update_existing_private_game(game_id: int, req: PrivateGameUpdate):
    """Updates parameters for an existing private project."""
    updated = update_private_game(game_id, req.model_dump(exclude_unset=True))
    if not updated:
        raise HTTPException(status_code=404, detail="Private game not found.")
    return updated


@app.delete("/api/private-games/{game_id}", tags=["Private Games"])
def delete_existing_private_game(game_id: int):
    """Deletes a private project from the SQLite database."""
    success = delete_private_game(game_id)
    if not success:
        raise HTTPException(status_code=404, detail="Private game not found.")
    return {"status": "deleted", "id": game_id}


@app.post("/api/private-games/{game_id}/analyze", response_model=PredictionResponse, tags=["Private Games"])
def analyze_private_game(game_id: int):
    """
    Executes full AI & ML prediction on a stored private game from SQLite,
    persists the analysis result in the SQLite database, and returns the forecast.
    """
    game = get_private_game(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Private game not found.")

    prediction = execute_game_prediction(game)
    save_game_prediction(game_id, prediction.model_dump())
    return prediction


@app.get("/similar", response_model=List[SimilarGame], tags=["Discovery"])
def get_similar_games(
    genre: str = Query(..., description="Genre to filter by (e.g. Action, RPG, Indie)", examples=["Action"]),
    price: float = Query(..., ge=0.0, description="Price in USD", examples=[19.99]),
    rating_text: Optional[str] = Query(None, description="Optional Steam rating tier filter", examples=["Very Positive"]),
    n: int = Query(3, ge=1, le=20, description="Number of similar games to return")
):
    """
    Filters the loaded 27,810 cleaned dataset to return matching historical games
    by genre, price bracket, and rating tier.
    """
    matches = find_similar_games(genre=genre, price=price, rating_text=rating_text, n=n)
    return [SimilarGame(**m) for m in matches]


@app.get("/games", tags=["Dataset"])
def list_dataset_games(
    skip: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(20, ge=1, le=100, description="Pagination page size"),
    query: Optional[str] = Query(None, description="Search keyword in title, developer, or publisher"),
    genre: Optional[str] = Query(None, description="Filter by genre"),
    rating_text: Optional[str] = Query(None, description="Filter by Steam rating text"),
    is_hit: Optional[int] = Query(None, ge=0, le=1, description="Filter by Hit (1) or Flop (0)")
):
    """
    Lists paginated historical games from the 27,810 games dataset for the Dataset Explorer UI.
    """
    if state.dataset_df is None or state.dataset_df.empty:
        raise HTTPException(status_code=503, detail="Dataset not loaded.")

    df = state.dataset_df

    if query:
        q = query.strip().lower()
        mask = (
            df["name"].str.lower().str.contains(q, regex=False, na=False)
            | df["developer"].str.lower().str.contains(q, regex=False, na=False)
            | df["publisher"].str.lower().str.contains(q, regex=False, na=False)
        )
        df = df[mask]

    if genre:
        g = genre.strip().lower()
        df = df[df["genres"].str.lower().str.contains(g, regex=False, na=False)]

    if rating_text:
        df = df[df["rating_text"].str.lower() == rating_text.strip().lower()]

    if is_hit is not None:
        df = df[df["is_hit"] == is_hit]

    total_count = len(df)
    page_items = df.iloc[skip:skip + limit]

    columns_to_return = [
        "appid", "name", "developer", "publisher", "primary_genre", "genres",
        "price", "rating_text", "approval_rate", "positive_ratings", "total_reviews",
        "is_hit", "hype_score", "story_score", "nostalgia_score", "short_description"
    ]
    existing_cols = [c for c in columns_to_return if c in page_items.columns]
    records = page_items[existing_cols].to_dict(orient="records")

    return {
        "total": total_count,
        "skip": skip,
        "limit": limit,
        "games": records
    }


@app.get("/metadata", tags=["Metadata"])
def get_metadata():
    """Returns unique categories, genres, and rating tiers for frontend studio dropdowns."""
    if state.dataset_df is None:
        return {}

    genres = sorted(state.dataset_df["primary_genre"].dropna().unique().tolist())
    rating_tiers = [
        "Overwhelmingly Positive", "Very Positive", "Positive",
        "Mostly Positive", "Mixed", "Mostly Negative", "Overwhelmingly Negative"
    ]
    return {
        "total_games": len(state.dataset_df),
        "primary_genres": [g for g in genres if g and g != "unknown"],
        "rating_tiers": rating_tiers,
        "price_brackets": ["Free ($0.00)", "Budget ($0.01 - $9.99)", "Mid-Tier ($10.00 - $29.99)", "Premium ($30.00+)"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
