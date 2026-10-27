"""
generate_vibe_scores.py
Heuristic Vibe Score Generator (Hype, Story, Nostalgia: 1.0 - 10.0)

Calculates ground-truth vibe scores purely from non-description game metadata
(genres, categories, steamspy tags, publisher prestige, player ownership brackets,
pricing signals, playtime engagement, and historical release date).

IMPORTANT:
Does NOT use any textual descriptions (short_description, about_the_game,
detailed_description, clean_description) to prevent data leakage during
downstream LLM fine-tuning and evaluation.
"""

import os
import re
import argparse
from typing import Dict, Any, Union
import numpy as np
import pandas as pd


# Major AAA / high-hype publishers that command inherent pre-release excitement
HIGH_HYPE_PUBLISHERS = {
    "valve", "electronic arts", "ubisoft", "bethesda softworks", "square enix",
    "capcom", "2k", "sega", "bandai namco entertainment", "activision", "warner bros. interactive entertainment",
    "cd projekt red", "rockstar games", "sony interactive entertainment", "xbox game studios",
    "blizzard entertainment", "paradox interactive", "devolver digital", "thq nordic", "deep silver"
}

# Owners bracket heuristic mapping (market footprint signal)
OWNERS_HYPE_WEIGHTS = {
    "10000000-20000000": 3.0,
    "20000000-50000000": 3.5,
    "5000000-10000000": 2.5,
    "2000000-5000000": 2.0,
    "1000000-2000000": 1.5,
    "500000-1000000": 1.0,
    "200000-500000": 0.5,
    "100000-200000": 0.3,
    "50000-100000": 0.1,
    "20000-50000": 0.0,
    "0-20000": -0.5
}


def _extract_release_year(val: Any) -> int:
    """Extracts integer release year from string or date."""
    if not val or pd.isna(val):
        return 2015
    match = re.search(r"\b(19\d\d|20\d\d)\b", str(val))
    if match:
        return int(match.group(1))
    return 2015


def calculate_game_vibe_scores(row: Union[pd.Series, Dict[str, Any]]) -> Dict[str, float]:
    """
    Computes calibrated hype_score, story_score, and nostalgia_score (1.0 to 10.0)
    for an individual Steam game purely from non-description metadata:
    1. Title name signals
    2. Genre classifications & steamspy community tags
    3. Category features (multiplayer, workshop, singleplayer, co-op)
    4. Publisher pedigree & market traction (owners bracket)
    5. Release date vintage & historical heritage
    6. Pricing and engagement signals

    NOTE: No description text is used to prevent data leakage in LLM training.
    """
    title = str(row.get("name", "")).lower()
    genres = str(row.get("genres", "")).lower().replace(";", " ")
    categories = str(row.get("categories", "")).lower().replace(";", " ")
    tags = str(row.get("steamspy_tags", "")).lower().replace(";", " ")
    publisher = str(row.get("publisher", "")).lower()
    owners_bracket = str(row.get("owners", "")).strip()
    price = float(row.get("price", 0.0) or 0.0)
    release_year = _extract_release_year(row.get("release_date", ""))
    playtime = float(row.get("average_playtime", 0.0) or 0.0)

    # =========================================================================
    # A. HYPE SCORE HEURISTIC (1.0 - 10.0)
    # Reflects viral potential, multiplayer reach, competitive scene & market buzz
    # =========================================================================
    hype = 4.5  # Neutral baseline

    # Multiplayer, competitive, and virality feature signals
    if any(m in categories or m in tags for m in ["multi-player", "online multi-player", "pvp", "online pvp", "cross-platform"]):
        hype += 1.0
    if "co-op" in categories or "co-op" in tags:
        hype += 0.5
    if "steam workshop" in categories or "steam workshop" in tags:
        hype += 0.5
    if any(g in genres or g in tags for g in ["action", "massively multiplayer", "mmo", "esports", "battle royale", "competitive"]):
        hype += 1.0
    if "free to play" in genres or "free to play" in tags:
        hype += 0.4

    # Publisher prestige bonus
    if any(pub in publisher for pub in HIGH_HYPE_PUBLISHERS):
        hype += 1.2

    # Market footprint (owners tiers)
    hype += OWNERS_HYPE_WEIGHTS.get(owners_bracket, 0.0)

    # Pricing signal (premium AAA tier typically carries substantial PR hype)
    if price >= 39.99:
        hype += 0.8
    elif price >= 19.99:
        hype += 0.4

    # Deep player retention signal
    if playtime > 1000:
        hype += 0.5

    hype_score = round(float(np.clip(hype, 1.0, 10.0)), 1)

    # =========================================================================
    # B. STORY SCORE HEURISTIC (1.0 - 10.0)
    # Reflects narrative depth, rich lore, dialogue, character arcs & worldbuilding
    # =========================================================================
    story = 4.0  # Neutral baseline

    # Primary narrative genres
    if "visual novel" in genres or "visual novel" in tags:
        story += 3.0
    if "rpg" in genres or "rpg" in tags:
        story += 2.0
    if "adventure" in genres or "adventure" in tags:
        story += 1.0
    if any(t in tags for t in ["interactive fiction", "choose your own adventure"]):
        story += 2.5

    # Narrative community tags
    if any(t in tags for t in ["story rich", "lore-rich", "deep lore", "choices matter"]):
        story += 2.2
    if any(t in tags for t in ["mystery", "detective", "investigation", "walking simulator", "point & click"]):
        story += 1.0
    if any(t in tags for t in ["atmospheric", "psychological", "psychological horror"]):
        story += 0.8

    # Single-player focus
    if "single-player" in categories or "single-player" in tags:
        story += 0.5

    # Mechanics-only / non-narrative genre penalties
    mechanical_genres = ["casual", "puzzle", "racing", "sports", "arcade", "clicker", "match 3"]
    if any(g in genres for g in mechanical_genres) and not any(t in tags for t in ["story rich", "rpg", "visual novel", "lore-rich"]):
        story -= 2.0

    if "multiplayer only" in tags or ("pvp" in tags and "single-player" not in categories):
        story -= 1.0

    story_score = round(float(np.clip(story, 1.0, 10.0)), 1)

    # =========================================================================
    # C. NOSTALGIA SCORE HEURISTIC (1.0 - 10.0)
    # Reflects retro aesthetic, classic homage, 8-bit/16-bit pixel art & vintage era
    # =========================================================================
    nostalgia = 3.5  # Neutral baseline

    # Vintage release year bonus
    if release_year < 1999:
        nostalgia += 4.5  # Historic pioneer era (Half-Life, Fallout, DOOM, Quake)
    elif release_year <= 2004:
        nostalgia += 3.5  # Golden era classic (Counter-Strike, Deus Ex, Half-Life 2)
    elif release_year <= 2008:
        nostalgia += 2.5  # Vintage PC era (Portal, Bioshock, Team Fortress 2)
    elif release_year <= 2012:
        nostalgia += 1.5  # Early modern era
    elif release_year <= 2015:
        nostalgia += 0.8

    # Retro community tags & genres
    if any(t in tags or t in genres for t in ["retro", "pixel graphics", "classic", "arcade", "old school"]):
        nostalgia += 2.2
    if any(t in tags or t in genres for t in ["remake", "remaster", "remastered", "8-bit", "16-bit", "boomer shooter"]):
        nostalgia += 1.8
    if any(t in tags for t in ["2d", "platformer", "metroidvania", "dungeon crawler", "crpg"]):
        nostalgia += 0.6

    # Title franchise legacy signal
    if re.search(r"\b(ii|iii|iv|v|vi|classic|remastered|hd|edition|anniversary|legacy|collection)\b", title):
        nostalgia += 0.8

    nostalgia_score = round(float(np.clip(nostalgia, 1.0, 10.0)), 1)

    return {
        "hype_score": hype_score,
        "story_score": story_score,
        "nostalgia_score": nostalgia_score
    }


def compute_vibe_scores_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes heuristic vibe scores across all rows in a DataFrame purely from non-description metadata:
    - hype_score
    - story_score
    - nostalgia_score
    """
    df_copy = df.copy()
    print(f"Calculating description-leakage-free heuristic vibe scores for {len(df_copy):,} games...")

    vibe_scores = [calculate_game_vibe_scores(row) for _, row in df_copy.iterrows()]
    df_vibe = pd.DataFrame(vibe_scores, index=df_copy.index)

    df_copy["hype_score"] = df_vibe["hype_score"]
    df_copy["story_score"] = df_vibe["story_score"]
    df_copy["nostalgia_score"] = df_vibe["nostalgia_score"]

    print("Heuristic vibe score calculation complete (description text excluded):")
    print(f"  * Hype score:      mean={df_copy['hype_score'].mean():.2f}, std={df_copy['hype_score'].std():.2f}, min={df_copy['hype_score'].min()}, max={df_copy['hype_score'].max()}")
    print(f"  * Story score:     mean={df_copy['story_score'].mean():.2f}, std={df_copy['story_score'].std():.2f}, min={df_copy['story_score'].min()}, max={df_copy['story_score'].max()}")
    print(f"  * Nostalgia score: mean={df_copy['nostalgia_score'].mean():.2f}, std={df_copy['nostalgia_score'].std():.2f}, min={df_copy['nostalgia_score'].min()}, max={df_copy['nostalgia_score'].max()}")

    return df_copy


def run_cli():
    parser = argparse.ArgumentParser(description="Steam Hit Predictor - Description-Free Vibe Score Generator")
    parser.add_argument("--input", type=str, default="cleaned_data/train/games.csv", help="Input CSV file path")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file path")
    parser.add_argument("--test_game", type=str, default=None, help="Evaluate a specific game name from dataset")
    args = parser.parse_args()

    if args.test_game:
        if os.path.exists(args.input):
            df = pd.read_csv(args.input)
            match = df[df["name"].str.contains(args.test_game, case=False, na=False)]
            if not match.empty:
                row = match.iloc[0]
                scores = calculate_game_vibe_scores(row)
                print(f"\nGame: '{row['name']}' ({row.get('genres', '')})")
                print(f"Calculated Vibe Scores (metadata-only): {scores}\n")
            else:
                print(f"Game '{args.test_game}' not found in {args.input}.")
        return

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        return

    df = pd.read_csv(args.input)
    df_scored = compute_vibe_scores_dataframe(df)

    out_path = args.output or args.input
    df_scored.to_csv(out_path, index=False)
    print(f"Saved scored dataset to: {out_path}")


if __name__ == "__main__":
    run_cli()
