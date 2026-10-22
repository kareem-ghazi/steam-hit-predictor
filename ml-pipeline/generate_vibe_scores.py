import os
import re
import argparse
from typing import Dict, Any, Tuple, Union
import numpy as np
import pandas as pd


# Major AAA / high-hype publishers that command inherent pre-release excitement
HIGH_HYPE_PUBLISHERS = {
    "valve", "electronic arts", "ubisoft", "bethesda softworks", "square enix",
    "capcom", "2k", "sega", "bandai namco entertainment", "activision", "warner bros. interactive entertainment",
    "cd projekt red", "rockstar games", "sony interactive entertainment", "xbox game studios",
    "blizzard entertainment", "paradox interactive", "devolver digital", "thq nordic", "deep silver"
}

# Hype & virality keywords in descriptions
HYPE_KEYWORDS = [
    r"\bmultiplayer\b", r"\bonline\b", r"\bpvp\b", r"\bbattle royale\b", r"\besports\b",
    r"\bcompetitive\b", r"\barena\b", r"\btournament\b", r"\bco-op\b", r"\bcooperative\b",
    r"\baction-packed\b", r"\bintense\b", r"\bexplosive\b", r"\bmassive\b", r"\bmmo\b",
    r"\bleaderboards?\b", r"\bclans?\b", r"\bstream(?:ing|ers?)?\b", r"\bsurvival\b",
    r"\bepic\b", r"\bphenomenon\b", r"\bblockbuster\b", r"\bviral\b", r"\bfranchise\b",
    r"\bunleash\b", r"\bthrilling\b", r"\bdominate\b", r"\bcombat\b", r"\bwarfare\b"
]

# Story & narrative keywords in descriptions
STORY_KEYWORDS = [
    r"\bstory(?:line)?\b", r"\bnarrative\b", r"\blore\b", r"\bcharacter(?:s)?\b",
    r"\bdialogue\b", r"\bplot\b", r"\bmystery\b", r"\bsecrets?\b", r"\bjourney\b",
    r"\bemotional\b", r"\bdestiny\b", r"\btragedy\b", r"\btragic\b", r"\bcinematic\b",
    r"\bvoice acting\b", r"\bcutscenes?\b", r"\bchoices matter\b", r"\bconsequences?\b",
    r"\bdeep\b", r"\bvisual novel\b", r"\bepic tale\b", r"\bdark fantasy\b", r"\brpg\b",
    r"\bworldbuilding\b", r"\buncover\b", r"\bdiscover the truth\b", r"\bchapter\b",
    r"\bprotagonist\b", r"\bmemories\b", r"\bphilosophical\b", r"\batmosphere\b", r"\batmospheric\b"
]

# Negative story indicators (mechanics-only games with zero story depth)
LOW_STORY_KEYWORDS = [
    r"\bmatch-3\b", r"\bmahjong\b", r"\bsolitaire\b", r"\bjigsaw\b", r"\bclicker\b",
    r"\bidle game\b", r"\bphysics sandbox\b", r"\bbenchmark\b", r"\btower defense\b"
]

# Nostalgia, retro & classic keywords in descriptions
NOSTALGIA_KEYWORDS = [
    r"\bretro\b", r"\bpixel(?:-art|\s+art|\s+graphics)?\b", r"\b8-bit\b", r"\b16-bit\b",
    r"\barcade\b", r"\bclassic\b", r"\bvintage\b", r"\boldschool\b", r"\bold-school\b",
    r"\bnostalg(?:ia|ic)\b", r"\bremake\b", r"\bremaster(?:ed)?\b", r"\bhomage\b",
    r"\btribute\b", r"\bgolden age\b", r"\bboomer shooter\b", r"\bcrt\b", r"\bdungeon crawler\b",
    r"\bturn-based\b", r"\bisometric\b", r"\bplatformer\b", r"\b2d\b", r"\bmetroidvania\b",
    r"\bpoint and click\b", r"\bshmup\b", r"\bshoot 'em up\b", r"\bchiptune\b",
    r"\blegendary\b", r"\bdefinitive edition\b", r"\banniversary\b", r"\b90s\b", r"\b80s\b"
]

# Owners bracket heuristic mapping
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
    "0-20000": -0.4
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
    for an individual Steam game by analyzing:
    1. Full description prose and title semantics
    2. Genre classifications & steamspy community tags
    3. Category features (multiplayer, workshop, singleplayer, co-op)
    4. Publisher pedigree & market traction (owners bracket)
    5. Release date vintage & historical heritage
    """
    # 1. Parse text & description
    title = str(row.get("name", "")).lower()
    desc_clean = str(row.get("clean_description", "")).lower()
    desc_short = str(row.get("short_description", "")).lower()
    desc_detailed = str(row.get("detailed_description", "")).lower()
    full_text = f"{title} {desc_clean} {desc_short} {desc_detailed}"

    # 2. Parse categories, genres, and tags
    genres = str(row.get("genres", "")).lower().replace(";", " ")
    categories = str(row.get("categories", "")).lower().replace(";", " ")
    tags = str(row.get("steamspy_tags", "")).lower().replace(";", " ")
    publisher = str(row.get("publisher", "")).lower()
    owners_bracket = str(row.get("owners", "")).strip()
    release_year = _extract_release_year(row.get("release_date", ""))

    # =========================================================================
    # A. HYPE SCORE HEURISTIC (1.0 - 10.0)
    # Reflects viral potential, multiplayer reach, competitive scene & buzz
    # =========================================================================
    hype = 4.5  # Baseline

    # Keyword density in description
    hype_kw_count = sum(len(re.findall(kw, full_text)) for kw in HYPE_KEYWORDS)
    hype += min(hype_kw_count * 0.22, 2.0)

    # Categories & tags
    if any(m in categories or m in tags for m in ["multi-player", "online multi-player", "pvp", "cross-platform"]):
        hype += 0.8
    if "co-op" in categories or "co-op" in tags:
        hype += 0.4
    if "steam workshop" in categories or "steam workshop" in tags:
        hype += 0.5
    if any(g in genres or g in tags for g in ["action", "massively multiplayer", "mmo", "esports"]):
        hype += 0.6
    if "free to play" in genres or "free to play" in tags:
        hype += 0.4

    # Publisher prestige
    if any(pub in publisher for pub in HIGH_HYPE_PUBLISHERS):
        hype += 1.0

    # Market footprint (owners tiers)
    hype += OWNERS_HYPE_WEIGHTS.get(owners_bracket, 0.0)

    # Price signal (premium AAA tier typically carries substantial PR hype)
    price = float(row.get("price", 0.0) or 0.0)
    if price >= 29.99:
        hype += 0.6
    elif price >= 14.99:
        hype += 0.3

    hype_score = round(float(np.clip(hype, 1.0, 10.0)), 1)

    # =========================================================================
    # B. STORY SCORE HEURISTIC (1.0 - 10.0)
    # Reflects narrative depth, rich lore, dialogue, character arcs & worldbuilding
    # =========================================================================
    story = 4.2  # Baseline

    # Keyword density in description
    story_kw_count = sum(len(re.findall(kw, full_text)) for kw in STORY_KEYWORDS)
    story += min(story_kw_count * 0.30, 3.2)

    # Genre & tag narrative indicators
    if "visual novel" in genres or "visual novel" in tags:
        story += 2.2
    if "rpg" in genres or "rpg" in tags:
        story += 1.5
    if "adventure" in genres or "adventure" in tags:
        story += 0.8
    if any(t in tags for t in ["story rich", "lore-rich", "choices matter", "mystery", "detective"]):
        story += 1.4
    if "atmospheric" in tags or "atmospheric" in categories:
        story += 0.6
    if "point & click" in tags:
        story += 0.8

    # Penalties for purely mechanical games lacking narrative intent
    low_story_hits = sum(len(re.findall(kw, full_text)) for kw in LOW_STORY_KEYWORDS)
    if low_story_hits > 0:
        story -= min(low_story_hits * 0.8, 2.5)

    if any(g in genres for g in ["casual", "puzzle", "racing", "sports"]) and story_kw_count == 0:
        story -= 1.2

    # Text length indicator: games with substantive descriptions usually have plot details
    desc_len = len(desc_clean)
    if desc_len > 600:
        story += 0.5
    elif desc_len < 100:
        story -= 0.6

    story_score = round(float(np.clip(story, 1.0, 10.0)), 1)

    # =========================================================================
    # C. NOSTALGIA SCORE HEURISTIC (1.0 - 10.0)
    # Reflects retro aesthetic, classic homage, 8-bit/16-bit pixel art & vintage era
    # =========================================================================
    nostalgia = 3.5  # Baseline

    # Vintage release year bonus
    if release_year < 1999:
        nostalgia += 4.5  # Historic pioneer era (Half-Life, Fallout, DOOM)
    elif release_year <= 2004:
        nostalgia += 3.5  # Golden era classic (Counter-Strike, Deus Ex)
    elif release_year <= 2008:
        nostalgia += 2.5  # Vintage PC era (Portal, Bioshock, Team Fortress 2)
    elif release_year <= 2012:
        nostalgia += 1.2  # Early modern era
    elif release_year <= 2015:
        nostalgia += 0.5

    # Retro & aesthetic keywords in description
    nostalgia_kw_count = sum(len(re.findall(kw, full_text)) for kw in NOSTALGIA_KEYWORDS)
    nostalgia += min(nostalgia_kw_count * 0.40, 3.5)

    # Retro tags
    if any(t in tags for t in ["retro", "pixel graphics", "classic", "arcade", "old school"]):
        nostalgia += 1.8
    if any(t in tags for t in ["remake", "remaster", "nostalgia", "great soundtrack"]):
        nostalgia += 1.0
    if "2d" in tags or "platformer" in tags:
        nostalgia += 0.4

    # Franchise legacy signal in title (e.g. Roman numerals, sequels, Definitive, Remastered)
    if re.search(r"\b(ii|iii|iv|v|vi|classic|remastered|hd|edition)\b", title):
        nostalgia += 0.6

    nostalgia_score = round(float(np.clip(nostalgia, 1.0, 10.0)), 1)

    return {
        "hype_score": hype_score,
        "story_score": story_score,
        "nostalgia_score": nostalgia_score
    }


def compute_vibe_scores_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes heuristic vibe scores across all rows in a DataFrame and updates:
    - hype_score
    - story_score
    - nostalgia_score
    """
    df_copy = df.copy()
    print(f"Calculating accurate heuristic vibe scores for {len(df_copy)} games...")

    vibe_scores = [calculate_game_vibe_scores(row) for _, row in df_copy.iterrows()]
    df_vibe = pd.DataFrame(vibe_scores, index=df_copy.index)

    df_copy["hype_score"] = df_vibe["hype_score"]
    df_copy["story_score"] = df_vibe["story_score"]
    df_copy["nostalgia_score"] = df_vibe["nostalgia_score"]

    print("Heuristic vibe score calculation complete:")
    print(f"  * Hype score:      mean={df_copy['hype_score'].mean():.2f}, std={df_copy['hype_score'].std():.2f}, min={df_copy['hype_score'].min()}, max={df_copy['hype_score'].max()}")
    print(f"  * Story score:     mean={df_copy['story_score'].mean():.2f}, std={df_copy['story_score'].std():.2f}, min={df_copy['story_score'].min()}, max={df_copy['story_score'].max()}")
    print(f"  * Nostalgia score: mean={df_copy['nostalgia_score'].mean():.2f}, std={df_copy['nostalgia_score'].std():.2f}, min={df_copy['nostalgia_score'].min()}, max={df_copy['nostalgia_score'].max()}")

    return df_copy


def run_cli():
    parser = argparse.ArgumentParser(description="Steam Hit Predictor - Heuristic Vibe Score Generator")
    parser.add_argument("--input", type=str, default="cleaned_data/train.csv", help="Input CSV file path")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file path (defaults to overwrite input)")
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
                print(f"Description: {str(row.get('clean_description', ''))[:200]}...")
                print(f"Calculated Vibe Scores: {scores}\n")
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
