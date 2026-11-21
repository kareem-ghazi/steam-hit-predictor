#!/usr/bin/env python3
"""
SQLite Database Layer for Steam Hit Predictor
Persists private studio games and their historical analysis forecasts.
"""

import os
import sqlite3
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "steam_hit_predictor.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes SQLite schema for private games."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS private_games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT DEFAULT 'guest',
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                developer TEXT DEFAULT 'Independent Studio',
                publisher TEXT DEFAULT 'Independent Studio',
                primary_genre TEXT DEFAULT 'Action',
                genres TEXT DEFAULT 'Action;Indie',
                categories TEXT DEFAULT 'Single-player;Steam Achievements',
                steamspy_tags TEXT DEFAULT 'Action;Indie;Difficult',
                platforms TEXT DEFAULT 'windows',
                price REAL DEFAULT 19.99,
                english INTEGER DEFAULT 1,
                required_age INTEGER DEFAULT 0,
                achievements INTEGER DEFAULT 25,
                latest_prediction TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()

        # Seed initial sample private game if empty
        cursor.execute("SELECT COUNT(*) as cnt FROM private_games")
        count = cursor.fetchone()["cnt"]
        if count == 0:
            cursor.execute("""
                INSERT INTO private_games (
                    username, name, description, developer, publisher,
                    primary_genre, genres, categories, steamspy_tags,
                    platforms, price, english, required_age, achievements
                ) VALUES (
                    'guest',
                    'Chronicles of Eldoria',
                    'An atmospheric dark-fantasy roguelike deckbuilder featuring deep branching narrative choices, procedural dungeons, hand-crafted pixel art, and deeply tactical turn-based card combat.',
                    'Mythic Forge Studios',
                    'Indie Publisher Co.',
                    'RPG',
                    'RPG;Strategy;Indie',
                    'Single-player;Steam Achievements;Steam Cloud',
                    'Roguelike;Deckbuilder;Dark Fantasy;Turn-Based;Pixel Art',
                    'windows;mac',
                    19.99,
                    1,
                    0,
                    25
                );
            """)
            conn.commit()


def row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    if not row:
        return {}
    d = dict(row)
    if d.get("latest_prediction"):
        try:
            d["latest_prediction"] = json.loads(d["latest_prediction"])
        except Exception:
            pass
    return d


def list_private_games(username: Optional[str] = None) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        if username and username != "all":
            cursor.execute("""
                SELECT * FROM private_games
                WHERE username = ? OR username = 'guest'
                ORDER BY updated_at DESC, id DESC
            """, (username,))
        else:
            cursor.execute("SELECT * FROM private_games ORDER BY updated_at DESC, id DESC")
        rows = cursor.fetchall()
        return [row_to_dict(r) for r in rows]


def get_private_game(game_id: int) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM private_games WHERE id = ?", (game_id,))
        row = cursor.fetchone()
        return row_to_dict(row) if row else None


def create_private_game(data: Dict[str, Any], username: str = "guest") -> Dict[str, Any]:
    with get_connection() as conn:
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()
        cursor.execute("""
            INSERT INTO private_games (
                username, name, description, developer, publisher,
                primary_genre, genres, categories, steamspy_tags,
                platforms, price, english, required_age, achievements,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            username or "guest",
            data.get("name", "Untitled Project"),
            data.get("description", ""),
            data.get("developer", "Independent Studio"),
            data.get("publisher", "Independent Studio"),
            data.get("primary_genre", "Action"),
            data.get("genres", "Action;Indie"),
            data.get("categories", "Single-player"),
            data.get("steamspy_tags", ""),
            data.get("platforms", "windows"),
            float(data.get("price", 19.99)),
            int(data.get("english", 1)),
            int(data.get("required_age", 0)),
            int(data.get("achievements", 0)),
            now,
            now
        ))
        conn.commit()
        game_id = cursor.lastrowid
        return get_private_game(game_id)


def update_private_game(game_id: int, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    existing = get_private_game(game_id)
    if not existing:
        return None

    with get_connection() as conn:
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()
        cursor.execute("""
            UPDATE private_games SET
                name = ?,
                description = ?,
                developer = ?,
                publisher = ?,
                primary_genre = ?,
                genres = ?,
                categories = ?,
                steamspy_tags = ?,
                platforms = ?,
                price = ?,
                english = ?,
                required_age = ?,
                achievements = ?,
                updated_at = ?
            WHERE id = ?
        """, (
            data.get("name", existing["name"]),
            data.get("description", existing["description"]),
            data.get("developer", existing["developer"]),
            data.get("publisher", existing["publisher"]),
            data.get("primary_genre", existing["primary_genre"]),
            data.get("genres", existing["genres"]),
            data.get("categories", existing["categories"]),
            data.get("steamspy_tags", existing["steamspy_tags"]),
            data.get("platforms", existing["platforms"]),
            float(data.get("price", existing["price"])),
            int(data.get("english", existing["english"])),
            int(data.get("required_age", existing["required_age"])),
            int(data.get("achievements", existing["achievements"])),
            now,
            game_id
        ))
        conn.commit()
        return get_private_game(game_id)


def save_game_prediction(game_id: int, prediction_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()
        json_str = json.dumps(prediction_data)
        cursor.execute("""
            UPDATE private_games SET
                latest_prediction = ?,
                updated_at = ?
            WHERE id = ?
        """, (json_str, now, game_id))
        conn.commit()
        return get_private_game(game_id)


def delete_private_game(game_id: int) -> bool:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM private_games WHERE id = ?", (game_id,))
        conn.commit()
        return cursor.rowcount > 0
