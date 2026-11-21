import React, { useState, useEffect } from 'react';
import {
  Sparkles, Play, RotateCcw, AlertCircle, Check, Plus, X,
  Gamepad2, Monitor, Apple, Terminal, DollarSign, Layers,
  Save, Trash2, FolderPlus, Database, CheckCircle2
} from 'lucide-react';

const API_BASE = 'http://127.0.0.1:8000';

// Curated standard Steam genres
const AVAILABLE_GENRES = [
  'Action', 'Adventure', 'RPG', 'Strategy', 'Simulation', 'Indie',
  'Casual', 'Massively Multiplayer', 'Racing', 'Sports', 'Early Access'
];

// Curated standard Steam categories
const AVAILABLE_CATEGORIES = [
  'Single-player', 'Multi-player', 'Online PvP', 'Co-op', 'Online Co-op',
  'Steam Achievements', 'Full controller support', 'Steam Cloud',
  'Steam Trading Cards', 'Steam Workshop', 'Cross-Platform Multiplayer',
  'In-App Purchases', 'Remote Play Together'
];

// Curated popular Steam tags
const POPULAR_TAGS = [
  'Roguelike', 'Souls-like', 'Metroidvania', 'Deckbuilder', 'Open World',
  'Crafting', 'Survival', 'Horror', 'Atmospheric', '2D', 'Pixel Art',
  'FPS', 'Boomer Shooter', 'Story Rich', 'Cozy', 'Difficult', 'Turn-Based',
  'Sci-fi', 'Fantasy', 'Cyberpunk', 'Space', 'Puzzle'
];

export default function PredictionStudio({
  initialGame,
  currentUser,
  onPredictionComplete,
  descriptionAnalyzerActive,
  hitAnalyzerActive
}) {
  // Private Games from SQLite
  const [privateGames, setPrivateGames] = useState([]);
  const [selectedGameId, setSelectedGameId] = useState(null);
  const [loadingProjects, setLoadingProjects] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);

  const [formData, setFormData] = useState({
    name: 'Chronicles of Eldoria',
    description: 'An atmospheric dark-fantasy roguelike deckbuilder featuring deep branching narrative choices, procedural dungeons, hand-crafted pixel art, and deeply tactical turn-based card combat.',
    developer: 'Mythic Forge Studios',
    publisher: 'Indie Publisher Co.',
    primary_genre: 'RPG',
    genres: ['RPG', 'Strategy', 'Indie'],
    categories: ['Single-player', 'Steam Achievements', 'Steam Cloud'],
    steamspy_tags: ['Roguelike', 'Deckbuilder', 'Dark Fantasy', 'Turn-Based', 'Pixel Art'],
    platforms: ['windows', 'mac'],
    price: 19.99,
    english: 1,
    required_age: 0,
    achievements: 25
  });

  const [customTagInput, setCustomTagInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch private games from SQLite backend
  const fetchPrivateGames = async (autoSelectId = null) => {
    setLoadingProjects(true);
    try {
      const username = currentUser?.username || 'guest';
      const res = await fetch(`${API_BASE}/api/private-games?username=${encodeURIComponent(username)}`);
      if (res.ok) {
        const games = await res.json();
        setPrivateGames(games);

        // Auto select specified or first game
        if (autoSelectId) {
          const target = games.find((g) => g.id === autoSelectId);
          if (target) loadGameIntoForm(target);
        } else if (games.length > 0 && !selectedGameId) {
          loadGameIntoForm(games[0]);
        }
      }
    } catch (err) {
      console.warn('Could not load private games:', err);
    } finally {
      setLoadingProjects(false);
    }
  };

  useEffect(() => {
    fetchPrivateGames();
  }, [currentUser]);

  // Load a private game from SQLite into form
  const loadGameIntoForm = (game) => {
    setSelectedGameId(game.id);
    const parsedGenres = (game.genres || 'Action;Indie')
      .split(';')
      .map((s) => s.trim())
      .filter(Boolean);
    const parsedCats = (game.categories || 'Single-player')
      .split(';')
      .map((s) => s.trim())
      .filter(Boolean);
    const parsedTags = (game.steamspy_tags || '')
      .split(';')
      .map((s) => s.trim())
      .filter(Boolean);
    const parsedPlats = (game.platforms || 'windows')
      .split(';')
      .map((s) => s.trim().toLowerCase())
      .filter(Boolean);

    setFormData({
      name: game.name || 'Untitled Game',
      description: game.description || '',
      developer: game.developer || 'Independent Studio',
      publisher: game.publisher || 'Independent Studio',
      primary_genre: game.primary_genre || parsedGenres[0] || 'Action',
      genres: parsedGenres.length > 0 ? parsedGenres : ['Action', 'Indie'],
      categories: parsedCats.length > 0 ? parsedCats : ['Single-player'],
      steamspy_tags: parsedTags,
      platforms: parsedPlats.length > 0 ? parsedPlats : ['windows'],
      price: game.price !== undefined ? game.price : 19.99,
      english: game.english !== undefined ? game.english : 1,
      required_age: game.required_age || 0,
      achievements: game.achievements || 0
    });

    // If this private game already has an analysis in SQLite, load it immediately!
    if (game.latest_prediction && onPredictionComplete) {
      onPredictionComplete(game.latest_prediction);
    }
  };

  // Sync if initialGame is sent from DatasetExplorer
  useEffect(() => {
    if (initialGame) {
      const parsedGenres = (initialGame.genres || '')
        .split(';')
        .map((s) => s.trim())
        .filter(Boolean);
      const parsedCats = (initialGame.categories || '')
        .split(';')
        .map((s) => s.trim())
        .filter(Boolean);
      const parsedTags = (initialGame.steamspy_tags || '')
        .split(';')
        .map((s) => s.trim())
        .filter(Boolean);
      const parsedPlats = (initialGame.platforms || 'windows')
        .split(';')
        .map((s) => s.trim().toLowerCase())
        .filter(Boolean);

      // Create new private game in SQLite from this explorer game
      handleCreateNewProject({
        name: `${initialGame.name} (Custom Project)`,
        description: initialGame.short_description || '',
        developer: initialGame.developer || 'Independent Studio',
        publisher: initialGame.publisher || 'Independent Studio',
        primary_genre: initialGame.primary_genre || parsedGenres[0] || 'Action',
        genres: parsedGenres.join(';') || 'Action;Indie',
        categories: parsedCats.join(';') || 'Single-player',
        steamspy_tags: parsedTags.join(';'),
        platforms: parsedPlats.join(';') || 'windows',
        price: initialGame.price !== undefined ? initialGame.price : 19.99,
        achievements: initialGame.achievements || 0
      });
    }
  }, [initialGame]);

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? (checked ? 1 : 0) : value
    }));
  };

  const toggleGenre = (genre) => {
    setFormData((prev) => {
      const exists = prev.genres.includes(genre);
      const updated = exists ? prev.genres.filter((g) => g !== genre) : [...prev.genres, genre];
      let newPrimary = prev.primary_genre;
      if (!updated.includes(newPrimary)) {
        newPrimary = updated[0] || 'Indie';
      }
      return { ...prev, genres: updated, primary_genre: newPrimary };
    });
  };

  const toggleCategory = (category) => {
    setFormData((prev) => {
      const exists = prev.categories.includes(category);
      const updated = exists ? prev.categories.filter((c) => c !== category) : [...prev.categories, category];
      return { ...prev, categories: updated };
    });
  };

  const toggleTag = (tag) => {
    setFormData((prev) => {
      const exists = prev.steamspy_tags.includes(tag);
      const updated = exists ? prev.steamspy_tags.filter((t) => t !== tag) : [...prev.steamspy_tags, tag];
      return { ...prev, steamspy_tags: updated };
    });
  };

  const handleAddCustomTag = (e) => {
    if (e.key === 'Enter' || e.type === 'click') {
      e.preventDefault();
      const tag = customTagInput.trim();
      if (tag && !formData.steamspy_tags.includes(tag)) {
        setFormData((prev) => ({
          ...prev,
          steamspy_tags: [...prev.steamspy_tags, tag]
        }));
        setCustomTagInput('');
      }
    }
  };

  const removeTag = (tagToRemove) => {
    setFormData((prev) => ({
      ...prev,
      steamspy_tags: prev.steamspy_tags.filter((t) => t !== tagToRemove)
    }));
  };

  const togglePlatform = (plat) => {
    setFormData((prev) => {
      const exists = prev.platforms.includes(plat);
      if (exists && prev.platforms.length === 1) return prev;
      const updated = exists ? prev.platforms.filter((p) => p !== plat) : [...prev.platforms, plat];
      return { ...prev, platforms: updated };
    });
  };

  const handlePriceSet = (p) => {
    setFormData((prev) => ({ ...prev, price: p }));
  };

  // Create a new private game in SQLite DB
  const handleCreateNewProject = async (overrideData = null) => {
    setError(null);
    try {
      const payload = overrideData || {
        name: `New Project ${privateGames.length + 1}`,
        description: 'A new private video game concept designed for commercial release on Steam.',
        developer: currentUser?.displayName || 'Independent Studio',
        publisher: currentUser?.displayName || 'Independent Studio',
        primary_genre: 'Action',
        genres: 'Action;Indie',
        categories: 'Single-player;Steam Achievements',
        steamspy_tags: 'Action;Indie;Difficult',
        platforms: 'windows',
        price: 19.99,
        english: 1,
        required_age: 0,
        achievements: 20,
        username: currentUser?.username || 'guest'
      };

      const res = await fetch(`${API_BASE}/api/private-games`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) throw new Error('Failed to create private game in database.');
      const newGame = await res.json();
      await fetchPrivateGames(newGame.id);
    } catch (err) {
      setError(err.message || 'Error creating private game.');
    }
  };

  // Save current edits to SQLite DB
  const handleSaveToDatabase = async () => {
    if (!selectedGameId) return;
    setError(null);
    try {
      const payload = {
        name: formData.name.trim(),
        description: formData.description.trim(),
        developer: formData.developer.trim() || 'Independent Studio',
        publisher: formData.publisher.trim() || 'Independent Studio',
        primary_genre: formData.primary_genre,
        genres: formData.genres.join(';'),
        categories: formData.categories.join(';'),
        steamspy_tags: formData.steamspy_tags.join(';'),
        platforms: formData.platforms.join(';'),
        price: parseFloat(formData.price) || 0.0,
        english: parseInt(formData.english, 10),
        required_age: parseInt(formData.required_age, 10) || 0,
        achievements: parseInt(formData.achievements, 10) || 0
      };

      const res = await fetch(`${API_BASE}/api/private-games/${selectedGameId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) throw new Error('Failed to update project in database.');
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 2500);
      fetchPrivateGames(selectedGameId);
    } catch (err) {
      setError(err.message || 'Error saving project.');
    }
  };

  // Delete private game from SQLite DB
  const handleDeleteProject = async () => {
    if (!selectedGameId) return;
    if (!window.confirm('Delete this private game project from your database?')) return;
    try {
      const res = await fetch(`${API_BASE}/api/private-games/${selectedGameId}`, {
        method: 'DELETE'
      });
      if (!res.ok) throw new Error('Failed to delete game.');
      setSelectedGameId(null);
      await fetchPrivateGames();
    } catch (err) {
      setError(err.message || 'Error deleting game.');
    }
  };

  // Submit prediction on the private game saved in SQLite
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      // 1. Ensure current edits are saved to SQLite first
      let activeId = selectedGameId;
      const gamePayload = {
        name: formData.name.trim(),
        description: formData.description.trim(),
        developer: formData.developer.trim() || 'Independent Studio',
        publisher: formData.publisher.trim() || 'Independent Studio',
        primary_genre: formData.primary_genre,
        genres: formData.genres.join(';'),
        categories: formData.categories.join(';'),
        steamspy_tags: formData.steamspy_tags.join(';'),
        platforms: formData.platforms.join(';'),
        price: parseFloat(formData.price) || 0.0,
        english: parseInt(formData.english, 10),
        required_age: parseInt(formData.required_age, 10) || 0,
        achievements: parseInt(formData.achievements, 10) || 0,
        username: currentUser?.username || 'guest'
      };

      if (!activeId) {
        // Create new project in SQLite
        const createRes = await fetch(`${API_BASE}/api/private-games`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(gamePayload)
        });
        const created = await createRes.json();
        activeId = created.id;
        setSelectedGameId(activeId);
      } else {
        // Update in SQLite
        await fetch(`${API_BASE}/api/private-games/${activeId}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(gamePayload)
        });
      }

      // 2. Trigger analysis on the stored SQLite private game
      const res = await fetch(`${API_BASE}/api/private-games/${activeId}/analyze`, {
        method: 'POST'
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Analysis failed with status ${res.status}`);
      }

      const predictionData = await res.json();
      onPredictionComplete(predictionData);

      // Refresh list to show updated prediction status
      fetchPrivateGames(activeId);
    } catch (err) {
      setError(err.message || 'Could not communicate with prediction service.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="surface-card" style={{ padding: '1.75rem', display: 'flex', flexDirection: 'column', gap: '1.4rem' }}>
      {/* Top Bar: Private Studio Projects Manager (SQLite Database) */}
      <div style={{ background: 'var(--surface-container-lowest)', padding: '12px 16px', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(102, 192, 244, 0.3)', display: 'flex', flexDirection: 'column', gap: '10px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Database size={16} style={{ color: 'var(--primary)' }} />
            <span style={{ fontSize: '0.825rem', fontWeight: 700, color: '#fff' }}>
              My Private Projects (SQLite Database)
            </span>
            <span className="font-mono" style={{ fontSize: '0.7rem', color: 'var(--outline)', background: 'rgba(51, 65, 85, 0.4)', padding: '2px 7px', borderRadius: 'var(--radius-sm)' }}>
              {privateGames.length} Saved
            </span>
          </div>

          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              type="button"
              onClick={() => handleCreateNewProject()}
              className="btn-secondary"
              style={{ padding: '5px 12px', fontSize: '0.78rem' }}
            >
              <Plus size={13} />
              <span>New Private Game</span>
            </button>

            <button
              type="button"
              onClick={handleSaveToDatabase}
              className="btn-secondary"
              style={{ padding: '5px 12px', fontSize: '0.78rem', color: saveSuccess ? 'var(--status-positive-bright)' : 'var(--primary)' }}
            >
              {saveSuccess ? <CheckCircle2 size={13} /> : <Save size={13} />}
              <span>{saveSuccess ? 'Saved' : 'Save Draft'}</span>
            </button>

            {privateGames.length > 1 && (
              <button
                type="button"
                onClick={handleDeleteProject}
                className="btn-secondary"
                style={{ padding: '5px 10px', fontSize: '0.78rem', color: '#fb7185', borderColor: 'rgba(244, 63, 94, 0.3)' }}
                title="Delete this project from database"
              >
                <Trash2 size={13} />
              </button>
            )}
          </div>
        </div>

        {/* Project Selector Dropdown */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <select
            value={selectedGameId || ''}
            onChange={(e) => {
              const id = parseInt(e.target.value, 10);
              const target = privateGames.find((g) => g.id === id);
              if (target) loadGameIntoForm(target);
            }}
            style={{
              flex: 1,
              background: '#0d1527',
              borderColor: 'rgba(51, 65, 85, 0.8)',
              fontSize: '0.85rem',
              fontWeight: 600,
              padding: '8px 12px'
            }}
          >
            {privateGames.map((g) => {
              const hitText = g.latest_prediction ? ` [${g.latest_prediction.prediction_label}: ${(g.latest_prediction.hit_probability * 100).toFixed(0)}%]` : ' [Unanalyzed]';
              return (
                <option key={g.id} value={g.id}>
                  {g.name} — {g.primary_genre} ({g.price === 0 ? 'Free' : `$${g.price}`}){hitText}
                </option>
              );
            })}
          </select>
        </div>
      </div>

      {error && (
        <div style={{ background: 'var(--status-negative-bg)', border: '1px solid var(--status-negative-border)', padding: '10px 14px', borderRadius: 'var(--radius-sm)', display: 'flex', alignItems: 'center', gap: '8px', color: '#ffb4ab' }}>
          <AlertCircle size={16} />
          <span style={{ fontSize: '0.85rem' }}>{error}</span>
        </div>
      )}

      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.35rem' }}>
        {/* Game Title */}
        <div>
          <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--on-surface-variant)', display: 'block', marginBottom: '6px' }}>
            Game Title *
          </label>
          <input
            type="text"
            name="name"
            required
            value={formData.name}
            onChange={handleInputChange}
            placeholder="e.g. Hollow Knight, Hades, Balatro"
          />
        </div>

        {/* Store Pitch / Description */}
        <div>
          <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--on-surface-variant)', display: 'block', marginBottom: '6px' }}>
            Steam Store Pitch / Description *
          </label>
          <textarea
            name="description"
            rows="3"
            required
            value={formData.description}
            onChange={handleInputChange}
            placeholder="Describe gameplay mechanics, world setting, visual aesthetic, and mood..."
          />
        </div>

        {/* Genres & Primary Genre Select Box */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px', flexWrap: 'wrap', gap: '8px' }}>
            <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--on-surface-variant)' }}>
              Select Genres ({formData.genres.length} active)
            </label>

            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '0.78rem', color: 'var(--outline)', fontWeight: 600 }}>
                Primary Genre:
              </span>
              <select
                name="primary_genre"
                value={formData.primary_genre}
                onChange={handleInputChange}
                style={{ width: 'auto', padding: '5px 10px', fontSize: '0.78rem', height: '32px' }}
              >
                {formData.genres.map((g) => (
                  <option key={g} value={g}>{g}</option>
                ))}
                {AVAILABLE_GENRES.filter((g) => !formData.genres.includes(g)).map((g) => (
                  <option key={g} value={g}>{g}</option>
                ))}
              </select>
            </div>
          </div>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {AVAILABLE_GENRES.map((g) => {
              const active = formData.genres.includes(g);
              const isPrimary = formData.primary_genre === g;
              return (
                <button
                  type="button"
                  key={g}
                  onClick={() => toggleGenre(g)}
                  className={`tag-chip ${active ? 'active' : ''}`}
                >
                  {active && <Check size={12} style={{ color: 'var(--primary)' }} />}
                  <span>{g}</span>
                  {isPrimary && (
                    <span style={{ fontSize: '0.65rem', background: 'var(--primary)', color: '#0b1326', padding: '1px 5px', borderRadius: '2px', fontWeight: 700 }}>
                      PRIMARY
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </div>

        {/* Categories Selection Chips */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--on-surface-variant)' }}>
              Game Modes & Steam Features ({formData.categories.length} active)
            </label>
            <span style={{ fontSize: '0.75rem', color: 'var(--outline)' }}>
              Select planned features
            </span>
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {AVAILABLE_CATEGORIES.map((c) => {
              const active = formData.categories.includes(c);
              return (
                <button
                  type="button"
                  key={c}
                  onClick={() => toggleCategory(c)}
                  className={`tag-chip ${active ? 'active' : ''}`}
                >
                  {active && <Check size={12} style={{ color: 'var(--primary)' }} />}
                  <span>{c}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* SteamSpy Tags */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--on-surface-variant)' }}>
              Steam Tags & Descriptors ({formData.steamspy_tags.length} selected)
            </label>
            <span style={{ fontSize: '0.75rem', color: 'var(--outline)' }}>
              Sub-genre and theme classification
            </span>
          </div>

          {formData.steamspy_tags.length > 0 && (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '8px' }}>
              {formData.steamspy_tags.map((t) => (
                <span
                  key={t}
                  style={{
                    background: 'rgba(102, 192, 244, 0.15)',
                    color: 'var(--primary)',
                    border: '1px solid rgba(102, 192, 244, 0.35)',
                    borderRadius: 'var(--radius-sm)',
                    padding: '3px 8px',
                    fontSize: '0.78rem',
                    fontWeight: 600,
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '4px'
                  }}
                >
                  {t}
                  <X
                    size={13}
                    style={{ cursor: 'pointer', opacity: 0.8 }}
                    onClick={() => removeTag(t)}
                  />
                </span>
              ))}
            </div>
          )}

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '8px' }}>
            {POPULAR_TAGS.filter((t) => !formData.steamspy_tags.includes(t)).slice(0, 12).map((t) => (
              <button
                type="button"
                key={t}
                onClick={() => toggleTag(t)}
                className="tag-chip"
                style={{ fontSize: '0.74rem', padding: '3px 8px' }}
              >
                <Plus size={11} /> {t}
              </button>
            ))}
          </div>

          <div style={{ display: 'flex', gap: '8px' }}>
            <input
              type="text"
              placeholder="Type custom tag and press Enter..."
              value={customTagInput}
              onChange={(e) => setCustomTagInput(e.target.value)}
              onKeyDown={handleAddCustomTag}
              style={{ fontSize: '0.8rem', padding: '7px 10px' }}
            />
            <button
              type="button"
              onClick={handleAddCustomTag}
              className="btn-secondary"
              style={{ padding: '7px 14px', whiteSpace: 'nowrap' }}
            >
              Add Tag
            </button>
          </div>
        </div>

        {/* Developer, Publisher */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
          <div>
            <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--on-surface-variant)', display: 'block', marginBottom: '6px' }}>
              Developer Studio
            </label>
            <input
              type="text"
              name="developer"
              value={formData.developer}
              onChange={handleInputChange}
              placeholder="e.g. Supergiant Games"
            />
          </div>

          <div>
            <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--on-surface-variant)', display: 'block', marginBottom: '6px' }}>
              Publisher
            </label>
            <input
              type="text"
              name="publisher"
              value={formData.publisher}
              onChange={handleInputChange}
              placeholder="e.g. Devolver Digital"
            />
          </div>
        </div>

        {/* Pricing Controls */}
        <div>
          <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--on-surface-variant)', display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
            <span>Launch Price ($ USD)</span>
            <span style={{ color: 'var(--outline)', fontSize: '0.75rem' }}>Continuous Normalized Signal</span>
          </label>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
            <div style={{ position: 'relative', width: '120px', flexShrink: 0 }}>
              <span style={{ position: 'absolute', left: '10px', top: '9px', color: 'var(--outline)' }}>$</span>
              <input
                type="number"
                step="0.01"
                min="0"
                name="price"
                value={formData.price}
                onChange={handleInputChange}
                className="font-mono"
                style={{ paddingLeft: '24px', fontWeight: 600 }}
              />
            </div>
            <div style={{ display: 'flex', gap: '5px', flexWrap: 'wrap' }}>
              {[0, 4.99, 9.99, 14.99, 19.99, 29.99, 39.99, 59.99].map((p) => (
                <button
                  type="button"
                  key={p}
                  onClick={() => handlePriceSet(p)}
                  className="tag-chip font-mono"
                  style={{
                    background: formData.price === p ? 'var(--primary)' : 'rgba(19, 27, 46, 0.85)',
                    color: formData.price === p ? '#0b1326' : 'var(--on-surface-variant)',
                    borderColor: formData.price === p ? 'var(--primary)' : 'rgba(51, 65, 85, 0.6)',
                    fontWeight: 700
                  }}
                >
                  {p === 0 ? 'FREE' : `$${p}`}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Platforms & Achievements */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', alignItems: 'center' }}>
          <div>
            <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--on-surface-variant)', display: 'block', marginBottom: '6px' }}>
              Target Platforms
            </label>
            <div style={{ display: 'flex', gap: '6px' }}>
              {[
                { id: 'windows', label: 'Windows', icon: Monitor },
                { id: 'mac', label: 'macOS', icon: Apple },
                { id: 'linux', label: 'Linux', icon: Terminal }
              ].map((plat) => {
                const active = formData.platforms.includes(plat.id);
                const IconComponent = plat.icon;
                return (
                  <button
                    type="button"
                    key={plat.id}
                    onClick={() => togglePlatform(plat.id)}
                    className={`tag-chip ${active ? 'active' : ''}`}
                    style={{ flex: 1, justifyContent: 'center', padding: '8px 10px' }}
                  >
                    <IconComponent size={14} />
                    <span>{plat.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          <div>
            <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--on-surface-variant)', display: 'block', marginBottom: '6px' }}>
              Steam Achievements
            </label>
            <input
              type="number"
              min="0"
              name="achievements"
              value={formData.achievements}
              onChange={handleInputChange}
              className="font-mono"
            />
          </div>
        </div>

        {/* Predict Action Button */}
        <button
          type="submit"
          disabled={loading || (!descriptionAnalyzerActive && !hitAnalyzerActive)}
          className="btn-primary"
          style={{ width: '100%', padding: '13px 20px', fontSize: '1rem', fontWeight: 700 }}
        >
          {loading ? (
            <>
              <div
                style={{
                  width: '18px', height: '18px', borderRadius: '50%',
                  border: '2px solid rgba(11, 19, 38, 0.4)',
                  borderTopColor: '#0b1326',
                  animation: 'spin 1s linear infinite'
                }}
              />
              <span>Persisting & Forecasting...</span>
            </>
          ) : (
            <>
              <Play size={18} fill="#0b1326" />
              <span>Predict</span>
            </>
          )}
        </button>
      </form>
    </div>
  );
}
