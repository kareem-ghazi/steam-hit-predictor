import React, { useState, useEffect } from 'react';
import {
  Sparkles, Sliders, Play, RotateCcw, AlertCircle, Check, Plus, X,
  Layers, Gamepad2, Monitor, Apple, Terminal, DollarSign, Tag, Info
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

// Benchmark Game Presets
const PRESETS = [
  {
    name: 'Metroidvania Masterpiece (Hit Archetype)',
    data: {
      name: 'Crypt of the Hollow Sovereign',
      description: 'An atmospheric hand-drawn 2D metroidvania through a ruined insect kingdom. Uncover deep forgotten lore, master fluid acrobatic combat, and challenge over 40 epic bosses in a haunting interconnected world with an orchestral soundtrack.',
      developer: 'Team Moonstone',
      publisher: 'Team Moonstone',
      primary_genre: 'Action',
      genres: ['Action', 'Indie', 'Adventure'],
      categories: ['Single-player', 'Steam Achievements', 'Full controller support', 'Steam Cloud'],
      steamspy_tags: ['Metroidvania', 'Souls-like', 'Difficult', 'Atmospheric', '2D', 'Indie'],
      platforms: ['windows', 'mac', 'linux'],
      price: 19.99,
      english: 1,
      required_age: 0,
      achievements: 45,
      use_manual_vibes: false,
      hype_score: 8.5,
      story_score: 9.0,
      nostalgia_score: 7.2
    }
  },
  {
    name: 'Retro 90s Boomer Shooter (Hit Archetype)',
    data: {
      name: 'Duskfall: Bloodline 1996',
      description: 'Fast-paced retro 90s FPS boomer shooter with pixelated bloody action, strafe jumping, dual-wield shotguns, industrial metal soundtrack, and labyrinthine level design reminiscent of Quake and Doom.',
      developer: 'Iron Pixel Studios',
      publisher: 'New Blood Interactive',
      primary_genre: 'Action',
      genres: ['Action', 'Indie'],
      categories: ['Single-player', 'Steam Achievements', 'Steam Cloud'],
      steamspy_tags: ['Boomer Shooter', 'FPS', 'Pixel Art', 'Atmospheric', 'Fast-Paced', 'Action'],
      platforms: ['windows', 'linux'],
      price: 14.99,
      english: 1,
      required_age: 18,
      achievements: 30,
      use_manual_vibes: false,
      hype_score: 8.0,
      story_score: 4.5,
      nostalgia_score: 9.6
    }
  },
  {
    name: 'Story-Rich Cozy Farming Sim (Hit Archetype)',
    data: {
      name: 'Starlight Orchard Valley',
      description: 'A cozy heartwarming narrative farming simulator. Inherit your grandmothers quaint countryside cottage, befriend quirky village townsfolk, uncover ancient forest spirits, and build your dream orchard.',
      developer: 'Breeze Studio',
      publisher: 'Chucklefish',
      primary_genre: 'Simulation',
      genres: ['Simulation', 'RPG', 'Casual', 'Indie'],
      categories: ['Single-player', 'Steam Achievements', 'Full controller support'],
      steamspy_tags: ['Cozy', 'Story Rich', 'Atmospheric', 'Crafting', '2D', 'Casual'],
      platforms: ['windows', 'mac'],
      price: 14.99,
      english: 1,
      required_age: 0,
      achievements: 36,
      use_manual_vibes: false,
      hype_score: 7.5,
      story_score: 8.8,
      nostalgia_score: 8.2
    }
  },
  {
    name: 'Generic Zombie Survival (Flop Archetype)',
    data: {
      name: 'Dead Survival Zombie Island Online',
      description: 'Survive in an open world island filled with walking dead zombies. Collect wood and stone, build a base, find loot, and shoot zombies. Early access survival game with unity assets.',
      developer: 'Asset Flip Games',
      publisher: 'Asset Flip Games',
      primary_genre: 'Action',
      genres: ['Action', 'Adventure', 'Massively Multiplayer'],
      categories: ['Multi-player', 'Online PvP'],
      steamspy_tags: ['Survival', 'Open World', 'Crafting', 'FPS', 'Action'],
      platforms: ['windows'],
      price: 29.99,
      english: 1,
      required_age: 0,
      achievements: 0,
      use_manual_vibes: false,
      hype_score: 2.1,
      story_score: 1.5,
      nostalgia_score: 1.0
    }
  }
];

export default function PredictionStudio({ initialGame, onPredictionComplete }) {
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
    achievements: 25,
    use_manual_vibes: false,
    hype_score: 7.5,
    story_score: 8.0,
    nostalgia_score: 6.5
  });

  const [customTagInput, setCustomTagInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

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

      setFormData((prev) => ({
        ...prev,
        name: initialGame.name || prev.name,
        description: initialGame.short_description || prev.description,
        developer: initialGame.developer || prev.developer,
        publisher: initialGame.publisher || prev.publisher,
        primary_genre: initialGame.primary_genre || parsedGenres[0] || prev.primary_genre,
        genres: parsedGenres.length > 0 ? parsedGenres : prev.genres,
        categories: parsedCats.length > 0 ? parsedCats : prev.categories,
        steamspy_tags: parsedTags.length > 0 ? parsedTags : prev.steamspy_tags,
        platforms: parsedPlats.length > 0 ? parsedPlats : prev.platforms,
        price: initialGame.price !== undefined ? initialGame.price : prev.price,
        achievements: initialGame.achievements || prev.achievements,
        use_manual_vibes: true,
        hype_score: initialGame.hype_score || 7.0,
        story_score: initialGame.story_score || 7.0,
        nostalgia_score: initialGame.nostalgia_score || 7.0
      }));
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
      // Ensure primary genre stays valid
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
      if (exists && prev.platforms.length === 1) return prev; // Keep at least one platform
      const updated = exists ? prev.platforms.filter((p) => p !== plat) : [...prev.platforms, plat];
      return { ...prev, platforms: updated };
    });
  };

  const handlePriceSet = (p) => {
    setFormData((prev) => ({ ...prev, price: p }));
  };

  const handleLoadPreset = (preset) => {
    setFormData({ ...preset.data });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
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
        achievements: parseInt(formData.achievements, 10) || 0,
        vibe_scores: formData.use_manual_vibes
          ? {
              hype_score: parseFloat(formData.hype_score),
              story_score: parseFloat(formData.story_score),
              nostalgia_score: parseFloat(formData.nostalgia_score)
            }
          : null
      };

      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Prediction failed with status ${res.status}`);
      }

      const predictionData = await res.json();
      onPredictionComplete(predictionData);
    } catch (err) {
      setError(err.message || 'Could not communicate with prediction service.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="surface-card" style={{ padding: '1.75rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Studio Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <div style={{ background: 'rgba(102, 192, 244, 0.15)', color: 'var(--primary)', padding: '6px', borderRadius: 'var(--radius-sm)' }}>
              <Gamepad2 size={20} />
            </div>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#fff', letterSpacing: '-0.01em' }}>
              Developer Studio
            </h2>
          </div>
          <p style={{ fontSize: '0.85rem', color: 'var(--on-surface-variant)' }}>
            Configure game attributes & evaluate Steam commercial viability
          </p>
        </div>

        {/* Archetype Quick Select */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '0.78rem', color: 'var(--outline)', fontWeight: 500 }}>
            Archetypes:
          </span>
          <select
            onChange={(e) => {
              const idx = parseInt(e.target.value, 10);
              if (!isNaN(idx)) handleLoadPreset(PRESETS[idx]);
            }}
            defaultValue=""
            style={{ width: 'auto', padding: '6px 12px', fontSize: '0.8rem' }}
          >
            <option value="" disabled>Load Preset...</option>
            {PRESETS.map((p, idx) => (
              <option key={p.name} value={idx}>{p.name}</option>
            ))}
          </select>
        </div>
      </div>

      {error && (
        <div style={{ background: 'var(--status-negative-bg)', border: '1px solid var(--status-negative-border)', padding: '10px 14px', borderRadius: 'var(--radius-sm)', display: 'flex', alignItems: 'center', gap: '8px', color: '#ffb4ab' }}>
          <AlertCircle size={16} />
          <span style={{ fontSize: '0.85rem' }}>{error}</span>
        </div>
      )}

      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        {/* Section 1: Title & Pitch Description */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1rem' }}>
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

            <div>
              <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--on-surface-variant)', display: 'block', marginBottom: '6px' }}>
                Primary Genre *
              </label>
              <select name="primary_genre" value={formData.primary_genre} onChange={handleInputChange}>
                {formData.genres.map((g) => (
                  <option key={g} value={g}>{g}</option>
                ))}
                {AVAILABLE_GENRES.filter((g) => !formData.genres.includes(g)).map((g) => (
                  <option key={g} value={g}>{g}</option>
                ))}
              </select>
            </div>
          </div>

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
              <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--on-surface-variant)' }}>
                Steam Store Pitch / Description *
              </label>
              <span className="font-mono" style={{ fontSize: '0.72rem', color: 'var(--primary)', letterSpacing: '0.02em' }}>
                Evaluated by Narrative Sentiment Engine
              </span>
            </div>
            <textarea
              name="description"
              rows="3"
              required
              value={formData.description}
              onChange={handleInputChange}
              placeholder="Describe gameplay mechanics, world setting, visual aesthetic, and mood..."
            />
          </div>
        </div>

        {/* Section 2: Interactive Genre Selection Chips */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--on-surface-variant)' }}>
              Select Genres ({formData.genres.length} active)
            </label>
            <span style={{ fontSize: '0.75rem', color: 'var(--outline)' }}>
              Click to toggle • Primary: <strong style={{ color: 'var(--primary)' }}>{formData.primary_genre}</strong>
            </span>
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

        {/* Section 3: Interactive Categories Selection Chips */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--on-surface-variant)' }}>
              Game Modes & Steam Features ({formData.categories.length} active)
            </label>
            <span style={{ fontSize: '0.75rem', color: 'var(--outline)' }}>
              Select all features planned for launch
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

        {/* Section 4: SteamSpy Tags with Autocomplete & Custom Addition */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--on-surface-variant)' }}>
              Steam Tags & Descriptors ({formData.steamspy_tags.length} selected)
            </label>
            <span style={{ fontSize: '0.75rem', color: 'var(--outline)' }}>
              Tags directly inform sub-genre classification
            </span>
          </div>

          {/* Active Tags Pills */}
          {formData.steamspy_tags.length > 0 && (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '10px' }}>
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

          {/* Tag Quick Select & Add */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '8px' }}>
            {POPULAR_TAGS.filter((t) => !formData.steamspy_tags.includes(t)).slice(0, 14).map((t) => (
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

        {/* Section 5: Developer, Publisher, Platforms, Price */}
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
            <div style={{ position: 'relative', width: '130px', flexShrink: 0 }}>
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

        {/* Section 6: Narrative Signals Calibration (Vibes) */}
        <div style={{ background: 'var(--surface-container-lowest)', border: '1px solid var(--outline-variant)', borderRadius: 'var(--radius-sm)', padding: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: formData.use_manual_vibes ? '1rem' : '0' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Sparkles size={18} style={{ color: formData.use_manual_vibes ? 'var(--status-warning)' : 'var(--primary)' }} />
              <div>
                <span style={{ fontSize: '0.875rem', fontWeight: 700, color: '#fff', display: 'block' }}>
                  {formData.use_manual_vibes ? 'Manual Signal Calibration Mode' : 'Autonomous Narrative Analysis'}
                </span>
                <span style={{ fontSize: '0.75rem', color: 'var(--outline)' }}>
                  {formData.use_manual_vibes
                    ? 'Manually tuning market excitement, lore depth, and nostalgia appeal'
                    : 'Store pitch text is evaluated dynamically to extract qualitative market signals'}
                </span>
              </div>
            </div>

            <button
              type="button"
              onClick={() => setFormData((prev) => ({ ...prev, use_manual_vibes: !prev.use_manual_vibes }))}
              className="btn-secondary"
              style={{ fontSize: '0.78rem', padding: '6px 12px' }}
            >
              <Sliders size={13} />
              <span>{formData.use_manual_vibes ? 'Switch to Autonomous Mode' : 'Calibrate Manually'}</span>
            </button>
          </div>

          {formData.use_manual_vibes && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem', marginTop: '0.75rem' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '4px' }}>
                  <span style={{ color: 'var(--primary)' }}>Hype & Excitement:</span>
                  <span className="font-mono" style={{ fontWeight: 700, color: '#fff' }}>
                    {parseFloat(formData.hype_score).toFixed(1)} / 10.0
                  </span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="10"
                  step="0.1"
                  name="hype_score"
                  value={formData.hype_score}
                  onChange={handleInputChange}
                />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '4px' }}>
                  <span style={{ color: '#c084fc' }}>Story & Lore Depth:</span>
                  <span className="font-mono" style={{ fontWeight: 700, color: '#fff' }}>
                    {parseFloat(formData.story_score).toFixed(1)} / 10.0
                  </span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="10"
                  step="0.1"
                  name="story_score"
                  value={formData.story_score}
                  onChange={handleInputChange}
                />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '4px' }}>
                  <span style={{ color: '#fbbf24' }}>Nostalgia Appeal:</span>
                  <span className="font-mono" style={{ fontWeight: 700, color: '#fff' }}>
                    {parseFloat(formData.nostalgia_score).toFixed(1)} / 10.0
                  </span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="10"
                  step="0.1"
                  name="nostalgia_score"
                  value={formData.nostalgia_score}
                  onChange={handleInputChange}
                />
              </div>
            </div>
          )}
        </div>

        {/* Submit Prediction Button */}
        <button
          type="submit"
          disabled={loading}
          className="btn-primary"
          style={{ width: '100%', padding: '13px 20px', fontSize: '0.975rem' }}
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
              <span>Forecasting Commercial Viability...</span>
            </>
          ) : (
            <>
              <Play size={18} fill="#0b1326" />
              <span>Run Commercial Viability Forecast</span>
            </>
          )}
        </button>
      </form>
    </div>
  );
}
