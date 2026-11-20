import React, { useState, useEffect } from 'react';
import { Play, Sparkles, Sliders, RotateCcw, AlertCircle, HelpCircle, Layers } from 'lucide-react';

const API_BASE = 'http://127.0.0.1:8000';

const PRESETS = [
  {
    name: 'Metroidvania Masterpiece (Hit Preset)',
    data: {
      name: 'Crypt of the Hollow Sovereign',
      description: 'An atmospheric hand-drawn 2D metroidvania through a ruined insect kingdom. Uncover deep forgotten lore, master fluid acrobatic combat, and challenge over 40 epic bosses in a haunting interconnected world with an orchestral soundtrack.',
      developer: 'Team Moonstone',
      publisher: 'Team Moonstone',
      primary_genre: 'Action',
      genres: 'Action;Indie;Adventure',
      categories: 'Single-player;Steam Achievements;Full controller support;Steam Cloud',
      steamspy_tags: 'Metroidvania;Souls-like;Difficult;Atmospheric;2D;Indie;Action',
      platforms: 'windows;mac;linux',
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
    name: 'Retro Boomer Shooter (Hit Preset)',
    data: {
      name: 'Duskfall: Bloodline 1996',
      description: 'Fast-paced retro 90s FPS boomer shooter with pixelated bloody action, strafe jumping, dual-wield shotguns, industrial metal soundtrack, and labyrinthine level design reminiscent of Quake and Doom.',
      developer: 'Iron Pixel Studios',
      publisher: 'New Blood Interactive',
      primary_genre: 'Action',
      genres: 'Action;Indie',
      categories: 'Single-player;Steam Achievements;Steam Cloud',
      steamspy_tags: 'Boomer Shooter;Retro;FPS;Fast-Paced;Gore;Action;90s',
      platforms: 'windows;linux',
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
    name: 'Generic Zombie Survival (Flop Preset)',
    data: {
      name: 'Dead Survival Zombie Island Online',
      description: 'Survive in an open world island filled with walking dead zombies. Collect wood and stone, build a base, find loot, and shoot zombies. Early access survival game with unity assets.',
      developer: 'Asset Flip Games',
      publisher: 'Asset Flip Games',
      primary_genre: 'Action',
      genres: 'Action;Adventure;Massively Multiplayer',
      categories: 'Multi-player;PvP',
      steamspy_tags: 'Zombies;Survival;Open World;Early Access;Shooter;Multiplayer',
      platforms: 'windows',
      price: 29.99,
      english: 1,
      required_age: 0,
      achievements: 0,
      use_manual_vibes: false,
      hype_score: 2.1,
      story_score: 1.5,
      nostalgia_score: 1.0
    }
  },
  {
    name: 'Story-Rich Cozy Farming Sim (Hit Preset)',
    data: {
      name: 'Starlight Orchard Valley',
      description: 'A cozy heartwarming narrative farming simulator. Inherit your grandmothers quaint countryside cottage, befriend quirky village townsfolk, uncover ancient forest spirits, and build your dream orchard.',
      developer: 'Breeze Studio',
      publisher: 'Chucklefish',
      primary_genre: 'Simulation',
      genres: 'Simulation;RPG;Casual;Indie',
      categories: 'Single-player;Steam Achievements;Controller support',
      steamspy_tags: 'Cozy;Farming Sim;Relaxing;Cute;Story Rich;Life Sim;RPG',
      platforms: 'windows;mac',
      price: 14.99,
      english: 1,
      required_age: 0,
      achievements: 36,
      use_manual_vibes: false,
      hype_score: 7.5,
      story_score: 8.8,
      nostalgia_score: 8.2
    }
  }
];

export default function PredictionStudio({ initialGame, onPredictionComplete }) {
  const [formData, setFormData] = useState({
    name: 'Chronicles of Eldoria',
    description: 'An epic dark-fantasy roguelike deckbuilder with branching narrative choices, procedural dungeons, atmospheric pixel art, and deeply tactical turn-based card combat.',
    developer: 'Mythic Forge Studios',
    publisher: 'Indie Publisher Co.',
    primary_genre: 'RPG',
    genres: 'RPG;Strategy;Indie',
    categories: 'Single-player;Steam Achievements;Steam Cloud',
    steamspy_tags: 'Roguelike Deckbuilder;Card Battler;Dark Fantasy;Turn-Based;Pixel Art',
    platforms: 'windows;mac',
    price: 19.99,
    english: 1,
    required_age: 0,
    achievements: 25,
    use_manual_vibes: false,
    hype_score: 7.5,
    story_score: 8.0,
    nostalgia_score: 6.5
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [genresList, setGenresList] = useState([]);

  // Load metadata for genre options
  useEffect(() => {
    fetch(`${API_BASE}/metadata`)
      .then((res) => res.json())
      .then((data) => {
        if (data.primary_genres) setGenresList(data.primary_genres);
      })
      .catch((err) => console.log('Metadata load note:', err));
  }, []);

  // Sync if initialGame is passed from DatasetExplorer
  useEffect(() => {
    if (initialGame) {
      setFormData((prev) => ({
        ...prev,
        name: initialGame.name || prev.name,
        description: initialGame.short_description || prev.description,
        developer: initialGame.developer || prev.developer,
        publisher: initialGame.publisher || prev.publisher,
        primary_genre: initialGame.primary_genre || prev.primary_genre,
        genres: initialGame.genres || prev.genres,
        categories: initialGame.categories || prev.categories,
        steamspy_tags: initialGame.steamspy_tags || prev.steamspy_tags,
        price: initialGame.price !== undefined ? initialGame.price : prev.price,
        achievements: initialGame.achievements || prev.achievements,
        use_manual_vibes: true,
        hype_score: initialGame.hype_score || 7.0,
        story_score: initialGame.story_score || 7.0,
        nostalgia_score: initialGame.nostalgia_score || 7.0
      }));
    }
  }, [initialGame]);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? (checked ? 1 : 0) : value
    }));
  };

  const handlePlatformToggle = (platform) => {
    const platforms = formData.platforms.split(';').filter(Boolean);
    const exists = platforms.includes(platform);
    const updated = exists ? platforms.filter((p) => p !== platform) : [...platforms, platform];
    setFormData((prev) => ({ ...prev, platforms: updated.join(';') || 'windows' }));
  };

  const handlePriceQuickSet = (priceVal) => {
    setFormData((prev) => ({ ...prev, price: priceVal }));
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
        genres: formData.genres.trim(),
        categories: formData.categories.trim(),
        steamspy_tags: formData.steamspy_tags.trim(),
        platforms: formData.platforms,
        price: parseFloat(formData.price) || 0.0,
        english: parseInt(formData.english, 10),
        required_age: parseInt(formData.required_age, 10) || 0,
        achievements: parseInt(formData.achievements, 10) || 0,
        vibe_scores: formData.use_manual_vibes ? {
          hype_score: parseFloat(formData.hype_score),
          story_score: parseFloat(formData.story_score),
          nostalgia_score: parseFloat(formData.nostalgia_score)
        } : null
      };

      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Prediction failed with HTTP ${res.status}`);
      }

      const predictionData = await res.json();
      onPredictionComplete(predictionData);
    } catch (err) {
      setError(err.message || 'Failed to communicate with prediction backend');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="panel-card" style={{ padding: '1.75rem' }}>
      {/* Header with Preset Selector */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h2 style={{ fontSize: '1.35rem', fontWeight: 800, color: '#fff', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Layers size={22} style={{ color: 'var(--steam-blue)' }} />
            Developer Studio
          </h2>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            Configure game attributes & evaluate market hit potential using XGBoost + QLoRA LLM
          </p>
        </div>

        {/* Preset quick buttons */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Quick Presets:</span>
          <select
            onChange={(e) => {
              const idx = parseInt(e.target.value, 10);
              if (!isNaN(idx)) handleLoadPreset(PRESETS[idx]);
            }}
            defaultValue=""
            style={{ width: 'auto', padding: '6px 10px', fontSize: '0.8rem' }}
          >
            <option value="" disabled>Load Archetype...</option>
            {PRESETS.map((p, idx) => (
              <option key={p.name} value={idx}>{p.name}</option>
            ))}
          </select>
        </div>
      </div>

      {error && (
        <div style={{ background: 'rgba(239, 68, 68, 0.15)', border: '1px solid var(--flop-red)', padding: '0.85rem 1rem', borderRadius: '8px', display: 'flex', alignItems: 'center', gap: '0.75rem', color: '#fca5a5', marginBottom: '1.25rem' }}>
          <AlertCircle size={18} />
          <span style={{ fontSize: '0.85rem' }}>{error}</span>
        </div>
      )}

      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
        {/* Game Title & Genre */}
        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1rem' }}>
          <div>
            <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: '6px' }}>
              Game Title *
            </label>
            <input
              type="text"
              name="name"
              required
              value={formData.name}
              onChange={handleChange}
              placeholder="e.g. Hades, Terraria, Lethal Company"
            />
          </div>

          <div>
            <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: '6px' }}>
              Primary Genre
            </label>
            <select name="primary_genre" value={formData.primary_genre} onChange={handleChange}>
              {genresList.length > 0 ? (
                genresList.map((g) => <option key={g} value={g}>{g}</option>)
              ) : (
                <>
                  <option value="Action">Action</option>
                  <option value="Adventure">Adventure</option>
                  <option value="RPG">RPG</option>
                  <option value="Strategy">Strategy</option>
                  <option value="Simulation">Simulation</option>
                  <option value="Indie">Indie</option>
                  <option value="Casual">Casual</option>
                </>
              )}
            </select>
          </div>
        </div>

        {/* Short Description */}
        <div>
          <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
            <span>Steam Store Pitch / Description *</span>
            <span style={{ color: 'var(--steam-blue)', fontSize: '0.75rem' }}>Processed by QLoRA LLM</span>
          </label>
          <textarea
            name="description"
            rows="3"
            required
            value={formData.description}
            onChange={handleChange}
            placeholder="Describe gameplay mechanics, world setting, visual aesthetic, and mood..."
          />
        </div>

        {/* Developer & Publisher */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
          <div>
            <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: '6px' }}>
              Developer Studio
            </label>
            <input
              type="text"
              name="developer"
              value={formData.developer}
              onChange={handleChange}
              placeholder="e.g. Supergiant Games"
            />
          </div>

          <div>
            <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: '6px' }}>
              Publisher
            </label>
            <input
              type="text"
              name="publisher"
              value={formData.publisher}
              onChange={handleChange}
              placeholder="e.g. Devolver Digital"
            />
          </div>
        </div>

        {/* Pricing with Quick Buttons */}
        <div>
          <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
            <span>Launch Price ($ USD)</span>
            <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>Normalized in ML Pipeline</span>
          </label>
          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
            <div style={{ position: 'relative', width: '130px', flexShrink: 0 }}>
              <span style={{ position: 'absolute', left: '12px', top: '10px', color: 'var(--text-muted)' }}>$</span>
              <input
                type="number"
                step="0.01"
                min="0"
                name="price"
                value={formData.price}
                onChange={handleChange}
                style={{ paddingLeft: '26px' }}
              />
            </div>
            <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
              {[0, 4.99, 9.99, 14.99, 19.99, 29.99, 59.99].map((p) => (
                <button
                  type="button"
                  key={p}
                  onClick={() => handlePriceQuickSet(p)}
                  style={{
                    padding: '6px 10px', fontSize: '0.75rem', borderRadius: '6px',
                    background: formData.price === p ? 'var(--steam-blue)' : 'var(--bg-secondary)',
                    color: formData.price === p ? '#0a0e17' : 'var(--text-secondary)',
                    border: '1px solid var(--border-subtle)', cursor: 'pointer', fontWeight: 600
                  }}
                >
                  {p === 0 ? 'Free' : `$${p}`}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Tags & Categories */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
          <div>
            <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: '6px' }}>
              SteamSpy Tags (semicolon-separated)
            </label>
            <input
              type="text"
              name="steamspy_tags"
              value={formData.steamspy_tags}
              onChange={handleChange}
              placeholder="Action;Roguelike;Indie;Difficult"
            />
          </div>

          <div>
            <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: '6px' }}>
              Categories (semicolon-separated)
            </label>
            <input
              type="text"
              name="categories"
              value={formData.categories}
              onChange={handleChange}
              placeholder="Single-player;Steam Achievements;Controller support"
            />
          </div>
        </div>

        {/* Platforms & Achievements */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem', alignItems: 'center' }}>
          <div>
            <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: '6px' }}>
              Target Platforms
            </label>
            <div style={{ display: 'flex', gap: '0.75rem' }}>
              {['windows', 'mac', 'linux'].map((plat) => {
                const active = formData.platforms.includes(plat);
                return (
                  <button
                    type="button"
                    key={plat}
                    onClick={() => handlePlatformToggle(plat)}
                    style={{
                      padding: '6px 12px', borderRadius: '6px', fontSize: '0.8rem', cursor: 'pointer', textTransform: 'capitalize',
                      background: active ? 'rgba(102, 192, 244, 0.2)' : 'var(--bg-secondary)',
                      color: active ? 'var(--steam-blue)' : 'var(--text-muted)',
                      border: `1px solid ${active ? 'var(--steam-blue)' : 'var(--border-subtle)'}`,
                      fontWeight: active ? 700 : 500
                    }}
                  >
                    {plat}
                  </button>
                );
              })}
            </div>
          </div>

          <div>
            <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: '6px' }}>
              Steam Achievements Count
            </label>
            <input
              type="number"
              min="0"
              name="achievements"
              value={formData.achievements}
              onChange={handleChange}
            />
          </div>
        </div>

        {/* Vibe Score Control: Auto QLoRA vs Manual Override */}
        <div style={{ background: 'rgba(10, 14, 23, 0.65)', border: '1px solid var(--border-subtle)', borderRadius: '10px', padding: '1rem 1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: formData.use_manual_vibes ? '1rem' : '0' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <Sparkles size={18} style={{ color: formData.use_manual_vibes ? '#f59e0b' : 'var(--steam-blue)' }} />
              <div>
                <span style={{ fontSize: '0.875rem', fontWeight: 700, color: '#fff', display: 'block' }}>
                  {formData.use_manual_vibes ? 'Manual Vibe Scores Mode' : 'AI Autonomous Vibe Mode'}
                </span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                  {formData.use_manual_vibes
                    ? 'Custom scores will override the QLoRA LLM extractor'
                    : 'Fine-tuned QLoRA Qwen3.5-0.8B will analyze store description on the fly'}
                </span>
              </div>
            </div>

            <button
              type="button"
              onClick={() => setFormData((prev) => ({ ...prev, use_manual_vibes: !prev.use_manual_vibes }))}
              style={{
                background: formData.use_manual_vibes ? 'rgba(245, 158, 11, 0.15)' : 'rgba(102, 192, 244, 0.15)',
                color: formData.use_manual_vibes ? '#fbbf24' : 'var(--steam-blue)',
                border: `1px solid ${formData.use_manual_vibes ? 'rgba(245, 158, 11, 0.4)' : 'rgba(102, 192, 244, 0.4)'}`,
                padding: '6px 12px', borderRadius: '6px', fontSize: '0.775rem', fontWeight: 600, cursor: 'pointer'
              }}
            >
              {formData.use_manual_vibes ? 'Switch to AI Auto-Extract' : 'Override Manually'}
            </button>
          </div>

          {formData.use_manual_vibes && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem', marginTop: '0.75rem' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '4px' }}>
                  <span style={{ color: 'var(--steam-blue)' }}>Hype Score:</span>
                  <span style={{ fontWeight: 700, color: '#fff' }}>{parseFloat(formData.hype_score).toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="10"
                  step="0.1"
                  name="hype_score"
                  value={formData.hype_score}
                  onChange={handleChange}
                />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '4px' }}>
                  <span style={{ color: '#c084fc' }}>Story Score:</span>
                  <span style={{ fontWeight: 700, color: '#fff' }}>{parseFloat(formData.story_score).toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="10"
                  step="0.1"
                  name="story_score"
                  value={formData.story_score}
                  onChange={handleChange}
                />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '4px' }}>
                  <span style={{ color: '#fbbf24' }}>Nostalgia Score:</span>
                  <span style={{ fontWeight: 700, color: '#fff' }}>{parseFloat(formData.nostalgia_score).toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="10"
                  step="0.1"
                  name="nostalgia_score"
                  value={formData.nostalgia_score}
                  onChange={handleChange}
                />
              </div>
            </div>
          )}
        </div>

        {/* Submit Action Button */}
        <button
          type="submit"
          disabled={loading}
          style={{
            background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
            color: '#fff',
            border: 'none',
            borderRadius: '10px',
            padding: '14px 20px',
            fontSize: '1rem',
            fontWeight: 700,
            cursor: loading ? 'not-allowed' : 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '10px',
            boxShadow: '0 4px 20px rgba(16, 185, 129, 0.3)',
            transition: 'all 0.2s ease',
            marginTop: '0.5rem'
          }}
        >
          {loading ? (
            <>
              <div
                style={{
                  width: '18px', height: '18px', borderRadius: '50%',
                  border: '2px solid rgba(255,255,255,0.3)',
                  borderTopColor: '#fff',
                  animation: 'spin 1s linear infinite'
                }}
              />
              <span>Evaluating Market Viability...</span>
            </>
          ) : (
            <>
              <Play size={18} fill="#fff" />
              <span>Predict Steam Market Success</span>
            </>
          )}
        </button>
      </form>
    </div>
  );
}
