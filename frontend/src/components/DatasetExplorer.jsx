import React, { useState, useEffect } from 'react';
import { Search, Filter, Database, Trophy, ArrowRight, RefreshCw, AlertCircle, ChevronLeft, ChevronRight } from 'lucide-react';

const API_BASE = 'http://127.0.0.1:8000';

export default function DatasetExplorer({ onSelectGameForStudio }) {
  const [games, setGames] = useState([]);
  const [totalGames, setTotalGames] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Filters & Pagination
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedGenre, setSelectedGenre] = useState('');
  const [selectedRating, setSelectedRating] = useState('');
  const [hitFilter, setHitFilter] = useState(''); // '' | '1' | '0'
  const [page, setPage] = useState(0);
  const pageSize = 12;

  // Metadata options
  const [genresList, setGenresList] = useState([]);

  useEffect(() => {
    // Fetch unique metadata for dropdowns
    fetch(`${API_BASE}/metadata`)
      .then((res) => res.json())
      .then((data) => {
        if (data.primary_genres) {
          setGenresList(data.primary_genres);
        }
      })
      .catch((err) => console.log('Metadata fetch note:', err));
  }, []);

  const fetchGames = () => {
    setLoading(true);
    setError(null);

    const params = new URLSearchParams({
      skip: (page * pageSize).toString(),
      limit: pageSize.toString()
    });

    if (searchQuery.trim()) params.append('query', searchQuery.trim());
    if (selectedGenre) params.append('genre', selectedGenre);
    if (selectedRating) params.append('rating_text', selectedRating);
    if (hitFilter !== '') params.append('is_hit', hitFilter);

    fetch(`${API_BASE}/games?${params.toString()}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP error ${res.status}`);
        return res.json();
      })
      .then((data) => {
        setGames(data.games || []);
        setTotalGames(data.total || 0);
        setLoading(false);
      })
      .catch((err) => {
        setError(`Could not connect to FastAPI server at ${API_BASE}. Make sure the backend is running.`);
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchGames();
  }, [page, selectedGenre, selectedRating, hitFilter]);

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    setPage(0);
    fetchGames();
  };

  const getRatingBadgeClass = (ratingText) => {
    const r = (ratingText || '').toLowerCase();
    if (r.includes('overwhelmingly positive')) return 'badge-overwhelmingly-positive';
    if (r.includes('very positive') || r === 'positive') return 'badge-very-positive';
    if (r.includes('mostly positive')) return 'badge-mostly-positive';
    if (r.includes('mixed')) return 'badge-mixed';
    return 'badge-mostly-negative';
  };

  const totalPages = Math.ceil(totalGames / pageSize);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Search & Filter Header */}
      <div className="panel-card" style={{ padding: '1.25rem 1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{ background: 'rgba(102, 192, 244, 0.15)', padding: '8px', borderRadius: '8px', color: 'var(--steam-blue)' }}>
              <Database size={22} />
            </div>
            <div>
              <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#fff' }}>Dataset Explorer</h2>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                Exploring {totalGames.toLocaleString()} verified Steam games across stratified train, valid, and test splits
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button
              onClick={() => { setHitFilter(''); setPage(0); }}
              style={{
                padding: '6px 14px', borderRadius: '6px', fontSize: '0.85rem', cursor: 'pointer', fontWeight: 600,
                background: hitFilter === '' ? 'var(--steam-blue)' : 'var(--bg-secondary)',
                color: hitFilter === '' ? '#0a0e17' : 'var(--text-secondary)',
                border: '1px solid var(--border-subtle)'
              }}
            >
              All ({totalGames.toLocaleString()})
            </button>
            <button
              onClick={() => { setHitFilter('1'); setPage(0); }}
              style={{
                padding: '6px 14px', borderRadius: '6px', fontSize: '0.85rem', cursor: 'pointer', fontWeight: 600,
                background: hitFilter === '1' ? 'rgba(16, 185, 129, 0.25)' : 'var(--bg-secondary)',
                color: hitFilter === '1' ? 'var(--hit-green)' : 'var(--text-secondary)',
                border: `1px solid ${hitFilter === '1' ? 'var(--hit-green)' : 'var(--border-subtle)'}`
              }}
            >
              ★ Hits (≥80% Approval)
            </button>
            <button
              onClick={() => { setHitFilter('0'); setPage(0); }}
              style={{
                padding: '6px 14px', borderRadius: '6px', fontSize: '0.85rem', cursor: 'pointer', fontWeight: 600,
                background: hitFilter === '0' ? 'rgba(239, 68, 68, 0.25)' : 'var(--bg-secondary)',
                color: hitFilter === '0' ? 'var(--flop-red)' : 'var(--text-secondary)',
                border: `1px solid ${hitFilter === '0' ? 'var(--flop-red)' : 'var(--border-subtle)'}`
              }}
            >
              Flops (&lt;80% Approval)
            </button>
          </div>
        </div>

        {/* Form controls */}
        <form onSubmit={handleSearchSubmit} style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.75rem' }}>
          <div style={{ position: 'relative' }}>
            <Search size={16} style={{ position: 'absolute', left: '12px', top: '12px', color: 'var(--text-muted)' }} />
            <input
              type="text"
              placeholder="Search by title, dev, pub..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{ paddingLeft: '36px' }}
            />
          </div>

          <select value={selectedGenre} onChange={(e) => { setSelectedGenre(e.target.value); setPage(0); }}>
            <option value="">All Genres</option>
            {genresList.map((g) => (
              <option key={g} value={g}>{g}</option>
            ))}
          </select>

          <select value={selectedRating} onChange={(e) => { setSelectedRating(e.target.value); setPage(0); }}>
            <option value="">All Rating Tiers</option>
            <option value="Overwhelmingly Positive">Overwhelmingly Positive (95%+)</option>
            <option value="Very Positive">Very Positive (80-94%)</option>
            <option value="Positive">Positive</option>
            <option value="Mostly Positive">Mostly Positive (70-79%)</option>
            <option value="Mixed">Mixed (40-69%)</option>
            <option value="Mostly Negative">Mostly Negative (20-39%)</option>
            <option value="Overwhelmingly Negative">Overwhelmingly Negative (&lt;20%)</option>
          </select>

          <button
            type="submit"
            style={{
              background: 'linear-gradient(135deg, #1a9fff 0%, #0077cc 100%)',
              color: '#fff', border: 'none', borderRadius: '8px', fontWeight: 600,
              padding: '10px 18px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px'
            }}
          >
            <Search size={16} /> Filter
          </button>
        </form>
      </div>

      {/* Error state */}
      {error && (
        <div style={{ background: 'rgba(239, 68, 68, 0.15)', border: '1px solid var(--flop-red)', padding: '1rem', borderRadius: '8px', display: 'flex', alignItems: 'center', gap: '0.75rem', color: '#fca5a5' }}>
          <AlertCircle size={20} />
          <span>{error}</span>
          <button onClick={fetchGames} style={{ marginLeft: 'auto', background: 'transparent', border: '1px solid #fca5a5', color: '#fff', padding: '4px 10px', borderRadius: '4px', cursor: 'pointer' }}>Retry</button>
        </div>
      )}

      {/* Loading state */}
      {loading ? (
        <div style={{ textAlign: 'center', padding: '4rem 1rem' }}>
          <RefreshCw className="animate-spin" size={32} style={{ color: 'var(--steam-blue)', margin: '0 auto 1rem' }} />
          <p style={{ color: 'var(--text-secondary)' }}>Loading games from cleaned dataset...</p>
        </div>
      ) : (
        <>
          {/* Games Grid */}
          <div className="grid-cards">
            {games.map((g) => (
              <div key={g.appid} className="panel-card" style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '0.5rem', marginBottom: '0.5rem' }}>
                    <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#fff', lineHeight: 1.3 }}>
                      {g.name}
                    </h3>
                    <span
                      style={{
                        fontSize: '0.75rem', fontWeight: 700, padding: '2px 8px', borderRadius: '4px', whiteSpace: 'nowrap',
                        background: g.is_hit === 1 ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                        color: g.is_hit === 1 ? 'var(--hit-green)' : 'var(--flop-red)',
                        border: `1px solid ${g.is_hit === 1 ? 'var(--hit-green)' : 'var(--flop-red)'}`
                      }}
                    >
                      {g.is_hit === 1 ? 'HIT' : 'FLOP'}
                    </span>
                  </div>

                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
                    by <span style={{ color: 'var(--text-secondary)' }}>{g.developer}</span> • {g.primary_genre}
                  </p>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.75rem' }}>
                    <span className={getRatingBadgeClass(g.rating_text)} style={{ fontSize: '0.75rem', fontWeight: 600, padding: '2px 8px', borderRadius: '4px' }}>
                      {g.rating_text}
                    </span>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                      {(g.approval_rate * 100).toFixed(1)}% ({g.total_reviews.toLocaleString()} revs)
                    </span>
                    <span style={{ marginLeft: 'auto', fontSize: '0.9rem', fontWeight: 700, color: g.price === 0 ? 'var(--hit-green)' : '#fff' }}>
                      {g.price === 0 ? 'FREE' : `$${g.price.toFixed(2)}`}
                    </span>
                  </div>

                  {g.short_description && (
                    <p style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', lineHeight: 1.4, marginBottom: '1rem', display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                      {g.short_description}
                    </p>
                  )}

                  {/* Vibe Score Bars */}
                  <div style={{ background: 'rgba(10, 14, 23, 0.6)', padding: '8px 10px', borderRadius: '6px', marginBottom: '1rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '4px' }}>
                      <span>Vibe Signals:</span>
                      <span>Hype {g.hype_score.toFixed(1)} • Story {g.story_score.toFixed(1)} • Retro {g.nostalgia_score.toFixed(1)}</span>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '4px', height: '4px' }}>
                      <div style={{ background: 'var(--steam-blue)', borderRadius: '2px', width: `${(g.hype_score / 10) * 100}%` }} title={`Hype: ${g.hype_score}`} />
                      <div style={{ background: '#a855f7', borderRadius: '2px', width: `${(g.story_score / 10) * 100}%` }} title={`Story: ${g.story_score}`} />
                      <div style={{ background: '#f59e0b', borderRadius: '2px', width: `${(g.nostalgia_score / 10) * 100}%` }} title={`Nostalgia: ${g.nostalgia_score}`} />
                    </div>
                  </div>
                </div>

                {onSelectGameForStudio && (
                  <button
                    onClick={() => onSelectGameForStudio(g)}
                    style={{
                      background: 'rgba(102, 192, 244, 0.1)', color: 'var(--steam-blue)',
                      border: '1px solid rgba(102, 192, 244, 0.3)', borderRadius: '6px',
                      padding: '8px 12px', fontSize: '0.825rem', fontWeight: 600,
                      cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
                      transition: 'all 0.2s ease'
                    }}
                  >
                    <span>Use in Developer Studio</span>
                    <ArrowRight size={14} />
                  </button>
                )}
              </div>
            ))}
          </div>

          {/* Pagination Footer */}
          {totalPages > 1 && (
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '1rem', padding: '1rem', background: 'var(--bg-secondary)', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                Page <span style={{ color: '#fff', fontWeight: 600 }}>{page + 1}</span> of <span style={{ color: '#fff', fontWeight: 600 }}>{totalPages}</span> ({totalGames.toLocaleString()} games)
              </p>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button
                  disabled={page === 0}
                  onClick={() => setPage(p => Math.max(0, p - 1))}
                  style={{
                    padding: '6px 12px', borderRadius: '6px', border: '1px solid var(--border-subtle)',
                    background: page === 0 ? 'transparent' : 'var(--bg-card)',
                    color: page === 0 ? 'var(--text-muted)' : '#fff', cursor: page === 0 ? 'not-allowed' : 'pointer',
                    display: 'flex', alignItems: 'center', gap: '4px'
                  }}
                >
                  <ChevronLeft size={16} /> Previous
                </button>
                <button
                  disabled={page >= totalPages - 1}
                  onClick={() => setPage(p => p + 1)}
                  style={{
                    padding: '6px 12px', borderRadius: '6px', border: '1px solid var(--border-subtle)',
                    background: page >= totalPages - 1 ? 'transparent' : 'var(--bg-card)',
                    color: page >= totalPages - 1 ? 'var(--text-muted)' : '#fff', cursor: page >= totalPages - 1 ? 'not-allowed' : 'pointer',
                    display: 'flex', alignItems: 'center', gap: '4px'
                  }}
                >
                  Next <ChevronRight size={16} />
                </button>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
