import React, { useState, useEffect } from 'react';
import {
  Search, Filter, Database, ArrowRight, RefreshCw, AlertCircle,
  ChevronLeft, ChevronRight, X, ExternalLink
} from 'lucide-react';

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
    fetch(`${API_BASE}/metadata`)
      .then((res) => res.json())
      .then((data) => {
        if (data.primary_genres) {
          setGenresList(data.primary_genres);
        }
      })
      .catch((err) => console.log('Metadata note:', err));
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
        setError(`Could not connect to FastAPI server at ${API_BASE}. Ensure backend is running.`);
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
    return 'badge-negative';
  };

  const totalPages = Math.ceil(totalGames / pageSize);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Search & Filter Header (Surface Tier 1) */}
      <div className="surface-card" style={{ padding: '1.25rem 1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{ background: 'rgba(102, 192, 244, 0.15)', padding: '8px', borderRadius: 'var(--radius-sm)', color: 'var(--primary)' }}>
              <Database size={20} />
            </div>
            <div>
              <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#fff' }}>Dataset Explorer</h2>
              <p style={{ fontSize: '0.825rem', color: 'var(--on-surface-variant)' }}>
                Exploring <span className="font-mono" style={{ color: '#fff', fontWeight: 600 }}>{totalGames.toLocaleString()}</span> verified Steam releases across stratified splits
              </p>
            </div>
          </div>

          {/* Quick Segmented Filter */}
          <div className="segmented-nav">
            <button
              onClick={() => { setHitFilter(''); setPage(0); }}
              className={`segmented-tab ${hitFilter === '' ? 'active' : ''}`}
              style={{ fontSize: '0.8rem', padding: '5px 12px' }}
            >
              All Releases
            </button>
            <button
              onClick={() => { setHitFilter('1'); setPage(0); }}
              className={`segmented-tab ${hitFilter === '1' ? 'active' : ''}`}
              style={{ fontSize: '0.8rem', padding: '5px 12px' }}
            >
              <span className="pulsating-dot" style={{ backgroundColor: 'var(--status-positive-bright)', width: '5px', height: '5px' }} />
              Hits (≥80%)
            </button>
            <button
              onClick={() => { setHitFilter('0'); setPage(0); }}
              className={`segmented-tab ${hitFilter === '0' ? 'active' : ''}`}
              style={{ fontSize: '0.8rem', padding: '5px 12px' }}
            >
              <span className="pulsating-dot" style={{ backgroundColor: 'var(--status-negative-bright)', width: '5px', height: '5px' }} />
              Flops (&lt;80%)
            </button>
          </div>
        </div>

        {/* Form controls */}
        <form onSubmit={handleSearchSubmit} style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.75rem' }}>
          <div style={{ position: 'relative' }}>
            <Search size={15} style={{ position: 'absolute', left: '12px', top: '11px', color: 'var(--outline)' }} />
            <input
              type="text"
              placeholder="Search title, developer, publisher..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{ paddingLeft: '34px' }}
            />
          </div>

          <select value={selectedGenre} onChange={(e) => { setSelectedGenre(e.target.value); setPage(0); }}>
            <option value="">All Genres</option>
            {genresList.map((g) => (
              <option key={g} value={g}>{g}</option>
            ))}
          </select>

          <select value={selectedRating} onChange={(e) => { setSelectedRating(e.target.value); setPage(0); }}>
            <option value="">All Community Tiers</option>
            <option value="Overwhelmingly Positive">Overwhelmingly Positive (95%+)</option>
            <option value="Very Positive">Very Positive (80-94%)</option>
            <option value="Positive">Positive</option>
            <option value="Mostly Positive">Mostly Positive (70-79%)</option>
            <option value="Mixed">Mixed (40-69%)</option>
            <option value="Mostly Negative">Mostly Negative (20-39%)</option>
            <option value="Overwhelmingly Negative">Overwhelmingly Negative (&lt;20%)</option>
          </select>

          <button type="submit" className="btn-primary" style={{ height: '38px', padding: '0 16px' }}>
            <Search size={15} /> Filter
          </button>
        </form>
      </div>

      {/* Error state */}
      {error && (
        <div style={{ background: 'var(--status-negative-bg)', border: '1px solid var(--status-negative-border)', padding: '1rem', borderRadius: 'var(--radius-sm)', display: 'flex', alignItems: 'center', gap: '0.75rem', color: '#ffb4ab' }}>
          <AlertCircle size={18} />
          <span style={{ fontSize: '0.85rem' }}>{error}</span>
          <button onClick={fetchGames} className="btn-secondary" style={{ marginLeft: 'auto', padding: '4px 10px', fontSize: '0.75rem' }}>
            Retry
          </button>
        </div>
      )}

      {/* Loading state */}
      {loading ? (
        <div style={{ textAlign: 'center', padding: '4rem 1rem' }}>
          <RefreshCw className="animate-spin" size={30} style={{ color: 'var(--primary)', margin: '0 auto 1rem' }} />
          <p style={{ color: 'var(--on-surface-variant)', fontSize: '0.9rem' }}>Loading releases from dataset archive...</p>
        </div>
      ) : (
        <>
          {/* Games Grid */}
          <div className="grid-cards">
            {games.map((g) => (
              <div
                key={g.appid}
                className="surface-card surface-card-hover"
                style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}
              >
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '0.5rem', marginBottom: '0.5rem' }}>
                    <div>
                      <span className="font-mono" style={{ fontSize: '0.68rem', color: 'var(--outline)', display: 'block', marginBottom: '2px' }}>
                        APPID: {g.appid}
                      </span>
                      <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#fff', lineHeight: 1.3 }}>
                        {g.name}
                      </h3>
                    </div>
                    <span
                      className={`badge-pill ${g.is_hit === 1 ? 'badge-overwhelmingly-positive' : 'badge-negative'}`}
                      style={{ fontSize: '0.68rem', padding: '2px 8px' }}
                    >
                      {g.is_hit === 1 ? 'HIT' : 'FLOP'}
                    </span>
                  </div>

                  <p style={{ fontSize: '0.78rem', color: 'var(--outline)', marginBottom: '0.75rem' }}>
                    by <span style={{ color: 'var(--on-surface-variant)' }}>{g.developer}</span> • {g.primary_genre}
                  </p>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '0.75rem' }}>
                    <span className={`badge-pill ${getRatingBadgeClass(g.rating_text)}`} style={{ padding: '2px 8px' }}>
                      {g.rating_text}
                    </span>
                    <span className="font-mono" style={{ fontSize: '0.78rem', color: 'var(--on-surface-variant)' }}>
                      {(g.approval_rate * 100).toFixed(1)}% ({g.total_reviews.toLocaleString()} revs)
                    </span>
                    <span className="font-mono" style={{ marginLeft: 'auto', fontSize: '0.85rem', fontWeight: 700, color: g.price === 0 ? 'var(--status-positive-bright)' : '#fff' }}>
                      {g.price === 0 ? 'FREE' : `$${g.price.toFixed(2)}`}
                    </span>
                  </div>

                  {g.short_description && (
                    <p style={{ fontSize: '0.8rem', color: 'var(--on-surface-variant)', lineHeight: 1.4, marginBottom: '1rem', display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                      {g.short_description}
                    </p>
                  )}

                  {/* Vibe Score Signals */}
                  <div style={{ background: 'var(--surface-container-lowest)', padding: '7px 10px', borderRadius: 'var(--radius-sm)', marginBottom: '1rem', border: '1px solid rgba(51, 65, 85, 0.4)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--outline)', marginBottom: '4px' }}>
                      <span>Narrative Signals:</span>
                      <span className="font-mono">H {g.hype_score.toFixed(1)} • S {g.story_score.toFixed(1)} • N {g.nostalgia_score.toFixed(1)}</span>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '4px', height: '3px' }}>
                      <div style={{ background: 'var(--primary)', borderRadius: '1px', width: `${(g.hype_score / 10) * 100}%` }} title={`Hype: ${g.hype_score}`} />
                      <div style={{ background: '#a855f7', borderRadius: '1px', width: `${(g.story_score / 10) * 100}%` }} title={`Story: ${g.story_score}`} />
                      <div style={{ background: '#f59e0b', borderRadius: '1px', width: `${(g.nostalgia_score / 10) * 100}%` }} title={`Nostalgia: ${g.nostalgia_score}`} />
                    </div>
                  </div>
                </div>

                {onSelectGameForStudio && (
                  <button
                    onClick={() => onSelectGameForStudio(g)}
                    className="btn-secondary"
                    style={{ width: '100%', fontSize: '0.8rem', padding: '7px 10px' }}
                  >
                    <span>Load into Developer Studio</span>
                    <ArrowRight size={13} />
                  </button>
                )}
              </div>
            ))}
          </div>

          {/* Pagination Footer */}
          {totalPages > 1 && (
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '1rem', padding: '0.85rem 1.25rem', background: 'var(--surface-container-low)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--outline-variant)' }}>
              <p className="font-mono" style={{ fontSize: '0.8rem', color: 'var(--on-surface-variant)' }}>
                Page <span style={{ color: '#fff', fontWeight: 600 }}>{page + 1}</span> of <span style={{ color: '#fff', fontWeight: 600 }}>{totalPages}</span> ({totalGames.toLocaleString()} releases)
              </p>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button
                  disabled={page === 0}
                  onClick={() => setPage(p => Math.max(0, p - 1))}
                  className="btn-secondary"
                  style={{ opacity: page === 0 ? 0.4 : 1, cursor: page === 0 ? 'not-allowed' : 'pointer', padding: '5px 10px', fontSize: '0.78rem' }}
                >
                  <ChevronLeft size={14} /> Previous
                </button>
                <button
                  disabled={page >= totalPages - 1}
                  onClick={() => setPage(p => p + 1)}
                  className="btn-secondary"
                  style={{ opacity: page >= totalPages - 1 ? 0.4 : 1, cursor: page >= totalPages - 1 ? 'not-allowed' : 'pointer', padding: '5px 10px', fontSize: '0.78rem' }}
                >
                  Next <ChevronRight size={14} />
                </button>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
