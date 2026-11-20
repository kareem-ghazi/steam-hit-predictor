import React, { useState, useEffect, useMemo } from 'react';
import {
  Search, Filter, Database, ArrowRight, RefreshCw, AlertCircle,
  ChevronLeft, ChevronRight, X, ArrowUpDown, DollarSign, Layers
} from 'lucide-react';

const API_BASE = 'http://127.0.0.1:8000';

export default function DatasetExplorer({ onSelectGameForStudio }) {
  const [games, setGames] = useState([]);
  const [totalGames, setTotalGames] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Filters & Controls
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedGenre, setSelectedGenre] = useState('');
  const [selectedRating, setSelectedRating] = useState('');
  const [selectedPriceBracket, setSelectedPriceBracket] = useState('');
  const [selectedPlatform, setSelectedPlatform] = useState('');
  const [sortBy, setSortBy] = useState('reviews_desc');
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
        let fetched = data.games || [];

        // Apply client-side price bracket filtering if set
        if (selectedPriceBracket) {
          if (selectedPriceBracket === 'free') {
            fetched = fetched.filter((g) => g.price === 0);
          } else if (selectedPriceBracket === 'budget') {
            fetched = fetched.filter((g) => g.price > 0 && g.price < 10);
          } else if (selectedPriceBracket === 'mid') {
            fetched = fetched.filter((g) => g.price >= 10 && g.price < 30);
          } else if (selectedPriceBracket === 'premium') {
            fetched = fetched.filter((g) => g.price >= 30);
          }
        }

        // Apply client-side sort
        if (sortBy === 'reviews_desc') {
          fetched.sort((a, b) => (b.total_reviews || 0) - (a.total_reviews || 0));
        } else if (sortBy === 'approval_desc') {
          fetched.sort((a, b) => (b.approval_rate || 0) - (a.approval_rate || 0));
        } else if (sortBy === 'approval_asc') {
          fetched.sort((a, b) => (a.approval_rate || 0) - (b.approval_rate || 0));
        } else if (sortBy === 'price_asc') {
          fetched.sort((a, b) => (a.price || 0) - (b.price || 0));
        } else if (sortBy === 'price_desc') {
          fetched.sort((a, b) => (b.price || 0) - (a.price || 0));
        } else if (sortBy === 'hype_desc') {
          fetched.sort((a, b) => (b.hype_score || 0) - (a.hype_score || 0));
        } else if (sortBy === 'story_desc') {
          fetched.sort((a, b) => (b.story_score || 0) - (a.story_score || 0));
        } else if (sortBy === 'nostalgia_desc') {
          fetched.sort((a, b) => (b.nostalgia_score || 0) - (a.nostalgia_score || 0));
        }

        setGames(fetched);
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
  }, [page, selectedGenre, selectedRating, hitFilter, selectedPriceBracket, sortBy]);

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    setPage(0);
    fetchGames();
  };

  const handleResetFilters = () => {
    setSearchQuery('');
    setSelectedGenre('');
    setSelectedRating('');
    setSelectedPriceBracket('');
    setSortBy('reviews_desc');
    setHitFilter('');
    setPage(0);
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
      {/* Header Banner */}
      <div className="surface-card" style={{ padding: '1.25rem 1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1.25rem' }}>
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

        {/* Grouped Filter Area (Hits / Flops segmented nav grouped directly with filters) */}
        <div style={{ background: 'var(--surface-container-lowest)', padding: '1.25rem', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(51, 65, 85, 0.5)' }}>
          {/* Top Row: Hit/Flop Segmented Nav + Search Bar */}
          <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'center', marginBottom: '1rem' }}>
            {/* Grouped Hit/Flop Segmented Controls */}
            <div className="segmented-nav" style={{ flexShrink: 0 }}>
              <button
                type="button"
                onClick={() => { setHitFilter(''); setPage(0); }}
                className={`segmented-tab ${hitFilter === '' ? 'active' : ''}`}
                style={{ fontSize: '0.78rem', padding: '6px 14px' }}
              >
                All Releases
              </button>
              <button
                type="button"
                onClick={() => { setHitFilter('1'); setPage(0); }}
                className={`segmented-tab ${hitFilter === '1' ? 'active' : ''}`}
                style={{ fontSize: '0.78rem', padding: '6px 14px' }}
              >
                <span className="pulsating-dot" style={{ backgroundColor: 'var(--status-positive-bright)', width: '5px', height: '5px' }} />
                Hits (≥80%)
              </button>
              <button
                type="button"
                onClick={() => { setHitFilter('0'); setPage(0); }}
                className={`segmented-tab ${hitFilter === '0' ? 'active' : ''}`}
                style={{ fontSize: '0.78rem', padding: '6px 14px' }}
              >
                <span className="pulsating-dot" style={{ backgroundColor: 'var(--status-negative-bright)', width: '5px', height: '5px' }} />
                Flops (&lt;80%)
              </button>
            </div>

            {/* Keyword Search */}
            <div style={{ position: 'relative', flex: 1, minWidth: '220px' }}>
              <Search size={15} style={{ position: 'absolute', left: '12px', top: '11px', color: 'var(--outline)' }} />
              <input
                type="text"
                placeholder="Search title, developer, publisher..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') handleSearchSubmit(e); }}
                style={{ paddingLeft: '34px', height: '36px' }}
              />
            </div>

            <button
              type="button"
              onClick={handleSearchSubmit}
              className="btn-primary"
              style={{ height: '36px', padding: '0 16px', fontSize: '0.85rem' }}
            >
              <Search size={14} /> Filter
            </button>
          </div>

          {/* Bottom Row: Additional Filter Options (Genre, Rating, Price, Sort) */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: '0.75rem', alignItems: 'center' }}>
            {/* Genre Filter */}
            <div>
              <label style={{ fontSize: '0.72rem', color: 'var(--outline)', fontWeight: 600, display: 'block', marginBottom: '3px' }}>
                Genre
              </label>
              <select
                value={selectedGenre}
                onChange={(e) => { setSelectedGenre(e.target.value); setPage(0); }}
                style={{ height: '34px', fontSize: '0.8rem', padding: '4px 10px' }}
              >
                <option value="">All Genres</option>
                {genresList.map((g) => (
                  <option key={g} value={g}>{g}</option>
                ))}
              </select>
            </div>

            {/* Rating Tier Filter */}
            <div>
              <label style={{ fontSize: '0.72rem', color: 'var(--outline)', fontWeight: 600, display: 'block', marginBottom: '3px' }}>
                Community Tier
              </label>
              <select
                value={selectedRating}
                onChange={(e) => { setSelectedRating(e.target.value); setPage(0); }}
                style={{ height: '34px', fontSize: '0.8rem', padding: '4px 10px' }}
              >
                <option value="">All Rating Tiers</option>
                <option value="Overwhelmingly Positive">Overwhelmingly Positive (95%+)</option>
                <option value="Very Positive">Very Positive (80-94%)</option>
                <option value="Positive">Positive</option>
                <option value="Mostly Positive">Mostly Positive (70-79%)</option>
                <option value="Mixed">Mixed (40-69%)</option>
                <option value="Mostly Negative">Mostly Negative (20-39%)</option>
                <option value="Overwhelmingly Negative">Overwhelmingly Negative (&lt;20%)</option>
              </select>
            </div>

            {/* Price Bracket Filter */}
            <div>
              <label style={{ fontSize: '0.72rem', color: 'var(--outline)', fontWeight: 600, display: 'block', marginBottom: '3px' }}>
                Price Bracket
              </label>
              <select
                value={selectedPriceBracket}
                onChange={(e) => { setSelectedPriceBracket(e.target.value); setPage(0); }}
                style={{ height: '34px', fontSize: '0.8rem', padding: '4px 10px' }}
              >
                <option value="">All Price Tiers</option>
                <option value="free">Free to Play ($0.00)</option>
                <option value="budget">Budget (&lt;$10.00)</option>
                <option value="mid">Mid-Tier ($10.00 - $29.99)</option>
                <option value="premium">Premium ($30.00+)</option>
              </select>
            </div>

            {/* Sort Filter */}
            <div>
              <label style={{ fontSize: '0.72rem', color: 'var(--outline)', fontWeight: 600, display: 'block', marginBottom: '3px' }}>
                Sort By
              </label>
              <select
                value={sortBy}
                onChange={(e) => { setSortBy(e.target.value); setPage(0); }}
                style={{ height: '34px', fontSize: '0.8rem', padding: '4px 10px' }}
              >
                <option value="reviews_desc">Most Community Reviews</option>
                <option value="approval_desc">Highest Approval %</option>
                <option value="approval_asc">Lowest Approval %</option>
                <option value="price_desc">Price: High to Low</option>
                <option value="price_asc">Price: Low to High</option>
                <option value="hype_desc">Highest Hype Signal</option>
                <option value="story_desc">Highest Story Depth</option>
                <option value="nostalgia_desc">Highest Nostalgia</option>
              </select>
            </div>
          </div>
        </div>
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

                  {/* High-Contrast Clear Progress Bars for Narrative Signals */}
                  <div style={{ background: '#0a0f1d', padding: '8px 10px', borderRadius: 'var(--radius-sm)', marginBottom: '1rem', border: '1px solid rgba(51, 65, 85, 0.6)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--outline)', marginBottom: '6px' }}>
                      <span style={{ fontWeight: 600, color: 'var(--on-surface-variant)' }}>Narrative Signals:</span>
                      <span className="font-mono" style={{ color: '#fff', fontSize: '0.72rem' }}>
                        H {g.hype_score.toFixed(1)} • S {g.story_score.toFixed(1)} • N {g.nostalgia_score.toFixed(1)}
                      </span>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '6px' }}>
                      {/* Hype Meter */}
                      <div>
                        <div style={{ height: '6px', background: '#1e293b', border: '1px solid #334155', borderRadius: '3px', overflow: 'hidden' }}>
                          <div
                            style={{
                              height: '100%',
                              width: `${(g.hype_score / 10) * 100}%`,
                              background: 'linear-gradient(90deg, #38bdf8 0%, #66c0f4 100%)',
                              borderRadius: '2px',
                              boxShadow: '0 0 6px rgba(102, 192, 244, 0.4)'
                            }}
                            title={`Hype: ${g.hype_score.toFixed(1)} / 10.0`}
                          />
                        </div>
                      </div>

                      {/* Story Meter */}
                      <div>
                        <div style={{ height: '6px', background: '#1e293b', border: '1px solid #334155', borderRadius: '3px', overflow: 'hidden' }}>
                          <div
                            style={{
                              height: '100%',
                              width: `${(g.story_score / 10) * 100}%`,
                              background: 'linear-gradient(90deg, #9333ea 0%, #c084fc 100%)',
                              borderRadius: '2px',
                              boxShadow: '0 0 6px rgba(192, 132, 252, 0.4)'
                            }}
                            title={`Story: ${g.story_score.toFixed(1)} / 10.0`}
                          />
                        </div>
                      </div>

                      {/* Nostalgia Meter */}
                      <div>
                        <div style={{ height: '6px', background: '#1e293b', border: '1px solid #334155', borderRadius: '3px', overflow: 'hidden' }}>
                          <div
                            style={{
                              height: '100%',
                              width: `${(g.nostalgia_score / 10) * 100}%`,
                              background: 'linear-gradient(90deg, #d97706 0%, #fbbf24 100%)',
                              borderRadius: '2px',
                              boxShadow: '0 0 6px rgba(251, 191, 36, 0.4)'
                            }}
                            title={`Nostalgia: ${g.nostalgia_score.toFixed(1)} / 10.0`}
                          />
                        </div>
                      </div>
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
