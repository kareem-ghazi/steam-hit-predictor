import React from 'react';
import {
  Sparkles, Trophy, Cpu, CheckCircle2, XCircle, TrendingUp,
  BookOpen, Clock, Flame, ShieldAlert, Award, Compass, BarChart3
} from 'lucide-react';

export default function ResultsPane({ result, loading }) {
  if (loading) {
    return (
      <div className="surface-card" style={{ padding: '3.5rem 2rem', textAlign: 'center', minHeight: '420px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ position: 'relative', width: '64px', height: '64px', marginBottom: '1.5rem' }}>
          <div
            style={{
              position: 'absolute', inset: 0, borderRadius: 'var(--radius-full)',
              border: '3px solid rgba(102, 192, 244, 0.15)',
              borderTopColor: 'var(--primary)',
              animation: 'spin 1s linear infinite'
            }}
          />
          <BarChart3 size={28} style={{ position: 'absolute', top: '18px', left: '18px', color: 'var(--primary)' }} />
        </div>
        <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: '#fff', marginBottom: '0.4rem' }}>
          Calculating Commercial Forecast...
        </h3>
        <p style={{ color: 'var(--on-surface-variant)', maxWidth: '380px', fontSize: '0.875rem', lineHeight: 1.5 }}>
          Evaluating 194 market dimensions, multi-label genre indicators, narrative pitch depth, and historical Steam benchmarks...
        </p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="surface-card" style={{ padding: '3.5rem 2rem', textAlign: 'center', minHeight: '420px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ background: 'rgba(255, 255, 255, 0.03)', borderRadius: 'var(--radius-full)', padding: '20px', marginBottom: '1.25rem', border: '1px solid rgba(51, 65, 85, 0.5)' }}>
          <Compass size={36} style={{ color: 'var(--primary)' }} />
        </div>
        <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#fff', marginBottom: '0.4rem' }}>
          Awaiting Game Concept
        </h3>
        <p style={{ color: 'var(--on-surface-variant)', maxWidth: '360px', fontSize: '0.85rem', lineHeight: 1.5 }}>
          Configure your game attributes on the left or load an archetype preset, then click <strong>Run Commercial Viability Forecast</strong> to evaluate release performance.
        </p>
      </div>
    );
  }

  const isHit = result.is_hit === 1;
  const hitProbPct = (result.hit_probability * 100).toFixed(1);
  const confidencePct = (result.confidence_score * 100).toFixed(1);

  const getRatingBadgeClass = (ratingText) => {
    const r = (ratingText || '').toLowerCase();
    if (r.includes('overwhelmingly positive')) return 'badge-overwhelmingly-positive';
    if (r.includes('very positive') || r === 'positive') return 'badge-very-positive';
    if (r.includes('mostly positive')) return 'badge-mostly-positive';
    if (r.includes('mixed')) return 'badge-mixed';
    return 'badge-negative';
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Primary ML Forecast Card (DESIGN.md ML Hit-Prediction Card) */}
      <div
        className="surface-card"
        style={{
          padding: '1.5rem',
          position: 'relative',
          overflow: 'hidden',
          border: `1px solid ${isHit ? 'var(--status-positive-border)' : 'var(--status-negative-border)'}`,
          boxShadow: `0 8px 30px ${isHit ? 'var(--status-positive-glow)' : 'var(--status-negative-glow)'}`
        }}
      >
        {/* Top Header Zone */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem', marginBottom: '1.25rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
              {isHit ? (
                <div style={{ background: 'var(--status-positive)', color: '#0b1326', borderRadius: 'var(--radius-full)', padding: '3px', display: 'flex' }}>
                  <CheckCircle2 size={20} />
                </div>
              ) : (
                <div style={{ background: 'var(--status-negative)', color: '#fff', borderRadius: 'var(--radius-full)', padding: '3px', display: 'flex' }}>
                  <XCircle size={20} />
                </div>
              )}
              <h2
                style={{
                  fontSize: '1.5rem',
                  fontWeight: 800,
                  letterSpacing: '0.02em',
                  color: isHit ? 'var(--status-positive-bright)' : 'var(--status-negative-bright)'
                }}
              >
                {isHit ? 'COMMERCIAL HIT' : 'MARKET UNDERPERFORMER'}
              </h2>
            </div>
            <p style={{ fontSize: '0.95rem', fontWeight: 600, color: '#fff' }}>
              {result.game_name}
            </p>
          </div>

          <div style={{ textAlign: 'right' }}>
            <div
              className="font-mono"
              style={{
                fontSize: '2.5rem',
                fontWeight: 800,
                color: isHit ? 'var(--status-positive-bright)' : 'var(--status-negative-bright)',
                lineHeight: 1
              }}
            >
              {hitProbPct}%
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--outline)', textTransform: 'uppercase', letterSpacing: '0.04em', fontWeight: 600, marginTop: '3px' }}>
              Hit Probability
            </div>
          </div>
        </div>

        {/* Predictive Gauge Progress Bar */}
        <div style={{ background: 'var(--surface-container-lowest)', borderRadius: 'var(--radius-sm)', height: '8px', overflow: 'hidden', marginBottom: '1.25rem' }}>
          <div
            style={{
              height: '100%',
              width: `${hitProbPct}%`,
              background: isHit
                ? 'linear-gradient(90deg, #10b981 0%, #4ade80 100%)'
                : 'linear-gradient(90deg, #e11d48 0%, #f43f5e 100%)',
              transition: 'width 0.8s ease-out'
            }}
          />
        </div>

        {/* Telemetry Matrix (4 Columns) */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '0.75rem' }}>
          <div style={{ background: 'rgba(6, 14, 32, 0.65)', border: '1px solid rgba(51, 65, 85, 0.5)', padding: '10px 12px', borderRadius: 'var(--radius-sm)' }}>
            <span style={{ fontSize: '0.7rem', color: 'var(--outline)', display: 'block', marginBottom: '4px' }}>
              Projected Community Tier
            </span>
            <span className={`badge-pill ${getRatingBadgeClass(result.projected_rating_tier)}`}>
              <span
                className="pulsating-dot"
                style={{
                  backgroundColor: isHit ? 'var(--status-positive-bright)' : 'var(--status-negative-bright)'
                }}
              />
              <span>{result.projected_rating_tier}</span>
            </span>
          </div>

          <div style={{ background: 'rgba(6, 14, 32, 0.65)', border: '1px solid rgba(51, 65, 85, 0.5)', padding: '10px 12px', borderRadius: 'var(--radius-sm)' }}>
            <span style={{ fontSize: '0.7rem', color: 'var(--outline)', display: 'block', marginBottom: '4px' }}>
              Forecast Confidence
            </span>
            <span className="font-mono" style={{ fontSize: '1rem', fontWeight: 700, color: '#fff' }}>
              {confidencePct}%
            </span>
          </div>

          <div style={{ background: 'rgba(6, 14, 32, 0.65)', border: '1px solid rgba(51, 65, 85, 0.5)', padding: '10px 12px', borderRadius: 'var(--radius-sm)' }}>
            <span style={{ fontSize: '0.7rem', color: 'var(--outline)', display: 'block', marginBottom: '4px' }}>
              Pipeline Latency
            </span>
            <span className="font-mono" style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--primary)' }}>
              {result.inference_duration_ms.toFixed(1)} ms
            </span>
          </div>

          <div style={{ background: 'rgba(6, 14, 32, 0.65)', border: '1px solid rgba(51, 65, 85, 0.5)', padding: '10px 12px', borderRadius: 'var(--radius-sm)' }}>
            <span style={{ fontSize: '0.7rem', color: 'var(--outline)', display: 'block', marginBottom: '4px' }}>
              Market Dimensions
            </span>
            <span className="font-mono" style={{ fontSize: '1rem', fontWeight: 600, color: '#fff' }}>
              {result.features_used_count || 194} Features
            </span>
          </div>
        </div>
      </div>

      {/* Narrative Signals Breakdown (Vibe Scores) */}
      <div className="surface-card" style={{ padding: '1.25rem 1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Sparkles size={16} style={{ color: 'var(--primary)' }} />
            <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#fff' }}>
              Qualitative Narrative Sentiment Signals
            </h3>
          </div>
          <span
            className="badge-pill"
            style={{
              background: result.vibe_scores_source === 'user_override'
                ? 'var(--status-warning-bg)'
                : 'rgba(102, 192, 244, 0.12)',
              color: result.vibe_scores_source === 'user_override'
                ? 'var(--status-warning)'
                : 'var(--primary)',
              border: `1px solid ${result.vibe_scores_source === 'user_override' ? 'var(--status-warning-border)' : 'rgba(102, 192, 244, 0.35)'}`
            }}
          >
            {result.vibe_scores_source === 'user_override' ? 'User Calibrated Signals' : 'Autonomous Pitch Analysis'}
          </span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
          {/* Hype */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '4px' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--on-surface-variant)' }}>
                <Flame size={14} style={{ color: 'var(--primary)' }} /> Hype & Commercial Excitement
              </span>
              <strong className="font-mono" style={{ color: 'var(--primary)' }}>
                {result.vibe_scores.hype_score.toFixed(1)} / 10.0
              </strong>
            </div>
            <div style={{ height: '6px', background: 'var(--surface-container-lowest)', borderRadius: 'var(--radius-sm)', overflow: 'hidden' }}>
              <div
                style={{
                  height: '100%',
                  width: `${(result.vibe_scores.hype_score / 10) * 100}%`,
                  background: 'var(--primary)',
                  borderRadius: 'var(--radius-sm)'
                }}
              />
            </div>
          </div>

          {/* Story Depth */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '4px' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--on-surface-variant)' }}>
                <BookOpen size={14} style={{ color: '#c084fc' }} /> Narrative & World Lore Depth
              </span>
              <strong className="font-mono" style={{ color: '#c084fc' }}>
                {result.vibe_scores.story_score.toFixed(1)} / 10.0
              </strong>
            </div>
            <div style={{ height: '6px', background: 'var(--surface-container-lowest)', borderRadius: 'var(--radius-sm)', overflow: 'hidden' }}>
              <div
                style={{
                  height: '100%',
                  width: `${(result.vibe_scores.story_score / 10) * 100}%`,
                  background: '#a855f7',
                  borderRadius: 'var(--radius-sm)'
                }}
              />
            </div>
          </div>

          {/* Nostalgia */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '4px' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--on-surface-variant)' }}>
                <Clock size={14} style={{ color: '#fbbf24' }} /> Nostalgia & Vintage Heritage
              </span>
              <strong className="font-mono" style={{ color: '#fbbf24' }}>
                {result.vibe_scores.nostalgia_score.toFixed(1)} / 10.0
              </strong>
            </div>
            <div style={{ height: '6px', background: 'var(--surface-container-lowest)', borderRadius: 'var(--radius-sm)', overflow: 'hidden' }}>
              <div
                style={{
                  height: '100%',
                  width: `${(result.vibe_scores.nostalgia_score / 10) * 100}%`,
                  background: '#f59e0b',
                  borderRadius: 'var(--radius-sm)'
                }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Closest Historical Comparables */}
      <div className="surface-card" style={{ padding: '1.25rem 1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '1rem' }}>
          <TrendingUp size={16} style={{ color: 'var(--status-positive-bright)' }} />
          <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#fff' }}>
            Closest Historical Comparables (Nearest Neighbors)
          </h3>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {result.similar_games && result.similar_games.length > 0 ? (
            result.similar_games.map((sg, idx) => (
              <div
                key={sg.appid || idx}
                style={{
                  background: 'rgba(6, 14, 32, 0.7)',
                  border: '1px solid rgba(51, 65, 85, 0.6)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '10px 14px'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '4px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span className="font-mono" style={{ fontSize: '0.7rem', color: 'var(--outline)', background: 'rgba(51, 65, 85, 0.4)', padding: '2px 6px', borderRadius: '2px' }}>
                      #{idx + 1}
                    </span>
                    <h4 style={{ fontSize: '0.9rem', fontWeight: 700, color: '#fff' }}>
                      {sg.name}
                    </h4>
                  </div>
                  <span
                    className={`badge-pill ${sg.is_hit === 1 ? 'badge-overwhelmingly-positive' : 'badge-negative'}`}
                  >
                    {sg.is_hit === 1 ? 'HIT' : 'FLOP'}
                  </span>
                </div>

                <div style={{ display: 'flex', gap: '6px', alignItems: 'center', fontSize: '0.75rem', color: 'var(--outline)', marginBottom: '6px', flexWrap: 'wrap' }}>
                  <span>by <strong style={{ color: 'var(--on-surface-variant)' }}>{sg.developer}</strong></span>
                  <span>•</span>
                  <span className="font-mono" style={{ color: '#fff', fontWeight: 600 }}>
                    {sg.price === 0 ? 'FREE' : `$${sg.price.toFixed(2)}`}
                  </span>
                  <span>•</span>
                  <span className={`badge-pill ${getRatingBadgeClass(sg.rating_text)}`} style={{ padding: '1px 6px' }}>
                    {sg.rating_text}
                  </span>
                  <span className="font-mono">
                    ({((sg.approval_rate || 0) * 100).toFixed(0)}% • {sg.total_reviews?.toLocaleString()} revs)
                  </span>
                </div>

                {sg.short_description && (
                  <p style={{ fontSize: '0.78rem', color: 'var(--on-surface-variant)', lineHeight: 1.4, display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                    {sg.short_description}
                  </p>
                )}
              </div>
            ))
          ) : (
            <p style={{ fontSize: '0.85rem', color: 'var(--outline)' }}>No historical matches found.</p>
          )}
        </div>
      </div>
    </div>
  );
}
