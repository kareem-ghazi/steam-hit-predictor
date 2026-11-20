import React from 'react';
import { Sparkles, Trophy, AlertTriangle, Cpu, CheckCircle2, XCircle, TrendingUp, BookOpen, Clock, Heart, Flame } from 'lucide-react';

export default function ResultsPane({ result, loading }) {
  if (loading) {
    return (
      <div className="panel-card" style={{ padding: '3rem 2rem', textAlign: 'center', minHeight: '400px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ position: 'relative', width: '70px', height: '70px', marginBottom: '1.5rem' }}>
          <div
            style={{
              position: 'absolute', inset: 0, borderRadius: '50%',
              border: '3px solid rgba(102, 192, 244, 0.2)',
              borderTopColor: 'var(--steam-blue)',
              animation: 'spin 1s linear infinite'
            }}
          />
          <Cpu size={30} style={{ position: 'absolute', top: '20px', left: '20px', color: 'var(--steam-blue)' }} />
        </div>
        <h3 style={{ fontSize: '1.3rem', fontWeight: 700, color: '#fff', marginBottom: '0.5rem' }}>
          Simulating Steam Market Launch...
        </h3>
        <p style={{ color: 'var(--text-secondary)', maxWidth: '420px', fontSize: '0.9rem', lineHeight: 1.5 }}>
          Running QLoRA LLM vibe extraction, 194-feature MinMax & one-hot transformations, and XGBoost gradient-boosted inference...
        </p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="panel-card" style={{ padding: '3rem 2rem', textAlign: 'center', minHeight: '400px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ background: 'rgba(255, 255, 255, 0.04)', borderRadius: '50%', padding: '24px', marginBottom: '1.5rem' }}>
          <Sparkles size={40} style={{ color: 'var(--steam-blue)' }} />
        </div>
        <h3 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#fff', marginBottom: '0.5rem' }}>
          Awaiting Game Concept
        </h3>
        <p style={{ color: 'var(--text-secondary)', maxWidth: '380px', fontSize: '0.875rem', lineHeight: 1.5 }}>
          Define your game parameters on the left or load a preset, then click <strong>Analyze & Predict</strong> to run the dual ML+LLM evaluation.
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
    return 'badge-mostly-negative';
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Hit / Flop Hero Banner */}
      <div
        className="panel-card"
        style={{
          padding: '1.75rem',
          position: 'relative',
          overflow: 'hidden',
          border: `1px solid ${isHit ? 'var(--hit-green)' : 'var(--flop-red)'}`,
          boxShadow: `0 0 35px ${isHit ? 'var(--hit-green-glow)' : 'var(--flop-red-glow)'}`
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.4rem' }}>
              {isHit ? (
                <div style={{ background: 'var(--hit-green)', color: '#0a0e17', borderRadius: '50%', padding: '4px', display: 'flex' }}>
                  <CheckCircle2 size={24} />
                </div>
              ) : (
                <div style={{ background: 'var(--flop-red)', color: '#fff', borderRadius: '50%', padding: '4px', display: 'flex' }}>
                  <XCircle size={24} />
                </div>
              )}
              <h2 style={{ fontSize: '1.75rem', fontWeight: 800, letterSpacing: '0.02em', color: isHit ? 'var(--hit-green)' : 'var(--flop-red)' }}>
                {isHit ? 'COMMERCIAL HIT' : 'MARKET FLOP'}
              </h2>
            </div>
            <p style={{ fontSize: '1rem', fontWeight: 600, color: '#fff' }}>
              {result.game_name}
            </p>
          </div>

          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '2.25rem', fontWeight: 800, color: isHit ? 'var(--hit-green)' : 'var(--flop-red)', lineHeight: 1 }}>
              {hitProbPct}%
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Hit Probability
            </div>
          </div>
        </div>

        {/* Probability bar */}
        <div style={{ margin: '1.25rem 0 1rem', background: 'rgba(0, 0, 0, 0.4)', borderRadius: '6px', height: '10px', overflow: 'hidden' }}>
          <div
            style={{
              height: '100%',
              width: `${hitProbPct}%`,
              background: isHit
                ? 'linear-gradient(90deg, #059669 0%, #10b981 100%)'
                : 'linear-gradient(90deg, #dc2626 0%, #ef4444 100%)',
              transition: 'width 0.8s ease-out'
            }}
          />
        </div>

        {/* Status Indicators */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '0.75rem', marginTop: '1rem' }}>
          <div style={{ background: 'rgba(0, 0, 0, 0.25)', padding: '8px 12px', borderRadius: '6px' }}>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'block' }}>Projected Rating</span>
            <span className={getRatingBadgeClass(result.projected_rating_tier)} style={{ fontSize: '0.8rem', fontWeight: 700, padding: '2px 6px', borderRadius: '4px', display: 'inline-block', marginTop: '2px' }}>
              {result.projected_rating_tier}
            </span>
          </div>

          <div style={{ background: 'rgba(0, 0, 0, 0.25)', padding: '8px 12px', borderRadius: '6px' }}>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'block' }}>Model Confidence</span>
            <span style={{ fontSize: '0.9rem', fontWeight: 700, color: '#fff' }}>{confidencePct}%</span>
          </div>

          <div style={{ background: 'rgba(0, 0, 0, 0.25)', padding: '8px 12px', borderRadius: '6px' }}>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'block' }}>Inference Latency</span>
            <span style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--steam-blue)' }}>{result.inference_duration_ms.toFixed(1)} ms</span>
          </div>
        </div>
      </div>

      {/* LLM Vibe Scores Breakdown */}
      <div className="panel-card" style={{ padding: '1.25rem 1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Sparkles size={18} style={{ color: 'var(--steam-blue)' }} />
            <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#fff' }}>Extracted LLM Vibe Scores</h3>
          </div>
          <span
            style={{
              fontSize: '0.7rem', padding: '2px 8px', borderRadius: '4px', fontWeight: 600,
              background: result.vibe_scores_source === 'qlora_adapter' ? 'rgba(102, 192, 244, 0.15)' : 'rgba(245, 158, 11, 0.15)',
              color: result.vibe_scores_source === 'qlora_adapter' ? 'var(--steam-blue)' : '#f59e0b',
              border: `1px solid ${result.vibe_scores_source === 'qlora_adapter' ? 'rgba(102, 192, 244, 0.3)' : 'rgba(245, 158, 11, 0.3)'}`
            }}
          >
            {result.vibe_scores_source === 'qlora_adapter' ? 'QLoRA Qwen3.5-0.8B' : result.vibe_scores_source}
          </span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
          {/* Hype Score */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '5px', color: 'var(--text-secondary)' }}>
                <Flame size={14} style={{ color: 'var(--steam-blue)' }} /> Hype & Excitement
              </span>
              <strong style={{ color: 'var(--steam-blue)' }}>{result.vibe_scores.hype_score.toFixed(1)} / 10.0</strong>
            </div>
            <div style={{ height: '8px', background: 'rgba(0,0,0,0.4)', borderRadius: '4px', overflow: 'hidden' }}>
              <div
                style={{
                  height: '100%',
                  width: `${(result.vibe_scores.hype_score / 10) * 100}%`,
                  background: 'var(--steam-blue)',
                  borderRadius: '4px'
                }}
              />
            </div>
          </div>

          {/* Story Score */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '5px', color: 'var(--text-secondary)' }}>
                <BookOpen size={14} style={{ color: '#c084fc' }} /> Narrative & Story Depth
              </span>
              <strong style={{ color: '#c084fc' }}>{result.vibe_scores.story_score.toFixed(1)} / 10.0</strong>
            </div>
            <div style={{ height: '8px', background: 'rgba(0,0,0,0.4)', borderRadius: '4px', overflow: 'hidden' }}>
              <div
                style={{
                  height: '100%',
                  width: `${(result.vibe_scores.story_score / 10) * 100}%`,
                  background: '#a855f7',
                  borderRadius: '4px'
                }}
              />
            </div>
          </div>

          {/* Nostalgia Score */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '5px', color: 'var(--text-secondary)' }}>
                <Clock size={14} style={{ color: '#fbbf24' }} /> Nostalgia & Retro Feel
              </span>
              <strong style={{ color: '#fbbf24' }}>{result.vibe_scores.nostalgia_score.toFixed(1)} / 10.0</strong>
            </div>
            <div style={{ height: '8px', background: 'rgba(0,0,0,0.4)', borderRadius: '4px', overflow: 'hidden' }}>
              <div
                style={{
                  height: '100%',
                  width: `${(result.vibe_scores.nostalgia_score / 10) * 100}%`,
                  background: '#f59e0b',
                  borderRadius: '4px'
                }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* 3 Similar Historical Games */}
      <div className="panel-card" style={{ padding: '1.25rem 1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
          <TrendingUp size={18} style={{ color: 'var(--hit-green)' }} />
          <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#fff' }}>
            Closest Historical Comparables (Nearest Neighbors)
          </h3>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
          {result.similar_games && result.similar_games.length > 0 ? (
            result.similar_games.map((sg, idx) => (
              <div
                key={sg.appid || idx}
                style={{
                  background: 'rgba(11, 16, 26, 0.7)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: '8px',
                  padding: '12px 14px'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '4px' }}>
                  <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#fff' }}>
                    #{idx + 1} {sg.name}
                  </h4>
                  <span
                    style={{
                      fontSize: '0.7rem', fontWeight: 700, padding: '2px 6px', borderRadius: '4px',
                      background: sg.is_hit === 1 ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                      color: sg.is_hit === 1 ? 'var(--hit-green)' : 'var(--flop-red)',
                      border: `1px solid ${sg.is_hit === 1 ? 'var(--hit-green)' : 'var(--flop-red)'}`
                    }}
                  >
                    {sg.is_hit === 1 ? 'HIT' : 'FLOP'}
                  </span>
                </div>

                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '6px', flexWrap: 'wrap' }}>
                  <span>by {sg.developer}</span>
                  <span>•</span>
                  <span>${sg.price ? sg.price.toFixed(2) : '0.00'}</span>
                  <span>•</span>
                  <span className={getRatingBadgeClass(sg.rating_text)} style={{ padding: '1px 5px', borderRadius: '3px', fontWeight: 600 }}>
                    {sg.rating_text}
                  </span>
                  <span>({((sg.approval_rate || 0) * 100).toFixed(0)}% of {sg.total_reviews?.toLocaleString()} reviews)</span>
                </div>

                {sg.short_description && (
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.4, display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                    {sg.short_description}
                  </p>
                )}
              </div>
            ))
          ) : (
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>No historical matches found.</p>
          )}
        </div>
      </div>
    </div>
  );
}
