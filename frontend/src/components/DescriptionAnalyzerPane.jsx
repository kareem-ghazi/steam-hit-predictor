import React from 'react';
import { Sparkles, Flame, BookOpen, Clock, FileText, CheckCircle2, AlertCircle } from 'lucide-react';

export default function DescriptionAnalyzerPane({
  active,
  onToggleActive,
  result,
  loading,
  onPredict
}) {
  return (
    <div
      className="surface-card"
      style={{
        padding: '1.25rem 1.5rem',
        opacity: active ? 1 : 0.65,
        transition: 'all 0.2s ease',
        border: active ? '1px solid var(--outline-variant)' : '1px dashed rgba(51, 65, 85, 0.4)'
      }}
    >
      {/* Pane Header with Toggle */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', paddingBottom: '0.75rem', borderBottom: '1px solid rgba(51, 65, 85, 0.4)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ background: active ? 'rgba(102, 192, 244, 0.15)' : 'rgba(51, 65, 85, 0.3)', color: active ? 'var(--primary)' : 'var(--outline)', padding: '5px', borderRadius: 'var(--radius-sm)' }}>
            <FileText size={18} />
          </div>
          <div>
            <h3 style={{ fontSize: '0.975rem', fontWeight: 700, color: active ? '#fff' : 'var(--outline)' }}>
              Game Description Analyzer
            </h3>
            <span style={{ fontSize: '0.72rem', color: 'var(--outline)' }}>
              Narrative sentiment & pitch resonance signals
            </span>
          </div>
        </div>

        {/* Toggle Switch */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: active ? 'var(--primary)' : 'var(--outline)' }}>
            {active ? 'ACTIVE' : 'OFF'}
          </span>
          <label className="switch-toggle">
            <input
              type="checkbox"
              checked={active}
              onChange={(e) => onToggleActive(e.target.checked)}
            />
            <span className="switch-slider" />
          </label>
        </div>
      </div>

      {/* Inactive State */}
      {!active && (
        <div style={{ padding: '1.5rem 1rem', textAlign: 'center', color: 'var(--outline)' }}>
          <p style={{ fontSize: '0.85rem' }}>
            Description Analyzer is turned off. Toggle to <strong>ACTIVE</strong> to evaluate narrative momentum and qualitative pitch signals.
          </p>
        </div>
      )}

      {/* Active State */}
      {active && (
        <>
          {loading ? (
            <div style={{ padding: '2rem 1rem', textAlign: 'center' }}>
              <div
                style={{
                  width: '24px', height: '24px', borderRadius: '50%',
                  border: '2px solid rgba(102, 192, 244, 0.2)',
                  borderTopColor: 'var(--primary)',
                  animation: 'spin 1s linear infinite',
                  margin: '0 auto 0.75rem'
                }}
              />
              <p style={{ fontSize: '0.825rem', color: 'var(--on-surface-variant)' }}>
                Analyzing store pitch text and extracting qualitative narrative signals...
              </p>
            </div>
          ) : result?.vibe_scores ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
              {/* Archetype Highlight */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(6, 14, 32, 0.6)', padding: '6px 12px', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(51, 65, 85, 0.4)' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--on-surface-variant)' }}>
                  Narrative Profile:
                </span>
                <span className="font-mono" style={{ fontSize: '0.75rem', color: 'var(--primary)', fontWeight: 600 }}>
                  {result.vibe_scores.hype_score >= 8.0 ? 'High-Momentum Commercial Concept' :
                   result.vibe_scores.story_score >= 8.0 ? 'Lore-Rich Narrative Universe' :
                   result.vibe_scores.nostalgia_score >= 8.0 ? 'Classic Retro Nostalgia' : 'Balanced Gameplay Focus'}
                </span>
              </div>

              {/* Hype */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '4px' }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--on-surface-variant)' }}>
                    <Flame size={14} style={{ color: 'var(--primary)' }} /> Hype & Excitement
                  </span>
                  <strong className="font-mono" style={{ color: 'var(--primary)', fontSize: '0.825rem' }}>
                    {result.vibe_scores.hype_score.toFixed(1)} <span style={{ color: 'var(--outline)', fontWeight: 400 }}>/ 10.0</span>
                  </strong>
                </div>
                <div style={{ height: '7px', background: '#101726', border: '1px solid #334155', borderRadius: 'var(--radius-sm)', overflow: 'hidden' }}>
                  <div
                    style={{
                      height: '100%',
                      width: `${(result.vibe_scores.hype_score / 10) * 100}%`,
                      background: 'linear-gradient(90deg, #38bdf8 0%, #66c0f4 100%)',
                      borderRadius: 'var(--radius-sm)',
                      boxShadow: '0 0 8px rgba(102, 192, 244, 0.4)'
                    }}
                  />
                </div>
              </div>

              {/* Story */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '4px' }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--on-surface-variant)' }}>
                    <BookOpen size={14} style={{ color: '#c084fc' }} /> Narrative & Lore Depth
                  </span>
                  <strong className="font-mono" style={{ color: '#c084fc', fontSize: '0.825rem' }}>
                    {result.vibe_scores.story_score.toFixed(1)} <span style={{ color: 'var(--outline)', fontWeight: 400 }}>/ 10.0</span>
                  </strong>
                </div>
                <div style={{ height: '7px', background: '#101726', border: '1px solid #334155', borderRadius: 'var(--radius-sm)', overflow: 'hidden' }}>
                  <div
                    style={{
                      height: '100%',
                      width: `${(result.vibe_scores.story_score / 10) * 100}%`,
                      background: 'linear-gradient(90deg, #9333ea 0%, #c084fc 100%)',
                      borderRadius: 'var(--radius-sm)',
                      boxShadow: '0 0 8px rgba(192, 132, 252, 0.4)'
                    }}
                  />
                </div>
              </div>

              {/* Nostalgia */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '4px' }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--on-surface-variant)' }}>
                    <Clock size={14} style={{ color: '#fbbf24' }} /> Nostalgia & Vintage Appeal
                  </span>
                  <strong className="font-mono" style={{ color: '#fbbf24', fontSize: '0.825rem' }}>
                    {result.vibe_scores.nostalgia_score.toFixed(1)} <span style={{ color: 'var(--outline)', fontWeight: 400 }}>/ 10.0</span>
                  </strong>
                </div>
                <div style={{ height: '7px', background: '#101726', border: '1px solid #334155', borderRadius: 'var(--radius-sm)', overflow: 'hidden' }}>
                  <div
                    style={{
                      height: '100%',
                      width: `${(result.vibe_scores.nostalgia_score / 10) * 100}%`,
                      background: 'linear-gradient(90deg, #d97706 0%, #fbbf24 100%)',
                      borderRadius: 'var(--radius-sm)',
                      boxShadow: '0 0 8px rgba(251, 191, 36, 0.4)'
                    }}
                  />
                </div>
              </div>
            </div>
          ) : (
            <div style={{ padding: '1.5rem 1rem', textAlign: 'center', color: 'var(--outline)' }}>
              <Sparkles size={28} style={{ color: 'var(--primary)', margin: '0 auto 0.5rem', opacity: 0.8 }} />
              <p style={{ fontSize: '0.85rem', color: 'var(--on-surface-variant)', marginBottom: '0.75rem' }}>
                Awaiting Pitch Input
              </p>
              <p style={{ fontSize: '0.78rem', color: 'var(--outline)' }}>
                Fill in your game description on the left and click <strong>Predict</strong> to evaluate pitch signals.
              </p>
            </div>
          )}
        </>
      )}
    </div>
  );
}
