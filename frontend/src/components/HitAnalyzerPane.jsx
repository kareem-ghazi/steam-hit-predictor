import React from 'react';
import { Trophy, CheckCircle2, XCircle, TrendingUp, Cpu, BarChart3, ShieldAlert } from 'lucide-react';

export default function HitAnalyzerPane({
  active,
  onToggleActive,
  result,
  loading
}) {
  const isHit = result?.is_hit === 1;
  const hitProbPct = result ? (result.hit_probability * 100).toFixed(1) : '0.0';
  const confidencePct = result ? (result.confidence_score * 100).toFixed(1) : '0.0';

  const getRatingBadgeClass = (ratingText) => {
    const r = (ratingText || '').toLowerCase();
    if (r.includes('overwhelmingly positive')) return 'badge-overwhelmingly-positive';
    if (r.includes('very positive') || r === 'positive') return 'badge-very-positive';
    if (r.includes('mostly positive')) return 'badge-mostly-positive';
    if (r.includes('mixed')) return 'badge-mixed';
    return 'badge-negative';
  };

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
          <div style={{ background: active ? 'rgba(16, 185, 129, 0.15)' : 'rgba(51, 65, 85, 0.3)', color: active ? 'var(--status-positive-bright)' : 'var(--outline)', padding: '5px', borderRadius: 'var(--radius-sm)' }}>
            <Trophy size={18} />
          </div>
          <div>
            <h3 style={{ fontSize: '0.975rem', fontWeight: 700, color: active ? '#fff' : 'var(--outline)' }}>
              Commercial Hit Analyzer
            </h3>
            <span style={{ fontSize: '0.72rem', color: 'var(--outline)' }}>
              Viability probability & historical benchmarks (requires Description Analyzer)
            </span>
          </div>
        </div>

        {/* Toggle Switch */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: active ? 'var(--status-positive-bright)' : 'var(--outline)' }}>
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
            Hit Analyzer is turned off. Toggle to <strong>ACTIVE</strong> to calculate commercial hit probability and compare against 27,810 historical Steam games.
          </p>
        </div>
      )}

      {/* Active State */}
      {active && (
        <>
          {loading ? (
            <div style={{ padding: '2.5rem 1rem', textAlign: 'center' }}>
              <div
                style={{
                  width: '28px', height: '28px', borderRadius: '50%',
                  border: '2px solid rgba(16, 185, 129, 0.2)',
                  borderTopColor: 'var(--status-positive)',
                  animation: 'spin 1s linear infinite',
                  margin: '0 auto 0.75rem'
                }}
              />
              <p style={{ fontSize: '0.85rem', color: 'var(--on-surface-variant)' }}>
                Forecasting market hit potential across 194 continuous dimensions...
              </p>
            </div>
          ) : result ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {/* Hit / Flop Hero Banner */}
              <div
                style={{
                  background: isHit ? 'rgba(16, 185, 129, 0.08)' : 'rgba(244, 63, 94, 0.08)',
                  border: `1px solid ${isHit ? 'var(--status-positive-border)' : 'var(--status-negative-border)'}`,
                  borderRadius: 'var(--radius-sm)',
                  padding: '1rem',
                  boxShadow: `0 4px 15px ${isHit ? 'var(--status-positive-glow)' : 'var(--status-negative-glow)'}`
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    {isHit ? (
                      <div style={{ background: 'var(--status-positive)', color: '#0b1326', borderRadius: 'var(--radius-full)', padding: '2px', display: 'flex' }}>
                        <CheckCircle2 size={18} />
                      </div>
                    ) : (
                      <div style={{ background: 'var(--status-negative)', color: '#fff', borderRadius: 'var(--radius-full)', padding: '2px', display: 'flex' }}>
                        <XCircle size={18} />
                      </div>
                    )}
                    <span
                      style={{
                        fontSize: '1.1rem',
                        fontWeight: 800,
                        letterSpacing: '0.02em',
                        color: isHit ? 'var(--status-positive-bright)' : 'var(--status-negative-bright)'
                      }}
                    >
                      {isHit ? 'COMMERCIAL HIT' : 'MARKET UNDERPERFORMER'}
                    </span>
                  </div>

                  <div className="font-mono" style={{ fontSize: '1.75rem', fontWeight: 800, color: isHit ? 'var(--status-positive-bright)' : 'var(--status-negative-bright)' }}>
                    {hitProbPct}%
                  </div>
                </div>

                {/* Progress bar */}
                <div style={{ background: '#101726', height: '6px', borderRadius: 'var(--radius-sm)', overflow: 'hidden', border: '1px solid #334155' }}>
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
              </div>

              {/* 3-Column Telemetry Matrix */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.5rem' }}>
                <div style={{ background: 'rgba(6, 14, 32, 0.65)', border: '1px solid rgba(51, 65, 85, 0.5)', padding: '8px 10px', borderRadius: 'var(--radius-sm)' }}>
                  <span style={{ fontSize: '0.68rem', color: 'var(--outline)', display: 'block', marginBottom: '3px' }}>
                    Projected Tier
                  </span>
                  <span className={`badge-pill ${getRatingBadgeClass(result.projected_rating_tier)}`} style={{ padding: '2px 6px', fontSize: '0.7rem' }}>
                    {result.projected_rating_tier}
                  </span>
                </div>

                <div style={{ background: 'rgba(6, 14, 32, 0.65)', border: '1px solid rgba(51, 65, 85, 0.5)', padding: '8px 10px', borderRadius: 'var(--radius-sm)' }}>
                  <span style={{ fontSize: '0.68rem', color: 'var(--outline)', display: 'block', marginBottom: '3px' }}>
                    Confidence
                  </span>
                  <span className="font-mono" style={{ fontSize: '0.85rem', fontWeight: 700, color: '#fff' }}>
                    {confidencePct}%
                  </span>
                </div>

                <div style={{ background: 'rgba(6, 14, 32, 0.65)', border: '1px solid rgba(51, 65, 85, 0.5)', padding: '8px 10px', borderRadius: 'var(--radius-sm)' }}>
                  <span style={{ fontSize: '0.68rem', color: 'var(--outline)', display: 'block', marginBottom: '3px' }}>
                    Latency
                  </span>
                  <span className="font-mono" style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--primary)' }}>
                    {result.inference_duration_ms.toFixed(1)} ms
                  </span>
                </div>
              </div>

              {/* Historical Comparables */}
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
                  <TrendingUp size={14} style={{ color: 'var(--status-positive-bright)' }} />
                  <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#fff' }}>
                    Historical Comparables
                  </span>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  {result.similar_games?.map((sg, idx) => (
                    <div
                      key={sg.appid || idx}
                      style={{
                        background: 'rgba(6, 14, 32, 0.65)',
                        border: '1px solid rgba(51, 65, 85, 0.5)',
                        borderRadius: 'var(--radius-sm)',
                        padding: '8px 10px'
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2px' }}>
                        <span style={{ fontSize: '0.825rem', fontWeight: 700, color: '#fff' }}>
                          #{idx + 1} {sg.name}
                        </span>
                        <span className={`badge-pill ${sg.is_hit === 1 ? 'badge-overwhelmingly-positive' : 'badge-negative'}`} style={{ fontSize: '0.65rem', padding: '1px 6px' }}>
                          {sg.is_hit === 1 ? 'HIT' : 'FLOP'}
                        </span>
                      </div>
                      <div style={{ display: 'flex', gap: '6px', fontSize: '0.72rem', color: 'var(--outline)', flexWrap: 'wrap' }}>
                        <span>by {sg.developer}</span>
                        <span>•</span>
                        <span className="font-mono" style={{ color: '#fff' }}>${sg.price ? sg.price.toFixed(2) : '0.00'}</span>
                        <span>•</span>
                        <span className="font-mono">{((sg.approval_rate || 0) * 100).toFixed(0)}% Approval</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div style={{ padding: '1.5rem 1rem', textAlign: 'center', color: 'var(--outline)' }}>
              <BarChart3 size={28} style={{ color: 'var(--status-positive)', margin: '0 auto 0.5rem', opacity: 0.8 }} />
              <p style={{ fontSize: '0.85rem', color: 'var(--on-surface-variant)', marginBottom: '0.75rem' }}>
                Awaiting Forecast Request
              </p>
              <p style={{ fontSize: '0.78rem', color: 'var(--outline)' }}>
                Configure parameters and click <strong>Predict</strong> to project hit viability and historical matches.
              </p>
            </div>
          )}
        </>
      )}
    </div>
  );
}
