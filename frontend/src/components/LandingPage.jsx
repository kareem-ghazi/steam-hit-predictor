import React from 'react';
import {
  Sparkles, Layers, TrendingUp, Database, ArrowRight, ShieldCheck,
  CheckCircle2, Compass, Play, FileText, Trophy, BarChart3, Clock
} from 'lucide-react';

export default function LandingPage({ onNavigateToStudio, onNavigateToExplorer }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '3.5rem', padding: '1rem 0 3rem' }}>
      {/* Hero Section */}
      <section style={{ textAlign: 'center', maxWidth: '900px', margin: '0 auto', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <div
          className="badge-pill"
          style={{
            background: 'rgba(102, 192, 244, 0.12)',
            color: 'var(--primary)',
            border: '1px solid rgba(102, 192, 244, 0.35)',
            marginBottom: '1.25rem',
            padding: '5px 14px'
          }}
        >
          <span className="pulsating-dot" style={{ backgroundColor: 'var(--primary)' }} />
          <span>GAMEMETRICS // COMMERCIAL INTELLIGENCE</span>
        </div>

        <h1
          style={{
            fontSize: '2.75rem',
            fontWeight: 800,
            lineHeight: 1.18,
            letterSpacing: '-0.025em',
            color: '#fff',
            marginBottom: '1.25rem'
          }}
        >
          Predict Steam Hit Potential <br />
          <span style={{ background: 'linear-gradient(90deg, #66c0f4 0%, #38bdf8 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            Before Writing Code
          </span>
        </h1>

        <p
          style={{
            fontSize: '1.05rem',
            color: 'var(--on-surface-variant)',
            lineHeight: 1.6,
            maxWidth: '680px',
            marginBottom: '2rem'
          }}
        >
          Empirical commercial viability forecasting powered by <strong>27,810 verified historical releases</strong>, <strong>194 continuous market dimensions</strong>, and automated store pitch narrative analysis.
        </p>

        {/* Action CTAs */}
        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', justifyContent: 'center' }}>
          <button
            onClick={onNavigateToStudio}
            className="btn-primary"
            style={{ padding: '12px 24px', fontSize: '0.95rem' }}
          >
            <span>Launch Developer Studio</span>
            <ArrowRight size={16} />
          </button>

          <button
            onClick={onNavigateToExplorer}
            className="btn-secondary"
            style={{ padding: '12px 24px', fontSize: '0.95rem' }}
          >
            <Database size={16} />
            <span>Explore Dataset Archive</span>
          </button>
        </div>
      </section>

      {/* Live Benchmark Metric Strip (DESIGN.md Telemetry) */}
      <section style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
        <div className="surface-card" style={{ padding: '1.5rem', textAlign: 'center' }}>
          <div className="font-mono" style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--primary)', marginBottom: '4px' }}>
            27,810
          </div>
          <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#fff', marginBottom: '2px' }}>
            Verified Steam Releases
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--outline)' }}>
            Filtered strictly ≥50 community reviews
          </div>
        </div>

        <div className="surface-card" style={{ padding: '1.5rem', textAlign: 'center' }}>
          <div className="font-mono" style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--status-positive-bright)', marginBottom: '4px' }}>
            96.26%
          </div>
          <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#fff', marginBottom: '2px' }}>
            Holdout Test Accuracy
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--outline)' }}>
            96.70% F1-Score on unseen test split
          </div>
        </div>

        <div className="surface-card" style={{ padding: '1.5rem', textAlign: 'center' }}>
          <div className="font-mono" style={{ fontSize: '2rem', fontWeight: 800, color: '#c084fc', marginBottom: '4px' }}>
            194
          </div>
          <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#fff', marginBottom: '2px' }}>
            Market Dimensions
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--outline)' }}>
            Tags, categories, genres & pricing
          </div>
        </div>

        <div className="surface-card" style={{ padding: '1.5rem', textAlign: 'center' }}>
          <div className="font-mono" style={{ fontSize: '2rem', fontWeight: 800, color: '#fbbf24', marginBottom: '4px' }}>
            &lt; 15 ms
          </div>
          <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#fff', marginBottom: '2px' }}>
            Inference Latency
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--outline)' }}>
            Real-time scoring & similarity search
          </div>
        </div>
      </section>

      {/* Feature Architecture Cards */}
      <section style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <div style={{ textAlign: 'center', maxWidth: '600px', margin: '0 auto' }}>
          <h2 style={{ fontSize: '1.75rem', fontWeight: 800, color: '#fff', letterSpacing: '-0.015em', marginBottom: '0.5rem' }}>
            Dual-Phase Predictive Intelligence
          </h2>
          <p style={{ fontSize: '0.9rem', color: 'var(--on-surface-variant)' }}>
            Unifying qualitative pitch narrative depth with quantitative tabular dimensions
          </p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.5rem' }}>
          {/* Card 1: Narrative */}
          <div className="surface-card surface-card-hover" style={{ padding: '1.75rem' }}>
            <div style={{ background: 'rgba(102, 192, 244, 0.15)', color: 'var(--primary)', width: '42px', height: '42px', borderRadius: 'var(--radius-sm)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1.25rem' }}>
              <FileText size={22} />
            </div>
            <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#fff', marginBottom: '0.6rem' }}>
              Game Description Analyzer
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--on-surface-variant)', lineHeight: 1.5, marginBottom: '1rem' }}>
              Evaluates your store pitch copy, visual world setting, and mechanics to extract commercial Hype, Narrative Depth, and Nostalgia appeal.
            </p>
            <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '0.8rem', color: 'var(--outline)' }}>
              <li style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <CheckCircle2 size={14} style={{ color: 'var(--primary)' }} /> Automated pitch copy evaluation
              </li>
              <li style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <CheckCircle2 size={14} style={{ color: 'var(--primary)' }} /> Qualitative vibe resonance scoring
              </li>
            </ul>
          </div>

          {/* Card 2: Viability */}
          <div className="surface-card surface-card-hover" style={{ padding: '1.75rem' }}>
            <div style={{ background: 'rgba(16, 185, 129, 0.15)', color: 'var(--status-positive-bright)', width: '42px', height: '42px', borderRadius: 'var(--radius-sm)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1.25rem' }}>
              <Trophy size={22} />
            </div>
            <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#fff', marginBottom: '0.6rem' }}>
              Commercial Hit Analyzer
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--on-surface-variant)', lineHeight: 1.5, marginBottom: '1rem' }}>
              Synthesizes platform availability, price tiers, multi-label categories, and genre indicators to calculate commercial hit probability.
            </p>
            <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '0.8rem', color: 'var(--outline)' }}>
              <li style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <CheckCircle2 size={14} style={{ color: 'var(--status-positive-bright)' }} /> Commercial Hit vs Flop classification
              </li>
              <li style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <CheckCircle2 size={14} style={{ color: 'var(--status-positive-bright)' }} /> Projected community reception tier
              </li>
            </ul>
          </div>

          {/* Card 3: Comparables */}
          <div className="surface-card surface-card-hover" style={{ padding: '1.75rem' }}>
            <div style={{ background: 'rgba(168, 85, 247, 0.15)', color: '#c084fc', width: '42px', height: '42px', borderRadius: 'var(--radius-sm)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1.25rem' }}>
              <TrendingUp size={22} />
            </div>
            <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#fff', marginBottom: '0.6rem' }}>
              Historical Nearest Neighbors
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--on-surface-variant)', lineHeight: 1.5, marginBottom: '1rem' }}>
              Matches your game concept against 27,810 verified historical games to retrieve the 3 most relevant real-world release benchmarks.
            </p>
            <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '0.8rem', color: 'var(--outline)' }}>
              <li style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <CheckCircle2 size={14} style={{ color: '#c084fc' }} /> Real market performance verification
              </li>
              <li style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <CheckCircle2 size={14} style={{ color: '#c084fc' }} /> Approval ratings & review volume insights
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* Bottom CTA Card */}
      <section
        className="surface-card"
        style={{
          padding: '2.5rem',
          textAlign: 'center',
          background: 'linear-gradient(180deg, #131b2e 0%, #0b1326 100%)',
          border: '1px solid rgba(102, 192, 244, 0.25)',
          borderRadius: 'var(--radius-md)'
        }}
      >
        <h3 style={{ fontSize: '1.5rem', fontWeight: 800, color: '#fff', marginBottom: '0.5rem' }}>
          Ready to Forecast Your Game's Steam Success?
        </h3>
        <p style={{ fontSize: '0.9rem', color: 'var(--on-surface-variant)', maxWidth: '520px', margin: '0 auto 1.5rem' }}>
          Test your game concept, optimize store tags, and benchmark against verified market hits in the Developer Studio.
        </p>
        <button
          onClick={onNavigateToStudio}
          className="btn-primary"
          style={{ padding: '12px 28px', fontSize: '0.95rem' }}
        >
          <span>Open Developer Studio</span>
          <ArrowRight size={16} />
        </button>
      </section>
    </div>
  );
}
