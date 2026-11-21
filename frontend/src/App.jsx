import React, { useState, useEffect } from 'react';
import {
  Gamepad2, Database, Sparkles, Activity, ShieldCheck, AlertCircle,
  Home, Compass, Layers, Github, LogIn, LogOut, User
} from 'lucide-react';
import LandingPage from './components/LandingPage';
import PredictionStudio from './components/PredictionStudio';
import DescriptionAnalyzerPane from './components/DescriptionAnalyzerPane';
import HitAnalyzerPane from './components/HitAnalyzerPane';
import DatasetExplorer from './components/DatasetExplorer';
import LoginPage from './components/LoginPage';

const API_BASE = 'http://127.0.0.1:8000';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview'); // 'overview' | 'studio' | 'explorer' | 'login'
  const [predictionResult, setPredictionResult] = useState(null);
  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [selectedGameForStudio, setSelectedGameForStudio] = useState(null);

  // User Auth State
  const [user, setUser] = useState(null);

  useEffect(() => {
    try {
      const savedUser = localStorage.getItem('steam_hit_user');
      if (savedUser) {
        setUser(JSON.parse(savedUser));
      }
    } catch (e) {
      console.warn('Auth state read note:', e);
    }
  }, []);

  const handleLoginSuccess = (userObj) => {
    setUser(userObj);
    setActiveTab('studio'); // Redirect to studio upon login
  };

  const handleLogout = () => {
    localStorage.removeItem('steam_hit_user');
    setUser(null);
  };

  // Dual-Pane Toggles in Developer Studio
  const [descriptionAnalyzerActive, setDescriptionAnalyzerActive] = useState(true);
  const [hitAnalyzerActive, setHitAnalyzerActive] = useState(true);

  // Backend health state
  const [backendStatus, setBackendStatus] = useState('checking'); // 'connected' | 'disconnected' | 'checking'
  const [backendMeta, setBackendMeta] = useState(null);

  const checkHealth = () => {
    fetch(`${API_BASE}/health`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        setBackendStatus('connected');
        setBackendMeta(data);
      })
      .catch((err) => {
        console.warn('Backend connection note:', err);
        setBackendStatus('disconnected');
      });
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  // Enforce rule: if Hit Analyzer is activated, Game Description Analyzer MUST be activated too!
  const handleToggleDescriptionAnalyzer = (active) => {
    setDescriptionAnalyzerActive(active);
    if (!active && hitAnalyzerActive) {
      setHitAnalyzerActive(false); // Hit analyzer depends on description analyzer
    }
  };

  const handleToggleHitAnalyzer = (active) => {
    setHitAnalyzerActive(active);
    if (active && !descriptionAnalyzerActive) {
      setDescriptionAnalyzerActive(true); // Auto-activate description analyzer
    }
  };

  const handleGameSelectFromExplorer = (game) => {
    setSelectedGameForStudio(game);
    setActiveTab('studio');
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Surface Tier 3: Glass Header (DESIGN.md 64px, sticky top, backdrop blur) */}
      <header
        className="glass-header"
        style={{
          position: 'sticky',
          top: 0,
          zIndex: 50,
          display: 'flex',
          alignItems: 'center',
          padding: '0 2rem'
        }}
      >
        <div style={{ maxWidth: '1600px', width: '100%', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          {/* Brand & Logo */}
          <div
            onClick={() => setActiveTab('overview')}
            style={{ display: 'flex', alignItems: 'center', gap: '0.85rem', cursor: 'pointer' }}
          >
            <div
              style={{
                background: 'linear-gradient(135deg, #1a9fff 0%, #0077cc 100%)',
                color: '#fff',
                padding: '7px',
                borderRadius: 'var(--radius-sm)',
                display: 'flex',
                boxShadow: '0 0 14px var(--primary-glow)'
              }}
            >
              <Gamepad2 size={22} />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span className="font-mono" style={{ fontSize: '0.72rem', color: 'var(--primary)', fontWeight: 700, letterSpacing: '0.05em' }}>
                  GAMEMETRICS
                </span>
                <span style={{ color: 'var(--outline)', fontSize: '0.75rem' }}>//</span>
                <h1 style={{ fontSize: '1.05rem', fontWeight: 800, color: '#fff', letterSpacing: '-0.01em', lineHeight: 1 }}>
                  Steam Hit Predictor
                </h1>
              </div>
              <p style={{ fontSize: '0.72rem', color: 'var(--on-surface-variant)', marginTop: '2px' }}>
                Commercial Viability Intelligence & Market Forecasting for Steam Releases
              </p>
            </div>
          </div>

          {/* Segmented Navigation Tabs */}
          <div className="segmented-nav">
            <button
              onClick={() => setActiveTab('overview')}
              className={`segmented-tab ${activeTab === 'overview' ? 'active' : ''}`}
            >
              <Home size={15} style={{ color: activeTab === 'overview' ? 'var(--primary)' : 'inherit' }} />
              <span>Overview</span>
            </button>

            <button
              onClick={() => setActiveTab('studio')}
              className={`segmented-tab ${activeTab === 'studio' ? 'active' : ''}`}
            >
              <Sparkles size={15} style={{ color: activeTab === 'studio' ? 'var(--primary)' : 'inherit' }} />
              <span>Developer Studio</span>
            </button>

            <button
              onClick={() => setActiveTab('explorer')}
              className={`segmented-tab ${activeTab === 'explorer' ? 'active' : ''}`}
            >
              <Database size={15} style={{ color: activeTab === 'explorer' ? 'var(--primary)' : 'inherit' }} />
              <span>Dataset Explorer</span>
            </button>
          </div>

          {/* User Auth & Engine Status */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            {/* Minimal Engine Status Badge ("Online" or "Offline") */}
            <span
              className={`badge-pill ${backendStatus === 'connected' ? 'badge-overwhelmingly-positive' : 'badge-negative'}`}
              style={{ padding: '4px 12px', fontSize: '0.75rem' }}
            >
              <span
                className="pulsating-dot"
                style={{
                  backgroundColor: backendStatus === 'connected' ? 'var(--status-positive-bright)' : 'var(--status-negative-bright)'
                }}
              />
              <span>{backendStatus === 'connected' ? 'Online' : 'Offline'}</span>
            </span>

            {/* Auth Button or User Badge */}
            {user ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', background: 'var(--surface-container-high)', padding: '3px 10px 3px 5px', borderRadius: 'var(--radius-full)', border: '1px solid rgba(102, 192, 244, 0.3)' }}>
                  <img
                    src={user.avatarUrl}
                    alt={user.displayName}
                    style={{ width: '22px', height: '22px', borderRadius: '50%' }}
                  />
                  <span style={{ fontSize: '0.78rem', fontWeight: 600, color: '#fff' }}>
                    {user.displayName}
                  </span>
                </div>
                <button
                  onClick={handleLogout}
                  className="btn-secondary"
                  style={{ padding: '4px 8px', fontSize: '0.75rem' }}
                  title="Sign Out"
                >
                  <LogOut size={13} />
                </button>
              </div>
            ) : (
              <button
                onClick={() => setActiveTab('login')}
                className="btn-secondary"
                style={{ padding: '5px 12px', fontSize: '0.78rem' }}
              >
                <LogIn size={13} />
                <span>Sign In</span>
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main style={{ flex: 1, maxWidth: '1600px', width: '100%', margin: '0 auto', padding: '1.75rem 2rem' }}>
        {activeTab === 'overview' && (
          <LandingPage
            onNavigateToStudio={() => setActiveTab('studio')}
            onNavigateToExplorer={() => setActiveTab('explorer')}
          />
        )}

        {activeTab === 'login' && (
          <LoginPage
            onLogin={handleLoginSuccess}
            onCancel={() => setActiveTab('studio')}
          />
        )}

        {activeTab === 'studio' && (
          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.25fr) minmax(0, 0.95fr)', gap: '1.75rem', alignItems: 'start' }}>
            {/* Left Column: Developer Studio Form */}
            <PredictionStudio
              initialGame={selectedGameForStudio}
              currentUser={user}
              onPredictionComplete={(result) => {
                setPredictionResult(result);
              }}
              descriptionAnalyzerActive={descriptionAnalyzerActive}
              hitAnalyzerActive={hitAnalyzerActive}
            />

            {/* Right Column: TWO Panes Stacked on Top of Each Other */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
              {/* Top Pane: Game Description Analyzer */}
              <DescriptionAnalyzerPane
                active={descriptionAnalyzerActive}
                onToggleActive={handleToggleDescriptionAnalyzer}
                result={predictionResult}
                loading={loadingPrediction}
              />

              {/* Bottom Pane: Commercial Hit Analyzer */}
              <HitAnalyzerPane
                active={hitAnalyzerActive}
                onToggleActive={handleToggleHitAnalyzer}
                result={predictionResult}
                loading={loadingPrediction}
              />
            </div>
          </div>
        )}

        {activeTab === 'explorer' && (
          <DatasetExplorer
            onSelectGameForStudio={handleGameSelectFromExplorer}
          />
        )}
      </main>

      {/* Professional SaaS Footer */}
      <footer
        style={{
          borderTop: '1px solid rgba(51, 65, 85, 0.5)',
          padding: '2.5rem 2rem 1.75rem',
          background: 'var(--surface-container-lowest)'
        }}
      >
        <div style={{ maxWidth: '1600px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          {/* Top Footer Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '2rem' }}>
            {/* Column 1: Brand & Identity */}
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '0.75rem' }}>
                <div style={{ background: 'var(--primary)', color: '#0b1326', padding: '4px', borderRadius: 'var(--radius-sm)', display: 'flex' }}>
                  <Gamepad2 size={16} />
                </div>
                <span className="font-mono" style={{ fontSize: '0.85rem', fontWeight: 800, color: '#fff', letterSpacing: '0.04em' }}>
                  GAMEMETRICS // STEAM HIT PREDICTOR
                </span>
              </div>
              <p style={{ fontSize: '0.8rem', color: 'var(--outline)', lineHeight: 1.5, maxWidth: '340px' }}>
                Commercial viability forecasting and video game market intelligence, transforming historical Steam telemetry into predictive success models.
              </p>
            </div>

            {/* Column 2: Dataset Statistics */}
            <div>
              <h4 style={{ fontSize: '0.85rem', fontWeight: 700, color: '#fff', marginBottom: '0.75rem' }}>
                Empirical Archive
              </h4>
              <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '0.78rem', color: 'var(--outline)' }}>
                <li>• 27,810 Verified Steam Releases (≥50 Reviews)</li>
                <li>• 194 Multi-label, Categorical & Scaled Dimensions</li>
                <li>• Stratified 80/10/10 Train, Validation, and Test Partition</li>
                <li>• Harmonized Multi-Source Metadata & Pricing</li>
              </ul>
            </div>

            {/* Column 3: Platform Capabilities */}
            <div>
              <h4 style={{ fontSize: '0.85rem', fontWeight: 700, color: '#fff', marginBottom: '0.75rem' }}>
                Predictive Architecture
              </h4>
              <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '0.78rem', color: 'var(--outline)' }}>
                <li>• Autonomous Pitch Narrative Sentiment Engine</li>
                <li>• Commercial Hit vs Underperformer Classification</li>
                <li>• SQLite Private Projects Persistence</li>
                <li>• Nearest Neighbor Historical Comparables Engine</li>
              </ul>
            </div>

            {/* Column 4: Quick Navigation */}
            <div>
              <h4 style={{ fontSize: '0.85rem', fontWeight: 700, color: '#fff', marginBottom: '0.75rem' }}>
                Platform Navigation
              </h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '0.78rem' }}>
                <span
                  onClick={() => setActiveTab('overview')}
                  style={{ color: 'var(--primary)', cursor: 'pointer', display: 'inline-block' }}
                >
                  → Platform Overview & Benchmarks
                </span>
                <span
                  onClick={() => setActiveTab('studio')}
                  style={{ color: 'var(--primary)', cursor: 'pointer', display: 'inline-block' }}
                >
                  → Developer Prediction Studio
                </span>
                <span
                  onClick={() => setActiveTab('explorer')}
                  style={{ color: 'var(--primary)', cursor: 'pointer', display: 'inline-block' }}
                >
                  → Explore 27,810 Games Archive
                </span>
              </div>
            </div>
          </div>

          {/* Bottom Copyright Strip */}
          <div style={{ borderTop: '1px solid rgba(51, 65, 85, 0.4)', paddingTop: '1.25rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem', fontSize: '0.75rem', color: 'var(--outline)' }}>
            <div>
              © 2026 GameMetrics • Steam Hit Predictor Platform. Built for game developers, analysts & publishers.
            </div>
            <div className="font-mono" style={{ display: 'flex', gap: '1rem' }}>
              <span>Engine Status: {backendStatus === 'connected' ? 'ONLINE (0.1096 Logloss)' : 'OFFLINE'}</span>
              <span>•</span>
              <span>FastAPI & SQLite & Vite</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
