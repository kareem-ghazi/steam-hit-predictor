import React, { useState, useEffect } from 'react';
import { Gamepad2, Database, Sparkles, Activity, ShieldCheck, AlertCircle } from 'lucide-react';
import PredictionStudio from './components/PredictionStudio';
import ResultsPane from './components/ResultsPane';
import DatasetExplorer from './components/DatasetExplorer';

const API_BASE = 'http://127.0.0.1:8000';

export default function App() {
  const [activeTab, setActiveTab] = useState('studio'); // 'studio' | 'explorer'
  const [predictionResult, setPredictionResult] = useState(null);
  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [selectedGameForStudio, setSelectedGameForStudio] = useState(null);

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
          {/* Brand & Title */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
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

          {/* Real-time Status Badge */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <span
              className={`badge-pill ${backendStatus === 'connected' ? 'badge-overwhelmingly-positive' : 'badge-negative'}`}
              style={{ padding: '5px 12px' }}
            >
              <span
                className="pulsating-dot"
                style={{
                  backgroundColor: backendStatus === 'connected' ? 'var(--status-positive-bright)' : 'var(--status-negative-bright)'
                }}
              />
              <span>
                {backendStatus === 'connected'
                  ? `Engine Online (${backendMeta?.games_in_memory?.toLocaleString() || '27,810'} Releases)`
                  : 'Engine Offline'}
              </span>
            </span>
          </div>
        </div>
      </header>

      {/* Main Content Area (1600px max width, 12-column grid layout) */}
      <main style={{ flex: 1, maxWidth: '1600px', width: '100%', margin: '0 auto', padding: '1.75rem 2rem' }}>
        {activeTab === 'studio' ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.25fr) minmax(0, 0.95fr)', gap: '1.75rem', alignItems: 'start' }}>
            <PredictionStudio
              initialGame={selectedGameForStudio}
              onPredictionComplete={(result) => {
                setPredictionResult(result);
              }}
            />
            <ResultsPane
              result={predictionResult}
              loading={loadingPrediction}
            />
          </div>
        ) : (
          <DatasetExplorer
            onSelectGameForStudio={handleGameSelectFromExplorer}
          />
        )}
      </main>

      {/* Footer */}
      <footer
        style={{
          borderTop: '1px solid rgba(51, 65, 85, 0.5)',
          padding: '1.25rem 2rem',
          fontSize: '0.78rem',
          color: 'var(--outline)',
          background: 'var(--surface-container-lowest)'
        }}
      >
        <div style={{ maxWidth: '1600px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
          <div>
            Steam Hit Predictor • Trained on 27,810 verified Steam games with ≥50 reviews
          </div>
          <div className="font-mono" style={{ display: 'flex', gap: '1.25rem' }}>
            <span>27,810 Historical Releases</span>
            <span>•</span>
            <span>194 Market Dimensions</span>
            <span>•</span>
            <span>Autonomous Narrative Sentiment Engine</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
