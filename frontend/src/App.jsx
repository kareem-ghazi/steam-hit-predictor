import React, { useState, useEffect } from 'react';
import { Gamepad2, Database, Sparkles, Activity, ShieldCheck, AlertCircle, ExternalLink, Cpu } from 'lucide-react';
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
      {/* Top Navigation Bar */}
      <header
        style={{
          background: 'linear-gradient(180deg, #171d25 0%, #101822 100%)',
          borderBottom: '1px solid var(--border-subtle)',
          padding: '0.75rem 2rem',
          position: 'sticky',
          top: 0,
          zIndex: 50,
          backdropFilter: 'blur(10px)'
        }}
      >
        <div style={{ maxWidth: '1440px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          {/* Logo & Title */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div
              style={{
                background: 'linear-gradient(135deg, #1a9fff 0%, #0077cc 100%)',
                color: '#fff',
                padding: '8px',
                borderRadius: '10px',
                display: 'flex',
                boxShadow: '0 0 15px var(--steam-blue-glow)'
              }}
            >
              <Gamepad2 size={24} />
            </div>
            <div>
              <h1 style={{ fontSize: '1.25rem', fontWeight: 800, color: '#fff', letterSpacing: '-0.01em', lineHeight: 1.2 }}>
                Steam Hit Predictor
              </h1>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Hybrid ML (XGBoost 96.3% Acc) + LLM (Fine-Tuned QLoRA Qwen3.5-0.8B)
              </p>
            </div>
          </div>

          {/* Tab Navigation */}
          <div style={{ display: 'flex', background: 'var(--bg-secondary)', padding: '4px', borderRadius: '10px', border: '1px solid var(--border-subtle)' }}>
            <button
              onClick={() => setActiveTab('studio')}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '8px 16px',
                borderRadius: '8px',
                fontSize: '0.875rem',
                fontWeight: 600,
                cursor: 'pointer',
                border: 'none',
                background: activeTab === 'studio' ? 'var(--steam-accent)' : 'transparent',
                color: activeTab === 'studio' ? '#fff' : 'var(--text-secondary)',
                boxShadow: activeTab === 'studio' ? '0 2px 8px rgba(0,0,0,0.3)' : 'none',
                transition: 'all 0.2s ease'
              }}
            >
              <Sparkles size={16} style={{ color: activeTab === 'studio' ? 'var(--steam-blue)' : 'inherit' }} />
              <span>Developer Studio</span>
            </button>

            <button
              onClick={() => setActiveTab('explorer')}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '8px 16px',
                borderRadius: '8px',
                fontSize: '0.875rem',
                fontWeight: 600,
                cursor: 'pointer',
                border: 'none',
                background: activeTab === 'explorer' ? 'var(--steam-accent)' : 'transparent',
                color: activeTab === 'explorer' ? '#fff' : 'var(--text-secondary)',
                boxShadow: activeTab === 'explorer' ? '0 2px 8px rgba(0,0,0,0.3)' : 'none',
                transition: 'all 0.2s ease'
              }}
            >
              <Database size={16} style={{ color: activeTab === 'explorer' ? 'var(--steam-blue)' : 'inherit' }} />
              <span>Dataset Explorer</span>
            </button>
          </div>

          {/* Backend Status Badge */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                fontSize: '0.75rem',
                fontWeight: 600,
                padding: '5px 10px',
                borderRadius: '20px',
                background: backendStatus === 'connected' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                color: backendStatus === 'connected' ? 'var(--hit-green)' : 'var(--flop-red)',
                border: `1px solid ${backendStatus === 'connected' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`
              }}
            >
              <span
                style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  backgroundColor: backendStatus === 'connected' ? 'var(--hit-green)' : 'var(--flop-red)',
                  display: 'inline-block'
                }}
              />
              <span>
                {backendStatus === 'connected'
                  ? `FastAPI Online (${backendMeta?.games_in_memory?.toLocaleString() || '27.8k'} Games)`
                  : 'Backend Offline'}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main style={{ flex: 1, maxWidth: '1440px', width: '100%', margin: '0 auto', padding: '1.75rem 2rem' }}>
        {activeTab === 'studio' ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.15fr) minmax(0, 0.85fr)', gap: '1.75rem', alignItems: 'start' }}>
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
          borderTop: '1px solid var(--border-subtle)',
          padding: '1.25rem 2rem',
          textAlign: 'center',
          fontSize: '0.8rem',
          color: 'var(--text-muted)',
          background: 'var(--bg-secondary)'
        }}
      >
        <div style={{ maxWidth: '1440px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
          <div>
            Steam Hit Predictor © 2026 • Trained on 27,810 verified Steam games with ≥50 reviews
          </div>
          <div style={{ display: 'flex', gap: '1.25rem' }}>
            <span>XGBoost 96.3% Acc</span>
            <span>•</span>
            <span>Qwen3.5-0.8B QLoRA</span>
            <span>•</span>
            <span>194 Engineered Features</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
