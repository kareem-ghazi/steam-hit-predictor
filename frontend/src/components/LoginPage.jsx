import React, { useState } from 'react';
import { Gamepad2, Lock, User, ArrowRight, ShieldCheck, CheckCircle2 } from 'lucide-react';

export default function LoginPage({ onLogin, onCancel }) {
  const [username, setUsername] = useState('developer_pro');
  const [password, setPassword] = useState('password123');

  const handleStandardLogin = (e) => {
    e.preventDefault();
    const finalUsername = username.trim() || 'developer_user';
    const userObj = {
      username: finalUsername,
      displayName: finalUsername.charAt(0).toUpperCase() + finalUsername.slice(1),
      provider: 'credentials',
      avatarUrl: `https://api.dicebear.com/7.x/bottts/svg?seed=${finalUsername}`
    };
    localStorage.setItem('steam_hit_user', JSON.stringify(userObj));
    onLogin(userObj);
  };

  const handleSocialLogin = (provider, defaultName) => {
    const userObj = {
      username: defaultName.toLowerCase().replace(/\s+/g, '_'),
      displayName: defaultName,
      provider: provider,
      avatarUrl: `https://api.dicebear.com/7.x/bottts/svg?seed=${provider}_${defaultName}`
    };
    localStorage.setItem('steam_hit_user', JSON.stringify(userObj));
    onLogin(userObj);
  };

  return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '75vh', padding: '1rem' }}>
      <div
        className="surface-card"
        style={{
          width: '100%',
          maxWidth: '460px',
          padding: '2.25rem 2rem',
          borderRadius: 'var(--radius-lg)',
          boxShadow: '0 12px 40px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(102, 192, 244, 0.25)',
          background: 'linear-gradient(180deg, #171f33 0%, #0d1527 100%)'
        }}
      >
        {/* Header Logo */}
        <div style={{ textAlign: 'center', marginBottom: '1.75rem' }}>
          <div
            style={{
              width: '48px',
              height: '48px',
              borderRadius: 'var(--radius-md)',
              background: 'linear-gradient(135deg, #1a9fff 0%, #0077cc 100%)',
              color: '#fff',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 0.85rem',
              boxShadow: '0 0 20px var(--primary-glow)'
            }}
          >
            <Gamepad2 size={26} />
          </div>
          <span className="font-mono" style={{ fontSize: '0.72rem', color: 'var(--primary)', fontWeight: 700, letterSpacing: '0.06em' }}>
            GAMEMETRICS ACCESS
          </span>
          <h2 style={{ fontSize: '1.45rem', fontWeight: 800, color: '#fff', marginTop: '2px', letterSpacing: '-0.01em' }}>
            Studio Authentication
          </h2>
          <p style={{ fontSize: '0.825rem', color: 'var(--on-surface-variant)', marginTop: '4px' }}>
            Access private studio game projects and commercial forecasts
          </p>
        </div>

        {/* Demo Mode Badge */}
        <div
          style={{
            background: 'rgba(16, 185, 129, 0.1)',
            border: '1px solid rgba(74, 222, 128, 0.25)',
            borderRadius: 'var(--radius-sm)',
            padding: '8px 12px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            marginBottom: '1.25rem'
          }}
        >
          <CheckCircle2 size={16} style={{ color: 'var(--status-positive-bright)', flexShrink: 0 }} />
          <span style={{ fontSize: '0.75rem', color: '#dae2fd' }}>
            <strong>Demo Mode Active:</strong> Sign in with <em>any</em> username and password to create private projects.
          </span>
        </div>

        {/* Credentials Form */}
        <form onSubmit={handleStandardLogin} style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginBottom: '1.5rem' }}>
          <div>
            <label style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--on-surface-variant)', display: 'block', marginBottom: '5px' }}>
              Studio Username or Email
            </label>
            <div style={{ position: 'relative' }}>
              <User size={15} style={{ position: 'absolute', left: '12px', top: '11px', color: 'var(--outline)' }} />
              <input
                type="text"
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter any studio username..."
                style={{ paddingLeft: '34px' }}
              />
            </div>
          </div>

          <div>
            <label style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--on-surface-variant)', display: 'block', marginBottom: '5px' }}>
              Password
            </label>
            <div style={{ position: 'relative' }}>
              <Lock size={15} style={{ position: 'absolute', left: '12px', top: '11px', color: 'var(--outline)' }} />
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter any password..."
                style={{ paddingLeft: '34px' }}
              />
            </div>
          </div>

          <button
            type="submit"
            className="btn-primary"
            style={{ width: '100%', padding: '11px', marginTop: '4px', fontSize: '0.925rem' }}
          >
            <span>Sign In to Studio</span>
            <ArrowRight size={16} />
          </button>
        </form>

        {/* Divider */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1.25rem' }}>
          <div style={{ flex: 1, height: '1px', background: 'rgba(51, 65, 85, 0.6)' }} />
          <span style={{ fontSize: '0.7rem', color: 'var(--outline)', fontWeight: 600, letterSpacing: '0.04em' }}>
            OR CONTINUE WITH
          </span>
          <div style={{ flex: 1, height: '1px', background: 'rgba(51, 65, 85, 0.6)' }} />
        </div>

        {/* Social Login Connectors (Google, Steam, GitHub) */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
          {/* Steam Connector */}
          <button
            type="button"
            onClick={() => handleSocialLogin('steam', 'Steam Developer')}
            className="btn-secondary"
            style={{
              width: '100%',
              padding: '10px 14px',
              justifyContent: 'center',
              background: 'rgba(23, 26, 33, 0.9)',
              borderColor: 'rgba(102, 192, 244, 0.35)',
              color: '#fff'
            }}
          >
            {/* Steam Logo SVG */}
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <path d="M11.979 0C5.678 0 .511 4.86.022 11.037l6.432 2.658c.545-.371 1.203-.59 1.912-.59.063 0 .125.004.188.006l2.861-4.142V8.91c0-2.495 2.028-4.524 4.524-4.524 2.494 0 4.524 2.029 4.524 4.524s-2.03 4.524-4.524 4.524h-.105l-4.076 2.871c0 .052.004.105.004.159 0 1.875-1.515 3.396-3.39 3.396-1.635 0-3.016-1.173-3.331-2.727L.436 14.87C1.815 20.088 6.556 24 12 24c6.627 0 12-5.373 12-12S18.605 0 11.979 0zM7.54 18.216c-.33-.518-.47-1.127-.378-1.74l-3.69-1.526c-.347 1.055-.262 2.215.309 3.204.936 1.621 2.973 2.19 4.606 1.258l-1.37-1.385c.18-.236.377-.525.523-.811zm8.4-11.554c-1.246 0-2.256 1.01-2.256 2.256s1.01 2.256 2.256 2.256 2.256-1.01 2.256-2.256-1.01-2.256-2.256-2.256zm0 3.61c-.747 0-1.354-.606-1.354-1.354 0-.747.607-1.354 1.354-1.354.748 0 1.354.607 1.354 1.354 0 .748-.606 1.354-1.354 1.354z"/>
            </svg>
            <span>Continue with Steam</span>
          </button>

          {/* Google Connector */}
          <button
            type="button"
            onClick={() => handleSocialLogin('google', 'Google Developer')}
            className="btn-secondary"
            style={{
              width: '100%',
              padding: '10px 14px',
              justifyContent: 'center',
              background: 'rgba(23, 26, 33, 0.9)',
              borderColor: 'rgba(255, 255, 255, 0.2)',
              color: '#fff'
            }}
          >
            {/* Google G Logo SVG */}
            <svg width="17" height="17" viewBox="0 0 24 24">
              <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
              <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
              <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"/>
              <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"/>
            </svg>
            <span>Continue with Google</span>
          </button>

          {/* GitHub Connector */}
          <button
            type="button"
            onClick={() => handleSocialLogin('github', 'GitHub Developer')}
            className="btn-secondary"
            style={{
              width: '100%',
              padding: '10px 14px',
              justifyContent: 'center',
              background: 'rgba(23, 26, 33, 0.9)',
              borderColor: 'rgba(255, 255, 255, 0.2)',
              color: '#fff'
            }}
          >
            {/* GitHub Octocat SVG */}
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"/>
            </svg>
            <span>Continue with GitHub</span>
          </button>
        </div>

        {/* Guest fallback button */}
        {onCancel && (
          <div style={{ textAlign: 'center', marginTop: '1.25rem' }}>
            <button
              type="button"
              onClick={onCancel}
              style={{ background: 'transparent', border: 'none', color: 'var(--outline)', fontSize: '0.8rem', cursor: 'pointer', textDecoration: 'underline' }}
            >
              Skip and continue as Guest
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
