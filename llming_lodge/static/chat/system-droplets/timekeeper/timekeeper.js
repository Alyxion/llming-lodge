/**
 * Timekeeper — A premium work timer mini-app
 * Stopwatch, Countdown Timer, Work Log, and Settings
 * Self-contained floating window app for llming-lodge
 */
(function () {
  'use strict';

  /* ======================================================================
   *  CSS — injected once
   * ====================================================================== */
  const STYLE_ID = 'bolt-tk-styles';

  function injectStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent = /* css */`
      /* ---- Custom Properties ---- */
      .bolt-tk-root {
        --tk-bg: #ffffff;
        --tk-bg2: #f4f6f8;
        --tk-bg3: #e8ecf0;
        --tk-fg: #1a1a2e;
        --tk-fg2: #5a6072;
        --tk-accent: #0066ff;
        --tk-accent-soft: rgba(0, 102, 255, 0.12);
        --tk-accent-glow: rgba(0, 102, 255, 0.3);
        --tk-danger: #ff4060;
        --tk-success: #00c853;
        --tk-warn: #ffab00;
        --tk-border: rgba(0,0,0,0.08);
        --tk-shadow: 0 2px 8px rgba(0,0,0,0.08);
        --tk-radius: 12px;
        --tk-radius-sm: 8px;
        --tk-clock-face: #f0f2f5;
        --tk-clock-rim: #d0d4dc;
        --tk-clock-hand: #1a1a2e;
        --tk-clock-second: #ff4060;
        --tk-font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        --tk-mono: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
        --tk-transition: 0.2s cubic-bezier(0.4,0,0.2,1);
        font-family: var(--tk-font);
        color: var(--tk-fg);
        background: var(--tk-bg);
        height: 100%;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        user-select: none;
        -webkit-user-select: none;
        font-size: 14px;
        line-height: 1.4;
      }
      .bolt-tk-root.bolt-tk-dark {
        --tk-bg: #0f1729;
        --tk-bg2: #172038;
        --tk-bg3: #1e2a4a;
        --tk-fg: #e8ecf4;
        --tk-fg2: #8892a8;
        --tk-accent: #60a5fa;
        --tk-accent-soft: rgba(96, 165, 250, 0.12);
        --tk-accent-glow: rgba(96, 165, 250, 0.25);
        --tk-border: rgba(255,255,255,0.06);
        --tk-shadow: 0 2px 12px rgba(0,0,0,0.3);
        --tk-clock-face: #172038;
        --tk-clock-rim: #2a3658;
        --tk-clock-hand: #e8ecf4;
        --tk-clock-second: #00d4ff;
      }

      /* ---- Tab bar ---- */
      .bolt-tk-tabs {
        display: flex;
        gap: 2px;
        padding: 6px 8px;
        background: var(--tk-bg2);
        border-bottom: 1px solid var(--tk-border);
        flex-shrink: 0;
        position: relative;
      }
      .bolt-tk-tab {
        flex: 1;
        padding: 8px 4px;
        border: none;
        background: transparent;
        color: var(--tk-fg2);
        font-size: 12px;
        font-weight: 500;
        cursor: pointer;
        border-radius: var(--tk-radius-sm);
        transition: all var(--tk-transition);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 5px;
        position: relative;
        font-family: var(--tk-font);
      }
      .bolt-tk-tab:hover { background: var(--tk-accent-soft); color: var(--tk-fg); }
      .bolt-tk-tab.bolt-tk-active {
        background: var(--tk-accent);
        color: #fff;
        font-weight: 600;
        box-shadow: 0 2px 8px var(--tk-accent-glow);
      }
      .bolt-tk-tab-icon {
        font-size: 16px;
        line-height: 1;
      }
      .bolt-tk-tracking-dot {
        position: absolute;
        top: 4px;
        right: 4px;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--tk-danger);
        animation: bolt-tk-pulse 1.5s ease-in-out infinite;
      }
      @keyframes bolt-tk-pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.7); }
      }

      /* ---- View container ---- */
      .bolt-tk-view {
        flex: 1;
        overflow-y: auto;
        overflow-x: hidden;
        padding: 16px;
        overscroll-behavior-y: contain;
      }
      .bolt-tk-view::-webkit-scrollbar { width: 4px; }
      .bolt-tk-view::-webkit-scrollbar-thumb {
        background: var(--tk-fg2);
        border-radius: 4px;
        opacity: 0.3;
      }

      /* ---- Tracking banner ---- */
      .bolt-tk-tracking-banner {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 12px;
        background: var(--tk-accent-soft);
        border-bottom: 1px solid var(--tk-border);
        font-size: 12px;
        color: var(--tk-accent);
        flex-shrink: 0;
        cursor: pointer;
        transition: background var(--tk-transition);
      }
      .bolt-tk-tracking-banner:hover { background: var(--tk-accent-glow); }
      .bolt-tk-tracking-banner-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: var(--tk-danger);
        animation: bolt-tk-pulse 1.5s ease-in-out infinite;
        flex-shrink: 0;
      }
      .bolt-tk-tracking-banner-text { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .bolt-tk-tracking-banner-time { font-family: var(--tk-mono); font-weight: 600; flex-shrink: 0; }

      /* ---- Stopwatch ---- */
      .bolt-tk-sw-display {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 16px;
        padding: 8px 0 16px;
      }
      .bolt-tk-sw-toggle {
        display: flex;
        gap: 4px;
        background: var(--tk-bg2);
        border-radius: 20px;
        padding: 3px;
      }
      .bolt-tk-sw-toggle-btn {
        padding: 4px 12px;
        border: none;
        background: transparent;
        color: var(--tk-fg2);
        font-size: 11px;
        font-weight: 500;
        cursor: pointer;
        border-radius: 16px;
        transition: all var(--tk-transition);
        font-family: var(--tk-font);
      }
      .bolt-tk-sw-toggle-btn.bolt-tk-active {
        background: var(--tk-accent);
        color: #fff;
      }

      /* Digital display */
      .bolt-tk-digital {
        font-family: var(--tk-mono);
        font-size: 48px;
        font-weight: 300;
        letter-spacing: 2px;
        color: var(--tk-fg);
        text-align: center;
        line-height: 1;
      }
      .bolt-tk-digital-cs {
        font-size: 28px;
        opacity: 0.5;
      }

      /* Analog clock */
      .bolt-tk-clock-container {
        width: 220px;
        height: 220px;
        position: relative;
      }
      .bolt-tk-clock-svg {
        width: 100%;
        height: 100%;
        filter: drop-shadow(0 4px 12px rgba(0,0,0,0.12));
      }
      .bolt-tk-dark .bolt-tk-clock-svg {
        filter: drop-shadow(0 4px 16px rgba(0,0,0,0.4));
      }

      /* Buttons row */
      .bolt-tk-btn-row {
        display: flex;
        gap: 10px;
        justify-content: center;
        flex-wrap: wrap;
      }
      .bolt-tk-btn {
        padding: 10px 24px;
        border: none;
        border-radius: 24px;
        font-size: 13px;
        font-weight: 600;
        cursor: pointer;
        transition: all var(--tk-transition);
        font-family: var(--tk-font);
        display: flex;
        align-items: center;
        gap: 6px;
      }
      .bolt-tk-btn:active { transform: scale(0.95); }
      .bolt-tk-btn-primary {
        background: var(--tk-accent);
        color: #fff;
        box-shadow: 0 2px 10px var(--tk-accent-glow);
      }
      .bolt-tk-btn-primary:hover { filter: brightness(1.1); box-shadow: 0 4px 16px var(--tk-accent-glow); }
      .bolt-tk-btn-danger {
        background: var(--tk-danger);
        color: #fff;
      }
      .bolt-tk-btn-danger:hover { filter: brightness(1.1); }
      .bolt-tk-btn-ghost {
        background: var(--tk-bg2);
        color: var(--tk-fg2);
        border: 1px solid var(--tk-border);
      }
      .bolt-tk-btn-ghost:hover { background: var(--tk-bg3); color: var(--tk-fg); }
      .bolt-tk-btn-success {
        background: var(--tk-success);
        color: #fff;
      }
      .bolt-tk-btn-success:hover { filter: brightness(1.1); }
      .bolt-tk-btn-sm {
        padding: 6px 14px;
        font-size: 12px;
      }
      .bolt-tk-btn-lg {
        padding: 14px 36px;
        font-size: 15px;
      }
      .bolt-tk-btn-icon-only {
        padding: 8px;
        border-radius: 50%;
        width: 36px; height: 36px;
        justify-content: center;
      }

      /* Lap list */
      .bolt-tk-laps {
        margin-top: 16px;
        border-top: 1px solid var(--tk-border);
        max-height: 200px;
        overflow-y: auto;
      }
      .bolt-tk-lap {
        display: flex;
        padding: 8px 12px;
        font-size: 12px;
        border-bottom: 1px solid var(--tk-border);
        transition: background var(--tk-transition);
      }
      .bolt-tk-lap:hover { background: var(--tk-bg2); }
      .bolt-tk-lap-num { width: 40px; color: var(--tk-fg2); font-weight: 600; }
      .bolt-tk-lap-time { flex: 1; font-family: var(--tk-mono); }
      .bolt-tk-lap-total { font-family: var(--tk-mono); color: var(--tk-fg2); }
      .bolt-tk-lap-fastest { color: var(--tk-success); }
      .bolt-tk-lap-slowest { color: var(--tk-danger); }

      /* ---- Timer / Countdown ---- */
      .bolt-tk-timer-setup {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 20px;
        padding-top: 8px;
      }
      .bolt-tk-time-picker {
        display: flex;
        align-items: center;
        gap: 4px;
      }
      .bolt-tk-time-input {
        width: 64px;
        height: 56px;
        border: 2px solid var(--tk-border);
        border-radius: var(--tk-radius-sm);
        background: var(--tk-bg2);
        color: var(--tk-fg);
        font-family: var(--tk-mono);
        font-size: 24px;
        text-align: center;
        outline: none;
        transition: border-color var(--tk-transition);
        font-weight: 500;
      }
      .bolt-tk-time-input:focus { border-color: var(--tk-accent); }
      .bolt-tk-time-sep {
        font-size: 24px;
        font-weight: 300;
        color: var(--tk-fg2);
      }
      .bolt-tk-presets {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
        justify-content: center;
      }
      .bolt-tk-preset {
        padding: 6px 14px;
        border: 1px solid var(--tk-border);
        border-radius: 20px;
        background: var(--tk-bg2);
        color: var(--tk-fg2);
        font-size: 12px;
        font-weight: 500;
        cursor: pointer;
        transition: all var(--tk-transition);
        font-family: var(--tk-font);
      }
      .bolt-tk-preset:hover { border-color: var(--tk-accent); color: var(--tk-accent); background: var(--tk-accent-soft); }

      /* Progress ring */
      .bolt-tk-progress-ring-container {
        position: relative;
        width: 200px;
        height: 200px;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      .bolt-tk-progress-ring-svg {
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        transform: rotate(-90deg);
      }
      .bolt-tk-progress-ring-bg {
        fill: none;
        stroke: var(--tk-bg3);
        stroke-width: 6;
      }
      .bolt-tk-progress-ring-fg {
        fill: none;
        stroke: var(--tk-accent);
        stroke-width: 6;
        stroke-linecap: round;
        transition: stroke-dashoffset 0.5s linear;
        filter: drop-shadow(0 0 6px var(--tk-accent-glow));
      }
      .bolt-tk-progress-inner-time {
        font-family: var(--tk-mono);
        font-size: 36px;
        font-weight: 300;
        z-index: 1;
      }

      /* Pomodoro */
      .bolt-tk-pomo-row {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        color: var(--tk-fg2);
      }
      .bolt-tk-pomo-dot {
        width: 10px; height: 10px;
        border-radius: 50%;
        border: 2px solid var(--tk-accent);
        transition: all var(--tk-transition);
      }
      .bolt-tk-pomo-dot.bolt-tk-filled { background: var(--tk-accent); }
      .bolt-tk-pomo-label { font-weight: 600; color: var(--tk-accent); }

      /* ---- Work Log ---- */
      .bolt-tk-wl-input-row {
        display: flex;
        gap: 8px;
        margin-bottom: 16px;
      }
      .bolt-tk-input {
        flex: 1;
        padding: 10px 14px;
        border: 1px solid var(--tk-border);
        border-radius: var(--tk-radius-sm);
        background: var(--tk-bg2);
        color: var(--tk-fg);
        font-size: 13px;
        outline: none;
        transition: border-color var(--tk-transition);
        font-family: var(--tk-font);
      }
      .bolt-tk-input:focus { border-color: var(--tk-accent); }
      .bolt-tk-input::placeholder { color: var(--tk-fg2); opacity: 0.6; }
      .bolt-tk-select {
        padding: 10px 12px;
        border: 1px solid var(--tk-border);
        border-radius: var(--tk-radius-sm);
        background: var(--tk-bg2);
        color: var(--tk-fg);
        font-size: 13px;
        outline: none;
        cursor: pointer;
        font-family: var(--tk-font);
        min-width: 100px;
      }

      /* Track button */
      .bolt-tk-track-btn {
        width: 100%;
        padding: 14px;
        border: none;
        border-radius: var(--tk-radius);
        font-size: 15px;
        font-weight: 700;
        cursor: pointer;
        transition: all var(--tk-transition);
        font-family: var(--tk-font);
        text-transform: uppercase;
        letter-spacing: 1px;
      }
      .bolt-tk-track-btn.bolt-tk-start {
        background: var(--tk-accent);
        color: #fff;
        box-shadow: 0 4px 16px var(--tk-accent-glow);
      }
      .bolt-tk-track-btn.bolt-tk-start:hover { filter: brightness(1.1); transform: translateY(-1px); }
      .bolt-tk-track-btn.bolt-tk-stop {
        background: var(--tk-danger);
        color: #fff;
        box-shadow: 0 4px 16px rgba(255,64,96,0.3);
      }
      .bolt-tk-track-btn.bolt-tk-stop:hover { filter: brightness(1.1); }

      /* Entries */
      .bolt-tk-section-title {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--tk-fg2);
        font-weight: 700;
        margin: 20px 0 10px;
      }
      .bolt-tk-entry {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 12px;
        background: var(--tk-bg2);
        border-radius: var(--tk-radius-sm);
        margin-bottom: 6px;
        transition: background var(--tk-transition);
        cursor: pointer;
      }
      .bolt-tk-entry:hover { background: var(--tk-bg3); }
      .bolt-tk-entry-color {
        width: 4px;
        height: 32px;
        border-radius: 2px;
        flex-shrink: 0;
      }
      .bolt-tk-entry-info { flex: 1; min-width: 0; }
      .bolt-tk-entry-name {
        font-size: 13px;
        font-weight: 600;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .bolt-tk-entry-times {
        font-size: 11px;
        color: var(--tk-fg2);
        font-family: var(--tk-mono);
      }
      .bolt-tk-entry-duration {
        font-family: var(--tk-mono);
        font-size: 13px;
        font-weight: 600;
        color: var(--tk-accent);
        flex-shrink: 0;
      }
      .bolt-tk-entry-delete {
        opacity: 0;
        transition: opacity var(--tk-transition);
        background: none;
        border: none;
        color: var(--tk-danger);
        cursor: pointer;
        font-size: 16px;
        padding: 4px;
        flex-shrink: 0;
      }
      .bolt-tk-entry:hover .bolt-tk-entry-delete { opacity: 1; }

      .bolt-tk-daily-total {
        display: flex;
        justify-content: space-between;
        padding: 12px 14px;
        background: var(--tk-accent-soft);
        border-radius: var(--tk-radius-sm);
        font-weight: 700;
        margin-top: 12px;
      }
      .bolt-tk-daily-total-value {
        font-family: var(--tk-mono);
        color: var(--tk-accent);
      }

      /* Weekly chart */
      .bolt-tk-weekly {
        margin-top: 16px;
      }
      .bolt-tk-chart {
        display: flex;
        align-items: flex-end;
        gap: 6px;
        height: 80px;
        padding: 0 4px;
      }
      .bolt-tk-chart-bar-wrap {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 4px;
        height: 100%;
        justify-content: flex-end;
      }
      .bolt-tk-chart-bar {
        width: 100%;
        border-radius: 4px 4px 0 0;
        background: var(--tk-accent);
        opacity: 0.7;
        transition: all var(--tk-transition);
        min-height: 2px;
      }
      .bolt-tk-chart-bar:hover { opacity: 1; }
      .bolt-tk-chart-bar.bolt-tk-today { opacity: 1; box-shadow: 0 0 8px var(--tk-accent-glow); }
      .bolt-tk-chart-label {
        font-size: 10px;
        color: var(--tk-fg2);
        font-weight: 600;
      }
      .bolt-tk-chart-hours {
        font-size: 9px;
        color: var(--tk-fg2);
        font-family: var(--tk-mono);
      }

      /* ---- Settings ---- */
      .bolt-tk-settings-group {
        margin-bottom: 20px;
      }
      .bolt-tk-settings-group-title {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--tk-fg2);
        font-weight: 700;
        margin-bottom: 10px;
      }
      .bolt-tk-setting-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid var(--tk-border);
      }
      .bolt-tk-setting-label {
        font-size: 13px;
        font-weight: 500;
      }
      .bolt-tk-setting-desc {
        font-size: 11px;
        color: var(--tk-fg2);
        margin-top: 2px;
      }
      /* Toggle switch */
      .bolt-tk-toggle {
        position: relative;
        width: 44px;
        height: 24px;
        cursor: pointer;
        flex-shrink: 0;
      }
      .bolt-tk-toggle input { display: none; }
      .bolt-tk-toggle-track {
        position: absolute;
        inset: 0;
        background: var(--tk-bg3);
        border-radius: 12px;
        transition: background var(--tk-transition);
      }
      .bolt-tk-toggle input:checked + .bolt-tk-toggle-track { background: var(--tk-accent); }
      .bolt-tk-toggle-knob {
        position: absolute;
        top: 2px; left: 2px;
        width: 20px; height: 20px;
        background: #fff;
        border-radius: 50%;
        transition: transform var(--tk-transition);
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
      }
      .bolt-tk-toggle input:checked ~ .bolt-tk-toggle-knob { transform: translateX(20px); }
      .bolt-tk-number-input {
        width: 60px;
        padding: 6px 8px;
        border: 1px solid var(--tk-border);
        border-radius: var(--tk-radius-sm);
        background: var(--tk-bg2);
        color: var(--tk-fg);
        font-family: var(--tk-mono);
        font-size: 13px;
        text-align: center;
        outline: none;
      }
      .bolt-tk-number-input:focus { border-color: var(--tk-accent); }

      /* Project management */
      .bolt-tk-project-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 12px;
        background: var(--tk-bg2);
        border-radius: var(--tk-radius-sm);
        margin-bottom: 6px;
      }
      .bolt-tk-project-color-swatch {
        width: 20px; height: 20px;
        border-radius: 50%;
        cursor: pointer;
        border: 2px solid var(--tk-border);
        flex-shrink: 0;
      }
      .bolt-tk-project-name-edit {
        flex: 1;
        background: transparent;
        border: none;
        color: var(--tk-fg);
        font-size: 13px;
        outline: none;
        font-family: var(--tk-font);
      }
      .bolt-tk-add-project-row {
        display: flex;
        gap: 8px;
        margin-top: 8px;
      }

      /* ---- Edit modal overlay ---- */
      .bolt-tk-modal-overlay {
        position: absolute;
        inset: 0;
        background: rgba(0,0,0,0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 100;
        padding: 20px;
      }
      .bolt-tk-modal {
        background: var(--tk-bg);
        border-radius: var(--tk-radius);
        padding: 20px;
        width: 100%;
        max-width: 320px;
        box-shadow: var(--tk-shadow);
      }
      .bolt-tk-modal-title {
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 16px;
      }
      .bolt-tk-modal-field {
        margin-bottom: 12px;
      }
      .bolt-tk-modal-field label {
        display: block;
        font-size: 11px;
        font-weight: 600;
        color: var(--tk-fg2);
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 1px;
      }
      .bolt-tk-modal-actions {
        display: flex;
        gap: 8px;
        justify-content: flex-end;
        margin-top: 16px;
      }

      /* ---- Animations ---- */
      @keyframes bolt-tk-fade-in {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
      }
      .bolt-tk-fade-in { animation: bolt-tk-fade-in 0.25s ease-out; }

      /* ---- Responsive ---- */
      @media (max-width: 400px) {
        .bolt-tk-digital { font-size: 36px; }
        .bolt-tk-digital-cs { font-size: 20px; }
        .bolt-tk-clock-container { width: 180px; height: 180px; }
        .bolt-tk-btn { padding: 8px 16px; font-size: 12px; }
      }
    `;
    document.head.appendChild(style);
  }

  /* ======================================================================
   *  Utility helpers
   * ====================================================================== */
  function h(tag, attrs, ...children) {
    const el = document.createElement(tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (k === 'style' && typeof v === 'object') {
          Object.assign(el.style, v);
        } else if (k.startsWith('on') && typeof v === 'function') {
          el.addEventListener(k.slice(2).toLowerCase(), v);
        } else if (k === 'className') {
          el.className = v;
        } else if (k === 'htmlContent') {
          el.innerHTML = v;
        } else {
          el.setAttribute(k, v);
        }
      }
    }
    for (const c of children) {
      if (c == null) continue;
      if (typeof c === 'string' || typeof c === 'number') {
        el.appendChild(document.createTextNode(String(c)));
      } else if (c instanceof Node) {
        el.appendChild(c);
      }
    }
    return el;
  }

  function pad(n, len = 2) { return String(n).padStart(len, '0'); }

  function fmtDuration(ms) {
    const totalSec = Math.floor(ms / 1000);
    const hrs = Math.floor(totalSec / 3600);
    const mins = Math.floor((totalSec % 3600) / 60);
    const secs = totalSec % 60;
    const cs = Math.floor((ms % 1000) / 10);
    if (hrs > 0) return `${pad(hrs)}:${pad(mins)}:${pad(secs)}`;
    return `${pad(mins)}:${pad(secs)}.${pad(cs)}`;
  }

  function fmtHMS(ms) {
    const totalSec = Math.floor(ms / 1000);
    const hrs = Math.floor(totalSec / 3600);
    const mins = Math.floor((totalSec % 3600) / 60);
    const secs = totalSec % 60;
    return `${pad(hrs)}:${pad(mins)}:${pad(secs)}`;
  }

  function fmtTime(date, use24) {
    const d = new Date(date);
    if (use24) return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
    let h = d.getHours();
    const ampm = h >= 12 ? 'PM' : 'AM';
    h = h % 12 || 12;
    return `${h}:${pad(d.getMinutes())} ${ampm}`;
  }

  function fmtHoursDecimal(ms) {
    return (ms / 3600000).toFixed(1) + 'h';
  }

  function todayKey() {
    const d = new Date();
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  }

  function getWeekDates() {
    const now = new Date();
    const day = now.getDay();
    const mon = new Date(now);
    mon.setDate(now.getDate() - ((day + 6) % 7));
    mon.setHours(0, 0, 0, 0);
    const dates = [];
    for (let i = 0; i < 7; i++) {
      const d = new Date(mon);
      d.setDate(mon.getDate() + i);
      dates.push(`${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`);
    }
    return dates;
  }

  const DAY_LABELS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

  const DEFAULT_COLORS = ['#0066ff', '#00d4ff', '#ff4060', '#ffab00', '#00c853', '#9c27b0', '#ff6d00', '#607d8b'];

  function uid() { return Date.now().toString(36) + Math.random().toString(36).slice(2, 6); }

  /* ======================================================================
   *  Parse duration strings like "30m", "30s", "1h", "5m30s", "1h30m"
   * ====================================================================== */
  function parseDuration(input) {
    if (!input) return 0;
    const text = input.trim().toLowerCase();
    let totalMs = 0;
    const re = /(\d+(?:\.\d+)?)\s*(h|hr|m|min|s|sec)/gi;
    let match;
    let hasUnit = false;
    while ((match = re.exec(text)) !== null) {
      hasUnit = true;
      const val = parseFloat(match[1]);
      const unit = match[2].toLowerCase();
      if (unit === 'h' || unit === 'hr') totalMs += val * 3600000;
      else if (unit === 'm' || unit === 'min') totalMs += val * 60000;
      else if (unit === 's' || unit === 'sec') totalMs += val * 1000;
    }
    // Plain number without unit → assume minutes
    if (!hasUnit) {
      const n = parseFloat(text);
      if (!isNaN(n) && n > 0) totalMs = n * 60000;
    }
    return totalMs;
  }

  /**
   * Split input like "30m Design Review" or "Design Review" into
   * { durationMs, taskName }. If the first token looks like a duration
   * (e.g. "30m", "1h30m", "90"), it's consumed; the rest is the task name.
   */
  function parseTimerInput(input) {
    if (!input) return { durationMs: 0, taskName: '' };
    const text = input.trim();
    // Try parsing just the first whitespace-delimited token as a duration
    const firstSpace = text.search(/\s/);
    if (firstSpace > 0) {
      const firstToken = text.slice(0, firstSpace);
      const dur = parseDuration(firstToken);
      if (dur > 0) {
        return { durationMs: dur, taskName: text.slice(firstSpace).trim() };
      }
    }
    // No leading duration — entire input is the task name
    return { durationMs: 0, taskName: text };
  }

  /* ======================================================================
   *  Web Audio beep
   * ====================================================================== */
  function playAlarm() {
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const notes = [880, 1100, 880, 1100, 880];
      let t = ctx.currentTime;
      for (const freq of notes) {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.value = freq;
        gain.gain.setValueAtTime(0.3, t);
        gain.gain.exponentialRampToValueAtTime(0.01, t + 0.25);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(t);
        osc.stop(t + 0.25);
        t += 0.3;
      }
      setTimeout(() => ctx.close(), 2000);
    } catch (_) { /* no audio */ }
  }

  /* ======================================================================
   *  SVG Analog Clock for Stopwatch
   * ====================================================================== */
  function createAnalogClock() {
    const NS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', '0 0 200 200');
    svg.setAttribute('class', 'bolt-tk-clock-svg');

    const CX = 100, CY = 100, R = 88;

    // Defs for gradients
    const defs = document.createElementNS(NS, 'defs');

    // Clock face gradient
    const radGrad = document.createElementNS(NS, 'radialGradient');
    radGrad.id = 'bolt-tk-face-grad-' + uid();
    const s1 = document.createElementNS(NS, 'stop');
    s1.setAttribute('offset', '0%');
    s1.setAttribute('class', 'bolt-tk-grad-s1');
    const s2 = document.createElementNS(NS, 'stop');
    s2.setAttribute('offset', '100%');
    s2.setAttribute('class', 'bolt-tk-grad-s2');
    radGrad.append(s1, s2);
    defs.appendChild(radGrad);

    // Shadow filter
    const filter = document.createElementNS(NS, 'filter');
    filter.id = 'bolt-tk-hand-shadow-' + uid();
    filter.setAttribute('x', '-50%'); filter.setAttribute('y', '-50%');
    filter.setAttribute('width', '200%'); filter.setAttribute('height', '200%');
    const feDS = document.createElementNS(NS, 'feDropShadow');
    feDS.setAttribute('dx', '0'); feDS.setAttribute('dy', '1');
    feDS.setAttribute('stdDeviation', '1.5'); feDS.setAttribute('flood-opacity', '0.3');
    filter.appendChild(feDS);
    defs.appendChild(filter);

    svg.appendChild(defs);

    // Outer rim
    const rimOuter = document.createElementNS(NS, 'circle');
    rimOuter.setAttribute('cx', CX); rimOuter.setAttribute('cy', CY); rimOuter.setAttribute('r', R + 6);
    rimOuter.setAttribute('fill', 'none');
    rimOuter.setAttribute('stroke', 'var(--tk-clock-rim)');
    rimOuter.setAttribute('stroke-width', '3');
    svg.appendChild(rimOuter);

    // Face
    const face = document.createElementNS(NS, 'circle');
    face.setAttribute('cx', CX); face.setAttribute('cy', CY); face.setAttribute('r', R);
    face.setAttribute('fill', 'var(--tk-clock-face)');
    face.setAttribute('stroke', 'var(--tk-clock-rim)');
    face.setAttribute('stroke-width', '1');
    svg.appendChild(face);

    // Inner ring
    const innerRing = document.createElementNS(NS, 'circle');
    innerRing.setAttribute('cx', CX); innerRing.setAttribute('cy', CY); innerRing.setAttribute('r', R - 8);
    innerRing.setAttribute('fill', 'none');
    innerRing.setAttribute('stroke', 'var(--tk-border)');
    innerRing.setAttribute('stroke-width', '0.5');
    svg.appendChild(innerRing);

    // Tick marks
    for (let i = 0; i < 60; i++) {
      const angle = (i * 6 - 90) * (Math.PI / 180);
      const isMajor = i % 5 === 0;
      const outerR = R - 4;
      const innerR = isMajor ? R - 16 : R - 10;
      const line = document.createElementNS(NS, 'line');
      line.setAttribute('x1', CX + Math.cos(angle) * innerR);
      line.setAttribute('y1', CY + Math.sin(angle) * innerR);
      line.setAttribute('x2', CX + Math.cos(angle) * outerR);
      line.setAttribute('y2', CY + Math.sin(angle) * outerR);
      line.setAttribute('stroke', isMajor ? 'var(--tk-fg)' : 'var(--tk-fg2)');
      line.setAttribute('stroke-width', isMajor ? '2' : '0.8');
      line.setAttribute('stroke-linecap', 'round');
      if (isMajor) line.setAttribute('opacity', '0.8');
      else line.setAttribute('opacity', '0.4');
      svg.appendChild(line);
    }

    // Hour numerals
    for (let i = 1; i <= 12; i++) {
      const angle = (i * 30 - 90) * (Math.PI / 180);
      const numR = R - 26;
      const text = document.createElementNS(NS, 'text');
      text.setAttribute('x', CX + Math.cos(angle) * numR);
      text.setAttribute('y', CY + Math.sin(angle) * numR + 4.5);
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('font-size', '11');
      text.setAttribute('font-weight', '600');
      text.setAttribute('fill', 'var(--tk-fg)');
      text.setAttribute('font-family', 'var(--tk-font)');
      text.setAttribute('opacity', '0.7');
      text.textContent = i;
      svg.appendChild(text);
    }

    // Minute hand
    const minuteHand = document.createElementNS(NS, 'line');
    minuteHand.setAttribute('x1', CX); minuteHand.setAttribute('y1', CY + 12);
    minuteHand.setAttribute('x2', CX); minuteHand.setAttribute('y2', CY - 60);
    minuteHand.setAttribute('stroke', 'var(--tk-clock-hand)');
    minuteHand.setAttribute('stroke-width', '3');
    minuteHand.setAttribute('stroke-linecap', 'round');
    minuteHand.setAttribute('filter', `url(#${filter.id})`);
    svg.appendChild(minuteHand);

    // Second hand
    const secondHand = document.createElementNS(NS, 'line');
    secondHand.setAttribute('x1', CX); secondHand.setAttribute('y1', CY + 18);
    secondHand.setAttribute('x2', CX); secondHand.setAttribute('y2', CY - 72);
    secondHand.setAttribute('stroke', 'var(--tk-clock-second)');
    secondHand.setAttribute('stroke-width', '1.5');
    secondHand.setAttribute('stroke-linecap', 'round');
    svg.appendChild(secondHand);

    // Center cap
    const cap = document.createElementNS(NS, 'circle');
    cap.setAttribute('cx', CX); cap.setAttribute('cy', CY); cap.setAttribute('r', 5);
    cap.setAttribute('fill', 'var(--tk-clock-second)');
    svg.appendChild(cap);

    const capInner = document.createElementNS(NS, 'circle');
    capInner.setAttribute('cx', CX); capInner.setAttribute('cy', CY); capInner.setAttribute('r', 2.5);
    capInner.setAttribute('fill', 'var(--tk-bg)');
    svg.appendChild(capInner);

    function update(elapsedMs) {
      const totalSec = elapsedMs / 1000;
      const secAngle = (totalSec % 60) * 6;
      const minAngle = ((totalSec / 60) % 60) * 6;
      secondHand.setAttribute('transform', `rotate(${secAngle}, ${CX}, ${CY})`);
      minuteHand.setAttribute('transform', `rotate(${minAngle}, ${CX}, ${CY})`);
    }

    return { svg, update };
  }

  /* ======================================================================
   *  MAIN APP
   * ====================================================================== */
  function createApp(container, args, context) {
    injectStyles();

    const isDark = context.isDark;
    const locale = context.locale || 'en';

    // Memory (with localStorage fallback)
    const mem = context.memory || {
      async get(k) { try { return JSON.parse(localStorage.getItem('bolt-tk-' + k)); } catch { return null; } },
      async set(k, v) { localStorage.setItem('bolt-tk-' + k, JSON.stringify(v)); },
      async delete(k) { localStorage.removeItem('bolt-tk-' + k); },
      async sessionGet(k) { try { return JSON.parse(sessionStorage.getItem('bolt-tk-' + k)); } catch { return null; } },
      async sessionSet(k, v) { sessionStorage.setItem('bolt-tk-' + k, JSON.stringify(v)); },
      async registerRevive(cfg) { localStorage.setItem('bolt-tk-__revive__', JSON.stringify(cfg)); },
      async unregisterRevive() { localStorage.removeItem('bolt-tk-__revive__'); },
    };

    /* ---- State ---- */
    const state = {
      activeView: (args && args.view) || 'stopwatch',
      // Stopwatch
      sw: { running: false, startTime: 0, elapsed: 0, laps: [], displayMode: 'analog' },
      // Timer
      timer: {
        running: false, paused: false, startTime: 0, duration: 0, remaining: 0,
        pomodoroMode: false, pomodoroSession: 0, pomodoroOnBreak: false, taskName: '',
      },
      // Work log
      wl: { tracking: false, currentTask: '', currentProject: '', trackingStart: 0, entries: {}, projects: [] },
      // Settings
      settings: {
        displayMode: 'analog', use24h: true, pomoDuration: 25, breakDuration: 5, longBreakDuration: 15,
        reminderInterval: 30, notify: true,
      },
    };

    let animFrame = null;
    let timerInterval = null;
    let reminderInterval = null;
    let destroyed = false;
    let _maxW = 0, _maxH = 0;

    /* ---- Root element ---- */
    const root = h('div', { className: `bolt-tk-root${isDark ? ' bolt-tk-dark' : ''} bolt-tk-fade-in` });
    container.appendChild(root);

    /* ---- Load persisted state ---- */
    async function loadState() {
      const saved = await mem.get('state');
      if (saved) {
        // Merge saved into state, keeping defaults for missing keys
        if (saved.sw) Object.assign(state.sw, saved.sw);
        if (saved.timer) Object.assign(state.timer, saved.timer);
        if (saved.wl) {
          Object.assign(state.wl, saved.wl);
          if (!state.wl.entries) state.wl.entries = {};
          if (!state.wl.projects) state.wl.projects = [];
        }
        if (saved.settings) Object.assign(state.settings, saved.settings);
        // Only restore saved view if no explicit view arg was passed
        if (saved.activeView && !(args && args.view)) state.activeView = saved.activeView;
      }
      // Default projects
      if (state.wl.projects.length === 0) {
        state.wl.projects = [
          { id: uid(), name: 'Development', color: '#0066ff' },
          { id: uid(), name: 'Meetings', color: '#ff6d00' },
          { id: uid(), name: 'Design', color: '#9c27b0' },
        ];
      }
      // Revive running states
      if (state.sw.running && state.sw.startTime) {
        // Recalculate elapsed from saved start time
        state.sw.elapsed = Date.now() - state.sw.startTime;
      }
      if (state.timer.running && state.timer.startTime) {
        const elapsed = Date.now() - state.timer.startTime;
        state.timer.remaining = Math.max(0, state.timer.duration - elapsed);
        if (state.timer.remaining <= 0) {
          state.timer.running = false;
          state.timer.paused = false;
        }
      }
    }

    async function saveState() {
      if (destroyed) return;
      await mem.set('state', {
        activeView: state.activeView,
        sw: state.sw,
        timer: state.timer,
        wl: { ...state.wl },
        settings: state.settings,
      });
    }

    /* ---- Tab bar ---- */
    const tabs = [
      { id: 'stopwatch', icon: '\u23F1', label: 'Stopwatch' },
      { id: 'timer', icon: '\u23F3', label: 'Timer' },
      { id: 'worklog', icon: '\uD83D\uDCCB', label: 'Work Log' },
      { id: 'settings', icon: '\u2699', label: 'Settings' },
    ];

    let trackingBanner = null;
    let tabBar = null;
    let viewContainer = null;

    function buildChrome() {
      root.innerHTML = '';

      // Tracking banner (always visible if tracking)
      trackingBanner = h('div', { className: 'bolt-tk-tracking-banner', style: { display: 'none' } });
      root.appendChild(trackingBanner);

      // Tab bar
      tabBar = h('div', { className: 'bolt-tk-tabs' });
      for (const tab of tabs) {
        const btn = h('button', {
          className: `bolt-tk-tab${state.activeView === tab.id ? ' bolt-tk-active' : ''}`,
          onClick: () => switchView(tab.id),
        },
          h('span', { className: 'bolt-tk-tab-icon' }, tab.icon),
          h('span', null, tab.label),
        );
        if (tab.id === 'worklog' && state.wl.tracking) {
          btn.appendChild(h('span', { className: 'bolt-tk-tracking-dot' }));
        }
        tabBar.appendChild(btn);
      }
      root.appendChild(tabBar);

      // View container
      viewContainer = h('div', { className: 'bolt-tk-view' });
      root.appendChild(viewContainer);

      updateTrackingBanner();
    }

    function switchView(id) {
      state.activeView = id;
      // Update tab active states
      const tabBtns = tabBar.querySelectorAll('.bolt-tk-tab');
      tabs.forEach((t, i) => {
        tabBtns[i].className = `bolt-tk-tab${t.id === id ? ' bolt-tk-active' : ''}`;
      });
      renderView();
      // Resize window to fit the current view
      const sz = VIEW_SIZES[id];
      if (sz && context.resize) context.resize(sz.w, sz.h);
      saveState();
    }

    function updateTrackingBanner() {
      if (!state.wl.tracking) {
        trackingBanner.style.display = 'none';
        return;
      }
      trackingBanner.style.display = 'flex';
      const elapsed = Date.now() - state.wl.trackingStart;
      const proj = state.wl.projects.find(p => p.id === state.wl.currentProject);
      trackingBanner.innerHTML = '';
      trackingBanner.appendChild(h('span', { className: 'bolt-tk-tracking-banner-dot' }));
      trackingBanner.appendChild(h('span', { className: 'bolt-tk-tracking-banner-text' },
        `Tracking: ${state.wl.currentTask || (proj ? proj.name : 'Task')}`));
      trackingBanner.appendChild(h('span', { className: 'bolt-tk-tracking-banner-time' }, fmtHMS(elapsed)));
      trackingBanner.onclick = () => switchView('worklog');

      // Update dot on worklog tab
      const tabBtns = tabBar.querySelectorAll('.bolt-tk-tab');
      const wlIdx = tabs.findIndex(t => t.id === 'worklog');
      const existing = tabBtns[wlIdx].querySelector('.bolt-tk-tracking-dot');
      if (!existing) tabBtns[wlIdx].appendChild(h('span', { className: 'bolt-tk-tracking-dot' }));
    }

    /* ==================================================================
     *  STOPWATCH VIEW
     * ================================================================== */
    // Persistent refs for stopwatch (avoid full rebuild)
    const LAPS_REGION_H = 180; // fixed height for the laps scroll area
    let _swWrap = null, _swBtnRow = null, _swLapsEl = null;
    let _swClockObj = null, _swDigitalEl = null, _swDisplayMode = null;

    function _swBuildDisplay(wrap) {
      _swDisplayMode = state.sw.displayMode;
      // Remove old clock/digital
      const oldClock = wrap.querySelector('.bolt-tk-clock-container');
      const oldDigital = wrap.querySelector('.bolt-tk-digital');
      if (oldClock) oldClock.remove();
      if (oldDigital) oldDigital.remove();
      _swClockObj = null; _swDigitalEl = null;

      const btnRow = wrap.querySelector('.bolt-tk-btn-row');
      if (state.sw.displayMode === 'analog') {
        const clockContainer = h('div', { className: 'bolt-tk-clock-container' });
        _swClockObj = createAnalogClock();
        clockContainer.appendChild(_swClockObj.svg);
        wrap.insertBefore(clockContainer, btnRow);
        _swDigitalEl = h('div', { className: 'bolt-tk-digital', style: { fontSize: '24px' } });
        wrap.insertBefore(_swDigitalEl, btnRow);
      } else {
        _swDigitalEl = h('div', { className: 'bolt-tk-digital' });
        wrap.insertBefore(_swDigitalEl, btnRow);
      }
    }

    function _swUpdateButtons() {
      if (!_swBtnRow) return;
      _swBtnRow.innerHTML = '';
      if (!state.sw.running && state.sw.elapsed === 0) {
        _swBtnRow.appendChild(h('button', {
          className: 'bolt-tk-btn bolt-tk-btn-primary bolt-tk-btn-lg',
          onClick: startStopwatch,
        }, '\u25B6 Start'));
      } else if (state.sw.running) {
        _swBtnRow.appendChild(h('button', {
          className: 'bolt-tk-btn bolt-tk-btn-ghost',
          onClick: lapStopwatch,
        }, '\u25CF Lap'));
        _swBtnRow.appendChild(h('button', {
          className: 'bolt-tk-btn bolt-tk-btn-danger bolt-tk-btn-lg',
          onClick: stopStopwatch,
        }, '\u25A0 Stop'));
      } else {
        _swBtnRow.appendChild(h('button', {
          className: 'bolt-tk-btn bolt-tk-btn-ghost',
          onClick: resetStopwatch,
        }, '\u21BB Reset'));
        _swBtnRow.appendChild(h('button', {
          className: 'bolt-tk-btn bolt-tk-btn-primary bolt-tk-btn-lg',
          onClick: startStopwatch,
        }, '\u25B6 Resume'));
      }
    }

    function _swUpdateLaps() {
      if (!_swLapsEl) return;
      _swLapsEl.innerHTML = '';
      if (state.sw.laps.length === 0) {
        _swLapsEl.style.display = 'none';
        return;
      }
      _swLapsEl.style.display = '';
      const lapTimes = state.sw.laps.map(l => l.lapTime);
      const fastest = Math.min(...lapTimes);
      const slowest = Math.max(...lapTimes);
      const showHighlight = state.sw.laps.length > 2;

      for (let i = state.sw.laps.length - 1; i >= 0; i--) {
        const lap = state.sw.laps[i];
        let cls = 'bolt-tk-lap';
        let timeCls = 'bolt-tk-lap-time';
        if (showHighlight && lap.lapTime === fastest) timeCls += ' bolt-tk-lap-fastest';
        else if (showHighlight && lap.lapTime === slowest) timeCls += ' bolt-tk-lap-slowest';
        _swLapsEl.appendChild(h('div', { className: cls },
          h('span', { className: 'bolt-tk-lap-num' }, `#${i + 1}`),
          h('span', { className: timeCls }, fmtDuration(lap.lapTime)),
          h('span', { className: 'bolt-tk-lap-total' }, fmtDuration(lap.totalTime)),
        ));
      }
    }

    function renderStopwatch() {
      // Build stable DOM once, then update in place
      if (!_swWrap || !viewContainer.contains(_swWrap)) {
        viewContainer.innerHTML = '';
        _swWrap = h('div', { className: 'bolt-tk-sw-display bolt-tk-fade-in' });
        _swDisplayMode = null;

        // Display mode toggle
        const modeToggle = h('div', { className: 'bolt-tk-sw-toggle' });
        const analogBtn = h('button', {
          className: `bolt-tk-sw-toggle-btn${state.sw.displayMode === 'analog' ? ' bolt-tk-active' : ''}`,
          onClick: () => { state.sw.displayMode = 'analog'; _swBuildDisplay(_swWrap); _swStartTick(); analogBtn.className = 'bolt-tk-sw-toggle-btn bolt-tk-active'; digitalBtn.className = 'bolt-tk-sw-toggle-btn'; saveState(); },
        }, 'Analog');
        const digitalBtn = h('button', {
          className: `bolt-tk-sw-toggle-btn${state.sw.displayMode === 'digital' ? ' bolt-tk-active' : ''}`,
          onClick: () => { state.sw.displayMode = 'digital'; _swBuildDisplay(_swWrap); _swStartTick(); digitalBtn.className = 'bolt-tk-sw-toggle-btn bolt-tk-active'; analogBtn.className = 'bolt-tk-sw-toggle-btn'; saveState(); },
        }, 'Digital');
        modeToggle.append(analogBtn, digitalBtn);
        _swWrap.appendChild(modeToggle);

        _swBtnRow = h('div', { className: 'bolt-tk-btn-row', style: { alignSelf: 'stretch' } });
        _swWrap.appendChild(_swBtnRow);

        _swLapsEl = h('div', { className: 'bolt-tk-laps' });
        _swLapsEl.style.cssText = `height: ${LAPS_REGION_H}px; overflow-y: auto; flex-shrink: 0; align-self: stretch; width: 100%; display: none;`;
        _swWrap.appendChild(_swLapsEl);

        viewContainer.appendChild(_swWrap);
        _swBuildDisplay(_swWrap);
      }

      // Update only buttons and laps (no full rebuild)
      _swUpdateButtons();
      _swUpdateLaps();
      _swStartTick();
      // If restored with existing laps, expand window
      if (state.sw.laps.length > 0 && context.resize) {
        const sz = VIEW_SIZES.stopwatch;
        context.resize(sz.w, sz.h + LAPS_REGION_H);
      }
    }

    function _swStartTick() {
      if (animFrame) cancelAnimationFrame(animFrame);

      function tick() {
        if (destroyed) return;
        let elapsed = state.sw.elapsed;
        if (state.sw.running) {
          elapsed = Date.now() - state.sw.startTime;
          state.sw.elapsed = elapsed;
        }

        // Current lap time (elapsed since last lap marker)
        const lastLapTotal = state.sw.laps.length > 0
          ? state.sw.laps[state.sw.laps.length - 1].totalTime : 0;
        const lapElapsed = elapsed - lastLapTotal;
        const displayMs = state.sw.laps.length > 0 ? lapElapsed : elapsed;

        const totalSec = Math.floor(displayMs / 1000);
        const hrs = Math.floor(totalSec / 3600);
        const mins = Math.floor((totalSec % 3600) / 60);
        const secs = totalSec % 60;
        const cs = Math.floor((displayMs % 1000) / 10);

        if (_swDigitalEl) {
          let html = '';
          // Show total time as small label when laps exist
          if (state.sw.laps.length > 0) {
            const tSec = Math.floor(elapsed / 1000);
            const tH = Math.floor(tSec / 3600), tM = Math.floor((tSec % 3600) / 60), tS = tSec % 60;
            const tCs = Math.floor((elapsed % 1000) / 10);
            const totalStr = tH > 0 ? `${pad(tH)}:${pad(tM)}:${pad(tS)}.${pad(tCs)}` : `${pad(tM)}:${pad(tS)}.${pad(tCs)}`;
            html += `<div style="font-size:12px;opacity:0.45;margin-bottom:2px">Total ${totalStr}</div>`;
          }
          if (hrs > 0) {
            html += `${pad(hrs)}:${pad(mins)}:${pad(secs)}<span class="bolt-tk-digital-cs">.${pad(cs)}</span>`;
          } else {
            html += `${pad(mins)}:${pad(secs)}<span class="bolt-tk-digital-cs">.${pad(cs)}</span>`;
          }
          _swDigitalEl.innerHTML = html;
        }
        if (_swClockObj) _swClockObj.update(displayMs);

        if (state.sw.running && state.activeView === 'stopwatch') {
          animFrame = requestAnimationFrame(tick);
        }
      }

      tick();
      if (state.sw.running) animFrame = requestAnimationFrame(tick);
    }

    function startStopwatch() {
      if (state.sw.running) return;
      if (state.sw.elapsed > 0) {
        state.sw.startTime = Date.now() - state.sw.elapsed;
      } else {
        state.sw.startTime = Date.now();
        state.sw.elapsed = 0;
      }
      state.sw.running = true;
      _swUpdateButtons();
      _swStartTick();
      saveState();
    }

    function stopStopwatch() {
      if (!state.sw.running) return;
      state.sw.elapsed = Date.now() - state.sw.startTime;
      state.sw.running = false;
      if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
      _swUpdateButtons();
      saveState();
    }

    function resetStopwatch() {
      state.sw.running = false;
      state.sw.elapsed = 0;
      state.sw.startTime = 0;
      state.sw.laps = [];
      if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
      _swUpdateButtons();
      _swUpdateLaps();
      _swStartTick(); // reset display to 00:00.00
      // Shrink window back to default stopwatch size
      if (context.resize) {
        const sz = VIEW_SIZES.stopwatch;
        context.resize(sz.w, sz.h);
      }
      saveState();
    }

    function lapStopwatch() {
      if (!state.sw.running) return;
      const now = Date.now();
      const totalTime = now - state.sw.startTime;
      const prevTotal = state.sw.laps.length > 0 ? state.sw.laps[state.sw.laps.length - 1].totalTime : 0;
      const lapTime = totalTime - prevTotal;
      state.sw.laps.push({ lapTime, totalTime });
      _swUpdateLaps();
      // On first lap, instantly grow window to accommodate laps scroll region
      if (state.sw.laps.length === 1 && context.resize) {
        const sz = VIEW_SIZES.stopwatch;
        context.resize(sz.w, sz.h + LAPS_REGION_H);
      }
      if (_swLapsEl) _swLapsEl.scrollTop = 0;
      saveState();
    }

    /* ==================================================================
     *  TIMER / COUNTDOWN VIEW
     * ================================================================== */
    function renderTimer() {
      viewContainer.innerHTML = '';

      if (state.timer.running || state.timer.paused) {
        renderTimerRunning();
      } else {
        renderTimerSetup();
      }
    }

    function renderTimerSetup() {
      const wrap = h('div', { className: 'bolt-tk-timer-setup bolt-tk-fade-in' });

      // Pre-fill with last used duration (or default 25m)
      const lastMs = state.timer.duration || 25 * 60000;
      const lastTotalSec = Math.round(lastMs / 1000);
      const defH = Math.floor(lastTotalSec / 3600);
      const defM = Math.floor((lastTotalSec % 3600) / 60);
      const defS = lastTotalSec % 60;

      // Time picker
      const picker = h('div', { className: 'bolt-tk-time-picker' });
      const hInput = h('input', {
        className: 'bolt-tk-time-input', type: 'number', min: '0', max: '23', value: String(defH), placeholder: 'HH',
      });
      const mInput = h('input', {
        className: 'bolt-tk-time-input', type: 'number', min: '0', max: '59', value: String(defM), placeholder: 'MM',
      });
      const sInput = h('input', {
        className: 'bolt-tk-time-input', type: 'number', min: '0', max: '59', value: String(defS), placeholder: 'SS',
      });
      picker.append(hInput, h('span', { className: 'bolt-tk-time-sep' }, ':'), mInput,
        h('span', { className: 'bolt-tk-time-sep' }, ':'), sInput);
      wrap.appendChild(picker);

      // Quick presets
      const presets = h('div', { className: 'bolt-tk-presets' });
      const presetValues = [
        { label: '5m', s: 300 }, { label: '10m', s: 600 }, { label: '15m', s: 900 },
        { label: '25m', s: 1500 }, { label: '30m', s: 1800 }, { label: '1h', s: 3600 },
      ];
      for (const p of presetValues) {
        presets.appendChild(h('button', {
          className: 'bolt-tk-preset',
          onClick: () => {
            const mins = Math.floor(p.s / 60);
            const hrs = Math.floor(mins / 60);
            hInput.value = hrs;
            mInput.value = mins % 60;
            sInput.value = p.s % 60;
          },
        }, p.label));
      }
      wrap.appendChild(presets);

      // Pomodoro toggle
      const pomoRow = h('div', { className: 'bolt-tk-pomo-row', style: { marginTop: '8px' } });
      const pomoLabel = h('label', { className: 'bolt-tk-toggle' });
      const pomoCheck = h('input', { type: 'checkbox' });
      if (state.timer.pomodoroMode) pomoCheck.checked = true;
      pomoCheck.addEventListener('change', () => { state.timer.pomodoroMode = pomoCheck.checked; });
      pomoLabel.append(pomoCheck, h('span', { className: 'bolt-tk-toggle-track' }), h('span', { className: 'bolt-tk-toggle-knob' }));
      pomoRow.append(h('span', null, 'Pomodoro Mode'), pomoLabel);
      wrap.appendChild(pomoRow);

      // Start button
      wrap.appendChild(h('button', {
        className: 'bolt-tk-btn bolt-tk-btn-primary bolt-tk-btn-lg',
        style: { marginTop: '12px' },
        onClick: () => {
          const totalSec = (parseInt(hInput.value) || 0) * 3600 + (parseInt(mInput.value) || 0) * 60 + (parseInt(sInput.value) || 0);
          if (totalSec <= 0) return;
          // Request notification permission on user gesture (Start click)
          if (state.settings.notify && 'Notification' in window && Notification.permission === 'default') {
            Notification.requestPermission();
          }
          startTimer(totalSec * 1000);
        },
      }, '\u25B6 Start'));

      viewContainer.appendChild(wrap);
    }

    function renderTimerRunning() {
      const wrap = h('div', { className: 'bolt-tk-timer-setup bolt-tk-fade-in' });

      // Progress ring
      const ringContainer = h('div', { className: 'bolt-tk-progress-ring-container' });
      const NS = 'http://www.w3.org/2000/svg';
      const ringSvg = document.createElementNS(NS, 'svg');
      ringSvg.setAttribute('viewBox', '0 0 200 200');
      ringSvg.setAttribute('class', 'bolt-tk-progress-ring-svg');

      const R = 90;
      const C = Math.PI * 2 * R;

      const bgCircle = document.createElementNS(NS, 'circle');
      bgCircle.setAttribute('cx', '100'); bgCircle.setAttribute('cy', '100'); bgCircle.setAttribute('r', R);
      bgCircle.setAttribute('class', 'bolt-tk-progress-ring-bg');
      ringSvg.appendChild(bgCircle);

      const fgCircle = document.createElementNS(NS, 'circle');
      fgCircle.setAttribute('cx', '100'); fgCircle.setAttribute('cy', '100'); fgCircle.setAttribute('r', R);
      fgCircle.setAttribute('class', 'bolt-tk-progress-ring-fg');
      fgCircle.setAttribute('stroke-dasharray', C);
      fgCircle.setAttribute('stroke-dashoffset', '0');
      ringSvg.appendChild(fgCircle);

      ringContainer.appendChild(ringSvg);

      const timeDisplay = h('div', { className: 'bolt-tk-progress-inner-time' });
      ringContainer.appendChild(timeDisplay);
      wrap.appendChild(ringContainer);

      // Pomodoro indicator
      if (state.timer.pomodoroMode) {
        const pomoRow = h('div', { className: 'bolt-tk-pomo-row' });
        const focusLabel = state.timer.pomodoroOnBreak ? 'Break'
          : state.timer.taskName || `Focus #${state.timer.pomodoroSession + 1}`;
        pomoRow.appendChild(h('span', { className: 'bolt-tk-pomo-label' }, focusLabel));
        for (let i = 0; i < 4; i++) {
          const dot = h('span', { className: 'bolt-tk-pomo-dot' });
          if (i < state.timer.pomodoroSession) dot.classList.add('bolt-tk-filled');
          pomoRow.appendChild(dot);
        }
        wrap.appendChild(pomoRow);
      }

      // Buttons
      const btnRow = h('div', { className: 'bolt-tk-btn-row' });
      if (state.timer.running) {
        btnRow.appendChild(h('button', { className: 'bolt-tk-btn bolt-tk-btn-ghost', onClick: pauseTimer }, '\u23F8 Pause'));
      } else {
        btnRow.appendChild(h('button', { className: 'bolt-tk-btn bolt-tk-btn-primary', onClick: resumeTimer }, '\u25B6 Resume'));
      }
      btnRow.appendChild(h('button', { className: 'bolt-tk-btn bolt-tk-btn-danger', onClick: cancelTimer }, '\u25A0 Cancel'));
      wrap.appendChild(btnRow);

      viewContainer.appendChild(wrap);

      // Tick
      let tickCompleted = false;
      function tick() {
        if (destroyed || state.activeView !== 'timer') return;
        let remaining = state.timer.remaining;
        if (state.timer.running) {
          remaining = state.timer.duration - (Date.now() - state.timer.startTime);
          state.timer.remaining = remaining;
        }

        if (remaining <= 0) {
          if (tickCompleted) return;
          tickCompleted = true;
          remaining = 0;
          state.timer.running = false;
          state.timer.paused = false;
          playAlarm();
          // Browser notification
          if (state.settings.notify && 'Notification' in window && Notification.permission === 'granted') {
            const task = state.timer.taskName;
            const label = state.timer.pomodoroMode
              ? (state.timer.pomodoroOnBreak ? 'Break over!' : task ? `${task} — done!` : 'Focus session complete!')
              : task ? `${task} — done!` : 'Timer finished!';
            try { new Notification('Timekeeper', { body: label, icon: '/static/logo/Lechler_Company-Logo.svg' }); } catch (_) {}
          }
          handleTimerComplete();
          return;
        }

        const progress = 1 - (remaining / state.timer.duration);
        const offset = C * progress;
        fgCircle.setAttribute('stroke-dashoffset', offset);
        timeDisplay.textContent = fmtHMS(remaining);

        if (state.timer.running) {
          timerInterval = requestAnimationFrame(tick);
        }
      }

      if (timerInterval) cancelAnimationFrame(timerInterval);
      tick();
      if (state.timer.running) timerInterval = requestAnimationFrame(tick);
    }

    function startTimer(durationMs) {
      state.timer.duration = durationMs;
      state.timer.remaining = durationMs;
      state.timer.startTime = Date.now();
      state.timer.running = true;
      state.timer.paused = false;
      if (!state.timer.pomodoroMode) {
        state.timer.pomodoroSession = 0;
        state.timer.pomodoroOnBreak = false;
      }
      if (state.timer.pomodoroMode && context.lockClose) context.lockClose();
      saveState();
      renderTimer();
    }

    function pauseTimer() {
      state.timer.remaining = state.timer.duration - (Date.now() - state.timer.startTime);
      state.timer.running = false;
      state.timer.paused = true;
      if (timerInterval) { cancelAnimationFrame(timerInterval); timerInterval = null; }
      saveState();
      renderTimer();
    }

    function resumeTimer() {
      state.timer.startTime = Date.now() - (state.timer.duration - state.timer.remaining);
      state.timer.running = true;
      state.timer.paused = false;
      saveState();
      renderTimer();
    }

    function cancelTimer() {
      state.timer.running = false;
      state.timer.paused = false;
      state.timer.pomodoroSession = 0;
      state.timer.pomodoroOnBreak = false;
      state.timer.taskName = '';
      if (timerInterval) { cancelAnimationFrame(timerInterval); timerInterval = null; }
      if (context.unlockClose) context.unlockClose();
      saveState();
      renderTimer();
    }

    function handleTimerComplete() {
      if (state.timer.pomodoroMode) {
        if (!state.timer.pomodoroOnBreak) {
          state.timer.pomodoroSession++;
          state.timer.pomodoroOnBreak = true;
          const isLong = state.timer.pomodoroSession % 4 === 0;
          const breakMs = (isLong ? state.settings.longBreakDuration : state.settings.breakDuration) * 60000;
          setTimeout(() => startTimer(breakMs), 500);
        } else {
          state.timer.pomodoroOnBreak = false;
          const workMs = state.settings.pomoDuration * 60000;
          setTimeout(() => startTimer(workMs), 500);
        }
      } else {
        state.timer.running = false;
        state.timer.paused = false;
        if (context.unlockClose) context.unlockClose();
        saveState();
        renderTimer();
      }
    }

    /* ==================================================================
     *  WORK LOG VIEW
     * ================================================================== */
    function renderWorkLog() {
      viewContainer.innerHTML = '';
      const wrap = h('div', { className: 'bolt-tk-fade-in' });

      // Input row
      const inputRow = h('div', { className: 'bolt-tk-wl-input-row' });
      const taskInput = h('input', {
        className: 'bolt-tk-input',
        placeholder: 'What are you working on?',
        value: state.wl.currentTask,
      });
      taskInput.addEventListener('input', () => { state.wl.currentTask = taskInput.value; });

      const projSelect = h('select', { className: 'bolt-tk-select' });
      projSelect.appendChild(h('option', { value: '' }, 'No project'));
      for (const p of state.wl.projects) {
        const opt = h('option', { value: p.id }, p.name);
        if (state.wl.currentProject === p.id) opt.selected = true;
        projSelect.appendChild(opt);
      }
      projSelect.addEventListener('change', () => { state.wl.currentProject = projSelect.value; });

      inputRow.append(taskInput, projSelect);
      wrap.appendChild(inputRow);

      // Track button
      const trackBtn = h('button', {
        className: `bolt-tk-track-btn ${state.wl.tracking ? 'bolt-tk-stop' : 'bolt-tk-start'}`,
        onClick: () => toggleTracking(taskInput.value, projSelect.value),
      }, state.wl.tracking ? '\u25A0  Stop Tracking' : '\u25B6  Start Tracking');
      wrap.appendChild(trackBtn);

      // Current tracking info
      if (state.wl.tracking) {
        const elapsed = Date.now() - state.wl.trackingStart;
        const currentInfo = h('div', {
          style: {
            textAlign: 'center', marginTop: '12px', fontSize: '12px', color: 'var(--tk-fg2)',
          },
        }, `Tracking since ${fmtTime(state.wl.trackingStart, state.settings.use24h)} (${fmtHMS(elapsed)})`);
        wrap.appendChild(currentInfo);
      }

      // Today's entries
      const tk = todayKey();
      const todayEntries = (state.wl.entries[tk] || []).slice().reverse();

      wrap.appendChild(h('div', { className: 'bolt-tk-section-title' }, "Today's Log"));

      if (todayEntries.length === 0) {
        wrap.appendChild(h('div', {
          style: { textAlign: 'center', padding: '20px', color: 'var(--tk-fg2)', fontSize: '13px' },
        }, 'No entries yet. Start tracking to log your work.'));
      } else {
        for (const entry of todayEntries) {
          const proj = state.wl.projects.find(p => p.id === entry.projectId);
          const entryEl = h('div', { className: 'bolt-tk-entry' });

          entryEl.appendChild(h('div', {
            className: 'bolt-tk-entry-color',
            style: { background: proj ? proj.color : 'var(--tk-fg2)' },
          }));

          const info = h('div', { className: 'bolt-tk-entry-info' });
          info.appendChild(h('div', { className: 'bolt-tk-entry-name' }, entry.task || (proj ? proj.name : 'Untitled')));
          info.appendChild(h('div', { className: 'bolt-tk-entry-times' },
            `${fmtTime(entry.start, state.settings.use24h)} - ${fmtTime(entry.end, state.settings.use24h)}`));
          entryEl.appendChild(info);

          entryEl.appendChild(h('div', { className: 'bolt-tk-entry-duration' }, fmtHMS(entry.end - entry.start)));

          const delBtn = h('button', {
            className: 'bolt-tk-entry-delete',
            onClick: (e) => {
              e.stopPropagation();
              const entries = state.wl.entries[tk];
              const idx = entries.findIndex(en => en.id === entry.id);
              if (idx >= 0) { entries.splice(idx, 1); saveState(); renderWorkLog(); }
            },
          }, '\u2715');
          entryEl.appendChild(delBtn);

          // Click to edit
          entryEl.addEventListener('click', () => showEditEntry(entry, tk));

          wrap.appendChild(entryEl);
        }
      }

      // Daily total
      let totalMs = 0;
      for (const e of (state.wl.entries[tk] || [])) totalMs += (e.end - e.start);
      if (state.wl.tracking) totalMs += Date.now() - state.wl.trackingStart;

      const totalRow = h('div', { className: 'bolt-tk-daily-total' },
        h('span', null, 'Daily Total'),
        h('span', { className: 'bolt-tk-daily-total-value' }, fmtHMS(totalMs)),
      );
      wrap.appendChild(totalRow);

      // Weekly chart
      wrap.appendChild(renderWeeklyChart());

      viewContainer.appendChild(wrap);
    }

    function toggleTracking(task, projectId) {
      if (state.wl.tracking) {
        // Stop tracking, add entry
        const now = Date.now();
        const tk = todayKey();
        if (!state.wl.entries[tk]) state.wl.entries[tk] = [];
        state.wl.entries[tk].push({
          id: uid(),
          task: state.wl.currentTask,
          projectId: state.wl.currentProject,
          start: state.wl.trackingStart,
          end: now,
        });
        state.wl.tracking = false;
        state.wl.trackingStart = 0;
      } else {
        // Start tracking
        state.wl.currentTask = task;
        state.wl.currentProject = projectId;
        state.wl.tracking = true;
        state.wl.trackingStart = Date.now();
      }
      saveState();
      updateTrackingBanner();
      renderWorkLog();
    }

    function showEditEntry(entry, dateKey) {
      // Create modal overlay
      const overlay = h('div', { className: 'bolt-tk-modal-overlay' });
      const modal = h('div', { className: 'bolt-tk-modal bolt-tk-fade-in' });
      modal.appendChild(h('div', { className: 'bolt-tk-modal-title' }, 'Edit Entry'));

      // Task field
      const taskField = h('div', { className: 'bolt-tk-modal-field' });
      taskField.appendChild(h('label', null, 'Task'));
      const taskIn = h('input', { className: 'bolt-tk-input', value: entry.task || '' });
      taskField.appendChild(taskIn);
      modal.appendChild(taskField);

      // Project field
      const projField = h('div', { className: 'bolt-tk-modal-field' });
      projField.appendChild(h('label', null, 'Project'));
      const projSel = h('select', { className: 'bolt-tk-select', style: { width: '100%' } });
      projSel.appendChild(h('option', { value: '' }, 'No project'));
      for (const p of state.wl.projects) {
        const opt = h('option', { value: p.id }, p.name);
        if (p.id === entry.projectId) opt.selected = true;
        projSel.appendChild(opt);
      }
      projField.appendChild(projSel);
      modal.appendChild(projField);

      // Start time
      const startField = h('div', { className: 'bolt-tk-modal-field' });
      startField.appendChild(h('label', null, 'Start'));
      const sd = new Date(entry.start);
      const startIn = h('input', {
        className: 'bolt-tk-input', type: 'time',
        value: `${pad(sd.getHours())}:${pad(sd.getMinutes())}`,
      });
      startField.appendChild(startIn);
      modal.appendChild(startField);

      // End time
      const endField = h('div', { className: 'bolt-tk-modal-field' });
      endField.appendChild(h('label', null, 'End'));
      const ed = new Date(entry.end);
      const endIn = h('input', {
        className: 'bolt-tk-input', type: 'time',
        value: `${pad(ed.getHours())}:${pad(ed.getMinutes())}`,
      });
      endField.appendChild(endIn);
      modal.appendChild(endField);

      // Actions
      const actions = h('div', { className: 'bolt-tk-modal-actions' });
      actions.appendChild(h('button', {
        className: 'bolt-tk-btn bolt-tk-btn-ghost bolt-tk-btn-sm',
        onClick: () => overlay.remove(),
      }, 'Cancel'));
      actions.appendChild(h('button', {
        className: 'bolt-tk-btn bolt-tk-btn-primary bolt-tk-btn-sm',
        onClick: () => {
          const entries = state.wl.entries[dateKey];
          const idx = entries.findIndex(e => e.id === entry.id);
          if (idx >= 0) {
            const [sh, sm] = startIn.value.split(':').map(Number);
            const [eh, em] = endIn.value.split(':').map(Number);
            const newStart = new Date(entry.start);
            newStart.setHours(sh, sm, 0, 0);
            const newEnd = new Date(entry.end);
            newEnd.setHours(eh, em, 0, 0);
            entries[idx] = {
              ...entry,
              task: taskIn.value,
              projectId: projSel.value,
              start: newStart.getTime(),
              end: newEnd.getTime(),
            };
            saveState();
          }
          overlay.remove();
          renderWorkLog();
        },
      }, 'Save'));
      modal.appendChild(actions);

      overlay.appendChild(modal);
      overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });
      root.appendChild(overlay);
    }

    function renderWeeklyChart() {
      const weekWrap = h('div', { className: 'bolt-tk-weekly' });
      weekWrap.appendChild(h('div', { className: 'bolt-tk-section-title' }, 'This Week'));

      const weekDates = getWeekDates();
      const tk = todayKey();
      const hoursPerDay = weekDates.map(d => {
        const entries = state.wl.entries[d] || [];
        let total = 0;
        for (const e of entries) total += (e.end - e.start);
        // Add current tracking if today
        if (d === tk && state.wl.tracking) total += Date.now() - state.wl.trackingStart;
        return total / 3600000;
      });
      const maxHours = Math.max(...hoursPerDay, 1);

      const chart = h('div', { className: 'bolt-tk-chart' });
      for (let i = 0; i < 7; i++) {
        const barWrap = h('div', { className: 'bolt-tk-chart-bar-wrap' });
        if (hoursPerDay[i] > 0) {
          barWrap.appendChild(h('div', { className: 'bolt-tk-chart-hours' }, hoursPerDay[i].toFixed(1)));
        }
        const barHeight = Math.max(2, (hoursPerDay[i] / maxHours) * 60);
        const bar = h('div', {
          className: `bolt-tk-chart-bar${weekDates[i] === tk ? ' bolt-tk-today' : ''}`,
          style: { height: barHeight + 'px' },
        });
        barWrap.appendChild(bar);
        barWrap.appendChild(h('div', { className: 'bolt-tk-chart-label' }, DAY_LABELS[i]));
        chart.appendChild(barWrap);
      }
      weekWrap.appendChild(chart);
      return weekWrap;
    }

    /* ==================================================================
     *  SETTINGS VIEW
     * ================================================================== */
    function renderSettings() {
      viewContainer.innerHTML = '';
      const wrap = h('div', { className: 'bolt-tk-fade-in' });

      // Display
      const displayGroup = h('div', { className: 'bolt-tk-settings-group' });
      displayGroup.appendChild(h('div', { className: 'bolt-tk-settings-group-title' }, 'Display'));

      displayGroup.appendChild(settingRow('Default Clock', 'Analog or digital stopwatch display',
        makeToggle(state.settings.displayMode === 'analog', (v) => {
          state.settings.displayMode = v ? 'analog' : 'digital';
          state.sw.displayMode = state.settings.displayMode;
          saveState();
        }, 'Analog', 'Digital')
      ));

      displayGroup.appendChild(settingRow('24-Hour Format', 'Use 24h time in work log',
        makeToggleSwitch(state.settings.use24h, (v) => { state.settings.use24h = v; saveState(); })
      ));
      wrap.appendChild(displayGroup);

      // Pomodoro
      const pomoGroup = h('div', { className: 'bolt-tk-settings-group' });
      pomoGroup.appendChild(h('div', { className: 'bolt-tk-settings-group-title' }, 'Pomodoro'));

      pomoGroup.appendChild(settingRow('Focus Duration', 'Minutes per work session',
        makeNumberInput(state.settings.pomoDuration, 1, 120, (v) => { state.settings.pomoDuration = v; saveState(); })
      ));
      pomoGroup.appendChild(settingRow('Break Duration', 'Minutes per short break',
        makeNumberInput(state.settings.breakDuration, 1, 30, (v) => { state.settings.breakDuration = v; saveState(); })
      ));
      pomoGroup.appendChild(settingRow('Long Break', 'Minutes per long break (every 4)',
        makeNumberInput(state.settings.longBreakDuration, 1, 60, (v) => { state.settings.longBreakDuration = v; saveState(); })
      ));
      wrap.appendChild(pomoGroup);

      // Reminders
      const remGroup = h('div', { className: 'bolt-tk-settings-group' });
      remGroup.appendChild(h('div', { className: 'bolt-tk-settings-group-title' }, 'Reminders'));
      remGroup.appendChild(settingRow('Idle Reminder', 'Minutes before reminder when tracking',
        makeNumberInput(state.settings.reminderInterval, 0, 180, (v) => { state.settings.reminderInterval = v; saveState(); })
      ));
      remGroup.appendChild(settingRow('Browser Notifications', 'Show a notification when timer ends',
        makeToggleSwitch(state.settings.notify, (v) => {
          state.settings.notify = v;
          saveState();
          if (v && 'Notification' in window && Notification.permission !== 'granted') {
            Notification.requestPermission();
          }
        })
      ));
      wrap.appendChild(remGroup);

      // Projects
      const projGroup = h('div', { className: 'bolt-tk-settings-group' });
      projGroup.appendChild(h('div', { className: 'bolt-tk-settings-group-title' }, 'Projects'));

      for (const proj of state.wl.projects) {
        const item = h('div', { className: 'bolt-tk-project-item' });

        const colorInput = h('input', {
          type: 'color', value: proj.color,
          style: { width: '28px', height: '28px', border: 'none', padding: '0', cursor: 'pointer', background: 'transparent', borderRadius: '50%' },
        });
        colorInput.addEventListener('change', () => { proj.color = colorInput.value; saveState(); });
        item.appendChild(colorInput);

        const nameInput = h('input', {
          className: 'bolt-tk-project-name-edit',
          value: proj.name,
        });
        nameInput.addEventListener('change', () => { proj.name = nameInput.value; saveState(); });
        item.appendChild(nameInput);

        item.appendChild(h('button', {
          className: 'bolt-tk-btn bolt-tk-btn-ghost bolt-tk-btn-icon-only bolt-tk-btn-sm',
          style: { width: '28px', height: '28px', fontSize: '14px' },
          onClick: () => {
            state.wl.projects = state.wl.projects.filter(p => p.id !== proj.id);
            saveState();
            renderSettings();
          },
        }, '\u2715'));

        projGroup.appendChild(item);
      }

      // Add project
      const addRow = h('div', { className: 'bolt-tk-add-project-row' });
      const addInput = h('input', { className: 'bolt-tk-input', placeholder: 'New project name...' });
      addRow.appendChild(addInput);
      addRow.appendChild(h('button', {
        className: 'bolt-tk-btn bolt-tk-btn-primary bolt-tk-btn-sm',
        onClick: () => {
          const name = addInput.value.trim();
          if (!name) return;
          state.wl.projects.push({
            id: uid(),
            name,
            color: DEFAULT_COLORS[state.wl.projects.length % DEFAULT_COLORS.length],
          });
          saveState();
          renderSettings();
        },
      }, '+ Add'));
      projGroup.appendChild(addRow);

      wrap.appendChild(projGroup);
      viewContainer.appendChild(wrap);
    }

    function settingRow(label, desc, control) {
      const row = h('div', { className: 'bolt-tk-setting-row' });
      const left = h('div');
      left.appendChild(h('div', { className: 'bolt-tk-setting-label' }, label));
      if (desc) left.appendChild(h('div', { className: 'bolt-tk-setting-desc' }, desc));
      row.append(left, control);
      return row;
    }

    function makeToggleSwitch(value, onChange) {
      const label = h('label', { className: 'bolt-tk-toggle' });
      const input = h('input', { type: 'checkbox' });
      if (value) input.checked = true;
      input.addEventListener('change', () => onChange(input.checked));
      label.append(input, h('span', { className: 'bolt-tk-toggle-track' }), h('span', { className: 'bolt-tk-toggle-knob' }));
      return label;
    }

    function makeToggle(isFirst, onChange, labelA, labelB) {
      const wrap = h('div', { className: 'bolt-tk-sw-toggle' });
      const btnA = h('button', {
        className: `bolt-tk-sw-toggle-btn${isFirst ? ' bolt-tk-active' : ''}`,
        onClick: () => { btnA.classList.add('bolt-tk-active'); btnB.classList.remove('bolt-tk-active'); onChange(true); },
      }, labelA);
      const btnB = h('button', {
        className: `bolt-tk-sw-toggle-btn${!isFirst ? ' bolt-tk-active' : ''}`,
        onClick: () => { btnB.classList.add('bolt-tk-active'); btnA.classList.remove('bolt-tk-active'); onChange(false); },
      }, labelB);
      wrap.append(btnA, btnB);
      return wrap;
    }

    function makeNumberInput(value, min, max, onChange) {
      const input = h('input', {
        className: 'bolt-tk-number-input', type: 'number',
        min: String(min), max: String(max), value: String(value),
      });
      input.addEventListener('change', () => {
        let v = parseInt(input.value) || min;
        v = Math.max(min, Math.min(max, v));
        input.value = v;
        onChange(v);
      });
      return input;
    }

    /* ==================================================================
     *  View Router
     * ================================================================== */
    function renderView() {
      if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
      if (timerInterval) { cancelAnimationFrame(timerInterval); timerInterval = null; }

      switch (state.activeView) {
        case 'stopwatch': renderStopwatch(); break;
        case 'timer': renderTimer(); break;
        case 'worklog': renderWorkLog(); break;
        case 'settings': renderSettings(); break;
        default: renderStopwatch();
      }
    }

    /* ==================================================================
     *  Global refresh interval (tracking banner, work log live timer)
     * ================================================================== */
    let globalTick = null;
    function startGlobalTick() {
      globalTick = setInterval(() => {
        if (destroyed) { clearInterval(globalTick); return; }
        updateTrackingBanner();

        // Idle reminder
        if (state.wl.tracking && state.settings.reminderInterval > 0) {
          const elapsed = Date.now() - state.wl.trackingStart;
          const intervalMs = state.settings.reminderInterval * 60000;
          if (elapsed > 0 && elapsed % intervalMs < 1000) {
            const proj = state.wl.projects.find(p => p.id === state.wl.currentProject);
            const taskName = state.wl.currentTask || (proj ? proj.name : 'a task');
            // Simple visual reminder via banner flash
            if (trackingBanner) {
              trackingBanner.style.background = 'var(--tk-warn)';
              setTimeout(() => {
                if (trackingBanner) trackingBanner.style.background = '';
              }, 2000);
            }
          }
        }
      }, 1000);
    }

    /* ==================================================================
     *  Keyboard shortcuts
     * ================================================================== */
    function handleKeydown(e) {
      // Don't capture when typing in inputs
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
      // Don't capture if modifier keys
      if (e.ctrlKey || e.metaKey || e.altKey) return;

      switch (e.key) {
        case ' ':
          e.preventDefault();
          if (state.activeView === 'stopwatch') {
            state.sw.running ? stopStopwatch() : startStopwatch();
          } else if (state.activeView === 'timer') {
            if (state.timer.running) pauseTimer();
            else if (state.timer.paused) resumeTimer();
          }
          break;
        case 'r':
        case 'R':
          if (state.activeView === 'stopwatch') resetStopwatch();
          break;
        case 'l':
        case 'L':
          if (state.activeView === 'stopwatch' && state.sw.running) lapStopwatch();
          break;
        case '1': switchView('stopwatch'); break;
        case '2': switchView('timer'); break;
        case '3': switchView('worklog'); break;
      }
    }

    /* ==================================================================
     *  Init & API
     * ================================================================== */
    async function init() {
      await loadState();
      buildChrome();
      switchView(state.activeView);
      startGlobalTick();
      root.addEventListener('keydown', handleKeydown);
      root.setAttribute('tabindex', '0');

      // Auto-start from preset (e.g. /focus → pomodoro, /focus 30m Design Review)
      if (args && args.preset === 'pomodoro' && !state.timer.running && !state.timer.paused) {
        const { durationMs, taskName } = parseTimerInput(args.input || '');
        state.timer.pomodoroMode = true;
        state.timer.taskName = taskName;
        startTimer(durationMs || state.settings.pomoDuration * 60000);
      }

      // Auto-start timer from input (e.g. "30m", "30s", "1h30m", "90")
      if (args?.input && !args?.preset && !state.timer.running && !state.timer.paused) {
        const durationMs = parseDuration(args.input);
        if (durationMs > 0) {
          state.activeView = 'timer';
          state.timer.pomodoroMode = false;
          state.timer.taskName = '';
          buildChrome();
          startTimer(durationMs);
        }
      }

      // Register for revive
      mem.registerRevive({
        app: 'timekeeper',
        args: { view: state.activeView },
      });
    }

    init();

    return {
      getState() {
        return {
          activeView: state.activeView,
          sw: { ...state.sw },
          timer: { ...state.timer },
          wl: { ...state.wl },
          settings: { ...state.settings },
        };
      },
      setState(newState) {
        if (newState.sw) Object.assign(state.sw, newState.sw);
        if (newState.timer) Object.assign(state.timer, newState.timer);
        if (newState.wl) Object.assign(state.wl, newState.wl);
        if (newState.settings) Object.assign(state.settings, newState.settings);
        if (newState.activeView) state.activeView = newState.activeView;
        buildChrome();
        renderView();
        saveState();
      },
      destroy() {
        destroyed = true;
        if (animFrame) cancelAnimationFrame(animFrame);
        if (timerInterval) cancelAnimationFrame(timerInterval);
        if (globalTick) clearInterval(globalTick);
        root.removeEventListener('keydown', handleKeydown);
        root.remove();
        mem.unregisterRevive();
      },
    };
  }

  /* ======================================================================
   *  Register
   * ====================================================================== */
  // View-specific sizes
  const VIEW_SIZES = {
    stopwatch: { w: 380, h: 520 },
    timer:     { w: 380, h: 520 },
    worklog:   { w: 460, h: 600 },
    settings:  { w: 400, h: 480 },
  };

  window._boltApps = window._boltApps || {};
  window._boltApps.timekeeper = {
    name: 'timekeeper',
    icon: 'timer',
    width: 380,
    height: 520,
    devices: ['desktop', 'tablet', 'mobile'],
    render: createApp,
  };
})();
