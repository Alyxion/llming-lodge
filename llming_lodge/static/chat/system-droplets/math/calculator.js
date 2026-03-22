/* ============================================================
 *  Bolt Mini-App: Calculator (Normal + Scientific + Programmer)
 *  Self-contained floating window calculator.
 * ============================================================ */
(function () {
  'use strict';

  const CSS_ID = 'bolt-calc-styles';
  const MODES = ['normal', 'scientific', 'programmer'];
  const SIZES = { normal: [340, 480], scientific: [420, 520], programmer: [420, 580] };

  /* ---------- safe math helpers ---------- */
  function factorial(n) {
    if (n < 0) return NaN;
    if (n === 0 || n === 1) return 1;
    if (n > 170) return Infinity;
    let r = 1;
    for (let i = 2; i <= n; i++) r *= i;
    return r;
  }

  /* ---- Standard Tokenizer & expression parser (NO eval) ---- */
  function tokenize(expr) {
    const tokens = [];
    let i = 0;
    const src = expr.replace(/\s+/g, '');
    while (i < src.length) {
      if (/[0-9.]/.test(src[i])) {
        let num = '';
        while (i < src.length && /[0-9.]/.test(src[i])) num += src[i++];
        tokens.push({ type: 'num', value: parseFloat(num) });
        continue;
      }
      const rest = src.slice(i);
      const fnMatch = rest.match(/^(asin|acos|atan|sin|cos|tan|log2|log10|log|ln|sqrt|cbrt|exp|abs|fact)/);
      if (fnMatch) {
        tokens.push({ type: 'fn', value: fnMatch[1] });
        i += fnMatch[1].length;
        continue;
      }
      if (rest.startsWith('pi') || rest.startsWith('\u03C0')) {
        tokens.push({ type: 'num', value: Math.PI });
        i += rest.startsWith('pi') ? 2 : 1;
        continue;
      }
      if (rest[0] === 'e' && (i + 1 >= src.length || !/[a-z]/i.test(src[i + 1]))) {
        tokens.push({ type: 'num', value: Math.E });
        i++;
        continue;
      }
      if ('+-*/%^'.includes(src[i])) {
        tokens.push({ type: 'op', value: src[i] });
        i++;
        continue;
      }
      if (src[i] === '(' || src[i] === ')') {
        tokens.push({ type: src[i], value: src[i] });
        i++;
        continue;
      }
      i++;
    }
    return tokens;
  }

  function parseExpr(tokens, pos, degMode) {
    let [left, p] = parseTerm(tokens, pos, degMode);
    while (p < tokens.length && tokens[p].type === 'op' && (tokens[p].value === '+' || tokens[p].value === '-')) {
      const op = tokens[p].value; p++;
      let [right, np] = parseTerm(tokens, p, degMode);
      left = op === '+' ? left + right : left - right;
      p = np;
    }
    return [left, p];
  }

  function parseTerm(tokens, pos, degMode) {
    let [left, p] = parsePower(tokens, pos, degMode);
    while (p < tokens.length && tokens[p].type === 'op' && ('*/%'.includes(tokens[p].value))) {
      const op = tokens[p].value; p++;
      let [right, np] = parsePower(tokens, p, degMode);
      if (op === '*') left *= right;
      else if (op === '/') left = right === 0 ? (left === 0 ? NaN : (left > 0 ? Infinity : -Infinity)) : left / right;
      else left = right === 0 ? NaN : left % right;
      p = np;
    }
    return [left, p];
  }

  function parsePower(tokens, pos, degMode) {
    let [base, p] = parseUnary(tokens, pos, degMode);
    if (p < tokens.length && tokens[p].type === 'op' && tokens[p].value === '^') {
      p++;
      let [exp, np] = parsePower(tokens, p, degMode);
      base = Math.pow(base, exp);
      p = np;
    }
    return [base, p];
  }

  function parseUnary(tokens, pos, degMode) {
    if (pos < tokens.length && tokens[pos].type === 'op' && (tokens[pos].value === '+' || tokens[pos].value === '-')) {
      const sign = tokens[pos].value; pos++;
      let [val, p] = parseUnary(tokens, pos, degMode);
      return [sign === '-' ? -val : val, p];
    }
    if (pos < tokens.length && tokens[pos].type === 'fn') {
      const fn = tokens[pos].value; pos++;
      let [arg, p] = parseAtom(tokens, pos, degMode);
      const toRad = degMode ? (v) => v * Math.PI / 180 : (v) => v;
      const fromRad = degMode ? (v) => v * 180 / Math.PI : (v) => v;
      switch (fn) {
        case 'sin':   return [Math.sin(toRad(arg)), p];
        case 'cos':   return [Math.cos(toRad(arg)), p];
        case 'tan':   return [Math.tan(toRad(arg)), p];
        case 'asin':  return [fromRad(Math.asin(arg)), p];
        case 'acos':  return [fromRad(Math.acos(arg)), p];
        case 'atan':  return [fromRad(Math.atan(arg)), p];
        case 'ln':    return [Math.log(arg), p];
        case 'log':
        case 'log10': return [Math.log10(arg), p];
        case 'log2':  return [Math.log2(arg), p];
        case 'sqrt':  return [Math.sqrt(arg), p];
        case 'cbrt':  return [Math.cbrt(arg), p];
        case 'exp':   return [Math.exp(arg), p];
        case 'abs':   return [Math.abs(arg), p];
        case 'fact':  return [factorial(Math.round(arg)), p];
        default:      return [NaN, p];
      }
    }
    return parseAtom(tokens, pos, degMode);
  }

  function parseAtom(tokens, pos, degMode) {
    if (pos >= tokens.length) return [NaN, pos];
    const t = tokens[pos];
    if (t.type === 'num') return [t.value, pos + 1];
    if (t.type === '(') {
      let [val, p] = parseExpr(tokens, pos + 1, degMode);
      if (p < tokens.length && tokens[p].type === ')') p++;
      return [val, p];
    }
    return [NaN, pos + 1];
  }

  function safeEval(expression, degMode) {
    try {
      const sanitized = expression
        .replace(/\u00D7/g, '*')
        .replace(/\u00F7/g, '/')
        .replace(/\u03C0/g, 'pi');
      const tokens = tokenize(sanitized);
      if (tokens.length === 0) return '';
      const [result] = parseExpr(tokens, 0, degMode);
      return result;
    } catch {
      return NaN;
    }
  }

  /* ---- Programmer Tokenizer & Parser ---- */
  function tokenizeProg(expr, base) {
    const tokens = [];
    let i = 0;
    const src = expr.replace(/\s+/g, '');
    while (i < src.length) {
      // Two-char operators
      if (i + 1 < src.length) {
        const two = src[i] + src[i + 1];
        if (two === '<<' || two === '>>') {
          tokens.push({ type: 'op', value: two });
          i += 2;
          continue;
        }
      }
      // Number in current base
      const digitRe = base === 16 ? /[0-9a-fA-F]/ : base === 8 ? /[0-7]/ : base === 2 ? /[01]/ : /[0-9]/;
      if (digitRe.test(src[i])) {
        let num = '';
        while (i < src.length && digitRe.test(src[i])) num += src[i++];
        tokens.push({ type: 'num', value: parseInt(num, base) });
        continue;
      }
      if ('+-*/%&|^~'.includes(src[i])) {
        tokens.push({ type: 'op', value: src[i] });
        i++;
        continue;
      }
      if (src[i] === '(' || src[i] === ')') {
        tokens.push({ type: src[i], value: src[i] });
        i++;
        continue;
      }
      i++;
    }
    return tokens;
  }

  /* Operator-precedence parser (Pratt-style) */
  const PROG_OPS = {
    '|':  { prec: 1, fn: (a, b) => (a | b) },
    '^':  { prec: 2, fn: (a, b) => (a ^ b) },
    '&':  { prec: 3, fn: (a, b) => (a & b) },
    '<<': { prec: 4, fn: (a, b) => (a << b) },
    '>>': { prec: 4, fn: (a, b) => (a >> b) },
    '+':  { prec: 5, fn: (a, b) => (a + b) },
    '-':  { prec: 5, fn: (a, b) => (a - b) },
    '*':  { prec: 6, fn: (a, b) => (a * b) },
    '/':  { prec: 6, fn: (a, b) => b === 0 ? NaN : Math.trunc(a / b) },
    '%':  { prec: 6, fn: (a, b) => b === 0 ? NaN : (a % b) },
  };

  function parseProgExpr(tokens, pos, minPrec) {
    let [left, p] = parseProgUnary(tokens, pos);
    while (p < tokens.length && tokens[p].type === 'op' && PROG_OPS[tokens[p].value] && PROG_OPS[tokens[p].value].prec >= minPrec) {
      const op = tokens[p].value; p++;
      const prec = PROG_OPS[op].prec;
      let [right, np] = parseProgExpr(tokens, p, prec + 1);
      left = PROG_OPS[op].fn(left, right);
      p = np;
    }
    return [left, p];
  }

  function parseProgUnary(tokens, pos) {
    if (pos < tokens.length && tokens[pos].type === 'op') {
      if (tokens[pos].value === '~') { pos++; const [v, p] = parseProgUnary(tokens, pos); return [~v, p]; }
      if (tokens[pos].value === '-') { pos++; const [v, p] = parseProgUnary(tokens, pos); return [-v, p]; }
      if (tokens[pos].value === '+') { pos++; return parseProgUnary(tokens, pos); }
    }
    return parseProgAtom(tokens, pos);
  }

  function parseProgAtom(tokens, pos) {
    if (pos >= tokens.length) return [NaN, pos];
    const t = tokens[pos];
    if (t.type === 'num') return [t.value, pos + 1];
    if (t.type === '(') {
      let [val, p] = parseProgExpr(tokens, pos + 1, 0);
      if (p < tokens.length && tokens[p].type === ')') p++;
      return [val, p];
    }
    return [NaN, pos + 1];
  }

  function safeEvalProg(expression, base) {
    try {
      const sanitized = expression
        .replace(/\u00D7/g, '*')
        .replace(/\u00F7/g, '/')
        .replace(/\u2212/g, '-');
      const tokens = tokenizeProg(sanitized, base);
      if (tokens.length === 0) return '';
      const [result] = parseProgExpr(tokens, 0, 0);
      return Math.trunc(result) | 0;
    } catch {
      return NaN;
    }
  }

  /* ---------- format helpers ---------- */
  function formatNum(n) {
    if (n === '' || n === undefined || n === null) return '0';
    if (typeof n === 'string') return n;
    if (!isFinite(n)) return isNaN(n) ? 'Error' : (n > 0 ? '\u221E' : '-\u221E');
    if (Math.abs(n) > 1e15 || (Math.abs(n) < 1e-10 && n !== 0)) return n.toExponential(8);
    const s = parseFloat(n.toPrecision(12));
    return String(s);
  }

  function formatInBase(n, base) {
    if (n === '' || n === undefined || n === null) return '0';
    if (typeof n === 'string') return n;
    if (!isFinite(n) || isNaN(n)) return isNaN(n) ? 'Error' : (n > 0 ? '\u221E' : '-\u221E');
    const v = Math.trunc(n) | 0;
    const u = v >>> 0;
    if (base === 16) return u.toString(16).toUpperCase();
    if (base === 8) return u.toString(8);
    if (base === 2) return u.toString(2);
    return v.toString(10);
  }

  function allBases(n) {
    if (typeof n !== 'number' || isNaN(n) || !isFinite(n)) return { hex: '\u2014', dec: '\u2014', oct: '\u2014', bin: '\u2014' };
    const v = Math.trunc(n) | 0;
    const u = v >>> 0;
    let bin = u.toString(2);
    // Group binary in 4-bit nibbles
    bin = bin.padStart(Math.ceil(bin.length / 4) * 4, '0').replace(/(.{4})(?=.)/g, '$1 ');
    return {
      hex: u.toString(16).toUpperCase(),
      dec: v.toString(10),
      oct: u.toString(8),
      bin: bin,
    };
  }

  /* ---------- inject CSS ---------- */
  function injectStyles() {
    if (document.getElementById(CSS_ID)) return;
    const style = document.createElement('style');
    style.id = CSS_ID;
    style.textContent = `
      /* ---- theme variables ---- */
      .bolt-calc-root {
        --bc-bg:        #1a1a2e;
        --bc-surface:   rgba(255,255,255,0.06);
        --bc-surface-h: rgba(255,255,255,0.10);
        --bc-text:      #f0f0f0;
        --bc-text-dim:  #888;
        --bc-accent:    #ff9500;
        --bc-accent-g:  linear-gradient(135deg, #ff9500, #ff6a00);
        --bc-eq-bg:     linear-gradient(135deg, #ff9500, #ff6a00);
        --bc-num-bg:    rgba(255,255,255,0.08);
        --bc-num-h:     rgba(255,255,255,0.14);
        --bc-fn-bg:     rgba(255,255,255,0.04);
        --bc-fn-h:      rgba(255,255,255,0.09);
        --bc-border:    rgba(255,255,255,0.06);
        --bc-shadow:    0 2px 8px rgba(0,0,0,0.3);
        --bc-radius:    12px;
      }
      .bolt-calc-root.bolt-calc-light {
        --bc-bg:        #f2f2f7;
        --bc-surface:   rgba(0,0,0,0.04);
        --bc-surface-h: rgba(0,0,0,0.08);
        --bc-text:      #1c1c1e;
        --bc-text-dim:  #6e6e73;
        --bc-num-bg:    #ffffff;
        --bc-num-h:     #e8e8ed;
        --bc-fn-bg:     rgba(0,0,0,0.06);
        --bc-fn-h:      rgba(0,0,0,0.10);
        --bc-border:    rgba(0,0,0,0.06);
        --bc-shadow:    0 2px 8px rgba(0,0,0,0.08);
      }

      .bolt-calc-root {
        display: flex; flex-direction: column;
        width: 100%; height: 100%;
        background: var(--bc-bg);
        color: var(--bc-text);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        user-select: none;
        overflow: hidden;
        border-radius: 8px;
        box-sizing: border-box;
        padding: 10px;
        gap: 8px;
      }

      /* ---- mode toggle ---- */
      .bolt-calc-toggle-bar {
        display: flex; align-items: center; justify-content: center; gap: 8px;
        padding: 2px 0 4px;
      }
      .bolt-calc-pill {
        position: relative; display: flex;
        background: var(--bc-surface);
        border-radius: 20px; overflow: hidden;
        border: 1px solid var(--bc-border);
      }
      .bolt-calc-pill button {
        position: relative; z-index: 1;
        background: none; border: none; color: var(--bc-text-dim);
        font-size: 11px; font-weight: 600;
        padding: 5px 12px; cursor: pointer;
        transition: color 0.2s;
      }
      .bolt-calc-pill button.active { color: #fff; }
      .bolt-calc-pill-bg {
        position: absolute; top: 2px; bottom: 2px;
        width: calc(33.33% - 1px); border-radius: 18px;
        background: var(--bc-accent);
        transition: left 0.25s cubic-bezier(.4,0,.2,1);
      }
      .bolt-calc-pill-bg.pos-0 { left: 2px; }
      .bolt-calc-pill-bg.pos-1 { left: 33.33%; }
      .bolt-calc-pill-bg.pos-2 { left: calc(66.66% - 1px); }

      .bolt-calc-deg-rad {
        font-size: 10px; font-weight: 700;
        color: var(--bc-accent);
        background: var(--bc-surface);
        border: 1px solid var(--bc-border);
        border-radius: 6px; padding: 3px 8px;
        cursor: pointer; transition: background 0.15s;
      }
      .bolt-calc-deg-rad:hover { background: var(--bc-surface-h); }

      /* ---- display ---- */
      .bolt-calc-display {
        background: var(--bc-surface);
        border: 1px solid var(--bc-border);
        border-radius: var(--bc-radius);
        padding: 12px 16px 10px;
        min-height: 60px;
        display: flex; flex-direction: column; justify-content: flex-end;
        overflow: hidden;
        flex-shrink: 0;
      }
      .bolt-calc-history {
        font-size: 12px; color: var(--bc-text-dim);
        text-align: right; min-height: 16px;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
      }
      .bolt-calc-expr {
        font-size: 22px; color: var(--bc-text);
        text-align: right; min-height: 30px;
        white-space: nowrap; overflow-x: auto; overflow-y: hidden;
        font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
        font-weight: 500;
        scrollbar-width: none;
      }
      .bolt-calc-expr::-webkit-scrollbar { display: none; }
      .bolt-calc-preview {
        font-size: 14px; color: var(--bc-text-dim);
        text-align: right; min-height: 18px;
        font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
        transition: opacity 0.15s;
      }

      /* ---- base display (programmer) ---- */
      .bolt-calc-bases {
        margin-top: 6px;
        border-top: 1px solid var(--bc-border);
        padding-top: 4px;
        display: flex; flex-direction: column; gap: 1px;
        flex-shrink: 0;
      }
      .bolt-calc-base-row {
        display: flex; align-items: baseline; gap: 8px;
        padding: 1px 4px; border-radius: 4px;
        cursor: pointer; transition: background 0.12s;
      }
      .bolt-calc-base-row:hover { background: var(--bc-surface-h); }
      .bolt-calc-base-row.active { background: rgba(255,149,0,0.10); }
      .bolt-calc-base-lbl {
        width: 28px; font-size: 10px; font-weight: 700;
        color: var(--bc-text-dim);
        font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
      }
      .bolt-calc-base-row.active .bolt-calc-base-lbl { color: var(--bc-accent); }
      .bolt-calc-base-val {
        flex: 1; font-size: 11px; color: var(--bc-text);
        font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
        overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
        text-align: right; min-width: 0;
      }

      /* ---- button grid ---- */
      .bolt-calc-grid {
        display: grid; gap: 6px; flex: 1;
        grid-auto-rows: 1fr;
      }
      .bolt-calc-grid.normal {
        grid-template-columns: repeat(4, 1fr);
      }
      .bolt-calc-grid.scientific,
      .bolt-calc-grid.programmer {
        grid-template-columns: repeat(5, 1fr);
      }

      .bolt-calc-btn {
        display: flex; align-items: center; justify-content: center;
        border: none; border-radius: 10px;
        font-size: 15px; font-weight: 500;
        cursor: pointer;
        transition: transform 0.1s ease, background 0.15s ease, box-shadow 0.15s ease;
        box-shadow: var(--bc-shadow);
        padding: 0; min-height: 36px;
        font-family: inherit;
        -webkit-tap-highlight-color: transparent;
      }
      .bolt-calc-btn:active {
        transform: scale(0.92);
      }

      .bolt-calc-btn.num {
        background: var(--bc-num-bg); color: var(--bc-text);
      }
      .bolt-calc-btn.num:hover { background: var(--bc-num-h); }

      .bolt-calc-btn.op {
        background: var(--bc-accent-g); color: #fff; font-weight: 700;
      }
      .bolt-calc-btn.op:hover { filter: brightness(1.1); }

      .bolt-calc-btn.fn {
        background: var(--bc-fn-bg); color: var(--bc-text-dim);
        font-size: 13px;
      }
      .bolt-calc-btn.fn:hover { background: var(--bc-fn-h); color: var(--bc-text); }

      .bolt-calc-btn.eq {
        background: var(--bc-eq-bg); color: #fff;
        font-size: 20px; font-weight: 700;
      }
      .bolt-calc-btn.eq:hover { filter: brightness(1.1); }

      .bolt-calc-btn.wide { grid-column: span 2; }

      .bolt-calc-btn.mem {
        font-size: 11px; font-weight: 600;
        background: var(--bc-fn-bg); color: var(--bc-accent);
      }
      .bolt-calc-btn.mem:hover { background: var(--bc-fn-h); }

      .bolt-calc-btn.second-active {
        background: var(--bc-accent); color: #fff;
      }

      .bolt-calc-btn.disabled {
        opacity: 0.2;
        pointer-events: none;
        box-shadow: none;
      }

      /* flash animation */
      @keyframes bolt-calc-flash {
        0%   { filter: brightness(1.4); }
        100% { filter: brightness(1); }
      }
      .bolt-calc-btn.flash {
        animation: bolt-calc-flash 0.15s ease;
      }
    `;
    document.head.appendChild(style);
  }

  /* ---------- build UI ---------- */
  function render(container, args, context) {
    injectStyles();

    const isDark = context?.isDark !== false;
    const state = context?.reviveState || {
      mode: args?.mode || 'normal',
      expr: '',
      history: '',
      memory: 0,
      degMode: true,
      secondFn: false,
      base: 10,
    };

    const root = document.createElement('div');
    root.className = 'bolt-calc-root' + (isDark ? '' : ' bolt-calc-light');
    container.appendChild(root);

    /* ---- state ---- */
    let expr = state.expr || '';
    let history = state.history || '';
    let memory = state.memory || 0;
    let degMode = state.degMode !== false;
    let secondFn = !!state.secondFn;
    let mode = state.mode || 'normal';
    let base = state.base || 10;
    let justEvaluated = false;

    /* ---- elements ---- */
    // Toggle bar
    const toggleBar = document.createElement('div');
    toggleBar.className = 'bolt-calc-toggle-bar';

    const pill = document.createElement('div');
    pill.className = 'bolt-calc-pill';
    const pillBg = document.createElement('div');
    const modeIdx = () => MODES.indexOf(mode);
    pillBg.className = 'bolt-calc-pill-bg pos-' + modeIdx();

    const pillBtns = MODES.map(m => {
      const b = document.createElement('button');
      b.textContent = m === 'normal' ? 'Normal' : m === 'scientific' ? 'Sci' : 'Prog';
      if (m === mode) b.classList.add('active');
      return b;
    });
    pill.appendChild(pillBg);
    pillBtns.forEach(b => pill.appendChild(b));

    const degBtn = document.createElement('button');
    degBtn.className = 'bolt-calc-deg-rad';
    degBtn.textContent = degMode ? 'DEG' : 'RAD';
    degBtn.style.display = mode === 'scientific' ? '' : 'none';

    toggleBar.append(pill, degBtn);
    root.appendChild(toggleBar);

    // Display
    const display = document.createElement('div');
    display.className = 'bolt-calc-display';
    const historyEl = document.createElement('div');
    historyEl.className = 'bolt-calc-history';
    historyEl.textContent = history;
    const exprEl = document.createElement('div');
    exprEl.className = 'bolt-calc-expr';
    exprEl.textContent = expr || '0';
    const previewEl = document.createElement('div');
    previewEl.className = 'bolt-calc-preview';
    display.append(historyEl, exprEl, previewEl);

    // Base display (programmer mode)
    const basesEl = document.createElement('div');
    basesEl.className = 'bolt-calc-bases';
    basesEl.style.display = mode === 'programmer' ? '' : 'none';
    const BASE_KEYS = [
      { key: 16, label: 'HEX' },
      { key: 10, label: 'DEC' },
      { key: 8,  label: 'OCT' },
      { key: 2,  label: 'BIN' },
    ];
    const baseRows = BASE_KEYS.map(b => {
      const row = document.createElement('div');
      row.className = 'bolt-calc-base-row' + (b.key === base ? ' active' : '');
      const lbl = document.createElement('span');
      lbl.className = 'bolt-calc-base-lbl';
      lbl.textContent = b.label;
      const val = document.createElement('span');
      val.className = 'bolt-calc-base-val';
      val.textContent = '0';
      row.append(lbl, val);
      row.addEventListener('click', () => switchBase(b.key));
      basesEl.appendChild(row);
      return { base: b.key, el: row, valEl: val };
    });
    display.appendChild(basesEl);
    root.appendChild(display);

    // Grid container
    const grid = document.createElement('div');
    grid.className = 'bolt-calc-grid ' + mode;
    root.appendChild(grid);

    /* ---- helpers ---- */
    function evalCurrent() {
      if (mode === 'programmer') return safeEvalProg(expr, base);
      return safeEval(expr, degMode);
    }

    function updateDisplay() {
      if (mode === 'programmer') {
        exprEl.textContent = expr || '0';
        // Live preview
        if (expr && !justEvaluated) {
          const r = evalCurrent();
          previewEl.textContent = (r !== '' && r !== undefined && !isNaN(r)) ? '= ' + formatInBase(r, base) : '';
        } else {
          previewEl.textContent = '';
        }
        // Update all base rows
        const val = expr ? evalCurrent() : 0;
        const bases = allBases(typeof val === 'number' && isFinite(val) ? val : 0);
        baseRows.forEach(br => {
          br.valEl.textContent = br.base === 16 ? bases.hex : br.base === 10 ? bases.dec : br.base === 8 ? bases.oct : bases.bin;
          br.el.classList.toggle('active', br.base === base);
        });
      } else {
        exprEl.textContent = expr || '0';
        historyEl.textContent = history;
        if (expr && !justEvaluated) {
          const r = safeEval(expr, degMode);
          previewEl.textContent = (r !== '' && r !== undefined && !isNaN(r)) ? '= ' + formatNum(r) : '';
        } else {
          previewEl.textContent = '';
        }
      }
      exprEl.scrollLeft = exprEl.scrollWidth;
    }

    function flash(btn) {
      btn.classList.remove('flash');
      void btn.offsetWidth;
      btn.classList.add('flash');
    }

    function appendToExpr(str) {
      if (justEvaluated) {
        if (mode === 'programmer') {
          if (/[0-9a-fA-F(]/.test(str)) expr = '';
        } else {
          if (/[0-9.(]/.test(str) || str === '\u03C0' || str === 'e') expr = '';
        }
        justEvaluated = false;
      }
      expr += str;
      updateDisplay();
    }

    function doEvaluate() {
      if (!expr) return;
      const result = evalCurrent();
      if (mode === 'programmer') {
        const disp = formatInBase(result, base);
        history = expr + ' = ' + disp;
        expr = disp;
      } else {
        history = expr + ' = ' + formatNum(result);
        expr = formatNum(result);
      }
      justEvaluated = true;
      updateDisplay();
    }

    function doClear() {
      expr = '';
      justEvaluated = false;
      updateDisplay();
    }

    function doAllClear() {
      expr = '';
      history = '';
      justEvaluated = false;
      updateDisplay();
    }

    function doBackspace() {
      if (justEvaluated) { doClear(); return; }
      // Handle two-char operators
      if (expr.endsWith('<<') || expr.endsWith('>>')) {
        expr = expr.slice(0, -2);
      } else {
        expr = expr.slice(0, -1);
      }
      updateDisplay();
    }

    function doPercent() {
      const r = evalCurrent();
      if (!isNaN(r)) {
        expr = formatNum(r / 100);
        justEvaluated = true;
        updateDisplay();
      }
    }

    function doNegate() {
      if (!expr) return;
      const r = evalCurrent();
      if (!isNaN(r)) {
        if (mode === 'programmer') {
          expr = formatInBase(-r, base);
        } else {
          expr = formatNum(-r);
        }
        justEvaluated = true;
        updateDisplay();
      }
    }

    /* ---- button definitions ---- */
    function normalButtons() {
      return [
        ['MC', 'mem', () => { memory = 0; }],
        ['MR', 'mem', () => { appendToExpr(formatNum(memory)); }],
        ['M+', 'mem', () => { const r = safeEval(expr, degMode); if (!isNaN(r)) memory += r; }],
        ['M-', 'mem', () => { const r = safeEval(expr, degMode); if (!isNaN(r)) memory -= r; }],

        ['AC', 'fn', doAllClear],
        ['C', 'fn', doClear],
        ['\u232B', 'fn', doBackspace],
        ['\u00F7', 'op', () => appendToExpr('\u00F7')],

        ['7', 'num', () => appendToExpr('7')],
        ['8', 'num', () => appendToExpr('8')],
        ['9', 'num', () => appendToExpr('9')],
        ['\u00D7', 'op', () => appendToExpr('\u00D7')],

        ['4', 'num', () => appendToExpr('4')],
        ['5', 'num', () => appendToExpr('5')],
        ['6', 'num', () => appendToExpr('6')],
        ['\u2212', 'op', () => appendToExpr('-')],

        ['1', 'num', () => appendToExpr('1')],
        ['2', 'num', () => appendToExpr('2')],
        ['3', 'num', () => appendToExpr('3')],
        ['+', 'op', () => appendToExpr('+')],

        ['\u00B1', 'fn', doNegate],
        ['0', 'num', () => appendToExpr('0')],
        ['.', 'num', () => appendToExpr('.')],
        ['=', 'eq', doEvaluate],
      ];
    }

    function sciButtons() {
      const s = secondFn;
      return [
        ['MC', 'mem', () => { memory = 0; }],
        ['MR', 'mem', () => { appendToExpr(formatNum(memory)); }],
        ['M+', 'mem', () => { const r = safeEval(expr, degMode); if (!isNaN(r)) memory += r; }],
        ['M-', 'mem', () => { const r = safeEval(expr, degMode); if (!isNaN(r)) memory -= r; }],
        ['%', 'fn', doPercent],

        ['2nd', 'fn' + (s ? ' second-active' : ''), () => { secondFn = !secondFn; buildGrid(); }],
        ['(', 'fn', () => appendToExpr('(')],
        [')', 'fn', () => appendToExpr(')')],
        ['AC', 'fn', doAllClear],
        ['\u232B', 'fn', doBackspace],

        [s ? 'sin\u207B\u00B9' : 'sin', 'fn', () => appendToExpr(s ? 'asin(' : 'sin(')],
        [s ? 'cos\u207B\u00B9' : 'cos', 'fn', () => appendToExpr(s ? 'acos(' : 'cos(')],
        [s ? 'tan\u207B\u00B9' : 'tan', 'fn', () => appendToExpr(s ? 'atan(' : 'tan(')],
        [s ? 'e\u02E3' : 'ln', 'fn', () => appendToExpr(s ? 'exp(' : 'ln(')],
        [s ? '10\u02E3' : 'log', 'fn', () => appendToExpr(s ? '10^(' : 'log(')],

        ['x\u00B2', 'fn', () => appendToExpr('^(2)')],
        ['x\u00B3', 'fn', () => appendToExpr('^(3)')],
        [s ? '\u00B3\u221A' : 'x\u207F', 'fn', () => appendToExpr(s ? 'cbrt(' : '^(')],
        [s ? '2\u02E3' : '\u221A', 'fn', () => appendToExpr(s ? '2^(' : 'sqrt(')],
        ['n!', 'fn', () => appendToExpr('fact(')],

        ['\u03C0', 'fn', () => appendToExpr('\u03C0')],
        ['7', 'num', () => appendToExpr('7')],
        ['8', 'num', () => appendToExpr('8')],
        ['9', 'num', () => appendToExpr('9')],
        ['\u00F7', 'op', () => appendToExpr('\u00F7')],

        ['e', 'fn', () => appendToExpr('e')],
        ['4', 'num', () => appendToExpr('4')],
        ['5', 'num', () => appendToExpr('5')],
        ['6', 'num', () => appendToExpr('6')],
        ['\u00D7', 'op', () => appendToExpr('\u00D7')],

        ['mod', 'fn', () => appendToExpr('%')],
        ['1', 'num', () => appendToExpr('1')],
        ['2', 'num', () => appendToExpr('2')],
        ['3', 'num', () => appendToExpr('3')],
        ['\u2212', 'op', () => appendToExpr('-')],

        ['\u00B1', 'fn', doNegate],
        ['0', 'num', () => appendToExpr('0')],
        ['.', 'num', () => appendToExpr('.')],
        ['+', 'op', () => appendToExpr('+')],
        ['=', 'eq', doEvaluate],
      ];
    }

    function progButtons() {
      const cd = (d) => d < base ? 'num' : 'num disabled';
      return [
        // Row 1: bitwise + clear
        ['AND', 'fn', () => appendToExpr('&')],
        ['OR', 'fn', () => appendToExpr('|')],
        ['XOR', 'fn', () => appendToExpr('^')],
        ['NOT', 'fn', () => appendToExpr('~')],
        ['AC', 'fn', doAllClear],
        // Row 2: shifts, grouping, backspace
        ['<<', 'fn', () => appendToExpr('<<')],
        ['>>', 'fn', () => appendToExpr('>>')],
        ['(', 'fn', () => appendToExpr('(')],
        [')', 'fn', () => appendToExpr(')')],
        ['\u232B', 'fn', doBackspace],
        // Row 3: hex D-F, multiply, divide
        ['D', cd(13), () => appendToExpr('D')],
        ['E', cd(14), () => appendToExpr('E')],
        ['F', cd(15), () => appendToExpr('F')],
        ['\u00D7', 'op', () => appendToExpr('\u00D7')],
        ['\u00F7', 'op', () => appendToExpr('\u00F7')],
        // Row 4: hex A-C, add, subtract
        ['A', cd(10), () => appendToExpr('A')],
        ['B', cd(11), () => appendToExpr('B')],
        ['C', cd(12), () => appendToExpr('C')],
        ['+', 'op', () => appendToExpr('+')],
        ['\u2212', 'op', () => appendToExpr('-')],
        // Row 5: 7-9, mod, %
        ['7', cd(7), () => appendToExpr('7')],
        ['8', cd(8), () => appendToExpr('8')],
        ['9', cd(9), () => appendToExpr('9')],
        ['mod', 'fn', () => appendToExpr('%')],
        ['CE', 'fn', doClear],
        // Row 6: 4-6, negate, decimal (disabled)
        ['4', cd(4), () => appendToExpr('4')],
        ['5', cd(5), () => appendToExpr('5')],
        ['6', cd(6), () => appendToExpr('6')],
        ['\u00B1', 'fn', doNegate],
        ['.', 'num disabled', () => {}],
        // Row 7: 1-3, 0, equals
        ['1', cd(1), () => appendToExpr('1')],
        ['2', cd(2), () => appendToExpr('2')],
        ['3', cd(3), () => appendToExpr('3')],
        ['0', 'num', () => appendToExpr('0')],
        ['=', 'eq', doEvaluate],
      ];
    }

    function buildGrid() {
      grid.innerHTML = '';
      grid.className = 'bolt-calc-grid ' + mode;
      const buttons = mode === 'normal' ? normalButtons() : mode === 'scientific' ? sciButtons() : progButtons();
      buttons.forEach(([label, cls, action]) => {
        const btn = document.createElement('button');
        btn.className = 'bolt-calc-btn ' + cls;
        btn.textContent = label;
        if (!cls.includes('disabled')) {
          btn.addEventListener('pointerdown', (e) => {
            e.preventDefault();
            flash(btn);
            action();
          });
        }
        grid.appendChild(btn);
      });
    }

    /* ---- mode switching ---- */
    function switchMode(newMode) {
      mode = newMode;
      const idx = MODES.indexOf(mode);
      pillBtns.forEach((b, i) => b.classList.toggle('active', i === idx));
      pillBg.className = 'bolt-calc-pill-bg pos-' + idx;
      degBtn.style.display = mode === 'scientific' ? '' : 'none';
      basesEl.style.display = mode === 'programmer' ? '' : 'none';
      buildGrid();
      updateDisplay();
      if (context?.resize) context.resize(...SIZES[mode]);
      if (context?.memory) context.memory.set('calc_mode', mode);
    }

    function switchBase(newBase) {
      // Convert current value to new base display
      if (expr && mode === 'programmer') {
        const r = safeEvalProg(expr, base);
        if (typeof r === 'number' && !isNaN(r) && isFinite(r)) {
          expr = formatInBase(r, newBase);
          justEvaluated = true;
        }
      }
      base = newBase;
      buildGrid();
      updateDisplay();
      if (context?.memory) context.memory.set('calc_base', newBase);
    }

    /* ---- toggle handlers ---- */
    pillBtns.forEach((btn, i) => {
      btn.addEventListener('click', () => switchMode(MODES[i]));
    });
    degBtn.addEventListener('click', () => {
      degMode = !degMode;
      degBtn.textContent = degMode ? 'DEG' : 'RAD';
      updateDisplay();
    });

    /* ---- keyboard support ---- */
    function onKey(e) {
      if (!root.isConnected) { document.removeEventListener('keydown', onKey); return; }
      const k = e.key;
      if (k >= '0' && k <= '9') {
        if (mode === 'programmer' && parseInt(k, 10) >= base) return;
        appendToExpr(k); e.preventDefault();
      }
      else if (k === '.') {
        if (mode !== 'programmer') { appendToExpr('.'); e.preventDefault(); }
      }
      else if (k === '+') { appendToExpr('+'); e.preventDefault(); }
      else if (k === '-') { appendToExpr('-'); e.preventDefault(); }
      else if (k === '*') { appendToExpr('\u00D7'); e.preventDefault(); }
      else if (k === '/') { appendToExpr('\u00F7'); e.preventDefault(); }
      else if (k === '%') {
        if (mode === 'programmer') appendToExpr('%');
        else doPercent();
        e.preventDefault();
      }
      else if (k === '^') {
        appendToExpr(mode === 'programmer' ? '^' : '^(');
        e.preventDefault();
      }
      else if (k === '(' || k === ')') { appendToExpr(k); e.preventDefault(); }
      else if (k === 'Enter' || k === '=') { doEvaluate(); e.preventDefault(); }
      else if (k === 'Escape') { doAllClear(); e.preventDefault(); }
      else if (k === 'Backspace') { doBackspace(); e.preventDefault(); }
      else if (k === 'Delete') { doClear(); e.preventDefault(); }
      else if (mode === 'programmer' && base === 16) {
        const lower = k.toLowerCase();
        if (lower >= 'a' && lower <= 'f') {
          appendToExpr(lower.toUpperCase());
          e.preventDefault();
        }
      }
      // Programmer bitwise ops
      if (mode === 'programmer') {
        if (k === '&') { appendToExpr('&'); e.preventDefault(); }
        else if (k === '|') { appendToExpr('|'); e.preventDefault(); }
        else if (k === '~') { appendToExpr('~'); e.preventDefault(); }
        else if (k === '<' && e.shiftKey) { /* << via shift combo handled by two keys */ }
      }
    }
    document.addEventListener('keydown', onKey);

    /* ---- load saved mode from memory (async) ---- */
    if (!context?.reviveState && context?.memory) {
      Promise.all([
        context.memory.get('calc_mode'),
        context.memory.get('calc_base'),
      ]).then(([savedMode, savedBase]) => {
        let changed = false;
        if (savedMode && MODES.includes(savedMode) && savedMode !== mode) {
          mode = savedMode;
          changed = true;
        }
        if (savedBase && [2, 8, 10, 16].includes(savedBase) && savedBase !== base) {
          base = savedBase;
          changed = true;
        }
        if (changed) {
          const idx = MODES.indexOf(mode);
          pillBtns.forEach((b, i) => b.classList.toggle('active', i === idx));
          pillBg.className = 'bolt-calc-pill-bg pos-' + idx;
          degBtn.style.display = mode === 'scientific' ? '' : 'none';
          basesEl.style.display = mode === 'programmer' ? '' : 'none';
          buildGrid();
          updateDisplay();
          if (context?.resize) context.resize(...SIZES[mode]);
        }
      });
    }

    /* ---- initial render ---- */
    buildGrid();
    updateDisplay();

    /* ---- public interface ---- */
    return {
      getState() {
        return { mode, expr, history, memory, degMode, secondFn, base };
      },
      setState(s) {
        if (!s) return;
        mode = s.mode || 'normal';
        expr = s.expr || '';
        history = s.history || '';
        memory = s.memory || 0;
        degMode = s.degMode !== false;
        secondFn = !!s.secondFn;
        base = s.base || 10;
        justEvaluated = false;

        const idx = MODES.indexOf(mode);
        pillBtns.forEach((b, i) => b.classList.toggle('active', i === idx));
        pillBg.className = 'bolt-calc-pill-bg pos-' + idx;
        degBtn.style.display = mode === 'scientific' ? '' : 'none';
        basesEl.style.display = mode === 'programmer' ? '' : 'none';
        degBtn.textContent = degMode ? 'DEG' : 'RAD';
        buildGrid();
        updateDisplay();
      }
    };
  }

  /* ---------- register bolt app ---------- */
  window._boltApps = window._boltApps || {};
  window._boltApps.calculator = {
    name: 'calculator',
    icon: 'calculate',
    width: 340,
    height: 480,
    devices: ['desktop', 'tablet', 'mobile'],
    render: render,
  };
})();
