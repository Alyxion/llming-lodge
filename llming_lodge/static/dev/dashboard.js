(function () {
  "use strict";

  const LS_KEY = "dev-dashboard-key";
  const MAX_PTS = 600;

  let key = localStorage.getItem(LS_KEY) || "";
  let since = 0;
  let hb = [], sys = [];
  let polling = null, infoPoll = null;
  let infoData = null, llmData = null;
  let activeTab = "graphs";
  let lastStateMs = null, lastInfoMs = null;

  const $ = (id) => document.getElementById(id);

  /* ── Tabs ──────────────────────────────────── */

  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const tab = btn.dataset.tab;
      $("tab-" + tab).classList.add("active");
      activeTab = tab;
      if (tab === "graphs") drawAll();
    });
  });

  /* ── Auth ──────────────────────────────────── */

  function showAuth(msg) {
    $("auth-screen").style.display = "flex";
    $("dashboard").style.display = "none";
    if (msg) { $("auth-hint").textContent = msg; $("auth-hint").style.display = "block"; }
    else { $("auth-hint").style.display = "none"; }
    $("key-input").value = "";
    $("key-input").focus();
    stopPolling();
  }

  function showDashboard() {
    $("auth-screen").style.display = "none";
    $("dashboard").style.display = "block";
  }

  function stopPolling() {
    if (polling) { clearInterval(polling); polling = null; }
    if (infoPoll) { clearInterval(infoPoll); infoPoll = null; }
  }

  function tryConnect() {
    key = $("key-input").value.trim();
    if (!key) return;
    localStorage.setItem(LS_KEY, key);
    startSession();
  }

  $("auth-btn").addEventListener("click", tryConnect);
  $("key-input").addEventListener("keydown", (e) => { if (e.key === "Enter") tryConnect(); });

  function startSession() {
    showDashboard();
    since = 0; hb = []; sys = []; infoData = null; llmData = null;
    fetchState();
    polling = setInterval(fetchState, 1000);
    setTimeout(() => { fetchInfo(); infoPoll = setInterval(fetchInfo, 5000); }, 500);
    setTimeout(fetchLlm, 800);
  }

  if (key) { startSession(); } else { showAuth(); }

  /* ── Data fetching ─────────────────────────── */

  async function fetchState() {
    try {
      const t0 = performance.now();
      const r = await fetch("/dev/state?since=" + since, { headers: { "X-Dev-Key": key } });
      if (r.status === 401) { localStorage.removeItem(LS_KEY); key = ""; showAuth("Invalid password"); return; }
      const d = await r.json();
      lastStateMs = performance.now() - t0;
      since = d.now;
      if (d.heartbeat.length) { hb.push(...d.heartbeat); if (hb.length > MAX_PTS) hb = hb.slice(-MAX_PTS); }
      if (d.system.length) { sys.push(...d.system); if (sys.length > MAX_PTS) sys = sys.slice(-MAX_PTS); }
      updateHeader(d);
      if (activeTab === "graphs") drawAll();
    } catch (e) { console.error("state fetch error", e); }
  }

  async function fetchInfo() {
    try {
      const t0 = performance.now();
      const r = await fetch("/dev/info", { headers: { "X-Dev-Key": key } });
      if (r.status === 401) return;
      infoData = await r.json();
      lastInfoMs = performance.now() - t0;
      renderProcess();
      renderSystem();
      renderAsyncio();
      renderEnv();
    } catch (e) { console.error("info fetch error", e); }
  }

  /* ── Header ────────────────────────────────── */

  function updateHeader(d) {
    const up = d.uptime_s;
    const h = Math.floor(up / 3600), m = Math.floor((up % 3600) / 60), s = Math.floor(up % 60);
    const upStr = h > 0 ? `${h}h${m}m` : `${m}m${s}s`;
    const ls = sys.length ? sys[sys.length - 1] : null;
    const cpu = ls ? ls.cpu_percent.toFixed(1) : "--";
    const nCpu = ls && ls.per_cpu ? ls.per_cpu.length : "--";
    const la = ls && ls.load_avg ? ls.load_avg.map((v) => v.toFixed(1)).join("/") : "--";
    const memU = ls ? ls.mem_used_mb.toFixed(0) : "--";
    const memT = ls ? ls.mem_total_mb.toFixed(0) : "--";
    const pRSS = ls ? ls.proc_rss_mb.toFixed(0) : "--";
    const lh = hb.length ? hb[hb.length - 1] : null;
    const loopA = lh ? lh.avg_ms.toFixed(1) : "--";
    const pid = infoData ? infoData.process.pid : "--";

    $("header").innerHTML =
      `<span><span class="label">Server</span><span class="value">${d.server_name}</span></span>` +
      `<span class="sep">|</span>` +
      `<span><span class="label">Worker</span><span class="value">${(d.worker_id || "--").slice(0, 8)}</span></span>` +
      `<span class="sep">|</span>` +
      `<span><span class="label">PID</span><span class="value">${pid}</span></span>` +
      `<span class="sep">|</span>` +
      `<span><span class="label">Uptime</span><span class="value">${upStr}</span></span>` +
      `<span class="sep">|</span>` +
      `<span><span class="label">CPU</span><span class="value">${cpu}% x${nCpu}</span></span>` +
      `<span class="sep">|</span>` +
      `<span><span class="label">Load</span><span class="value">${la}</span></span>` +
      `<span class="sep">|</span>` +
      `<span><span class="label">System</span><span class="value">${memU}/${memT} MB</span></span>` +
      `<span class="sep">|</span>` +
      `<span><span class="label">Process</span><span class="value">${pRSS} MB</span></span>` +
      `<span class="sep">|</span>` +
      `<span><span class="label">Loop</span><span class="value">${loopA} ms</span></span>` +
      `<span class="sep">|</span>` +
      `<span><span class="label">Fetch</span><span class="value">${lastStateMs != null ? lastStateMs.toFixed(0) : "--"}ms / ${lastInfoMs != null ? lastInfoMs.toFixed(0) : "--"}ms</span></span>`;
  }

  /* ── Canvas drawing ────────────────────────── */

  function setupCanvas(c) {
    const r = window.devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight;
    c.width = w * r; c.height = h * r;
    const ctx = c.getContext("2d"); ctx.scale(r, r);
    return { ctx, w, h };
  }

  function drawGraph(cid, data, getValue, opts) {
    const c = $(cid); if (!c) return;
    const { ctx, w, h } = setupCanvas(c);
    const barW = opts.currentBar ? 34 : 0;
    const pad = { l: 44, r: 8 + barW + 4, t: 4, b: 16 };
    const gw = w - pad.l - pad.r, gh = h - pad.t - pad.b;
    ctx.clearRect(0, 0, w, h);
    if (!data.length) return;

    const vals = data.map((d) => getValue(d));

    let maxY = 0;
    vals.forEach((v) => {
      if (Array.isArray(v)) v.forEach((x) => { if (x > maxY) maxY = x; });
      else if (v > maxY) maxY = v;
    });
    if (opts.minMax) maxY = Math.max(maxY, opts.minMax);
    maxY = Math.ceil(maxY * 1.15) || 1;

    // Grid
    ctx.strokeStyle = "#333"; ctx.lineWidth = 1;
    ctx.fillStyle = "#666"; ctx.font = "10px monospace"; ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const yv = maxY * (1 - i / 4), y = pad.t + gh * (i / 4);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + gw, y); ctx.stroke();
      ctx.fillText(yv.toFixed(yv >= 10 ? 0 : 1), pad.l - 4, y + 3);
    }

    // Fixed-width padding
    const pc = MAX_PTS - vals.length;
    const av = new Array(pc).fill(null).concat(vals);

    // Lines
    const colors = opts.colors || ["#4a9eff"];
    const sample = av.find((v) => v !== null);
    const lc = Array.isArray(sample) ? sample.length : 1;
    for (let li = 0; li < lc; li++) {
      ctx.strokeStyle = colors[li % colors.length]; ctx.lineWidth = 1.5; ctx.beginPath();
      let s = false;
      for (let i = 0; i < av.length; i++) {
        if (av[i] === null) continue;
        const v = Array.isArray(av[i]) ? av[i][li] : av[i];
        const x = pad.l + (i / (MAX_PTS - 1)) * gw;
        const y = pad.t + gh * (1 - v / maxY);
        if (!s) { ctx.moveTo(x, y); s = true; } else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // Jitter bars
    if (opts.barGetter) {
      const ad = new Array(pc).fill(null).concat(data);
      ctx.fillStyle = "rgba(128,128,128,0.25)";
      const bw = Math.max(1, gw / MAX_PTS);
      let bM = 0; data.forEach((d) => { const b = opts.barGetter(d); if (b > bM) bM = b; }); bM = bM || 1;
      for (let i = 0; i < ad.length; i++) {
        if (ad[i] === null) continue;
        const bv = opts.barGetter(ad[i]); if (bv <= 0) continue;
        const bh = (bv / bM) * gh * 0.3, x = pad.l + (i / (MAX_PTS - 1)) * gw - bw / 2;
        ctx.fillRect(x, pad.t + gh - bh, bw, bh);
      }
    }

    // Current value bar
    if (opts.currentBar && vals.length) {
      const last = vals[vals.length - 1];
      const v = Array.isArray(last) ? last[0] : last;
      const pct = Math.min(v / maxY, 1);
      const bx = w - barW - 2, bh = gh * pct, by = pad.t + gh - bh;
      ctx.fillStyle = "#222"; ctx.fillRect(bx, pad.t, barW, gh);
      ctx.fillStyle = opts.currentBar; ctx.fillRect(bx, by, barW, bh);
      ctx.fillStyle = "#fff"; ctx.font = "bold 10px monospace"; ctx.textAlign = "center";
      const lbl = v.toFixed(v >= 100 ? 0 : 1) + (opts.currentBarUnit || "");
      ctx.fillText(lbl, bx + barW / 2, by - 3 > pad.t + 8 ? by - 3 : by + 12);
    }

    // Time axis
    ctx.fillStyle = "#555"; ctx.textAlign = "center"; ctx.font = "9px monospace";
    if (data.length >= 2) {
      const fmt = (ts) => new Date(ts * 1000).toLocaleTimeString();
      const fx = pad.l + (pc / (MAX_PTS - 1)) * gw;
      ctx.fillText(fmt(data[0].ts), Math.max(fx, pad.l + 24), h - 2);
      ctx.fillText(fmt(data[data.length - 1].ts), pad.l + gw, h - 2);
    }
  }

  function drawPerCPU() {
    const c = $("c-percpu"); if (!c) return;
    const { ctx, w, h } = setupCanvas(c);
    const ls = sys.length ? sys[sys.length - 1] : null;
    if (!ls || !ls.per_cpu || !ls.per_cpu.length) { ctx.clearRect(0, 0, w, h); return; }
    const cpus = ls.per_cpu;
    const n = cpus.length;
    const pad = { l: 24, r: 8, t: 2, b: 14 };
    const gw = w - pad.l - pad.r, gh = h - pad.t - pad.b;
    const gap = 2;
    const bw = Math.max(4, (gw - gap * (n - 1)) / n);

    ctx.clearRect(0, 0, w, h);

    // Grid lines at 50% and 100%
    ctx.strokeStyle = "#333"; ctx.lineWidth = 1;
    [0, 50, 100].forEach((v) => {
      const y = pad.t + gh * (1 - v / 100);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    });
    ctx.fillStyle = "#555"; ctx.font = "9px monospace"; ctx.textAlign = "right";
    ctx.fillText("100", pad.l - 3, pad.t + 4);
    ctx.fillText("50", pad.l - 3, pad.t + gh / 2 + 3);

    for (let i = 0; i < n; i++) {
      const v = Math.min(cpus[i], 100);
      const bh = gh * (v / 100);
      const x = pad.l + i * (bw + gap);
      const y = pad.t + gh - bh;
      const color = v > 80 ? "#f44" : v > 50 ? "#ff9800" : "#4caf50";
      ctx.fillStyle = color;
      ctx.fillRect(x, y, bw, bh);
      ctx.fillStyle = "#888"; ctx.font = "8px monospace"; ctx.textAlign = "center";
      ctx.fillText(i, x + bw / 2, h - 2);
    }
  }

  function drawAll() {
    drawGraph("c-loop", hb, (d) => [d.avg_ms, d.max_ms], { colors: ["#4a9eff", "#f44"], minMax: 20, barGetter: (d) => d.count });
    drawGraph("c-cpu", sys, (d) => [d.cpu_percent, d.proc_cpu], { colors: ["#4caf50", "#81c784"], minMax: 100, currentBar: "#4caf50", currentBarUnit: "%" });
    drawGraph("c-mem", sys, (d) => d.mem_percent, { colors: ["#ff9800"], minMax: 100, currentBar: "#ff9800", currentBarUnit: "%" });
    drawGraph("c-proc", sys, (d) => d.proc_rss_mb, { colors: ["#ce93d8"], currentBar: "#ce93d8", currentBarUnit: " MB" });
    drawPerCPU();
  }

  window.addEventListener("resize", () => { if (activeTab === "graphs") drawAll(); });

  /* ── Info renderers ────────────────────────── */

  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function row(k, v, cls) {
    return `<div class="info-row"><span class="k">${k}</span><span class="v ${cls || ""}">${v}</span></div>`;
  }

  function card(title, rows) {
    return `<div class="info-card"><h3>${title}</h3>${rows}</div>`;
  }

  function renderProcess() {
    if (!infoData) return;
    const p = infoData.process;
    const r = infoData.runtime;

    let rows = "";
    rows += row("PID", p.pid);
    rows += row("Parent PID", p.ppid || "--");
    rows += row("Status", p.status || "--");
    rows += row("Started", p.create_time || "--");
    rows += row("Command", `<span title="${esc(p.cmdline || "")}">${esc((p.cmdline || "").substring(0, 60))}</span>`);
    rows += row("Working Dir", esc(p.cwd || "--"));

    let rows2 = "";
    rows2 += row("RSS", p.mem_rss_mb + " MB");
    rows2 += row("VMS", p.mem_vms_mb + " MB");
    rows2 += row("CPU Time (user)", p.cpu_user_s != null ? p.cpu_user_s + "s" : "--");
    rows2 += row("CPU Time (sys)", p.cpu_system_s != null ? p.cpu_system_s + "s" : "--");
    rows2 += row("Threads", p.num_threads);
    rows2 += row("File Descriptors", p.num_fds >= 0 ? p.num_fds : "N/A");
    rows2 += row("Connections", p.connection_count >= 0 ? p.connection_count : "N/A");

    let rows3 = "";
    rows3 += row("Python", r.python_version + " (" + r.python_impl + ")");
    rows3 += row("Platform", r.platform);
    rows3 += row("Architecture", r.arch);
    rows3 += row("Hostname", r.hostname);

    let rows4 = "";
    if (p.children && p.children.length) {
      p.children.forEach((c) => { rows4 += row("PID " + c.pid, c.name + " (" + c.status + ")"); });
    } else {
      rows4 += row("--", "No child processes");
    }

    let connRows = "";
    if (p.connections && Object.keys(p.connections).length) {
      Object.entries(p.connections).forEach(([st, cnt]) => { connRows += row(st, cnt); });
    } else {
      connRows += row("--", "No connections");
    }

    $("process-grid").innerHTML =
      card("Process", rows) +
      card("Memory & Resources", rows2) +
      card("Runtime", rows3) +
      card("Children", rows4) +
      card("Connections by State", connRows);
  }

  function renderSystem() {
    if (!infoData) return;
    const s = infoData.system;

    let rows = "";
    rows += row("CPU Cores", s.cpu_count_physical + " physical / " + s.cpu_count_logical + " logical");
    if (s.cpu_freq_mhz) rows += row("CPU Freq", s.cpu_freq_mhz + " MHz");
    rows += row("Total Memory", s.mem_total_mb + " MB");
    rows += row("Available Memory", s.mem_available_mb + " MB");
    if (s.load_avg) rows += row("Load Avg (1/5/15)", s.load_avg.join(" / "));
    rows += row("Boot Time", s.boot_time);

    let diskHtml = '<div class="info-card" style="grid-column:1/-1"><h3>Disk Usage</h3>';
    if (s.disks && s.disks.length) {
      diskHtml += '<table class="info-table"><tr><th>Mount</th><th>Device</th><th>FS</th><th>Total</th><th>Used</th><th>Free</th><th>Usage</th></tr>';
      s.disks.forEach((d) => {
        const cls = d.percent > 90 ? "crit" : d.percent > 75 ? "warn" : "";
        const barColor = d.percent > 90 ? "#f44" : d.percent > 75 ? "#ff9800" : "#4caf50";
        diskHtml += `<tr><td>${esc(d.mount)}</td><td>${esc(d.device)}</td><td>${d.fstype}</td><td>${d.total_gb} GB</td><td>${d.used_gb} GB</td><td>${d.free_gb} GB</td><td><span class="${cls}">${d.percent}%</span><div class="disk-bar"><div class="disk-bar-fill" style="width:${d.percent}%;background:${barColor}"></div></div></td></tr>`;
      });
      diskHtml += "</table>";
    } else {
      diskHtml += '<div class="info-row"><span class="k">--</span><span class="v">No disk info</span></div>';
    }
    diskHtml += "</div>";

    let netHtml = "";
    if (s.network_interfaces && s.network_interfaces.length) {
      s.network_interfaces.forEach((n) => {
        let nr = "";
        n.addrs.forEach((a) => { nr += row(a.family, a.addr); });
        netHtml += card(n.name, nr);
      });
    }

    $("system-grid").innerHTML = card("System", rows) + diskHtml + netHtml;
  }

  function renderAsyncio() {
    if (!infoData) return;
    const tasks = infoData.asyncio_tasks || [];
    let html = `<div class="info-card" style="max-width:100%"><h3>Asyncio Tasks (${infoData.asyncio_task_count})</h3>`;
    if (tasks.length) {
      html += '<table class="info-table"><tr><th>Name</th><th>Coroutine</th><th>Status</th></tr>';
      tasks.forEach((t) => {
        const st = t.done
          ? '<span style="color:#888">done</span>'
          : '<span style="color:#4caf50">running</span>';
        html += `<tr><td>${esc(t.name)}</td><td style="font-size:11px;color:#aaa">${esc(t.coro)}</td><td>${st}</td></tr>`;
      });
      html += "</table>";
    } else {
      html += '<div class="info-row"><span class="k">--</span><span class="v">No tasks</span></div>';
    }
    html += "</div>";
    $("asyncio-content").innerHTML = html;
  }

  function renderEnv() {
    if (!infoData) return;
    const env = infoData.environment || {};
    const keys = Object.keys(env).sort();
    let html = '<div class="info-card" style="max-width:100%"><h3>Environment Variables (filtered)</h3>';
    if (keys.length) {
      keys.forEach((k) => { html += row(k, esc(env[k])); });
    } else {
      html += '<div class="info-row"><span class="k">--</span><span class="v">No matching env vars</span></div>';
    }
    html += "</div>";
    $("env-content").innerHTML = html;
  }

  /* ── LLM Provider Cascade ───────────────── */

  async function fetchLlm() {
    try {
      const r = await fetch("/dev/llm", { headers: { "X-Dev-Key": key } });
      if (r.status === 401) return;
      llmData = await r.json();
      renderLlm();
    } catch (e) { console.error("llm fetch error", e); }
  }

  function renderLlm() {
    if (!llmData) { $("llm-content").innerHTML = ""; return; }
    const d = llmData;

    // Cascade order card
    let cascadeRows = "";
    d.cascade_order.forEach((p, i) => {
      const active = d.active_providers.includes(p);
      const cls = active ? "ok" : "crit";
      const badge = active ? "active" : "no keys";
      cascadeRows += row("#" + (i + 1) + " " + p, '<span class="' + cls + '">' + badge + "</span>");
    });

    // Default categories card
    let catRows = "";
    if (d.default_categories) {
      Object.entries(d.default_categories).forEach(([cat, info]) => {
        const prov = info.resolved_provider || "?";
        catRows += row(cat, esc(info.model) + '  <span style="color:#4a9eff">\u2192 ' + esc(prov) + "</span>");
      });
    }

    // Models table
    let modelsHtml = '<div class="info-card" style="grid-column:1/-1"><h3>Models &mdash; Cascade Resolution</h3>';
    if (d.models && d.models.length) {
      modelsHtml += '<table class="info-table"><tr><th>Model</th><th>Size</th><th>Active Provider</th><th>All Providers</th></tr>';
      d.models.forEach((m) => {
        const provBadges = m.all_providers.map((p) => {
          if (p === m.active_provider) return '<span style="color:#4caf50;font-weight:600">' + esc(p) + "</span>";
          return '<span style="color:#888">' + esc(p) + "</span>";
        }).join(", ");
        modelsHtml += "<tr><td>" + esc(m.model) + "</td><td>" + esc(m.size) + "</td><td><span style='color:#4caf50'>" + esc(m.active_provider || "--") + "</span></td><td>" + provBadges + "</td></tr>";
      });
      modelsHtml += "</table>";
    } else {
      modelsHtml += row("--", "No models available");
    }
    modelsHtml += "</div>";

    $("llm-content").innerHTML =
      '<div class="info-grid">' +
      card("Provider Cascade", cascadeRows) +
      card("Default Categories", catRows) +
      modelsHtml +
      "</div>";
  }
})();
