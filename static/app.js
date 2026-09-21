// HRGCN CloudGuard Interactive Dashboard Frontend Engine

let currentGraphData = null;
let activeNode = null;
let showEdgeLabels = true;
let simulationActive = true;
let isDarkTheme = false;

// Color maps matching theme tokens
const NODE_COLORS = {
  0: "#2563eb", // User (Blue)
  1: "#d97706", // Device (Amber)
  2: "#7c3aed", // Role (Purple)
  3: "#059669"  // Resource (Emerald)
};

const NODE_COLORS_DARK = {
  0: "#3b82f6",
  1: "#f59e0b",
  2: "#a855f7",
  3: "#10b981"
};

// Canvas State
let canvas, ctx;
let nodes = [];
let edges = [];
let transform = { x: 0, y: 0, scale: 1 };
let isDragging = false;
let draggedNode = null;
let lastMouse = { x: 0, y: 0 };

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initTabs();
  initCanvas();
  initEventListeners();
  loadInitialData();
});

// 1. Theme Management (Light by Default, Dark Toggle)
function initTheme() {
  const themeToggle = document.getElementById("themeToggle");
  const html = document.documentElement;

  // Default is light
  html.setAttribute("data-theme", "light");
  isDarkTheme = false;

  themeToggle.addEventListener("click", () => {
    isDarkTheme = !isDarkTheme;
    html.setAttribute("data-theme", isDarkTheme ? "dark" : "light");
    
    const icon = themeToggle.querySelector(".theme-icon");
    const label = themeToggle.querySelector(".theme-label");
    if (isDarkTheme) {
      icon.textContent = "☀️";
      label.textContent = "Light Mode";
    } else {
      icon.textContent = "🌙";
      label.textContent = "Dark Mode";
    }
    drawGraph();
    drawSvddPlot();
  });
}

// 2. Tab Navigation
function initTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".view-panel").forEach(p => p.classList.remove("active"));

      tab.classList.add("active");
      const targetPanel = document.getElementById(tab.getAttribute("data-tab"));
      if (targetPanel) {
        targetPanel.classList.add("active");
        if (tab.getAttribute("data-tab") === "tab-monitor") {
          resizeCanvas();
        } else if (tab.getAttribute("data-tab") === "tab-explain") {
          drawSvddPlot();
        }
      }
    });
  });
}

// 3. Canvas & Physics Graph Visualization
function initCanvas() {
  canvas = document.getElementById("graphCanvas");
  ctx = canvas.getContext("2d");
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);

  // Mouse Interactivity
  canvas.addEventListener("mousedown", onMouseDown);
  canvas.addEventListener("mousemove", onMouseMove);
  canvas.addEventListener("mouseup", onMouseUp);
  canvas.addEventListener("wheel", onWheel);

  // Animation Loop
  requestAnimationFrame(updatePhysicsAndDraw);
}

function resizeCanvas() {
  if (!canvas) return;
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
  centerGraph();
  drawGraph();
}

function centerGraph() {
  if (!nodes.length) return;
  transform.x = canvas.width / 2;
  transform.y = canvas.height / 2;
  transform.scale = 1.0;
}

function updatePhysicsAndDraw() {
  if (nodes.length > 0 && simulationActive) {
    // Simple spring force layout
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[j].x - nodes[i].x;
        const dy = nodes[j].y - nodes[i].y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const force = (120 - dist) * 0.02;
        if (dist < 250) {
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          if (nodes[i] !== draggedNode) { nodes[i].x -= fx; nodes[i].y -= fy; }
          if (nodes[j] !== draggedNode) { nodes[j].x += fx; nodes[j].y += fy; }
        }
      }
    }

    // Edge attraction
    edges.forEach(e => {
      const src = nodes[e.source];
      const dst = nodes[e.target];
      if (src && dst) {
        const dx = dst.x - src.x;
        const dy = dst.y - src.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const force = (dist - 140) * 0.03;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        if (src !== draggedNode) { src.x += fx; src.y += fy; }
        if (dst !== draggedNode) { dst.x -= fx; dst.y -= fy; }
      }
    });
  }

  drawGraph();
  requestAnimationFrame(updatePhysicsAndDraw);
}

function drawGraph() {
  if (!ctx || !canvas) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.save();
  ctx.translate(transform.x, transform.y);
  ctx.scale(transform.scale, transform.scale);

  // Draw Edges
  edges.forEach(e => {
    const src = nodes[e.source];
    const dst = nodes[e.target];
    if (!src || !dst) return;

    ctx.beginPath();
    ctx.moveTo(src.x, src.y);
    ctx.lineTo(dst.x, dst.y);

    const isAnomaly = currentGraphData && currentGraphData.is_anomaly;
    ctx.strokeStyle = isAnomaly ? "rgba(239, 68, 68, 0.7)" : (isDarkTheme ? "rgba(148, 163, 184, 0.4)" : "rgba(100, 116, 139, 0.5)");
    ctx.lineWidth = isAnomaly ? 2.5 : 1.8;
    ctx.stroke();

    // Arrowhead
    const angle = Math.atan2(dst.y - src.y, dst.x - src.x);
    const targetRadius = 24;
    const arrowX = dst.x - Math.cos(angle) * targetRadius;
    const arrowY = dst.y - Math.sin(angle) * targetRadius;

    ctx.save();
    ctx.translate(arrowX, arrowY);
    ctx.rotate(angle);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(-8, -4);
    ctx.lineTo(-8, 4);
    ctx.closePath();
    ctx.fill();
    ctx.restore();

    // Edge Label
    if (showEdgeLabels) {
      const midX = (src.x + dst.x) / 2;
      const midY = (src.y + dst.y) / 2;
      ctx.font = "10px 'JetBrains Mono', monospace";
      ctx.fillStyle = isDarkTheme ? "#94a3b8" : "#475569";
      ctx.textAlign = "center";
      ctx.fillText(e.relation, midX, midY - 6);
    }
  });

  // Draw Nodes
  nodes.forEach(n => {
    const isAnomaly = currentGraphData && currentGraphData.is_anomaly;
    const colorMap = isDarkTheme ? NODE_COLORS_DARK : NODE_COLORS;
    const baseColor = colorMap[n.type_id] || "#2563eb";

    // Pulsing halo for anomalies
    if (isAnomaly) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, 32, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(239, 68, 68, 0.2)";
      ctx.fill();
    }

    // Node Circle
    ctx.beginPath();
    ctx.arc(n.x, n.y, 22, 0, Math.PI * 2);
    ctx.fillStyle = baseColor;
    ctx.fill();
    ctx.lineWidth = n === activeNode ? 3 : 1.5;
    ctx.strokeStyle = n === activeNode ? "#00f0ff" : (isDarkTheme ? "#1e293b" : "#ffffff");
    ctx.stroke();

    // Node Type Icon / Initial
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 11px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(n.type[0], n.x, n.y);

    // Node Label
    ctx.fillStyle = isDarkTheme ? "#f8fafc" : "#0f172a";
    ctx.font = "11px Inter, sans-serif";
    ctx.fillText(n.name, n.x, n.y + 34);
  });

  ctx.restore();
}

// Mouse Handlers
function onMouseDown(e) {
  const rect = canvas.getBoundingClientRect();
  const mouseX = (e.clientX - rect.left - transform.x) / transform.scale;
  const mouseY = (e.clientY - rect.top - transform.y) / transform.scale;

  for (let n of nodes) {
    const dx = n.x - mouseX;
    const dy = n.y - mouseY;
    if (Math.sqrt(dx * dx + dy * dy) < 24) {
      draggedNode = n;
      activeNode = n;
      inspectNode(n);
      isDragging = true;
      lastMouse = { x: e.clientX, y: e.clientY };
      return;
    }
  }

  isDragging = true;
  lastMouse = { x: e.clientX, y: e.clientY };
}

function onMouseMove(e) {
  if (!isDragging) return;
  const dx = e.clientX - lastMouse.x;
  const dy = e.clientY - lastMouse.y;

  if (draggedNode) {
    draggedNode.x += dx / transform.scale;
    draggedNode.y += dy / transform.scale;
  } else {
    transform.x += dx;
    transform.y += dy;
  }
  lastMouse = { x: e.clientX, y: e.clientY };
}

function onMouseUp() {
  isDragging = false;
  draggedNode = null;
}

function onWheel(e) {
  e.preventDefault();
  const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
  transform.scale = Math.max(0.4, Math.min(2.5, transform.scale * zoomFactor));
  drawGraph();
}

// 4. Data Loading & API Calls
async function loadInitialData() {
  await loadScenarios();
  await loadMetrics();
  await evaluateScenario("normal_dev");
}

async function loadScenarios() {
  try {
    const res = await fetch("/api/scenarios");
    const data = await res.json();
    renderScenarioCards(data);
    renderStreamList(data);
  } catch (err) {
    console.error("Failed to load scenarios:", err);
  }
}

function renderScenarioCards(scenarios) {
  const grid = document.getElementById("scenarioGrid");
  grid.innerHTML = scenarios.map(s => `
    <div class="scenario-card">
      <div class="scenario-info">
        <div class="stream-item-header">
          <span class="badge ${s.category === 'NORMAL' ? 'badge-success' : 'badge-danger'}">
            ${s.threat_type || s.category}
          </span>
        </div>
        <div class="scenario-name">${s.name}</div>
        <div class="scenario-desc">${s.description}</div>
      </div>
      <button class="btn btn-primary btn-sm" onclick="evaluateScenario('${s.id}')">
        Simulate
      </button>
    </div>
  `).join("");
}

function renderStreamList(scenarios) {
  const list = document.getElementById("streamList");
  list.innerHTML = scenarios.map((s, idx) => `
    <div class="stream-item ${idx === 0 ? 'active' : ''}" onclick="evaluateScenario('${s.id}')">
      <div class="stream-item-header">
        <span class="badge ${s.category === 'NORMAL' ? 'badge-success' : 'badge-danger'}">
          ${s.category}
        </span>
        <span class="font-mono text-subtle" style="font-size: 0.7rem;">00:${10 + idx * 5}s</span>
      </div>
      <div class="stream-item-title">${s.name}</div>
      <div class="stream-item-desc">${s.description.substring(0, 75)}...</div>
    </div>
  `).join("");
}

async function evaluateScenario(scenarioId) {
  try {
    const res = await fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario_id: scenarioId })
    });
    const result = await res.json();
    applyEvaluationResult(result);
  } catch (err) {
    console.error("Evaluation error:", err);
  }
}

function applyEvaluationResult(result) {
  currentGraphData = result;

  // Title & description
  document.getElementById("currentScenarioTitle").textContent = result.scenario;
  document.getElementById("currentScenarioDesc").textContent = result.risk_level + " • Action: " + result.action;

  // Verdict Banner
  const banner = document.getElementById("verdictBanner");
  const statusEl = document.getElementById("verdictStatus");
  const riskLevelEl = document.getElementById("verdictRiskLevel");
  const iconEl = document.getElementById("verdictIcon");

  statusEl.textContent = result.verdict;
  riskLevelEl.textContent = result.risk_level;

  banner.className = "verdict-banner";
  if (result.action === "ALLOW") {
    banner.classList.add("verdict-allow");
    iconEl.textContent = "✔";
  } else if (result.action === "BLOCK") {
    banner.classList.add("verdict-block");
    iconEl.textContent = "⛔";
  } else {
    banner.classList.add("verdict-challenge");
    iconEl.textContent = "⚠️";
  }

  // Risk Gauge
  const gaugeVal = document.getElementById("gaugeValue");
  const gaugeProg = document.getElementById("gaugeProgress");
  const pct = result.risk_percentage;
  gaugeVal.textContent = pct + "%";

  // Stroke offset calculation: 408 is full circumference
  const offset = 408 - (408 * (pct / 100));
  gaugeProg.style.strokeDashoffset = offset;
  gaugeProg.style.stroke = pct < 30 ? "var(--accent-emerald)" : (pct < 60 ? "var(--accent-amber)" : "var(--accent-crimson)");

  // Metric Boxes
  document.getElementById("metricSvddDist").textContent = result.scores.svdd_distance;
  document.getElementById("metricSsScore").textContent = result.scores.self_supervised_confidence;
  document.getElementById("metricTotalScore").textContent = result.scores.total_score;

  // Populate Graph Canvas Nodes
  setupGraphNodes(result.graph);

  // Populate Explainability Table
  renderAttributionTable(result.attributions);

  // Redraw SVDD plot
  drawSvddPlot(result.scores.svdd_distance);
}

function setupGraphNodes(graph) {
  nodes = graph.nodes.map((n, i) => {
    // Arrange in circle initially
    const angle = (i / graph.nodes.length) * Math.PI * 2;
    return {
      id: n.id,
      name: n.name,
      type: n.type,
      type_id: n.type_id,
      features: n.features,
      x: Math.cos(angle) * 110,
      y: Math.sin(angle) * 110
    };
  });
  edges = graph.edges;
  centerGraph();
}

function inspectNode(node) {
  const content = document.getElementById("inspectorContent");
  content.innerHTML = `
    <div style="margin-bottom: 6px;"><strong>${node.name}</strong> (${node.type})</div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px; font-size: 0.72rem;">
      <div>Feature Vector:</div>
      <div class="font-mono">[${node.features.join(", ")}]</div>
    </div>
  `;
}

function renderAttributionTable(attributions) {
  const tbody = document.getElementById("attributionTableBody");
  if (!tbody) return;

  tbody.innerHTML = attributions.map(a => `
    <tr>
      <td><span class="badge ${a.is_suspicious ? 'badge-danger' : 'badge-info'} font-mono">${a.source_type} → ${a.target_type} [${a.relation}]</span></td>
      <td><strong>${a.source_node}</strong></td>
      <td><strong>${a.target_node}</strong></td>
      <td class="font-mono ${a.is_suspicious ? 'text-crimson' : ''}">${a.deviation_score}</td>
      <td>${a.explanation}</td>
    </tr>
  `).join("");
}

function drawSvddPlot(currentDist = 0.1) {
  const sCanvas = document.getElementById("svddCanvas");
  if (!sCanvas) return;
  const sCtx = sCanvas.getContext("2d");
  const rect = sCanvas.parentElement.getBoundingClientRect();
  sCanvas.width = rect.width;
  sCanvas.height = 240;

  const cx = sCanvas.width / 2;
  const cy = sCanvas.height / 2;
  const radius = 65;

  sCtx.clearRect(0, 0, sCanvas.width, sCanvas.height);

  // Draw Hypersphere Boundary
  sCtx.beginPath();
  sCtx.arc(cx, cy, radius, 0, Math.PI * 2);
  sCtx.strokeStyle = "rgba(16, 185, 129, 0.5)";
  sCtx.lineWidth = 2;
  sCtx.fillStyle = isDarkTheme ? "rgba(16, 185, 129, 0.05)" : "rgba(16, 185, 129, 0.08)";
  sCtx.fill();
  sCtx.stroke();

  // Draw Center c
  sCtx.beginPath();
  sCtx.arc(cx, cy, 6, 0, Math.PI * 2);
  sCtx.fillStyle = "#00f0ff";
  sCtx.fill();

  // Draw Active Graph point
  const isAnomaly = currentGraphData && currentGraphData.is_anomaly;
  const angle = Math.random() * Math.PI * 2;
  const dist = isAnomaly ? radius + (currentDist * 80) : radius * Math.min(0.85, currentDist * 2);

  const px = cx + Math.cos(angle) * dist;
  const py = cy + Math.sin(angle) * dist;

  sCtx.beginPath();
  sCtx.arc(px, py, 8, 0, Math.PI * 2);
  sCtx.fillStyle = isAnomaly ? "#ef4444" : "#10b981";
  sCtx.fill();
  sCtx.strokeStyle = "#ffffff";
  sCtx.lineWidth = 2;
  sCtx.stroke();
}

async function loadMetrics() {
  try {
    const res = await fetch("/api/metrics");
    const data = await res.json();
    renderAblationTable(data);
  } catch (err) {
    console.error("Failed to load metrics:", err);
  }
}

function renderAblationTable(metrics) {
  const tbody = document.getElementById("ablationTableBody");
  if (!tbody) return;

  const full = metrics.full || {};
  const baseline = metrics["vanilla-hetgcn"] || {};

  // Dynamically update KPI Cards
  const kpiAuc = document.getElementById("kpiAuc");
  const kpiAp = document.getElementById("kpiAp");
  const kpiLatency = document.getElementById("kpiLatency");
  const kpiAucSub = document.getElementById("kpiAucSub");
  const kpiApSub = document.getElementById("kpiApSub");

  if (kpiAuc && full.auc) kpiAuc.textContent = `${full.auc}%`;
  if (kpiAp && full.ap) kpiAp.textContent = `${full.ap}%`;
  if (kpiLatency && full.latency_ms) kpiLatency.textContent = `${full.latency_ms} ms`;

  if (kpiAucSub && full.auc && baseline.auc) {
    const diff = roundNum(full.auc - baseline.auc);
    kpiAucSub.textContent = `${diff >= 0 ? '+' : ''}${diff}% vs Baseline HetGCN`;
  }
  if (kpiApSub && full.ap && baseline.ap) {
    const diff = roundNum(full.ap - baseline.ap);
    kpiApSub.textContent = `${diff >= 0 ? '+' : ''}${diff}% vs Baseline HetGCN`;
  }

  tbody.innerHTML = Object.entries(metrics).map(([key, item]) => {
    const diff = roundNum(item.auc - (baseline.auc || 80.0));
    return `
      <tr>
        <td><strong>${item.name}</strong></td>
        <td class="font-mono text-emerald">${item.auc}%</td>
        <td class="font-mono text-cyan">${item.ap}%</td>
        <td class="font-mono">${item.latency_ms} ms</td>
        <td><span class="badge ${diff >= 0 ? 'badge-success' : 'badge-danger'} font-mono">${diff >= 0 ? '+' : ''}${diff}%</span></td>
      </tr>
    `;
  }).join("");
}

function roundNum(n) {
  return Math.round(n * 10) / 10;
}

// 5. Event Listeners
function initEventListeners() {
  // Test Random Event button
  document.getElementById("btnGenerateSample").addEventListener("click", async () => {
    const isAttack = Math.random() > 0.5;
    const res = await fetch(`/api/sample_graph?mode=${isAttack ? 'attack' : 'normal'}`);
    const graphData = await res.json();
    const evalRes = await fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ custom_graph: graphData })
    });
    const evalResult = await evalRes.json();
    applyEvaluationResult(evalResult);
  });

  // Canvas Reset & Toggle Buttons
  document.getElementById("btnResetCanvas").addEventListener("click", centerGraph);
  document.getElementById("btnToggleLabels").addEventListener("click", () => {
    showEdgeLabels = !showEdgeLabels;
    drawGraph();
  });

  // Custom Request Form Submission
  const form = document.getElementById("customRequestForm");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const user = document.getElementById("custUser").value;
    const device = document.getElementById("custDevice").value;
    const role = document.getElementById("custRole").value;
    const resource = document.getElementById("custResource").value;
    const mfa = parseFloat(document.getElementById("custMfa").value);
    const bypass = document.getElementById("custBypass").value;

    const isHighRisk = (mfa === 0.0) || (bypass !== "none") || device.includes("Android") || device.includes("Tor");

    const customGraph = {
      scenario: `Custom Request (${user.split("_")[0]} -> ${resource.split("_")[0]})`,
      label: isHighRisk ? 1 : 0,
      nodes: [
        { name: user, type_id: 0, features: [0.2, 0.5, mfa, 0.1, 0.3, 0.0, 0.4, 0.9] },
        { name: device, type_id: 1, features: [device.includes("Mac") ? 0.95 : 0.2, 1.0, 0.8, 0.9, 1.0, 1.0, 0.9, 1.0] },
        { name: role, type_id: 2, features: [role.includes("SuperAdmin") ? 1.0 : 0.4, 0.0, 0.3, 0.3, 1.0, 0.0, 0.4, 0.5] },
        { name: resource, type_id: 3, features: [resource.includes("KMS") ? 1.0 : 0.4, 1.0, 0.5, 1.0, 0.0, 0.0, 0.2, 1.0] }
      ],
      edges: [
        { source: 0, target: 1, relation_id: 0 },
        { source: 0, target: 2, relation_id: 1 },
        { source: bypass === "direct_device" ? 1 : (bypass === "direct_user" ? 0 : 2), target: 3, relation_id: bypass === "direct_device" ? 3 : (bypass === "direct_user" ? 2 : 4) }
      ]
    };

    const res = await fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ custom_graph: customGraph })
    });
    const result = await res.json();
    applyEvaluationResult(result);

    // Switch to Monitor tab to view graph
    document.querySelector('[data-tab="tab-monitor"]').click();
  });
}
