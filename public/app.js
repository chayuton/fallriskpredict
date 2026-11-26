/* fallriskpredict web app: reads CSVs in the browser, keeps only the needed columns,
   gzips them, sends them to /api/analyze and renders the results. */
"use strict";

const $ = (id) => document.getElementById(id);
const FRAME_REQ = ["subject_id", "video_id", "frame_idx"];
const FRAME_OPT = ["timestamp", "frame_w", "frame_h"];
const FRAME_LM = [11, 12, 23, 24].flatMap((i) => ["x", "y", "v"].map((a) => `lm${i}_${a}`));
const SETTING_IDS = ["p_thresh", "persist", "persist_on", "velocity_mode", "v_thresh", "frame_h_meters",
  "latency_ms", "fps", "frame_w", "frame_h", "vel_lag", "vel_window", "max_pre_impact_s", "onset_tol_s",
  "max_post_impact_s", "refractory_s"];
const UNITS = { meters: "m/s", torso: "TL/s", pixels: "px/s" };
const MAX_UPLOAD = 4.3e6;
let result = null;
const charts = {};

// ---------------------------------------------------------------- helpers
const fmt = (x, d = 0) => (x === null || x === undefined || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
const pct = (x, d = 1) => (x === null || x === undefined ? "–" : (100 * x).toFixed(d) + "%");
const fmtP = (p) => (typeof p !== "number" ? p : p < 0.001 ? "<0.001" : p.toFixed(3));
function setStatus(msg, cls = "") { const s = $("status"); s.textContent = msg; s.className = cls; }
function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

function settings() {
  const s = {};
  for (const id of SETTING_IDS) { const el = $("s-" + id); if (el && el.value !== "") s[id] = el.value; }
  return s;
}

function download(name, text, type = "text/csv") {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: type + ";charset=utf-8" }));
  a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ---------------------------------------------------------------- CSV reduction in the browser
function parseHeader(file) {
  return new Promise((res, rej) => Papa.parse(file, { preview: 1, header: true, complete: (r) => res(r.meta.fields || []), error: rej }));
}

function reduceCsv(file, cols, rowFn = null, outFields = null) {
  return new Promise((resolve, reject) => {
    const data = [];
    Papa.parse(file, {
      header: true, skipEmptyLines: true, chunkSize: 4 * 1024 * 1024,
      chunk: (r) => { for (const row of r.data) data.push(rowFn ? rowFn(row) : cols.map((c) => row[c])); },
      complete: () => resolve(Papa.unparse({ fields: outFields || cols, data })),
      error: reject,
    });
  });
}

async function readFrames(file) {
  const head = await parseHeader(file);
  const miss = FRAME_REQ.filter((c) => !head.includes(c));
  if (miss.length) throw new Error(`frames.csv ขาดคอลัมน์ ${miss.join(", ")}`);
  const lm = FRAME_LM.filter((c) => head.includes(c));
  for (const c of ["lm23_y", "lm24_y"]) if (!lm.includes(c)) throw new Error(`frames.csv ขาดคอลัมน์ ${c} (ใช้คำนวณความเร็วสะโพก)`);
  const cols = [...FRAME_REQ, ...FRAME_OPT.filter((c) => head.includes(c)), ...lm];
  return reduceCsv(file, cols);
}

async function readPreds(file) {
  const head = await parseHeader(file);
  for (const c of ["video_id", "frame_idx"]) if (!head.includes(c)) throw new Error(`predictions ขาดคอลัมน์ ${c}`);
  if (head.includes("p_pos")) return reduceCsv(file, ["video_id", "frame_idx", "p_pos"]);
  const pos = [...document.querySelectorAll("#pos-list input:checked")].map((e) => e.value);
  if (!pos.length) throw new Error("เลือกคลาสที่นับเป็นสัญญาณเตือนอย่างน้อย 1 คลาส");
  return reduceCsv(file, null, (r) => {
    let s = 0, any = false;
    for (const c of pos) { const v = parseFloat(r[c]); if (!Number.isNaN(v)) { s += v; any = true; } }
    return [r.video_id, r.frame_idx, any ? s.toFixed(5) : ""];
  }, ["video_id", "frame_idx", "p_pos"]);
}

function readText(file) {
  if (file.text) return file.text();
  return new Promise((res, rej) => { const r = new FileReader(); r.onload = () => res(r.result); r.onerror = rej; r.readAsText(file); });
}

async function gzip(text) {
  if (!("CompressionStream" in window)) return { body: new Blob([text]), gz: false };
  const stream = new Blob([text]).stream().pipeThrough(new CompressionStream("gzip"));
  return { body: await new Response(stream).blob(), gz: true };
}

$("f-preds").addEventListener("change", async (e) => {
  const f = e.target.files[0]; const box = $("pos-box"), list = $("pos-list");
  list.innerHTML = ""; box.hidden = true;
  if (!f) return;
  const head = await parseHeader(f);
  const probs = head.filter((c) => c.startsWith("prob_"));
  if (head.includes("p_pos") || !probs.length) return;
  for (const c of probs) {
    const name = c.slice(5);
    const on = /loss|balance|pre/i.test(name) || (probs.length === 2 && c === probs[1]);
    list.insertAdjacentHTML("beforeend", `<label><input type="checkbox" value="${esc(c)}" ${on ? "checked" : ""}> ${esc(name)}</label>`);
  }
  box.hidden = false;
});

// ---------------------------------------------------------------- run
async function call(fn) {
  $("run").disabled = $("demo").disabled = true;
  try { await fn(); } catch (err) { setStatus(String(err.message || err), "bad"); }
  finally { $("run").disabled = $("demo").disabled = false; }
}

$("run").addEventListener("click", () => call(async () => {
  const ff = $("f-frames").files[0], ft = $("f-trials").files[0], fp = $("f-preds").files[0];
  if (!ff || !ft || !fp) throw new Error("เลือกไฟล์ให้ครบทั้ง 3 ไฟล์");
  setStatus("กำลังอ่านไฟล์…");
  const [frames, trials, preds] = await Promise.all([readFrames(ff), readText(ft), readPreds(fp)]);
  const payload = JSON.stringify({ frames, trials, preds, settings: settings(), grid: $("s-grid").checked });
  const { body, gz } = await gzip(payload);
  if (body.size > MAX_UPLOAD) throw new Error(`ข้อมูลหลังบีบอัดยังใหญ่ ${(body.size / 1e6).toFixed(1)} MB (จำกัด 4.3 MB) ให้แบ่งไฟล์ตามผู้ทดลองหรือตามชุดข้อมูล`);
  setStatus(`กำลังวิเคราะห์ (ส่ง ${(body.size / 1e6).toFixed(2)} MB)…`);
  const headers = { "Content-Type": "application/octet-stream" };
  if (gz) headers["X-Content-Encoding"] = "gzip";
  const r = await fetch("/api/analyze", { method: "POST", headers, body });
  await show(r);
}));

$("demo").addEventListener("click", () => call(async () => {
  setStatus("กำลังวิเคราะห์ข้อมูลจำลอง…");
  const q = new URLSearchParams({ demo: "1", ...settings() });
  await show(await fetch("/api/analyze?" + q.toString(), { cache: "no-store" }));
}));

async function show(r) {
  let j;
  try { j = await r.json(); } catch { throw new Error(`เซิร์ฟเวอร์ตอบกลับผิดรูปแบบ (HTTP ${r.status})`); }
  if (!r.ok || j.status !== "ok") throw new Error(j.error || `HTTP ${r.status}`);
  result = j;
  render(j);
  setStatus(`เสร็จ (${j.runtime_s} s)`, "ok");
  $("results").hidden = false;
  $("results").scrollIntoView({ behavior: "smooth" });
}

// ---------------------------------------------------------------- render
function table(el, rows, hl = () => false) {
  if (!rows || !rows.length) { el.innerHTML = "<p class='note'>ไม่มีข้อมูล</p>"; return; }
  const cols = Object.keys(rows[0]);
  el.innerHTML = `<table><thead><tr>${cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>${
    rows.map((r) => `<tr class="${hl(r) ? "hl" : ""}">${cols.map((c) => `<td>${esc(c.startsWith("McNemar") ? fmtP(r[c]) : r[c])}</td>`).join("")}</tr>`).join("")
  }</tbody></table>`;
}

function chart(id, cfg) { if (charts[id]) charts[id].destroy(); charts[id] = new Chart($(id), cfg); }

function render(j) {
  const s = j.summary, st = j.settings, unit = st.velocity_unit;
  $("demo-banner").hidden = !j.demo;
  const kp = [
    ["เตือนก่อนกระแทก", `${s.n_pre_impact}/${s.n_falls}`, pct(s.pre_impact_rate)],
    ["Lead time มัธยฐาน", `${fmt(s.lead_median_ms)} ms`, `IQR ${fmt(s.lead_q1_ms)}–${fmt(s.lead_q3_ms)}`],
    [`เตือน ≥ 100 ms`, pct(s.rate_lead_ge_target_min), "ของการล้มทั้งหมด"],
    ["ADL ที่เตือนผิด", `${s.adl_trials_with_false_alarm}/${s.n_adl}`, `specificity ${pct(s.specificity)}`],
    ["Trial accuracy", pct(s.trial_accuracy), `${j.meta.subjects} คน, ${j.meta.trials} trials`],
  ];
  $("kpis").innerHTML = kp.map(([k, v, sub]) => `<div class="kpi"><div class="k">${k}</div><div class="v">${v}</div><div class="k">${sub}</div></div>`).join("");

  // histogram
  const step = 1000 / st.fps, L = j.lead_times_ms;
  if (L.length) {
    const lo = Math.floor(Math.min(...L) / step), hi = Math.ceil(Math.max(...L) / step);
    const labels = [], counts = [], colors = [];
    for (let b = lo; b <= hi; b++) {
      const c = b * step; labels.push(c.toFixed(0));
      counts.push(L.filter((v) => Math.abs(v - c) < step / 2 + 1e-6).length);
      colors.push(c <= 0 ? "#b3261e" : c >= 100 && c <= 300 ? "#1f7a4d" : "#2456a6");
    }
    chart("c-hist", { type: "bar", data: { labels, datasets: [{ label: "จำนวนการล้ม", data: counts, backgroundColor: colors }] },
      options: { plugins: { legend: { display: false }, title: { display: true, text: `Lead time (ms, หัก latency ${st.latency_ms} ms) — เขียว = 100–300 ms, แดง = ไม่ทันก่อนกระแทก` } },
        scales: { x: { title: { display: true, text: "ms ก่อนกระแทก" } }, y: { beginAtZero: true, ticks: { precision: 0 } } } } });
  }
  table($("t-scen"), j.by_scenario.map((r) => { const { Configuration, ...rest } = r; return { Scenario: Configuration, ...rest }; }));
  table($("t-abl"), j.ablation, (r) => r.Configuration.endsWith("(proposed)"));

  // grid
  $("sec-grid").hidden = !j.grid;
  if (j.grid) renderGrid(j, unit);

  // timelines
  const sel = $("ex-select");
  sel.innerHTML = j.examples.map((e, i) => `<option value="${i}">${esc(e.trial_id)} (${esc(e.scenario)}) ${e.detected ? fmt(e.lead_ms) + " ms" : "พลาด"}</option>`).join("");
  sel.onchange = () => timeline(j.examples[+sel.value], st);
  if (j.examples.length) timeline(j.examples[0], st);

  // paper text
  const abl = j.ablation, gru = abl[0], prop = abl.find((r) => r.Configuration.endsWith("(proposed)"));
  $("paper").textContent =
`Decision rule: GRU p >= ${st.p_thresh} for ${st.persist} consecutive frames (${st.persist_on}) AND hip descent velocity >= ${st.v_thresh} ${unit}.
Of ${s.n_falls} falls, ${s.n_pre_impact} (${pct(s.pre_impact_rate)}) were warned before impact; median effective lead time ${fmt(s.lead_median_ms)} ms (IQR ${fmt(s.lead_q1_ms)}–${fmt(s.lead_q3_ms)} ms, range ${fmt(s.lead_min_ms)} to ${fmt(s.lead_max_ms)} ms) after subtracting ${st.latency_ms} ms processing latency; ${pct(s.rate_lead_ge_target_min)} of falls were warned at least 100 ms before impact (temporal resolution ±${fmt(1000 / st.fps, 1)} ms).
False alarms occurred in ${s.adl_trials_with_false_alarm}/${s.n_adl} ADL trials (specificity ${pct(s.specificity)}).
Ablation: the GRU alone produced false alarms in ${gru["ADL trials with false alarm"]} ADL trials versus ${prop["ADL trials with false alarm"]} with the full rule (exact McNemar p = ${fmtP(gru["McNemar p (ADL FA)"])}).` +
(j.loso_thresholds ? `\nWhen thresholds were selected on the remaining subjects (LOSO), the chosen values matched the fixed thresholds in ${j.loso_thresholds.same_as_chosen}/${j.loso_thresholds.folds} folds.` : "");
}

function renderGrid(j, unit) {
  const st = j.settings, rows = j.grid.rows, ax = j.grid.axes;
  const pUsed = ax.P.reduce((a, b) => (Math.abs(b - st.p_thresh) < Math.abs(a - st.p_thresh) ? b : a));
  $("grid-note").textContent = `ที่ GRU threshold P = ${pUsed}; กรอบดำ = ค่าที่ตั้งไว้ (N = ${st.persist}, V = ${st.v_thresh} ${unit})`;
  const sub = rows.filter((r) => r.P === pUsed);
  const panels = [["pre_impact_rate", "การล้มที่เตือนก่อนกระแทก (%)", 100, [31, 122, 77]],
    ["adl_false_alarm_rate", "ADL ที่เตือนผิด (%)", 100, [179, 38, 30]], ["lead_median_ms", "Lead time มัธยฐาน (ms)", 1, [36, 86, 166]]];
  $("heatmaps").innerHTML = panels.map(([key, title, sc, rgb]) => {
    const vals = sub.map((r) => r[key]).filter((v) => v !== null).map((v) => v * sc);
    const mn = Math.min(...vals), mx = Math.max(...vals);
    const cell = (n, v) => {
      const r = sub.find((x) => x.N === n && x.V === v); const val = r && r[key] !== null ? r[key] * sc : null;
      const t = val === null || mx === mn ? 0 : (val - mn) / (mx - mn);
      const bg = `rgba(${rgb.join(",")},${0.08 + 0.8 * t})`, fg = t > 0.6 ? "#fff" : "#1d2433";
      const selC = n === st.persist && Math.abs(v - st.v_thresh) < 1e-9 ? "sel" : "";
      return `<td class="${selC}" style="background:${bg};color:${fg}">${val === null ? "–" : val.toFixed(0)}</td>`;
    };
    return `<div class="hm"><h3>${title}</h3><table><tr><th>N \\ V</th>${ax.V.map((v) => `<th>${v}</th>`).join("")}</tr>${
      ax.N.map((n) => `<tr><th>${n}</th>${ax.V.map((v) => cell(n, v)).join("")}</tr>`).join("")}</table></div>`;
  }).join("");
  const pts = rows.filter((r) => r.pre_impact_rate !== null && r.adl_false_alarm_rate !== null);
  const me = pts.filter((r) => r.N === st.persist && Math.abs(r.V - st.v_thresh) < 1e-9 && Math.abs(r.P - st.p_thresh) < 1e-9);
  chart("c-trade", { type: "scatter", data: { datasets: [
    { label: "ทุกชุดค่า (N, V, P)", data: pts.map((r) => ({ x: 100 * r.adl_false_alarm_rate, y: 100 * r.pre_impact_rate, r })), backgroundColor: "rgba(36,86,166,.35)" },
    { label: "ค่าที่ตั้งไว้", data: me.map((r) => ({ x: 100 * r.adl_false_alarm_rate, y: 100 * r.pre_impact_rate, r })), backgroundColor: "#b3261e", pointRadius: 9, pointStyle: "star" }] },
    options: { plugins: { tooltip: { callbacks: { label: (c) => { const r = c.raw.r; return `N=${r.N}, V=${r.V}, P=${r.P}: เตือนทัน ${c.raw.y.toFixed(0)}%, FA ${c.raw.x.toFixed(0)}%, lead ${fmt(r.lead_median_ms)} ms`; } } } },
      scales: { x: { title: { display: true, text: "ADL ที่เตือนผิด (%)" } }, y: { title: { display: true, text: "การล้มที่เตือนก่อนกระแทก (%)" } } } } });
  const lt = j.loso_thresholds;
  if (lt) {
    $("loso-box").innerHTML = `<h3>เลือก threshold โดยไม่เห็นผู้ทดลองที่ทดสอบ (LOSO)</h3><div class="table" id="t-loso"></div>
      <p class="note">ค่าที่ตั้งไว้ถูกเลือกใน ${lt.same_as_chosen}/${lt.folds} fold</p>`;
    table($("t-loso"), [lt.row]);
  } else $("loso-box").innerHTML = "";
}

function timeline(e, st) {
  if (!e) return;
  const xs = e.t_ms;
  const vline = (x, color, dash) => ({ type: "line", data: [{ x, y: 0 }, { x, y: 1 }], yAxisID: "yp", borderColor: color,
    borderDash: dash || [], borderWidth: 1.5, pointRadius: 0, label: "_" });
  const ds = [
    { label: `ความเร็วการลงของสะโพก (${st.velocity_unit})`, data: xs.map((x, i) => ({ x, y: e.velocity[i] })), yAxisID: "yv", borderColor: "#c26a00", pointRadius: 0, borderWidth: 1.5 },
    { label: `V threshold`, data: [{ x: xs[0], y: st.v_thresh }, { x: xs[xs.length - 1], y: st.v_thresh }], yAxisID: "yv", borderColor: "#c26a00", borderDash: [5, 4], pointRadius: 0, borderWidth: 1 },
    { label: "GRU p(warning)", data: xs.map((x, i) => ({ x, y: e.p[i] })), yAxisID: "yp", borderColor: "#6a3fb0", pointRadius: 0, borderWidth: 1.5 },
    { label: "เงื่อนไขเตือนเป็นจริง", data: xs.map((x, i) => ({ x, y: e.alert[i] ? 1 : 0 })), yAxisID: "yp", stepped: true, fill: true, backgroundColor: "rgba(179,38,30,.12)", borderWidth: 0, pointRadius: 0 },
    vline(0, "#000"),
  ];
  if (e.detected) ds.push(vline(-e.lead_ms, "#b3261e"));
  if (e.onset_ms !== null) ds.push(vline(e.onset_ms, "#888", [3, 3]));
  chart("c-tl", { type: "line", data: { datasets: ds }, options: { parsing: true, animation: false,
    plugins: { legend: { labels: { filter: (i) => i.text !== "_" } }, title: { display: true,
      text: `${e.trial_id} — ${e.detected ? `lead ${fmt(e.lead_ms)} ms (effective ${fmt(e.eff_lead_ms)} ms)` : "ไม่ได้เตือน"}` } },
    scales: { x: { type: "linear", title: { display: true, text: "เวลาเทียบกับการกระแทก (ms)" } },
      yv: { position: "left", title: { display: true, text: st.velocity_unit } },
      yp: { position: "right", min: 0, max: 1, grid: { drawOnChartArea: false }, title: { display: true, text: "p" } } } } });
}

// ---------------------------------------------------------------- downloads
document.querySelectorAll("[data-dl]").forEach((b) => b.addEventListener("click", () => {
  if (!result) return;
  const k = b.dataset.dl;
  if (k === "per_trial") download("per_trial.csv", Papa.unparse(result.per_trial));
  if (k === "ablation") download("ablation.csv", Papa.unparse(result.ablation));
  if (k === "grid" && result.grid) download("threshold_grid.csv", Papa.unparse(result.grid.rows));
  if (k === "summary") { const { examples, ...rest } = result; download("summary.json", JSON.stringify(rest, null, 2), "application/json"); }
}));

$("s-velocity_mode").addEventListener("change", (e) => {
  const v = $("s-v_thresh"); if (e.target.value === "torso" && v.value === "1.5") v.value = "3";
  if (e.target.value === "meters" && v.value === "3") v.value = "1.5";
});
