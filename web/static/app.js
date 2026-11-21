/* NIDS web demo – vanilla JS + Chart.js */
const FAM_COLORS = { BENIGN: "#6F9AC7", Botnet: "#B3261E", BruteForce: "#7B3FA0", DDoS: "#D9480F", DoS: "#F18B28",
  Infiltration: "#1E7B4F", WebAttack: "#C2185B" };
const NAVY = "#002E5D", ORANGE = "#F18B28";
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const fmt = (v, d = 0) => v == null || Number.isNaN(v) ? "–" : Number(v).toLocaleString("vi-VN", { maximumFractionDigits: d, minimumFractionDigits: d });
const pct = (v, d = 1) => v == null ? "–" : fmt(v * 100, d) + "%";
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const sev = (s) => `<span class="sev" style="background:var(--${s})">${s}</span>`;
const fam = (f) => `<span class="fam" style="background:${(FAM_COLORS[f] || "#9AA9BC")}22;color:${FAM_COLORS[f] || "#3C4C61"}">${esc(f)}</span>`;
Chart.defaults.font.family = "Arial, 'Segoe UI', sans-serif";
Chart.defaults.color = "#3C4C61";
const charts = {};
function chart(id, cfg) { if (charts[id]) charts[id].destroy(); const el = document.getElementById(id); if (!el) return; charts[id] = new Chart(el, cfg); return charts[id]; }
function toast(msg, err = false) { const t = $("#toast"); t.textContent = msg; t.className = "show" + (err ? " err" : ""); clearTimeout(t._h); t._h = setTimeout(() => (t.className = ""), 4200); }
async function api(url, opt) { const r = await fetch(url, opt); if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m; } catch {} throw new Error(m); } return r.json(); }

/* ------------------------------------------------------------- navigation */
let INFO = null, SAMPLES = [];
function go(page) {
  $$("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.page === page));
  $$(".page").forEach((p) => p.classList.toggle("active", p.id === "page-" + page));
  if (!location.hash.startsWith("#" + page)) history.replaceState(null, "", "#" + page);
  if (page === "results") loadResults();
  if (page === "flow" && !FLOW.loaded) randomFlow();
  window.scrollTo(0, 0);
}
$$("#nav a").forEach((a) => a.addEventListener("click", () => go(a.dataset.page)));
document.addEventListener("click", (e) => { const g = e.target.closest("[data-goto]"); if (g) go(g.dataset.goto); });

async function init() {
  try {
    [INFO, SAMPLES] = await Promise.all([api("/api/info"), api("/api/samples")]);
  } catch (e) { toast("Không kết nối được server: " + e.message, true); return; }
  const m = INFO.meta || {};
  $("#hero-stats").innerHTML = [
    [INFO.classifier_features.length, "đặc trưng parity-safe của LightGBM"],
    [INFO.class_names.length, "họ lưu lượng (BENIGN + 6 tấn công)"],
    [(m.gaussian || "GMM").replace("GaussianMixture", "GMM "), "mô hình mật độ lưu lượng bình thường"],
    [fmt(INFO.anomaly_threshold, 2), "ngưỡng −log ε (FPR 1% validation)"],
  ].map(([b, s]) => `<div class="hstat"><b>${esc(b)}</b><span>${s}</span></div>`).join("");
  $("#ov-gmm").textContent = `−log p(x) > ${fmt(INFO.anomaly_threshold, 2)} → bất thường`;
  const card = (s) => `<div class="scard" data-sample="${s.id}"><b>${esc(s.title)}</b><p>${esc(s.description)}</p>
     <div class="meta"><span class="tag ${s.kind}">${s.kind.toUpperCase()}</span>${fmt(s.size_mb, 1)} MB · ${esc(s.truth || "")}</div></div>`;
  $("#ov-samples").innerHTML = SAMPLES.map(card).join("");
  $("#sample-list").innerHTML = SAMPLES.map(card).join("");
  $$("[data-sample]").forEach((c) => c.addEventListener("click", () => { go("analyze"); runSample(c.dataset.sample); }));
  $("#rt-sample").innerHTML = SAMPLES.filter((s) => s.kind === "pcap").map((s) => `<option value="${s.id}">${esc(s.title)}</option>`).join("");
  $("#rt-iface").innerHTML = (INFO.interfaces || []).map((i) => `<option>${esc(i)}</option>`).join("");
  $("#fam-chips").innerHTML = ["any", ...INFO.class_names].map((f, i) => `<button class="chip ${i === 0 ? "on" : ""}" data-f="${f}">${f === "any" ? "Bất kỳ" : f}</button>`).join("");
  $$("#fam-chips .chip").forEach((c) => c.addEventListener("click", () => { $$("#fam-chips .chip").forEach((x) => x.classList.remove("on")); c.classList.add("on"); randomFlow(); }));
  const [page, arg] = location.hash.slice(1).split("/");     // e.g. #analyze/synthetic
  if (page && document.getElementById("page-" + page)) go(page);
  if (page === "analyze" && arg) runSample(arg);
  if (page === "realtime" && arg && SAMPLES.some((s) => s.id === arg)) { $("#rt-sample").value = arg; startRT(); }
}

/* ------------------------------------------------------------- analysis */
const drop = $("#drop");
$("#file").addEventListener("change", (e) => e.target.files[0] && upload(e.target.files[0]));
["dragover", "dragenter"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && upload(e.dataTransfer.files[0]));

async function upload(file) {
  const fd = new FormData(); fd.append("file", file);
  showJob({ name: file.name, progress: 0, message: "Đang tải lên…" });
  try { follow(await api("/api/analyze/upload", { method: "POST", body: fd })); } catch (e) { jobError(e.message); }
}
async function runSample(id) {
  const s = SAMPLES.find((x) => x.id === id);
  showJob({ name: s ? s.title : id, progress: 0, message: "Đang khởi động…" });
  try { follow(await api(`/api/analyze/sample/${id}`, { method: "POST" })); } catch (e) { jobError(e.message); }
}
function showJob(j) {
  $("#job").classList.remove("hidden"); $("#result").classList.add("hidden");
  $("#job-name").textContent = j.name; $("#job-msg").textContent = j.message; $("#job-bar").style.width = (j.progress * 100) + "%";
}
function jobError(m) { $("#job-msg").textContent = "Lỗi: " + m; $("#job-bar").style.width = "0"; toast(m, true); }
async function follow(job) {
  for (;;) {
    const j = await api(`/api/jobs/${job.id}`);
    showJob(j); $("#job-msg").textContent = `${j.message} · ${fmt(j.elapsed_s, 1)} s`;
    if (j.status === "done") { $("#job").classList.add("hidden"); renderResult(j.result); return; }
    if (j.status === "error") { jobError(j.error); return; }
    await new Promise((r) => setTimeout(r, 700));
  }
}

function kpis(items) { return `<div class="kpis">${items.map(([v, l, c]) => `<div class="kpi ${c || ""}"><b>${v}</b><span>${l}</span></div>`).join("")}</div>`; }

function renderResult(r) {
  const s = r.summary, el = $("#result");
  el.classList.remove("hidden");
  const src = r.source || {};
  const isPcap = s.packets != null;
  let h = `<h2 class="sec">Kết quả: ${esc(src.title || src.name || "")} ${r.from_cache ? '<small class="muted">(kết quả đã lưu)</small>' : ""}</h2>`;
  h += kpis([
    ...(isPcap ? [[fmt(s.packets), "gói tin"], [fmt(s.capture_span_s / 60, 1) + " phút", "khoảng thời gian capture"]] : []),
    [fmt(s.flows_total), "flow"], [fmt(s.flows_scored), "flow được chấm (≥ 2 gói)"],
    [fmt(s.attack_flows), "flow bị phân loại tấn công", s.attack_flows ? "warn" : ""],
    [fmt(s.anomalies), "flow bất thường (GMM)"], [fmt(s.alerts), "cảnh báo", "dark"],
    ...(s.hosts != null ? [[fmt(s.hosts), "IP nguồn"]] : []), [fmt(s.us_per_flow, 0) + " µs", "suy luận / flow"],
  ]);
  if (s.flows_skipped_short) h += `<div class="note"><b>${fmt(s.flows_skipped_short)} flow chỉ có 1 gói</b> không được chấm ở mức flow (CICFlowMeter không bao giờ xuất loại flow này) nhưng vẫn dùng ở mức host – ví dụ SYN không được trả lời của một đợt quét cổng.</div>`;
  if (src.id === "exercise" || src.id === "ctu_neris" || src.uploaded) h += `<div class="note"><b>Lưu ý dịch chuyển phân phối:</b> mô hình học trên testbed CICIDS2018. Trên mạng khác, LightGBM có thể gán nhầm một phần lưu lượng thường (vd. DoS) và GMM báo “Unknown” nhiều hơn mức 1% – báo cáo khuyến nghị hiệu chỉnh ε bằng <code>scripts/calibrate_anomaly.py</code> trên lưu lượng bình thường của chính mạng đó. Các cảnh báo mức host (quét, beaconing, botnet) dựa trên hành vi nên ít bị ảnh hưởng hơn.</div>`;
  if (s.flagged_flows > s.alert_flows_detailed) h += `<div class="note"><b>${fmt(s.flagged_flows)} flow bị gắn cờ</b> (tấn công hoặc bất thường); bảng cảnh báo chi tiết lấy mẫu đều theo từng họ ${fmt(s.alert_flows_detailed)} flow để giữ trang nhẹ. Số đếm và biểu đồ phía dưới vẫn tính trên toàn bộ flow.</div>`;
  if (r.evaluation) h += evalBlock(r.evaluation);
  h += `<div class="res-grid"><div class="card"><h3>Dự đoán theo họ</h3><canvas id="ch-fam" height="230"></canvas></div>
        <div class="card"><h3>Dòng thời gian <small>${r.timeline.unit === "time" ? "mỗi cột ≈ " + fmt(r.timeline.bin_seconds, 0) + " s" : "theo thứ tự flow"}</small></h3><canvas id="ch-time" height="115"></canvas></div></div>`;
  h += alertBlock(r);
  if (s.has_host_analysis) h += hostBlock(r);
  h += flowTable(r.flows);
  el.innerHTML = h;
  const famE = Object.entries(r.family_counts).sort((a, b) => b[1] - a[1]);
  chart("ch-fam", { type: "doughnut", data: { labels: famE.map((e) => e[0]), datasets: [{ data: famE.map((e) => e[1]), backgroundColor: famE.map((e) => FAM_COLORS[e[0]] || "#999"), borderWidth: 1 }] },
    options: { plugins: { legend: { position: "right" }, tooltip: { callbacks: { label: (c) => `${c.label}: ${fmt(c.raw)} flow` } } }, cutout: "58%" } });
  const tl = r.timeline, n = tl.anomalies.length;
  const labels = [...Array(n).keys()].map((i) => tl.unit === "time" ? new Date((tl.start + i * tl.bin_seconds) * 1000).toISOString().slice(11, 19) : i);
  const ds = Object.entries(tl.series).sort((a, b) => (a[0] === "BENIGN") - (b[0] === "BENIGN")).map(([k, v]) => ({ label: k, data: v, backgroundColor: FAM_COLORS[k] || "#999", stack: "f" }));
  ds.push({ type: "line", label: "Bất thường (GMM)", data: tl.anomalies, borderColor: NAVY, borderWidth: 1.5, pointRadius: 0, tension: .25 });
  chart("ch-time", { type: "bar", data: { labels, datasets: ds }, options: { scales: { x: { stacked: true, ticks: { maxTicksLimit: 10 } }, y: { stacked: true, beginAtZero: true } }, plugins: { legend: { labels: { boxWidth: 12 } } } } });
  wireAlertTable(r); wireFlowTable(r.flows);
  el.scrollIntoView({ behavior: "smooth" });
}

function evalBlock(e) {
  const max = Math.max(...e.confusion.map((r) => Math.max(...r)));
  const rowsum = e.confusion.map((r) => r.reduce((a, b) => a + b, 0));
  let cm = `<table class="t cm"><tr><th>Thực tế ↓ / Dự đoán →</th>${e.labels.map((l) => `<th>${l}</th>`).join("")}</tr>`;
  e.confusion.forEach((row, i) => {
    cm += `<tr><th>${e.labels[i]}</th>${row.map((v, j) => { const a = rowsum[i] ? v / rowsum[i] : 0; const bg = i === j ? `rgba(0,46,93,${0.12 + a * 0.85})` : (v ? `rgba(241,139,40,${0.15 + a * 0.8})` : "#fff");
      return `<td style="background:${bg};color:${a > 0.5 ? "#fff" : "#2B3A4D"}">${fmt(v)}<br><small>${pct(a, 0)}</small></td>`; }).join("")}</tr>`;
  });
  cm += "</table>";
  const pc = `<table class="t"><tr><th>Lớp</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th><th class="num">Số mẫu</th></tr>${e.per_class.map((p) => `<tr><td>${fam(p.class)}</td><td class="num">${fmt(p.precision, 3)}</td><td class="num">${fmt(p.recall, 3)}</td><td class="num">${fmt(p.f1, 3)}</td><td class="num">${fmt(p.support)}</td></tr>`).join("")}</table>`;
  return `<div class="card"><h3>So sánh với nhãn thật <small>${fmt(e.n)} flow có nhãn</small></h3>
    ${kpis([[fmt(e.macro_f1, 3), "Macro-F1", "dark"], [fmt(e.accuracy, 3), "Accuracy"], [pct(e.fpr), "FPR (BENIGN bị báo)"], [pct(e.fnr), "FNR (tấn công bị lọt)"]])}
    <div class="res-grid2"><div style="overflow:auto">${cm}</div><div>${pc}<p class="muted" style="font-size:12px">Infiltration khó vì CIC gán nhãn mọi flow của máy bị chiếm, kể cả lướt web bình thường – xem mục 4.4 của báo cáo.</p></div></div></div>`;
}

let ALERTS = [], ALERT_FILTER = "all";
function alertBlock(r) {
  ALERTS = r.alert_groups;
  const mods = [...new Set(ALERTS.map((a) => a.detection_module))];
  const sevs = Object.entries(r.severity_counts).reverse().map(([k, v]) => `${sev(k)} ${fmt(v)}`).join(" &nbsp; ");
  return `<div class="card"><h3>Cảnh báo <small>gộp theo (nguồn, đích, loại, module) – bấm một dòng để xem lý do</small></h3>
    <div class="filters"><div class="chips" id="alert-mods"><button class="chip on" data-m="all">Tất cả</button>${mods.map((m) => `<button class="chip" data-m="${m}">${m}</button>`).join("")}</div><span style="margin-left:auto">${sevs}</span></div>
    <div class="scroll"><table class="t" id="alert-table"></table></div></div>`;
}
function wireAlertTable() {
  $$("#alert-mods .chip").forEach((c) => c.addEventListener("click", () => { $$("#alert-mods .chip").forEach((x) => x.classList.remove("on")); c.classList.add("on"); ALERT_FILTER = c.dataset.m; drawAlerts(); }));
  drawAlerts();
}
function drawAlerts() {
  const rows = ALERTS.filter((a) => ALERT_FILTER === "all" || a.detection_module === ALERT_FILTER);
  let h = `<tr><th>Mức độ</th><th>Loại</th><th>Module</th><th>Nguồn</th><th>Đích</th><th class="num">Số lần</th><th class="num">Điểm</th></tr>`;
  rows.forEach((a, i) => {
    const dst = a.dst_ip === "*" ? "*" : `${esc(a.dst_ip)}${a.dst_port ? ":" + a.dst_port : ""}`;
    h += `<tr class="clickable" data-i="${i}"><td>${sev(a.severity)}</td><td>${fam(a.attack_type)}</td><td>${esc(a.detection_module)}</td><td>${esc(String(a.src_ip).slice(0, 60))}</td><td>${dst}</td><td class="num">${fmt(a.count)}</td><td class="num">${fmt(a.confidence, 3)}</td></tr>`;
  });
  if (!rows.length) h += `<tr><td colspan="7" class="muted">Không có cảnh báo.</td></tr>`;
  const t = $("#alert-table"); t.innerHTML = h;
  $$("tr.clickable", t).forEach((tr) => tr.addEventListener("click", () => {
    const nx = tr.nextElementSibling; if (nx && nx.classList.contains("detail")) { nx.remove(); return; }
    const a = rows[+tr.dataset.i];
    const kv = Object.entries(a.relevant_features || {}).map(([k, v]) => `<span><b>${esc(k)}</b>: ${typeof v === "object" ? esc(JSON.stringify(v)) : esc(v)}</span>`).join("");
    tr.insertAdjacentHTML("afterend", `<tr class="detail"><td colspan="7">${esc(a.reason)}<br><small class="muted">Lần đầu ${esc(a.first_seen)} · lần cuối ${esc(a.last_seen)}${a.anomaly_score != null ? " · điểm bất thường " + fmt(a.anomaly_score, 1) : ""}</small><div class="kv">${kv}</div></td></tr>`);
  }));
}

function hostBlock(r) {
  let h = `<div class="res-grid2">`;
  h += `<div class="card"><h3>IP nguồn đáng chú ý</h3><div class="scroll"><table class="t"><tr><th>IP</th><th class="num">Flow</th><th class="num">Flow tấn công</th><th class="num">Bất thường</th><th class="num">Số đích</th><th>Họ chính</th></tr>
    ${r.top_sources.map((s) => `<tr><td>${esc(s.ip)}</td><td class="num">${fmt(s.flows)}</td><td class="num">${fmt(s.attack_flows)}</td><td class="num">${fmt(s.anomalies)}</td><td class="num">${fmt(s.dsts)}</td><td>${fam(s.top_family)}</td></tr>`).join("")}</table></div></div>`;
  const hs = r.hosts.filter((x) => x.botnet_score != null);
  h += `<div class="card"><h3>Mô hình hành vi host <small>Random Forest học trên CTU-13, cửa sổ 300 s</small></h3><div class="scroll"><table class="t"><tr><th>Host</th><th class="num">Điểm botnet</th><th class="num">Cửa sổ ≥ 0,5</th><th class="num">IP đích</th><th class="num">Cổng đích</th><th class="num">Tỷ lệ thất bại</th></tr>
    ${hs.map((x) => `<tr><td>${esc(x.src_ip)}</td><td class="num"><b style="color:${x.botnet_score >= 0.8 ? "var(--CRITICAL)" : x.botnet_score >= 0.5 ? "var(--HIGH)" : "inherit"}">${fmt(x.botnet_score, 2)}</b></td><td class="num">${fmt(x.flagged_windows)} / ${fmt(x.windows)}</td><td class="num">${fmt(x.unique_dst_ips)}</td><td class="num">${fmt(x.unique_dst_ports)}</td><td class="num">${pct(x.failed_ratio, 0)}</td></tr>`).join("") || '<tr><td colspan="6" class="muted">Không có dữ liệu host.</td></tr>'}</table></div></div></div>`;
  const tp = r.topology || {};
  const vtxt = { centralized: "Tập trung (một C&C chung)", p2p: "Ngang hàng (P2P)", no_coordinated_structure: "Không có cấu trúc phối hợp", insufficient_evidence: "Không đủ bằng chứng" };
  h += `<div class="res-grid2"><div class="card"><h3>Beaconing <small>min(CV, CV bền vững) ≤ 0,25 · ≥ 8 kết nối · chu kỳ ≥ 2 s</small></h3><div class="scroll"><table class="t"><tr><th>Nguồn</th><th>Đích</th><th class="num">Kết nối</th><th class="num">Chu kỳ (s)</th><th class="num">CV</th><th class="num">CV bền vững</th><th>Beacon</th></tr>
    ${r.beacons.map((b) => `<tr><td>${esc(b.src)}</td><td>${esc(b.dst)}:${b.port}</td><td class="num">${b.n}</td><td class="num">${fmt(b.interval, 1)}</td><td class="num">${fmt(b.cv, 3)}</td><td class="num">${fmt(b.robust_cv, 3)}</td><td>${b.is_beacon ? '<b style="color:var(--HIGH)">có</b>' : "không"}</td></tr>`).join("") || '<tr><td colspan="7" class="muted">Không có cặp đủ điều kiện.</td></tr>'}</table></div>
    <p class="muted" style="font-size:12px;margin:8px 0 0">Ứng dụng hợp lệ (keep-alive, cập nhật) cũng có chu kỳ → beaconing chỉ là tín hiệu bổ trợ.</p></div>
    <div class="card"><h3>Phân tích topology</h3>${tp.verdict ? `<div class="verdict">${vtxt[tp.verdict] || tp.verdict}</div><p class="muted">Phiếu: ${esc(JSON.stringify(tp.votes || {}))}</p>` : '<p class="muted">Cần ít nhất 2 host nghi vấn để phân tích cấu trúc phối hợp.</p>'}
    <p style="font-size:12.5px;color:var(--body)">Bỏ phiếu bằng: đích hiếm được ≥ 50% máy nghi vấn kết nối (C&C chung), peer hiếm dùng chung, liên kết trực tiếp giữa các máy nghi vấn; so với mức phổ biến toàn mạng để loại DNS/web phổ biến.</p></div></div>`;
  return h;
}

let FLOWS = [], FLOW_FILTER = "flagged";
function flowTable(rows) {
  FLOWS = rows;
  return `<div class="card"><h3>Flow <small>tối đa ${rows.length} dòng: flow đáng chú ý trước</small></h3>
    <div class="filters"><div class="chips" id="flow-f"><button class="chip on" data-f="flagged">Tấn công / bất thường</button><button class="chip" data-f="all">Tất cả</button>${rows.some((r) => r.truth) ? '<button class="chip" data-f="wrong">Dự đoán sai</button>' : ""}</div></div>
    <div class="scroll"><table class="t" id="flow-table"></table></div></div>`;
}
function wireFlowTable() {
  $$("#flow-f .chip").forEach((c) => c.addEventListener("click", () => { $$("#flow-f .chip").forEach((x) => x.classList.remove("on")); c.classList.add("on"); FLOW_FILTER = c.dataset.f; drawFlows(); }));
  drawFlows();
}
function drawFlows() {
  const rows = FLOWS.filter((r) => FLOW_FILTER === "all" || (FLOW_FILTER === "flagged" ? (r.pred !== "BENIGN" || r.anom) : (r.truth && r.truth !== r.pred)));
  const truth = FLOWS.some((r) => r.truth);
  let h = `<tr><th>Nguồn</th><th>Đích</th><th>Proto</th><th class="num">Gói</th><th class="num">Byte</th><th class="num">Thời lượng (s)</th><th>Dự đoán</th><th class="num">Xác suất</th><th class="num">−log p(x)</th>${truth ? "<th>Nhãn thật</th>" : ""}</tr>`;
  rows.forEach((r) => {
    h += `<tr><td>${esc(r.src)}${r.sport ? ":" + r.sport : ""}</td><td>${esc(r.dst)}:${r.dport ?? ""}</td><td>${r.proto}</td><td class="num">${fmt(r.pkts)}</td><td class="num">${fmt(r.bytes)}</td><td class="num">${fmt(r.dur_s, 3)}</td><td>${fam(r.pred)}</td><td class="num">${fmt(r.conf, 3)}</td><td class="num" style="${r.anom ? "color:var(--HIGH);font-weight:bold" : ""}">${fmt(r.score, 1)}</td>${truth ? `<td>${fam(r.truth)}</td>` : ""}</tr>`;
  });
  if (!rows.length) h += `<tr><td colspan="10" class="muted">Không có flow phù hợp.</td></tr>`;
  $("#flow-table").innerHTML = h;
}

/* ------------------------------------------------------------- single flow */
const FLOW = { loaded: false, original: {}, model: [] };
async function randomFlow() {
  const f = ($("#fam-chips .chip.on") || {}).dataset?.f || "any";
  try {
    const d = await api(`/api/flow/random?family=${encodeURIComponent(f)}`);
    FLOW.loaded = true; FLOW.original = d.features; FLOW.model = d.model_features;
    $("#flow-truth").innerHTML = `nhãn thật: ${fam(d.family)} <span class="muted">${esc(d.fine_label || "")}</span>`;
    const model = d.model_features.filter((k) => k in d.features), other = Object.keys(d.features).filter((k) => !d.model_features.includes(k));
    const row = (k) => `<label class="${d.model_features.includes(k) ? "model" : ""}" for="f_${k}">${esc(k)}</label><input id="f_${k}" data-k="${esc(k)}" value="${d.features[k] ?? ""}">`;
    $("#feat-form").innerHTML = `<div class="feat-sep">24 ĐẶC TRƯNG MÔ HÌNH DÙNG</div>${model.map(row).join("")}<div class="feat-sep">CÁC CỘT KHÁC (BỘ LÀM SẠCH / GMM)</div>${other.map(row).join("")}`;
    $$("#feat-form input").forEach((i) => i.addEventListener("input", () => i.classList.toggle("changed", String(FLOW.original[i.dataset.k]) !== i.value)));
    predict();
  } catch (e) { toast(e.message, true); }
}
async function predict() {
  const features = {}; $$("#feat-form input").forEach((i) => (features[i.dataset.k] = i.value === "" ? null : Number(i.value)));
  try {
    const p = await api("/api/flow/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ features }) });
    const al = p.decision.alert;
    $("#flow-decision").innerHTML = `<h3>Quyết định của pipeline</h3><div class="decision"><div class="big" style="color:${FAM_COLORS[p.predicted]}">${p.predicted}</div>
      <div>xác suất <b>${fmt(p.confidence, 3)}</b> · GMM: ${p.is_anomaly ? '<b style="color:var(--HIGH)">bất thường</b>' : "bình thường"}<br>
      ${al ? `${sev(al.severity)} <b>${esc(al.attack_type)}</b> – ${esc(al.reason)}` : '<span class="muted">Không sinh cảnh báo (BENIGN và không bất thường).</span>'}
      ${p.decision.to_host_analysis ? '<br><small class="muted">→ host được chuyển xuống tầng phân tích hành vi.</small>' : ""}</div></div>`;
    const pe = Object.entries(p.probabilities);
    chart("ch-proba", { type: "bar", data: { labels: pe.map((e) => e[0]), datasets: [{ data: pe.map((e) => e[1]), backgroundColor: pe.map((e) => FAM_COLORS[e[0]]) }] },
      options: { indexAxis: "y", scales: { x: { min: 0, max: 1 } }, plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => fmt(c.raw, 4) } } } } });
    const ce = p.contributions.slice().reverse();
    chart("ch-contrib", { type: "bar", data: { labels: ce.map((c) => `${c.feature} = ${fmt(c.value, 2)}`), datasets: [{ data: ce.map((c) => c.contribution), backgroundColor: ce.map((c) => (c.contribution >= 0 ? ORANGE : "#6F9AC7")) }] },
      options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: `đóng góp vào log-odds của lớp ${p.predicted}` } } } } });
    const thr = p.anomaly_threshold, sc = p.anomaly_score, maxv = Math.max(thr * 2, sc * 1.1, 1);
    const posS = Math.min(100, Math.max(0, (sc / maxv) * 100)), posT = (thr / maxv) * 100;
    $("#gauge").innerHTML = `<div style="position:absolute;left:0;right:0;top:20px;height:12px;border-radius:6px;background:linear-gradient(90deg,#6F9AC7 0%,#6F9AC7 ${posT}%,#F6B36B ${posT}%,#D9480F 100%)"></div>
      <div style="position:absolute;left:${posT}%;top:12px;height:28px;border-left:2px dashed #002E5D"></div><div style="position:absolute;left:calc(${posT}% + 4px);top:0;font-size:11px;color:#002E5D">ε: ${fmt(thr, 2)}</div>
      <div style="position:absolute;left:calc(${posS}% - 7px);top:16px;width:14px;height:20px;background:#002E5D;border-radius:3px"></div><div style="position:absolute;left:calc(${posS}% - 30px);top:38px;font-size:12px;font-weight:bold">−log p(x) = ${fmt(sc, 2)}</div>`;
    $("#z-list").innerHTML = `<div style="grid-column:1/-1;border:0;color:var(--muted)">Đặc trưng lệch nhiều nhất so với lưu lượng bình thường (z-score sau log + chuẩn hoá):</div>` +
      p.anomaly_features.map((z) => `<div><span>${esc(z.feature)}</span><b style="color:${Math.abs(z.z) > 3 ? "var(--HIGH)" : "inherit"}">${fmt(z.z, 2)}</b></div>`).join("");
  } catch (e) { toast(e.message, true); }
}
$("#btn-random").addEventListener("click", randomFlow);
$("#btn-predict").addEventListener("click", predict);

/* ------------------------------------------------------------- realtime */
let WS = null, RT = null;
$$("#rt-mode button").forEach((b) => b.addEventListener("click", () => {
  $$("#rt-mode button").forEach((x) => x.classList.remove("on")); b.classList.add("on");
  const live = b.dataset.v === "live";
  $("#rt-sample").classList.toggle("hidden", live); $("#rt-iface").classList.toggle("hidden", !live); $("#rt-speed").parentElement.classList.toggle("hidden", live);
  if (live && !INFO.live_enabled) toast("Bắt gói trực tiếp đang tắt: chạy server trên Kali với NIDS_ALLOW_LIVE=1 và quyền root.", true);
}));
$("#rt-start").addEventListener("click", startRT);
$("#rt-stop").addEventListener("click", () => WS && WS.readyState === 1 && WS.send(JSON.stringify({ action: "stop" })));
function startRT() {
  if (WS) WS.close();
  const live = $("#rt-mode .on").dataset.v === "live";
  RT = { pts: [], fams: {}, alerts: 0 };
  $("#rt-feed").innerHTML = ""; drawRT({ stats: {}, families: {} });
  WS = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/realtime`);
  WS.onopen = () => WS.send(JSON.stringify(live ? { mode: "live", interface: $("#rt-iface").value } : { mode: "replay", sample: $("#rt-sample").value, speed: +$("#rt-speed").value }));
  WS.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "started") { $("#rt-state").textContent = "● đang chạy"; $("#rt-state").style.color = "var(--ok)"; $("#rt-start").disabled = true; $("#rt-stop").disabled = false; }
    else if (m.type === "error") { toast(m.message, true); endRT("lỗi"); }
    else { RT.pts.push(m.point); m.new_alerts.forEach(feed); drawRT(m); if (m.type === "done") endRT("đã xong"); }
  };
  WS.onclose = () => endRT($("#rt-state").textContent.includes("xong") ? "đã xong" : "đã dừng");
}
function endRT(txt) { $("#rt-state").textContent = "■ " + txt; $("#rt-state").style.color = "var(--muted)"; $("#rt-start").disabled = false; $("#rt-stop").disabled = true; }
function feed(a) {
  const d = document.createElement("div"); d.className = "it";
  d.innerHTML = `<div>${sev(a.severity)}</div><div>${fam(a.attack_type)}<br><small class="muted">${esc(a.detection_module)}</small></div><div><b>${esc(String(a.src_ip).slice(0, 60))}</b> → ${esc(a.dst_ip)}${a.dst_port ? ":" + a.dst_port : ""} <small class="muted">${esc(a.timestamp)}</small><br><small>${esc(a.reason)}</small></div>`;
  const f = $("#rt-feed"); f.prepend(d); while (f.children.length > 150) f.lastChild.remove();
}
function drawRT(m) {
  const s = m.stats || {};
  $("#rt-kpis").innerHTML = kpis([
    [fmt(s.packets), "gói đã xử lý"], [fmt(s.pps), "gói / giây"], [fmt(s.flows_scored), "flow đã chấm"],
    [fmt(s.anomalies), "bất thường (GMM)"], [fmt(s.alerts), "cảnh báo", "dark"], [fmt(s.detection_runs), "lần chạy phát hiện"],
    [s.batch_ms != null ? fmt(s.batch_ms, 0) + " ms" : "–", "suy luận lô gần nhất"], [s.packet_time ? new Date(s.packet_time * 1000).toISOString().slice(11, 19) : "–", "thời gian gói (UTC)"],
  ]).replace(/^<div class="kpis">|<\/div>$/g, "");
  const p = RT ? RT.pts : [];
  chart("ch-rt", { type: "line", data: { labels: p.map((x) => x.t + "s"), datasets: [
    { label: "Gói (nghìn)", data: p.map((x) => x.packets / 1000), borderColor: "#6F9AC7", pointRadius: 0, yAxisID: "y" },
    { label: "Flow đã chấm", data: p.map((x) => x.flows), borderColor: NAVY, pointRadius: 0, yAxisID: "y1" },
    { label: "Cảnh báo", data: p.map((x) => x.alerts), borderColor: ORANGE, pointRadius: 0, yAxisID: "y1" }] },
    options: { animation: false, scales: { x: { ticks: { maxTicksLimit: 10 } }, y: { beginAtZero: true, title: { display: true, text: "nghìn gói" } }, y1: { beginAtZero: true, position: "right", grid: { display: false } } } } });
  const fe = Object.entries(m.families || {}).sort((a, b) => b[1] - a[1]);
  chart("ch-rt-fam", { type: "bar", data: { labels: fe.map((e) => e[0]), datasets: [{ data: fe.map((e) => e[1]), backgroundColor: fe.map((e) => FAM_COLORS[e[0]]) }] },
    options: { animation: false, indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { type: "logarithmic" } } } });
}

/* ------------------------------------------------------------- results */
let RES = null;
async function loadResults() {
  if (RES) return;
  try { RES = await api("/api/results"); } catch (e) { toast(e.message, true); return; }
  const r = RES; let h = "";
  const tbl = (rows, cols) => `<div class="scroll" style="max-height:none"><table class="t"><tr>${cols.map((c) => `<th class="${c[2] || ""}">${c[1]}</th>`).join("")}</tr>${rows.map((x) => `<tr>${cols.map((c) => `<td class="${c[2] || ""}">${c[3] ? c[3](x[c[0]], x) : esc(x[c[0]])}</td>`).join("")}</tr>`).join("")}</table></div>`;
  const f3 = (v) => fmt(v, 3), p1 = (v) => pct(v, 1);
  for (const task of ["multiclass", "binary"]) {
    const v = (r.validation[task] || []).slice().sort((a, b) => b.f1_macro - a.f1_macro);
    if (!v.length) continue;
    h += `<h2 class="sec">${task === "multiclass" ? "Bài toán 7 lớp" : "Bài toán nhị phân BENIGN / ATTACK"} – validation (266.305 flow)</h2>
      <div class="res-grid2"><div class="card"><canvas id="ch-v-${task}" height="230"></canvas></div><div class="card">${tbl(v, [["model", "Mô hình"], ["f1_macro", "Macro-F1", "num", f3], ["accuracy", "Accuracy", "num", f3], ["fpr", "FPR", "num", p1], ["fnr", "FNR", "num", p1], ["train_seconds", "Train (s)", "num", (x) => fmt(x, 0)], ["size_mb", "MB", "num", (x) => fmt(x, 1)]])}</div></div>`;
  }
  for (const task of ["multiclass", "binary"]) {
    const t = r.test[task];
    h += `<h2 class="sec">Tập test độc lập – ${task === "multiclass" ? "7 lớp" : "nhị phân"}</h2>`;
    h += t ? `<div class="card">${tbl(t.sort((a, b) => b.f1_macro - a.f1_macro), [["model", "Mô hình"], ["f1_macro", "Macro-F1", "num", f3], ["f1_weighted", "Weighted-F1", "num", f3], ["accuracy", "Accuracy", "num", f3], ["fpr", "FPR", "num", p1], ["fnr", "FNR", "num", p1], ["inference_us_per_flow", "µs/flow", "num", (x) => fmt(x, 1)], ["single_flow_latency_ms", "1 flow (ms)", "num", (x) => fmt(x, 2)]])}</div>`
      : `<div class="note"><b>Đang chạy:</b> <code>scripts/evaluate.py --stage all</code> chưa xong; bảng test sẽ tự hiện khi có <code>reports/metrics/model_comparison_${task}.csv</code>.</div>`;
  }
  if (r.temporal) h += `<h2 class="sec">Chia ngẫu nhiên vs chia theo thời gian</h2><div class="card">${tbl(r.temporal, [["split", "Cách chia"], ["model", "Mô hình"], ["f1_macro", "Macro-F1", "num", f3], ["fpr", "FPR", "num", p1], ["fnr", "FNR", "num", p1]])}</div>`;
  if (r.ablation) h += `<h2 class="sec">Ablation đặc trưng lối tắt</h2><div class="card">${tbl(r.ablation, [["variant", "Biến thể"], ["n_features", "Số đặc trưng", "num"], ["f1_macro", "Macro-F1", "num", f3], ["fpr", "FPR", "num", p1]])}</div>`;
  if (r.anomaly) h += `<h2 class="sec">Phát hiện bất thường – test (ngưỡng chọn trên validation)</h2><div class="res-grid2"><div class="card"><canvas id="ch-anom" height="220"></canvas></div><div class="card">${tbl(r.anomaly, [["detector", "Detector"], ["roc_auc", "ROC-AUC", "num", f3], ["pr_auc", "PR-AUC", "num", f3], ["fpr_1pct_recall", "Recall @1%", "num", f3], ["fpr_5pct_recall", "Recall @5%", "num", f3], ["fpr_5pct_precision", "Precision @5%", "num", f3]])}</div></div>`;
  if (r.lofo) h += `<h2 class="sec">Tấn công chưa từng thấy (leave-one-family-out, FPR ≈ 5%)</h2><div class="card"><canvas id="ch-lofo" height="110"></canvas></div>`;
  if (r.botnet) h += `<h2 class="sec">Phát hiện máy nhiễm – CTU-13 (test: 82 cửa sổ, 11 của bot)</h2><div class="card">${tbl(r.botnet, [["Unnamed: 0", "Bộ phát hiện"], ["precision", "Precision", "num", f3], ["recall", "Recall", "num", f3], ["f1", "F1", "num", f3], ["fpr", "FPR", "num", p1], ["roc_auc", "ROC-AUC", "num", f3]])}<p class="muted" style="font-size:12px">F1 = 1,0 không chứng minh khả năng tổng quát: chỉ có một máy nhiễm trong tập test.</p></div>`;
  if (r.deployment) { const d = r.deployment, ct = d.classifier_test || {}, g = d.gaussian_test_at_1pct_val_fpr || {};
    h += `<h2 class="sec">Mô hình triển khai (đang chạy trong web này)</h2>${kpis([[fmt(ct.f1_macro, 3), "Macro-F1 test – LightGBM 24 đặc trưng", "dark"], [pct(ct.fpr, 2), "FPR test"], [pct(ct.fnr, 2), "FNR test"], [fmt(g.roc_auc, 3), "ROC-AUC GMM (test)"], [fmt(g.recall, 3), "Recall GMM @ FPR 1% val"]])}`; }
  if (r.figures?.length) h += `<h2 class="sec">Hình trong reports/figures</h2><div class="fig-gallery">${r.figures.map((f) => `<a href="/figures/${f}" target="_blank"><img loading="lazy" src="/figures/${f}" alt="${esc(f)}">${esc(f)}</a>`).join("")}</div>`;
  $("#res").innerHTML = h;
  for (const task of ["multiclass", "binary"]) {
    const v = (r.validation[task] || []).slice().sort((a, b) => b.f1_macro - a.f1_macro);
    if (!v.length) continue;
    chart(`ch-v-${task}`, { type: "bar", data: { labels: v.map((x) => x.model), datasets: [{ label: "Macro-F1", data: v.map((x) => x.f1_macro), backgroundColor: NAVY, xAxisID: "x" }, { label: "FPR", data: v.map((x) => x.fpr), backgroundColor: ORANGE, xAxisID: "x1" }] },
      options: { indexAxis: "y", scales: { x: { min: Math.min(0.6, ...v.map((x) => x.f1_macro)) - 0.02, max: 1, position: "bottom", title: { display: true, text: "Macro-F1" } }, x1: { min: 0, max: 0.45, position: "top", grid: { display: false }, title: { display: true, text: "FPR" } } } } });
  }
  if (r.anomaly) chart("ch-anom", { type: "bar", data: { labels: r.anomaly.map((x) => x.detector), datasets: [{ label: "ROC-AUC", data: r.anomaly.map((x) => x.roc_auc), backgroundColor: NAVY }, { label: "PR-AUC", data: r.anomaly.map((x) => x.pr_auc), backgroundColor: ORANGE }] }, options: { scales: { y: { min: 0, max: 1 } } } });
  if (r.lofo) { const l = r.lofo.filter((x) => x.target_val_fpr === 0.05);
    chart("ch-lofo", { type: "bar", data: { labels: l.map((x) => x.held_out_family), datasets: [{ label: "Gaussian mixture (chỉ học BENIGN)", data: l.map((x) => x.gaussian_recall_unknown), backgroundColor: NAVY }, { label: "LightGBM chưa thấy họ này", data: l.map((x) => x.supervised_recall_unknown), backgroundColor: ORANGE }, { label: "Kết hợp hai tầng", data: l.map((x) => x.either_recall_unknown), backgroundColor: "#6F9AC7" }] }, options: { scales: { y: { min: 0, max: 1, title: { display: true, text: "recall trên họ bị giấu" } } } } }); }
}

init();
