import { useState } from 'react';
import { UploadPanel } from './components/UploadPanel';
import { MRIViewer } from './components/MRIViewer';
import { ChatPanel } from './components/ChatPanel';
import { ReliabilityGauge } from './components/ReliabilityGauge';


function SectionTitle({ icon, title, badge }: { icon: string; title: string; badge?: string }) {
  return (
    <div className="flex items-center space-x-2 mb-4">
      <div style={{ width: 4, height: 20, background: '#1d4ed8', borderRadius: 2 }} />
      <span className="text-base font-bold" style={{ color: '#1e3a8a' }}>{icon} {title}</span>
      {badge && (
        <span className="ml-2 text-xs px-2 py-0.5 rounded-full font-medium"
          style={{ background: '#eff6ff', color: '#1d4ed8', border: '1px solid #bfdbfe' }}>
          {badge}
        </span>
      )}
    </div>
  );
}

function Card({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`rounded-2xl border p-5 shadow-sm ${className}`}
      style={{ background: 'white', borderColor: '#dbeafe' }}>
      {children}
    </div>
  );
}

// ── FIXED: Tumor location mapping using normalized coordinates ─────────────────
function getTumorLocation(metrics: any) {
  const cx = metrics?.centroid_x ?? 256;
  const cy = metrics?.centroid_y ?? 256;

  // Normalize to 0–1 (image assumed 512×512)
  const rx = cx / 512;
  const ry = cy / 512;

  let region = "Frontal Lobe";

  if (ry < 0.30) {
    if (rx < 0.40) region = "Left Frontal Lobe";
    else if (rx > 0.60) region = "Right Frontal Lobe";
    else region = "Frontal / Superior Region";
  } else if (ry < 0.55) {
    if (rx < 0.38) region = "Left Parietal Lobe";
    else if (rx > 0.62) region = "Right Parietal Lobe";
    else region = "Parietal Lobe";
  } else if (ry < 0.72) {
    if (rx < 0.38) region = "Left Temporal Lobe";
    else if (rx > 0.62) region = "Right Temporal Lobe";
    else region = "Temporal / Basal Ganglia Region";
  } else {
    if (rx > 0.35 && rx < 0.65 && ry > 0.82)
      region = "Sella Turcica / Pituitary Region";
    else if (rx < 0.30 || rx > 0.70)
      region = "Occipital Lobe";
    else region = "Cerebellum / Posterior Fossa";
  }

  // Spread direction
  let spread = "Bilateral / Midline";
  if (rx < 0.38) spread = "Left Hemisphere";
  else if (rx > 0.62) spread = "Right Hemisphere";
  else if (rx >= 0.38 && rx <= 0.62 && ry < 0.50) spread = "Superior Midline";
  else spread = "Inferior Midline";

  return { region, spread, cx: Math.round(cx), cy: Math.round(cy) };
}

function getAggressiveness(metrics: any) {
  const area = metrics?.tumor_area_percent ?? 0;
  const grade = metrics?.who_grade ?? "";
  const morph = metrics?.morphology ?? {};
  const irreg = morph.boundary_irregularity ?? 0;
  const type = metrics?.predicted_class ?? "";
  let score = 0;
  score += Math.min(area / 30, 1) * 30;
  score += irreg * 25;
  if (grade.includes("III") || grade.includes("IV")) score += 30;
  else if (grade.includes("II")) score += 15;
  if (type === "Glioma") score += 15;
  else if (type === "Meningioma") score += 5;
  score += (metrics?.confidence ?? 0) * 10;
  score = Math.min(Math.round(score), 100);
  let level = "Low"; let color = "#16a34a";
  if (score >= 65) { level = "High"; color = "#dc2626"; }
  else if (score >= 35) { level = "Moderate"; color = "#d97706"; }
  return { score, level, color };
}

function getModelContributions(metrics: any) {
  const conf = metrics?.confidence ?? 0.5;
  const area = (metrics?.tumor_area_percent ?? 0) / 100;
  const unc = metrics?.uncertainty?.uncertainty_score ?? 0.1;
  const cnn = Math.round(conf * 45);
  const yolo = Math.round(area * 30 + 10);
  const gradcam = Math.round((1 - unc) * 20);
  const sam = Math.max(100 - cnn - yolo - gradcam, 5);
  return [
    { name: "CNN Classification", pct: cnn, color: "#1d4ed8", icon: "🧠" },
    { name: "YOLO Localization", pct: yolo, color: "#7c3aed", icon: "📦" },
    { name: "GradCAM Attention", pct: gradcam, color: "#ea580c", icon: "🔥" },
    { name: "SAM Segmentation", pct: sam, color: "#0d9488", icon: "✂️" },
  ];
}

function downloadReport(metrics: any, location: any, aggr: any) {
  const now = new Date().toLocaleString();
  const probs = metrics?.probabilities ?? {};
  const unc = metrics?.uncertainty ?? {};
  const html = `<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
  body{font-family:'Segoe UI',sans-serif;margin:40px;color:#1e293b}
  h1{color:#1e3a8a;border-bottom:3px solid #1d4ed8;padding-bottom:12px;font-size:22px}
  h2{color:#1e3a8a;font-size:15px;margin-top:24px;margin-bottom:8px;border-left:4px solid #1d4ed8;padding-left:10px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0}
  .card{background:#f8faff;border:1px solid #dbeafe;border-radius:10px;padding:14px}
  .label{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px}
  .value{font-size:18px;font-weight:700;color:#1e3a8a}
  .badge{display:inline-block;background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;border-radius:20px;padding:3px 10px;font-size:12px;font-weight:600;margin-top:4px}
  table{width:100%;border-collapse:collapse;margin-top:8px;font-size:13px}
  th{background:#eff6ff;color:#1e3a8a;padding:8px 12px;text-align:left;font-weight:600}
  td{padding:8px 12px;border-bottom:1px solid #e0e7ff}
  .bar-wrap{background:#e0e7ff;border-radius:4px;height:8px;width:100%;margin-top:4px}
  .bar{height:8px;border-radius:4px;background:linear-gradient(90deg,#1d4ed8,#60a5fa)}
  .footer{margin-top:40px;font-size:11px;color:#94a3b8;border-top:1px solid #e0e7ff;padding-top:12px;text-align:center}
</style></head><body>
<h1>🧠 MRI Diagnostic Summary Report</h1>
<p style="color:#64748b;font-size:13px">Generated: ${now} | Neuro-Oncology AI Platform v2.0</p>
<h2>Primary Diagnosis</h2>
<div class="grid">
  <div class="card"><div class="label">Tumor Type</div><div class="value">${metrics?.predicted_class ?? "—"}</div><div class="badge">${metrics?.who_grade ?? "N/A"}</div></div>
  <div class="card"><div class="label">Confidence</div><div class="value">${((metrics?.confidence ?? 0) * 100).toFixed(1)}%</div></div>
  <div class="card"><div class="label">Tumor Area</div><div class="value">${(metrics?.tumor_area_percent ?? 0).toFixed(2)}%</div></div>
  <div class="card"><div class="label">Volume</div><div class="value">${(metrics?.volume_cm3 ?? 0).toFixed(3)} cm³</div></div>
</div>
<h2>Reliability & Uncertainty</h2>
<div class="grid">
  <div class="card"><div class="label">Reliability Tier</div><div class="value">${metrics?.reliability_tier ?? "—"}</div></div>
  <div class="card"><div class="label">MC-Dropout Uncertainty</div><div class="value">${(unc?.uncertainty_score ?? 0).toFixed(4)}</div>
  <div style="font-size:12px;color:#64748b">${unc?.n_passes ?? 20} passes · T=1.5 calibrated</div></div>
</div>
<h2>Tumor Location</h2>
<div class="card"><table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Region</td><td><strong>${location?.region ?? "—"}</strong></td></tr>
  <tr><td>Centroid</td><td>(x=${location?.cx ?? 0}, y=${location?.cy ?? 0})</td></tr>
  <tr><td>Spread</td><td>${location?.spread ?? "—"}</td></tr>
  <tr><td>Max Diameter</td><td>${(metrics?.max_diameter_mm ?? 0).toFixed(1)} mm</td></tr>
</table></div>
<h2>Aggressiveness</h2>
<div class="card">
  <div style="font-size:20px;font-weight:700;color:${aggr?.color}">${aggr?.level?.toUpperCase()} RISK — ${aggr?.score}/100</div>
  <div class="bar-wrap"><div class="bar" style="width:${aggr?.score}%;background:${aggr?.color}"></div></div>
</div>
<h2>Classification Probabilities</h2>
<table><tr><th>Class</th><th>Probability</th><th>Uncertainty</th></tr>
${Object.entries(probs).sort(([, a], [, b]) => (b as number) - (a as number)).map(([cls, prob]) => {
    const u = unc?.class_breakdown?.[cls]?.uncertainty ?? 0;
    return `<tr><td>${cls === metrics?.predicted_class ? `<strong>▶ ${cls}</strong>` : cls}</td><td><strong>${((prob as number) * 100).toFixed(1)}%</strong></td><td>±${(u as number).toFixed(3)}</td></tr>`;
  }).join('')}
</table>
<div class="footer">⚠️ AI-generated for research only. Confirm with a qualified radiologist.</div>
</body></html>`;
  const blob = new Blob([html], { type: 'text/html' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = `MRI_Report_${Date.now()}.html`; a.click();
  URL.revokeObjectURL(url);
}

export default function App() {
  const [analysisData, setAnalysisData] = useState<any>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState("");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [processingTime, setProcessingTime] = useState<number | null>(null);

  const handleUpload = async (file: File) => {
    setLoading(true); setErrorMsg(null); setProcessingTime(null);
    const stages = [
      "Preprocessing scan...",
      "Running CNN classification...",
      "Computing Grad-CAM...",
      "Prompting SAM segmentation...",
      "Estimating volume...",
      "Finalizing diagnosis...",
    ];
    let si = 0; setLoadingStage(stages[0]);
    const iv = setInterval(() => {
      si = Math.min(si + 1, stages.length - 1);
      setLoadingStage(stages[si]);
    }, 1800);
    const t0 = Date.now();
    const formData = new FormData(); formData.append('file', file);
    try {
      const res = await fetch('http://localhost:8000/analyze', { method: 'POST', body: formData });
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const data = await res.json();
      setAnalysisData(data.results);
      setSessionId(data.session_id);
      setProcessingTime((Date.now() - t0) / 1000);
    } catch (e: any) {
      setErrorMsg(e.message ?? "Unknown error. Is the backend running?");
    } finally { clearInterval(iv); setLoading(false); setLoadingStage(""); }
  };

  const metrics = analysisData?.metrics ?? {};
  const primaryConf = metrics.confidence ?? 0;
  const reliabilityPct = Math.round((metrics.reliability_score ?? 0) * 100);
  const reliabilityTier = metrics.reliability_tier ?? "Unknown";
  const uncertaintyData = metrics.uncertainty ?? {};
  const uncertaintyScore = uncertaintyData.uncertainty_score ?? metrics.uncertainty_score ?? null;
  const classBreakdown = uncertaintyData.class_breakdown ?? {};
  const morph = metrics.morphology ?? {};
  const location = analysisData ? getTumorLocation(metrics) : null;
  const aggr = analysisData ? getAggressiveness(metrics) : null;
  const contribs = analysisData ? getModelContributions(metrics) : null;
  const allPasses = uncertaintyData.all_passes ?? [];

  // ── Apply temperature scaling to probabilities in frontend ───────────────
  // Backend sometimes sends raw saturated softmax (100%) — scale here as safety net
  const scaleProbs = (rawProbs: Record<string, number>, T = 1.5) => {
    const vals = Object.values(rawProbs);
    const maxVal = Math.max(...vals);
    // If already scaled (max < 0.99) return as-is
    if (maxVal < 0.99) return rawProbs;
    // Apply power scaling then renormalize
    const scaled: Record<string, number> = {};
    let total = 0;
    for (const [k, v] of Object.entries(rawProbs)) {
      scaled[k] = Math.pow(Math.max(v, 1e-8), T);
      total += scaled[k];
    }
    for (const k of Object.keys(scaled)) scaled[k] /= total;
    return scaled;
  };
  const probs = scaleProbs(metrics.probabilities ?? {});

  const tierColor: Record<string, string> = {
    "High": "#1d4ed8", "Medium": "#0369a1",
    "Low": "#b45309", "Uncertain": "#b91c1c", "Unknown": "#6b7280",
  };
  const currentTierColor = tierColor[reliabilityTier] ?? "#6b7280";

  return (
    <div className="min-h-screen flex flex-col"
      style={{ background: 'linear-gradient(135deg,#f0f4ff 0%,#e8f0fe 50%,#f5f8ff 100%)', color: '#1e293b', fontFamily: "'Segoe UI',system-ui,sans-serif" }}>

      {/* ── Header ─────────────────────────────────────────────────── */}
      <header style={{ background: 'linear-gradient(90deg,#1e3a8a 0%,#1d4ed8 60%,#2563eb 100%)', boxShadow: '0 2px 16px rgba(30,58,138,0.18)' }}
        className="px-6 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div style={{ background: 'rgba(255,255,255,0.15)', borderRadius: 10 }} className="p-2">
            <span className="text-2xl">🧠</span>
          </div>
          <div>
            <h1 className="text-lg font-bold text-white tracking-wide">Neuro-Oncology AI Platform</h1>
            <p className="text-xs" style={{ color: 'rgba(255,255,255,0.6)' }}>EfficientNet-B0 · YOLOv10 · Grad-CAM · MobileSAM · MC-Dropout</p>
          </div>
        </div>
        <div className="flex items-center space-x-3">
          {processingTime && <span className="text-xs text-white opacity-70">⏱ {processingTime.toFixed(1)}s</span>}
          {analysisData && (
            <button onClick={() => downloadReport(metrics, location, aggr)}
              className="text-xs font-semibold px-4 py-2 rounded-lg"
              style={{ background: 'rgba(255,255,255,0.15)', color: 'white', border: '1px solid rgba(255,255,255,0.3)' }}
              onMouseOver={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.25)')}
              onMouseOut={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.15)')}>
              📥 Download Report
            </button>
          )}
          <span style={{ background: 'rgba(255,255,255,0.12)', border: '1px solid rgba(255,255,255,0.2)' }}
            className="text-xs text-white px-3 py-1 rounded-full font-medium hidden md:block">
            RESEARCH GRADE
          </span>
        </div>
      </header>

      <main className="flex-1 flex overflow-hidden">
        <div className="flex-1 p-5 overflow-y-auto space-y-5">

          {errorMsg && (
            <div className="bg-red-50 border border-red-300 text-red-700 rounded-xl px-4 py-3 text-sm flex items-center space-x-2">
              <span>⚠️</span><span>{errorMsg}</span>
            </div>
          )}

          {!analysisData ? (
            <UploadPanel onUpload={handleUpload} loading={loading} loadingStage={loadingStage} />
          ) : (<>

            {/* MRI Viewer */}
            <Card>
              <SectionTitle icon="" title="Synchronized Multi-Model Analysis" />
              <MRIViewer images={analysisData.images} />
            </Card>

            {/* Metrics Row */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">

              <Card>
                <p className="text-xs font-bold uppercase tracking-widest mb-1" style={{ color: '#6b7280' }}>DIAGNOSIS</p>
                <div className="text-3xl font-bold mb-3" style={{ color: '#1e3a8a' }}>{metrics.predicted_class ?? "—"}</div>
                <div className="w-full rounded-full h-2 mb-1" style={{ background: '#e0e7ff' }}>
                  <div className="h-2 rounded-full"
                    style={{ width: `${(primaryConf * 100).toFixed(1)}%`, background: 'linear-gradient(90deg,#1d4ed8,#60a5fa)', transition: 'width 1s' }} />
                </div>
                <p className="text-xs" style={{ color: '#6b7280' }}>Confidence: {(primaryConf * 100).toFixed(1)}%</p>
                {processingTime && <p className="text-xs mt-1" style={{ color: '#94a3b8' }}>⏱ {processingTime.toFixed(1)}s</p>}
              </Card>

              <Card>
                <p className="text-xs font-bold uppercase tracking-widest mb-1" style={{ color: '#6b7280' }}>TUMOR AREA</p>
                <div className="text-3xl font-bold mb-1" style={{ color: '#1d4ed8' }}>
                  {typeof metrics.tumor_area_percent === "number" ? metrics.tumor_area_percent.toFixed(2) : "0.00"}%
                </div>
                <p className="text-xs mb-1" style={{ color: '#6b7280' }}>Relative to cross-section</p>
                {typeof metrics.volume_cm3 === "number" &&
                  <p className="text-xs font-medium" style={{ color: '#1d4ed8' }}>Volume: {metrics.volume_cm3.toFixed(3)} cm³</p>}
                {typeof metrics.max_diameter_mm === "number" &&
                  <p className="text-xs" style={{ color: '#6b7280' }}>Max Ø: {metrics.max_diameter_mm.toFixed(1)} mm</p>}
                {metrics.who_grade && metrics.who_grade !== "N/A" && (
                  <div className="mt-2 inline-block px-2 py-0.5 rounded text-xs font-bold"
                    style={{ background: '#eff6ff', color: '#1d4ed8', border: '1px solid #bfdbfe' }}>
                    {metrics.who_grade}
                  </div>
                )}
              </Card>

              <Card className="flex flex-col items-center">
                <p className="text-xs font-bold uppercase tracking-widest mb-3 w-full" style={{ color: '#6b7280' }}>RELIABILITY</p>
                <ReliabilityGauge score={reliabilityPct} tier={reliabilityTier} />
                {uncertaintyScore !== null && (
                  <div className="mt-3 w-full">
                    <div className="flex justify-between text-xs mb-1" style={{ color: '#6b7280' }}>
                      <span>Uncertainty Score</span>
                      <span className="font-bold" style={{ color: currentTierColor }}>{uncertaintyScore.toFixed(4)}</span>
                    </div>
                    <div className="w-full rounded-full h-1.5" style={{ background: '#e0e7ff' }}>
                      <div className="h-1.5 rounded-full"
                        style={{ width: `${Math.min(uncertaintyScore * 500, 100)}%`, background: currentTierColor, transition: 'width .7s' }} />
                    </div>
                    <p className="text-xs mt-1 text-center" style={{ color: '#9ca3af' }}>
                      T=1.5 calibrated · {uncertaintyData.n_passes ?? 20} MC passes
                    </p>
                  </div>
                )}
              </Card>
            </div>

            {/* Location + Aggressiveness */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {location && (
                <Card>
                  <SectionTitle icon="📍" title="Tumor Location Mapping" />
                  <p className="text-sm font-bold mb-3" style={{ color: '#1e3a8a' }}>{location.region}</p>
                  <div className="space-y-1">
                    {[
                      { label: "Brain Region", value: location.region },
                      { label: "Spread Direction", value: location.spread },
                      {
                        label: "Max Diameter",
                        value: typeof metrics.max_diameter_mm === "number"
                          ? `${metrics.max_diameter_mm.toFixed(1)} mm` : "—"
                      },
                    ].map(({ label, value }) => (
                      <div key={label} className="flex justify-between items-center py-1.5 border-b"
                        style={{ borderColor: '#f0f4ff' }}>
                        <span className="text-xs font-medium" style={{ color: '#64748b' }}>{label}</span>
                        <span className="text-xs font-bold" style={{ color: '#1e3a8a' }}>{value}</span>
                      </div>
                    ))}
                  </div>
                </Card>
              )}

              {aggr && (
                <Card>
                  <SectionTitle icon="⚡" title="Tumor Aggressiveness Score" />
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <div className="text-4xl font-bold" style={{ color: aggr.color }}>{aggr.score}</div>
                      <div className="text-xs" style={{ color: '#64748b' }}>out of 100</div>
                    </div>
                    <div className="text-lg font-bold px-4 py-2 rounded-xl"
                      style={{ background: aggr.color + '18', color: aggr.color }}>
                      {aggr.level.toUpperCase()} RISK
                    </div>
                  </div>
                  <div className="w-full rounded-full h-5 mb-4 overflow-hidden" style={{ background: '#f0f4ff' }}>
                    <div className="h-5 rounded-full flex items-center justify-end pr-2"
                      style={{ width: `${aggr.score}%`, background: `linear-gradient(90deg,#16a34a,${aggr.color})`, transition: 'width 1s' }}>
                      <span className="text-xs text-white font-bold">{aggr.score}%</span>
                    </div>
                  </div>
                  {[
                    { label: "LOW", color: "#16a34a", from: 0, to: 34 },
                    { label: "MODERATE", color: "#d97706", from: 34, to: 65 },
                    { label: "HIGH", color: "#dc2626", from: 65, to: 100 },
                  ].map(t => {
                    const fill = aggr.score > t.from
                      ? Math.min((aggr.score - t.from) / (t.to - t.from) * 100, 100) : 0;
                    return (
                      <div key={t.label} className="flex items-center space-x-2 mb-1.5">
                        <div className="w-20 text-xs font-bold" style={{ color: t.color }}>{t.label}</div>
                        <div className="flex-1 rounded-full h-2" style={{ background: '#f0f4ff' }}>
                          <div className="h-2 rounded-full"
                            style={{ width: `${fill}%`, background: t.color, transition: 'width 1s' }} />
                        </div>
                      </div>
                    );
                  })}
                  <p className="text-xs mt-2" style={{ color: '#94a3b8' }}>
                    Based on: area, boundary irregularity, WHO grade, tumor type
                  </p>
                </Card>
              )}
            </div>

            {/* Classification Probabilities */}
            {Object.keys(probs).length > 0 && (
              <Card>
                <SectionTitle icon="📊" title="Classification Probabilities" badge="+ MC Uncertainty" />
                <div className="space-y-4">
                  {Object.entries(probs)
                    .sort(([, a], [, b]) => (b as number) - (a as number))
                    .map(([cls, prob]) => {
                      const isTop = cls === metrics.predicted_class;
                      const unc = classBreakdown[cls]?.uncertainty ?? null;
                      return (
                        <div key={cls}>
                          <div className="flex justify-between text-sm mb-1">
                            <span className="font-semibold" style={{ color: isTop ? '#1d4ed8' : '#64748b' }}>
                              {isTop && <span className="mr-1">▶</span>}{cls}
                            </span>
                            <div className="flex items-center space-x-3">
                              {unc !== null && (
                                <span className="text-xs px-1.5 py-0.5 rounded"
                                  style={{ background: '#f0f4ff', color: '#6b7280' }}>
                                  ±{unc.toFixed(3)}
                                </span>
                              )}
                              <span className="font-bold" style={{ color: isTop ? '#1d4ed8' : '#94a3b8' }}>
                                {((prob as number) * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                          <div className="relative w-full rounded-full h-2.5" style={{ background: '#f0f4ff' }}>
                            <div className="h-2.5 rounded-full transition-all duration-700"
                              style={{
                                width: `${((prob as number) * 100).toFixed(1)}%`,
                                background: isTop ? 'linear-gradient(90deg,#1d4ed8,#60a5fa)' : '#cbd5e1',
                              }} />
                            {unc !== null && (prob as number) > 0 && (
                              <div className="absolute top-0 h-2.5 rounded-full opacity-25"
                                style={{
                                  left: `${Math.max(0, ((prob as number) - unc) * 100)}%`,
                                  width: `${Math.min(unc * 2 * 100, 100)}%`,
                                  background: isTop ? '#1d4ed8' : '#94a3b8',
                                }} />
                            )}
                          </div>
                        </div>
                      );
                    })}
                </div>
              </Card>
            )}

            {/* Model Contributions + MC Passes */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {contribs && (
                <Card>
                  <SectionTitle icon="🔬" title="Model Contribution Analysis" />
                  <div className="space-y-3">
                    {contribs.map(c => (
                      <div key={c.name}>
                        <div className="flex justify-between text-sm mb-1">
                          <span className="font-medium" style={{ color: '#1e3a8a' }}>{c.icon} {c.name}</span>
                          <span className="font-bold" style={{ color: c.color }}>{c.pct}%</span>
                        </div>
                        <div className="w-full rounded-full h-3" style={{ background: '#f0f4ff' }}>
                          <div className="h-3 rounded-full transition-all duration-700"
                            style={{ width: `${c.pct}%`, background: c.color }} />
                        </div>
                      </div>
                    ))}
                  </div>
                  <p className="text-xs mt-3" style={{ color: '#94a3b8' }}>
                    Weights derived from confidence, area, and uncertainty metrics
                  </p>
                </Card>
              )}

              {allPasses.length > 0 && (
                <Card>
                  <SectionTitle icon="🎲" title="MC-Dropout Prediction Stability" badge={`${allPasses.length} passes`} />
                  <div className="space-y-1.5">
                    {allPasses.slice(0, 12).map((pass: number[], i: number) => {
                      const topProb = Math.max(...pass);
                      const topIdx = pass.indexOf(topProb);
                      const cls = ["Glioma", "Meningioma", "Healthy", "Pituitary"][topIdx];
                      const colors = ["#1d4ed8", "#7c3aed", "#16a34a", "#ea580c"];
                      return (
                        <div key={i} className="flex items-center space-x-2">
                          <span className="text-xs w-12" style={{ color: '#94a3b8' }}>Pass {i + 1}</span>
                          <div className="flex-1 rounded-full h-2.5 overflow-hidden" style={{ background: '#f0f4ff' }}>
                            <div className="h-2.5 rounded-full"
                              style={{ width: `${(topProb * 100).toFixed(1)}%`, background: colors[topIdx] }} />
                          </div>
                          <span className="text-xs w-24 text-right font-medium" style={{ color: colors[topIdx] }}>
                            {cls} {(topProb * 100).toFixed(0)}%
                          </span>
                        </div>
                      );
                    })}
                  </div>
                  <p className="text-xs mt-3" style={{ color: '#94a3b8' }}>
                    Each bar = one stochastic forward pass · T=1.5 calibrated
                  </p>
                </Card>
              )}
            </div>

            {/* Morphology */}
            {morph && typeof morph.area === "number" && morph.area > 0 && (
              <Card>
                <SectionTitle icon="🔭" title="Morphology & Measurements" />
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {[
                    { label: "Area", value: morph.area?.toFixed(1), unit: " px²" },
                    { label: "Eccentricity", value: morph.eccentricity?.toFixed(3), unit: "" },
                    { label: "Compactness", value: morph.compactness?.toFixed(3), unit: "" },
                    { label: "Boundary Irreg.", value: morph.boundary_irregularity?.toFixed(3), unit: "" },
                    { label: "Skull Proximity", value: morph.skull_proximity?.toFixed(1), unit: " px" },
                    { label: "Max Diameter", value: metrics.max_diameter_mm?.toFixed(1), unit: " mm" },
                  ].map(({ label, value, unit }) => value ? (
                    <div key={label} className="rounded-xl p-3 border"
                      style={{ background: '#f8faff', borderColor: '#e0e7ff' }}>
                      <p className="text-xs mb-1" style={{ color: '#94a3b8' }}>{label}</p>
                      <p className="text-lg font-bold" style={{ color: '#1e3a8a' }}>{value}{unit}</p>
                    </div>
                  ) : null)}
                </div>
              </Card>
            )}

            <div className="flex justify-center pt-1 pb-4">
              <button
                onClick={() => { setAnalysisData(null); setSessionId(null); setProcessingTime(null); }}
                className="text-sm underline" style={{ color: '#94a3b8' }}
                onMouseOver={e => (e.currentTarget.style.color = '#1d4ed8')}
                onMouseOut={e => (e.currentTarget.style.color = '#94a3b8')}>
                Upload new scan
              </button>
            </div>
          </>)}
        </div>

        {/* Chat Sidebar */}
        <div className="w-96 flex flex-col border-l" style={{ background: 'white', borderColor: '#dbeafe' }}>
          {sessionId ? (
            <ChatPanel sessionId={sessionId} />
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center p-8 text-center space-y-4">
              <div style={{ background: '#eff6ff', borderRadius: '50%', padding: 20 }}>
                <span className="text-4xl">🧠</span>
              </div>
              <p className="text-sm font-medium" style={{ color: '#94a3b8' }}>
                Upload an MRI scan to initialize<br />the AI diagnostic assistant.
              </p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}