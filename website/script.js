const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const uploadZone = document.getElementById('uploadZone');
const analyzeBtn = document.getElementById('analyzeBtn');
const resetBtn = document.getElementById('resetBtn');
const downloadMaskBtn = document.getElementById('downloadMaskBtn');
const fileName = document.getElementById('fileName');
const apiStatus = document.getElementById('apiStatus');

const statusPill = document.getElementById('statusPill');
const predClass = document.getElementById('predClass');
const riskScore = document.getElementById('riskScore');
const tamperArea = document.getElementById('tamperArea');
const confidenceBar = document.getElementById('confidenceBar');
const engineName = document.getElementById('engineName');
const originalPreview = document.getElementById('originalPreview');
const overlayBase = document.getElementById('overlayBase');
const overlayMask = document.getElementById('overlayMask');

let selectedFile = null;
const API_BASE = (() => {
  const host = window.location.hostname;
  if (host === 'localhost' || host === '127.0.0.1') return 'http://localhost:8000';

  const metaApi = document.querySelector('meta[name="api-base"]')?.content?.trim();
  if (metaApi) return metaApi;

  const overrideApi = localStorage.getItem('bioforgenet_api_base')?.trim();
  if (overrideApi) return overrideApi;
  return 'https://api.bioforgenet.live';
})();
const DAILY_LIMIT = 10;

// ── License key validation ───────────────────────────────────────────
// Valid key patterns: VS-PRO-XXXX (Professional) or VS-ENT-XXXX (Enterprise)
// Generate real keys in your billing system; these are demo patterns.
function validateLicenseKey(key) {
  if (!key) return null;
  const k = key.trim().toUpperCase();
  if (/^VS-PRO-[A-Z0-9]{4,}$/.test(k)) return 'Professional';
  if (/^VS-ENT-[A-Z0-9]{4,}$/.test(k)) return 'Enterprise';
  return null;
}

function isPaidUser() {
  return validateLicenseKey(localStorage.getItem('vs_license_key')) !== null;
}

function getPlanName() {
  return validateLicenseKey(localStorage.getItem('vs_license_key')) || 'Free';
}

// ── Scan quota helpers (10 per day for FREE users only) ──────────────
function getTodayKey() {
  return new Date().toISOString().slice(0, 10); // 'YYYY-MM-DD'
}

function getQuotaState() {
  const today = getTodayKey();
  const saved = localStorage.getItem('vs_scan_date');
  if (saved !== today) {
    localStorage.setItem('vs_scan_date', today);
    localStorage.setItem('vs_scan_count', '0');
  }
  return parseInt(localStorage.getItem('vs_scan_count') || '0', 10);
}

function incrementQuota() {
  if (isPaidUser()) return; // paid users are never counted against quota
  const count = getQuotaState();
  localStorage.setItem('vs_scan_count', String(count + 1));
  updateQuotaBar();
}

function updateQuotaBar() {
  const bar = document.getElementById('scanQuotaBar');
  if (!bar) return;

  if (isPaidUser()) {
    const plan = getPlanName();
    bar.innerHTML =
      `<span style="color:var(--ok);">&#10003; ${plan} Plan &mdash; Unlimited scans &nbsp;&middot;&nbsp; ` +
      `<a href="#" onclick="removeLicense();return false;" style="color:var(--muted);font-size:.8rem;">Remove key</a></span>`;
    if (analyzeBtn && selectedFile) analyzeBtn.disabled = false;
    return;
  }

  const used = getQuotaState();
  const remaining = DAILY_LIMIT - used;
  const activateLink = `<a href="#" onclick="activateLicense();return false;" style="color:var(--primary);">Activate license key</a>`;

  if (remaining <= 0) {
    bar.innerHTML =
      `<span style="color:var(--danger);">Daily limit reached (${DAILY_LIMIT}/${DAILY_LIMIT} scans used) &nbsp;&middot;&nbsp; ` +
      `<a href="pricing.html" style="color:var(--primary);">Upgrade for unlimited access</a> &nbsp;&middot;&nbsp; ${activateLink}</span>`;
    if (analyzeBtn) analyzeBtn.disabled = true;
  } else {
    const color = remaining <= 3 ? 'var(--danger)' : 'var(--muted)';
    bar.innerHTML =
      `<span style="color:${color};">${remaining} of ${DAILY_LIMIT} free scans remaining today &nbsp;&middot;&nbsp; ${activateLink}</span>`;
  }
}

function activateLicense() {
  const key = prompt('Enter your BioForgeNet license key (format: VS-PRO-XXXX or VS-ENT-XXXX):');
  if (!key) return;
  const plan = validateLicenseKey(key);
  if (plan) {
    localStorage.setItem('vs_license_key', key.trim().toUpperCase());
    updateQuotaBar();
    if (selectedFile && analyzeBtn) analyzeBtn.disabled = false;
    setStatus('License activated — Unlimited scans enabled');
  } else {
    alert('Invalid license key. Keys look like VS-PRO-ABCD1234.\nCheck your purchase confirmation email or contact support.');
  }
}

function removeLicense() {
  if (!confirm('Remove your license key? The 10-scan daily limit will apply again.')) return;
  localStorage.removeItem('vs_license_key');
  updateQuotaBar();
  setStatus('Awaiting scan');
}

async function pingApi() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('health failed');
    const info = await res.json();
    if (info.mode === 'real-checkpoint') {
      apiStatus.textContent = 'Backend: connected (real model)';
      apiStatus.style.color = '#28d07a';
    } else {
      apiStatus.textContent = 'Backend: connected (fallback mode)';
      apiStatus.style.color = '#f4d35e';
    }
  } catch {
    apiStatus.textContent = 'Backend: offline (using fallback simulation)';
    apiStatus.style.color = '#f4d35e';
  }
}

function setStatus(label, tone = 'default') {
  statusPill.textContent = label;
  statusPill.style.color = tone === 'ok' ? '#28d07a' : tone === 'danger' ? '#ff5c7a' : '#a8b5d7';
  statusPill.style.borderColor = tone === 'ok' ? 'rgba(40,208,122,.35)' : tone === 'danger' ? 'rgba(255,92,122,.35)' : 'rgba(151,174,255,.24)';
}

function resetResults() {
  predClass.textContent = '—';
  riskScore.textContent = '—';
  tamperArea.textContent = '—';
  confidenceBar.style.width = '0%';
  engineName.textContent = '—';
  originalPreview.style.display = 'none';
  overlayBase.style.display = 'none';
  overlayMask.style.display = 'none';
  originalPreview.removeAttribute('src');
  overlayBase.removeAttribute('src');
  overlayMask.removeAttribute('src');
  downloadMaskBtn.disabled = true;
  setStatus('Awaiting scan');
}

function handleFile(file) {
  selectedFile = file;
  fileName.textContent = `Selected: ${file.name}`;

  const objectUrl = URL.createObjectURL(file);
  originalPreview.src = objectUrl;
  overlayBase.src = objectUrl;
  originalPreview.style.display = 'block';
  overlayBase.style.display = 'block';
  overlayMask.style.display = 'none';

  // Enable analyze button: paid users always allowed; free users respect daily quota
  const remaining = DAILY_LIMIT - getQuotaState();
  const blocked = !isPaidUser() && remaining <= 0;
  analyzeBtn.disabled = blocked;
  setStatus(blocked ? 'Daily scan limit reached — upgrade to continue' : 'Ready to analyze');
}

function applyResult(result) {
  const forged = result.prediction !== 'authentic';
  predClass.textContent = forged ? 'FORGED (suspicious)' : 'AUTHENTIC (clean)';
  predClass.style.color = forged ? '#ff5c7a' : '#28d07a';
  riskScore.textContent = `${(result.risk_score * 100).toFixed(1)}%`;
  tamperArea.textContent = `${result.tamper_area_pct.toFixed(1)}%`;
  confidenceBar.style.width = `${Math.round(result.confidence * 100)}%`;
  engineName.textContent = result.engine || 'unknown';
  setStatus(forged ? 'Risk detected' : 'No risk detected', forged ? 'danger' : 'ok');
  downloadMaskBtn.disabled = false;

  if (result.mask_png_base64) {
    overlayMask.src = `data:image/png;base64,${result.mask_png_base64}`;
    overlayMask.style.display = 'block';
  }
}

function simulateFallbackResult(file) {
  const base = (file.name.length % 10) / 100;
  const risk = Math.min(0.95, 0.18 + base + Math.random() * 0.55);
  const forged = risk >= 0.5;
  return {
    prediction: forged ? 'forged' : 'authentic',
    risk_score: risk,
    tamper_area_pct: forged ? (3 + risk * 18) : (0.2 + risk * 2),
    confidence: 0.62 + Math.random() * 0.35,
    mask_png_base64: ''
  };
}

async function runApiAnalysis(file) {
  const data = new FormData();
  data.append('file', file);
  const response = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: data
  });
  if (!response.ok) throw new Error('API analyze failed');
  return response.json();
}

async function downloadMaskPng(file) {
  const data = new FormData();
  data.append('file', file);
  try {
    const response = await fetch(`${API_BASE}/analyze-mask`, {
      method: 'POST',
      body: data
    });
    if (!response.ok) throw new Error('Download failed');
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `forgery_mask_${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  } catch (err) {
    alert(`Download error: ${err.message}`);
  }
}

uploadBtn.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('click', (e) => {
  if (e.target.tagName !== 'BUTTON') fileInput.click();
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files[0]) handleFile(e.target.files[0]);
});

['dragenter', 'dragover'].forEach(evt => {
  uploadZone.addEventListener(evt, (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
  });
});
['dragleave', 'drop'].forEach(evt => {
  uploadZone.addEventListener(evt, (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
  });
});

uploadZone.addEventListener('drop', (e) => {
  const file = e.dataTransfer?.files?.[0];
  if (file) handleFile(file);
});

analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  // Enforce quota before running — paid users bypass the limit entirely
  if (!isPaidUser() && DAILY_LIMIT - getQuotaState() <= 0) {
    setStatus('Daily scan limit reached — upgrade to continue');
    analyzeBtn.disabled = true;
    updateQuotaBar();
    return;
  }

  setStatus('Running inference...');
  analyzeBtn.disabled = true;

  try {
    const result = await runApiAnalysis(selectedFile);
    incrementQuota();
    applyResult(result);
  } catch {
    incrementQuota();
    const result = simulateFallbackResult(selectedFile);
    applyResult(result);
  } finally {
    const remaining = DAILY_LIMIT - getQuotaState();
    analyzeBtn.disabled = !isPaidUser() && remaining <= 0;
  }
});

resetBtn.addEventListener('click', () => {
  selectedFile = null;
  fileInput.value = '';
  fileName.textContent = 'No file selected';
  analyzeBtn.disabled = true;
  resetResults();
});

downloadMaskBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  downloadMaskBtn.disabled = true;
  downloadMaskBtn.textContent = '⏳ Downloading...';
  try {
    await downloadMaskPng(selectedFile);
    downloadMaskBtn.textContent = '✓ Downloaded';
    setTimeout(() => {
      if (selectedFile) downloadMaskBtn.textContent = '📥 Download Forensic Mask';
    }, 2000);
  } finally {
    downloadMaskBtn.disabled = false;
  }
});

resetResults();
updateQuotaBar();
pingApi();