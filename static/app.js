// ============================================================
// Resume Shortlister — Frontend Logic
// ============================================================

// --- Slider helpers ---
function updateSlider(sliderId, valueId) {
  const slider = document.getElementById(sliderId);
  const display = document.getElementById(valueId);
  display.textContent = parseFloat(slider.value).toFixed(2);
}

// --- Query row management ---
function addQueryRow(listId, tier, value = '') {
  const list = document.getElementById(listId);
  const row = document.createElement('div');
  row.className = 'query-chip';
  row.innerHTML = `
    <input type="text" class="query-chip-input" placeholder="Requirement..." value="${escapeHtml(value)}" data-tier="${tier}">
    <button class="query-chip-remove" onclick="this.parentElement.remove()" title="Remove">
      <i data-lucide="x" class="icon-sm"></i>
    </button>
  `;
  list.appendChild(row);
  if (window.lucide) lucide.createIcons({ root: row });
  
  // Focus the new input
  const input = row.querySelector('input');
  if (!value) input.focus();
  return input;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function getQueriesFromList(listId) {
  const inputs = document.querySelectorAll(`#${listId} .query-chip-input`);
  return Array.from(inputs).map(i => i.value.trim()).filter(v => v.length > 0);
}

function clearAllTiers() {
  ['mustHaveList', 'goodToHaveList', 'bonusList', 'disqualifiersList'].forEach(id => {
    document.getElementById(id).innerHTML = '';
  });
  // Add one empty row per tier
  addQueryRow('mustHaveList', 'must');
  addQueryRow('goodToHaveList', 'good');
  addQueryRow('bonusList', 'bonus');
  addQueryRow('disqualifiersList', 'disq');
}

// --- Load default JD ---
async function loadDefaultJD() {
  try {
    const resp = await fetch('/api/default-jd');
    const jd = await resp.json();
    
    // Clear existing
    ['mustHaveList', 'goodToHaveList', 'bonusList', 'disqualifiersList'].forEach(id => {
      document.getElementById(id).innerHTML = '';
    });

    // Populate tiers
    (jd.must_have || []).forEach(q => addQueryRow('mustHaveList', 'must', q));
    (jd.good_to_have || []).forEach(q => addQueryRow('goodToHaveList', 'good', q));
    (jd.bonus || []).forEach(q => addQueryRow('bonusList', 'bonus', q));
    (jd.disqualifiers || []).forEach(q => addQueryRow('disqualifiersList', 'disq', q));

    // Set tier weight sliders
    if (jd.tier_weights) {
      setSlider('mustWeight', 'mustWeightVal', jd.tier_weights.must_have || 1.0);
      setSlider('goodWeight', 'goodWeightVal', jd.tier_weights.good_to_have || 0.5);
      setSlider('bonusWeight', 'bonusWeightVal', jd.tier_weights.bonus || 0.25);
    }
    if (jd.disqualifier_penalty !== undefined) {
      setSlider('disqPenalty', 'disqPenaltyVal', jd.disqualifier_penalty);
    }

    // Set penalty sliders
    if (jd.penalty_config) {
      const pc = jd.penalty_config;
      setSlider('ghostPen', 'ghostPenVal', pc.ghost_penalty || 0.20);
      setSlider('mismatchPen', 'mismatchPenVal', pc.mismatch_penalty || 0.50);
      setSlider('hopperPen', 'hopperPenVal', pc.hopper_penalty || 0.60);
      setSlider('codingPen', 'codingPenVal', pc.coding_penalty || 0.70);
      setSlider('consultPen', 'consultPenVal', pc.consulting_penalty || 0.65);
      setSlider('lowProfPen', 'lowProfPenVal', pc.low_profile_penalty || 0.80);
      setSlider('cvPen', 'cvPenVal', pc.cv_speech_penalty || 0.55);
      setSlider('resPen', 'resPenVal', pc.research_penalty || 0.40);
      setSlider('lcPen', 'lcPenVal', pc.langchain_penalty || 0.45);
    }

    showToast('Default JD loaded — Redrob Senior AI Engineer');
  } catch (err) {
    showToast('Failed to load default JD: ' + err.message, true);
  }
}

function setSlider(sliderId, valueId, val) {
  const slider = document.getElementById(sliderId);
  slider.value = val;
  document.getElementById(valueId).textContent = parseFloat(val).toFixed(2);
}

// --- Top-K selector ---
function setTopK(val) {
  document.getElementById('topKValue').value = val;
  document.querySelectorAll('.topk-btn').forEach(btn => {
    btn.classList.toggle('active', parseInt(btn.dataset.val) === val);
  });
  document.getElementById('topkCustom').value = '';
}

function setTopKCustom(val) {
  const n = parseInt(val);
  if (!isNaN(n) && n > 0) {
    document.getElementById('topKValue').value = Math.min(n, 500);
    document.querySelectorAll('.topk-btn').forEach(btn => btn.classList.remove('active'));
  }
}

// --- Run pipeline ---
async function runPipeline() {
  const btn = document.getElementById('btnRun');
  
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<i data-lucide="loader-2" class="icon-md spin"></i> Searching...';
    if (window.lucide) lucide.createIcons({ root: btn });
  }

  const mustHave = getQueriesFromList('mustHaveList');
  const goodToHave = getQueriesFromList('goodToHaveList');
  const bonus = getQueriesFromList('bonusList');
  const disqualifiers = getQueriesFromList('disqualifiersList');

  // Helper to safely get value from DOM
  const getVal = (id, def) => {
    const el = document.getElementById(id);
    return el ? parseFloat(el.value) : def;
  };

  if (mustHave.length === 0) {
    showToast('Add at least one "Must Have" requirement', true);
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '<i data-lucide="search" class="icon-md"></i> Search Candidates';
      if (window.lucide) lucide.createIcons({ root: btn });
    }
    return;
  }

  // Show progress in engine banner
  const engineStatus = document.getElementById('engineStatus');
  const engineDot = document.getElementById('engineDot');
  if (engineStatus) engineStatus.textContent = 'Running hybrid search...';
  if (engineDot) engineDot.className = 'status-dot loading';

  const payload = {
    must_have: mustHave,
    good_to_have: goodToHave,
    bonus: bonus,
    disqualifiers: disqualifiers,
    top_k: parseInt(document.getElementById('topKValue').value) || 50,
    tier_weights: {
      must_have: getVal('mustWeight', 1.0),
      good_to_have: getVal('goodWeight', 0.5),
      bonus: getVal('bonusWeight', 0.25),
    },
    disqualifier_penalty: getVal('disqPenalty', -1.0),
    penalty_config: {
      ghost_penalty: getVal('ghostPen', 0.2),
      mismatch_penalty: getVal('mismatchPen', 0.5),
      hopper_penalty: getVal('hopperPen', 0.6),
      coding_penalty: getVal('codingPen', 0.7),
      consulting_penalty: getVal('consultPen', 0.65),
      low_profile_penalty: getVal('lowProfPen', 0.8),
      cv_speech_penalty: getVal('cvPen', 0.55),
      research_penalty: getVal('resPen', 0.4),
      langchain_penalty: getVal('lcPen', 0.45),
    },
    dataset_name: document.getElementById('datasetSelect').value,
  };

  try {
    const resp = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await resp.json();
    
    if (!resp.ok) {
      throw new Error(data.error || 'Pipeline failed');
    }

    renderResults(data);
    
    // Status text update safely
    const statusText = document.getElementById('engineStatus');
    if (statusText) {
      const elapsed = data.stats.elapsed_seconds ? ` in ${data.stats.elapsed_seconds}s` : '';
      statusText.textContent = `Completed${elapsed} — ${data.results.length} ranked`;
    }
    
    // Update badge in results header
    const badge = document.getElementById('resultsCountBadge');
    if (badge) badge.textContent = `Showing ${data.results.length}`;

    showToast(`✅ Done! Top candidate: ${data.results[0]?.candidate_id} (${data.results[0]?.final_score.toFixed(2)})`, false, true);
    
    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth', block: 'start' });

  } catch (err) {
    const statusText = document.getElementById('engineStatus');
    if (statusText) statusText.textContent = 'Error: ' + err.message;
    showToast('Pipeline error: ' + err.message, true);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '<i data-lucide="search" class="icon-md"></i> Search Candidates';
      if (window.lucide) lucide.createIcons({ root: btn });
    }
    const engineDot = document.getElementById('engineDot');
    if (engineDot) engineDot.className = 'status-dot healthy';
  }
}

// --- Render results ---
function renderResults(data) {
  const section = document.getElementById('resultsSection');
  section.classList.remove('hidden');

  // Stats bar
  const stats = data.stats;
  document.getElementById('statsBar').innerHTML = `
    <div class="stat-pill">
      <div>
        <div class="stat-value">${stats.total_candidates.toLocaleString()}</div>
        <div class="stat-label">Total Indexed</div>
      </div>
    </div>
    <div class="stat-pill">
      <div>
        <div class="stat-value">${stats.scored_candidates}</div>
        <div class="stat-label">Scored</div>
      </div>
    </div>
    <div class="stat-pill">
      <div>
        <div class="stat-value">${stats.score_max}</div>
        <div class="stat-label">Top Score</div>
      </div>
    </div>
    <div class="stat-pill">
      <div>
        <div class="stat-value">${stats.score_mean}</div>
        <div class="stat-label">Mean Score</div>
      </div>
    </div>
    <div class="stat-pill">
      <div>
        <div class="stat-value">${stats.penalized_count}</div>
        <div class="stat-label">Penalized</div>
      </div>
    </div>
    <div class="stat-pill">
      <div>
        <div class="stat-value">${stats.bonused_count}</div>
        <div class="stat-label">Bonused</div>
      </div>
    </div>
    ${stats.elapsed_seconds ? `<div class="stat-pill">
      <div>
        <div class="stat-value">${stats.elapsed_seconds}s</div>
        <div class="stat-label">Elapsed</div>
      </div>
    </div>` : ''}
  `;

  // Table
  const tbody = document.getElementById('resultsBody');
  tbody.innerHTML = '';

  data.results.forEach(r => {
    const rankClass = r.rank === 1 ? 'gold' : r.rank === 2 ? 'silver' : r.rank === 3 ? 'bronze' : '';
    const scorePercent = Math.round(r.final_score * 100);
    const semPercent = Math.round(r.semantic_score * 100);

    const badgeBaseStyle = "padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; white-space: nowrap;";
    const penaltyStyle = badgeBaseStyle + " background: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5;";
    const bonusStyle = badgeBaseStyle + " background: #DCFCE7; color: #166534; border: 1px solid #86EFAC;";
    const noneStyle = badgeBaseStyle + " background: #F3F4F6; color: #6B7280; border: 1px solid #E5E7EB;";

    // Parse penalty/bonus flags
    const penaltyBadges = r.penalties === 'None' 
      ? `<span style="${noneStyle}">No Penalties</span>`
      : r.penalties.split(/\s+/).filter(Boolean).map(p => 
          `<span style="${penaltyStyle}">${p}</span>`
        ).join('');
    
    const bonusBadges = r.bonuses === 'None'
      ? ''
      : r.bonuses.split(/\s+/).filter(Boolean).map(b => 
          `<span style="${bonusStyle}">${b}</span>`
        ).join('');

    const row = document.createElement('tr');
    row.innerHTML = `
      <td class="rank-cell ${rankClass}">#${r.rank}</td>
      <td class="candidate-id">
        <button class="candidate-link" title="Copy to clipboard" onclick="copyCandidateId('${r.candidate_id}')">
          <i data-lucide="copy" class="icon-sm" style="margin-right:4px;"></i>${r.candidate_id}
        </button>
      </td>
      <td>
        <div class="score-bar-container">
          <div class="score-bar"><div class="score-bar-fill" style="width:${scorePercent}%"></div></div>
          <span class="score-value" style="font-weight:600;">${r.final_score.toFixed(4)}</span>
        </div>
      </td>
      <td>
        <div style="display: flex; flex-direction: column; gap: 6px;">
          <div class="reasoning-cell" style="line-height: 1.4;">${buildReasoning(r)}</div>
          <div style="display: flex; gap: 6px; flex-wrap: wrap;">
            ${penaltyBadges} ${bonusBadges}
          </div>
        </div>
      </td>
    `;
    tbody.appendChild(row);
    if (window.lucide) lucide.createIcons({ root: row });
  });
}

function buildReasoning(r) {
  const sem = (r.semantic_score * 100).toFixed(0);
  const penStr = r.penalty_multiplier < 1.0 ? `<span style="color:#991B1B">penalty ×${r.penalty_multiplier.toFixed(2)}</span>` : 'no penalty';
  const bonusStr = r.bonus_total > 0 ? `<span style="color:#166534">bonus +${r.bonus_total.toFixed(2)}</span>` : 'no bonus';
  return `<span class="reasoning-text" style="color:var(--text-muted); font-size:12px;">Semantic <b>${sem}%</b> &bull; ${penStr} &bull; ${bonusStr}</span>`;
}

// --- CSV Export ---
function exportCSV() {
  const link = document.createElement('a');
  link.href = '/api/export-csv';
  link.download = 'shortlisted_candidates.csv';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  showToast('✅ CSV download started!');
}

// --- Click to Copy ---
function copyCandidateId(id) {
  navigator.clipboard.writeText(id).then(() => {
    showToast('✅ Copied ' + id + ' to clipboard!');
  }).catch(err => {
    showToast('❌ Failed to copy to clipboard', true);
  });
}


function signalPill(label, value) {
  return `<div class="signal-pill"><div class="signal-val">${value}</div><div class="signal-lbl">${label}</div></div>`;
}

// --- Toast notifications ---
function showToast(msg, isError = false, isSuccess = false) {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  let cls = 'toast';
  if (isError) cls += ' error';
  else if (isSuccess) cls += ' success';
  toast.className = cls;
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 4500);
}

// --- Datasets ---
async function loadDatasets() {
  try {
    const resp = await fetch('/api/datasets');
    const datasets = await resp.json();
    const select = document.getElementById('datasetSelect');
    select.innerHTML = '';
    
    datasets.forEach(ds => {
      const opt = document.createElement('option');
      opt.value = ds.name;
      opt.textContent = `${ds.name} (${ds.candidate_count.toLocaleString()} candidates)${ds.status !== 'ready' ? ' ['+ds.status+']' : ''}`;
      if (ds.is_default) {
        opt.selected = true;
      }
      select.appendChild(opt);
    });
  } catch (err) {
    showToast('Failed to load datasets: ' + err.message, true);
  }
}

async function uploadDataset() {
  const name = document.getElementById('dsName').value.trim();
  const fileInput = document.getElementById('dsFile');
  const btn = document.getElementById('btnUpload');
  const status = document.getElementById('uploadStatus');

  if (!name || !fileInput.files.length) {
    status.textContent = 'Please provide a name and select a file.';
    status.style.color = 'var(--error)';
    return;
  }

  const formData = new FormData();
  formData.append('name', name);
  formData.append('file', fileInput.files[0]);

  btn.disabled = true;
  btn.textContent = 'Uploading & Processing...';
  status.textContent = 'This may take several minutes if the dataset is large...';
  status.style.color = 'var(--text-muted)';

  try {
    const resp = await fetch('/api/datasets', {
      method: 'POST',
      body: formData
    });
    const data = await resp.json();
    
    if (!resp.ok) throw new Error(data.error || 'Upload failed');
    
    status.textContent = '✅ Success! Dataset embedded and indexed.';
    status.style.color = 'var(--success)';
    
    setTimeout(() => {
      document.getElementById('uploadModal').classList.add('hidden');
      loadDatasets(); // refresh list
    }, 1500);
    
  } catch (err) {
    status.textContent = '❌ Error: ' + err.message;
    status.style.color = 'var(--error)';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Upload & Process';
  }
}

// --- Init ---
document.addEventListener('DOMContentLoaded', async () => {
  // Add one empty row per tier
  addQueryRow('mustHaveList', 'must');
  addQueryRow('goodToHaveList', 'good');
  addQueryRow('bonusList', 'bonus');
  addQueryRow('disqualifiersList', 'disq');

  // Load available datasets
  await loadDatasets();

  // Check engine status
  try {
    const resp = await fetch('/api/status');
    const data = await resp.json();
    if (data.ready) {
      document.getElementById('statusDot').className = 'status-dot ready';
      document.getElementById('engineDot').className = 'status-dot ready';
      document.getElementById('statusText').textContent =
        `Engine ready — ${data.candidates_indexed.toLocaleString()} candidates indexed with pre-computed embeddings`;
      document.getElementById('engineStatus').textContent =
        `${data.candidates_indexed.toLocaleString()} candidates indexed`;
      document.getElementById('btnRun').disabled = false;
    }
  } catch (err) {
    document.getElementById('statusDot').className = 'status-dot error';
    document.getElementById('engineDot').className = 'status-dot error';
    document.getElementById('statusText').textContent = 'Failed to connect to backend: ' + err.message;
  }
});
