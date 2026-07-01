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
  row.className = 'query-row';
  row.innerHTML = `
    <input type="text" class="query-input" placeholder="e.g. Production experience with vector databases FAISS Elasticsearch"
           value="${escapeHtml(value)}" data-tier="${tier}">
    <button class="btn-icon" onclick="this.parentElement.remove()" title="Remove">×</button>
  `;
  list.appendChild(row);
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
  const inputs = document.querySelectorAll(`#${listId} .query-input`);
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
  const btnText = document.getElementById('btnRunText');
  
  // Collect data
  const mustHave = getQueriesFromList('mustHaveList');
  const goodToHave = getQueriesFromList('goodToHaveList');
  const bonus = getQueriesFromList('bonusList');
  const disqualifiers = getQueriesFromList('disqualifiersList');

  if (mustHave.length === 0) {
    showToast('Add at least one "Must Have" requirement', true);
    return;
  }

  // Disable button
  btn.disabled = true;
  btnText.innerHTML = '<span class="spinner"></span> Running pipeline...';

  // Show progress
  const statusBar = document.getElementById('statusBar');
  statusBar.classList.remove('hidden');
  document.getElementById('statusDot').className = 'status-dot loading';
  document.getElementById('statusText').textContent = 'Running hybrid search + cross-encoder reranking...';

  const payload = {
    must_have: mustHave,
    good_to_have: goodToHave,
    bonus: bonus,
    disqualifiers: disqualifiers,
    top_k: parseInt(document.getElementById('topKValue').value) || 50,
    tier_weights: {
      must_have: parseFloat(document.getElementById('mustWeight').value),
      good_to_have: parseFloat(document.getElementById('goodWeight').value),
      bonus: parseFloat(document.getElementById('bonusWeight').value),
    },
    disqualifier_penalty: parseFloat(document.getElementById('disqPenalty').value),
    penalty_config: {
      ghost_penalty: parseFloat(document.getElementById('ghostPen').value),
      mismatch_penalty: parseFloat(document.getElementById('mismatchPen').value),
      hopper_penalty: parseFloat(document.getElementById('hopperPen').value),
      coding_penalty: parseFloat(document.getElementById('codingPen').value),
      consulting_penalty: parseFloat(document.getElementById('consultPen').value),
      low_profile_penalty: parseFloat(document.getElementById('lowProfPen').value),
      cv_speech_penalty: parseFloat(document.getElementById('cvPen').value),
      research_penalty: parseFloat(document.getElementById('resPen').value),
      langchain_penalty: parseFloat(document.getElementById('lcPen').value),
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
    document.getElementById('statusDot').className = 'status-dot ready';
    const elapsed = data.stats.elapsed_seconds ? ` in ${data.stats.elapsed_seconds}s` : '';
    document.getElementById('statusText').textContent = 
      `Pipeline complete${elapsed} — ${data.results.length} candidates ranked from ${data.stats.total_candidates.toLocaleString()} total`;
    
    // Update badge in results header
    const badge = document.getElementById('resultsCountBadge');
    if (badge) badge.textContent = `Showing ${data.results.length}`;

    showToast(`✅ Done! Top candidate: ${data.results[0]?.candidate_id} (${data.results[0]?.final_score})`, false, true);
    
    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth', block: 'start' });

  } catch (err) {
    document.getElementById('statusDot').className = 'status-dot error';
    document.getElementById('statusText').textContent = 'Error: ' + err.message;
    showToast('Pipeline error: ' + err.message, true);
  } finally {
    btn.disabled = false;
    btnText.innerHTML = '🚀 Run Pipeline';
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

    // Parse penalty/bonus flags
    const penaltyBadges = r.penalties === 'None' 
      ? '<span class="penalty-badge none">—</span>'
      : r.penalties.split(/\s+/).filter(Boolean).map(p => 
          `<span class="penalty-badge penalty">${p}</span>`
        ).join('');
    
    const bonusBadges = r.bonuses === 'None'
      ? '<span class="penalty-badge none">—</span>'
      : r.bonuses.split(/\s+/).filter(Boolean).map(b => 
          `<span class="penalty-badge bonus">${b}</span>`
        ).join('');

    const row = document.createElement('tr');
    row.innerHTML = `
      <td class="rank-cell ${rankClass}">#${r.rank}</td>
      <td class="candidate-id">
        <button class="candidate-link" onclick="openCandidateProfile('${r.candidate_id}')">${r.candidate_id}</button>
      </td>
      <td>
        <div class="score-bar-container">
          <div class="score-bar"><div class="score-bar-fill" style="width:${scorePercent}%"></div></div>
          <span class="score-value">${r.final_score.toFixed(4)}</span>
        </div>
      </td>
      <td>
        <div class="score-bar-container">
          <div class="score-bar"><div class="score-bar-fill" style="width:${semPercent}%"></div></div>
          <span class="score-value">${r.semantic_score.toFixed(4)}</span>
        </div>
      </td>
      <td><span class="score-value">×${r.penalty_multiplier.toFixed(2)}</span></td>
      <td><span class="score-value" style="color:var(--success)">+${r.bonus_total.toFixed(2)}</span></td>
      <td class="reasoning-cell">${buildReasoning(r)}</td>
      <td>${penaltyBadges} ${bonusBadges}</td>
    `;
    tbody.appendChild(row);
  });
}

// --- Build reasoning summary (like the screenshot CSV) ---
function buildReasoning(r) {
  // Pull from penalty/bonus text if present
  const parts = [];
  if (r.penalties && r.penalties !== 'None') parts.push(`⚠ ${r.penalties}`);
  if (r.bonuses && r.bonuses !== 'None') parts.push(`⭐ ${r.bonuses}`);
  const sem = (r.semantic_score * 100).toFixed(0);
  const penStr = r.penalty_multiplier < 1.0 ? `penalty ×${r.penalty_multiplier.toFixed(2)}` : 'no penalty';
  const bonusStr = r.bonus_total > 0 ? `bonus +${r.bonus_total.toFixed(2)}` : 'no bonus';
  return `<span class="reasoning-text">Semantic ${sem}%; ${penStr}; ${bonusStr}${parts.length ? ' | ' + parts.join(', ') : ''}</span>`;
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

// --- Candidate Profile Modal ---
async function openCandidateProfile(candidateId) {
  const modal = document.getElementById('profileModal');
  const title = document.getElementById('profileModalTitle');
  const body = document.getElementById('profileModalBody');

  title.textContent = candidateId;
  body.innerHTML = '<div class="profile-loading">Loading profile…</div>';
  modal.classList.remove('hidden');

  try {
    const resp = await fetch(`/api/candidate/${candidateId}`);
    if (!resp.ok) throw new Error('Profile not found');
    const data = await resp.json();
    body.innerHTML = renderCandidateProfile(data);
  } catch (err) {
    body.innerHTML = `<div class="profile-loading" style="color:var(--error)">❌ ${err.message}</div>`;
  }
}

function closeProfileModal(event) {
  document.getElementById('profileModal').classList.add('hidden');
}

function renderCandidateProfile(c) {
  const p = c.profile || {};
  const signals = c.redrob_signals || {};
  const career = c.career_history || [];
  const skills = c.skills || [];
  const education = c.education || [];
  const certs = c.certifications || [];

  // Profile header
  let html = `
    <div class="prof-header">
      <div class="prof-avatar">${(p.anonymized_name || '?').charAt(0)}</div>
      <div>
        <div class="prof-name">${p.anonymized_name || 'Anonymous'}</div>
        <div class="prof-headline">${p.headline || ''}</div>
        <div class="prof-meta">
          ${p.current_title ? `<span>💼 ${p.current_title} @ ${p.current_company || ''}</span>` : ''}
          ${p.location ? `<span>📍 ${p.location}${p.country && p.country !== p.location ? ', '+p.country : ''}</span>` : ''}
          ${p.years_of_experience ? `<span>⏳ ${p.years_of_experience} yrs exp</span>` : ''}
        </div>
      </div>
    </div>`;

  // Summary
  if (p.summary) {
    html += `<div class="prof-section"><div class="prof-section-title">Summary</div><p class="prof-summary">${p.summary}</p></div>`;
  }

  // Redrob signals
  html += `<div class="prof-section">
    <div class="prof-section-title">📊 Redrob Signals</div>
    <div class="prof-signals-grid">
      ${signalPill('Completeness', signals.profile_completeness_score + '%')}
      ${signalPill('Open to Work', signals.open_to_work_flag ? '✅ Yes' : '❌ No')}
      ${signalPill('Response Rate', signals.recruiter_response_rate != null ? (signals.recruiter_response_rate*100).toFixed(0)+'%' : 'N/A')}
      ${signalPill('GitHub Score', signals.github_activity_score >= 0 ? signals.github_activity_score : 'N/A')}
      ${signalPill('Notice Period', signals.notice_period_days != null ? signals.notice_period_days + ' days' : 'N/A')}
      ${signalPill('Connections', signals.connection_count ?? 'N/A')}
      ${signalPill('Verified Email', signals.verified_email ? '✅' : '❌')}
      ${signalPill('Willing to Relocate', signals.willing_to_relocate ? '✅ Yes' : '❌ No')}
    </div>
  </div>`;

  // Career history
  if (career.length) {
    html += `<div class="prof-section"><div class="prof-section-title">💼 Career History</div>`;
    career.forEach(job => {
      const dur = job.duration_months ? `${Math.floor(job.duration_months/12)}y ${job.duration_months%12}m` : '';
      html += `<div class="prof-job">
        <div class="prof-job-header">
          <span class="prof-job-title">${job.title}</span>
          <span class="prof-job-company">${job.company}</span>
          <span class="prof-job-dur">${dur}${job.is_current ? ' (current)' : ''}</span>
        </div>
        ${job.description ? `<p class="prof-job-desc">${job.description}</p>` : ''}
      </div>`;
    });
    html += `</div>`;
  }

  // Skills
  if (skills.length) {
    html += `<div class="prof-section"><div class="prof-section-title">🛠 Skills</div><div class="prof-skills">`;
    skills.forEach(s => {
      html += `<span class="prof-skill-tag ${s.proficiency}">${s.name} <small>${s.proficiency}</small></span>`;
    });
    html += `</div></div>`;
  }

  // Education
  if (education.length) {
    html += `<div class="prof-section"><div class="prof-section-title">🎓 Education</div>`;
    education.forEach(e => {
      html += `<div class="prof-edu"><strong>${e.degree} in ${e.field_of_study}</strong> — ${e.institution} (${e.start_year}–${e.end_year}) <span class="prof-edu-tier">${e.tier || ''}</span></div>`;
    });
    html += `</div>`;
  }

  // Certifications
  if (certs.length) {
    html += `<div class="prof-section"><div class="prof-section-title">🏅 Certifications</div><ul class="prof-certs">`;
    certs.forEach(cert => {
      html += `<li>${cert.name} — ${cert.issuer} (${cert.year})</li>`;
    });
    html += `</ul></div>`;
  }

  // Skill assessment scores
  const assessments = signals.skill_assessment_scores || {};
  const assessKeys = Object.keys(assessments);
  if (assessKeys.length) {
    html += `<div class="prof-section"><div class="prof-section-title">🧠 Skill Assessments</div><div class="prof-signals-grid">`;
    assessKeys.forEach(k => {
      html += signalPill(k, assessments[k].toFixed(1));
    });
    html += `</div></div>`;
  }

  return html;
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
