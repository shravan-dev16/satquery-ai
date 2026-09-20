/**
 * SatQuery AI — Developer Test Console Client
 *
 * Dedicated lightweight testing harness for manual backend verification.
 * Completely decoupled from production frontend.
 * Direct REST consumer of POST /api/v1/analyze.
 */

const API_BASE_URL = window.__SATQUERY_API_BASE_URL__ || window.location.origin;

class SatQueryDeveloperConsole {
  constructor() {
    this.primaryFile = null;
    this.secondaryFile = null;
    this.currentResult = null;
    this.manifestData = [];

    this.initElements();
    this.bindEvents();
    this.checkHealth();
    this.loadDemoManifest();
  }

  initElements() {
    // Header status
    this.backendTarget = document.getElementById('backend-target');
    this.statusDot = document.getElementById('status-dot');
    this.healthText = document.getElementById('health-text');
    if (this.backendTarget) this.backendTarget.textContent = `Origin: ${API_BASE_URL}`;

    // Workflow & Inputs
    this.workflowSelect = document.getElementById('workflow-select');
    this.demoCount = document.getElementById('demo-count');
    this.demoSamplesList = document.getElementById('demo-samples-list');

    this.slotPrimary = document.getElementById('slot-primary');
    this.slotSecondary = document.getElementById('slot-secondary');
    this.lblPrimary = document.getElementById('lbl-primary');
    this.lblSecondary = document.getElementById('lbl-secondary');
    this.metaPrimary = document.getElementById('meta-primary');
    this.metaSecondary = document.getElementById('meta-secondary');
    this.inputPrimary = document.getElementById('input-primary');
    this.inputSecondary = document.getElementById('input-secondary');
    this.fileStatusTag = document.getElementById('file-status-tag');

    // Query & Action
    this.queryInput = document.getElementById('query-input');
    this.quickQueryBtns = document.querySelectorAll('.btn-quick-query[data-query]');
    this.btnAnalyze = document.getElementById('btn-analyze');
    this.btnAnalyzeText = document.getElementById('btn-analyze-text');

    // Results container
    this.errorBanner = document.getElementById('error-banner');
    this.errorTitle = document.getElementById('error-title');
    this.errorDetail = document.getElementById('error-detail');

    this.resultsEmpty = document.getElementById('results-empty');
    this.resultsContainer = document.getElementById('results-container');

    // Result summary
    this.valObservedRoute = document.getElementById('val-observed-route');
    this.valTaskName = document.getElementById('val-task-name');
    this.valHttpStatus = document.getElementById('val-http-status');
    this.valLatency = document.getElementById('val-latency');
    this.valAnswer = document.getElementById('val-answer');
    this.answerBox = document.getElementById('answer-box');

    // Semantic disclaimer
    this.semanticDisclaimerCard = document.getElementById('semantic-disclaimer-card');
    this.semanticDisclaimerText = document.getElementById('semantic-disclaimer-text');

    // Statistics
    this.statChangedPixels = document.getElementById('stat-changed-pixels');
    this.statPhysicalArea = document.getElementById('stat-physical-area');
    this.statAreaNote = document.getElementById('stat-area-note');
    this.statChangeRatio = document.getElementById('stat-change-ratio');
    this.statClusters = document.getElementById('stat-clusters');
    this.statSemanticClass = document.getElementById('stat-semantic-class');
    this.statSemanticStatus = document.getElementById('stat-semantic-status');

    // Confidence M9
    this.badgeConfLevel = document.getElementById('badge-conf-level');
    this.valConfScore = document.getElementById('val-conf-score');
    this.confCalibType = document.getElementById('conf-calib-type');
    this.subSpecialist = document.getElementById('sub-specialist');
    this.subEvidence = document.getElementById('sub-evidence');
    this.subAlignment = document.getElementById('sub-alignment');
    this.subInput = document.getElementById('sub-input');
    this.subPenalty = document.getElementById('sub-penalty');

    // Consistency M8
    this.badgeConsistencyStatus = document.getElementById('badge-consistency-status');
    this.valConsistencyNarrative = document.getElementById('val-consistency-narrative');
    this.valEvidenceQuality = document.getElementById('val-evidence-quality');
    this.valFusedCount = document.getElementById('val-fused-count');
    this.conflictsBox = document.getElementById('conflicts-box');
    this.conflictsList = document.getElementById('conflicts-list');

    // Visuals & Evidence
    this.visualsCount = document.getElementById('visuals-count');
    this.visualsGrid = document.getElementById('visuals-grid');
    this.evidenceUnitsCount = document.getElementById('evidence-units-count');
    this.evidenceTableBody = document.getElementById('evidence-table-body');

    // Trace
    this.traceStepCount = document.getElementById('trace-step-count');
    this.traceTimeline = document.getElementById('trace-timeline');

    // M11 Report
    this.reportPanel = document.getElementById('report-panel');
    this.reportIdLabel = document.getElementById('report-id-label');
    this.reportPreviewBox = document.getElementById('report-preview-box');
    this.btnViewReportJson = document.getElementById('btn-view-report-json');
    this.btnViewReportMd = document.getElementById('btn-view-report-md');
    this.btnDownloadReport = document.getElementById('btn-download-report');

    // Raw JSON
    this.rawJsonPre = document.getElementById('raw-json-pre');
    this.btnCopyJson = document.getElementById('btn-copy-json');
  }

  bindEvents() {
    // Workflow template selection changes slot labels/visibility
    this.workflowSelect.addEventListener('change', () => this.handleWorkflowChange());

    // Native file input change handlers
    this.inputPrimary.addEventListener('change', (e) => this.handleFileSelect(e, 'primary'));
    this.inputSecondary.addEventListener('change', (e) => this.handleFileSelect(e, 'secondary'));

    // Quick query buttons
    this.quickQueryBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        this.queryInput.value = btn.dataset.query;
      });
    });

    // Primary Action
    this.btnAnalyze.addEventListener('click', () => this.executeAnalysis());

    // Copy JSON
    this.btnCopyJson.addEventListener('click', () => {
      if (this.currentResult) {
        navigator.clipboard.writeText(JSON.stringify(this.currentResult, null, 2)).then(() => {
          this.btnCopyJson.textContent = 'Copied!';
          setTimeout(() => { this.btnCopyJson.textContent = 'Copy JSON'; }, 1500);
        });
      }
    });

    // Report buttons
    if (this.btnViewReportJson) {
      this.btnViewReportJson.addEventListener('click', () => this.renderReportView('json'));
    }
    if (this.btnViewReportMd) {
      this.btnViewReportMd.addEventListener('click', () => this.renderReportView('md'));
    }
    if (this.btnDownloadReport) {
      this.btnDownloadReport.addEventListener('click', () => this.downloadReportJson());
    }

    // Initialize slots
    this.handleWorkflowChange();
  }

  async checkHealth() {
    try {
      const res = await fetch(`${API_BASE_URL}/health`);
      if (res.ok) {
        const data = await res.json();
        this.statusDot.className = 'status-dot dot-connected';
        this.healthText.textContent = `CONNECTED (${data.registered_specialists || 6} Specialists)`;
      } else {
        throw new Error(`HTTP ${res.status}`);
      }
    } catch (e) {
      this.statusDot.className = 'status-dot dot-error';
      this.healthText.textContent = 'BACKEND OFFLINE';
    }
  }

  async loadDemoManifest() {
    try {
      // Fetch manifest from backend API or static mount
      let manifest = null;
      try {
        const res = await fetch(`${API_BASE_URL}/api/v1/demo/manifest`);
        if (res.ok) manifest = await res.json();
      } catch (e) {
        // Fallback
      }

      if (!manifest || manifest.length === 0) {
        // Fallback to static manifest file
        const res2 = await fetch(`${API_BASE_URL}/api/v1/demo/files/manifest.json`);
        if (res2.ok) manifest = await res2.json();
      }

      if (manifest && Array.isArray(manifest)) {
        this.manifestData = manifest;
        this.demoCount.textContent = `${manifest.length} Samples`;
        this.renderDemoSamplesList(manifest);
      } else {
        this.demoCount.textContent = 'Manual upload';
        this.demoSamplesList.innerHTML = '<div style="color: var(--text-muted); font-size: 11px;">Manual file upload mode active.</div>';
      }
    } catch (err) {
      this.demoCount.textContent = 'Offline';
      this.demoSamplesList.innerHTML = '<div style="color: var(--text-muted); font-size: 11px;">Upload sample files manually via file inputs below.</div>';
    }
  }

  renderDemoSamplesList(samples) {
    this.demoSamplesList.innerHTML = '';
    samples.forEach(sample => {
      const item = document.createElement('div');
      item.className = 'demo-sample-item';

      const info = document.createElement('div');
      info.className = 'demo-sample-info';

      const name = document.createElement('span');
      name.className = 'demo-sample-name';
      name.textContent = sample.id;

      const desc = document.createElement('span');
      desc.className = 'demo-sample-desc';
      desc.textContent = `${sample.workflow} • "${sample.suggested_query}"`;

      info.appendChild(name);
      info.appendChild(desc);

      const btn = document.createElement('button');
      btn.className = 'btn-load-sample';
      btn.textContent = 'Load';
      btn.title = `Pre-fill ${sample.id} into inputs`;
      btn.addEventListener('click', () => this.loadDemoSample(sample));

      item.appendChild(info);
      item.appendChild(btn);
      this.demoSamplesList.appendChild(item);
    });
  }

  async loadDemoSample(sample) {
    this.queryInput.value = sample.suggested_query;

    // Match workflow selector
    if (sample.workflow.includes('vqa')) this.workflowSelect.value = 'single_vqa';
    else if (sample.workflow.includes('grounding')) this.workflowSelect.value = 'grounding';
    else if (sample.workflow.includes('pure')) this.workflowSelect.value = 'bitemporal_pure';
    else if (sample.workflow.includes('optical_sar')) this.workflowSelect.value = 'optical_sar';
    else if (sample.workflow.includes('semantic') || sample.workflow.includes('quantity')) this.workflowSelect.value = 'semantic_change';
    else this.workflowSelect.value = 'combined';

    this.handleWorkflowChange();

    // Fetch binary files from demo files mount
    try {
      const files = sample.files || [];
      if (files.length > 0) {
        const p1Name = files[0].split('/').pop();
        const res1 = await fetch(`${API_BASE_URL}/api/v1/demo/files/${p1Name}`);
        if (res1.ok) {
          const blob1 = await res1.blob();
          this.primaryFile = new File([blob1], p1Name, { type: blob1.type || 'image/tiff' });
          this.metaPrimary.textContent = `${p1Name} (${this.formatBytes(this.primaryFile.size)})`;
        }
      }

      if (files.length > 1) {
        const p2Name = files[1].split('/').pop();
        const res2 = await fetch(`${API_BASE_URL}/api/v1/demo/files/${p2Name}`);
        if (res2.ok) {
          const blob2 = await res2.blob();
          this.secondaryFile = new File([blob2], p2Name, { type: blob2.type || 'image/tiff' });
          this.metaSecondary.textContent = `${p2Name} (${this.formatBytes(this.secondaryFile.size)})`;
        }
      } else {
        this.secondaryFile = null;
        this.metaSecondary.textContent = 'No file';
      }

      this.updateAnalyzeButtonState();
    } catch (e) {
      console.warn('Could not auto-fetch demo blob:', e);
    }
  }

  handleWorkflowChange() {
    const wf = this.workflowSelect.value;
    if (wf === 'single_vqa') {
      this.lblPrimary.textContent = 'Satellite Image (Single Observation)';
      this.slotSecondary.classList.add('hidden');
    } else if (wf === 'grounding') {
      this.lblPrimary.textContent = 'Target Image to Ground';
      this.slotSecondary.classList.add('hidden');
    } else if (wf === 'bitemporal_pure' || wf === 'semantic_change') {
      this.lblPrimary.textContent = 'Time 1 Observation (T1 - Earlier)';
      this.lblSecondary.textContent = 'Time 2 Observation (T2 - Later)';
      this.slotSecondary.classList.remove('hidden');
    } else if (wf === 'optical_sar') {
      this.lblPrimary.textContent = 'Optical Observation (RGB / Multispectral)';
      this.lblSecondary.textContent = 'SAR Observation (Radar Amplitude / Backscatter)';
      this.slotSecondary.classList.remove('hidden');
    } else {
      this.lblPrimary.textContent = 'Primary Observation (T1 / Optical)';
      this.lblSecondary.textContent = 'Secondary Observation (T2 / SAR)';
      this.slotSecondary.classList.remove('hidden');
    }
    this.updateAnalyzeButtonState();
  }

  handleFileSelect(e, slot) {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      if (slot === 'primary') {
        this.primaryFile = file;
        this.metaPrimary.textContent = `${file.name} (${this.formatBytes(file.size)})`;
      } else {
        this.secondaryFile = file;
        this.metaSecondary.textContent = `${file.name} (${this.formatBytes(file.size)})`;
      }
    }
    this.updateAnalyzeButtonState();
  }

  updateAnalyzeButtonState() {
    const hasPrimary = Boolean(this.primaryFile);
    const count = (this.primaryFile ? 1 : 0) + (this.secondaryFile ? 1 : 0);
    this.fileStatusTag.textContent = `${count} File(s) Loaded`;
    this.btnAnalyze.disabled = !hasPrimary;
  }

  async executeAnalysis() {
    if (!this.primaryFile) return;

    this.btnAnalyze.disabled = true;
    this.btnAnalyzeText.textContent = 'Running Model Execution...';
    this.errorBanner.classList.add('hidden');
    this.resultsContainer.classList.add('hidden');
    this.resultsEmpty.classList.remove('hidden');

    const startTime = performance.now();

    const formData = new FormData();
    formData.append('query', this.queryInput.value || 'Describe this scene.');
    formData.append('image_primary', this.primaryFile);
    if (this.secondaryFile) {
      formData.append('image_secondary', this.secondaryFile);
    }

    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/analyze`, {
        method: 'POST',
        body: formData,
      });

      const latencyMs = Math.round(performance.now() - startTime);
      const data = await res.json();

      if (!res.ok) {
        const detail = data.detail || `Server returned HTTP ${res.status}`;
        this.showError(`Analysis Request Failed (HTTP ${res.status})`, typeof detail === 'string' ? detail : JSON.stringify(detail, null, 2));
        return;
      }

      this.currentResult = data;
      this.renderResults(data, latencyMs, res.status);
    } catch (err) {
      this.showError('Network / Client Error', err.message);
    } finally {
      this.btnAnalyze.disabled = false;
      this.btnAnalyzeText.textContent = 'Analyze Query';
    }
  }

  renderResults(contract, latencyMs, httpStatus) {
    this.resultsEmpty.classList.add('hidden');
    this.resultsContainer.classList.remove('hidden');

    // Top Status & Route Bar
    this.valHttpStatus.textContent = `HTTP ${httpStatus}`;
    this.valHttpStatus.className = httpStatus === 200 ? 'status-badge badge-200' : 'status-badge badge-err';
    this.valLatency.textContent = `${contract.execution_time_ms || latencyMs} ms`;
    this.valTaskName.textContent = `TASK: ${(contract.task || 'UNKNOWN').toUpperCase()}`;

    // Compute Observed Route from trace & models
    let observedRoute = 'AUTO-AGENT';
    if (contract.models && contract.models.length > 0) {
      observedRoute = contract.models.map(m => m.identifier || m.model_name || m).join(' → ');
    } else if (contract.parameters?.candidate_model) {
      observedRoute = contract.parameters.candidate_model.toUpperCase();
    } else if (contract.task === 'single_image_grounding') {
      observedRoute = 'RS_GROUND';
    } else if (contract.task === 'single_image_vqa') {
      observedRoute = 'RS_VQA';
    } else if (contract.task === 'optical_sar_analysis') {
      observedRoute = 'OPTICAL_SAR_FUSION';
    }
    this.valObservedRoute.textContent = observedRoute;

    // Final Answer & Gating
    this.valAnswer.textContent = contract.answer || 'Analysis complete.';
    this.answerBox.className = 'answer-box';
    if (contract.answer && contract.answer.includes('[CONTRADICTION DETECTED]')) {
      this.answerBox.classList.add('gated-contradiction');
    } else if (contract.answer && contract.answer.includes('[INSUFFICIENT EVIDENCE]')) {
      this.answerBox.classList.add('gated-insufficient');
    }

    // Semantic Area Limitation Check
    const params = contract.parameters || {};
    const semStatus = params.semantic_area_status;
    const isSemUnmeasured = semStatus === 'unmeasured_from_spatial_evidence';
    if (isSemUnmeasured || params.semantic_area_limitation) {
      this.semanticDisclaimerCard.classList.remove('hidden');
      this.semanticDisclaimerText.textContent = params.semantic_area_limitation ||
        'Semantic-specific area is not directly measurable from current spatial evidence without pixel-level multi-class semantic segmentation. Total detected change encompasses all verified physical surface transitions.';
    } else {
      this.semanticDisclaimerCard.classList.add('hidden');
    }

    // Quantitative Statistics
    const changedPixels = params.changed_pixels !== undefined ? params.changed_pixels.toLocaleString() + ' px' : '--';
    this.statChangedPixels.textContent = changedPixels;

    const statM2 = contract.evidence?.statistics?.find(s => s.metric_name === 'changed_area_m2');
    const statHa = contract.evidence?.statistics?.find(s => s.metric_name === 'changed_area_hectares');
    if (statM2 && statHa) {
      this.statPhysicalArea.textContent = `${statHa.value.toFixed(2)} ha`;
      this.statAreaNote.textContent = `${statM2.value.toLocaleString()} m²`;
    } else {
      this.statPhysicalArea.textContent = 'Pixel Space';
      this.statAreaNote.textContent = 'No CRS bounds';
    }

    const ratio = params.change_ratio_pct !== undefined ? params.change_ratio_pct.toFixed(2) + '%' : '--';
    this.statChangeRatio.textContent = ratio;

    const clusters = params.total_clusters !== undefined ? params.total_clusters : (contract.evidence?.boxes?.length || 0);
    this.statClusters.textContent = clusters;

    this.statSemanticClass.textContent = params.semantic_class_detected || 'N/A';
    this.statSemanticStatus.textContent = semStatus || 'Standard analysis';

    // M9 Defensible System Confidence
    const breakdown = contract.confidence_breakdown || {};
    const overallConf = contract.confidence !== null && contract.confidence !== undefined ? contract.confidence : 0.0;
    const confLevel = contract.confidence_level || breakdown.confidence_level || 'EVALUATED';

    this.valConfScore.textContent = `${(overallConf * 100).toFixed(1)}%`;
    this.badgeConfLevel.textContent = confLevel;
    this.badgeConfLevel.className = `badge-pill badge-${confLevel}`;

    this.confCalibType.textContent = breakdown.is_calibrated_probability
      ? 'Calibrated Statistical Probability'
      : 'Multi-Factor Grounded Heuristic (M9)';

    this.subSpecialist.textContent = breakdown.specialist_confidence !== undefined ? (breakdown.specialist_confidence * 100).toFixed(1) + '%' : '--';
    this.subEvidence.textContent = breakdown.evidence_quality_score !== undefined ? (breakdown.evidence_quality_score * 100).toFixed(1) + '%' : '--';
    this.subAlignment.textContent = breakdown.spatial_alignment_score !== undefined ? (breakdown.spatial_alignment_score * 100).toFixed(1) + '%' : '--';
    this.subInput.textContent = breakdown.input_quality_score !== undefined ? (breakdown.input_quality_score * 100).toFixed(1) + '%' : '--';
    this.subPenalty.textContent = breakdown.consistency_penalty !== undefined ? (breakdown.consistency_penalty > 0 ? `-${breakdown.consistency_penalty.toFixed(2)}` : '0.00') : '0.00';

    // M8 Consistency & Reliability
    const consReport = contract.evidence?.consistency_report;
    const consStatus = contract.evidence_status || consReport?.status || 'CONSISTENT';
    this.badgeConsistencyStatus.textContent = consStatus;
    this.badgeConsistencyStatus.className = `badge-pill badge-${consStatus}`;
    this.valConsistencyNarrative.textContent = consReport?.summary_narrative || 'Evidence is consistent across specialists.';
    this.valEvidenceQuality.textContent = consReport?.evidence_quality_score !== undefined ? `${Math.round(consReport.evidence_quality_score * 100)}%` : '100%';

    const fused = contract.evidence?.fused_items || [];
    this.valFusedCount.textContent = fused.length;

    const conflicts = consReport?.conflicts || [];
    if (conflicts.length > 0) {
      this.conflictsBox.classList.remove('hidden');
      this.conflictsList.innerHTML = conflicts.map(c => `<li><strong>[${c.rule_violated}]</strong> ${c.description}</li>`).join('');
    } else {
      this.conflictsBox.classList.add('hidden');
      this.conflictsList.innerHTML = '';
    }

    // Visual Artifacts Gallery
    this.renderVisuals(contract);

    // Evidence Units Table
    this.renderEvidenceUnits(contract);

    // Execution Trace
    this.renderTrace(contract.execution_trace || []);

    // M11 Analyst Report
    if (contract.report) {
      this.reportPanel.classList.remove('hidden');
      const repId = contract.report.metadata?.report_id || contract.report.report_id || 'Attached';
      this.reportIdLabel.textContent = repId;
      this.renderReportView('json');
    } else {
      this.reportPanel.classList.add('hidden');
    }

    // Raw JSON Pretty Print
    this.rawJsonPre.textContent = JSON.stringify(contract, null, 2);
  }

  renderVisuals(contract) {
    this.visualsGrid.innerHTML = '';
    const images = contract.evidence?.images || [];
    const masks = contract.evidence?.masks || [];

    const artifacts = [];
    images.forEach(img => {
      if (img.url) artifacts.push({ title: (img.role || 'Preview Image').replace(/_/g, ' '), url: img.url });
    });
    masks.forEach(m => {
      if (m.url) artifacts.push({ title: m.label || 'Mask Overlay', url: m.url });
    });

    this.visualsCount.textContent = `${artifacts.length} Previews`;

    if (artifacts.length === 0) {
      this.visualsGrid.innerHTML = '<div style="color: var(--text-muted); font-size: 11px;">No visual previews attached to response.</div>';
      return;
    }

    artifacts.forEach(item => {
      const card = document.createElement('div');
      card.className = 'visual-item';

      const title = document.createElement('div');
      title.className = 'visual-title';
      title.textContent = item.title;

      const imgWrap = document.createElement('div');
      imgWrap.className = 'visual-img-wrap';

      const img = document.createElement('img');
      img.className = 'visual-img';
      img.alt = item.title;
      // Resolve relative path against API_BASE_URL
      const fullUrl = item.url.startsWith('/') ? `${API_BASE_URL}${item.url}` : item.url;
      img.src = fullUrl;

      imgWrap.appendChild(img);

      const link = document.createElement('a');
      link.className = 'visual-link';
      link.href = fullUrl;
      link.target = '_blank';
      link.textContent = item.url;

      card.appendChild(title);
      card.appendChild(imgWrap);
      card.appendChild(link);
      this.visualsGrid.appendChild(card);
    });
  }

  renderEvidenceUnits(contract) {
    this.evidenceTableBody.innerHTML = '';
    const fused = contract.evidence?.fused_items || [];
    const boxes = contract.evidence?.boxes || [];

    if (fused.length > 0) {
      this.evidenceUnitsCount.textContent = fused.length;
      fused.forEach(item => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><code>${item.item_id}</code></td>
          <td>${item.evidence_type}</td>
          <td>${item.source_specialist}</td>
          <td>${item.claim}</td>
          <td>${(item.specialist_confidence * 100).toFixed(1)}%</td>
        `;
        this.evidenceTableBody.appendChild(tr);
      });
    } else if (boxes.length > 0) {
      this.evidenceUnitsCount.textContent = boxes.length;
      boxes.forEach(b => {
        const coords = b.coordinates_pixel ? `[${b.coordinates_pixel.join(', ')}]` : 'N/A';
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><code>${b.box_id}</code></td>
          <td>bounding_box</td>
          <td>RS_GROUND</td>
          <td>${b.label} bbox ${coords}</td>
          <td>${(b.confidence * 100).toFixed(1)}%</td>
        `;
        this.evidenceTableBody.appendChild(tr);
      });
    } else {
      this.evidenceUnitsCount.textContent = '0';
      this.evidenceTableBody.innerHTML = '<tr><td colspan="5" style="color: var(--text-muted); text-align: center;">No discrete evidence items.</td></tr>';
    }
  }

  renderTrace(trace) {
    this.traceTimeline.innerHTML = '';
    this.traceStepCount.textContent = `${trace.length} Steps`;

    trace.forEach(step => {
      const row = document.createElement('div');
      row.className = 'trace-step-row';
      row.innerHTML = `
        <span class="trace-step-num">${step.step_number}.</span>
        <span class="trace-step-name">${step.step_name}</span>
        <span class="trace-step-dur">${step.duration_ms} ms</span>
        <span class="trace-step-detail">${step.details || ''}</span>
      `;
      this.traceTimeline.appendChild(row);
    });
  }

  renderReportView(mode) {
    if (!this.currentResult?.report) return;
    const rpt = this.currentResult.report;

    if (mode === 'json') {
      this.reportPreviewBox.textContent = JSON.stringify(rpt, null, 2);
    } else {
      // Build markdown summary
      const repId = rpt.metadata?.report_id || rpt.report_id || 'REPORT';
      const repDate = rpt.metadata?.timestamp || rpt.timestamp_utc || 'N/A';
      const repTask = rpt.task || rpt.task_type || 'N/A';
      const repAnswer = rpt.answer || rpt.executive_summary || 'No summary available.';
      const stats = rpt.statistics || [];
      const statsLines = stats.map(s => `- ${s.display_name || s.metric_name}: ${s.value} ${s.unit || ''}`);

      const md = [
        `# SatQuery Earth Observation Intelligence Report`,
        `**Report ID:** ${repId} | **Timestamp:** ${repDate}`,
        `**Task Type:** ${repTask}`,
        '',
        `## Executive Summary`,
        repAnswer,
        '',
        `## Quantitative Metrics`,
        statsLines.length > 0 ? statsLines.join('\n') : '- No quantitative metrics recorded.',
      ].join('\n');
      this.reportPreviewBox.textContent = md;
    }
  }

  downloadReportJson() {
    if (!this.currentResult?.report) return;
    const repId = this.currentResult.report.metadata?.report_id || this.currentResult.report.report_id || 'export';
    const blob = new Blob([JSON.stringify(this.currentResult.report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `satquery_report_${repId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  showError(title, detail) {
    this.errorTitle.textContent = title;
    this.errorDetail.textContent = detail;
    this.errorBanner.classList.remove('hidden');
    this.resultsContainer.classList.add('hidden');
    this.resultsEmpty.classList.remove('hidden');
  }

  formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }
}

// Instantiate on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  window.devConsole = new SatQueryDeveloperConsole();
});
