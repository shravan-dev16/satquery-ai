/**
 * SatQuery AI — Interactive Remote Sensing Analysis Web Client
 * Milestone M12 Final Product Integration
 *
 * Adheres to:
 * - docs/API_CONTRACT.md
 * - docs/UI_HANDOFF.md
 * - AGENTS.md (Rules 7, 8, 9, 10, 11, 12, 18, 19, 29, 37)
 *
 * Strictly interfaces with live backend:
 * - GET /health & GET /api/v1/health (readiness & telemetry)
 * - GET /api/v1/demo/manifest & /api/v1/demo/files/{filename} (curated samples)
 * - POST /api/v1/analyze (autonomous agent orchestration & multi-specialist DAG)
 */

const API_BASE_URL = window.__SATQUERY_API_BASE_URL__ || window.location.origin;

class SatQueryClient {
  constructor() {
    this.t1File = null;
    this.t2File = null;
    this.currentResult = null;
    this.isAnalyzing = false;
    this.activeViewMode = 'slider'; // 'slider' | 'side'
    this.manifestData = [];

    this.initElements();
    this.bindEvents();
    this.checkBackendHealth();
    this.loadDemoManifest();
  }

  initElements() {
    // Header & Status
    this.statusDot = document.getElementById('backend-status-dot');
    this.statusText = document.getElementById('backend-status-text');

    // Demo Sample Showcase
    this.demoSampleSelect = document.getElementById('demo-sample-select');
    this.btnLoadSample = document.getElementById('btn-load-sample');

    // T1 / Primary Elements
    this.t1Card = document.getElementById('t1-card');
    this.t1HeaderLabel = document.getElementById('t1-header-label');
    this.t1Dropzone = document.getElementById('t1-dropzone');
    this.t1Input = document.getElementById('t1-file-input');
    this.t1EmptyView = document.getElementById('t1-empty-view');
    this.t1LoadedView = document.getElementById('t1-loaded-view');
    this.t1Thumb = document.getElementById('t1-thumb');
    this.t1ThumbFallback = document.getElementById('t1-thumb-fallback');
    this.t1Filename = document.getElementById('t1-filename');
    this.t1Filesize = document.getElementById('t1-filesize');
    this.t1FileCrs = document.getElementById('t1-filecrs');
    this.t1RemoveBtn = document.getElementById('t1-remove-btn');

    // T2 / Secondary Elements
    this.t2Card = document.getElementById('t2-card');
    this.t2HeaderLabel = document.getElementById('t2-header-label');
    this.t2Dropzone = document.getElementById('t2-dropzone');
    this.t2Input = document.getElementById('t2-file-input');
    this.t2EmptyView = document.getElementById('t2-empty-view');
    this.t2LoadedView = document.getElementById('t2-loaded-view');
    this.t2Thumb = document.getElementById('t2-thumb');
    this.t2ThumbFallback = document.getElementById('t2-thumb-fallback');
    this.t2Filename = document.getElementById('t2-filename');
    this.t2Filesize = document.getElementById('t2-filesize');
    this.t2FileCrs = document.getElementById('t2-filecrs');
    this.t2RemoveBtn = document.getElementById('t2-remove-btn');

    // Query & Action
    this.queryInput = document.getElementById('query-input');
    this.candidateSelect = document.getElementById('candidate-select');
    this.analyzeBtn = document.getElementById('analyze-btn');
    this.queryPills = document.querySelectorAll('.query-pill');

    // Progress & Pipeline Stepper
    this.progressContainer = document.getElementById('progress-container');
    this.progressHeadline = document.getElementById('progress-headline');
    this.progressSubtext = document.getElementById('progress-subtext');
    this.stepperItems = document.querySelectorAll('.stepper-item');

    // Error Banner
    this.errorBanner = document.getElementById('error-banner');
    this.errorTitle = document.getElementById('error-title');
    this.errorDetail = document.getElementById('error-detail');
    this.errorCloseBtn = document.getElementById('error-close-btn');

    // Results Section & Narrative
    this.resultsSection = document.getElementById('results-section');
    this.summaryAnswer = document.getElementById('summary-answer');
    this.summaryConfidence = document.getElementById('summary-confidence');
    this.summaryTask = document.getElementById('summary-task');

    // Warnings Banner
    this.warningsCard = document.getElementById('warnings-card');
    this.warningsList = document.getElementById('warnings-list');

    // Defensible System Confidence Card (M9)
    this.confidenceCard = document.getElementById('confidence-card');
    this.confLevelBadge = document.getElementById('conf-level-badge');
    this.confOverallScore = document.getElementById('conf-overall-score');
    this.confCalibrationBadge = document.getElementById('conf-calibration-badge');
    this.subfactorSpecialist = document.getElementById('subfactor-specialist');
    this.subfactorEvidence = document.getElementById('subfactor-evidence');
    this.subfactorAlignment = document.getElementById('subfactor-alignment');
    this.subfactorInput = document.getElementById('subfactor-input');
    this.subfactorConsistency = document.getElementById('subfactor-consistency');
    this.confFactorsList = document.getElementById('conf-factors-list');
    this.confWarningsList = document.getElementById('conf-warnings-list');

    // Quantitative Telemetry Metrics
    this.valChangedPixels = document.getElementById('val-changed-pixels');
    this.valPhysicalArea = document.getElementById('val-physical-area');
    this.subPhysicalArea = document.getElementById('sub-physical-area');
    this.valChangeRatio = document.getElementById('val-change-ratio');
    this.valTotalClusters = document.getElementById('val-total-clusters');
    this.valSpecialist = document.getElementById('val-specialist');
    this.valLatency = document.getElementById('val-latency');

    // Visual Hub Containers
    this.visualHubCard = document.querySelector('.visual-hub-card');
    this.hubMainTitle = document.getElementById('hub-main-title');
    this.hubMainSubtitle = document.getElementById('hub-main-subtitle');
    this.bitemporalModeSwitch = document.getElementById('bitemporal-mode-switch');
    this.btnModeSlider = document.getElementById('btn-mode-slider');
    this.btnModeSide = document.getElementById('btn-mode-side');

    // Mode 1: Split Slider
    this.sliderView = document.getElementById('slider-view');
    this.sliderFrame = document.getElementById('split-slider-frame');
    this.sliderImgBefore = document.getElementById('slider-img-before');
    this.sliderImgAfter = document.getElementById('slider-img-after');
    this.sliderOverlayWrapper = document.getElementById('slider-overlay-wrapper');
    this.sliderDivider = document.getElementById('slider-divider');

    // Mode 2: Side-by-Side Gallery
    this.sideBySideGrid = document.getElementById('side-by-side-grid');
    this.galleryT1 = document.getElementById('gallery-t1');
    this.galleryT2 = document.getElementById('gallery-t2');
    this.galleryMask = document.getElementById('gallery-mask');
    this.galleryOverlay = document.getElementById('gallery-overlay');

    // Mode 3: Grounding / Single-Image View
    this.singleViewContainer = document.getElementById('single-view-container');
    this.singlePreviewImg = document.getElementById('single-preview-img');
    this.singlePreviewTag = document.getElementById('single-preview-tag');
    this.singleCaption = document.getElementById('single-caption');

    // Mode 4: Optical-SAR Preview & Complementarity (M6)
    this.opticalSarPreviewCard = document.getElementById('optical-sar-preview-card');
    this.opticalSarCompositeImg = document.getElementById('optical-sar-composite-img');
    this.complementarityCard = document.getElementById('complementarity-card');
    this.compOpticalText = document.getElementById('comp-optical-text');
    this.compSarText = document.getElementById('comp-sar-text');
    this.compJointText = document.getElementById('comp-joint-text');

    // Semantic Card (M5)
    this.semanticCard = document.getElementById('semantic-card');
    this.semanticDirectionBadge = document.getElementById('semantic-direction-badge');
    this.predomTransitionBadge = document.getElementById('predom-transition-badge');
    this.semanticUncertaintyBadge = document.getElementById('semantic-uncertainty-badge');
    this.semanticTransitionsContainer = document.getElementById('semantic-transitions-container');

    // Evidence Consistency & Reliability Card (M8)
    this.evidenceStatusCard = document.getElementById('evidence-status-card');
    this.consistencyStatusBadge = document.getElementById('consistency-status-badge');
    this.evidenceQualityBadge = document.getElementById('evidence-quality-badge');
    this.provenanceCountBadge = document.getElementById('provenance-count-badge');
    this.consistencyNarrative = document.getElementById('consistency-narrative');
    this.conflictsContainer = document.getElementById('conflicts-container');
    this.conflictsList = document.getElementById('conflicts-list');
    this.evidencePillsRow = document.getElementById('evidence-pills-row');

    // Bounding Box Telemetry
    this.telemetryRegionsList = document.getElementById('telemetry-regions-list');

    // Structured Report Download Controls (M11)
    this.btnDownloadReportJson = document.getElementById('btn-download-report-json');
    this.btnDownloadReportMd = document.getElementById('btn-download-report-md');
    this.btnToggleReportPreview = document.getElementById('btn-toggle-report-preview');
    this.reportPreviewBox = document.getElementById('report-preview-box');
    this.reportPreviewText = document.getElementById('report-preview-text');

    // Execution Trace Accordion (Rule 12)
    this.traceToggleBtn = document.getElementById('trace-toggle-btn');
    this.traceContent = document.getElementById('trace-content');
    this.traceArrow = document.getElementById('accordion-arrow');
    this.traceTimeline = document.getElementById('trace-timeline-container');
    this.traceStepsCount = document.getElementById('trace-steps-count');

    // Developer & Diagnostics Console
    this.debugToggleBtn = document.getElementById('debug-toggle-btn');
    this.debugContent = document.getElementById('debug-content');
    this.debugArrow = document.getElementById('debug-accordion-arrow');
    this.debugJsonViewer = document.getElementById('debug-json-viewer');
  }

  bindEvents() {
    // T1 Upload Events
    this.t1Input.addEventListener('change', (e) => this.handleFileSelect(e, 't1'));
    this.setupDragAndDrop(this.t1Dropzone, 't1');
    this.t1RemoveBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.clearFile('t1');
    });

    // T2 Upload Events
    this.t2Input.addEventListener('change', (e) => this.handleFileSelect(e, 't2'));
    this.setupDragAndDrop(this.t2Dropzone, 't2');
    this.t2RemoveBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.clearFile('t2');
    });

    // Demo Sample Picker Events
    if (this.btnLoadSample) {
      this.btnLoadSample.addEventListener('click', () => {
        const selectedId = this.demoSampleSelect.value;
        if (selectedId) this.loadDemoSampleById(selectedId);
      });
    }
    if (this.demoSampleSelect) {
      this.demoSampleSelect.addEventListener('change', () => {
        const selectedId = this.demoSampleSelect.value;
        if (selectedId) this.loadDemoSampleById(selectedId);
      });
    }

    // Query Pills
    this.queryPills.forEach(pill => {
      pill.addEventListener('click', () => {
        this.queryInput.value = pill.dataset.query;
      });
    });

    // Analyze CTA Button
    this.analyzeBtn.addEventListener('click', () => this.executeAnalysis());

    // Error Close Button
    this.errorCloseBtn.addEventListener('click', () => {
      this.errorBanner.classList.add('hidden');
    });

    // View Mode Toggle (Slider vs Side-by-Side)
    if (this.btnModeSlider) {
      this.btnModeSlider.addEventListener('click', () => this.setViewMode('slider'));
    }
    if (this.btnModeSide) {
      this.btnModeSide.addEventListener('click', () => this.setViewMode('side'));
    }

    // Split Slider Interactive Drag
    this.setupSplitSlider();

    // Report Download & Preview Actions (M11)
    if (this.btnDownloadReportJson) {
      this.btnDownloadReportJson.addEventListener('click', () => this.downloadReportJson());
    }
    if (this.btnDownloadReportMd) {
      this.btnDownloadReportMd.addEventListener('click', () => this.downloadReportMarkdown());
    }
    if (this.btnToggleReportPreview) {
      this.btnToggleReportPreview.addEventListener('click', () => this.toggleReportPreview());
    }

    // Trace Accordion Toggle
    if (this.traceToggleBtn) {
      this.traceToggleBtn.addEventListener('click', () => {
        const isHidden = this.traceContent.classList.toggle('hidden');
        if (this.traceArrow) this.traceArrow.classList.toggle('open', !isHidden);
      });
    }

    // Developer / Debug Accordion Toggle
    if (this.debugToggleBtn) {
      this.debugToggleBtn.addEventListener('click', () => {
        const isHidden = this.debugContent.classList.toggle('hidden');
        if (this.debugArrow) this.debugArrow.classList.toggle('open', !isHidden);
      });
    }
  }

  async checkBackendHealth() {
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/health`);
      if (res.ok) {
        const data = await res.json();
        this.statusDot.className = 'status-dot dot-online';
        this.statusText.textContent = `Online • ${data.registered_specialists || 6} Specialists Active`;
      } else {
        throw new Error(`HTTP ${res.status}`);
      }
    } catch (err) {
      this.statusDot.className = 'status-dot dot-offline';
      this.statusText.textContent = 'Backend Offline';
    }
  }

  async loadDemoManifest() {
    try {
      let manifest = null;
      try {
        const res = await fetch(`${API_BASE_URL}/api/v1/demo/manifest`);
        if (res.ok) manifest = await res.json();
      } catch (e) {
        // Fallback below
      }

      if (!manifest || manifest.length === 0) {
        const res2 = await fetch(`${API_BASE_URL}/api/v1/demo/files/manifest.json`);
        if (res2.ok) manifest = await res2.json();
      }

      if (manifest && Array.isArray(manifest) && manifest.length > 0) {
        this.manifestData = manifest;
        this.populateDemoDropdown(manifest);
      } else {
        this.demoSampleSelect.innerHTML = '<option value="" disabled>Manual file upload mode</option>';
      }
    } catch (err) {
      this.demoSampleSelect.innerHTML = '<option value="" disabled>Manifest unavailable — upload files manually</option>';
    }
  }

  populateDemoDropdown(samples) {
    this.demoSampleSelect.innerHTML = '<option value="" disabled selected>Select a remote-sensing workflow sample...</option>';
    samples.forEach(sample => {
      const opt = document.createElement('option');
      opt.value = sample.id;
      const title = sample.title || sample.id;
      opt.textContent = `${title} (${sample.workflow})`;
      this.demoSampleSelect.appendChild(opt);
    });
  }

  async loadDemoSampleById(sampleId) {
    const sample = this.manifestData.find(s => s.id === sampleId);
    if (!sample) return;

    this.queryInput.value = sample.suggested_query;

    // Adjust labels based on workflow type
    const wf = sample.workflow || '';
    if (wf.includes('vqa')) {
      if (this.t1HeaderLabel) this.t1HeaderLabel.textContent = 'Observation Raster (Single Scene)';
      if (this.t2HeaderLabel) this.t2HeaderLabel.textContent = '(Not required for single-image VQA)';
    } else if (wf.includes('grounding')) {
      if (this.t1HeaderLabel) this.t1HeaderLabel.textContent = 'Target Image to Ground';
      if (this.t2HeaderLabel) this.t2HeaderLabel.textContent = '(Not required for grounding)';
    } else if (wf.includes('optical_sar')) {
      if (this.t1HeaderLabel) this.t1HeaderLabel.textContent = 'Optical Observation';
      if (this.t2HeaderLabel) this.t2HeaderLabel.textContent = 'SAR Microwave Observation';
    } else {
      if (this.t1HeaderLabel) this.t1HeaderLabel.textContent = 'Earlier Observation (T1)';
      if (this.t2HeaderLabel) this.t2HeaderLabel.textContent = 'Later Observation (T2)';
    }

    try {
      const files = sample.files || [];
      if (files.length > 0) {
        const p1Name = files[0].split('/').pop();
        const res1 = await fetch(`${API_BASE_URL}/api/v1/demo/files/${p1Name}`);
        if (res1.ok) {
          const blob1 = await res1.blob();
          const file1 = new File([blob1], p1Name, { type: blob1.type || 'image/tiff' });
          this.setFile('t1', file1);
        }
      }

      if (files.length > 1) {
        const p2Name = files[1].split('/').pop();
        const res2 = await fetch(`${API_BASE_URL}/api/v1/demo/files/${p2Name}`);
        if (res2.ok) {
          const blob2 = await res2.blob();
          const file2 = new File([blob2], p2Name, { type: blob2.type || 'image/tiff' });
          this.setFile('t2', file2);
        }
      } else {
        this.clearFile('t2');
      }

      this.updateAnalyzeButtonState();
    } catch (e) {
      console.warn('Could not auto-fetch demo sample files:', e);
    }
  }

  setupDragAndDrop(dropzone, slot) {
    ['dragenter', 'dragover'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        this.setFile(slot, files[0]);
      }
    });
  }

  handleFileSelect(e, slot) {
    if (e.target.files && e.target.files.length > 0) {
      this.setFile(slot, e.target.files[0]);
    }
  }

  setFile(slot, file) {
    const isT1 = slot === 't1';
    const filenameEl = isT1 ? this.t1Filename : this.t2Filename;
    const filesizeEl = isT1 ? this.t1Filesize : this.t2Filesize;
    const filecrsEl = isT1 ? this.t1FileCrs : this.t2FileCrs;
    const emptyView = isT1 ? this.t1EmptyView : this.t2EmptyView;
    const loadedView = isT1 ? this.t1LoadedView : this.t2LoadedView;
    const thumbEl = isT1 ? this.t1Thumb : this.t2Thumb;
    const fallbackEl = isT1 ? this.t1ThumbFallback : this.t2ThumbFallback;

    if (isT1) this.t1File = file;
    else this.t2File = file;

    filenameEl.textContent = file.name;
    filesizeEl.textContent = this.formatBytes(file.size);
    if (filecrsEl) {
      const ext = file.name.split('.').pop().toUpperCase();
      filecrsEl.textContent = ext.includes('TIF') ? 'Raster / Ready' : `${ext} Mode`;
    }

    emptyView.classList.add('hidden');
    loadedView.classList.remove('hidden');

    const isWebImg = file.type === 'image/png' || file.type === 'image/jpeg' || file.name.endsWith('.png') || file.name.endsWith('.jpg') || file.name.endsWith('.jpeg');
    if (isWebImg) {
      thumbEl.src = URL.createObjectURL(file);
      thumbEl.classList.remove('hidden');
      fallbackEl.classList.add('hidden');
    } else {
      thumbEl.classList.add('hidden');
      fallbackEl.classList.remove('hidden');
    }

    this.updateAnalyzeButtonState();
  }

  clearFile(slot) {
    if (slot === 't1') {
      this.t1File = null;
      this.t1Input.value = '';
      this.t1LoadedView.classList.add('hidden');
      this.t1EmptyView.classList.remove('hidden');
    } else {
      this.t2File = null;
      this.t2Input.value = '';
      this.t2LoadedView.classList.add('hidden');
      this.t2EmptyView.classList.remove('hidden');
    }
    this.updateAnalyzeButtonState();
  }

  updateAnalyzeButtonState() {
    const hasInput = Boolean(this.t1File || this.t2File);
    this.analyzeBtn.disabled = !hasInput || this.isAnalyzing;
  }

  setViewMode(mode) {
    this.activeViewMode = mode;
    if (mode === 'slider') {
      if (this.btnModeSlider) this.btnModeSlider.classList.add('active');
      if (this.btnModeSide) this.btnModeSide.classList.remove('active');
      if (this.sliderView) this.sliderView.classList.remove('hidden');
      if (this.sideBySideGrid) this.sideBySideGrid.classList.add('hidden');
      this.updateSliderDimensions();
    } else {
      if (this.btnModeSide) this.btnModeSide.classList.add('active');
      if (this.btnModeSlider) this.btnModeSlider.classList.remove('active');
      if (this.sideBySideGrid) this.sideBySideGrid.classList.remove('hidden');
      if (this.sliderView) this.sliderView.classList.add('hidden');
    }
  }

  setupSplitSlider() {
    if (!this.sliderDivider || !this.sliderFrame) return;

    let isDragging = false;

    const onMove = (clientX) => {
      if (!isDragging) return;
      const rect = this.sliderFrame.getBoundingClientRect();
      let pos = (clientX - rect.left) / rect.width;
      pos = Math.max(0.02, Math.min(0.98, pos));

      const pct = (pos * 100).toFixed(2) + '%';
      if (this.sliderOverlayWrapper) this.sliderOverlayWrapper.style.width = pct;
      this.sliderDivider.style.left = pct;
    };

    this.sliderDivider.addEventListener('mousedown', () => { isDragging = true; });
    this.sliderFrame.addEventListener('mousedown', (e) => {
      isDragging = true;
      onMove(e.clientX);
    });

    window.addEventListener('mousemove', (e) => onMove(e.clientX));
    window.addEventListener('mouseup', () => { isDragging = false; });

    // Touch support for tablets/trackpads
    this.sliderDivider.addEventListener('touchstart', () => { isDragging = true; }, { passive: true });
    this.sliderFrame.addEventListener('touchstart', (e) => {
      isDragging = true;
      if (e.touches.length > 0) onMove(e.touches[0].clientX);
    }, { passive: true });
    window.addEventListener('touchmove', (e) => {
      if (e.touches.length > 0) onMove(e.touches[0].clientX);
    }, { passive: true });
    window.addEventListener('touchend', () => { isDragging = false; });
  }

  updateSliderDimensions() {
    if (!this.sliderFrame || !this.sliderImgAfter) return;
    const frameWidth = this.sliderFrame.offsetWidth || 800;
    this.sliderImgAfter.style.width = frameWidth + 'px';
  }

  async executeAnalysis() {
    if (!this.t1File && !this.t2File) {
      this.showError('Missing Image', 'Please upload at least one remote-sensing observation.');
      return;
    }

    this.isAnalyzing = true;
    this.updateAnalyzeButtonState();
    this.errorBanner.classList.add('hidden');
    this.progressContainer.classList.remove('hidden');
    this.resultsSection.classList.add('hidden');

    // Reset Visual Stepper
    this.stepperItems.forEach(item => item.classList.remove('completed'));
    this.progressHeadline.textContent = 'Executing Agentic Remote Sensing Analysis...';
    this.progressSubtext.textContent = 'AgentRouter inspecting query intent, input modalities, and specialist models...';

    const formData = new FormData();
    formData.append('query', this.queryInput.value || 'What changed between these two dates?');
    if (this.t1File) {
      formData.append('image_primary', this.t1File);
    }
    if (this.t2File) {
      formData.append('image_secondary', this.t2File);
    }
    const candidate = this.candidateSelect ? this.candidateSelect.value : 'auto';
    if (candidate && candidate !== 'auto') {
      formData.append('task_hint', candidate);
    }

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/analyze`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        const errorDetail = data.detail || 'Analysis request failed.';
        throw new Error(this.sanitizePath(errorDetail));
      }

      this.currentResult = data;
      this.renderResults(data);
    } catch (err) {
      this.showError('Analysis Failed', this.sanitizePath(err.message));
    } finally {
      this.isAnalyzing = false;
      this.updateAnalyzeButtonState();
      this.progressContainer.classList.add('hidden');
    }
  }

  renderResults(contract) {
    this.resultsSection.classList.remove('hidden');

    // 0. Task Routing Tag
    if (this.summaryTask) {
      const taskFormatted = (contract.task || 'remote_sensing_analysis').toUpperCase().replace(/_/g, ' ');
      this.summaryTask.textContent = `Auto-Agent Routed: ${taskFormatted}`;
    }

    // 1. Answer Narrative & System Confidence
    this.summaryAnswer.textContent = this.sanitizePath(contract.answer || 'Analysis completed.');
    const confVal = contract.confidence !== null && contract.confidence !== undefined 
      ? (contract.confidence * 100).toFixed(1) + '%' 
      : 'N/A';
    const confLevel = contract.confidence_level || 'EVALUATED';
    this.summaryConfidence.textContent = `System Confidence: ${confLevel} (${confVal})`;

    // 1b. Defensible System Confidence Breakdown (M9)
    this.renderConfidenceDetails(contract);

    // 1c. Semantic Change Reasoning (M5)
    this.renderSemanticInterpretation(contract.evidence?.semantic_interpretation);

    // 1d. Evidence Reliability & Consistency Status (M8)
    this.renderEvidenceStatus(contract);

    // 1e. Adaptive Visual Hub (Bi-Temporal, Grounding, Single, Optical+SAR)
    this.renderAdaptiveVisualHub(contract);

    // 2. Quantitative Telemetry Metrics
    const params = contract.parameters || {};
    const changedPixels = params.changed_pixels !== undefined ? params.changed_pixels.toLocaleString() + ' px' : '--';
    this.valChangedPixels.textContent = changedPixels;

    // Physical Metric Area (Strict separation of verified metric vs pixel space)
    const statAreaM2 = contract.evidence?.statistics?.find(s => s.metric_name === 'changed_area_m2');
    const statAreaHa = contract.evidence?.statistics?.find(s => s.metric_name === 'changed_area_hectares');
    if (statAreaM2 && statAreaHa && statAreaM2.value !== null && statAreaHa.value !== null) {
      this.valPhysicalArea.textContent = `${statAreaM2.value.toLocaleString()} m² (${statAreaHa.value.toFixed(2)} ha)`;
      if (this.subPhysicalArea) this.subPhysicalArea.textContent = 'Geospatial metric area verified (Projected CRS)';
    } else {
      this.valPhysicalArea.textContent = 'Pixel space only';
      if (this.subPhysicalArea) this.subPhysicalArea.textContent = 'Metric area unmeasured from spatial evidence';
    }

    // Ratio & Spatial Clusters
    const changeRatio = params.change_ratio_pct !== undefined ? params.change_ratio_pct.toFixed(2) + '%' : '--';
    this.valChangeRatio.textContent = changeRatio;

    const clusters = params.total_clusters !== undefined ? params.total_clusters : (contract.evidence?.boxes?.length || 0);
    this.valTotalClusters.textContent = clusters;

    // Specialist Execution & Latency
    let specialistName = 'Auto Orchestrator';
    if (contract.models && contract.models.length > 0) {
      specialistName = contract.models.map(m => m.model_name || m.identifier || m).join(' → ');
    } else if (params.candidate_model) {
      specialistName = params.candidate_model;
    }
    this.valSpecialist.textContent = specialistName;

    const latency = contract.execution_time_ms !== undefined ? `${contract.execution_time_ms} ms` : '--';
    this.valLatency.textContent = latency;

    // 3. Geospatial Bounding Box Telemetry
    this.renderTelemetryBoxes(contract.evidence?.boxes || []);

    // 4. Auditable Execution Trace
    this.renderTrace(contract.execution_trace || []);

    // 5. Warnings & Advisories
    this.renderWarnings(contract.warnings || []);

    // 6. Developer / Debug Raw JSON
    if (this.debugJsonViewer) {
      this.debugJsonViewer.textContent = JSON.stringify(contract, null, 2);
    }

    // Scroll smoothly to results
    this.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  renderAdaptiveVisualHub(contract) {
    const task = (contract.task || '').toLowerCase();
    const isOpticalSar = task === 'optical_sar_analysis' || task === 'optical_sar';
    const isGrounding = task.includes('grounding');
    const isSingleVqa = task.includes('vqa') && !task.includes('bitemporal') && !task.includes('change');
    const isBiTemporal = task.includes('bitemporal') || task.includes('change');

    // Reset visibility of visual containers
    if (this.opticalSarPreviewCard) this.opticalSarPreviewCard.classList.add('hidden');
    if (this.complementarityCard) this.complementarityCard.classList.add('hidden');
    if (this.singleViewContainer) this.singleViewContainer.classList.add('hidden');
    if (this.sliderView) this.sliderView.classList.add('hidden');
    if (this.sideBySideGrid) this.sideBySideGrid.classList.add('hidden');
    if (this.bitemporalModeSwitch) this.bitemporalModeSwitch.classList.add('hidden');
    if (this.visualHubCard) this.visualHubCard.classList.remove('hidden');

    if (isOpticalSar) {
      // -------------------------------------------------------------
      // OPTICAL + SAR CROSS-MODAL FUSION
      // -------------------------------------------------------------
      if (this.visualHubCard) this.visualHubCard.classList.add('hidden');
      if (this.semanticCard) this.semanticCard.classList.add('hidden');
      if (this.opticalSarPreviewCard) {
        this.opticalSarPreviewCard.classList.remove('hidden');
        const compImg = contract.evidence?.images?.find(img => img.role === 'semantic_composite');
        if (compImg && this.opticalSarCompositeImg) {
          this.opticalSarCompositeImg.src = compImg.url;
        }
      }
      if (contract.evidence?.complementarity_report && this.complementarityCard) {
        this.complementarityCard.classList.remove('hidden');
        const cr = contract.evidence.complementarity_report;
        if (this.compOpticalText) this.compOpticalText.textContent = cr.optical_limitations || 'Visible spectral reflectance context.';
        if (this.compSarText) this.compSarText.textContent = cr.sar_penetration || 'Microwave dielectric backscatter and structural contrast.';
        if (this.compJointText) this.compJointText.textContent = cr.structural_contrast || 'Joint fused multimodal land-cover finding.';
      }
    } else if (isGrounding) {
      // -------------------------------------------------------------
      // TEXT-GUIDED REGION GROUNDING
      // -------------------------------------------------------------
      if (this.semanticCard) this.semanticCard.classList.add('hidden');
      if (this.hubMainTitle) this.hubMainTitle.textContent = 'Text-Guided Spatial Grounding Preview';
      if (this.hubMainSubtitle) this.hubMainSubtitle.textContent = 'Zero-shot bounding box spatial localization grounded on query';
      if (this.singleViewContainer) {
        this.singleViewContainer.classList.remove('hidden');
        const groundImg = contract.evidence?.images?.find(img => img.role === 'grounding_preview');
        const primaryImg = contract.evidence?.images?.find(img => img.role === 'primary');
        const fallbackUrl = this.t1File ? URL.createObjectURL(this.t1File) : '';
        this.singlePreviewImg.src = groundImg ? groundImg.url : (primaryImg ? primaryImg.url : fallbackUrl);
        if (this.singlePreviewTag) this.singlePreviewTag.textContent = 'GROUNDED REGION OVERLAY';
        if (this.singleCaption) this.singleCaption.textContent = 'Visual bounding box detections projected from natural-language query.';
      }
    } else if (isSingleVqa) {
      // -------------------------------------------------------------
      // SINGLE-IMAGE VQA / SCENE DESCRIPTION
      // -------------------------------------------------------------
      if (this.semanticCard) this.semanticCard.classList.add('hidden');
      if (this.hubMainTitle) this.hubMainTitle.textContent = 'Remote Sensing Scene Inspection';
      if (this.hubMainSubtitle) this.hubMainSubtitle.textContent = 'Single-observation vision-language question answering';
      if (this.singleViewContainer) {
        this.singleViewContainer.classList.remove('hidden');
        const primaryImg = contract.evidence?.images?.find(img => img.role === 'primary');
        const fallbackUrl = this.t1File ? URL.createObjectURL(this.t1File) : '';
        this.singlePreviewImg.src = primaryImg ? primaryImg.url : fallbackUrl;
        if (this.singlePreviewTag) this.singlePreviewTag.textContent = 'PRIMARY SCENE OBSERVATION';
        if (this.singleCaption) this.singleCaption.textContent = 'Remote sensing multispectral/optical observation inspected.';
      }
    } else {
      // -------------------------------------------------------------
      // BI-TEMPORAL CHANGE DETECTION & VQA
      // -------------------------------------------------------------
      if (this.bitemporalModeSwitch) this.bitemporalModeSwitch.classList.remove('hidden');
      if (this.hubMainTitle) this.hubMainTitle.textContent = 'Bi-Temporal Spatial Evidence Inspection';
      if (this.hubMainSubtitle) this.hubMainSubtitle.textContent = 'High-resolution multi-temporal comparison with verified change overlay';

      const maskObj = contract.evidence?.masks && contract.evidence.masks.length > 0 ? contract.evidence.masks[0] : null;
      const overlayImg = contract.evidence?.images?.find(img => img.role === 'change_overlay');
      const primaryImg = contract.evidence?.images?.find(img => img.role === 'primary');
      const secondaryImg = contract.evidence?.images?.find(img => img.role === 'secondary');

      const isT1Web = this.t1File && (this.t1File.type === 'image/png' || this.t1File.type === 'image/jpeg');
      const isT2Web = this.t2File && (this.t2File.type === 'image/png' || this.t2File.type === 'image/jpeg');
      const t1Url = isT1Web ? URL.createObjectURL(this.t1File) : (primaryImg ? primaryImg.url : '');
      const t2Url = isT2Web ? URL.createObjectURL(this.t2File) : (secondaryImg ? secondaryImg.url : '');
      const overlayUrl = overlayImg ? overlayImg.url : t2Url;
      const maskUrl = maskObj ? maskObj.url : overlayUrl;

      if (this.sliderImgBefore) this.sliderImgBefore.src = t1Url;
      if (this.sliderImgAfter) this.sliderImgAfter.src = overlayUrl || t2Url;

      if (this.galleryT1) this.galleryT1.src = t1Url;
      if (this.galleryT2) this.galleryT2.src = t2Url;
      if (this.galleryMask) this.galleryMask.src = maskUrl;
      if (this.galleryOverlay) this.galleryOverlay.src = overlayUrl;

      this.setViewMode(this.activeViewMode);
      setTimeout(() => this.updateSliderDimensions(), 120);
    }
  }

  renderSemanticInterpretation(semanticInterp) {
    if (!semanticInterp) {
      if (this.semanticCard) this.semanticCard.classList.add('hidden');
      return;
    }

    if (this.semanticCard) this.semanticCard.classList.remove('hidden');

    const dir = (semanticInterp.temporal_direction || 'modified').toLowerCase();
    if (this.semanticDirectionBadge) {
      this.semanticDirectionBadge.textContent = dir.toUpperCase();
      this.semanticDirectionBadge.className = `direction-badge direction-${dir}`;
    }

    if (this.predomTransitionBadge) {
      this.predomTransitionBadge.textContent = semanticInterp.predominant_transition || 'Unspecified';
    }

    const uncPct = Math.round((semanticInterp.semantic_uncertainty || 0) * 100);
    if (this.semanticUncertaintyBadge) {
      this.semanticUncertaintyBadge.textContent = `Model Uncertainty: ${uncPct}%`;
    }

    if (this.semanticTransitionsContainer) {
      this.semanticTransitionsContainer.innerHTML = '';
      const transitions = semanticInterp.transitions || [];

      if (transitions.length === 0) {
        const emptyRow = document.createElement('div');
        emptyRow.className = 'transition-row';
        emptyRow.innerHTML = '<span style="color: var(--text-muted); font-size: 12px;">No discrete semantic transitions observed (zero detected surface change).</span>';
        this.semanticTransitionsContainer.appendChild(emptyRow);
        return;
      }

      transitions.forEach((t) => {
        const row = document.createElement('div');
        row.className = 'transition-row';

        const left = document.createElement('div');
        left.className = 'transition-left';

        const classRow = document.createElement('div');
        classRow.className = 'transition-classes';
        classRow.innerHTML = `<span>${t.from_class}</span><span class="transition-arrow-symbol">&rarr;</span><span>${t.to_class}</span>`;

        const desc = document.createElement('span');
        desc.className = 'transition-desc';
        desc.textContent = t.description;

        left.appendChild(classRow);
        left.appendChild(desc);

        const right = document.createElement('div');
        right.className = 'transition-right';

        if (t.region_id) {
          const regTag = document.createElement('span');
          regTag.className = 'transition-region-tag';
          regTag.textContent = t.region_id;
          right.appendChild(regTag);
        }

        if (t.is_uncertain) {
          const uncTag = document.createElement('span');
          uncTag.className = 'transition-uncertain-tag';
          uncTag.textContent = 'Uncertain Class';
          right.appendChild(uncTag);
        }

        const provTag = document.createElement('span');
        provTag.className = 'transition-prov-tag';
        provTag.textContent = t.evidence_support || 'visual_crop_comparison';
        right.appendChild(provTag);

        row.appendChild(left);
        row.appendChild(right);
        this.semanticTransitionsContainer.appendChild(row);
      });
    }
  }

  renderTelemetryBoxes(boxes) {
    if (!this.telemetryRegionsList) return;
    this.telemetryRegionsList.innerHTML = '';
    if (boxes.length === 0) {
      this.telemetryRegionsList.innerHTML = '<div class="cluster-row"><span>No discrete spatial change bounding boxes generated (zero detected surface change).</span></div>';
      return;
    }

    boxes.forEach((box, idx) => {
      const row = document.createElement('div');
      row.className = 'cluster-row';
      const coords = box.coordinates_pixel ? `[${box.coordinates_pixel.join(', ')}] px` : 'N/A';
      row.innerHTML = `
        <span class="cluster-id">${box.label || `Region ${idx + 1}`}</span>
        <span class="cluster-bbox">BBox: ${coords}</span>
        <span class="cluster-area">Conf: ${(box.confidence * 100).toFixed(1)}%</span>
      `;
      this.telemetryRegionsList.appendChild(row);
    });
  }

  renderTrace(traceSteps) {
    if (!this.traceTimeline) return;
    this.traceTimeline.innerHTML = '';
    if (this.traceStepsCount) this.traceStepsCount.textContent = `${traceSteps.length} Steps Recorded`;

    traceSteps.forEach(step => {
      // Highlight matching visual stepper item
      const stepItem = document.querySelector(`.stepper-item[data-step="${step.step_name}"]`);
      if (stepItem) stepItem.classList.add('completed');

      const item = document.createElement('div');
      item.className = 'trace-step-item';
      const statusClass = step.status === 'completed' ? 'badge-completed' : (step.status === 'warning' ? 'badge-warning' : 'badge-failed');

      item.innerHTML = `
        <span class="trace-badge ${statusClass}">${step.status}</span>
        <div class="trace-step-info">
          <div class="trace-step-name-row">
            <span class="trace-step-name">${step.step_number}. ${step.step_name}</span>
            <span class="trace-step-duration">${step.duration_ms} ms</span>
          </div>
          <span class="trace-step-details">${this.sanitizePath(step.details || '')}</span>
        </div>
      `;
      this.traceTimeline.appendChild(item);
    });
  }

  renderEvidenceStatus(contract) {
    if (!this.evidenceStatusCard) return;

    const report = contract.evidence?.consistency_report;
    const status = contract.evidence_status || report?.status || 'CONSISTENT';

    this.consistencyStatusBadge.textContent = status;
    this.consistencyStatusBadge.className = 'consistency-status-badge';
    if (status === 'CONSISTENT') {
      this.consistencyStatusBadge.classList.add('status-consistent');
    } else if (status === 'PARTIALLY_CONSISTENT') {
      this.consistencyStatusBadge.classList.add('status-partially-consistent');
    } else if (status === 'UNCERTAIN') {
      this.consistencyStatusBadge.classList.add('status-uncertain');
    } else if (status === 'CONTRADICTORY') {
      this.consistencyStatusBadge.classList.add('status-contradictory');
    } else if (status === 'INSUFFICIENT_EVIDENCE') {
      this.consistencyStatusBadge.classList.add('status-insufficient-evidence');
    }

    const quality = report?.evidence_quality_score !== undefined
      ? Math.round(report.evidence_quality_score * 100) + '%'
      : '100%';
    this.evidenceQualityBadge.textContent = `Quality: ${quality}`;

    const itemCount = contract.evidence?.fused_items?.length || 0;
    this.provenanceCountBadge.textContent = `${itemCount} Evidence Units`;

    this.consistencyNarrative.textContent = this.sanitizePath(report?.summary_narrative || 'Evidence is consistent across specialists.');

    const conflicts = report?.conflicts || [];
    if (conflicts.length > 0) {
      this.conflictsContainer.classList.remove('hidden');
      this.conflictsList.innerHTML = conflicts.map(c => `
        <li><strong>[${c.rule_violated}]</strong> ${this.sanitizePath(c.description)} (Sources: ${c.conflicting_sources.join(', ')})</li>
      `).join('');
    } else {
      this.conflictsContainer.classList.add('hidden');
      this.conflictsList.innerHTML = '';
    }

    const fusedItems = contract.evidence?.fused_items || [];
    if (fusedItems.length > 0) {
      this.evidencePillsRow.innerHTML = fusedItems.map(item => `
        <div class="evidence-pill" title="${this.sanitizePath(item.claim || '')}">
          <span class="evidence-pill-spec">${item.source_specialist}</span>: ${item.evidence_type}
        </div>
      `).join('');
    } else {
      this.evidencePillsRow.innerHTML = '<span style="font-size: 11px; color: var(--text-muted);">Standard evidence records indexed.</span>';
    }
  }

  renderConfidenceDetails(contract) {
    if (!this.confidenceCard) return;

    const breakdown = contract.confidence_breakdown || {};
    const overallConf = contract.confidence !== null && contract.confidence !== undefined
      ? contract.confidence
      : (breakdown.overall_confidence || 0.0);
    const level = contract.confidence_level || breakdown.confidence_level || 'MEDIUM';

    this.confLevelBadge.textContent = level;
    this.confLevelBadge.className = 'confidence-level-badge';
    if (level === 'HIGH') {
      this.confLevelBadge.classList.add('level-high');
    } else if (level === 'MEDIUM') {
      this.confLevelBadge.classList.add('level-medium');
    } else if (level === 'LOW') {
      this.confLevelBadge.classList.add('level-low');
    } else if (level === 'UNSUPPORTED') {
      this.confLevelBadge.classList.add('level-unsupported');
    }

    this.confOverallScore.textContent = `${(overallConf * 100).toFixed(1)}%`;

    const isCalibrated = breakdown.calculation_details?.is_calibrated_probability === true;
    this.confCalibrationBadge.textContent = isCalibrated 
      ? 'Calibrated Statistical Probability' 
      : 'Heuristic Multi-Factor Grounding';

    const specScore = breakdown.specialist_confidence !== undefined 
      ? (breakdown.specialist_confidence * 100).toFixed(1) + '%' 
      : '--';
    const evidScore = breakdown.evidence_quality_score !== undefined 
      ? (breakdown.evidence_quality_score * 100).toFixed(1) + '%' 
      : '--';
    const alignScore = breakdown.spatial_alignment_score !== undefined 
      ? (breakdown.spatial_alignment_score * 100).toFixed(1) + '%' 
      : '--';
    const inputScore = breakdown.input_quality_score !== undefined 
      ? (breakdown.input_quality_score * 100).toFixed(1) + '%' 
      : '--';
    const consPenalty = breakdown.consistency_penalty !== undefined 
      ? (breakdown.consistency_penalty > 0 ? `-${breakdown.consistency_penalty.toFixed(2)}` : '0.00') 
      : '0.00';

    this.subfactorSpecialist.textContent = specScore;
    this.subfactorEvidence.textContent = evidScore;
    this.subfactorAlignment.textContent = alignScore;
    this.subfactorInput.textContent = inputScore;
    this.subfactorConsistency.textContent = consPenalty;
    if (breakdown.consistency_penalty > 0) {
      this.subfactorConsistency.style.color = 'var(--crimson-danger)';
    } else {
      this.subfactorConsistency.style.color = 'var(--emerald-success)';
    }

    const factors = breakdown.confidence_factors || [];
    if (factors.length > 0) {
      this.confFactorsList.innerHTML = factors.map(f => `
        <span class="factor-pill">✓ ${f.replace(/_/g, ' ')}</span>
      `).join('');
    } else {
      this.confFactorsList.innerHTML = '<span style="font-size: 11px; color: var(--text-muted);">Standard telemetry signals evaluated.</span>';
    }

    const warnings = breakdown.confidence_warnings || [];
    if (warnings.length > 0) {
      this.confWarningsList.classList.remove('hidden');
      this.confWarningsList.innerHTML = warnings.map(w => `
        <span class="warning-pill">⚠️ ${this.sanitizePath(w)}</span>
      `).join('');
    } else {
      this.confWarningsList.classList.add('hidden');
      this.confWarningsList.innerHTML = '';
    }
  }

  renderWarnings(warnings) {
    if (!this.warningsCard || !this.warningsList) return;
    if (!warnings || warnings.length === 0) {
      this.warningsCard.classList.add('hidden');
      return;
    }
    this.warningsCard.classList.remove('hidden');
    this.warningsList.innerHTML = '';
    warnings.forEach(warn => {
      const li = document.createElement('li');
      li.textContent = this.sanitizePath(warn);
      this.warningsList.appendChild(li);
    });
  }

  // -----------------------------------------------------------------
  // M11 Structured Report Downloads & Preview
  // -----------------------------------------------------------------
  downloadReportJson() {
    if (!this.currentResult?.report) {
      this.showError('Report Unavailable', 'No structured report artifact is present in the response.');
      return;
    }
    const report = this.currentResult.report;
    const repId = report.metadata?.report_id || report.report_id || 'satquery_export';
    const jsonStr = JSON.stringify(report, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `satquery_report_${repId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  downloadReportMarkdown() {
    if (!this.currentResult?.report) {
      this.showError('Report Unavailable', 'No structured report artifact is present in the response.');
      return;
    }
    const md = this.formatReportMarkdown(this.currentResult.report);
    const repId = this.currentResult.report.metadata?.report_id || 'satquery_export';
    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `satquery_report_${repId}.md`;
    a.click();
    URL.revokeObjectURL(url);
  }

  toggleReportPreview() {
    if (!this.reportPreviewBox || !this.reportPreviewText) return;
    const isHidden = this.reportPreviewBox.classList.toggle('hidden');
    if (!isHidden) {
      if (this.currentResult?.report) {
        this.reportPreviewText.textContent = this.formatReportMarkdown(this.currentResult.report);
      } else {
        this.reportPreviewText.textContent = 'No report loaded. Execute an analysis to preview the executive briefing.';
      }
    }
  }

  formatReportMarkdown(report) {
    const meta = report.metadata || {};
    const inp = report.input_summary || {};
    const conf = report.confidence || {};
    const cons = report.consistency || {};
    const stats = report.statistics || {};

    const lines = [
      '# SatQuery AI — Executive Analysis Report',
      `**Mission:** ${meta.mission || 'SIH26167'}  `,
      `**Authority:** ${meta.organization || 'Indian Space Research Organisation (ISRO)'}  `,
      `**Report ID:** \`${meta.report_id || 'N/A'}\` | **Generated:** ${meta.timestamp || new Date().toISOString()} | **Runtime:** ${meta.execution_time_ms || 0} ms  `,
      `**Engine Version:** v${meta.app_version || '0.3.0'}`,
      '',
      '---',
      '',
      '## 1. Executive Summary',
      `- **Task:** \`${report.task || 'remote_sensing_analysis'}\``,
      `- **Query:** *"${inp.query || ''}"*`,
      '- **Primary Finding:**',
      `  > ${this.sanitizePath(report.answer || 'Analysis completed.')}`,
      '',
      '## 2. System Confidence & Consistency Audit',
      `- **System Confidence:** **${conf.level || 'EVALUATED'} (${((conf.score || 0) * 100).toFixed(1)}%)** *(Heuristic multi-factor score; not uncalibrated probability)*`,
      `- **Consistency Status:** \`${cons.status || 'CONSISTENT'}\` (Gating action: \`${cons.gating_action || 'allow'}\`)`,
      `- **Consistency Narrative:** ${this.sanitizePath(cons.summary_narrative || 'Evidence is consistent.')}`,
      '',
      '## 3. Quantitative Measurements',
      `- **Changed Pixels:** ${stats.changed_pixels !== undefined ? stats.changed_pixels.toLocaleString() + ' px' : 'N/A'}`,
      `- **Change Ratio:** ${stats.change_ratio_pct !== undefined ? stats.change_ratio_pct.toFixed(2) + '%' : 'N/A'}`,
      `- **Connected Clusters:** ${stats.total_clusters !== undefined ? stats.total_clusters : 'N/A'}`,
      `- **Physical Area (Metric):** ${stats.changed_area_hectares ? `${stats.changed_area_hectares.toFixed(2)} ha (${stats.changed_area_m2?.toLocaleString()} m²)` : 'Pixel space only (Metric area unmeasured)'}`,
      `- **Semantic Class Limitation:** ${stats.semantic_area_status || 'unmeasured_from_spatial_evidence'}`,
      '',
      '## 4. Auditable Execution Trace',
    ];

    const trace = report.audit_trail || [];
    if (trace.length > 0) {
      trace.forEach(step => {
        lines.push(`- **Step ${step.step_number} (${step.step_name}):** \`${step.status}\` in ${step.duration_ms} ms — ${this.sanitizePath(step.details || '')}`);
      });
    } else {
      lines.push('- No trace steps recorded.');
    }

    return lines.join('\n');
  }

  showError(title, detail) {
    this.errorTitle.textContent = title;
    this.errorDetail.textContent = detail;
    this.errorBanner.classList.remove('hidden');
    this.errorBanner.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  sanitizePath(str) {
    if (!str || typeof str !== 'string') return str;
    // Replace local server paths like C:\Users\... or /home/... with base filenames
    return str.replace(/[A-Za-z]:\\[^"'\s\n\r]+\\([A-Za-z0-9_.-]+)/g, '$1')
              .replace(/\/home\/[^"'\s\n\r]+\/([A-Za-z0-9_.-]+)/g, '$1');
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
  window.satQueryClient = new SatQueryClient();
});
