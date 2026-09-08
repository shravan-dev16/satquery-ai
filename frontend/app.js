/**
 * SatQuery AI — Bi-Temporal Change Analysis Web Client
 * 
 * Interacts with real backend endpoints:
 * - GET /api/v1/health (readiness check)
 * - POST /api/v1/validate (pre-flight validation)
 * - POST /api/v1/analyze (9-step change detection pipeline)
 */

class SatQueryClient {
  constructor() {
    this.t1File = null;
    this.t2File = null;
    this.currentResult = null;
    this.isAnalyzing = false;
    this.activeViewMode = 'slider'; // 'slider' | 'side'

    this.initElements();
    this.bindEvents();
    this.checkBackendHealth();
  }

  initElements() {
    // Header
    this.statusDot = document.getElementById('backend-status-dot');
    this.statusText = document.getElementById('backend-status-text');

    // T1 Elements
    this.t1Dropzone = document.getElementById('t1-dropzone');
    this.t1Input = document.getElementById('t1-file-input');
    this.t1EmptyView = document.getElementById('t1-empty-view');
    this.t1LoadedView = document.getElementById('t1-loaded-view');
    this.t1Thumb = document.getElementById('t1-thumb');
    this.t1ThumbFallback = document.getElementById('t1-thumb-fallback');
    this.t1Filename = document.getElementById('t1-filename');
    this.t1Filesize = document.getElementById('t1-filesize');
    this.t1RemoveBtn = document.getElementById('t1-remove-btn');

    // T2 Elements
    this.t2Dropzone = document.getElementById('t2-dropzone');
    this.t2Input = document.getElementById('t2-file-input');
    this.t2EmptyView = document.getElementById('t2-empty-view');
    this.t2LoadedView = document.getElementById('t2-loaded-view');
    this.t2Thumb = document.getElementById('t2-thumb');
    this.t2ThumbFallback = document.getElementById('t2-thumb-fallback');
    this.t2Filename = document.getElementById('t2-filename');
    this.t2Filesize = document.getElementById('t2-filesize');
    this.t2RemoveBtn = document.getElementById('t2-remove-btn');

    // Query & Action
    this.queryInput = document.getElementById('query-input');
    this.candidateSelect = document.getElementById('candidate-select');
    this.analyzeBtn = document.getElementById('analyze-btn');
    this.queryPills = document.querySelectorAll('.query-pill');

    // Progress & Stepper
    this.progressContainer = document.getElementById('progress-container');
    this.progressHeadline = document.getElementById('progress-headline');
    this.progressSubtext = document.getElementById('progress-subtext');
    this.stepperItems = document.querySelectorAll('.stepper-item');

    // Error Banner
    this.errorBanner = document.getElementById('error-banner');
    this.errorTitle = document.getElementById('error-title');
    this.errorDetail = document.getElementById('error-detail');
    this.errorCloseBtn = document.getElementById('error-close-btn');

    // Results
    this.resultsSection = document.getElementById('results-section');
    this.summaryAnswer = document.getElementById('summary-answer');
    this.summaryConfidence = document.getElementById('summary-confidence');
    this.summaryTask = document.getElementById('summary-task');

    // Metrics
    this.valChangedPixels = document.getElementById('val-changed-pixels');
    this.valPhysicalArea = document.getElementById('val-physical-area');
    this.valChangeRatio = document.getElementById('val-change-ratio');
    this.valTotalClusters = document.getElementById('val-total-clusters');
    this.valSpecialist = document.getElementById('val-specialist');
    this.valLatency = document.getElementById('val-latency');

    // Visual Views
    this.btnModeSlider = document.getElementById('btn-mode-slider');
    this.btnModeSide = document.getElementById('btn-mode-side');
    this.sliderView = document.getElementById('slider-view');
    this.sideBySideGrid = document.getElementById('side-by-side-grid');

    // Split Slider Frame
    this.sliderFrame = document.getElementById('split-slider-frame');
    this.sliderImgBefore = document.getElementById('slider-img-before');
    this.sliderImgAfter = document.getElementById('slider-img-after');
    this.sliderOverlayWrapper = document.getElementById('slider-overlay-wrapper');
    this.sliderDivider = document.getElementById('slider-divider');

    // Gallery Images
    this.galleryT1 = document.getElementById('gallery-t1');
    this.galleryT2 = document.getElementById('gallery-t2');
    this.galleryMask = document.getElementById('gallery-mask');
    this.galleryOverlay = document.getElementById('gallery-overlay');

    // Semantic Card (M5)
    this.semanticCard = document.getElementById('semantic-card');
    this.semanticDirectionBadge = document.getElementById('semantic-direction-badge');
    this.predomTransitionBadge = document.getElementById('predom-transition-badge');
    this.semanticUncertaintyBadge = document.getElementById('semantic-uncertainty-badge');
    this.semanticTransitionsContainer = document.getElementById('semantic-transitions-container');

    // Cross-Modal Card & Preview (M6)
    this.complementarityCard = document.getElementById('complementarity-card');
    this.compOpticalText = document.getElementById('comp-optical-text');
    this.compSarText = document.getElementById('comp-sar-text');
    this.compJointText = document.getElementById('comp-joint-text');
    this.opticalSarPreviewCard = document.getElementById('optical-sar-preview-card');
    this.opticalSarCompositeImg = document.getElementById('optical-sar-composite-img');
    this.visualHubCard = document.querySelector('.visual-hub-card');

    // Telemetry & Trace
    this.telemetryRegionsList = document.getElementById('telemetry-regions-list');
    this.traceToggleBtn = document.getElementById('trace-toggle-btn');
    this.traceContent = document.getElementById('trace-content');
    this.traceArrow = document.getElementById('accordion-arrow');
    this.traceTimeline = document.getElementById('trace-timeline-container');
    this.traceStepsCount = document.getElementById('trace-steps-count');

    // Warnings
    this.warningsCard = document.getElementById('warnings-card');
    this.warningsList = document.getElementById('warnings-list');
  }

  bindEvents() {
    // T1 Events
    this.t1Input.addEventListener('change', (e) => this.handleFileSelect(e, 't1'));
    this.setupDragAndDrop(this.t1Dropzone, 't1');
    this.t1RemoveBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.clearFile('t1');
    });

    // T2 Events
    this.t2Input.addEventListener('change', (e) => this.handleFileSelect(e, 't2'));
    this.setupDragAndDrop(this.t2Dropzone, 't2');
    this.t2RemoveBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.clearFile('t2');
    });

    // Query Pills
    this.queryPills.forEach(pill => {
      pill.addEventListener('click', () => {
        this.queryInput.value = pill.dataset.query;
      });
    });

    // Analyze Action
    this.analyzeBtn.addEventListener('click', () => this.executeAnalysis());

    // Error Close
    this.errorCloseBtn.addEventListener('click', () => {
      this.errorBanner.classList.add('hidden');
    });

    // View Mode Toggle
    this.btnModeSlider.addEventListener('click', () => this.setViewMode('slider'));
    this.btnModeSide.addEventListener('click', () => this.setViewMode('side'));

    // Split Slider Interactive Drag
    this.setupSplitSlider();

    // Trace Accordion Toggle
    this.traceToggleBtn.addEventListener('click', () => {
      const isHidden = this.traceContent.classList.toggle('hidden');
      this.traceArrow.classList.toggle('open', !isHidden);
    });
  }

  async checkBackendHealth() {
    try {
      const res = await fetch('/api/v1/health');
      if (res.ok) {
        const data = await res.json();
        this.statusDot.className = 'status-dot dot-online';
        this.statusText.textContent = `Online • ${data.registered_specialists} Specialists`;
      } else {
        throw new Error(`HTTP ${res.status}`);
      }
    } catch (err) {
      this.statusDot.className = 'status-dot dot-offline';
      this.statusText.textContent = 'Backend Offline';
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
    if (slot === 't1') {
      this.t1File = file;
      this.t1Filename.textContent = file.name;
      this.t1Filesize.textContent = this.formatBytes(file.size);
      this.t1EmptyView.classList.add('hidden');
      this.t1LoadedView.classList.remove('hidden');

      const isWebImg = file.type === 'image/png' || file.type === 'image/jpeg';
      if (isWebImg) {
        this.t1Thumb.src = URL.createObjectURL(file);
        this.t1Thumb.classList.remove('hidden');
        this.t1ThumbFallback.classList.add('hidden');
      } else {
        this.t1Thumb.classList.add('hidden');
        this.t1ThumbFallback.classList.remove('hidden');
      }
    } else {
      this.t2File = file;
      this.t2Filename.textContent = file.name;
      this.t2Filesize.textContent = this.formatBytes(file.size);
      this.t2EmptyView.classList.add('hidden');
      this.t2LoadedView.classList.remove('hidden');

      const isWebImg = file.type === 'image/png' || file.type === 'image/jpeg';
      if (isWebImg) {
        this.t2Thumb.src = URL.createObjectURL(file);
        this.t2Thumb.classList.remove('hidden');
        this.t2ThumbFallback.classList.add('hidden');
      } else {
        this.t2Thumb.classList.add('hidden');
        this.t2ThumbFallback.classList.remove('hidden');
      }
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
      this.btnModeSlider.classList.add('active');
      this.btnModeSide.classList.remove('active');
      this.sliderView.classList.remove('hidden');
      this.sideBySideGrid.classList.add('hidden');
      this.updateSliderDimensions();
    } else {
      this.btnModeSide.classList.add('active');
      this.btnModeSlider.classList.remove('active');
      this.sideBySideGrid.classList.remove('hidden');
      this.sliderView.classList.add('hidden');
    }
  }

  setupSplitSlider() {
    let isDragging = false;

    const onMove = (clientX) => {
      if (!isDragging) return;
      const rect = this.sliderFrame.getBoundingClientRect();
      let pos = (clientX - rect.left) / rect.width;
      pos = Math.max(0.02, Math.min(0.98, pos));

      const pct = (pos * 100).toFixed(2) + '%';
      this.sliderOverlayWrapper.style.width = pct;
      this.sliderDivider.style.left = pct;
    };

    this.sliderDivider.addEventListener('mousedown', () => { isDragging = true; });
    this.sliderFrame.addEventListener('mousedown', (e) => {
      isDragging = true;
      onMove(e.clientX);
    });

    window.addEventListener('mousemove', (e) => onMove(e.clientX));
    window.addEventListener('mouseup', () => { isDragging = false; });

    // Touch support
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

    // Reset Stepper
    this.stepperItems.forEach(item => item.classList.remove('completed'));
    this.progressHeadline.textContent = 'Executing Agentic Remote Sensing Analysis...';
    this.progressSubtext.textContent = 'Orchestrating specialist models via AgentRouter & ModelRegistry...';

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
      const response = await fetch('/api/v1/analyze', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        const errorDetail = data.detail || 'Analysis request failed.';
        throw new Error(errorDetail);
      }

      this.currentResult = data;
      this.renderResults(data);
    } catch (err) {
      this.showError('Analysis Failed', err.message);
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
      this.summaryTask.textContent = `Task: ${taskFormatted}`;
    }

    // 1. Answer Narrative & Confidence
    this.summaryAnswer.textContent = contract.answer || 'Analysis completed.';
    const confVal = contract.confidence !== null && contract.confidence !== undefined 
      ? (contract.confidence * 100).toFixed(1) + '%' 
      : 'N/A';
    this.summaryConfidence.textContent = `Evidence Confidence: ${confVal}`;

    // 1b. Semantic Change Interpretation (Milestone M5)
    this.renderSemanticInterpretation(contract.evidence?.semantic_interpretation);

    // 1c. Cross-Modal Complementarity & 3-Panel Preview (Milestone M6)
    const isOpticalSar = contract.task === 'optical_sar_analysis';
    if (isOpticalSar) {
      if (this.semanticCard) this.semanticCard.classList.add('hidden');
      if (this.visualHubCard) this.visualHubCard.classList.add('hidden');
      if (this.opticalSarPreviewCard) {
        this.opticalSarPreviewCard.classList.remove('hidden');
        const compImg = contract.evidence.images?.find(img => img.role === 'semantic_composite');
        if (compImg && this.opticalSarCompositeImg) {
          this.opticalSarCompositeImg.src = compImg.url;
        }
      }
      if (contract.evidence.complementarity_report && this.complementarityCard) {
        this.complementarityCard.classList.remove('hidden');
        const cr = contract.evidence.complementarity_report;
        if (this.compOpticalText) this.compOpticalText.textContent = cr.optical_limitations || '';
        if (this.compSarText) this.compSarText.textContent = cr.sar_penetration || '';
        if (this.compJointText) this.compJointText.textContent = cr.structural_contrast || '';
      }
    } else {
      if (this.opticalSarPreviewCard) this.opticalSarPreviewCard.classList.add('hidden');
      if (this.complementarityCard) this.complementarityCard.classList.add('hidden');
      if (this.visualHubCard) this.visualHubCard.classList.remove('hidden');
    }

    // 2. Telemetry Cards
    const params = contract.parameters || {};
    const changedPixels = params.changed_pixels !== undefined ? params.changed_pixels.toLocaleString() + ' px' : '--';
    this.valChangedPixels.textContent = changedPixels;

    // Physical Area
    const statAreaM2 = contract.evidence.statistics?.find(s => s.metric_name === 'changed_area_m2');
    const statAreaHa = contract.evidence.statistics?.find(s => s.metric_name === 'changed_area_hectares');
    if (statAreaM2 && statAreaHa) {
      this.valPhysicalArea.textContent = `${statAreaM2.value.toLocaleString()} m² (${statAreaHa.value.toFixed(2)} ha)`;
    } else {
      this.valPhysicalArea.textContent = 'Pixel space only';
    }

    // Ratio & Clusters
    const changeRatio = params.change_ratio_pct !== undefined ? params.change_ratio_pct.toFixed(2) + '%' : '--';
    this.valChangeRatio.textContent = changeRatio;

    const clusters = params.total_clusters !== undefined ? params.total_clusters : (contract.evidence.boxes?.length || 0);
    this.valTotalClusters.textContent = clusters;

    // Specialist & Latency
    let specialistName = 'Auto Orchestrator';
    if (contract.models && contract.models.length > 0) {
      specialistName = contract.models.map(m => m.model_name || m).join(' → ');
    } else if (params.candidate_model) {
      specialistName = params.candidate_model;
    }
    this.valSpecialist.textContent = specialistName;

    const latency = contract.execution_time_ms !== undefined ? `${contract.execution_time_ms} ms` : '--';
    this.valLatency.textContent = latency;

    // 3. Visual Preview Images
    const maskObj = contract.evidence.masks && contract.evidence.masks.length > 0 ? contract.evidence.masks[0] : null;
    const overlayImg = contract.evidence.images?.find(img => img.role === 'change_overlay');
    const primaryImg = contract.evidence.images?.find(img => img.role === 'primary');
    const secondaryImg = contract.evidence.images?.find(img => img.role === 'secondary');

    const overlayUrl = overlayImg ? overlayImg.url : (params.overlay_path || '');
    const maskUrl = maskObj ? maskObj.url : (params.mask_path || '');
    
    // For T1 & T2: if PNG/JPEG, browser can use local object URL; otherwise use backend rendered PNG previews
    const isT1Web = this.t1File && (this.t1File.type === 'image/png' || this.t1File.type === 'image/jpeg');
    const isT2Web = this.t2File && (this.t2File.type === 'image/png' || this.t2File.type === 'image/jpeg');
    const t1Url = isT1Web ? URL.createObjectURL(this.t1File) : (primaryImg ? primaryImg.url : overlayUrl);
    const t2Url = isT2Web ? URL.createObjectURL(this.t2File) : (secondaryImg ? secondaryImg.url : overlayUrl);

    this.sliderImgBefore.src = t1Url;
    this.sliderImgAfter.src = overlayUrl || t2Url;

    this.galleryT1.src = t1Url;
    this.galleryT2.src = t2Url;
    this.galleryMask.src = maskUrl;
    this.galleryOverlay.src = overlayUrl;

    setTimeout(() => this.updateSliderDimensions(), 100);

    // 4. Geospatial Bounding Box Telemetry
    this.renderTelemetryBoxes(contract.evidence.boxes || []);

    // 5. Auditable Execution Trace
    this.renderTrace(contract.execution_trace || []);

    // 6. Warnings
    this.renderWarnings(contract.warnings || []);

    // Scroll smoothly to results
    this.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  renderSemanticInterpretation(semanticInterp) {
    if (!semanticInterp) {
      this.semanticCard.classList.add('hidden');
      return;
    }

    this.semanticCard.classList.remove('hidden');

    const dir = (semanticInterp.temporal_direction || 'modified').toLowerCase();
    this.semanticDirectionBadge.textContent = dir.toUpperCase();
    this.semanticDirectionBadge.className = `direction-badge direction-${dir}`;

    this.predomTransitionBadge.textContent = semanticInterp.predominant_transition || 'Unspecified';

    const uncPct = Math.round((semanticInterp.semantic_uncertainty || 0) * 100);
    this.semanticUncertaintyBadge.textContent = `Model Uncertainty: ${uncPct}%`;

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

  renderTelemetryBoxes(boxes) {
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
    this.traceTimeline.innerHTML = '';
    this.traceStepsCount.textContent = `${traceSteps.length} Steps Recorded`;

    traceSteps.forEach(step => {
      // Highlight matching stepper item
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
          <span class="trace-step-details">${step.details || ''}</span>
        </div>
      `;
      this.traceTimeline.appendChild(item);
    });
  }

  renderWarnings(warnings) {
    if (!warnings || warnings.length === 0) {
      this.warningsCard.classList.add('hidden');
      return;
    }
    this.warningsCard.classList.remove('hidden');
    this.warningsList.innerHTML = '';
    warnings.forEach(warn => {
      const li = document.createElement('li');
      li.textContent = warn;
      this.warningsList.appendChild(li);
    });
  }

  showError(title, detail) {
    this.errorTitle.textContent = title;
    this.errorDetail.textContent = detail;
    this.errorBanner.classList.remove('hidden');
    this.errorBanner.scrollIntoView({ behavior: 'smooth', block: 'center' });
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
