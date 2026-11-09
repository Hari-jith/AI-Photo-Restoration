/* ==========================================================================
   Old Photo Restoration: the page's script (plain JavaScript, no libraries)

   What it does
     1. Asks the server what it can do (/api/health, /api/options) and builds the
        option form from that answer.
     2. Lets you choose a photo (click or drag and drop) and checks it.
     3. Sends the photo and options to the server (POST /api/jobs).
     4. Asks the server how it is going about once a second (polling) and
        shows progress, then the before/after comparison and the downloads.

   The page's screen is controlled by ONE attribute:  <body data-state="...">
     idle -> ready -> running -> done      (the stylesheet decides what shows)

   Safety rule used everywhere: text that comes from the server is inserted with
   textContent (as plain text), never as HTML, so it can never run as code.
   ========================================================================== */
(() => {
  'use strict';

  // ---------------------------------------------------------------- settings
  const POLL_FIRST_MS = 300;
  const POLL_EVERY_MS = 800;
  const POLL_MAX_FAILURES = 5;          // lost-connection retries before giving up
  const ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tif', 'tiff'];
  const BLANK_PICTURE = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';

  // Quick settings. Only the listed options change; everything else is left alone.
  const PRESETS = {
    recommended: { values: null,   // null = the server's defaults
      hint: 'Recommended: repairs light damage, restores faces and enlarges the photo twice.' },
    fast: { values: { repair: 'light', upscale: 1, faces: 'off' },
      hint: 'Fast: skips enlarging and face restoration. Useful to check what colorizing and repair do.' },
    best: { values: { repair: 'strong', upscale: 2, upscale_model: 'x4plus', faces: 'on', face_strength: 0.6 },
      hint: 'Best quality: stronger repair and the better enlarging model. Without a graphics card this can take many minutes.' },
  };

  // ---------------------------------------------------------------- elements
  const byId = (id) => {
    const element = document.getElementById(id);
    if (!element) console.error('app.js: the page has no element with id "' + id + '"');
    return element;
  };
  const el = {};
  ['status-chip', 'card-options', 'error-box', 'error-text', 'error-dismiss', 'drop-zone', 'file-input', 'limits-text',
   'preview-img', 'file-name', 'file-meta', 'remove-file', 'preset-hint', 'options-form', 'options-loading',
   'options-basic', 'options-advanced', 'options-advanced-wrap', 'start-button', 'start-hint',
   'progress-bar', 'progress-percent', 'progress-message', 'queue-info', 'steps-list', 'elapsed',
   'cancel-button', 'h-progress', 'h-result', 'notes', 'notes-list', 'compare', 'img-before', 'img-after',
   'compare-range', 'open-full', 'download-png', 'download-jpeg', 'again-button', 'new-button',
   'steps-table-body', 'analysis-summary', 'live-region'].forEach((id) => { el[id] = byId(id); });
  const presetButtons = Array.from(document.querySelectorAll('[data-preset]'));
  const stepItems = () => Array.from(el['steps-list'].querySelectorAll('li[data-step]'));

  // ---------------------------------------------------------------- state
  const app = {
    file: null, previewUrl: null,
    schema: null, defaults: {}, stepNames: [], stepWeights: {},
    maxUploadMb: 10,
    jobId: null, oldJobId: null, pollToken: 0, pollFailures: 0, pollTimer: null,
    startedAt: 0, clock: null, knownStepStatus: {},
  };

  // ---------------------------------------------------------------- tiny utilities
  const setState = (name) => { document.body.dataset.state = name; };
  const getState = () => document.body.dataset.state;
  const announce = (text) => { el['live-region'].textContent = ''; setTimeout(() => { el['live-region'].textContent = text; }, 50); };

  function showError(message) {
    el['error-text'].textContent = message;
    el['error-box'].hidden = false;
    el['error-box'].scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
  const hideError = () => { el['error-box'].hidden = true; };

  const formatBytes = (n) => (n < 1024 * 1024 ? Math.max(1, Math.round(n / 1024)) + ' KB' : (n / 1048576).toFixed(1) + ' MB');
  const formatDuration = (s) => (s < 60 ? Math.round(s) + ' s' : Math.floor(s / 60) + ' min ' + String(Math.round(s % 60)).padStart(2, '0') + ' s');

  class ApiError extends Error {
    constructor(status, message) { super(message); this.status = status; }
  }

  // Talks to the server. Every failure becomes an ApiError with a message fit for the user.
  async function api(path, options) {
    let response;
    try {
      response = await fetch(path, options);
    } catch (error) {
      throw new ApiError(0, 'Cannot reach the server. Is it still running? (Start it with: python -m src.api.main)');
    }
    if (response.ok) {
      return response.status === 204 ? null : response.json();
    }
    let message = 'The server answered with an error (' + response.status + ').';
    try {
      const data = await response.json();
      if (typeof data.detail === 'string') message = data.detail;
      else if (Array.isArray(data.detail)) message = data.detail.map((d) => d.msg).join('; ');
    } catch (ignored) { /* the body was not JSON: keep the generic message */ }
    throw new ApiError(response.status, message);
  }

  // ---------------------------------------------------------------- start-up
  async function loadHealth() {
    const chip = el['status-chip'];
    try {
      const health = await api('/api/health');
      app.maxUploadMb = health.limits.max_upload_mb;
      el['limits-text'].textContent = 'JPEG, PNG, WebP, BMP or TIFF, up to ' + app.maxUploadMb + ' MB';
      const models = health.models_installed;
      const missing = [];
      if (!models.colorizer && !models.backup_colorizer) missing.push('colorizing');
      if (!models.faces) missing.push('face restoration');
      if (!models.upscaler_general) missing.push('enlarging');
      const where = health.device.kind === 'cuda' ? 'GPU: ' + health.device.name : 'CPU only (slower)';
      if (missing.length) {
        chip.textContent = 'Missing models for: ' + missing.join(', ');
        chip.className = 'chip chip--warn';
        showError('Some AI models are not installed (' + missing.join(', ') + '). Those steps will be skipped. ' +
                  'Install them with: python scripts\\download_weights.py');
      } else {
        chip.textContent = 'Ready \u00b7 ' + where;
        chip.className = 'chip chip--ok';
      }
    } catch (error) {
      chip.textContent = 'Server not reachable';
      chip.className = 'chip chip--warn';
      throw error;
    }
  }

  async function loadOptions() {
    const data = await api('/api/options');
    app.schema = data.options;
    app.defaults = data.defaults;
    app.stepNames = data.steps.map((s) => s.name);
    app.stepWeights = Object.fromEntries(data.steps.map((s) => [s.name, s.weight]));
    buildOptionFields();
    el['options-loading'].hidden = true;
    updateStartButton();
  }

  // ---------------------------------------------------------------- the options form
  const decimalsOf = (step) => { const t = String(step); return t.includes('.') ? t.split('.')[1].length : 0; };
  const capitalize = (t) => t.charAt(0).toUpperCase() + t.slice(1);

  function formatOutput(option, value) {
    if (option.name === 'upscale') return Number(value) === 1 ? 'Off' : Number(value) + '\u00d7';
    return Number(value).toFixed(decimalsOf(option.step || 1));
  }

  function buildOptionFields() {
    for (const option of app.schema) {
      const id = 'opt-' + option.name;
      const field = document.createElement('div');
      field.className = 'field';

      const label = document.createElement('label');
      label.className = 'field__label';
      label.htmlFor = id;
      label.textContent = option.label;
      field.appendChild(label);

      let control;
      const initial = app.defaults[option.name];
      if (option.type === 'choice') {
        control = document.createElement('select');
        control.className = 'control';
        for (const choice of option.choices) {
          const item = document.createElement('option');
          item.value = choice;
          item.textContent = capitalize(choice);
          control.appendChild(item);
        }
        control.value = initial;
        field.appendChild(control);
      } else if (option.type === 'number') {
        const row = document.createElement('div');
        row.className = 'field__range';
        control = document.createElement('input');
        control.type = 'range';
        control.className = 'range';
        control.min = option.min; control.max = option.max; control.step = option.step;
        control.value = initial;
        const output = document.createElement('output');
        output.className = 'field__value';
        output.htmlFor = id;
        output.textContent = formatOutput(option, initial);
        control.addEventListener('input', () => { output.textContent = formatOutput(option, control.value); });
        row.append(control, output);
        field.appendChild(row);
      } else {                                     // a whole number; empty means "automatic"
        control = document.createElement('input');
        control.type = 'number';
        control.className = 'control';
        control.min = option.min;
        control.step = 1;
        control.placeholder = 'automatic';
        if (initial !== null && initial !== undefined) control.value = initial;
        field.appendChild(control);
      }
      control.id = id;
      control.name = option.name;
      control.addEventListener('input', onOptionChanged);
      control.addEventListener('change', onOptionChanged);

      const help = document.createElement('p');
      help.className = 'field__help';
      help.id = id + '-help';
      help.textContent = option.help;
      control.setAttribute('aria-describedby', help.id);
      field.appendChild(help);

      (option.advanced ? el['options-advanced'] : el['options-basic']).appendChild(field);
    }
    syncDisabledControls();
    updatePresetButtons();
  }

  const control = (name) => document.getElementById('opt-' + name);

  function getValues() {
    const values = {};
    for (const option of app.schema || []) {
      const c = control(option.name);
      if (!c) continue;
      if (option.type === 'number') values[option.name] = Number(c.value);
      else if (option.type === 'integer') values[option.name] = c.value === '' ? null : Number(c.value);
      else values[option.name] = c.value;
    }
    return values;
  }

  function setValues(values) {
    for (const option of app.schema) {
      if (!(option.name in values)) continue;
      const c = control(option.name);
      const v = values[option.name];
      c.value = v === null ? '' : v;
      c.dispatchEvent(new Event('input', { bubbles: true }));   // refreshes the number shown beside a slider
    }
  }

  // Options that make no sense together are greyed out (their values are still sent).
  function syncDisabledControls() {
    const faces = control('faces'), strength = control('face_strength');
    const upscale = control('upscale'), model = control('upscale_model');
    if (faces && strength) strength.disabled = faces.value === 'off';
    if (upscale && model) model.disabled = Number(upscale.value) === 1;
  }

  function presetValues(name) {
    const preset = PRESETS[name];
    return preset.values ? { ...app.defaults, ...preset.values } : { ...app.defaults };
  }

  function updatePresetButtons() {
    const now = getValues();
    let matched = null;
    for (const name of Object.keys(PRESETS)) {
      const wanted = presetValues(name);
      const same = Object.keys(wanted).every((key) => key === 'colorize' || key === 'max_side' || wanted[key] === now[key]);
      if (same && !matched) matched = name;
    }
    presetButtons.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.preset === matched)));
    el['preset-hint'].textContent = matched ? PRESETS[matched].hint
      : 'Custom settings. Pick a quick setting above to start again from a known one.';
  }

  function onOptionChanged() { syncDisabledControls(); updatePresetButtons(); }

  function applyPreset(name) {
    const wanted = presetValues(name);
    delete wanted.colorize;            // presets never change your colorize choice
    delete wanted.max_side;
    setValues(wanted);
    syncDisabledControls();
    updatePresetButtons();
    announce(PRESETS[name].hint);
  }

  // ---------------------------------------------------------------- choosing a photo
  function describeProblem(file) {
    const extension = (file.name.split('.').pop() || '').toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(extension)) {
      return '"' + file.name + '" is not a supported picture type. Please choose a JPEG, PNG, WebP, BMP or TIFF file.';
    }
    if (file.size > app.maxUploadMb * 1048576) {
      return 'That file is ' + formatBytes(file.size) + ', but the limit is ' + app.maxUploadMb + ' MB. Please choose a smaller file.';
    }
    if (file.size === 0) return 'That file is empty.';
    return null;
  }

  function clearFile() {
    if (app.previewUrl) URL.revokeObjectURL(app.previewUrl);
    app.file = null; app.previewUrl = null;
    el['file-input'].value = '';
    el['preview-img'].src = BLANK_PICTURE;
  }

  function chooseFile(file) {
    hideError();
    const problem = describeProblem(file);
    if (problem) { showError(problem); return false; }
    clearFile();
    app.file = file;
    app.previewUrl = URL.createObjectURL(file);
    el['file-name'].textContent = file.name;
    el['file-meta'].textContent = formatBytes(file.size);
    const preview = el['preview-img'];
    preview.onload = () => {
      el['file-meta'].textContent = preview.naturalWidth + ' \u00d7 ' + preview.naturalHeight + ' pixels \u00b7 ' + formatBytes(file.size);
    };
    preview.onerror = () => {      // some browsers cannot show TIFF/BMP: the photo can still be restored
      preview.src = BLANK_PICTURE;
      el['file-meta'].textContent = formatBytes(file.size) + ' \u00b7 no preview available for this file type';
    };
    preview.src = app.previewUrl;
    setState('ready');
    updateStartButton();
    announce('Photo chosen: ' + file.name);
    return true;
  }

  function updateStartButton() {
    const ready = !!app.file && !!app.schema && getState() === 'ready';
    el['start-button'].disabled = !ready;
    el['start-hint'].textContent = !app.file ? 'Choose a photo first.'
      : 'Without a graphics card this usually takes one to two minutes.';
  }

  // ---------------------------------------------------------------- running a job
  async function startJob(event) {
    event.preventDefault();
    if (!app.file || getState() !== 'ready') return;
    hideError();
    const form = new FormData();
    form.append('file', app.file, app.file.name);
    for (const [name, value] of Object.entries(getValues())) {
      if (value !== null && value !== undefined && !Number.isNaN(value)) form.append(name, String(value));
    }
    el['start-button'].disabled = true;
    let job;
    try {
      job = await api('/api/jobs', { method: 'POST', body: form });
    } catch (error) {
      showError(error.message);
      updateStartButton();
      return;
    }
    if (app.oldJobId) { api('/api/jobs/' + app.oldJobId, { method: 'DELETE' }).catch(() => {}); app.oldJobId = null; }
    app.jobId = job.job_id;
    beginProgressScreen();
    schedulePoll(POLL_FIRST_MS);
  }

  function beginProgressScreen() {
    app.startedAt = Date.now();
    app.knownStepStatus = {};
    app.pollFailures = 0;
    renderProgress({ progress: 0, message: 'Sending the photo\u2026', status: 'queued', queue_position: null });
    for (const item of stepItems()) { item.dataset.status = 'pending'; item.querySelector('.steps__time').textContent = ''; }
    el['cancel-button'].disabled = false;
    setState('running');
    el['h-progress'].tabIndex = -1;
    el['h-progress'].focus({ preventScroll: true });
    announce('Restoration started.');
    clearInterval(app.clock);
    const tick = () => { el['elapsed'].textContent = formatDuration((Date.now() - app.startedAt) / 1000); };
    tick();
    app.clock = setInterval(tick, 1000);
  }

  function stopTimers() {
    app.pollToken += 1;                    // makes any answer still on its way irrelevant
    clearTimeout(app.pollTimer);
    clearInterval(app.clock);
  }

  function schedulePoll(delay) {
    const token = ++app.pollToken;
    clearTimeout(app.pollTimer);
    app.pollTimer = setTimeout(() => poll(token), delay);
  }

  async function poll(token) {
    let view;
    try {
      view = await api('/api/jobs/' + app.jobId);
    } catch (error) {
      if (token !== app.pollToken) return;
      if (error.status === 404) {
        return failBackToReady('The server no longer has this photo (it may have been restarted, or the result expired). Please try again.');
      }
      app.pollFailures += 1;
      if (app.pollFailures >= POLL_MAX_FAILURES) {
        return failBackToReady('Lost contact with the server. If it is still running, the result may be lost. Please try again.');
      }
      return schedulePoll(POLL_EVERY_MS * (app.pollFailures + 1));
    }
    if (token !== app.pollToken) return;
    app.pollFailures = 0;
    if (view.status === 'done') return finishJob(view);
    if (view.status === 'failed') return failBackToReady(view.error || 'The restoration failed.');
    if (view.status === 'cancelled') return failBackToReady('Cancelled. You can change the settings and try again.', true);
    renderProgress(view);
    schedulePoll(POLL_EVERY_MS);
  }

  // Which step is running? The server gives progress as a fraction; the step weights
  // (also from the server) say where each step starts and ends.
  function activeStepIndex(fraction) {
    const total = app.stepNames.reduce((sum, name) => sum + app.stepWeights[name], 0);
    let end = 0;
    for (let i = 0; i < app.stepNames.length; i += 1) {
      end += app.stepWeights[app.stepNames[i]];
      if (fraction * total + 1e-6 < end) return i;
    }
    return app.stepNames.length - 1;
  }

  function renderProgress(view) {
    const percent = Math.min(100, Math.max(0, Math.round(view.progress * 100)));
    el['progress-bar'].style.setProperty('--progress', percent + '%');
    el['progress-bar'].setAttribute('aria-valuenow', String(percent));
    el['progress-percent'].textContent = percent + '%';
    el['progress-message'].textContent = view.message;

    const waiting = view.status === 'queued' && view.queue_position !== null && view.queue_position !== undefined;
    el['queue-info'].hidden = !waiting;
    if (waiting) {
      el['queue-info'].textContent = view.queue_position === 0
        ? 'You are next in line.'
        : view.queue_position + ' photo(s) are waiting ahead of yours.';
    }

    const match = /^(\w+): (done|skipped|failed)/.exec(view.message || '');
    if (match) app.knownStepStatus[match[1]] = match[2];
    const active = view.status === 'queued' ? -1 : activeStepIndex(view.progress);
    stepItems().forEach((item, index) => {
      const name = item.dataset.step;
      if (index < active) item.dataset.status = app.knownStepStatus[name] || 'done';
      else if (index === active) item.dataset.status = 'active';
      else item.dataset.status = 'pending';
    });
  }

  async function cancelJob() {
    el['cancel-button'].disabled = true;
    el['progress-message'].textContent = 'Cancelling (the current step has to finish first)\u2026';
    try {
      await api('/api/jobs/' + app.jobId + '/cancel', { method: 'POST' });
    } catch (error) {
      if (error.status !== 409) showError(error.message);   // 409 = it had just finished
    }
  }

  // Go back to the "ready" screen after a failure or cancellation.
  function failBackToReady(message, isCancel = false) {
    stopTimers();
    app.jobId = null;
    setState('ready');
    updateStartButton();
    if (isCancel) { el['start-hint'].textContent = message; announce(message); }
    else { showError(message); announce('The restoration did not finish.'); }
  }

  // ---------------------------------------------------------------- the result
  const loadPicture = (img, url) => new Promise((resolve, reject) => {
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('The picture could not be loaded.'));
    img.src = url;
  });

  async function finishJob(view) {
    stopTimers();
    renderProgress({ ...view, progress: 1, message: 'Done' });
    const base = '/api/jobs/' + app.jobId;
    try {
      await Promise.all([loadPicture(el['img-before'], base + '/original'), loadPicture(el['img-after'], base + '/result')]);
    } catch (error) {
      return failBackToReady('The restored picture could not be loaded. Please try again.');
    }
    const after = el['img-after'];
    el['compare'].style.setProperty('--aspect', String(after.naturalWidth / after.naturalHeight));
    el['compare-range'].value = 50;
    moveSlider();

    el['open-full'].href = base + '/result';
    el['download-png'].href = base + '/result?download=1';
    el['download-jpeg'].href = base + '/result?format=jpeg&download=1';

    el['notes-list'].replaceChildren(...view.notes.map((text) => {
      const item = document.createElement('li');
      item.textContent = text;
      return item;
    }));
    el['notes'].hidden = view.notes.length === 0;

    el['steps-table-body'].replaceChildren(...view.steps.map((step) => {
      const row = document.createElement('tr');
      for (const text of [step.name, step.status, step.seconds.toFixed(1) + ' s', step.message]) {
        const cell = document.createElement('td');
        cell.textContent = text;
        row.appendChild(cell);
      }
      return row;
    }));
    const byName = Object.fromEntries(view.steps.map((s) => [s.name, s]));
    for (const item of stepItems()) {
      const step = byName[item.dataset.step];
      if (step) { item.dataset.status = step.status; item.querySelector('.steps__time').textContent = step.seconds.toFixed(1) + ' s'; }
    }
    const a = view.analysis || {};
    const facts = [];
    if (a.color_kind) facts.push('photo type: ' + a.color_kind.replace('_', ' '));
    if (typeof a.noise_sigma === 'number') facts.push('grain level ' + a.noise_sigma.toFixed(1));
    if (typeof a.contrast_range === 'number') facts.push('contrast ' + Math.round(a.contrast_range) + ' of 255');
    el['analysis-summary'].textContent = facts.length ? 'What the program measured: ' + facts.join(', ') + '.' : '';

    app.oldJobId = app.jobId;
    setState('done');
    el['h-result'].tabIndex = -1;
    el['h-result'].focus({ preventScroll: false });
    announce('Your restored photo is ready. Use the slider to compare it with the original.');
  }

  function moveSlider() {
    const value = Number(el['compare-range'].value);
    el['compare'].style.setProperty('--split', value + '%');
    el['compare-range'].setAttribute('aria-valuetext', value + ' percent of the original photo shown');
  }

  // ---------------------------------------------------------------- starting over
  function backToReady() {                       // "Try different settings"
    app.jobId = null;
    setState('ready');
    updateStartButton();
    el['start-hint'].textContent = 'Change the settings, then restore the same photo again.';
    el['card-options'].scrollIntoView({ block: 'start' });
  }

  function backToIdle() {                        // "Restore another photo" / "Choose a different photo"
    stopTimers();
    for (const id of [app.jobId, app.oldJobId]) {
      if (id && getState() !== 'running') api('/api/jobs/' + id, { method: 'DELETE' }).catch(() => {});
    }
    app.jobId = null; app.oldJobId = null;
    clearFile();
    hideError();
    setState('idle');
    updateStartButton();
    window.scrollTo({ top: 0 });
  }

  // ---------------------------------------------------------------- wiring
  function wireEvents() {
    el['file-input'].addEventListener('change', () => {
      const file = el['file-input'].files && el['file-input'].files[0];
      if (file) chooseFile(file);
    });

    const zone = el['drop-zone'];
    ['dragenter', 'dragover'].forEach((type) => zone.addEventListener(type, (e) => {
      e.preventDefault();
      zone.classList.add('is-dragover');
    }));
    ['dragleave', 'dragend'].forEach((type) => zone.addEventListener(type, () => zone.classList.remove('is-dragover')));
    zone.addEventListener('drop', (e) => {
      e.preventDefault();
      zone.classList.remove('is-dragover');
      const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (file) chooseFile(file);
    });
    // A photo dropped anywhere else must not make the browser leave the page.
    ['dragover', 'drop'].forEach((type) => window.addEventListener(type, (e) => e.preventDefault()));

    el['remove-file'].addEventListener('click', backToIdle);
    el['options-form'].addEventListener('submit', startJob);
    presetButtons.forEach((b) => b.addEventListener('click', () => applyPreset(b.dataset.preset)));
    el['cancel-button'].addEventListener('click', cancelJob);
    el['again-button'].addEventListener('click', backToReady);
    el['new-button'].addEventListener('click', backToIdle);
    el['compare-range'].addEventListener('input', moveSlider);
    el['error-dismiss'].addEventListener('click', hideError);

    window.addEventListener('beforeunload', (e) => {
      if (getState() === 'running') { e.preventDefault(); e.returnValue = ''; }
    });
  }

  async function init() {
    wireEvents();
    if (location.protocol === 'file:') {
      el['status-chip'].textContent = 'Not connected to the server';
      el['status-chip'].className = 'chip chip--warn';
      showError('This page has to be opened through the restoration server, not as a file. ' +
                'Start the server with "python -m src.api.main" and open http://127.0.0.1:8000 in your browser.');
      return;
    }
    try {
      await loadHealth();
      await loadOptions();
    } catch (error) {
      showError(error.message);
    }
  }

  init();
})();
