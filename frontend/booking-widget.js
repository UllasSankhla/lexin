/**
 * Voice Booking Widget
 * Embeddable voice appointment booking assistant
 *
 * Usage:
 *   <script src="booking-widget.js"
 *     data-api-url="http://localhost:8001"
 *     data-customer-key="ck_live_..."
 *     data-theme="light"
 *     data-position="bottom-right"
 *     data-primary-color="#4F46E5"
 *     data-button-label="Book Appointment">
 *   </script>
 */
(function () {
  'use strict';

  // ── Config ──────────────────────────────────────────────────────────────────
  const scriptEl = document.currentScript || (() => {
    const scripts = document.querySelectorAll('script[data-api-url]');
    return scripts[scripts.length - 1];
  })();

  const cfg = {
    apiUrl: (scriptEl && scriptEl.dataset.apiUrl) || 'http://localhost:8001',
    customerKey: (scriptEl && scriptEl.dataset.customerKey) || '',
    theme: (scriptEl && scriptEl.dataset.theme) || 'light',
    position: (scriptEl && scriptEl.dataset.position) || 'bottom-right',
    primaryColor: (scriptEl && scriptEl.dataset.primaryColor) || '#4F46E5',
    buttonLabel: (scriptEl && scriptEl.dataset.buttonLabel) || 'Book Appointment',
    debug: (scriptEl && scriptEl.dataset.debug) === 'true',
  };

  const log = (...args) => cfg.debug && console.log('[BookingWidget]', ...args);

  // ── State ────────────────────────────────────────────────────────────────────
  const STATES = {
    IDLE: 'idle',
    CONNECTING: 'connecting',
    ACTIVE: 'active',
    THINKING: 'thinking',
    SPEAKING: 'speaking',
    COMPLETING: 'completing',
    DONE: 'done',
    ERROR: 'error',
  };

  let state = STATES.IDLE;
  let ws = null;
  let audioCtx = null;
  let mediaStream = null;
  let audioWorkletNode = null;
  let scriptProcessor = null;
  let wsSeq = 0;
  let playbackCtx = null;     // single AudioContext reused for the whole call
  let nextStartTime = 0;      // Web Audio clock position for gapless scheduling
  let activeSources = [];     // BufferSourceNodes in flight (for interruption)
  let jitterBuffer = [];      // PCM chunks buffered before jitter threshold is met
  let jitterReady = false;    // true once initial jitter buffer has been flushed

  const JITTER_BUFFER_MIN_CHUNKS = 2;  // accumulate this many chunks before starting playback
  let callId = null;
  let shadowRoot = null;
  let animFrame = null;
  let collectedParams = {};

  // ── UI Injection ─────────────────────────────────────────────────────────────
  function createWidget() {
    const host = document.createElement('div');
    host.id = 'booking-widget-host';
    host.style.cssText = `
      position: fixed;
      ${cfg.position.includes('right') ? 'right: 24px;' : 'left: 24px;'}
      bottom: 24px;
      z-index: 2147483647;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    `;
    document.body.appendChild(host);
    shadowRoot = host.attachShadow({ mode: 'open' });

    const styles = `
      :host { all: initial; }
      * { box-sizing: border-box; }

      .widget-btn {
        width: 64px; height: 64px;
        border-radius: 50%;
        background: ${cfg.primaryColor};
        border: none;
        cursor: pointer;
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        transition: transform 0.2s, box-shadow 0.2s;
        position: relative;
        outline: none;
      }
      .widget-btn:hover { transform: scale(1.08); box-shadow: 0 6px 20px rgba(0,0,0,0.25); }
      .widget-btn:active { transform: scale(0.96); }

      .pulse-ring {
        position: absolute;
        top: -8px; left: -8px; right: -8px; bottom: -8px;
        border-radius: 50%;
        border: 3px solid ${cfg.primaryColor};
        opacity: 0;
        animation: none;
      }
      .pulse-ring.active {
        animation: pulse 1.4s ease-out infinite;
      }
      @keyframes pulse {
        0% { transform: scale(0.85); opacity: 0.8; }
        100% { transform: scale(1.3); opacity: 0; }
      }

      .mic-icon svg { width: 28px; height: 28px; fill: white; }

      .panel {
        position: absolute;
        bottom: 80px;
        ${cfg.position.includes('right') ? 'right: 0;' : 'left: 0;'}
        width: 340px;
        background: ${cfg.theme === 'dark' ? '#1e1e2e' : '#ffffff'};
        color: ${cfg.theme === 'dark' ? '#e0e0f0' : '#1a1a2e'};
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.18);
        padding: 20px;
        display: none;
        flex-direction: column;
        gap: 12px;
        animation: slideUp 0.25s ease;
      }
      .panel.visible { display: flex; }
      @keyframes slideUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
      }

      .panel-header {
        display: flex; align-items: center; justify-content: space-between;
        padding-bottom: 12px;
        border-bottom: 1px solid ${cfg.theme === 'dark' ? '#333' : '#eee'};
      }
      .persona-name {
        font-weight: 700; font-size: 16px; color: ${cfg.primaryColor};
      }
      .call-status {
        font-size: 12px; color: ${cfg.theme === 'dark' ? '#888' : '#999'};
        text-transform: uppercase; letter-spacing: 0.5px;
      }
      .close-btn {
        background: none; border: none; cursor: pointer;
        color: ${cfg.theme === 'dark' ? '#888' : '#aaa'};
        font-size: 20px; padding: 0; line-height: 1;
      }

      .waveform-container {
        height: 48px;
        display: flex; align-items: center; justify-content: center;
        gap: 3px;
      }
      .waveform-bar {
        width: 4px; border-radius: 2px;
        background: ${cfg.primaryColor};
        transition: height 0.05s ease;
        min-height: 4px;
      }

      .transcript-box {
        background: ${cfg.theme === 'dark' ? '#2a2a3e' : '#f5f5ff'};
        border-radius: 10px;
        padding: 12px;
        font-size: 14px;
        line-height: 1.5;
        min-height: 60px;
        max-height: 120px;
        overflow-y: auto;
      }
      .transcript-caller { color: ${cfg.theme === 'dark' ? '#a0a0d0' : '#555'}; }
      .transcript-assistant { color: ${cfg.primaryColor}; font-weight: 500; }

      .thinking-dots {
        display: inline-flex; gap: 4px; align-items: center;
      }
      .thinking-dots span {
        width: 8px; height: 8px; border-radius: 50%;
        background: ${cfg.primaryColor}; opacity: 0.3;
        animation: blink 1.2s infinite;
      }
      .thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
      .thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
      @keyframes blink {
        0%, 80%, 100% { opacity: 0.3; }
        40% { opacity: 1; }
      }

      .params-summary {
        background: ${cfg.theme === 'dark' ? '#1a2e1a' : '#f0fff4'};
        border: 1px solid ${cfg.theme === 'dark' ? '#2a5a2a' : '#b8f0cc'};
        border-radius: 10px;
        padding: 12px;
        font-size: 13px;
      }
      .params-summary h4 { margin: 0 0 8px; font-size: 13px; color: #4caf50; }
      .param-item { display: flex; justify-content: space-between; padding: 2px 0; }
      .param-label { color: ${cfg.theme === 'dark' ? '#888' : '#666'}; }
      .param-value { font-weight: 600; }

      .action-row {
        display: flex; gap: 8px; justify-content: flex-end;
      }
      .end-call-btn {
        background: #ef4444; color: white;
        border: none; border-radius: 8px;
        padding: 8px 16px; font-size: 14px; font-weight: 600;
        cursor: pointer; transition: background 0.2s;
      }
      .end-call-btn:hover { background: #dc2626; }

      .done-message {
        text-align: center; padding: 12px;
        font-size: 15px; font-weight: 600;
        color: #4caf50;
      }

      .label-tag {
        display: inline-block;
        background: ${cfg.primaryColor}22;
        color: ${cfg.primaryColor};
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 11px;
        font-weight: 700;
        margin-bottom: 6px;
      }
    `;

    const numBars = 20;
    const barsHtml = Array.from({ length: numBars }, (_, i) =>
      `<div class="waveform-bar" id="bar-${i}" style="height:4px"></div>`
    ).join('');

    shadowRoot.innerHTML = `
      <style>${styles}</style>
      <div class="pulse-ring" id="pulse-ring"></div>
      <button class="widget-btn" id="main-btn" aria-label="${cfg.buttonLabel}">
        <span class="mic-icon">
          <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 1a4 4 0 0 1 4 4v6a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4zm6.5 9.5A6.5 6.5 0 0 1 5.5 10.5H3.5a8.5 8.5 0 0 0 7.5 8.45V21h-3v2h7v-2h-3v-2.05A8.5 8.5 0 0 0 19.5 10.5h-2z"/>
          </svg>
        </span>
      </button>
      <div class="panel" id="panel">
        <div class="panel-header">
          <div>
            <div class="label-tag">AI Assistant</div>
            <div class="persona-name" id="persona-name">Assistant</div>
          </div>
          <div style="display:flex;align-items:center;gap:12px">
            <span class="call-status" id="call-status">Idle</span>
            <button class="close-btn" id="close-btn" aria-label="Close">✕</button>
          </div>
        </div>

        <div class="waveform-container" id="waveform">${barsHtml}</div>

        <div class="transcript-box" id="transcript-box">
          <span style="color:#999;font-size:13px">Your conversation will appear here…</span>
        </div>

        <div id="params-section" style="display:none">
          <div class="params-summary" id="params-summary">
            <h4>✓ Information Collected</h4>
            <div id="params-list"></div>
          </div>
        </div>

        <div class="action-row">
          <button class="end-call-btn" id="end-call-btn" style="display:none">End Call</button>
        </div>

        <div class="done-message" id="done-message" style="display:none">
          ✓ Booking request submitted! You'll receive a confirmation shortly.
        </div>
      </div>
    `;

    // Event listeners
    shadowRoot.getElementById('main-btn').addEventListener('click', onMainButtonClick);
    shadowRoot.getElementById('close-btn').addEventListener('click', closePanel);
    shadowRoot.getElementById('end-call-btn').addEventListener('click', endCall);
  }

  // ── State Transitions ─────────────────────────────────────────────────────────
  function setState(newState) {
    log('State:', state, '->', newState);
    state = newState;
    updateUI();
  }

  function updateUI() {
    const panel = shadowRoot.getElementById('panel');
    const pulseRing = shadowRoot.getElementById('pulse-ring');
    const statusEl = shadowRoot.getElementById('call-status');
    const endBtn = shadowRoot.getElementById('end-call-btn');
    const doneMsg = shadowRoot.getElementById('done-message');

    switch (state) {
      case STATES.IDLE:
        pulseRing.classList.remove('active');
        statusEl.textContent = 'Idle';
        endBtn.style.display = 'none';
        break;
      case STATES.CONNECTING:
        pulseRing.classList.add('active');
        panel.classList.add('visible');
        statusEl.textContent = 'Connecting…';
        endBtn.style.display = 'inline-block';
        setTranscript('<span style="color:#999">Connecting to assistant…</span>');
        break;
      case STATES.ACTIVE:
        pulseRing.classList.add('active');
        statusEl.textContent = 'Listening';
        endBtn.style.display = 'inline-block';
        break;
      case STATES.THINKING:
        statusEl.textContent = 'Thinking';
        break;
      case STATES.SPEAKING:
        statusEl.textContent = 'Speaking';
        break;
      case STATES.COMPLETING:
        statusEl.textContent = 'Wrapping up';
        break;
      case STATES.DONE:
        pulseRing.classList.remove('active');
        statusEl.textContent = 'Done';
        endBtn.style.display = 'none';
        doneMsg.style.display = 'block';
        stopAudioCapture();
        setTimeout(() => {
          panel.classList.remove('visible');
          doneMsg.style.display = 'none';
          setState(STATES.IDLE);
        }, 6000);
        break;
      case STATES.ERROR:
        pulseRing.classList.remove('active');
        statusEl.textContent = 'Error';
        endBtn.style.display = 'none';
        stopAudioCapture();
        break;
    }
  }

  function setTranscript(html) {
    shadowRoot.getElementById('transcript-box').innerHTML = html;
  }

  function appendTranscript(speaker, text) {
    const box = shadowRoot.getElementById('transcript-box');
    const cls = speaker === 'caller' ? 'transcript-caller' : 'transcript-assistant';
    const label = speaker === 'caller' ? 'You' : shadowRoot.getElementById('persona-name').textContent;
    const entry = document.createElement('div');
    entry.style.marginBottom = '6px';
    entry.innerHTML = `<strong class="${cls}">${label}:</strong> ${escapeHtml(text)}`;
    if (box.querySelector('span[style]')) box.innerHTML = '';
    box.appendChild(entry);
    box.scrollTop = box.scrollHeight;
  }

  function showThinking() {
    const box = shadowRoot.getElementById('transcript-box');
    let thinkingEl = box.querySelector('.thinking-indicator');
    if (!thinkingEl) {
      thinkingEl = document.createElement('div');
      thinkingEl.className = 'thinking-indicator';
      thinkingEl.innerHTML = `<div class="thinking-dots"><span></span><span></span><span></span></div>`;
      box.appendChild(thinkingEl);
    }
  }

  function hideThinking() {
    const thinkingEl = shadowRoot.getElementById('transcript-box').querySelector('.thinking-indicator');
    if (thinkingEl) thinkingEl.remove();
  }

  function updateParamsSummary() {
    const section = shadowRoot.getElementById('params-section');
    const list = shadowRoot.getElementById('params-list');
    if (Object.keys(collectedParams).length === 0) return;

    section.style.display = 'block';
    list.innerHTML = Object.entries(collectedParams).map(([k, v]) =>
      `<div class="param-item">
        <span class="param-label">${formatLabel(k)}</span>
        <span class="param-value">${escapeHtml(v)}</span>
      </div>`
    ).join('');
  }

  // ── Button / Panel Controls ───────────────────────────────────────────────────
  function onMainButtonClick() {
    if (state === STATES.IDLE) {
      startCall();
    } else {
      const panel = shadowRoot.getElementById('panel');
      panel.classList.toggle('visible');
    }
  }

  function closePanel() {
    shadowRoot.getElementById('panel').classList.remove('visible');
  }

  function endCall() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'client.hangup', seq: ++wsSeq, ts: Date.now() / 1000, payload: {} }));
    }
    cleanup();
    setState(STATES.DONE);
  }

  // ── Call Flow ──────────────────────────────────────────────────────────────────
  async function startCall() {
    setState(STATES.CONNECTING);
    collectedParams = {};

    try {
      const initiateHeaders = { 'Content-Type': 'application/json' };
      if (cfg.customerKey) initiateHeaders['X-Customer-Key'] = cfg.customerKey;
      const resp = await fetch(`${cfg.apiUrl}/api/v1/calls/initiate`, {
        method: 'POST',
        headers: initiateHeaders,
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      callId = data.call_id;
      const wsUrl = data.ws_url;
      log('Call initiated:', callId, 'ws:', wsUrl);

      await openWebSocket(wsUrl);
      await startAudioCapture();
    } catch (err) {
      log('Failed to start call:', err);
      setTranscript(`<span style="color:#ef4444">Failed to connect: ${escapeHtml(err.message)}</span>`);
      setState(STATES.ERROR);
      setTimeout(() => setState(STATES.IDLE), 4000);
    }
  }

  function openWebSocket(url) {
    return new Promise((resolve, reject) => {
      ws = new WebSocket(url);
      ws.binaryType = 'arraybuffer';

      ws.onopen = () => { log('WebSocket open'); resolve(); };
      ws.onerror = (e) => { log('WebSocket error', e); reject(new Error('WebSocket failed')); };
      ws.onclose = (e) => {
        log('WebSocket closed', e.code, e.reason);
        if (state !== STATES.DONE && state !== STATES.ERROR) {
          cleanup();
          setState(STATES.DONE);
        }
      };
      ws.onmessage = handleWSMessage;
    });
  }

  function handleWSMessage(event) {
    if (event.data instanceof ArrayBuffer) {
      handleAudioFrame(event.data);
      return;
    }
    try {
      const msg = JSON.parse(event.data);
      handleTextMessage(msg);
    } catch (e) {
      log('Failed to parse message', e);
    }
  }

  function handleTextMessage(msg) {
    const { type, payload } = msg;
    log('MSG:', type, payload);

    switch (type) {
      case 'server.session_ready':
        shadowRoot.getElementById('persona-name').textContent = payload.persona_name || 'Assistant';
        setState(STATES.ACTIVE);
        break;

      case 'server.transcript_interim':
        if (state === STATES.ACTIVE) {
          // Update interim display without persisting
          const box = shadowRoot.getElementById('transcript-box');
          let interimEl = box.querySelector('.interim-transcript');
          if (!interimEl) {
            interimEl = document.createElement('div');
            interimEl.className = 'interim-transcript';
            interimEl.style.cssText = 'color:#aaa;font-style:italic;font-size:13px;';
            box.appendChild(interimEl);
          }
          interimEl.textContent = payload.text;
        }
        break;

      case 'server.transcript_final':
        // Remove interim, show final
        const interimEl2 = shadowRoot.getElementById('transcript-box').querySelector('.interim-transcript');
        if (interimEl2) interimEl2.remove();
        appendTranscript('caller', payload.text);
        break;

      case 'server.greeting':
        setState(STATES.SPEAKING);
        appendTranscript('assistant', payload.text);
        break;

      case 'server.thinking':
        setState(STATES.THINKING);
        showThinking();
        break;

      case 'server.tts_stream_start':
        setState(STATES.SPEAKING);
        jitterBuffer = [];
        jitterReady = false;
        break;

      case 'server.tts_stream_end':
        // Flush chunks still in the jitter buffer (short responses may never hit threshold)
        if (jitterBuffer.length > 0) {
          jitterReady = true;
          jitterBuffer.forEach(pcm => _scheduleNow(pcm));
          jitterBuffer = [];
        }
        break;

      case 'server.tts_interrupted':
        stopAudioPlayback();
        setState(STATES.ACTIVE);
        break;

      case 'server.response_text':
        hideThinking();
        appendTranscript('assistant', payload.text);
        break;

      case 'server.parameter_collected':
        collectedParams[payload.parameter_name] = payload.value;
        updateParamsSummary();
        break;

      case 'server.call_completing':
        setState(STATES.COMPLETING);
        break;

      case 'server.call_ended':
        cleanup();
        setState(STATES.DONE);
        break;

      case 'server.error':
        log('Server error:', payload);
        if (payload.fatal) {
          cleanup();
          setState(STATES.ERROR);
          setTranscript(`<span style="color:#ef4444">Error: ${escapeHtml(payload.message)}</span>`);
        }
        break;
    }
  }

  // ── Audio Capture ─────────────────────────────────────────────────────────────
  async function startAudioCapture() {
    // Create the playback context here, inside the user-gesture chain, so the
    // browser starts it in 'running' state.  Creating it lazily (on the first
    // binary frame) results in a 'suspended' context: scheduled buffers pile up
    // and then fire all at once when the context auto-resumes, causing the
    // greeting to sound choppy and subsequent audio to rush.
    playbackCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
    nextStartTime = 0;
    activeSources = [];
    log('Playback AudioContext created | sampleRate:', playbackCtx.sampleRate, 'state:', playbackCtx.state);

    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });

    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: {
      echoCancellation: true,
      noiseSuppression: true,
      sampleRate: 16000,
      channelCount: 1,
    }});

    const source = audioCtx.createMediaStreamSource(mediaStream);

    // Try AudioWorklet first, fallback to ScriptProcessor
    try {
      await audioCtx.audioWorklet.addModule(PCM_WORKLET_BLOB_URL);
      audioWorkletNode = new AudioWorkletNode(audioCtx, 'pcm-processor');
      audioWorkletNode.port.onmessage = (e) => sendAudioChunk(e.data);
      source.connect(audioWorkletNode);
      audioWorkletNode.connect(audioCtx.destination);
      log('Using AudioWorklet for PCM capture');
    } catch (_) {
      log('AudioWorklet unavailable, using ScriptProcessor');
      scriptProcessor = audioCtx.createScriptProcessor(8192, 1, 1);
      scriptProcessor.onaudioprocess = (e) => {
        const float32 = e.inputBuffer.getChannelData(0);
        sendAudioChunk(floatTo16BitPCM(float32));
      };
      source.connect(scriptProcessor);
      scriptProcessor.connect(audioCtx.destination);
    }

    // VAD waveform animation
    startWaveformAnimation(source);

    // Send ready signal
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'client.ready',
        seq: ++wsSeq,
        ts: Date.now() / 1000,
        payload: { audio_format: 'pcm_s16le', sample_rate: 16000, channels: 1, chunk_ms: 20 },
      }));
    }
  }

  function sendAudioChunk(pcmData) {
    // Send during ACTIVE, SPEAKING (barge-in), and THINKING (early interruption).
    // Echo cancellation on the mic handles TTS bleed-through during SPEAKING.
    const captureStates = [STATES.ACTIVE, STATES.SPEAKING, STATES.THINKING];
    if (ws && ws.readyState === WebSocket.OPEN && captureStates.includes(state)) {
      ws.send(pcmData instanceof ArrayBuffer ? pcmData : pcmData.buffer);
    }
  }

  function floatTo16BitPCM(float32Array) {
    const int16 = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      const s = Math.max(-1, Math.min(1, float32Array[i]));
      int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return int16;
  }

  function stopAudioCapture() {
    if (mediaStream) {
      mediaStream.getTracks().forEach(t => t.stop());
      mediaStream = null;
    }
    if (audioWorkletNode) { audioWorkletNode.disconnect(); audioWorkletNode = null; }
    if (scriptProcessor) { scriptProcessor.disconnect(); scriptProcessor = null; }
    if (audioCtx) { audioCtx.close(); audioCtx = null; }
    if (playbackCtx) { playbackCtx.close(); playbackCtx = null; }
    if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
    resetWaveform();
  }

  // ── Waveform Animation ────────────────────────────────────────────────────────
  function startWaveformAnimation(source) {
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 64;
    source.connect(analyser);

    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    const bars = shadowRoot.querySelectorAll('.waveform-bar');

    function draw() {
      animFrame = requestAnimationFrame(draw);
      analyser.getByteFrequencyData(dataArray);
      bars.forEach((bar, i) => {
        const idx = Math.floor(i * dataArray.length / bars.length);
        const height = Math.max(4, (dataArray[idx] / 255) * 44);
        bar.style.height = height + 'px';
      });
    }
    draw();
  }

  function resetWaveform() {
    const bars = shadowRoot ? shadowRoot.querySelectorAll('.waveform-bar') : [];
    bars.forEach(b => b.style.height = '4px');
  }

  // ── Audio Playback ────────────────────────────────────────────────────────────

  function getPlaybackCtx() {
    // Context is created eagerly in startAudioCapture(); this is just a safety net.
    if (!playbackCtx || playbackCtx.state === 'closed') {
      log('Warning: playbackCtx missing or closed — recreating');
      playbackCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
      nextStartTime = 0;
      activeSources = [];
    }
    if (playbackCtx.state === 'suspended') {
      playbackCtx.resume();
    }
    return playbackCtx;
  }

  function handleAudioFrame(arrayBuffer) {
    // Parse binary frame: [4-byte ID length][ID bytes][PCM audio]
    const view = new DataView(arrayBuffer);
    const idLen = view.getUint32(0, true);
    const pcmData = arrayBuffer.slice(4 + idLen);
    scheduleAudioChunk(pcmData);
  }

  function scheduleAudioChunk(pcmBuffer) {
    if (!jitterReady) {
      jitterBuffer.push(pcmBuffer);
      if (jitterBuffer.length >= JITTER_BUFFER_MIN_CHUNKS) {
        jitterReady = true;
        jitterBuffer.forEach(pcm => _scheduleNow(pcm));
        jitterBuffer = [];
      }
      return;
    }
    _scheduleNow(pcmBuffer);
  }

  function _scheduleNow(pcmBuffer) {
    const ctx = getPlaybackCtx();

    const int16 = new Int16Array(pcmBuffer);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 32768.0;
    }

    const audioBuffer = ctx.createBuffer(1, float32.length, 24000);
    audioBuffer.copyToChannel(float32, 0);

    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(ctx.destination);

    // Schedule exactly when the previous chunk ends; snap to now if behind
    const startAt = Math.max(ctx.currentTime, nextStartTime);
    nextStartTime = startAt + audioBuffer.duration;

    source.onended = () => {
      activeSources = activeSources.filter(s => s !== source);
      if (activeSources.length === 0 && state === STATES.SPEAKING) {
        setState(STATES.ACTIVE);
      }
    };

    activeSources.push(source);
    source.start(startAt);
  }

  function stopAudioPlayback() {
    activeSources.forEach(s => { try { s.stop(); } catch (_) {} });
    activeSources = [];
    nextStartTime = 0;
    jitterBuffer = [];
    jitterReady = false;
  }

  // ── Cleanup ───────────────────────────────────────────────────────────────────
  function cleanup() {
    stopAudioPlayback();
    stopAudioCapture();  // also closes playbackCtx
    if (ws) {
      ws.onclose = null;
      ws.close();
      ws = null;
    }
  }

  // ── Utilities ─────────────────────────────────────────────────────────────────
  function escapeHtml(str) {
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function formatLabel(key) {
    return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }

  // ── AudioWorklet PCM Processor (Blob URL) ────────────────────────────────────
  const PCM_WORKLET_CODE = `
    class PCMProcessor extends AudioWorkletProcessor {
      constructor() {
        super();
        this._buffer = [];
        this._targetSamples = 640; // 40ms at 16kHz
      }
      process(inputs) {
        const input = inputs[0];
        if (!input || !input[0]) return true;
        const channel = input[0];
        for (let i = 0; i < channel.length; i++) {
          const s = Math.max(-1, Math.min(1, channel[i]));
          this._buffer.push(s < 0 ? s * 32768 : s * 32767);
        }
        while (this._buffer.length >= this._targetSamples) {
          const chunk = this._buffer.splice(0, this._targetSamples);
          const int16 = new Int16Array(chunk);
          this.port.postMessage(int16.buffer, [int16.buffer]);
        }
        return true;
      }
    }
    registerProcessor('pcm-processor', PCMProcessor);
  `;

  const PCM_WORKLET_BLOB = new Blob([PCM_WORKLET_CODE], { type: 'application/javascript' });
  const PCM_WORKLET_BLOB_URL = URL.createObjectURL(PCM_WORKLET_BLOB);

  // ── Init ──────────────────────────────────────────────────────────────────────
  function init() {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', createWidget);
    } else {
      createWidget();
    }
    log('Widget initialized', cfg);
  }

  init();
})();
