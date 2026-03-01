/**
 * chat-realtime.js — Realtime (live voice) conversation feature
 * Extracted from chat-app.js
 */
(function() {
  Object.assign(window._ChatAppProto, {

    _openRealtimeDialog() {
      // Use inline voice bar instead of overlay
      this._showInlineVoiceBar('realtime');
      this._rtStatusText = this.el.ivbStatus;
      this._rtCanvas = this.el.ivbWave;

      this._rtChatBody = null;
      this._rtChatText = '';
      this._rtStatusText.textContent = this.t('chat.realtime_connecting');

      // Ask server to open Azure Realtime WS
      this.ws.send({ type: 'realtime_start' });
    },

    _onRealtimeReady(msg) {
      // Server has connected to Azure Realtime — start mic capture
      if (msg.error) {
        if (this._rtStatusText) this._rtStatusText.textContent = `Error: ${msg.error}`;
        return;
      }
      this._rtActive = true;
      // PTT mode: start muted, show idle phase
      if (this._pushToTalk) {
        this._rtMicMuted = true;
        this._setSpeechPhase('idle');
        if (this._rtStatusText) this._rtStatusText.textContent = this.t('chat.voice_ptt_ready');
      } else {
        this._rtMicMuted = false;
        if (this._rtStatusText) this._rtStatusText.textContent = this.t('chat.realtime_listening');
      }
      if (this._rtOrb) this._rtOrb.classList.add('cv2-rt-active');
      // Show AI avatar in inline bar
      const slot = this.el.ivbAvatar;
      if (slot && this.el.ivbBar.style.display !== 'none') {
        if (this.config.appMascot) {
          slot.innerHTML = '<img class="cv2-voice-avatar" src="' + this.config.appMascot + '" alt="">';
        } else {
          slot.innerHTML = '<div class="cv2-voice-avatar cv2-voice-avatar-initials">AI</div>';
        }
      }
      this._startRealtimeMic();
    },

    _onRealtimeEvent(msg) {
      // Events relayed from server (Azure → our backend → browser)
      const event = msg.event;
      if (!event) return;

      switch (event.type) {
        case 'session.created':
        case 'session.updated':
          break;

        case 'session.closed':
          // Server closed the Azure WS
          this._stopRealtimeMic();
          this._stopRealtimePlayback();
          this._rtActive = false;
          break;

        case 'input_audio_buffer.speech_started':
          if (this._rtStatusText) this._rtStatusText.textContent = this.t('chat.realtime_listening');
          this._setSpeechPhase('recording');
          this._stopRealtimePlayback();
          // Cancel any in-progress AI response
          this.ws.send({ type: 'rt_send', event: { type: 'response.cancel' } });
          break;

        case 'input_audio_buffer.speech_stopped':
          break;

        case 'response.created':
          if (this._rtStatusText) this._rtStatusText.textContent = this.t('chat.realtime_speaking');
          this._setSpeechPhase('responding');
          break;

        case 'response.audio.delta':
          this._enqueueRealtimeAudio(event.delta);
          break;

        case 'response.audio.done':
          // Audio finished — return to appropriate state
          if (this._pushToTalk) {
            this._setSpeechPhase('idle');
            if (this._rtStatusText) this._rtStatusText.textContent = this.t('chat.voice_ptt_ready');
          } else {
            this._setSpeechPhase('recording');
            if (this._rtStatusText) this._rtStatusText.textContent = this.t('chat.realtime_listening');
          }
          break;

        case 'response.audio_transcript.delta':
          this._appendRealtimeTranscript(event.delta, 'assistant');
          break;

        case 'response.audio_transcript.done':
          this._finalizeRealtimeTranscript('assistant');
          break;

        case 'conversation.item.input_audio_transcription.completed':
          this._addRealtimeTranscript(event.transcript || '', 'user');
          break;

        // Tool calls are handled server-side — no client bridging needed

        case 'error':
          console.error('[Realtime] Error:', event.error);
          // Show error as a chat message (not in the compact status bar)
          const errMsg = event.error?.message || 'unknown error';
          const errEl = document.createElement('div');
          errEl.className = 'cv2-msg-assistant';
          errEl.innerHTML = `
            <div class="cv2-msg-header">
              <span class="material-icons" style="font-size:18px;color:#ef4444">error</span>
              <span>Live Voice Error</span>
            </div>
            <div class="cv2-msg-body" style="color:#ef4444;font-size:13px">${this._escapeHtml(errMsg)}</div>
          `;
          this.el.messages.appendChild(errEl);
          this._scrollToBottom();
          break;
      }
    },

    // ── Realtime Mic Capture ──────────────────────────────

    async _startRealtimeMic() {
      try {
        this._rtMicStream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 24000, channelCount: 1 } });
        this._rtAudioCtx = new AudioContext({ sampleRate: 24000 });
        const source = this._rtAudioCtx.createMediaStreamSource(this._rtMicStream);

        // Use AudioWorkletNode (inline processor via Blob URL)
        const workletCode = `
          class PcmCaptureProcessor extends AudioWorkletProcessor {
            process(inputs) {
              const input = inputs[0]?.[0];
              if (input) this.port.postMessage(input);
              return true;
            }
          }
          registerProcessor('pcm-capture', PcmCaptureProcessor);
        `;
        const blob = new Blob([workletCode], { type: 'application/javascript' });
        const url = URL.createObjectURL(blob);
        await this._rtAudioCtx.audioWorklet.addModule(url);
        URL.revokeObjectURL(url);

        const workletNode = new AudioWorkletNode(this._rtAudioCtx, 'pcm-capture');
        workletNode.port.onmessage = (e) => {
          if (!this._rtActive || this._rtMicMuted) return;
          const float32 = e.data;
          const int16 = new Int16Array(float32.length);
          for (let i = 0; i < float32.length; i++) {
            const s = Math.max(-1, Math.min(1, float32[i]));
            int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
          }
          const b64 = this._arrayBufferToBase64(int16.buffer);
          this.ws.send({
            type: 'rt_send',
            event: { type: 'input_audio_buffer.append', audio: b64 },
          });
        };
        source.connect(workletNode);
        workletNode.connect(this._rtAudioCtx.destination);
        this._rtProcessor = workletNode;
        this._rtMicSource = source;

        // Visualize mic input on canvas
        this._startRealtimeVisualization(source);
      } catch (err) {
        console.error('[Realtime] Mic error:', err);
        if (this._rtStatusText) this._rtStatusText.textContent = 'Microphone access denied';
      }
    },

    _stopRealtimeMic() {
      if (this._rtProcessor) { this._rtProcessor.disconnect(); this._rtProcessor = null; }
      if (this._rtMicSource) { this._rtMicSource.disconnect(); this._rtMicSource = null; }
      if (this._rtMicStream) {
        this._rtMicStream.getTracks().forEach(t => t.stop());
        this._rtMicStream = null;
      }
      if (this._rtVisRaf) { cancelAnimationFrame(this._rtVisRaf); this._rtVisRaf = null; }
    },

    _startRealtimeVisualization(source) {
      if (!this._rtAudioCtx || !this._rtCanvas) return;
      const analyser = this._rtAudioCtx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      const data = new Uint8Array(analyser.frequencyBinCount);
      const canvas = this._rtCanvas;
      const ctx = canvas.getContext('2d');

      const draw = () => {
        if (!this._rtActive) return;
        // Ensure canvas internal size matches display size (DPR-aware)
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        const w = Math.max(1, Math.floor(rect.width * dpr));
        const h = Math.max(1, Math.floor(rect.height * dpr));
        if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }

        analyser.getByteFrequencyData(data);
        ctx.clearRect(0, 0, w, h);
        const barW = (w / data.length) * 2.5;
        const mid = h / 2;
        ctx.fillStyle = 'rgba(75, 143, 231, 0.6)';
        for (let i = 0; i < data.length; i++) {
          const barH = (data[i] / 255) * mid;
          const x = i * barW;
          ctx.fillRect(x, mid - barH, barW - 1, barH * 2);
        }
        this._rtVisRaf = requestAnimationFrame(draw);
      };
      this._rtVisRaf = requestAnimationFrame(draw);
    },

    // ── Realtime Audio Playback ───────────────────────────

    _enqueueRealtimeAudio(b64Delta) {
      if (!this._rtPlaybackCtx) {
        this._rtPlaybackCtx = new AudioContext({ sampleRate: 24000 });
        this._rtPlaybackQueue = [];
        this._rtPlaybackPlaying = false;
      }
      // Decode base64 PCM16 to Float32
      const raw = atob(b64Delta);
      const bytes = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
      const int16 = new Int16Array(bytes.buffer);
      const float32 = new Float32Array(int16.length);
      for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;

      this._rtPlaybackQueue.push(float32);
      if (!this._rtPlaybackPlaying) this._playRealtimeQueue();
    },

    async _playRealtimeQueue() {
      this._rtPlaybackPlaying = true;
      const ctx = this._rtPlaybackCtx;
      while (this._rtPlaybackQueue.length > 0) {
        // Batch a few chunks together for smoother playback
        const chunks = this._rtPlaybackQueue.splice(0, Math.min(3, this._rtPlaybackQueue.length));
        let totalLen = 0;
        for (const c of chunks) totalLen += c.length;
        const merged = new Float32Array(totalLen);
        let offset = 0;
        for (const c of chunks) { merged.set(c, offset); offset += c.length; }

        const buf = ctx.createBuffer(1, merged.length, 24000);
        buf.getChannelData(0).set(merged);
        const src = ctx.createBufferSource();
        src.buffer = buf;
        src.connect(ctx.destination);
        src.start(0);

        // Wait for this buffer to finish
        await new Promise(resolve => { src.onended = resolve; });
      }
      this._rtPlaybackPlaying = false;
    },

    _stopRealtimePlayback() {
      if (this._rtPlaybackQueue) this._rtPlaybackQueue.length = 0;
      this._rtPlaybackPlaying = false;
    },

    // ── Realtime Transcript ───────────────────────────────

    _appendRealtimeTranscript(delta, role) {
      if (role !== 'assistant') return;
      if (!this._rtChatBody) {
        const msgEl = document.createElement('div');
        msgEl.className = 'cv2-msg-assistant';
        msgEl.innerHTML = `
          <div class="cv2-msg-header">
            <span class="material-icons" style="font-size:18px;color:var(--chat-accent)">call</span>
            <span>Live</span>
          </div>
          <div class="cv2-msg-body"></div>
        `;
        this.el.messages.appendChild(msgEl);
        this._rtChatBody = msgEl.querySelector('.cv2-msg-body');
        this._rtChatText = '';
      }
      this._rtChatText += delta;
      this._rtChatBody.innerHTML = this.md.render(this._rtChatText);
      this._scrollToBottom();
    },

    _finalizeRealtimeTranscript(role) {
      this._rtChatBody = null;
      this._rtChatText = '';
    },

    _addRealtimeTranscript(text, role) {
      if (!text) return;
      if (role === 'user') {
        const userEl = document.createElement('div');
        userEl.className = 'cv2-msg-user';
        userEl.innerHTML = `<div class="cv2-msg-user-bubble">${this.md.render(text)}</div>`;
        this.el.messages.appendChild(userEl);
        this._scrollToBottom();
      }
    },

    // ── Realtime Close ────────────────────────────────────

    _closeRealtimeDialog() {
      this._stopRealtimeMic();
      this._stopRealtimePlayback();
      this._rtActive = false;
      this._rtMicMuted = false;
      // Tell server to close the Azure WS
      this.ws.send({ type: 'realtime_stop' });
      if (this._rtAudioCtx) {
        this._rtAudioCtx.close().catch(() => {});
        this._rtAudioCtx = null;
      }
      if (this._rtPlaybackCtx) {
        this._rtPlaybackCtx.close().catch(() => {});
        this._rtPlaybackCtx = null;
      }
      // Hide inline voice bar (restores input area)
      this._hideInlineVoiceBar();
      // Also hide legacy overlay if somehow visible
      if (this._rtOverlay) {
        this._rtOverlay.style.display = 'none';
      }
      if (this._rtOrb) this._rtOrb.classList.remove('cv2-rt-active');
    },

    // ── Helpers ───────────────────────────────────────────

    _escapeHtml(str) {
      const d = document.createElement('div');
      d.textContent = str;
      return d.innerHTML;
    },

    _arrayBufferToBase64(buffer) {
      const bytes = new Uint8Array(buffer);
      let binary = '';
      for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
      return btoa(binary);
    },

  });

  ChatFeatures.register('realtime', {
    handleMessage: {
      'realtime_ready': '_onRealtimeReady',
      'rt_event': '_onRealtimeEvent',
    },
  });
})();
