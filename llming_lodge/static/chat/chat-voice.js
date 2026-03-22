/**
 * chat-voice.js — Voice recording, TTS playback, speech mode
 * Extracted from chat-app.js
 */
(function() {
  // Static constants for voice
  ChatApp.VOICE_MAX_SECONDS = 90;
  ChatApp.VOICE_LOUDNESS_HISTORY = 160;

  Object.assign(window._ChatAppProto, {

    // ── Voice Recording ─────────────────────────────────

    async _startVoiceRecording() {
      if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
        this.el.voiceNote.textContent = this.t('chat.voice_no_support');
        return;
      }

      this._voiceChunks = [];
      this._voiceLoudnessHistory = Array(ChatApp.VOICE_LOUDNESS_HISTORY).fill(0);

      try {
        this._voiceStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      } catch (err) {
        this.el.voiceNote.textContent = this.t('chat.voice_no_support');
        return;
      }

      const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : '';

      this._voiceRecorder = mime
        ? new MediaRecorder(this._voiceStream, { mimeType: mime })
        : new MediaRecorder(this._voiceStream);

      this._voiceAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
      this._voiceSourceNode = this._voiceAudioCtx.createMediaStreamSource(this._voiceStream);
      this._voiceAnalyser = this._voiceAudioCtx.createAnalyser();
      this._voiceAnalyser.fftSize = 2048;
      this._voiceSourceNode.connect(this._voiceAnalyser);
      const pcm = new Uint8Array(this._voiceAnalyser.fftSize);
      const freq = new Uint8Array(this._voiceAnalyser.frequencyBinCount);

      this._voiceRecorder.addEventListener('dataavailable', (e) => {
        if (e.data?.size > 0) this._voiceChunks.push(e.data);
      });

      this._voiceRecorder.addEventListener('stop', () => {
        this._clearVoiceTimers();
        this._stopVoiceStream();
        const seconds = Math.max(1, Math.round((Date.now() - this._voiceStartedAt) / 1000));
        if (this._voiceChunks.length) {
          const blob = new Blob(this._voiceChunks, { type: this._voiceRecorder?.mimeType || 'audio/webm' });
          this._sendVoiceRecording(blob, seconds);
        } else if (this._speechMode) {
          // No audio captured — restart recording cycle
          this._speechModeNextCycle();
        }
        this._voiceRecorder = null;
      });

      this._voiceRecorder.start(250);
      this._voiceStartedAt = Date.now();
      this._voiceLastVoiceAt = this._voiceStartedAt;
      this._voiceActiveMs = 0;  // cumulative ms of voice activity
      this.el.speechModeBtn.classList.add('cv2-recording');

      // Visual frame loop
      const renderFrame = () => {
        if (!this._voiceAnalyser) return;
        this._voiceAnalyser.getByteTimeDomainData(pcm);
        this._voiceAnalyser.getByteFrequencyData(freq);
        let sum = 0;
        for (let i = 0; i < pcm.length; i++) {
          const c = (pcm[i] - 128) / 128;
          sum += c * c;
        }
        const rms = Math.sqrt(sum / pcm.length);
        if (rms > this._voiceSilenceThreshold) this._voiceLastVoiceAt = Date.now();
        this._drawVoiceVisuals(pcm, freq, rms);
        this._voiceRafId = requestAnimationFrame(renderFrame);
      };
      this._voiceRafId = requestAnimationFrame(renderFrame);

      // Elapsed timer
      this._voiceElapsedTimer = setInterval(() => {
        const elapsed = Math.floor((Date.now() - this._voiceStartedAt) / 1000);
        const mm = String(Math.floor(elapsed / 60)).padStart(2, '0');
        const ss = String(elapsed % 60).padStart(2, '0');
        if (this._pushToTalk) {
          this.el.voiceNote.textContent = `${this.t('chat.voice_listening')} ${mm}:${ss}`;
        } else {
          const silence = Math.max(0, this._voiceSilenceMs - (Date.now() - this._voiceLastVoiceAt)) / 1000;
          this.el.voiceNote.textContent = `${this.t('chat.voice_listening')} ${mm}:${ss} (${silence.toFixed(1)}s)`;
        }
        if (elapsed >= ChatApp.VOICE_MAX_SECONDS) {
          this._stopVoiceRecorder();
        }
      }, 1000);

      // Silence detection — disabled in push-to-talk mode
      if (!this._pushToTalk) {
        const SILENCE_CHECK_MS = 180;
        const MIN_VOICE_ACTIVITY_MS = 250; // require at least 250ms of actual speech
        this._voiceSilenceTimer = setInterval(() => {
          if (!this._voiceAnalyser) return;
          this._voiceAnalyser.getByteTimeDomainData(pcm);
          let sum = 0;
          for (let i = 0; i < pcm.length; i++) {
            const c = (pcm[i] - 128) / 128;
            sum += c * c;
          }
          const rms = Math.sqrt(sum / pcm.length);
          if (rms > this._voiceSilenceThreshold) {
            this._voiceLastVoiceAt = Date.now();
            this._voiceActiveMs += SILENCE_CHECK_MS;
          }
          const silenceDuration = Date.now() - this._voiceLastVoiceAt;
          const elapsed = Date.now() - this._voiceStartedAt;
          // Only stop if: enough total time passed, silence long enough, AND user actually spoke
          if (elapsed > 600 && silenceDuration >= this._voiceSilenceMs && this._voiceActiveMs >= MIN_VOICE_ACTIVITY_MS) {
            this.el.voiceNote.textContent = this.t('chat.voice_silence');
            this._stopVoiceRecorder();
          }
        }, SILENCE_CHECK_MS);
      }
    },

    _drawVoiceVisuals(pcm, freq, rms) {
      const ensureSize = (canvas) => {
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        const w = Math.max(1, Math.floor(rect.width * dpr));
        const h = Math.max(1, Math.floor(rect.height * dpr));
        if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
        return { w, h };
      };

      // Waveform
      const wCtx = this.el.voiceWave.getContext('2d');
      if (wCtx) {
        const { w, h } = ensureSize(this.el.voiceWave);
        wCtx.clearRect(0, 0, w, h);
        wCtx.strokeStyle = 'rgba(255,255,255,0.20)';
        wCtx.lineWidth = 1;
        wCtx.beginPath(); wCtx.moveTo(0, h * 0.5); wCtx.lineTo(w, h * 0.5); wCtx.stroke();
        wCtx.strokeStyle = '#f0f2f8';
        wCtx.lineWidth = 2;
        wCtx.beginPath();
        for (let i = 0; i < pcm.length; i++) {
          const x = (i / (pcm.length - 1)) * w;
          const y = (pcm[i] / 255) * h;
          i === 0 ? wCtx.moveTo(x, y) : wCtx.lineTo(x, y);
        }
        wCtx.stroke();
      }

      // Frequency bands (not present in inline voice bar)
      const bCtx = this.el.voiceBands?.getContext('2d');
      if (bCtx) {
        const { w, h } = ensureSize(this.el.voiceBands);
        const barCount = 28, barGap = 3;
        const barW = Math.max(2, (w - barGap * (barCount - 1)) / barCount);
        bCtx.clearRect(0, 0, w, h);
        for (let i = 0; i < barCount; i++) {
          const idx = Math.floor((i / barCount) * freq.length);
          const v = (freq[idx] || 0) / 255;
          const bh = Math.max(2, v * (h - 4));
          bCtx.fillStyle = `rgba(255,255,255,${0.2 + v * 0.75})`;
          bCtx.fillRect(i * (barW + barGap), h - bh, barW, bh);
        }
      }

      // Loudness history (not present in inline voice bar)
      this._voiceLoudnessHistory.push(Math.min(1, rms * 4.8));
      if (this._voiceLoudnessHistory.length > ChatApp.VOICE_LOUDNESS_HISTORY) this._voiceLoudnessHistory.shift();
      const lCtx = this.el.voiceLoudness?.getContext('2d');
      if (lCtx) {
        const { w, h } = ensureSize(this.el.voiceLoudness);
        lCtx.clearRect(0, 0, w, h);
        lCtx.strokeStyle = 'rgba(255,255,255,0.28)';
        lCtx.lineWidth = 1;
        lCtx.beginPath(); lCtx.moveTo(0, h - 1); lCtx.lineTo(w, h - 1); lCtx.stroke();
        lCtx.strokeStyle = '#ffd878';
        lCtx.lineWidth = 2;
        lCtx.beginPath();
        this._voiceLoudnessHistory.forEach((v, i) => {
          const x = (i / Math.max(1, this._voiceLoudnessHistory.length - 1)) * w;
          const y = h - v * (h - 4) - 2;
          i === 0 ? lCtx.moveTo(x, y) : lCtx.lineTo(x, y);
        });
        lCtx.stroke();
      }
    },

    _stopVoiceStream() {
      if (this._voiceStream) {
        this._voiceStream.getTracks().forEach(t => t.stop());
        this._voiceStream = null;
      }
      if (this._voiceSourceNode) { this._voiceSourceNode.disconnect(); this._voiceSourceNode = null; }
      if (this._voiceAnalyser) { this._voiceAnalyser.disconnect(); this._voiceAnalyser = null; }
      if (this._voiceAudioCtx) { this._voiceAudioCtx.close().catch(() => {}); this._voiceAudioCtx = null; }
      if (this._voiceRafId) { cancelAnimationFrame(this._voiceRafId); this._voiceRafId = null; }
      this._voiceLoudnessHistory = [];
    },

    _clearVoiceTimers() {
      if (this._voiceElapsedTimer) { clearInterval(this._voiceElapsedTimer); this._voiceElapsedTimer = null; }
      if (this._voiceSilenceTimer) { clearInterval(this._voiceSilenceTimer); this._voiceSilenceTimer = null; }
    },

    _stopVoiceRecorder() {
      if (this._voiceRecorder && this._voiceRecorder.state !== 'inactive') {
        this._voiceRecorder.stop();
      }
      this._clearVoiceTimers();
    },

    _sendNowVoiceRecording() {
      if (!this._voiceRecorder || this._voiceRecorder.state === 'inactive') return;
      // If no voice activity was detected, don't try to transcribe
      // (skip in PTT mode — user's explicit press/release is the intent)
      if (!this._pushToTalk && this._voiceActiveMs < 200) {
        console.log('[Voice] No voice activity detected (' + this._voiceActiveMs + 'ms), skipping send');
        if (this._speechMode) {
          this._speechModeNextCycle();
        } else {
          this._closeVoicePopup();
        }
        return;
      }
      if (this._speechMode) this._setSpeechPhase('processing');
      this.el.voiceNote.textContent = this.t('chat.voice_transcribing');
      this._voiceRecorder.requestData();
      this._stopVoiceRecorder();
    },

    _closeVoicePopup() {
      this._speechMode = false;
      if (this._voiceRecorder && this._voiceRecorder.state !== 'inactive') {
        this._voiceRecorder.stop();
      }
      this._voiceRecorder = null;
      this._clearVoiceTimers();
      this._stopVoiceStream();
      this._voiceChunks = [];
      this.el.speechModeBtn.classList.remove('cv2-recording');
      // Hide inline voice bar (restores input area)
      this._hideInlineVoiceBar();
      // Also hide legacy overlay if somehow visible
      this.el.voicePopup.classList.remove('cv2-speech-mode');
      this.el.voicePopup.style.display = 'none';
      if (this.el.voiceThinking) this.el.voiceThinking.style.display = 'none';
      // Disable speech response when closing voice popup
      if (this._speechResponse) {
        this._toggleSpeechResponse();
      }
    },

    async _sendVoiceRecording(blob, seconds) {
      // Guard against empty or too-small recordings (e.g. immediate cancel, no speech)
      // - Very short recordings (< 500ms) produce truncated WebM that OpenAI rejects
      // - In non-PTT mode, also require minimum voice activity
      const elapsed = Date.now() - this._voiceStartedAt;
      if (!blob || blob.size < 1000 || elapsed < 500 || (!this._pushToTalk && this._voiceActiveMs < 200)) {
        console.log('[Voice] Skipping send: blob=' + (blob ? blob.size + 'B' : 'null') + ', elapsed=' + elapsed + 'ms, voiceActive=' + this._voiceActiveMs + 'ms');

        if (this._speechMode) {
          this._speechModeNextCycle();
        } else {
          this._closeVoicePopup();
        }
        return;
      }
      this.el.voiceNote.textContent = this.t('chat.voice_transcribing');
      try {
        const buf = await blob.arrayBuffer();
        const bytes = new Uint8Array(buf);
        let binary = '';
        for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
        const audioB64 = btoa(binary);

        // Strip codecs param (e.g. "audio/webm;codecs=opus" → "audio/webm")
        const ct = (blob.type || 'audio/webm').split(';')[0];
        const mimeType = blob.type || '(none)';
        console.log(`[Voice] Sending audio: ${blob.size} bytes, blob.type="${mimeType}", content_type="${ct}", recorder.mimeType="${this._voiceRecorder?.mimeType || '(stopped)'}"`);
        this.ws.send({
          type: 'transcribe',
          audio_b64: audioB64,
          filename: 'voice.webm',
          content_type: ct,
        });
      } catch (err) {
        console.error('[Voice] Send error:', err);
        this.el.voiceNote.textContent = err.message || 'Transcription failed';
      }
    },

    // ── Transcription + TTS ─────────────────────────────

    handleTranscriptionResult(msg) {
      const text = (msg.text || '').trim();
      if (text) {
        const current = this.el.textarea.value.trim();
        this.el.textarea.value = current ? `${current} ${text}` : text;
        this._autoResizeTextarea();
      }
      if (this._speechMode) {
        if (text) {
          this.sendMessage();
          this._setSpeechPhase('responding');
          this.el.voiceNote.textContent = '...';
          // Show thinking animation while waiting for TTS
          this._showVoiceThinking();
        } else {
          // Empty transcription — restart recording
          this._speechModeNextCycle();
        }
      } else {
        this._closeVoicePopup();
      }
    },

    // ── TTS Playback ─────────────────────────────────────

    handleTTSAudio(msg) {
      const audioB64 = msg.audio_b64;
      if (!audioB64) return;

      if (msg.segment != null) {
        // Streaming segment (sentence-level TTS, no word timings)
        this._ttsSegments[msg.segment] = audioB64;
        this._tryPlayNextSegment();
      } else {
        // Single-shot TTS (explicit request, has word_timings)
        this._playSingleShotTTS(msg);
      }
    },

    handleTTSDone(msg) {
      // Code block or skipped — no audio was sent
      if (msg && msg.skipped) {
        this._stopTTSPlayback();
        if (this.el.voiceThinking) this.el.voiceThinking.style.display = 'none';
        if (this._speechMode) {
          this._speechModeNextCycle();
        }
        return;
      }
      this._ttsDone = true;
      if (!this._ttsPlaying && !this._ttsSegments[this._ttsNextSegment]) {
        this._onAllSegmentsPlayed();
      }
    },

    // ── Streaming TTS queue ───────────────────────────────

    _tryPlayNextSegment() {
      if (this._ttsPlaying) return;
      const b64 = this._ttsSegments[this._ttsNextSegment];
      if (!b64) return;
      delete this._ttsSegments[this._ttsNextSegment];
      this._ttsNextSegment++;
      this._ttsPlaying = true;
      this._ttsAllB64.push(b64);
      // Hide thinking animation — audio is arriving
      this._hideVoiceThinking();

      const binary = atob(b64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      const blob = new Blob([bytes], { type: 'audio/mpeg' });
      const url = URL.createObjectURL(blob);

      if (this._ttsAudio) { this._ttsAudio.pause(); this._ttsAudio = null; }
      this._ttsAudio = new Audio(url);
      this._ttsAudio.play().catch(() => {});

      // Visualize TTS audio on voice canvases (inline bar or popup)
      if (this._speechMode) {
        this._connectTTSVisualizer();
      }

      this._ttsAudio.addEventListener('ended', () => {
        URL.revokeObjectURL(url);
        this._ttsAudio = null;
        this._ttsPlaying = false;
        if (this._ttsSegments[this._ttsNextSegment]) {
          this._tryPlayNextSegment();
        } else if (this._ttsDone) {
          this._onAllSegmentsPlayed();
        }
      });
    },

    _showVoiceThinking() {
      if (this.el.voiceThinking) {
        this.el.voiceThinking.style.display = 'flex';
        // Hide canvas while thinking
        if (this.el.voiceWave) this.el.voiceWave.style.display = 'none';
      }
    },

    _hideVoiceThinking() {
      if (this.el.voiceThinking) {
        this.el.voiceThinking.style.display = 'none';
        if (this.el.voiceWave) this.el.voiceWave.style.display = 'block';
      }
    },

    _onAllSegmentsPlayed() {
      this._stopTTSVisualization();
      this._ttsNextSegment = 1;
      this._ttsDone = false;

      // Replay is available via the action bar "Anhören" button — no extra inline icon.
      this._ttsAllB64 = [];

      if (this._speechMode) {
        this._speechModeNextCycle();
      }
    },

    // ── Single-shot TTS (explicit request with word timings) ─

    _playSingleShotTTS(msg) {
      const audioB64 = msg.audio_b64;
      const wordTimings = msg.word_timings || [];

      this._stopTTSPlayback();

      const binary = atob(audioB64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      const blob = new Blob([bytes], { type: 'audio/mpeg' });
      const url = URL.createObjectURL(blob);

      this._ttsAudio = new Audio(url);
      this._ttsWordTimings = wordTimings;

      // No speaker button — single-shot TTS is triggered from the existing
      // action bar button ("Anhören"), so no extra replay element needed.

      this._ttsAudio.play().catch(() => {});

      this._ttsAudio.addEventListener('ended', () => {
        URL.revokeObjectURL(url);
        this._ttsAudio = null;
      });
    },

    // ── Speaker replay button ─────────────────────────────

    _addSpeakerButton(bodyEl, audioB64Arr, wordTimings) {
      // Remove existing replay button if any
      const existing = bodyEl.querySelector('.cv2-tts-replay-btn');
      if (existing) existing.remove();

      const btn = document.createElement('button');
      btn.className = 'cv2-tts-replay-btn';
      btn.innerHTML = '<span class="material-icons">volume_up</span>';
      btn.title = this.t('chat.speech_response');
      const storedB64 = [...audioB64Arr];
      const storedTimings = wordTimings || [];
      btn.addEventListener('click', () => {
        this._stopTTSPlayback();
        // Combine all segments into one blob
        let totalLen = 0;
        const parts = storedB64.map(b64 => {
          const bin = atob(b64);
          const arr = new Uint8Array(bin.length);
          for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
          totalLen += arr.length;
          return arr;
        });
        const combined = new Uint8Array(totalLen);
        let offset = 0;
        for (const p of parts) { combined.set(p, offset); offset += p.length; }
        const blob = new Blob([combined], { type: 'audio/mpeg' });
        const replayUrl = URL.createObjectURL(blob);
        this._ttsAudio = new Audio(replayUrl);
        this._ttsWordTimings = storedTimings;
        this._ttsAudio.play().catch(() => {});
        if (storedTimings.length) this._startWordHighlighting();
        this._ttsAudio.addEventListener('ended', () => {
          this._stopWordHighlighting();
          URL.revokeObjectURL(replayUrl);
          this._ttsAudio = null;
        });
      });
      bodyEl.appendChild(btn);
    },

    // ── Word highlighting ─────────────────────────────────

    _wrapWordsForHighlight(bodyEl, wordTimings) {
      const walker = document.createTreeWalker(bodyEl, NodeFilter.SHOW_TEXT, null);
      const textNodes = [];
      while (walker.nextNode()) textNodes.push(walker.currentNode);

      let wordIdx = 0;
      for (const node of textNodes) {
        if (wordIdx >= wordTimings.length) break;
        const words = node.textContent.split(/(\s+)/);
        if (words.length <= 1 && !words[0].trim()) continue;

        const frag = document.createDocumentFragment();
        for (const part of words) {
          if (!part.trim() || wordIdx >= wordTimings.length) {
            frag.appendChild(document.createTextNode(part));
            continue;
          }
          const span = document.createElement('span');
          span.className = 'cv2-tts-word';
          span.dataset.tStart = wordTimings[wordIdx].start;
          span.dataset.tEnd = wordTimings[wordIdx].end;
          span.textContent = part;
          frag.appendChild(span);
          wordIdx++;
        }
        node.parentNode.replaceChild(frag, node);
      }
    },

    _startWordHighlighting() {
      if (!this._ttsAudio) return;
      const highlightFrame = () => {
        if (!this._ttsAudio) return;
        const t = this._ttsAudio.currentTime;
        const words = document.querySelectorAll('.cv2-tts-word');
        words.forEach(w => {
          const start = parseFloat(w.dataset.tStart);
          const end = parseFloat(w.dataset.tEnd);
          w.classList.toggle('cv2-word-active', t >= start && t < end);
        });
        this._ttsRafId = requestAnimationFrame(highlightFrame);
      };
      this._ttsRafId = requestAnimationFrame(highlightFrame);
    },

    _stopWordHighlighting() {
      if (this._ttsRafId) { cancelAnimationFrame(this._ttsRafId); this._ttsRafId = null; }
      document.querySelectorAll('.cv2-word-active').forEach(w => w.classList.remove('cv2-word-active'));
    },

    // ── TTS Visualization (audio analysis during playback) ─

    _connectTTSVisualizer() {
      if (!this._ttsAudio) return;
      if (!this._ttsAudioCtx) {
        this._ttsAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (this._ttsAudioCtx.state === 'suspended') {
        this._ttsAudioCtx.resume();
      }
      // Disconnect previous source
      if (this._ttsSourceNode) {
        try { this._ttsSourceNode.disconnect(); } catch (_) {}
        this._ttsSourceNode = null;
      }
      this._ttsSourceNode = this._ttsAudioCtx.createMediaElementSource(this._ttsAudio);
      if (!this._ttsAnalyser) {
        this._ttsAnalyser = this._ttsAudioCtx.createAnalyser();
        this._ttsAnalyser.fftSize = 2048;
        this._ttsAnalyser.connect(this._ttsAudioCtx.destination);
      }
      this._ttsSourceNode.connect(this._ttsAnalyser);

      if (!this._ttsVisRafId) {
        const pcm = new Uint8Array(this._ttsAnalyser.fftSize);
        const freq = new Uint8Array(this._ttsAnalyser.frequencyBinCount);
        const render = () => {
          if (!this._ttsAnalyser) { this._ttsVisRafId = null; return; }
          this._ttsAnalyser.getByteTimeDomainData(pcm);
          this._ttsAnalyser.getByteFrequencyData(freq);
          let sum = 0;
          for (let i = 0; i < pcm.length; i++) {
            const c = (pcm[i] - 128) / 128;
            sum += c * c;
          }
          const rms = Math.sqrt(sum / pcm.length);
          this._drawVoiceVisuals(pcm, freq, rms);
          this._ttsVisRafId = requestAnimationFrame(render);
        };
        this._ttsVisRafId = requestAnimationFrame(render);
      }
    },

    _stopTTSVisualization() {
      if (this._ttsVisRafId) {
        cancelAnimationFrame(this._ttsVisRafId);
        this._ttsVisRafId = null;
      }
      if (this._ttsSourceNode) {
        try { this._ttsSourceNode.disconnect(); } catch (_) {}
        this._ttsSourceNode = null;
      }
    },

    _stopTTSPlayback() {
      if (this._ttsAudio) {
        this._ttsAudio.pause();
        this._ttsAudio = null;
      }
      this._stopWordHighlighting();
      this._stopTTSVisualization();
      this._ttsSegments = {};
      this._ttsPlaying = false;
      this._ttsNextSegment = 1;
      this._ttsDone = false;
      this._ttsAllB64 = [];
    },

    // ── Speech Response Toggle ────────────────────────────

    _toggleSpeechResponse() {
      this._speechResponse = !this._speechResponse;
      localStorage.setItem('cv2-speech-response', this._speechResponse);
      this.el.speechIcon.textContent = this._speechResponse ? 'volume_off' : 'volume_up';
      this.ws.send({ type: 'update_settings', speech_response: this._speechResponse });
    },

    // ── Inline Voice Bar (show/hide in input area) ─────

    _showInlineVoiceBar(mode) {
      // Save original refs so we can restore them later
      this._origVoiceRefs = {
        voiceWave: this.el.voiceWave,
        voiceNote: this.el.voiceNote,
        voiceAvatarSlot: this.el.voiceAvatarSlot,
        voiceThinking: this.el.voiceThinking,
      };
      // Redirect to inline bar elements
      this.el.voiceWave = this.el.ivbWave;
      this.el.voiceNote = this.el.ivbStatus;
      // Mobile: add voice-active class for big mic button
      const chatRoot = document.getElementById('chat-app');
      if (chatRoot && window.innerWidth <= 600) chatRoot.classList.add('cv2-voice-active-mobile');
      this.el.voiceAvatarSlot = this.el.ivbAvatar;
      this.el.voiceThinking = this.el.ivbThinking;

      // Hide input children via CSS class + hide disclaimer
      this.el.inputWrapper.classList.add('cv2-voice-active');
      if (this.el.disclaimer) this.el.disclaimer.style.display = 'none';

      // Show bar with mode class
      const bar = this.el.ivbBar;
      bar.classList.remove('cv2-ivb-speech', 'cv2-ivb-realtime');
      bar.classList.add(mode === 'realtime' ? 'cv2-ivb-realtime' : 'cv2-ivb-speech');
      bar.style.display = '';
    },

    _hideInlineVoiceBar() {
      // Restore original refs
      if (this._origVoiceRefs) {
        this.el.voiceWave = this._origVoiceRefs.voiceWave;
        this.el.voiceNote = this._origVoiceRefs.voiceNote;
        this.el.voiceAvatarSlot = this._origVoiceRefs.voiceAvatarSlot;
        this.el.voiceThinking = this._origVoiceRefs.voiceThinking;
        this._origVoiceRefs = null;
      }

      // Show input children + disclaimer
      this.el.inputWrapper.classList.remove('cv2-voice-active');
      if (this.el.disclaimer) this.el.disclaimer.style.display = '';

      // Hide bar and remove phase/mode classes
      const bar = this.el.ivbBar;
      bar.style.display = 'none';
      bar.classList.remove('cv2-ivb-speech', 'cv2-ivb-realtime',
        'cv2-speech-idle', 'cv2-speech-recording', 'cv2-speech-processing', 'cv2-speech-responding');

      // Mobile: remove voice-active + phase classes
      const chatRoot = document.getElementById('chat-app');
      if (chatRoot) chatRoot.classList.remove('cv2-voice-active-mobile',
        'cv2-speech-idle', 'cv2-speech-recording', 'cv2-speech-processing', 'cv2-speech-responding');
      if (this.el.mobileMicBtn) this.el.mobileMicBtn.classList.remove('cv2-recording');
    },

    // ── Push-to-Talk ────────────────────────────────────

    _pttPress() {
      if (!this._pushToTalk) return;
      // Realtime (live voice) mode PTT
      if (this._rtActive) {
        // Interrupt AI response if currently speaking
        this._stopRealtimePlayback();
        this.ws.send({
          type: 'rt_send',
          event: { type: 'response.cancel' },
        });
        this._rtMicMuted = false;
        this._setSpeechPhase('recording');
        if (this._rtStatusText) this._rtStatusText.textContent = this.t('chat.realtime_listening');
        return;
      }
      // Speech-input mode PTT
      if (!this._speechMode) return;
      // Don't start if already recording
      if (this._voiceRecorder && this._voiceRecorder.state !== 'inactive') return;
      // Interrupt AI response/TTS if currently speaking
      if (this.streaming || this._ttsAudio || this._ttsPlaying) {
        this._interruptSpeechMode();
      }
      this._setSpeechPhase('recording');
      this.el.voiceNote.textContent = this.t('chat.voice_listening');
      this._startVoiceRecording();
    },

    _pttRelease() {
      if (!this._pushToTalk) return;
      // Realtime (live voice) mode PTT
      if (this._rtActive) {
        this._rtMicMuted = true;
        this._setSpeechPhase('idle');
        if (this._rtStatusText) this._rtStatusText.textContent = this.t('chat.voice_ptt_ready');
        // Commit the audio buffer so the server processes what was said
        this.ws.send({
          type: 'rt_send',
          event: { type: 'input_audio_buffer.commit' },
        });
        return;
      }
      // Speech-input mode PTT
      if (!this._speechMode) return;
      if (!this._voiceRecorder || this._voiceRecorder.state === 'inactive') return;
      this._sendNowVoiceRecording();
    },

    // ── Speech Mode (continuous voice conversation) ─────

    _enterSpeechMode() {
      this._speechMode = true;
      // Enable speech response (auto-TTS) if not already active
      if (!this._speechResponse) {
        this._toggleSpeechResponse();
      }
      // Show inline voice bar instead of overlay popup
      this._showInlineVoiceBar('speech');
      // PTT mode: show idle state, wait for user to press mic/space
      if (this._pushToTalk) {
        this._setSpeechPhase('idle');
        this.el.voiceNote.textContent = this.t('chat.voice_ptt_ready');
        return;
      }
      this._setSpeechPhase('recording');
      // If not already recording (e.g. entered from a non-recording state), start
      if (!this._voiceRecorder || this._voiceRecorder.state === 'inactive') {
        this.el.voiceNote.textContent = this.t('chat.voice_listening');
        this._startVoiceRecording();
      }
    },

    _exitSpeechMode() {
      this._speechMode = false;
      this._stopTTSPlayback();
      this._closeVoicePopup();
      this._setSpeechPhase(null);
    },

    _interruptSpeechMode() {
      this._stopTTSPlayback();
      this._hideVoiceThinking();
      // Immediately switch to idle so UI updates (hide interrupt, show mic)
      this._setSpeechPhase('idle');
      this.el.voiceNote.textContent = this.t('chat.voice_ptt_ready');
      if (this.streaming) {
        this.ws.send({ type: 'stop_streaming' });
        // handleResponseCancelled will call _speechModeNextCycle
      } else {
        this._speechModeNextCycle();
      }
    },

    _setSpeechPhase(phase) {
      const phases = ['cv2-speech-idle', 'cv2-speech-recording', 'cv2-speech-processing', 'cv2-speech-responding'];
      // Apply to inline voice bar
      const bar = this.el.ivbBar;
      if (bar) {
        bar.classList.remove(...phases);
        if (phase) bar.classList.add(`cv2-speech-${phase}`);
      }
      // Apply to #chat-app so CSS can target mobile mic visibility by phase
      const chatRoot = document.getElementById('chat-app');
      if (chatRoot) {
        chatRoot.classList.remove(...phases);
        if (phase) chatRoot.classList.add(`cv2-speech-${phase}`);
      }
      // Also apply to legacy popup elements (backward compat)
      const el = this.el.voiceModeActive;
      if (el) {
        el.classList.remove(...phases);
        if (phase) el.classList.add(`cv2-speech-${phase}`);
      }
      const card = this.el.voicePopup?.querySelector('.cv2-voice-card');
      if (card) {
        card.classList.remove(...phases);
        if (phase) card.classList.add(`cv2-speech-${phase}`);
      }
      this._updateVoiceAvatar(phase);
      // Clear waveform canvas when idle (no stale waveform visible)
      if (phase === 'idle') {
        const canvas = this.el.voiceWave;
        if (canvas) {
          const ctx = canvas.getContext('2d');
          if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
      }
    },

    _updateVoiceAvatar(phase) {
      const slot = this.el.voiceAvatarSlot;
      const title = this.el.voiceTitle;
      if (!slot) return;
      if (phase === 'idle' || phase === 'recording' || phase === 'processing') {
        // Show user avatar
        const name = this.fullName || this.userName || '';
        const initials = name.split(/[\s,]+/).map(w => (w[0] || '')).join('').substring(0, 2).toUpperCase() || '?';
        const photoUrl = this.config.userAvatar || '';
        if (photoUrl) {
          slot.innerHTML = '<img class="cv2-voice-avatar" src="' + photoUrl + '" alt="">';
        } else {
          slot.innerHTML = '<div class="cv2-voice-avatar cv2-voice-avatar-initials">' + initials + '</div>';
        }
        if (title) {
          const icon = phase === 'recording' ? 'mic' : 'hourglass_top';
          title.innerHTML = '<span class="material-icons">' + icon + '</span> ' + (phase === 'recording' ? this.t('chat.voice_listening') : this.t('chat.voice_transcribing'));
        }
      } else if (phase === 'responding') {
        // Show AI avatar (mascot or default)
        if (this.config.appMascot) {
          slot.innerHTML = '<img class="cv2-voice-avatar" src="' + this.config.appMascot + '" alt="">';
        } else {
          slot.innerHTML = '<div class="cv2-voice-avatar cv2-voice-avatar-initials">AI</div>';
        }
        if (title) title.innerHTML = '<span class="material-icons">volume_up</span> ' + (this.t('chat.speech_response') || 'Speaking');
      } else {
        slot.innerHTML = '';
        if (title) title.innerHTML = '<span class="material-icons">mic</span> ' + this.t('chat.voice_title');
      }
    },

    _speechModeNextCycle() {
      if (!this._speechMode) return;
      if (this._pushToTalk) {
        this._setSpeechPhase('idle');
        this.el.voiceNote.textContent = this.t('chat.voice_ptt_ready');
        return;
      }
      this._setSpeechPhase('recording');
      this.el.voiceNote.textContent = this.t('chat.voice_listening');
      this._startVoiceRecording();
    },

    // ── Speech helpers ────────────────────────────────────

    _isSpeakable(text) {
      if (!text || text.length > (this._speechMaxTokens || 2000) * 4) return false;
      // Check if code blocks dominate (>50% of text)
      const codeBlockRe = /```[\s\S]*?```/g;
      let codeLen = 0;
      let m;
      while ((m = codeBlockRe.exec(text)) !== null) codeLen += m[0].length;
      if (codeLen > text.length * 0.5) return false;
      return true;
    },

    _cleanTextForSpeech(text) {
      text = text.replace(/```[\s\S]*?```/g, '');              // remove code blocks
      text = text.replace(/`[^`]+`/g, '');                      // remove inline code
      text = text.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');      // markdown links → label
      text = text.replace(/https?:\/\/\S+/g, '');               // bare URLs
      text = text.replace(/#{1,6}\s/g, '');                      // headers
      text = text.replace(/\*{1,2}([^*]+)\*{1,2}/g, '$1');      // bold/italic
      text = text.replace(/<[^>]+>/g, '');                       // HTML tags
      text = text.replace(/\s+/g, ' ').trim();
      if (text.length > 8000) text = text.substring(0, 8000) + '...';
      return text;
    },

    _speakMessage(rawText, btn) {
      // Stop any existing TTS
      this._stopTTSPlayback();

      const cleaned = this._cleanTextForSpeech(rawText);
      if (!cleaned) return;

      // Show loading state
      const origHtml = btn.innerHTML;
      btn.innerHTML = '<span class="material-icons cv2-spin" style="font-size:16px">sync</span>';
      btn.disabled = true;

      // Send to existing TTS handler
      this.ws.send({ type: 'tts', text: cleaned });

      // Listen for response — the existing handleTTSAudio will play it
      // Restore button after audio plays or on error
      const restore = () => {
        btn.innerHTML = origHtml;
        btn.disabled = false;
      };
      // Set a timeout fallback in case no audio comes back
      const timeout = setTimeout(restore, 30000);

      // Hook into the next tts_audio to clear timeout
      const origHandler = this.handleTTSAudio.bind(this);
      this.handleTTSAudio = (msg) => {
        clearTimeout(timeout);
        this.handleTTSAudio = origHandler;
        origHandler(msg);
        // Restore button when audio ends
        if (this._ttsAudio) {
          this._ttsAudio.addEventListener('ended', restore);
        } else {
          restore();
        }
      };
    },

  });

  ChatFeatures.register('voice', {
    initState(app) {
      // Voice/Speech state — always start disabled; user opts in via toggle or speech mode
      app._speechResponse = false;
      app._speechMode = false;
      app._enableVoiceInput = app.config.enableVoiceInput !== false;   // default true
      app._enableLiveVoice = app.config.enableLiveVoice !== false;     // default true
      app._pushToTalk = localStorage.getItem('cv2-push-to-talk') !== 'false'; // default: true
      app._pttKeyDown = false;
      app._ttsVoice = localStorage.getItem('cv2-tts-voice') || '';
      app._voiceSilenceMs = parseInt(localStorage.getItem('cv2-voice-silence-ms') || '1500', 10);
      app._voiceSilenceThreshold = parseFloat(localStorage.getItem('cv2-voice-silence-threshold') || '0.012');
      app._voiceRecorder = null;
      app._voiceStream = null;
      app._voiceAudioCtx = null;
      app._voiceAnalyser = null;
      app._voiceSourceNode = null;
      app._voiceChunks = [];
      app._voiceRafId = null;
      app._voiceElapsedTimer = null;
      app._voiceSilenceTimer = null;
      app._voiceStartedAt = 0;
      app._voiceLastVoiceAt = 0;
      app._voiceLoudnessHistory = [];
      app._ttsAudio = null;
      app._ttsRafId = null;
      app._ttsWordTimings = [];
      app._ttsSegments = {};
      app._ttsNextSegment = 1;
      app._ttsPlaying = false;
      app._ttsDone = false;
      app._ttsAllB64 = [];
      app._ttsAudioCtx = null;
      app._ttsSourceNode = null;
      app._ttsAnalyser = null;
      app._ttsVisRafId = null;
    },
    handleMessage: {
      'transcription_result': 'handleTranscriptionResult',
      'tts_audio': 'handleTTSAudio',
      'tts_done': 'handleTTSDone',
    },
  });
})();
