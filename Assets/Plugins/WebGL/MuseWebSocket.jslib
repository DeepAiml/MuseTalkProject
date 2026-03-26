mergeInto(LibraryManager.library, {

  // Resume all audio contexts to unlock browser audio
  UnlockAudio: function () {
    try {
      if (typeof unityAudioContext !== "undefined" && unityAudioContext.state !== "running") {
        unityAudioContext.resume();
      }
      var state = Module.MuseWebSocketState;
      if (state) {
        if (state.micAudioCtx && state.micAudioCtx.state !== "running") {
          state.micAudioCtx.resume();
        }
        if (state.playbackAudioCtx && state.playbackAudioCtx.state !== "running") {
          state.playbackAudioCtx.resume();
        }
      }
    } catch (e) {
      console.error("UnlockAudio failed:", e);
    }
  },

  MuseWebSocketConnect: function (urlPtr, gameObjectPtr) {
    var url = UTF8ToString(urlPtr);
    var go = UTF8ToString(gameObjectPtr);

    // Convert PCM16 bytes to Float32 and send to playback worklet
    function processAudioPacket(bytes) {
      var state = Module.MuseWebSocketState;
      if (!state || !state.playbackStarted || !state.playbackNode) return;

      try {
        var pcmOffset = 1;
        var pcmBytes = bytes.length - pcmOffset;
        if (pcmBytes <= 0) return;

        var sampleCount = Math.floor(pcmBytes / 2);
        var samples = new Float32Array(sampleCount);
        
        for (var i = 0; i < sampleCount; i++) {
          var idx = pcmOffset + (i * 2);
          var pcm16 = bytes[idx] | (bytes[idx + 1] << 8);
          if (pcm16 >= 32768) pcm16 -= 65536;
          samples[i] = pcm16 / 32768.0;
        }

        state.playbackNode.port.postMessage({
          samples: samples,
          maxSize: state.maxBufferSize
        });
      } catch (e) {
        console.error("ProcessAudioPacket failed:", e);
      }
    }

    // Handle WebSocket reconnection with exponential backoff
    function attemptReconnect(url, go) {
      var state = Module.MuseWebSocketState;
      if (!state.reconnectionEnabled) return;

      if (!state.reconnectAttempts[url]) {
        state.reconnectAttempts[url] = 0;
      }
      state.reconnectAttempts[url]++;

      if (state.reconnectAttempts[url] > state.maxReconnectAttempts) {
        console.error("Max reconnection attempts reached for " + url);
        SendMessage(go, "OnWebSocketError", "max_reconnect_attempts_reached");
        return;
      }

      var delay = Math.min(
        state.reconnectDelay * Math.pow(2, state.reconnectAttempts[url] - 1),
        state.maxReconnectDelay
      );

      if (state.reconnectTimeouts[url]) {
        clearTimeout(state.reconnectTimeouts[url]);
      }

      state.reconnectTimeouts[url] = setTimeout(function() {
        _MuseWebSocketConnect(
          allocate(intArrayFromString(url), ALLOC_NORMAL),
          allocate(intArrayFromString(go), ALLOC_NORMAL)
        );
      }, delay);
    }

    // Initialize global state
    if (!Module.MuseWebSocketState) {
      Module.MuseWebSocketState = {
        ws: null,
        gameObjectName: null,
        // Microphone
        micOn: false,
        micAudioCtx: null,
        micStream: null,
        micSource: null,
        micNode: null,
        workletLoaded: false,
        micPacketsSent: 0,
        micPacketsBlocked: 0,
        // Audio Playback
        playbackAudioCtx: null,
        playbackWorkletLoaded: false,
        playbackNode: null,
        audioBuffer: [],
        isAIPlaying: false,
        lastAudioTime: 0,
        audioPacketsReceived: 0,
        bufferUnderruns: 0,
        maxBufferSize: 48000,
        minBufferSize: 3200,
        playbackStarted: false,
        // Reconnection
        reconnectionEnabled: true,
        reconnectAttempts: {},
        maxReconnectAttempts: 10,
        reconnectDelay: 1000,
        maxReconnectDelay: 30000,
        socketUrls: {},
        socketGameObjects: {},
        reconnectTimeouts: {}
      };
    }

    var state = Module.MuseWebSocketState;
    state.gameObjectName = go;

    try {
      var ws = new WebSocket(url);
      ws.binaryType = "arraybuffer";
      state.ws = ws;

      ws.onopen = function () {
        _InitializeAudioPlayback();
        SendMessage(go, "OnWebSocketConnected", "");
      };

      ws.onclose = function (event) {
        SendMessage(go, "OnWebSocketClosed", "");
        
        if (state.playbackNode) {
          try { state.playbackNode.disconnect(); } catch (e) {}
          state.playbackNode = null;
        }
        state.playbackStarted = false;

        if (event.code !== 1000) {
          attemptReconnect(url, go);
        }
      };

      ws.onerror = function (err) {
        console.error("WebSocket error:", err);
        SendMessage(go, "OnWebSocketError", "error");
      };

      ws.onmessage = function (e) {
        if (!(e.data instanceof ArrayBuffer)) return;
        var bytes = new Uint8Array(e.data);
        if (bytes.length === 0) return;

        var packetType = bytes[0];

        // Audio packets (AI speaking)
        if (packetType === 0x02 || packetType === 0x03) {
          state.audioPacketsReceived++;
          state.isAIPlaying = true;
          state.lastAudioTime = Date.now();
          processAudioPacket(bytes);
        } else if (packetType === 0x12) {
          state.isAIPlaying = false;
        }

        // Forward all packets to Unity
        var ptr = _malloc(bytes.length);
        HEAPU8.set(bytes, ptr);
        SendMessage(go, "OnWebSocketBinary", ptr + "," + bytes.length);
        _free(ptr);
      };

      return 1;
    } catch (e) {
      console.error("WebSocket connection failed:", e);
      attemptReconnect(url, go);
      return 0;
    }
  },

  MuseWebSocketSetReconnection: function(enabled) {
    var state = Module.MuseWebSocketState;
    if (state) {
      state.reconnectionEnabled = !!enabled;
    }
  },

  MuseWebSocketReconnect: function() {
    var state = Module.MuseWebSocketState;
    if (!state) return 0;
    
    var url = Object.keys(state.socketUrls)[0];
    var go = state.socketGameObjects[url];
    
    if (url && go) {
      state.reconnectAttempts[url] = 0;
      _MuseWebSocketConnect(
        allocate(intArrayFromString(url), ALLOC_NORMAL),
        allocate(intArrayFromString(go), ALLOC_NORMAL)
      );
      return 1;
    }
    return 0;
  },
  
  InitializeAudioPlayback: function() {
    var state = Module.MuseWebSocketState;
    if (!state) {
      console.error("MuseWebSocketState not initialized");
      return 0;
    }

    try {
      if (!state.playbackAudioCtx) {
        state.playbackAudioCtx = new (window.AudioContext || window.webkitAudioContext)({ 
          sampleRate: 22050,
          latencyHint: 'interactive'
        });
      }

      state.playbackAudioCtx.resume().catch(function() {});

      if (!state.playbackWorkletLoaded) {
        var code = [
          'class PlaybackProc extends AudioWorkletProcessor {',
          '  constructor() {',
          '    super();',
          '    this.buffer = [];',
          '    this.underruns = 0;',
          '    this.speedMultiplier = 1;',
          '    this.readPosition = 0.0;',
          '    this.port.onmessage = (e) => {',
          '      if (e.data && e.data.samples) {',
          '        this.buffer.push(...e.data.samples);',
          '        if (this.buffer.length > e.data.maxSize) {',
          '          this.buffer.splice(0, this.buffer.length - e.data.maxSize);',
          '        }',
          '      }',
          '      if (e.data && e.data.speedMultiplier !== undefined) {',
          '        this.speedMultiplier = e.data.speedMultiplier;',
          '      }',
          '    };',
          '  }',
          '  process(inputs, outputs) {',
          '    var channel = outputs[0] && outputs[0][0];',
          '    if (!channel) return true;',
          '    for (var i = 0; i < channel.length; i++) {',
          '      var idx = Math.floor(this.readPosition);',
          '      channel[i] = idx < this.buffer.length ? this.buffer[idx] : 0;',
          '      this.readPosition += this.speedMultiplier;',
          '    }',
          '    var consumed = Math.floor(this.readPosition);',
          '    if (consumed > 0) {',
          '      this.buffer.splice(0, consumed);',
          '      this.readPosition -= consumed;',
          '    }',
          '    return true;',
          '  }',
          '}',
          'registerProcessor("playback-proc", PlaybackProc);'
        ].join('\n');
        
        var blob = new Blob([code], { type: "application/javascript" });
        var blobURL = URL.createObjectURL(blob);
        
        state.playbackAudioCtx.audioWorklet.addModule(blobURL)
          .then(function() {
            state.playbackWorkletLoaded = true;
            URL.revokeObjectURL(blobURL);
            
            state.playbackNode = new AudioWorkletNode(state.playbackAudioCtx, "playback-proc");
            state.playbackNode.connect(state.playbackAudioCtx.destination);
            state.playbackStarted = true;
            
            state.playbackAudioCtx.resume().catch(function() {});
          })
          .catch(function(err) {
            console.error("Playback AudioWorklet load failed:", err);
          });
      }

      return 1;
    } catch (e) {
      console.error("InitializeAudioPlayback failed:", e);
      return 0;
    }
  },

  GetAudioBufferSize: function() {
    var state = Module.MuseWebSocketState;
    return (state && state.audioBuffer) ? state.audioBuffer.length : 0;
  },

  ClearAudioBuffer: function() {
    var state = Module.MuseWebSocketState;
    if (!state) return;
    state.audioBuffer = [];
    state.audioPacketsReceived = 0;
    state.bufferUnderruns = 0;
  },

  MuseWebSocketSendText: function (textPtr) {
    var state = Module.MuseWebSocketState;
    if (!state || !state.ws || state.ws.readyState !== 1) return 0;
    try {
      state.ws.send(UTF8ToString(textPtr));
      return 1;
    } catch (e) {
      console.error("Send text failed:", e);
      return 0;
    }
  },

  MuseWebSocketSendBinary: function (ptr, len) {
    var state = Module.MuseWebSocketState;
    if (!state || !state.ws || state.ws.readyState !== 1) return 0;
    try {
      var data = HEAPU8.slice(ptr, ptr + len);
      state.ws.send(data.buffer);
      return 1;
    } catch (e) {
      console.error("Send binary failed:", e);
      return 0;
    }
  },

  MuseWebSocketClose: function () {
    var state = Module.MuseWebSocketState;
    if (state && state.ws) {
      state.ws.close();
      state.ws = null;
    }
  },

  MuseWebSocketGetState: function () {
    var state = Module.MuseWebSocketState;
    return (state && state.ws) ? state.ws.readyState : 3;
  },

  MuseToggleMic: function (enable) {
    var state = Module.MuseWebSocketState;
    if (!state) {
      console.error("MuseWebSocketState not initialized");
      return;
    }

    // Create AudioContext if needed
    if (!state.micAudioCtx) {
      try {
        state.micAudioCtx = new (window.AudioContext || window.webkitAudioContext)({
          sampleRate: 22050
        });
      } catch (e) {
        console.error("Failed to create AudioContext:", e);
        SendMessage(state.gameObjectName, "OnMicrophoneError", "audio_context_failed");
        return;
      }
    }

    state.micAudioCtx.resume().then(function() {
      // Load worklet once
      if (!state.workletLoaded) {
        var code = [
          'class MicProc extends AudioWorkletProcessor {',
          '  constructor() {',
          '    super();',
          '    this.micGain = 0.7;',
          '  }',
          '  process(inputs) {',
          '    var ch = inputs[0] && inputs[0][0];',
          '    if (!ch || ch.length === 0) return true;',
          '    var pcm = new Int16Array(ch.length);',
          '    for (var i = 0; i < ch.length; i++) {',
          '      var sample = ch[i] * this.micGain;',
          '      var s = Math.max(-1, Math.min(1, sample));',
          '      pcm[i] = s < 0 ? s * 32768 : s * 32767;',
          '    }',
          '    this.port.postMessage(pcm);',
          '    return true;',
          '  }',
          '}',
          'registerProcessor("mic-proc", MicProc);'
        ].join('\n');
        
        var blob = new Blob([code], { type: "application/javascript" });
        var blobURL = URL.createObjectURL(blob);
        
        state.micAudioCtx.audioWorklet.addModule(blobURL)
          .then(function() {
            state.workletLoaded = true;
            URL.revokeObjectURL(blobURL);
            if (enable) {
              setTimeout(function() { _MuseToggleMic(1); }, 100);
            }
          })
          .catch(function(err) {
            console.error("Mic AudioWorklet load failed:", err);
            SendMessage(state.gameObjectName, "OnMicrophoneError", "worklet_failed");
          });
        return;
      }

      // Start microphone
      if (enable && !state.micOn) {
        state.micPacketsSent = 0;
        state.micPacketsBlocked = 0;

        navigator.mediaDevices.getUserMedia({ 
          audio: { 
            echoCancellation: false, 
            noiseSuppression: false, 
            autoGainControl: false,
            sampleRate: 22050
          } 
        })
        .then(function(stream) {
          state.micStream = stream;
          state.micSource = state.micAudioCtx.createMediaStreamSource(stream);
          state.micNode = new AudioWorkletNode(state.micAudioCtx, "mic-proc");

          state.micNode.port.onmessage = function(e) {
            var pcm = e.data;
            
            // Block mic while AI is speaking
            if (state.isAIPlaying && (Date.now() - state.lastAudioTime < 300)) {
              state.micPacketsBlocked++;
              return;
            }
            
            if (Date.now() - state.lastAudioTime >= 300 && state.isAIPlaying) {
              state.isAIPlaying = false;
            }
            
            if (!state.ws || state.ws.readyState !== 1) return;

            // Voice activity detection
            var sum = 0;
            for (var i = 0; i < pcm.length; i++) {
              sum += Math.abs(pcm[i]);
            }
            if (sum / pcm.length < 500) return;

            // Send audio packet
            var buf = new Uint8Array(1 + pcm.byteLength);
            buf[0] = 0x10;
            buf.set(new Uint8Array(pcm.buffer), 1);
            
            try {
              state.ws.send(buf.buffer);
              state.micPacketsSent++;
            } catch (err) {
              console.error("Failed to send audio:", err);
            }
          };

          state.micSource.connect(state.micNode);
          state.micOn = true;
          state.isAIPlaying = false;
          SendMessage(state.gameObjectName, "OnMicrophoneStarted", "");
        })
        .catch(function(err) {
          console.error("Microphone access denied:", err);
          SendMessage(state.gameObjectName, "OnMicrophoneError", err.message || "permission_denied");
        });
      }
      // Stop microphone
      else if (!enable && state.micOn) {
        if (state.micStream) {
          state.micStream.getTracks().forEach(function(t) { t.stop(); });
        }
        if (state.micSource) state.micSource.disconnect();
        if (state.micNode) state.micNode.disconnect();
        
        state.micStream = null;
        state.micSource = null;
        state.micNode = null;
        state.micOn = false;
        state.isAIPlaying = false;
        
        SendMessage(state.gameObjectName, "OnMicrophoneStopped", "");
      }
    }).catch(function(err) {
      console.error("AudioContext resume failed:", err);
      SendMessage(state.gameObjectName, "OnMicrophoneError", "context_resume_failed");
    });
  },

  MuseIsMicOn: function () {
    var state = Module.MuseWebSocketState;
    return (state && state.micOn) ? 1 : 0;
  }
});

// Token expiry bridge
var LibraryTokenBridge = {
  OnTokenExpired: function () {
    if (window.dispatchEvent) {
      window.dispatchEvent(new CustomEvent('unity-token-expired'));
    }
    if (typeof WS !== 'undefined' && WS.onTokenExpired) {
      WS.onTokenExpired();
    }
  }
};
mergeInto(LibraryManager.library, LibraryTokenBridge);

// Speak completion bridge
var LibrarySpeakBridge = {
  OnSpeakRequestCompleted: function () {
    if (window.dispatchEvent) {
      window.dispatchEvent(new CustomEvent('unity-speak-complete'));
    }
    if (typeof WS !== 'undefined' && WS.onSpeakCompleted) {
      WS.onSpeakCompleted();
    }
  }
};
mergeInto(LibraryManager.library, LibrarySpeakBridge);

// Socket event dispatchers
mergeInto(LibraryManager.library, {
  DispatchSocketConnectedEvent: function() {
    window.dispatchEvent(new Event('unity-socket-connected'));
  },
  
  DispatchSocketDisconnectedEvent: function(messagePtr) {
    var message = UTF8ToString(messagePtr);
    window.dispatchEvent(new CustomEvent('unity-socket-disconnected', {
      detail: { message: message }
    }));
  }
});