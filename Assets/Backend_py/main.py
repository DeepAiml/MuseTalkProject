session_state = {}

# PyTorch 2.1 compatibility patch for transformers
import queue
import threading
import torch.utils._pytree as _pytree

from utils.websocket_helper import start_websocket_heartbeat, stop_websocket_heartbeat
if not hasattr(_pytree, 'register_pytree_node'):
    _pytree.register_pytree_node = _pytree._register_pytree_node
import ssl  
import base64
from concurrent.futures import ThreadPoolExecutor
import re
from types import NoneType
import pysbd
from scipy import signal
import sys
import os
import tempfile
import traceback  
import wave
import io
import torchaudio
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import soundfile as sf
from fastapi import Request
import websockets
from g2p_en import G2p
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from typing import Tuple, AsyncGenerator, Optional, Dict, Any
from multiprocessing import cpu_count
import nltk
import asyncio
import librosa
from faster_whisper import WhisperModel
from TTS.api import TTS
from TTS.utils.synthesizer import Synthesizer
import TTS.utils.io
import warnings
import logging
import time
import numpy as np
import torch
import noisereduce as nr
import hashlib
import torch
import torch.nn.functional as F
from enum import Enum
from typing import Optional
import math
from functools import lru_cache
import time
from integrated_s2s_system import (
    IntegratedSpeechToSpeechHandler,
    integrate_s2s_with_websocket
)
import os
import json
import asyncio
import websockets
from dotenv import load_dotenv
import base64
import hashlib
import logging
from typing import Tuple, AsyncGenerator
import numpy as np
from io import BytesIO
import soundfile as sf
import time
import websockets
import ssl
import json
import ssl
import threading
import websocket
from elevenlabs import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface
load_dotenv()
# =============================================================================
# SAM SAGILITY WEBSOCKET CONFIGURATION
# =============================================================================


SAM_WS_BASE_URL = "wss://localhost:12345/api/ws"
client_llm_tokens = {}
client_session_ids: Dict[str, str] = {}  # Add this with other token storage
client_token_storage = {}  # {client_id: TokenStatus}
# =============================================================================
# PAUSE TOKEN SYSTEM
# =============================================================================
class SAMLLMClient:
    def __init__(self, base_url, session_id, access_token):
        self.base_url = base_url
        self.session_id = session_id
        self.access_token = access_token
        self.ws = None
        self.full_response = ""
        
        

        # ─── used by send_query to block until "done" arrives ───
        self._response_ready = threading.Event()
        self._ws_ready = threading.Event()   # fires once connection is open
        self._error = None                   # stores any error string from server

    # ──────────────────────────────────────────────────────────────────────────
    # WebSocketApp callbacks
    # ──────────────────────────────────────────────────────────────────────────

    def on_open(self, ws):
        print(" Connected to SAM successfully!\n")
        self._ws_ready.set()          # unblock send_query if it's waiting

    def on_message(self, ws, message):
        try:
            data = json.loads(message)

            if data.get("type") == "token":
                token = data.get("data", "")
                self.full_response += token


            elif data.get("type") == "done":
                print("\n" + "-" * 60)
                print(" Stream completed\n")
                self._response_ready.set()          # unblock send_query

            elif "error" in data:
                self._error = data["error"]
                self._response_ready.set()          # unblock so we don't hang

        except json.JSONDecodeError:
            print(f"Non-JSON message: {message}")

    def on_error(self, ws, error):
        print(f" WebSocket error: {error}")
        self._error = str(error)
        self._response_ready.set()    

    def on_close(self, ws, close_status_code, close_msg):
        print(f"🔌 Connection closed – code: {close_status_code}")
        if not self._response_ready.is_set():
            self._error = "Connection closed before response completed"
            self._response_ready.set()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def connect(self):
        url = f"{self.base_url}/query-stream?session_id={self.session_id}"
        print(f"Connecting to: {url}")

        self.ws = websocket.WebSocketApp(
            url,
            header={"Authorization": f"Bearer {self.access_token}"},
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )

        # Run WebSocket in background thread
        ws_thread = threading.Thread(
            target=self.ws.run_forever,
            kwargs={"sslopt": {"cert_reqs": ssl.CERT_NONE}},
            daemon=True,
        )
        ws_thread.start()

        # Wait until on_open fires (max 10 s)
        if not self._ws_ready.wait(timeout=10):
            raise ConnectionError("SAM WebSocket did not connect within 10 s")

    def send_query(self, query: str, timeout: float = 30.0) -> str:
        if not self.ws:
            raise ConnectionError("Not connected – call connect() first")

        # Reset state for this query
        # self.current_token_count = 0
        self.full_response = ""
        self._error = None
        self._response_ready.clear()

        # Send the query
        message = {"query": query}
        print(f" Query: {query}\n")
        print(" Response:")
        print("-" * 60)
        self.ws.send(json.dumps(message))

        # Block until on_message sets the event (or timeout)
        finished = self._response_ready.wait(timeout=timeout)

        if not finished:
            raise TimeoutError(f"SAM did not respond within {timeout}s")

        if self._error:
            raise RuntimeError(f"SAM error: {self._error}")

        return self.full_response

    def close(self):
        if self.ws:
            self.ws.close()

sam_clients = {}  

async def generate_llm_response_streaming_sam_with_fallback(
    user_input: str,
    conversation_history: list,
    tts_callback,
    emotion: str = "neutral",
    interruption_check=None,
    client_id: str = None,
    sam_timeout: float = 6.0
) -> Tuple[str, str, str]:
    """
     ENHANCED: SAM → Groq with live interruption capability
    """
    
    # Shared state for coordinating SAM and Groq
    response_ready = asyncio.Event()
    sam_response = {"text": "", "emotion": emotion, "role": "assistant", "source": None}
    groq_task = None
    sam_task = None
    
    # Check if we have SAM credentials
    token = get_client_llm_token(client_id)
    session_id = get_client_session_id(client_id)
    
    has_sam = bool(token and session_id)
    
    async def sam_task_func():
        """Try SAM WebSocket"""
        nonlocal sam_response
        
        try:
            logger.info(f"🔵 Starting SAM WebSocket for {client_id}")
            
            if client_id not in sam_clients:
                sam_clients[client_id] = SAMLLMClient(
                    SAM_WS_BASE_URL, 
                    session_id, 
                    token
                )
            
            sam_client = sam_clients[client_id]
            
            if not sam_client.connected:
                await sam_client.connect()
            
            # Prepare query
            context = "\n".join([
                f"User: {msg.get('User', '')}" if 'User' in msg 
                else f"Assistant: {msg.get('Assistant', '')}"
                for msg in conversation_history[-6:]
            ])
            
            full_query = f"{context}\nUser: {user_input}" if context else user_input
            
            # Call SAM with timeout
            response_text, detected_emotion, role = await asyncio.wait_for(
                sam_client.send_query(
                    query=full_query,
                    tts_callback=tts_callback,
                    interruption_check=interruption_check,
                    timeout=sam_timeout
                ),
                timeout=sam_timeout + 0.5
            )
            
            if response_text and len(response_text.strip()) > 0:
                sam_response["text"] = response_text
                sam_response["emotion"] = detected_emotion or emotion
                sam_response["role"] = role
                sam_response["source"] = "sam"
                
                logger.info(f" SAM response ready: {len(response_text)} chars")
                response_ready.set()
                
                #  CRITICAL: Cancel Groq task if it's running
                if groq_task and not groq_task.done():
                    groq_task.cancel()
                    logger.info(f"🛑 Cancelled Groq task - SAM responded")
                
                return True
            else:
                logger.warning(f"SAM returned empty response")
                return False
                
        except asyncio.TimeoutError:
            logger.warning(f" SAM timeout after {sam_timeout}s")
            return False
        except Exception as e:
            logger.error(f" SAM error: {e}")
            return False
    
    async def groq_task_func():
        """Fallback to Groq"""
        nonlocal sam_response
        
        try:
            # Wait a bit to give SAM a chance
            await asyncio.sleep(0.8)
            
            # Check if SAM already responded
            if response_ready.is_set():
                logger.info(f"Groq skipped - SAM already responded")
                return
            logger.info(f"🟢 Starting Groq LLM fallback")
            
            from utils.localllm import generate_llm_response_streaming
            
            async def groq_tts_callback(phrase: str, phrase_emotion: dict, phrase_idx: int):
                # Check if SAM has responded
                if response_ready.is_set():
                    logger.info(f"🛑 Groq phrase #{phrase_idx} skipped - SAM ready")
                    raise asyncio.CancelledError("SAM interrupted")
                
                # Forward to actual TTS callback
                await tts_callback(phrase, phrase_emotion, phrase_idx)
            
            response_text, llm_emotion, role = await generate_llm_response_streaming(
                user_input=user_input,
                conversation_history=conversation_history[-6:],
                tts_callback=groq_tts_callback,
                emotion=emotion,
                interruption_check=interruption_check
            )
            
            # Check one more time before committing
            if response_ready.is_set():
                logger.info(f"Groq response discarded - SAM won")
                return
            
            if response_text and len(response_text.strip()) > 0:
                sam_response["text"] = response_text
                sam_response["emotion"] = llm_emotion or emotion
                sam_response["role"] = role
                sam_response["source"] = "groq"
                
                logger.info(f" Groq response ready: {len(response_text)} chars")
                response_ready.set()
                
        except asyncio.CancelledError:
            logger.info(f"🛑 Groq task cancelled (SAM won)")
        except Exception as e:
            logger.error(f" Groq error: {e}")
    
    #  RACE: Start both tasks if SAM available, otherwise just Groq
    if has_sam:
        logger.info(f"🏁 Starting SAM + Groq race")
        sam_task = asyncio.create_task(sam_task_func())
        groq_task = asyncio.create_task(groq_task_func())
        
        # Wait for first to complete
        done, pending = await asyncio.wait(
            [sam_task, groq_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    else:
        # No SAM - just use Groq
        logger.info(f"🟢 Using Groq only (no SAM credentials)")
        await groq_task_func()
    
    # Wait a bit for response to be set
    try:
        await asyncio.wait_for(response_ready.wait(), timeout=0.5)
    except asyncio.TimeoutError:
        pass
    
    # Return result
    if sam_response["text"]:
        source = sam_response.get("source", "unknown")
        logger.info(f" Returning {source.upper()} response: {len(sam_response['text'])} chars")
        
        return (
            sam_response["text"],
            sam_response["emotion"],
            sam_response["role"]
        )
    else:
        logger.error(f" No response from either SAM or Groq")
        return (
            "I'm experiencing technical difficulties. Please try again.",
            "concerned",
            "assistant"
        )
class PauseToken:
    """Represents a pause in the text"""
    def __init__(self, duration: float, position: int):
        self.duration = duration  
        self.position = position  
    
    def __repr__(self):
        return f"PauseToken(duration={self.duration}s, pos={self.position})"

def parse_pause_tokens(text: str) -> Tuple[str, List[PauseToken]]:
    """
    Parse pause tokens from text and return clean text + pause list
    
    Supported formats:
    - <pause:1.5> - Pause for 1.5 seconds
    - <pause:0.5> - Pause for 0.5 seconds
    - <break:2.0> - Alternative syntax (same as pause)
    
    Example:
        Input:  "Hello <pause:1.0> how are you <pause:0.5> today?"
        Output: ("Hello  how are you  today?", [PauseToken(1.0, 6), PauseToken(0.5, 20)])
    
    Args:
        text: Text containing pause tokens
    
    Returns:
        tuple: (cleaned_text, list_of_pause_tokens)
    """
    
    # Pattern to match pause tokens: <pause:NUMBER> or <break:NUMBER>
    pause_pattern = r'<(?:pause|break):(\d+(?:\.\d+)?)>'
    
    pauses = []
    cleaned_text = text
    offset = 0  # Track position offset as we remove tokens
    
    for match in re.finditer(pause_pattern, text):
        duration_str = match.group(1)
        try:
            duration = float(duration_str)
            
            # Clamp duration to reasonable range (0.1s to 10s)
            duration = max(0.1, min(10.0, duration))
            
            # Calculate position in cleaned text (accounting for previous removals)
            position = match.start() - offset
            
            pauses.append(PauseToken(duration, position))
            
            # Remove the token from text
            cleaned_text = cleaned_text[:match.start() - offset] + cleaned_text[match.end() - offset:]
            
            # Update offset for next iteration
            offset += len(match.group(0))
            
            logger.debug(f"📍 Found pause: {duration}s at position {position}")
            
        except ValueError:
            logger.warning(f"Invalid pause duration: {duration_str}")
            continue
    
    # Clean up any double spaces created by token removal
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text).strip()
    
    return cleaned_text, pauses

def split_text_by_pauses(text: str, pauses: List[PauseToken]) -> List[Tuple[str, float]]:
    """
    Split text into segments based on pause tokens
    
    Args:
        text: Cleaned text (without pause tokens)
        pauses: List of PauseToken objects
    
    Returns:
        list: [(segment_text, pause_after_duration), ...]
    
    Example:
        Input:  "Hello how are you today?", [PauseToken(1.0, 5), PauseToken(0.5, 18)]
        Output: [("Hello", 1.0), ("how are you", 0.5), ("today?", 0.0)]
    """
    
    if not pauses:
        return [(text, 0.0)]
    
    # Sort pauses by position
    sorted_pauses = sorted(pauses, key=lambda p: p.position)
    
    segments = []
    last_pos = 0
    
    for pause in sorted_pauses:
        # Extract segment before this pause
        segment = text[last_pos:pause.position].strip()
        
        if segment:  # Only add non-empty segments
            segments.append((segment, pause.duration))
        
        last_pos = pause.position
    
    # Add final segment (after last pause)
    final_segment = text[last_pos:].strip()
    if final_segment:
        segments.append((final_segment, 0.0))  # No pause after last segment
    
    return segments

async def apply_pause(duration: float, client_id: str, interaction_id: str, 
                      interruption_manager) -> bool:

    check_interval = 0.015  # Check every 15ms for interruptions
    pause_start = time.time()
    
    logger.info(f"⏸️ Applying pause: {duration}s for {client_id}")
    
    while (time.time() - pause_start) < duration:
        # Check for interruption
        if interaction_id != "-1" and await interruption_manager.check_interrupted(client_id, interaction_id):
            logger.info(f"🛑 Pause interrupted after {time.time() - pause_start:.2f}s")
            return False
        
        # Sleep in small intervals
        await asyncio.sleep(check_interval)
    
    logger.info(f" Pause completed: {duration}s")
    return True
# Add these constants after the imports section (around line 150)
FILLER_WORDS = {"um", "uh", "ah", "hmm", "er", "uhh", "huh", "like", "you know"}
GRAMMAR_END_PATTERN = re.compile(r"[.!?]$")  # Proper sentence ending
FILLER_PATTERN = re.compile(r"\b(um|uh|ah|hmm|er|uhh|like|you know)\b", re.I)
class SpeechTurnController:
    def __init__(self):
        self.last_speech_time = None
        self.partial_text = ""
        self.silence_start = None
        self.is_waiting_for_completion = False
        
    def is_grammatically_complete(self, text: str) -> bool:
        """Check if text ends in proper grammar (not just a filler)"""
        text = text.strip().lower()
        
        # Remove trailing fillers
        words = text.split()
        while words and words[-1] in FILLER_WORDS:
            words.pop()
        
        if not words:
            return False
        
        cleaned = " ".join(words)
        return bool(GRAMMAR_END_PATTERN.search(cleaned))
    
    def clean_user_input_for_llm(self, text: str) -> str:
        """Remove fillers before sending to LLM"""
        text = FILLER_PATTERN.sub("", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text
    
    async def handle_user_pause(self, current_text: str, silence_duration: float = 0.0) -> dict:
        """
         FIXED: Better handling of pauses
        """
        now = time.time()
        
        # Check if grammatically complete
        is_complete = self.is_grammatically_complete(current_text)
        
        #  Case 1: Grammatically complete → immediate commit
        if is_complete:
            cleaned = self.clean_user_input_for_llm(current_text)
            self.reset()
            return {
                "action": "commit",
                "cleaned_text": cleaned
            }
        
        #  Case 2: Very short input (<3 words) + long silence → commit anyway
        word_count = len(current_text.split())
        if word_count < 3 and silence_duration >= 1.5:
            cleaned = self.clean_user_input_for_llm(current_text)
            self.reset()
            return {
                "action": "commit",
                "cleaned_text": cleaned
            }
        
        #  Case 3: 1.2–2.0s pause AND not complete → clarify
        if 1.2 <= silence_duration < 2.0 and not is_complete and word_count >= 3:
            clarification_messages = [
                "Would you like to continue?",
                "Is there more you'd like to add?",
                "Should I respond now?"
            ]
            import random
            return {
                "action": "clarify",
                "message": random.choice(clarification_messages)
            }
        
        #  Case 4: ≥ 2.0s pause → commit anyway
        if silence_duration >= 2.0:
            cleaned = self.clean_user_input_for_llm(current_text)
            self.reset()
            return {
                "action": "commit",
                "cleaned_text": cleaned
            }
        
        #  Case 5: Still building (< 1.2s pause)
        return {"action": "wait"}
    
    def reset(self):
        """Reset state after committing to LLM"""
        self.last_speech_time = None
        self.partial_text = ""
        self.silence_start = None
        self.is_waiting_for_completion = False

# Global instance (add after the class definition)
speech_controllers = {}  # {client_id: SpeechTurnController}

import numpy as np
from typing import Dict, Tuple
import time

class EmotionStateManager:
     
    def __init__(self, num_blendshapes=None, fps=60):
        if num_blendshapes is None:
            num_blendshapes = NUM_BLENDSHAPES 
        
        self.num_blendshapes = num_blendshapes
        self.fps = fps
        self.dt = 1.0 / fps
        
        # === Persistent State ===
        self.current_intensity = 0.5
        self.target_intensity = 0.5
        self.current_emotion = "neutral"
        self.target_emotion = "neutral"
        
        #  Blendshape state - use num_blendshapes from parameter
        self.B_face = np.zeros(self.num_blendshapes, dtype=np.float32)
        self.B_micro = np.zeros(self.num_blendshapes, dtype=np.float32)
        
        #  Micro-motion noise state
        self.noise_state = np.random.randn(self.num_blendshapes).astype(np.float32) * 0.05
        
        # === Emotion Parameters Database ===
        self.emotion_params = {
            "neutral": {
                "tau": 0.5,      # Time constant (seconds)
                "k_audio": 0.15, # Audio coupling
                "sigma": 0.02,   # Micro-motion scale
                "beta": 0.3      # Noise update rate
            },
            "happy": {
                "tau": 0.3,
                "k_audio": 0.25,
                "sigma": 0.04,
                "beta": 0.4
            },
            "cheerful": {
                "tau": 0.25,
                "k_audio": 0.30,
                "sigma": 0.05,
                "beta": 0.45
            },
            "encouraging": {
                "tau": 0.35,
                "k_audio": 0.28,
                "sigma": 0.045,
                "beta": 0.42
            },
            "excited": {
                "tau": 0.2,
                "k_audio": 0.35,
                "sigma": 0.06,
                "beta": 0.5
            },
            "sad": {
                "tau": 0.8,
                "k_audio": 0.10,
                "sigma": 0.015,
                "beta": 0.2
            },
            "concerned": {
                "tau": 0.6,
                "k_audio": 0.18,
                "sigma": 0.025,
                "beta": 0.3
            },
            "curious": {
                "tau": 0.4,
                "k_audio": 0.22,
                "sigma": 0.055,
                "beta": 0.48
            },
            "angry": {
                "tau": 0.25,
                "k_audio": 0.30,
                "sigma": 0.05,
                "beta": 0.45
            },
            "surprised": {
                "tau": 0.15,
                "k_audio": 0.40,
                "sigma": 0.07,
                "beta": 0.55
            }
        }
        
         # === Lip-Sync Priority Weights ===
        self.priority_weights = self._build_priority_weights()
        
        # === Emotion Basis Vectors ===
        self.emotion_bases = self._build_emotion_bases()
        
        # Tracking
        self.last_update = time.time()
    
    def _build_priority_weights(self) -> np.ndarray:
        """
        Build priority vector using NUM_BLENDSHAPES
        """
        W = np.zeros(self.num_blendshapes, dtype=np.float32)
        
        #  Mouth shapes get HIGH priority (lip-sync first)
        mouth_shapes = [
            "jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthPucker",
            "mouthFunnel", "mouthStretchLeft", "mouthStretchRight",
            "mouthUpperUpLeft", "mouthUpperUpRight", "mouthLowerDownLeft",
            "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight"
        ]
        
        for shape_name in mouth_shapes:
            if shape_name in BLENDSHAPE_INDEX:
                idx = BLENDSHAPE_INDEX[shape_name]
                if idx < self.num_blendshapes:  #  Safety check
                    W[idx] = 0.90
        
        #  Brows/eyes get MEDIUM priority
        eye_brow_shapes = [
            "eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight",
            "eyeWideLeft", "eyeWideRight", "browInnerUp", "browDownLeft",
            "browDownRight", "browOuterUpLeft", "browOuterUpRight"
        ]
        
        for shape_name in eye_brow_shapes:
            if shape_name in BLENDSHAPE_INDEX:
                idx = BLENDSHAPE_INDEX[shape_name]
                if idx < self.num_blendshapes:  #  Safety check
                    W[idx] = 0.35

        #  NEW: Allow neck to blend (low priority, but nonzero)
        neck_shapes = [
            "neckTurnLeft", "neckTurnRight",
            "neckUp", "neckDown",
            "neckDownTiltLeft", "neckDownTiltRight",
            "neckUpTiltLeft", "neckUpTiltRight"
        ]

        for shape_name in neck_shapes:
            if shape_name in BLENDSHAPE_INDEX:
                idx = BLENDSHAPE_INDEX[shape_name]
                if idx < self.num_blendshapes:
                    W[idx] = 0.15   # LOW priority (emotion + micro can influence it)
        return W
    
    def _build_emotion_bases(self) -> Dict[str, np.ndarray]:
        """
        Pre-compute emotion basis vectors using NUM_BLENDSHAPES
        """
        bases = {}
        
        for emotion_name, modifiers in EMOTION_MODIFIERS.items():
            basis = np.zeros(self.num_blendshapes, dtype=np.float32)
            
            for shape_name, value in modifiers.items():
                if shape_name in BLENDSHAPE_INDEX:
                    idx = BLENDSHAPE_INDEX[shape_name]
                    if idx < self.num_blendshapes:  #  Safety check
                        # Scale down to 0-1 range (was 0-100)
                        scaled_value = value * 0.60
                        basis[idx] = scaled_value
            
            bases[emotion_name] = basis
        
        return bases
    def _build_emotion_bases_SCALED(self) -> Dict[str, np.ndarray]:
        """
         ENHANCED: Pre-compute emotion bases with REDUCED intensity
        FIXES: Over-animation by scaling down all emotion modifiers
        """
        bases = {}
        
        from collections import defaultdict
        
        for emotion_name, modifiers in EMOTION_MODIFIERS.items():
            basis = np.zeros(self.num_blendshapes, dtype=np.float32)
            
            for shape_name, value in modifiers.items():
                if shape_name in BLENDSHAPE_INDEX:
                    idx = BLENDSHAPE_INDEX[shape_name]
                    
                    #  CRITICAL: Reduce ALL emotion values by 40%
                    # This prevents emotions from overwhelming lip-sync
                    scaled_value = value * 0.60  # ↓ Was 1.0
                    
                    basis[idx] = scaled_value
            
            bases[emotion_name] = basis
        
        return bases
    def set_target_emotion(self, emotion: str, intensity: float):
        """
         Called when LLM updates emotion
        Does NOT reset state - just changes target
        """
        if emotion != self.target_emotion or abs(intensity - self.target_intensity) > 0.05:
            logger.info(
                f"🎭 Emotion target changed: {self.current_emotion}@{self.current_intensity:.2f} "
                f"→ {emotion}@{intensity:.2f}"
            )
        
        self.target_emotion = emotion
        self.target_intensity = intensity
    
    def update_frame(self, B_lipsync: np.ndarray, audio_energy: float = 0.0, is_speaking: bool = True) -> np.ndarray:
        """
         FIXED: Added is_speaking parameter
        """
        #  Validate input size
        if len(B_lipsync) != self.num_blendshapes:
            logger.error(
                f" Size mismatch: B_lipsync has {len(B_lipsync)} elements, "
                f"expected {self.num_blendshapes}"
            )
            # Resize if needed
            if len(B_lipsync) > self.num_blendshapes:
                B_lipsync = B_lipsync[:self.num_blendshapes]
            else:
                # Pad with zeros
                padded = np.zeros(self.num_blendshapes, dtype=np.float32)
                padded[:len(B_lipsync)] = B_lipsync
                B_lipsync = padded
        
        current_time = time.time()
        actual_dt = current_time - self.last_update
        self.last_update = current_time
        
        dt = min(actual_dt, self.dt * 2)
        
        # === Get emotion parameters ===
        params = self.emotion_params.get(
            self.target_emotion, 
            self.emotion_params["neutral"]
        )
        tau = params["tau"]
        k_audio = params["k_audio"]
        sigma = params["sigma"]
        beta = params["beta"]
        
        # === Smooth intensity transition ===
        lambda_e = 1.0 - np.exp(-dt / tau)
        self.current_intensity += lambda_e * (self.target_intensity - self.current_intensity)
        
        # === Audio-reactive modulation ===
        A_hat = np.clip(audio_energy, 0.0, 1.0)
        alpha_mod = self.current_intensity * (1.0 + k_audio * A_hat)
        alpha_mod = np.clip(alpha_mod, 0.0, 1.2)
        
        # === Apply emotion basis ===
        emotion_basis = self.emotion_bases.get(
            self.target_emotion, 
            self.emotion_bases.get("neutral", np.zeros(self.num_blendshapes))
        )
        
        B_emo = alpha_mod * emotion_basis
        
        #  Validate sizes before blending
        if len(B_emo) != self.num_blendshapes:
            logger.error(f" B_emo size mismatch: {len(B_emo)} vs {self.num_blendshapes}")
            B_emo = np.zeros(self.num_blendshapes, dtype=np.float32)
        
        # === Priority blend ===
        W = self.priority_weights
        
        #  Final size check
        if len(W) != self.num_blendshapes or len(B_lipsync) != self.num_blendshapes or len(B_emo) != self.num_blendshapes:
            logger.error(
                f" Array size mismatch before blend: "
                f"W={len(W)}, B_lipsync={len(B_lipsync)}, B_emo={len(B_emo)}, "
                f"expected={self.num_blendshapes}"
            )
            return B_lipsync  # Return lipsync only if error
        
        self.B_face = W * B_lipsync + (1.0 - W) * B_emo
        
        #  FIX: Only add micro-motion when NOT speaking
        if is_speaking:
            #  NEW: Only zero FACE micro, not NECK micro
            self.B_micro = np.zeros(self.num_blendshapes, dtype=np.float32)

            # But keep small neck motion
            for neck in [
                "neckDownTiltLeft","neckDownTiltRight",
                "neckUpTiltLeft","neckUpTiltRight",
                "neckTurnLeft","neckTurnRight"
            ]:
                if neck in BLENDSHAPE_INDEX:
                    idx = BLENDSHAPE_INDEX[neck]
                    # tiny breathing-like drift
                    self.B_micro[idx] = 0.002 * np.sin(time.time())

        else:
            # Update micro-motion only during silence
            eta = np.random.randn(self.num_blendshapes).astype(np.float32)
            self.noise_state = (1.0 - beta) * self.noise_state + beta * eta
            self.B_micro = sigma * 0.3 * self.noise_state  #  Reduced by 70%
        
        # === Combine ===
        B_final = self.B_face + self.B_micro

        #  AUDIO-REACTIVE NECK (SAFE, AFTER B_final EXISTS)
        energy = np.clip(audio_energy, 0.0, 1.0)

        for neck in ["neckDownTiltLeft", "neckDownTiltRight"]:
            if neck in BLENDSHAPE_INDEX:
                idx = BLENDSHAPE_INDEX[neck]
                B_final[idx] += 0.01 * energy   # gentle nod with speech

        # === Final clamp ===
        B_final = np.clip(B_final, 0.0, 1.0)

        return B_final

    
    def get_current_state(self) -> Dict:
        """Get current emotion state for debugging/logging"""
        return {
            "current_emotion": self.current_emotion,
            "target_emotion": self.target_emotion,
            "current_intensity": round(self.current_intensity, 3),
            "target_intensity": round(self.target_intensity, 3),
            "transition_progress": round(
                1.0 - abs(self.current_intensity - self.target_intensity), 3
            )
        }

# === Global emotion state managers (one per client) ===
emotion_state_managers = {}  # {client_id: EmotionStateManager}
# # =============================================================================
# # ADD THIS AT THE TOP OF YOUR SCRIPT (After imports, before other code)
# # =============================================================================

class MicrophoneSetup(str, Enum):
    """Microphone setup types with different noise characteristics"""
    DEDICATED = "dedicated"             # Professional USB mic, XLR setup
    LAPTOP_EXTERNAL = "laptop_external"  # Laptop with external USB mic
    LAPTOP_BUILTIN = "laptop_builtin"    # Laptop built-in microphone
    PHONE = "phone"                      # Mobile phone microphone
    HEADSET = "headset"                     # Gaming/call headset
    AUTO = "auto"                # Auto-detect based on audio characteristics

# VAD threshold configurations for different mic setups
VAD_THRESHOLD_PROFILES = {
    MicrophoneSetup.DEDICATED: {
        "vad_threshold": 0.08,  #  LOWERED from 0.10
        "min_speech_duration": 0.25,  #  LOWERED from 0.3
        "speech_pad_ms": 300,
        "description": "Professional mic - low noise floor"
    },
    MicrophoneSetup.LAPTOP_EXTERNAL: {
        "vad_threshold": 0.09,  #  LOWERED from 0.11
        "min_speech_duration": 0.35,  #  LOWERED from 0.4
        "speech_pad_ms": 400,
        "description": "External USB mic on laptop"
    },
    MicrophoneSetup.LAPTOP_BUILTIN: {
        "vad_threshold": 0.13,  #  LOWERED from 0.15
        "min_speech_duration": 0.45,  #  LOWERED from 0.5
        "speech_pad_ms": 500,
        "description": "Laptop built-in mic - noisy environment"
    },
    MicrophoneSetup.PHONE: {
        "vad_threshold": 0.15,  #  LOWERED from 0.17
        "min_speech_duration": 0.40,  #  LOWERED from 0.45
        "speech_pad_ms": 450,
        "description": "Mobile phone microphone"
    },
    MicrophoneSetup.HEADSET: {
        "vad_threshold": 0.10,  #  LOWERED from 0.12
        "min_speech_duration": 0.30,  #  LOWERED from 0.35
        "speech_pad_ms": 350,
        "description": "Gaming/call headset"
    },
    MicrophoneSetup.AUTO: {
        "vad_threshold": 0.09,  #  LOWERED from 0.11
        "min_speech_duration": 0.30,  #  LOWERED from 0.35
        "speech_pad_ms": 350,
        "description": "Auto-detect mode"
    }
}

def detect_mic_setup_from_audio(audio: np.ndarray, sample_rate: int) -> MicrophoneSetup:
    """
    Auto-detect microphone setup based on audio characteristics
    
    Heuristics:
    - Noise floor analysis
    - Signal-to-noise ratio
    - Frequency spectrum analysis
    """
    if audio is None or len(audio) < sample_rate:
        return MicrophoneSetup.AUTO
    
    # Calculate noise floor (RMS of quietest 20% of audio)
    sorted_audio = np.sort(np.abs(audio))
    noise_floor = np.mean(sorted_audio[:len(sorted_audio)//5])
    
    # Calculate overall RMS
    overall_rms = np.sqrt(np.mean(audio**2))
    
    # Signal-to-noise ratio
    snr = overall_rms / (noise_floor + 1e-9)
    
    # Heuristic thresholds
    if snr > 15.0 and noise_floor < 0.005:
        return MicrophoneSetup.DEDICATED
    elif snr > 10.0 and noise_floor < 0.01:
        return MicrophoneSetup.HEADSET
    elif snr > 8.0 and noise_floor < 0.015:
        return MicrophoneSetup.LAPTOP_EXTERNAL
    elif snr > 5.0:
        return MicrophoneSetup.LAPTOP_BUILTIN
    else:
        return MicrophoneSetup.PHONE

try:
    from utils.localllm import generate_llm_response_streaming, set_llm_config, get_llm_config, get_available_providers #,   ollama_client 
except ImportError:
    async def generate_llm_response_streaming(*args, **kwargs):
        await asyncio.sleep(0.1)
        return "I understand your question.", "neutral", {}
    
    def set_llm_config(*args, **kwargs):
        return True
        
    def get_llm_config():
        return {}
        
    def get_available_providers():
        return ["groq", "openai"]
    
    # ollama_client = None

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("nltk").setLevel(logging.ERROR)

# =============================================================================
# CONFIGURATION
# =============================================================================
CHUNK_SIZE = 4096
target_sample_rate = 24000
CPU_CORES = cpu_count()
AUDIO_WORKERS = min(CPU_CORES - 1, 4)
MAX_TTS_WORKERS = min(CPU_CORES - 1, 2)  

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

logging.getLogger("main2").setLevel(logging.DEBUG)
logger.info("="*60)
logger.info("PARALLEL Voice Assistant - 1000+ Client Support")
logger.info("Production Audio Pipeline - 95%+ Accuracy in Noisy Environments")
logger.info(f"CPU Cores: {CPU_CORES} | Audio Workers: {AUDIO_WORKERS} |  TTS Workers: {MAX_TTS_WORKERS}")
logger.info("="*60)

# PyTorch 2.6 compatibility patch
_original_load_fsspec = TTS.utils.io.load_fsspec

def _patched_load_fsspec(path, map_location=None, **kwargs):
    kwargs['weights_only'] = False
    return torch.load(path, map_location=map_location, **kwargs)

TTS.utils.io.load_fsspec = _patched_load_fsspec
logger.info("🔧 Applied PyTorch 2.6 compatibility patch")

# Global variables 
client_sessions = {}
selected_llm = "groq"
selected_bot_config = None
#  NEW: Audio timing tracking (for silence detection)
client_audio_timings = {}  # {client_id: {'last_audio_time': float, 'partial_text': str}}

stats = {
    "connections_total": 0, "connections_active": 0, "messages_received": 0,
    "messages_broadcast": 0, "audio_chunks_processed": 0, "tts_generations": 0,
    "rag_queries": 0, "errors": 0, "interruptions": 0, "voice_detections": 0,
    "noise_suppressed_chunks": 0
}
# =============================================================================
# TOKEN MANAGEMENT
# =============================================================================
client_llm_tokens = {}  # {client_id: token_string}

class TokenStatus:
    """Track token status per client"""
    def __init__(self):
        self.valid = False
        self.expired = False
        self.token = ""
        self.session_id = ""
        self.checked_at = None
        self.expires_at = None
        self.last_validated = None
        self.last_check_time = 0
        self.check_interval = 300  # Check every 5 minutes
        self.is_valid = False
        self.is_expired = False
        self.warnings_sent = set()

client_token_status = {}  # {client_id: TokenStatus}

class LLMStatus:
    """Track LLM API call status"""
    def __init__(self):
        self.consecutive_failures = 0
        self.last_failure_time = 0
        self.last_success_time = time.time()
        self.is_degraded = False

client_llm_status = {}  # {client_id: LLMStatus}


def get_client_session_id(client_id: str) -> Optional[str]:
    """Retrieve session ID for client"""
    return client_session_ids.get(client_id)

def get_client_llm_token(client_id: str) -> Optional[str]:
    """Get token for client"""
    return client_llm_tokens.get(client_id)

def remove_client_token(client_id: str):
    """Remove token when client disconnects"""
    if client_id in client_llm_tokens:
        del client_llm_tokens[client_id]
        logger.info(f"🗑️ Removed token for {client_id}")


async def check_token_with_sam_websocket(token: str, session_id: str) -> dict:
    """
    Validate token with SAM WebSocket
    Now returns credentials for persistent connection
    """
    try:
        if not token or len(token) < 100:
            return {
                "valid": False,
                "expired": False,
                "message": "Invalid token format"
            }
        
        if not session_id:
            return {
                "valid": False,
                "expired": False,
                "message": "Session ID is required"
            }
        
        # Test connection
        test_client = SAMLLMClient(SAM_WS_BASE_URL, session_id, token)
        
        if await test_client.connect():
            # Test query
            test_response, _, _ = await test_client.send_query("Hello")
            await test_client.close()
            
            if test_response:
                return {
                    "valid": True,
                    "expired": False,
                    "message": "Token validated successfully",
                    "credentials": {
                        "base_url": SAM_WS_BASE_URL,
                        "session_id": session_id,
                        "access_token": token
                    }
                }
        
        return {
            "valid": False,
            "expired": False,
            "message": "Token validation failed"
        }
        
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        return {
            "valid": False,
            "expired": False,
            "message": str(e)
        }

async def periodic_token_check(client_id: str):
    """
    Periodically check if stored token is still valid
    Sends warnings to frontend if token expires
    """
    if client_id not in client_sessions:
        return
    
    token = get_client_llm_token(client_id)
    if not token:
        return
    
    # Get or create token status
    if client_id not in client_token_status:
        client_token_status[client_id] = TokenStatus()
    
    status = client_token_status[client_id]
    current_time = time.time()
    
    # Check if it's time to validate again
    if current_time - status.last_check_time < status.check_interval:
        return
    
    status.last_check_time = current_time
    
    # Validate token
    logger.info(f"🔍 Periodic token check for {client_id}")
    result = await check_token_with_sam_websocket(token)
    
    # Update status
    was_valid = status.is_valid
    status.is_valid = result["valid"]
    status.is_expired = result["expired"]
    
    # Send notification to frontend
    if result["expired"] and "expired" not in status.warnings_sent:
        status.warnings_sent.add("expired")
        await send_to_client(client_id, {
            "type": "token_expired_during_session",
            "status": "warning",
            "message": "Your LLM token has expired during this session. You can continue talking, but please provide a new token for optimal performance.",
            "can_continue": True,
            "severity": "high",
            "log": f"[{datetime.now().strftime('%H:%M:%S')}] Token expired - service degraded"
        })
        logger.warning(f"⏰ Token expired during session for {client_id}")
    
    elif not result["valid"] and was_valid and "invalid" not in status.warnings_sent:
        status.warnings_sent.add("invalid")
        await send_to_client(client_id, {
            "type": "token_invalid_during_session",
            "status": "warning",
            "message": f"Token validation failed: {result['message']}. Service continues with fallback.",
            "can_continue": True,
            "severity": "medium",
            "log": f"[{datetime.now().strftime('%H:%M:%S')}] Token invalid - {result['message']}"
        })
        logger.error(f" Token became invalid for {client_id}: {result['message']}")

async def send_log_to_frontend(client_id: str, log_type: str, message: str, level: str = "info"):
    """
    Send log messages to frontend for display
    
    Args:
        client_id: Client identifier
        log_type: Type of log (token, audio, emotion, etc.)
        message: Log message
        level: Log level (info, warning, error, success)
    """
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    
    await send_to_client(client_id, {
        "type": "log_message",
        "log_type": log_type,
        "level": level,
        "message": message,
        "timestamp": timestamp,
        "can_continue": True
    })

#  NEW: Add helper to check token status and send warning
async def check_and_warn_token_status(client_id: str) -> dict:
    """
    Check token status and return warning message if needed
    Returns: {"has_token": bool, "valid": bool, "expired": bool, "warning": str or None}
    """
    token = get_client_llm_token(client_id)
    
    if not token:
        return {
            "has_token": False,
            "valid": False,
            "expired": False,
            "warning": "No LLM token provided. Please submit a valid token for full functionality."
        }
    
    # Check if token is valid
    token_status = await check_token_with_sam_websocket(token)
    
    if token_status["expired"]:
        return {
            "has_token": True,
            "valid": False,
            "expired": True,
            "warning": "Your LLM token has expired. Please submit a new token."
        }
    elif not token_status["valid"]:
        return {
            "has_token": True,
            "valid": False,
            "expired": False,
            "warning": f"Invalid token: {token_status['message']}"
        }
    
    # Token is valid
    return {
        "has_token": True,
        "valid": True,
        "expired": False,
        "warning": None
    }
   
# =============================================================================
# ELEVENLABS SCRIBE V2 REAL-TIME SPEECH-TO-TEXT
# =============================================================================

# class ElevenLabsScribeProcessor:
#     """
#     Real-time speech-to-text using ElevenLabs Scribe v2
#     Replaces Faster Whisper for better accuracy and speed
#     """
    
#     def __init__(
#         self,
#         api_key: str,
#         sample_rate: int = 16000,
#         mic_setup: MicrophoneSetup = MicrophoneSetup.AUTO
#     ):
#         self.api_key = api_key
#         self.sample_rate = sample_rate
#         self.mic_setup = mic_setup
        
#         # Base URL for ElevenLabs API
#         self.base_url = "https://api.elevenlabs.io/v1"
        
#         logger.info(" ElevenLabs Scribe v2 initialized")
    
#     async def transcribe_audio_stream(
#         self, 
#         audio_data: bytes, 
#         client_id: str
#     ) -> tuple[str, list]:
#         """
#         Transcribe audio using ElevenLabs Scribe v2
        
#         Returns:
#             tuple: (transcribed_text, timestamps)
#         """
#         try:
#             import httpx
            
#             # Create temporary audio file
#             temp_audio_path = f"/tmp/scribe_audio_{client_id}_{int(time.time()*1000)}.wav"
            
#             # Write audio data to WAV file
#             with wave.open(temp_audio_path, 'wb') as wav_file:
#                 wav_file.setnchannels(1)  # Mono
#                 wav_file.setsampwidth(2)   # 16-bit
#                 wav_file.setframerate(self.sample_rate)
#                 wav_file.writeframes(audio_data)
            
#             # Prepare the API request
#             url = f"{self.base_url}/speech-to-text"
            
#             headers = {
#                 "xi-api-key": self.api_key
#             }
            
#             #  FIX: Use 'file' as the parameter name (not 'audio')
#             with open(temp_audio_path, 'rb') as audio_file:
#                 files = {
#                     'file': ('audio.wav', audio_file, 'audio/wav')  #  Changed from 'audio' to 'file'
#                 }
                
#                 data = {
#                     'model_id': 'scribe_v2',
#                     'language': 'en'
#                 }
                
#                 async with httpx.AsyncClient(timeout=30.0) as client:
#                     response = await client.post(
#                         url,
#                         headers=headers,
#                         files=files,
#                         data=data
#                     )
            
#             # Remove temp file
#             try:
#                 os.remove(temp_audio_path)
#             except:
#                 pass
            
#             if response.status_code == 200:
#                 result = response.json()
#                 text = result.get('text', '')
                
#                 # Extract timestamps if available
#                 timestamps = []
#                 if 'words' in result:
#                     timestamps = result['words']
                
#                 logger.info(f"[Scribe v2] Transcribed: {text}")
#                 return text, timestamps
#             else:
#                 logger.error(f"[Scribe v2] API error: {response.status_code} - {response.text}")
#                 return "", []
            
#         except Exception as e:
#             logger.error(f"[Scribe v2] Transcription failed: {e}")
#             import traceback
#             traceback.print_exc()
#             return "", []
            
#         except Exception as e:
#             logger.error(f"[Scribe v2] Transcription failed: {e}")
#             import traceback
#             traceback.print_exc()
#             return "", []
    
#     async def cleanup(self):
#         """Cleanup resources"""
#         pass
   









# =============================================================================
# FASTER WHISPER FALLBACK (OPTIONAL)
# =============================================================================

# class OptimizedFasterWhisperProcessor:
#     """
#      OPTIMIZED: Faster Whisper fallback with batching
#     """
    
#     def __init__(
#         self,
#         model_name: str = "small.en",
#         device: str = "cpu",
#         compute_type: str = "int8"
#     ):
#         from faster_whisper import WhisperModel
        
#         self.model = WhisperModel(
#             model_name,
#             device=device,
#             compute_type=compute_type,
#             num_workers=4
#         )
        
#         # Batching
#         self.batch_size = 3
#         self.pending_audio = []
#         self.batch_lock = asyncio.Lock()
        
#         logger.info(f" Faster Whisper loaded: {model_name} on {device}")
    
#     async def transcribe_audio(self, audio: np.ndarray) -> str:
#         """Transcribe audio with batching"""
#         loop = asyncio.get_running_loop()
        
#         def _transcribe():
#             segments, info = self.model.transcribe(
#                 audio,
#                 beam_size=1,  # Faster
#                 best_of=1,
#                 vad_filter=False,  # We already did VAD
#                 language="en"
#             )
#             text = " ".join([s.text for s in segments])
#             return text.strip()
        
#         return await loop.run_in_executor(None, _transcribe)

# =============================================================================
# ENHANCED AUDIO PROCESSOR WITH SCRIBE V2 INTEGRATION
# =============================================================================


class OptimizedElevenLabsScribeProcessor:
    """
     OPTIMIZED: ElevenLabs Scribe v2 with batching, caching, and no temp files
    """
    
    
    def __init__(
        self,
        api_key: str,
        sample_rate: int = 16000,
        mic_setup: MicrophoneSetup = MicrophoneSetup.AUTO,
        max_cache_size: int = 100
    ):
        self.api_key = api_key
        self.sample_rate = sample_rate
        self.mic_setup = mic_setup
        self.base_url = "https://api.elevenlabs.io/v1"
        
        #  NEW: Caching system
        self.transcription_cache = {}
        self.max_cache_size = max_cache_size
        
        #  NEW: Request batching
        self.pending_requests = {}
        self.batch_lock = asyncio.Lock()
        
        #  NEW: Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0
        
        logger.info(" Optimized ElevenLabs Scribe v2 initialized")
    print("using scribe here >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    
    def _generate_cache_key(self, audio_data: bytes) -> str:
        """Generate cache key from audio data"""
        return hashlib.md5(audio_data).hexdigest()
    
    async def transcribe_audio_stream(
        self, 
        audio_data: bytes, 
        client_id: str
    ) -> Tuple[str, list]:
        """
         OPTIMIZED: Transcribe with caching and batching
        
        Returns:
            tuple: (transcribed_text, timestamps)
        """
        self.total_requests += 1
        
        # Generate cache key
        cache_key = self._generate_cache_key(audio_data)
        
        # Check cache first
        if cache_key in self.transcription_cache:
            self.cache_hits += 1
            logger.debug(
                f"[Scribe] Cache hit for {client_id} | "
                f"Hit rate: {self.cache_hits}/{self.total_requests} "
                f"({self.cache_hits/self.total_requests*100:.1f}%)"
            )
            return self.transcription_cache[cache_key]
        
        self.cache_misses += 1
        
        # Check if already processing this audio
        async with self.batch_lock:
            if cache_key in self.pending_requests:
                logger.debug(f"[Scribe] Waiting for existing request: {client_id}")
                # Wait for existing request to complete
                return await self.pending_requests[cache_key]
            
            # Create new request task
            request_task = asyncio.create_task(
                self._transcribe_internal(audio_data, client_id, cache_key)
            )
            self.pending_requests[cache_key] = request_task
        
        try:
            result = await request_task
            return result
        finally:
            # Cleanup
            async with self.batch_lock:
                self.pending_requests.pop(cache_key, None)
    
    async def _transcribe_internal(
        self, 
        audio_data: bytes, 
        client_id: str, 
        cache_key: str
    ) -> Tuple[str, list]:
        """
         OPTIMIZED: Internal transcription with BytesIO (no temp files)
        """
        import httpx
        
        max_retries = 2
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                #  FIX: Use BytesIO instead of temp files
                audio_buffer = io.BytesIO()
                
                with wave.open(audio_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)   # 16-bit
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(audio_data)
                
                # Reset buffer position
                audio_buffer.seek(0)
                
                # Prepare multipart form data
                files = {
                    'file': ('audio.wav', audio_buffer, 'audio/wav')
                }
                
                data = {
                    'model_id': 'scribe_v2',
                    'language_code': 'en',
                    'tag_audio_events': False, 
                    
                }
                
                #  OPTIMIZED: Timeout with retry logic
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        f"{self.base_url}/speech-to-text",
                        headers={"xi-api-key": self.api_key},
                        files=files,
                        data=data
                    )
                
                # Handle response
                if response.status_code == 200:
                    result = response.json()
                    text = result.get('text', '').strip()
                    
                    
                    #  ADD: Reject if text contains non-Latin characters (Hindi, etc.)
                    if re.search(r'[^\x00-\x7F]', text):
                        logger.warning(f"[Scribe v2] Non-English detected, discarding: {text[:50]}")
                        return "", []
                
                    timestamps = result.get('words', [])

                    
                    # Cache result
                    self._add_to_cache(cache_key, (text, timestamps))
                    
                    if text:
                        logger.info(f"[Scribe v2]  {client_id}: {text}")
                    
                    return text, timestamps
                
                elif response.status_code == 429:  # Rate limit
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        logger.warning(
                            f"[Scribe v2] Rate limited, retry in {wait_time}s "
                            f"({attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    logger.error(f"[Scribe v2] Rate limit exceeded")
                    return "", []
                
                else:
                    logger.error(
                        f"[Scribe v2] API error: {response.status_code} - "
                        f"{response.text[:200]}"
                    )
                    return "", []
            
            except httpx.TimeoutException:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"[Scribe v2] Timeout, retry {attempt + 1}/{max_retries}"
                    )
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                logger.error(f"[Scribe v2] Timeout after {max_retries} attempts")
                return "", []
            
            except Exception as e:
                logger.error(f"[Scribe v2] Transcription failed: {e}")
                import traceback
                traceback.print_exc()
                return "", []
        
        return "", []
    
    def _add_to_cache(self, key: str, value: Tuple[str, list]):
        """Add result to cache with size limit"""
        self.transcription_cache[key] = value
        
        # Limit cache size (LRU-style)
        if len(self.transcription_cache) > self.max_cache_size:
            # Remove oldest entry (first key)
            oldest_key = next(iter(self.transcription_cache))
            del self.transcription_cache[oldest_key]
            logger.debug(f"[Scribe] Cache eviction: {len(self.transcription_cache)}/{self.max_cache_size}")
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        hit_rate = (self.cache_hits / self.total_requests * 100) if self.total_requests > 0 else 0
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "cache_size": len(self.transcription_cache),
            "pending_requests": len(self.pending_requests)
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        self.transcription_cache.clear()
        async with self.batch_lock:
            self.pending_requests.clear()
        logger.info(f"[Scribe v2] Cleaned up | Stats: {self.get_stats()}")

# =============================================================================
# ENHANCED AUDIO PROCESSOR WITH OPTIMIZED ASR
# =============================================================================

class EnhancedAudioProcessorWithOptimizedASR:
    """
     PRODUCTION: Complete audio processor with optimized ASR
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        device: Optional[str] = None,
        mic_setup: MicrophoneSetup = MicrophoneSetup.AUTO,
        elevenlabs_api_key: Optional[str] = None,
        use_whisper_fallback: bool = True
    ):
        self.sample_rate = sample_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mic_setup = mic_setup
        self.vad_profile = VAD_THRESHOLD_PROFILES[mic_setup]
        
        # VAD settings
        self.vad_threshold = self.vad_profile["vad_threshold"]
        self.min_speech_duration = self.vad_profile["min_speech_duration"]
        self.speech_pad_ms = self.vad_profile["speech_pad_ms"]
        
        logger.info(f"Audio pipeline init — device: {self.device}")
        logger.info(f"Mic setup: {mic_setup.value} | VAD threshold: {self.vad_threshold}")
        
        # Model placeholders
        self.deepfilter_model = None
        self.deepfilter_state = None
        self.vad_model = None
        self.vad_utils = None
        
        #  OPTIMIZED: Primary ASR - ElevenLabs Scribe v2
        self.elevenlabs_api_key = elevenlabs_api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.elevenlabs_api_key:
            logger.warning("ELEVENLABS_API_KEY not found - Scribe v2 disabled")
            self.scribe_processor = None
        else:
            self.scribe_processor = OptimizedElevenLabsScribeProcessor(
                api_key=self.elevenlabs_api_key,
                sample_rate=sample_rate,
                mic_setup=mic_setup,
                max_cache_size=100
            )
        
        #  NEW: Fallback ASR - Faster Whisper
        self.use_whisper_fallback = use_whisper_fallback
        self.whisper_processor = None
        # if use_whisper_fallback:
        #     try:
        #         self.whisper_processor = OptimizedFasterWhisperProcessor(
        #             model_name="small.en",
        #             device="cpu",
        #             compute_type="int8"
        #         )
        #     except Exception as e:
        #         logger.warning(f"Whisper fallback init failed: {e}")
        
        # Tunables
        self.max_speech_duration = 80.0
        
        # Concurrency
        self.executor = ThreadPoolExecutor(max_workers=8)
        self._initialized = False
        self._init_lock = asyncio.Lock()
        
        # Auto-calibration
        self.audio_samples_processed = 0
        self.auto_calibration_window = 10
        
        #  NEW: Performance tracking
        self.scribe_successes = 0
        self.scribe_failures = 0
        self.whisper_fallbacks = 0
    
    print("using EnhancedAudioProcessorWithOptimizedASR >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    async def _load_deepfilter_force_cpu(self):
        try:
            from df.enhance import init_df
            self.deepfilter_model, self.deepfilter_state, _ = init_df()
            self.deepfilter_model = self.deepfilter_model.cpu()  # already there
            #  ADD: force all params to cpu
            self.deepfilter_model = self.deepfilter_model.float()
            logger.info(" DeepFilter loaded (CPU)")
        except Exception as e:
            logger.warning(f"DeepFilter load failed: {e}")
    
    async def _load_silero_vad(self):
        """Load Silero VAD"""
        try:
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.vad_model = model.cpu()
            self.vad_utils = utils
            logger.info(" Silero VAD loaded")
        except Exception as e:
            logger.error(f" Silero VAD load failed: {e}")
    
    async def initialize(self):
        """Initialize models"""
        async with self._init_lock:
            if self._initialized:
                return
            
            start = time.time()
            logger.info("Loading models...")
            
            # Load in parallel
            await asyncio.gather(
                self._load_deepfilter_force_cpu(),
                self._load_silero_vad()
            )
            
            self._initialized = True
            logger.info(f" Pipeline ready in {time.time() - start:.2f}s")
    
    def _decode_audio(self, audio_b64: str) -> Optional[np.ndarray]:
        """Decode base64 audio"""
        try:
            raw = base64.b64decode(audio_b64)
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            return pcm / 32768.0
        except Exception as e:
            logger.error(f"Decode failed: {e}")
            return None

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio"""
        if audio is None or len(audio) == 0:
            return audio
        audio = audio.astype(np.float32)
        audio = audio - np.mean(audio)
        peak = np.max(np.abs(audio)) + 1e-9
        return np.clip(audio / peak, -1.0, 1.0)

    def _rms(self, audio: np.ndarray, eps: float = 1e-9) -> float:
        """Calculate RMS"""
        if audio is None or len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio**2) + eps))

    def _adaptive_rms(self, audio: np.ndarray, frame_ms: int = 50) -> float:
        """Calculate adaptive RMS"""
        if audio is None or len(audio) == 0:
            return 0.0
        frame_len = int(self.sample_rate * frame_ms / 1000)
        frame_len = max(1, frame_len)
        rms_values = []
        for start in range(0, len(audio), frame_len):
            frame = audio[start : start + frame_len]
            rms_values.append(np.sqrt(np.mean(frame**2) + 1e-9))
        return float(np.mean(rms_values))

    async def _apply_deepfilter(self, audio: np.ndarray) -> np.ndarray:
        """Apply DeepFilter noise reduction"""
        if self.deepfilter_model is None or audio is None or len(audio) < 400:
            return audio.astype(np.float32)
        try:
            from df.enhance import enhance
            loop = asyncio.get_running_loop()

            # def _run_enhance():
            #     # audio_t = torch.from_numpy(audio).float().unsqueeze(0).cpu()
            #     # audio_t = torch.from_numpy(audio).float().unsqueeze(0)
            #     # Force model and tensor to same device
            #     device = next(self.deepfilter_model.parameters()).device
            #     audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(device)

            #     enhanced = enhance(self.deepfilter_model, self.deepfilter_state, audio_t)
            #     return enhanced.squeeze().cpu().numpy().astype(np.float32)
            def _run_enhance():
                #  Force tensor to CPU always
                audio_t = torch.from_numpy(audio).float().unsqueeze(0).cpu()
                enhanced = enhance(self.deepfilter_model, self.deepfilter_state, audio_t)
                return enhanced.squeeze().cpu().numpy().astype(np.float32)

            enhanced = await loop.run_in_executor(self.executor, _run_enhance)
            return enhanced
        except Exception as e:
            logger.warning(f"DeepFilter failed: {e}")
            return audio.astype(np.float32)

    async def _detect_speech_segments(self, audio: np.ndarray) -> List[Dict]:
        """Detect speech segments using Silero VAD"""
        if self.vad_model is None or self.vad_utils is None:
            return []
        try:
            get_ts = self.vad_utils[0]
            loop = asyncio.get_running_loop()

            def _run_vad():
                audio_t = torch.from_numpy(audio).float().cpu()
                with torch.no_grad():
                    ts = get_ts(
                        audio_t,
                        self.vad_model,
                        sampling_rate=self.sample_rate,
                        threshold=self.vad_threshold,
                        min_speech_duration_ms=int(self.min_speech_duration * 1000),
                        max_speech_duration_s=self.max_speech_duration,
                        min_silence_duration_ms=150,
                        speech_pad_ms=self.speech_pad_ms,
                    )
                return ts

            ts = await loop.run_in_executor(self.executor, _run_vad)
            results = []
            for t in ts or []:
                start = int(t.get("start", 0))
                end = int(t.get("end", 0))
                results.append({"start_sample": start, "end_sample": end})
            return results
        except Exception as e:
            logger.error(f" VAD failed: {e}")
            return []
    
    def _add_smart_punctuation(self, text: str) -> str:
        """Add intelligent punctuation"""
        if re.search(r'[.!?]$', text):
            return text
        
        question_starters = [
            "who", "what", "when", "where", "why", "how",
            "is", "are", "was", "were", "can", "could",
            "would", "should", "will", "do", "does", "did"
        ]
        
        first_word = text.lower().split()[0] if text.split() else ""
        
        if first_word in question_starters:
            return text + "?"
        
        exclamation_words = ["wow", "amazing", "incredible", "oh", "ah", "hey"]
        if first_word in exclamation_words or text.isupper():
            return text + "!"
        
        return text + "."
    
    def _post_process_transcript(self, text: str) -> str:
        """Clean and improve transcript quality"""
        if not text:
            return text
        
        text = re.sub(r'[^\x00-\x7F]+', '', text).strip()
        
        corrections = {
            r'\btheres\b': "there's",
            r'\bits\b': "it's",
            r'\bim\b': "I'm",
            r'\bwont\b': "won't",
            r'\bcant\b': "can't",
            r'\bdont\b': "don't",
        }
        
        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        if text:
            text = text[0].upper() + text[1:]
        
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        text = re.sub(r'([.,!?])(\w)', r'\1 \2', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _join_transcripts_intelligently(self, results: list) -> str:
        """Join multiple transcript segments"""
        if not results:
            return ""
        
        if len(results) == 1:
            return results[0]
        
        joined = []
        
        for i, segment in enumerate(results):
            segment = segment.strip()
            
            if i < len(results) - 1:
                if not re.search(r'[.!?,;:]$', segment):
                    next_segment = results[i + 1].strip()
                    if next_segment and next_segment[0].isupper():
                        segment = segment + "."
                    else:
                        segment = segment + ","
            
            joined.append(segment)
        
        return " ".join(joined)
    
    async def _transcribe_with_fallback(
        self,
        audio_chunk: np.ndarray,
        chunk_bytes: bytes,
        client_id: str
    ) -> Optional[str]:
        """
         NEW: Transcribe with automatic fallback
        """
        # Try Scribe v2 first
        if self.scribe_processor:
            try:
                text, timestamps = await self.scribe_processor.transcribe_audio_stream(
                    chunk_bytes,
                    client_id
                )
                
                if text:
                    self.scribe_successes += 1
                    return text
                else:
                    self.scribe_failures += 1
            except Exception as e:
                self.scribe_failures += 1
                logger.warning(f"[ASR] Scribe v2 failed: {e}")
        
        # Fallback to Whisper
        if self.whisper_processor:
            try:
                self.whisper_fallbacks += 1
                logger.info(f"[ASR] Using Whisper fallback for {client_id}")
                
                text = await self.whisper_processor.transcribe_audio(audio_chunk)
                
                if text:
                    return text
            except Exception as e:
                logger.error(f"[ASR] Whisper fallback failed: {e}")
        
        return None
    
    def update_mic_setup(self, new_setup: MicrophoneSetup):
        """Update microphone setup"""
        if new_setup == self.mic_setup:
            return
        
        old_setup = self.mic_setup
        self.mic_setup = new_setup
        self.vad_profile = VAD_THRESHOLD_PROFILES[new_setup]
        self.vad_threshold = self.vad_profile["vad_threshold"]
        self.min_speech_duration = self.vad_profile["min_speech_duration"]
        self.speech_pad_ms = self.vad_profile["speech_pad_ms"]
        
        logger.info(
            f"Mic setup: {old_setup.value} → {new_setup.value} | "
            f"VAD: {self.vad_threshold}"
        )
    
    async def process_audio_stream(
        self, 
        client_id: str, 
        audio_data_base64: str,
        auto_detect_mic: bool = True,
        return_partial: bool = True
    ) -> Tuple[Optional[str], Optional[str], Optional[float], Optional[bool]]:
        """
         OPTIMIZED: Main audio processing pipeline
        """
        start_time = time.time()
        
        if not self._initialized:
            await self.initialize()
        
        # Decode audio
        audio = self._decode_audio(audio_data_base64)
        if audio is None or len(audio) < 400:
            return None, None, None, None
        
        # Auto-detect mic setup
        if auto_detect_mic and self.mic_setup == MicrophoneSetup.AUTO:
            self.audio_samples_processed += 1
            
            if self.audio_samples_processed % self.auto_calibration_window == 0:
                detected_setup = detect_mic_setup_from_audio(audio, self.sample_rate)
                if detected_setup != MicrophoneSetup.AUTO:
                    self.update_mic_setup(detected_setup)
        
        # Calculate energy
        raw_energy = self._adaptive_rms(audio)
        
        #  OPTIMIZED: Skip DeepFilter for energy check
        audio_norm = self._normalize(audio)
        quick_energy = self._adaptive_rms(audio_norm)
        
        MIN_ASR_ENERGY = 0.003
        
        if quick_energy < MIN_ASR_ENERGY:
            logger.debug(
                f"[{client_id}] Dropped: energy={quick_energy:.5f} < {MIN_ASR_ENERGY}"
            )
            return None, None, quick_energy, False
        
        #  PARALLEL: VAD + DeepFilter
        async def run_vad():
            return await self._detect_speech_segments(audio_norm)
        
        async def run_deepfilter():
            return await self._apply_deepfilter(audio_norm)
        
        segments, audio_clean = await asyncio.gather(run_vad(), run_deepfilter())
        audio_clean = self._normalize(audio_clean)
        
        if not segments:
            return None, None, None, None
        
        #  OPTIMIZED: Transcribe with fallback
        results = []
        for seg in segments:
            start, end = seg["start_sample"], seg["end_sample"]
            start = max(0, min(len(audio_clean) - 1, start))
            end = max(start + 1, min(len(audio_clean), end))
            chunk = audio_clean[start:end]
            
            if len(chunk) < 400:
                continue
            
            # Convert to bytes
            chunk_int16 = (chunk * 32768).astype(np.int16)
            chunk_bytes = chunk_int16.tobytes()
            
            # Transcribe with automatic fallback
            text = await self._transcribe_with_fallback(chunk, chunk_bytes, client_id)
            
            if text:
                # Post-process
                text = self._add_smart_punctuation(text)
                text = self._post_process_transcript(text)
                results.append(text)
        
        if not results:
            return None, None, None, None
        
        # Join transcripts
        final_text = self._join_transcripts_intelligently(results)
        
        # Check if complete
        is_complete = bool(GRAMMAR_END_PATTERN.search(final_text))
        
        # Adaptive emotion
        df_energy = self._adaptive_rms(audio_clean)
        if df_energy > 0.02:
            emotion = "excited"
        elif df_energy > 0.008:
            emotion = "happy"
        else:
            emotion = "neutral"
        
        elapsed = (time.time() - start_time) * 1000
        
        # Log with stats
        stats = self.get_performance_stats()
        logger.info(
            f" [{client_id}] {final_text} | "
            f"{elapsed:.0f}ms | Complete: {is_complete} | "
            f"Scribe: {stats['scribe_success_rate']:.1f}%"
        )
        
        return final_text, emotion, df_energy, is_complete
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        total_asr = self.scribe_successes + self.scribe_failures
        scribe_rate = (self.scribe_successes / total_asr * 100) if total_asr > 0 else 0
        
        stats = {
            "scribe_successes": self.scribe_successes,
            "scribe_failures": self.scribe_failures,
            "whisper_fallbacks": self.whisper_fallbacks,
            "scribe_success_rate": round(scribe_rate, 2),
            "total_transcriptions": total_asr
        }
        
        # Add Scribe cache stats if available
        if self.scribe_processor:
            stats.update(self.scribe_processor.get_stats())
        
        return stats
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.executor:
                self.executor.shutdown(wait=False)
            self.deepfilter_model = None
            self.deepfilter_state = None
            self.vad_model = None
            self.vad_utils = None
            
            if self.scribe_processor:
                asyncio.create_task(self.scribe_processor.cleanup())
            
            if self.device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            
            logger.info(f"🧹 ASR cleanup complete | Stats: {self.get_performance_stats()}")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def __del__(self):
        self.cleanup()

# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

#  Initialize with all optimizations
audio_processor = EnhancedAudioProcessorWithOptimizedASR(
    sample_rate=16000,
    mic_setup=MicrophoneSetup.AUTO,
    elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY"),
    use_whisper_fallback=True  # Enable automatic fallback
)


# ============================================================================
# TTS WORKER POOL
# ============================================================================
"""
========================================================================================================
||| >>> ||| >>> ||| >>> ||| Voice Streaming Method Apply ||| >>> ||| >>> ||| >>> ||| >>> ||| >>> ||| >>>
========================================================================================================
"""
# =============================================================================
# STREAMING CHUNK BUFFER
# =============================================================================
class StreamingChunkBuffer:
    """Buffer for streaming audio chunks"""
    def __init__(self, chunk_size_samples: int = 1024):
        self.chunk_size = chunk_size_samples
        self.buffer = queue.Queue(maxsize=100)  # Buffer up to 100 chunks
        self.finished = False
        self.error = None
        
    def put_chunk(self, chunk: np.ndarray):
        """Add a chunk to the buffer"""
        if not self.finished:
            self.buffer.put(chunk)
    
    def put_finished(self):
        """Signal that no more chunks will be added"""
        self.finished = True
        self.buffer.put(None)  # Sentinel for end
    
    def put_error(self, error: Exception):
        """Signal an error occurred"""
        self.error = error
        self.finished = True
        self.buffer.put(None)
    
    def get_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get next chunk (blocks until available)"""
        try:
            chunk = self.buffer.get(timeout=timeout)
            if chunk is None:
                return None  # End of stream
            return chunk
        except queue.Empty:
            return None
    
    def chunks_available(self) -> bool:
        """Check if chunks are available"""
        return not self.buffer.empty() or not self.finished

# =============================================================================
#  TEXT AMERICANIZATION (Lightweight - No Phonemes)
# =============================================================================
class TextAmericanizer:
    """
    Convert British text patterns to American WITHOUT using phonemes.
    Only modifies TEXT that would be pronounced differently.
    """
    
    @staticmethod
    def americanize(text: str) -> str:
        """Convert British spellings/words to American equivalents"""
        
        # 1. Spelling changes (these DO affect TTS pronunciation in some models)
        spelling_map = {
            # -our → -or
            r'\b(col|hon|fav|flav|lab|neigh|rum|sav|vig)our\b': r'\1or',
            
            # -re → -er (some TTS models pronounce these differently)
            r'\b(cent|met|theat|fib|calibr|meag)re\b': r'\1er',
            
            # -ise → -ize
            r'\b(\w+)ise\b': r'\1ize',
            
            # Specific words
            r'\baluminium\b': 'aluminum',
            r'\bprogramme\b': 'program',
        }
        
        for pattern, replacement in spelling_map.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # 2. Words with DIFFERENT PRONUNCIATIONS (critical for accent)
        # Format: "british_word" → "american_respelling"
        pronunciation_map = {
            # These are spelled the same but pronounced differently
            # We'll use respelling tricks that work with TTS
            'schedule': 'skedule',      # British: "shed-yule" → American: "sked-yule"
            'tomato': 'tomayto',        # British: "tuh-mah-toe" → American: "tuh-may-toe"
            'garage': 'guh-rahj',       # British: "ga-ridge" → American: "guh-rahj"
            'vase': 'vayse',            # British: "vahz" → American: "vayse"
            'route': 'rout',            # British: "root" → American: "rout"
            'either': 'eether',         # British: "eye-ther" → American: "ee-ther"
            'neither': 'neether',       # British: "nye-ther" → American: "nee-ther"
            'advertisement': 'adver-tize-ment',  # Stress difference
            'privacy': 'pry-vuh-see',   # British: "priv-uh-see" → American: "pry-vuh-see"
        }
        
        for british, american in pronunciation_map.items():
            # Case-insensitive word boundary replacement
            pattern = r'\b' + re.escape(british) + r'\b'
            text = re.sub(pattern, american, text, flags=re.IGNORECASE)
        
        # 3. Vocabulary changes (different words entirely)
        vocab_map = {
            r'\blorry\b': 'truck',
            r'\bpetrol\b': 'gas',
            r'\blift\b': 'elevator',
            r'\bflat\b': 'apartment',
            r'\bholiday\b': 'vacation',
            r'\bqueue\b': 'line',
            r'\bbin\b': 'trash can',
            r'\bmaths\b': 'math',
        }
        
        for british, american in vocab_map.items():
            text = re.sub(british, american, text, flags=re.IGNORECASE)
        
        return text

# ElevenLabs Configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Voice IDs for different audio types
ELEVENLABS_VOICE_IDS = {
    "boy": "ipwVQXREGpbLFfvHevWY",  
    "girl": "7N5GkVRhUIV0JiQC4huQ"  
}   

MODEL_ID = "eleven_turbo_v2_5"  # Fast, high-quality model

# High quality voice settings
HIGH_QUALITY_SETTINGS = {
    "stability": 0.7,
    "similarity_boost": 0.95,
    "style": 0.3,
    "use_speaker_boost": True
}

logger = logging.getLogger(__name__)


class OptimizedXTTSEngine:
    """
     CORRECT: ElevenLabs streaming implementation
    """
    
    def __init__(self, worker_id: int = 0, default_speed: float = 1.0):
        self.worker_id = worker_id
        self.websocket = None
        self.context_counter = 0
        self._initialized = False
        self.sample_rate = 24000
        self.output_sample_rate = 48000
        self.default_speed = float(default_speed)
        self.device = "elevenlabs_api"
        
        logger.info(f"TTS Worker #{worker_id} → ElevenLabs API")
        
    async def initialize(self):
        """Initialize - we'll create connections per request"""
        if self._initialized:
            return
        
        try:
            logger.info(f" Worker #{self.worker_id} ready for ElevenLabs")
            self._initialized = True
            
        except Exception as e:
            logger.exception(f" Worker #{self.worker_id} init failed: {e}")
            raise
    
    def _get_websocket_uri(self, audio_type="boy"):
        """Generate WebSocket URI"""
        voice_id = ELEVENLABS_VOICE_IDS.get(audio_type, ELEVENLABS_VOICE_IDS["boy"])
        # Use the correct streaming endpoint
        return f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id={MODEL_ID}"
    
    async def generate(self, text: str, audio_type: str = "boy", language: str = "en-us", 
               speed: float = None) -> Tuple[np.ndarray, int]:
        """
        Generate complete audio using ElevenLabs streaming API
        """
        if not self._initialized:
            await self.initialize()
        
        #  Clean text before sending
        text = re.sub(r'\[(?:EMOTION|TENSION)\s+[^\]]*(?:\]|$)', '', text).strip()
        text = re.sub(r'^\[.*?\]\s*', '', text).strip()
        
        if not text or len(text) < 2:
            logger.warning(f"Worker #{self.worker_id} received empty text after cleaning")
            silence = np.zeros(int(self.output_sample_rate * 0.5), dtype=np.int16)
            return silence, self.output_sample_rate
        
        logger.info(f" Worker #{self.worker_id} sending to ElevenLabs: '{text}'")
        
        start_time = time.time()
        
        try:
            # Create new WebSocket connection for this request
            uri = self._get_websocket_uri(audio_type)
            
            logger.info(f"🔗 Worker #{self.worker_id} connecting to {uri}")
            
            async with websockets.connect(
                uri,
                additional_headers={"xi-api-key": ELEVENLABS_API_KEY}
            ) as websocket:
                
                logger.info(f" Worker #{self.worker_id} connected to ElevenLabs")
                
                # Collect all audio chunks
                audio_chunks = []
                
                # Send BOS (Beginning of Stream) message
                bos_message = {
                    "text": " ",
                    "voice_settings": {
                        "stability": 0.7,
                        "similarity_boost": 0.95
                    },
                    "language_code": "en",
                }
                
                await websocket.send(json.dumps(bos_message))
                logger.debug(f" Worker #{self.worker_id} sent BOS")
                
                # Send the actual text
                text_message = {
                    "text": text,
                    "try_trigger_generation": True,
                    "language_code": "en",
                }
                
                await websocket.send(json.dumps(text_message))
                logger.debug(f" Worker #{self.worker_id} sent text: '{text[:50]}...'")
                
                # Send EOS (End of Stream) message
                eos_message = {
                    "text": ""
                }
                
                await websocket.send(json.dumps(eos_message))
                logger.debug(f" Worker #{self.worker_id} sent EOS")
                
                # Receive audio chunks
                async for message in websocket:
                    try:
                        response = json.loads(message)
                        
                        #  CHECK FOR ERRORS
                        if "error" in response:
                            logger.error(f" ElevenLabs API Error: {response['error']}")
                            break
                        
                        if "message" in response:
                            logger.warning(f"ElevenLabs Message: {response['message']}")
                        
                        # Check for audio data
                        if "audio" in response:
                            audio_base64 = response["audio"]
                            if audio_base64 is not None:
                                audio_chunk = base64.b64decode(audio_base64)
                                audio_chunks.append(audio_chunk)
                                logger.debug(f"Worker #{self.worker_id} received chunk: {len(audio_chunk)} bytes")
                            else:
                                logger.warning(f"Worker #{self.worker_id} received None audio data")
                        
                        # Check if this is the final message
                        if response.get("isFinal", False):
                            logger.info(f" Worker #{self.worker_id} received final chunk")
                            break
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode message: {e}")
                        continue
                
                # Combine all chunks
                if not audio_chunks:
                    logger.error(f" Worker #{self.worker_id} received NO audio chunks!")
                    # Return silence
                    silence = np.zeros(int(self.output_sample_rate * 0.5), dtype=np.int16)
                    return silence, self.output_sample_rate
                
                combined_audio = b''.join(audio_chunks)
                logger.info(f"🎵 Worker #{self.worker_id} combined {len(audio_chunks)} chunks = {len(combined_audio)} bytes")
                
                # Convert to numpy array
                audio_array, sample_rate = self._mp3_to_numpy(combined_audio)
                
                
                # Convert to int16
                audio_int16 = (audio_array * 32767).astype(np.int16)
                
                elapsed = time.time() - start_time
                audio_duration = len(audio_int16) / sample_rate
                
                logger.info(
                    f"⚡ Worker #{self.worker_id} | {audio_type} | {len(text)} chars | "
                    f"{audio_duration:.2f}s audio | {elapsed:.2f}s generation"
                )
                
                return audio_int16, sample_rate
                
        except asyncio.TimeoutError:
            logger.error(f" Worker {self.worker_id} timeout")
            silence = np.zeros(int(self.output_sample_rate * 0.5), dtype=np.int16)
            return silence, self.output_sample_rate
            
        except Exception as e:
            logger.error(f" Worker {self.worker_id} generation error: {e}")
            import traceback
            traceback.print_exc()
            silence = np.zeros(int(self.output_sample_rate * 0.5), dtype=np.int16)
            return silence, self.output_sample_rate
    
    async def generate_streaming(self, text: str, audio_type: str = "boy", 
                                  language: str = "en-us", speed: float = None,
                                  chunk_duration_ms: int = 100) -> AsyncGenerator[np.ndarray, None]:
        """
        Stream audio chunks as they arrive
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            uri = self._get_websocket_uri(audio_type)
            
            async with websockets.connect(
                uri,
                extra_headers={"xi-api-key": ELEVENLABS_API_KEY}
            ) as websocket:
                
                # Send BOS
                await websocket.send(json.dumps({
                    "text": " ",
                    "voice_settings": HIGH_QUALITY_SETTINGS,
                    "xi_api_key": ELEVENLABS_API_KEY,
                }))
                
                # Send text
                await websocket.send(json.dumps({
                    "text": text,
                    "try_trigger_generation": True,
                }))
                
                # Send EOS
                await websocket.send(json.dumps({"text": ""}))
                
                # Stream chunks
                async for message in websocket:
                    try:
                        response = json.loads(message)
                        logger.debug(f" Worker #{self.worker_id} received: {response}")
                        
                        if "audio" in response:
                            audio_chunk = base64.b64decode(response["audio"])
                            
                            # Convert to numpy
                            audio_array, sr = self._mp3_to_numpy(audio_chunk)
                        
                        if response.get("isFinal", False):
                            break
                            
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"Streaming error: {e}")
    
    def _mp3_to_numpy(self, mp3_data: bytes) -> Tuple[np.ndarray, int]:
        """Convert MP3 bytes to numpy array"""
        try:
            audio_array, sample_rate = sf.read(BytesIO(mp3_data))
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            return audio_array.astype(np.float32), sample_rate
            
        except Exception as e:
            logger.error(f"MP3 conversion error: {e}")
            return np.array([], dtype=np.float32), 44100
    
    def cleanup(self):
        """Cleanup"""
        logger.info(f"🧹 Worker #{self.worker_id} cleaned up")


# =============================================================================
# TTS WORKER POOL
# =============================================================================
class TTSWorkerPool:
    def __init__(self, num_workers: int = 8):
        self.workers = []
        self.semaphore = asyncio.Semaphore(num_workers)
        self.num_workers = num_workers
        self._initialized = False
        self._lock = asyncio.Lock()
        self.active_tasks = {}  #  NEW: Track active TTS tasks
    
    async def initialize(self):
        if self._initialized:
            return
        
        logger.info(f" Initializing {self.num_workers} TTS workers with ElevenLabs...")
        
        for i in range(self.num_workers):
            worker = OptimizedXTTSEngine(worker_id=i)
            self.workers.append(worker)
            await worker.initialize()
        
        self._initialized = True
        logger.info(f" {self.num_workers} TTS workers ready (ElevenLabs)")
    
    async def generate(self, text: str, audio_type: str = "boy", 
                       language: str = "en-us", speed: float = None,
                       task_id: str = None) -> Tuple[np.ndarray, int]:  #  ADD task_id
        """Generate audio"""
        async with self.semaphore:
            worker_idx = self._select_worker(text, audio_type)
            worker = self.workers[worker_idx]
            
            #  NEW: Create cancellable task
            if task_id:
                task = asyncio.create_task(
                    worker.generate(text, audio_type, language, speed)
                )
                self.active_tasks[task_id] = task
                
                try:
                    result = await task
                    return result
                finally:
                    # Clean up
                    if task_id in self.active_tasks:
                        del self.active_tasks[task_id]
            else:
                return await worker.generate(text, audio_type, language, speed)
        
    def cancel_task(self, task_id: str):
        """ NEW: Cancel a specific TTS task"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            if not task.done():
                task.cancel()
                logger.info(f"🛑 Cancelled TTS task: {task_id}")
    
    async def generate_streaming(self, text: str, audio_type: str = "boy", 
                                  language: str = "en-us", speed: float = None,
                                  chunk_duration_ms: int = 100) -> AsyncGenerator[np.ndarray, None]:
        """Stream audio"""
        async with self.semaphore:
            worker_idx = self._select_worker(text, audio_type)
            worker = self.workers[worker_idx]
            
            async for chunk in worker.generate_streaming(text, audio_type, language, speed, chunk_duration_ms):
                yield chunk
    
    def _select_worker(self, text: str, audio_type: str) -> int:
        """Select worker"""
        cache_key = f"{text[:50]}:{audio_type}"
        hash_val = int(hashlib.md5(cache_key.encode()).hexdigest(), 16)
        return hash_val % len(self.workers)
    
    def cleanup(self):
        for worker in self.workers:
            worker.cleanup()


# Initialize pool
tts_pool = TTSWorkerPool()
# =============================================================================
# INTERRUPTION MANAGER
# =============================================================================
@dataclass
class InteractionState:
    id: str
    start_time: float
    is_interrupted: bool = False
    is_generating: bool = False
    is_speaking: bool = False

# class InterruptionManager:
#     def __init__(self):
#         self.client_interactions = {}
#         self._lock = asyncio.Lock()
#         self.interruption_events = {}
#         self.interruption_callbacks = {}
#         self.active_tts_tasks = {}  #  NEW: Track TTS tasks per client
    
    

#     async def start_interaction(self, client_id: str, interaction_id: str) -> InteractionState:
#         async with self._lock:  #  This is inside async method, so it's fine
#             # Clear any previous interruption state
#             if client_id in self.interruption_events:
#                 self.interruption_events[client_id].clear()
#             else:
#                 self.interruption_events[client_id] = asyncio.Event()
            
#             # Clear any callbacks
#             if client_id in self.interruption_callbacks:
#                 del self.interruption_callbacks[client_id]
            
#             interaction = InteractionState(id=interaction_id, start_time=time.time())
#             self.client_interactions[client_id] = interaction
#             logger.info(f"🆕 Started interaction {interaction_id} for client {client_id}")
#             return interaction
    
#     # async def clear_interruption(self, client_id: str) -> None:
#     #     """Clear interruption state for new interaction"""
#     #     async with self._lock:
#     #         if client_id in self.interruption_events:
#     #             self.interruption_events[client_id].clear()
#     #             logger.debug(f" Cleared interruption event for {client_id}")
            
#     #         # Reset interrupted flag for new interaction
#     #         if client_id in self.client_interactions:
#     #             self.client_interactions[client_id].is_interrupted = False
    
#     async def clear_interruption(self, client_id: str) -> None:
#         async with self._lock:
#             if client_id in self.interruption_events:
#                 self.interruption_events[client_id].clear()
#             if client_id in self.client_interactions:
#                 self.client_interactions[client_id].is_interrupted = False
#             if client_id in self.active_tts_tasks:
#                 self.active_tts_tasks[client_id].clear()
                
#     async def signal_interruption(self, client_id: str) -> bool:
#         """
#         Signal interruption - IMMEDIATELY stops and clears everything
#          UPDATED VERSION with queue clearing
#         """
#         async with self._lock:
#             if client_id not in self.client_interactions:
#                 logger.debug(f"No active interaction to interrupt for {client_id}")
#                 return False
            
#             # Set the interruption event
#             if client_id in self.interruption_events:
#                 self.interruption_events[client_id].set()
#                 logger.debug(f"Interruption event set for {client_id}")
            
#             interaction = self.client_interactions[client_id]
#             if interaction.is_interrupted:
#                 logger.debug(f"Interaction already interrupted for {client_id}")
#                 return False
            
#             interaction.is_interrupted = True
            
#             #  Clear TTS tasks immediately
#             if client_id in self.active_tts_tasks:
#                 for task_id in list(self.active_tts_tasks[client_id]):
#                     tts_pool.cancel_task(task_id)
#                     logger.info(f"🛑 Cancelled TTS task {task_id}")
#                 self.active_tts_tasks[client_id].clear()
            
#             #  Clear any pending query queue for this client
#             if client_id in client_sessions:
#                 if 'pending_queries' in client_sessions[client_id]:
#                     client_sessions[client_id]['pending_queries'] = []
#                     logger.info(f"🧹 Cleared pending queries for {client_id}")
            
#             # Send stop signal to frontend
#             if client_id in client_sessions:
#                 try:
#                     ws = client_sessions[client_id]['websocket']
#                     stop_message = {
#                         "type": "immediate_stop",
#                         "status": "interrupted",
#                         "timestamp": time.time(),
#                         "interaction_id": interaction.id if hasattr(interaction, 'id') else "unknown",
#                         "clear_queue": True,
#                         "flush_pending": True,
#                         "ready_for_input": True  #  Tell frontend we're ready for new input
#                     }
#                     asyncio.create_task(ws.send_json(stop_message))
#                 except Exception as e:
#                     logger.error(f"Failed to send stop signal: {e}")
            
#             logger.info(f"🛑 Interruption ACCEPTED for client {client_id}")
#             stats["interruptions"] += 1
#             return True
    
    
#     async def clear_interruption(self, client_id: str) -> None:
#         """
#         Clear interruption state to allow new interaction
#          NEW METHOD - Call this before processing new queries
#         """
#         async with self._lock:
#             # Clear the interruption event
#             if client_id in self.interruption_events:
#                 self.interruption_events[client_id].clear()
#                 logger.debug(f" Cleared interruption event for {client_id}")
            
#             # Reset interrupted flag for new interaction
#             if client_id in self.client_interactions:
#                 self.client_interactions[client_id].is_interrupted = False
#                 logger.info(f" Reset interruption state for {client_id} - ready for new query")
            
#             # Optional: Clear any stale task references
#             if client_id in self.active_tts_tasks:
#                 self.active_tts_tasks[client_id].clear()
#                 logger.debug(f"Cleared stale TTS task references for {client_id}")
    
#     def register_tts_task(self, client_id: str, task_id: str):
#         """ NEW: Register a TTS task for tracking"""
#         if client_id not in self.active_tts_tasks:
#             self.active_tts_tasks[client_id] = set()
#         self.active_tts_tasks[client_id].add(task_id)
    
#     def unregister_tts_task(self, client_id: str, task_id: str):
#         """ NEW: Unregister a completed TTS task"""
#         if client_id in self.active_tts_tasks:
#             self.active_tts_tasks[client_id].discard(task_id)
    
#     async def check_interrupted(self, client_id: str, interaction_id: str) -> bool:
#         """Check if current interaction should be interrupted"""
#         async with self._lock:
#             if client_id not in self.client_interactions:
#                 logger.debug(f"🔍 No interaction found for {client_id} - NOT interrupted")
#                 return False  #  FIX: Don't interrupt if no interaction exists
            
#             interaction = self.client_interactions[client_id]
            
#             # Check if this is the wrong interaction ID
#             if interaction.id != interaction_id:
#                 logger.debug(f"🔍 Wrong interaction ID: expected {interaction_id}, got {interaction.id}")
#                 return True        
#             # Check if interruption event is set
#             if client_id in self.interruption_events and self.interruption_events[client_id].is_set():
#                 logger.debug(f"Interruption event detected for {client_id}")
#                 return True
            
#             return interaction.is_interrupted

#     def register_cancellation_callback(self, client_id: str, callback):
#         """Register a callback to be called when interrupted"""
#         # This is a synchronous method, so DON'T use async with here
#         with asyncio.Lock():  
#             if client_id not in self.interruption_callbacks:
#                 self.interruption_callbacks[client_id] = []
#             self.interruption_callbacks[client_id].append(callback)

#     async def clear_interruption(self, client_id: str):
#         """Clear interruption state for client"""
#         async with self._lock:  #  This is inside async method
#             if client_id in self.interruption_events:
#                 self.interruption_events[client_id].clear()
#             if client_id in self.interruption_callbacks:
#                 del self.interruption_callbacks[client_id]
#             if client_id in self.client_interactions:
#                 del self.client_interactions[client_id]
#             logger.debug(f"Cleared interruption state for {client_id}")





class InterruptionManager:
    def __init__(self):
        self.client_interactions = {}
        self._lock = asyncio.Lock()
        self.interruption_events = {}
        self.active_tts_tasks = {}

    async def start_interaction(self, client_id: str, interaction_id: str) -> InteractionState:
        async with self._lock:
            # Set event first time if not exists
            if client_id not in self.interruption_events:
                self.interruption_events[client_id] = asyncio.Event()
            else:
                # Clear any previous interruption for fresh start
                self.interruption_events[client_id].clear()

            interaction = InteractionState(id=interaction_id, start_time=time.time())
            self.client_interactions[client_id] = interaction
            logger.info(f"🆕 Started interaction {interaction_id} for {client_id}")
            return interaction

    async def signal_interruption(self, client_id: str) -> bool:
        async with self._lock:
            if client_id not in self.client_interactions:
                logger.debug(f"No active interaction to interrupt for {client_id}")
                return False

            interaction = self.client_interactions[client_id]

            if interaction.is_interrupted:
                logger.debug(f"Already interrupted for {client_id}")
                return False

            interaction.is_interrupted = True

            # Set the event so all awaiters unblock immediately
            if client_id in self.interruption_events:
                self.interruption_events[client_id].set()

            # Cancel active TTS tasks
            if client_id in self.active_tts_tasks:
                for task_id in list(self.active_tts_tasks[client_id]):
                    tts_pool.cancel_task(task_id)
                    logger.info(f"🛑 Cancelled TTS task {task_id}")
                self.active_tts_tasks[client_id].clear()

            # Clear pending queries
            if client_id in client_sessions:
                client_sessions[client_id].pop('pending_queries', None)

            # Send stop to frontend
            if client_id in client_sessions:
                try:
                    ws = client_sessions[client_id]['websocket']
                    if ws:
                        asyncio.create_task(ws.send_json({
                            "type": "immediate_stop",
                            "status": "interrupted",
                            "timestamp": time.time(),
                            "interaction_id": interaction.id,
                            "clear_queue": True,
                            "flush_pending": True,
                            "ready_for_input": True
                        }))
                except Exception as e:
                    logger.error(f"Failed to send stop signal: {e}")

            logger.info(f"🛑 Interruption ACCEPTED for {client_id}")
            stats["interruptions"] += 1
            return True

    async def clear_interruption(self, client_id: str) -> None:
        """
        Reset interruption state so a NEW interaction can start cleanly.
        Does NOT delete the interaction — only resets the flags.
        """
        async with self._lock:
            # Clear the event so future checks return False
            if client_id in self.interruption_events:
                self.interruption_events[client_id].clear()

            # Reset interrupted flag — keep the interaction entry alive
            if client_id in self.client_interactions:
                self.client_interactions[client_id].is_interrupted = False

            # Clear stale TTS task references
            if client_id in self.active_tts_tasks:
                self.active_tts_tasks[client_id].clear()

            logger.debug(f"✅ Cleared interruption state for {client_id}")

    async def check_interrupted(self, client_id: str, interaction_id: str) -> bool:
        async with self._lock:
            if client_id not in self.client_interactions:
                return False

            interaction = self.client_interactions[client_id]

            # Wrong interaction ID means this one is stale
            if interaction.id != interaction_id:
                return True

            # Check event (set by signal_interruption)
            if client_id in self.interruption_events:
                if self.interruption_events[client_id].is_set():
                    return True

            return interaction.is_interrupted

    def register_tts_task(self, client_id: str, task_id: str):
        if client_id not in self.active_tts_tasks:
            self.active_tts_tasks[client_id] = set()
        self.active_tts_tasks[client_id].add(task_id)

    def unregister_tts_task(self, client_id: str, task_id: str):
        if client_id in self.active_tts_tasks:
            self.active_tts_tasks[client_id].discard(task_id)

interruption_manager = InterruptionManager()





# =============================================================================
# CONTEXT-AWARE EMOTION DETECTION
# =============================================================================

import re
from typing import Tuple, Dict
from collections import defaultdict

def detect_emotion_from_context_enhanced(
    user_input: str,
    llm_emotion: str = None
) -> Tuple[str, str]:
    """
    🧠 DEEP CONTEXT-AWARE EMOTION DETECTION

    Signals used:
    - Explicit emotion tags
    - LLM emotion (weighted)
    - Keyword scoring
    - Negation handling
    - Intensity modifiers
    - Emojis & punctuation
    - Question + concern blending
    - Sarcasm / resignation cues
    """

    text = user_input.strip()
    text_lower = text.lower()

    # ------------------------------------------------------------------
    # 1. Explicit emotion tag [emotion]
    # ------------------------------------------------------------------
    emotion_match = re.match(r'\[([^\]]+)\]\s*(.+)', text)
    if emotion_match:
        emotion = emotion_match.group(1).lower()
        clean_input = emotion_match.group(2)
        logger.info(f"🎭 Explicit emotion tag: {emotion}")
        return emotion, clean_input

    # ------------------------------------------------------------------
    # 2. Emotion buckets with weights
    # ------------------------------------------------------------------
    emotion_scores: Dict[str, float] = defaultdict(float)

    emotion_keywords = {
        "happy": [
            "happy", "joy", "love", "great", "awesome", "amazing", "fantastic",
            "delighted", "excited", "pleased"
        ],
        "sad": [
            "sad", "depressed", "lonely", "heartbroken", "miserable",
            "hopeless", "crying", "grief", "lost"
        ],
        "angry": [
            "angry", "furious", "mad", "rage", "annoyed", "irritated",
            "hate", "pissed", "damn"
        ],
        "fear": [
            "scared", "afraid", "terrified", "panic", "fear",
            "danger", "threat"
        ],
        "anxious": [
            "anxious", "worried", "stress", "tension", "nervous",
            "overthinking", "uncertain"
        ],
        "concerned": [
            "help", "issue", "problem", "suffering", "pain",
            "symptom", "diagnosis", "treatment", "condition"
        ],
        "excited": [
            "omg", "wow", "can't wait", "thrilled", "pumped", "stoked"
        ],
        "surprised": [
            "shocked", "unexpected", "no way", "really", "astonished"
        ],
        "resigned": [
            "whatever", "it is what it is", "fine i guess", "doesn't matter",
            "no choice"
        ],
        "confused": [
            "confused", "lost", "don't understand", "what is going on",
            "unclear"
        ],
        "curious": [
            "why", "how", "what", "can you explain", "i wonder"
        ]
    }

    # ------------------------------------------------------------------
    # 3. Negation handling
    # ------------------------------------------------------------------
    negations = {"not", "never", "no", "hardly", "rarely"}

    def is_negated(keyword: str) -> bool:
        pattern = r'(not|never|no)\s+\w*\s*' + re.escape(keyword)
        return re.search(pattern, text_lower) is not None

    # ------------------------------------------------------------------
    # 4. Intensity modifiers
    # ------------------------------------------------------------------
    amplifiers = {
        "very": 1.5,
        "extremely": 2.0,
        "so": 1.3,
        "really": 1.3,
        "too": 1.2
    }

    dampeners = {
        "slightly": 0.5,
        "a bit": 0.6,
        "somewhat": 0.7
    }

    def intensity_multiplier() -> float:
        multiplier = 1.0
        for word, value in amplifiers.items():
            if word in text_lower:
                multiplier *= value
        for word, value in dampeners.items():
            if word in text_lower:
                multiplier *= value
        return multiplier

    intensity = intensity_multiplier()

    # ------------------------------------------------------------------
    # 5. Score emotions
    # ------------------------------------------------------------------
    for emotion, keywords in emotion_keywords.items():
        for kw in keywords:
            if kw in text_lower:
                if not is_negated(kw):
                    emotion_scores[emotion] += 1.0 * intensity
                else:
                    emotion_scores[emotion] -= 0.8  # negation penalty

    # ------------------------------------------------------------------
    # 6. Emojis & punctuation
    # ------------------------------------------------------------------
    if re.search(r"[!]{2,}", text):
        emotion_scores["excited"] += 0.8
        emotion_scores["angry"] += 0.5

    emoji_map = {
        "😢": "sad",
        "😭": "sad",
        "😡": "angry",
        "😠": "angry",
        "😨": "fear",
        "😰": "anxious",
        "😍": "happy",
        "😊": "happy",
        "😲": "surprised"
    }

    for emoji, emotion in emoji_map.items():
        if emoji in text:
            emotion_scores[emotion] += 1.5

    # ------------------------------------------------------------------
    # 7. Question + concern blending
    # ------------------------------------------------------------------
    if "?" in text and emotion_scores["concerned"] > 0:
        emotion_scores["anxious"] += 0.8

    if "?" in text and emotion_scores["curious"] > 0:
        emotion_scores["curious"] += 0.5

    # ------------------------------------------------------------------
    # 8. LLM emotion influence (soft weight)
    # ------------------------------------------------------------------
    if llm_emotion and llm_emotion.lower() not in {"neutral", "none", ""}:
        emotion_scores[llm_emotion.lower()] += 1.2

    # ------------------------------------------------------------------
    # 9. Select dominant emotion
    # ------------------------------------------------------------------
    if emotion_scores:
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        if emotion_scores[dominant_emotion] > 0.7:
            logger.info(f"🎭 Detected emotion → {dominant_emotion} | scores={dict(emotion_scores)}")
            return dominant_emotion, text

    # ------------------------------------------------------------------
    # 10. Intelligent fallback
    # ------------------------------------------------------------------
    if "?" in text:
        return "curious", text

    if len(text.split()) <= 4:
        return "neutral", text

    return "attentive", text

# =============================================================================
# BLENDSHAPE ORDER (0-103 ONLY - Excludes everything after temp)
# =============================================================================

BLENDSHAPE_ORDER = [
    # 0-11: Base, Cheeks, Eyes, Neck
    "Basis", "cheekPuffIn", "cheekPuffOut", "eyeLookDownRight",
    "neckUp", "neckDown", "neckLeft", "neckRight",
    "neckTurnLeft", "neckTurnRight", "neckForward", "neckBackward",
    
    # 12-16: NEW - Neck Tilts and Cheek Talk
    "neckDownTiltLeft", "neckDownTiltRight", "neckUpTiltLeft", "neckUpTiltRight",
    "cheekTalk",
    
    # 17-29: Eyes (indices shifted by 5)
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft",
    "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft",
    "eyeBlinkRight", "eyeLookInRight", "eyeLookOutRight",
    "eyeLookUpRight", "eyeSquintRight", "eyeWideRight",
    
    # 30-35: Jaw
    "jawForward", "jawLeft", "jawRight", "jawOpenOriginal", "jawOpen", "jawDrop",
    
    # 36-56: Mouth
    "mouthClose", "mouthFunnel", "mouthPucker", "mouthRight", "mouthLeft",
    "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthPressLeft", "mouthPressRight",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    
    # 57-71: Brows, Cheeks, Nose
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight",
    
    # 72-87: Tongue V-series
    "V_Tongue_up", "V_Tongue_Raise", "V_Tongue_Out", "V_Tongue_Narrow",
    "V_Tongue_Lower", "V_Tongue_Curl_U", "V_Tongue_Curl_D",
    "Tongue_In", "Tongue_Up", "Tongue_Down", "Tongue_Narrow", "Tongue_Roll",
    "Tongue_Tip_L", "Tongue_Tip_R", "Tongue_Twist_L", "Tongue_Twist_R",
    
    # 88-108: Visemes
    "e", "r", "oh", "ow", "oo", "bmp", "fv", "th", "d", "g", "n",
    "sh", "s", "ch", "k", "l", "w", "y", "ih", "sil", "aa",
    
    # 109-112: Shoulders
    "shoulderUp", "shoulderDown", "shoulderForward", "shoulderBackward"
]

NUM_BLENDSHAPES = len(BLENDSHAPE_ORDER)  # Now 113
BLENDSHAPE_INDEX = {name: i for i, name in enumerate(BLENDSHAPE_ORDER)}
# Right after BLENDSHAPE_INDEX = {...}

#  Validate blendshape system
print("=" * 60)
print("🔍 BLENDSHAPE SYSTEM VALIDATION")
print("=" * 60)
print(f"Total blendshapes: {NUM_BLENDSHAPES}")
print(f"BLENDSHAPE_ORDER length: {len(BLENDSHAPE_ORDER)}")
print(f"BLENDSHAPE_INDEX length: {len(BLENDSHAPE_INDEX)}")

if NUM_BLENDSHAPES != len(BLENDSHAPE_ORDER):
    raise ValueError(
        f" MISMATCH: NUM_BLENDSHAPES={NUM_BLENDSHAPES} but "
        f"BLENDSHAPE_ORDER has {len(BLENDSHAPE_ORDER)} items"
    )

if NUM_BLENDSHAPES != len(BLENDSHAPE_INDEX):
    raise ValueError(
        f" MISMATCH: NUM_BLENDSHAPES={NUM_BLENDSHAPES} but "
        f"BLENDSHAPE_INDEX has {len(BLENDSHAPE_INDEX)} items"
    )

# Check for duplicates
if len(set(BLENDSHAPE_ORDER)) != len(BLENDSHAPE_ORDER):
    duplicates = [x for x in BLENDSHAPE_ORDER if BLENDSHAPE_ORDER.count(x) > 1]
    raise ValueError(f" Duplicate blendshapes found: {set(duplicates)}")

print(f" All validation passed")
print(f" Unity value range: 0-100")
print(f" First blendshape: {BLENDSHAPE_ORDER[0]} (index 0)")
print(f" Last blendshape: {BLENDSHAPE_ORDER[-1]} (index {NUM_BLENDSHAPES-1})")
print("=" * 60 + "\n")
print(f" Using {NUM_BLENDSHAPES} blendshapes (indices 0-{NUM_BLENDSHAPES-1})")

def parse_groq_emotion(text: str) -> Tuple[str, str, float, str]:
    """
    Parse emotion metadata from Groq response.
    Your LLM already does this, but this is a backup parser.
    """
    pattern = r'\[EMOTION\s+name=(\w+)\s+intensity=([\d.]+)\s+facial_state=(\w+)\]\s*'
    match = re.search(pattern, text)
    
    if match:
        emotion_name = match.group(1).lower()
        intensity = float(match.group(2))
        facial_state = match.group(3).lower()
        clean_text = re.sub(pattern, '', text).strip()
        return clean_text, emotion_name, intensity, facial_state
    
    return text, "neutral", 0.5, "neutral"

def apply_emotion_with_intensity_LOCKED(
    base_shapes: Dict[str, float],
    emotion: str,
    intensity: float,
    facial_state: str = "neutral",
    is_speaking: bool = False
) -> Dict[str, float]:
    """
     FIXED: Only lock MOUTH shapes, NOT eyebrows/eyes
    """
    result = base_shapes.copy()
    emotion_mods = EMOTION_MODIFIERS.get(emotion, EMOTION_MODIFIERS["neutral"])
    
    #  CRITICAL: Define ONLY mouth-articulation shapes
    MOUTH_ARTICULATION_ONLY = {
        "mouthSmileLeft", "mouthSmileRight",
        "mouthFrownLeft", "mouthFrownRight",
        "mouthPucker", "mouthFunnel",
        "mouthStretchLeft", "mouthStretchRight",
        "jawOpen"
    }
    
    for shape, base_value in emotion_mods.items():
        scaled_value = base_value * intensity
        
        #  ONLY reduce MOUTH shapes during speech
        if is_speaking and shape in MOUTH_ARTICULATION_ONLY:
            scaled_value *= 0.30  # Reduce mouth expressions
        #  Eyebrows/eyes get FULL intensity!
        
        # Apply facial state (only for non-mouth)
        if shape not in MOUTH_ARTICULATION_ONLY:
            if facial_state == "tense" and "brow" in shape.lower():
                scaled_value *= 1.3
            elif facial_state == "soft" and "cheek" in shape.lower():
                scaled_value *= 0.9
        
        result[shape] = min(1.0, result.get(shape, 0.0) + scaled_value)
    
    return result

def generate_blendshape_keyframe(
    viseme: str,
    emotion: str,
    intensity: float,
    facial_state: str
) -> Dict:
    """
    Generate blendshapes for a single phoneme with Groq emotion.
    
    Returns values in Unity 0-100 range for indices 0-103.
    """
    # Get base mouth shapes from viseme (already in 0-100 scale)
    base_shapes = VISEME_TO_MOUTH_SHAPES.get(viseme, {}).copy()
    
    # Apply emotion with intensity scaling
    final_shapes = apply_emotion_with_intensity_LOCKED(
        base_shapes,
        emotion,
        intensity,
        facial_state
    )
    
    blendshape_array = [0.0] * NUM_BLENDSHAPES  # Was: [0.0] * 104
    
    for shape_name, value in final_shapes.items():
        if shape_name in BLENDSHAPE_INDEX:
            idx = BLENDSHAPE_INDEX[shape_name]
            #  Add safety check
            if idx < NUM_BLENDSHAPES:
                blendshape_array[idx] = min(100.0, max(0.0, value))
    
    return {
        "values": blendshape_array,
        "values": blendshape_array,
        "metadata": {
            "viseme": viseme,
            "emotion": emotion,
            "intensity": intensity,
            "facial_state": facial_state
        }
    }
# =============================================================================
#  EMOTION-BASED NECK TURN PATTERNS (NEW)
# =============================================================================
NECK_TURN_PATTERNS = {
    "happy": {
        "turns": [
            {"neckTurnLeft": 0.18, "neckTurnRight": 0.0, "duration": 1.2},
            {"neckTurnRight": 0.18, "neckTurnLeft": 0.0, "duration": 1.2},
            {"neckTurnLeft": 0.12, "neckTurnRight": 0.0, "duration": 0.9}
        ],
        "frequency": 1.5,
        "intensity_mod": 1.0,
        "description": "Engaging happy turns"
    },
    "excited": {
        "turns": [
            {"neckTurnLeft": 0.25, "neckTurnRight": 0.0, "duration": 0.8},
            {"neckTurnRight": 0.25, "neckTurnLeft": 0.0, "duration": 0.8},
            {"neckTurnLeft": 0.20, "duration": 0.6}
        ],
        "frequency": 1.0,
        "intensity_mod": 1.3,
        "description": "Quick energetic turns"
    },
    "sad": {
        "turns": [
            {"neckTurnLeft": 0.08, "neckTurnRight": 0.0, "duration": 2.0},
            {"neckTurnRight": 0.08, "neckTurnLeft": 0.0, "duration": 2.0}
        ],
        "frequency": 3.0,
        "intensity_mod": 0.6,
        "description": "Slow melancholic turns"
    },
    "angry": {
        "turns": [
            {"neckTurnLeft": 0.22, "neckTurnRight": 0.0, "duration": 0.7},
            {"neckTurnRight": 0.22, "neckTurnLeft": 0.0, "duration": 0.7},
            {"neckTurnLeft": 0.18, "duration": 0.5}
        ],
        "frequency": 1.2,
        "intensity_mod": 1.2,
        "description": "Sharp aggressive turns"
    },
    "surprised": {
        "turns": [
            {"neckTurnLeft": 0.20, "neckTurnRight": 0.0, "duration": 0.5},
            {"neckTurnRight": 0.20, "neckTurnLeft": 0.0, "duration": 0.5},
            {"neckTurnLeft": 0.15, "duration": 0.4}
        ],
        "frequency": 2.0,
        "intensity_mod": 1.1,
        "description": "Quick surprised looks"
    },
    "concerned": {
        "turns": [
            {"neckTurnLeft": 0.12, "neckTurnRight": 0.0, "duration": 1.5},
            {"neckTurnRight": 0.12, "neckTurnLeft": 0.0, "duration": 1.5}
        ],
        "frequency": 2.5,
        "intensity_mod": 0.8,
        "description": "Concerned evaluating turns"
    },
    "curious": {
        "turns": [
            {"neckTurnLeft": 0.15, "neckTurnRight": 0.0, "duration": 1.0},
            {"neckTurnRight": 0.15, "neckTurnLeft": 0.0, "duration": 1.0},
            {"neckTurnLeft": 0.10, "duration": 0.7}
        ],
        "frequency": 2.0,
        "intensity_mod": 0.9,
        "description": "Inquisitive head turns"
    },
    "friendly": {
        "turns": [
            {"neckTurnLeft": 0.14, "neckTurnRight": 0.0, "duration": 1.3},
            {"neckTurnRight": 0.14, "neckTurnLeft": 0.0, "duration": 1.3}
        ],
        "frequency": 2.0,
        "intensity_mod": 0.95,
        "description": "Welcoming friendly turns"
    },
    "encouraging": {
        "turns": [
            {"neckTurnLeft": 0.16, "neckTurnRight": 0.0, "duration": 1.1},
            {"neckTurnRight": 0.16, "neckTurnLeft": 0.0, "duration": 1.1},
            {"neckTurnLeft": 0.12, "duration": 0.8}
        ],
        "frequency": 1.8,
        "intensity_mod": 1.05,
        "description": "Motivational engaging turns"
    },
    "professional_friendly": {
        "turns": [
            {"neckTurnLeft": 0.10, "neckTurnRight": 0.0, "duration": 1.6},
            {"neckTurnRight": 0.10, "neckTurnLeft": 0.0, "duration": 1.6}
        ],
        "frequency": 2.8,
        "intensity_mod": 0.7,
        "description": "Professional subtle turns"
    },
    "neutral": {
        "turns": [
            {"neckTurnLeft": 0.08, "neckTurnRight": 0.0, "duration": 1.8},
            {"neckTurnRight": 0.08, "neckTurnLeft": 0.0, "duration": 1.8}
        ],
        "frequency": 3.5,
        "intensity_mod": 0.8,
        "description": "Neutral minimal turns"
    }
}
# =============================================================================
# VISEME MAP
# =============================================================================
VISEME_MAP = {
    "AA": "aa","AE": "e","AH": "aa","AO": "oh","AW": "ow","AY": "ih",
    "EH": "e","ER": "r","EY": "e","IH": "ih","IY": "ih","OW": "ow",
    "UH": "oo","UW": "oo",
    "B": "bmp","P": "bmp","M": "bmp",
    "F": "fv","V": "fv",
    "TH": "th","DH": "th",
    "T": "d","D": "d",
    "S": "s","Z": "s",
    "SH": "sh","CH": "ch","JH": "ch",
    "K": "k","G": "g",
    "N": "n","L": "l","R": "r",
    "W": "w","Y": "y",
    "HH": "sil"
}

# =============================================================================
# REDUCED JAW ARTICULATION - Prevents over-opening and teeth exposure
# =============================================================================

VISEME_JAW_OPENING = {
    "aa": 0.08,   # was 0.45 - natural wide vowel
    "oh": 0.07,   # was 0.40
    "ow": 0.07,   # was 0.35
    "e": 0.07,    # was 0.30
    "ih": 0.02,   # was 0.25
    "oo": 0.03,   # was 0.20
    "r": 0.08,
    "l": 0.08,
    "w": 0.08,
    "y": 0.03,
    "bmp": 0.0,   # Fully closed
    "fv": 0.03,
    "th": 0.01,
    "d": 0.03,
    "s": 0.03,
    "sh": 0.03,
    "ch": 0.03,
    "k": 0.09,
    "g": 0.6,
    "n": 0.07,
    "sil": 0.02
}

# Emotion-based jaw opening modifiers
EMOTION_JAW_MODIFIERS = {
    "happy": 0.95,      # Slightly more open when happy
    "excited":1,    # More expressive opening
    "sad": 0.65,        # Less jaw movement
    "angry": 1.02,      # Slightly more tension
    "surprised": 1.20,  # Much more open
    "fear": 0.95,       # Slight tension
    "disgust": 0.75,    # Restricted
    "neutral": 0.8,      # Base level
    "concerned": 0.75,   # Less jaw movement (serious tone)
    "curious": 0.90 
}

# =============================================================================
#  EMOTION MODIFIERS - INCREASED INTENSITY (3-4x boost for visibility)
# =============================================================================
EMOTION_MODIFIERS = {
    "happy": {
        "cheekSquintLeft": 0.49,
        "cheekSquintRight": 0.49,
        "cheekPuff": -0.2625,

        "browOuterUpLeft": 1.1375,
        "browOuterUpRight": 1.1375,
        "browInnerUp": 0.525,

        "eyeSquintLeft": 0.6125,
        "eyeSquintRight": 0.6125,

        "mouthSmileLeft": 0.875,
        "mouthSmileRight": 0.875,
    },

    "cheerful": {
        "cheekSquintLeft": 0.6125,
        "cheekSquintRight": 0.6125,
        "cheekPuff": -0.21,

        "browOuterUpLeft": 1.3125,
        "browOuterUpRight": 1.3125,
        "browInnerUp": 0.70,

        "eyeSquintLeft": 0.735,
        "eyeSquintRight": 0.735,

        "mouthSmileLeft": 1.1375,
        "mouthSmileRight": 1.1375,
    },

    "encouraging": {
        "browInnerUp": 0.9625,
        "browOuterUpLeft": 1.05,
        "browOuterUpRight": 1.05,

        "eyeWideLeft": 0.49,
        "eyeWideRight": 0.49,

        "cheekSquintLeft": 0.35,
        "cheekSquintRight": 0.35,

        "mouthSmileLeft": 0.6125,
        "mouthSmileRight": 0.6125,
    },

    "excited": {
        "eyeWideLeft": 1.085,
        "eyeWideRight": 1.085,

        "browOuterUpLeft": 1.19,
        "browOuterUpRight": 1.19,
        "browInnerUp": 0.7875,

        "cheekSquintLeft": 0.70,
        "cheekSquintRight": 0.70,

        "mouthSmileLeft": 0.91,
        "mouthSmileRight": 0.91,
    },

    "sad": {
        "mouthFrownLeft": 1.40,
        "mouthFrownRight": 1.40,
        "mouthLowerDownLeft": 0.6125,
        "mouthLowerDownRight": 0.6125,

        "browInnerUp": 0.6125,
        "browDownLeft": 1.3125,
        "browDownRight": 1.3125,

        "eyeSquintLeft": 0.49,
        "eyeSquintRight": 0.49,

        "cheekSquintLeft": -0.2625,
        "cheekSquintRight": -0.2625,
    },

    "concerned": {
        "browInnerUp": 0.735,
        "browDownLeft": 0.4375,
        "browDownRight": 0.4375,

        "mouthFrownLeft": 0.70,
        "mouthFrownRight": 0.70,

        "eyeSquintLeft": 0.35,
        "eyeSquintRight": 0.35,
    },

    "curious": {
        "browOuterUpLeft": 0.9625,
        "browOuterUpRight": 0.9625,
        "browInnerUp": 0.665,

        "eyeWideLeft": 0.35,
        "eyeWideRight": 0.35,

        "cheekSquintLeft": 0.245,
        "cheekSquintRight": 0.245,
    },

    "calm": {
        "eyeSquintLeft": 0.175,
        "eyeSquintRight": 0.175,

        "mouthSmileLeft": 0.315,
        "mouthSmileRight": 0.315,

        "browOuterUpLeft": 0.35,
        "browOuterUpRight": 0.35,
    },

    "attentive": {
        "browOuterUpLeft": 0.665,
        "browOuterUpRight": 0.665,
        "browInnerUp": 0.49,

        "eyeWideLeft": 0.315,
        "eyeWideRight": 0.315,
    },

    "neutral": {
        "cheekSquintLeft": 0.105,
        "cheekSquintRight": 0.105,
        "eyeSquintLeft": 0.0525,
        "eyeSquintRight": 0.0525,
    }
}

EXPRESSION_POOL = [f"n{i}" for i in range(1, 11)]

EXPRESSION_TO_SHAPES = {
    "n1": {"browInnerUp": 0.08},
    "n2": {"browOuterUpLeft": 0.04, "browOuterUpRight": 0.07},
    "n3": {"eyeSquintLeft": 0.12, "eyeSquintRight": 0.12},
    "n4": {"cheekSquintLeft": 0.15, "cheekSquintRight": 0.09},
    "n5": {"noseSneerLeft": 0.11, "noseSneerRight": 0.11},
    "n6": {"browDownLeft": 0.07, "browDownRight": 0.03},
    "n7": {"eyeWideLeft": 0.16, "eyeWideRight": 0.16},
    "n8": {"mouthShrugUpper": 0.05},
    "n9": {"mouthShrugLower": 0.05},
    "n10": {"jawLeft": 0.04}
}

class ExpressionManager:
    def __init__(self, change_probability=0.25):
        self.current = random.choice(EXPRESSION_POOL)
        self.change_probability = change_probability
    
    def next(self):
        if random.random() < self.change_probability:
            new_expression = random.choice([e for e in EXPRESSION_POOL if e != self.current])
            self.current = new_expression
        return self.current

expression_manager = ExpressionManager(change_probability=0.25)
# =============================================================================
#  FIX #3: PREVENT MOUTH OVER-ANIMATION DURING VISEMES
# =============================================================================
# Add this NEW constant after EMOTION_MODIFIERS (around line 900)
MOUTH_LOCK_DURING_SPEECH = True  #  NEW: Lock mouth shapes during active speech

def apply_emotion_with_intensity_LOCKED(
    base_shapes: Dict[str, float],
    emotion: str,
    intensity: float,
    facial_state: str = "neutral",
    is_speaking: bool = False
) -> Dict[str, float]:
    """
     ENHANCED: Apply emotion BUT lock mouth shapes during speech
    """
    result = base_shapes.copy()
    
    # Get emotion modifiers
    emotion_mods = EMOTION_MODIFIERS.get(emotion, EMOTION_MODIFIERS["neutral"])
    
    #  NEW: Log eyebrow shapes being applied
    eyebrow_shapes = {k: v for k, v in emotion_mods.items() if 'brow' in k.lower()}
    if eyebrow_shapes:
        logger.info(f"👁️ Applying eyebrow shapes for {emotion}: {eyebrow_shapes}")
    
    
    #  NEW: Define mouth-related blendshapes that should be LOCKED during speech
    MOUTH_SHAPES_TO_LOCK = {
        "mouthSmileLeft", "mouthSmileRight",
        "mouthFrownLeft", "mouthFrownRight",
        "mouthPucker", "mouthFunnel",
        "mouthStretchLeft", "mouthStretchRight",
        "mouthUpperUpLeft", "mouthUpperUpRight",
        "mouthLowerDownLeft", "mouthLowerDownRight",
        "mouthPressLeft", "mouthPressRight",
        "jawOpen"  #  CRITICAL: Lock jaw during speech
    }
    
    # Scale by intensity
    for shape, base_value in emotion_mods.items():
        scaled_value = base_value * intensity
        
        #  CRITICAL: Skip mouth shapes if actively speaking
        if is_speaking and MOUTH_LOCK_DURING_SPEECH and shape in MOUTH_SHAPES_TO_LOCK:
            # Reduce emotion impact on mouth to only 10% during speech
            scaled_value *= 0.10
        
        # Apply facial state adjustments (only for non-mouth shapes)
        if shape not in MOUTH_SHAPES_TO_LOCK:
            if facial_state == "tense" and "brow" in shape.lower():
                scaled_value *= 1.2
            elif facial_state == "soft" and "cheek" in shape.lower():
                scaled_value *= 0.8
            elif facial_state == "bright" and "eye" in shape.lower():
                scaled_value *= 1.15
            elif facial_state == "focused" and "brow" in shape.lower():
                scaled_value *= 1.1
            elif facial_state == "relaxed":
                scaled_value *= 0.9
        
        # Add to result
        if shape in result:
            result[shape] = min(1.0, result[shape] + scaled_value)
        else:
            result[shape] = min(1.0, max(0.0, scaled_value))
    
    return result
# =============================================================================
# TONGUE POSITION RESOLVER - Maps visemes to actual tongue blendshapes
# =============================================================================

TONGUE_POSITIONS = {
    # TH sound - tongue between/against teeth
    "th": {
        "Tongue_Out": 0.3,
        "Tongue_Tip_Up": 0.6,
        "Tongue_Up": 0.4,
        "V_Tongue_Out": 0.5
    },
    
    # D, T, N - tongue tip behind upper teeth
    "d": {
        "Tongue_Tip_Up": 0.7,
        "Tongue_Up": 0.5,
        "V_Tongue_up": 0.6
    },
    
    # N - nasal, tongue tip up
    "n": {
        "Tongue_Tip_Up": 0.6,
        "Tongue_Up": 0.45,
        "V_Tongue_up": 0.5
    },
    
    # L - lateral, tongue tip up, sides down
    "l": {
        "Tongue_Tip_Up": 0.7,
        "Tongue_Up": 0.5,
        "Tongue_Wide": 0.3,
        "V_Tongue_up": 0.55
    },
    
    # S, Z - tongue groove for air, tip near ridge
    "s": {
        "Tongue_Tip_Up": 0.5,
        "Tongue_Narrow": 0.6,
        "Tongue_Up": 0.3,
        "V_Tongue_Narrow": 0.5
    },
    
    # SH - tongue further back, rounded
    "sh": {
        "Tongue_Up": 0.4,
        "Tongue_Curl_U": 0.4,
        "V_Tongue_Curl_U": 0.35
    },
    
    # CH, J - affricate, tongue release
    "ch": {
        "Tongue_Tip_Up": 0.5,
        "Tongue_Up": 0.45,
        "V_Tongue_up": 0.4
    },
    
    # K, G - tongue back raised (velar)
    "k": {
        "Tongue_Down": 0.3,
        "Tongue_In": 0.5,
        "V_Tongue_Lower": 0.3
    },
    
    "g": {
        "Tongue_Down": 0.3,
        "Tongue_In": 0.5,
        "V_Tongue_Lower": 0.3
    },
    
    # R - tongue curl/retroflection
    "r": {
        "Tongue_Curl_U": 0.6,
        "Tongue_Mid_Up": 0.5,
        "V_Tongue_Curl_U": 0.6,
        "V_Tongue_Raise": 0.4
    },
    
    # W - tongue back, rounded
    "w": {
        "Tongue_In": 0.4,
        "V_Tongue_Lower": 0.2
    },
    
    # Y - tongue body raised
    "y": {
        "Tongue_Up": 0.5,
        "V_Tongue_up": 0.4
    },
    
    # Vowels
    "aa": {  # Open vowel - tongue low and back
        "Tongue_Down": 0.6,
        "V_Tongue_Lower": 0.5
    },
    
    "e": {  # Front vowel - tongue forward and mid
        "Tongue_Up": 0.3,
        "V_Tongue_up": 0.25
    },
    
    "ih": {  # Front vowel - tongue slightly forward
        "Tongue_Up": 0.2,
        "V_Tongue_up": 0.2
    },
    
    "oh": {  # Back rounded - tongue back
        "Tongue_In": 0.3,
        "V_Tongue_Lower": 0.2
    },
    
    "ow": {  # Diphthong - tongue moves back
        "Tongue_In": 0.3,
        "V_Tongue_Lower": 0.25
    },
    
    "oo": {  # High back rounded - tongue back and up
        "Tongue_In": 0.4,
        "Tongue_Up": 0.2,
        "V_Tongue_Raise": 0.3
    },
    
    # Labials - neutral tongue
    "bmp": {},
    "fv": {},
    
    # Silence - rest position
    "sil": {
        "Tongue_In": 0.1
    }
}
# =============================================================================
#  ENHANCED TONGUE POSITIONS FOR REALISTIC ARTICULATION
# =============================================================================
def get_tongue_shapes_enhanced(viseme_name):
    """
    More realistic tongue positions based on phonetic articulation
    """
    if viseme_name == "th":
        return {
            "Tongue_Out": 0.65,          # Tongue between teeth
            "Tongue_Tip_Up": 0.25,       # Tip raised slightly
            "Tongue_Narrow": 0.35,       # Narrow tongue
            "Tongue_Tip_Down": 0.15,     # Tip visible
        }
    elif viseme_name in ["d", "t"]:
        return {
            "Tongue_Tip_Up": 0.55,       # Tip to alveolar ridge
            "Tongue_Up": 0.35,
            "Tongue_In": 0.20,           # Pulled back slightly
        }
    elif viseme_name in ["s", "z"]:
        return {
            "Tongue_Tip_Up": 0.40,       # Near alveolar ridge
            "Tongue_Narrow": 0.45,       # Groove for airflow
            "Tongue_Up": 0.25,
            "Tongue_Tip_L": 0.10,        # Slight asymmetry for realism
        }
    elif viseme_name in ["sh", "zh"]:
        return {
            "Tongue_Up": 0.50,           # Body raised
            "Tongue_Tip_Up": 0.15,       # Tip lowered
            "Tongue_Wide": 0.30,         # Wider for sh
            "Tongue_Curl_U": 0.25,       # Slight curl
        }
    elif viseme_name in ["ch", "j"]:
        return {
            "Tongue_Up": 0.40,
            "Tongue_Tip_Up": 0.35,
            "Tongue_In": 0.20,
        }
    elif viseme_name in ["k", "g"]:
        return {
            "Tongue_Back": 0.60,         # Back of tongue raised
            "Tongue_Up": 0.40,
            "Tongue_Down": 0.20,         # Front lowered
            "Tongue_In": 0.35,           # Retracted
        }
    elif viseme_name == "n":
        return {
            "Tongue_Tip_Up": 0.60,       # Tip to alveolar
            "Tongue_Up": 0.40,
            "Tongue_In": 0.15,
        }
    elif viseme_name == "l":
        return {
            "Tongue_Tip_Up": 0.70,       # Strong tip up
            "Tongue_Up": 0.45,
            "Tongue_Wide": 0.25,         # Sides down
            "Tongue_Tip_R": 0.10,        # Slight right bias
        }
    elif viseme_name == "r":
        return {
            "Tongue_Curl_U": 0.55,       # Bunched or retroflex
            "Tongue_Mid_Up": 0.40,
            "Tongue_Wide": 0.35,
            "Tongue_In": 0.30,           # Retracted
        }
    elif viseme_name == "y":
        return {
            "Tongue_Up": 0.45,           # Body to palate
            "Tongue_Tip_Up": 0.30,
            "Tongue_Front": 0.25,        # Forward
        }
    elif viseme_name == "aa":
        return {
            "Tongue_Down": 0.40,         # Low and back
            "Tongue_Back": 0.35,
            "Tongue_Wide": 0.20,
        }
    elif viseme_name in ["e", "ih"]:
        return {
            "Tongue_Up": 0.30,           # Front raised
            "Tongue_Front": 0.35,
            "Tongue_Tip_Up": 0.20,
        }
    elif viseme_name in ["oh", "oo", "ow"]:
        return {
            "Tongue_Back": 0.40,         # Back position
            "Tongue_Up": 0.25,
            "Tongue_In": 0.30,           # Retracted
        }
    elif viseme_name == "bmp":
        return {
            "Tongue_In": 0.20,           # Neutral position
            "Tongue_Up": 0.10,
        }
    elif viseme_name == "fv":
        return {
            "Tongue_In": 0.15,           # Back slightly
            "Tongue_Down": 0.10,
        }
    
    # Default for silence
    return {
        "Tongue_In": 0.15,               # Resting position
        "Tongue_Up": 0.05,
    }
# =============================================================================
# ENHANCED MOUTH CLOSURE HANDLING - PREVENT JAW EXTENSION ISSUES
# =============================================================================
def handle_mouth_close_intelligently(mouth_shapes_dict, current_viseme, next_viseme):
    """
    Intelligent mouth closure handling to prevent jaw extension artifacts
    
    Note: 'mouthClose' blendshape extends the jaw unnaturally in some rigs.
    We'll simulate mouth closure using other blendshapes instead.
    """
    
    # Check if we're transitioning to a closed mouth sound
    closed_visemes = ["bmp", "fv", "p", "b", "m"]
    
    if current_viseme in closed_visemes or next_viseme in closed_visemes:
        # Instead of using mouthClose, use mouthPress and mouthShrug
        if "mouthClose" in mouth_shapes_dict:
            close_intensity = mouth_shapes_dict.pop("mouthClose")
            
            # Distribute closure intensity to other shapes
            mouth_shapes_dict["mouthPressLeft"] = mouth_shapes_dict.get("mouthPressLeft", 0.0) + close_intensity * 0.6
            mouth_shapes_dict["mouthPressRight"] = mouth_shapes_dict.get("mouthPressRight", 0.0) + close_intensity * 0.6
            mouth_shapes_dict["mouthShrugLower"] = mouth_shapes_dict.get("mouthShrugLower", 0.0) + close_intensity * 0.4
            mouth_shapes_dict["mouthShrugUpper"] = mouth_shapes_dict.get("mouthShrugUpper", 0.0) + close_intensity * 0.2
        
        # For plosives (b, p), add quick jaw closure
        if current_viseme == "bmp" or next_viseme == "bmp":
            mouth_shapes_dict["jawOpen"] = max(0.0, mouth_shapes_dict.get("jawOpen", 0.0) - 0.15)
    
    return mouth_shapes_dict

# =============================================================================
# ENHANCED TONGUE VISIBILITY HANDLING
# =============================================================================
def adjust_tongue_for_visibility(tongue_shapes_dict, viseme_name, jaw_openness):
    """
    Adjust tongue positions based on jaw openness
    When jaw is more open, tongue should be more visible/pronounced
    """
    adjusted = tongue_shapes_dict.copy()
    
    # Tongue visibility multiplier based on jaw openness
    visibility_multiplier = 1.0 + (jaw_openness * 0.8)
    
    # Specific adjustments for different sounds
    if viseme_name == "th":
        # Tongue between teeth - more visible when jaw is open
        if "Tongue_Out" in adjusted:
            adjusted["Tongue_Out"] *= visibility_multiplier
        if "Tongue_Tip_Down" in adjusted:
            adjusted["Tongue_Tip_Down"] *= visibility_multiplier
    
    elif viseme_name in ["d", "t", "n", "l"]:
        # Tongue tip up sounds - adjust based on jaw
        if "Tongue_Tip_Up" in adjusted:
            adjusted["Tongue_Tip_Up"] *= (1.0 + jaw_openness * 0.3)
    
    elif viseme_name in ["s", "z", "sh", "zh"]:
        # Sibilants - tongue groove needs visibility
        if "Tongue_Narrow" in adjusted:
            adjusted["Tongue_Narrow"] *= visibility_multiplier
    
    # Cap all values
    for key in adjusted:
        adjusted[key] = min(1.0, max(0.0, adjusted[key]))
    
    return adjusted
# =============================================================================
# ENHANCED LIP SYNC TRANSITION HELPERS
# =============================================================================

def get_transition_lip_shape(current_viseme, next_viseme, progress):
    """
    Create smooth transitions between visemes with realistic lip movements
    """
    current_shapes = VISEME_TO_MOUTH_SHAPES.get(current_viseme, {})
    next_shapes = VISEME_TO_MOUTH_SHAPES.get(next_viseme, {})
    
    # Special transition cases
    if current_viseme == "bmp" and next_viseme in ["aa", "e", "oh"]:
        # From closed to open - add anticipatory opening
        transition_shapes = {}
        for shape in set(current_shapes.keys()) | set(next_shapes.keys()):
            if shape == "jawOpen":
                # Early jaw opening before lips fully relax
                transition_shapes[shape] = current_shapes.get(shape, 0) * (1-progress) + \
                                          next_shapes.get(shape, 0) * progress * 1.2
            else:
                transition_shapes[shape] = current_shapes.get(shape, 0) * (1-progress) + \
                                          next_shapes.get(shape, 0) * progress
        return transition_shapes
    
    elif next_viseme == "bmp" and current_viseme in ["aa", "e", "oh"]:
        # From open to closed - early lip closure
        transition_shapes = {}
        for shape in set(current_shapes.keys()) | set(next_shapes.keys()):
            if shape == "mouthClose":
                # Start closing earlier
                transition_shapes[shape] = next_shapes.get(shape, 0) * (progress * 1.3)
            elif shape in ["mouthPressLeft", "mouthPressRight"]:
                # Prepare press earlier
                transition_shapes[shape] = next_shapes.get(shape, 0) * (progress * 1.2)
            else:
                transition_shapes[shape] = current_shapes.get(shape, 0) * (1-progress) + \
                                          next_shapes.get(shape, 0) * progress
        return transition_shapes
    
    # Default linear interpolation
    transition_shapes = {}
    all_shapes = set(current_shapes.keys()) | set(next_shapes.keys())
    
    for shape in all_shapes:
        current_val = current_shapes.get(shape, 0)
        next_val = next_shapes.get(shape, 0)
        
        #  CUBIC BEZIER EASING for ultra-smooth transitions
        # Approximates CSS cubic-bezier(0.4, 0.0, 0.2, 1.0)
        if progress < 0.5:
            # Ease in (cubic)
            eased_progress = 4 * progress * progress * progress
        else:
            # Ease out (cubic)
            p = (progress - 1)
            eased_progress = 1 + 4 * p * p * p
        
        transition_shapes[shape] = current_val * (1 - eased_progress) + \
                                   next_val * eased_progress
    
    return transition_shapes

def get_anticipatory_value(blendshape_name, current_val, target_val):
    """
    Calculate anticipatory values for realistic speech
    """
    # Special anticipatory movements for certain blendshapes
    if blendshape_name == "jawOpen":
        # Jaw opens slightly in anticipation
        return min(target_val, current_val * 1.1 + 0.05)
    
    elif blendshape_name == "mouthClose" and target_val > 0.5:
        # Start closing earlier for plosives
        return min(target_val, current_val + 0.15)
    
    elif blendshape_name in ["mouthPucker", "mouthFunnel"] and target_val > 0.3:
        # Start rounding earlier
        return min(target_val, current_val * 1.2)
    
    # Default: slight movement toward target
    diff = target_val - current_val
    return current_val + diff * 0.3
# =============================================================================
# NECK MOVEMENT SYSTEM - Adds natural head movements during speech
# =============================================================================
NECK_MOVEMENT_PATTERNS = {
    "happy": {
        "movements": [
            {"neckUp": 0.15, "duration": 0.8},
            {"neckDown": 0.10, "duration": 0.6},
            {"neckLeft": 0.12, "neckRight": 0.0, "duration": 0.7},
            {"neckRight": 0.12, "neckLeft": 0.0, "duration": 0.7}
        ],
        "frequency": 1.2  # More frequent movements
    },
    "excited": {
        "movements": [
            {"neckUp": 0.25, "duration": 0.5},
            {"neckDown": 0.15, "duration": 0.4},
            {"neckLeft": 0.20, "duration": 0.5},
            {"neckRight": 0.20, "duration": 0.5}
        ],
        "frequency": 0.8  # Very frequent
    },
    "sad": {
        "movements": [
            {"neckDown": 0.20, "duration": 1.2},
            {"neckUp": 0.08, "duration": 1.0},
            {"neckLeft": 0.08, "duration": 1.1},
            {"neckRight": 0.08, "duration": 1.1}
        ],
        "frequency": 2.0  # Slower movements
    },
    "angry": {
        "movements": [
            {"neckForward": 0.25, "duration": 0.8},
            {"neckLeft": 0.15, "duration": 0.7},
            {"neckRight": 0.15, "duration": 0.7},
            {"neckUp": 0.10, "duration": 0.6}
        ],
        "frequency": 1.0
    },
    "surprised": {
        "movements": [
            {"neckUp": 0.30, "duration": 0.4},
            {"neckDown": 0.20, "duration": 0.5}
        ],
        "frequency": 1.5
    },
    "fear": {
        "movements": [
            {"neckUp": 0.18, "duration": 0.7},
            {"neckLeft": 0.15, "duration": 0.6},
            {"neckRight": 0.15, "duration": 0.6},
            {"neckDown": 0.12, "duration": 0.8}
        ],
        "frequency": 1.0
    },
    "disgust": {
        "movements": [
            {"neckUp": 0.20, "duration": 0.9},
            {"neckLeft": 0.12, "duration": 0.8},
            {"neckRight": 0.12, "duration": 0.8}
        ],
        "frequency": 1.3
    },
    "neutral": {
        "movements": [
            {"neckUp": 0.08, "duration": 1.0},
            {"neckDown": 0.06, "duration": 1.0},
            {"neckLeft": 0.08, "duration": 1.0},
            {"neckRight": 0.08, "duration": 1.0}
        ],
        "frequency": 2.5  # Subtle, infrequent
    }
}
# =============================================================================
# AUDIO ENERGY
# =============================================================================

def load_audio_energy(audio_bytes, sample_rate):
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    energy = np.abs(audio)
    energy = np.convolve(energy, np.ones(400)/400, mode="same")
    return energy

def energy_at_time(energy, sr, t):
    idx = int(t * sr)
    if idx < 0 or idx >= len(energy):
        return 0.0
    return np.clip(energy[idx] * 2.0, 0.0, 1.0)

# =============================================================================
# PHONEME → VISEME ALIGNMENT
# =============================================================================
g2p = G2p()

def align_visemes_with_timestamps(audio_bytes, text, timestamps, sample_rate):
    """
     NEW: Use Scribe v2 word timestamps for precise viseme alignment
    """
    energy = load_audio_energy(audio_bytes, sample_rate)
    duration = len(energy) / sample_rate
    
    segments = []
    
    if not timestamps or len(timestamps) == 0:
        # Fallback to original method
        return align_visemes(audio_bytes, text, sample_rate)
    
    # Use word timestamps from Scribe v2
    for word_data in timestamps:
        word_text = word_data.get('text', '')
        start_time = word_data.get('start', 0)
        end_time = word_data.get('end', start_time + 0.1)
        
        # Get phonemes for this word
        phonemes = g2p(word_text)
        
        # Distribute phonemes across word duration
        word_duration = end_time - start_time
        phoneme_duration = word_duration / max(1, len(phonemes))
        
        for i, p in enumerate(phonemes):
            p_clean = re.sub(r"\d", "", p).upper()
            if p_clean in VISEME_MAP:
                viseme = VISEME_MAP[p_clean]
                
                phoneme_start = start_time + (i * phoneme_duration)
                phoneme_end = phoneme_start + phoneme_duration
                
                segments.append({
                    "name": viseme,
                    "start": phoneme_start,
                    "end": phoneme_end,
                    "expression": expression_manager.next()
                })
    
    return segments, energy
def align_visemes(audio_bytes, text, sample_rate):
    """
     PERFECT: Ultra-precise viseme timing with predictive lookahead
    """
    energy = load_audio_energy(audio_bytes, sample_rate)
    duration = len(energy) / sample_rate
    
    # Get phonemes and convert to visemes
    phonemes = g2p(text)
    visemes = []
    for p in phonemes:
        p_clean = re.sub(r"\d", "", p).upper()
        if p_clean in VISEME_MAP:
            visemes.append(VISEME_MAP[p_clean])
    
    if not visemes:
        visemes = ["sil"]
    
    #  ADAPTIVE TIMING: Adjust based on viseme complexity
    segments = []
    
    if len(visemes) == 1:
        segments.append({
            "name": visemes[0],
            "start": 0.0,
            "end": duration,
            "expression": expression_manager.next()
        })
    else:
        #  Calculate individual viseme durations based on complexity
        viseme_weights = []
        for v in visemes:
            if v in ["bmp", "sil"]:
                weight = 0.7  # Closure sounds are shorter
            elif v in ["th", "fv", "s", "sh", "ch"]:
                weight = 1.3  # Fricatives need more time
            elif v in ["l", "r", "w", "y"]:
                weight = 1.2  # Liquid/glide sounds
            elif v in ["aa", "oh", "ow", "oo"]:
                weight = 1.1  # Open vowels
            else:
                weight = 1.0  # Standard consonants/vowels
            viseme_weights.append(weight)
        
        total_weight = sum(viseme_weights)
        base_duration = duration / total_weight
        
        #  ANTICIPATORY START: Begin mouth movement 40ms early
        anticipation_offset = 0.04
        
        t = 0.0
        for i, v in enumerate(visemes):
            # Calculate this viseme's duration
            viseme_duration = base_duration * viseme_weights[i]
            
            #  ANTICIPATORY ADJUSTMENT: Start preparing for next sound
            if i > 0:
                start = max(0, t - anticipation_offset)
            else:
                start = 0.0
            
            end = t + viseme_duration
            
            #  COARTICULATION: Blend into next viseme
            if i < len(visemes) - 1:
                next_weight = viseme_weights[i + 1]
                # Extend into next viseme based on its complexity
                blend_factor = 0.15 if next_weight > 1.0 else 0.10
                end = min(end + viseme_duration * blend_factor, duration)
            
            # Ensure last viseme reaches END
            if i == len(visemes) - 1:
                end = duration
            
            segments.append({
                "name": v,
                "start": round(start, 4),
                "end": round(end, 4),
                "expression": expression_manager.next()
            })
            
            # Move forward by base duration (no overlap in time counter)
            t += viseme_duration
        
        #  POST-PROCESSING: Ensure no gaps and smooth transitions
        for i in range(len(segments) - 1):
            curr_end = segments[i]['end']
            next_start = segments[i + 1]['start']
            
            # Fill any gaps
            if next_start > curr_end + 0.005:  # Gap > 5ms
                segments[i]['end'] = next_start
            
            # Smooth overlaps
            if next_start < curr_end:
                overlap_mid = (curr_end + next_start) / 2
                segments[i]['end'] = round(overlap_mid, 4)
                segments[i + 1]['start'] = round(overlap_mid, 4)
    
    # Final validation
    last_end = segments[-1]['end']
    if abs(last_end - duration) > 0.01:
        logger.warning(f"Correcting viseme gap: {last_end:.3f}s → {duration:.3f}s")
        segments[-1]['end'] = duration
    
    logger.info(
        f" PERFECT lip sync: {len(visemes)} visemes | "
        f"Duration: {duration:.3f}s | "
        f"Avg per viseme: {duration/len(visemes)*1000:.1f}ms"
    )
    
    return segments, energy
# =============================================================================
#  VISEME MAPPING — UPDATED AGAIN
# JawOpen: unchanged (0–1)
# All other shapes: previous values reduced by ~70% (× 0.3)
# =============================================================================
VISEME_TO_MOUTH_SHAPES = {
"aa": {   # A as in "father"
    
    "mouthLowerDownLeft": 0.08,
    "mouthLowerDownRight": 0.08,
},

"e": {   # "ay"
    
    "mouthSmileLeft": 0.22,
    "mouthSmileRight": 0.22,
    "mouthStretchLeft": 0.05,
    "mouthStretchRight": 0.05,
},

"ih": {  # "ih"
    
    "mouthStretchLeft": 0.05,
    "mouthStretchRight": 0.05,
},

"oh": {  
    
    "mouthPucker": 0.00,      # reduced so upper teeth don’t drift
    "mouthFunnel": 0.00,
    "cheekTalk": 0.25,
},

"ow": {  
    
    "mouthPucker": 0.00,
    "mouthFunnel": 0.00,
    "cheekTalk": 0.22,
},

"oo": {  
    
    "mouthPucker": 0.00,
    "mouthFunnel": 0.00,
    "cheekTalk": 0.30,
},

# ==================== CONSONANTS ====================

"bmp": {  # B, P, M — clean seal, stable teeth
    
    "mouthPressLeft": 0.65,
    "mouthPressRight": 0.65,
    "mouthShrugLower": 0.22,
    "mouthRollLower": 0.14,
},

"fv": {  
    
    "mouthLowerDownLeft": 0.48,
    "mouthLowerDownRight": 0.48,
},

"th": {  
    
    "V_Tongue_Out": 0.40,
},

"d": {  
    
    "Tongue_Up": 0.35,
},

"s": {  
    
    "Tongue_Narrow": 0.40,
},

"sh": {  
    
    "mouthPucker": 0.18,
    "mouthFunnel": 0.15,
    "V_Tongue_Curl_U": 0.35,
},

"ch": {  
    
    "Tongue_Up": 0.40,
},

"k": {  
    
    "Tongue_In": 0.35,
},

"g": {  
    "Tongue_In": 0.35,
},

"n": {  
    "Tongue_Up": 0.38,
},

"l": {  
    
    "V_Tongue_up": 0.42,
},

"r": {  
    
    "mouthPucker": 0.22,
    "V_Tongue_Curl_U": 0.50,
},

"w": {  
    "mouthPucker": 0.15,
    "mouthFunnel": 0.00,
    "cheekTalk": 0.18, 
},

"y": {  
    "mouthSmileLeft": 0.18,
    "mouthSmileRight": 0.18,
    "Tongue_Up": 0.30,
},

"sil": {  
}
}
# =============================================================================
#  FIXED GENDER MULTIPLIERS - 1.5x scaled
# =============================================================================

GENDER_BLENDSHAPE_MULTIPLIERS = {
    "girl": {
        # Lips / mouth (3× stronger)
        "mouthSmileLeft": 4.95,
        "mouthSmileRight": 4.95,
        "mouthFunnel": 2.025,
        "mouthPucker": 2.925,
        "mouthStretchLeft": 2.25,
        "mouthStretchRight": 2.25,
        "mouthDimpleLeft": 5.4,
        "mouthDimpleRight": 5.4,
        "mouthUpperUpLeft": 2.25,
        "mouthUpperUpRight": 2.25,
        "mouthLowerDownLeft": 2.25,
        "mouthLowerDownRight": 2.25,
        "mouthPressLeft": 1.8,
        "mouthPressRight": 1.8,
        "mouthFrownLeft": 1.8,
        "mouthFrownRight": 1.8,

        # Eyes / brows (3× stronger)
        "eyeWideLeft": 1.125,
        "eyeWideRight": 1.125,
        "eyeSquintLeft": 1.35,
        "eyeSquintRight": 1.35,
        "browInnerUp": 4.05,
        "browOuterUpLeft": 4.5,
        "browOuterUpRight": 4.5,
        "browDownLeft": 4.275,
        "browDownRight": 4.275,

        # Neck (3× stronger)
        "neckUp": 2.25,
        "neckDown": 2.25,
        "neckLeft": 2.7,
        "neckRight": 2.7,

        # Shoulder breathing (3× stronger)
        "shoulderUp": 1.8,
        "shoulderDown": 1.8,
    },

    "boy": {
        "mouthStretchLeft": 4.725,
        "mouthStretchRight": 4.725,
        "mouthLowerDownLeft": 4.05,
        "mouthLowerDownRight": 4.05,
    }
}

def apply_gender_multiplier(blendshape_name: str, base_value: float, audio_type: str) -> float:
    """
     NOW ONLY USED IN add_keyframe FOR FINAL CLAMPING
    Main application happens inline in generate_time_based_blendshapes
    """
    if audio_type not in GENDER_BLENDSHAPE_MULTIPLIERS:
        return base_value
    
    multipliers = GENDER_BLENDSHAPE_MULTIPLIERS[audio_type]
    multiplier = multipliers.get(blendshape_name, 1.0)
    adjusted_value = base_value * multiplier
    return adjusted_value

# =============================================================================
#  FIX #4: KEYFRAME OPTIMIZATION - Less Aggressive Filtering
# =============================================================================

def optimize_for_60fps_aggressive(animation_tracks, total_duration):
    """
     FIXED: Balanced optimization - preserves lip sync quality
    """
    TARGET_FPS = 50
    frame_duration = 1.0 / TARGET_FPS
    MAX_KF_PER_FRAME = 1.5  #  INCREASED from 1.2 (less aggressive)
    
    optimized = {}
    
    for name, keyframes in animation_tracks.items():
        if not keyframes:
            continue
        
        keyframes.sort(key=lambda k: k["start"])
        
        # Remove duplicates within 15ms (was 20ms)
        unique = []
        for kf in keyframes:
            if not unique or abs(kf["start"] - unique[-1]["start"]) > 0.015:
                unique.append(kf)
        
        #  Essential lip sync shapes - NEVER over-filter
        is_critical = name in [
            "jawOpen", "mouthSmileLeft", "mouthSmileRight", 
            "mouthPucker", "mouthFunnel", "mouthStretchLeft", "mouthStretchRight",
            "mouthUpperUpLeft", "mouthUpperUpRight"
        ]
        
        if len(unique) <= 3:
            optimized[name] = unique
            continue
        
        # Calculate limit
        total_frames = int(total_duration * TARGET_FPS)
        max_kf_for_track = int(total_frames * MAX_KF_PER_FRAME / max(1, len(animation_tracks)))
        max_kf_for_track = max(4, max_kf_for_track)  # Minimum 4 keyframes
        
        if is_critical:
            max_kf_for_track = int(max_kf_for_track * 1.5)  #  50% more for critical shapes
        
        if len(unique) <= max_kf_for_track:
            optimized[name] = unique
            continue
        
        #  LESS AGGRESSIVE FILTERING
        filtered = [unique[0]]
        
        for i in range(1, len(unique) - 1):
            if len(filtered) >= max_kf_for_track:
                break
            
            kf = unique[i]
            time_since_last = kf['start'] - filtered[-1]['start']
            intensity_change = abs(kf['intensity'] - filtered[-1]['intensity'])
            
            if is_critical:
                #  RELAXED: 30ms AND 8% change (was 40ms/12%)
                keep = (time_since_last >= 0.03 and intensity_change > 0.08)
            else:
                #  RELAXED: 50ms AND 20% change (was 60ms/25%)
                keep = (time_since_last >= 0.05 and intensity_change > 0.20)
            
            if keep:
                frame_idx = round(kf['start'] / frame_duration)
                kf['start'] = frame_idx * frame_duration
                filtered.append(kf)
        
        # Keep last
        if unique[-1] not in filtered:
            filtered.append(unique[-1])
        
        # Ensure last frame at duration
        if filtered and abs(filtered[-1]['start'] - total_duration) > 0.01:
            filtered[-1]['start'] = round(total_duration, 4)
        
        optimized[name] = filtered
    
    # Stats
    total_before = sum(len(tracks) for tracks in animation_tracks.values())
    total_after = sum(len(tracks) for tracks in optimized.values())
    reduction = ((total_before - total_after) / total_before * 100) if total_before > 0 else 0
    
    total_frames = int(total_duration * TARGET_FPS)
    avg_kf_per_frame = total_after / total_frames if total_frames > 0 else 0
    
    print(f" Optimized: {total_before} → {total_after} keyframes ({reduction:.1f}% reduction)")
    print(f" Density: {avg_kf_per_frame:.2f} kf/frame | Frames: {total_frames}")
    
    return optimized
# =============================================================================
#  FIX #5: ENSURE ALL BLENDSHAPES START AT 0 (Prevents Frozen Face)
# =============================================================================

def ensure_animation_initialization(animation_tracks, total_duration):
    """
     CRITICAL FIX: Ensure ALL tracks have initial and final keyframes
    Prevents "frozen face" bug where some blendshapes never activate
    """
    ALL_MOUTH_SHAPES = [
        "jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthPucker",
        "mouthFunnel", "mouthStretchLeft", "mouthStretchRight",
        "mouthUpperUpLeft", "mouthUpperUpRight", "mouthLowerDownLeft",
        "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight"
    ]
    for shape in ALL_MOUTH_SHAPES:
        if shape not in animation_tracks:
            animation_tracks[shape] = []
        
        track = animation_tracks[shape]
        
        # Ensure starts at 0
        if not track or track[0]['start'] > 0.01:
            track.insert(0, {
                "name": shape,
                "start": 0.0,
                "intensity": 0.0
            })
        
        # Ensure ends at duration
        if not track or abs(track[-1]['start'] - total_duration) > 0.01:
            track.append({
                "name": shape,
                "start": round(total_duration, 4),
                "intensity": 0.0
            })
    
    return animation_tracks

def smooth_transitions(animation_tracks):
    """
    ULTRA-SMOOTH: Bezier-curve-like smoothing with intelligent filtering
    """
    smoothed = {}
    
    for name, keyframes in animation_tracks.items():
        if len(keyframes) < 3:
            smoothed[name] = keyframes
            continue
        
        smoothed_kf = [keyframes[0]]
        
        for i in range(1, len(keyframes) - 1):
            prev = keyframes[i-1]
            curr = keyframes[i]
            next_kf = keyframes[i+1]
            
            time_diff = curr['start'] - prev['start']
            intensity_diff = abs(curr['intensity'] - prev['intensity'])
            
            #  ANTI-JITTER: Remove rapid micro-movements
            if time_diff < 0.04 and intensity_diff < 0.15:  # <40ms & <15% change
                continue
            
            #  BEZIER-STYLE SMOOTHING: Weighted average with lookahead
            if intensity_diff < 0.12:  # Small changes get smoothed more
                # Weight: 20% prev, 50% curr, 30% next
                smoothed_intensity = (
                    prev['intensity'] * 0.20 +
                    curr['intensity'] * 0.50 +
                    next_kf['intensity'] * 0.30
                )
            elif intensity_diff < 0.25:  # Medium changes
                # Weight: 15% prev, 60% curr, 25% next
                smoothed_intensity = (
                    prev['intensity'] * 0.15 +
                    curr['intensity'] * 0.60 +
                    next_kf['intensity'] * 0.25
                )
            else:  # Large changes - keep sharp
                smoothed_intensity = curr['intensity']
            
            smoothed_curr = curr.copy()
            smoothed_curr['intensity'] = round(smoothed_intensity, 4)
            smoothed_kf.append(smoothed_curr)
        
        # Always keep last keyframe
        smoothed_kf.append(keyframes[-1])
        
        #  SECONDARY PASS: Ensure minimum spacing between keyframes
        final_kf = [smoothed_kf[0]]
        min_spacing = 0.025  # 25ms minimum between keyframes
        
        for kf in smoothed_kf[1:]:
            if kf['start'] - final_kf[-1]['start'] >= min_spacing:
                final_kf.append(kf)
            else:
                # Merge with previous keyframe (average intensity)
                final_kf[-1]['intensity'] = round(
                    (final_kf[-1]['intensity'] + kf['intensity']) / 2, 4
                )
        
        smoothed[name] = final_kf
    
    return smoothed
# =============================================================================
# CHEEK CONTROL FUNCTION - PREVENT BUMPY FACE
# =============================================================================
def control_cheek_movements(animation_tracks, emotion):
    """
    Control cheek movements to prevent bumpy face appearance
    """
    # Emotions that should have minimal cheek movement
    minimal_cheek_emotions = ["sad", "concerned", "neutral", "curious", "professional_friendly"]
    
    if emotion in minimal_cheek_emotions:
        # Reduce cheek movements for these emotions
        cheek_shapes = ["cheekSquintLeft", "cheekSquintRight", "cheekPuff", "cheekPuffOut", "cheekPuffIn"]
        
        for cheek_shape in cheek_shapes:
            if cheek_shape in animation_tracks:
                # Reduce intensity by 50-70%
                for kf in animation_tracks[cheek_shape]:
                    kf["intensity"] = round(kf["intensity"] * 0.4, 4)
    
    # Always cap cheek movements
    max_cheek_intensity = 0.25
    for shape in ["cheekSquintLeft", "cheekSquintRight", "cheekPuff"]:
        if shape in animation_tracks:
            for kf in animation_tracks[shape]:
                kf["intensity"] = min(max_cheek_intensity, kf["intensity"])
    
    return animation_tracks
# =============================================================================
#  ENHANCED: Dynamic Emotion Modifiers with Intensity & Facial State
# =============================================================================

def get_dynamic_emotion_modifiers(groq_emotion: str, intensity: float = 1.0, facial_state: str = "neutral"):
    """
    Generate emotion modifiers dynamically based on:
    - groq_emotion: The detected/LLM emotion
    - intensity: Strength of emotion (0.0 - 1.0+)
    - facial_state: Additional facial quality ("tense", "soft", "bright", "heavy", "neutral")
    
    Returns: Dict of blendshape modifiers
    """
    
    # Get base modifiers from existing EMOTION_MODIFIERS
    base_modifiers = EMOTION_MODIFIERS.get(groq_emotion, EMOTION_MODIFIERS.get("neutral", {})).copy()
    
    # Scale by intensity
    scaled_modifiers = {
        k: v * intensity for k, v in base_modifiers.items()
    }
    
    # Apply facial state tweaks
    if facial_state == "tense":
        scaled_modifiers["jawOpen"] = scaled_modifiers.get("jawOpen", 0) + 0.05
        scaled_modifiers["mouthPressLeft"] = scaled_modifiers.get("mouthPressLeft", 0) + 0.08
        scaled_modifiers["mouthPressRight"] = scaled_modifiers.get("mouthPressRight", 0) + 0.08
        scaled_modifiers["browDownLeft"] = scaled_modifiers.get("browDownLeft", 0) + 0.06
        scaled_modifiers["browDownRight"] = scaled_modifiers.get("browDownRight", 0) + 0.06
        
    elif facial_state == "soft":
        scaled_modifiers["cheekSquintLeft"] = scaled_modifiers.get("cheekSquintLeft", 0) * 0.8
        scaled_modifiers["cheekSquintRight"] = scaled_modifiers.get("cheekSquintRight", 0) * 0.8
        scaled_modifiers["mouthSmileLeft"] = scaled_modifiers.get("mouthSmileLeft", 0) * 1.1
        scaled_modifiers["mouthSmileRight"] = scaled_modifiers.get("mouthSmileRight", 0) * 1.1
        
    elif facial_state == "bright":
        scaled_modifiers["eyeWideLeft"] = scaled_modifiers.get("eyeWideLeft", 0) + 0.05
        scaled_modifiers["eyeWideRight"] = scaled_modifiers.get("eyeWideRight", 0) + 0.05
        scaled_modifiers["browOuterUpLeft"] = scaled_modifiers.get("browOuterUpLeft", 0) + 0.04
        scaled_modifiers["browOuterUpRight"] = scaled_modifiers.get("browOuterUpRight", 0) + 0.04
        
    elif facial_state == "heavy":
        scaled_modifiers["browInnerUp"] = scaled_modifiers.get("browInnerUp", 0) + 0.05
        scaled_modifiers["browDownLeft"] = scaled_modifiers.get("browDownLeft", 0) + 0.03
        scaled_modifiers["browDownRight"] = scaled_modifiers.get("browDownRight", 0) + 0.03
        scaled_modifiers["eyeSquintLeft"] = scaled_modifiers.get("eyeSquintLeft", 0) + 0.04
        scaled_modifiers["eyeSquintRight"] = scaled_modifiers.get("eyeSquintRight", 0) + 0.04
    
    # Clamp all values to valid range [0, 1]
    for key in scaled_modifiers:
        scaled_modifiers[key] = max(0.0, min(1.0, scaled_modifiers[key]))
    
    return scaled_modifiers

# =============================================================================
#  APPLY ALL FIXES TO MAIN FUNCTION
# =============================================================================

def generate_blendshapes_realtime_FIXED(
    segments, 
    energy, 
    sample_rate, 
    emotion="neutral", 
    audio_type="boy", 
    actual_audio_duration=None,
    intensity=1.0,
    facial_state="neutral"
):
    """
     COMPLETE FIXED VERSION with all patches applied
    """
    start_time = time.time()
    
    if not segments:
        return {}
    
    if actual_audio_duration is not None:
        total_duration = actual_audio_duration
    else:
        total_duration = segments[-1]["end"]
    
    # Ensure last segment matches duration
    if segments[-1]["end"] != total_duration:
        segments[-1]["end"] = total_duration
    
    # Track selection (keep existing code)
    ESSENTIAL_MOUTH_SHAPES = {
        "jawOpen", "mouthSmileLeft", "mouthSmileRight",
        "mouthPucker", "mouthFunnel", "mouthStretchLeft", "mouthStretchRight",
    }
    
    # =============================================================================
    #  ESSENTIAL EMOTION SHAPES - COMPLETE VERSION (All facial features)
    # =============================================================================
    ESSENTIAL_EMOTION_SHAPES = {
        "happy": {
            # Eyebrows
            "browOuterUpLeft", "browOuterUpRight", "browInnerUp",
            # Eyes
            "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
            # Cheeks
            "cheekSquintLeft", "cheekSquintRight", "cheekPuff"
        },
        
        "cheerful": {
            # Eyebrows
            "browOuterUpLeft", "browOuterUpRight", "browInnerUp",
            # Eyes
            "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
            # Cheeks
            "cheekSquintLeft", "cheekSquintRight", "cheekPuff"
        },
        
        "encouraging": {
            # Eyebrows (empathetic expression)
            "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
            # Eyes
            "eyeWideLeft", "eyeWideRight", "eyeSquintLeft", "eyeSquintRight",
            # Cheeks
            "cheekSquintLeft", "cheekSquintRight"
        },
        
        "excited": {
            # Eyebrows (raised high)
            "browOuterUpLeft", "browOuterUpRight", "browInnerUp",
            # Eyes (wide open)
            "eyeWideLeft", "eyeWideRight", "eyeSquintLeft", "eyeSquintRight",
            # Cheeks
            "cheekSquintLeft", "cheekSquintRight", "cheekPuff"
        },
        
        "sad": {
            # Eyebrows (sad expression)
            "browInnerUp", "browDownLeft", "browDownRight",
            # Eyes
            "eyeSquintLeft", "eyeSquintRight",
            # Mouth
            "mouthFrownLeft", "mouthFrownRight",
            "mouthLowerDownLeft", "mouthLowerDownRight",
            # Cheeks
            "cheekSquintLeft", "cheekSquintRight"
        },
        
        "concerned": {
            # Eyebrows (worried)
            "browInnerUp", "browDownLeft", "browDownRight",
            # Eyes
            "eyeSquintLeft", "eyeSquintRight",
            # Mouth
            "mouthFrownLeft", "mouthFrownRight"
        },
        
        "curious": {
            # Eyebrows (questioning)
            "browOuterUpLeft", "browOuterUpRight", "browInnerUp",
            # Eyes
            "eyeWideLeft", "eyeWideRight", "eyeSquintLeft", "eyeSquintRight",
            # Cheeks
            "cheekSquintLeft", "cheekSquintRight"
        },
        
        "attentive": {
            # Eyebrows (engaged)
            "browOuterUpLeft", "browOuterUpRight", "browInnerUp",
            # Eyes
            "eyeWideLeft", "eyeWideRight"
        },
        
        "calm": {
            # Eyes (relaxed)
            "eyeSquintLeft", "eyeSquintRight",
            # Eyebrows (slightly relaxed)
            "browOuterUpLeft", "browOuterUpRight"
        },
        
        "neutral": set(),
    }
  
    essential_shapes = set()
    essential_shapes.update(ESSENTIAL_MOUTH_SHAPES)
    essential_shapes.update(ESSENTIAL_EMOTION_SHAPES.get(emotion, set()))
        
    animation_tracks = {}
    

    def add_keyframe(name, time, value):
        if name not in essential_shapes:
            return
        
        if name not in animation_tracks:
            animation_tracks[name] = []
        
        animation_tracks[name].append({
            "name": name,
            "start": round(float(time), 4),
            "intensity": round(float(value), 4)
        })
    
    # Lip sync generation (keep existing code but ensure initialization)
    ESSENTIAL_MOUTH_SHAPES_LIPSYNC = {
        "jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", 
        "mouthStretchRight", "mouthPucker", "mouthFunnel", "mouthPressLeft", 
        "mouthPressRight", "mouthUpperUpLeft", "mouthUpperUpRight",
        "mouthLowerDownLeft", "mouthLowerDownRight",
        "cheekTalk"
   }
    all_shapes = set()
    for seg in segments:
        shapes = VISEME_TO_MOUTH_SHAPES.get(seg["name"], {})
        for shape in shapes.keys():
            if shape in ESSENTIAL_MOUTH_SHAPES_LIPSYNC:
                all_shapes.add(shape)
    
    #  CRITICAL: Initialize ALL shapes at time 0
    for shape in all_shapes:
        add_keyframe(shape, 0.0, 0.0)
    
    # Generate lip sync keyframes (keep existing loop)
    for i, seg in enumerate(segments):
        viseme = seg["name"]
        start = seg["start"]
        end = seg["end"]
        duration = end - start
        
        if duration <= 0:
            continue
        
        target_raw = VISEME_TO_MOUTH_SHAPES.get(viseme, {}).copy()
        target = {shape: value for shape, value in target_raw.items() 
                 if shape in ESSENTIAL_MOUTH_SHAPES_LIPSYNC}
        
        prev_viseme = segments[i-1]["name"] if i > 0 else "sil"
        prev_shapes_raw = VISEME_TO_MOUTH_SHAPES.get(prev_viseme, {})
        prev_shapes = {shape: value for shape, value in prev_shapes_raw.items() 
                      if shape in ESSENTIAL_MOUTH_SHAPES_LIPSYNC}
        
        transition_in = duration * 0.15
        transition_out = duration * 0.15
        hold_start = start + transition_in
        hold_end = end - transition_out
        
        for shape in all_shapes:
            prev_val = prev_shapes.get(shape, 0.0)
            target_val = target.get(shape, 0.0)
            
            if abs(target_val - prev_val) > 0.05:
                add_keyframe(shape, start, prev_val)
                add_keyframe(shape, hold_start, target_val)
        
        for shape, value in target.items():
            if value > 0.05:
                add_keyframe(shape, hold_start, value)
                add_keyframe(shape, hold_end, value)
        
        if i < len(segments) - 1:
            next_viseme = segments[i+1]["name"]
            next_shapes_raw = VISEME_TO_MOUTH_SHAPES.get(next_viseme, {})
            next_shapes = {shape: value for shape, value in next_shapes_raw.items() 
                          if shape in ESSENTIAL_MOUTH_SHAPES_LIPSYNC}
            
            for shape in all_shapes:
                curr_val = target.get(shape, 0.0)
                next_val = next_shapes.get(shape, 0.0)
                
                if abs(next_val - curr_val) > 0.05:
                    add_keyframe(shape, hold_end, curr_val)
                    add_keyframe(shape, end, next_val)
        else:
            for shape, value in target.items():
                if value > 0.05:
                    add_keyframe(shape, hold_end, value)
                    add_keyframe(shape, end, value * 0.3)
                    add_keyframe(shape, total_duration, 0.0)
    
    # Emotion overlay - USE NEW FUNCTION
    base_emotion_shapes = {}
    overlay = apply_emotion_with_intensity_LOCKED(  #  Use the LOCKED version
        base_shapes=base_emotion_shapes,
        emotion=emotion,
        intensity=intensity,
        facial_state=facial_state,
        is_speaking=True 
    )
    eyebrow_shapes = {k: v for k, v in overlay.items() if 'brow' in k.lower()}
    mouth_shapes = {k: v for k, v in overlay.items() if 'mouth' in k.lower()}

    logger.info(
        f"🎭 Emotion overlay | {emotion}@{intensity:.2f}x | "
        f"Eyebrows: {eyebrow_shapes} | Mouth: {mouth_shapes}"
    )
    # Check if they're in essential_shapes
    missing = [k for k in eyebrow_shapes.keys() if k not in essential_shapes]
    if missing:
        logger.warning(f"Eyebrow shapes NOT in essential_shapes: {missing}")

    for shape, peak_value in overlay.items():
        if 'brow' in shape.lower():
            logger.info(f"👁️ Eyebrow shape: {shape} = {peak_value:.4f} (in essential: {shape in essential_shapes})")
        
        if shape in essential_shapes:
            fade_in_time = min(0.8, total_duration * 0.25)
            fade_out_start = max(total_duration - 0.8, total_duration * 0.75)
            
            add_keyframe(shape, 0.0, 0.0)
            add_keyframe(shape, fade_in_time, peak_value)
            add_keyframe(shape, fade_out_start, peak_value)
            add_keyframe(shape, total_duration, 0.0)
    
    animation_tracks = ensure_animation_initialization(animation_tracks, total_duration)
    optimized = optimize_for_60fps_aggressive(animation_tracks, total_duration)
    
    generation_time = (time.time() - start_time) * 1000
    total_kf = sum(len(track) for track in optimized.values())
    total_tracks = len(optimized)
    kf_per_second = total_kf / total_duration if total_duration > 0 else 0
    
    print(f" Generated {total_tracks} tracks, {total_kf} keyframes in {generation_time:.2f}ms")
    print(f" Keyframe density: {kf_per_second:.1f} kf/s | Emotion: {emotion} @ {intensity:.1f}x")
    
    return optimized
# # =============================================================================
# #  PATCH 2: ADD MATRIX GENERATION WRAPPER
# # Location: After generate_blendshapes_realtime_FIXED function (~line 1800)
# # =============================================================================

EMOTION_JAW_MODIFIERS = {
    "happy": 0.90,
    "cheerful": 0.92,              # Slightly more open (cheerful expression)
    "encouraging": 0.88,            # Moderate opening
    "friendly": 0.89,
    "happy_friendly": 0.91,
    "professional_friendly": 0.86,  # More controlled
    "excited": 0.95,
    "sad": 0.60,
    "angry": 0.95,
    "surprised": 1.05,
    "fear": 0.90,
    "disgust": 0.70,
    "neutral": 0.85,
    "concerned": 0.70,
    "curious": 0.85
}
# =============================================================================
# VERIFICATION CODE - Add after generate_time_based_blendshapes
# =============================================================================

def verify_gender_scaling():
    """Test that gender multipliers work correctly"""
    # Test base mouth shape
    test_shapes = {"jawOpen": 0.05, "mouthSmileLeft": 0.25}
    
    # Apply girl multiplier
    girl_shapes = {}
    for name, value in test_shapes.items():
        multipliers = GENDER_BLENDSHAPE_MULTIPLIERS.get("girl", {})
        multiplier = multipliers.get(name, 1.0)
        girl_shapes[name] = value * multiplier
    
    print(f"🧪 Verification:")
    print(f"   Boy jawOpen: {test_shapes['jawOpen']:.2f}")
    print(f"   Girl jawOpen: {girl_shapes['jawOpen']:.2f} (1.8x = {0.05 * 1.8:.2f})")
    print(f"   Boy mouthSmile: {test_shapes['mouthSmileLeft']:.2f}")
    print(f"   Girl mouthSmile: {girl_shapes['mouthSmileLeft']:.2f} (1.4x = {0.25 * 1.4:.2f})")
# =============================================================================
# DIAGNOSTIC: Compare jaw behavior
# =============================================================================
def calculate_audio_duration(audio_bytes, sample_rate):
    """Calculate audio duration"""
    try:
        with io.BytesIO(audio_bytes) as f:
            with wave.open(f, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / rate
                return duration
    except Exception:
        try:
            num_samples = len(audio_bytes) // 2
            duration = num_samples / sample_rate
            return max(0.1, duration)
        except Exception:
            return 0.1

def validate_and_fix_durations(animation_tracks, audio_duration):
    """
    Ensure all animation tracks match audio duration
    """
    for track_name, keyframes in animation_tracks.items():
        if keyframes:
            # Check last keyframe
            last_kf = keyframes[-1]
            if abs(last_kf['start'] - audio_duration) > 0.01:
                # Add proper fadeout
                if len(keyframes) > 1:
                    # Create smooth fadeout over last 0.2 seconds
                    fade_start = max(0, audio_duration - 0.2)
                    fade_intensity = keyframes[-2]['intensity'] if len(keyframes) > 1 else last_kf['intensity']
                    
                    # Remove any keyframes after fade_start
                    keyframes = [kf for kf in keyframes if kf['start'] <= fade_start]
                    
                    # Add fadeout keyframes
                    keyframes.append({
                        'name': track_name,
                        'start': fade_start,
                        'intensity': fade_intensity
                    })
                    keyframes.append({
                        'name': track_name,
                        'start': audio_duration,
                        'intensity': 0.0
                    })
                    
                    animation_tracks[track_name] = keyframes
    
    return animation_tracks

# =============================================================================
#  REDUCED / SMOOTHER NATURAL PAUSE CALCULATION
# =============================================================================

import random
import re

def calculate_natural_pause_optimized(text, current_idx, emotion=None,
                                      speech_rate=1.0, prev_pause=0.0):
    # -----------------------
    # Base micro pause (breathing)
    # -----------------------
    base_pause = 0.035

    # -----------------------
    # Emotion influence
    # -----------------------
    emotion_modifiers = {
        "excited": -0.025,
        "cheerful": -0.02,
        "happy": -0.01,
        "friendly": -0.008,
        "neutral": 0.0,
        "professional_friendly": 0.01,
        "curious": 0.012,
        "sad": 0.045,
        "concerned": 0.03,
        "angry": -0.02,
        "surprised": 0.015,
    }

    base_pause += emotion_modifiers.get(emotion, 0.0)

    lowered = text.lower().strip()
    # -----------------------
    # 🇺🇸 American Discourse Markers (Expanded)
    # -----------------------

    conversation_starters = [
        "well", "so", "okay", "alright", "look", "listen"
    ]

    thinking_markers = [
        "actually", "basically", "honestly", "seriously",
        "literally", "frankly", "clearly", "probably"
    ]

    light_fillers = [
        "i mean", "you know", "kind of", "sort of"
    ]
    discourse_markers = (
        conversation_starters +
        thinking_markers +
        light_fillers
    )

    if any(lowered.startswith(w) for w in discourse_markers):
        base_pause += 0.055

    # -----------------------
    #  Thought transition words
    # -----------------------

    transition_words = [
        "but", "however", "because", "although",
        "therefore", "instead", "meanwhile",
        "otherwise", "so", "yet"
    ]

    if any(f" {w} " in lowered for w in transition_words):
        base_pause += 0.045

    # -----------------------
    #  Intro phrase breathing
    # -----------------------

    intro_phrases = [
        "in my opinion",
        "for example",
        "to be honest",
        "in fact",
        "the truth is",
        "as a result",
        "from my perspective",
        "to be clear"
    ]

    if any(lowered.startswith(p) for p in intro_phrases):
        base_pause += 0.07

    # -----------------------
    # Smart punctuation timing
    # -----------------------

    punctuation_modifiers = {
        '.': 0.13,
        '!': 0.10,
        '?': 0.15,
        ',': 0.02,
        ';': 0.055,
        ':': 0.045,
        '—': 0.065,
        '...': 0.15
    }
    stripped = text.strip()

    if stripped.endswith("..."):
        base_pause += punctuation_modifiers["..."]
    elif stripped:
        base_pause += punctuation_modifiers.get(stripped[-1], 0.0)

    # -----------------------
    # Emphasis detection
    # -----------------------
    # Numbers sound more natural with slight pause
    if re.search(r"\d", text):
        base_pause += 0.035

    # Proper nouns emphasis
    if text.istitle():
        base_pause += 0.03
    # -----------------------
    # Length-aware phrasing
    # -----------------------
    word_count = len(text.split())

    if word_count >= 18:
        base_pause += 0.03
    elif word_count >= 12:
        base_pause += 0.02
    elif word_count <= 2:
        base_pause -= 0.01

    # -----------------------
    # Question lead pacing
    # -----------------------
    question_starters = ("who", "what", "when", "where", "why", "how")

    if lowered.startswith(question_starters):
        base_pause += 0.03

    # -----------------------
    # Speech rate sync
    # -----------------------
    base_pause /= max(0.65, speech_rate)
    # -----------------------
    # Prevent pause stacking
    # -----------------------

    if prev_pause > 0.22:
        base_pause *= 0.75
    # -----------------------
    # Human micro variation
    # -----------------------
    variation = random.uniform(-0.01, 0.02)
    # XTTS-safe clamp
    final_pause = min(0.35, max(0.04, base_pause + variation))
    return final_pause

def parse_emotion_from_llm_response(text: str) -> Tuple[str, str, float, str]:
    """
    Parse emotion metadata from LLM response and remove it from text.
    FIXED: Remove friendly/interesting tags - only keep core emotions
    """
    # Pattern for complete tags - ONLY EMOTION, no friendly/interesting
    complete_pattern = r'\[TENSION\s+name=(\w+)\s+intensity=([\d.]+)\s+facial_state=(\w+)\]\s*'
    match = re.search(complete_pattern, text)
    
    if match:
        emotion_name = match.group(1)
        intensity = float(match.group(2))
        facial_state = match.group(3)
        
        #  FIXED: Remove ALL emotion tags (including incomplete ones)
        clean_text = re.sub(r'\[TENSION\s+[^\]]*(?:\]|$)', '', text).strip()
        
        #  NEW: Also remove any "friendly" or "interesting" tags
        clean_text = re.sub(r'\[(?:friendly|interesting)\s*\]', '', clean_text, flags=re.IGNORECASE).strip()
        
        if not clean_text:
            return "", emotion_name, intensity, facial_state
        
        return clean_text, emotion_name, intensity, facial_state
    
    # Handle incomplete tags
    incomplete_pattern = r'\[TENSION\s+name=(\w+)\s+intensity=([\d.]*).*?(?:\]|$)'
    incomplete_match = re.search(incomplete_pattern, text)
    
    if incomplete_match:
        emotion_name = incomplete_match.group(1)
        intensity_str = incomplete_match.group(2)
        
        try:
            intensity = float(intensity_str) if intensity_str else 0.5
        except ValueError:
            intensity = 0.5
        
        clean_text = re.sub(r'\[TENSION\s+[^\]]*(?:\]|$)', '', text).strip()
        clean_text = re.sub(r'\[(?:friendly|interesting)\s*\]', '', clean_text, flags=re.IGNORECASE).strip()
        
        logger.warning(f"Incomplete emotion tag detected: {incomplete_match.group(0)}")
        
        return clean_text, emotion_name, intensity, "neutral"
    
    #  NEW: Clean any remaining friendly/interesting tags even without emotion tags
    clean_text = re.sub(r'\[(?:friendly|interesting)\s*\]', '', text, flags=re.IGNORECASE).strip()
    
    return clean_text, "neutral", 1.0, "neutral"

async def process_client_message_fixed(client_id: str, message_data: dict):
    session = client_sessions[client_id]
    user_input = message_data.get('text', '')
    audio_type = message_data.get('audio_type', 'boy') 
    
    if not user_input or user_input.lower() == 'exit':
        return
    
    #  Generate unique interaction ID
    interaction_id = f"int_{client_id}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
    
    # Interrupt previous interaction if exists
    previous_interaction = session.get('current_interaction_id')
    if previous_interaction and previous_interaction != interaction_id:
        logger.info(f"🛑 Interrupting previous interaction: {previous_interaction}")
        await interruption_manager.signal_interruption(client_id)
        await asyncio.sleep(0.05)  # Small delay to ensure interruption is processed
    
    #  Clear any previous interruption state for this client
    interruption_manager.clear_interruption(client_id)
    
    #  Start new interaction
    await interruption_manager.start_interaction(client_id, interaction_id)
    
    #  Update session state
    session['current_interaction_id'] = interaction_id
    session['is_processing'] = True
    
    try:
        #  FIX: ALWAYS detect emotion fresh (ignore previous emotion tag)
        # Strip any existing emotion tags first
        clean_input = re.sub(r'^\[([^\]]+)\]\s*', '', user_input)
        
        # Now detect emotion from the ACTUAL content
        initial_emotion, clean_input = detect_emotion_from_context_enhanced(
            clean_input,  # Use cleaned input without old tags
            llm_emotion=None  # Don't pass previous LLM emotion
        )
        
        logger.info(
            f"[CLIENT {client_id}] [{audio_type.upper()}] "
            f"Emotion: {initial_emotion} | Input: '{clean_input[:50]}...'"
        )
        
        #  Process and respond (with interruption handling inside)
        await process_and_respond(
            client_id=client_id,
            user_input=clean_input,
            emotion=initial_emotion,  # Use freshly detected emotion
            interaction_id=interaction_id,
            conversation_history=session['conversation_history'],
            audio_type=audio_type
        )
        
    except asyncio.CancelledError:
        logger.info(f"🛑 Processing cancelled for {client_id}")
    except Exception as e:
        logger.error(f" Processing error for {client_id}: {e}")
        traceback.print_exc()
    finally:
        #  Clear processing state
        session['is_processing'] = False
        
        # Only clear if this is still the current interaction
        if session.get('current_interaction_id') == interaction_id:
            session['current_interaction_id'] = None# =============================================================================
# UPDATE process_and_respond TO USE NEW FALLBACK SYSTEM
# =============================================================================
async def process_and_respond(
    client_id: str, 
    user_input: str, 
    emotion: str,
    interaction_id: str, 
    conversation_history: list,
    audio_type: str = "boy",
    intensity: float = 1.0,
    facial_state: str = "neutral"
):
    """
    Updated with SAM → Groq fallback system + proper interruption handling
    """
    try:
        if client_id not in client_llm_status:
            client_llm_status[client_id] = LLMStatus()
        
        llm_status = client_llm_status[client_id]

        client_sessions[client_id]['current_interaction_id'] = interaction_id

        #  Check if interrupted before starting
        if await interruption_manager.check_interrupted(client_id, interaction_id):
            logger.info(f"🛑 process_and_respond cancelled - interrupted before start")
            return

        response_text = ""
        final_emotion = emotion
        final_intensity = intensity
        final_facial_state = facial_state

        await send_to_client(client_id, {
            "type": "emotion_state_update",
            "emotion": final_emotion,
            "intensity": final_intensity,
            "facial_state": final_facial_state,
            "interaction_id": interaction_id,
            "timestamp": time.time()
        })

        if selected_bot_config:
            response_text = "I understand your question."
            await stream_response_audio_with_emotion_REALTIME(
                client_id=client_id,
                phrase=response_text,
                emotion=final_emotion,
                interaction_id=interaction_id,
                phrase_index=1,
                is_last=True,
                audio_type=audio_type,
                intensity=final_intensity,
                facial_state=final_facial_state
            )
        else:
            try:
                phrase_queue = asyncio.Queue()
                streaming_complete = asyncio.Event()

                async def tts_callback(phrase: str, phrase_emotion: dict, phrase_idx: int):
                    nonlocal final_emotion, final_intensity, final_facial_state

                    if await interruption_manager.check_interrupted(client_id, interaction_id):
                        return

                    if phrase_emotion is None:
                        effective_emotion = final_emotion
                        effective_intensity = final_intensity
                        effective_state = final_facial_state
                        clean_phrase = phrase
                    elif isinstance(phrase_emotion, dict):
                        effective_emotion = phrase_emotion.get("name", final_emotion)
                        effective_intensity = phrase_emotion.get("intensity", final_intensity)
                        effective_state = phrase_emotion.get("facial_state", final_facial_state)
                        clean_phrase = phrase

                        if phrase_idx == 1:
                            final_emotion = effective_emotion
                            final_intensity = effective_intensity
                            final_facial_state = effective_state
                    elif isinstance(phrase_emotion, str):
                        effective_emotion = phrase_emotion
                        effective_intensity = final_intensity
                        effective_state = final_facial_state
                        clean_phrase = phrase
                    else:
                        effective_emotion = final_emotion
                        effective_intensity = final_intensity
                        effective_state = final_facial_state
                        clean_phrase = phrase

                    clean_phrase = re.sub(r'\[EMOTION\s+[^\]]*\]\s*', '', clean_phrase).strip()
                    clean_phrase = re.sub(r'^\[.*?\]\s*', '', clean_phrase).strip()

                    if not clean_phrase or len(clean_phrase.split()) < 2:
                        return

                    await phrase_queue.put({
                        'phrase': clean_phrase,
                        'emotion': effective_emotion,
                        'intensity': effective_intensity,
                        'facial_state': effective_state,
                        'index': phrase_idx,
                        'is_last': False
                    })

                async def llm_task():
                    nonlocal response_text, final_emotion, final_intensity, final_facial_state
                    
                    try:
                        # CRITICAL: Check if interrupted BEFORE starting LLM call
                        if await interruption_manager.check_interrupted(client_id, interaction_id):
                            logger.info(f"🛑 LLM task cancelled - interrupted before start")
                            await phrase_queue.put({'is_last': True})
                            streaming_complete.set()
                            return

                        # USE FALLBACK SYSTEM (will automatically try SAM then Groq)
                        response_text, llm_emotion, role = await generate_llm_response_streaming_sam_with_fallback(
                            user_input=user_input,
                            conversation_history=conversation_history[-6:],
                            tts_callback=tts_callback,
                            emotion=final_emotion,
                            interruption_check=interruption_manager.check_interrupted,
                            client_id=client_id,
                            sam_timeout=1.0
                        )

                        # Check interruption after LLM completes
                        if await interruption_manager.check_interrupted(client_id, interaction_id):
                            logger.info(f"🛑 LLM task interrupted after completion")
                            await phrase_queue.put({'is_last': True})
                            streaming_complete.set()
                            return

                        llm_status.consecutive_failures = 0
                        llm_status.last_success_time = time.time()
                        llm_status.is_degraded = False

                        clean_response, parsed_emotion, parsed_intensity, parsed_state = \
                            parse_emotion_from_llm_response(response_text)

                        if parsed_emotion != "neutral":
                            final_emotion = parsed_emotion
                            final_intensity = parsed_intensity
                            final_facial_state = parsed_state

                            await send_to_client(client_id, {
                                "type": "emotion_state_update",
                                "emotion": final_emotion,
                                "intensity": final_intensity,
                                "facial_state": final_facial_state,
                                "interaction_id": interaction_id,
                                "timestamp": time.time()
                            })

                        response_text = clean_response

                    except asyncio.CancelledError:
                        logger.info(f"🛑 LLM task cancelled")
                        await phrase_queue.put({'is_last': True})
                        streaming_complete.set()
                        raise
                    except Exception as e:
                        llm_status.consecutive_failures += 1
                        llm_status.last_failure_time = time.time()
                        llm_status.is_degraded = True

                        logger.error(f" LLM error: {e}")

                        await send_to_client(client_id, {
                            "type": "llm_critical_error",
                            "status": "error",
                            "message": "LLM error occurred",
                            "error_details": str(e),
                            "can_continue": True,
                            "fallback_active": True
                        })

                        response_text = "I'm experiencing technical difficulties right now."
                        final_emotion = "concerned"

                    finally:
                        await phrase_queue.put({'is_last': True})
                        streaming_complete.set()

                async def tts_task():
                    phrase_count = 0

                    while True:
                        #  Check interruption FIRST - before waiting for queue
                        if await interruption_manager.check_interrupted(client_id, interaction_id):
                            #  DISCARD entire queue
                            discarded = 0
                            while not phrase_queue.empty():
                                try:
                                    phrase_queue.get_nowait()
                                    discarded += 1
                                except asyncio.QueueEmpty:
                                    break
                            logger.info(f"🛑 TTS task cancelled - discarded {discarded} phrases")
                            break

                        try:
                            phrase_data = await asyncio.wait_for(phrase_queue.get(), timeout=0.5)

                            if phrase_data.get('is_last'):
                                logger.debug(f"TTS task received final marker")
                                break
                            
                            #  Double-check before processing each phrase
                            if await interruption_manager.check_interrupted(client_id, interaction_id):
                                logger.info(f"🛑 Skipping phrase #{phrase_data.get('index', '?')} - interrupted")
                                # Discard remaining queue
                                discarded = 0
                                while not phrase_queue.empty():
                                    try:
                                        phrase_queue.get_nowait()
                                        discarded += 1
                                    except asyncio.QueueEmpty:
                                        break
                                logger.info(f"🛑 Discarded {discarded} remaining phrases")
                                break

                            phrase_count += 1

                            # Stream the phrase
                            await stream_response_audio_with_emotion_REALTIME(
                                client_id=client_id,
                                phrase=phrase_data['phrase'],
                                emotion=phrase_data['emotion'],
                                interaction_id=interaction_id,
                                phrase_index=phrase_count,
                                is_last=False,
                                audio_type=audio_type,
                                intensity=phrase_data.get('intensity', final_intensity),
                                facial_state=phrase_data.get('facial_state', final_facial_state)
                            )

                        except asyncio.TimeoutError:
                            # Check if streaming is complete and queue is empty
                            if streaming_complete.is_set() and phrase_queue.empty():
                                logger.debug(f"TTS task ending - streaming complete")
                                break
                            continue
                        except asyncio.CancelledError:
                            logger.info(f"🛑 TTS task cancelled")
                            break
                        except Exception as e:
                            logger.error(f"TTS task error: {e}")
                            break

                    # Final completion check - only send if not interrupted
                    if not await interruption_manager.check_interrupted(client_id, interaction_id):
                        await send_to_client(client_id, {
                            "type": "completion",
                            "status": "done",
                            "interaction_id": interaction_id,
                            "timestamp": time.time()
                        })
                        logger.info(f" Response complete for {interaction_id}")
                    else:
                        logger.info(f"🛑 Response interrupted for {interaction_id}")

                # Run both tasks concurrently
                await asyncio.gather(llm_task(), tts_task(), return_exceptions=True)

            except asyncio.CancelledError:
                logger.info(f"🛑 process_and_respond cancelled for {interaction_id}")
                raise
            except Exception as e:
                logger.error(f" Outer error in process_and_respond: {e}")
                response_text = "I understand what you're saying."
                final_emotion = "concerned"

        # Update conversation history
        conversation_history.append({"User": user_input})
        conversation_history.append({"Assistant": response_text})

        if len(conversation_history) > 8:
            conversation_history[:] = conversation_history[-8:]

    except asyncio.CancelledError:
        logger.info(f"🛑 process_and_respond fully cancelled")
        raise
    except Exception as e:
        logger.error(f" Process error: {e}")

    finally:
        if client_id in client_sessions:
            client_sessions[client_id]['current_interaction_id'] = None
            logger.debug(f"Cleared current_interaction_id for {client_id}")
# =============================================================================
# UPDATED: stream_response_audio_with_emotion_REALTIME - Fix Speak API blocking
# =============================================================================

async def stream_response_audio_with_emotion_REALTIME(
    client_id: str,
    phrase: str,
    emotion: str,
    interaction_id: str,
    phrase_index: int,
    is_last: bool = False,
    audio_type: str = "boy",
    intensity: float = 1.0,
    facial_state: str = "neutral"
):
    """
     FIXED: Allow Speak API to bypass interaction ID check
    """
    #  FIX: Check if this is a Speak API call
    is_speak_api = interaction_id.startswith("speak_")
    
    #  UPDATED: Only check interaction ID for non-Speak API calls
    if not is_speak_api and client_id in client_sessions:
        current_interaction = client_sessions[client_id].get('current_interaction_id')
        if current_interaction != interaction_id:
            logger.info(
                f"Phrase #{phrase_index} skipped - wrong interaction "
                f"(current: {current_interaction}, got: {interaction_id})"
            )
            return  # Exit immediately for WebSocket calls
    
    #  NEW: For Speak API, set the interaction ID in the session
    if is_speak_api and client_id in client_sessions:
        client_sessions[client_id]['current_interaction_id'] = interaction_id
        logger.debug(f" Speak API set interaction ID: {interaction_id}")
    
    #  START: Overall timing
    pipeline_start = time.time()
    
    #  EARLY INTERRUPTION CHECK - but skip for Speak API
    if not is_speak_api and interaction_id != "-1" and await interruption_manager.check_interrupted(client_id, interaction_id):
        logger.info(f"Phrase #{phrase_index} interrupted before starting")
        return
    
    # === STEP 1: Initialize or get emotion state manager ===
    step1_start = time.time()
    if client_id not in emotion_state_managers:
        emotion_state_managers[client_id] = EmotionStateManager(num_blendshapes=NUM_BLENDSHAPES, fps=60)
        logger.info(f"Created real-time emotion state for client {client_id}")
    
    state_manager = emotion_state_managers[client_id]
    state_manager.set_target_emotion(emotion, intensity)
    step1_elapsed = (time.time() - step1_start) * 1000
    logger.info(f" [STEP 1] Emotion state init: {step1_elapsed:.2f}ms")

    # === STEP 2: Parse emotion from dict or string ===
    step2_start = time.time()
    if isinstance(emotion, dict):
        target_emotion = emotion.get("name", "neutral")
        target_intensity = emotion.get("intensity", intensity)
        target_state = emotion.get("facial_state", facial_state)
    else:
        target_emotion = emotion
        target_intensity = intensity
        target_state = facial_state
    
    # Clean ALL metadata tags before TTS
    phrase = re.sub(r'\[(?:EMOTION|TENSION)\s+[^\]]*(?:\]|$)', '', phrase).strip()
    phrase = re.sub(r'^\[.*?\]\s*', '', phrase).strip()
    
    if not phrase or len(phrase.split()) < 2:
        logger.warning(f"Phrase #{phrase_index} empty after cleaning, skipping")
        return
    
    step2_elapsed = (time.time() - step2_start) * 1000
    logger.info(f" [STEP 2] Text cleaning: {step2_elapsed:.2f}ms | Text: '{phrase[:50]}...'")
    
    try:
        
        """
        *******************************************************************************************
                                    # === STEP 3: Generate TTS audio ===
        *******************************************************************************************
        """
        # === STEP 3: Generate TTS audio ===
        tts_start = time.time()
        logger.info(f" [STEP 3] Starting TTS generation for {len(phrase)} chars...")
        
        # tts_task = asyncio.create_task(tts_pool.generate(text=phrase, audio_type=audio_type, language="en-US"))
        tts_task_id = f"tts_{client_id}_{interaction_id}_{phrase_index}_{int(time.time()*1000)}"
        
        #  UPDATED: Only register for interruption if not Speak API
        if not is_speak_api:
            interruption_manager.register_tts_task(client_id, tts_task_id)
        try:
            audio_data, sample_rate = await tts_pool.generate(
                text=phrase,
                audio_type=audio_type,
                language="en-US",
                task_id=tts_task_id if not is_speak_api else None
            )
        except asyncio.TimeoutError:
            logger.warning(f" TTS generation timeout for phrase #{phrase_index}")
            # tts_task.cancel()
            return
        except asyncio.CancelledError:
            logger.info(f"🛑 TTS generation cancelled for phrase #{phrase_index}")
            return
        finally:
            if not is_speak_api:
                interruption_manager.unregister_tts_task(client_id, tts_task_id)

        tts_elapsed = (time.time() - tts_start) * 1000
        audio_duration = len(audio_data) / sample_rate
        logger.info(
            f" [STEP 3] TTS COMPLETE: {tts_elapsed:.2f}ms | "
            f"Audio duration: {audio_duration:.2f}s | "
            f"RTF: {tts_elapsed/1000/audio_duration:.2f}x"
        )
        
        #  UPDATED: Check interruption (skip for Speak API)
        if not is_speak_api and interaction_id != "-1" and await interruption_manager.check_interrupted(client_id, interaction_id):
            logger.info(f"Phrase #{phrase_index} interrupted after TTS generation")
            return
        
        # === STEP 4: Parallel processing ===
        parallel_start = time.time()
        logger.info(f"[STEP 4] Starting parallel processing (visemes + encoding)...")
        
        async def parallel_processing():
            loop = asyncio.get_running_loop()
            
            def generate_lipsync_animations():
                lipsync_start = time.time()
                
                #  UPDATED: Check interruption (skip for Speak API)
                if not is_speak_api and interaction_id != "-1" and asyncio.run_coroutine_threadsafe(
                    interruption_manager.check_interrupted(client_id, interaction_id),
                    loop
                ).result():
                    return None
                
                audio_bytes = audio_data.tobytes() if hasattr(audio_data, 'tobytes') else audio_data
                actual_duration = len(audio_data) / sample_rate
                
                # Viseme alignment
                viseme_start = time.time()
                segments, energy = align_visemes(audio_bytes, phrase, sample_rate)
                viseme_elapsed = (time.time() - viseme_start) * 1000
                logger.info(f"   [4a] Viseme alignment: {viseme_elapsed:.2f}ms | {len(segments)} visemes")
                
                #  UPDATED: Check interruption again (skip for Speak API)
                if not is_speak_api and interaction_id != "-1" and asyncio.run_coroutine_threadsafe(
                    interruption_manager.check_interrupted(client_id, interaction_id),
                    loop
                ).result():
                    return None
                # Generate blendshapes
                blendshape_start = time.time()
                blendshapes_lipsync = generate_blendshapes_realtime_FIXED(
                    segments=segments,
                    energy=energy,
                    sample_rate=sample_rate,
                    emotion=target_emotion,
                    audio_type=audio_type,
                    actual_audio_duration=actual_duration,
                    intensity=target_intensity,
                    facial_state="neutral"
                )
                blendshape_elapsed = (time.time() - blendshape_start) * 1000
                logger.info(f"   [4b] Blendshape generation: {blendshape_elapsed:.2f}ms")
                
                lipsync_total = (time.time() - lipsync_start) * 1000
                logger.info(f"   [4] Lip-sync TOTAL: {lipsync_total:.2f}ms")
                
                return blendshapes_lipsync, segments, energy, actual_duration
            
            # Run in parallel
            encode_start = time.time()
            animations_task = loop.run_in_executor(None, generate_lipsync_animations)
            encode_task = loop.run_in_executor(None, lambda: base64.b64encode(audio_data.tobytes()).decode("utf-8"))
            
            try:
                (blendshapes_base, segments, energy, duration), audio_b64 = await asyncio.wait_for(
                    asyncio.gather(animations_task, encode_task),
                    timeout=10.0
                )
                encode_elapsed = (time.time() - encode_start) * 1000
                logger.info(f"   [4c] Audio encoding: {encode_elapsed:.2f}ms")
            except asyncio.TimeoutError:
                logger.warning(f" Animation generation timeout for phrase #{phrase_index}")
                animations_task.cancel()
                encode_task.cancel()
                return None, None, None, None, None
            
            if blendshapes_base is None:
                return None, None, None, None, None
            
            #  UPDATED: Check interruption (skip for Speak API)
            if not is_speak_api and interaction_id != "-1" and await interruption_manager.check_interrupted(client_id, interaction_id):
                return None, None, None, None, None
            
            return blendshapes_base, segments, energy, duration, audio_b64
        
        result_task = asyncio.create_task(parallel_processing())
        try:
            result = await asyncio.wait_for(result_task, timeout=12.0)
        except asyncio.TimeoutError:
            logger.warning(f" Parallel processing timeout for phrase #{phrase_index}")
            result_task.cancel()
            return
        except asyncio.CancelledError:
            logger.info(f"🛑 Processing cancelled for phrase #{phrase_index}")
            return
        
        parallel_elapsed = (time.time() - parallel_start) * 1000
        logger.info(f" [STEP 4] Parallel processing COMPLETE: {parallel_elapsed:.2f}ms")
        
        if result is None or result[0] is None:
            logger.info(f"Phrase #{phrase_index} interrupted during generation")
            return
        
        blendshapes_base, segments, energy_data, audio_duration, audio_b64 = result
        
        # === STEP 5: Frame-by-frame emotion blending ===
        frame_start = time.time()
        logger.info(f"[STEP 5] Starting frame-by-frame emotion blending...")
        
        fps = 60
        total_frames = int(audio_duration * fps)
        
        # Pre-compute audio energy timeline
        audio_bytes = audio_data.tobytes() if hasattr(audio_data, 'tobytes') else audio_data
        energy_timeline = load_audio_energy(audio_bytes, sample_rate)
        
        def get_audio_energy_at_time(t: float) -> float:
            idx = int(t * sample_rate)
            if idx < 0 or idx >= len(energy_timeline):
                return 0.0
            return np.clip(energy_timeline[idx] * 2.0, 0.0, 1.0)
        
        # Convert keyframes to dense
        def keyframes_to_dense(keyframes_dict: Dict, duration: float, fps: int) -> np.ndarray:
            num_frames = int(duration * fps)
            dense = np.zeros((num_frames, NUM_BLENDSHAPES), dtype=np.float32)
            
            for shape_name, kf_list in keyframes_dict.items():
                if shape_name not in BLENDSHAPE_INDEX:
                    continue
                
                shape_idx = BLENDSHAPE_INDEX[shape_name]
                if shape_idx >= NUM_BLENDSHAPES:
                    continue
                
                kf_list = sorted(kf_list, key=lambda k: k['start'])
                
                for frame_idx in range(num_frames):
                    t = frame_idx / fps
                    value = 0.0
                    
                    if t <= kf_list[0]['start']:
                        value = kf_list[0]['intensity']
                    elif t >= kf_list[-1]['start']:
                        value = kf_list[-1]['intensity']
                    else:
                        for i in range(len(kf_list) - 1):
                            if kf_list[i]['start'] <= t <= kf_list[i+1]['start']:
                                t0, v0 = kf_list[i]['start'], kf_list[i]['intensity']
                                t1, v1 = kf_list[i+1]['start'], kf_list[i+1]['intensity']
                                
                                if t1 == t0:
                                    value = v0
                                else:
                                    alpha = (t - t0) / (t1 - t0)
                                    value = v0 + (v1 - v0) * alpha
                                break
                    
                    dense[frame_idx, shape_idx] = value
            return dense
        
        dense_start = time.time()
        B_lipsync_dense = keyframes_to_dense(blendshapes_base, audio_duration, fps)
        dense_elapsed = (time.time() - dense_start) * 1000
        logger.info(f"   [5a] Keyframe to dense conversion: {dense_elapsed:.2f}ms | {total_frames} frames")
        
        # Frame processing
        blend_start = time.time()
        B_final_dense = np.zeros((total_frames, NUM_BLENDSHAPES), dtype=np.float32)
        user_is_speaking = client_sessions.get(client_id, {}).get('is_speaking', False)
        
        for frame_idx in range(total_frames):
            t = frame_idx / fps
            
            #  UPDATED: Check interruption every 10 frames (skip for Speak API)
            if not is_speak_api and frame_idx % 10 == 0 and interaction_id != "-1":
                if await interruption_manager.check_interrupted(client_id, interaction_id):
                    logger.info(f"🛑 Phrase #{phrase_index} interrupted during frame processing")
                    return
            
            B_lipsync_frame = B_lipsync_dense[frame_idx]
            audio_energy = get_audio_energy_at_time(t)
            
            if len(B_lipsync_frame) != NUM_BLENDSHAPES:
                if len(B_lipsync_frame) > NUM_BLENDSHAPES:
                    B_lipsync_frame = B_lipsync_frame[:NUM_BLENDSHAPES]
                else:
                    padded = np.zeros(NUM_BLENDSHAPES, dtype=np.float32)
                    padded[:len(B_lipsync_frame)] = B_lipsync_frame
                    B_lipsync_frame = padded
            
            disable_micro = user_is_speaking
            B_final_frame = state_manager.update_frame(
                B_lipsync_frame, 
                audio_energy,
                is_speaking=disable_micro
            )
            
            # Neck amplification
            NECK_SCALE = 1.8
            for neck in ["neckDownTiltLeft", "neckDownTiltRight", "neckUpTiltLeft", 
                        "neckUpTiltRight", "neckTurnLeft", "neckTurnRight"]:
                if neck in BLENDSHAPE_INDEX:
                    idx = BLENDSHAPE_INDEX[neck]
                    B_final_frame[idx] *= NECK_SCALE
            
            B_final_dense[frame_idx] = B_final_frame
        
        blend_elapsed = (time.time() - blend_start) * 1000
        logger.info(f"   [5b] Frame blending: {blend_elapsed:.2f}ms | {total_frames} frames @ {total_frames/(blend_elapsed/1000):.0f} fps")
        
        frame_elapsed = (time.time() - frame_start) * 1000
        logger.info(f" [STEP 5] Frame processing COMPLETE: {frame_elapsed:.2f}ms")
        
        # === STEP 6: Dense to keyframes conversion ===
        conversion_start = time.time()
        logger.info(f"[STEP 6] Converting dense back to keyframes...")
        
        def dense_to_keyframes(dense: np.ndarray, fps: int, threshold: float = 0.01) -> Dict:
            num_frames, num_shapes = dense.shape
            
            if num_shapes != NUM_BLENDSHAPES:
                if num_shapes > NUM_BLENDSHAPES:
                    dense = dense[:, :NUM_BLENDSHAPES]
                    num_shapes = NUM_BLENDSHAPES
            
            keyframes = {}
            NECK_MOVEMENT_SHAPES = {
                "neckTurnLeft", "neckTurnRight", "neckUp", "neckDown",
                "neckLeft", "neckRight", "neckForward", "neckBackward",
                "neckDownTiltLeft", "neckDownTiltRight", 
                "neckUpTiltLeft", "neckUpTiltRight"
            }
            
            for shape_idx in range(num_shapes):
                if shape_idx >= len(BLENDSHAPE_ORDER):
                    continue
                
                shape_name = BLENDSHAPE_ORDER[shape_idx]
                values = dense[:, shape_idx]
                
                if shape_name in NECK_MOVEMENT_SHAPES:
                    active_threshold = 0.0001
                    change_threshold = 0.0002
                else:
                    active_threshold = threshold
                    change_threshold = threshold
                
                kf_list = []
                kf_list.append({
                    "name": shape_name,
                    "start": 0.0,
                    "intensity": round(float(values[0]), 4)
                })
                
                for i in range(1, num_frames - 1):
                    t = i / fps
                    curr_val = values[i]
                    prev_val = values[i-1]
                    
                    change = abs(curr_val - prev_val)
                    
                    if change > change_threshold:
                        kf_list.append({
                            "name": shape_name,
                            "start": round(t, 4),
                            "intensity": round(float(curr_val), 4)
                        })
                
                kf_list.append({
                    "name": shape_name,
                    "start": round((num_frames - 1) / fps, 4),
                    "intensity": round(float(values[-1]), 4)
                })
                max_intensity = max(abs(kf['intensity']) for kf in kf_list)
                
                if max_intensity >= active_threshold:
                    keyframes[shape_name] = kf_list
            return keyframes
        
        blendshapes_final = dense_to_keyframes(B_final_dense, fps, threshold=0.02)
        conversion_elapsed = (time.time() - conversion_start) * 1000
        total_keyframes = sum(len(kf) for kf in blendshapes_final.values())
        logger.info(f" [STEP 6] Dense to keyframes COMPLETE: {conversion_elapsed:.2f}ms | {total_keyframes} keyframes")
        
        # === STEP 7: Build message ===
        message_start = time.time()
        logger.info(f"[STEP 7] Building message payload...")
        
        #  UPDATED: Skip interruption check for Speak API phrase #1
        should_skip_interrupt_check = is_speak_api
        if not should_skip_interrupt_check and phrase_index == 1:
            interaction_status = await interruption_manager.check_interrupted(client_id, interaction_id)
            current_interaction = client_sessions.get(client_id, {}).get('current_interaction_id')
            
            if current_interaction == interaction_id:
                should_skip_interrupt_check = True

        if not should_skip_interrupt_check:
            if interaction_id != "-1" and await interruption_manager.check_interrupted(client_id, interaction_id):
                logger.info(f"Phrase #{phrase_index} interrupted after blendshape generation")
                return
        
        # Build viseme timeline
        viseme_timeline = [
            {
                "name": seg["name"],
                "intensity": 1.0,
                "start": round(seg["start"], 4),
                "end": round(seg["end"], 4),
            }
            for seg in segments
        ]
        # Build emotion metadata
        emotion_metadata = None
        if phrase_index == 1: 
            emotion_metadata = {
                "name": target_emotion,
                "intensity": target_intensity,
                "facial_state": target_state
            }
        # Build message
        if emotion_metadata:
            message = {
                "type": "audio_chunk",
                "transcript": phrase,
                "pcm_base64": audio_b64,
                "sample_rate": sample_rate,
                "blendshape_animations": blendshapes_final,
                "visemes": viseme_timeline,
                "duration": audio_duration,
                "flag": 1,
                "utterance_index": phrase_index,
                "is_last": is_last,
                "interaction_id": interaction_id,
                "audio_type": audio_type,
                "emotion": emotion_metadata["name"],
                "emotion_intensity": emotion_metadata["intensity"],
                "facial_state": emotion_metadata["facial_state"],
                "realtime_emotion": True,
                "audio_reactive": True,
                "micro_motion": True,
                "frame_rate": fps,
                "is_speak_api": is_speak_api  #  NEW: Flag for frontend
            }
        else:
            message = {
                "type": "audio_chunk",
                "transcript": phrase,
                "pcm_base64": audio_b64,
                "sample_rate": sample_rate,
                "blendshape_animations": blendshapes_final,
                "visemes": viseme_timeline,
                "duration": audio_duration,
                "flag": 1,
                "utterance_index": phrase_index,
                "is_last": is_last,
                "interaction_id": interaction_id,
                "audio_type": audio_type,
                "emotion": target_emotion,
                "realtime_emotion": True,
                "audio_reactive": True,
                "micro_motion": True,
                "frame_rate": fps,
                "is_speak_api": is_speak_api  #  NEW: Flag for frontend
            }
        message_elapsed = (time.time() - message_start) * 1000
        logger.info(f" [STEP 7] Message building COMPLETE: {message_elapsed:.2f}ms")
        
        # === STEP 8: Send to client ===
        send_start = time.time()
        logger.info(f" [STEP 8] Sending to client...")
        
        try:
            if client_id in {"1", "0", "-1"}:
                send_task = asyncio.create_task(_broadcast_to_all(message))
            else:
                send_task = asyncio.create_task(send_to_client(client_id, message))
            
            await asyncio.wait_for(send_task, timeout=3.0)
            send_elapsed = (time.time() - send_start) * 1000
            logger.info(f" [STEP 8] Send COMPLETE: {send_elapsed:.2f}ms")
        except asyncio.TimeoutError:
            logger.warning(f"Send timeout for phrase #{phrase_index}")
            return
        except Exception as e:
            logger.error(f" Send error for phrase #{phrase_index}: {e}")
            return
        
        # === FINAL SUMMARY ===
        pipeline_elapsed = (time.time() - pipeline_start) * 1000
        
        logger.info("=" * 80)
        logger.info(f" PHRASE #{phrase_index} COMPLETE - TIMING BREAKDOWN:")
        logger.info(f"    Text: '{phrase[:60]}...'")
        logger.info(f"     TTS Generation:       {tts_elapsed:>8.2f}ms ({tts_elapsed/pipeline_elapsed*100:>5.1f}%)")
        logger.info(f"    Parallel Processing:  {parallel_elapsed:>8.2f}ms ({parallel_elapsed/pipeline_elapsed*100:>5.1f}%)")
        logger.info(f"    Frame Blending:        {frame_elapsed:>8.2f}ms ({frame_elapsed/pipeline_elapsed*100:>5.1f}%)")
        logger.info(f"    Keyframe Conversion:   {conversion_elapsed:>8.2f}ms ({conversion_elapsed/pipeline_elapsed*100:>5.1f}%)")
        logger.info(f"    Message Building:      {message_elapsed:>8.2f}ms ({message_elapsed/pipeline_elapsed*100:>5.1f}%)")
        logger.info(f"     Network Send:          {send_elapsed:>8.2f}ms ({send_elapsed/pipeline_elapsed*100:>5.1f}%)")
        logger.info(f"      TOTAL PIPELINE:        {pipeline_elapsed:>8.2f}ms")
        logger.info(f"    Audio Duration:        {audio_duration:.2f}s")
        logger.info(f"     Real-time Factor:      {pipeline_elapsed/1000/audio_duration:.2f}x")
        logger.info(f"    Source: {'Speak API' if is_speak_api else 'WebSocket'}")
        logger.info("=" * 80)
        
        # Playback monitoring
        playback_start = time.time()
        check_interval = 0.015
        
        while (time.time() - playback_start) < audio_duration:
            #  UPDATED: Skip interruption check for Speak API
            if not is_speak_api and interaction_id != "-1" and await interruption_manager.check_interrupted(client_id, interaction_id):
                stop_msg = {
                    "type": "immediate_stop",
                    "interaction_id": interaction_id,
                    "stop_audio": True,
                    "stop_animations": True,
                    "reset_face": True,
                    "reason": "user_interruption"
                }
                try:
                    if client_id in {"1", "0", "-1"}:
                        await _broadcast_to_all(stop_msg)
                    else:
                        await send_to_client(client_id, stop_msg)
                except:
                    pass
                
                logger.info(f"🛑 Phrase #{phrase_index} interrupted during playback")
                return
            await asyncio.sleep(check_interval)
        
        # Natural pause
        if not is_last:
            if 'last_pause' not in locals():
                last_pause = 0.0

            speech_rate = max(0.6, len(phrase.split()) / audio_duration)
            pause_duration = calculate_natural_pause_optimized(
                phrase,
                phrase_index,
                emotion=target_emotion,
                speech_rate=speech_rate,
                prev_pause=last_pause
            )

            last_pause = pause_duration

            pause_start = time.time()
            while (time.time() - pause_start) < pause_duration:
                #  UPDATED: Skip interruption check for Speak API
                if not is_speak_api and interaction_id != "-1" and await interruption_manager.check_interrupted(client_id, interaction_id):
                    logger.info(f"🛑 Phrase #{phrase_index} interrupted during pause")
                    return
                await asyncio.sleep(check_interval)

        logger.info(f" Phrase #{phrase_index} playback complete")
        
    except asyncio.CancelledError:
        logger.info(f"🛑 Phrase #{phrase_index} cancelled")
    except Exception as e:
        logger.error(f" Phrase #{phrase_index} error: {e}")
        traceback.print_exc()
    finally:
        #  NEW: Clear interaction ID for Speak API when last phrase completes
        if is_speak_api and is_last and client_id in client_sessions:
            client_sessions[client_id]['current_interaction_id'] = None
            logger.debug(f" Speak API cleared interaction ID (last phrase)")

async def send_to_client(client_id: str, message: dict):
    """Send message to specific client"""
    if client_id not in client_sessions:
        return
    try:
        ws = client_sessions[client_id]['websocket']
        await ws.send_json(message)
        stats["messages_broadcast"] += 1
    except Exception as e:
        logger.error(f"Send to {client_id} failed: {e}")

async def _broadcast_to_all(message: dict):
    """Broadcast to all connected WS clients. Removes dead sessions safely."""
    dead_clients = []
   
    for client_id, session in list(client_sessions.items()):
        ws = session.get("websocket")
        if ws is None:
            continue
        try:
            await ws.send_json(message)
        except Exception:
            logger.exception(f"Broadcast: failed to send to {client_id}, marking dead")
            dead_clients.append(client_id)

    for cid in dead_clients:
        client_sessions.pop(cid, None)

    stats["messages_broadcast"] = stats.get("messages_broadcast", 0) + 1

# =============================================================================
# FASTAPI LIFESPAN
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🎬 [STARTUP] Starting Production Voice Assistant...")
    
    try:
        await audio_processor.initialize()       
        await tts_pool.initialize()
        
        #  NEW: Initialize S2S handler
        global s2s_handler
        s2s_handler = IntegratedSpeechToSpeechHandler(
            audio_processor=audio_processor,
            tts_pool=tts_pool,
            generate_llm_response_streaming=generate_llm_response_streaming
        )
        logger.info(" Speech-to-Speech handler initialized")
        
        logger.info("All production components initialized")
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
    
    try:
        set_llm_config("groq")
        logger.info("Default LLM: Groq")
    except Exception as e:
        logger.warning(f"LLM config failed: {e}")
    
    # logger.info("Production system ready for 1000+ clients in noisy environments")
    
    yield
    
    logger.info("🔌 [SHUTDOWN] Cleaning up production system...")
    tts_pool.cleanup()
    audio_processor.cleanup()


app = FastAPI(lifespan=lifespan)

origins = [
    "https://talkinglady.instaviv.in",
    "https://talking.instaviv.in",
    "http://localhost:3000",
    "http://localhost:5173",
    "https://talkinglady.instavivai.com",
    "https://talking.instavivai.com",
    "https://talkingbackend.datavivservers.in",
    "https://talking.datavivservers.in",
    "http://localhost:4200",
    "https://unity_web.datavivservers.in/",
    "https://unity_web.datavivservers.in",
    "*"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# =============================================================================
# API ENDPOINTS
# =============================================================================
class SpeakRequest(BaseModel):
    text: str
    client_id: Optional[str] = None
    emotion : Optional[str] = None
    audio_type: Optional[str] = None
 
class TokenSubmitRequest(BaseModel):
    token: str
    client_id: Optional[str] = None
    session_id: str  #  NEW: Required from frontend

# =============================================================================
# UPDATED SPEAK API WITH PAUSE SUPPORT
# =============================================================================

# @app.post("/speak")
# async def speak(request: SpeakRequest):
#     try:
#         text = request.text.strip()
#         if not text:
#             raise HTTPException(status_code=400, detail="Text cannot be empty")

#         client_id = request.client_id or f"http_{int(time.time() * 1000)}"
#         audio_type = request.audio_type or "boy"
#         emotion = request.emotion or "neutral"
        
#         #  NEW: Parse pause tokens
#         cleaned_text, pauses = parse_pause_tokens(text)
        
#         if not cleaned_text:
#             raise HTTPException(status_code=400, detail="Text contains only pause tokens")
        
#         #  NEW: Split text by pauses
#         segments = split_text_by_pauses(cleaned_text, pauses)
        
#         logger.info(
#             f"Speak API | Client: {client_id} | "
#             f"Voice: {audio_type} | Emotion: {emotion} | "
#             f"Segments: {len(segments)} | "
#             f"Pauses: {len(pauses)} | "
#             f"Text: '{cleaned_text[:50]}...'"
#         )
        
#         # Initialize session if needed
#         if client_id not in client_sessions:
#             client_sessions[client_id] = {
#                 'websocket': None,
#                 'input_queue': asyncio.Queue(),
#                 'conversation_history': [],
#                 'connected_at': time.time(),
#                 'playback_end_time': 0,
#                 'audio_type': audio_type,
#                 'is_speaking': False,
#                 'last_interrupt_time': 0.0,
#                 'current_interaction_id': None
#             }
#         # Reset emotion state manager
#         if client_id in emotion_state_managers:
#             del emotion_state_managers[client_id]
        
#         # Create fresh emotion state manager
#         emotion_state_managers[client_id] = EmotionStateManager(
#             num_blendshapes=NUM_BLENDSHAPES, 
#             fps=60
#         )
#         # Set target emotion
#         state_manager = emotion_state_managers[client_id]
#         state_manager.set_target_emotion(emotion, intensity=1.0)
        
#         #  NEW: Create unique interaction ID
#         interaction_id = f"speak_{client_id}_{int(time.time() * 1000)}"
        
#         #  NEW: Mark Speak API as active (blocks WebSocket)
#         await set_speak_api_active(client_id, interaction_id)
        
#         try:
#             # Process each segment with pauses
#             for segment_idx, (segment_text, pause_duration) in enumerate(segments):
#                 is_last_segment = (segment_idx == len(segments) - 1)
                
#                 logger.info(
#                     f" Segment {segment_idx + 1}/{len(segments)}: "
#                     f"'{segment_text[:30]}...' | "
#                     f"Pause after: {pause_duration}s"
#                 )
#                 # Generate and stream audio for this segment
#                 await stream_response_audio_with_emotion_REALTIME(
#                     client_id=client_id,
#                     phrase=segment_text,
#                     emotion=emotion,
#                     interaction_id=interaction_id,
#                     phrase_index=segment_idx + 1,
#                     is_last=is_last_segment,
#                     audio_type=audio_type,
#                     intensity=1.0,
#                     facial_state="neutral"
#                 )
#                 #  NEW: Apply pause if this isn't the last segment
#                 if not is_last_segment and pause_duration > 0:
#                     pause_completed = await apply_pause(
#                         duration=pause_duration,
#                         client_id=client_id,
#                         interaction_id=interaction_id,
#                         interruption_manager=interruption_manager
#                     )
                    
#                     if not pause_completed:
#                         logger.warning(f"Speak API interrupted during pause")
#                         break
            
#             return { 
#                 "status": "success", 
#                 "message": "Audio streaming completed",
#                 "client_id": client_id,
#                 "emotion": emotion,
#                 "audio_type": audio_type,
#                 "segments": len(segments),
#                 "total_pauses": len(pauses),
#                 "cleaned_text": cleaned_text
#             }
#         finally:
#             await clear_speak_api_active(client_id)

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Speak API failed: {e}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail="Audio generation failed")



@app.post("/speak")
async def speak(request: SpeakRequest):
    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        client_id  = request.client_id or f"http_{int(time.time() * 1000)}"
        audio_type = request.audio_type or "boy"
        emotion    = request.emotion or "neutral"

        # Parse pause tokens
        cleaned_text, pauses = parse_pause_tokens(text)

        if not cleaned_text:
            raise HTTPException(status_code=400, detail="Text contains only pause tokens")

        # Split text by pauses
        segments = split_text_by_pauses(cleaned_text, pauses)

        logger.info(
            f"Speak API | Client: {client_id} | "
            f"Voice: {audio_type} | Emotion: {emotion} | "
            f"Segments: {len(segments)} | "
            f"Pauses: {len(pauses)} | "
            f"Text: '{cleaned_text[:50]}...'"
        )

        # Initialize session if needed
        if client_id not in client_sessions:
            client_sessions[client_id] = {
                'websocket':              None,
                'input_queue':            asyncio.Queue(),
                'conversation_history':   [],
                'connected_at':           time.time(),
                'playback_end_time':      0,
                'audio_type':             audio_type,
                'is_speaking':            False,
                'last_interrupt_time':    0.0,
                'current_interaction_id': None
            }

        # Reset emotion state manager
        if client_id in emotion_state_managers:
            del emotion_state_managers[client_id]

        # Create fresh emotion state manager
        emotion_state_managers[client_id] = EmotionStateManager(
            num_blendshapes=NUM_BLENDSHAPES,
            fps=60
        )

        # Set target emotion
        state_manager = emotion_state_managers[client_id]
        state_manager.set_target_emotion(emotion, intensity=1.0)

        # Create unique interaction ID
        interaction_id = f"speak_{client_id}_{int(time.time() * 1000)}"

        # Mark Speak API as active (blocks WebSocket)
        await set_speak_api_active(client_id, interaction_id)

        try:
            # Process each segment with pauses
            for segment_idx, (segment_text, pause_duration) in enumerate(segments):
                is_last_segment = (segment_idx == len(segments) - 1)

                logger.info(
                    f" Segment {segment_idx + 1}/{len(segments)}: "
                    f"'{segment_text[:30]}...' | "
                    f"Pause after: {pause_duration}s"
                )

                # Generate and stream audio for this segment
                await stream_response_audio_with_emotion_REALTIME(
                    client_id=client_id,
                    phrase=segment_text,
                    emotion=emotion,
                    interaction_id=interaction_id,
                    phrase_index=segment_idx + 1,
                    is_last=is_last_segment,
                    audio_type=audio_type,
                    intensity=1.0,
                    facial_state="neutral"
                )

                # Apply pause if this isn't the last segment
                if not is_last_segment and pause_duration > 0:
                    pause_completed = await apply_pause(
                        duration=pause_duration,
                        client_id=client_id,
                        interaction_id=interaction_id,
                        interruption_manager=interruption_manager
                    )

                    if not pause_completed:
                        logger.warning(f"Speak API interrupted during pause")
                        break

            # ── Build completion payload ──────────────────────────────────────
            completion_payload = {
                "status":       "success",
                "message":      "Speak API Audio streaming completed",
                "client_id":    client_id,
                "emotion":      emotion,
                "audio_type":   audio_type,
                "segments":     len(segments),
                "total_pauses": len(pauses),
                "cleaned_text": cleaned_text
            }

            # ── Send to WebSocket if client is connected ──────────────────────
            ws = client_sessions.get(client_id, {}).get('websocket')
            if ws is not None:
                try:
                    await ws.send_json(completion_payload)
                    logger.info(f" Speak completion sent to WebSocket: {client_id}")
                except Exception as ws_err:
                    logger.warning(f"Could not send completion to WebSocket: {ws_err}")
            else:
                logger.debug(f"[Speak] No WebSocket for {client_id} — skipping WS notify")

            # ── Return HTTP response ──────────────────────────────────────────
            return completion_payload

        finally:
            await clear_speak_api_active(client_id)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Speak API failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Audio generation failed")

@app.get("/")
async def root():
    return {
        "status": "running",
        "connections_active": stats["connections_active"],
        "connections_total": stats["connections_total"],
        "tts_generations": stats["tts_generations"],
        "interruptions": stats["interruptions"],
        "noise_suppressed_chunks": stats["noise_suppressed_chunks"],
        "tts_workers": tts_pool.num_workers,
        "audio_pipeline": "Production (95%+ noisy accuracy)"
    }

@app.get("/stats")
async def get_stats():
    return stats

@app.post("/configure_mic")
async def configure_mic(mic_setup: str):
    """
    Configure microphone setup for VAD tuning
    Options: dedicated, laptop_external, laptop_builtin, phone, headset, auto
    """
    try:
        setup = MicrophoneSetup(mic_setup.lower())
        audio_processor.update_mic_setup(setup)
        
        return {
            "status": "success",
            "mic_setup": setup.value,
            "vad_threshold": audio_processor.vad_threshold,
            "min_speech_duration": audio_processor.min_speech_duration,
            "description": audio_processor.vad_profile["description"]
        }
    except ValueError:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid mic_setup. Options: {[s.value for s in MicrophoneSetup]}"
        )

@app.get("/mic_info")
async def get_mic_info():
    """Get current microphone configuration"""
    return {
        "current_setup": audio_processor.mic_setup.value,
        "vad_threshold": audio_processor.vad_threshold,
        "min_speech_duration": audio_processor.min_speech_duration,
        "speech_pad_ms": audio_processor.speech_pad_ms,
        "description": audio_processor.vad_profile["description"],
        "available_setups": {
            setup.value: VAD_THRESHOLD_PROFILES[setup]
            for setup in MicrophoneSetup
        }
    }
# =============================================================================
#  PATCH 4: ADD EMOTION CHANGE ENDPOINT (Optional - for real-time changes)
# Location: After /mic_info endpoint (~line 2650)
# =============================================================================

class EmotionChangeRequest(BaseModel):
    client_id: str
    new_emotion: str
    new_intensity: float = 1.0
    new_facial_state: str = "neutral"

async def process_client_message_immediate(client_id: str, message_data: dict):
    """
     NEW: Process message IMMEDIATELY without queue
    Replaces the queue-based system
    """
    session = client_sessions.get(client_id)
    if not session:
        return
    
    user_input = message_data.get('text', '')
    audio_type = message_data.get('audio_type', 'boy')
    
    if not user_input or user_input.lower() == 'exit':
        return
    
    interaction_id = f"int_{client_id}_{int(time.time() * 1000)}"
    
    # Interrupt PREVIOUS interaction
    previous_interaction = session.get('current_interaction_id')
    if previous_interaction and previous_interaction != interaction_id:
        logger.info(f"🛑 Interrupting previous interaction: {previous_interaction}")
        await interruption_manager.signal_interruption(client_id)
        await asyncio.sleep(0.05)
    
    await interruption_manager.start_interaction(client_id, interaction_id)
    
    # Strip emotion tags and detect fresh
    clean_input = re.sub(r'^\[([^\]]+)\]\s*', '', user_input)
    
    initial_emotion, clean_input = detect_emotion_from_context_enhanced(
        clean_input,
        llm_emotion=None
    )
    
    logger.info(
        f"[CLIENT {client_id}] [{audio_type.upper()}] "
        f"Emotion: {initial_emotion} | Input: '{clean_input[:50]}...'"
    )
    
    # Process immediately
    await process_and_respond(
        client_id=client_id,
        user_input=clean_input,
        emotion=initial_emotion,
        interaction_id=interaction_id,
        conversation_history=session['conversation_history'],
        audio_type=audio_type
    )
# =============================================================================
# SPEAK API PRIORITY SYSTEM (Add before WebSocket endpoint)
# =============================================================================

speak_api_active = {} 

async def set_speak_api_active(client_id: str, interaction_id: str):
    """Mark Speak API as active for this client"""
    speak_api_active[client_id] = {
        "active": True,
        "interaction_id": interaction_id,
        "start_time": time.time()
    }
    logger.info(f"Speak API active for {client_id} (interaction: {interaction_id})")

async def clear_speak_api_active(client_id: str):
    """Clear Speak API active state"""
    if client_id in speak_api_active:
        del speak_api_active[client_id]
        logger.info(f" Speak API cleared for {client_id}")

async def is_speak_api_active(client_id: str) -> bool:
    """Check if Speak API is currently active for this client"""
    return client_id in speak_api_active and speak_api_active[client_id]["active"]

async def get_speak_api_interaction_id(client_id: str) -> str:
    """Get current Speak API interaction ID"""
    if client_id in speak_api_active:
        return speak_api_active[client_id].get("interaction_id", "")
    return ""
#  =============================================================================
# UPDATED WEBSOCKET ENDPOINT
# =============================================================================
def create_client_session(client_id,websocket):
     # Initialize client session WITHOUT input_queue
        client_sessions[client_id] = {
            'websocket': websocket,
            'conversation_history': [],
            'connected_at': time.time(),
            'playback_end_time': 0,
            'audio_type': 'boy',
            'is_speaking': False,
            'last_interrupt_time': 0.0,
            'current_interaction_id': None,
            'processing_task': None,  
            'sam_client': None  
        }

        client_token_status[client_id] = TokenStatus()
        stats["connections_total"] += 1
        stats["connections_active"] = len(client_sessions)

token = None
session_id = None

@app.websocket("/mic_input")
async def mic_input_ws(websocket: WebSocket):
    client_id = f"{websocket.client.host}:{websocket.client.port}_{int(time.time() * 1000)}" \
        if websocket.client else f"unknown_{int(time.time() * 1000)}"

    audio_type = 'boy'
    is_groq = False

    # FIX: track SAM client at handler scope so it persists across messages
    sam_client = None

    # Heartbeat task reference
    heartbeat_task = None
    
    try:
        await websocket.accept()
        logger.info(f" [WS CONNECTED] {client_id}")
        await websocket.send_json({
            "status": "connected",
            "message": "WebSocket connected with backend successfully",
            "client_id": client_id,
            "token_required": False,
            "can_continue": True
        })
        create_client_session(client_id, websocket)

        # Start heartbeat with custom interval (optional, default is 30s)
        heartbeat_task = await start_websocket_heartbeat(
            websocket=websocket,
            client_id=client_id,
            interval=5  # Send heartbeat every 30 seconds
        )
        
        
        async for message in websocket.iter_text():
            stats["messages_received"] += 1
            try:
                data = json.loads(message)

                if data.get("type") in ["ping","Ping"]:
                    await websocket.send_json({"type": "pong"})

                # ─────────────────────────────────────────────────────────────
                # MESSAGE 1: Token handshake
                # ─────────────────────────────────────────────────────────────
                if 'token' in data.keys():
                    token = data.get("token", "").strip()
                    session_id = (
                        data.get("session_id")
                        or data.get("session_Id")
                        or data.get("sessionID")
                        or ""
                    ).strip()

                    if not token:
                        logger.info("🟡 No token provided — using Groq fallback")
                        is_groq = True
                    else:
                        # FIX: Connect SAM ONCE here when token arrives,
                        # not on every subsequent audio message
                        try:
                            sam_client = SAMLLMClient(
                                base_url=SAM_WS_BASE_URL,
                                session_id=session_id,
                                access_token=token
                            )
                            sam_client.connect()  # raises on failure
                            client_sessions[client_id]['sam_client'] = sam_client
                            client_sessions[client_id]['session_id'] = session_id
                            sam_clients[client_id] = sam_client
                            logger.info(f" SAM client ready for {client_id}")
                            is_groq = False
                        except Exception as e:
                            logger.error(f" SAM connect failed: {e} — falling back to Groq")
                            sam_client = None
                            is_groq = True

                # ─────────────────────────────────────────────────────────────
                # MESSAGE 2+: Audio / control messages
                # ─────────────────────────────────────────────────────────────
                else:
                    # ── audio_type config ────────────────────────────────────
                    if "audio_type" in data:
                        audio_type = data.get("audio_type", "boy")
                        client_sessions[client_id]['audio_type'] = audio_type
                        await send_log_to_frontend(client_id, "audio", f"Voice changed to: {audio_type}", "info")

                    # ── isSpeak (INTERRUPTION HANDLING) ──────────────────────
                    # if data.get("type") == "isSpeak":
                    #     is_speaking_value = data.get("value", False)
                    #     previous_state = client_sessions[client_id].get('is_speaking', False)
                    #     client_sessions[client_id]['is_speaking'] = is_speaking_value

                    #     if is_speaking_value and not previous_state:
                    #         current_time = time.time()
                    #         last_interrupt = client_sessions[client_id].get('last_interrupt_time', 0.0)

                    #         #  Debounce interruptions (0.5s minimum between interrupts)
                    #         if current_time - last_interrupt >= 0.5:
                    #             client_sessions[client_id]['last_interrupt_time'] = current_time
                    #             current_interaction = client_sessions[client_id].get('current_interaction_id')
                                
                    #             logger.info(f"🛑 User started speaking - interrupting {current_interaction}")
                                
                    #             #  CRITICAL: Cancel current processing task and WAIT
                    #             processing_task = client_sessions[client_id].get('processing_task')
                    #             if processing_task and not processing_task.done():
                    #                 logger.info(f"🛑 Cancelling ongoing task (SAM or Groq)...")
                    #                 processing_task.cancel()
                                    
                    #                 #  WAIT for task to actually cancel (critical!)
                    #                 try:
                    #                     await asyncio.wait_for(processing_task, timeout=0.5)
                    #                     logger.info(f" Task cancelled successfully")
                    #                 except asyncio.CancelledError:
                    #                     logger.info(f" Task cancelled via CancelledError")
                    #                 except asyncio.TimeoutError:
                    #                     logger.warning(f"Task cancellation timed out (took >500ms)")
                    #                 except Exception as e:
                    #                     logger.error(f"Error waiting for task cancellation: {e}")
                    #             else:
                    #                 logger.debug(f"No active task to cancel")
                                
                    #             #  Signal interruption to stop TTS and LLM
                    #             if current_interaction:
                    #                 interrupted = await interruption_manager.signal_interruption(client_id)
                    #                 logger.info(f"🛑 Interruption signal sent: {interrupted}")
                                
                    #             #  Send stop signal to frontend
                    #             try:
                    #                 await websocket.send_json({
                    #                     "type": "immediate_stop",
                    #                     "reason": "user_speaking",
                    #                     "stop_audio": True,
                    #                     "stop_animations": True,
                    #                     "reset_face": True,
                    #                     "clear_queue": True,
                    #                     "flush_pending": True,
                    #                     "ready_for_input": True
                    #                 })
                    #                 logger.info(f" Stop signal sent to frontend")
                    #             except Exception as e:
                    #                 logger.error(f"Failed to send stop signal: {e}")
                                
                    #             await send_log_to_frontend(client_id, "interaction", "⏸️ Interrupted by user", "info")

                    #     # Acknowledge the isSpeak message
                    #     await websocket.send_json({
                    #         "type": "isSpeak_ack",
                    #         "value": is_speaking_value,
                    #         "timestamp": time.time()
                    #     })
                    #     continue
                    
                    # # ── isSpeak (INTERRUPTION HANDLING) ──────────────────────
                    
                    if data.get("type") == "isSpeak":
                        is_speaking_value = data.get("value", False)
                        previous_state = client_sessions[client_id].get('is_speaking', False)
                        client_sessions[client_id]['is_speaking'] = is_speaking_value

                        if is_speaking_value and not previous_state:
                            current_time = time.time()
                            last_interrupt = client_sessions[client_id].get('last_interrupt_time', 0.0)

                            if current_time - last_interrupt >= 0.5:
                                client_sessions[client_id]['last_interrupt_time'] = current_time
                                current_interaction = client_sessions[client_id].get('current_interaction_id')

                                logger.info(f"🛑 User started speaking - interrupting {current_interaction}")

                                # Step 1: Signal interruption FIRST (before cancelling task)
                                # so any running coroutine sees it immediately
                                if current_interaction:
                                    await interruption_manager.signal_interruption(client_id)
                                    logger.info(f"🛑 Interruption signalled")

                                # Step 2: Close SAM WebSocket to unblock blocking thread
                                _sam = client_sessions[client_id].get('sam_client')
                                if _sam and _sam.ws:
                                    try:
                                        _sam.ws.close()
                                        logger.info(f"🛑 SAM WebSocket closed to unblock thread")
                                    except Exception as e:
                                        logger.warning(f"SAM ws close error: {e}")

                                # Step 3: Cancel the asyncio task
                                processing_task = client_sessions[client_id].get('processing_task')
                                if processing_task and not processing_task.done():
                                    logger.info(f"🛑 Cancelling processing task...")
                                    processing_task.cancel()
                                    try:
                                        await asyncio.wait_for(
                                            asyncio.shield(processing_task), timeout=0.3
                                        )
                                    except (asyncio.CancelledError, asyncio.TimeoutError):
                                        logger.info(f"✅ Task cancelled")
                                    except Exception as e:
                                        logger.error(f"Task cancel error: {e}")
                                else:
                                    logger.debug(f"No active task to cancel")

                                # Step 4: Send stop to frontend explicitly
                                try:
                                    await websocket.send_json({
                                        "type": "immediate_stop",
                                        "reason": "user_speaking",
                                        "stop_audio": True,
                                        "stop_animations": True,
                                        "reset_face": True,
                                        "clear_queue": True,
                                        "flush_pending": True,
                                        "ready_for_input": True
                                    })
                                    logger.info(f"✅ Stop signal sent to frontend")
                                except Exception as e:
                                    logger.error(f"Failed to send stop signal: {e}")

                                await send_log_to_frontend(
                                    client_id, "interaction", "⏸️ Interrupted by user", "info"
                                )

                        await websocket.send_json({
                            "type": "isSpeak_ack",
                            "value": is_speaking_value,
                            "timestamp": time.time()
                        })
                        continue

                    # ── configure_mic ────────────────────────────────────────
                    if data.get("type") == "configure_mic":
                        mic_setup_value = data.get("mic_setup", "auto")
                        try:
                            setup = MicrophoneSetup(mic_setup_value.lower())
                            audio_processor.update_mic_setup(setup)
                            await websocket.send_json({
                                "type": "mic_configured",
                                "status": "success",
                                "mic_setup": setup.value,
                                "vad_threshold": audio_processor.vad_threshold,
                                "description": audio_processor.vad_profile["description"]
                            })
                            await send_log_to_frontend(client_id, "audio", f"Mic configured: {setup.value}", "success")
                        except ValueError:
                            await websocket.send_json({"type": "mic_config_failed", "status": "error", "message": f"Invalid mic setup: {mic_setup_value}"})
                            await send_log_to_frontend(client_id, "audio", f" Invalid mic setup: {mic_setup_value}", "error")
                        continue

                    # ── manual interruption ──────────────────────────────────
                    if data.get("type") == "interruption":
                        logger.info(f"🛑 Manual interruption requested for {client_id}")
                        
                        # Cancel current task
                        processing_task = client_sessions[client_id].get('processing_task')
                        if processing_task and not processing_task.done():
                            processing_task.cancel()
                            try:
                                await asyncio.wait_for(processing_task, timeout=0.2)
                            except (asyncio.CancelledError, asyncio.TimeoutError):
                                pass
                        
                        await interruption_manager.signal_interruption(client_id)
                        await send_log_to_frontend(client_id, "interaction", "🛑 User interruption", "info")
                        continue

                    # ── audio data (SAM PATH WITH INTERRUPTION SUPPORT) ──
                    # ── audio data (SAM PATH WITH INTERRUPTION SUPPORT) ──
                    if "audio_data" in data:
                        audio_base64 = data["audio_data"]

                        if not isinstance(audio_base64, str):
                            continue

                        # ─────────────────────────────────────────────────────────────────────
                        # STEP 0: If any task is currently running, INTERRUPT IT FIRST
                        # This handles the case where new audio arrives mid-response
                        # ─────────────────────────────────────────────────────────────────────
                        processing_task = client_sessions[client_id].get('processing_task')
                        if processing_task and not processing_task.done():
                            logger.info(f"🛑 New audio arrived — interrupting current response")

                            current_interaction = client_sessions[client_id].get('current_interaction_id')

                            # 1. Signal interruption so the running task's checks fire
                            if current_interaction:
                                await interruption_manager.signal_interruption(client_id)

                            # 2. Close SAM WebSocket to unblock the blocking thread in executor
                            _sam = client_sessions[client_id].get('sam_client')
                            if _sam and hasattr(_sam, 'ws') and _sam.ws:
                                try:
                                    _sam.ws.close()
                                    logger.info(f"🛑 SAM WebSocket closed to unblock executor thread")
                                except Exception as e:
                                    logger.warning(f"SAM ws close on new audio: {e}")

                            # 3. Cancel the asyncio task
                            processing_task.cancel()
                            try:
                                await asyncio.wait_for(asyncio.shield(processing_task), timeout=0.4)
                            except (asyncio.CancelledError, asyncio.TimeoutError):
                                logger.info(f"✅ Previous task cancelled for new audio")
                            except Exception as e:
                                logger.error(f"Task cancel error: {e}")

                            # 4. Tell frontend to stop playing current audio immediately
                            try:
                                await websocket.send_json({
                                    "type": "immediate_stop",
                                    "reason": "new_audio_received",
                                    "stop_audio": True,
                                    "stop_animations": True,
                                    "reset_face": True,
                                    "clear_queue": True,
                                    "flush_pending": True,
                                    "ready_for_input": True
                                })
                            except Exception as e:
                                logger.error(f"Failed to send stop signal: {e}")

                            # 5. Reconnect SAM if it was closed (so next query works)
                            if not is_groq and sam_client is not None:
                                try:
                                    sam_client.connect()
                                    client_sessions[client_id]['sam_client'] = sam_client
                                    logger.info(f"✅ SAM WebSocket reconnected after interruption")
                                except Exception as e:
                                    logger.warning(f"SAM reconnect failed: {e} — will use Groq")
                                    is_groq = True

                        # ─────────────────────────────────────────────────────────────────────
                        # SAM PATH
                        # ─────────────────────────────────────────────────────────────────────
                        if not is_groq and sam_client is not None:
                            logger.info(f"🤖 Using SAM for {client_id}")
                            try:
                                # Clear interruption state FRESH before new interaction
                                await interruption_manager.clear_interruption(client_id)

                                final_text, emotion, df_energy, is_complete = await EnhancedAudioProcessorWithOptimizedASR().process_audio_stream(
                                    client_id=client_id,
                                    audio_data_base64=audio_base64
                                )
                                if not final_text:
                                    continue

                                interaction_id = f"sam_{int(time.time()*1000)}"

                                async def sam_query_and_stream():
                                    try:
                                        client_sessions[client_id]['current_interaction_id'] = interaction_id

                                        if await interruption_manager.check_interrupted(client_id, interaction_id):
                                            logger.info(f"🛑 SAM task cancelled before start")
                                            return

                                        logger.info(f"Starting SAM query: '{final_text[:50]}...'")

                                        loop = asyncio.get_event_loop()

                                        query_future = loop.run_in_executor(
                                            None,
                                            lambda: sam_client.send_query(query=final_text, timeout=30.0)
                                        )

                                        # Poll for interruption while SAM blocking call runs
                                        while not query_future.done():
                                            if await interruption_manager.check_interrupted(client_id, interaction_id):
                                                logger.info(f"🛑 SAM interrupted during query — closing socket")
                                                try:
                                                    if sam_client.ws:
                                                        sam_client.ws.close()
                                                except Exception:
                                                    pass
                                                query_future.cancel()
                                                return
                                            await asyncio.sleep(0.05)

                                        try:
                                            full_response = await query_future
                                        except Exception as e:
                                            logger.error(f"SAM query failed: {e}")
                                            full_response = None

                                        if not full_response:
                                            return

                                        if await interruption_manager.check_interrupted(client_id, interaction_id):
                                            logger.info(f"🛑 SAM interrupted after query")
                                            return

                                        logger.info(f"✅ SAM response ({len(full_response)} chars)")

                                        sentences = re.split(r'(?<=[.!?])\s+', full_response.strip())
                                        sentences = [s for s in sentences if s.strip()] or [full_response]

                                        for idx, sentence in enumerate(sentences):
                                            # Check BEFORE each sentence
                                            if await interruption_manager.check_interrupted(client_id, interaction_id):
                                                logger.info(f"🛑 SAM streaming interrupted at sentence {idx+1}")
                                                break

                                            is_last = (idx == len(sentences) - 1)

                                            await stream_response_audio_with_emotion_REALTIME(
                                                client_id=client_id,
                                                phrase=sentence.strip(),
                                                emotion=emotion,
                                                interaction_id=interaction_id,
                                                phrase_index=idx + 1,
                                                is_last=is_last,
                                                audio_type=client_sessions[client_id].get('audio_type', 'girl'),
                                                intensity=1.0,
                                                facial_state="neutral"
                                            )

                                            if not is_last:
                                                for _ in range(5):  # 5 × 100ms = 500ms gap
                                                    if await interruption_manager.check_interrupted(client_id, interaction_id):
                                                        logger.info(f"🛑 Interrupted during sentence gap")
                                                        return
                                                    await asyncio.sleep(0.1)

                                        if not await interruption_manager.check_interrupted(client_id, interaction_id):
                                            await send_to_client(client_id, {
                                                "type": "completion",
                                                "status": "done",
                                                "interaction_id": interaction_id,
                                                "response_from": "SAM llm"
                                            })

                                    except asyncio.CancelledError:
                                        logger.info(f"🛑 SAM task CancelledError")
                                        raise
                                    except Exception as e:
                                        logger.error(f"❌ SAM task error: {e}")
                                        traceback.print_exc()
                                    finally:
                                        if client_id in client_sessions:
                                            if client_sessions[client_id].get('current_interaction_id') == interaction_id:
                                                client_sessions[client_id]['current_interaction_id'] = None

                                # Clear state and start fresh interaction
                                await interruption_manager.clear_interruption(client_id)
                                await interruption_manager.start_interaction(client_id, interaction_id)

                                task = asyncio.create_task(sam_query_and_stream())
                                client_sessions[client_id]['processing_task'] = task

                            except Exception as e:
                                logger.error(f"❌ SAM processing failed: {e} — falling back to Groq")
                                is_groq = True
                                await interruption_manager.clear_interruption(client_id)

                        # ─────────────────────────────────────────────────────────────────────
                        # GROQ PATH
                        # ─────────────────────────────────────────────────────────────────────
                        if is_groq:
                            logger.info(f"🟢 Using Groq for {client_id}")

                            await interruption_manager.clear_interruption(client_id)

                            result = await integrate_s2s_with_websocket(
                                client_id=client_id,
                                audio_data_base64=audio_base64,
                                audio_type=audio_type,
                                client_sessions=client_sessions,
                                audio_processor=audio_processor,
                                tts_pool=tts_pool,
                                generate_llm_response_streaming=generate_llm_response_streaming_sam_with_fallback,
                                stream_response_audio_with_emotion_REALTIME=stream_response_audio_with_emotion_REALTIME,
                                interruption_manager=interruption_manager
                            )
                            if result['status'] == 'success':
                                logger.info(f"✅ [S2S] Complete | User: '{result['transcription'][:50]}...' | Response: {result['phrase_count']} phrases | Emotion: {result['emotion']}")
                                await websocket.send_json({
                                    "type": "s2s_complete",
                                    "transcription": result['transcription'],
                                    "emotion": result['emotion'],
                                    "phrase_count": result['phrase_count'],
                                    "timestamp": time.time()
                                })
                            elif result['status'] == 'no_speech':
                                logger.debug(f"ℹ️ [S2S] No speech detected")
                            else:
                                logger.error(f"❌ [S2S] Failed: {result.get('error')}")
                                await send_log_to_frontend(client_id, "error", f"❌ S2S processing failed", "error")
                                
            except Exception as e:
                logger.error(f"Message handling error: {e}")
                await send_log_to_frontend(client_id, "error", f" Error: {str(e)[:100]}", "error")
                import traceback
                traceback.print_exc()

    except WebSocketDisconnect:
        logger.info(f"🔌 Disconnected: {client_id}")
    except Exception as e:
        logger.error(f" WebSocket error for {client_id}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        remove_client_token(client_id)
        if client_id in client_token_status:    del client_token_status[client_id]
        if client_id in client_token_storage:   del client_token_storage[client_id]
        if client_id in client_llm_status:      del client_llm_status[client_id]
        if client_id in client_session_ids:     del client_session_ids[client_id]

        if client_id in client_sessions:
            processing_task = client_sessions[client_id].get('processing_task')
            if processing_task and not processing_task.done():
                processing_task.cancel()
            token_task = client_sessions[client_id].get('token_task')
            if token_task and not token_task.done():
                token_task.cancel()
            _sam = client_sessions[client_id].get('sam_client')
            if _sam:
                try:
                    _sam.close()
                except Exception as e:
                    logger.error(f"SAM client cleanup error: {e}")
            del client_sessions[client_id]

        if heartbeat_task:
            await stop_websocket_heartbeat(heartbeat_task, client_id)
            
        if client_id in sam_clients:
            try:
                sam_clients[client_id].close()
                del sam_clients[client_id]
            except Exception as e:
                logger.error(f"SAM client cleanup error: {e}")

        if client_id in client_audio_timings:   del client_audio_timings[client_id]
        if client_id in speech_controllers:     del speech_controllers[client_id]
        if client_id in emotion_state_managers: del emotion_state_managers[client_id]

        stats["connections_active"] = len(client_sessions)
        logger.info(f" Cleanup complete for {client_id}")
        
        
"""  with client llm working  """