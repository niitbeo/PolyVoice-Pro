"""
VOICERECORE Local Bidirectional Real-Time Speech Translation Server (v0.7.0 — Production Ready).

Supports:
- 3-Tier Smart Offline Translation:
    1) Tencent Hy-MT2-1.8B (GGUF Q4_K_M, Apache 2.0, 38 languages incl. VI/EN/TH/KM/ZH/JA/KO/FR/DE)
       with native Terminology Intervention + Context-Aware Background
    2) VietAI EnViT5 INT8 (266 MB, OpenRAIL, ultra-fast VI <-> EN)
    3) Meta NLLB-200-600M (default: 2x faster, 750 MB lighter, highest Lao fidelity) & 1.3B INT8 (202 languages, lazy-loaded)
- 2-Pass Vietnamese ASR (+BPE Glossary Hotwords + Keyword Preservation Guard) & 2-Pass English ASR (Moonshine-v2-Base 106 MB, 1.82% TED WER)
- Audio Modes: Micro (`mic`), YouTube/System (`system`: macOS ScreenCaptureKit + Windows WASAPI Loopback), and 2-Way Meeting Dual Audio (`dual`: Mic + System!)
- Conference Suite: QR Code LAN Audience View (`/audience`, multi-language fan-out),
  OBS Transparent Subtitles (`/obs`), Batch File Translator (`.mp3/.wav/.mp4 -> .SRT`), & AI Meeting Summary
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
import queue
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

# Windows PyInstaller `--noconsole` & SSL CA bootstrap:
# In frozen GUI builds on Windows, sys.stdout/sys.stderr can be None or cp1252, causing
# print/logging/tqdm to crash with AttributeError or UnicodeEncodeError. Also ensure
# bundled certifi CA roots are registered before any HTTPS calls.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
elif hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")
elif hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
try:
    import certifi as _certifi

    _ca_bundle = _certifi.where()
    if _ca_bundle and os.path.isfile(_ca_bundle):
        os.environ.setdefault("SSL_CERT_FILE", _ca_bundle)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", _ca_bundle)
except Exception:
    pass

import numpy as np
import segno
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from asr_engine import (
    ASROutput,
    MoonshineV2EnRefiner,
    VietASRBatchRefiner,
    VietASRStreamingEngine,
    ZipformerStreamingEnEngine,
    ZipformerStreamingViEngine,
    restore_vietnamese_casing,
)
from audience_web import AUDIENCE_HTML, OBS_HTML, get_local_lan_ip
from audio_capture import (
    SAMPLE_RATE,
    MacSystemAudioStream,
    MicrophoneStream,
    MixedAudioStream,
    WindowsLoopbackStream,
    find_macos_helper,
    has_windows_loopback,
    list_input_devices,
    list_output_loopback_devices,
)
from file_translator import FileTranslateCancelled, translate_media_file
from glossary import GlossaryManager
from languages import (
    get_hymt2_lang_name,
    get_language,
    get_source_language,
    list_languages,
    list_source_languages,
)
from model_manager import ModelManager
from quota_tracker import CloudQuotaManager
from session_logger import SessionLogger
from translator import (
    AzureTranslator,
    EnViT5Translator,
    FastOnlineTranslator,
    GeminiCloudTranslator,
    GoogleCloudTranslator,
    HyMT2Translator,
    LocalLLMTranslator,
    NLLBTranslator,
    OpusMTTranslator,
    TranslationResult,
    find_llama_server,
    summarize_meeting_entries,
)

logger = logging.getLogger("voicerecore.server")


def _adaptive_mt_threads() -> int:
    return max(2, min(4, (os.cpu_count() or 4) // 2))


def _drain_stderr_async(proc: Optional[subprocess.Popen], label: str) -> None:
    """Drain subprocess stderr in a daemon thread so the 64 KB pipe buffer never deadlocks."""
    if proc is None or proc.stderr is None:
        return
    pipe = proc.stderr

    def _reader() -> None:
        try:
            for _ in pipe:
                pass
        except Exception:
            pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    threading.Thread(target=_reader, daemon=True, name=f"stderr-drain-{label}").start()


@dataclass
class SessionConfig:
    device_id: Optional[int] = None
    loopback_device_id: Optional[int] = None
    audio_source: str = "mic"          # "mic" | "system" | "dual" | "mix"
    input_channel: str = "mix"         # "mix" | "left" | "right"
    input_gain_db: float = 0.0         # -12.0 .. +24.0 dB
    auto_gain: bool = False            # Broadcast AGC + soft limiter
    source_lang: str = "vie_Latn"
    target_lang: str = "eng_Latn"
    asr_engine: str = "auto"           # "auto" | "apple-native" | "zipformer-30m-streaming" | "zipformer-en-20m" | "viet-asr"
    mt_mode: str = "auto-offline"      # "auto-offline" | "hy-mt2" | "opus" | "nllb" | "fast-online" | "local-llm"
    paragraph_mode: str = "long"       # "short" | "medium" | "long"
    translate_partials: bool = True
    pass2_refine: bool = True          # 2-Pass Vietnamese refinement (punctuation + ITN + error fix)
    eco_mode: bool = False             # Low-CPU mode for 2-3h seminars (relaxed partial MT debounce)
    beam_size: int = 1
    llm_url: str = "http://127.0.0.1:11434/v1"
    llm_model: str = "translategemma:4b"
    llm_api_key: str = "local"


class StartSessionRequest(BaseModel):
    device_id: Optional[int] = None
    loopback_device_id: Optional[int] = None
    audio_source: str = "mic"
    input_channel: str = "mix"
    input_gain_db: float = 0.0
    auto_gain: bool = False
    source_lang: str = "vie_Latn"
    target_lang: str = "eng_Latn"
    asr_engine: str = "auto"
    mt_mode: str = "auto-offline"
    paragraph_mode: str = "long"
    translate_partials: bool = True
    pass2_refine: bool = True
    eco_mode: bool = False
    beam_size: int = 1
    llm_url: str = "http://127.0.0.1:11434/v1"
    llm_model: str = "translategemma:4b"
    llm_api_key: str = "local"


class UpdateConfigRequest(BaseModel):
    device_id: Optional[int] = None
    loopback_device_id: Optional[int] = None
    audio_source: Optional[str] = None
    input_channel: Optional[str] = None
    input_gain_db: Optional[float] = None
    auto_gain: Optional[bool] = None
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None
    asr_engine: Optional[str] = None
    mt_mode: Optional[str] = None
    paragraph_mode: Optional[str] = None
    translate_partials: Optional[bool] = None
    pass2_refine: Optional[bool] = None
    eco_mode: Optional[bool] = None
    beam_size: Optional[int] = None
    llm_url: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None


class GlossaryUpdateRequest(BaseModel):
    enabled: Optional[bool] = None
    asr_fixes: Optional[list[dict[str, str]]] = None
    term_locks: Optional[list[dict[str, str]]] = None


class TranslateTextRequest(BaseModel):
    text: str
    source_lang: str = "vie_Latn"
    target_lang: str = "eng_Latn"
    mt_mode: str = "auto-offline"
    beam_size: int = 1
    llm_url: str = "http://127.0.0.1:11434/v1"
    llm_model: str = "translategemma:4b"
    llm_api_key: str = "local"


class DemoRequest(BaseModel):
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None
    text: Optional[str] = None


class CloudConfigUpdateRequest(BaseModel):
    provider: Optional[str] = None           # "google" | "azure" | "gemini"
    target_provider: Optional[str] = None
    api_key: Optional[str] = None
    region: Optional[str] = None
    endpoint: Optional[str] = None
    model: Optional[str] = None
    monthly_limit: Optional[int] = None
    hard_cap_enabled: Optional[bool] = None
    draft_offline: Optional[bool] = None


class CloudTestRequest(BaseModel):
    provider: Optional[str] = None
    api_key: Optional[str] = None
    region: Optional[str] = None
    endpoint: Optional[str] = None
    model: Optional[str] = None
    target_lang: str = "eng_Latn"


class CloudQuotaResetRequest(BaseModel):
    provider: Optional[str] = None


_NON_SPACED_SCRIPT_PREFIXES = ("tha_", "lao_", "khm_", "zho_", "jpn_", "mya_")

# Legitimate Vietnamese reduplicative words (từ láy / lặp nghĩa) that Clean Verbatim must preserve
_VI_REDUPLICATIVE_KEEP = frozenset({
    "từng", "ngày", "nhà", "người", "rất", "xa", "gần", "dần", "nhanh", "chậm",
    "khác", "lần", "mãi", "luôn", "đời", "lớp", "tầng", "nơi", "khắp", "trùng",
    "điệp", "sâu", "rõ", "đều", "lớn", "nhỏ", "nhẹ", "khẽ", "từ", "lâu", "cao",
    "thấp", "sáng", "tối", "xanh", "đỏ", "trắng", "đen", "nóng", "lạnh", "vui",
    "buồn", "khỏe", "mạnh", "thật", "lắm", "quá", "vừa", "sát", "chặt", "kỹ",
    "gọn", "đẹp", "tốt", "hay", "đúng", "sai", "đủ", "nhiều", "ít", "sớm",
    "muộn", "trước", "sau", "trên", "dưới", "trong", "ngoài", "quanh", "đi",
    "lại", "lên", "xuống", "ra", "vào", "qua", "về",
})


class StablePrefixTracker:
    """
    Conference Local Agreement-2 monotonic prefix stabilizer for live streaming MT.

    Locks the longest common prefix agreed across consecutive draft translations so the
    displayed translation grows smoothly left-to-right (bold locked prefix + soft tentative tail)
    without flickering or rewriting already-read words.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seg_id: str = ""
        self._target_lang: str = ""
        self._locked_tokens: list[str] = []
        self._prev_tokens: list[str] = []
        self._locked_chars: str = ""
        self._prev_chars: str = ""

    @staticmethod
    def _norm_tok(tok: str) -> str:
        return re.sub(r"^[^\w]+|[^\w]+$", "", tok.lower())

    def reset(self) -> None:
        with self._lock:
            self._seg_id = ""
            self._target_lang = ""
            self._locked_tokens = []
            self._prev_tokens = []
            self._locked_chars = ""
            self._prev_chars = ""

    def clear_if(self, seg_id: str) -> None:
        with self._lock:
            if seg_id and seg_id == self._seg_id:
                self._seg_id = ""
                self._target_lang = ""
                self._locked_tokens = []
                self._prev_tokens = []
                self._locked_chars = ""
                self._prev_chars = ""

    def update(self, seg_id: str, target_lang: str, text: str) -> tuple[str, str, str]:
        """Return `(stable_text, unstable_text, full_stabilized_text)`."""
        clean = " ".join((text or "").strip().split())
        if not clean:
            return ("", "", "")

        with self._lock:
            if seg_id != self._seg_id or target_lang != self._target_lang:
                self._seg_id = seg_id
                self._target_lang = target_lang
                self._locked_tokens = []
                self._prev_tokens = []
                self._locked_chars = ""
                self._prev_chars = ""

            # Character-level agreement for non-space-delimited scripts (Thai/Lao/CJK/Khmer/Burmese)
            is_non_spaced = (
                target_lang.startswith(_NON_SPACED_SCRIPT_PREFIXES)
                and len(clean.split()) <= 2
                and len(clean) >= 8
            )
            if is_non_spaced:
                if not self._prev_chars:
                    self._prev_chars = clean
                    return ("", clean, clean)
                lcp = 0
                lim = min(len(self._prev_chars), len(clean))
                while lcp < lim and self._prev_chars[lcp] == clean[lcp]:
                    lcp += 1
                cand = max(0, lcp - 3) if lcp == len(clean) else lcp
                if self._locked_chars:
                    if clean.startswith(self._locked_chars):
                        if cand > len(self._locked_chars):
                            self._locked_chars = clean[:cand]
                        unstable_c = clean[len(self._locked_chars) :]
                    else:
                        # Check character overlap before breaking monotonic lock
                        match_c = 0
                        mlim = min(len(self._locked_chars), len(clean))
                        while match_c < mlim and self._locked_chars[match_c] == clean[match_c]:
                            match_c += 1
                        if match_c < len(self._locked_chars) * 0.4 and len(clean) >= len(self._locked_chars) + 10:
                            self._locked_chars = clean[:cand]
                            unstable_c = clean[len(self._locked_chars) :]
                        else:
                            unstable_c = clean[len(self._locked_chars) :]
                else:
                    self._locked_chars = clean[:cand]
                    unstable_c = clean[cand:]
                self._prev_chars = clean
                full_c = self._locked_chars + unstable_c
                return (self._locked_chars, unstable_c, full_c)

            # Word-level Local Agreement-2 for space-delimited languages (EN, VI, FR, DE, KO, etc.)
            new_tokens = clean.split()
            if not self._prev_tokens:
                self._prev_tokens = new_tokens
                return ("", clean, clean)

            prev_norms = [self._norm_tok(t) for t in self._prev_tokens]
            new_norms = [self._norm_tok(t) for t in new_tokens]

            lcp = 0
            lim = min(len(prev_norms), len(new_norms))
            while lcp < lim and prev_norms[lcp] == new_norms[lcp]:
                lcp += 1

            # Keep final token unlocked if no new tokens arrived yet so punctuation/tense can settle
            cand_lock = max(0, lcp - 1) if lcp == len(new_tokens) else lcp

            if self._locked_tokens:
                locked_norms = [self._norm_tok(t) for t in self._locked_tokens]
                pos_match = 0
                mlim = min(len(locked_norms), len(new_norms))
                while pos_match < mlim and locked_norms[pos_match] == new_norms[pos_match]:
                    pos_match += 1

                if pos_match == len(self._locked_tokens):
                    if cand_lock > len(self._locked_tokens):
                        self._locked_tokens = list(self._locked_tokens) + new_tokens[len(self._locked_tokens) : cand_lock]
                    unstable_tokens = new_tokens[len(self._locked_tokens) :]
                else:
                    window = set(new_norms[: len(locked_norms) + 3])
                    overlap = sum(1 for t in locked_norms if t and t in window) / max(len(locked_norms), 1)
                    if overlap < 0.30 and len(new_tokens) >= len(self._locked_tokens) + 5:
                        # Major semantic re-parse (e.g., sentence-final negation) — clean re-anchor
                        self._locked_tokens = new_tokens[:cand_lock]
                        unstable_tokens = new_tokens[cand_lock:]
                    else:
                        # Keep monotonic lock; align tail cleanly after last locked anchor word
                        anchor_pos = -1
                        lo = max(0, len(self._locked_tokens) - 3)
                        hi = min(len(new_norms), len(self._locked_tokens) + 4)
                        for back in range(1, min(3, len(locked_norms)) + 1):
                            needle = locked_norms[-back]
                            if not needle:
                                continue
                            for j in range(hi - 1, lo - 1, -1):
                                if new_norms[j] == needle:
                                    anchor_pos = j + back
                                    break
                            if anchor_pos >= 0:
                                break
                        tail_start = anchor_pos if anchor_pos >= 0 else len(self._locked_tokens)
                        unstable_tokens = new_tokens[tail_start:] if tail_start < len(new_tokens) else []
            else:
                self._locked_tokens = new_tokens[:cand_lock]
                unstable_tokens = new_tokens[cand_lock:]

            # Prevent any duplicate seam word where locked prefix meets aligned unstable tail
            while (
                self._locked_tokens
                and unstable_tokens
                and self._norm_tok(self._locked_tokens[-1])
                and self._norm_tok(self._locked_tokens[-1]) == self._norm_tok(unstable_tokens[0])
            ):
                unstable_tokens = unstable_tokens[1:]

            self._prev_tokens = new_tokens
            stable_str = " ".join(self._locked_tokens).strip()
            unstable_str = " ".join(unstable_tokens).strip()
            if stable_str and unstable_str:
                full_str = f"{stable_str} {unstable_str}"
            else:
                full_str = stable_str or unstable_str
            return (stable_str, unstable_str, full_str)


class EngineRuntime:
    def __init__(self) -> None:
        self.model_manager = ModelManager()
        self.glossary = GlossaryManager()
        self.cloud_quota = CloudQuotaManager()
        self.logger = SessionLogger()
        self.config = SessionConfig()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.mac_helper = find_macos_helper()
        self._llama_server_ready: bool = find_llama_server() is not None
        self.server_port: int = 18765
        self.apple_caps: dict[str, Any] = {
            "available": False,
            "on_device_locales": [],
            "system_audio": False,
        }

        self._ws_clients: set[WebSocket] = set()
        self._audience_ws: dict[WebSocket, str] = {}
        self._ws_lock = asyncio.Lock()

        self._asr_zf_vi: Optional[ZipformerStreamingViEngine] = None
        self._asr_zf_en: Optional[ZipformerStreamingEnEngine] = None
        self._asr_viet: Optional[VietASRStreamingEngine] = None
        self._pass2_refiner: Optional[VietASRBatchRefiner] = None
        self._en_pass2_refiner: Optional[MoonshineV2EnRefiner] = None
        self._hymt2: Optional[HyMT2Translator] = None
        self._envit5: Optional[EnViT5Translator] = None
        self._opus_ve: Optional[OpusMTTranslator] = None
        self._opus_ev: Optional[OpusMTTranslator] = None
        self._nllb_13b: Optional[NLLBTranslator] = None
        self._nllb: Optional[NLLBTranslator] = None
        self._fast_online = FastOnlineTranslator()
        self._google_cloud: Optional[GoogleCloudTranslator] = None
        self._azure_mt: Optional[AzureTranslator] = None
        self._gemini_mt: Optional[GeminiCloudTranslator] = None
        self._llm: Optional[LocalLLMTranslator] = None
        self._model_lock = threading.Lock()

        self._audio_stream: Any = None
        self._remote_audio_stream: Any = None
        self._apple_proc: Optional[subprocess.Popen] = None
        self._remote_apple_proc: Optional[subprocess.Popen] = None
        self._asr_thread: Optional[threading.Thread] = None
        self._remote_asr_thread: Optional[threading.Thread] = None
        self._mt_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._listening = False
        self._active_asr_label: str = ""
        self._remote_asr_label: str = ""

        self._final_mt_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._partial_lock = threading.Lock()
        self._pending_partial: Optional[dict[str, Any]] = None
        self._stable_prefix = StablePrefixTracker()
        self._mt_wakeup = threading.Event()

        # Primary channel semantic sentence buffer + PCM ring buffer for Pass-2 Vietnamese refinement
        self._para_lock = threading.Lock()
        self._para_parts: list[str] = []
        self._para_pcm_chunks: list[np.ndarray] = []
        self._para_pcm_truncated: bool = False
        self._para_has_carry_prefix: bool = False
        self._para_has_active_partial: bool = False
        self._para_last_speech_ts: float = 0.0
        self._para_last_asr_ms: float = 0.0
        self._para_src_lang: str = "vie_Latn"
        self._para_tgt_lang: str = "eng_Latn"

        # Dual-Audio remote channel buffer (System Audio partner turn in `audio_source="dual"`)
        self._remote_lock = threading.Lock()
        self._remote_parts: list[str] = []
        self._remote_pcm_chunks: list[np.ndarray] = []
        self._remote_has_active_partial: bool = False
        self._remote_last_speech_ts: float = 0.0
        self._remote_last_asr_ms: float = 0.0
        self._remote_seg_id: str = ""

        # Visual Paragraph Grouping (YouTube-style decoupling: translate by sentence, display by paragraph)
        self._group_lock = threading.Lock()
        self._active_group_id: str = ""
        self._active_group_words: int = 0
        self._active_group_last_ts: float = 0.0
        self._active_group_langs: tuple[str, str] = ("", "")
        self._active_group_speaker: str = ""
        self._recent_context: deque[tuple[str, str]] = deque(maxlen=2)

        self._current_segment_id: str = ""
        self._last_level_emit = 0.0
        self._history: list[dict[str, Any]] = []
        self._boot_error: str = ""
        self._file_job_progress: dict[str, Any] = {"active": False}
        self._file_job_cancel: Optional[threading.Event] = None
        self._file_job_lock = threading.Lock()
        self._file_job_busy: bool = False
        self._active_downloads: set[str] = set()
        self._downloads_lock = threading.Lock()
        self._fanout_queue: queue.Queue[Optional[dict[str, Any]]] = queue.Queue(maxsize=32)
        self._fanout_lock = threading.Lock()
        self._fanout_thread: Optional[threading.Thread] = None
        self._fanout_stop = threading.Event()
        self._uvicorn_server: Optional[uvicorn.Server] = None
        self._cloud_cooldown_until: float = 0.0
        self._cloud_fallback_reason: str = ""
        self._cloud_notice_lock = threading.Lock()

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def broadcast_threadsafe(self, payload: dict[str, Any]) -> None:
        if self.loop is None or self.loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self.broadcast(payload), self.loop)

    async def register_ws(self, ws: WebSocket) -> None:
        async with self._ws_lock:
            self._ws_clients.add(ws)
        await ws.send_json({"type": "status", "data": self.get_status_payload()})
        if self._history:
            await ws.send_json({"type": "history", "data": self._history[-100:]})

    async def unregister_ws(self, ws: WebSocket) -> None:
        async with self._ws_lock:
            self._ws_clients.discard(ws)

    def get_audience_status_payload(self) -> dict[str, Any]:
        """Minimal, sanitized status for untrusted LAN `/ws/audience` clients (no API config or host paths)."""
        src = get_source_language(self.config.source_lang)
        tgt = get_language(self.config.target_lang)
        return {
            "listening": self._listening,
            "source_lang": self.config.source_lang,
            "target_lang": self.config.target_lang,
            "source_lang_meta": asdict(src) if src else None,
            "target_lang_meta": asdict(tgt) if tgt else None,
        }

    async def register_audience_ws(self, ws: WebSocket, lang: str = "follow") -> None:
        req_lang = lang.strip() if lang and lang.strip() else "follow"
        async with self._ws_lock:
            self._audience_ws[ws] = req_lang
        await ws.send_json({"type": "status", "data": self.get_audience_status_payload()})
        # Send recent history (capped at last 8 entries to avoid MT lock starvation on connect)
        recent = list(self._history)[-8:]
        if recent:
            if req_lang == "follow":
                await ws.send_json({"type": "history", "data": recent})
            else:
                customized = []
                for e in recent:
                    if e.get("target_lang") == req_lang:
                        customized.append(e)
                    else:
                        try:
                            tr = await asyncio.to_thread(
                                self.translate_text,
                                e["source_text"],
                                e["source_lang"],
                                req_lang,
                                self.config.mt_mode,
                                1,
                            )
                            customized.append({**e, "target_lang": req_lang, "translated_text": tr.translated_text})
                        except Exception as exc:
                            logger.warning("Audience history backfill failed for %s: %s", req_lang, exc)
                            customized.append(e)
                await ws.send_json({"type": "history", "data": customized})
        await self.broadcast({"type": "audience_count", "data": self.get_audience_summary()})

    async def unregister_audience_ws(self, ws: WebSocket) -> None:
        async with self._ws_lock:
            self._audience_ws.pop(ws, None)
        await self.broadcast({"type": "audience_count", "data": self.get_audience_summary()})

    def get_audience_summary(self) -> dict[str, Any]:
        langs = sorted({l for l in self._audience_ws.values() if l != "follow"})
        return {
            "connected_count": len(self._audience_ws),
            "active_languages": langs,
        }

    def get_audience_info(self) -> dict[str, Any]:
        lan_ip = get_local_lan_ip()
        audience_url = f"http://{lan_ip}:{self.server_port}/audience"
        obs_url = f"http://127.0.0.1:{self.server_port}/obs"
        qr = segno.make(audience_url)
        qr_svg = qr.svg_inline(omitsize=True, border=0, dark="#0f172a", light="#ffffff")
        return {
            "lan_ip": lan_ip,
            "port": self.server_port,
            "audience_url": audience_url,
            "obs_url": obs_url,
            "qr_svg": qr_svg,
            **self.get_audience_summary(),
        }

    async def broadcast(self, payload: dict[str, Any]) -> None:
        async with self._ws_lock:
            clients = list(self._ws_clients)
            aud_items = list(self._audience_ws.items())
        dead: list[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)

        ev_type = payload.get("type")
        ev_data = payload.get("data")
        aud_status_payload = (
            {"type": "status", "data": self.get_audience_status_payload()}
            if ev_type == "status" and aud_items
            else None
        )
        dead_aud: list[WebSocket] = []
        for aws, alang in aud_items:
            try:
                if ev_type == "status" and aud_status_payload is not None:
                    await aws.send_json(aud_status_payload)
                elif ev_type == "partial":
                    await aws.send_json(payload)
                elif ev_type == "history" and isinstance(ev_data, list) and len(ev_data) == 0:
                    await aws.send_json(payload)
                elif ev_type in ("partial_mt", "final") and isinstance(ev_data, dict):
                    if alang == "follow" or alang == ev_data.get("target_lang"):
                        await aws.send_json(payload)
            except Exception:
                dead_aud.append(aws)

        if dead or dead_aud:
            async with self._ws_lock:
                for ws in dead:
                    self._ws_clients.discard(ws)
                for aws in dead_aud:
                    self._audience_ws.pop(aws, None)

    def _ensure_fanout_worker(self) -> None:
        with self._fanout_lock:
            if self._fanout_thread is None or not self._fanout_thread.is_alive():
                self._fanout_stop.clear()
                self._fanout_thread = threading.Thread(
                    target=self._fanout_worker_loop,
                    daemon=True,
                    name="audience-fanout-worker",
                )
                self._fanout_thread.start()

    def _fanout_worker_loop(self) -> None:
        while not self._fanout_stop.is_set():
            try:
                entry = self._fanout_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if entry is None:
                break
            if not self._audience_ws or self.loop is None or self.loop.is_closed():
                continue
            extra_langs = sorted(
                {
                    lang
                    for lang in list(self._audience_ws.values())
                    if lang != "follow" and lang != entry.get("target_lang")
                }
            )
            for tgt in extra_langs:
                if self._fanout_stop.is_set():
                    break
                try:
                    tr = self.translate_text(
                        entry["source_text"],
                        entry["source_lang"],
                        tgt,
                        mt_mode=self.config.mt_mode,
                        beam_size=1,
                        llm_url=self.config.llm_url,
                        llm_model=self.config.llm_model,
                        llm_api_key=self.config.llm_api_key,
                    )
                    custom_entry = {
                        **entry,
                        "target_lang": tgt,
                        "translated_text": tr.translated_text,
                        "mt_ms": tr.latency_ms,
                        "mt_engine": tr.engine,
                    }

                    async def _push(target_code=tgt, payload={"type": "final", "data": custom_entry}) -> None:
                        async with self._ws_lock:
                            targets = [ws for ws, l in self._audience_ws.items() if l == target_code]
                        for ws in targets:
                            try:
                                await ws.send_json(payload)
                            except Exception:
                                pass

                    if self.loop and not self.loop.is_closed():
                        asyncio.run_coroutine_threadsafe(_push(), self.loop)
                except Exception as exc:
                    logger.warning("Audience fan-out translation failed for %s: %s", tgt, exc)

    def _fanout_audience_final(self, entry: dict[str, Any]) -> None:
        """Enqueue a committed sentence onto the bounded FIFO audience fan-out worker."""
        if not self._audience_ws or self.loop is None or self.loop.is_closed():
            return
        extra_langs = {
            lang
            for lang in list(self._audience_ws.values())
            if lang != "follow" and lang != entry.get("target_lang")
        }
        if not extra_langs:
            return
        self._ensure_fanout_worker()
        try:
            self._fanout_queue.put_nowait(dict(entry))
        except queue.Full:
            try:
                self._fanout_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._fanout_queue.put_nowait(dict(entry))
            except queue.Full:
                pass

    def _probe_mac_helper(self) -> None:
        if self.mac_helper is None:
            return
        try:
            proc = subprocess.run(
                [str(self.mac_helper), "--mode", "check"],
                capture_output=True,
                text=True,
                timeout=6.0,
            )
            for line in reversed(proc.stdout.splitlines()):
                line = line.strip()
                if line.startswith("{"):
                    data = json.loads(line)
                    if data.get("type") == "capabilities":
                        self.apple_caps = {
                            "available": True,
                            "on_device_locales": data.get("on_device_locales", []),
                            "system_audio": bool(data.get("system_audio", True)),
                        }
                        break
        except Exception:
            pass

    def warmup_background(self) -> None:
        def _run() -> None:
            try:
                self._probe_mac_helper()
                if self.model_manager.is_zipformer_streaming_ready():
                    self.ensure_onnx_asr_loaded("zipformer-30m-streaming")
                    if self.config.pass2_refine:
                        self.ensure_pass2_refiner()
                else:
                    self.ensure_onnx_asr_loaded("viet-asr")
                # Lazy-load strategy: warm up primary VI<->EN model (EnViT5) only.
                # Do NOT preload 615 MB NLLB-200 or unused Opus models at startup (saves ~800 MB RAM!).
                if self.model_manager.is_envit5_ready():
                    self.ensure_envit5_loaded()
                else:
                    if self.model_manager.is_opus_vi_en_ready():
                        self.ensure_opus_loaded("vi-en")
                    if self.model_manager.is_opus_en_vi_ready():
                        self.ensure_opus_loaded("en-vi")
                self.broadcast_threadsafe({"type": "status", "data": self.get_status_payload()})
            except Exception as exc:
                self._boot_error = str(exc)
                self.broadcast_threadsafe({"type": "error", "data": {"message": f"Khởi động engine gặp lỗi: {exc}"}})

        threading.Thread(target=_run, daemon=True, name="engine-warmup").start()

    def ensure_pass2_refiner(self) -> VietASRBatchRefiner:
        with self._model_lock:
            if self._pass2_refiner is None:
                self._pass2_refiner = VietASRBatchRefiner(provider="cpu", sample_rate=float(SAMPLE_RATE))
            return self._pass2_refiner

    def ensure_en_pass2_refiner(self) -> Optional[MoonshineV2EnRefiner]:
        with self._model_lock:
            if self._en_pass2_refiner is None and self.model_manager.is_moonshine_en_ready():
                self._en_pass2_refiner = MoonshineV2EnRefiner(
                    self.model_manager.moonshine_en_dir,
                    sample_rate=SAMPLE_RATE,
                    num_threads=_adaptive_mt_threads(),
                )
            return self._en_pass2_refiner

    def _get_glossary_hotwords(self) -> list[str]:
        g = self.glossary.get()
        if not g.get("enabled", True):
            return []
        phrases: list[str] = []
        for rule in g.get("term_locks", []):
            if rule.get("source"):
                phrases.append(rule["source"])
        for rule in g.get("asr_fixes", []):
            if rule.get("correct"):
                phrases.append(rule["correct"])
        return phrases

    def sync_asr_hotwords(self) -> int:
        if self._asr_zf_vi is not None:
            return self._asr_zf_vi.update_hotwords(self._get_glossary_hotwords())
        return 0

    def ensure_onnx_asr_loaded(self, engine_name: str):
        with self._model_lock:
            if engine_name == "zipformer-30m-streaming":
                if self._asr_zf_vi is None:
                    if not self.model_manager.is_zipformer_streaming_ready():
                        if self._asr_viet is None:
                            self._asr_viet = VietASRStreamingEngine(provider="cpu", sample_rate=float(SAMPLE_RATE))
                        return self._asr_viet
                    self._asr_zf_vi = ZipformerStreamingViEngine(
                        self.model_manager.zipformer_streaming_dir,
                        hotwords=self._get_glossary_hotwords(),
                    )
                return self._asr_zf_vi

            if engine_name == "zipformer-en-20m":
                if self._asr_zf_en is None:
                    if not self.model_manager.is_zipformer_en_20m_ready():
                        raise RuntimeError("Model tiếng Anh Zipformer-20M chưa được tải. Hãy bấm tải hoặc chọn Apple Neural Engine.")
                    self._asr_zf_en = ZipformerStreamingEnEngine(self.model_manager.zipformer_en_20m_dir)
                return self._asr_zf_en

            if self._asr_viet is None:
                self._asr_viet = VietASRStreamingEngine(provider="cpu", sample_rate=float(SAMPLE_RATE))
            return self._asr_viet

    def ensure_hymt2_loaded(self) -> Optional[HyMT2Translator]:
        with self._model_lock:
            if not self._llama_server_ready or not self.model_manager.is_hymt2_ready():
                return None
            if self._hymt2 is None:
                self._hymt2 = HyMT2Translator(
                    self.model_manager.hy_mt2_gguf_path,
                    port=18766,
                    threads=_adaptive_mt_threads(),
                )
            return self._hymt2

    def ensure_envit5_loaded(self) -> Optional[EnViT5Translator]:
        with self._model_lock:
            if self._envit5 is None and self.model_manager.is_envit5_ready():
                self._envit5 = EnViT5Translator(
                    self.model_manager.envit5_dir,
                    intra_threads=_adaptive_mt_threads(),
                )
            return self._envit5

    def ensure_opus_loaded(self, direction: str) -> Optional[OpusMTTranslator]:
        with self._model_lock:
            if direction == "vi-en":
                if self._opus_ve is None and self.model_manager.is_opus_vi_en_ready():
                    self._opus_ve = OpusMTTranslator(
                        self.model_manager.opus_vi_en_dir, "vie_Latn", "eng_Latn", "opus-mt-vi-en", intra_threads=2
                    )
                return self._opus_ve
            if direction == "en-vi":
                if self._opus_ev is None and self.model_manager.is_opus_en_vi_ready():
                    self._opus_ev = OpusMTTranslator(
                        self.model_manager.opus_en_vi_dir,
                        "eng_Latn",
                        "vie_Latn",
                        "opus-mt-en-vi",
                        src_prefix=">>vie<<",
                        intra_threads=2,
                    )
                return self._opus_ev
            return None

    def ensure_nllb_13b_loaded(self) -> Optional[NLLBTranslator]:
        with self._model_lock:
            if self._nllb_13b is None and self.model_manager.is_nllb_13b_ready():
                self._nllb_13b = NLLBTranslator(
                    self.model_manager.nllb_13b_dir,
                    device="cpu",
                    intra_threads=_adaptive_mt_threads(),
                    engine_name="nllb-200-1.3b-int8",
                )
            return self._nllb_13b

    def ensure_nllb_loaded(self, prefer_13b: bool = False) -> NLLBTranslator:
        """
        Empirical YouTube Audit Result (v0.7):
        NLLB-200-600M is 2x faster (1.22s vs 2.38s), uses 750 MB less RAM (1.21 GB vs 1.95 GB),
        AND scores higher on Lao fidelity (55.47 vs 52.50 BackTrans-chrF) than NLLB-1.3B!
        So default prefers NLLB-600M, while `prefer_13b=True` (explicit `nllb-1.3b` mode) uses 1.3B.
        """
        if prefer_13b:
            n13 = self.ensure_nllb_13b_loaded()
            if n13 is not None:
                return n13
        with self._model_lock:
            if self._nllb is None and self.model_manager.is_nllb_ready():
                self._nllb = NLLBTranslator(
                    self.model_manager.nllb_600m_dir,
                    device="cpu",
                    intra_threads=_adaptive_mt_threads(),
                    engine_name="nllb-200-600m-int8",
                )
            if self._nllb is not None:
                return self._nllb
        n13 = self.ensure_nllb_13b_loaded()
        if n13 is not None:
            return n13
        raise RuntimeError(
            "Chưa tải mô hình dịch Offline trên máy. Hãy bấm 'Tải bộ mô hình lõi' trong mục Mô hình Offline hoặc chuyển sang Cloud API."
        )

    def get_llm_translator(self, url: str, model: str, api_key: str) -> LocalLLMTranslator:
        with self._model_lock:
            if (
                self._llm is None
                or self._llm.base_url != url.rstrip("/")
                or self._llm.model != model
                or self._llm.api_key != api_key
            ):
                self._llm = LocalLLMTranslator(base_url=url, model=model, api_key=api_key)
            return self._llm

    def _get_cloud_translator(self, creds: dict[str, Any]):
        pid = creds["id"]
        key = str(creds.get("api_key", "")).strip()
        with self._model_lock:
            if pid == "google":
                if self._google_cloud is None or self._google_cloud.api_key != key:
                    self._google_cloud = GoogleCloudTranslator(api_key=key)
                return self._google_cloud
            if pid == "azure":
                region = str(creds.get("region", "")).strip()
                endpoint = str(creds.get("endpoint", "")).strip() or "https://api.cognitive.microsofttranslator.com"
                if (
                    self._azure_mt is None
                    or self._azure_mt.api_key != key
                    or self._azure_mt.region != region
                    or self._azure_mt.endpoint != endpoint.rstrip("/")
                ):
                    self._azure_mt = AzureTranslator(api_key=key, region=region, endpoint=endpoint)
                return self._azure_mt
            if pid == "gemini":
                model = str(creds.get("model", "")).strip() or "gemini-2.5-flash"
                if self._gemini_mt is None or self._gemini_mt.api_key != key or self._gemini_mt.model != model:
                    self._gemini_mt = GeminiCloudTranslator(api_key=key, model=model)
                return self._gemini_mt
        raise ValueError(f"Unsupported cloud provider: {pid}")

    def clear_cloud_cooldown(self) -> None:
        changed = False
        with self._cloud_notice_lock:
            if self._cloud_cooldown_until > 0.0 or self._cloud_fallback_reason:
                self._cloud_cooldown_until = 0.0
                self._cloud_fallback_reason = ""
                changed = True
        if changed:
            self.broadcast_threadsafe({"type": "cloud_notice", "data": {"message": "", "cooldown_s": 0}})

    def get_cloud_fallback_notice(self) -> str:
        with self._cloud_notice_lock:
            if self._cloud_cooldown_until > 0.0 and time.time() >= self._cloud_cooldown_until:
                self._cloud_cooldown_until = 0.0
                self._cloud_fallback_reason = ""
            return self._cloud_fallback_reason

    def _record_cloud_error(self, pid: str, exc: Exception) -> None:
        raw = str(exc or "").strip()
        low = raw.lower()
        pname = {"gemini": "Gemini", "azure": "Azure Translator", "google": "Google Cloud"}.get(pid, "Cloud API")
        if "429" in low or "quota" in low or "rate" in low or "resource_exhausted" in low:
            if "free_tier" in low or "per day" in low or "limit: 20" in low:
                cooldown_s = 300
                reason = f"{pname} đã hết 20 lượt miễn phí trong ngày (HTTP 429) — đang tự dịch bằng Offline."
            else:
                cooldown_s = 45
                reason = f"{pname} đang quá tải tạm thời (HTTP 429) — dùng Offline dự phòng 45 giây."
        elif "401" in low or "403" in low or "api_key" in low or "unauthorized" in low or "invalid" in low:
            cooldown_s = 120
            reason = f"{pname} API Key không hợp lệ hoặc bị từ chối — đã tự chuyển sang Offline."
        else:
            cooldown_s = 30
            reason = f"Không kết nối được {pname} — đang tự chuyển mượt sang Offline."

        with self._cloud_notice_lock:
            prev_reason = self._cloud_fallback_reason
            was_cooling = time.time() < self._cloud_cooldown_until
            self._cloud_cooldown_until = time.time() + cooldown_s
            self._cloud_fallback_reason = reason
            should_notify = (not was_cooling) or (prev_reason != reason)

        if should_notify:
            self.broadcast_threadsafe(
                {"type": "cloud_notice", "data": {"message": reason, "cooldown_s": cooldown_s}}
            )

    def test_cloud_connection(self, req: CloudTestRequest) -> dict[str, Any]:
        creds = self.cloud_quota.get_credentials(req.provider)
        pid = creds["id"]
        api_key = (req.api_key if req.api_key is not None and req.api_key.strip() else creds.get("api_key", "")).strip()
        if not api_key:
            return {"ok": False, "provider": pid, "error": "Chưa nhập API Key."}
        sample = "Xin chào, hệ thống dịch giọng nói đã kết nối thành công."
        tgt = req.target_lang or "eng_Latn"
        try:
            if pid == "google":
                tr = GoogleCloudTranslator(api_key=api_key).translate(sample, "vie_Latn", tgt)
            elif pid == "azure":
                region = req.region if req.region is not None else creds.get("region", "")
                endpoint = req.endpoint if req.endpoint is not None else creds.get("endpoint", "")
                tr = AzureTranslator(api_key=api_key, region=region, endpoint=endpoint).translate(sample, "vie_Latn", tgt)
            elif pid == "gemini":
                model = req.model if req.model is not None else creds.get("model", "gemini-2.5-flash")
                tr = GeminiCloudTranslator(api_key=api_key, model=model).translate(sample, "vie_Latn", tgt)
            else:
                return {"ok": False, "provider": pid, "error": f"Nhà cung cấp không hỗ trợ: {pid}"}
            self.cloud_quota.record_usage(pid, len(sample))
            self.clear_cloud_cooldown()
            return {
                "ok": True,
                "provider": pid,
                "engine": tr.engine,
                "latency_ms": tr.latency_ms,
                "translated_text": tr.translated_text,
                "cloud_api": self.cloud_quota.get_public_status(),
            }
        except Exception as exc:
            return {"ok": False, "provider": pid, "error": str(exc)}

    def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        mt_mode: str = "auto-offline",
        beam_size: int = 1,
        llm_url: str = "http://127.0.0.1:11434/v1",
        llm_model: str = "translategemma:4b",
        llm_api_key: str = "local",
        context: Optional[list[tuple[str, str]]] = None,
        is_draft: bool = False,
    ) -> TranslationResult:
        clean = self.glossary.apply_asr_fixes(text.strip())
        if not clean:
            return TranslationResult("", "", source_lang, target_lang, 0.0, mt_mode)
        if source_lang == "vie_Latn":
            clean = self.glossary.normalize_session_entities(clean, source_lang)

        if source_lang == target_lang:
            return TranslationResult(clean, clean, source_lang, target_lang, 0.0, "passthrough")

        active_term_locks = self.glossary.get_effective_term_locks(clean, source_lang, target_lang)
        is_vi_en_pair = {source_lang, target_lang} == {"vie_Latn", "eng_Latn"}
        can_hymt2 = (
            self._llama_server_ready
            and self.model_manager.is_hymt2_ready()
            and bool(get_hymt2_lang_name(source_lang))
            and bool(get_hymt2_lang_name(target_lang))
        )

        # 0) Official Cloud API Mode (`cloud-api`: Google Cloud / Microsoft Azure F0 / Google Gemini)
        #    HYBRID ARCHITECTURE (saves ~85% monthly quota!):
        #    - Live drafts (`is_draft=True` when `draft_offline=True`): ALWAYS skip Cloud API and use Local Offline!
        #    - Final sentences (`is_draft=False`): Check monthly hard cap ($0 guard) -> call official Cloud API
        #      -> record usage only on cache misses -> fall back seamlessly to Offline SOTA on cap/offline/error!
        if mt_mode == "cloud-api":
            creds = self.cloud_quota.get_credentials()
            pid = creds["id"]
            should_call_cloud = (not is_draft) or (not creds.get("draft_offline", True))
            in_cooldown = time.time() < self._cloud_cooldown_until
            if should_call_cloud and not in_cooldown:
                if not self.cloud_quota.can_spend(pid, len(clean)):
                    with self._cloud_notice_lock:
                        cap_msg = "Đã chạm ngưỡng miễn phí tháng này ($0) — đang tự dịch bằng Offline."
                        if self._cloud_fallback_reason != cap_msg:
                            self._cloud_fallback_reason = cap_msg
                            self._cloud_cooldown_until = time.time() + 120
                            self.broadcast_threadsafe(
                                {"type": "cloud_notice", "data": {"message": cap_msg, "cooldown_s": 120}}
                            )
                else:
                    try:
                        adapter = self._get_cloud_translator(creds)
                        if pid == "gemini":
                            res = adapter.translate(
                                clean,
                                source_lang,
                                target_lang,
                                context=context,
                                term_locks=active_term_locks,
                            )
                            if not res.cached:
                                self.cloud_quota.record_usage(pid, len(clean))
                            self.clear_cloud_cooldown()
                            return res
                        else:
                            masked_c, restore_c = self.glossary.protect_terms(clean, source_lang, target_lang)
                            res = adapter.translate(masked_c, source_lang, target_lang)
                            if not res.cached:
                                self.cloud_quota.record_usage(pid, len(clean))
                            self.clear_cloud_cooldown()
                            restored_c = self.glossary.restore_terms(res.translated_text, restore_c)
                            return TranslationResult(
                                source_text=clean,
                                translated_text=restored_c,
                                source_lang=res.source_lang,
                                target_lang=res.target_lang,
                                latency_ms=res.latency_ms,
                                engine=res.engine,
                                cached=res.cached,
                            )
                    except Exception as exc:
                        # Record friendly status + enter cooldown, then fall back seamlessly to Offline SOTA!
                        self._record_cloud_error(pid, exc)

        # 2-Tier Smart Offline Architecture (v0.8.6):
        # - Live Drafts (`is_draft=True`) on VI <-> EN: ALWAYS stay on ultra-fast EnViT5 (176-341ms on any CPU)!
        # - Final Committed Sentences (`is_draft=False`) across ALL 38 Hy-MT2 languages (INCLUDING VI <-> EN!):
        #   Route directly to Hy-MT2-1.8B (9.1-9.3/10 VI->EN, 8.7/10 EN->VI, native Context + Glossary) in
        #   `auto-offline`, `hy-mt2`, and as offline fallback for `cloud-api`.
        # - `light-offline` skips Hy-MT2 completely (EnViT5 + NLLB-600M only for low-RAM PCs).
        use_hymt2 = can_hymt2 and (
            (mt_mode in ("auto-offline", "hy-mt2", "cloud-api") and not is_draft)
            or (mt_mode == "hy-mt2" and not is_vi_en_pair)
        )
        if use_hymt2:
            try:
                hymt = self.ensure_hymt2_loaded()
                if hymt is not None:
                    return hymt.translate(
                        clean,
                        source_lang,
                        target_lang,
                        context=context,
                        term_locks=active_term_locks,
                    )
            except Exception:
                pass

        # Placeholder-protected paths for CTranslate2 / Online / External LLM
        masked, restore_pairs = self.glossary.protect_terms(clean, source_lang, target_lang)

        # Standalone proper-name utterance (e.g. "Trương Gia Bình.") -> preserve verbatim with 0ms MT!
        if restore_pairs:
            stripped_masked = masked.strip(".,!?;: ")
            for tok, tgt_name in restore_pairs:
                if stripped_masked == tok:
                    direct_out = self.glossary.restore_terms(masked, restore_pairs)
                    return TranslationResult(clean, direct_out, source_lang, target_lang, 0.0, "proper-name-lock")

        def _finish(res: TranslationResult) -> TranslationResult:
            restored = self.glossary.restore_terms(res.translated_text, restore_pairs)
            return TranslationResult(
                source_text=clean,
                translated_text=restored,
                source_lang=res.source_lang,
                target_lang=res.target_lang,
                latency_ms=res.latency_ms,
                engine=res.engine,
                cached=res.cached,
            )

        # 1) Fast online (0% CPU, ~80-160ms) — drafts stay Offline to prevent rapid-fire 429 throttling!
        if mt_mode == "fast-online" and not is_draft:
            try:
                return _finish(self._fast_online.translate(masked, source_lang, target_lang))
            except Exception:
                pass

        # 2) Pro Local LLM (context-aware across turns) with offline fallback
        if mt_mode == "local-llm":
            try:
                llm = self.get_llm_translator(llm_url, llm_model, llm_api_key)
                return _finish(llm.translate(masked, source_lang, target_lang, context=context))
            except Exception:
                pass

        # 3) VietAI EnViT5 (Ultra-fast SOTA VI <-> EN, 266 MB INT8, OpenRAIL)
        if mt_mode in ("auto-offline", "light-offline", "cloud-api", "hy-mt2", "fast-online", "local-llm") and is_vi_en_pair:
            envit5 = self.ensure_envit5_loaded()
            if envit5 is not None:
                return _finish(envit5.translate(masked, source_lang, target_lang, beam_size=beam_size))

        if mt_mode in ("opus", "auto-offline", "light-offline", "cloud-api", "hy-mt2", "fast-online", "local-llm") and is_vi_en_pair:
            if source_lang == "vie_Latn" and target_lang == "eng_Latn":
                opus = self.ensure_opus_loaded("vi-en")
                if opus is not None:
                    return _finish(opus.translate(masked, beam_size=beam_size))
            elif source_lang == "eng_Latn" and target_lang == "vie_Latn":
                opus = self.ensure_opus_loaded("en-vi")
                if opus is not None:
                    return _finish(opus.translate(masked, beam_size=beam_size))

        # 4) Offline NLLB-200 (prefers 600M INT8 — 2x faster & 750 MB lighter for Lao; 1.3B via `nllb-1.3b`)
        nllb = self.ensure_nllb_loaded(prefer_13b=(mt_mode == "nllb-1.3b"))
        return _finish(nllb.translate(masked, source_lang, target_lang, beam_size=beam_size))

    def _resolve_asr_choice_for_lang(self, lang_code: str, forced_engine: str = "auto") -> str:
        """Pick optimal ASR backend given language code and OS."""
        if forced_engine != "auto":
            return forced_engine

        if lang_code == "vie_Latn":
            if self.model_manager.is_zipformer_streaming_ready():
                return "zipformer-30m-streaming"
            return "viet-asr"

        if lang_code == "eng_Latn":
            if self.model_manager.is_zipformer_en_20m_ready():
                return "zipformer-en-20m"
            if self.mac_helper is not None:
                return "apple-native"
            raise RuntimeError("Chưa có model nhận dạng tiếng Anh (hãy tải Zipformer-20M EN).")

        # For Thai, Japanese, Korean, Chinese, French, German, etc. -> macOS Native Apple ASR
        if self.mac_helper is not None:
            return "apple-native"
        raise RuntimeError(
            f"Ngôn ngữ đầu vào '{lang_code}' cần Apple Native ASR trên macOS. Trên Windows hiện hỗ trợ offline Tiếng Việt & Tiếng Anh."
        )

    def _resolve_asr_choice(self, cfg: SessionConfig) -> str:
        return self._resolve_asr_choice_for_lang(cfg.source_lang, cfg.asr_engine)

    def get_status_payload(self) -> dict[str, Any]:
        src = get_source_language(self.config.source_lang)
        tgt = get_language(self.config.target_lang)
        g = self.glossary.get()
        cfg_public = asdict(self.config)
        raw_llm_key = str(cfg_public.get("llm_api_key", "") or "").strip()
        if raw_llm_key and raw_llm_key != "local":
            cfg_public["llm_api_key"] = (
                f"{raw_llm_key[:4]}••••{raw_llm_key[-4:]}"
                if len(raw_llm_key) > 8
                else "••••"
            )
        return {
            "listening": self._listening,
            "active_asr_label": self._active_asr_label,
            "remote_asr_label": self._remote_asr_label,
            "asr_loaded": (self._asr_zf_vi is not None) or (self._asr_viet is not None) or (self._asr_zf_en is not None),
            "pass2_loaded": self._pass2_refiner is not None,
            "hotwords_active": self._asr_zf_vi._hotwords_count if self._asr_zf_vi is not None else 0,
            "zipformer_ready": self.model_manager.is_zipformer_streaming_ready(),
            "zipformer_en_ready": self.model_manager.is_zipformer_en_20m_ready(),
            "moonshine_en_ready": self.model_manager.is_moonshine_en_ready(),
            "hymt2_ready": self.model_manager.is_hymt2_ready() and self._llama_server_ready,
            "hymt2_loaded": self._hymt2 is not None and self._hymt2.is_running,
            "hymt2_backend": self._hymt2.active_backend if (self._hymt2 is not None and self._hymt2.is_running) else "",
            "llama_server_ready": self._llama_server_ready,
            "envit5_ready": self.model_manager.is_envit5_ready(),
            "opus_en_ready": self.model_manager.is_opus_vi_en_ready(),
            "opus_en_vi_ready": self.model_manager.is_opus_en_vi_ready(),
            "nllb_13b_ready": self.model_manager.is_nllb_13b_ready(),
            "nllb_loaded": (self._nllb is not None) or (self._nllb_13b is not None),
            "nllb_ready": self.model_manager.is_nllb_ready() or self.model_manager.is_nllb_13b_ready(),
            "mac_helper_ready": self.mac_helper is not None,
            "system_audio_ready": (self.mac_helper is not None) or has_windows_loopback(),
            "apple_caps": self.apple_caps,
            "config": cfg_public,
            "source_lang_meta": asdict(src) if src else None,
            "target_lang_meta": asdict(tgt) if tgt else None,
            "logs": self.logger.get_status(),
            "audience": self.get_audience_summary(),
            "glossary_enabled": g["enabled"],
            "glossary_count": len(g["asr_fixes"]) + len(g["term_locks"]),
            "cloud_api": self.cloud_quota.get_public_status(),
            "cloud_fallback_notice": self.get_cloud_fallback_notice(),
            "envit5_device": self._envit5.device if self._envit5 is not None else "cpu",
            "platform": {
                "os": platform.system(),
                "arch": platform.machine(),
                "python": platform.python_version(),
            },
            "boot_error": self._boot_error,
        }

    def delete_model(self, model_id: str) -> dict[str, Any]:
        with self._model_lock:
            if model_id == "hy-mt2" and self._hymt2 is not None:
                try:
                    self._hymt2.stop()
                    getattr(self._hymt2, "close", lambda: None)()
                except Exception:
                    pass
                self._hymt2 = None
            elif model_id == "nllb-1.3b":
                self._nllb_13b = None
            elif model_id == "nllb-600m":
                self._nllb = None
            elif model_id == "envit5":
                self._envit5 = None
            elif model_id == "opus-vi-en":
                self._opus_ve = None
            elif model_id == "opus-en-vi":
                self._opus_ev = None
            elif model_id == "moonshine-en":
                self._en_pass2_refiner = None
            elif model_id == "zipformer-30m-streaming":
                self._asr_zf_vi = None
            elif model_id == "zipformer-en-20m":
                self._asr_zf_en = None
            elif model_id == "viet-asr":
                self._asr_viet = None
                self._pass2_refiner = None
        try:
            res = self.model_manager.delete_model(model_id)
        except PermissionError as exc:
            raise RuntimeError(
                f"Không thể xóa model '{model_id}' vì file đang được sử dụng. Hãy dừng phiên dịch rồi thử lại. ({exc})"
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"Lỗi hệ điều hành khi xóa model '{model_id}': {exc}") from exc
        self.broadcast_threadsafe({"type": "status", "data": self.get_status_payload()})
        return res

    def start_download(self, model_id: str) -> bool:
        with self._downloads_lock:
            if model_id in self._active_downloads:
                return False
            self._active_downloads.add(model_id)

        def _on_progress(state: dict[str, Any]) -> None:
            self.broadcast_threadsafe({"type": "model_progress", "data": state})

        _on_progress(
            {
                "model_id": model_id,
                "status": "downloading",
                "progress": 1.0,
                "downloaded_mb": 0.0,
                "total_mb": 0.0,
                "message": f"Đang bắt đầu tải {model_id}...",
            }
        )

        def _try_warm_load(fn) -> None:
            try:
                fn()
            except Exception as warm_exc:
                print(f"[engine] Post-download warm-load skipped for {model_id}: {warm_exc}", flush=True)

        def _worker() -> None:
            try:
                if model_id == "nllb-600m":
                    self.model_manager.ensure_nllb_600m(on_progress=_on_progress)
                elif model_id == "nllb-1.3b":
                    self.model_manager.ensure_nllb_13b(on_progress=_on_progress)
                elif model_id == "hy-mt2":
                    self.model_manager.ensure_hymt2(on_progress=_on_progress)
                elif model_id == "zipformer-30m-streaming":
                    self.model_manager.ensure_zipformer_streaming(on_progress=_on_progress)
                    _try_warm_load(lambda: self.ensure_onnx_asr_loaded("zipformer-30m-streaming"))
                elif model_id == "zipformer-en-20m":
                    self.model_manager.ensure_zipformer_en_20m(on_progress=_on_progress)
                    _try_warm_load(lambda: self.ensure_onnx_asr_loaded("zipformer-en-20m"))
                elif model_id == "moonshine-en":
                    self.model_manager.ensure_moonshine_en(on_progress=_on_progress)
                    _try_warm_load(self.ensure_en_pass2_refiner)
                elif model_id == "opus-vi-en":
                    self.model_manager.ensure_opus("vi-en", on_progress=_on_progress)
                    _try_warm_load(lambda: self.ensure_opus_loaded("vi-en"))
                elif model_id == "opus-en-vi":
                    self.model_manager.ensure_opus("en-vi", on_progress=_on_progress)
                    _try_warm_load(lambda: self.ensure_opus_loaded("en-vi"))
                elif model_id == "envit5":
                    self.model_manager.ensure_envit5(on_progress=_on_progress)
                    _try_warm_load(self.ensure_envit5_loaded)
                else:
                    raise ValueError(f"Unknown model_id: {model_id}")
                self.broadcast_threadsafe({"type": "status", "data": self.get_status_payload()})
            except Exception as exc:
                _on_progress(
                    {
                        "model_id": model_id,
                        "status": "error",
                        "progress": 0.0,
                        "downloaded_mb": 0.0,
                        "total_mb": 0.0,
                        "message": str(exc),
                    }
                )
                self.broadcast_threadsafe({"type": "error", "data": {"message": f"Lỗi tải {model_id}: {exc}"}})
            finally:
                with self._downloads_lock:
                    self._active_downloads.discard(model_id)

        threading.Thread(target=_worker, daemon=True, name=f"dl-{model_id}").start()
        return True

    def start_session(self, req: StartSessionRequest) -> dict[str, Any]:
        if self._listening:
            self.stop_session()

        self.config = SessionConfig(
            device_id=req.device_id,
            loopback_device_id=req.loopback_device_id,
            audio_source=req.audio_source if req.audio_source in ("mic", "system", "dual", "mix") else "mic",
            input_channel=req.input_channel if req.input_channel in ("mix", "left", "right") else "mix",
            input_gain_db=max(-12.0, min(24.0, float(req.input_gain_db))),
            auto_gain=bool(req.auto_gain),
            source_lang=req.source_lang,
            target_lang=req.target_lang,
            asr_engine=req.asr_engine,
            mt_mode=req.mt_mode,
            paragraph_mode=req.paragraph_mode if req.paragraph_mode in ("short", "medium", "long") else "long",
            translate_partials=req.translate_partials,
            pass2_refine=req.pass2_refine,
            eco_mode=req.eco_mode,
            beam_size=max(1, min(3, req.beam_size)),
            llm_url=req.llm_url,
            llm_model=req.llm_model,
            llm_api_key=req.llm_api_key,
        )

        chosen_asr = self._resolve_asr_choice(self.config)
        if self.config.audio_source == "mix" and chosen_asr == "apple-native":
            if self.config.source_lang == "vie_Latn" and self.model_manager.is_zipformer_streaming_ready():
                chosen_asr = "zipformer-30m-streaming"
            elif self.config.source_lang == "eng_Latn" and self.model_manager.is_zipformer_en_20m_ready():
                chosen_asr = "zipformer-en-20m"
        self._active_asr_label = chosen_asr
        self._remote_asr_label = ""

        # Lazy-load ONLY the models actually needed for this session (saves ~1+ GB RAM!)
        is_vi_en_pair = {self.config.source_lang, self.config.target_lang} == {"vie_Latn", "eng_Latn"}
        can_hymt2 = (
            self._llama_server_ready
            and self.model_manager.is_hymt2_ready()
            and bool(get_hymt2_lang_name(self.config.source_lang))
            and bool(get_hymt2_lang_name(self.config.target_lang))
        )
        if is_vi_en_pair and self.config.mt_mode not in ("opus", "nllb", "nllb-1.3b") and self.model_manager.is_envit5_ready():
            self.ensure_envit5_loaded()
        elif is_vi_en_pair and (self.config.mt_mode == "opus" or not self.model_manager.is_envit5_ready()):
            if self.config.source_lang == "vie_Latn" and self.model_manager.is_opus_vi_en_ready():
                self.ensure_opus_loaded("vi-en")
            elif self.config.source_lang == "eng_Latn" and self.model_manager.is_opus_en_vi_ready():
                self.ensure_opus_loaded("en-vi")

        if can_hymt2 and (self.config.mt_mode == "hy-mt2" or (self.config.mt_mode == "auto-offline" and not is_vi_en_pair)):
            # Pre-warm Hy-MT2 llama-server in background so session starts immediately
            threading.Thread(
                target=lambda: self.ensure_hymt2_loaded() and self.ensure_hymt2_loaded().ensure_started(),
                daemon=True,
                name="hymt2-warmup",
            ).start()
        else:
            need_nllb = (
                self.config.source_lang != self.config.target_lang
                and (
                    self.config.mt_mode in ("nllb", "nllb-1.3b")
                    or (not is_vi_en_pair and self.config.mt_mode in ("auto-offline", "hy-mt2", "opus"))
                )
            )
            if need_nllb and (self.model_manager.is_nllb_13b_ready() or self.model_manager.is_nllb_ready()):
                self.ensure_nllb_loaded(prefer_13b=(self.config.mt_mode == "nllb-1.3b"))

        if self.config.pass2_refine and (
            (chosen_asr == "zipformer-30m-streaming" and self.config.source_lang == "vie_Latn")
            or (
                self.config.audio_source == "dual"
                and self.config.target_lang == "vie_Latn"
                and self.model_manager.is_zipformer_streaming_ready()
            )
        ):
            self.ensure_pass2_refiner()
        if (
            self.config.pass2_refine
            and (self.config.source_lang == "eng_Latn" or (self.config.audio_source == "dual" and self.config.target_lang == "eng_Latn"))
            and self.model_manager.is_moonshine_en_ready()
        ):
            self.ensure_en_pass2_refiner()

        # Start real-time auto-save meeting log (.TXT & .SRT)
        self.logger.start_new_session(self.config.source_lang, self.config.target_lang)

        self._stop_event.clear()
        self._mt_wakeup.clear()
        while not self._final_mt_queue.empty():
            try:
                self._final_mt_queue.get_nowait()
            except queue.Empty:
                break
        with self._partial_lock:
            self._pending_partial = None
        self._stable_prefix.reset()
        with self._para_lock:
            self._para_parts.clear()
            self._para_pcm_chunks.clear()
            self._para_pcm_truncated = False
            self._para_has_carry_prefix = False
            self._para_has_active_partial = False
            self._para_last_speech_ts = 0.0
        with self._remote_lock:
            self._remote_parts.clear()
            self._remote_pcm_chunks.clear()
            self._remote_has_active_partial = False
            self._remote_last_speech_ts = 0.0
            self._remote_seg_id = uuid.uuid4().hex[:10]
        with self._group_lock:
            self._active_group_id = ""
            self._active_group_words = 0
            self._active_group_last_ts = 0.0
            self._active_group_langs = ("", "")
            self._active_group_speaker = ""
        self._recent_context.clear()
        self.glossary.clear_session_entities()

        self._current_segment_id = uuid.uuid4().hex[:10]

        self._mt_thread = threading.Thread(target=self._mt_worker_loop, daemon=True, name="mt-worker")
        self._mt_thread.start()

        # Pre-warm Hy-MT2 llama-server in the background so Sentence #1 has zero cold-start delay
        if (
            self.config.mt_mode in ("auto-offline", "hy-mt2")
            and self._llama_server_ready
            and self.model_manager.is_hymt2_ready()
            and bool(get_hymt2_lang_name(self.config.source_lang))
            and bool(get_hymt2_lang_name(self.config.target_lang))
        ):
            def _warmup_hymt2() -> None:
                try:
                    h = self.ensure_hymt2_loaded()
                    if h is not None:
                        h.ensure_started()
                        self.broadcast_threadsafe({"type": "status", "data": self.get_status_payload()})
                except Exception:
                    pass

            threading.Thread(target=_warmup_hymt2, daemon=True, name="hymt2-warmup").start()

        # Primary audio channel: "mic" in mic/dual mode, "system" in system mode, "mix" in hall-mixer mode
        primary_source = (
            "mix"
            if self.config.audio_source == "mix"
            else "system"
            if self.config.audio_source == "system"
            else "mic"
        )

        if chosen_asr == "apple-native":
            if self.mac_helper is None:
                raise RuntimeError("Apple Neural Engine ASR chỉ hỗ trợ trên macOS.")
            src_meta = get_source_language(self.config.source_lang)
            locale = src_meta.locale if src_meta else "vi-VN"
            self._apple_proc = subprocess.Popen(
                [
                    str(self.mac_helper),
                    "--mode",
                    "apple-asr",
                    "--locale",
                    locale,
                    "--source",
                    "system" if primary_source == "system" else "mic",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            _drain_stderr_async(self._apple_proc, "apple-asr")
            self._listening = True
            self._asr_thread = threading.Thread(
                target=self._apple_asr_loop,
                daemon=True,
                name="apple-asr-worker",
            )
            self._asr_thread.start()
        else:
            asr = self.ensure_onnx_asr_loaded(chosen_asr)
            asr.reset()

            def _on_meter(level: float, clipping: bool) -> None:
                now = time.perf_counter()
                if clipping or (now - self._last_level_emit >= 0.08):
                    self._last_level_emit = now
                    self.broadcast_threadsafe(
                        {"type": "level", "data": {"level": round(level, 3), "clipping": bool(clipping)}}
                    )

            def _make_system_stream(meter_cb=None, apply_cond: bool = True):
                err_cb = lambda m: self.broadcast_threadsafe({"type": "error", "data": {"message": m}})
                g_db = self.config.input_gain_db if apply_cond else 0.0
                agc = self.config.auto_gain if apply_cond else False
                if self.mac_helper is not None:
                    return MacSystemAudioStream(
                        self.mac_helper,
                        on_meter=meter_cb,
                        on_error=err_cb,
                        gain_db=g_db,
                        auto_gain=agc,
                    )
                if platform.system() == "Windows":
                    return WindowsLoopbackStream(
                        on_meter=meter_cb,
                        on_error=err_cb,
                        loopback_device_id=self.config.loopback_device_id,
                        gain_db=g_db,
                        auto_gain=agc,
                    )
                raise RuntimeError(
                    "Thu âm thanh hệ thống trực tiếp hỗ trợ trên macOS (ScreenCaptureKit) và Windows (WASAPI Loopback)."
                )

            if primary_source == "system":
                self._audio_stream = _make_system_stream(meter_cb=_on_meter, apply_cond=True)
            elif primary_source == "mix":
                mic_st = MicrophoneStream(
                    device_id=self.config.device_id,
                    channel_mode=self.config.input_channel,
                    gain_db=0.0,
                    auto_gain=False,
                )
                sys_st = _make_system_stream(meter_cb=None, apply_cond=False)
                self._audio_stream = MixedAudioStream(
                    mic_st,
                    sys_st,
                    on_meter=_on_meter,
                    gain_db=self.config.input_gain_db,
                    auto_gain=self.config.auto_gain,
                )
            else:
                self._audio_stream = MicrophoneStream(
                    device_id=self.config.device_id,
                    on_meter=_on_meter,
                    channel_mode=self.config.input_channel,
                    gain_db=self.config.input_gain_db,
                    auto_gain=self.config.auto_gain,
                )

            self._audio_stream.start()
            self._listening = True
            self._asr_thread = threading.Thread(
                target=self._audio_loop,
                args=(asr,),
                daemon=True,
                name="asr-worker",
            )
            self._asr_thread.start()

        # Dual Audio Mode (2-Way Zoom/Meet): Spawn concurrent System Audio partner channel!
        if self.config.audio_source == "dual":
            if self.mac_helper is None and platform.system() != "Windows":
                self.broadcast_threadsafe(
                    {
                        "type": "error",
                        "data": {"message": "Họp 2 chiều song song (Mic + Loa) hỗ trợ trực tiếp trên macOS và Windows."},
                    }
                )
            else:
                try:
                    remote_lang = self.config.target_lang
                    remote_asr_choice = self._resolve_asr_choice_for_lang(remote_lang, "auto")
                    self._remote_asr_label = remote_asr_choice
                    if remote_asr_choice == "apple-native":
                        rmeta = get_source_language(remote_lang)
                        rlocale = rmeta.locale if rmeta else "en-US"
                        self._remote_apple_proc = subprocess.Popen(
                            [
                                str(self.mac_helper),
                                "--mode",
                                "apple-asr",
                                "--locale",
                                rlocale,
                                "--source",
                                "system",
                            ],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            bufsize=1,
                        )
                        _drain_stderr_async(self._remote_apple_proc, "remote-apple-asr")
                        self._remote_asr_thread = threading.Thread(
                            target=self._remote_apple_asr_loop,
                            daemon=True,
                            name="remote-apple-asr",
                        )
                        self._remote_asr_thread.start()
                    else:
                        remote_asr = self.ensure_onnx_asr_loaded(remote_asr_choice)
                        remote_asr.reset()
                        if self.mac_helper is not None:
                            self._remote_audio_stream = MacSystemAudioStream(
                                self.mac_helper,
                                on_error=lambda m: self.broadcast_threadsafe({"type": "error", "data": {"message": m}}),
                            )
                        else:
                            self._remote_audio_stream = WindowsLoopbackStream(
                                loopback_device_id=self.config.loopback_device_id,
                                on_error=lambda m: self.broadcast_threadsafe({"type": "error", "data": {"message": m}}),
                            )
                        self._remote_audio_stream.start()
                        self._remote_asr_thread = threading.Thread(
                            target=self._remote_audio_loop,
                            args=(remote_asr,),
                            daemon=True,
                            name="remote-asr-worker",
                        )
                        self._remote_asr_thread.start()
                except Exception as exc:
                    self.broadcast_threadsafe(
                        {"type": "error", "data": {"message": f"Không khởi tạo được kênh Loa đối tác: {exc}"}}
                    )

        payload = self.get_status_payload()
        self.broadcast_threadsafe({"type": "status", "data": payload})
        return payload

    def stop_session(self) -> dict[str, Any]:
        self._stop_event.set()
        self._mt_wakeup.set()
        for stream_attr in ("_audio_stream", "_remote_audio_stream"):
            st = getattr(self, stream_attr, None)
            if st is not None:
                try:
                    st.stop()
                except Exception:
                    pass
                setattr(self, stream_attr, None)
        for proc_attr in ("_apple_proc", "_remote_apple_proc"):
            pr = getattr(self, proc_attr, None)
            if pr is not None:
                try:
                    pr.terminate()
                    pr.wait(timeout=1.5)
                except Exception:
                    try:
                        pr.kill()
                    except Exception:
                        pass
                setattr(self, proc_attr, None)
        for th_attr in ("_asr_thread", "_remote_asr_thread", "_mt_thread"):
            th = getattr(self, th_attr, None)
            if th is not None and th.is_alive():
                th.join(timeout=2.0)
            setattr(self, th_attr, None)
        self._listening = False
        self.broadcast_threadsafe({"type": "level", "data": {"level": 0.0}})
        payload = self.get_status_payload()
        self.broadcast_threadsafe({"type": "status", "data": payload})
        return payload

    def shutdown(self) -> None:
        self.stop_session()
        # Cancel any active batch file translation
        if self._file_job_cancel is not None:
            try:
                self._file_job_cancel.set()
            except Exception:
                pass
        # Stop audience fan-out worker cleanly
        self._fanout_stop.set()
        try:
            self._fanout_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._fanout_thread is not None and self._fanout_thread.is_alive():
            self._fanout_thread.join(timeout=1.5)
        self._fanout_thread = None

        with self._model_lock:
            if self._hymt2 is not None:
                try:
                    self._hymt2.stop()
                    getattr(self._hymt2, "close", lambda: None)()
                except Exception:
                    pass
                self._hymt2 = None
            for client_obj in (
                self._fast_online,
                self._google_cloud,
                self._azure_mt,
                self._gemini_mt,
                self._llm,
            ):
                if client_obj is not None:
                    try:
                        getattr(client_obj, "close", lambda: None)()
                    except Exception:
                        pass
            self._google_cloud = None
            self._azure_mt = None
            self._gemini_mt = None
            self._llm = None
            self._pass2_refiner = None
            self._en_pass2_refiner = None
            self._asr_viet = None
            self._asr_zf_vi = None
            self._asr_zf_en = None

    def update_config(self, req: UpdateConfigRequest) -> dict[str, Any]:
        need_restart = False
        if req.device_id is not None and req.device_id != self.config.device_id:
            self.config.device_id = req.device_id
            if self.config.audio_source in ("mic", "dual", "mix"):
                need_restart = True
        if req.loopback_device_id is not None and req.loopback_device_id != self.config.loopback_device_id:
            self.config.loopback_device_id = req.loopback_device_id
            if self.config.audio_source in ("system", "dual", "mix"):
                need_restart = True
        if req.source_lang is not None and req.source_lang != self.config.source_lang:
            self.config.source_lang = req.source_lang
            need_restart = True
        if req.audio_source is not None and req.audio_source != self.config.audio_source:
            self.config.audio_source = req.audio_source
            need_restart = True
        if req.input_channel is not None and req.input_channel in ("mix", "left", "right"):
            prev_ch = self.config.input_channel
            self.config.input_channel = req.input_channel
            # Switching between mono ("mix") and discrete stereo ("left"/"right") may need 2ch reopen if currently 1ch
            if self._audio_stream is not None and hasattr(self._audio_stream, "set_conditioning"):
                self._audio_stream.set_conditioning(channel_mode=self.config.input_channel)
                src_ch = getattr(self._audio_stream, "_src_channels", 2)
                if prev_ch != req.input_channel and req.input_channel in ("left", "right") and src_ch < 2:
                    need_restart = True
        if req.input_gain_db is not None:
            self.config.input_gain_db = max(-12.0, min(24.0, float(req.input_gain_db)))
            if self._audio_stream is not None and hasattr(self._audio_stream, "set_conditioning"):
                self._audio_stream.set_conditioning(gain_db=self.config.input_gain_db)
        if req.auto_gain is not None:
            self.config.auto_gain = bool(req.auto_gain)
            if self._audio_stream is not None and hasattr(self._audio_stream, "set_conditioning"):
                self._audio_stream.set_conditioning(auto_gain=self.config.auto_gain)
        if req.asr_engine is not None and req.asr_engine != self.config.asr_engine:
            self.config.asr_engine = req.asr_engine
            need_restart = True
        if req.target_lang is not None and req.target_lang != self.config.target_lang:
            self.config.target_lang = req.target_lang
            if self.config.audio_source == "dual":
                need_restart = True
        if req.mt_mode is not None:
            self.config.mt_mode = req.mt_mode
        if req.paragraph_mode is not None and req.paragraph_mode in ("short", "medium", "long"):
            self.config.paragraph_mode = req.paragraph_mode
        if req.translate_partials is not None:
            self.config.translate_partials = req.translate_partials
        if req.pass2_refine is not None:
            self.config.pass2_refine = req.pass2_refine
        if req.eco_mode is not None:
            self.config.eco_mode = req.eco_mode
        if req.beam_size is not None:
            self.config.beam_size = max(1, min(3, req.beam_size))
        if req.llm_url is not None:
            self.config.llm_url = req.llm_url
        if req.llm_model is not None:
            self.config.llm_model = req.llm_model
        if req.llm_api_key is not None:
            self.config.llm_api_key = req.llm_api_key

        if need_restart and self._listening:
            return self.start_session(StartSessionRequest(**asdict(self.config)))

        payload = self.get_status_payload()
        self.broadcast_threadsafe({"type": "status", "data": payload})
        return payload

    def clear_history(self) -> None:
        self._history.clear()
        with self._group_lock:
            self._active_group_id = ""
            self._active_group_words = 0
            self._active_group_last_ts = 0.0
            self._active_group_langs = ("", "")
        self._recent_context.clear()
        self.glossary.clear_session_entities()
        self.broadcast_threadsafe({"type": "history", "data": []})

    def _get_capitalized_glossary_words(self) -> set[str]:
        """Collect lowercased leading tokens of capitalized glossary terms/proper nouns to preserve casing."""
        words: set[str] = set()
        for phrase in self._get_glossary_hotwords():
            toks = phrase.strip().split()
            if toks and toks[0] and toks[0][0].isupper():
                words.add(toks[0].lower())
        return words

    @classmethod
    def _join_fragments(cls, parts: list[str], preserve_case: Optional[set[str]] = None) -> str:
        valid_single = {"i", "a", "à", "ừ", "ờ", "ý", "ở", "ồ", "ô", "ơ", "ư"}
        clean_parts: list[str] = []
        for p in parts:
            s = (p or "").strip()
            if not s:
                continue
            # Filter isolated 1-char acoustic noise artifacts (e.g. stray "O" at tail silence)
            if len(s) == 1 and s.lower() not in valid_single:
                continue
            clean_parts.append(s)

        if not clean_parts:
            return ""
        merged = clean_parts[0]
        for nxt in clean_parts[1:]:
            if merged.endswith((".", "!", "?", "…")):
                merged = f"{merged} {nxt}"
            else:
                words_nxt = nxt.split()
                first_word = words_nxt[0]
                first_clean = first_word.strip(".,!?;:\"'()[]")
                is_proper_or_acronym = (
                    first_clean.isupper()
                    or first_clean in {"I", "I'm", "I've", "I'll", "I'd"}
                    or any(ch.isupper() for ch in first_clean[1:])
                    or (len(words_nxt) >= 2 and bool(words_nxt[1]) and words_nxt[1][0].isupper())
                    or (bool(preserve_case) and first_clean.lower() in preserve_case)
                )
                if len(nxt) > 1 and nxt[0].isupper() and not is_proper_or_acronym:
                    nxt_norm = nxt[0].lower() + nxt[1:]
                else:
                    nxt_norm = nxt
                # Avoid awkward commas after connectors ("to the, cloud") or before 1-2 word tails ("triệu, đồng")
                if cls._ends_with_connector(merged) or len(nxt_norm.split()) <= 2:
                    merged = f"{merged} {nxt_norm}"
                else:
                    merged = f"{merged}, {nxt_norm}"
        return merged

    @staticmethod
    def _clean_verbatim(
        text: str,
        source_lang: str = "vie_Latn",
        protected_terms: Optional[set[str]] = None,
    ) -> str:
        """
        Conference Clean Verbatim pre-processor:
        Removes pure spoken hesitation fillers ('ờ, ờm, ừm, um, uh, erm') and immediate stutter
        repeats ('chúng ta chúng ta sẽ' -> 'chúng ta sẽ', 'we we need' -> 'we need') while
        preserving legitimate Vietnamese reduplicative words, numbers, and glossary terms.
        """
        raw = " ".join((text or "").strip().split())
        if not raw:
            return ""

        had_leading_cap = raw[0].isupper()
        prot_lower = {p.lower() for p in (protected_terms or set()) if p}

        # 1) Strip standalone hesitation fillers (Unicode-safe word boundaries)
        if source_lang == "vie_Latn":
            filler_pat = r"(?i)(?:^|(?<=[\s,;:\-—]))(?:à\s+ờm?|ờ\s+thì|ờm+|ờ|ừm+|uhm|um+|uh+|erm)(?=$|[\s,.;:!?])"
        else:
            filler_pat = r"(?i)(?:^|(?<=[\s,;:\-—]))(?:uhm|um+|uh+|erm|ờm+|ừm+)(?=$|[\s,.;:!?])"

        cleaned = re.sub(filler_pat, " ", raw)
        cleaned = re.sub(r"\s*,\s*(?:,\s*)+", ", ", cleaned)
        cleaned = re.sub(r"^[\s,;:\-—]+", "", cleaned)
        cleaned = " ".join(cleaned.split())
        if not cleaned:
            return raw

        # 2) Collapse immediate 2-word and 1-word stutters (skip numbers, glossary, & VI reduplicatives)
        tokens = cleaned.split()
        out: list[str] = []
        i = 0
        while i < len(tokens):
            # 2-word stutter: e.g. "chúng ta chúng ta" or "in the in the"
            if i + 3 < len(tokens):
                a1 = tokens[i].lower().strip(".,!?;:\"'()[]")
                a2 = tokens[i + 1].lower().strip(".,!?;:\"'()[]")
                b1 = tokens[i + 2].lower().strip(".,!?;:\"'()[]")
                b2 = tokens[i + 3].lower().strip(".,!?;:\"'()[]")
                pair = f"{a1} {a2}"
                if (
                    a1
                    and a2
                    and a1 == b1
                    and a2 == b2
                    and not a1.isdigit()
                    and not a2.isdigit()
                    and pair not in prot_lower
                    and not tokens[i + 1].endswith((".", "!", "?"))
                ):
                    out.append(tokens[i])
                    out.append(tokens[i + 1])
                    i += 4
                    continue

            # 1-word stutter: e.g. "tôi tôi muốn" or "we we need"
            if i + 1 < len(tokens):
                w1 = tokens[i].lower().strip(".,!?;:\"'()[]")
                w2 = tokens[i + 1].lower().strip(".,!?;:\"'()[]")
                if (
                    w1
                    and w1 == w2
                    and not w1.isdigit()
                    and w1 not in _VI_REDUPLICATIVE_KEEP
                    and w1 not in prot_lower
                    and not tokens[i].endswith((".", "!", "?"))
                ):
                    out.append(tokens[i + 1])
                    i += 2
                    continue

            out.append(tokens[i])
            i += 1

        res = " ".join(out).strip()
        res = re.sub(r"\s+([,.;:!?])", r"\1", res)
        if source_lang == "vie_Latn":
            res = restore_vietnamese_casing(res)
        if had_leading_cap and res and res[0].islower():
            res = res[0].upper() + res[1:]
        return res

    @staticmethod
    def _ends_with_connector(text: str) -> bool:
        words = text.lower().rstrip(".,!?;: ").split()
        if not words:
            return False
        single_connectors = {
            "và", "hoặc", "nhưng", "mà", "thì", "là", "của", "vì", "nếu", "để",
            "nên", "với", "cho", "trong", "khi", "cũng", "hay", "như", "tại",
            "cái", "những", "các", "một", "đôi", "sẽ", "đang", "có", "bị", "được",
            "tận", "cả", "đối",
            "and", "but", "so", "to", "that", "because", "with", "the", "of", "in", "for", "or", "a", "an",
        }
        if words[-1] in single_connectors:
            return True
        if len(words) >= 2:
            last2 = f"{words[-2]} {words[-1]}"
            if last2 in {
                "nhưng mà", "để mà", "có thể", "tức là", "gọi là", "bất kể",
                "cũng như", "hay là", "cho nên", "sẽ có", "có cái", "thì đôi",
            }:
                return True
        return False

    @classmethod
    def _split_clean_commit(cls, text: str, min_words: int) -> tuple[str, str]:
        """
        If `text` ends mid-phrase on a dangling modifier/connector (e.g. '...sẽ có cái'),
        split at the last clean boundary (punctuation, comma, or clause starter) and
        carry the trailing tail over into the NEXT paragraph so nothing is cut awkwardly.
        """
        words = text.split()
        n = len(words)
        if n <= min_words or not cls._ends_with_connector(text):
            return text, ""

        # Search backwards up to 10 words for a clean boundary while keeping >= min_words
        lo = max(min_words - 2, n - 10)
        clause_starters = {
            "nhưng", "tại", "tức", "cho", "sau", "còn", "để", "và", "vì", "nếu", "khi",
            "because", "but", "and", "so", "which", "while", "when", "that",
        }

        # 1) Sentence punctuation or comma
        for i in range(n - 1, lo, -1):
            if words[i - 1].endswith((".", "!", "?", ",", ";")):
                head = " ".join(words[:i]).rstrip(" ,;")
                tail = " ".join(words[i:]).strip()
                if head and tail:
                    return head, tail

        # 2) Right before a clause starter
        for i in range(n - 1, lo, -1):
            if words[i].lower().strip(",.?!;:") in clause_starters:
                head = " ".join(words[:i]).rstrip(" ,;")
                tail = " ".join(words[i:]).strip()
                if head and tail and not cls._ends_with_connector(head):
                    return head, tail

        # 3) Strip trailing dangling tokens (1..3 words) so they carry over cleanly
        cut = n
        while cut > lo and (n - cut) < 3 and cls._ends_with_connector(" ".join(words[:cut])):
            cut -= 1
        if cut < n and cut >= lo:
            return " ".join(words[:cut]).rstrip(" ,;"), " ".join(words[cut:]).strip()

        return text, ""

    @classmethod
    def _trim_dangling_tail_for_draft(cls, text: str) -> str:
        """
        YouTube Live Anti-Hallucination Guard:
        If an in-progress speech draft ends on a dangling connector (e.g. '...nhưng mà',
        '...để mà', '...because', '...to the'), strip the trailing dangling connector ONLY
        from the string sent to partial MT (keep it visible in the live source transcript).
        This prevents neural MT from hallucinating fake continuations like "but it's not".
        """
        words = text.strip().split()
        if len(words) < 5:
            return text.strip()
        cut = len(words)
        while cut > 3 and (len(words) - cut) < 3 and cls._ends_with_connector(" ".join(words[:cut])):
            cut -= 1
        trimmed = " ".join(words[:cut]).rstrip(" ,;:-")
        return trimmed if trimmed else text.strip()

    def _paragraph_thresholds(self) -> tuple[int, int, float]:
        """
        Conference Semantic Sentence Commit thresholds (Backend Translation Unit):
        Commit clean semantic clauses (12-28 words) so Pass-2 & MT operate at peak
        accuracy and comfortable reading pace, while UI merges sentences into paragraphs.
        """
        mode = self.config.paragraph_mode
        if mode == "short":
            return (7, 15, 0.95)
        if mode == "medium":
            return (10, 22, 1.15)
        # Default "long": commit natural conference clauses (~12-28 words, 1.35s pause);
        # breath-pause soft commit catches 0.45s pauses cleanly once >= 12 words are spoken.
        return (12, 28, 1.35)

    def _visual_paragraph_limits(self) -> tuple[int, float]:
        """Display grouping budget (max_words_per_paragraph, max_gap_seconds) by paragraph_mode."""
        mode = self.config.paragraph_mode
        if mode == "short":
            return (1, 0.0)
        if mode == "medium":
            return (34, 2.8)
        return (62, 4.0)

    def _assign_paragraph_id(
        self,
        word_count: int,
        src_lang: str,
        tgt_lang: str,
        speaker: str = "",
    ) -> str:
        """Group consecutive semantic sentences from the same speaker & turn into one visual paragraph_id."""
        now = time.monotonic()
        max_words, max_gap = self._visual_paragraph_limits()
        with self._group_lock:
            if (
                self.config.paragraph_mode == "short"
                or not self._active_group_id
                or (src_lang, tgt_lang) != self._active_group_langs
                or speaker != self._active_group_speaker
                or (now - self._active_group_last_ts) > max_gap
                or (self._active_group_words > 0 and self._active_group_words + word_count > max_words)
            ):
                self._active_group_id = f"p-{uuid.uuid4().hex[:8]}"
                self._active_group_words = word_count
                self._active_group_langs = (src_lang, tgt_lang)
                self._active_group_speaker = speaker
            else:
                self._active_group_words += word_count
            self._active_group_last_ts = now
            return self._active_group_id

    def _flush_paragraph_locked(self, allow_carry_over: bool = False) -> None:
        if not self._para_parts:
            return
        raw_text = self._join_fragments(self._para_parts, preserve_case=self._get_capitalized_glossary_words())
        src_lang = self._para_src_lang or self.config.source_lang
        tgt_lang = self._para_tgt_lang or self.config.target_lang
        asr_ms = self._para_last_asr_ms

        carry_tail = ""
        if allow_carry_over:
            min_words, _, _ = self._paragraph_thresholds()
            full_text, carry_tail = self._split_clean_commit(raw_text, min_words)
        else:
            full_text = raw_text

        # Snapshot exact paragraph PCM for Pass-2 refinement when boundaries are clean
        pcm_snapshot: Optional[np.ndarray] = None
        if (
            self._para_pcm_chunks
            and not carry_tail
            and not self._para_has_carry_prefix
            and not self._para_pcm_truncated
        ):
            try:
                pcm_snapshot = np.concatenate(self._para_pcm_chunks)
            except Exception:
                pcm_snapshot = None

        self._para_pcm_chunks.clear()
        self._para_pcm_truncated = False
        self._para_has_active_partial = False
        self._para_has_carry_prefix = bool(carry_tail)

        self._para_parts.clear()
        if carry_tail:
            # Capitalize carried-over start cleanly for the next paragraph
            carry_fmt = carry_tail[0].upper() + carry_tail[1:] if len(carry_tail) > 1 else carry_tail
            self._para_parts.append(carry_fmt)
            self._para_last_speech_ts = time.monotonic()
        else:
            self._para_last_speech_ts = 0.0

        seg_id = self._current_segment_id or uuid.uuid4().hex[:10]
        self._current_segment_id = uuid.uuid4().hex[:10]
        with self._partial_lock:
            self._pending_partial = None
        self._stable_prefix.reset()

        if not full_text:
            return

        prot_terms = set(self._get_glossary_hotwords())
        display_text = self._clean_verbatim(
            self.glossary.apply_asr_fixes(full_text),
            src_lang,
            protected_terms=prot_terms,
        )
        speaker_tag = "self" if self.config.audio_source == "dual" else ""
        self.broadcast_threadsafe(
            {
                "type": "partial",
                "data": {
                    "id": seg_id,
                    "source_text": display_text,
                    "source_lang": src_lang,
                    "target_lang": tgt_lang,
                    "asr_ms": asr_ms,
                    "speaker": speaker_tag,
                },
            }
        )
        self._final_mt_queue.put(
            {
                "id": seg_id,
                "source_text": display_text,
                "source_lang": src_lang,
                "target_lang": tgt_lang,
                "asr_ms": asr_ms,
                "pcm": pcm_snapshot,
                "speaker": speaker_tag,
                "timestamp": time.strftime("%H:%M:%S"),
            }
        )
        if carry_tail:
            # Immediately show the carried-over start of the new paragraph
            self.broadcast_threadsafe(
                {
                    "type": "partial",
                    "data": {
                        "id": self._current_segment_id,
                        "source_text": self.glossary.apply_asr_fixes(self._para_parts[0]),
                        "source_lang": src_lang,
                        "target_lang": tgt_lang,
                        "asr_ms": asr_ms,
                    },
                }
            )
        self._mt_wakeup.set()

    def _flush_paragraph_now(self) -> None:
        with self._para_lock:
            self._flush_paragraph_locked(allow_carry_over=False)

    def _flush_paragraph_if_idle(self) -> None:
        _, _, pause_flush_s = self._paragraph_thresholds()
        with self._para_lock:
            if not self._para_parts or self._para_last_speech_ts <= 0.0:
                return
            if (time.monotonic() - self._para_last_speech_ts) >= pause_flush_s:
                self._flush_paragraph_locked(allow_carry_over=False)

    def _enqueue_asr_output(self, out: ASROutput) -> None:
        if out.kind == "none" or not out.text:
            return

        cfg = self.config
        src_lang = cfg.source_lang
        tgt_lang = cfg.target_lang
        now_mono = time.monotonic()

        with self._para_lock:
            self._para_last_speech_ts = now_mono
            self._para_last_asr_ms = out.latency_ms
            self._para_src_lang = src_lang
            self._para_tgt_lang = tgt_lang

            if not self._current_segment_id:
                self._current_segment_id = uuid.uuid4().hex[:10]

            if out.kind == "partial":
                self._para_has_active_partial = True
                parts_snapshot = [*self._para_parts, out.text]
                combined = self._clean_verbatim(
                    self.glossary.apply_asr_fixes(
                        self._join_fragments(parts_snapshot, preserve_case=self._get_capitalized_glossary_words())
                    ),
                    src_lang,
                    protected_terms=set(self._get_glossary_hotwords()),
                )
                seg_id = self._current_segment_id
                self.broadcast_threadsafe(
                    {
                        "type": "partial",
                        "data": {
                            "id": seg_id,
                            "source_text": combined,
                            "source_lang": src_lang,
                            "target_lang": tgt_lang,
                            "asr_ms": out.latency_ms,
                            "speaker": "self" if cfg.audio_source == "dual" else "",
                        },
                    }
                )
                if cfg.translate_partials:
                    with self._partial_lock:
                        self._pending_partial = {
                            "id": seg_id,
                            "source_text": combined,
                            "parts": parts_snapshot,
                            "source_lang": src_lang,
                            "target_lang": tgt_lang,
                        }
                    self._mt_wakeup.set()
                return

            if out.kind == "final":
                self._para_has_active_partial = False
                self._para_parts.append(out.text)
                combined = self._join_fragments(self._para_parts, preserve_case=self._get_capitalized_glossary_words())
                wc = len(combined.split())
                min_words, target_words, _ = self._paragraph_thresholds()

                should_commit = (
                    wc >= target_words
                    or (wc >= min_words and not self._ends_with_connector(combined))
                    or (
                        wc >= 6
                        and combined.endswith((".", "!", "?"))
                        and not self._ends_with_connector(combined)
                    )
                )

                if should_commit:
                    self._flush_paragraph_locked(allow_carry_over=True)
                else:
                    seg_id = self._current_segment_id
                    parts_snapshot = list(self._para_parts)
                    combined_display = self._clean_verbatim(
                        self.glossary.apply_asr_fixes(combined),
                        src_lang,
                        protected_terms=set(self._get_glossary_hotwords()),
                    )
                    self.broadcast_threadsafe(
                        {
                            "type": "partial",
                            "data": {
                                "id": seg_id,
                                "source_text": combined_display,
                                "source_lang": src_lang,
                                "target_lang": tgt_lang,
                                "asr_ms": out.latency_ms,
                            },
                        }
                    )
                    if cfg.translate_partials:
                        with self._partial_lock:
                            self._pending_partial = {
                                "id": seg_id,
                                "source_text": combined_display,
                                "parts": parts_snapshot,
                                "source_lang": src_lang,
                                "target_lang": tgt_lang,
                            }
                        self._mt_wakeup.set()

    def _apple_asr_loop(self) -> None:
        proc = self._apple_proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                if self._stop_event.is_set():
                    break
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                kind = ev.get("type")
                if kind == "level":
                    self.broadcast_threadsafe({"type": "level", "data": {"level": ev.get("level", 0.0)}})
                elif kind == "partial":
                    self._enqueue_asr_output(
                        ASROutput("partial", str(ev.get("text", "")), float(ev.get("asr_ms", 20.0)), "apple-ane")
                    )
                elif kind == "final":
                    self._enqueue_asr_output(
                        ASROutput("final", str(ev.get("text", "")), float(ev.get("asr_ms", 25.0)), "apple-ane")
                    )
                elif kind == "error":
                    self.broadcast_threadsafe({"type": "error", "data": {"message": str(ev.get("message", ""))}})
        except Exception as exc:
            if not self._stop_event.is_set():
                self.broadcast_threadsafe({"type": "error", "data": {"message": f"Apple ASR lỗi: {exc}"}})
        finally:
            self._flush_paragraph_now()

    def _append_para_pcm(self, chunk: np.ndarray) -> None:
        with self._para_lock:
            self._para_pcm_chunks.append(chunk)
            if not self._para_parts and not self._para_has_active_partial:
                # Keep only ~0.8s lead-in before speech starts so Pass-2 doesn't scan dead silence
                if len(self._para_pcm_chunks) > 8:
                    self._para_pcm_chunks = self._para_pcm_chunks[-8:]
            elif len(self._para_pcm_chunks) > 240:
                self._para_pcm_chunks = self._para_pcm_chunks[-240:]
                self._para_pcm_truncated = True

    def _audio_loop(self, asr) -> None:
        silent_chunks = 0
        breath_chunks = 0
        speech_rms = 0.04
        gated = False
        preroll: deque[np.ndarray] = deque(maxlen=2)
        try:
            while not self._stop_event.is_set() and self._audio_stream is not None:
                chunk = self._audio_stream.read_chunk(timeout=0.15)
                if chunk is None:
                    continue

                # Silence Gate: skip ONNX ASR calls after 2.5s complete silence when idle (~0% CPU)
                rms = float(np.sqrt(np.mean(np.square(chunk.astype(np.float32) / 32768.0))))
                if rms < 0.0045:
                    silent_chunks += 1
                elif rms >= 0.0055:
                    silent_chunks = 0

                # Adaptive Speech/Noise-Floor Breath Detector (works on studio mic, quiet classroom mic & hall!)
                if rms > 0.012:
                    speech_rms = speech_rms * 0.92 + rms * 0.08
                dip_thresh = max(0.007, min(0.028, speech_rms * 0.28))
                if rms < dip_thresh:
                    breath_chunks += 1
                elif rms >= dip_thresh * 1.25:
                    breath_chunks = 0

                if silent_chunks > 25:
                    with self._para_lock:
                        can_gate = (not self._para_parts) and (not self._para_has_active_partial)
                    if can_gate:
                        gated = True
                        breath_chunks = 0
                        preroll.append(chunk)
                        continue

                if gated:
                    gated = False
                    while preroll:
                        pre = preroll.popleft()
                        self._append_para_pcm(pre)
                        self._enqueue_asr_output(asr.accept_pcm(pre))

                self._append_para_pcm(chunk)
                out = asr.accept_pcm(chunk)
                self._enqueue_asr_output(out)

                # Conference Breath-Pause Soft Commit (native rule2=0.72s handles clean pauses;
                # soft commit with 0.45s tail-flush assists on >=0.55s breath dips in long runs)
                if out.kind != "final" and hasattr(asr, "soft_commit_if_ready"):
                    min_words, _, _ = self._paragraph_thresholds()
                    soft_min = 24 if breath_chunks == 5 else (min_words if breath_chunks == 6 else 0)
                    if soft_min > 0:
                        soft_out = asr.soft_commit_if_ready(
                            min_words=soft_min,
                            block_tail_fn=self._ends_with_connector,
                        )
                        if soft_out.kind == "final":
                            self._enqueue_asr_output(soft_out)
                            breath_chunks = 0
        except Exception as exc:
            self.broadcast_threadsafe({"type": "error", "data": {"message": f"Lỗi thu âm / ASR: {exc}"}})
        finally:
            try:
                flushed = asr.flush_final()
                self._enqueue_asr_output(flushed)
            except Exception:
                pass
            self._flush_paragraph_now()

    def _remote_flush_locked(self) -> None:
        if not self._remote_parts:
            return
        r_src = self.config.target_lang
        r_tgt = self.config.source_lang
        full_text = self._clean_verbatim(
            self.glossary.apply_asr_fixes(
                self._join_fragments(self._remote_parts, preserve_case=self._get_capitalized_glossary_words())
            ),
            r_src,
            protected_terms=set(self._get_glossary_hotwords()),
        )
        pcm_snap: Optional[np.ndarray] = None
        if self._remote_pcm_chunks:
            try:
                pcm_snap = np.concatenate(self._remote_pcm_chunks)
            except Exception:
                pcm_snap = None
        self._remote_parts.clear()
        self._remote_pcm_chunks.clear()
        self._remote_has_active_partial = False
        self._remote_last_speech_ts = 0.0
        seg_id = self._remote_seg_id or uuid.uuid4().hex[:10]
        self._remote_seg_id = uuid.uuid4().hex[:10]
        if not full_text:
            return
        # Remote partner speaks `target_lang` -> translated back into our `source_lang`!
        self._final_mt_queue.put(
            {
                "id": seg_id,
                "source_text": full_text,
                "source_lang": r_src,
                "target_lang": r_tgt,
                "asr_ms": self._remote_last_asr_ms,
                "pcm": pcm_snap,
                "speaker": "remote",
                "timestamp": time.strftime("%H:%M:%S"),
            }
        )
        self._mt_wakeup.set()

    def _remote_flush_if_idle(self) -> None:
        _, _, pause_flush_s = self._paragraph_thresholds()
        with self._remote_lock:
            if not self._remote_parts or self._remote_last_speech_ts <= 0.0:
                return
            if (time.monotonic() - self._remote_last_speech_ts) >= pause_flush_s:
                self._remote_flush_locked()

    def _remote_enqueue_asr_output(self, out: ASROutput) -> None:
        if out.kind == "none" or not out.text:
            return
        r_src = self.config.target_lang
        r_tgt = self.config.source_lang
        with self._remote_lock:
            self._remote_last_speech_ts = time.monotonic()
            self._remote_last_asr_ms = out.latency_ms
            if not self._remote_seg_id:
                self._remote_seg_id = uuid.uuid4().hex[:10]

            if out.kind == "partial":
                self._remote_has_active_partial = True
                combined = self._clean_verbatim(
                    self.glossary.apply_asr_fixes(
                        self._join_fragments(
                            [*self._remote_parts, out.text],
                            preserve_case=self._get_capitalized_glossary_words(),
                        )
                    ),
                    r_src,
                    protected_terms=set(self._get_glossary_hotwords()),
                )
                self.broadcast_threadsafe(
                    {
                        "type": "partial",
                        "data": {
                            "id": self._remote_seg_id,
                            "source_text": combined,
                            "source_lang": r_src,
                            "target_lang": r_tgt,
                            "asr_ms": out.latency_ms,
                            "speaker": "remote",
                        },
                    }
                )
                return

            if out.kind == "final":
                self._remote_has_active_partial = False
                self._remote_parts.append(out.text)
                combined = self._join_fragments(
                    self._remote_parts,
                    preserve_case=self._get_capitalized_glossary_words(),
                )
                wc = len(combined.split())
                min_words, target_words, _ = self._paragraph_thresholds()
                if wc >= target_words or (wc >= min_words and not self._ends_with_connector(combined)):
                    self._remote_flush_locked()

    def _append_remote_pcm(self, chunk: np.ndarray) -> None:
        with self._remote_lock:
            self._remote_pcm_chunks.append(chunk)
            if not self._remote_parts and not self._remote_has_active_partial:
                # Keep only ~0.8s lead-in before speech starts so Pass-2 doesn't scan dead silence
                if len(self._remote_pcm_chunks) > 8:
                    self._remote_pcm_chunks = self._remote_pcm_chunks[-8:]
            elif len(self._remote_pcm_chunks) > 240:
                self._remote_pcm_chunks = self._remote_pcm_chunks[-240:]

    def _remote_audio_loop(self, asr) -> None:
        silent_chunks = 0
        breath_chunks = 0
        speech_rms = 0.04
        gated = False
        preroll: deque[np.ndarray] = deque(maxlen=2)
        try:
            while not self._stop_event.is_set() and self._remote_audio_stream is not None:
                chunk = self._remote_audio_stream.read_chunk(timeout=0.15)
                if chunk is None:
                    continue
                rms = float(np.sqrt(np.mean(np.square(chunk.astype(np.float32) / 32768.0))))
                if rms < 0.0045:
                    silent_chunks += 1
                elif rms >= 0.0055:
                    silent_chunks = 0
                if rms > 0.012:
                    speech_rms = speech_rms * 0.92 + rms * 0.08
                dip_thresh = max(0.007, min(0.028, speech_rms * 0.28))
                if rms < dip_thresh:
                    breath_chunks += 1
                elif rms >= dip_thresh * 1.25:
                    breath_chunks = 0
                if silent_chunks > 25:
                    with self._remote_lock:
                        can_gate = (not self._remote_parts) and (not self._remote_has_active_partial)
                    if can_gate:
                        gated = True
                        breath_chunks = 0
                        preroll.append(chunk)
                        continue
                if gated:
                    gated = False
                    while preroll:
                        pre = preroll.popleft()
                        self._append_remote_pcm(pre)
                        self._remote_enqueue_asr_output(asr.accept_pcm(pre))
                self._append_remote_pcm(chunk)
                out = asr.accept_pcm(chunk)
                self._remote_enqueue_asr_output(out)
                if out.kind != "final" and hasattr(asr, "soft_commit_if_ready"):
                    min_words, _, _ = self._paragraph_thresholds()
                    soft_min = 24 if breath_chunks == 5 else (min_words if breath_chunks == 6 else 0)
                    if soft_min > 0:
                        soft_out = asr.soft_commit_if_ready(
                            min_words=soft_min,
                            block_tail_fn=self._ends_with_connector,
                        )
                        if soft_out.kind == "final":
                            self._remote_enqueue_asr_output(soft_out)
                            breath_chunks = 0
        except Exception as exc:
            if not self._stop_event.is_set():
                self.broadcast_threadsafe({"type": "error", "data": {"message": f"Lỗi kênh đối tác: {exc}"}})
        finally:
            try:
                self._remote_enqueue_asr_output(asr.flush_final())
            except Exception:
                pass
            with self._remote_lock:
                self._remote_flush_locked()

    def _remote_apple_asr_loop(self) -> None:
        proc = self._remote_apple_proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                if self._stop_event.is_set():
                    break
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                kind = ev.get("type")
                if kind == "partial":
                    self._remote_enqueue_asr_output(
                        ASROutput("partial", str(ev.get("text", "")), float(ev.get("asr_ms", 20.0)), "apple-ane")
                    )
                elif kind == "final":
                    self._remote_enqueue_asr_output(
                        ASROutput("final", str(ev.get("text", "")), float(ev.get("asr_ms", 25.0)), "apple-ane")
                    )
                elif kind == "error":
                    self.broadcast_threadsafe({"type": "error", "data": {"message": str(ev.get("message", ""))}})
        except Exception as exc:
            if not self._stop_event.is_set():
                self.broadcast_threadsafe({"type": "error", "data": {"message": f"Apple ASR đối tác lỗi: {exc}"}})
        finally:
            with self._remote_lock:
                self._remote_flush_locked()

    def _mt_worker_loop(self) -> None:
        last_partial_time = 0.0
        last_partial_key: tuple[str, str, str] = ("", "", "")
        last_partial_seg_id: str = ""
        last_partial_wc: int = 0
        last_final_mt_ms = 0.0

        while not self._stop_event.is_set() or not self._final_mt_queue.empty():
            if not self._stop_event.is_set():
                self._flush_paragraph_if_idle()
                self._remote_flush_if_idle()
            try:
                item = self._final_mt_queue.get_nowait()
                # Drop any stale draft belonging to the exact utterance now being finalized
                with self._partial_lock:
                    if self._pending_partial is not None and self._pending_partial.get("id") == item.get("id"):
                        self._pending_partial = None
                self._stable_prefix.clear_if(str(item.get("id") or ""))
                cfg = self.config
                source_text = item["source_text"]
                speaker = item.get("speaker", "")
                pass2_ms = 0.0

                # Pass-2 Async Refinement (never blocks live Pass-1 streaming):
                # - Vietnamese: VietASRBatchRefiner (punctuation + ITN) with Glossary-Keyword Preservation Guard!
                # - English: Moonshine-v2-Base (106 MB, 44x RT, 1.82% TED WER + full punctuation!)
                if cfg.pass2_refine and item.get("pcm") is not None:
                    asr_label = self._remote_asr_label if speaker == "remote" else self._active_asr_label
                    if item["source_lang"] == "vie_Latn" and asr_label == "zipformer-30m-streaming":
                        try:
                            p1_fixed = self.glossary.apply_asr_fixes(source_text)
                            refiner = self.ensure_pass2_refiner()
                            refined, pass2_ms = refiner.refine(
                                item["pcm"],
                                p1_fixed,
                                protected_terms=self._get_glossary_hotwords(),
                            )
                            if refined:
                                source_text = refined
                        except Exception:
                            pass
                    elif item["source_lang"] == "eng_Latn":
                        try:
                            en_refiner = self.ensure_en_pass2_refiner()
                            if en_refiner is not None:
                                refined, pass2_ms = en_refiner.refine(item["pcm"], source_text)
                                if refined:
                                    source_text = refined
                        except Exception:
                            pass

                source_text = self._clean_verbatim(
                    self.glossary.apply_asr_fixes(source_text),
                    item["source_lang"],
                    protected_terms=set(self._get_glossary_hotwords()),
                )
                source_text = self.glossary.normalize_session_entities(source_text, item["source_lang"])
                para_id = self._assign_paragraph_id(
                    len(source_text.split()),
                    item["source_lang"],
                    item["target_lang"],
                    speaker=speaker,
                )

                translated = ""
                mt_ms = 0.0
                mt_engine = cfg.mt_mode
                try:
                    res = self.translate_text(
                        source_text,
                        item["source_lang"],
                        item["target_lang"],
                        mt_mode=cfg.mt_mode,
                        beam_size=cfg.beam_size,
                        llm_url=cfg.llm_url,
                        llm_model=cfg.llm_model,
                        llm_api_key=cfg.llm_api_key,
                        context=list(self._recent_context),
                        is_draft=False,
                    )
                    translated = res.translated_text
                    mt_ms = res.latency_ms
                    mt_engine = res.engine
                    last_final_mt_ms = mt_ms
                except Exception as exc:
                    self.broadcast_threadsafe({"type": "error", "data": {"message": f"Lỗi dịch: {exc}"}})

                if source_text and translated:
                    self._recent_context.append((source_text, translated))

                entry = {
                    "id": item["id"],
                    "paragraph_id": para_id,
                    "speaker": speaker,
                    "source_text": source_text,
                    "translated_text": translated,
                    "source_lang": item["source_lang"],
                    "target_lang": item["target_lang"],
                    "asr_ms": item["asr_ms"],
                    "pass2_ms": pass2_ms,
                    "mt_ms": mt_ms,
                    "mt_engine": mt_engine,
                    "timestamp": item["timestamp"],
                }
                self._history.append(entry)
                if len(self._history) > 300:
                    self._history = self._history[-300:]
                try:
                    self.logger.append_entry(entry)
                    entry["log_saved_count"] = self.logger.get_status()["saved_count"]
                except Exception:
                    pass
                self.broadcast_threadsafe({"type": "final", "data": entry})
                self._fanout_audience_final(entry)
                continue
            except queue.Empty:
                pass

            if self._stop_event.is_set():
                break

            # Strict Final Priority: NEVER start a draft translation while a committed sentence is waiting!
            if not self._final_mt_queue.empty():
                continue

            partial_job = None
            with self._partial_lock:
                if self._pending_partial is not None:
                    partial_job = self._pending_partial
                    self._pending_partial = None

            if partial_job is not None:
                now = time.perf_counter()
                cfg = self.config
                # Adaptive Draft Throttle: if MT is under heavy load (>3.2s/sentence), skip drafts so finals never fall behind!
                if last_final_mt_ms > 3200.0:
                    continue
                # Conference-grade calm cadence (~0.85s normal, ~1.05s eco) prevents visual flicker
                debounce_s = 1.05 if cfg.eco_mode else 0.85
                if last_final_mt_ms > 1400.0:
                    debounce_s = max(debounce_s, (last_final_mt_ms / 1000.0) * 0.85)
                draft_src = self._trim_dangling_tail_for_draft(partial_job["source_text"])
                key = (draft_src, partial_job["source_lang"], partial_job["target_lang"])
                if not draft_src or key == last_partial_key:
                    continue

                seg_id = str(partial_job["id"])
                wc_now = len(draft_src.split())
                same_seg = seg_id == last_partial_seg_id
                elapsed_s = now - last_partial_time
                # Wait for calm cadence OR at least 2 new words (unless >=1.25s pause on same segment)
                need_wait = elapsed_s < debounce_s or (
                    same_seg and (wc_now - last_partial_wc) < 2 and elapsed_s < 1.25
                )
                if need_wait:
                    # Re-queue latest draft if nothing newer arrived so final words are never dropped
                    with self._partial_lock:
                        if self._pending_partial is None:
                            self._pending_partial = partial_job
                    self._mt_wakeup.wait(timeout=0.12)
                    self._mt_wakeup.clear()
                    continue

                try:
                    r = self.translate_text(
                        draft_src,
                        partial_job["source_lang"],
                        partial_job["target_lang"],
                        mt_mode=cfg.mt_mode,
                        beam_size=1,
                        llm_url=cfg.llm_url,
                        llm_model=cfg.llm_model,
                        llm_api_key=cfg.llm_api_key,
                        context=list(self._recent_context),
                        is_draft=True,
                    )

                    last_partial_time = time.perf_counter()
                    last_partial_key = key
                    last_partial_seg_id = seg_id
                    last_partial_wc = wc_now
                    if r.latency_ms > 2800.0:
                        last_final_mt_ms = max(last_final_mt_ms, r.latency_ms)

                    stable_text, unstable_text, stabilized = self._stable_prefix.update(
                        seg_id, partial_job["target_lang"], r.translated_text
                    )
                    self.broadcast_threadsafe(
                        {
                            "type": "partial_mt",
                            "data": {
                                "id": seg_id,
                                "translated_text": stabilized,
                                "stable_text": stable_text,
                                "unstable_text": unstable_text,
                                "source_lang": partial_job["source_lang"],
                                "target_lang": partial_job["target_lang"],
                                "mt_ms": round(r.latency_ms, 1),
                                "mt_engine": r.engine,
                            },
                        }
                    )
                except Exception:
                    pass
                continue

            self._mt_wakeup.wait(timeout=0.1)
            self._mt_wakeup.clear()

    def simulate_demo(
        self,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        custom_text: Optional[str] = None,
    ) -> dict[str, Any]:
        src = source_lang or self.config.source_lang
        tgt = target_lang or self.config.target_lang
        if custom_text and custom_text.strip():
            sample = custom_text.strip()
        elif src == "eng_Latn":
            sample = "Hello everyone, today we are comparing MCP, API, and CLI tools for real-time translation."
        else:
            sample = "Xin chào mọi người, hệ thống dịch giọng nói hai chiều đang chạy rất mượt với tốc độ siêu nhanh."

        words = sample.split()
        seg_id = uuid.uuid4().hex[:10]

        def _run_demo() -> None:
            demo_tracker = StablePrefixTracker()
            for step in range(3, len(words) + 1, 3):
                partial_src = self._clean_verbatim(" ".join(words[:step]), src)
                self.broadcast_threadsafe(
                    {
                        "type": "partial",
                        "data": {
                            "id": seg_id,
                            "source_text": partial_src,
                            "source_lang": src,
                            "target_lang": tgt,
                            "asr_ms": 22.0,
                        },
                    }
                )
                try:
                    mt = self.translate_text(partial_src, src, tgt, mt_mode=self.config.mt_mode, beam_size=1)
                    st_txt, unst_txt, full_txt = demo_tracker.update(seg_id, tgt, mt.translated_text)
                    self.broadcast_threadsafe(
                        {
                            "type": "partial_mt",
                            "data": {
                                "id": seg_id,
                                "translated_text": full_txt,
                                "stable_text": st_txt,
                                "unstable_text": unst_txt,
                                "source_lang": src,
                                "target_lang": tgt,
                                "mt_ms": mt.latency_ms,
                                "mt_engine": mt.engine,
                            },
                        }
                    )
                except Exception:
                    pass
                time.sleep(0.35)

            tr_final = ""
            mt_ms = 0.0
            mt_engine = self.config.mt_mode
            try:
                mt = self.translate_text(sample, src, tgt, mt_mode=self.config.mt_mode, beam_size=self.config.beam_size)
                tr_final = mt.translated_text
                mt_ms = mt.latency_ms
                mt_engine = mt.engine
            except Exception:
                pass

            entry = {
                "id": seg_id,
                "paragraph_id": self._assign_paragraph_id(len(sample.split()), src, tgt),
                "source_text": self.glossary.apply_asr_fixes(sample),
                "translated_text": tr_final,
                "source_lang": src,
                "target_lang": tgt,
                "asr_ms": 28.0,
                "mt_ms": mt_ms,
                "mt_engine": mt_engine,
                "timestamp": time.strftime("%H:%M:%S"),
            }
            self._history.append(entry)
            self.broadcast_threadsafe({"type": "final", "data": entry})
            self._fanout_audience_final(entry)

        threading.Thread(target=_run_demo, daemon=True, name="demo-sim").start()
        return {"ok": True, "source_lang": src, "target_lang": tgt}


runtime = EngineRuntime()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    runtime.set_loop(asyncio.get_running_loop())
    runtime.warmup_background()
    yield
    runtime.shutdown()


app = FastAPI(title="VOICERECORE Local Engine", version="0.8.4", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/shutdown")
async def api_shutdown():
    """Graceful sidecar shutdown called by Tauri on app exit or before Windows/macOS auto-update."""
    def _do_exit() -> None:
        # Brief pause so HTTP 200 response flushes cleanly before tearing down streams & process
        time.sleep(0.12)
        try:
            runtime.shutdown()
        except Exception:
            pass
        srv = runtime._uvicorn_server
        if srv is not None:
            srv.should_exit = True
        # Guarantee fast process exit on Windows/macOS so auto-updater never hits file locks
        time.sleep(0.35)
        os._exit(0)

    threading.Thread(target=_do_exit, daemon=True, name="api-shutdown").start()
    return JSONResponse({"ok": True, "shutting_down": True}, headers={"Connection": "close"})


@app.get("/api/health")
def api_health():
    return {"ok": True, **runtime.get_status_payload()}


@app.get("/api/devices")
def api_devices():
    return {
        "devices": list_input_devices(),
        "loopback_devices": list_output_loopback_devices(),
    }


@app.get("/api/languages")
def api_languages():
    langs = list_languages()
    return {"count": len(langs), "languages": langs}


@app.get("/api/source-languages")
def api_source_languages():
    srcs = list_source_languages()
    return {"count": len(srcs), "languages": srcs}


@app.get("/api/models/status")
def api_models_status():
    return runtime.model_manager.get_status()


@app.post("/api/models/download/{model_id}")
def api_download_model(model_id: str):
    if model_id not in (
        "hy-mt2",
        "nllb-1.3b",
        "nllb-600m",
        "zipformer-30m-streaming",
        "zipformer-en-20m",
        "moonshine-en",
        "opus-vi-en",
        "opus-en-vi",
        "envit5",
    ):
        raise HTTPException(status_code=400, detail="Unsupported model_id")
    started = runtime.start_download(model_id)
    return {"ok": True, "model_id": model_id, "started": started}


@app.delete("/api/models/{model_id}")
def api_delete_model(model_id: str):
    try:
        return runtime.delete_model(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/translate")
def api_translate(req: TranslateTextRequest):
    try:
        res = runtime.translate_text(
            req.text,
            req.source_lang,
            req.target_lang,
            mt_mode=req.mt_mode,
            beam_size=req.beam_size,
            llm_url=req.llm_url,
            llm_model=req.llm_model,
            llm_api_key=req.llm_api_key,
        )
        return asdict(res)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/session/start")
def api_session_start(req: StartSessionRequest):
    try:
        return runtime.start_session(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/session/stop")
def api_session_stop():
    return runtime.stop_session()


@app.post("/api/session/config")
def api_session_config(req: UpdateConfigRequest):
    try:
        return runtime.update_config(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/history/clear")
def api_history_clear():
    runtime.clear_history()
    return {"ok": True}


@app.get("/api/glossary")
def api_glossary_get():
    return runtime.glossary.get()


@app.post("/api/glossary")
def api_glossary_update(req: GlossaryUpdateRequest):
    data = runtime.glossary.update(
        enabled=req.enabled,
        asr_fixes=req.asr_fixes,
        term_locks=req.term_locks,
    )
    runtime.sync_asr_hotwords()
    runtime.broadcast_threadsafe({"type": "status", "data": runtime.get_status_payload()})
    return data


@app.post("/api/glossary/preset/{preset_name}")
def api_glossary_preset(preset_name: str):
    try:
        data = runtime.glossary.apply_preset(preset_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    runtime.sync_asr_hotwords()
    runtime.broadcast_threadsafe({"type": "status", "data": runtime.get_status_payload()})
    return data


@app.get("/api/cloud/config")
def api_cloud_get():
    return runtime.cloud_quota.get_public_status()


@app.post("/api/cloud/config")
def api_cloud_update(req: CloudConfigUpdateRequest):
    data = runtime.cloud_quota.update(req.model_dump(exclude_none=True))
    runtime.clear_cloud_cooldown()
    runtime.broadcast_threadsafe({"type": "status", "data": runtime.get_status_payload()})
    return data


@app.post("/api/cloud/test")
def api_cloud_test(req: CloudTestRequest):
    return runtime.test_cloud_connection(req)


@app.post("/api/cloud/quota/reset")
def api_cloud_quota_reset(req: CloudQuotaResetRequest):
    data = runtime.cloud_quota.reset_usage(req.provider)
    runtime.broadcast_threadsafe({"type": "status", "data": runtime.get_status_payload()})
    return data


@app.get("/api/logs/status")
def api_logs_status():
    return runtime.logger.get_status()


@app.post("/api/logs/open")
def api_logs_open():
    return {"ok": runtime.logger.open_logs_folder(), **runtime.logger.get_status()}


@app.get("/audience", response_class=HTMLResponse)
def page_audience():
    return HTMLResponse(AUDIENCE_HTML)


@app.get("/obs", response_class=HTMLResponse)
def page_obs():
    return HTMLResponse(OBS_HTML)


@app.get("/api/audience/info")
def api_audience_info():
    return runtime.get_audience_info()


@app.post("/api/session/summarize")
def api_session_summarize():
    return summarize_meeting_entries(
        list(runtime._history),
        runtime.config.source_lang,
        runtime.config.target_lang,
        translate_fn=lambda txt, s, t: runtime.translate_text(txt, s, t, mt_mode=runtime.config.mt_mode, beam_size=1),
    )


@app.get("/api/files/progress")
def api_files_progress():
    return runtime._file_job_progress


@app.post("/api/files/cancel")
def api_files_cancel():
    ev = runtime._file_job_cancel
    if ev is not None:
        ev.set()
        return {"ok": True, "cancelling": True}
    return {"ok": True, "cancelling": False}


@app.post("/api/files/translate")
async def api_files_translate(
    file: UploadFile = File(...),
    source_lang: str = Form("vie_Latn"),
    target_lang: str = Form("eng_Latn"),
    mt_mode: str = Form("auto-offline"),
    job_id: str = Form(""),
):
    with runtime._file_job_lock:
        if runtime._file_job_busy:
            raise HTTPException(
                status_code=409,
                detail="Đang có một tác vụ dịch file khác chạy. Vui lòng đợi xong hoặc bấm Hủy trước.",
            )
        runtime._file_job_busy = True

    active_job_id = job_id.strip() or uuid.uuid4().hex[:10]
    orig_name = file.filename or "audio.wav"
    suffix = Path(orig_name).suffix or ".wav"
    cancel_ev = threading.Event()
    runtime._file_job_cancel = cancel_ev
    tmp_path: Optional[Path] = None

    def _on_file_progress(evt: dict[str, Any]) -> None:
        runtime._file_job_progress = evt
        runtime.broadcast_threadsafe({"type": "file_progress", "data": evt})

    _on_file_progress(
        {
            "job_id": active_job_id,
            "filename": orig_name,
            "active": True,
            "stage": "uploading",
            "percent": 1.0,
            "message": "Đang tiếp nhận file âm thanh/video...",
            "current_chunk": 0,
            "total_chunks": 0,
            "processed_audio_s": 0.0,
            "duration_s": 0.0,
            "elapsed_s": 0.0,
            "eta_s": None,
            "eta_estimated": True,
            "speed_x": 0.0,
        }
    )

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            tmp_path = Path(tf.name)
            while True:
                if cancel_ev.is_set():
                    raise FileTranslateCancelled("Đã hủy dịch file.")
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                tf.write(chunk)

        result = await asyncio.to_thread(
            translate_media_file,
            tmp_path,
            orig_name,
            source_lang,
            target_lang,
            mt_mode,
            runtime,
            runtime.logger.logs_dir,
            None,
            _on_file_progress,
            cancel_ev,
            active_job_id,
        )
        return result
    except FileTranslateCancelled:
        _on_file_progress(
            {
                **runtime._file_job_progress,
                "active": False,
                "stage": "cancelled",
                "message": "Đã hủy dịch file.",
            }
        )
        return {"ok": False, "cancelled": True}
    except Exception as exc:
        _on_file_progress(
            {
                **runtime._file_job_progress,
                "active": False,
                "stage": "error",
                "message": str(exc),
            }
        )
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if runtime._file_job_cancel is cancel_ev:
            runtime._file_job_cancel = None
        with runtime._file_job_lock:
            runtime._file_job_busy = False
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


@app.post("/api/demo/simulate")
def api_demo_simulate(req: DemoRequest):
    return runtime.simulate_demo(req.source_lang, req.target_lang, req.text)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    await runtime.register_ws(ws)
    try:
        while True:
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        await runtime.unregister_ws(ws)


@app.websocket("/ws/audience")
async def ws_audience_endpoint(ws: WebSocket, lang: str = "follow"):
    await ws.accept()
    await runtime.register_audience_ws(ws, lang=lang)
    try:
        while True:
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        await runtime.unregister_audience_ws(ws)


def main() -> None:
    parser = argparse.ArgumentParser(description="VOICERECORE Local Engine")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args()
    runtime.server_port = args.port
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="warning")
    server = uvicorn.Server(config)
    runtime._uvicorn_server = server
    server.run()


if __name__ == "__main__":
    main()
