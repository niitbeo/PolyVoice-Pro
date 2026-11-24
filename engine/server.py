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
import gc
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
from collections import OrderedDict, deque
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
# Prevent OpenMP / CTranslate2 / ONNX Runtime worker threads from busy-spinning at 200%+ CPU on laptops
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
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
    HybridEnStreamingEngine,
    MoonshineEnStreamingEngine,
    MoonshineJaStreamingEngine,
    MoonshineV2EnRefiner,
    ParakeetEuStreamingEngine,
    VietASRBatchRefiner,
    VietASRStreamingEngine,
    WhisperTurboStreamingEngine,
    ZipformerStreamingMultiEngine,
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
    PARAKEET_EU_LANGS,
    get_hymt2_lang_name,
    get_language,
    get_source_asr_profile,
    get_source_language,
    get_whisper_lang_code,
    list_languages,
    list_source_languages,
)
from model_manager import ModelManager
from quota_tracker import CloudQuotaManager
from session_logger import SessionLogger
from slide_context import SlideContextManager
from translator import (
    AzureTranslator,
    EnViT5Translator,
    FastOnlineTranslator,
    GeminiCloudTranslator,
    GoogleCloudTranslator,
    HyMT2Translator,
    LocalLLMTranslator,
    NLLBTranslator,
    TranslationResult,
    evaluate_translations_with_gemini,
    evaluate_translations_with_google_cloud,
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
    asr_engine: str = "auto"           # "auto" | "apple-native" | "zipformer-30m-streaming" | "moonshine-en" | "parakeet-eu" | "whisper-turbo" | "viet-asr"
    mt_mode: str = "auto-offline"      # "auto-offline" | "hy-mt2" | "nllb" | "fast-online" | "cloud-api" | "local-llm"
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


class EvaluateItemInput(BaseModel):
    id: str
    source_text: str
    translated_text: str = ""
    source_lang: str = "vie_Latn"
    target_lang: str = "eng_Latn"
    timestamp: str = ""


class EvaluateRequest(BaseModel):
    mode: str = "session"  # "session" | "item"
    items: Optional[list[EvaluateItemInput]] = None


class HistoryUpdateRequest(BaseModel):
    id: str
    item_ids: Optional[list[str]] = None
    translated_text: str
    source_text: Optional[str] = None


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
    "mơ", "màng", "lờ", "mờ", "rập", "rình", "lác", "đác", "lẻ", "tẻ",
    "rộn", "ràng", "vội", "vã", "thấp", "thoáng", "lập", "lòe",
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
                    last_locked_missing = bool(locked_norms) and bool(locked_norms[-1]) and (locked_norms[-1] not in window)
                    if (
                        (overlap < 0.30 and len(new_tokens) >= len(self._locked_tokens) + 5)
                        or (last_locked_missing and len(self._locked_tokens) <= 4 and len(new_tokens) >= len(self._locked_tokens) + 2)
                        or (overlap < 0.50 and len(new_tokens) >= len(self._locked_tokens) + 3)
                    ):
                        # Major semantic re-parse / mid-sentence rephrase (e.g. "And come" -> "And he goes back...")
                        # Cleanly re-anchor instead of gluing a stale locked prefix onto a rephrased tail!
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
        _cache_dir = Path(os.environ.get("VOICERECORE_CACHE_DIR", str(Path.home() / ".cache" / "voicerecore")))
        self.slide_context = SlideContextManager(_cache_dir / "slide_context.json")
        self.cloud_quota = CloudQuotaManager()
        self.logger = SessionLogger()
        self.config = SessionConfig()
        self._display_frozen: bool = False
        self._display_freeze_message: str = "Tạm nghỉ — Phiên dịch hội nghị sẽ tiếp tục trong giây lát"
        self._last_slide_match_idx: int = 0
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
        self._audience_snap_lock = threading.Lock()
        self._aud_backfill_lock = threading.Lock()
        self._aud_backfill_cache: OrderedDict[tuple[str, str], tuple[str, str]] = OrderedDict()
        self._aud_backfill_inflight: dict[tuple[str, str, str], asyncio.Task[str]] = {}
        self._aud_backfill_sem: Optional[asyncio.Semaphore] = None
        self._aud_backfill_sem_loop: Optional[asyncio.AbstractEventLoop] = None

        self._asr_zf_vi: Optional[ZipformerStreamingViEngine] = None
        self._asr_hybrid_en: Optional[HybridEnStreamingEngine] = None
        self._asr_hybrid_en_remote: Optional[HybridEnStreamingEngine] = None
        self._asr_moonshine_en: Optional[MoonshineEnStreamingEngine] = None
        self._asr_moonshine_en_remote: Optional[MoonshineEnStreamingEngine] = None
        self._asr_moonshine_ja: Optional[MoonshineJaStreamingEngine] = None
        self._asr_moonshine_ja_remote: Optional[MoonshineJaStreamingEngine] = None
        self._asr_moonshine_ko: Optional[MoonshineJaStreamingEngine] = None
        self._asr_moonshine_ko_remote: Optional[MoonshineJaStreamingEngine] = None
        self._asr_zf_multi: Optional[ZipformerStreamingMultiEngine] = None
        self._asr_zf_multi_remote: Optional[ZipformerStreamingMultiEngine] = None
        self._asr_parakeet_eu: Optional[ParakeetEuStreamingEngine] = None
        self._asr_parakeet_eu_remote: Optional[ParakeetEuStreamingEngine] = None
        self._asr_viet: Optional[VietASRStreamingEngine] = None
        self._asr_whisper: Optional[WhisperTurboStreamingEngine] = None
        self._asr_whisper_remote: Optional[WhisperTurboStreamingEngine] = None
        self._pass2_refiner: Optional[VietASRBatchRefiner] = None
        self._en_pass2_refiner: Optional[MoonshineV2EnRefiner] = None
        self._hymt2: Optional[HyMT2Translator] = None
        self._hymt2_idle_timer: Optional[threading.Timer] = None
        self._envit5: Optional[EnViT5Translator] = None
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
        self._draft_mt_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._listening = False
        self._active_asr_label: str = ""
        self._remote_asr_label: str = ""

        self._final_mt_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=32)
        self._partial_lock = threading.Lock()
        self._pending_partial: Optional[dict[str, Any]] = None
        self._committed_seg_ids: deque[str] = deque(maxlen=64)
        self._stable_prefix = StablePrefixTracker()
        self._mt_wakeup = threading.Event()
        self._draft_wakeup = threading.Event()
        self._final_mt_busy: bool = False

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
        self._post_pass2_carry: dict[tuple[str, str], str] = {}

        self._current_segment_id: str = ""
        self._last_level_emit = 0.0
        self._history_lock = threading.Lock()
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

    @staticmethod
    async def _send_json_safe(ws: WebSocket, payload: dict[str, Any], timeout_s: float = 2.0) -> bool:
        try:
            await asyncio.wait_for(ws.send_json(payload), timeout=timeout_s)
            return True
        except Exception:
            return False

    def _get_backfill_semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._aud_backfill_sem is None or self._aud_backfill_sem_loop is not loop:
            self._aud_backfill_sem = asyncio.Semaphore(2)
            self._aud_backfill_sem_loop = loop
        return self._aud_backfill_sem

    def _cache_audience_translation(
        self,
        entry_id: str,
        lang: str,
        source_text: str,
        translated_text: str,
    ) -> None:
        if not entry_id or not lang:
            return
        key = (str(entry_id), str(lang))
        with self._aud_backfill_lock:
            self._aud_backfill_cache[key] = (source_text, translated_text)
            self._aud_backfill_cache.move_to_end(key)
            while len(self._aud_backfill_cache) > 512:
                self._aud_backfill_cache.popitem(last=False)

    def _invalidate_audience_cache(self, entry_ids: Optional[set[str]] = None) -> None:
        with self._aud_backfill_lock:
            if entry_ids is None:
                self._aud_backfill_cache.clear()
                return
            for key in [k for k in self._aud_backfill_cache if k[0] in entry_ids]:
                self._aud_backfill_cache.pop(key, None)

    async def _translate_audience_backfill(
        self,
        entry_id: str,
        source_text: str,
        source_lang: str,
        req_lang: str,
    ) -> str:
        """Single-flight + LRU-cached + concurrency-capped backfill translation so 20+ simultaneous
        QR joins never queue 160 redundant MT calls or starve live LED translation."""
        cache_key = (str(entry_id), str(req_lang))
        with self._aud_backfill_lock:
            cached = self._aud_backfill_cache.get(cache_key)
            if cached is not None and cached[0] == source_text:
                self._aud_backfill_cache.move_to_end(cache_key)
                return cached[1]

        inflight_key = (str(entry_id), str(req_lang), source_text)
        task = self._aud_backfill_inflight.get(inflight_key)
        if task is None or task.done():
            sem = self._get_backfill_semaphore()

            async def _run() -> str:
                try:
                    async with sem:
                        tr = await asyncio.to_thread(
                            self.translate_text,
                            source_text,
                            source_lang,
                            req_lang,
                            self.config.mt_mode,
                            1,
                            update_slide_state=False,
                        )
                    self._cache_audience_translation(entry_id, req_lang, source_text, tr.translated_text)
                    return tr.translated_text
                finally:
                    if self._aud_backfill_inflight.get(inflight_key) is task_ref:
                        self._aud_backfill_inflight.pop(inflight_key, None)

            task_ref = asyncio.create_task(_run())
            self._aud_backfill_inflight[inflight_key] = task_ref
            task = task_ref

        return await asyncio.shield(task)

    async def register_ws(self, ws: WebSocket) -> None:
        async with self._ws_lock:
            self._ws_clients.add(ws)
        ok = await self._send_json_safe(ws, {"type": "status", "data": self.get_status_payload()})
        if not ok:
            return
        with self._history_lock:
            hist_snap = list(self._history[-100:])
        if hist_snap:
            await self._send_json_safe(ws, {"type": "history", "data": hist_snap})

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
            "display_frozen": self._display_frozen,
            "display_freeze_message": self._display_freeze_message,
        }

    async def register_audience_ws(self, ws: WebSocket, lang: str = "follow") -> None:
        req_lang = lang.strip() if lang and lang.strip() else "follow"
        async with self._ws_lock:
            with self._audience_snap_lock:
                self._audience_ws[ws] = req_lang
        if not await self._send_json_safe(ws, {"type": "status", "data": self.get_audience_status_payload()}):
            return  # Client disconnected immediately — unregister handled by ws_audience_endpoint finally
        # Send recent history (capped at last 8 entries + single-flight cached to avoid MT lock starvation)
        with self._history_lock:
            recent = list(self._history[-8:])
        if recent:
            if req_lang == "follow":
                if not await self._send_json_safe(ws, {"type": "history", "data": recent}):
                    return
            else:
                customized = []
                for e in recent:
                    if e.get("target_lang") == req_lang:
                        customized.append(e)
                    else:
                        try:
                            tr_txt = await self._translate_audience_backfill(
                                str(e.get("id", "")),
                                str(e.get("source_text", "")),
                                str(e.get("source_lang", self.config.source_lang)),
                                req_lang,
                            )
                            customized.append({**e, "target_lang": req_lang, "translated_text": tr_txt})
                        except Exception as exc:
                            logger.warning("Audience history backfill failed for %s: %s", req_lang, exc)
                            customized.append(e)
                if not await self._send_json_safe(ws, {"type": "history", "data": customized}):
                    return
        await self.broadcast({"type": "audience_count", "data": self.get_audience_summary()})

    async def unregister_audience_ws(self, ws: WebSocket) -> None:
        async with self._ws_lock:
            with self._audience_snap_lock:
                self._audience_ws.pop(ws, None)
        await self.broadcast({"type": "audience_count", "data": self.get_audience_summary()})

    def get_audience_summary(self) -> dict[str, Any]:
        with self._audience_snap_lock:
            count = len(self._audience_ws)
            langs = sorted({l for l in self._audience_ws.values() if l != "follow"})
        return {
            "connected_count": count,
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

        ev_type = payload.get("type")
        ev_data = payload.get("data")
        aud_status_payload = (
            {"type": "status", "data": self.get_audience_status_payload()}
            if ev_type == "status" and aud_items
            else None
        )

        # Build bounded-parallel send tasks (timeout=2.0s per socket) so one half-open conference Wi-Fi phone
        # NEVER head-of-line stalls delivery to 50+ other attendees.
        admin_jobs: list[tuple[WebSocket, dict[str, Any]]] = (
            [] if ev_type == "history_patch" else [(ws, payload) for ws in clients]
        )
        aud_jobs: list[tuple[WebSocket, dict[str, Any]]] = []
        for aws, alang in aud_items:
            if ev_type == "status" and aud_status_payload is not None:
                aud_jobs.append((aws, aud_status_payload))
            elif ev_type in ("partial", "display_freeze"):
                aud_jobs.append((aws, payload))
            elif ev_type == "history" and isinstance(ev_data, list):
                if len(ev_data) == 0 or alang == "follow":
                    aud_jobs.append((aws, {"type": "history", "data": ev_data[-80:]}))
            elif ev_type == "history_patch" and isinstance(ev_data, dict):
                patch_entry = ev_data.get("entry")
                if isinstance(patch_entry, dict) and alang != "follow" and alang == patch_entry.get("target_lang"):
                    aud_jobs.append((aws, payload))
            elif ev_type in ("partial_mt", "final") and isinstance(ev_data, dict):
                if alang == "follow" or alang == ev_data.get("target_lang"):
                    aud_jobs.append((aws, payload))

        if not admin_jobs and not aud_jobs:
            return

        sem = asyncio.Semaphore(32)

        async def _bounded_send(sock: WebSocket, msg: dict[str, Any]) -> bool:
            async with sem:
                return await self._send_json_safe(sock, msg, timeout_s=2.0)

        all_results = await asyncio.gather(
            *(_bounded_send(ws, msg) for ws, msg in admin_jobs),
            *(_bounded_send(aws, msg) for aws, msg in aud_jobs),
        )
        n_admin = len(admin_jobs)
        dead = [admin_jobs[i][0] for i in range(n_admin) if not all_results[i]]
        dead_aud = [aud_jobs[i][0] for i in range(len(aud_jobs)) if not all_results[n_admin + i]]

        if dead or dead_aud:
            async with self._ws_lock:
                for ws in dead:
                    self._ws_clients.discard(ws)
                if dead_aud:
                    with self._audience_snap_lock:
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
            if self.loop is None or self.loop.is_closed():
                continue
            with self._audience_snap_lock:
                snap_langs = list(self._audience_ws.values())
            if not snap_langs:
                continue
            removed_ids = list(entry.get("_removed_ids") or [])
            base_entry = {k: v for k, v in entry.items() if k != "_removed_ids"}
            extra_langs = sorted(
                {
                    lang
                    for lang in snap_langs
                    if lang != "follow" and lang != base_entry.get("target_lang")
                }
            )
            for tgt in extra_langs:
                if self._fanout_stop.is_set():
                    break
                try:
                    tr = self.translate_text(
                        base_entry["source_text"],
                        base_entry["source_lang"],
                        tgt,
                        mt_mode=self.config.mt_mode,
                        beam_size=1,
                        llm_url=self.config.llm_url,
                        llm_model=self.config.llm_model,
                        llm_api_key=self.config.llm_api_key,
                        update_slide_state=False,
                    )
                    self._cache_audience_translation(
                        str(base_entry.get("id", "")),
                        tgt,
                        str(base_entry.get("source_text", "")),
                        tr.translated_text,
                    )
                    custom_entry = {
                        **base_entry,
                        "target_lang": tgt,
                        "translated_text": tr.translated_text,
                        "mt_ms": tr.latency_ms,
                        "mt_engine": tr.engine,
                    }
                    wire_payload = (
                        {"type": "history_patch", "data": {"entry": custom_entry, "removed_ids": removed_ids}}
                        if removed_ids
                        else {"type": "final", "data": custom_entry}
                    )

                    async def _push(target_code=tgt, payload=wire_payload) -> None:
                        async with self._ws_lock:
                            targets = [ws for ws, l in self._audience_ws.items() if l == target_code]
                        if not targets:
                            return
                        sem = asyncio.Semaphore(32)

                        async def _one(sock: WebSocket) -> bool:
                            async with sem:
                                return await self._send_json_safe(sock, payload, timeout_s=2.0)

                        results = await asyncio.gather(*(_one(ws) for ws in targets))
                        dead_targets = [targets[i] for i, ok in enumerate(results) if not ok]
                        if dead_targets:
                            async with self._ws_lock:
                                with self._audience_snap_lock:
                                    for dws in dead_targets:
                                        self._audience_ws.pop(dws, None)

                    if self.loop and not self.loop.is_closed():
                        asyncio.run_coroutine_threadsafe(_push(), self.loop)
                except Exception as exc:
                    logger.warning("Audience fan-out translation failed for %s: %s", tgt, exc)

    def _fanout_audience_final(self, entry: dict[str, Any]) -> None:
        """Enqueue a committed sentence onto the bounded FIFO audience fan-out worker."""
        if self.loop is None or self.loop.is_closed():
            return
        with self._audience_snap_lock:
            snap_langs = list(self._audience_ws.values())
        if not snap_langs:
            return
        extra_langs = {
            lang
            for lang in snap_langs
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
                # Do NOT preload 615 MB NLLB-200 at startup (saves ~800 MB RAM!).
                if self.model_manager.is_envit5_ready():
                    self.ensure_envit5_loaded()
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
        phrases: list[str] = []
        seen: set[str] = set()
        g = self.glossary.get()
        if g.get("enabled", True):
            for rule in g.get("term_locks", []):
                src = rule.get("source")
                if src and src.lower() not in seen:
                    seen.add(src.lower())
                    phrases.append(src)
            for rule in g.get("asr_fixes", []):
                cor = rule.get("correct")
                if cor and cor.lower() not in seen:
                    seen.add(cor.lower())
                    phrases.append(cor)
        for hw in self.slide_context.get_hotwords():
            if hw and hw.lower() not in seen:
                seen.add(hw.lower())
                phrases.append(hw)
        return phrases

    def sync_asr_hotwords(self) -> int:
        if self._asr_zf_vi is not None:
            return self._asr_zf_vi.update_hotwords(self._get_glossary_hotwords())
        return 0

    def ensure_onnx_asr_loaded(self, engine_name: str, lang_code: str = "", remote: bool = False):
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

            if engine_name == "hybrid-en":
                if not self.model_manager.is_moonshine_en_ready():
                    raise RuntimeError("Model tiếng Anh Moonshine-v2 chưa được tải.")
                use_remote_slot = remote and self._active_asr_label == "hybrid-en"
                slot = self._asr_hybrid_en_remote if use_remote_slot else self._asr_hybrid_en
                if slot is None:
                    if self._en_pass2_refiner is None:
                        self._en_pass2_refiner = MoonshineV2EnRefiner(
                            self.model_manager.moonshine_en_dir,
                            sample_rate=SAMPLE_RATE,
                            num_threads=_adaptive_mt_threads(),
                        )
                    slot = HybridEnStreamingEngine(
                        self._en_pass2_refiner,
                        self.model_manager.silero_vad_path,
                        sample_rate=SAMPLE_RATE,
                    )
                    if use_remote_slot:
                        self._asr_hybrid_en_remote = slot
                    else:
                        self._asr_hybrid_en = slot
                return slot

            if engine_name == "moonshine-en":
                if not self.model_manager.is_moonshine_en_ready():
                    raise RuntimeError("Model tiếng Anh Moonshine-v2 chưa được tải. Hãy bấm tải trong mục Mô hình Offline.")
                use_remote_slot = remote and self._active_asr_label == "moonshine-en"
                slot = self._asr_moonshine_en_remote if use_remote_slot else self._asr_moonshine_en
                if slot is None:
                    shared_rec = self._en_pass2_refiner._recognizer if (self._en_pass2_refiner is not None and not use_remote_slot) else None
                    slot = MoonshineEnStreamingEngine(
                        self.model_manager.moonshine_en_dir,
                        sample_rate=SAMPLE_RATE,
                        num_threads=_adaptive_mt_threads(),
                        recognizer=shared_rec,
                    )
                    if use_remote_slot:
                        self._asr_moonshine_en_remote = slot
                    else:
                        self._asr_moonshine_en = slot
                return slot

            if engine_name == "moonshine-ja":
                if not (self.model_manager.is_moonshine_ja_ready() and self.model_manager.is_silero_vad_ready()):
                    raise RuntimeError("Model tiếng Nhật Moonshine-v2 chưa được tải. Hãy bấm tải trong mục Mô hình Offline (~99 MB).")
                use_remote_slot = remote and self._active_asr_label == "moonshine-ja"
                slot = self._asr_moonshine_ja_remote if use_remote_slot else self._asr_moonshine_ja
                if slot is None:
                    slot = MoonshineJaStreamingEngine(
                        self.model_manager.moonshine_ja_dir,
                        vad_model=self.model_manager.silero_vad_path,
                        sample_rate=SAMPLE_RATE,
                        num_threads=min(4, _adaptive_mt_threads()),
                        language="ja",
                    )
                    if use_remote_slot:
                        self._asr_moonshine_ja_remote = slot
                    else:
                        self._asr_moonshine_ja = slot
                return slot

            if engine_name == "moonshine-ko":
                if not (self.model_manager.is_moonshine_ko_ready() and self.model_manager.is_silero_vad_ready()):
                    raise RuntimeError("Model tiếng Hàn Moonshine-v2 chưa được tải. Hãy bấm tải trong mục Mô hình Offline (~47 MB).")
                use_remote_slot = remote and self._active_asr_label == "moonshine-ko"
                slot = self._asr_moonshine_ko_remote if use_remote_slot else self._asr_moonshine_ko
                if slot is None:
                    slot = MoonshineJaStreamingEngine(
                        self.model_manager.moonshine_ko_dir,
                        vad_model=self.model_manager.silero_vad_path,
                        sample_rate=SAMPLE_RATE,
                        num_threads=min(4, _adaptive_mt_threads()),
                        language="ko",
                    )
                    if use_remote_slot:
                        self._asr_moonshine_ko_remote = slot
                    else:
                        self._asr_moonshine_ko = slot
                return slot

            if engine_name == "zipformer-multi":
                if not self.model_manager.is_zipformer_multi_ready():
                    raise RuntimeError("Model Zipformer Streaming Đa ngôn ngữ (Thái/Trung/Indo) chưa được tải (~247 MB).")
                use_remote_slot = remote and self._active_asr_label == "zipformer-multi"
                slot = self._asr_zf_multi_remote if use_remote_slot else self._asr_zf_multi
                if slot is None:
                    slot = ZipformerStreamingMultiEngine(
                        self.model_manager.zipformer_multi_dir,
                        language=lang_code or "tha_Thai",
                        sample_rate=SAMPLE_RATE,
                        num_threads=min(4, _adaptive_mt_threads()),
                    )
                    if use_remote_slot:
                        self._asr_zf_multi_remote = slot
                    else:
                        self._asr_zf_multi = slot
                else:
                    slot.set_language(lang_code or "tha_Thai")
                return slot

            if engine_name == "parakeet-eu":
                if not self.model_manager.is_parakeet_eu_ready():
                    raise RuntimeError(
                        "Model Parakeet-Ultra 25 NN Châu Âu chưa được tải (~465 MB). "
                        "Hãy bấm nút 'Tải ~465 MB' ngay dưới ô chọn ngôn ngữ hoặc tải Gói 3."
                    )
                use_remote_slot = remote and self._active_asr_label == "parakeet-eu"
                slot = self._asr_parakeet_eu_remote if use_remote_slot else self._asr_parakeet_eu
                if slot is None:
                    slot = ParakeetEuStreamingEngine(
                        self.model_manager.parakeet_eu_dir,
                        vad_model=self.model_manager.silero_vad_path,
                        sample_rate=SAMPLE_RATE,
                        num_threads=min(4, _adaptive_mt_threads()),
                    )
                    if use_remote_slot:
                        self._asr_parakeet_eu_remote = slot
                    else:
                        self._asr_parakeet_eu = slot
                return slot

            if engine_name == "whisper-turbo":
                if not self.model_manager.is_whisper_turbo_ready():
                    raise RuntimeError(
                        "Model nghe đa ngôn ngữ Whisper-Turbo chưa được tải. "
                        "Hãy mở mục 'Mô hình Offline' và bấm tải 'Nghe Đa ngôn ngữ 99 NN (Whisper-Turbo)' (~540 MB)."
                    )
                iso = get_whisper_lang_code(lang_code) if lang_code else ""
                use_remote_slot = remote and self._active_asr_label == "whisper-turbo"
                slot = self._asr_whisper_remote if use_remote_slot else self._asr_whisper
                if slot is None:
                    slot = WhisperTurboStreamingEngine(
                        self.model_manager.whisper_turbo_dir,
                        vad_model=self.model_manager.silero_vad_path,
                        language=iso,
                        sample_rate=SAMPLE_RATE,
                        num_threads=min(4, _adaptive_mt_threads()),
                    )
                    if use_remote_slot:
                        self._asr_whisper_remote = slot
                    else:
                        self._asr_whisper = slot
                else:
                    slot.set_language(iso)
                return slot

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
                # 2 intra-threads is empirically 4-6x faster than 4 under concurrent ASR/Pass-2 load on laptops
                self._envit5 = EnViT5Translator(
                    self.model_manager.envit5_dir,
                    intra_threads=2,
                )
            return self._envit5

    def ensure_nllb_loaded(self) -> NLLBTranslator:
        """Load lightweight CPU NLLB-200-600M INT8 for all 202 x 202 languages."""
        with self._model_lock:
            if self._nllb is None and self.model_manager.is_nllb_ready():
                self._nllb = NLLBTranslator(
                    self.model_manager.nllb_600m_dir,
                    device="cpu",
                    intra_threads=2,
                    engine_name="nllb-200-600m-int8",
                )
            if self._nllb is not None:
                return self._nllb
        raise RuntimeError(
            "Chưa tải mô hình dịch Offline trên máy. Hãy bấm 'Tải Gói 1' trong mục Mô hình Offline hoặc chuyển sang Cloud API."
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
        update_slide_state: bool = False,
    ) -> TranslationResult:
        clean = self.glossary.apply_asr_fixes(text.strip())
        if not clean:
            return TranslationResult("", "", source_lang, target_lang, 0.0, mt_mode)
        if source_lang == "vie_Latn":
            clean = self.glossary.normalize_session_entities(clean, source_lang)

        if source_lang == target_lang:
            return TranslationResult(clean, clean, source_lang, target_lang, 0.0, "passthrough")

        active_term_locks = list(self.glossary.get_effective_term_locks(clean, source_lang, target_lang))
        slide_matched_terms: list[dict[str, Any]] = []
        slide_domain_hint = ""
        if not is_draft:
            rag = self.slide_context.retrieve_for_utterance(
                clean, source_lang, target_lang, update_state=update_slide_state
            )
            if rag is not None:
                slide_domain_hint = rag.domain_hint
                if rag.matched_terms:
                    slide_matched_terms = list(rag.matched_terms)
                    seen_tl = {str(r.get("source", "")).lower() for r in active_term_locks}
                    for st in rag.matched_terms:
                        s_low = str(st.get("source", "")).lower()
                        if s_low and s_low not in seen_tl:
                            seen_tl.add(s_low)
                            active_term_locks.append(st)
                if update_slide_state and rag.slide_index > 0 and rag.slide_index != self._last_slide_match_idx:
                    self._last_slide_match_idx = rag.slide_index
                    self.broadcast_threadsafe(
                        {
                            "type": "slide_match",
                            "data": {
                                "slide_index": rag.slide_index,
                                "slide_title": rag.slide_title,
                                "score": rag.score,
                            },
                        }
                    )
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
        #    - Final sentences (`is_draft=False`): Check API key & monthly hard cap ($0 guard) -> call official Cloud API
        #      -> record usage only on cache misses -> fall back seamlessly to Offline SOTA on cap/offline/error!
        if mt_mode == "cloud-api":
            creds = self.cloud_quota.get_credentials()
            pid = creds["id"]
            should_call_cloud = (not is_draft) or (not creds.get("draft_offline", True))
            in_cooldown = time.time() < self._cloud_cooldown_until
            if should_call_cloud and not in_cooldown:
                if not str(creds.get("api_key", "")).strip():
                    with self._cloud_notice_lock:
                        key_msg = "Chưa cấu hình Cloud API Key — đang tự dịch bằng Offline."
                        if self._cloud_fallback_reason != key_msg:
                            self._cloud_fallback_reason = key_msg
                            self._cloud_cooldown_until = time.time() + 120
                            self.broadcast_threadsafe(
                                {"type": "cloud_notice", "data": {"message": key_msg, "cooldown_s": 120}}
                            )
                elif not self.cloud_quota.can_spend(pid, len(clean)):
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
                            masked_c, restore_c = self.glossary.protect_terms(
                                clean, source_lang, target_lang, extra_locks=slide_matched_terms
                            )
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

        # v0.9.2 Lightweight Faithful Offline Architecture (0% GPU by default!):
        # - `auto-offline` & `light-offline`: NEVER launch heavy `Hy-MT2` llama-server (saves 1.31 GB RAM & GPU!).
        #   Uses ultra-light CPU CTranslate2 models with faithful beam=2 for final sentences (beam=1 for live drafts):
        #     * VI <-> EN: VietAI EnViT5 (~380 MB RAM, ~260-450ms, length_penalty=1.15 -> zero dropped words!)
        #     * 202 Languages (Lao, Thai, Japanese, Korean, Chinese, Khmer...): NLLB-200-600M INT8 (beam=2 finals!)
        # - `hy-mt2` (explicit opt-in only): runs Hy-MT2-1.8B GPU/CPU when the user intentionally selects it.
        use_hymt2 = can_hymt2 and (mt_mode == "hy-mt2") and (not is_draft or not is_vi_en_pair)
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
                        domain_hint=slide_domain_hint,
                    )
            except Exception:
                pass

        # Placeholder-protected paths for CTranslate2 / Online / External LLM
        masked, restore_pairs = self.glossary.protect_terms(
            clean, source_lang, target_lang, extra_locks=slide_matched_terms
        )

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

        # Keep greedy beam=1 by default (5x faster & 3x cooler CPU than beam=2 under live ASR/Pass-2 load);
        # finals use gentle repetition_penalty=1.05 in EnViT5Translator for maximum word faithfulness.
        eff_beam = 1 if is_draft else max(1, beam_size)

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

        # 3) VietAI EnViT5 (Ultra-light, faithful VI <-> EN, 266 MB INT8, 0% GPU)
        # Gracefully accept retired "opus" / "nllb-1.3b" values from older client configs
        eff_mt_mode = "auto-offline" if mt_mode == "opus" else ("nllb" if mt_mode == "nllb-1.3b" else mt_mode)
        if eff_mt_mode in ("auto-offline", "light-offline", "cloud-api", "hy-mt2", "fast-online", "local-llm") and is_vi_en_pair:
            envit5 = self.ensure_envit5_loaded()
            if envit5 is not None:
                return _finish(
                    envit5.translate(masked, source_lang, target_lang, beam_size=eff_beam, is_draft=is_draft)
                )

        # 4) Offline NLLB-200-600M INT8 (0% GPU, faithful beam=2 for finals across all 202 langs)
        nllb = self.ensure_nllb_loaded()
        return _finish(nllb.translate(masked, source_lang, target_lang, beam_size=eff_beam))

    def _is_asr_model_ready(self, model_id: str) -> bool:
        if model_id == "zipformer-30m-streaming":
            return self.model_manager.is_zipformer_streaming_ready()
        if model_id == "moonshine-en":
            return self.model_manager.is_moonshine_en_ready()
        if model_id == "moonshine-ja":
            return self.model_manager.is_moonshine_ja_ready() and self.model_manager.is_silero_vad_ready()
        if model_id == "moonshine-ko":
            return self.model_manager.is_moonshine_ko_ready() and self.model_manager.is_silero_vad_ready()
        if model_id == "zipformer-multi":
            return self.model_manager.is_zipformer_multi_ready()
        if model_id == "parakeet-eu":
            return self.model_manager.is_parakeet_eu_ready()
        if model_id == "whisper-turbo":
            return self.model_manager.is_whisper_turbo_ready()
        return False

    @staticmethod
    def _is_asr_compatible_with_lang(engine_id: str, lang_code: str) -> bool:
        """Verify whether a manually pinned ASR engine actually supports `lang_code`."""
        if not engine_id or engine_id in ("auto", "whisper-turbo", "apple-native"):
            return True
        if engine_id in ("zipformer-30m-streaming", "viet-asr"):
            return lang_code == "vie_Latn"
        if engine_id in ("hybrid-en", "moonshine-en"):
            return lang_code == "eng_Latn"
        if engine_id == "moonshine-ja":
            return lang_code == "jpn_Jpan"
        if engine_id == "moonshine-ko":
            return lang_code == "kor_Hang"
        if engine_id == "zipformer-multi":
            return lang_code in {
                "tha_Thai", "zho_Hans", "zho_Hant", "ind_Latn", "rus_Cyrl", "arb_Arab",
                "vie_Latn", "eng_Latn", "jpn_Jpan",
            }
        if engine_id == "parakeet-eu":
            return (lang_code in PARAKEET_EU_LANGS) or (lang_code == "eng_Latn")
        return False

    def _is_apple_on_device_ready(self, lang_code: str) -> bool:
        """Strictly require macOS on-device speech assets so `apple-native` NEVER streams audio to Apple servers."""
        if self.mac_helper is None or lang_code in ("lao_Laoo", "khm_Khmr", "mya_Mymr"):
            return False
        on_dev = [str(loc).lower().replace("_", "-") for loc in (self.apple_caps.get("on_device_locales") or [])]
        if not on_dev:
            return False
        src_meta = get_source_language(lang_code)
        if not src_meta or not src_meta.locale:
            return False
        want = src_meta.locale.lower().replace("_", "-")
        if want in on_dev:
            return True
        want_prefix = want.split("-", 1)[0] + "-"
        return any(loc.startswith(want_prefix) for loc in on_dev)

    def get_source_readiness(self, lang_code: str) -> dict[str, Any]:
        """Return declarative 3-tier readiness metadata for any source language."""
        prof = get_source_asr_profile(lang_code)
        pref_ready = self._is_asr_model_ready(prof.preferred_model)
        # Vietnamese also has built-in viet-asr
        if lang_code == "vie_Latn":
            pref_ready = pref_ready or True
        fallback_ready = self._is_asr_model_ready(prof.fallback_model)
        apple_ready = self._is_apple_on_device_ready(lang_code)

        if pref_ready:
            active_tier = prof.asr_tier
            ready = True
        elif fallback_ready:
            active_tier = "experimental" if prof.asr_tier == "experimental" else "universal"
            ready = True
        elif apple_ready:
            active_tier = "apple"
            ready = True
        else:
            active_tier = "missing"
            ready = False

        return {
            "lang_code": lang_code,
            "ready": ready,
            "preferred_ready": pref_ready,
            "fallback_ready": fallback_ready,
            "active_tier": active_tier,
            "preferred_model": prof.preferred_model,
            "preferred_model_mb": prof.preferred_model_mb,
            "fallback_model": prof.fallback_model,
            "recommended_pack": prof.recommended_pack,
            "asr_tier": prof.asr_tier,
            "note": prof.asr_note,
        }

    def _resolve_asr_choice_for_lang(self, lang_code: str, forced_engine: str = "auto") -> str:
        """Pick optimal ASR backend via 3-tier declarative registry: Fast-Lane -> Universal 99 NN -> OS Native (strict on-device)."""
        if forced_engine == "zipformer-en-20m" or not self._is_asr_compatible_with_lang(forced_engine, lang_code):
            forced_engine = "auto"
        if forced_engine == "apple-native" and not self._is_apple_on_device_ready(lang_code):
            forced_engine = "auto"
        if forced_engine != "auto":
            return forced_engine

        prof = get_source_asr_profile(lang_code)

        # Tier 1: Fast-Lane dedicated streaming models (30-400ms, 0% GPU)
        if lang_code == "vie_Latn":
            return "zipformer-30m-streaming" if self.model_manager.is_zipformer_streaming_ready() else "viet-asr"

        if lang_code == "eng_Latn":
            if self.model_manager.is_moonshine_en_ready() and self.model_manager.is_silero_vad_ready():
                return "hybrid-en"
            if self.model_manager.is_moonshine_en_ready():
                return "moonshine-en"
        elif prof.preferred_model != "whisper-turbo" and self._is_asr_model_ready(prof.preferred_model):
            return prof.preferred_model

        # Tier 2: Universal 99-language cross-platform safety net (Whisper Large-v3-Turbo + Silero VAD v5)
        if self.model_manager.is_whisper_turbo_ready():
            return "whisper-turbo"

        # Tier 3: macOS Apple Neural Engine fallback — ONLY when strict on-device asset is installed!
        if self._is_apple_on_device_ready(lang_code):
            return "apple-native"

        src_meta = get_source_language(lang_code)
        lang_label = src_meta.name_vi if src_meta else lang_code
        if prof.preferred_model != "whisper-turbo":
            raise RuntimeError(
                f"Chưa tải gói nghe offline cho {lang_label}. Hãy bấm nút 'Tải ngay (~{prof.preferred_model_mb} MB)' hoặc tải Gói Toàn cầu 99 NN."
            )
        raise RuntimeError(
            f"Để nhận dạng giọng nói {lang_label} offline (Windows & macOS), hãy bấm tải 'Gói Toàn cầu 99 NN (Whisper-Turbo • 540 MB)'."
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
            "asr_loaded": (self._asr_zf_vi is not None) or (self._asr_viet is not None) or (self._asr_zf_multi is not None) or (self._asr_hybrid_en is not None) or (self._asr_moonshine_en is not None) or (self._asr_moonshine_ja is not None) or (self._asr_moonshine_ko is not None) or (self._asr_parakeet_eu is not None) or (self._asr_whisper is not None),
            "pass2_loaded": self._pass2_refiner is not None,
            "hotwords_active": self._asr_zf_vi._hotwords_count if self._asr_zf_vi is not None else 0,
            "zipformer_ready": self.model_manager.is_zipformer_streaming_ready(),
            "zipformer_multi_ready": self.model_manager.is_zipformer_multi_ready(),
            "moonshine_en_ready": self.model_manager.is_moonshine_en_ready(),
            "moonshine_ja_ready": self.model_manager.is_moonshine_ja_ready(),
            "moonshine_ko_ready": self.model_manager.is_moonshine_ko_ready(),
            "parakeet_eu_ready": self.model_manager.is_parakeet_eu_ready(),
            "whisper_turbo_ready": self.model_manager.is_whisper_turbo_ready(),
            "hymt2_ready": self.model_manager.is_hymt2_ready() and self._llama_server_ready,
            "hymt2_loaded": self._hymt2 is not None and self._hymt2.is_running,
            "hymt2_backend": self._hymt2.active_backend if (self._hymt2 is not None and self._hymt2.is_running) else "",
            "llama_server_ready": self._llama_server_ready,
            "envit5_ready": self.model_manager.is_envit5_ready(),
            "nllb_loaded": self._nllb is not None,
            "nllb_ready": self.model_manager.is_nllb_ready(),
            "mac_helper_ready": self.mac_helper is not None,
            "system_audio_ready": (self.mac_helper is not None) or has_windows_loopback(),
            "apple_caps": self.apple_caps,
            "config": cfg_public,
            "source_lang_meta": asdict(src) if src else None,
            "active_source_readiness": self.get_source_readiness(self.config.source_lang),
            "target_lang_meta": asdict(tgt) if tgt else None,
            "logs": self.logger.get_status(),
            "audience": self.get_audience_summary(),
            "glossary_enabled": g["enabled"],
            "glossary_count": len(g["asr_fixes"]) + len(g["term_locks"]),
            "slide_context": self.slide_context.get_status_summary(),
            "display_frozen": self._display_frozen,
            "display_freeze_message": self._display_freeze_message,
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
            elif model_id == "nllb-600m":
                self._nllb = None
            elif model_id == "envit5":
                self._envit5 = None
            elif model_id == "moonshine-en":
                self._en_pass2_refiner = None
                self._asr_hybrid_en = None
                self._asr_hybrid_en_remote = None
                self._asr_moonshine_en = None
                self._asr_moonshine_en_remote = None
            elif model_id == "moonshine-ja":
                self._asr_moonshine_ja = None
                self._asr_moonshine_ja_remote = None
            elif model_id == "moonshine-ko":
                self._asr_moonshine_ko = None
                self._asr_moonshine_ko_remote = None
            elif model_id == "zipformer-multi":
                self._asr_zf_multi = None
                self._asr_zf_multi_remote = None
            elif model_id == "parakeet-eu":
                self._asr_parakeet_eu = None
                self._asr_parakeet_eu_remote = None
            elif model_id == "whisper-turbo":
                for w in (self._asr_whisper, self._asr_whisper_remote):
                    if w is not None:
                        try:
                            w.close()
                        except Exception:
                            pass
                self._asr_whisper = None
                self._asr_whisper_remote = None
                self._asr_hybrid_en = None
                self._asr_hybrid_en_remote = None
            elif model_id == "zipformer-30m-streaming":
                self._asr_zf_vi = None
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
                elif model_id == "hy-mt2":
                    self.model_manager.ensure_hymt2(on_progress=_on_progress)
                elif model_id == "zipformer-30m-streaming":
                    self.model_manager.ensure_zipformer_streaming(on_progress=_on_progress)
                    _try_warm_load(lambda: self.ensure_onnx_asr_loaded("zipformer-30m-streaming"))
                elif model_id == "moonshine-en":
                    self.model_manager.ensure_moonshine_en(on_progress=_on_progress)
                    _try_warm_load(self.ensure_en_pass2_refiner)
                elif model_id == "moonshine-ja":
                    self.model_manager.ensure_moonshine_ja(on_progress=_on_progress)
                elif model_id == "moonshine-ko":
                    self.model_manager.ensure_moonshine_ko(on_progress=_on_progress)
                elif model_id == "zipformer-multi":
                    self.model_manager.ensure_zipformer_multi(on_progress=_on_progress)
                elif model_id == "parakeet-eu":
                    self.model_manager.ensure_parakeet_eu(on_progress=_on_progress)
                elif model_id == "whisper-turbo":
                    self.model_manager.ensure_whisper_turbo(on_progress=_on_progress)
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

    def _cancel_hymt2_idle_timer(self) -> None:
        tmr = self._hymt2_idle_timer
        self._hymt2_idle_timer = None
        if tmr is not None:
            try:
                tmr.cancel()
            except Exception:
                pass

    def _schedule_hymt2_idle_sleep(self, delay_s: float = 60.0) -> None:
        self._cancel_hymt2_idle_timer()
        if self._hymt2 is None or not self._hymt2.is_running:
            return

        def _sleep_now() -> None:
            if self._listening:
                return
            with self._model_lock:
                if self._hymt2 is not None and not self._listening:
                    try:
                        self._hymt2.stop()
                        getattr(self._hymt2, "close", lambda: None)()
                    except Exception:
                        pass
                    self._hymt2 = None
            gc.collect()
            self.broadcast_threadsafe({"type": "status", "data": self.get_status_payload()})

        t = threading.Timer(delay_s, _sleep_now)
        t.daemon = True
        self._hymt2_idle_timer = t
        t.start()

    def _release_unused_models(
        self,
        keep_asr: set[str],
        keep_mt: set[str],
        keep_pass2: set[str],
    ) -> None:
        """
        Automatic RAM Hygiene (v0.9.2):
        Unload models not needed for the current language pair / MT mode so switching across
        VI/EN/TH/JA/LO or leaving Hy-MT2 never accumulates 3-4 GB of resident RAM!
        """
        released = False
        with self._model_lock:
            if "hy-mt2" not in keep_mt and self._hymt2 is not None:
                try:
                    self._hymt2.stop()
                    getattr(self._hymt2, "close", lambda: None)()
                except Exception:
                    pass
                self._hymt2 = None
                released = True
            if "nllb-600m" not in keep_mt and self._nllb is not None:
                self._nllb = None
                released = True
            if "envit5" not in keep_mt and self._envit5 is not None:
                self._envit5 = None
                released = True

            if "whisper-turbo" not in keep_asr:
                for w in (self._asr_whisper, self._asr_whisper_remote):
                    if w is not None:
                        try:
                            w.close()
                        except Exception:
                            pass
                        released = True
                self._asr_whisper = None
                self._asr_whisper_remote = None

            if "zipformer-30m-streaming" not in keep_asr and self._asr_zf_vi is not None:
                self._asr_zf_vi = None
                released = True
            if "viet-asr" not in keep_asr and self._asr_viet is not None:
                self._asr_viet = None
                released = True
            if "vi-pass2" not in keep_pass2 and self._pass2_refiner is not None:
                try:
                    self._pass2_refiner.close()
                except Exception:
                    pass
                self._pass2_refiner = None
                released = True

            if "hybrid-en" not in keep_asr:
                if self._asr_hybrid_en is not None or self._asr_hybrid_en_remote is not None:
                    self._asr_hybrid_en = None
                    self._asr_hybrid_en_remote = None
                    released = True
            if "moonshine-en" not in keep_asr:
                if self._asr_moonshine_en is not None or self._asr_moonshine_en_remote is not None:
                    self._asr_moonshine_en = None
                    self._asr_moonshine_en_remote = None
                    released = True
            if "moonshine-ja" not in keep_asr:
                if self._asr_moonshine_ja is not None or self._asr_moonshine_ja_remote is not None:
                    self._asr_moonshine_ja = None
                    self._asr_moonshine_ja_remote = None
                    released = True
            if "moonshine-ko" not in keep_asr:
                if self._asr_moonshine_ko is not None or self._asr_moonshine_ko_remote is not None:
                    self._asr_moonshine_ko = None
                    self._asr_moonshine_ko_remote = None
                    released = True
            if "zipformer-multi" not in keep_asr:
                if self._asr_zf_multi is not None or self._asr_zf_multi_remote is not None:
                    self._asr_zf_multi = None
                    self._asr_zf_multi_remote = None
                    released = True
            if "parakeet-eu" not in keep_asr:
                if self._asr_parakeet_eu is not None or self._asr_parakeet_eu_remote is not None:
                    self._asr_parakeet_eu = None
                    self._asr_parakeet_eu_remote = None
                    released = True
            if "en-pass2" not in keep_pass2 and "hybrid-en" not in keep_asr and self._en_pass2_refiner is not None:
                self._en_pass2_refiner = None
                released = True

        if released:
            gc.collect()

    def start_session(self, req: StartSessionRequest) -> dict[str, Any]:
        self._cancel_hymt2_idle_timer()
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
            asr_engine=(
                "auto"
                if (req.asr_engine == "zipformer-en-20m" or not self._is_asr_compatible_with_lang(req.asr_engine, req.source_lang))
                else req.asr_engine
            ),
            mt_mode="auto-offline" if req.mt_mode == "opus" else ("nllb" if req.mt_mode == "nllb-1.3b" else req.mt_mode),
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
            elif (
                self.config.source_lang == "eng_Latn"
                and self.model_manager.is_moonshine_en_ready()
                and self.model_manager.is_silero_vad_ready()
            ):
                chosen_asr = "hybrid-en"
            elif self.config.source_lang == "eng_Latn" and self.model_manager.is_moonshine_en_ready():
                chosen_asr = "moonshine-en"
            elif self.model_manager.is_whisper_turbo_ready():
                chosen_asr = "whisper-turbo"
        self._active_asr_label = chosen_asr
        self._remote_asr_label = ""

        # Determine exact minimal model set for this session and unload everything else from RAM!
        is_vi_en_pair = {self.config.source_lang, self.config.target_lang} == {"vie_Latn", "eng_Latn"}
        can_hymt2 = (
            self._llama_server_ready
            and self.model_manager.is_hymt2_ready()
            and bool(get_hymt2_lang_name(self.config.source_lang))
            and bool(get_hymt2_lang_name(self.config.target_lang))
        )
        want_hymt2 = can_hymt2 and (self.config.mt_mode == "hy-mt2")
        keep_asr: set[str] = {chosen_asr}
        if self.config.audio_source == "dual":
            try:
                keep_asr.add(self._resolve_asr_choice_for_lang(self.config.target_lang, "auto"))
            except Exception:
                pass
        keep_mt: set[str] = set()
        if is_vi_en_pair and self.config.mt_mode != "nllb" and self.model_manager.is_envit5_ready():
            keep_mt.add("envit5")
        if want_hymt2:
            keep_mt.add("hy-mt2")
        else:
            need_nllb = (
                self.config.source_lang != self.config.target_lang
                and (
                    self.config.mt_mode == "nllb"
                    or (not is_vi_en_pair and self.config.mt_mode in ("auto-offline", "light-offline", "hy-mt2", "cloud-api"))
                )
            )
            if need_nllb:
                keep_mt.add("nllb-600m")

        keep_pass2: set[str] = set()
        need_vi_pass2 = self.config.pass2_refine and (
            (chosen_asr == "zipformer-30m-streaming" and self.config.source_lang == "vie_Latn")
            or (
                self.config.audio_source == "dual"
                and self.config.target_lang == "vie_Latn"
                and self.model_manager.is_zipformer_streaming_ready()
            )
        )
        need_en_pass2 = (
            self.config.pass2_refine
            and chosen_asr not in ("moonshine-en", "hybrid-en")
            and (self.config.source_lang == "eng_Latn" or (self.config.audio_source == "dual" and self.config.target_lang == "eng_Latn"))
            and self.model_manager.is_moonshine_en_ready()
        )
        if need_vi_pass2:
            keep_pass2.add("vi-pass2")
        if need_en_pass2 or "hybrid-en" in keep_asr:
            keep_pass2.add("en-pass2")

        self._release_unused_models(keep_asr=keep_asr, keep_mt=keep_mt, keep_pass2=keep_pass2)

        if "envit5" in keep_mt:
            self.ensure_envit5_loaded()

        if want_hymt2:
            # Pre-warm Hy-MT2 llama-server in background ONLY when explicitly selected
            def _warmup_hymt2() -> None:
                try:
                    h = self.ensure_hymt2_loaded()
                    if h is not None:
                        h.ensure_started()
                except Exception:
                    pass

            threading.Thread(
                target=_warmup_hymt2,
                daemon=True,
                name="hymt2-warmup",
            ).start()
        elif "nllb-600m" in keep_mt and self.model_manager.is_nllb_ready():
            self.ensure_nllb_loaded()

        if need_vi_pass2:
            self.ensure_pass2_refiner()
        if need_en_pass2:
            self.ensure_en_pass2_refiner()

        # Start real-time auto-save meeting log (.TXT & .SRT)
        self.logger.start_new_session(self.config.source_lang, self.config.target_lang)

        self._stop_event.clear()
        self._mt_wakeup.clear()
        self._draft_wakeup.clear()
        while not self._final_mt_queue.empty():
            try:
                self._final_mt_queue.get_nowait()
            except queue.Empty:
                break
        with self._partial_lock:
            self._pending_partial = None
            self._committed_seg_ids.clear()
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
        self._post_pass2_carry.clear()
        self.glossary.clear_session_entities()

        self._current_segment_id = uuid.uuid4().hex[:10]

        self._mt_thread = threading.Thread(target=self._mt_worker_loop, daemon=True, name="mt-worker")
        self._mt_thread.start()
        self._draft_mt_thread = threading.Thread(target=self._draft_mt_loop, daemon=True, name="draft-mt-worker")
        self._draft_mt_thread.start()

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
            asr = self.ensure_onnx_asr_loaded(chosen_asr, lang_code=self.config.source_lang)
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
                        remote_asr = self.ensure_onnx_asr_loaded(
                            remote_asr_choice,
                            lang_code=remote_lang,
                            remote=True,
                        )
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
        self._draft_wakeup.set()
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
        for th_attr in ("_asr_thread", "_remote_asr_thread", "_mt_thread", "_draft_mt_thread"):
            th = getattr(self, th_attr, None)
            if th is not None and th.is_alive():
                th.join(timeout=2.0)
            setattr(self, th_attr, None)
        self._listening = False
        self._schedule_hymt2_idle_sleep(delay_s=60.0)
        self.broadcast_threadsafe({"type": "level", "data": {"level": 0.0}})
        payload = self.get_status_payload()
        self.broadcast_threadsafe({"type": "status", "data": payload})
        return payload

    def shutdown(self) -> None:
        self._cancel_hymt2_idle_timer()
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
            self._asr_hybrid_en = None
            self._asr_hybrid_en_remote = None
            self._asr_moonshine_en = None
            self._asr_moonshine_en_remote = None
            self._asr_moonshine_ja = None
            self._asr_moonshine_ja_remote = None
            self._asr_moonshine_ko = None
            self._asr_moonshine_ko_remote = None
            self._asr_zf_multi = None
            self._asr_zf_multi_remote = None
            self._asr_parakeet_eu = None
            self._asr_parakeet_eu_remote = None
            self._asr_whisper = None
            self._asr_whisper_remote = None

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
            if not self._is_asr_compatible_with_lang(self.config.asr_engine, req.source_lang):
                self.config.asr_engine = "auto"
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
            next_asr = req.asr_engine
            if next_asr == "zipformer-en-20m" or not self._is_asr_compatible_with_lang(next_asr, self.config.source_lang):
                next_asr = "auto"
            if next_asr != self.config.asr_engine:
                self.config.asr_engine = next_asr
                need_restart = True
        if req.target_lang is not None and req.target_lang != self.config.target_lang:
            self.config.target_lang = req.target_lang
            # Immediately flush prior-language MT context & live draft state so switching
            # target_lang mid-session (without restarting ASR) never bleeds the old language
            # into the first translated utterance or live preview.
            self._recent_context.clear()
            self._stable_prefix.reset()
            with self._partial_lock:
                self._pending_partial = None
            with self._para_lock:
                self._para_tgt_lang = req.target_lang
            if (
                self.config.mt_mode == "hy-mt2"
                and self._nllb is not None
                and bool(get_hymt2_lang_name(self.config.source_lang))
                and bool(get_hymt2_lang_name(req.target_lang))
            ):
                with self._model_lock:
                    self._nllb = None
                gc.collect()
            if self.config.audio_source == "dual":
                need_restart = True
        if req.mt_mode is not None:
            self.config.mt_mode = req.mt_mode
            if req.mt_mode != "hy-mt2" and self._hymt2 is not None:
                self._cancel_hymt2_idle_timer()
                with self._model_lock:
                    if self._hymt2 is not None:
                        try:
                            self._hymt2.stop()
                            getattr(self._hymt2, "close", lambda: None)()
                        except Exception:
                            pass
                        self._hymt2 = None
                gc.collect()
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
        with self._history_lock:
            self._history.clear()
        self._invalidate_audience_cache(None)
        with self._group_lock:
            self._active_group_id = ""
            self._active_group_words = 0
            self._active_group_last_ts = 0.0
            self._active_group_langs = ("", "")
        self._recent_context.clear()
        self._post_pass2_carry.clear()
        self.glossary.clear_session_entities()
        self.broadcast_threadsafe({"type": "history", "data": []})

    def update_history_entry(
        self,
        target_id: str,
        translated_text: str,
        item_ids: Optional[list[str]] = None,
        source_text: Optional[str] = None,
    ) -> bool:
        """
        Apply a Quick-Edit or Gemini-verified improved translation onto a History entry or merged paragraph,
        syncing main UI, LED overlay, OBS, matching-language audience phones, and re-translating for
        other-language audience phones when `source_text` changes.
        """
        ids_set = {str(x) for x in (item_ids or []) if x}
        if target_id:
            ids_set.add(str(target_id))
        if not ids_set:
            return False

        removed_ids: list[str] = []
        with self._history_lock:
            matched_indices = [
                idx
                for idx, e in enumerate(self._history)
                if str(e.get("id", "")) in ids_set or str(e.get("paragraph_id", "")) in ids_set
            ]
            if not matched_indices:
                return False

            first_idx = matched_indices[0]
            if len(matched_indices) == 1:
                entry = dict(self._history[first_idx])
                old_source = str(entry.get("source_text", "")).strip()
                entry["translated_text"] = translated_text.strip()
                if source_text is not None and source_text.strip():
                    entry["source_text"] = source_text.strip()
                self._history[first_idx] = entry
                updated_entry = dict(entry)
            else:
                # Merge grouped paragraph entries cleanly into the first entry so exports & UI never duplicate text
                first = dict(self._history[first_idx])
                old_source = " ".join(
                    str(self._history[i].get("source_text", "")).strip()
                    for i in matched_indices
                    if str(self._history[i].get("source_text", "")).strip()
                ).strip()
                if source_text is not None and source_text.strip():
                    first["source_text"] = source_text.strip()
                elif old_source:
                    first["source_text"] = old_source
                first["translated_text"] = translated_text.strip()
                self._history[first_idx] = first
                for idx in reversed(matched_indices[1:]):
                    rem = self._history.pop(idx)
                    if rem.get("id"):
                        removed_ids.append(str(rem["id"]))
                updated_entry = dict(first)

            history_snap = list(self._history)
            source_changed = str(updated_entry.get("source_text", "")).strip() != old_source

        inv_ids = set(removed_ids)
        if updated_entry.get("id"):
            inv_ids.add(str(updated_entry["id"]))
        self._invalidate_audience_cache(inv_ids)

        # Sync main UI, overlay, `/obs` & `"follow"` phones via full history
        self.broadcast_threadsafe({"type": "history", "data": history_snap})
        # Sync non-`"follow"` phones explicitly set to `entry["target_lang"]`
        self.broadcast_threadsafe(
            {
                "type": "history_patch",
                "data": {"entry": updated_entry, "removed_ids": removed_ids},
            }
        )
        # Re-translate & fan out to other-language phones when source_text changed (or merged)
        if source_changed or removed_ids:
            try:
                fanout_item = dict(updated_entry)
                if removed_ids:
                    fanout_item["_removed_ids"] = removed_ids
                self._fanout_audience_final(fanout_item)
            except Exception as exc:
                logger.warning("Audience fan-out after quick-edit failed: %s", exc)
        return True

    def set_display_freeze(self, frozen: bool, message: Optional[str] = None) -> dict[str, Any]:
        self._display_frozen = bool(frozen)
        if message is not None and message.strip():
            self._display_freeze_message = message.strip()[:160]
        payload = {
            "frozen": self._display_frozen,
            "message": self._display_freeze_message,
        }
        self.broadcast_threadsafe({"type": "display_freeze", "data": payload})
        self.broadcast_threadsafe({"type": "status", "data": self.get_status_payload()})
        return payload

    def evaluate_translations(self, req: EvaluateRequest) -> dict[str, Any]:
        """Evaluate semantic translation quality using saved Google Gemini (or fallback Google Cloud) API key."""
        if req.items:
            raw_items = [it.model_dump() for it in req.items if it.source_text.strip()]
        else:
            with self._history_lock:
                hist_tail = list(self._history[-40:])
            raw_items = [
                {
                    "id": str(e.get("id", "")),
                    "source_text": str(e.get("source_text", "")),
                    "translated_text": str(e.get("translated_text", "")),
                    "source_lang": str(e.get("source_lang", self.config.source_lang)),
                    "target_lang": str(e.get("target_lang", self.config.target_lang)),
                    "timestamp": str(e.get("timestamp", "")),
                }
                for e in hist_tail
                if str(e.get("source_text", "")).strip()
            ]
        if not raw_items:
            raise ValueError("Chưa có câu dịch nào trong lịch sử để đánh giá.")

        gemini_creds = self.cloud_quota.get_credentials("gemini")
        google_creds = self.cloud_quota.get_credentials("google")
        gemini_key = str(gemini_creds.get("api_key", "")).strip()
        google_key = str(google_creds.get("api_key", "")).strip()

        est_chars = sum(len(it["source_text"]) + len(it["translated_text"]) for it in raw_items)

        if gemini_key:
            if not self.cloud_quota.can_spend("gemini", est_chars):
                raise RuntimeError(
                    "Google Gemini API đã chạm giới hạn ký tự tháng này trong mục Cloud API. "
                    "Hãy tăng hạn mức hoặc reset bộ đếm để tiếp tục đánh giá."
                )
            report = evaluate_translations_with_gemini(
                api_key=gemini_key,
                items=raw_items,
                model=str(gemini_creds.get("model") or "gemini-2.5-flash"),
            )
            self.cloud_quota.record_usage("gemini", int(report.get("chars_used", est_chars)))
            report["cloud_api"] = self.cloud_quota.get_public_status()
            return report

        if google_key:
            if not self.cloud_quota.can_spend("google", est_chars):
                raise RuntimeError(
                    "Google Cloud Translation API đã chạm giới hạn ký tự miễn phí tháng này."
                )
            report = evaluate_translations_with_google_cloud(
                api_key=google_key,
                items=raw_items,
            )
            self.cloud_quota.record_usage("google", int(report.get("chars_used", est_chars)))
            report["cloud_api"] = self.cloud_quota.get_public_status()
            return report

        raise LookupError("missing_google_api_key")

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
                if nxt and nxt[0].islower():
                    nxt = nxt[0].upper() + nxt[1:]
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
                # Avoid awkward commas after connectors ("Liệu rằng, thành công" / "to the, cloud"),
                # before Vietnamese relative/subordinating linkers ("điều, mà"), or 1-2 word tails ("triệu, đồng")
                nxt_first_low = first_clean.lower()
                no_comma_before = nxt_first_low in {
                    "mà", "rằng", "là", "của", "cho", "về", "với", "trong", "để", "nên", "thì",
                }
                if cls._ends_with_connector(merged) or no_comma_before or len(nxt_norm.split()) <= 2:
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
        if source_lang.startswith(_NON_SPACED_SCRIPT_PREFIXES):
            return raw

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
        elif source_lang == "eng_Latn":
            # Polish common STEM dimension tokens (e.g. "3d world" -> "3D world", "2d screen" -> "2D screen")
            res = re.sub(r"\b([234])d\b", r"\1D", res)
        if had_leading_cap and res and res[0].islower():
            res = res[0].upper() + res[1:]
        return res

    @staticmethod
    def _capitalize_sentence_start(text: str) -> str:
        """Capitalize sentence-initial lowercase letter while preserving camelCase brands (iPhone, iOS, macOS)."""
        s = (text or "").strip()
        if not s or not s[0].islower():
            return s
        first_tok = s.split(None, 1)[0].strip(".,!?;:\"'()[]")
        if len(first_tok) > 1 and any(ch.isupper() for ch in first_tok[1:]):
            return s
        return s[0].upper() + s[1:]

    @staticmethod
    def _ends_with_open_en_modifier(text: str) -> bool:
        """
        Detect unpunctuated English fragments ending mid-noun-phrase on `<determiner> <adjective/adverb>`
        (e.g. '...and enter a vibrant' before 'physical 3D world.'), so brief VAD pauses don't split
        an adjective from its noun and make MT hallucinate a missing noun.
        """
        stripped = (text or "").strip()
        if not stripped or stripped.endswith((".", "!", "?", "…")):
            return False
        words = stripped.rstrip(",;: ").split()
        if len(words) < 2:
            return False
        det = words[-2].lower().strip("\"'()[]")
        mod = words[-1].lower().strip("\"'()[],;:")
        if det not in {
            "a", "an", "the", "this", "that", "these", "those",
            "my", "our", "your", "their", "his", "her", "its",
        }:
            return False
        if not mod or not mod.isalpha():
            return False
        open_mods = {
            "new", "old", "big", "small", "large", "great", "good", "bad", "high", "low",
            "long", "short", "fast", "slow", "smart", "deep", "wide", "full", "whole",
            "next", "first", "last", "same", "other", "real", "true", "main", "key",
            "major", "minor", "modern", "global", "national", "human", "physical",
            "digital", "virtual", "vibrant", "brilliant", "brave", "separate", "complex",
            "simple", "single", "double", "special", "specific", "certain", "entire",
            "unique", "common", "clear", "strong", "weak", "early", "late", "young",
            "free", "open", "safe", "clean", "pure", "rich", "poor", "dark", "bright",
            "sharp", "smooth", "soft", "hard", "heavy", "huge", "vast", "tiny", "massive",
            "rapid", "quick", "steady", "broad", "narrow", "direct", "exact", "complete",
            "proper", "solid", "stable", "very", "more", "most", "really", "truly",
            "highly", "deeply", "largely", "totally", "fully", "nearly", "almost",
            "important", "different", "powerful", "beautiful", "wonderful", "amazing",
            "exciting", "surprising", "interesting", "intelligent", "autonomous",
            "mechanical", "electrical", "chemical", "biological", "medical", "technical",
            "practical", "academic", "scientific", "economic", "social", "cultural",
            "political", "historical", "environmental", "educational", "international",
            "largest", "smallest", "biggest", "greatest", "highest", "lowest", "best",
            "worst", "fastest", "smartest", "strongest", "richest", "deepest",
        }
        if mod in open_mods:
            return True
        if len(mod) >= 5 and mod.endswith(("ous", "ful", "less")):
            return True
        if len(mod) >= 6 and mod.endswith(("able", "ible")) and mod not in {"vegetable", "timetable", "syllable"}:
            return True
        return False

    @staticmethod
    def _ends_with_connector(text: str) -> bool:
        words = text.lower().rstrip(".,!?;: ").split()
        if not words:
            return False
        single_connectors = {
            "và", "hoặc", "nhưng", "mà", "thì", "là", "của", "vì", "nếu", "để",
            "nên", "với", "cho", "trong", "khi", "cũng", "hay", "như", "tại",
            "cái", "những", "các", "một", "đôi", "sẽ", "đang", "có", "bị", "được",
            "tận", "cả", "đối", "rằng", "liệu", "tuy", "dù", "mặc", "bởi", "chứ",
            "giữa", "bằng", "về", "từ", "sau", "trước", "nhằm", "sang", "vào", "ra",
            "lên", "xuống", "so", "theo", "qua", "bên", "ngoài",
            "phải", "cần", "đã", "luôn", "vẫn", "chỉ", "càng",
            "and", "but", "so", "to", "that", "because", "with", "the", "of", "in",
            "for", "or", "a", "an", "between", "from", "into", "about", "through",
        }
        if words[-1] in single_connectors:
            return True
        if len(words) >= 2:
            last2 = f"{words[-2]} {words[-1]}"
            if last2 in {
                "nhưng mà", "để mà", "có thể", "tức là", "gọi là", "bất kể",
                "cũng như", "hay là", "cho nên", "sẽ có", "có cái", "thì đôi",
                "so với", "như là", "khi mà", "đối với", "từ phía", "về phía",
                "giữa các", "tách biệt", "liên quan", "dựa trên",
                "luôn phải", "cần phải", "sẽ phải", "đang phải", "vẫn đang", "không thể",
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
        if n <= min_words:
            return text, ""

        # 0) If a long Vietnamese clause opens a contrastive/conditional sub-clause in its last 2..6 words
        #    without a closing "thì"/"nên" yet (e.g. "...có hạnh phúc, thế nhưng không có tiền"),
        #    split cleanly right BEFORE that opener so both halves translate as complete sentences!
        if n >= min_words + 4:
            open_pairs = {
                ("thế", "nhưng"),
                ("nhưng", "mà"),
                ("tuy", "nhiên"),
                ("nếu", "như"),
                ("và", "khi"),
            }
            search_start = max(min_words, n - 7)
            for i in range(n - 2, search_start - 1, -1):
                w_a = words[i].lower().strip(".,!?;:\"'()[]")
                w_b = words[i + 1].lower().strip(".,!?;:\"'()[]")
                if (w_a, w_b) in open_pairs:
                    tail_words = words[i:]
                    tail_low = {w.lower().strip(".,!?;:\"'()[]") for w in tail_words[2:]}
                    if 2 <= len(tail_words) <= 7 and "thì" not in tail_low and "nên" not in tail_low:
                        head = " ".join(words[:i]).rstrip(" ,;")
                        tail = " ".join(tail_words).strip()
                        if head and tail:
                            return head, tail

        if not cls._ends_with_connector(text):
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
        If an in-progress speech draft ends on a dangling connector/determiner/auxiliary
        (e.g. '...nhưng mà', '...to this', '...my', '...because'), strip the trailing dangling
        words ONLY from the string sent to partial MT (keep them visible in live source transcript).
        This prevents neural MT from hallucinating fake continuations like "bài kiểm tra này".
        """
        words = text.strip().split()
        if len(words) < 3:
            return text.strip()
        extra_draft_dangling = {
            "this", "that", "these", "those", "my", "your", "our", "their", "its", "his", "her",
            "very", "more", "most", "just", "will", "can", "could", "would", "should", "may", "might",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "about", "into",
            "from", "at", "by", "on", "as", "if", "when", "while", "which", "who",
        }
        cut = len(words)
        while cut > 2 and (len(words) - cut) < 4:
            tail_w = words[cut - 1].lower().strip(".,!?;:\"'()[]")
            if tail_w in extra_draft_dangling or cls._ends_with_connector(" ".join(words[:cut])):
                cut -= 1
            else:
                break
        trimmed = " ".join(words[:cut]).rstrip(" ,;:-")
        return trimmed if trimmed else text.strip()

    def _paragraph_thresholds(self, src_lang: Optional[str] = None) -> tuple[int, int, float]:
        """
        Conference Semantic Sentence Commit thresholds (Backend Translation Unit):
        Commit crisp YouTube-paced clauses so Pass-2 & MT run fast (~550-850ms) and smoothly,
        while UI merges consecutive sentences into clean paragraphs.
        """
        mode = self.config.paragraph_mode
        lang = src_lang or self.config.source_lang
        if lang == "eng_Latn":
            if mode == "short":
                return (5, 11, 0.78)
            if mode == "medium":
                return (7, 15, 0.92)
            # Crisp 8-18 word YouTube-paced English clauses (4-7.5s); UI groups 3-4 into paragraphs
            return (8, 18, 1.05)
        if mode == "short":
            return (7, 15, 0.95)
        if mode == "medium":
            return (10, 22, 1.15)
        return (12, 28, 1.35)

    @staticmethod
    def _apply_post_pass2_carry(text: str, prev_carry: str, allow_peel: bool = True) -> tuple[str, str]:
        """
        Text-Only Dangling-Tail Carry-Over after English Pass-2 Moonshine-v2:
        1. Prepends `prev_carry` cleanly (deduplicating 1-2 overlapping seam words).
        2. Peels off trailing dangling clause openers AFTER a sentence period (e.g.
           '...naturally over time. And to' -> head='...naturally over time.', carry='And to')
           OR trailing unfinished connectors ('...my 3000 most' -> carry='most') so zero
           words are cut across segment boundaries and ZERO audio is ever duplicated!
        """
        s = (text or "").strip()
        if prev_carry:
            c_toks = prev_carry.strip().split()
            s_toks = s.split()
            if c_toks and s_toks:
                norm = lambda w: re.sub(r"[^a-z0-9']+", "", w.lower())
                # Deduplicate 1-2 overlapping words at the seam ("And to" + "to build" -> "And to build")
                overlap = 0
                for k in (2, 1):
                    if len(c_toks) >= k and len(s_toks) >= k:
                        if [norm(w) for w in c_toks[-k:]] == [norm(w) for w in s_toks[:k]]:
                            overlap = k
                            break
                # Keep lowercase join if prev_carry didn't end a sentence
                tail_toks = s_toks[overlap:]
                if tail_toks and not prev_carry.endswith((".", "!", "?")) and tail_toks[0][0].isupper():
                    w0 = tail_toks[0]
                    if len(w0) > 1 and w0[1:].islower() and w0 not in {"I", "I'm", "I've", "I'll", "I'd", "English"}:
                        tail_toks[0] = w0[0].lower() + w0[1:]
                s = " ".join(c_toks + tail_toks).strip()
            elif c_toks:
                s = prev_carry.strip()

        words = s.split()
        if not allow_peel or len(words) < 6:
            return s, ""

        # Case 1: Complete sentence ending in .?! followed by 1..4 short unpunctuated trailing words
        # e.g. "...over time. And to" / "...Let's get started. If you" / "...in English. Today,"
        for i in range(len(words) - 1, max(3, len(words) - 5), -1):
            if words[i - 1].endswith((".", "!", "?")) and not words[-1].endswith((".", "!", "?")):
                tail_words = words[i:]
                if 1 <= len(tail_words) <= 4:
                    head = " ".join(words[:i]).strip()
                    carry = " ".join(tail_words).strip()
                    return head, carry

        # Case 2: Unpunctuated tail ending on a clear dangling English connector / determiner / auxiliary
        if not words[-1].endswith((".", "!", "?")):
            dangling = {
                "and", "or", "but", "so", "to", "for", "of", "in", "on", "at", "by", "from",
                "with", "about", "into", "if", "when", "while", "because", "that", "which",
                "the", "a", "an", "my", "your", "our", "their", "this", "these", "those",
                "most", "more", "very", "can", "could", "will", "would", "should", "between",
            }
            cut = len(words)
            while cut > 5 and (len(words) - cut) < 3:
                w_low = re.sub(r"[^a-z']+", "", words[cut - 1].lower())
                if w_low in dangling:
                    cut -= 1
                else:
                    break
            if cut < len(words):
                head = " ".join(words[:cut]).rstrip(" ,;")
                carry = " ".join(words[cut:]).strip()
                if head and carry:
                    return head, carry

        return s, ""

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

    def _flush_paragraph_locked(self, allow_carry_over: bool = False, is_stop_flush: bool = False) -> None:
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

        # Snapshot paragraph PCM for Pass-2 VietASR punctuation/ITN refinement:
        # `_merge_pass2_punct_and_itn` aligns strictly onto Pass-1 `full_text` and ignores
        # extra trailing words, so Pass-2 is safe even when `carry_tail` starts the next turn!
        # Keep the last ~1.8s (18 chunks) of audio for the carried-over start of the next paragraph.
        pcm_snapshot: Optional[np.ndarray] = None
        carry_pcm_tail: list[np.ndarray] = []
        if self._para_pcm_chunks and not self._para_pcm_truncated:
            try:
                pcm_snapshot = np.concatenate(self._para_pcm_chunks)
            except Exception:
                pcm_snapshot = None
            if carry_tail:
                carry_pcm_tail = list(self._para_pcm_chunks[-18:])

        self._para_pcm_chunks.clear()
        if carry_pcm_tail:
            self._para_pcm_chunks.extend(carry_pcm_tail)
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
            self._committed_seg_ids.append(seg_id)
        self._stable_prefix.reset()

        if not full_text:
            return

        prot_terms = set(self._get_glossary_hotwords())
        display_text = self._capitalize_sentence_start(
            self._clean_verbatim(
                self.glossary.apply_asr_fixes(full_text),
                src_lang,
                protected_terms=prot_terms,
            )
        )
        # Drop orphaned 1-2 word dangling connectors (e.g. "Là", "So với") so LED/History
        # never display meaningless stray fragments when speech pauses or stops mid-clause.
        if len(display_text.split()) <= 2 and self._ends_with_connector(display_text):
            return
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
        self._enqueue_final_mt(
            {
                "id": seg_id,
                "source_text": display_text,
                "source_lang": src_lang,
                "target_lang": tgt_lang,
                "asr_ms": asr_ms,
                "pcm": pcm_snapshot,
                "speaker": speaker_tag,
                "is_stop_flush": is_stop_flush,
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

    def _enqueue_final_mt(self, item: dict[str, Any]) -> None:
        """Bounded FIFO enqueue for final MT jobs: drops oldest item with warning on overflow to cap RAM & latency."""
        try:
            self._final_mt_queue.put_nowait(item)
        except queue.Full:
            dropped = None
            try:
                dropped = self._final_mt_queue.get_nowait()
            except queue.Empty:
                pass
            logger.warning(
                "Final MT queue full (maxsize=%d); dropped oldest queued utterance %s to bound RAM/latency.",
                self._final_mt_queue.maxsize,
                dropped.get("id") if isinstance(dropped, dict) else "?",
            )
            try:
                self._final_mt_queue.put_nowait(item)
            except queue.Full:
                pass

    def _flush_paragraph_now(self, is_stop_flush: bool = True) -> None:
        with self._para_lock:
            self._flush_paragraph_locked(allow_carry_over=False, is_stop_flush=is_stop_flush)

    def _flush_paragraph_if_idle(self) -> None:
        _, _, pause_flush_s = self._paragraph_thresholds()
        with self._para_lock:
            if not self._para_parts or self._para_last_speech_ts <= 0.0:
                return
            if (time.monotonic() - self._para_last_speech_ts) >= pause_flush_s:
                self._flush_paragraph_locked(allow_carry_over=False)

    def _enqueue_asr_output(self, out: ASROutput, is_stop_flush: bool = False) -> None:
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
                    self._draft_wakeup.set()
                return

            if out.kind == "final":
                self._para_has_active_partial = False
                self._para_parts.append(out.text)
                combined = self._join_fragments(self._para_parts, preserve_case=self._get_capitalized_glossary_words())
                wc = len(combined.split())
                is_non_spaced = src_lang.startswith(_NON_SPACED_SCRIPT_PREFIXES)
                is_short_unpunctuated_cjk = (
                    (out.provider.startswith("whisper-turbo") or out.provider == "moonshine-v2-ja")
                    and is_non_spaced
                    and not is_stop_flush
                    and len(combined.strip()) <= 6
                    and not combined.strip().endswith(("。", "！", "？", ".", "!", "?", "…"))
                )
                is_short_ko = (
                    out.provider == "moonshine-v2-ko"
                    and not is_stop_flush
                    and wc < 4
                    and not combined.rstrip().endswith((".", "!", "?", "…"))
                )
                is_incomplete_en = (
                    out.provider == "hybrid-en-vad-moonshine"
                    and not is_stop_flush
                    and wc < 24
                    and not combined.rstrip().endswith((".", "!", "?", "…"))
                    and (self._ends_with_connector(combined) or self._ends_with_open_en_modifier(combined))
                )
                is_sentence_asr = (
                    ((out.provider.startswith("whisper-turbo") or out.provider == "moonshine-v2-ja") and not is_short_unpunctuated_cjk)
                    or (out.provider == "moonshine-v2-ko" and not is_short_ko)
                    or out.provider == "moonshine-v2-en"
                    or (out.provider == "parakeet-tdt-0.6b-ultra" and (wc >= 5 or is_stop_flush))
                    or (out.provider == "hybrid-en-vad-moonshine" and wc >= 5 and not is_incomplete_en)
                )
                if is_sentence_asr:
                    # Moonshine-v2 & Whisper-Turbo already commit clean punctuated sentences -> flush immediately!
                    self._flush_paragraph_locked(allow_carry_over=False, is_stop_flush=is_stop_flush)
                    return

                if is_non_spaced:
                    stripped = combined.strip()
                    if len(stripped) >= 14 or stripped.endswith(("。", "！", "？", ".", "!", "?")):
                        self._flush_paragraph_locked(allow_carry_over=False)
                        return
                    if is_short_unpunctuated_cjk:
                        # Hold ultra-short CJK fragment (e.g. "今私は" / "しりとりとは") 1 beat so it joins cleanly
                        seg_id = self._current_segment_id
                        self.broadcast_threadsafe(
                            {
                                "type": "partial",
                                "data": {
                                    "id": seg_id,
                                    "source_text": stripped,
                                    "source_lang": src_lang,
                                    "target_lang": tgt_lang,
                                    "asr_ms": out.latency_ms,
                                },
                            }
                        )
                        return

                min_words, target_words, _ = self._paragraph_thresholds()

                should_commit = (
                    not is_incomplete_en
                    and (
                        wc >= target_words
                        or (wc >= min_words and not self._ends_with_connector(combined))
                        or (
                            wc >= 6
                            and combined.endswith((".", "!", "?"))
                            and not self._ends_with_connector(combined)
                        )
                    )
                )

                if should_commit:
                    # For English with Moonshine-v2 Pass-2, keep full PCM snapshot (allow_carry_over=False)
                    # and let Post-Pass-2 `_apply_post_pass2_carry` handle clean text-only boundary carry!
                    pre_carry = (not is_non_spaced) and (src_lang != "eng_Latn")
                    self._flush_paragraph_locked(allow_carry_over=pre_carry)
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
                        self._draft_wakeup.set()

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
                    if can_gate and hasattr(asr, "is_speech_active") and asr.is_speech_active():
                        can_gate = False
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
                self._enqueue_asr_output(flushed, is_stop_flush=True)
            except Exception:
                pass
            self._flush_paragraph_now(is_stop_flush=True)

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
        self._enqueue_final_mt(
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
                is_short_ko = (
                    out.provider == "moonshine-v2-ko"
                    and wc < 4
                    and not combined.rstrip().endswith((".", "!", "?", "…"))
                )
                is_incomplete_en = (
                    out.provider == "hybrid-en-vad-moonshine"
                    and wc < 24
                    and not combined.rstrip().endswith((".", "!", "?", "…"))
                    and (self._ends_with_connector(combined) or self._ends_with_open_en_modifier(combined))
                )
                is_sentence_asr = (
                    out.provider.startswith("whisper-turbo")
                    or out.provider == "moonshine-v2-en"
                    or out.provider == "moonshine-v2-ja"
                    or (out.provider == "moonshine-v2-ko" and not is_short_ko)
                    or (out.provider == "parakeet-tdt-0.6b-ultra" and wc >= 5)
                    or (out.provider == "hybrid-en-vad-moonshine" and wc >= 5 and not is_incomplete_en)
                )
                is_non_spaced = r_src.startswith(_NON_SPACED_SCRIPT_PREFIXES)
                if is_sentence_asr:
                    self._remote_flush_locked()
                    return
                if is_non_spaced:
                    stripped = combined.strip()
                    if len(stripped) >= 14 or stripped.endswith(("。", "！", "？", ".", "!", "?")):
                        self._remote_flush_locked()
                        return
                min_words, target_words, _ = self._paragraph_thresholds()
                if not is_incomplete_en and (
                    wc >= target_words or (wc >= min_words and not self._ends_with_connector(combined))
                ):
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
                    if can_gate and hasattr(asr, "is_speech_active") and asr.is_speech_active():
                        can_gate = False
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
        """Dedicated Final MT Worker (Pass-2 + Hy-MT2/EnViT5/NLLB/Cloud -> History + Logs + Audience)."""
        while not self._stop_event.is_set() or not self._final_mt_queue.empty():
            if not self._stop_event.is_set():
                self._flush_paragraph_if_idle()
                self._remote_flush_if_idle()
            try:
                item = self._final_mt_queue.get_nowait()
                # Drop any stale draft belonging to the exact utterance now being finalized
                seg_id_str = str(item.get("id") or "")
                with self._partial_lock:
                    if seg_id_str and seg_id_str not in self._committed_seg_ids:
                        self._committed_seg_ids.append(seg_id_str)
                    if self._pending_partial is not None and self._pending_partial.get("id") == item.get("id"):
                        self._pending_partial = None
                self._stable_prefix.clear_if(seg_id_str)
                cfg = self.config
                source_text = item["source_text"]
                speaker = item.get("speaker", "")
                active_tgt = item["target_lang"]
                if speaker != "remote" and not item.get("is_stop_flush", False) and cfg.audio_source != "dual":
                    active_tgt = cfg.target_lang
                asr_label = self._remote_asr_label if speaker == "remote" else self._active_asr_label
                pass2_ms = 0.0

                # Pass-2 Async Refinement + Final MT under `_final_mt_busy` so `_draft_mt_loop`
                # yields CPU cores during Pass-2 as well (cuts Pass-2 latency 30-50% on Windows CPU laptops!).
                translated = ""
                mt_ms = 0.0
                mt_engine = cfg.mt_mode
                para_id = ""
                self._final_mt_busy = True
                try:
                    if cfg.pass2_refine and item.get("pcm") is not None:
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
                        elif item["source_lang"] == "eng_Latn" and asr_label not in ("moonshine-en", "hybrid-en"):
                            try:
                                en_refiner = self.ensure_en_pass2_refiner()
                                if en_refiner is not None:
                                    refined, pass2_ms = en_refiner.refine(item["pcm"], source_text)
                                    if refined:
                                        source_text = refined
                            except Exception:
                                pass

                    # Post-Pass-2 Text-Only Dangling-Tail Carry-Over for English (zero audio overlap, 0% duplicates!)
                    if item["source_lang"] == "eng_Latn" and asr_label != "moonshine-en":
                        ckey = (speaker, item["source_lang"])
                        prev_c = self._post_pass2_carry.pop(ckey, "")
                        source_text, next_c = self._apply_post_pass2_carry(
                            source_text,
                            prev_c,
                            allow_peel=not bool(item.get("is_stop_flush", False)),
                        )
                        if next_c:
                            self._post_pass2_carry[ckey] = next_c

                    source_text = self._clean_verbatim(
                        self.glossary.apply_asr_fixes(source_text),
                        item["source_lang"],
                        protected_terms=set(self._get_glossary_hotwords()),
                    )
                    source_text = self.glossary.normalize_session_entities(source_text, item["source_lang"])
                    source_text = self._capitalize_sentence_start(source_text)
                    eff_wc = (
                        max(len(source_text.split()), len(source_text) // 4)
                        if item["source_lang"].startswith(_NON_SPACED_SCRIPT_PREFIXES)
                        else len(source_text.split())
                    )
                    para_id = self._assign_paragraph_id(
                        eff_wc,
                        item["source_lang"],
                        active_tgt,
                        speaker=speaker,
                    )

                    res = self.translate_text(
                        source_text,
                        item["source_lang"],
                        active_tgt,
                        mt_mode=cfg.mt_mode,
                        beam_size=cfg.beam_size,
                        llm_url=cfg.llm_url,
                        llm_model=cfg.llm_model,
                        llm_api_key=cfg.llm_api_key,
                        context=list(self._recent_context),
                        is_draft=False,
                        update_slide_state=True,
                    )
                    translated = res.translated_text
                    mt_ms = res.latency_ms
                    mt_engine = res.engine
                except Exception as exc:
                    self.broadcast_threadsafe({"type": "error", "data": {"message": f"Lỗi dịch: {exc}"}})
                finally:
                    self._final_mt_busy = False

                if (
                    source_text
                    and translated
                    and item["source_lang"] == self.config.source_lang
                    and active_tgt == self.config.target_lang
                ):
                    self._recent_context.append((source_text, translated))

                entry = {
                    "id": item["id"],
                    "paragraph_id": para_id,
                    "speaker": speaker,
                    "source_text": source_text,
                    "translated_text": translated,
                    "source_lang": item["source_lang"],
                    "target_lang": active_tgt,
                    "asr_ms": item["asr_ms"],
                    "pass2_ms": pass2_ms,
                    "mt_ms": mt_ms,
                    "mt_engine": mt_engine,
                    "timestamp": item["timestamp"],
                }
                with self._history_lock:
                    self._history.append(entry)
                    if len(self._history) > 300:
                        self._history = self._history[-300:]
                try:
                    self.logger.append_entry(entry)
                    entry["log_saved_count"] = self.logger.get_status()["saved_count"]
                except Exception:
                    pass
                self.broadcast_threadsafe({"type": "final", "data": entry})
                try:
                    self._fanout_audience_final(entry)
                except Exception as exc:
                    logger.warning("Audience fan-out enqueue failed: %s", exc)
                continue
            except queue.Empty:
                pass

            if self._stop_event.is_set():
                break

            self._mt_wakeup.wait(timeout=0.1)
            self._mt_wakeup.clear()

    def _draft_mt_loop(self) -> None:
        """
        Decoupled Live-Draft MT Worker (YouTube-Grade Word-by-Word Streaming):
        Runs completely in parallel with Final Hy-MT2 GPU translations so live EnViT5 drafts
        NEVER freeze behind a 700ms-2000ms final sentence, eliminating Spotlight spinners!
        """
        last_partial_time = 0.0
        last_partial_key: tuple[str, str, str] = ("", "", "")
        last_partial_seg_id: str = ""
        last_partial_wc: int = 0
        last_draft_mt_ms = 0.0

        while not self._stop_event.is_set():
            partial_job = None
            with self._partial_lock:
                if self._pending_partial is not None:
                    partial_job = self._pending_partial
                    self._pending_partial = None

            if partial_job is None:
                self._draft_wakeup.wait(timeout=0.1)
                self._draft_wakeup.clear()
                continue

            cfg = self.config
            seg_id = str(partial_job["id"])
            if cfg.audio_source != "dual" and partial_job["target_lang"] != cfg.target_lang:
                continue
            with self._partial_lock:
                if seg_id in self._committed_seg_ids:
                    continue

            # Always yield draft MT while any Final sentence is translating or queued (including Hy-MT2 single-slot!)
            # so Finals never wait behind live drafts!
            if self._final_mt_busy or not self._final_mt_queue.empty():
                with self._partial_lock:
                    if self._pending_partial is None and seg_id not in self._committed_seg_ids:
                        self._pending_partial = partial_job
                self._draft_wakeup.wait(timeout=0.08)
                self._draft_wakeup.clear()
                continue

            draft_src = self._trim_dangling_tail_for_draft(partial_job["source_text"])
            key = (draft_src, partial_job["source_lang"], partial_job["target_lang"])
            if not draft_src or key == last_partial_key:
                continue

            wc_now = (
                max(len(draft_src.split()), len(draft_src) // 4)
                if partial_job["source_lang"].startswith(_NON_SPACED_SCRIPT_PREFIXES)
                else len(draft_src.split())
            )
            now = time.perf_counter()
            same_seg = seg_id == last_partial_seg_id
            if not same_seg:
                # Never carry an old latency spike into a brand-new sentence
                last_draft_mt_ms = min(last_draft_mt_ms, 450.0)
            elapsed_s = now - last_partial_time

            is_hymt2_non_vien = (
                cfg.mt_mode == "hy-mt2"
                and {partial_job["source_lang"], partial_job["target_lang"]} != {"eng_Latn", "vie_Latn"}
            )
            # Calm, readable YouTube cadence: ~0.45s normal (0.75s Eco), gracefully capped at 1.1s under load.
            # When Hy-MT2 handles live non-VI/EN drafts on the shared llama-server slot, widen debounce (>= 0.90s)
            # and skip prior-turn context on drafts (~40% faster, 0 context-leak retries) so Finals never stall.
            base_debounce = 0.75 if cfg.eco_mode else 0.45
            if is_hymt2_non_vien:
                base_debounce = max(base_debounce, 0.90)
            if last_draft_mt_ms > 550.0:
                cap_s = 1.25 if is_hymt2_non_vien else 1.10
                base_debounce = min(cap_s, max(base_debounce, (last_draft_mt_ms / 1000.0) * 0.85))

            if not same_seg:
                min_first_wc = 2 if partial_job["source_lang"].startswith(_NON_SPACED_SCRIPT_PREFIXES) else 3
                need_wait = wc_now < min_first_wc and elapsed_s < 0.55
            else:
                min_delta = 3 if (cfg.eco_mode or is_hymt2_non_vien) else 2
                need_wait = elapsed_s < base_debounce or (
                    (wc_now - last_partial_wc) < min_delta and elapsed_s < (1.25 if is_hymt2_non_vien else 0.95)
                )

            if need_wait:
                with self._partial_lock:
                    if self._pending_partial is None and seg_id not in self._committed_seg_ids:
                        self._pending_partial = partial_job
                self._draft_wakeup.wait(timeout=0.10)
                self._draft_wakeup.clear()
                continue

            try:
                draft_context = [] if is_hymt2_non_vien else list(self._recent_context)
                r = self.translate_text(
                    draft_src,
                    partial_job["source_lang"],
                    partial_job["target_lang"],
                    mt_mode=cfg.mt_mode,
                    beam_size=1,
                    llm_url=cfg.llm_url,
                    llm_model=cfg.llm_model,
                    llm_api_key=cfg.llm_api_key,
                    context=draft_context,
                    is_draft=True,
                )

                last_partial_time = time.perf_counter()
                last_partial_key = key
                last_partial_seg_id = seg_id
                last_partial_wc = wc_now
                last_draft_mt_ms = r.latency_ms

                # Guard against race: if this segment committed or target_lang changed while draft MT ran, drop stale draft
                if self.config.audio_source != "dual" and partial_job["target_lang"] != self.config.target_lang:
                    continue
                with self._partial_lock:
                    if seg_id in self._committed_seg_ids:
                        continue

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
        elif src == "tha_Thai":
            sample = "สวัสดีครับทุกท่าน วันนี้เรามาประชุมร่วมกันเรื่องความร่วมมือด้านการค้าระหว่างไทยและเวียดนาม"
        elif src == "lao_Laoo":
            sample = "ສະບາຍດີທຸກທ່ານ ມື້ນີ້ພວກເຮົາມາປະຊຸມຮ່ວມກັນກ່ຽວກັບການຮ່ວມມືລະຫວ່າງລາວແລະຫວຽດນາມ"
        elif src == "jpn_Jpan":
            sample = "皆様こんにちは。本日はベトナムと日本の技術協力について打ち合わせを行います。"
        elif src in ("zho_Hans", "zho_Hant"):
            sample = "大家好，今天我们一起讨论越南与中国在人工智能领域的合作项目。"
        elif src == "kor_Hang":
            sample = "안녕하세요 여러분, 오늘은 베트남과 한국의 협력 프로젝트에 대해 논의하겠습니다."
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
                    mt = self.translate_text(
                        partial_src,
                        src,
                        tgt,
                        mt_mode=self.config.mt_mode,
                        beam_size=1,
                        is_draft=True,
                        update_slide_state=False,
                    )
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
                mt = self.translate_text(
                    sample,
                    src,
                    tgt,
                    mt_mode=self.config.mt_mode,
                    beam_size=self.config.beam_size,
                    update_slide_state=False,
                )
                tr_final = mt.translated_text or sample  # fallback: show source if MT empty
                mt_ms = mt.latency_ms
                mt_engine = mt.engine
            except Exception:
                tr_final = sample  # fallback: show source text instead of blank

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
            with self._history_lock:
                self._history.append(entry)
                if len(self._history) > 300:
                    self._history = self._history[-300:]
            self.broadcast_threadsafe({"type": "final", "data": entry})
            try:
                self._fanout_audience_final(entry)
            except Exception as exc:
                logger.warning("Demo audience fan-out failed: %s", exc)

        threading.Thread(target=_run_demo, daemon=True, name="demo-sim").start()
        return {"ok": True, "source_lang": src, "target_lang": tgt}


runtime = EngineRuntime()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    runtime.set_loop(asyncio.get_running_loop())
    runtime.warmup_background()
    yield
    runtime.shutdown()


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient", "::ffff:127.0.0.1"})
_PUBLIC_HTTP_PATHS = frozenset({"/audience", "/obs", "/school-logo.png", "/favicon.ico"})
_PUBLIC_WS_PATHS = frozenset({"/ws/audience"})


def _is_loopback_host(host: Optional[str]) -> bool:
    return (host or "").strip().lower() in _LOOPBACK_HOSTS


class LoopbackGuardMiddleware:
    """Restrict admin `/api/*` and `/ws` routes strictly to loopback clients.
    Non-loopback LAN clients may ONLY access `GET /audience`, `GET /obs`,
    `GET /school-logo.png`, `GET /favicon.ico`, and WS `/ws/audience`.
    """

    def __init__(self, asgi_app: Any) -> None:
        self.app = asgi_app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        scope_type = scope.get("type")
        if scope_type not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        client_host = client[0] if client and len(client) >= 1 else ""
        if _is_loopback_host(client_host):
            await self.app(scope, receive, send)
            return

        path = scope.get("path") or ""
        if scope_type == "http":
            method = (scope.get("method") or "GET").upper()
            if method in ("GET", "HEAD") and path in _PUBLIC_HTTP_PATHS:
                await self.app(scope, receive, send)
                return
            resp = JSONResponse(
                {"detail": "Forbidden: admin endpoint is restricted to localhost"},
                status_code=403,
            )
            await resp(scope, receive, send)
            return

        if path in _PUBLIC_WS_PATHS:
            await self.app(scope, receive, send)
            return
        await send(
            {
                "type": "websocket.close",
                "code": 1008,
                "reason": "Admin WebSocket restricted to localhost",
            }
        )


app = FastAPI(title="VOICERECORE Local Engine", version="1.1.2", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoopbackGuardMiddleware)


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
        "nllb-600m",
        "zipformer-30m-streaming",
        "zipformer-multi",
        "moonshine-en",
        "moonshine-ja",
        "moonshine-ko",
        "parakeet-eu",
        "whisper-turbo",
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
    with runtime._history_lock:
        hist_snap = list(runtime._history)
    return summarize_meeting_entries(
        hist_snap,
        runtime.config.source_lang,
        runtime.config.target_lang,
        translate_fn=lambda txt, s, t: runtime.translate_text(
            txt, s, t, mt_mode=runtime.config.mt_mode, beam_size=1, update_slide_state=False
        ),
    )


@app.post("/api/session/evaluate")
def api_session_evaluate(req: EvaluateRequest):
    try:
        return runtime.evaluate_translations(req)
    except LookupError:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "missing_google_api_key",
                "message": "Chưa cấu hình Google Gemini (hoặc Google Cloud) API Key. Hãy dán Key trong mục Cài đặt Cloud API.",
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi đánh giá qua Google API: {exc}") from exc


@app.post("/api/history/update")
def api_history_update(req: HistoryUpdateRequest):
    updated = runtime.update_history_entry(
        target_id=req.id,
        translated_text=req.translated_text,
        item_ids=req.item_ids,
        source_text=req.source_text,
    )
    return {"ok": True, "updated": updated}


class SlideContextTextRequest(BaseModel):
    text: Optional[str] = None
    topic: Optional[str] = None
    terms: Optional[list[dict[str, Any]]] = None
    enabled: Optional[bool] = None
    apply_dau_preset: bool = False
    active_slide_index: Optional[int] = None


class DisplayFreezeRequest(BaseModel):
    frozen: bool
    message: Optional[str] = None


@app.get("/api/context")
def api_get_context():
    return runtime.slide_context.get_payload()


@app.post("/api/context/upload")
async def api_upload_context(file: UploadFile = File(...)):
    orig_name = file.filename or "slides.txt"
    max_bytes = 40 * 1024 * 1024
    chunks: list[bytes] = []
    total_bytes = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise HTTPException(status_code=400, detail="File tài liệu vượt quá giới hạn 40 MB.")
        chunks.append(chunk)
    data = b"".join(chunks)
    try:
        payload = await asyncio.to_thread(runtime.slide_context.ingest_file, orig_name, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi đọc tài liệu: {exc}") from exc
    runtime._last_slide_match_idx = payload.get("active_slide_index", 0)
    await asyncio.to_thread(runtime.sync_asr_hotwords)
    runtime.broadcast_threadsafe({"type": "status", "data": runtime.get_status_payload()})
    return {"ok": True, **payload}


@app.post("/api/context/text")
def api_update_context_text(req: SlideContextTextRequest):
    payload = runtime.slide_context.ingest_text(
        text=req.text,
        topic=req.topic,
        terms=req.terms,
        enabled=req.enabled,
        apply_dau_preset=req.apply_dau_preset,
        active_slide_index=req.active_slide_index,
    )
    runtime._last_slide_match_idx = payload.get("active_slide_index", 0)
    runtime.sync_asr_hotwords()
    runtime.broadcast_threadsafe({"type": "status", "data": runtime.get_status_payload()})
    return {"ok": True, **payload}


@app.delete("/api/context")
def api_clear_context():
    payload = runtime.slide_context.clear()
    runtime._last_slide_match_idx = 0
    runtime.sync_asr_hotwords()
    runtime.broadcast_threadsafe({"type": "status", "data": runtime.get_status_payload()})
    return {"ok": True, **payload}


@app.post("/api/display/freeze")
def api_display_freeze(req: DisplayFreezeRequest):
    res = runtime.set_display_freeze(req.frozen, req.message)
    return {"ok": True, **res}


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
    asr_mode: str = Form("auto"),
    normalize_audio: bool = Form(True),
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
            asr_mode,
            normalize_audio,
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
    client_host = ws.client.host if ws.client else ""
    if not _is_loopback_host(client_host):
        await ws.close(code=1008, reason="Admin WebSocket restricted to localhost")
        return
    await ws.accept()
    try:
        await runtime.register_ws(ws)
        while True:
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_json({"type": "pong"})
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        await runtime.unregister_ws(ws)


@app.websocket("/ws/audience")
async def ws_audience_endpoint(ws: WebSocket, lang: str = "follow"):
    await ws.accept()
    try:
        await runtime.register_audience_ws(ws, lang=lang)
        while True:
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_json({"type": "pong"})
    except (WebSocketDisconnect, Exception):
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
