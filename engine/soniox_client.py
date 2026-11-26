"""
Soniox Cloud Real-Time Speech Recognition + Inline Translation Client (`stt-rt-v6`).

Streams 16 kHz mono PCM (`pcm_s16le`) over a single WebSocket to
`wss://stt-rt.soniox.com/transcribe-websocket` with:
- Automatic token-level language identification & mid-sentence code-switching (60 languages)
- Inline one-way real-time translation (`translation.type = "one_way"`)
- Semantic endpoint detection (`enable_endpoint_detection`, `<end>` token)
- Custom Slide Context & Glossary biasing (`context.general`, `context.terms`, `context.translation_terms`)
- Automatic keepalive frames during speaker pauses & graceful EOS flushing
- ~0% local CPU/GPU usage (no local ASR/MT models required while online)
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import ssl
import threading
import time
from typing import Any, Callable, Optional

import certifi
import httpx
import numpy as np
import websockets

from languages import WHISPER_LANG_TO_NLLB

logger = logging.getLogger("voicerecore.soniox")

SONIOX_WS_URL = "wss://stt-rt.soniox.com/transcribe-websocket"
SONIOX_MODELS_URL = "https://api.soniox.com/v1/models"
DEFAULT_SONIOX_MODEL = "stt-rt-v6"

# Official 60 languages supported by Soniox `stt-rt-v6` (verified via GET /v1/models)
SONIOX_SUPPORTED_ISO: frozenset[str] = frozenset(
    {
        "af", "sq", "ar", "az", "eu", "be", "bn", "bs", "bg", "ca",
        "zh", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "gl",
        "de", "el", "gu", "he", "hi", "hu", "id", "it", "ja", "kn",
        "kk", "ko", "lv", "lt", "mk", "ms", "ml", "mr", "no", "fa",
        "pl", "pt", "pa", "ro", "ru", "sr", "sk", "sl", "es", "sw",
        "sv", "tl", "ta", "te", "th", "tr", "uk", "ur", "vi", "cy",
    }
)

# Canonical NLLB FLORES-200 -> Soniox ISO-639-1 map
NLLB_TO_SONIOX_ISO: dict[str, str] = {
    "vie_Latn": "vi",
    "eng_Latn": "en",
    "jpn_Jpan": "ja",
    "kor_Hang": "ko",
    "zho_Hans": "zh",
    "zho_Hant": "zh",
    "yue_Hant": "zh",
    "tha_Thai": "th",
    "fra_Latn": "fr",
    "deu_Latn": "de",
    "spa_Latn": "es",
    "rus_Cyrl": "ru",
    "ita_Latn": "it",
    "por_Latn": "pt",
    "ind_Latn": "id",
    "zsm_Latn": "ms",
    "tgl_Latn": "tl",
    "arb_Arab": "ar",
    "hin_Deva": "hi",
    "nld_Latn": "nl",
    "pol_Latn": "pl",
    "tur_Latn": "tr",
    "ukr_Cyrl": "uk",
    "ces_Latn": "cs",
    "swe_Latn": "sv",
    "fin_Latn": "fi",
    "dan_Latn": "da",
    "nob_Latn": "no",
    "ell_Grek": "el",
    "heb_Hebr": "he",
    "hun_Latn": "hu",
    "ron_Latn": "ro",
    "slk_Latn": "sk",
    "bul_Cyrl": "bg",
    "hrv_Latn": "hr",
    "ben_Beng": "bn",
    "tam_Taml": "ta",
    "urd_Arab": "ur",
    "pes_Arab": "fa",
    "swh_Latn": "sw",
    "kaz_Cyrl": "kk",
    "afr_Latn": "af",
    "als_Latn": "sq",
    "azj_Latn": "az",
    "eus_Latn": "eu",
    "bel_Cyrl": "be",
    "bos_Latn": "bs",
    "cat_Latn": "ca",
    "est_Latn": "et",
    "glg_Latn": "gl",
    "guj_Gujr": "gu",
    "kan_Knda": "kn",
    "lvs_Latn": "lv",
    "lit_Latn": "lt",
    "mkd_Cyrl": "mk",
    "mal_Mlym": "ml",
    "mar_Deva": "mr",
    "pan_Guru": "pa",
    "srp_Cyrl": "sr",
    "slv_Latn": "sl",
    "tel_Telu": "te",
    "cym_Latn": "cy",
}

_EXTRA_SONIOX_TO_NLLB: dict[str, str] = {
    v: k for k, v in NLLB_TO_SONIOX_ISO.items() if k not in ("zho_Hant", "yue_Hant")
}


def nllb_to_soniox_iso(nllb_code: str) -> str:
    """Map an NLLB code to a Soniox ISO-639-1 code, or '' if unsupported / auto."""
    code = (nllb_code or "").strip()
    if not code or code == "auto":
        return ""
    iso = NLLB_TO_SONIOX_ISO.get(code, "")
    if iso:
        return iso
    low = code.lower().split("-", 1)[0].split("_", 1)[0]
    return low if low in SONIOX_SUPPORTED_ISO else ""


def soniox_iso_to_nllb(iso: str, fallback: str = "eng_Latn") -> str:
    """Map a Soniox token language code (e.g. 'ja', 'vi', 'en') back to canonical NLLB code."""
    clean = (iso or "").strip().lower().replace("_", "-")
    if "-" in clean:
        clean = clean.split("-", 1)[0]
    if not clean:
        return fallback
    return WHISPER_LANG_TO_NLLB.get(clean) or _EXTRA_SONIOX_TO_NLLB.get(clean) or fallback


def _build_ssl_context() -> ssl.SSLContext:
    """Return reliable TLS context using certifi CA bundle for frozen PyInstaller binaries."""
    try:
        ca = certifi.where()
        if ca:
            return ssl.create_default_context(cafile=ca)
    except Exception:
        pass
    return ssl.create_default_context()


def _friendly_soniox_error(error_code: Any, error_type: str, error_message: str) -> str:
    code_int = 0
    try:
        code_int = int(error_code or 0)
    except Exception:
        pass
    etype = (error_type or "").lower()
    emsg = (error_message or "").strip()
    if code_int == 402 or "balance" in etype or "exhausted" in etype:
        return (
            "Tài khoản Soniox đang có số dư $0. Hãy vào console.soniox.com → Billing "
            "nạp tối thiểu $5 (~28 giờ hội nghị) rồi thử lại."
        )
    if code_int in (401, 403) or "auth" in etype or "invalid_api_key" in etype:
        return "API Key Soniox không hợp lệ hoặc đã bị thu hồi."
    if code_int == 429 or "rate" in etype:
        return "Soniox đang giới hạn tốc độ tạm thời (HTTP 429) — đang tự động dự phòng Offline."
    return f"Soniox API lỗi ({code_int or etype or 'ws'}): {emsg or 'Không xác định'}"


def test_soniox_connection(
    api_key: str,
    model: str = DEFAULT_SONIOX_MODEL,
    ws_url: str = SONIOX_WS_URL,
    models_url: str = SONIOX_MODELS_URL,
) -> dict[str, Any]:
    """
    Validate a Soniox API Key & balance cleanly at $0 cost:
    1. `GET /v1/models` verifies API key authenticity and available real-time models.
    2. Zero-audio WebSocket handshake verifies active account balance (catches 402 immediately).
    """
    key = (api_key or "").strip()
    if not key:
        return {"ok": False, "error": "Chưa nhập API Key Soniox."}

    chosen_model = (model or DEFAULT_SONIOX_MODEL).strip() or DEFAULT_SONIOX_MODEL
    ssl_ctx = _build_ssl_context()
    t0 = time.perf_counter()

    # Step 1: Verify key against REST /v1/models
    try:
        with httpx.Client(timeout=10.0, verify=certifi.where()) as client:
            resp = client.get(models_url, headers={"Authorization": f"Bearer {key}"})
    except Exception as exc:
        return {"ok": False, "error": f"Không kết nối được máy chủ Soniox: {exc}"}

    if resp.status_code in (401, 403):
        return {"ok": False, "error": "API Key Soniox không hợp lệ hoặc đã bị thu hồi."}
    if resp.status_code >= 400:
        return {"ok": False, "error": f"Soniox REST HTTP {resp.status_code}: {resp.text[:180]}"}

    rt_models: list[str] = []
    lang_count = 60
    try:
        payload = resp.json()
        for m in payload.get("models") or []:
            if m.get("transcription_mode") == "real_time":
                mid = str(m.get("id") or "")
                if mid:
                    rt_models.append(mid)
                if mid == chosen_model and isinstance(m.get("languages"), list):
                    lang_count = len(m["languages"])
        if rt_models and chosen_model not in rt_models:
            chosen_model = rt_models[0]
    except Exception:
        pass

    # Step 2: Zero-audio WebSocket handshake to confirm billing balance & real-time access ($0 cost)
    async def _probe_ws() -> dict[str, Any]:
        async with websockets.connect(
            ws_url,
            additional_headers={"Authorization": f"Bearer {key}"},
            ssl=ssl_ctx if ws_url.startswith("wss://") else None,
            open_timeout=10,
            close_timeout=5,
        ) as ws:
            init_cfg = {
                "model": chosen_model,
                "audio_format": "pcm_s16le",
                "sample_rate": 16000,
                "num_channels": 1,
                "enable_language_identification": True,
                "enable_endpoint_detection": True,
            }
            await ws.send(json.dumps(init_cfg))
            await ws.send("")  # graceful immediate EOS (0 audio bytes)
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=8.0)
                data = json.loads(raw)
                if data.get("error_code"):
                    return {
                        "ok": False,
                        "key_valid": True,
                        "error_code": data.get("error_code"),
                        "error_type": str(data.get("error_type") or ""),
                        "error_message": str(data.get("error_message") or ""),
                    }
                if data.get("finished"):
                    return {"ok": True, "key_valid": True}

    try:
        ws_res = asyncio.run(_probe_ws())
    except websockets.exceptions.ConnectionClosedOK:
        ws_res = {"ok": True, "key_valid": True}
    except Exception as exc:
        return {"ok": False, "error": f"Lỗi bắt tay WebSocket Soniox: {exc}"}

    ms = round((time.perf_counter() - t0) * 1000.0, 1)
    if not ws_res.get("ok"):
        ec = ws_res.get("error_code")
        et = str(ws_res.get("error_type") or "")
        em = str(ws_res.get("error_message") or "")
        friendly = _friendly_soniox_error(ec, et, em)
        if ec == 402 or "balance" in et.lower():
            friendly = (
                f"API Key Soniox HỢP LỆ ({chosen_model} • {lang_count} NN), nhưng tài khoản đang có số dư $0. "
                "Vào console.soniox.com → Billing nạp tối thiểu $5 (~28 giờ hội nghị) rồi bấm Kiểm tra lại."
            )
        return {
            "ok": False,
            "key_valid": True,
            "balance_exhausted": bool(ec == 402 or "balance" in et.lower()),
            "model": chosen_model,
            "latency_ms": ms,
            "error": friendly,
        }

    return {
        "ok": True,
        "key_valid": True,
        "provider": "soniox",
        "engine": f"soniox:{chosen_model}",
        "model": chosen_model,
        "languages_count": lang_count,
        "latency_ms": ms,
        "translated_text": f"Soniox {chosen_model} ({lang_count} NN) sẵn sàng — nghe & dịch trực tiếp ~0% CPU!",
    }


def build_soniox_context(
    *,
    topic: str = "",
    terms: Optional[list[str]] = None,
    translation_terms: Optional[list[dict[str, str]]] = None,
) -> Optional[dict[str, Any]]:
    """Build compact Soniox v2 `context` object from DAU Slide RAG & Glossary."""
    ctx: dict[str, Any] = {}
    clean_topic = (topic or "").strip()
    if clean_topic:
        ctx["general"] = [{"key": "topic", "value": clean_topic[:140]}]

    clean_terms: list[str] = []
    seen_t: set[str] = set()
    for t in terms or []:
        s = str(t or "").strip()
        if s and len(s) <= 64 and s.lower() not in seen_t:
            seen_t.add(s.lower())
            clean_terms.append(s)
            if len(clean_terms) >= 80:
                break
    if clean_terms:
        ctx["terms"] = clean_terms

    clean_tt: list[dict[str, str]] = []
    seen_tt: set[tuple[str, str]] = set()
    for item in translation_terms or []:
        src = str(item.get("source") or "").strip()
        tgt = str(item.get("target") or "").strip()
        if src and tgt and (src.lower(), tgt.lower()) not in seen_tt:
            seen_tt.add((src.lower(), tgt.lower()))
            clean_tt.append({"source": src, "target": tgt})
            if len(clean_tt) >= 60:
                break
    if clean_tt:
        ctx["translation_terms"] = clean_tt

    return ctx or None


class SonioxRealtimeClient:
    """
    Thread-safe live WebSocket streamer for Soniox Real-Time STT + Translation (`stt-rt-v6`).

    Usage:
      client = SonioxRealtimeClient(
          api_key=..., source_lang="auto", target_lang="vie_Latn",
          on_partial=..., on_final=..., on_detected_lang=..., on_usage_seconds=..., on_error=...,
      )
      client.start()
      client.feed_pcm(pcm_s16)   # 100ms 16kHz mono int16 chunks
      ...
      client.stop()
    """

    def __init__(
        self,
        api_key: str,
        *,
        source_lang: str = "auto",
        target_lang: str = "vie_Latn",
        model: str = DEFAULT_SONIOX_MODEL,
        context: Optional[dict[str, Any]] = None,
        sample_rate: int = 16000,
        ws_url: str = SONIOX_WS_URL,
        on_partial: Optional[Callable[[str, str, str, float], None]] = None,
        on_final: Optional[Callable[[str, str, str, float], None]] = None,
        on_detected_lang: Optional[Callable[[str], None]] = None,
        on_usage_seconds: Optional[Callable[[int], None]] = None,
        on_error: Optional[Callable[[int, str], None]] = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.model = (model or DEFAULT_SONIOX_MODEL).strip() or DEFAULT_SONIOX_MODEL
        self.context = context
        self.sample_rate = sample_rate
        self.ws_url = ws_url
        self._on_partial = on_partial
        self._on_final = on_final
        self._on_detected_lang = on_detected_lang
        self._on_usage_seconds = on_usage_seconds
        self._on_error = on_error

        self._pcm_queue: queue.Queue[Optional[bytes]] = queue.Queue(maxsize=160)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._connected_event = threading.Event()
        self._fatal_error: str = ""
        self._fatal_code: int = 0

        # Token accumulators for the active utterance
        self._final_orig: list[str] = []
        self._final_trans: list[str] = []
        self._last_src_iso: str = ""
        self._last_reported_nllb: str = ""
        self._last_partial_sig: tuple[str, str] = ("", "")
        self._last_send_mono: float = 0.0
        self._sent_bytes_unreported: int = 0

    def start(self) -> None:
        self._stop_event.clear()
        self._connected_event.clear()
        self._fatal_error = ""
        self._fatal_code = 0
        self._thread = threading.Thread(
            target=self._run_thread,
            daemon=True,
            name="soniox-rt-worker",
        )
        self._thread.start()

    def wait_connected(self, timeout_s: float = 6.0) -> tuple[bool, int, str]:
        """Wait briefly for initial WebSocket connect so immediate auth/402 errors surface fast."""
        self._connected_event.wait(timeout=timeout_s)
        if self._fatal_error:
            return False, self._fatal_code, self._fatal_error
        return self._connected_event.is_set(), 0, ""

    def feed_pcm(self, pcm_s16: np.ndarray) -> None:
        if self._stop_event.is_set() or pcm_s16 is None or len(pcm_s16) == 0:
            return
        raw = pcm_s16.astype(np.int16, copy=False).tobytes()
        try:
            self._pcm_queue.put_nowait(raw)
        except queue.Full:
            # Drop oldest 100ms chunk to stay real-time on brief Wi-Fi hiccups
            try:
                self._pcm_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._pcm_queue.put_nowait(raw)
            except queue.Full:
                pass

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._pcm_queue.put_nowait(None)
        except queue.Full:
            try:
                self._pcm_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._pcm_queue.put_nowait(None)
            except queue.Full:
                pass
        th = self._thread
        if th is not None and th.is_alive():
            th.join(timeout=3.5)
        self._thread = None

    def _build_start_config(self) -> dict[str, Any]:
        src_iso = nllb_to_soniox_iso(self.source_lang)
        tgt_iso = nllb_to_soniox_iso(self.target_lang)
        cfg: dict[str, Any] = {
            "model": self.model,
            "audio_format": "pcm_s16le",
            "sample_rate": self.sample_rate,
            "num_channels": 1,
            "enable_language_identification": True,
            "enable_endpoint_detection": True,
        }
        hints: list[str] = []
        if src_iso:
            hints.append(src_iso)
        if tgt_iso and tgt_iso not in hints:
            hints.append(tgt_iso)
        if hints:
            cfg["language_hints"] = hints
        # Enable inline Soniox translation whenever target is supported (and != fixed source)
        if tgt_iso and (self.source_lang == "auto" or src_iso != tgt_iso):
            cfg["translation"] = {
                "type": "one_way",
                "target_language": tgt_iso,
            }
        if self.context:
            cfg["context"] = self.context
        return cfg

    @staticmethod
    def _join_tokens(tokens: list[str]) -> str:
        if not tokens:
            return ""
        raw = "".join(tokens)
        return " ".join(raw.split()).strip()

    def _flush_usage_locked(self, force: bool = False) -> None:
        bytes_per_sec = self.sample_rate * 2  # 16-bit mono
        secs = self._sent_bytes_unreported // bytes_per_sec
        if secs >= 10 or (force and secs >= 1):
            self._sent_bytes_unreported -= secs * bytes_per_sec
            if self._on_usage_seconds:
                try:
                    self._on_usage_seconds(int(secs))
                except Exception:
                    pass

    def _resolve_current_src_nllb(self) -> str:
        if self.source_lang != "auto":
            return self.source_lang
        if self._last_src_iso:
            return soniox_iso_to_nllb(self._last_src_iso, "eng_Latn")
        return self._last_reported_nllb or "eng_Latn"

    def _emit_final_utterance(self, latency_ms: float, override_src_nllb: str = "") -> None:
        orig = self._join_tokens(self._final_orig)
        trans = self._join_tokens(self._final_trans)
        self._final_orig.clear()
        self._final_trans.clear()
        self._last_partial_sig = ("", "")
        if not orig and not trans:
            return
        src_nllb = override_src_nllb or self._resolve_current_src_nllb()
        if not orig and trans:
            orig = trans
        if self._on_final:
            try:
                self._on_final(orig, trans, src_nllb, round(latency_ms, 1))
            except Exception as exc:
                logger.warning("Soniox on_final callback failed: %s", exc)

    def _handle_response_message(self, data: dict[str, Any]) -> bool:
        """Process one JSON frame from Soniox WebSocket. Returns True when stream is finished."""
        if data.get("error_code"):
            ec = int(data.get("error_code") or 0)
            et = str(data.get("error_type") or "")
            em = str(data.get("error_message") or "")
            msg = _friendly_soniox_error(ec, et, em)
            self._fatal_code = ec
            self._fatal_error = msg
            self._connected_event.set()
            if self._on_error:
                try:
                    self._on_error(ec, msg)
                except Exception:
                    pass
            return True

        now = time.monotonic()
        latency_ms = max(20.0, (now - self._last_send_mono) * 1000.0) if self._last_send_mono else 120.0
        tokens = data.get("tokens") or []

        nonfinal_orig: list[str] = []
        nonfinal_trans: list[str] = []
        saw_endpoint = False

        for tok in tokens:
            if not isinstance(tok, dict):
                continue
            txt = str(tok.get("text") or "")
            if not txt:
                continue
            if txt == "<end>":
                saw_endpoint = True
                continue
            if txt == "<fin>":
                continue

            is_final = bool(tok.get("is_final", False))
            t_status = str(tok.get("translation_status") or "none").lower()

            if t_status == "translation":
                src_iso = str(tok.get("source_language") or "").strip().lower()
                if src_iso and (is_final or not self._final_orig):
                    self._last_src_iso = src_iso
                if is_final:
                    self._final_trans.append(txt)
                else:
                    nonfinal_trans.append(txt)
            else:
                lang_iso = str(tok.get("language") or "").strip().lower()
                if lang_iso:
                    nllb = soniox_iso_to_nllb(lang_iso, "")
                    if is_final:
                        if nllb and nllb != self._last_reported_nllb:
                            prev_nllb = self._last_reported_nllb
                            # Flush previous language buffer cleanly WITH its previous language tag!
                            if prev_nllb and (self._final_orig or self._final_trans):
                                self._emit_final_utterance(latency_ms, override_src_nllb=prev_nllb)
                            self._last_src_iso = lang_iso
                            self._last_reported_nllb = nllb
                            if self._on_detected_lang:
                                try:
                                    self._on_detected_lang(nllb)
                                except Exception:
                                    pass
                        else:
                            self._last_src_iso = lang_iso
                    elif not self._final_orig:
                        self._last_src_iso = lang_iso
                        if nllb and nllb != self._last_reported_nllb:
                            self._last_reported_nllb = nllb
                            if self._on_detected_lang:
                                try:
                                    self._on_detected_lang(nllb)
                                except Exception:
                                    pass
                if is_final:
                    self._final_orig.append(txt)
                else:
                    nonfinal_orig.append(txt)

        # Commit cleanly on semantic `<end>` token OR long sentence boundary (>36 words)
        if saw_endpoint:
            self._emit_final_utterance(latency_ms)
        else:
            cur_final_orig = self._join_tokens(self._final_orig)
            cur_final_trans = self._join_tokens(self._final_trans)
            long_punct_boundary = (
                cur_final_orig
                and (len(cur_final_orig.split()) >= 34 or len(cur_final_orig) >= 140)
                and cur_final_orig.endswith((".", "!", "?", "。", "！", "？"))
                and (bool(cur_final_trans) or self._resolve_current_src_nllb() == self.target_lang)
            )
            if long_punct_boundary:
                self._emit_final_utterance(latency_ms)
            else:
                live_orig = self._join_tokens([*self._final_orig, *nonfinal_orig])
                live_trans = self._join_tokens([*self._final_trans, *nonfinal_trans])
                sig = (live_orig, live_trans)
                if (live_orig or live_trans) and sig != self._last_partial_sig:
                    self._last_partial_sig = sig
                    if self._on_partial:
                        try:
                            self._on_partial(
                                live_orig,
                                live_trans,
                                self._resolve_current_src_nllb(),
                                round(latency_ms, 1),
                            )
                        except Exception:
                            pass

        if data.get("finished"):
            self._emit_final_utterance(latency_ms)
            return True
        return False

    def _run_thread(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_session_loop())
        finally:
            self._flush_usage_locked(force=True)
            try:
                loop.close()
            except Exception:
                pass

    async def _async_session_loop(self) -> None:
        ssl_ctx = _build_ssl_context() if self.ws_url.startswith("wss://") else None
        max_attempts = 2
        attempt = 0

        while not self._stop_event.is_set() and attempt < max_attempts:
            attempt += 1
            try:
                async with websockets.connect(
                    self.ws_url,
                    additional_headers={"Authorization": f"Bearer {self.api_key}"},
                    ssl=ssl_ctx,
                    open_timeout=10,
                    close_timeout=5,
                    ping_interval=20,
                    ping_timeout=20,
                ) as ws:
                    await ws.send(json.dumps(self._build_start_config()))
                    self._last_send_mono = time.monotonic()
                    # Wait briefly (up to 0.35s) for an immediate config/402/401 error before marking connected
                    first_err = False
                    try:
                        first_raw = await asyncio.wait_for(ws.recv(), timeout=0.35)
                        first_data = json.loads(first_raw)
                        if self._handle_response_message(first_data):
                            first_err = True
                    except asyncio.TimeoutError:
                        pass
                    if first_err:
                        return

                    self._connected_event.set()
                    attempt = 0  # Reset reconnect budget after healthy connection

                    async def _sender() -> None:
                        last_tx = time.monotonic()
                        while True:
                            try:
                                chunk = await asyncio.to_thread(self._pcm_queue.get, True, 0.25)
                            except queue.Empty:
                                if self._stop_event.is_set():
                                    break
                                if (time.monotonic() - last_tx) >= 12.0:
                                    await ws.send(json.dumps({"type": "keepalive"}))
                                    last_tx = time.monotonic()
                                continue
                            if chunk is None:
                                break
                            await ws.send(chunk)
                            last_tx = time.monotonic()
                            self._last_send_mono = last_tx
                            self._sent_bytes_unreported += len(chunk)
                            self._flush_usage_locked(force=False)
                        # Graceful Soniox End-of-Stream frame so final tokens & translations flush
                        try:
                            await ws.send("")
                        except Exception:
                            pass

                    async def _receiver() -> None:
                        async for raw in ws:
                            data = json.loads(raw)
                            if self._handle_response_message(data):
                                break

                    send_task = asyncio.create_task(_sender())
                    recv_task = asyncio.create_task(_receiver())
                    done, pending = await asyncio.wait(
                        {send_task, recv_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    # If sender finished first (user pressed Stop), give receiver up to 2.5s to flush `<end>`/`finished`
                    if send_task in done and recv_task in pending:
                        try:
                            await asyncio.wait_for(recv_task, timeout=2.5)
                        except Exception:
                            recv_task.cancel()
                    else:
                        for task in pending:
                            task.cancel()
                        for task in done:
                            exc = task.exception()
                            if exc is not None and not isinstance(exc, asyncio.CancelledError):
                                raise exc

                    if self._fatal_error or self._stop_event.is_set():
                        self._emit_final_utterance(100.0)
                        return
            except Exception as exc:
                if self._stop_event.is_set():
                    self._emit_final_utterance(100.0)
                    return
                if self._fatal_error:
                    return
                logger.warning("Soniox WebSocket attempt %d failed: %s", attempt, exc)
                if attempt < max_attempts:
                    await asyncio.sleep(0.8)
                    continue
                msg = f"Mất kết nối Soniox Cloud ({exc}) — đang tự chuyển về chế độ Offline."
                self._fatal_code = 503
                self._fatal_error = msg
                self._connected_event.set()
                if self._on_error:
                    try:
                        self._on_error(503, msg)
                    except Exception:
                        pass
                return
