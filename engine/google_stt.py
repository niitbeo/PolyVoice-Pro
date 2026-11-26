"""
100% Google Cloud Real-Time Speech-to-Text Engine (`speech.googleapis.com/v1/speech:recognize`).

Uses the user's SAME Google Cloud API Key (`AIza...`) — zero Service Account JSON,
zero heavy `grpcio` binaries, billed per 1-second increment on Google Cloud.

Architecture:
- Lightweight Silero VAD v5 (2.2 MB, ~0.4% CPU) — or RMS energy VAD fallback — segments
  continuous microphone / YouTube system audio into crisp natural clauses (0.48s pause or
  7.5s max continuous speech, with 320ms pre-roll so onset syllables are never clipped).
- Non-blocking background rolling partial worker calls Google Cloud STT v1 every ~1.25s
  over a persistent keep-alive TLS pool (`httpx.Client`) to stream live partial transcripts
  without ever stalling PCM audio capture.
- On pause / clause end, fires one clean final Google Cloud STT v1 call with:
    - `enableAutomaticPunctuation: True` (proper commas, periods, question marks)
    - `speechContexts` (DAU glossary & slide terminology boosting!)
    - `alternativeLanguageCodes` (when `source_lang == "auto"`)
- Supports 125+ Google Cloud STT locales (Vietnamese `vi-VN`, English `en-US`, Thai `th-TH`,
  Lao `lo-LA`, Khmer `km-KH`, Chinese `zh`, Japanese `ja-JP`, Korean `ko-KR`, French, German...).
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import httpx
import numpy as np

from asr_engine import ASROutput

logger = logging.getLogger("voicerecore.google_stt")

GOOGLE_STT_V1_URL = "https://speech.googleapis.com/v1/speech:recognize"

NLLB_TO_GOOGLE_STT_LOCALE: dict[str, str] = {
    "vie_Latn": "vi-VN",
    "eng_Latn": "en-US",
    "tha_Thai": "th-TH",
    "lao_Laoo": "lo-LA",
    "khm_Khmr": "km-KH",
    "zho_Hans": "zh",
    "zho_Hant": "zh-TW",
    "jpn_Jpan": "ja-JP",
    "kor_Hang": "ko-KR",
    "fra_Latn": "fr-FR",
    "deu_Latn": "de-DE",
    "spa_Latn": "es-ES",
    "rus_Cyrl": "ru-RU",
    "ind_Latn": "id-ID",
    "zsm_Latn": "ms-MY",
    "mya_Mymr": "my-MM",
    "ita_Latn": "it-IT",
    "por_Latn": "pt-BR",
    "nld_Latn": "nl-NL",
    "pol_Latn": "pl-PL",
    "tur_Latn": "tr-TR",
    "arb_Arab": "ar-SA",
    "hin_Deva": "hi-IN",
    "tgl_Latn": "fil-PH",
}

GOOGLE_LOCALE_TO_NLLB: dict[str, str] = {
    "vi": "vie_Latn",
    "en": "eng_Latn",
    "th": "tha_Thai",
    "lo": "lao_Laoo",
    "km": "khm_Khmr",
    "zh": "zho_Hans",
    "cmn": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "ru": "rus_Cyrl",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "my": "mya_Mymr",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "tr": "tur_Latn",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "fil": "tgl_Latn",
}


def nllb_to_google_stt_locale(nllb_code: str) -> str:
    code = (nllb_code or "").strip()
    if code == "auto" or not code:
        return "vi-VN"
    return NLLB_TO_GOOGLE_STT_LOCALE.get(code, "en-US")


def google_locale_to_nllb(locale_code: str, fallback: str = "vie_Latn") -> str:
    low = (locale_code or "").strip().lower()
    if not low:
        return fallback
    if low in ("zh-tw", "zh-hk", "cmn-hant-tw"):
        return "zho_Hant"
    prefix = low.split("-", 1)[0]
    return GOOGLE_LOCALE_TO_NLLB.get(prefix, fallback)


def probe_google_stt_api(api_key: str, timeout_s: float = 6.0) -> tuple[bool, int, str]:
    """Quick 0.2s silent probe to verify whether `speech.googleapis.com` is enabled for `api_key`."""
    key = (api_key or "").strip()
    if not key:
        return (False, 400, "Chưa có Google Cloud API Key.")
    silence_b64 = base64.b64encode(b"\x00" * 6400).decode("ascii")
    payload = {
        "config": {
            "encoding": "LINEAR16",
            "sampleRateHertz": 16000,
            "languageCode": "vi-VN",
            "enableAutomaticPunctuation": True,
        },
        "audio": {"content": silence_b64},
    }
    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(
                GOOGLE_STT_V1_URL,
                headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                json=payload,
            )
        if resp.status_code == 200:
            return (True, 200, "Cloud Speech-to-Text API sẵn sàng (100% Google Cloud Nghe + Dịch).")
        err_msg = ""
        try:
            err_msg = resp.json().get("error", {}).get("message", "")
        except Exception:
            pass
        if resp.status_code in (401, 403):
            return (
                False,
                resp.status_code,
                "Key Google Cloud chưa bật quyền Cloud Speech-to-Text API (cần bấm ENABLE trên Google Cloud Console và thêm vào API restrictions của Key).",
            )
        return (False, resp.status_code, f"Google STT HTTP {resp.status_code}: {err_msg or resp.reason_phrase}")
    except Exception as exc:
        return (False, 0, f"Không kết nối được Google Speech-to-Text: {exc}")


class GoogleCloudStreamingASR:
    """
    VAD-driven Real-Time Google Cloud Speech-to-Text v1 Engine.
    Drop-in compatible with `EngineRuntime._audio_loop` (`accept_pcm`, `flush_final`,
    `is_speech_active`, `reset`, `close`), plus async live partial callback so PCM audio
    capture NEVER blocks waiting for network responses.
    """

    def __init__(
        self,
        api_key: str,
        source_lang: str = "vie_Latn",
        target_lang: str = "eng_Latn",
        sample_rate: int = 16000,
        vad_model_path: Optional[Path] = None,
        hotwords: Optional[list[str]] = None,
        on_async_output: Optional[Callable[[ASROutput], None]] = None,
        on_fatal_error: Optional[Callable[[int, str], None]] = None,
        pause_commit_s: float = 0.52,
        max_utterance_s: float = 7.5,
        partial_interval_s: float = 1.25,
    ) -> None:
        self.api_key = api_key.strip()
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.sample_rate = sample_rate
        self.active_provider = "google-cloud-stt"
        self._hotwords = [h.strip() for h in (hotwords or []) if h and h.strip()][:80]
        self._on_async_output = on_async_output
        self._on_fatal_error = on_fatal_error
        self._pause_commit_s = max(0.35, pause_commit_s)
        self._max_utterance_s = max(4.0, max_utterance_s)
        self._partial_interval_s = max(0.85, partial_interval_s)

        self._client = httpx.Client(
            timeout=httpx.Timeout(10.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=6, max_connections=10),
        )
        self._lock = threading.Lock()
        self._vad = self._try_build_silero_vad(vad_model_path)
        self._vad_window = 512
        self._vad_buf = np.zeros(0, dtype=np.float32)

        # Speech turn state
        self._preroll: deque[np.ndarray] = deque(maxlen=4)  # ~400ms pre-roll
        self._speech_chunks: list[np.ndarray] = []
        self._speech_samples: int = 0
        self._in_speech: bool = False
        self._quiet_s: float = 0.0
        self._speech_rms: float = 0.03
        self._turn_seq: int = 0

        # Non-blocking rolling partial worker state
        self._partial_inflight: bool = False
        self._last_partial_ts: float = 0.0
        self._last_partial_text: str = ""
        self._closed: bool = False

    def _try_build_silero_vad(self, vad_model_path: Optional[Path]):
        if vad_model_path is None or not Path(vad_model_path).is_file():
            return None
        try:
            import sherpa_onnx

            silero = sherpa_onnx.SileroVadModelConfig(
                model=str(vad_model_path),
                threshold=0.40,
                min_silence_duration=self._pause_commit_s,
                min_speech_duration=0.20,
                window_size=512,
                max_speech_duration=self._max_utterance_s,
            )
            cfg = sherpa_onnx.VadModelConfig(
                silero_vad=silero,
                sample_rate=self.sample_rate,
                num_threads=1,
                provider="cpu",
                debug=False,
            )
            return sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60.0)
        except Exception as exc:
            logger.debug("Silero VAD unavailable for GoogleCloudStreamingASR, using RMS VAD: %s", exc)
            return None

    def set_hotwords(self, hotwords: list[str]) -> None:
        with self._lock:
            self._hotwords = [h.strip() for h in (hotwords or []) if h and h.strip()][:80]

    def set_language(self, source_lang: str, target_lang: Optional[str] = None) -> None:
        with self._lock:
            self.source_lang = source_lang
            if target_lang:
                self.target_lang = target_lang

    def is_speech_active(self) -> bool:
        with self._lock:
            return self._in_speech or self._speech_samples > 0

    def reset(self) -> None:
        with self._lock:
            self._turn_seq += 1
            self._preroll.clear()
            self._speech_chunks.clear()
            self._speech_samples = 0
            self._in_speech = False
            self._quiet_s = 0.0
            self._last_partial_text = ""
            self._last_partial_ts = 0.0
            self._vad_buf = np.zeros(0, dtype=np.float32)
            if self._vad is not None:
                try:
                    self._vad.reset()
                except Exception:
                    pass

    def close(self) -> None:
        self._closed = True
        self.reset()
        try:
            self._client.close()
        except Exception:
            pass

    def _recognize_pcm(self, pcm_s16: np.ndarray, *, is_final: bool) -> tuple[str, float, str]:
        """Call Google Cloud Speech-to-Text REST v1 with LINEAR16 16kHz PCM."""
        if self._closed or pcm_s16 is None or len(pcm_s16) < int(self.sample_rate * 0.25):
            return ("", 0.0, "")

        with self._lock:
            src_lang = self.source_lang
            tgt_lang = self.target_lang
            phrases = list(self._hotwords)

        primary_locale = nllb_to_google_stt_locale(src_lang)
        cfg: dict = {
            "encoding": "LINEAR16",
            "sampleRateHertz": self.sample_rate,
            "languageCode": primary_locale,
            "enableAutomaticPunctuation": True,
        }
        if src_lang == "auto":
            alt_locales: list[str] = []
            for cand in ("en-US", "vi-VN", nllb_to_google_stt_locale(tgt_lang)):
                if cand != primary_locale and cand not in alt_locales:
                    alt_locales.append(cand)
            if alt_locales:
                cfg["alternativeLanguageCodes"] = alt_locales[:3]

        if phrases and is_final:
            cfg["speechContexts"] = [{"phrases": phrases[:80], "boost": 8.0}]

        audio_b64 = base64.b64encode(pcm_s16.astype(np.int16, copy=False).tobytes()).decode("ascii")
        payload = {"config": cfg, "audio": {"content": audio_b64}}

        t0 = time.perf_counter()
        resp = self._client.post(
            GOOGLE_STT_V1_URL,
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            json=payload,
        )
        ms = round((time.perf_counter() - t0) * 1000.0, 1)

        if resp.status_code >= 400:
            err_msg = ""
            try:
                err_msg = resp.json().get("error", {}).get("message", "")
            except Exception:
                pass
            if resp.status_code in (401, 403) and self._on_fatal_error is not None:
                self._on_fatal_error(resp.status_code, err_msg or f"HTTP {resp.status_code}")
            raise RuntimeError(f"Google STT HTTP {resp.status_code}: {err_msg or resp.reason_phrase}")

        data = resp.json()
        results = data.get("results") or []
        parts: list[str] = []
        det_locale = ""
        for r in results:
            if not det_locale and r.get("languageCode"):
                det_locale = str(r["languageCode"])
            alts = r.get("alternatives") or []
            if alts:
                tr = str(alts[0].get("transcript") or "").strip()
                if tr:
                    parts.append(tr)
        transcript = " ".join(parts).strip()
        det_nllb = google_locale_to_nllb(det_locale, fallback=src_lang if src_lang != "auto" else "vie_Latn") if det_locale else ""
        return (transcript, ms, det_nllb)

    def _spawn_partial_worker(self, pcm_snap: np.ndarray, turn_seq: int) -> None:
        def _run() -> None:
            try:
                text, ms, det = self._recognize_pcm(pcm_snap, is_final=False)
                if not text or self._closed:
                    return
                with self._lock:
                    if turn_seq != self._turn_seq or text == self._last_partial_text:
                        return
                    self._last_partial_text = text
                if self._on_async_output is not None:
                    self._on_async_output(
                        ASROutput("partial", text, ms, self.active_provider, detected_lang=det)
                    )
            except Exception as exc:
                logger.debug("Google Cloud STT rolling partial skipped: %s", exc)
            finally:
                with self._lock:
                    self._partial_inflight = False

        threading.Thread(target=_run, daemon=True, name="google-stt-partial").start()

    def _spawn_final_worker(self, pcm_snap: np.ndarray) -> None:
        def _run() -> None:
            try:
                text, ms, det = self._recognize_pcm(pcm_snap, is_final=True)
                if not text or self._closed:
                    return
                if self._on_async_output is not None:
                    self._on_async_output(
                        ASROutput("final", text, ms, self.active_provider, detected_lang=det)
                    )
            except Exception as exc:
                logger.warning("Google Cloud STT final failed: %s", exc)

        threading.Thread(target=_run, daemon=True, name="google-stt-final").start()

    def accept_pcm(self, pcm_s16: np.ndarray) -> ASROutput:
        """Accept 16kHz int16 audio chunk (~100ms). Dispatches non-blocking Google STT requests."""
        if pcm_s16 is None or len(pcm_s16) == 0 or self._closed:
            return ASROutput("none", "", 0.0, self.active_provider)

        chunk = pcm_s16.astype(np.int16, copy=False)
        dur_s = len(chunk) / float(self.sample_rate)
        rms = float(np.sqrt(np.mean(np.square(chunk.astype(np.float32) / 32768.0))))

        # Determine speech activity via Silero VAD (when available) or adaptive RMS energy
        vad_active = False
        vad_segments: list[np.ndarray] = []
        with self._lock:
            if self._vad is not None:
                f32 = chunk.astype(np.float32) / 32768.0
                self._vad_buf = np.concatenate([self._vad_buf, f32]) if len(self._vad_buf) else f32
                while len(self._vad_buf) >= self._vad_window:
                    win = self._vad_buf[: self._vad_window]
                    self._vad_buf = self._vad_buf[self._vad_window :]
                    self._vad.accept_waveform(win)
                vad_active = bool(self._vad.is_speech_detected())
                while not self._vad.empty():
                    seg = np.asarray(self._vad.front.samples, dtype=np.float32)
                    self._vad.pop()
                    if len(seg) >= int(self.sample_rate * 0.28):
                        vad_segments.append(np.clip(seg * 32767.0, -32768, 32767).astype(np.int16))
            else:
                if rms > 0.012:
                    self._speech_rms = self._speech_rms * 0.92 + rms * 0.08
                vad_thresh = max(0.008, min(0.025, self._speech_rms * 0.28))
                vad_active = rms >= vad_thresh

            # If Silero VAD emitted complete clean speech segment(s), commit immediately!
            if vad_segments:
                self._turn_seq += 1
                self._speech_chunks.clear()
                self._speech_samples = 0
                self._in_speech = vad_active
                self._quiet_s = 0.0
                self._last_partial_text = ""
                for seg_pcm in vad_segments:
                    self._spawn_final_worker(seg_pcm)
                return ASROutput("none", "", 0.0, self.active_provider)

            if not self._in_speech:
                self._preroll.append(chunk)
                if vad_active:
                    self._in_speech = True
                    self._quiet_s = 0.0
                    self._speech_chunks = list(self._preroll)
                    self._speech_samples = sum(len(c) for c in self._speech_chunks)
                    self._preroll.clear()
                    self._last_partial_ts = time.monotonic()
                    self._last_partial_text = ""
                return ASROutput("none", "", 0.0, self.active_provider)

            # Currently inside speech turn
            self._speech_chunks.append(chunk)
            self._speech_samples += len(chunk)
            if vad_active:
                self._quiet_s = 0.0
            else:
                self._quiet_s += dur_s

            speech_dur_s = self._speech_samples / float(self.sample_rate)
            should_commit = (
                (self._vad is None and self._quiet_s >= self._pause_commit_s and speech_dur_s >= 0.45)
                or (speech_dur_s >= self._max_utterance_s and (self._quiet_s >= 0.18 or speech_dur_s >= self._max_utterance_s + 1.5))
            )

            if should_commit:
                pcm_snap = np.concatenate(self._speech_chunks)
                self._turn_seq += 1
                self._speech_chunks.clear()
                self._speech_samples = 0
                self._in_speech = False
                self._quiet_s = 0.0
                self._last_partial_text = ""
                if self._vad is not None:
                    try:
                        self._vad.reset()
                    except Exception:
                        pass
                self._spawn_final_worker(pcm_snap)
                return ASROutput("none", "", 0.0, self.active_provider)

            # Fire rolling non-blocking partial every `partial_interval_s` for responsive live captions
            now = time.monotonic()
            if (
                not self._partial_inflight
                and speech_dur_s >= 0.95
                and (now - self._last_partial_ts) >= self._partial_interval_s
            ):
                self._partial_inflight = True
                self._last_partial_ts = now
                pcm_snap = np.concatenate(self._speech_chunks)
                t_seq = self._turn_seq
                self._spawn_partial_worker(pcm_snap, t_seq)

        return ASROutput("none", "", 0.0, self.active_provider)

    def flush_final(self) -> ASROutput:
        """Synchronous final flush when user stops the session."""
        with self._lock:
            self._turn_seq += 1
            pcm_snap = np.concatenate(self._speech_chunks) if self._speech_chunks else None
            self._speech_chunks.clear()
            self._speech_samples = 0
            self._in_speech = False
            self._quiet_s = 0.0
            self._last_partial_text = ""
        if pcm_snap is None or len(pcm_snap) < int(self.sample_rate * 0.30):
            return ASROutput("none", "", 0.0, self.active_provider)
        try:
            text, ms, det = self._recognize_pcm(pcm_snap, is_final=True)
            if text:
                return ASROutput("final", text, ms, self.active_provider, detected_lang=det)
        except Exception as exc:
            logger.debug("Google Cloud STT stop flush skipped: %s", exc)
        return ASROutput("none", "", 0.0, self.active_provider)
