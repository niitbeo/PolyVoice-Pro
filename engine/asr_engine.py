"""
Cross-Platform Real-Time Streaming ASR Engines (Vietnamese & English ONNX).

1. `ZipformerStreamingViEngine`:
   `hynt/Zipformer-30M-RNNT-Streaming-6000h` (51 MB, chunk-16) — TRUE O(1) Vietnamese streaming (~10–45ms/chunk).
2. `ZipformerStreamingEnEngine`:
   `sherpa-onnx-streaming-zipformer-en-20M-2023-02-17` (30 MB INT8) — TRUE O(1) English streaming (~10–30ms/chunk).
3. `VietASRStreamingEngine`:
   Embedded 78 MB INT8 Zipformer2 + Silero VAD + Punctuation/ITN with max-utterance auto-commit.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sentencepiece as spm
import sherpa_onnx
import vietasr


@dataclass
class ASROutput:
    kind: str  # "partial" | "final" | "none"
    text: str
    latency_ms: float
    provider: str = "CPUExecutionProvider"


def _format_sentence(raw: str) -> str:
    txt = " ".join(raw.strip().split())
    if not txt:
        return ""
    if txt.isupper():
        txt = txt.lower()
        txt = txt[0].upper() + txt[1:]
    return txt


class _SherpaOnlineTransducerBase:
    """Base O(1) stateful streaming transducer engine via sherpa-onnx OnlineRecognizer."""

    def __init__(
        self,
        tokens: Path,
        encoder: Path,
        decoder: Path,
        joiner: Path,
        provider_label: str,
        sample_rate: int = 16000,
        num_threads: int = 2,
        bpe_model: Path | None = None,
        hotwords: list[str] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.active_provider = provider_label
        self._tokens = Path(tokens)
        self._encoder = Path(encoder)
        self._decoder = Path(decoder)
        self._joiner = Path(joiner)
        self._num_threads = num_threads
        self._bpe_model = Path(bpe_model) if bpe_model and Path(bpe_model).exists() else None
        self._hotwords_file = self._tokens.parent / "active_hotwords.txt"
        self._hotwords_count = 0
        self._lock = threading.Lock()
        self._last_partial = ""

        self._valid_tokens: set[str] = set()
        try:
            for line in self._tokens.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if parts:
                    self._valid_tokens.add(parts[0])
        except Exception:
            pass

        self._recognizer = self._build_recognizer(hotwords)
        self._stream = self._recognizer.create_stream()

    def _encode_hotwords_file(self, phrases: list[str] | None) -> Path | None:
        if not phrases or self._bpe_model is None:
            self._hotwords_count = 0
            return None
        try:
            sp = spm.SentencePieceProcessor(model_file=str(self._bpe_model))
            seen: set[str] = set()
            lines: list[str] = []
            for raw in phrases:
                p = " ".join(str(raw).strip().split()).upper()
                if len(p) < 2 or p in seen:
                    continue
                seen.add(p)
                toks = sp.encode(p, out_type=str)
                if not toks or any(t == "<unk>" or (self._valid_tokens and t not in self._valid_tokens) for t in toks):
                    continue
                lines.append(" ".join(toks) + " :2.2")
                if len(lines) >= 80:
                    break
            if not lines:
                self._hotwords_count = 0
                return None
            self._hotwords_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self._hotwords_count = len(lines)
            return self._hotwords_file
        except Exception:
            self._hotwords_count = 0
            return None

    def _build_recognizer(self, hotwords: list[str] | None = None):
        hw_path = self._encode_hotwords_file(hotwords)
        base_kwargs = dict(
            tokens=str(self._tokens),
            encoder=str(self._encoder),
            decoder=str(self._decoder),
            joiner=str(self._joiner),
            num_threads=self._num_threads,
            sample_rate=self.sample_rate,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.0,
            rule2_min_trailing_silence=1.15,
            rule3_min_utterance_length=9.5,
            provider="cpu",
        )
        if hw_path is not None:
            try:
                return sherpa_onnx.OnlineRecognizer.from_transducer(
                    **base_kwargs,
                    decoding_method="modified_beam_search",
                    hotwords_file=str(hw_path),
                    hotwords_score=1.5,
                )
            except Exception:
                self._hotwords_count = 0
        return sherpa_onnx.OnlineRecognizer.from_transducer(
            **base_kwargs,
            decoding_method="greedy_search",
        )

    def update_hotwords(self, phrases: list[str] | None) -> int:
        """Dynamically reload acoustic hotwords (e.g. when user updates Meeting Glossary)."""
        with self._lock:
            self._recognizer = self._build_recognizer(phrases)
            self._stream = self._recognizer.create_stream()
            self._last_partial = ""
            return self._hotwords_count

    def reset(self) -> None:
        with self._lock:
            self._recognizer.reset(self._stream)
            self._last_partial = ""

    def accept_pcm(self, pcm_s16: np.ndarray) -> ASROutput:
        t0 = time.perf_counter()
        audio_f32 = pcm_s16.astype(np.float32) / 32768.0

        with self._lock:
            self._stream.accept_waveform(self.sample_rate, audio_f32)
            while self._recognizer.is_ready(self._stream):
                self._recognizer.decode_stream(self._stream)

            raw_text = self._recognizer.get_result(self._stream).strip()
            is_ep = self._recognizer.is_endpoint(self._stream)
            ms = round((time.perf_counter() - t0) * 1000.0, 1)

            if is_ep:
                self._recognizer.reset(self._stream)
                self._last_partial = ""
                formatted = _format_sentence(raw_text)
                if formatted:
                    return ASROutput("final", formatted, ms, self.active_provider)
                return ASROutput("none", "", ms, self.active_provider)

            formatted = _format_sentence(raw_text)
            if formatted and formatted != self._last_partial:
                self._last_partial = formatted
                return ASROutput("partial", formatted, ms, self.active_provider)
            return ASROutput("none", "", ms, self.active_provider)

    def flush_final(self) -> ASROutput:
        t0 = time.perf_counter()
        with self._lock:
            self._stream.accept_waveform(
                self.sample_rate,
                np.zeros(int(self.sample_rate * 0.4), dtype=np.float32),
            )
            while self._recognizer.is_ready(self._stream):
                self._recognizer.decode_stream(self._stream)
            raw_text = self._recognizer.get_result(self._stream).strip()
            self._recognizer.reset(self._stream)
            self._last_partial = ""
            ms = round((time.perf_counter() - t0) * 1000.0, 1)
            formatted = _format_sentence(raw_text)
            if formatted:
                return ASROutput("final", formatted, ms, self.active_provider)
            return ASROutput("none", "", ms, self.active_provider)

    def close(self) -> None:
        self.reset()


class ZipformerStreamingViEngine(_SherpaOnlineTransducerBase):
    """Vietnamese 30M 6000h streaming Zipformer with BPE Glossary Hotwords."""

    def __init__(
        self,
        model_dir: Path,
        sample_rate: int = 16000,
        num_threads: int = 2,
        hotwords: list[str] | None = None,
    ) -> None:
        d = Path(model_dir)
        super().__init__(
            tokens=d / "tokens.txt",
            encoder=d / "encoder-epoch-31-avg-11-chunk-16-left-128.fp16.onnx",
            decoder=d / "decoder-epoch-31-avg-11-chunk-16-left-128.fp16.onnx",
            joiner=d / "joiner-epoch-31-avg-11-chunk-16-left-128.fp16.onnx",
            provider_label="sherpa-onnx-vi-30m",
            sample_rate=sample_rate,
            num_threads=num_threads,
            bpe_model=d / "bpe.model",
            hotwords=hotwords,
        )


class ZipformerStreamingEnEngine(_SherpaOnlineTransducerBase):
    """English 20M INT8 streaming Zipformer (cross-platform Windows & macOS)."""

    def __init__(self, model_dir: Path, sample_rate: int = 16000, num_threads: int = 2) -> None:
        d = Path(model_dir)
        super().__init__(
            tokens=d / "tokens.txt",
            encoder=d / "encoder-epoch-99-avg-1.int8.onnx",
            decoder=d / "decoder-epoch-99-avg-1.int8.onnx",
            joiner=d / "joiner-epoch-99-avg-1.int8.onnx",
            provider_label="sherpa-onnx-en-20m",
            sample_rate=sample_rate,
            num_threads=num_threads,
        )


class VietASRStreamingEngine:
    """Embedded 78 MB Vietnamese ASR (`viet-asr`) with built-in Silero VAD, punctuation, and ITN."""

    def __init__(self, provider: str = "cpu", sample_rate: float = 16000.0) -> None:
        self.provider = provider
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._pipeline = (
            vietasr.Pipeline.new()
            .set_execution_provider(provider)
            .add("vietasr")
            .build()
        )
        self._session = self._pipeline.stream(sample_rate)
        self.active_provider = self._session.active_execution_provider() or provider
        self._last_partial = ""
        self._active_chunks = 0
        self._quiet_chunks = 0

    def reset(self) -> None:
        with self._lock:
            self._session.reset()
            self._last_partial = ""
            self._active_chunks = 0
            self._quiet_chunks = 0

    def accept_pcm(self, pcm_s16: np.ndarray) -> ASROutput:
        t0 = time.perf_counter()
        rms = float(np.sqrt(np.mean(np.square(pcm_s16.astype(np.float32) / 32768.0))))

        with self._lock:
            is_final = self._session.accept(pcm_s16)
            if is_final:
                res = self._session.result()
                text = (res.text or "").strip()
                self._last_partial = ""
                self._active_chunks = 0
                self._quiet_chunks = 0
                ms = round((time.perf_counter() - t0) * 1000.0, 1)
                if text:
                    return ASROutput("final", text, ms, self.active_provider)
                return ASROutput("none", "", ms, self.active_provider)

            pr = self._session.partial()
            partial_text = (pr.partial or "").strip()

            if partial_text:
                self._active_chunks += 1
                if rms < 0.015:
                    self._quiet_chunks += 1
                else:
                    self._quiet_chunks = 0

                should_split = (
                    (self._active_chunks >= 55 and self._quiet_chunks >= 12)
                    or (self._active_chunks >= 150)
                    or (self._active_chunks >= 95 and partial_text.endswith((".", "!", "?")))
                )
                if should_split:
                    fin = self._session.final()
                    text = (fin.text or partial_text).strip()
                    self._session.reset()
                    self._last_partial = ""
                    self._active_chunks = 0
                    self._quiet_chunks = 0
                    ms = round((time.perf_counter() - t0) * 1000.0, 1)
                    if text:
                        return ASROutput("final", text, ms, self.active_provider)
                    return ASROutput("none", "", ms, self.active_provider)

            ms = round((time.perf_counter() - t0) * 1000.0, 1)
            if partial_text and partial_text != self._last_partial:
                self._last_partial = partial_text
                return ASROutput("partial", partial_text, ms, self.active_provider)
            return ASROutput("none", "", ms, self.active_provider)

    def flush_final(self) -> ASROutput:
        t0 = time.perf_counter()
        with self._lock:
            res = self._session.final()
            text = (res.text or self._last_partial or "").strip()
            self._session.reset()
            self._last_partial = ""
            self._active_chunks = 0
            self._quiet_chunks = 0
            ms = round((time.perf_counter() - t0) * 1000.0, 1)
            if text:
                return ASROutput("final", text, ms, self.active_provider)
            return ASROutput("none", "", ms, self.active_provider)

    def close(self) -> None:
        with self._lock:
            try:
                self._session.close()
            except Exception:
                pass
            try:
                self._pipeline.close()
            except Exception:
                pass


class VietASRBatchRefiner:
    """
    Pass-2 Vietnamese batch refiner using embedded `viet-asr` (78 MB, 0 extra download).
    Runs asynchronously on committed paragraph audio (~5x realtime on CPU) to restore
    punctuation, capitalization, numbers (ITN), and fix acoustic errors without ever
    blocking Pass-1 streaming.
    """

    def __init__(self, provider: str = "cpu", sample_rate: float = 16000.0) -> None:
        self.provider = provider
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._pipeline = (
            vietasr.Pipeline.new()
            .set_execution_provider(provider)
            .add("vietasr")
            .build()
        )

    def refine(
        self,
        pcm_s16: np.ndarray,
        fallback_text: str,
        protected_terms: list[str] | None = None,
    ) -> tuple[str, float]:
        if pcm_s16 is None or len(pcm_s16) < int(self.sample_rate * 0.4):
            return fallback_text, 0.0
        t0 = time.perf_counter()
        try:
            with self._lock:
                res = self._pipeline.transcribe(pcm_s16, sample_rate=self.sample_rate)
            refined = (res.text or "").strip()
            ms = round((time.perf_counter() - t0) * 1000.0, 1)
            # Safety guard 1: never drop a valid Pass-1 transcript if Pass-2 VAD clipped soft audio
            fb_words = len(fallback_text.split())
            rf_words = len(refined.split())
            if not refined or (fb_words > 0 and rf_words < max(2, int(fb_words * 0.55))):
                return fallback_text, ms

            # Safety guard 2: if Pass-1 (+BPE Hotwords / ASR fixes) heard protected domain/brand terms
            # (e.g. MCP, API, CLI, Cloudflare, proper names) that VietASR Pass-2 dropped/corrupted
            # (e.g. MCP -> MB), keep Pass-1 so glossary keywords are NEVER lost!
            if fallback_text and protected_terms:
                dropped = 0
                for term in protected_terms:
                    t = term.strip()
                    if len(t) < 2:
                        continue
                    pat = re.compile(rf"(?<!\w){re.escape(t)}(?!\w)", re.IGNORECASE | re.UNICODE)
                    if pat.search(fallback_text) and not pat.search(refined):
                        dropped += 1
                if dropped >= 1:
                    return _format_sentence(fallback_text), ms

            return refined, ms
        except Exception:
            return fallback_text, round((time.perf_counter() - t0) * 1000.0, 1)

    def close(self) -> None:
        with self._lock:
            try:
                self._pipeline.close()
            except Exception:
                pass


class MoonshineV2EnRefiner:
    """
    Pass-2 & Batch English ASR Refiner using `sherpa-onnx-moonshine-base-en-quantized-2026-02-27`
    (106 MB .ort model, 100% cross-platform Win & Mac CPU).

    Benchmarked on real YouTube TED Talks:
    - Speed: ~42-44x real-time on CPU (~150-250ms per committed sentence!)
    - Accuracy: 1.82% WER on TED UK (down from 20.71% on Zipformer-20M!) & 11.18% on fast TED US
    - Built-in punctuation, capitalization, and proper handling of AI/tech terms (AI, AGI, Sora, MCP).
    """

    def __init__(self, model_dir: Path, sample_rate: int = 16000, num_threads: int = 4) -> None:
        d = Path(model_dir)
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_moonshine_v2(
            encoder=str(d / "encoder_model.ort"),
            decoder=str(d / "decoder_model_merged.ort"),
            tokens=str(d / "tokens.txt"),
            num_threads=num_threads,
            provider="cpu",
        )

    def refine(self, pcm_s16: np.ndarray, fallback_text: str = "") -> tuple[str, float]:
        if pcm_s16 is None or len(pcm_s16) < int(self.sample_rate * 0.35):
            return fallback_text, 0.0
        t0 = time.perf_counter()
        try:
            audio_f32 = pcm_s16.astype(np.float32) / 32768.0
            with self._lock:
                stream = self._recognizer.create_stream()
                stream.accept_waveform(self.sample_rate, audio_f32)
                self._recognizer.decode_stream(stream)
                refined = (stream.result.text or "").strip()
            ms = round((time.perf_counter() - t0) * 1000.0, 1)
            fb_words = len(fallback_text.split())
            rf_words = len(refined.split())
            if refined and (fb_words == 0 or rf_words >= max(2, int(fb_words * 0.5))):
                return refined, ms
            return fallback_text, ms
        except Exception:
            return fallback_text, round((time.perf_counter() - t0) * 1000.0, 1)
