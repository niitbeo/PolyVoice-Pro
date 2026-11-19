"""
Multilingual Bidirectional Real-Time Translation Engines.

1. `OpusMTTranslator` (Offline Ultra-Fast VI <-> EN, ~77 MB INT8 each):
   - `opus-mt-vi-en-ct2-int8`: Vietnamese -> English (~40–180ms on CPU)
   - `opus-mt-en-vi-ct2-int8`: English -> Vietnamese (~80–250ms on CPU)
2. `NLLBTranslator` (Offline All 202 x 202 Languages, ~615 MB INT8):
   `NLLB-200-distilled-600M` via CTranslate2 + SentencePiece (ZERO PyTorch).
3. `FastOnlineTranslator` (Optional 0%-CPU Bidirectional Mode, ~80–160ms, 48+ major languages):
   Keeps ASR 100% local while translating text via lightweight HTTPS endpoint,
   with automatic fallback to offline Opus-MT / NLLB-200.
4. `LocalLLMTranslator` (Optional Pro Mode):
   Local Ollama / llama.cpp / LM Studio (TranslateGemma-4B / HY-MT-1.8B).
"""

from __future__ import annotations

import atexit
import html
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import ctranslate2
import httpx
import sentencepiece as spm

from languages import get_hymt2_lang_name, get_language


@dataclass
class TranslationResult:
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    latency_ms: float
    engine: str
    cached: bool = False


class LRUCache:
    def __init__(self, maxsize: int = 1024) -> None:
        self._maxsize = maxsize
        self._data: OrderedDict[tuple[str, str, str, int], str] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple[str, str, str, int]) -> Optional[str]:
        with self._lock:
            val = self._data.get(key)
            if val is not None:
                self._data.move_to_end(key)
            return val

    def set(self, key: tuple[str, str, str, int], value: str) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            if len(self._data) > self._maxsize:
                self._data.popitem(last=False)


_CLAUSE_SPLITTERS = {
    "và", "nhưng", "tức", "tại", "cho", "sau", "còn", "để", "khi", "nếu", "vì", "hoặc",
    "and", "but", "because", "so", "which", "while", "when", "however", "although", "then",
}

_ABBREV_SET = {
    "dr.", "mr.", "mrs.", "ms.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.",
    "e.g.", "i.e.", "a.m.", "p.m.", "tp.", "ts.", "ths.", "pgs.", "gs.", "no.", "fig.",
}


def _clean_punctuation(text: str) -> str:
    """Fix minor SentencePiece spacing artifacts (e.g. \"did n't\" -> \"didn't\", \"don 't\" -> \"don't\", \"a. m.\" -> \"a.m.\")."""
    s = re.sub(r"\s+(n['’]t\b)", r"\1", text, flags=re.IGNORECASE)
    s = re.sub(r"\s+(['’](?:t|s|m|re|ve|ll|d)\b)", r"\1", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\b([ap])\.\s+m\.", r"\1.m.", s, flags=re.IGNORECASE)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


def _split_semantic_sentences(text: str, max_words: int = 32) -> list[str]:
    """
    YouTube-style Semantic Sentence Splitter:
    1. Always split on true sentence boundaries (. ? ! … 。 ？ ！) so sentence-level MT models
       (EnViT5 / NLLB-200 / Opus-MT) translate each complete sentence cleanly in a single batch
       (faster O(L^2) attention + zero cross-sentence subject/object mixing).
    2. Protect decimals (3.14, 9.30), abbreviations (Dr., a.m., TP.), and numbered markers (1.).
    3. Only fall back to clause splitting (`_split_long_text`) if a single sentence exceeds `max_words`.
    """
    clean = text.strip()
    if not clean:
        return []

    # First handle CJK full-width sentence terminators (no spaces required)
    pre_segments = [seg.strip() for seg in re.split(r"(?<=[。？！])\s*", clean) if seg.strip()]

    sentences: list[str] = []
    for seg in pre_segments:
        words = seg.split()
        cur: list[str] = []
        for idx, w in enumerate(words):
            cur.append(w)
            stripped = w.rstrip('"\'”’)]')
            if not stripped:
                continue
            ends_sentence = stripped[-1] in ".?!…"
            if not ends_sentence:
                continue

            low = stripped.lower()
            is_abbrev = low in _ABBREV_SET
            is_initial = len(stripped) <= 2 and stripped.endswith(".") and stripped[0].isalpha()
            is_list_num = bool(re.fullmatch(r"\d+\.", stripped))
            is_last = idx == len(words) - 1

            if is_last or (not is_abbrev and not is_initial and not is_list_num):
                sentences.append(" ".join(cur))
                cur = []

        if cur:
            sentences.append(" ".join(cur))

    # Sub-split any abnormally long run-on sentence (> max_words) at natural commas / clauses
    final_chunks: list[str] = []
    for sent in sentences:
        if len(sent.split()) > max_words:
            final_chunks.extend(_split_long_text(sent, max_words=max_words))
        else:
            final_chunks.append(sent)

    return [c for c in final_chunks if c.strip()]


def _split_long_text(text: str, max_words: int = 34) -> list[str]:
    """
    Split overly long spoken paragraphs (> max_words) at natural punctuation or clause boundaries
    so encoder-decoder models (EnViT5 / NLLB-200) never truncate tails or hit O(L^2) slowdowns.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(words)
    while start < n:
        remaining = n - start
        if remaining <= max_words:
            chunks.append(" ".join(words[start:]))
            break

        # Target a balanced split between 55% and 100% of max_words
        lo = start + max(10, int(max_words * 0.55))
        hi = min(n - 5, start + max_words)
        split_at = hi

        # Priority 1: sentence-ending punctuation
        found = False
        for i in range(hi, lo - 1, -1):
            if words[i - 1].endswith((".", "!", "?")):
                split_at = i
                found = True
                break

        # Priority 2: comma
        if not found:
            for i in range(hi, lo - 1, -1):
                if words[i - 1].endswith((",", ";")):
                    split_at = i
                    found = True
                    break

        # Priority 3: natural clause starter
        if not found:
            for i in range(hi, lo - 1, -1):
                if words[i].lower().strip(",.?!;:") in _CLAUSE_SPLITTERS:
                    split_at = i
                    found = True
                    break

        chunks.append(" ".join(words[start:split_at]).rstrip(" ,"))
        start = split_at

    return [c for c in chunks if c.strip()]


class OpusMTTranslator:
    """Ultra-lightweight 77 MB offline MarianMT translator (CTranslate2 INT8) for VI->EN or EN->VI."""

    def __init__(
        self,
        model_dir: Path,
        src_lang: str,
        tgt_lang: str,
        engine_name: str,
        src_prefix: Optional[str] = None,
        intra_threads: int = 2,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.engine_name = engine_name
        self.src_prefix = src_prefix
        self._sp_src = spm.SentencePieceProcessor(model_file=str(self.model_dir / "source.spm"))
        self._sp_tgt = spm.SentencePieceProcessor(model_file=str(self.model_dir / "target.spm"))
        self._translator = ctranslate2.Translator(
            str(self.model_dir),
            device="cpu",
            compute_type="default",
            intra_threads=max(1, intra_threads),
            inter_threads=1,
        )
        self._lock = threading.Lock()
        self._cache = LRUCache(maxsize=1024)

    def translate(self, text: str, beam_size: int = 1) -> TranslationResult:
        clean = text.strip()
        if not clean:
            return TranslationResult("", "", self.src_lang, self.tgt_lang, 0.0, self.engine_name)

        cache_key = (clean, self.src_lang, self.tgt_lang, beam_size)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return TranslationResult(clean, cached, self.src_lang, self.tgt_lang, 0.0, self.engine_name, True)

        t0 = time.perf_counter()
        chunks = _split_semantic_sentences(clean, max_words=30)
        batch_tokens = []
        for c in chunks:
            subwords = self._sp_src.encode(c, out_type=str)
            if self.src_prefix:
                batch_tokens.append([self.src_prefix, *subwords, "</s>"])
            else:
                batch_tokens.append([*subwords, "</s>"])
        max_src_len = max(len(toks) for toks in batch_tokens)
        with self._lock:
            results = self._translator.translate_batch(
                batch_tokens,
                beam_size=max(1, beam_size),
                max_decoding_length=min(160, max(36, max_src_len * 3)),
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
            )
        decoded_parts: list[str] = []
        for res in results:
            out_tokens = [t for t in res.hypotheses[0] if t not in ("</s>", "<s>", "<pad>")]
            part = self._sp_tgt.decode(out_tokens).strip()
            if part:
                decoded_parts.append(part)
        translated = _clean_punctuation(" ".join(decoded_parts))
        ms = round((time.perf_counter() - t0) * 1000.0, 1)
        self._cache.set(cache_key, translated)
        return TranslationResult(clean, translated, self.src_lang, self.tgt_lang, ms, self.engine_name)


class EnViT5Translator:
    """
    VietAI EnViT5-Translation (CTranslate2 INT8, ~266 MB) — Single bidirectional model
    for high-accuracy Vietnamese <-> English offline translation (OpenRAIL commercial license).
    """

    def __init__(self, model_dir: Path, intra_threads: int = 4) -> None:
        self.model_dir = Path(model_dir)
        sp_path = self.model_dir / "spiece.model"
        if not sp_path.exists():
            raise FileNotFoundError(f"Missing EnViT5 SentencePiece model: {sp_path}")
        self._sp = spm.SentencePieceProcessor(model_file=str(sp_path))
        self._translator = ctranslate2.Translator(
            str(self.model_dir),
            device="cpu",
            compute_type="default",
            intra_threads=max(2, min(4, intra_threads)),
            inter_threads=1,
        )
        self._lock = threading.Lock()
        self._cache = LRUCache(maxsize=1024)

    def translate(
        self,
        text: str,
        src_lang: str = "vie_Latn",
        tgt_lang: str = "eng_Latn",
        beam_size: int = 1,
    ) -> TranslationResult:
        clean = text.strip()
        if not clean:
            return TranslationResult("", "", src_lang, tgt_lang, 0.0, "vietai-envit5")
        if src_lang == tgt_lang:
            return TranslationResult(clean, clean, src_lang, tgt_lang, 0.0, "passthrough")

        prefix = "vi" if src_lang == "vie_Latn" else "en"
        cache_key = (clean, src_lang, tgt_lang, beam_size)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return TranslationResult(clean, cached, src_lang, tgt_lang, 0.0, "vietai-envit5", True)

        t0 = time.perf_counter()
        chunks = _split_semantic_sentences(clean, max_words=32)
        batch_tokens = [self._sp.encode(f"{prefix}: {c}", out_type=str) + ["</s>"] for c in chunks]
        max_src_len = max(len(toks) for toks in batch_tokens)
        with self._lock:
            results = self._translator.translate_batch(
                batch_tokens,
                beam_size=max(1, beam_size),
                max_decoding_length=min(256, max(64, max_src_len * 3)),
                repetition_penalty=1.06,
                no_repeat_ngram_size=6,
            )

        decoded_parts: list[str] = []
        for res in results:
            out_tokens = [t for t in res.hypotheses[0] if t not in ("</s>", "<s>", "<pad>")]
            part = self._sp.decode(out_tokens).strip()
            for tag in ("en:", "vi:", "En:", "Vi:"):
                if part.startswith(tag):
                    part = part[len(tag) :].strip()
                    break
            if part:
                decoded_parts.append(part)

        translated = _clean_punctuation(" ".join(decoded_parts))
        ms = round((time.perf_counter() - t0) * 1000.0, 1)
        self._cache.set(cache_key, translated)
        return TranslationResult(clean, translated, src_lang, tgt_lang, ms, "vietai-envit5")


class NLLBTranslator:
    """Fast CPU offline translator for all 202 x 202 languages using CTranslate2 + SentencePiece."""

    def __init__(
        self,
        model_dir: Path,
        device: str = "cpu",
        intra_threads: int = 0,
        inter_threads: int = 1,
        engine_name: str = "nllb-200-ct2-int8",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self.engine_name = engine_name
        sp_path = self.model_dir / "sentencepiece.bpe.model"
        if not sp_path.exists():
            raise FileNotFoundError(f"Missing SentencePiece model: {sp_path}")

        cpu_count = os.cpu_count() or 4
        if intra_threads <= 0:
            intra_threads = max(2, min(4, cpu_count // 2))

        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(str(sp_path))

        compute_type = "default" if device == "cpu" else "auto"
        self._translator = ctranslate2.Translator(
            str(self.model_dir),
            device=device,
            compute_type=compute_type,
            intra_threads=intra_threads,
            inter_threads=inter_threads,
        )
        self._lock = threading.Lock()
        self._cache = LRUCache(maxsize=1024)

    def translate(
        self,
        text: str,
        src_lang: str = "vie_Latn",
        tgt_lang: str = "eng_Latn",
        beam_size: int = 1,
        max_decoding_length: int = 256,
    ) -> TranslationResult:
        clean = text.strip()
        if not clean:
            return TranslationResult("", "", src_lang, tgt_lang, 0.0, self.engine_name)

        if src_lang == tgt_lang:
            return TranslationResult(clean, clean, src_lang, tgt_lang, 0.0, "passthrough")

        cache_key = (clean, src_lang, tgt_lang, beam_size)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return TranslationResult(clean, cached, src_lang, tgt_lang, 0.0, self.engine_name, True)

        t0 = time.perf_counter()
        chunks = _split_semantic_sentences(clean, max_words=32)
        batch_tokens = [[src_lang, *self._sp.encode(c, out_type=str), "</s>"] for c in chunks]
        max_src_len = max(len(toks) for toks in batch_tokens)

        min_src_len = min(len(toks) for toks in batch_tokens)
        min_dec = max(2, int(min_src_len * 0.58)) if min_src_len >= 12 else 1

        with self._lock:
            results = self._translator.translate_batch(
                batch_tokens,
                target_prefix=[[tgt_lang]] * len(batch_tokens),
                beam_size=max(1, beam_size),
                min_decoding_length=min_dec,
                max_decoding_length=min(max_decoding_length, max(64, max_src_len * 3)),
                length_penalty=1.2,
                repetition_penalty=1.08,
                no_repeat_ngram_size=6,
            )

        decoded_parts: list[str] = []
        for res in results:
            out_tokens = res.hypotheses[0]
            filtered = [tok for tok in out_tokens if tok not in (tgt_lang, "</s>", "<s>", "<pad>")]
            part = self._sp.decode(filtered).strip()
            if part:
                decoded_parts.append(part)

        translated = _clean_punctuation(" ".join(decoded_parts))
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        self._cache.set(cache_key, translated)
        return TranslationResult(clean, translated, src_lang, tgt_lang, latency_ms, self.engine_name)


NLLB_TO_ISO: dict[str, str] = {
    "eng_Latn": "en",
    "vie_Latn": "vi",
    "zho_Hans": "zh-CN",
    "zho_Hant": "zh-TW",
    "jpn_Jpan": "ja",
    "kor_Hang": "ko",
    "fra_Latn": "fr",
    "deu_Latn": "de",
    "spa_Latn": "es",
    "rus_Cyrl": "ru",
    "tha_Thai": "th",
    "ita_Latn": "it",
    "por_Latn": "pt",
    "ind_Latn": "id",
    "arb_Arab": "ar",
    "hin_Deva": "hi",
    "khm_Khmr": "km",
    "lao_Laoo": "lo",
    "zsm_Latn": "ms",
    "tgl_Latn": "tl",
    "nld_Latn": "nl",
    "pol_Latn": "pl",
    "tur_Latn": "tr",
    "ukr_Cyrl": "uk",
    "ces_Latn": "cs",
    "swe_Latn": "sv",
    "dan_Latn": "da",
    "fin_Latn": "fi",
    "nob_Latn": "no",
    "ell_Grek": "el",
    "heb_Hebr": "iw",
    "hun_Latn": "hu",
    "ron_Latn": "ro",
    "bul_Cyrl": "bg",
    "hrv_Latn": "hr",
    "slk_Latn": "sk",
    "slv_Latn": "sl",
    "cat_Latn": "ca",
    "ben_Beng": "bn",
    "tam_Taml": "ta",
    "tel_Telu": "te",
    "urd_Arab": "ur",
    "pes_Arab": "fa",
    "mya_Mymr": "my",
    "npi_Deva": "ne",
    "sin_Sinh": "si",
    "swh_Latn": "sw",
    "afr_Latn": "af",
}


class FastOnlineTranslator:
    """
    Optional ultra-fast (~80–160ms, 0% CPU) bidirectional text translator while ASR remains local.
    Automatically falls back to offline NLLB-200 if offline or for unmapped rare languages.
    """

    def __init__(self, timeout_s: float = 4.0) -> None:
        self._client = httpx.Client(
            timeout=timeout_s,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        self._cache = LRUCache(maxsize=1024)

    def translate(self, text: str, src_lang: str = "vie_Latn", tgt_lang: str = "eng_Latn") -> TranslationResult:
        clean = text.strip()
        if not clean:
            return TranslationResult("", "", src_lang, tgt_lang, 0.0, "fast-online")
        if src_lang == tgt_lang:
            return TranslationResult(clean, clean, src_lang, tgt_lang, 0.0, "passthrough")

        sl = NLLB_TO_ISO.get(src_lang, "auto")
        tl = NLLB_TO_ISO.get(tgt_lang)
        if not tl:
            raise ValueError(f"Language {tgt_lang} uses offline NLLB-200")

        cache_key = (clean, src_lang, tgt_lang, 99)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return TranslationResult(clean, cached, src_lang, tgt_lang, 0.0, "fast-online", True)

        t0 = time.perf_counter()
        resp = self._client.get(
            "https://translate.googleapis.com/translate_a/single",
            params={"client": "gtx", "sl": sl, "tl": tl, "dt": "t", "q": clean},
        )
        resp.raise_for_status()
        data = resp.json()
        translated = "".join(part[0] for part in (data[0] or []) if part and part[0]).strip()
        ms = round((time.perf_counter() - t0) * 1000.0, 1)
        self._cache.set(cache_key, translated)
        return TranslationResult(clean, translated, src_lang, tgt_lang, ms, "fast-online")


class LocalLLMTranslator:
    """Optional Pro Mode bidirectional translator via local Ollama / llama.cpp / LM Studio."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434/v1",
        model: str = "translategemma:4b",
        api_key: str = "local",
        timeout_s: float = 12.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self._client = httpx.Client(timeout=timeout_s)
        self._cache = LRUCache(maxsize=512)

    def translate(
        self,
        text: str,
        src_lang: str = "vie_Latn",
        tgt_lang: str = "eng_Latn",
        context: Optional[list[tuple[str, str]]] = None,
    ) -> TranslationResult:
        clean = text.strip()
        if not clean:
            return TranslationResult("", "", src_lang, tgt_lang, 0.0, f"llm:{self.model}")

        ctx_pairs = (context or [])[-2:]
        ctx_key = "|".join(f"{s}=>{t}" for s, t in ctx_pairs)
        cache_key = (f"{ctx_key}||{clean}" if ctx_key else clean, src_lang, tgt_lang, 0)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return TranslationResult(clean, cached, src_lang, tgt_lang, 0.0, f"llm:{self.model}", True)

        src_entry = get_language(src_lang)
        tgt_entry = get_language(tgt_lang)
        src_name = src_entry.name_en if src_entry else src_lang
        tgt_name = tgt_entry.name_en if tgt_entry else tgt_lang

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    f"You are a professional real-time speech interpreter. "
                    f"Translate the spoken {src_name} naturally and accurately into {tgt_name}, "
                    f"keeping pronouns and terminology consistent with prior turns. "
                    f"Output ONLY the translation of the latest turn with no explanations, quotes, or notes."
                ),
            }
        ]
        for prev_src, prev_tgt in ctx_pairs:
            if prev_src and prev_tgt:
                messages.append({"role": "user", "content": prev_src})
                messages.append({"role": "assistant", "content": prev_tgt})
        messages.append({"role": "user", "content": clean})

        t0 = time.perf_counter()
        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": 0.0,
                "max_tokens": 256,
                "messages": messages,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        translated = data["choices"][0]["message"]["content"].strip()
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        self._cache.set(cache_key, translated)
        return TranslationResult(clean, translated, src_lang, tgt_lang, latency_ms, f"llm:{self.model}")


def find_llama_server() -> Optional[Path]:
    """Locate local `llama-server` binary (Homebrew macOS, PATH, or bundled sidecar on Win/Mac)."""
    env_bin = os.environ.get("VOICERECORE_LLAMA_SERVER")
    if env_bin and Path(env_bin).is_file():
        return Path(env_bin)

    which = shutil.which("llama-server")
    if which:
        return Path(which)

    here = Path(__file__).resolve().parent
    exe_dir = Path(sys.executable).resolve().parent
    candidates = [
        exe_dir / "llama-server",
        exe_dir / "llama-server.exe",
        exe_dir.parent / "llama-server",
        exe_dir.parent / "llama-server.exe",
        here / "llama-server",
        here / "llama-server.exe",
        here.parent / "llama-server",
        here.parent / "llama-server.exe",
        Path("/opt/homebrew/bin/llama-server"),
        Path("/usr/local/bin/llama-server"),
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


class HyMT2Translator:
    """
    Tencent Hy-MT2-1.8B (GGUF Q4_K_M, ~1.05 GB, Apache 2.0) — SOTA 2026 Offline Translator.

    Supports 38 languages/dialects (Vietnamese, English, Thai, Khmer, Burmese, Chinese,
    Japanese, Korean, French, German, Spanish, Russian, etc.) with:
    - Native Terminology Intervention (direct Glossary rules inside prompt — no placeholder hacks!)
    - Context-Aware Translation (uses previous 2 sentences for consistent pronouns & style)
    - Zero-thinking fast real-time mode (`--reasoning off`) via managed local `llama-server`
      (Apple Metal GPU on macOS, multi-thread CPU on Windows).
    """

    def __init__(self, gguf_path: Path, port: int = 18766, threads: int = 4) -> None:
        self.gguf_path = Path(gguf_path)
        self.port = port
        self.threads = max(2, min(6, threads))
        self.base_url = f"http://127.0.0.1:{port}"
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._client = httpx.Client(timeout=25.0)
        self._cache = LRUCache(maxsize=1024)
        atexit.register(self.stop)

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def ensure_started(self) -> None:
        with self._lock:
            if self.is_running:
                return
            llama_bin = find_llama_server()
            if llama_bin is None:
                raise RuntimeError("Không tìm thấy `llama-server` để chạy Hy-MT2-1.8B (cài bằng `brew install llama.cpp`).")
            if not self.gguf_path.exists():
                raise FileNotFoundError(f"Chưa tải file model Hy-MT2: {self.gguf_path}")

            cmd = [
                str(llama_bin),
                "-m", str(self.gguf_path),
                "--host", "127.0.0.1",
                "--port", str(self.port),
                "--jinja",
                "--reasoning", "off",
                "-c", "2048",
                "-ngl", "99",
                "-t", str(self.threads),
            ]
            kwargs: dict = {
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.PIPE,
            }
            if os.name == "nt":
                kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

            self._proc = subprocess.Popen(cmd, **kwargs)
            deadline = time.monotonic() + 25.0
            while time.monotonic() < deadline:
                if self._proc.poll() is not None:
                    err = ""
                    try:
                        err = self._proc.stderr.read().decode("utf-8", errors="ignore") if self._proc.stderr else ""
                    except Exception:
                        pass
                    self._proc = None
                    raise RuntimeError(f"Hy-MT2 llama-server khởi động thất bại: {err[:400]}")
                try:
                    r = self._client.get(f"{self.base_url}/health", timeout=1.0)
                    if r.status_code == 200:
                        return
                except Exception:
                    pass
                time.sleep(0.2)

            self.stop()
            raise RuntimeError("Hy-MT2 llama-server khởi động quá thời gian chờ (25s).")

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    @staticmethod
    def _clean_output(raw: str) -> str:
        s = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "“" and s[-1] == "”")):
            s = s[1:-1].strip()
        # Strip accidental prefix like "Translation:"
        s = re.sub(r"^(?:Translation|Translated text|Output)\s*:\s*", "", s, flags=re.IGNORECASE).strip()
        return _clean_punctuation(s)

    def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        context: Optional[list[tuple[str, str]]] = None,
        term_locks: Optional[list[dict[str, str]]] = None,
    ) -> TranslationResult:
        clean = text.strip()
        if not clean:
            return TranslationResult("", "", src_lang, tgt_lang, 0.0, "hy-mt2-1.8b")
        if src_lang == tgt_lang:
            return TranslationResult(clean, clean, src_lang, tgt_lang, 0.0, "passthrough")

        tgt_name = get_hymt2_lang_name(tgt_lang)
        if not tgt_name:
            raise ValueError(f"Language {tgt_lang} is outside Hy-MT2-1.8B supported 38 languages")

        # Filter only glossary terms actually appearing in this sentence (keeps prompt tiny & fast!)
        matched_terms: list[tuple[str, str]] = []
        low_clean = clean.lower()
        for rule in term_locks or []:
            s = str(rule.get("source", "")).strip()
            t = str(rule.get("target", "")).strip()
            if s and t and s.lower() in low_clean:
                matched_terms.append((s, t))

        ctx_pairs = (context or [])[-2:]
        term_sig = "|".join(f"{s}->{t}" for s, t in matched_terms)
        ctx_sig = "|".join(s for s, _ in ctx_pairs)
        cache_key = (f"{term_sig}||{ctx_sig}||{clean}", src_lang, tgt_lang, 2)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return TranslationResult(clean, cached, src_lang, tgt_lang, 0.0, "hy-mt2-1.8b", True)

        self.ensure_started()

        # Build official Hy-MT2 prompt combining Terminology + Context-Aware Background
        blocks: list[str] = []
        if matched_terms:
            lines = ["Reference the following translations:"]
            for s, t in matched_terms:
                lines.append(f"{s} translates to {t}")
            blocks.append("\n".join(lines))

        if ctx_pairs:
            bg = " ".join(s for s, _ in ctx_pairs if s).strip()
            if bg:
                blocks.append(f"[Background Information]\n{bg}")

        blocks.append(
            f"Translate the following text into {tgt_name}. "
            f"Note that you must ONLY output the translated result without any additional explanation:\n\n{clean}"
        )
        prompt = "\n\n".join(blocks)

        t0 = time.perf_counter()
        resp = self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": "hy-mt2-1.8b",
                "temperature": 0.0,
                "top_p": 0.6,
                "max_tokens": min(384, max(96, len(clean.split()) * 6)),
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        raw_out = data["choices"][0]["message"]["content"]
        translated = self._clean_output(raw_out)
        ms = round((time.perf_counter() - t0) * 1000.0, 1)
        self._cache.set(cache_key, translated)
        return TranslationResult(clean, translated, src_lang, tgt_lang, ms, "hy-mt2-1.8b")


def _to_azure_iso(nllb_code: str) -> Optional[str]:
    if nllb_code == "zho_Hans":
        return "zh-Hans"
    if nllb_code == "zho_Hant":
        return "zh-Hant"
    if nllb_code == "heb_Hebr":
        return "he"
    if nllb_code == "tgl_Latn":
        return "fil"
    return NLLB_TO_ISO.get(nllb_code)


class GoogleCloudTranslator:
    """
    Official Google Cloud Translation API v2 (`translation.googleapis.com/language/translate/v2`).
    - Free tier: 500,000 chars/month (~11 hours of speech when paired with Offline Drafts).
    - Supports Vietnamese, English, Thai, Lao, Khmer, and 130+ languages officially with zero 429 bans.
    """

    def __init__(self, api_key: str, timeout_s: float = 8.0) -> None:
        self.api_key = api_key.strip()
        self._client = httpx.Client(timeout=timeout_s)
        self._cache = LRUCache(maxsize=2048)

    def translate(self, text: str, src_lang: str = "vie_Latn", tgt_lang: str = "eng_Latn") -> TranslationResult:
        clean = text.strip()
        if not clean:
            return TranslationResult("", "", src_lang, tgt_lang, 0.0, "google-cloud")
        if src_lang == tgt_lang:
            return TranslationResult(clean, clean, src_lang, tgt_lang, 0.0, "passthrough")
        if not self.api_key:
            raise ValueError("Chưa cấu hình Google Cloud Translation API Key.")

        sl = NLLB_TO_ISO.get(src_lang)
        tl = NLLB_TO_ISO.get(tgt_lang)
        if not tl:
            raise ValueError(f"Ngôn ngữ {tgt_lang} chưa ánh xạ ISO cho Google Cloud; chuyển về NLLB Offline.")

        cache_key = (clean, src_lang, tgt_lang, 201)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return TranslationResult(clean, cached, src_lang, tgt_lang, 0.0, "google-cloud", True)

        t0 = time.perf_counter()
        payload: dict = {"q": clean, "target": tl, "format": "text"}
        if sl:
            payload["source"] = sl
        resp = self._client.post(
            "https://translation.googleapis.com/language/translate/v2",
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            json=payload,
        )
        if resp.status_code >= 400:
            err_msg = ""
            try:
                err_msg = resp.json().get("error", {}).get("message", "")
            except Exception:
                pass
            raise RuntimeError(f"Google Cloud HTTP {resp.status_code}: {err_msg or resp.reason_phrase}")
        data = resp.json()
        raw_out = data["data"]["translations"][0]["translatedText"]
        translated = html.unescape(raw_out).strip()
        ms = round((time.perf_counter() - t0) * 1000.0, 1)
        self._cache.set(cache_key, translated)
        return TranslationResult(clean, translated, src_lang, tgt_lang, ms, "google-cloud")


class AzureTranslator:
    """
    Official Microsoft Azure AI Translator v3 (`api.cognitive.microsofttranslator.com/translate`).
    - Free tier F0: 2,000,000 chars/month FREE forever (~44 hours/month with Offline Drafts!).
    - Full official support for Vietnamese (`vi`), English (`en`), Thai (`th`), Lao (`lo`), Khmer (`km`), 130+ languages.
    """

    def __init__(
        self,
        api_key: str,
        region: str = "southeastasia",
        endpoint: str = "https://api.cognitive.microsofttranslator.com",
        timeout_s: float = 8.0,
    ) -> None:
        self.api_key = api_key.strip()
        self.region = region.strip()
        self.endpoint = (endpoint or "https://api.cognitive.microsofttranslator.com").rstrip("/")
        self._client = httpx.Client(timeout=timeout_s)
        self._cache = LRUCache(maxsize=2048)

    def translate(self, text: str, src_lang: str = "vie_Latn", tgt_lang: str = "eng_Latn") -> TranslationResult:
        clean = text.strip()
        if not clean:
            return TranslationResult("", "", src_lang, tgt_lang, 0.0, "azure-translator")
        if src_lang == tgt_lang:
            return TranslationResult(clean, clean, src_lang, tgt_lang, 0.0, "passthrough")
        if not self.api_key:
            raise ValueError("Chưa cấu hình Microsoft Azure Translator API Key.")

        sl = _to_azure_iso(src_lang)
        tl = _to_azure_iso(tgt_lang)
        if not tl:
            raise ValueError(f"Ngôn ngữ {tgt_lang} chưa ánh xạ cho Azure Translator; chuyển về NLLB Offline.")

        cache_key = (clean, src_lang, tgt_lang, 202)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return TranslationResult(clean, cached, src_lang, tgt_lang, 0.0, "azure-translator", True)

        t0 = time.perf_counter()
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-type": "application/json",
        }
        if self.region:
            headers["Ocp-Apim-Subscription-Region"] = self.region
        params: dict = {"api-version": "3.0", "to": tl}
        if sl:
            params["from"] = sl

        resp = self._client.post(
            f"{self.endpoint}/translate",
            params=params,
            headers=headers,
            json=[{"Text": clean}],
        )
        if resp.status_code >= 400:
            err_msg = ""
            try:
                err_msg = resp.json().get("error", {}).get("message", "")
            except Exception:
                pass
            raise RuntimeError(f"Azure Translator HTTP {resp.status_code}: {err_msg or resp.reason_phrase}")
        data = resp.json()
        translated = str(data[0]["translations"][0]["text"]).strip()
        ms = round((time.perf_counter() - t0) * 1000.0, 1)
        self._cache.set(cache_key, translated)
        return TranslationResult(clean, translated, src_lang, tgt_lang, ms, "azure-translator")


class GeminiCloudTranslator:
    """
    Official Google Gemini API translator (`gemini-2.5-flash` default with `thinkingBudget: 0`).
    - Context-aware (uses previous 2 turns) + native Glossary Terminology enforcement.
    - Passes API key strictly via `x-goog-api-key` header (never leaked in URL/logs).
    - Supports Vietnamese, English, Thai, Lao, Khmer, and 100+ languages naturally.
    """

    _DEPRECATED_MODELS = {"gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash"}

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", timeout_s: float = 12.0) -> None:
        self.api_key = api_key.strip()
        m = (model or "gemini-2.5-flash").strip()
        if m in self._DEPRECATED_MODELS:
            m = "gemini-2.5-flash"
        self.model = m
        self._client = httpx.Client(timeout=timeout_s)
        self._cache = LRUCache(maxsize=2048)

    def translate(
        self,
        text: str,
        src_lang: str = "vie_Latn",
        tgt_lang: str = "eng_Latn",
        context: Optional[list[tuple[str, str]]] = None,
        term_locks: Optional[list[dict[str, str]]] = None,
    ) -> TranslationResult:
        clean = text.strip()
        if not clean:
            return TranslationResult("", "", src_lang, tgt_lang, 0.0, f"gemini:{self.model}")
        if src_lang == tgt_lang:
            return TranslationResult(clean, clean, src_lang, tgt_lang, 0.0, "passthrough")
        if not self.api_key:
            raise ValueError("Chưa cấu hình Google Gemini API Key.")

        src_entry = get_language(src_lang)
        tgt_entry = get_language(tgt_lang)
        src_name = src_entry.name_en if src_entry else src_lang
        tgt_name = tgt_entry.name_en if tgt_entry else tgt_lang

        active_terms = [
            r for r in (term_locks or [])
            if r.get("source") and r["source"].lower() in clean.lower()
        ]
        terms_key = "|".join(f"{r['source']}={r['target']}" for r in active_terms)
        ctx_key = context[-1][0][:40] if context else ""
        cache_key = (f"{clean}||{terms_key}||{ctx_key}", src_lang, tgt_lang, 203)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return TranslationResult(clean, cached, src_lang, tgt_lang, 0.0, f"gemini:{self.model}", True)

        parts: list[str] = [
            f"You are a professional live simultaneous interpreter translating spoken {src_name} into natural, accurate {tgt_name}.",
            "Output ONLY the final translation without explanations, quotes, or markdown.",
        ]
        if active_terms:
            rules = "; ".join(f'"{r["source"]}" -> "{r["target"]}"' for r in active_terms)
            parts.append(f"Mandatory glossary rules: {rules}.")
        if context:
            ctx_lines = [f"- {s} => {t}" for s, t in context[-2:] if s and t]
            if ctx_lines:
                parts.append("Recent conversation context (reference only, do NOT re-translate):\n" + "\n".join(ctx_lines))
        parts.append(f"Translate now:\n{clean}")
        prompt = "\n\n".join(parts)

        gen_cfg: dict = {"temperature": 0.1, "maxOutputTokens": 512}
        # Disable thinking tokens on 2.5 / 3.1 models for blazing-fast (<1s) real-time translation!
        if self.model.startswith(("gemini-2.5-", "gemini-3.1-")):
            gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}

        t0 = time.perf_counter()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        resp = self._client.post(
            url,
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": gen_cfg,
            },
        )
        if resp.status_code >= 400:
            err_msg = ""
            try:
                err_msg = resp.json().get("error", {}).get("message", "")
            except Exception:
                pass
            raise RuntimeError(f"Gemini HTTP {resp.status_code}: {err_msg or resp.reason_phrase}")
        data = resp.json()
        cand_parts = data["candidates"][0]["content"]["parts"]
        translated = "".join(p.get("text", "") for p in cand_parts if not p.get("thought")).strip().strip('"').strip("'")
        ms = round((time.perf_counter() - t0) * 1000.0, 1)
        self._cache.set(cache_key, translated)
        return TranslationResult(clean, translated, src_lang, tgt_lang, ms, f"gemini:{self.model}")


_ACTION_KEYWORDS = (
    "quyết định", "thống nhất", "triển khai", "kế hoạch", "cần", "phải", "sẽ", "gửi", "hoàn thành",
    "ngân sách", "deadline", "hạn", "báo cáo", "phụ trách", "tuần sau", "ngày mai", "tháng sau",
    "action", "plan", "budget", "deploy", "agree", "decide", "must", "will", "schedule", "follow up",
)


def summarize_meeting_entries(
    entries: list[dict],
    source_lang: str,
    target_lang: str,
    translate_fn: Optional[Callable[[str, str, str], TranslationResult]] = None,
) -> dict:
    """
    100% Offline Bilingual Meeting Minutes & Action-Item Summarizer.
    Extracts top informative highlights and action items deterministically (<50ms),
    and ensures every bullet has both original + translated text.
    """
    if not entries:
        return {
            "total_sentences": 0,
            "total_words": 0,
            "overview_src": "Chưa có nội dung hội thoại để tóm tắt.",
            "overview_tgt": "No transcript entries available to summarize.",
            "key_points": [],
            "action_items": [],
        }

    total_words = sum(len(str(e.get("source_text", "")).split()) for e in entries)
    scored: list[tuple[float, int, dict]] = []
    actions: list[dict] = []

    for idx, e in enumerate(entries):
        src = str(e.get("source_text", "")).strip()
        tgt = str(e.get("translated_text", "")).strip()
        if not src:
            continue
        low = src.lower() + " " + tgt.lower()
        words = src.split()
        score = min(len(words), 28) * 0.4
        # Boost sentences with concrete numbers, times, dates, currencies
        if re.search(r"\d+", src):
            score += 6.0
        # Boost decision / planning keywords
        kw_hits = sum(1 for k in _ACTION_KEYWORDS if k in low)
        score += kw_hits * 4.5
        if idx == 0 or idx == len(entries) - 1:
            score += 2.5
        scored.append((score, idx, e))
        if kw_hits >= 1 or re.search(r"\b(\d+\s*(?:giờ|triệu|tỷ|ngày|tuần|tháng|%|million|am|pm))\b", low):
            actions.append(
                {
                    "timestamp": e.get("timestamp", ""),
                    "speaker": e.get("speaker", ""),
                    "source_text": src,
                    "translated_text": tgt,
                }
            )

    # Pick top 5 key points preserving chronological order
    top_scored = sorted(scored, key=lambda x: x[0], reverse=True)[:5]
    top_chrono = sorted(top_scored, key=lambda x: x[1])
    key_points = [
        {
            "timestamp": e.get("timestamp", ""),
            "speaker": e.get("speaker", ""),
            "source_text": str(e.get("source_text", "")).strip(),
            "translated_text": str(e.get("translated_text", "")).strip(),
        }
        for _, _, e in top_chrono
    ]

    first_ts = entries[0].get("timestamp", "")
    last_ts = entries[-1].get("timestamp", "")
    overview_src = (
        f"Phiên họp ghi nhận {len(entries)} câu ({total_words} từ) từ {first_ts} đến {last_ts}, "
        f"trích xuất {len(key_points)} ý chính và {min(len(actions), 6)} mục hành động / số liệu quan trọng."
    )
    overview_tgt = (
        f"Session recorded {len(entries)} sentences ({total_words} words) from {first_ts} to {last_ts}, "
        f"with {len(key_points)} key highlights and {min(len(actions), 6)} action items / key figures."
    )
    if translate_fn is not None and target_lang not in ("eng_Latn", "vie_Latn"):
        try:
            tr_ov = translate_fn(overview_src, "vie_Latn", target_lang)
            if tr_ov.translated_text:
                overview_tgt = tr_ov.translated_text
        except Exception:
            pass

    return {
        "total_sentences": len(entries),
        "total_words": total_words,
        "time_range": f"{first_ts} – {last_ts}",
        "overview_src": overview_src,
        "overview_tgt": overview_tgt,
        "key_points": key_points,
        "action_items": actions[:6],
    }

