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
import difflib
import html
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import weakref
from collections import OrderedDict, deque
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


def _adaptive_mt_threads(requested: int = 0) -> int:
    """Choose optimal CTranslate2/llama CPU thread budget without oversubscribing low-core CPUs."""
    cpu_count = os.cpu_count() or 4
    auto_budget = max(2, min(4, cpu_count // 2))
    if requested <= 0:
        return auto_budget
    return max(1, min(requested, auto_budget))


def _load_sentencepiece(sp_path: Path) -> spm.SentencePieceProcessor:
    """Load SentencePiece via in-memory proto bytes (immune to Windows non-ASCII paths in C++ std::ifstream)."""
    return spm.SentencePieceProcessor(model_proto=sp_path.read_bytes())


def _clean_punctuation(text: str) -> str:
    """Fix minor SentencePiece spacing artifacts (e.g. \"did n't\" -> \"didn't\", \"do n'tthink\" -> \"don't think\", \"o 'clock\" -> \"o'clock\", \"two - way\" -> \"two-way\", \"a. m.\" -> \"a.m.\", \"9: 30\" -> \"9:30\")."""
    s = re.sub(r"\b(n['’]t)([a-zA-Z])", r"\1 \2", text, flags=re.IGNORECASE)
    s = re.sub(r"\s+(n['’]t\b)", r"\1", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+(['’](?:t|s|m|re|ve|ll|d)\b)", r"\1", s, flags=re.IGNORECASE)
    s = re.sub(r"\bo\s+['’]clock\b", "o'clock", s, flags=re.IGNORECASE)
    # Collapse SentencePiece spaced hyphens inside compound words & numbers ("two - way" -> "two-way", "Twenty - four" -> "Twenty-four")
    s = re.sub(r"(?<=[A-Za-z0-9])\s+-\s+(?=[A-Za-z0-9])", "-", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\b([ap])\.\s+m\.", r"\1.m.", s, flags=re.IGNORECASE)
    # Fix spaced times ("9: 30" -> "9:30") and short decimals ("9. 30" -> "9.30") without merging 4-digit years
    s = re.sub(r"(?<=\d):\s+(?=\d)", ":", s)
    s = re.sub(r"(?<!\d)(\d{1,3})\.\s+(\d{1,3})(?!\d)", r"\1.\2", s)
    # Remove spaces immediately inside parentheses/brackets and matched quotes
    s = re.sub(r"([(\[{])\s+", r"\1", s)
    s = re.sub(r"\s+([)\]}])", r"\1", s)
    s = re.sub(r'(?<!\S)"\s+([^"]+?)\s+"(?=$|[\s,.;:!?)\]])', r'"\1"', s)
    s = re.sub(r"“\s+([^”]+?)\s+”", r"“\1”", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    # Strip stray unmatched leading/trailing quote hallucinated by seq2seq MT (e.g. '" Then this hand...')
    if s.count('"') == 1 and (s.startswith('"') or s.endswith('"')):
        s = s.strip('"').strip()
    if (s.startswith("“") and "”" not in s) or (s.endswith("”") and "“" not in s):
        s = s.strip("“”").strip()
    return s


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


class EnViT5Translator:
    """
    VietAI EnViT5-Translation (CTranslate2 INT8, ~266 MB) — Single bidirectional model
    for high-accuracy Vietnamese <-> English offline translation (OpenRAIL commercial license).
    """

    def __init__(self, model_dir: Path, intra_threads: int = 0) -> None:
        self.model_dir = Path(model_dir)
        sp_path = self.model_dir / "spiece.model"
        if not sp_path.exists():
            raise FileNotFoundError(f"Missing EnViT5 SentencePiece model: {sp_path}")
        self._sp = _load_sentencepiece(sp_path)
        self.device = "cpu"
        translator = None
        # Default to CPU (empirically 3.0-3.8x faster than experimental CT2 MPS on Apple Silicon,
        # even under live RT Zipformer ASR). Keep opt-in override via VOICERECORE_ENVIT5_DEVICE=mps.
        want_mps = os.environ.get("VOICERECORE_ENVIT5_DEVICE", "").strip().lower() == "mps"
        if want_mps and platform.system() == "Darwin" and platform.machine() == "arm64":
            try:
                translator = ctranslate2.Translator(
                    str(self.model_dir),
                    device="mps",
                    compute_type="default",
                    intra_threads=1,
                    inter_threads=1,
                )
                self.device = "mps"
            except Exception:
                translator = None
        if translator is None:
            translator = ctranslate2.Translator(
                str(self.model_dir),
                device="cpu",
                compute_type="default",
                intra_threads=_adaptive_mt_threads(intra_threads),
                inter_threads=1,
            )
            self.device = "cpu"
        self._translator = translator
        self._lock = threading.Lock()
        self._cache = LRUCache(maxsize=1024)

    def translate(
        self,
        text: str,
        src_lang: str = "vie_Latn",
        tgt_lang: str = "eng_Latn",
        beam_size: int = 1,
        is_draft: bool = False,
    ) -> TranslationResult:
        clean = text.strip()
        if not clean:
            return TranslationResult("", "", src_lang, tgt_lang, 0.0, "vietai-envit5")
        if src_lang == tgt_lang:
            return TranslationResult(clean, clean, src_lang, tgt_lang, 0.0, "passthrough")

        prefix = "vi" if src_lang == "vie_Latn" else "en"
        cache_key = (clean, src_lang, tgt_lang, beam_size, is_draft)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return TranslationResult(clean, cached, src_lang, tgt_lang, 0.0, "vietai-envit5", True)

        chunks = _split_semantic_sentences(clean, max_words=60)
        batch_tokens = [self._sp.encode(f"{prefix}: {c}", out_type=str) + ["</s>"] for c in chunks]
        max_src_len = max(len(toks) for toks in batch_tokens)
        is_short_src = len(clean.split()) <= 6
        eff_beam = max(1, beam_size)
        max_dec = min(256, max(16, int(max_src_len * 2.5) + 6))
        # Finals use gentle rep_penalty=1.05 (preserves every word faithfully without beam=2 5x CPU spike);
        # Short live drafts use rep_penalty=1.18 to block prefix echoing!
        rep_pen = 1.18 if (is_draft and is_short_src) else (1.05 if not is_draft else 1.10)
        len_pen = 1.15 if (eff_beam >= 2 and not is_short_src) else 1.0
        with self._lock:
            t0 = time.perf_counter()
            results = self._translator.translate_batch(
                batch_tokens,
                beam_size=eff_beam,
                max_decoding_length=max_dec,
                length_penalty=len_pen,
                repetition_penalty=rep_pen,
                no_repeat_ngram_size=3 if is_short_src else 4,
            )
            ms = round((time.perf_counter() - t0) * 1000.0, 1)

        decoded_parts: list[str] = []
        for res in results:
            out_tokens = [t for t in res.hypotheses[0] if t not in ("</s>", "<s>", "<pad>")]
            part = self._sp.decode(out_tokens).strip()
            for tag in ("en:", "vi:", "En:", "Vi:"):
                if part.startswith(tag):
                    part = part[len(tag) :].strip()
                    break
            # Short-fragment echo guard: EnViT5 beam=1 on <=6-word drafts (even with trailing '.')
            # can emit multiple echoed variations ("Chào mừng anh chào đón em...", or 3 sentences).
            if is_short_src and part:
                src_sents = [s for s in re.split(r"[.!?]+", clean) if s.strip()]
                max_sents = max(1, len(src_sents))
                out_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", part) if s.strip()]
                if len(out_sents) > max_sents:
                    part = " ".join(out_sents[:max_sents]).strip()
                # Also clip unpunctuated opening-phrase echoes (e.g. "Chào mừng các bạn chào mừng các bạn...")
                ptoks = part.split()
                if len(ptoks) >= 6:
                    lead2 = (ptoks[0].lower(), ptoks[1].lower())
                    for k in range(2, len(ptoks) - 1):
                        if (ptoks[k].lower(), ptoks[k + 1].lower()) == lead2:
                            part = " ".join(ptoks[:k]).rstrip(" ,;:-")
                            break
                        if ptoks[k].lower() == lead2[0] and k >= 3 and len(clean.split()) <= 3:
                            part = " ".join(ptoks[:k]).rstrip(" ,;:-")
                            break
            if part:
                decoded_parts.append(part)

        translated = _clean_punctuation(" ".join(decoded_parts))
        self._cache.set(cache_key, translated)
        return TranslationResult(clean, translated, src_lang, tgt_lang, ms, "vietai-envit5")


_RE_VI_TRIEU_TY = re.compile(
    r"(?i)(?<!\w)(\d+(?:[.,]\d+)?)\s+triệu\s+t[ỷỉ](?:\s+(đồng|vnd))?(?!\w)"
)
_VI_SMALL_NUM_WORDS: dict[str, int] = {
    "một": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4, "năm": 5,
    "sáu": 6, "bảy": 7, "bẩy": 7, "tám": 8, "chín": 9, "mười": 10,
}
_RE_VI_WORD_TRIEU_TY = re.compile(
    r"(?i)(?<!\w)(" + "|".join(_VI_SMALL_NUM_WORDS.keys()) + r")\s+triệu\s+t[ỷỉ](?:\s+(đồng|vnd))?(?!\w)"
)
_CJK_TRILLION_UNITS: dict[str, tuple[str, str]] = {
    "kor_Hang": ("조", "조 동"),
    "jpn_Jpan": ("兆", "兆ドン"),
    "zho_Hans": ("万亿", "万亿越南盾"),
    "zho_Hant": ("兆", "兆越南盾"),
}


def _normalize_vi_large_units_for_cjk(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Prevent 1,000x magnitude collapse on Vietnamese `triệu tỷ` (10^15) into Korean/Japanese/Chinese
    trillion (`조`/`兆`, 10^12) and accidental USD/KRW currency hallucinations (`2 triệu tỷ đồng` -> `2조 달러`):
    convert `X triệu tỷ [đồng]` directly to the exact target-language trillion form
    (e.g. `5,2 triệu tỷ đồng` -> `5,200조 동` / `5,200兆ドン` / `5,200万亿越南盾`).
    """
    if src_lang != "vie_Latn" or tgt_lang not in _CJK_TRILLION_UNITS:
        return text

    unit_plain, unit_vnd = _CJK_TRILLION_UNITS[tgt_lang]

    def _format_val(val: float, has_currency: bool) -> str:
        scaled = int(round(val * 1000.0))
        return f"{scaled:,}{unit_vnd if has_currency else unit_plain}"

    def _repl_digit(m: re.Match[str]) -> str:
        raw_num = m.group(1).replace(",", ".")
        try:
            val = float(raw_num)
        except ValueError:
            return m.group(0)
        return _format_val(val, bool(m.group(2)))

    def _repl_word(m: re.Match[str]) -> str:
        val = _VI_SMALL_NUM_WORDS.get(m.group(1).lower())
        if val is None:
            return m.group(0)
        return _format_val(float(val), bool(m.group(2)))

    out = _RE_VI_TRIEU_TY.sub(_repl_digit, text)
    out = _RE_VI_WORD_TRIEU_TY.sub(_repl_word, out)
    return out


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

        intra_threads = _adaptive_mt_threads(intra_threads)

        self._sp = _load_sentencepiece(sp_path)

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

        norm_clean = _normalize_vi_large_units_for_cjk(clean, src_lang, tgt_lang)
        chunks = _split_semantic_sentences(norm_clean, max_words=60)
        batch_tokens = [[src_lang, *self._sp.encode(c, out_type=str), "</s>"] for c in chunks]
        max_src_len = max(len(toks) for toks in batch_tokens)

        min_src_len = min(len(toks) for toks in batch_tokens)
        min_dec = max(2, int(min_src_len * 0.58)) if min_src_len >= 12 else 1

        with self._lock:
            t0 = time.perf_counter()
            results = self._translator.translate_batch(
                batch_tokens,
                target_prefix=[[tgt_lang]] * len(batch_tokens),
                beam_size=max(1, beam_size),
                min_decoding_length=min_dec,
                max_decoding_length=min(max_decoding_length, max(64, max_src_len * 3)),
                length_penalty=1.2,
                repetition_penalty=1.08,
                no_repeat_ngram_size=6,
                suppress_sequences=[["<unk>"]],
            )
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        decoded_parts: list[str] = []
        for res in results:
            out_tokens = res.hypotheses[0]
            filtered = [tok for tok in out_tokens if tok not in (tgt_lang, "</s>", "<s>", "<pad>", "<unk>")]
            part = self._sp.decode(filtered).replace("\u2047", "").strip()
            if part:
                decoded_parts.append(part)

        translated = _clean_punctuation(" ".join(decoded_parts))

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
        self._timeout_s = timeout_s
        self._client_lock = threading.Lock()
        self._client = httpx.Client(
            timeout=timeout_s,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        self._cache = LRUCache(maxsize=1024)

    def _get_client(self) -> httpx.Client:
        if self._client.is_closed:
            self._client = httpx.Client(
                timeout=self._timeout_s,
                headers={"User-Agent": "Mozilla/5.0"},
            )
        return self._client

    def close(self) -> None:
        with self._client_lock:
            try:
                self._client.close()
            except Exception:
                pass

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
        with self._client_lock:
            resp = self._get_client().get(
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
        self._timeout_s = timeout_s
        self._client_lock = threading.Lock()
        self._client = httpx.Client(timeout=timeout_s)
        self._cache = LRUCache(maxsize=512)

    def _get_client(self) -> httpx.Client:
        if self._client.is_closed:
            self._client = httpx.Client(timeout=self._timeout_s)
        return self._client

    def close(self) -> None:
        with self._client_lock:
            try:
                self._client.close()
            except Exception:
                pass

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
        with self._client_lock:
            resp = self._get_client().post(
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
    """Locate portable bundled `llama-server` (Win Vulkan/CPU or Mac Metal) or system PATH/Homebrew."""
    env_bin = os.environ.get("VOICERECORE_LLAMA_SERVER")
    if env_bin and Path(env_bin).is_file():
        return Path(env_bin)

    here = Path(__file__).resolve().parent
    exe_dir = Path(sys.executable).resolve().parent
    candidates = [
        # Bundled portable llama-server inside packaged sidecar (highest priority)
        exe_dir / "llama" / "llama-server.exe",
        exe_dir / "llama" / "llama-server",
        exe_dir / "llama-server.exe",
        exe_dir / "llama-server",
        exe_dir.parent / "llama" / "llama-server.exe",
        exe_dir.parent / "llama" / "llama-server",
        exe_dir.parent / "llama-server.exe",
        exe_dir.parent / "llama-server",
        # Local repo vendor directory (dev / CI build staging)
        here / "vendor" / "llama-bin" / "llama-server.exe",
        here / "vendor" / "llama-bin" / "llama-server",
        here / "llama" / "llama-server.exe",
        here / "llama" / "llama-server",
        here / "llama-server.exe",
        here / "llama-server",
        here.parent / "llama-server.exe",
        here.parent / "llama-server",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p

    which = shutil.which("llama-server")
    if which:
        return Path(which)

    for sys_p in (Path("/opt/homebrew/bin/llama-server"), Path("/usr/local/bin/llama-server")):
        if sys_p.exists() and sys_p.is_file():
            return sys_p
    return None


def _is_local_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _pick_free_local_port(preferred: int) -> int:
    if _is_local_port_free(preferred):
        return preferred
    for candidate in range(preferred + 1, preferred + 20):
        if _is_local_port_free(candidate):
            return candidate
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _make_weak_stop(ref: "weakref.ReferenceType[HyMT2Translator]") -> Callable[[], None]:
    def _cleanup() -> None:
        inst = ref()
        if inst is not None:
            inst.stop()
    return _cleanup


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

    def __init__(self, gguf_path: Path, port: int = 18766, threads: int = 0) -> None:
        self.gguf_path = Path(gguf_path)
        self.port = port
        self.threads = _adaptive_mt_threads(threads)
        self.base_url = f"http://127.0.0.1:{port}"
        self.active_backend: str = ""
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_ring: deque[str] = deque()
        self._stderr_bytes: int = 0
        self._stderr_lock = threading.Lock()
        self._lock = threading.Lock()
        self._client_lock = threading.Lock()
        self._client = httpx.Client(timeout=25.0)
        self._cache = LRUCache(maxsize=1024)
        self._atexit_cb = _make_weak_stop(weakref.ref(self))
        atexit.register(self._atexit_cb)

    def _get_client(self) -> httpx.Client:
        if self._client.is_closed:
            self._client = httpx.Client(timeout=25.0)
        return self._client

    def _start_stderr_drainer(self, proc: subprocess.Popen) -> None:
        """Continuously drain llama-server stderr into a bounded 4 KB ring buffer to prevent pipe deadlocks."""
        stream = proc.stderr
        if stream is None:
            return
        with self._stderr_lock:
            self._stderr_ring.clear()
            self._stderr_bytes = 0

        def _drain() -> None:
            try:
                while True:
                    chunk = stream.readline()
                    if not chunk:
                        break
                    text = chunk.decode("utf-8", errors="ignore")
                    with self._stderr_lock:
                        self._stderr_ring.append(text)
                        self._stderr_bytes += len(text)
                        while self._stderr_bytes > 4096 and self._stderr_ring:
                            dropped = self._stderr_ring.popleft()
                            self._stderr_bytes -= len(dropped)
            except Exception:
                pass
            finally:
                try:
                    stream.close()
                except Exception:
                    pass

        threading.Thread(target=_drain, daemon=True, name="hymt2-stderr-drain").start()

    def _get_recent_stderr(self) -> str:
        with self._stderr_lock:
            return "".join(self._stderr_ring).strip()

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def ensure_started(self) -> None:
        with self._lock:
            if self.is_running:
                return
            llama_bin = find_llama_server()
            if llama_bin is None:
                raise RuntimeError(
                    "Không tìm thấy `llama-server` để chạy Hy-MT2-1.8B. Hãy cập nhật bản cài đặt mới nhất hoặc cài `llama.cpp`."
                )
            if not self.gguf_path.exists():
                raise FileNotFoundError(f"Chưa tải file model Hy-MT2: {self.gguf_path}")

            # Probe preferred port; if occupied by a stale/foreign process, switch cleanly to a free local port
            chosen_port = _pick_free_local_port(self.port)
            if chosen_port != self.port:
                self.port = chosen_port
                self.base_url = f"http://127.0.0.1:{chosen_port}"

            gpu_env = os.environ.get("VOICERECORE_LLAMA_GPU", "").strip().lower()
            is_mac_arm = platform.system() == "Darwin" and platform.machine() == "arm64"
            has_win_vulkan = os.name == "nt" and (llama_bin.parent / "ggml-vulkan.dll").is_file()

            # Build ordered launch profiles: prefer hardware GPU (Apple Metal / Windows Vulkan),
            # with guaranteed zero-crash automatic fallback to pure CPU (`-ngl 0`).
            attempts: list[tuple[str, list[str]]] = []
            if gpu_env not in ("0", "false", "no", "off"):
                if is_mac_arm:
                    attempts.append(("metal", ["-ngl", "99"]))
                elif has_win_vulkan or gpu_env in ("1", "true", "yes", "on"):
                    attempts.append(("vulkan" if has_win_vulkan else "gpu", ["-ngl", "99"]))
            attempts.append(("cpu", ["-ngl", "0"]))

            last_err = ""
            for backend_label, extra_flags in attempts:
                cmd = [
                    str(llama_bin),
                    "-m", str(self.gguf_path),
                    "--host", "127.0.0.1",
                    "--port", str(self.port),
                    "--jinja",
                    "--reasoning", "off",
                    "-c", "2048",
                    "-np", "1",
                    "--log-disable",
                    "-t", str(self.threads),
                    *extra_flags,
                ]
                kwargs: dict = {
                    "stdout": subprocess.DEVNULL,
                    "stderr": subprocess.PIPE,
                    "cwd": str(llama_bin.parent),
                }
                if os.name == "nt":
                    kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

                proc = subprocess.Popen(cmd, **kwargs)
                self._proc = proc
                self._start_stderr_drainer(proc)

                deadline = time.monotonic() + (20.0 if len(attempts) > 1 and backend_label != "cpu" else 45.0)
                started = False
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        time.sleep(0.05)
                        last_err = self._get_recent_stderr()
                        self._proc = None
                        break
                    try:
                        with self._client_lock:
                            r = self._get_client().get(f"{self.base_url}/health", timeout=1.0)
                        if r.status_code == 200:
                            started = True
                            break
                    except Exception:
                        pass
                    time.sleep(0.15)

                if started:
                    self.active_backend = backend_label
                    return
                self.stop()

            self.active_backend = ""
            raise RuntimeError(f"Hy-MT2 llama-server khởi động thất bại: {last_err[:400] or 'quá thời gian chờ'}")

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
                    proc.wait(timeout=1.0)
                except Exception:
                    pass

    def close(self) -> None:
        try:
            atexit.unregister(self._atexit_cb)
        except Exception:
            pass
        self.stop()
        with self._client_lock:
            try:
                self._client.close()
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

    @staticmethod
    def _content_words(s: str) -> set[str]:
        """
        Script-aware content-unit extractor for context-leak detection across ALL 38 Hy-MT2 languages:
        - Spaced scripts (Latin/Vietnamese/Cyrillic/Arabic/digits): >=3-char word tokens.
        - Hangul / CJK / Kana / Thai / Lao / Khmer: overlapping 2-char morpheme bigrams
          (immune to attached Korean/Japanese particles and unspaced Thai/Chinese runs).
        """
        low = (s or "").lower()
        units: set[str] = set(
            re.findall(r"[a-z\u00c0-\u024f\u1e00-\u1eff\u0400-\u04ff\u0600-\u06ff0-9]{3,}", low)
        )
        for run in re.findall(r"[\uac00-\ud7a3\u3040-\u30ff\u4e00-\u9fff\u0e00-\u0eff\u1780-\u17ff]+", low):
            if len(run) == 2:
                units.add(run)
            elif len(run) >= 3:
                for i in range(len(run) - 1):
                    units.add(run[i : i + 2])
        return units

    @classmethod
    def _looks_like_context_leak(cls, clean_src: str, translated: str, ctx_pairs: list[tuple[str, str]]) -> bool:
        """Detect when a 1.8B LLM accidentally re-translates the background context instead of the target segment."""
        if not ctx_pairs or not translated:
            return False
        out_words = cls._content_words(translated)
        if len(out_words) < 3:
            return False
        prev_src = " ".join(s for s, _ in ctx_pairs if s)
        prev_tgt = " ".join(t for _, t in ctx_pairs if t)
        prev_tgt_words = cls._content_words(prev_tgt)
        if len(prev_tgt_words) < 3:
            return False
        shared = len(out_words & prev_tgt_words)
        if shared < 3:
            return False
        overlap_out_self = shared / max(len(out_words), 1)
        overlap_out_min = shared / max(min(len(out_words), len(prev_tgt_words)), 1)
        src_words = cls._content_words(clean_src)
        prev_src_words = cls._content_words(prev_src)
        overlap_src = (
            len(src_words & prev_src_words) / max(min(len(src_words), len(prev_src_words)), 1)
            if src_words and prev_src_words
            else 0.0
        )
        # Catch both distinct-sentence leaks (48%+ of output OR 60%+ of min when prev_tgt >= 5 units)
        # and same-topic near-verbatim copies (68%+), while avoiding false triggers on 2 shared common words
        distinct_leak = overlap_src <= 0.35 and (
            overlap_out_self >= 0.48 or (len(prev_tgt_words) >= 5 and overlap_out_min >= 0.60)
        )
        verbatim_leak = overlap_out_min >= 0.68 and overlap_out_self >= 0.48 and overlap_src < 0.65
        return distinct_leak or verbatim_leak

    def _build_prompt(
        self,
        clean: str,
        tgt_name: str,
        matched_terms: list[tuple[str, str]],
        ctx_pairs: list[tuple[str, str]],
    ) -> str:
        blocks: list[str] = []
        if matched_terms:
            lines = ["Reference the following translations:"]
            for s, t in matched_terms:
                lines.append(f"{s} translates to {t}")
            blocks.append("\n".join(lines))

        tgt_spec = (
            "polite formal Korean (consistent 합니다/해요 style)"
            if tgt_name == "Korean"
            else tgt_name
        )
        bg = " ".join(s for s, _ in ctx_pairs if s).strip() if ctx_pairs else ""
        if bg:
            blocks.append(bg)
            blocks.append(
                f"Reference the above context (do NOT translate the context above). "
                f"Translate ONLY the following segment into {tgt_spec}, without additional explanation:\n\n{clean}"
            )
        else:
            blocks.append(
                f"Translate the following segment into {tgt_spec}, without additional explanation.\n\n{clean}"
            )
        return "\n\n".join(blocks)

    def _run_completion(self, prompt: str, clean: str) -> str:
        with self._client_lock:
            resp = self._get_client().post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": "hy-mt2-1.8b",
                    "temperature": 0.0,
                    "top_p": 0.6,
                    "max_tokens": min(384, max(96, len(clean.split()) * 6, len(clean) * 2)),
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
        resp.raise_for_status()
        data = resp.json()
        return self._clean_output(data["choices"][0]["message"]["content"])

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

        # Keep single most recent turn for pronoun/topic continuity on medium segments;
        # skip prior-turn context on very short (<=5 words) or long self-contained paragraphs (>=38 words)
        # to prevent 1.8B context-overwriting and keep latency crisp.
        src_wc = len(clean.split())
        ctx_pairs = [] if (src_wc <= 5 or src_wc >= 38) else (context or [])[-1:]
        norm_clean = _normalize_vi_large_units_for_cjk(clean, src_lang, tgt_lang)
        term_sig = "|".join(f"{s}->{t}" for s, t in matched_terms)
        ctx_sig = "|".join(s for s, _ in ctx_pairs)
        cache_key = (f"{term_sig}||{ctx_sig}||{norm_clean}", src_lang, tgt_lang, 2)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return TranslationResult(clean, cached, src_lang, tgt_lang, 0.0, "hy-mt2-1.8b", True)

        self.ensure_started()

        prompt = self._build_prompt(norm_clean, tgt_name, matched_terms, ctx_pairs)
        t0 = time.perf_counter()
        translated = self._run_completion(prompt, norm_clean)
        if ctx_pairs and self._looks_like_context_leak(norm_clean, translated, ctx_pairs):
            # Guardrail: if 1.8B leaked context into output, cleanly retry without context
            fallback_prompt = self._build_prompt(norm_clean, tgt_name, matched_terms, [])
            translated = self._run_completion(fallback_prompt, norm_clean)
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
        self._timeout_s = timeout_s
        self._client_lock = threading.Lock()
        self._client = httpx.Client(timeout=timeout_s)
        self._cache = LRUCache(maxsize=2048)

    def _get_client(self) -> httpx.Client:
        if self._client.is_closed:
            self._client = httpx.Client(timeout=self._timeout_s)
        return self._client

    def close(self) -> None:
        with self._client_lock:
            try:
                self._client.close()
            except Exception:
                pass

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
        with self._client_lock:
            resp = self._get_client().post(
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
        self._timeout_s = timeout_s
        self._client_lock = threading.Lock()
        self._client = httpx.Client(timeout=timeout_s)
        self._cache = LRUCache(maxsize=2048)

    def _get_client(self) -> httpx.Client:
        if self._client.is_closed:
            self._client = httpx.Client(timeout=self._timeout_s)
        return self._client

    def close(self) -> None:
        with self._client_lock:
            try:
                self._client.close()
            except Exception:
                pass

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

        with self._client_lock:
            resp = self._get_client().post(
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
        self._timeout_s = timeout_s
        self._client_lock = threading.Lock()
        self._client = httpx.Client(timeout=timeout_s)
        self._cache = LRUCache(maxsize=2048)

    def _get_client(self) -> httpx.Client:
        if self._client.is_closed:
            self._client = httpx.Client(timeout=self._timeout_s)
        return self._client

    def close(self) -> None:
        with self._client_lock:
            try:
                self._client.close()
            except Exception:
                pass

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
        with self._client_lock:
            resp = self._get_client().post(
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

_ACTION_PATTERNS = tuple(
    re.compile(rf"(?<!\w){re.escape(k)}(?!\w)", re.IGNORECASE | re.UNICODE)
    for k in _ACTION_KEYWORDS
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
        empty_vi = "Chưa có nội dung hội thoại để tóm tắt."
        empty_en = "No transcript entries available to summarize."
        return {
            "total_sentences": 0,
            "total_words": 0,
            "overview_src": empty_en if source_lang == "eng_Latn" else empty_vi,
            "overview_tgt": empty_vi if source_lang == "eng_Latn" else empty_en,
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
        # Boost decision / planning keywords (strict Unicode word boundaries — no "hạnh phúc" / "planet" false hits)
        kw_hits = sum(1 for pat in _ACTION_PATTERNS if pat.search(low))
        score += kw_hits * 4.5
        if idx == 0 or idx == len(entries) - 1:
            score += 2.5
        scored.append((score, idx, e))
        if kw_hits >= 1 or re.search(r"(?<!\w)\d+\s*(?:giờ|triệu|tỷ|tỉ|ngày|tuần|tháng|%|million|am|pm)(?!\w)", low, flags=re.UNICODE):
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
    vi_overview = (
        f"Phiên họp ghi nhận {len(entries)} câu ({total_words} từ) từ {first_ts} đến {last_ts}, "
        f"trích xuất {len(key_points)} ý chính và {min(len(actions), 6)} mục hành động / số liệu quan trọng."
    )
    en_overview = (
        f"Session recorded {len(entries)} sentences ({total_words} words) from {first_ts} to {last_ts}, "
        f"with {len(key_points)} key highlights and {min(len(actions), 6)} action items / key figures."
    )

    if source_lang == "eng_Latn":
        overview_src = en_overview
        overview_tgt = vi_overview if target_lang == "vie_Latn" else en_overview
        if translate_fn is not None and target_lang not in ("eng_Latn", "vie_Latn"):
            try:
                tr_ov = translate_fn(en_overview, "eng_Latn", target_lang)
                if tr_ov.translated_text:
                    overview_tgt = tr_ov.translated_text
            except Exception:
                pass
    else:
        overview_src = vi_overview
        overview_tgt = en_overview if target_lang == "eng_Latn" else vi_overview
        if translate_fn is not None and target_lang not in ("eng_Latn", "vie_Latn"):
            try:
                tr_ov = translate_fn(vi_overview, "vie_Latn", target_lang)
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


def _clamp_score(val: object, default: int = 0) -> int:
    try:
        n = int(round(float(val)))
        return max(0, min(100, n))
    except Exception:
        return default


def _verdict_from_score(score: int) -> str:
    if score >= 90:
        return "Xuất sắc — Chuẩn xác & tự nhiên"
    if score >= 80:
        return "Tốt — Đạt chuẩn thực tế"
    if score >= 65:
        return "Khá — Hiểu ý chính, nên chỉnh nhẹ"
    return "Cần kiểm tra — Có đoạn lệch nghĩa hoặc nghe nhầm"


def _parse_json_object(raw_text: str) -> dict:
    txt = (raw_text or "").strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.IGNORECASE)
        txt = re.sub(r"\s*```$", "", txt)
        txt = txt.strip()
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Fallback: locate outermost {...}
    start = txt.find("{")
    end = txt.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(txt[start : end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError("Phản hồi JSON từ Gemini không hợp lệ.")


def evaluate_translations_with_gemini(
    api_key: str,
    items: list[dict],
    model: str = "gemini-2.5-flash",
    timeout_s: float = 35.0,
) -> dict:
    """
    Semantic Translation Quality Audit powered by Official Google Gemini API.
    - Passes API key strictly via `x-goog-api-key` header (never leaked in URL or logs).
    - Evaluates semantic accuracy, fluency/context, and ASR input clarity per segment,
      distinguishing ASR mishearing vs MT mistranslation, and suggests improved translations
      plus 1-click Glossary fixes for recurring ASR brand/proper-noun errors.
    """
    key = (api_key or "").strip()
    if not key:
        raise ValueError("Chưa cấu hình Google Gemini API Key.")
    clean_items = [
        {
            "id": str(it.get("id") or f"item-{idx + 1}"),
            "source_text": str(it.get("source_text") or "").strip(),
            "translated_text": str(it.get("translated_text") or "").strip(),
            "source_lang": str(it.get("source_lang") or "vie_Latn"),
            "target_lang": str(it.get("target_lang") or "eng_Latn"),
            "timestamp": str(it.get("timestamp") or ""),
        }
        for idx, it in enumerate(items)
        if str(it.get("source_text") or "").strip()
    ]
    if not clean_items:
        raise ValueError("Chưa có câu dịch nào trong lịch sử để đánh giá.")

    # Cap to latest 40 segments so a single Gemini JSON response stays fast (<4s) and never truncates
    if len(clean_items) > 40:
        clean_items = clean_items[-40:]

    m = (model or "gemini-2.5-flash").strip()
    if m in GeminiCloudTranslator._DEPRECATED_MODELS:
        m = "gemini-2.5-flash"

    def _lang_name(code: str) -> str:
        meta = get_language(code)
        return meta.name_en if meta else code

    payload_rows = [
        {
            "id": it["id"],
            "src_lang": _lang_name(it["source_lang"]),
            "tgt_lang": _lang_name(it["target_lang"]),
            "source": it["source_text"],
            "translation": it["translated_text"],
        }
        for it in clean_items
    ]
    chars_used = sum(len(it["source_text"]) + len(it["translated_text"]) for it in clean_items)

    prompt = (
        "Bạn là chuyên gia thẩm định chất lượng phiên dịch hội nghị song ngữ (Translation QA Auditor).\n"
        "Nhiệm vụ: Đánh giá độ chính xác ngữ nghĩa giữa lời nói gốc (`source`, do ASR nhận dạng) và bản dịch (`translation`) cho từng mục sau.\n"
        "QUY TẮC BẮT BUỘC:\n"
        "1. Toàn bộ nhận xét (`verdict`, `summary_vi`, `comment_vi`) viết bằng tiếng Việt ngắn gọn, rõ ràng, thực tế.\n"
        "2. Phân biệt rõ 4 loại `issue_type` cho mỗi mục:\n"
        "   - \"ok\": Dịch chuẩn xác, sát nghĩa (score 85-100).\n"
        "   - \"nuance\": Hiểu đúng ý chính nhưng diễn đạt hơi cứng hoặc thiếu tự nhiên (score 70-84).\n"
        "   - \"mt_inaccurate\": Nguồn nghe rõ nhưng bản dịch sai nghĩa, bỏ sót ý/số liệu hoặc dịch nhầm chủ ngữ (score 0-69).\n"
        "   - \"asr_error\": Lời gốc `source` bị nhận dạng giọng nói nghe nhầm từ/tên riêng (VD: 'ocbian' thay vì 'Obsidian') dẫn tới bản dịch sai theo.\n"
        "3. `suggested_translation`: Đưa ra bản dịch chuẩn xác & tự nhiên nhất bằng ngôn ngữ đích (`tgt_lang`). Nếu đã chuẩn (`ok`), vẫn giữ bản dịch tốt nhất.\n"
        "4. `glossary_suggestions`: CHỈ liệt kê tối đa 6 cặp {\"wrong\": \"...\", \"correct\": \"...\"} khi thấy rõ tên riêng/thuật ngữ bị ASR nghe nhầm trong `source`; nếu không có thì trả mảng rỗng [].\n"
        "5. Trả về DUY NHẤT một JSON object hợp lệ đúng schema sau (không markdown, không văn bản thừa):\n"
        "{\n"
        '  "overall_score": 0-100,\n'
        '  "semantic_score": 0-100,\n'
        '  "fluency_score": 0-100,\n'
        '  "asr_score": 0-100,\n'
        '  "verdict": "nhận định ngắn 4-8 từ",\n'
        '  "summary_vi": "2-3 câu tóm tắt chất lượng toàn phiên bằng tiếng Việt",\n'
        '  "items": [\n'
        '    {"id": "...", "score": 0-100, "issue_type": "ok|nuance|mt_inaccurate|asr_error", "comment_vi": "nhận xét ngắn 1 câu", "suggested_translation": "..."}\n'
        "  ],\n"
        '  "glossary_suggestions": [{"wrong": "...", "correct": "..."}]\n'
        "}\n\n"
        f"Dữ liệu cần đánh giá ({len(payload_rows)} mục):\n"
        + json.dumps(payload_rows, ensure_ascii=False)
    )

    gen_cfg: dict = {
        "temperature": 0.1,
        "maxOutputTokens": 4096,
        "responseMimeType": "application/json",
    }
    if m.startswith(("gemini-2.5-", "gemini-3.1-")):
        gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
    t0 = time.perf_counter()
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(
            url,
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
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
    raw_out = "".join(p.get("text", "") for p in cand_parts if not p.get("thought")).strip()
    parsed = _parse_json_object(raw_out)
    ms = round((time.perf_counter() - t0) * 1000.0, 1)

    by_id: dict[str, dict] = {}
    for row in parsed.get("items") or []:
        if isinstance(row, dict) and row.get("id"):
            by_id[str(row["id"])] = row

    valid_issues = {"ok", "nuance", "mt_inaccurate", "asr_error"}
    norm_items: list[dict] = []
    for idx, src_it in enumerate(clean_items):
        raw_row = by_id.get(src_it["id"])
        if raw_row is None and isinstance(parsed.get("items"), list) and idx < len(parsed["items"]):
            cand = parsed["items"][idx]
            if isinstance(cand, dict):
                raw_row = cand
        raw_row = raw_row or {}
        sc = _clamp_score(raw_row.get("score"), 80)
        itype = str(raw_row.get("issue_type") or "").strip().lower()
        if itype not in valid_issues:
            itype = "ok" if sc >= 85 else ("nuance" if sc >= 70 else "mt_inaccurate")
        sugg = str(raw_row.get("suggested_translation") or "").strip() or src_it["translated_text"]
        comment = str(raw_row.get("comment_vi") or "").strip() or (
            "Bản dịch sát nghĩa." if sc >= 85 else "Nên xem lại độ sát nghĩa ở câu này."
        )
        norm_items.append(
            {
                "id": src_it["id"],
                "timestamp": src_it["timestamp"],
                "source_text": src_it["source_text"],
                "translated_text": src_it["translated_text"],
                "source_lang": src_it["source_lang"],
                "target_lang": src_it["target_lang"],
                "score": sc,
                "issue_type": itype,
                "comment_vi": comment,
                "suggested_translation": sugg,
            }
        )

    avg_item = int(round(sum(x["score"] for x in norm_items) / max(1, len(norm_items))))
    overall = _clamp_score(parsed.get("overall_score"), avg_item)
    semantic = _clamp_score(parsed.get("semantic_score"), overall)
    fluency = _clamp_score(parsed.get("fluency_score"), overall)
    asr_sc = _clamp_score(parsed.get("asr_score"), overall)
    verdict = str(parsed.get("verdict") or "").strip() or _verdict_from_score(overall)
    summary_vi = str(parsed.get("summary_vi") or "").strip() or (
        f"Đã kiểm định {len(norm_items)} đoạn hội thoại với điểm trung bình {overall}%."
    )

    seen_gloss: set[tuple[str, str]] = set()
    gloss_out: list[dict] = []
    for g in parsed.get("glossary_suggestions") or []:
        if not isinstance(g, dict):
            continue
        w = str(g.get("wrong") or "").strip()
        c = str(g.get("correct") or "").strip()
        if w and c and w.lower() != c.lower():
            gk = (w.lower(), c.lower())
            if gk not in seen_gloss:
                seen_gloss.add(gk)
                gloss_out.append({"wrong": w, "correct": c})
        if len(gloss_out) >= 6:
            break

    return {
        "provider": f"gemini:{m}",
        "mode": "semantic_audit",
        "overall_score": overall,
        "semantic_score": semantic,
        "fluency_score": fluency,
        "asr_score": asr_sc,
        "verdict": verdict,
        "summary_vi": summary_vi,
        "items": norm_items,
        "glossary_suggestions": gloss_out,
        "latency_ms": ms,
        "chars_used": chars_used,
    }


def evaluate_translations_with_google_cloud(
    api_key: str,
    items: list[dict],
    timeout_s: float = 15.0,
) -> dict:
    """
    Fallback Translation Reference Audit using Google Cloud Translation API v2
    when the user configured a Google Cloud Translate key without a Gemini API key.
    """
    clean_items = [
        {
            "id": str(it.get("id") or f"item-{idx + 1}"),
            "source_text": str(it.get("source_text") or "").strip(),
            "translated_text": str(it.get("translated_text") or "").strip(),
            "source_lang": str(it.get("source_lang") or "vie_Latn"),
            "target_lang": str(it.get("target_lang") or "eng_Latn"),
            "timestamp": str(it.get("timestamp") or ""),
        }
        for idx, it in enumerate(items)
        if str(it.get("source_text") or "").strip()
    ]
    if not clean_items:
        raise ValueError("Chưa có câu dịch nào trong lịch sử để đánh giá.")
    if len(clean_items) > 25:
        clean_items = clean_items[-25:]

    t0 = time.perf_counter()
    gct = GoogleCloudTranslator(api_key=api_key, timeout_s=timeout_s)
    norm_items: list[dict] = []
    chars_used = 0
    try:
        for it in clean_items:
            chars_used += len(it["source_text"])
            ref = gct.translate(it["source_text"], it["source_lang"], it["target_lang"]).translated_text
            a = re.sub(r"\s+", " ", it["translated_text"].lower()).strip()
            b = re.sub(r"\s+", " ", ref.lower()).strip()
            ratio = difflib.SequenceMatcher(None, a, b).ratio() if (a and b) else 0.0
            sc = _clamp_score(ratio * 100.0, 50)
            itype = "ok" if sc >= 80 else ("nuance" if sc >= 62 else "mt_inaccurate")
            comment = (
                "Khớp sát với bản dịch tham chiếu Google Translate."
                if sc >= 80
                else "Có khác biệt về cách diễn đạt/từ vựng so với Google Translate — nên đối chiếu bản gợi ý."
            )
            norm_items.append(
                {
                    "id": it["id"],
                    "timestamp": it["timestamp"],
                    "source_text": it["source_text"],
                    "translated_text": it["translated_text"],
                    "source_lang": it["source_lang"],
                    "target_lang": it["target_lang"],
                    "score": sc,
                    "issue_type": itype,
                    "comment_vi": comment,
                    "suggested_translation": ref or it["translated_text"],
                }
            )
    finally:
        gct.close()

    overall = int(round(sum(x["score"] for x in norm_items) / max(1, len(norm_items))))
    ms = round((time.perf_counter() - t0) * 1000.0, 1)
    return {
        "provider": "google-cloud",
        "mode": "google_reference",
        "overall_score": overall,
        "semantic_score": overall,
        "fluency_score": overall,
        "asr_score": 85,
        "verdict": _verdict_from_score(overall),
        "summary_vi": (
            f"Đã đối chiếu {len(norm_items)} đoạn với Google Cloud Translate (độ tương đồng trung bình {overall}%). "
            "Mẹo: cấu hình thêm Google Gemini API để nhận phân tích ngữ nghĩa sâu & phát hiện lỗi nghe nhầm ASR."
        ),
        "items": norm_items,
        "glossary_suggestions": [],
        "latency_ms": ms,
        "chars_used": chars_used,
    }

