"""
Model Manager — handles downloading, caching, and status reporting for local models.

Models managed:
1. ASR:
   - `zipformer-30m-streaming`: hynt/Zipformer-30M-RNNT-Streaming-6000h (~51MB, Vietnamese O(1) streaming)
   - `zipformer-en-20m`: sherpa-onnx-streaming-zipformer-en-20M-2023-02-17 (~128MB archive, English O(1) streaming)
   - `moonshine-en`: sherpa-onnx-moonshine-base-en-quantized-2026-02-27 (~106MB, English Pass-2 refiner)
   - `whisper-turbo`: sherpa-onnx-whisper-turbo (~538MB INT8) + Silero VAD v5 (~2.2MB) — 99-language offline ASR (TH/LO/JA/ZH/KO/KM/FR/DE...)
   - `viet-asr`: embedded 78MB INT8 Zipformer2 + Silero VAD + Punctuation/ITN (zero download)
2. MT (Translation):
   - `envit5`: luigi000/envit5-translation-ct2-int8 (~266MB, VietAI EnViT5 VI <-> EN)
   - `opus-vi-en`: dekthedev/opus-mt-vi-en-ct2-int8 (~77MB, ultra-fast VI -> EN)
   - `opus-en-vi`: dekthedev/opus-mt-en-vi-ct2-int8 (~77MB, ultra-fast EN -> VI)
   - `nllb-600m`: mijuanlo/nllb-200-distilled-600M-ct2-int8 (~615MB INT8, 202 x 202 languages offline)
   - `nllb-1.3b`: JustFrederik/nllb-200-distilled-1.3B-ct2-int8 (~1.34GB INT8, 202 languages)
   - `hy-mt2`: tencent/Hy-MT2-1.8B-GGUF (~1.05GB Q4_K_M, 38 languages)
"""

from __future__ import annotations

import os
import shutil
import ssl
import tarfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import quote

import certifi
import httpx
import sentencepiece as spm


def _ssl_verify_context():
    """Return reliable SSL verification for frozen Windows/macOS PyInstaller bundles."""
    try:
        ca = certifi.where()
        if ca and Path(ca).is_file():
            return ca
    except Exception:
        pass
    # Fallback to native OS cert store (Windows CryptoAPI / macOS Keychain via ssl)
    return ssl.create_default_context()


def _safe_replace(src: Path, dst: Path) -> None:
    """
    Windows Defender frequently scans newly written/extracted .onnx/.bin files for ~200-1500ms,
    causing immediate `os.replace()` to raise `PermissionError: [WinError 5]`.
    Retry cleanly with short backoff and fall back to `shutil.move`.
    """
    last_exc: Exception | None = None
    for attempt in range(6):
        try:
            if dst.exists() and dst.is_dir() and src.is_dir():
                shutil.rmtree(dst)
            os.replace(str(src), str(dst))
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(0.35 * (attempt + 1))
        except OSError as exc:
            last_exc = exc
            try:
                if dst.exists():
                    if dst.is_dir():
                        shutil.rmtree(dst)
                    else:
                        dst.unlink()
                shutil.move(str(src), str(dst))
                return
            except Exception as move_exc:
                last_exc = move_exc
                time.sleep(0.35 * (attempt + 1))
    if last_exc is not None:
        raise last_exc


def _hf_resolve_url(repo_id: str, filename: str) -> str:
    encoded = "/".join(quote(part, safe="") for part in filename.split("/"))
    return f"https://huggingface.co/{repo_id}/resolve/main/{encoded}"


def _hf_list_files(client: httpx.Client, repo_id: str) -> dict[str, int]:
    """Query Hugging Face public model metadata API (zero hf_hub / tqdm / xet dependency!)."""
    resp = client.get(f"https://huggingface.co/api/models/{repo_id}", params={"blobs": "true"})
    resp.raise_for_status()
    data = resp.json()
    out: dict[str, int] = {}
    for sib in data.get("siblings") or []:
        name = str(sib.get("rfilename") or "")
        if not name:
            continue
        size = int(sib.get("size") or (sib.get("lfs") or {}).get("size") or 0)
        out[name] = size
    return out


def _windows_short_path(path: Path) -> Optional[Path]:
    """Return Windows 8.3 ASCII short path if supported on the NTFS volume."""
    if os.name != "nt":
        return None
    try:
        import ctypes

        path.mkdir(parents=True, exist_ok=True)
        buf = ctypes.create_unicode_buffer(1024)
        n = ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, len(buf))  # type: ignore[attr-defined]
        if n > 0 and n < len(buf) and buf.value.isascii():
            return Path(buf.value)
    except Exception:
        pass
    return None


def _resolve_default_models_dir() -> Path:
    """
    Resolve default local model cache directory.
    On Windows with non-ASCII usernames (e.g. `C:\\Users\\Nguyễn Văn A`), C++ model
    loaders (CTranslate2 / sherpa-onnx) using `std::ifstream` fail on UTF-8 paths under
    the default Active Code Page. Prefer Windows 8.3 ASCII short path, falling back to
    `%PUBLIC%\\PolyVoicePro\\models` (always ASCII and standard-user writable).
    """
    env_dir = os.environ.get("VOICERECORE_MODEL_DIR")
    if env_dir:
        return Path(env_dir)

    default_dir = Path.home() / ".cache" / "voicerecore" / "models"
    if os.name == "nt" and not str(default_dir).isascii():
        short = _windows_short_path(default_dir)
        if short is not None:
            return short
        pub_root = Path(os.environ.get("PUBLIC", r"C:\Users\Public")) / "PolyVoicePro" / "models"
        if str(pub_root).isascii():
            return pub_root
    return default_dir


DEFAULT_MODELS_DIR = _resolve_default_models_dir()

NLLB_600M_REPO = "mijuanlo/nllb-200-distilled-600M-ct2-int8"
NLLB_600M_FALLBACK_REPO = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
NLLB_REQUIRED_FILES = [
    "config.json",
    "model.bin",
    "sentencepiece.bpe.model",
]
NLLB_OPTIONAL_VOCAB_FILES = [
    "shared_vocabulary.json",
    "shared_vocabulary.txt",
]

ZIPFORMER_STREAMING_REPO = "hynt/Zipformer-30M-RNNT-Streaming-6000h"
ZIPFORMER_STREAMING_FILES = [
    "encoder-epoch-31-avg-11-chunk-16-left-128.fp16.onnx",
    "decoder-epoch-31-avg-11-chunk-16-left-128.fp16.onnx",
    "joiner-epoch-31-avg-11-chunk-16-left-128.fp16.onnx",
    "bpe.model",
]

ZIPFORMER_EN_20M_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2"
)
ZIPFORMER_EN_20M_DIRNAME = "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"

MOONSHINE_EN_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-moonshine-base-en-quantized-2026-02-27.tar.bz2"
)
MOONSHINE_EN_DIRNAME = "sherpa-onnx-moonshine-base-en-quantized-2026-02-27"

WHISPER_TURBO_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-whisper-turbo.tar.bz2"
)
WHISPER_TURBO_DIRNAME = "sherpa-onnx-whisper-turbo"
SILERO_VAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "silero_vad_v5.onnx"
)

OPUS_VI_EN_REPO = "dekthedev/opus-mt-vi-en-ct2-int8"
OPUS_EN_VI_REPO = "dekthedev/opus-mt-en-vi-ct2-int8"
ENVIT5_REPO = "luigi000/envit5-translation-ct2-int8"
HYMT2_REPO = "tencent/Hy-MT2-1.8B-GGUF"
HYMT2_FILENAME = "Hy-MT2-1.8B-Q4_K_M.gguf"
NLLB_13B_REPO = "JustFrederik/nllb-200-distilled-1.3B-ct2-int8"

_DOWNLOADING_MARKER = ".downloading"
_READY_SENTINEL = ".ready"


def _file_ok(path: Path, min_bytes: int = 1) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def _dir_not_downloading(d: Path) -> bool:
    return d.exists() and not (d / _DOWNLOADING_MARKER).exists()


def _mark_downloading(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / _READY_SENTINEL).unlink(missing_ok=True)
    (d / _DOWNLOADING_MARKER).write_text(str(time.time()), encoding="utf-8")


def _mark_ready(d: Path) -> None:
    (d / _READY_SENTINEL).write_text("ok\n", encoding="utf-8")
    (d / _DOWNLOADING_MARKER).unlink(missing_ok=True)


def _clear_downloading(d: Path) -> None:
    (d / _DOWNLOADING_MARKER).unlink(missing_ok=True)


@dataclass
class DownloadState:
    model_id: str
    status: str = "idle"  # idle | missing | downloading | ready | error
    progress: float = 0.0
    downloaded_mb: float = 0.0
    total_mb: float = 0.0
    message: str = ""


class ModelManager:
    def __init__(self, models_dir: Path | None = None) -> None:
        self.models_dir = Path(models_dir or DEFAULT_MODELS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._states: dict[str, DownloadState] = {}

    @property
    def nllb_600m_dir(self) -> Path:
        return self.models_dir / "nllb-200-distilled-600M-ct2-int8"

    @property
    def zipformer_streaming_dir(self) -> Path:
        return self.models_dir / "zipformer-30m-streaming-vi"

    @property
    def zipformer_en_20m_dir(self) -> Path:
        return self.models_dir / ZIPFORMER_EN_20M_DIRNAME

    @property
    def moonshine_en_dir(self) -> Path:
        return self.models_dir / MOONSHINE_EN_DIRNAME

    @property
    def whisper_turbo_dir(self) -> Path:
        return self.models_dir / WHISPER_TURBO_DIRNAME

    @property
    def silero_vad_path(self) -> Path:
        return self.whisper_turbo_dir / "silero_vad_v5.onnx"

    @property
    def opus_vi_en_dir(self) -> Path:
        return self.models_dir / "opus-mt-vi-en-ct2-int8"

    @property
    def opus_en_vi_dir(self) -> Path:
        return self.models_dir / "opus-mt-en-vi-ct2-int8"

    @property
    def envit5_dir(self) -> Path:
        return self.models_dir / "envit5-translation-ct2-int8"

    @property
    def hy_mt2_dir(self) -> Path:
        return self.models_dir / "hy-mt2-1.8b-gguf"

    @property
    def hy_mt2_gguf_path(self) -> Path:
        return self.hy_mt2_dir / HYMT2_FILENAME

    @property
    def nllb_13b_dir(self) -> Path:
        return self.models_dir / "nllb-200-distilled-1.3B-ct2-int8"

    def is_hymt2_ready(self) -> bool:
        return _dir_not_downloading(self.hy_mt2_dir) and _file_ok(self.hy_mt2_gguf_path, 100_000_000)

    def is_nllb_13b_ready(self) -> bool:
        d = self.nllb_13b_dir
        if not _dir_not_downloading(d):
            return False
        has_core = (
            _file_ok(d / "model.bin", 100_000_000)
            and _file_ok(d / "config.json", 20)
            and _file_ok(d / "sentencepiece.bpe.model", 100_000)
        )
        has_vocab = any(_file_ok(d / f, 1000) for f in NLLB_OPTIONAL_VOCAB_FILES)
        return has_core and has_vocab

    def is_nllb_ready(self) -> bool:
        d = self.nllb_600m_dir
        if not _dir_not_downloading(d):
            return False
        has_core = (
            _file_ok(d / "model.bin", 100_000_000)
            and _file_ok(d / "config.json", 20)
            and _file_ok(d / "sentencepiece.bpe.model", 100_000)
        )
        has_vocab = any(_file_ok(d / f, 1000) for f in NLLB_OPTIONAL_VOCAB_FILES)
        return has_core and has_vocab

    def is_zipformer_streaming_ready(self) -> bool:
        d = self.zipformer_streaming_dir
        if not _dir_not_downloading(d):
            return False
        return (
            _file_ok(d / ZIPFORMER_STREAMING_FILES[0], 5_000_000)
            and _file_ok(d / ZIPFORMER_STREAMING_FILES[1], 100_000)
            and _file_ok(d / ZIPFORMER_STREAMING_FILES[2], 100_000)
            and _file_ok(d / "bpe.model", 10_000)
            and _file_ok(d / "tokens.txt", 100)
        )

    def is_zipformer_en_20m_ready(self) -> bool:
        d = self.zipformer_en_20m_dir
        if not _dir_not_downloading(d):
            return False
        return (
            _file_ok(d / "encoder-epoch-99-avg-1.int8.onnx", 5_000_000)
            and _file_ok(d / "decoder-epoch-99-avg-1.int8.onnx", 100_000)
            and _file_ok(d / "joiner-epoch-99-avg-1.int8.onnx", 100_000)
            and _file_ok(d / "tokens.txt", 100)
        )

    def is_moonshine_en_ready(self) -> bool:
        d = self.moonshine_en_dir
        if not _dir_not_downloading(d):
            return False
        return (
            _file_ok(d / "encoder_model.ort", 5_000_000)
            and _file_ok(d / "decoder_model_merged.ort", 5_000_000)
            and _file_ok(d / "tokens.txt", 100)
        )

    def _whisper_turbo_has_weights(self) -> bool:
        d = self.whisper_turbo_dir
        if not _dir_not_downloading(d):
            return False
        enc = d / "turbo-encoder.int8.onnx"
        dec = d / "turbo-decoder.int8.onnx"
        tok = d / "turbo-tokens.txt"
        if not enc.exists():
            enc = next(d.glob("*-encoder.int8.onnx"), enc)
        if not dec.exists():
            dec = next(d.glob("*-decoder.int8.onnx"), dec)
        if not tok.exists():
            tok = next(d.glob("*-tokens.txt"), tok)
        return _file_ok(enc, 100_000_000) and _file_ok(dec, 50_000_000) and _file_ok(tok, 100_000)

    def is_whisper_turbo_ready(self) -> bool:
        return self._whisper_turbo_has_weights() and _file_ok(self.silero_vad_path, 500_000)

    def is_opus_vi_en_ready(self) -> bool:
        d = self.opus_vi_en_dir
        if not _dir_not_downloading(d):
            return False
        return (
            _file_ok(d / "model.bin", 10_000_000)
            and _file_ok(d / "source.spm", 10_000)
            and _file_ok(d / "target.spm", 10_000)
            and _file_ok(d / "config.json", 20)
        )

    def is_opus_en_vi_ready(self) -> bool:
        d = self.opus_en_vi_dir
        if not _dir_not_downloading(d):
            return False
        return (
            _file_ok(d / "model.bin", 10_000_000)
            and _file_ok(d / "source.spm", 10_000)
            and _file_ok(d / "target.spm", 10_000)
            and _file_ok(d / "config.json", 20)
        )

    def is_envit5_ready(self) -> bool:
        d = self.envit5_dir
        if not _dir_not_downloading(d):
            return False
        return (
            _file_ok(d / "model.bin", 50_000_000)
            and _file_ok(d / "spiece.model", 10_000)
            and _file_ok(d / "config.json", 20)
        )

    def get_status(self) -> dict:
        with self._lock:
            nllb_state = self._states.get(
                "nllb-600m",
                DownloadState(
                    model_id="nllb-600m",
                    status="ready" if self.is_nllb_ready() else "missing",
                    progress=100.0 if self.is_nllb_ready() else 0.0,
                    total_mb=615.0,
                    downloaded_mb=615.0 if self.is_nllb_ready() else 0.0,
                    message="Sẵn sàng (202×202 ngôn ngữ offline)" if self.is_nllb_ready() else "Chưa tải (~615 MB)",
                ),
            )
            zf_state = self._states.get(
                "zipformer-30m-streaming",
                DownloadState(
                    model_id="zipformer-30m-streaming",
                    status="ready" if self.is_zipformer_streaming_ready() else "missing",
                    progress=100.0 if self.is_zipformer_streaming_ready() else 0.0,
                    total_mb=51.0,
                    downloaded_mb=51.0 if self.is_zipformer_streaming_ready() else 0.0,
                    message="Sẵn sàng (Việt Streaming 6000h ~30ms)"
                    if self.is_zipformer_streaming_ready()
                    else "Chưa tải (~51 MB)",
                ),
            )
            zf_en_state = self._states.get(
                "zipformer-en-20m",
                DownloadState(
                    model_id="zipformer-en-20m",
                    status="ready" if self.is_zipformer_en_20m_ready() else "missing",
                    progress=100.0 if self.is_zipformer_en_20m_ready() else 0.0,
                    total_mb=128.0,
                    downloaded_mb=128.0 if self.is_zipformer_en_20m_ready() else 0.0,
                    message="Sẵn sàng (Anh Streaming Pass-1 ~6ms)"
                    if self.is_zipformer_en_20m_ready()
                    else "Chưa tải (~128 MB)",
                ),
            )
            moonshine_en_state = self._states.get(
                "moonshine-en",
                DownloadState(
                    model_id="moonshine-en",
                    status="ready" if self.is_moonshine_en_ready() else "missing",
                    progress=100.0 if self.is_moonshine_en_ready() else 0.0,
                    total_mb=106.0,
                    downloaded_mb=106.0 if self.is_moonshine_en_ready() else 0.0,
                    message="Sẵn sàng (Moonshine-v2 EN • 1.8% WER • Dấu câu • 44× RT)"
                    if self.is_moonshine_en_ready()
                    else "Chưa tải (~106 MB)",
                ),
            )
            whisper_turbo_state = self._states.get(
                "whisper-turbo",
                DownloadState(
                    model_id="whisper-turbo",
                    status="ready" if self.is_whisper_turbo_ready() else "missing",
                    progress=100.0 if self.is_whisper_turbo_ready() else 0.0,
                    total_mb=540.0,
                    downloaded_mb=540.0 if self.is_whisper_turbo_ready() else 0.0,
                    message="Sẵn sàng (Whisper Large-v3-Turbo + Silero VAD • 99 NN Offline)"
                    if self.is_whisper_turbo_ready()
                    else "Chưa tải (~540 MB)",
                ),
            )
            opus_ve_state = self._states.get(
                "opus-vi-en",
                DownloadState(
                    model_id="opus-vi-en",
                    status="ready" if self.is_opus_vi_en_ready() else "missing",
                    progress=100.0 if self.is_opus_vi_en_ready() else 0.0,
                    total_mb=77.0,
                    downloaded_mb=77.0 if self.is_opus_vi_en_ready() else 0.0,
                    message="Sẵn sàng (Việt → Anh siêu tốc)"
                    if self.is_opus_vi_en_ready()
                    else "Chưa tải (~77 MB)",
                ),
            )
            opus_ev_state = self._states.get(
                "opus-en-vi",
                DownloadState(
                    model_id="opus-en-vi",
                    status="ready" if self.is_opus_en_vi_ready() else "missing",
                    progress=100.0 if self.is_opus_en_vi_ready() else 0.0,
                    total_mb=77.0,
                    downloaded_mb=77.0 if self.is_opus_en_vi_ready() else 0.0,
                    message="Sẵn sàng (Anh → Việt siêu tốc)"
                    if self.is_opus_en_vi_ready()
                    else "Chưa tải (~77 MB)",
                ),
            )
            envit5_state = self._states.get(
                "envit5",
                DownloadState(
                    model_id="envit5",
                    status="ready" if self.is_envit5_ready() else "missing",
                    progress=100.0 if self.is_envit5_ready() else 0.0,
                    total_mb=266.0,
                    downloaded_mb=266.0 if self.is_envit5_ready() else 0.0,
                    message="Sẵn sàng (VietAI EnViT5 Anh⇄Việt • OpenRAIL)"
                    if self.is_envit5_ready()
                    else "Chưa tải (~266 MB)",
                ),
            )
            hymt2_state = self._states.get(
                "hy-mt2",
                DownloadState(
                    model_id="hy-mt2",
                    status="ready" if self.is_hymt2_ready() else "missing",
                    progress=100.0 if self.is_hymt2_ready() else 0.0,
                    total_mb=1080.0,
                    downloaded_mb=1080.0 if self.is_hymt2_ready() else 0.0,
                    message="Sẵn sàng (Tencent Hy-MT2-1.8B • 38 NN • Ngữ cảnh & Thuật ngữ)"
                    if self.is_hymt2_ready()
                    else "Chưa tải (~1.05 GB)",
                ),
            )
            nllb13_state = self._states.get(
                "nllb-1.3b",
                DownloadState(
                    model_id="nllb-1.3b",
                    status="ready" if self.is_nllb_13b_ready() else "missing",
                    progress=100.0 if self.is_nllb_13b_ready() else 0.0,
                    total_mb=1341.0,
                    downloaded_mb=1341.0 if self.is_nllb_13b_ready() else 0.0,
                    message="Sẵn sàng (NLLB-200 1.3B • Chuẩn cao Lào / Thái / 202 NN)"
                    if self.is_nllb_13b_ready()
                    else "Chưa tải (~1.34 GB)",
                ),
            )
            return {
                "models_dir": str(self.models_dir),
                "asr_zipformer_streaming": asdict(zf_state),
                "asr_zipformer_en_20m": asdict(zf_en_state),
                "asr_moonshine_en": asdict(moonshine_en_state),
                "asr_whisper_turbo": asdict(whisper_turbo_state),
                "asr_vietasr": {
                    "model_id": "viet-asr",
                    "status": "ready",
                    "progress": 100.0,
                    "total_mb": 78.0,
                    "downloaded_mb": 78.0,
                    "message": "Tích hợp sẵn (Zipformer2 + Dấu câu/ITN)",
                },
                "mt_hymt2": asdict(hymt2_state),
                "mt_envit5": asdict(envit5_state),
                "mt_nllb_13b": asdict(nllb13_state),
                "mt_opus_vi_en": asdict(opus_ve_state),
                "mt_opus_en_vi": asdict(opus_ev_state),
                "mt_nllb_600m": asdict(nllb_state),
            }

    def _update_state(
        self,
        model_id: str,
        on_progress: Optional[Callable[[dict], None]] = None,
        **kwargs,
    ) -> None:
        with self._lock:
            cur = self._states.get(model_id, DownloadState(model_id=model_id))
            for k, v in kwargs.items():
                setattr(cur, k, v)
            self._states[model_id] = cur
            payload = asdict(cur)
        if on_progress:
            try:
                on_progress(payload)
            except Exception:
                pass

    def _make_http_client(self) -> httpx.Client:
        timeout = httpx.Timeout(connect=20.0, read=60.0, write=30.0, pool=20.0)
        return httpx.Client(
            follow_redirects=True,
            verify=_ssl_verify_context(),
            timeout=timeout,
            headers={"User-Agent": "PolyVoicePro/0.8.1"},
        )

    def _download_file_resumable(
        self,
        client: httpx.Client,
        url: str,
        dest_path: Path,
        *,
        on_bytes: Optional[Callable[[int, int], None]] = None,
        expected_bytes: int = 0,
        max_retries: int = 3,
    ) -> int:
        """
        Stream-download a single file to `dest_path.part` with automatic HTTP Range resume,
        3x retry backoff on flaky Wi-Fi, real byte progress callbacks, and Windows-safe replace.
        """
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = dest_path.with_name(dest_path.name + ".part")
        last_exc: Exception | None = None

        for attempt in range(max_retries):
            try:
                existing = part_path.stat().st_size if part_path.exists() else 0
                headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}
                with client.stream("GET", url, headers=headers) as resp:
                    if resp.status_code == 416 and existing > 0:
                        cr = resp.headers.get("content-range", "")
                        remote_total = 0
                        if "/" in cr:
                            tail = cr.rsplit("/", 1)[1].strip()
                            if tail.isdigit():
                                remote_total = int(tail)
                        if (remote_total > 0 and existing == remote_total) or (
                            remote_total == 0 and expected_bytes > 0 and existing == expected_bytes
                        ):
                            total_bytes = remote_total or existing
                            if on_bytes:
                                on_bytes(existing, total_bytes)
                            _safe_replace(part_path, dest_path)
                            return total_bytes
                        # Stale or mismatched partial file — wipe and retry cleanly from byte 0
                        part_path.unlink(missing_ok=True)
                        raise RuntimeError("File tải dở không khớp dung lượng, đang tự tải lại từ đầu...")

                    if resp.status_code == 206 and existing > 0:
                        mode = "ab"
                        done = existing
                        cr = resp.headers.get("content-range", "")
                        total_bytes = 0
                        if "/" in cr:
                            try:
                                total_bytes = int(cr.rsplit("/", 1)[1])
                            except ValueError:
                                total_bytes = 0
                        if total_bytes <= 0:
                            cl = int(resp.headers.get("content-length") or 0)
                            total_bytes = (existing + cl) if cl > 0 else expected_bytes
                    else:
                        resp.raise_for_status()
                        mode = "wb"
                        done = 0
                        cl = int(resp.headers.get("content-length") or 0)
                        total_bytes = cl if cl > 0 else expected_bytes

                    with open(part_path, mode) as out_f:
                        for chunk in resp.iter_bytes(chunk_size=131072):
                            if not chunk:
                                continue
                            out_f.write(chunk)
                            done += len(chunk)
                            if on_bytes:
                                on_bytes(done, total_bytes)

                if total_bytes > 0 and done < total_bytes:
                    raise RuntimeError(f"Tải thiếu dữ liệu ({done}/{total_bytes} bytes)")

                _safe_replace(part_path, dest_path)
                return done
            except Exception as exc:
                last_exc = exc
                if attempt + 1 < max_retries:
                    time.sleep(1.2 * (attempt + 1))
                else:
                    part_path.unlink(missing_ok=True)

        assert last_exc is not None
        raise last_exc

    def _download_hf_files(
        self,
        model_id: str,
        files: list[tuple[str, str, int]],
        target_dir: Path,
        expected_total_mb: float,
        on_progress: Optional[Callable[[dict], None]] = None,
    ) -> float:
        """Download a list of `(repo_id, filename, size_hint)` into `target_dir` with smooth byte progress."""
        target_dir.mkdir(parents=True, exist_ok=True)
        known_total = sum(max(0, sz) for _, _, sz in files)
        total_est = known_total if known_total > 0 else int(expected_total_mb * 1024 * 1024)
        total_mb = round(max(total_est, 1) / (1024 * 1024), 1)
        completed_before = 0
        last_emit = [0.0]

        with self._make_http_client() as client:
            for idx, (repo_id, filename, size_hint) in enumerate(files):
                dest = target_dir / Path(filename).name
                url = _hf_resolve_url(repo_id, filename)
                short_name = Path(filename).name

                def _on_bytes(done_in_file: int, file_total: int) -> None:
                    now = time.monotonic()
                    if (now - last_emit[0]) < 0.25 and (file_total <= 0 or done_in_file < file_total):
                        return
                    last_emit[0] = now
                    eff_total = max(total_est, completed_before + max(file_total, done_in_file), 1)
                    eff_done = completed_before + done_in_file
                    pct = round(min(96.0, 2.0 + (eff_done / eff_total) * 94.0), 1)
                    dl_mb = round(eff_done / (1024 * 1024), 1)
                    tot_mb_now = round(eff_total / (1024 * 1024), 1)
                    self._update_state(
                        model_id,
                        on_progress,
                        status="downloading",
                        progress=pct,
                        downloaded_mb=dl_mb,
                        total_mb=tot_mb_now,
                        message=f"Đang tải {short_name} ({dl_mb:.1f} / {tot_mb_now:.1f} MB)...",
                    )

                self._update_state(
                    model_id,
                    on_progress,
                    status="downloading",
                    progress=round(min(95.0, 2.0 + (completed_before / max(total_est, 1)) * 93.0), 1),
                    downloaded_mb=round(completed_before / (1024 * 1024), 1),
                    total_mb=total_mb,
                    message=f"Đang tải {short_name} ({idx + 1}/{len(files)})...",
                )
                got = self._download_file_resumable(
                    client,
                    url,
                    dest,
                    on_bytes=_on_bytes,
                    expected_bytes=size_hint,
                )
                completed_before += got

        return round(completed_before / (1024 * 1024), 1)

    def _download_and_extract_tar_bz2(
        self,
        model_id: str,
        url: str,
        dirname: str,
        dest_dir: Path,
        expected_mb: float,
        start_msg: str,
        ready_msg: str,
        on_progress: Optional[Callable[[dict], None]] = None,
        extra_files: Optional[list[tuple[str, str, int]]] = None,
    ) -> Path:
        archive_path = self.models_dir / f".{dirname}.tar.bz2"
        staging_dir = self.models_dir / f".{dirname}.staging"
        self._update_state(
            model_id,
            on_progress,
            status="downloading",
            progress=2.0,
            downloaded_mb=0.0,
            total_mb=expected_mb,
            message=start_msg,
        )
        try:
            archive_path.unlink(missing_ok=True)
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)

            last_emit = [0.0]
            total_mb_ref = [expected_mb]

            def _on_bytes(done_bytes: int, total_bytes: int) -> None:
                now = time.monotonic()
                if (now - last_emit[0]) < 0.25 and (total_bytes <= 0 or done_bytes < total_bytes):
                    return
                last_emit[0] = now
                t_mb = round(total_bytes / (1024 * 1024), 1) if total_bytes > 0 else expected_mb
                total_mb_ref[0] = t_mb
                dl_mb = round(done_bytes / (1024 * 1024), 1)
                pct = (
                    round(min(88.0, 2.0 + (done_bytes / total_bytes) * 86.0), 1)
                    if total_bytes > 0
                    else min(85.0, round(2.0 + (done_bytes / max(expected_mb * 1024 * 1024, 1)) * 83.0, 1))
                )
                self._update_state(
                    model_id,
                    on_progress,
                    status="downloading",
                    progress=pct,
                    downloaded_mb=dl_mb,
                    total_mb=t_mb,
                    message=f"Đang tải ({dl_mb:.1f} / {t_mb:.1f} MB)...",
                )

            with self._make_http_client() as client:
                got_bytes = self._download_file_resumable(
                    client,
                    url,
                    archive_path,
                    on_bytes=_on_bytes,
                    expected_bytes=int(expected_mb * 1024 * 1024),
                )
            total_mb = round(got_bytes / (1024 * 1024), 1) if got_bytes > 0 else total_mb_ref[0]

            self._update_state(
                model_id,
                on_progress,
                status="downloading",
                progress=92.0,
                downloaded_mb=total_mb,
                total_mb=total_mb,
                message="Đang giải nén mô hình...",
            )
            staging_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive_path, "r:bz2") as tf:
                try:
                    tf.extractall(staging_dir, filter="data")
                except TypeError:
                    tf.extractall(staging_dir)

            extracted = staging_dir / dirname
            if not extracted.exists():
                subdirs = [p for p in staging_dir.iterdir() if p.is_dir()]
                if len(subdirs) == 1:
                    extracted = subdirs[0]
                else:
                    raise RuntimeError(f"Không tìm thấy thư mục mô hình sau giải nén: {dirname}")

            if extra_files:
                with self._make_http_client() as client:
                    for extra_name, extra_url, extra_bytes in extra_files:
                        self._update_state(
                            model_id,
                            on_progress,
                            status="downloading",
                            progress=96.0,
                            downloaded_mb=total_mb,
                            total_mb=round(total_mb + extra_bytes / (1024 * 1024), 1),
                            message=f"Đang tải thành phần phụ {extra_name}...",
                        )
                        got_extra = self._download_file_resumable(
                            client,
                            extra_url,
                            extracted / extra_name,
                            expected_bytes=extra_bytes,
                        )
                        total_mb = round(total_mb + got_extra / (1024 * 1024), 1)

            _mark_ready(extracted)
            _safe_replace(extracted, dest_dir)

            self._update_state(
                model_id,
                on_progress,
                status="ready",
                progress=100.0,
                downloaded_mb=total_mb,
                total_mb=total_mb,
                message=ready_msg,
            )
            return dest_dir
        except Exception as exc:
            self._update_state(model_id, on_progress, status="error", message=f"Lỗi tải: {exc}")
            raise
        finally:
            archive_path.unlink(missing_ok=True)
            archive_path.with_name(archive_path.name + ".part").unlink(missing_ok=True)
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)

    def ensure_zipformer_streaming(self, on_progress: Optional[Callable[[dict], None]] = None) -> Path:
        if self.is_zipformer_streaming_ready():
            self._update_state("zipformer-30m-streaming", on_progress, status="ready", progress=100.0)
            return self.zipformer_streaming_dir

        target_dir = self.zipformer_streaming_dir
        _mark_downloading(target_dir)
        self._update_state(
            "zipformer-30m-streaming",
            on_progress,
            status="downloading",
            progress=2.0,
            downloaded_mb=0.0,
            total_mb=51.0,
            message="Đang tải Zipformer-30M Việt Streaming (~51 MB)...",
        )
        try:
            file_specs = [(ZIPFORMER_STREAMING_REPO, fn, 0) for fn in ZIPFORMER_STREAMING_FILES]
            got_mb = self._download_hf_files(
                "zipformer-30m-streaming", file_specs, target_dir, 51.0, on_progress=on_progress
            )

            sp = spm.SentencePieceProcessor(model_proto=(target_dir / "bpe.model").read_bytes())
            tmp_tokens = target_dir / "tokens.txt.tmp"
            with open(tmp_tokens, "w", encoding="utf-8") as tf:
                for i in range(sp.get_piece_size()):
                    tf.write(f"{sp.id_to_piece(i)} {i}\n")
            _safe_replace(tmp_tokens, target_dir / "tokens.txt")

            _mark_ready(target_dir)
            self._update_state(
                "zipformer-30m-streaming",
                on_progress,
                status="ready",
                progress=100.0,
                downloaded_mb=got_mb or 51.0,
                total_mb=got_mb or 51.0,
                message="Sẵn sàng (Việt Streaming 6000h ~30ms)",
            )
            return target_dir
        except Exception as exc:
            _clear_downloading(target_dir)
            self._update_state("zipformer-30m-streaming", on_progress, status="error", message=f"Lỗi tải: {exc}")
            raise

    def ensure_zipformer_en_20m(self, on_progress: Optional[Callable[[dict], None]] = None) -> Path:
        if self.is_zipformer_en_20m_ready():
            self._update_state("zipformer-en-20m", on_progress, status="ready", progress=100.0)
            return self.zipformer_en_20m_dir

        return self._download_and_extract_tar_bz2(
            model_id="zipformer-en-20m",
            url=ZIPFORMER_EN_20M_URL,
            dirname=ZIPFORMER_EN_20M_DIRNAME,
            dest_dir=self.zipformer_en_20m_dir,
            expected_mb=128.0,
            start_msg="Đang tải Zipformer-20M Tiếng Anh Streaming (~128 MB)...",
            ready_msg="Sẵn sàng (Anh Streaming Pass-1 ~6ms)",
            on_progress=on_progress,
        )

    def ensure_moonshine_en(self, on_progress: Optional[Callable[[dict], None]] = None) -> Path:
        if self.is_moonshine_en_ready():
            self._update_state("moonshine-en", on_progress, status="ready", progress=100.0)
            return self.moonshine_en_dir

        return self._download_and_extract_tar_bz2(
            model_id="moonshine-en",
            url=MOONSHINE_EN_URL,
            dirname=MOONSHINE_EN_DIRNAME,
            dest_dir=self.moonshine_en_dir,
            expected_mb=106.0,
            start_msg="Đang tải Moonshine-v2 Base Tiếng Anh (~106 MB • 1.8% WER)...",
            ready_msg="Sẵn sàng (Moonshine-v2 EN • 1.8% WER • Dấu câu • 44× RT)",
            on_progress=on_progress,
        )

    def ensure_whisper_turbo(self, on_progress: Optional[Callable[[dict], None]] = None) -> Path:
        if self.is_whisper_turbo_ready():
            self._update_state("whisper-turbo", on_progress, status="ready", progress=100.0)
            return self.whisper_turbo_dir

        # If Whisper weights already exist and only Silero VAD v5 is missing, fetch VAD alone
        if self._whisper_turbo_has_weights() and not _file_ok(self.silero_vad_path, 500_000):
            self._update_state(
                "whisper-turbo",
                on_progress,
                status="downloading",
                progress=90.0,
                downloaded_mb=538.0,
                total_mb=540.0,
                message="Đang tải Silero VAD v5 (~2.2 MB)...",
            )
            try:
                with self._make_http_client() as client:
                    self._download_file_resumable(
                        client,
                        SILERO_VAD_URL,
                        self.silero_vad_path,
                        expected_bytes=2_313_101,
                    )
                _mark_ready(self.whisper_turbo_dir)
                self._update_state(
                    "whisper-turbo",
                    on_progress,
                    status="ready",
                    progress=100.0,
                    downloaded_mb=540.0,
                    total_mb=540.0,
                    message="Sẵn sàng (Whisper Large-v3-Turbo + Silero VAD • 99 NN Offline)",
                )
                return self.whisper_turbo_dir
            except Exception as exc:
                self._update_state("whisper-turbo", on_progress, status="error", message=f"Lỗi tải Silero VAD: {exc}")
                raise

        return self._download_and_extract_tar_bz2(
            model_id="whisper-turbo",
            url=WHISPER_TURBO_URL,
            dirname=WHISPER_TURBO_DIRNAME,
            dest_dir=self.whisper_turbo_dir,
            expected_mb=537.7,
            start_msg="Đang tải Whisper Large-v3-Turbo INT8 + Silero VAD (~540 MB • 99 NN)...",
            ready_msg="Sẵn sàng (Whisper Large-v3-Turbo + Silero VAD • 99 NN Offline)",
            on_progress=on_progress,
            extra_files=[("silero_vad_v5.onnx", SILERO_VAD_URL, 2_313_101)],
        )

    def ensure_opus(self, direction: str, on_progress: Optional[Callable[[dict], None]] = None) -> Path:
        if direction == "en-vi":
            model_id, repo_id, target_dir, ready = (
                "opus-en-vi",
                OPUS_EN_VI_REPO,
                self.opus_en_vi_dir,
                self.is_opus_en_vi_ready(),
            )
            label = "Anh → Việt"
        else:
            model_id, repo_id, target_dir, ready = (
                "opus-vi-en",
                OPUS_VI_EN_REPO,
                self.opus_vi_en_dir,
                self.is_opus_vi_en_ready(),
            )
            label = "Việt → Anh"

        if ready:
            self._update_state(model_id, on_progress, status="ready", progress=100.0)
            return target_dir

        _mark_downloading(target_dir)
        self._update_state(
            model_id,
            on_progress,
            status="downloading",
            progress=2.0,
            downloaded_mb=0.0,
            total_mb=77.0,
            message=f"Đang tải Opus-MT {label} INT8 (~77 MB)...",
        )
        try:
            with self._make_http_client() as client:
                avail = _hf_list_files(client, repo_id)
            wanted = [
                (repo_id, fn, avail.get(fn, 0))
                for fn in avail
                if fn.endswith((".bin", ".spm", ".json", ".txt")) and not fn.startswith(".")
            ]
            got_mb = self._download_hf_files(model_id, wanted, target_dir, 77.0, on_progress=on_progress)
            _mark_ready(target_dir)
            self._update_state(
                model_id,
                on_progress,
                status="ready",
                progress=100.0,
                downloaded_mb=got_mb or 77.0,
                total_mb=got_mb or 77.0,
                message=f"Sẵn sàng ({label} siêu tốc)",
            )
            return target_dir
        except Exception as exc:
            _clear_downloading(target_dir)
            self._update_state(model_id, on_progress, status="error", message=f"Lỗi tải: {exc}")
            raise

    def ensure_envit5(self, on_progress: Optional[Callable[[dict], None]] = None) -> Path:
        if self.is_envit5_ready():
            self._update_state("envit5", on_progress, status="ready", progress=100.0)
            return self.envit5_dir

        target_dir = self.envit5_dir
        _mark_downloading(target_dir)
        self._update_state(
            "envit5",
            on_progress,
            status="downloading",
            progress=2.0,
            downloaded_mb=0.0,
            total_mb=266.0,
            message="Đang tải VietAI EnViT5 INT8 (~266 MB)...",
        )
        try:
            try:
                with self._make_http_client() as client:
                    avail = _hf_list_files(client, ENVIT5_REPO)
                wanted = [
                    (ENVIT5_REPO, fn, avail.get(fn, 0))
                    for fn in ("config.json", "shared_vocabulary.json", "shared_vocabulary.txt", "spiece.model", "model.bin")
                    if fn in avail
                ]
            except Exception:
                wanted = [
                    (ENVIT5_REPO, "config.json", 1024),
                    (ENVIT5_REPO, "shared_vocabulary.json", 950_000),
                    (ENVIT5_REPO, "model.bin", 277_000_000),
                ]
            if not any(fn == "spiece.model" for _, fn, _ in wanted):
                wanted.append(("VietAI/envit5-translation", "spiece.model", 1_102_207))

            got_mb = self._download_hf_files("envit5", wanted, target_dir, 266.0, on_progress=on_progress)
            _mark_ready(target_dir)
            self._update_state(
                "envit5",
                on_progress,
                status="ready",
                progress=100.0,
                downloaded_mb=got_mb or 266.0,
                total_mb=got_mb or 266.0,
                message="Sẵn sàng (VietAI EnViT5 Anh⇄Việt • OpenRAIL)",
            )
            return target_dir
        except Exception as exc:
            _clear_downloading(target_dir)
            self._update_state("envit5", on_progress, status="error", message=f"Lỗi tải VietAI EnViT5: {exc}")
            raise

    def ensure_nllb_600m(self, on_progress: Optional[Callable[[dict], None]] = None) -> Path:
        if self.is_nllb_ready():
            self._update_state("nllb-600m", on_progress, status="ready", progress=100.0)
            return self.nllb_600m_dir

        target_dir = self.nllb_600m_dir
        _mark_downloading(target_dir)
        self._update_state(
            "nllb-600m",
            on_progress,
            status="downloading",
            progress=2.0,
            total_mb=615.0,
            downloaded_mb=0.0,
            message="Đang kết nối Hugging Face để tải NLLB-200 INT8 (~615 MB)...",
        )

        last_err: Exception | None = None
        for repo_id in (NLLB_600M_REPO, NLLB_600M_FALLBACK_REPO):
            try:
                with self._make_http_client() as client:
                    available = _hf_list_files(client, repo_id)
                wanted_names = [f for f in NLLB_REQUIRED_FILES if f in available]
                for vf in NLLB_OPTIONAL_VOCAB_FILES:
                    if vf in available:
                        wanted_names.append(vf)
                        break
                if len(wanted_names) < 4:
                    raise RuntimeError(f"Repo {repo_id} thiếu file cấu hình NLLB.")
                file_specs = [(repo_id, fn, available.get(fn, 0)) for fn in wanted_names]
                got_mb = self._download_hf_files("nllb-600m", file_specs, target_dir, 615.0, on_progress=on_progress)

                _mark_ready(target_dir)
                if self.is_nllb_ready():
                    self._update_state(
                        "nllb-600m",
                        on_progress,
                        status="ready",
                        progress=100.0,
                        downloaded_mb=got_mb or 615.0,
                        total_mb=got_mb or 615.0,
                        message="Sẵn sàng (202×202 ngôn ngữ offline)",
                    )
                    return target_dir
                _mark_downloading(target_dir)
            except Exception as exc:
                last_err = exc
                continue

        _clear_downloading(target_dir)
        self._update_state("nllb-600m", on_progress, status="error", message=f"Lỗi tải NLLB-200: {last_err}")
        raise RuntimeError(f"Could not download NLLB-200 CTranslate2 model: {last_err}")

    def ensure_hymt2(self, on_progress: Optional[Callable[[dict], None]] = None) -> Path:
        if self.is_hymt2_ready():
            self._update_state("hy-mt2", on_progress, status="ready", progress=100.0)
            return self.hy_mt2_gguf_path

        _mark_downloading(self.hy_mt2_dir)
        self._update_state(
            "hy-mt2",
            on_progress,
            status="downloading",
            progress=2.0,
            downloaded_mb=0.0,
            total_mb=1080.0,
            message="Đang tải Tencent Hy-MT2-1.8B Q4_K_M (~1.05 GB)...",
        )
        try:
            got_mb = self._download_hf_files(
                "hy-mt2",
                [(HYMT2_REPO, HYMT2_FILENAME, 1_133_080_448)],
                self.hy_mt2_dir,
                1080.0,
                on_progress=on_progress,
            )
            _mark_ready(self.hy_mt2_dir)
            self._update_state(
                "hy-mt2",
                on_progress,
                status="ready",
                progress=100.0,
                downloaded_mb=got_mb or 1080.0,
                total_mb=got_mb or 1080.0,
                message="Sẵn sàng (Tencent Hy-MT2-1.8B • 38 NN • Ngữ cảnh & Thuật ngữ)",
            )
            return self.hy_mt2_gguf_path
        except Exception as exc:
            _clear_downloading(self.hy_mt2_dir)
            self._update_state("hy-mt2", on_progress, status="error", message=f"Lỗi tải Hy-MT2-1.8B: {exc}")
            raise

    def ensure_nllb_13b(self, on_progress: Optional[Callable[[dict], None]] = None) -> Path:
        if self.is_nllb_13b_ready():
            self._update_state("nllb-1.3b", on_progress, status="ready", progress=100.0)
            return self.nllb_13b_dir

        _mark_downloading(self.nllb_13b_dir)
        self._update_state(
            "nllb-1.3b",
            on_progress,
            status="downloading",
            progress=2.0,
            downloaded_mb=0.0,
            total_mb=1341.0,
            message="Đang tải NLLB-200 1.3B INT8 (~1.34 GB — Chuẩn cao Lào / Thái / 202 NN)...",
        )
        try:
            with self._make_http_client() as client:
                available = _hf_list_files(client, NLLB_13B_REPO)
            wanted_names = [f for f in NLLB_REQUIRED_FILES if f in available]
            for vf in NLLB_OPTIONAL_VOCAB_FILES:
                if vf in available:
                    wanted_names.append(vf)
                    break
            file_specs = [(NLLB_13B_REPO, fn, available.get(fn, 0)) for fn in wanted_names]
            got_mb = self._download_hf_files(
                "nllb-1.3b", file_specs, self.nllb_13b_dir, 1341.0, on_progress=on_progress
            )
            _mark_ready(self.nllb_13b_dir)
            self._update_state(
                "nllb-1.3b",
                on_progress,
                status="ready",
                progress=100.0,
                downloaded_mb=got_mb or 1341.0,
                total_mb=got_mb or 1341.0,
                message="Sẵn sàng (NLLB-200 1.3B • Chuẩn cao Lào / Thái / 202 NN)",
            )
            return self.nllb_13b_dir
        except Exception as exc:
            _clear_downloading(self.nllb_13b_dir)
            self._update_state("nllb-1.3b", on_progress, status="error", message=f"Lỗi tải NLLB-200 1.3B: {exc}")
            raise

    def delete_model(self, model_id: str) -> dict:
        """Delete an optional/redundant local model directory to free disk space cleanly."""
        target_map = {
            "hy-mt2": self.hy_mt2_dir,
            "nllb-1.3b": self.nllb_13b_dir,
            "nllb-600m": self.nllb_600m_dir,
            "opus-vi-en": self.opus_vi_en_dir,
            "opus-en-vi": self.opus_en_vi_dir,
            "envit5": self.envit5_dir,
            "whisper-turbo": self.whisper_turbo_dir,
            "moonshine-en": self.moonshine_en_dir,
            "zipformer-en-20m": self.zipformer_en_20m_dir,
            "zipformer-30m-streaming": self.zipformer_streaming_dir,
        }
        target_dir = target_map.get(model_id)
        if target_dir is None:
            raise ValueError(f"Unsupported model_id: {model_id}")

        freed_bytes = 0
        if target_dir.exists():
            for p in target_dir.rglob("*"):
                if p.is_file():
                    try:
                        freed_bytes += p.stat().st_size
                    except OSError:
                        pass
            try:
                shutil.rmtree(target_dir)
            except PermissionError as exc:
                raise PermissionError(
                    f"Không thể xóa model '{model_id}' vì file đang được sử dụng. Hãy dừng phiên dịch hoặc khởi động lại ứng dụng rồi thử lại: {exc}"
                ) from exc

        with self._lock:
            self._states.pop(model_id, None)

        return {
            "ok": True,
            "model_id": model_id,
            "freed_mb": round(freed_bytes / (1024 * 1024), 1),
        }
