"""
Model Manager — handles downloading, caching, and status reporting for local models.

Models managed (10 streamlined offline models, 0% PyTorch):
1. ASR:
   - `zipformer-30m-streaming`: hynt/Zipformer-30M-RNNT-Streaming-6000h (~49MB, Vietnamese O(1) streaming)
   - `viet-asr`: embedded 78MB INT8 Zipformer2 + Silero VAD + Punctuation/ITN (zero download)
   - `moonshine-en`: sherpa-onnx-moonshine-base-en-quantized-2026-02-27 (~106MB, English main + Pass-2)
   - `moonshine-ja`: sherpa-onnx-moonshine-base-ja-quantized-2026-02-27 (~99MB, Japanese Fast-Lane)
   - `moonshine-ko`: sherpa-onnx-moonshine-tiny-ko-quantized-2026-02-27 (~47MB, Korean Fast-Lane)
   - `zipformer-multi`: sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh (~247MB, Asian 8-lang Fast-Lane)
   - `parakeet-eu`: sherpa-onnx-nemo-parakeet-tdt-0.6b-ultra-int8 (~465MB, 25-lang European Fast-Lane)
   - `whisper-turbo`: sherpa-onnx-whisper-turbo (~538MB INT8) + Silero VAD v5 (~2.2MB) — optional 99-language extension
2. MT (Translation):
   - `envit5`: luigi000/envit5-translation-ct2-int8 (~266MB, VietAI EnViT5 VI <-> EN)
   - `nllb-600m`: mijuanlo/nllb-200-distilled-600M-ct2-int8 (~615MB INT8, 202 x 202 languages offline)
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

MOONSHINE_EN_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-moonshine-base-en-quantized-2026-02-27.tar.bz2"
)
MOONSHINE_EN_DIRNAME = "sherpa-onnx-moonshine-base-en-quantized-2026-02-27"

MOONSHINE_JA_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-moonshine-base-ja-quantized-2026-02-27.tar.bz2"
)
MOONSHINE_JA_DIRNAME = "sherpa-onnx-moonshine-base-ja-quantized-2026-02-27"

MOONSHINE_KO_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-moonshine-tiny-ko-quantized-2026-02-27.tar.bz2"
)
MOONSHINE_KO_DIRNAME = "sherpa-onnx-moonshine-tiny-ko-quantized-2026-02-27"

ZIPFORMER_MULTI_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10.tar.bz2"
)
ZIPFORMER_MULTI_DIRNAME = "sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10"

PARAKEET_EU_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-nemo-parakeet-tdt-0.6b-ultra-int8.tar.bz2"
)
PARAKEET_EU_DIRNAME = "sherpa-onnx-nemo-parakeet-tdt-0.6b-ultra-int8"

WHISPER_TURBO_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-whisper-turbo.tar.bz2"
)
WHISPER_TURBO_DIRNAME = "sherpa-onnx-whisper-turbo"
SILERO_VAD_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "silero_vad_v5.onnx"
)

ENVIT5_REPO = "luigi000/envit5-translation-ct2-int8"
HYMT2_REPO = "tencent/Hy-MT2-1.8B-GGUF"
HYMT2_FILENAME = "Hy-MT2-1.8B-Q4_K_M.gguf"

_DOWNLOADING_MARKER = ".downloading"
_READY_SENTINEL = ".ready"

_KNOWN_MODEL_SUBDIRS = (
    "zipformer-30m-streaming-vi",
    MOONSHINE_EN_DIRNAME,
    MOONSHINE_JA_DIRNAME,
    MOONSHINE_KO_DIRNAME,
    ZIPFORMER_MULTI_DIRNAME,
    PARAKEET_EU_DIRNAME,
    WHISPER_TURBO_DIRNAME,
    "envit5-translation-ct2-int8",
    "nllb-200-distilled-600M-ct2-int8",
    "hy-mt2-1.8b-gguf",
)


def _has_model_weights(model_subdir: Path) -> bool:
    """Return True if `model_subdir` contains at least one non-trivial model weight file."""
    if not model_subdir.is_dir():
        return False
    try:
        for p in model_subdir.iterdir():
            if (
                p.is_file()
                and p.suffix.lower() in (".bin", ".onnx", ".ort", ".gguf")
                and p.stat().st_size >= 5_000_000
            ):
                return True
    except OSError:
        return False
    return False


def _count_existing_models(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for sub in _KNOWN_MODEL_SUBDIRS if _has_model_weights(root / sub))


def _is_dir_writable(root: Path) -> bool:
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / f".pv_write_test_{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _migrate_missing_models(src_root: Path, dst_root: Path) -> int:
    """
    Safely recover/migrate downloaded model folders from `src_root` into `dst_root`
    when Windows username encoding or UAC elevation shifts `Path.home()` across updates.
    Never overwrites a model that already exists in `dst_root`.
    """
    if not src_root.is_dir():
        return 0
    try:
        if dst_root.exists() and src_root.resolve() == dst_root.resolve():
            return 0
    except OSError:
        pass
    moved = 0
    try:
        dst_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return 0
    for sub in _KNOWN_MODEL_SUBDIRS:
        src_sub = src_root / sub
        dst_sub = dst_root / sub
        if not _has_model_weights(src_sub) or _has_model_weights(dst_sub):
            continue
        try:
            if dst_sub.exists():
                shutil.rmtree(dst_sub, ignore_errors=True)
            shutil.move(str(src_sub), str(dst_sub))
            (dst_sub / _DOWNLOADING_MARKER).unlink(missing_ok=True)
            moved += 1
        except Exception:
            try:
                shutil.copytree(str(src_sub), str(dst_sub), dirs_exist_ok=True)
                (dst_sub / _DOWNLOADING_MARKER).unlink(missing_ok=True)
                moved += 1
            except Exception:
                pass
    return moved


def _resolve_default_models_dir() -> Path:
    """
    Resolve default local model cache directory.
    On Windows:
      - Handles non-ASCII usernames (e.g. `C:\\Users\\Nguyễn Văn A`) where C++ loaders fail on UTF-8 paths.
      - Handles UAC / MSI elevation shifts (Standard User vs Administrator) so auto-updates NEVER lose models.
      - Automatically recovers/migrates existing models between `~/.cache/voicerecore/models` and
        `%PUBLIC%\\PolyVoicePro\\models`.
      - For brand-new Windows installs with no prior models, prefers `%PUBLIC%\\PolyVoicePro\\models`
        (always 100% ASCII, standard-user writable, and shared across Standard & Elevated Admin sessions).
    """
    env_dir = os.environ.get("VOICERECORE_MODEL_DIR")
    if env_dir:
        return Path(env_dir)

    default_dir = Path.home() / ".cache" / "voicerecore" / "models"
    if os.name != "nt":
        return default_dir

    pub_root = Path(os.environ.get("PUBLIC", r"C:\Users\Public")) / "PolyVoicePro" / "models"
    ascii_user_dir: Optional[Path] = (
        default_dir if str(default_dir).isascii() else _windows_short_path(default_dir)
    )
    user_cnt = _count_existing_models(default_dir)
    pub_cnt = _count_existing_models(pub_root)

    # 1. Non-ASCII Windows username without 8.3 short paths -> must use ASCII %PUBLIC% dir,
    #    and auto-migrate any models previously downloaded into `default_dir`.
    if ascii_user_dir is None:
        if str(pub_root).isascii() and _is_dir_writable(pub_root):
            if user_cnt > 0:
                _migrate_missing_models(default_dir, pub_root)
            return pub_root
        return default_dir

    # 2. Existing models live in %PUBLIC% while user home cache is empty (e.g. after MSI/UAC update)
    if pub_cnt > 0 and user_cnt == 0 and str(pub_root).isascii() and _is_dir_writable(pub_root):
        return pub_root

    # 3. Existing models already live in the user's cache -> preserve them and merge any extras from %PUBLIC%
    if user_cnt > 0:
        if pub_cnt > 0:
            _migrate_missing_models(pub_root, ascii_user_dir)
        return ascii_user_dir

    # 4. Fresh Windows install (0 models in both): store in %PUBLIC%\PolyVoicePro\models so
    #    future MSI/NSIS updates (even under UAC Admin) and Unicode usernames never lose models!
    if str(pub_root).isascii() and _is_dir_writable(pub_root):
        return pub_root
    return ascii_user_dir


DEFAULT_MODELS_DIR = _resolve_default_models_dir()


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
        # Clear any stale `.downloading` markers left over from an abrupt exit or app update:
        # no download thread can exist yet at startup, and every `is_*_ready()` verifies real weight files.
        for sub in _KNOWN_MODEL_SUBDIRS:
            stale = self.models_dir / sub / _DOWNLOADING_MARKER
            if stale.exists():
                try:
                    stale.unlink(missing_ok=True)
                except OSError:
                    pass
        self._lock = threading.Lock()
        self._states: dict[str, DownloadState] = {}

    @property
    def nllb_600m_dir(self) -> Path:
        return self.models_dir / "nllb-200-distilled-600M-ct2-int8"

    @property
    def zipformer_streaming_dir(self) -> Path:
        return self.models_dir / "zipformer-30m-streaming-vi"

    @property
    def moonshine_en_dir(self) -> Path:
        return self.models_dir / MOONSHINE_EN_DIRNAME

    @property
    def moonshine_ja_dir(self) -> Path:
        return self.models_dir / MOONSHINE_JA_DIRNAME

    @property
    def moonshine_ko_dir(self) -> Path:
        return self.models_dir / MOONSHINE_KO_DIRNAME

    @property
    def zipformer_multi_dir(self) -> Path:
        return self.models_dir / ZIPFORMER_MULTI_DIRNAME

    @property
    def parakeet_eu_dir(self) -> Path:
        return self.models_dir / PARAKEET_EU_DIRNAME

    @property
    def whisper_turbo_dir(self) -> Path:
        return self.models_dir / WHISPER_TURBO_DIRNAME

    @property
    def silero_vad_path(self) -> Path:
        for candidate in (
            self.whisper_turbo_dir / "silero_vad_v5.onnx",
            self.parakeet_eu_dir / "silero_vad_v5.onnx",
            self.moonshine_en_dir / "silero_vad_v5.onnx",
            self.moonshine_ja_dir / "silero_vad_v5.onnx",
            self.moonshine_ko_dir / "silero_vad_v5.onnx",
            self.models_dir / "silero_vad_v5.onnx",
        ):
            if _file_ok(candidate, 500_000):
                return candidate
        return self.whisper_turbo_dir / "silero_vad_v5.onnx"

    def is_silero_vad_ready(self) -> bool:
        return _file_ok(self.silero_vad_path, 500_000)

    @property
    def envit5_dir(self) -> Path:
        return self.models_dir / "envit5-translation-ct2-int8"

    @property
    def hy_mt2_dir(self) -> Path:
        return self.models_dir / "hy-mt2-1.8b-gguf"

    @property
    def hy_mt2_gguf_path(self) -> Path:
        return self.hy_mt2_dir / HYMT2_FILENAME

    def is_hymt2_ready(self) -> bool:
        return _dir_not_downloading(self.hy_mt2_dir) and _file_ok(self.hy_mt2_gguf_path, 100_000_000)

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

    def is_moonshine_en_ready(self) -> bool:
        d = self.moonshine_en_dir
        if not _dir_not_downloading(d):
            return False
        return (
            _file_ok(d / "encoder_model.ort", 5_000_000)
            and _file_ok(d / "decoder_model_merged.ort", 5_000_000)
            and _file_ok(d / "tokens.txt", 100)
        )

    def is_moonshine_ja_ready(self) -> bool:
        d = self.moonshine_ja_dir
        if not _dir_not_downloading(d):
            return False
        return (
            _file_ok(d / "encoder_model.ort", 5_000_000)
            and _file_ok(d / "decoder_model_merged.ort", 5_000_000)
            and _file_ok(d / "tokens.txt", 100)
        )

    def is_moonshine_ko_ready(self) -> bool:
        d = self.moonshine_ko_dir
        if not _dir_not_downloading(d):
            return False
        return (
            _file_ok(d / "encoder_model.ort", 5_000_000)
            and _file_ok(d / "decoder_model_merged.ort", 5_000_000)
            and _file_ok(d / "tokens.txt", 100)
        )

    def is_zipformer_multi_ready(self) -> bool:
        d = self.zipformer_multi_dir
        if not _dir_not_downloading(d):
            return False
        enc = next(d.glob("encoder*.int8.onnx"), d / "encoder.int8.onnx")
        dec = next(d.glob("decoder*.onnx"), d / "decoder.onnx")
        joi = next(d.glob("joiner*.int8.onnx"), d / "joiner.int8.onnx")
        return (
            _file_ok(enc, 50_000_000)
            and _file_ok(dec, 5_000_000)
            and _file_ok(joi, 1_000_000)
            and _file_ok(d / "tokens.txt", 100)
        )

    def _parakeet_eu_has_weights(self) -> bool:
        d = self.parakeet_eu_dir
        if not _dir_not_downloading(d):
            return False
        enc = next(d.glob("encoder*.int8.onnx"), d / "encoder.int8.onnx")
        dec = next(d.glob("decoder*.int8.onnx"), d / "decoder.int8.onnx")
        joi = next(d.glob("joiner*.int8.onnx"), d / "joiner.int8.onnx")
        return (
            _file_ok(enc, 200_000_000)
            and _file_ok(dec, 5_000_000)
            and _file_ok(joi, 1_000_000)
            and _file_ok(d / "tokens.txt", 100)
        )

    def is_parakeet_eu_ready(self) -> bool:
        return self._parakeet_eu_has_weights() and _file_ok(self.silero_vad_path, 500_000)

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
            moonshine_ja_state = self._states.get(
                "moonshine-ja",
                DownloadState(
                    model_id="moonshine-ja",
                    status="ready" if self.is_moonshine_ja_ready() else "missing",
                    progress=100.0 if self.is_moonshine_ja_ready() else 0.0,
                    total_mb=99.0,
                    downloaded_mb=99.0 if self.is_moonshine_ja_ready() else 0.0,
                    message="Sẵn sàng (Moonshine-v2 Tiếng Nhật • ~120ms Streaming • 0% GPU)"
                    if self.is_moonshine_ja_ready()
                    else "Chưa tải (~99 MB)",
                ),
            )
            moonshine_ko_state = self._states.get(
                "moonshine-ko",
                DownloadState(
                    model_id="moonshine-ko",
                    status="ready" if self.is_moonshine_ko_ready() else "missing",
                    progress=100.0 if self.is_moonshine_ko_ready() else 0.0,
                    total_mb=47.0,
                    downloaded_mb=47.0 if self.is_moonshine_ko_ready() else 0.0,
                    message="Sẵn sàng (Moonshine-v2 Tiếng Hàn • ~150ms Streaming • 0% GPU)"
                    if self.is_moonshine_ko_ready()
                    else "Chưa tải (~47 MB)",
                ),
            )
            zf_multi_state = self._states.get(
                "zipformer-multi",
                DownloadState(
                    model_id="zipformer-multi",
                    status="ready" if self.is_zipformer_multi_ready() else "missing",
                    progress=100.0 if self.is_zipformer_multi_ready() else 0.0,
                    total_mb=247.0,
                    downloaded_mb=247.0 if self.is_zipformer_multi_ready() else 0.0,
                    message="Sẵn sàng (Thái/Trung/Indo Streaming ~15ms • 0% GPU)"
                    if self.is_zipformer_multi_ready()
                    else "Chưa tải (~247 MB)",
                ),
            )
            parakeet_eu_state = self._states.get(
                "parakeet-eu",
                DownloadState(
                    model_id="parakeet-eu",
                    status="ready" if self.is_parakeet_eu_ready() else "missing",
                    progress=100.0 if self.is_parakeet_eu_ready() else 0.0,
                    total_mb=465.0,
                    downloaded_mb=465.0 if self.is_parakeet_eu_ready() else 0.0,
                    message="Sẵn sàng (Parakeet-Ultra 25 NN Châu Âu • ~0.3s • 0% GPU)"
                    if self.is_parakeet_eu_ready()
                    else "Chưa tải (~465 MB)",
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
            return {
                "models_dir": str(self.models_dir),
                "asr_zipformer_streaming": asdict(zf_state),
                "asr_moonshine_en": asdict(moonshine_en_state),
                "asr_moonshine_ja": asdict(moonshine_ja_state),
                "asr_moonshine_ko": asdict(moonshine_ko_state),
                "asr_zipformer_multi": asdict(zf_multi_state),
                "asr_parakeet_eu": asdict(parakeet_eu_state),
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
            headers={"User-Agent": "PolyVoicePro/1.2.0"},
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

    def ensure_moonshine_en(self, on_progress: Optional[Callable[[dict], None]] = None) -> Path:
        if self.is_moonshine_en_ready():
            self._update_state("moonshine-en", on_progress, status="ready", progress=100.0)
            return self.moonshine_en_dir

        return self._download_and_extract_tar_bz2(
            model_id="moonshine-en",
            url=MOONSHINE_EN_URL,
            dirname=MOONSHINE_EN_DIRNAME,
            dest_dir=self.moonshine_en_dir,
            expected_mb=108.0,
            start_msg="Đang tải Moonshine-v2 Base Tiếng Anh + Silero VAD (~108 MB • 1.8% WER)...",
            ready_msg="Sẵn sàng (Moonshine-v2 EN • 1.8% WER • Dấu câu • 44× RT)",
            on_progress=on_progress,
            extra_files=[("silero_vad_v5.onnx", SILERO_VAD_URL, 2_313_101)],
        )

    def ensure_moonshine_ja(self, on_progress: Optional[Callable[[dict], None]] = None) -> Path:
        if self.is_moonshine_ja_ready():
            self._update_state("moonshine-ja", on_progress, status="ready", progress=100.0)
            return self.moonshine_ja_dir

        return self._download_and_extract_tar_bz2(
            model_id="moonshine-ja",
            url=MOONSHINE_JA_URL,
            dirname=MOONSHINE_JA_DIRNAME,
            dest_dir=self.moonshine_ja_dir,
            expected_mb=101.0,
            start_msg="Đang tải Moonshine-v2 Base Tiếng Nhật + Silero VAD (~99 MB • ~120ms)...",
            ready_msg="Sẵn sàng (Moonshine-v2 Tiếng Nhật • ~120ms Streaming • 0% GPU)",
            on_progress=on_progress,
            extra_files=[("silero_vad_v5.onnx", SILERO_VAD_URL, 2_313_101)],
        )

    def ensure_moonshine_ko(self, on_progress: Optional[Callable[[dict], None]] = None) -> Path:
        if self.is_moonshine_ko_ready():
            self._update_state("moonshine-ko", on_progress, status="ready", progress=100.0)
            return self.moonshine_ko_dir

        return self._download_and_extract_tar_bz2(
            model_id="moonshine-ko",
            url=MOONSHINE_KO_URL,
            dirname=MOONSHINE_KO_DIRNAME,
            dest_dir=self.moonshine_ko_dir,
            expected_mb=49.0,
            start_msg="Đang tải Moonshine-v2 Tiếng Hàn + Silero VAD (~47 MB • ~150ms)...",
            ready_msg="Sẵn sàng (Moonshine-v2 Tiếng Hàn • ~150ms Streaming • 0% GPU)",
            on_progress=on_progress,
            extra_files=[("silero_vad_v5.onnx", SILERO_VAD_URL, 2_313_101)],
        )

    def ensure_zipformer_multi(self, on_progress: Optional[Callable[[dict], None]] = None) -> Path:
        if self.is_zipformer_multi_ready():
            self._update_state("zipformer-multi", on_progress, status="ready", progress=100.0)
            return self.zipformer_multi_dir

        return self._download_and_extract_tar_bz2(
            model_id="zipformer-multi",
            url=ZIPFORMER_MULTI_URL,
            dirname=ZIPFORMER_MULTI_DIRNAME,
            dest_dir=self.zipformer_multi_dir,
            expected_mb=247.0,
            start_msg="Đang tải Zipformer Streaming Đa ngôn ngữ (Thái, Trung, Indo • ~247 MB • ~15ms)...",
            ready_msg="Sẵn sàng (Thái/Trung/Indo Streaming ~15ms • 0% GPU)",
            on_progress=on_progress,
        )

    def ensure_parakeet_eu(self, on_progress: Optional[Callable[[dict], None]] = None) -> Path:
        if self.is_parakeet_eu_ready():
            self._update_state("parakeet-eu", on_progress, status="ready", progress=100.0)
            return self.parakeet_eu_dir

        if self._parakeet_eu_has_weights() and not _file_ok(self.silero_vad_path, 500_000):
            self._update_state(
                "parakeet-eu",
                on_progress,
                status="downloading",
                progress=92.0,
                downloaded_mb=462.0,
                total_mb=465.0,
                message="Đang tải Silero VAD v5 (~2.2 MB)...",
            )
            try:
                with self._make_http_client() as client:
                    self._download_file_resumable(
                        client,
                        SILERO_VAD_URL,
                        self.parakeet_eu_dir / "silero_vad_v5.onnx",
                        expected_bytes=2_313_101,
                    )
                _mark_ready(self.parakeet_eu_dir)
                self._update_state(
                    "parakeet-eu",
                    on_progress,
                    status="ready",
                    progress=100.0,
                    downloaded_mb=465.0,
                    total_mb=465.0,
                    message="Sẵn sàng (Parakeet-Ultra 25 NN Châu Âu • ~0.3s • 0% GPU)",
                )
                return self.parakeet_eu_dir
            except Exception as exc:
                self._update_state("parakeet-eu", on_progress, status="error", message=f"Lỗi tải Silero VAD: {exc}")
                raise

        return self._download_and_extract_tar_bz2(
            model_id="parakeet-eu",
            url=PARAKEET_EU_URL,
            dirname=PARAKEET_EU_DIRNAME,
            dest_dir=self.parakeet_eu_dir,
            expected_mb=465.0,
            start_msg="Đang tải Parakeet-TDT-0.6B-Ultra 25 NN Châu Âu + Silero VAD (~465 MB • ~0.3s)...",
            ready_msg="Sẵn sàng (Parakeet-Ultra 25 NN Châu Âu • ~0.3s • 0% GPU)",
            on_progress=on_progress,
            extra_files=[("silero_vad_v5.onnx", SILERO_VAD_URL, 2_313_101)],
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

    def delete_model(self, model_id: str) -> dict:
        """Delete an optional/redundant local model directory to free disk space cleanly."""
        target_map = {
            "hy-mt2": self.hy_mt2_dir,
            "nllb-600m": self.nllb_600m_dir,
            "envit5": self.envit5_dir,
            "whisper-turbo": self.whisper_turbo_dir,
            "parakeet-eu": self.parakeet_eu_dir,
            "moonshine-en": self.moonshine_en_dir,
            "moonshine-ja": self.moonshine_ja_dir,
            "moonshine-ko": self.moonshine_ko_dir,
            "zipformer-multi": self.zipformer_multi_dir,
            "zipformer-30m-streaming": self.zipformer_streaming_dir,
        }
        target_dir = target_map.get(model_id)
        if target_dir is None:
            raise ValueError(f"Unsupported model_id: {model_id}")

        # Preserve shared Silero VAD v5 before removing an ASR folder (e.g. deleting whisper-turbo)
        vad_in_target = target_dir / "silero_vad_v5.onnx"
        shared_vad = self.models_dir / "silero_vad_v5.onnx"
        if _file_ok(vad_in_target, 500_000) and not _file_ok(shared_vad, 500_000):
            try:
                shutil.copy2(vad_in_target, shared_vad)
            except OSError:
                pass

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
