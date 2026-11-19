"""
Model Manager — handles downloading, caching, and status reporting for local models.

Models managed:
1. ASR:
   - `zipformer-30m-streaming`: hynt/Zipformer-30M-RNNT-Streaming-6000h (~51MB, Vietnamese O(1) streaming)
   - `zipformer-en-20m`: sherpa-onnx-streaming-zipformer-en-20M-2023-02-17 (~25MB INT8, English O(1) streaming)
   - `viet-asr`: embedded 78MB INT8 Zipformer2 + Silero VAD + Punctuation/ITN (zero download)
2. MT (Translation):
   - `opus-vi-en`: dekthedev/opus-mt-vi-en-ct2-int8 (~77MB, ultra-fast VI -> EN)
   - `opus-en-vi`: dekthedev/opus-mt-en-vi-ct2-int8 (~77MB, ultra-fast EN -> VI)
   - `nllb-600m`: mijuanlo/nllb-200-distilled-600M-ct2-int8 (~615MB INT8, 202 x 202 languages offline)
"""

from __future__ import annotations

import os
import shutil
import tarfile
import threading
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import sentencepiece as spm
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

DEFAULT_MODELS_DIR = Path(
    os.environ.get("VOICERECORE_MODEL_DIR", Path.home() / ".cache" / "voicerecore" / "models")
)

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

OPUS_VI_EN_REPO = "dekthedev/opus-mt-vi-en-ct2-int8"
OPUS_EN_VI_REPO = "dekthedev/opus-mt-en-vi-ct2-int8"
ENVIT5_REPO = "luigi000/envit5-translation-ct2-int8"
HYMT2_REPO = "tencent/Hy-MT2-1.8B-GGUF"
HYMT2_FILENAME = "Hy-MT2-1.8B-Q4_K_M.gguf"
NLLB_13B_REPO = "JustFrederik/nllb-200-distilled-1.3B-ct2-int8"


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
        p = self.hy_mt2_gguf_path
        return p.exists() and p.stat().st_size > 100_000_000

    def is_nllb_13b_ready(self) -> bool:
        d = self.nllb_13b_dir
        if not d.exists():
            return False
        has_core = all((d / f).exists() and (d / f).stat().st_size > 0 for f in NLLB_REQUIRED_FILES)
        has_vocab = any((d / f).exists() and (d / f).stat().st_size > 0 for f in NLLB_OPTIONAL_VOCAB_FILES)
        return has_core and has_vocab

    def is_nllb_ready(self) -> bool:
        d = self.nllb_600m_dir
        if not d.exists():
            return False
        has_core = all((d / f).exists() and (d / f).stat().st_size > 0 for f in NLLB_REQUIRED_FILES)
        has_vocab = any((d / f).exists() and (d / f).stat().st_size > 0 for f in NLLB_OPTIONAL_VOCAB_FILES)
        return has_core and has_vocab

    def is_zipformer_streaming_ready(self) -> bool:
        d = self.zipformer_streaming_dir
        if not d.exists():
            return False
        has_onnx = all((d / f).exists() and (d / f).stat().st_size > 0 for f in ZIPFORMER_STREAMING_FILES)
        return has_onnx and (d / "tokens.txt").exists()

    def is_zipformer_en_20m_ready(self) -> bool:
        d = self.zipformer_en_20m_dir
        req = [
            "encoder-epoch-99-avg-1.int8.onnx",
            "decoder-epoch-99-avg-1.int8.onnx",
            "joiner-epoch-99-avg-1.int8.onnx",
            "tokens.txt",
        ]
        return d.exists() and all((d / f).exists() and (d / f).stat().st_size > 0 for f in req)

    def is_moonshine_en_ready(self) -> bool:
        d = self.moonshine_en_dir
        req = ["encoder_model.ort", "decoder_model_merged.ort", "tokens.txt"]
        return d.exists() and all((d / f).exists() and (d / f).stat().st_size > 0 for f in req)

    def is_opus_vi_en_ready(self) -> bool:
        d = self.opus_vi_en_dir
        required = ["model.bin", "source.spm", "target.spm", "config.json"]
        return d.exists() and all((d / f).exists() and (d / f).stat().st_size > 0 for f in required)

    def is_opus_en_vi_ready(self) -> bool:
        d = self.opus_en_vi_dir
        required = ["model.bin", "source.spm", "target.spm", "config.json"]
        return d.exists() and all((d / f).exists() and (d / f).stat().st_size > 0 for f in required)

    def is_envit5_ready(self) -> bool:
        d = self.envit5_dir
        required = ["model.bin", "spiece.model", "config.json"]
        return d.exists() and all((d / f).exists() and (d / f).stat().st_size > 0 for f in required)

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
                    total_mb=30.0,
                    downloaded_mb=30.0 if self.is_zipformer_en_20m_ready() else 0.0,
                    message="Sẵn sàng (Anh Streaming Pass-1 ~6ms)"
                    if self.is_zipformer_en_20m_ready()
                    else "Chưa tải (~30 MB)",
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

    def ensure_zipformer_streaming(self, on_progress: Optional[Callable[[dict], None]] = None) -> Path:
        if self.is_zipformer_streaming_ready():
            self._update_state("zipformer-30m-streaming", on_progress, status="ready", progress=100.0)
            return self.zipformer_streaming_dir

        target_dir = self.zipformer_streaming_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        self._update_state(
            "zipformer-30m-streaming",
            on_progress,
            status="downloading",
            progress=5.0,
            total_mb=51.0,
            message="Đang tải Zipformer-30M Việt Streaming (~51 MB)...",
        )
        try:
            for idx, filename in enumerate(ZIPFORMER_STREAMING_FILES):
                self._update_state(
                    "zipformer-30m-streaming",
                    on_progress,
                    status="downloading",
                    progress=round(10.0 + (idx / len(ZIPFORMER_STREAMING_FILES)) * 80.0, 1),
                    message=f"Đang tải {filename}...",
                )
                hf_hub_download(repo_id=ZIPFORMER_STREAMING_REPO, filename=filename, local_dir=str(target_dir))

            sp = spm.SentencePieceProcessor(model_file=str(target_dir / "bpe.model"))
            with open(target_dir / "tokens.txt", "w", encoding="utf-8") as tf:
                for i in range(sp.get_piece_size()):
                    tf.write(f"{sp.id_to_piece(i)} {i}\n")

            self._update_state(
                "zipformer-30m-streaming",
                on_progress,
                status="ready",
                progress=100.0,
                downloaded_mb=51.0,
                total_mb=51.0,
                message="Sẵn sàng (Việt Streaming 6000h ~30ms)",
            )
            return target_dir
        except Exception as exc:
            self._update_state("zipformer-30m-streaming", on_progress, status="error", message=f"Lỗi tải: {exc}")
            raise

    def ensure_zipformer_en_20m(self, on_progress: Optional[Callable[[dict], None]] = None) -> Path:
        if self.is_zipformer_en_20m_ready():
            self._update_state("zipformer-en-20m", on_progress, status="ready", progress=100.0)
            return self.zipformer_en_20m_dir

        archive_path = self.models_dir / f"{ZIPFORMER_EN_20M_DIRNAME}.tar.bz2"
        self._update_state(
            "zipformer-en-20m",
            on_progress,
            status="downloading",
            progress=5.0,
            total_mb=30.0,
            message="Đang tải Zipformer-20M Tiếng Anh Streaming (~30 MB)...",
        )
        try:
            def reporthook(bnum: int, bsize: int, total: int) -> None:
                if total > 0:
                    pct = round(min(90.0, (bnum * bsize / total) * 90.0), 1)
                    self._update_state("zipformer-en-20m", on_progress, status="downloading", progress=pct)

            urllib.request.urlretrieve(ZIPFORMER_EN_20M_URL, archive_path, reporthook=reporthook)
            with tarfile.open(archive_path, "r:bz2") as tf:
                tf.extractall(self.models_dir)
            archive_path.unlink(missing_ok=True)
            self._update_state(
                "zipformer-en-20m",
                on_progress,
                status="ready",
                progress=100.0,
                downloaded_mb=30.0,
                total_mb=30.0,
                message="Sẵn sàng (Anh Streaming 20M ~20ms)",
            )
            return self.zipformer_en_20m_dir
        except Exception as exc:
            archive_path.unlink(missing_ok=True)
            self._update_state("zipformer-en-20m", on_progress, status="error", message=f"Lỗi tải: {exc}")
            raise

    def ensure_moonshine_en(self, on_progress: Optional[Callable[[dict], None]] = None) -> Path:
        if self.is_moonshine_en_ready():
            self._update_state("moonshine-en", on_progress, status="ready", progress=100.0)
            return self.moonshine_en_dir

        archive_path = self.models_dir / f"{MOONSHINE_EN_DIRNAME}.tar.bz2"
        self._update_state(
            "moonshine-en",
            on_progress,
            status="downloading",
            progress=5.0,
            total_mb=106.0,
            message="Đang tải Moonshine-v2 Base Tiếng Anh (~106 MB • 1.8% WER)...",
        )
        try:
            def reporthook(bnum: int, bsize: int, total: int) -> None:
                if total > 0:
                    pct = round(min(90.0, (bnum * bsize / total) * 90.0), 1)
                    self._update_state("moonshine-en", on_progress, status="downloading", progress=pct)

            urllib.request.urlretrieve(MOONSHINE_EN_URL, archive_path, reporthook=reporthook)
            with tarfile.open(archive_path, "r:bz2") as tf:
                tf.extractall(self.models_dir)
            archive_path.unlink(missing_ok=True)
            self._update_state(
                "moonshine-en",
                on_progress,
                status="ready",
                progress=100.0,
                downloaded_mb=106.0,
                total_mb=106.0,
                message="Sẵn sàng (Moonshine-v2 EN • 1.8% WER • Dấu câu • 44× RT)",
            )
            return self.moonshine_en_dir
        except Exception as exc:
            archive_path.unlink(missing_ok=True)
            self._update_state("moonshine-en", on_progress, status="error", message=f"Lỗi tải Moonshine-v2: {exc}")
            raise

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

        target_dir.mkdir(parents=True, exist_ok=True)
        self._update_state(
            model_id,
            on_progress,
            status="downloading",
            progress=10.0,
            total_mb=77.0,
            message=f"Đang tải Opus-MT {label} INT8 (~77 MB)...",
        )
        try:
            snapshot_download(repo_id=repo_id, local_dir=str(target_dir))
            self._update_state(
                model_id,
                on_progress,
                status="ready",
                progress=100.0,
                downloaded_mb=77.0,
                total_mb=77.0,
                message=f"Sẵn sàng ({label} siêu tốc)",
            )
            return target_dir
        except Exception as exc:
            self._update_state(model_id, on_progress, status="error", message=f"Lỗi tải: {exc}")
            raise

    def ensure_envit5(self, on_progress: Optional[Callable[[dict], None]] = None) -> Path:
        if self.is_envit5_ready():
            self._update_state("envit5", on_progress, status="ready", progress=100.0)
            return self.envit5_dir

        target_dir = self.envit5_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        self._update_state(
            "envit5",
            on_progress,
            status="downloading",
            progress=10.0,
            total_mb=266.0,
            message="Đang tải VietAI EnViT5 INT8 (~266 MB)...",
        )
        try:
            snapshot_download(repo_id=ENVIT5_REPO, local_dir=str(target_dir))
            if not (target_dir / "spiece.model").exists():
                hf_hub_download("VietAI/envit5-translation", "spiece.model", local_dir=str(target_dir))
            self._update_state(
                "envit5",
                on_progress,
                status="ready",
                progress=100.0,
                downloaded_mb=266.0,
                total_mb=266.0,
                message="Sẵn sàng (VietAI EnViT5 Anh⇄Việt • OpenRAIL)",
            )
            return target_dir
        except Exception as exc:
            self._update_state("envit5", on_progress, status="error", message=f"Lỗi tải VietAI EnViT5: {exc}")
            raise

    def ensure_nllb_600m(self, on_progress: Optional[Callable[[dict], None]] = None) -> Path:
        if self.is_nllb_ready():
            self._update_state("nllb-600m", on_progress, status="ready", progress=100.0)
            return self.nllb_600m_dir

        target_dir = self.nllb_600m_dir
        target_dir.mkdir(parents=True, exist_ok=True)
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
                api = HfApi()
                repo_info = api.repo_info(repo_id=repo_id, files_metadata=True)
                available = {f.rfilename: (f.size or 0) for f in (repo_info.siblings or [])}
                wanted = [f for f in NLLB_REQUIRED_FILES if f in available]
                for vf in NLLB_OPTIONAL_VOCAB_FILES:
                    if vf in available:
                        wanted.append(vf)
                        break

                total_bytes = max(sum(available.get(f, 0) for f in wanted), 1)
                total_mb = round(total_bytes / (1024 * 1024), 1)
                done_bytes = 0

                for idx, filename in enumerate(wanted):
                    file_size = available.get(filename, 0)
                    self._update_state(
                        "nllb-600m",
                        on_progress,
                        status="downloading",
                        progress=round(min(95.0, (done_bytes / total_bytes) * 100.0 + 1.0), 1),
                        downloaded_mb=round(done_bytes / (1024 * 1024), 1),
                        total_mb=total_mb,
                        message=f"Đang tải {filename} ({idx + 1}/{len(wanted)})...",
                    )
                    downloaded_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        local_dir=str(target_dir),
                    )
                    if not Path(downloaded_path).exists():
                        raise RuntimeError(f"Failed downloading {filename}")
                    done_bytes += file_size

                if self.is_nllb_ready():
                    self._update_state(
                        "nllb-600m",
                        on_progress,
                        status="ready",
                        progress=100.0,
                        downloaded_mb=total_mb,
                        total_mb=total_mb,
                        message="Sẵn sàng (202×202 ngôn ngữ offline)",
                    )
                    return target_dir
            except Exception as exc:
                last_err = exc
                continue

        self._update_state("nllb-600m", on_progress, status="error", message=f"Lỗi tải NLLB-200: {last_err}")
        raise RuntimeError(f"Could not download NLLB-200 CTranslate2 model: {last_err}")

    def ensure_hymt2(self, on_progress: Optional[Callable[[dict], None]] = None) -> Path:
        if self.is_hymt2_ready():
            self._update_state("hy-mt2", on_progress, status="ready", progress=100.0)
            return self.hy_mt2_gguf_path

        self.hy_mt2_dir.mkdir(parents=True, exist_ok=True)
        self._update_state(
            "hy-mt2",
            on_progress,
            status="downloading",
            progress=5.0,
            total_mb=1080.0,
            message="Đang tải Tencent Hy-MT2-1.8B Q4_K_M (~1.05 GB)...",
        )
        try:
            hf_hub_download(repo_id=HYMT2_REPO, filename=HYMT2_FILENAME, local_dir=str(self.hy_mt2_dir))
            self._update_state(
                "hy-mt2",
                on_progress,
                status="ready",
                progress=100.0,
                downloaded_mb=1080.0,
                total_mb=1080.0,
                message="Sẵn sàng (Tencent Hy-MT2-1.8B • 38 NN • Ngữ cảnh & Thuật ngữ)",
            )
            return self.hy_mt2_gguf_path
        except Exception as exc:
            self._update_state("hy-mt2", on_progress, status="error", message=f"Lỗi tải Hy-MT2-1.8B: {exc}")
            raise

    def ensure_nllb_13b(self, on_progress: Optional[Callable[[dict], None]] = None) -> Path:
        if self.is_nllb_13b_ready():
            self._update_state("nllb-1.3b", on_progress, status="ready", progress=100.0)
            return self.nllb_13b_dir

        self.nllb_13b_dir.mkdir(parents=True, exist_ok=True)
        self._update_state(
            "nllb-1.3b",
            on_progress,
            status="downloading",
            progress=5.0,
            total_mb=1341.0,
            message="Đang tải NLLB-200 1.3B INT8 (~1.34 GB — Chuẩn cao Lào / Thái / 202 NN)...",
        )
        try:
            snapshot_download(
                repo_id=NLLB_13B_REPO,
                local_dir=str(self.nllb_13b_dir),
                allow_patterns=[
                    "model.bin",
                    "config.json",
                    "sentencepiece.bpe.model",
                    "shared_vocabulary.*",
                    "tokenizer.json",
                ],
            )
            self._update_state(
                "nllb-1.3b",
                on_progress,
                status="ready",
                progress=100.0,
                downloaded_mb=1341.0,
                total_mb=1341.0,
                message="Sẵn sàng (NLLB-200 1.3B • Chuẩn cao Lào / Thái / 202 NN)",
            )
            return self.nllb_13b_dir
        except Exception as exc:
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
            shutil.rmtree(target_dir, ignore_errors=True)

        with self._lock:
            self._states.pop(model_id, None)

        return {
            "ok": True,
            "model_id": model_id,
            "freed_mb": round(freed_bytes / (1024 * 1024), 1),
        }
