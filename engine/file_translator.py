"""
Offline Batch Audio/Video File Translator (.mp3 / .wav / .m4a / .mp4 -> Bilingual .SRT & .TXT).

Uses local `ffmpeg` (or macOS built-in `afconvert` / Python `wave`) to extract 16 kHz mono PCM,
segments speech at natural pauses, runs offline ASR (`VietASR` with punctuation/ITN for Vietnamese,
`Zipformer-20M EN` for English), and translates with VOICERECORE's active offline MT engine
(`Hy-MT2-1.8B`, `VietAI EnViT5`, `NLLB-200-1.3B`).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any, Callable

import numpy as np

from languages import PARAKEET_EU_LANGS, infer_text_nllb_lang
from session_logger import _fmt_srt_time

SAMPLE_RATE = 16000


class FileTranslateCancelled(RuntimeError):
    """Raised when the user cancels an active batch file translation job."""


def _find_ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and Path(bundled).is_file():
            return bundled
    except Exception:
        pass
    return None


def _decode_wav_numpy(p: Path) -> np.ndarray | None:
    """Decode any PCM WAV (arbitrary rate/channels/bit-depth) to 16 kHz mono int16 with pure NumPy."""
    try:
        with wave.open(str(p), "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            n_frames = wf.getnframes()
            if sr <= 0 or ch <= 0 or n_frames <= 0:
                return None
            raw = wf.readframes(n_frames)
            if sr == SAMPLE_RATE and ch == 1 and sw == 2:
                return np.frombuffer(raw, dtype=np.int16).copy()

            if sw == 1:
                samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            elif sw == 2:
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            elif sw == 3:
                b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
                signed = (
                    b[:, 0].astype(np.int32)
                    | (b[:, 1].astype(np.int32) << 8)
                    | (b[:, 2].astype(np.int32) << 16)
                )
                signed = np.where(signed >= 0x800000, signed - 0x1000000, signed)
                samples = signed.astype(np.float32) / 8388608.0
            elif sw == 4:
                samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                return None

            if ch > 1:
                usable = (len(samples) // ch) * ch
                samples = samples[:usable].reshape(-1, ch).mean(axis=1)

            if sr != SAMPLE_RATE and len(samples) > 1:
                out_len = max(1, int(round(len(samples) * SAMPLE_RATE / sr)))
                x_old = np.linspace(0.0, 1.0, len(samples), endpoint=False)
                x_new = np.linspace(0.0, 1.0, out_len, endpoint=False)
                samples = np.interp(x_new, x_old, samples).astype(np.float32)

            return np.clip(samples * 32768.0, -32768.0, 32767.0).astype(np.int16)
    except Exception:
        return None


def _run_cancellable_proc(
    cmd: list[str],
    *,
    capture_stdout: bool,
    timeout_s: float,
    cancel_event: threading.Event | None = None,
) -> tuple[int, bytes]:
    kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

    proc = subprocess.Popen(cmd, **kwargs)
    chunks: list[bytes] = []
    reader_thread: threading.Thread | None = None
    if capture_stdout and proc.stdout is not None:
        stream = proc.stdout

        def _drain() -> None:
            try:
                while True:
                    block = stream.read(65536)
                    if not block:
                        break
                    chunks.append(block)
            except Exception:
                pass
            finally:
                try:
                    stream.close()
                except Exception:
                    pass

        reader_thread = threading.Thread(target=_drain, daemon=True, name="file-decode-stdout")
        reader_thread.start()

    deadline = time.monotonic() + timeout_s
    try:
        while proc.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=1.5)
                except Exception:
                    proc.kill()
                raise FileTranslateCancelled("Đã hủy dịch file.")
            if time.monotonic() > deadline:
                proc.terminate()
                try:
                    proc.wait(timeout=1.5)
                except Exception:
                    proc.kill()
                raise RuntimeError("Giải mã âm thanh quá thời gian chờ.")
            time.sleep(0.05)
    finally:
        if reader_thread is not None:
            reader_thread.join(timeout=2.0)

    return proc.returncode or 0, b"".join(chunks)


def _dynamic_normalize_pcm16(
    pcm: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    frame_ms: int = 150,
    target_rms: float = 0.085,
    max_gain: float = 12.0,
    noise_floor: float = 0.0007,
) -> np.ndarray:
    """
    Pure-NumPy windowed dynamic loudness normalizer (equivalent to ffmpeg `dynaudnorm`).
    Lifts quiet speakers (e.g. far from mic, RMS ~0.0018) cleanly so VAD never drops them,
    while keeping pure silence unamplified and clamping peaks softly below clipping.
    """
    if pcm is None or len(pcm) == 0:
        return pcm
    f32 = pcm.astype(np.float32) / 32768.0
    win = max(160, int(sample_rate * frame_ms / 1000))
    n = len(f32)
    n_frames = (n + win - 1) // win
    pad = n_frames * win - n
    padded = np.pad(f32, (0, pad)) if pad > 0 else f32
    frames = padded.reshape(n_frames, win)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    peaks = np.max(np.abs(frames), axis=1) + 1e-12

    gains = np.ones(n_frames, dtype=np.float32)
    active = rms >= noise_floor
    if np.any(active):
        desired = np.minimum(target_rms / rms[active], 0.90 / peaks[active])
        gains[active] = np.clip(desired, 0.5, max_gain)

    # Smooth gain trajectory across neighboring frames (~0.75s window) to avoid pumping artifacts
    k = min(5, n_frames)
    if k >= 3:
        kernel = np.ones(k, dtype=np.float32) / float(k)
        gains = np.convolve(np.pad(gains, (k // 2, k // 2), mode="edge"), kernel, mode="valid")[:n_frames]

    # Interpolate frame gains smoothly to per-sample envelope
    centers = (np.arange(n_frames, dtype=np.float32) + 0.5) * win
    sample_gains = np.interp(np.arange(n, dtype=np.float32), centers, gains).astype(np.float32)
    out = np.tanh((f32 * sample_gains) * 1.05) / np.tanh(1.05)
    return np.clip(out * 32767.0, -32768.0, 32767.0).astype(np.int16)


def decode_media_to_pcm16(
    file_path: Path,
    cancel_event: threading.Event | None = None,
    normalize_audio: bool = True,
) -> np.ndarray:
    """Decode any audio/video file to 16 kHz mono int16 numpy array (with optional dynamic normalization)."""
    if cancel_event is not None and cancel_event.is_set():
        raise FileTranslateCancelled("Đã hủy dịch file.")

    p = Path(file_path)
    # Fast & zero-dependency path: pure-NumPy PCM WAV decode (handles any rate/channels/bit-depth)
    if p.suffix.lower() == ".wav":
        decoded = _decode_wav_numpy(p)
        if decoded is not None and len(decoded) > 0:
            return _dynamic_normalize_pcm16(decoded) if normalize_audio else decoded

    ffmpeg = _find_ffmpeg()
    if ffmpeg:
        base_cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(p), "-vn"]
        tail_cmd = ["-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "pipe:1"]
        attempts: list[tuple[list[str], bool]] = []
        if normalize_audio:
            attempts.append((base_cmd + ["-af", "dynaudnorm=f=150:g=15:p=0.9:m=10"] + tail_cmd, False))
        attempts.append((base_cmd + tail_cmd, normalize_audio))

        for cmd, need_numpy_norm in attempts:
            rc, stdout_bytes = _run_cancellable_proc(
                cmd,
                capture_stdout=True,
                timeout_s=300.0,
                cancel_event=cancel_event,
            )
            if rc == 0 and stdout_bytes:
                usable = (len(stdout_bytes) // 2) * 2
                arr = np.frombuffer(stdout_bytes[:usable], dtype=np.int16).copy()
                return _dynamic_normalize_pcm16(arr) if need_numpy_norm else arr

    afconvert = shutil.which("afconvert")
    if afconvert:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_wav = Path(tf.name)
        try:
            rc, _ = _run_cancellable_proc(
                [afconvert, "-f", "WAVE", "-d", f"LEI16@{SAMPLE_RATE}", "-c", "1", str(p), str(tmp_wav)],
                capture_stdout=False,
                timeout_s=180.0,
                cancel_event=cancel_event,
            )
            if rc == 0 and tmp_wav.exists():
                decoded = _decode_wav_numpy(tmp_wav)
                if decoded is not None and len(decoded) > 0:
                    return _dynamic_normalize_pcm16(decoded) if normalize_audio else decoded
        finally:
            tmp_wav.unlink(missing_ok=True)

    raise RuntimeError(
        "Không thể giải mã định dạng file này. Hãy cài `ffmpeg` (hoặc tải lên file .WAV PCM)."
    )


def segment_pcm_by_silero_vad(
    pcm: np.ndarray,
    vad_model_path: Path,
    sample_rate: int = SAMPLE_RATE,
    max_seg_s: float = 13.0,
) -> list[tuple[float, float, np.ndarray]]:
    """
    Segment speech cleanly with Silero VAD v5 so SRT timestamps match real spoken turns
    and silent/music sections are never sent to ASR (eliminates Whisper hallucinations).
    """
    if pcm is None or len(pcm) == 0 or not Path(vad_model_path).is_file():
        return []
    try:
        import sherpa_onnx
    except Exception:
        return []

    win = 512
    silero = sherpa_onnx.SileroVadModelConfig(
        model=str(vad_model_path),
        threshold=0.40,
        min_silence_duration=0.45,
        min_speech_duration=0.22,
        window_size=win,
        max_speech_duration=max_seg_s,
    )
    cfg = sherpa_onnx.VadModelConfig(
        silero_vad=silero,
        sample_rate=sample_rate,
        num_threads=1,
        provider="cpu",
        debug=False,
    )
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=120.0)
    f32 = pcm.astype(np.float32) / 32768.0
    total_samples = len(f32)

    raw_spans: list[tuple[int, int]] = []
    pos = 0
    while pos + win <= total_samples:
        vad.accept_waveform(f32[pos : pos + win])
        pos += win
        while not vad.empty():
            seg = vad.front
            s_start = max(0, int(seg.start))
            s_end = min(total_samples, s_start + len(seg.samples))
            if s_end > s_start:
                raw_spans.append((s_start, s_end))
            vad.pop()

    if pos < total_samples:
        tail = np.zeros(win, dtype=np.float32)
        tail[: total_samples - pos] = f32[pos:]
        vad.accept_waveform(tail)
    vad.flush()
    while not vad.empty():
        seg = vad.front
        s_start = max(0, int(seg.start))
        s_end = min(total_samples, s_start + len(seg.samples))
        if s_end > s_start:
            raw_spans.append((s_start, s_end))
        vad.pop()

    if not raw_spans:
        return []

    # Merge very short adjacent speech islands (gap <= 0.65s) up to max_seg_s for cleaner subtitles
    merged: list[tuple[int, int]] = []
    max_samples = int(max_seg_s * sample_rate)
    max_gap = int(0.65 * sample_rate)
    pad_samples = int(0.12 * sample_rate)
    for s_start, s_end in raw_spans:
        if merged and (s_start - merged[-1][1]) <= max_gap and (s_end - merged[-1][0]) <= max_samples:
            merged[-1] = (merged[-1][0], max(merged[-1][1], s_end))
        else:
            merged.append((s_start, s_end))

    out: list[tuple[float, float, np.ndarray]] = []
    for s_start, s_end in merged:
        ps = max(0, s_start - pad_samples)
        pe = min(total_samples, s_end + pad_samples)
        chunk = pcm[ps:pe]
        if len(chunk) >= int(0.22 * sample_rate):
            out.append((s_start / sample_rate, s_end / sample_rate, chunk))
    return out


def segment_pcm_by_pauses(
    pcm: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    min_seg_s: float = 3.5,
    target_seg_s: float = 11.0,
    max_seg_s: float = 16.0,
) -> list[tuple[float, float, np.ndarray]]:
    """
    Split long audio into natural speech chunks (3.5s–16s) at low-energy pauses so
    ASR punctuation & SRT timestamps align cleanly with human phrasing, skipping dead silence.
    """
    total_samples = len(pcm)
    if total_samples == 0:
        return []

    def _is_audible(seg: np.ndarray) -> bool:
        if len(seg) == 0:
            return False
        s_f32 = seg.astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(s_f32 * s_f32))) >= 0.0035

    win = int(sample_rate * 0.1)  # 100ms frames
    n_frames = total_samples // win
    if n_frames <= int(max_seg_s * 10):
        return [(0.0, total_samples / sample_rate, pcm)] if _is_audible(pcm) else []

    f32 = pcm[: n_frames * win].reshape(n_frames, win).astype(np.float32) / 32768.0
    rms = np.sqrt(np.mean(f32 * f32, axis=1))
    silence_thresh = max(0.006, float(np.percentile(rms, 18)) * 1.45)

    segments: list[tuple[float, float, np.ndarray]] = []
    start_f = 0
    min_f = int(min_seg_s * 10)
    target_f = int(target_seg_s * 10)
    max_f = int(max_seg_s * 10)

    while start_f < n_frames:
        rem = n_frames - start_f
        if rem <= max_f:
            s_idx = start_f * win
            e_idx = total_samples
            seg = pcm[s_idx:e_idx]
            if _is_audible(seg):
                segments.append((s_idx / sample_rate, e_idx / sample_rate, seg))
            break

        # Search for quiet pause between target_f and max_f (or earliest >= 300ms pause after min_f)
        split_f = -1
        for f in range(start_f + min_f, min(n_frames - 5, start_f + max_f)):
            if f >= start_f + target_f and rms[f] <= silence_thresh and rms[f + 1] <= silence_thresh:
                split_f = f + 1
                break
        if split_f < 0:
            # Pick absolute quietest frame between 60% and 100% of max_f
            lo = start_f + int(max_f * 0.6)
            hi = min(n_frames - 2, start_f + max_f)
            split_f = lo + int(np.argmin(rms[lo:hi]))

        s_idx = start_f * win
        e_idx = split_f * win
        seg = pcm[s_idx:e_idx]
        if _is_audible(seg):
            segments.append((s_idx / sample_rate, e_idx / sample_rate, seg))
        start_f = split_f

    return segments


def translate_media_file(
    file_path: Path,
    orig_filename: str,
    source_lang: str,
    target_lang: str,
    mt_mode: str,
    runtime: Any,
    logs_dir: Path,
    on_progress: Callable[[float, str], None] | None = None,
    on_progress_event: Callable[[dict[str, Any]], None] | None = None,
    cancel_event: threading.Event | None = None,
    job_id: str = "",
    asr_mode: str = "auto",
    normalize_audio: bool = True,
) -> dict[str, Any]:
    """Full offline/hybrid pipeline: Media -> dynaudnorm PCM -> Silero VAD -> ASR -> Glossary -> MT -> .TXT & .SRT."""
    t0 = time.perf_counter()

    def _emit(
        stage: str,
        pct: float,
        message: str,
        *,
        current_chunk: int = 0,
        total_chunks: int = 0,
        processed_audio_s: float = 0.0,
        duration_s: float = 0.0,
        eta_s: float | None = None,
        eta_estimated: bool = False,
        speed_x: float = 0.0,
        active: bool = True,
    ) -> None:
        if on_progress:
            on_progress(pct, message)
        if on_progress_event:
            on_progress_event(
                {
                    "job_id": job_id,
                    "filename": orig_filename,
                    "active": active,
                    "stage": stage,
                    "percent": round(max(0.0, min(100.0, pct)), 1),
                    "message": message,
                    "current_chunk": current_chunk,
                    "total_chunks": total_chunks,
                    "processed_audio_s": round(processed_audio_s, 1),
                    "duration_s": round(duration_s, 1),
                    "elapsed_s": round(time.perf_counter() - t0, 1),
                    "eta_s": round(max(0.0, eta_s), 1) if eta_s is not None else None,
                    "eta_estimated": eta_estimated,
                    "speed_x": round(speed_x, 1),
                }
            )

    _emit(
        "decoding",
        3.0,
        "Đang giải mã & cân bằng âm lượng (dynaudnorm)..." if normalize_audio else "Đang giải mã âm thanh 16kHz...",
    )
    pcm = decode_media_to_pcm16(file_path, cancel_event=cancel_event, normalize_audio=normalize_audio)
    duration_s = round(len(pcm) / SAMPLE_RATE, 1)
    if duration_s < 0.3:
        raise RuntimeError("File âm thanh quá ngắn hoặc không có dữ liệu tiếng nói.")
    if cancel_event is not None and cancel_event.is_set():
        raise FileTranslateCancelled("Đã hủy dịch file.")

    vad_used = False
    chunks: list[tuple[float, float, np.ndarray]] = []
    if runtime.model_manager.is_silero_vad_ready():
        chunks = segment_pcm_by_silero_vad(pcm, runtime.model_manager.silero_vad_path)
        if chunks:
            vad_used = True
    if not chunks:
        chunks = segment_pcm_by_pauses(pcm)
    if not chunks:
        raise RuntimeError("Không phát hiện giọng nói rõ ràng trong file âm thanh.")

    force_whisper = (asr_mode or "auto").strip().lower() == "whisper-turbo"
    est_speed = 2.6 if force_whisper else (6.0 if mt_mode == "cloud-api" else 12.0)
    est_eta_s = max(1.0, round(duration_s / est_speed, 1))
    seg_mode_label = "Silero VAD v5" if vad_used else "nhịp nghỉ"
    _emit(
        "loading_models",
        8.0,
        f"Đã tách {len(chunks)} câu nói ({seg_mode_label}). Đang nạp mô hình AI...",
        total_chunks=len(chunks),
        duration_s=duration_s,
        eta_s=est_eta_s,
        eta_estimated=True,
    )

    use_vi_batch = (not force_whisper) and source_lang == "vie_Latn"
    refiner = runtime.ensure_pass2_refiner() if use_vi_batch else None
    zf_vi = (
        runtime.ensure_onnx_asr_loaded("zipformer-30m-streaming")
        if use_vi_batch and runtime.model_manager.is_zipformer_streaming_ready()
        else None
    )
    en_moonshine = (
        runtime.ensure_en_pass2_refiner()
        if (not force_whisper) and source_lang == "eng_Latn"
        else None
    )
    ja_moonshine = (
        runtime.ensure_onnx_asr_loaded("moonshine-ja", lang_code=source_lang)
        if (not force_whisper)
        and source_lang == "jpn_Jpan"
        and runtime.model_manager.is_moonshine_ja_ready()
        and runtime.model_manager.is_silero_vad_ready()
        else None
    )
    ko_moonshine = (
        runtime.ensure_onnx_asr_loaded("moonshine-ko", lang_code=source_lang)
        if (not force_whisper)
        and source_lang == "kor_Hang"
        and runtime.model_manager.is_moonshine_ko_ready()
        and runtime.model_manager.is_silero_vad_ready()
        else None
    )
    parakeet_eu = (
        runtime.ensure_onnx_asr_loaded("parakeet-eu", lang_code=source_lang)
        if (not force_whisper)
        and source_lang in PARAKEET_EU_LANGS
        and runtime.model_manager.is_parakeet_eu_ready()
        else None
    )
    whisper_asr = None
    if force_whisper or (
        not use_vi_batch
        and en_moonshine is None
        and ja_moonshine is None
        and ko_moonshine is None
        and parakeet_eu is None
    ):
        if runtime.model_manager.is_whisper_turbo_ready():
            whisper_asr = runtime.ensure_onnx_asr_loaded("whisper-turbo", lang_code=source_lang)
        else:
            if source_lang in PARAKEET_EU_LANGS:
                raise RuntimeError(
                    "Chưa tải 'Parakeet-Ultra 25 NN Châu Âu' (~465 MB). "
                    "Hãy mở mục 'Mô hình Offline' ở cột trái để tải Gói 3 (Châu Âu siêu tốc)."
                )
            raise RuntimeError(
                "Chưa tải 'Nghe Mở rộng 99 NN (Whisper-Turbo)' (~540 MB). "
                "Hãy mở mục 'Mô hình Offline → Chi tiết từng model' ở cột trái để tải thêm."
            )
    if cancel_event is not None and cancel_event.is_set():
        raise FileTranslateCancelled("Đã hủy dịch file.")

    asr_used_label = (
        "Whisper Large-v3-Turbo"
        if whisper_asr is not None
        else (
            "Parakeet-Ultra 25 NN Châu Âu"
            if parakeet_eu is not None
            else (
                "Moonshine-v2 Tiếng Nhật"
                if ja_moonshine is not None
                else (
                    "Moonshine-v2 Tiếng Hàn"
                    if ko_moonshine is not None
                    else ("VietASR + Zipformer" if use_vi_batch else "Moonshine-v2")
                )
            )
        )
    )

    protected_terms = runtime._get_glossary_hotwords()
    segments_out: list[dict[str, Any]] = []
    recent_ctx: list[tuple[str, str]] = []
    last_seg_lang = ""
    processed_audio_s = 0.0
    t_loop = time.perf_counter()

    _emit(
        "translating",
        12.0,
        f"Đang nhận dạng ({asr_used_label}) & dịch đoạn 1/{len(chunks)}...",
        current_chunk=0,
        total_chunks=len(chunks),
        processed_audio_s=0.0,
        duration_s=duration_s,
        eta_s=est_eta_s,
        eta_estimated=True,
    )

    for idx, (start_s, end_s, seg_pcm) in enumerate(chunks):
        if cancel_event is not None and cancel_event.is_set():
            raise FileTranslateCancelled("Đã hủy dịch file.")

        det_lang = ""
        if whisper_asr is not None:
            if source_lang == "auto":
                text, det_lang = whisper_asr.transcribe_pcm_isolated_with_lang(seg_pcm, language="auto")
            else:
                text = whisper_asr.transcribe_pcm_isolated(seg_pcm, language=source_lang)
        elif parakeet_eu is not None:
            text = parakeet_eu.transcribe_pcm_isolated(seg_pcm, language=source_lang)
        elif ja_moonshine is not None:
            text = ja_moonshine.transcribe_pcm_isolated(seg_pcm, language=source_lang)
        elif ko_moonshine is not None:
            text = ko_moonshine.transcribe_pcm_isolated(seg_pcm, language=source_lang)
        elif use_vi_batch and refiner is not None:
            p1_fallback = ""
            if zf_vi is not None:
                p1_raw = zf_vi.transcribe_pcm_isolated(seg_pcm)
                p1_fallback = runtime.glossary.apply_asr_fixes(p1_raw)
            text, _ = refiner.refine(seg_pcm, p1_fallback, protected_terms=protected_terms)
        elif en_moonshine is not None:
            text, _ = en_moonshine.refine(seg_pcm, "")
        else:
            text = ""

        text = runtime.glossary.apply_asr_fixes(text.strip())
        if text:
            eff_src_lang = (det_lang or infer_text_nllb_lang(text)) if source_lang == "auto" else source_lang
            if last_seg_lang and eff_src_lang != last_seg_lang:
                recent_ctx.clear()
            last_seg_lang = eff_src_lang
            tr = runtime.translate_text(
                text,
                eff_src_lang,
                target_lang,
                mt_mode=mt_mode,
                beam_size=runtime.config.beam_size,
                llm_url=runtime.config.llm_url,
                llm_model=runtime.config.llm_model,
                llm_api_key=runtime.config.llm_api_key,
                context=recent_ctx[-2:],
            )
            if eff_src_lang != target_lang:
                recent_ctx.append((text, tr.translated_text))
            segments_out.append(
                {
                    "index": len(segments_out) + 1,
                    "start_s": round(start_s, 2),
                    "end_s": round(end_s, 2),
                    "start_srt": _fmt_srt_time(start_s),
                    "end_srt": _fmt_srt_time(end_s),
                    "source_lang": eff_src_lang,
                    "source_text": text,
                    "translated_text": tr.translated_text,
                    "mt_engine": tr.engine,
                }
            )

        processed_audio_s = min(duration_s, processed_audio_s + max(0.1, end_s - start_s))
        loop_elapsed = max(0.05, time.perf_counter() - t_loop)
        cur_speed = processed_audio_s / loop_elapsed
        rem_audio_s = max(0.0, duration_s - processed_audio_s)
        eta_s = rem_audio_s / max(cur_speed, 0.05)
        pct = min(97.0, 12.0 + (processed_audio_s / max(duration_s, 0.1)) * 85.0)
        _emit(
            "translating",
            pct,
            f"Đã xử lý {idx + 1}/{len(chunks)} phân đoạn...",
            current_chunk=idx + 1,
            total_chunks=len(chunks),
            processed_audio_s=processed_audio_s,
            duration_s=duration_s,
            eta_s=eta_s,
            eta_estimated=False,
            speed_x=cur_speed,
        )

    _emit(
        "saving",
        98.5,
        "Đang xuất file biên bản .TXT & .SRT (song ngữ / đơn ngữ)...",
        current_chunk=len(chunks),
        total_chunks=len(chunks),
        processed_audio_s=duration_s,
        duration_s=duration_s,
        eta_s=0.5,
        speed_x=round(duration_s / max(time.perf_counter() - t_loop, 0.1), 1),
    )

    # Build .TXT and all 3 .SRT variants (bilingual, target-only, source-only)
    txt_lines = [
        f"=== DAU Voice Pro — Dịch File Offline: {orig_filename} ({duration_s}s) ===",
        f"Hướng dịch: {source_lang} -> {target_lang} | ASR: {asr_used_label} | Tổng: {len(segments_out)} đoạn",
        "=" * 64,
        "",
    ]
    srt_lines: list[str] = []
    srt_target_lines: list[str] = []
    srt_source_lines: list[str] = []
    for seg in segments_out:
        txt_lines.append(f"[{seg['start_srt']} --> {seg['end_srt']}]")
        txt_lines.append(f"Gốc : {seg['source_text']}")
        txt_lines.append(f"Dịch: {seg['translated_text']}")
        txt_lines.append("")

        srt_lines.append(
            f"{seg['index']}\n{seg['start_srt']} --> {seg['end_srt']}\n"
            f"{seg['translated_text']}\n{seg['source_text']}\n"
        )
        srt_target_lines.append(
            f"{seg['index']}\n{seg['start_srt']} --> {seg['end_srt']}\n"
            f"{seg['translated_text']}\n"
        )
        srt_source_lines.append(
            f"{seg['index']}\n{seg['start_srt']} --> {seg['end_srt']}\n"
            f"{seg['source_text']}\n"
        )

    txt_content = "\n".join(txt_lines)
    srt_content = "\n".join(srt_lines)
    srt_target_content = "\n".join(srt_target_lines)
    srt_source_content = "\n".join(srt_source_lines)

    # Auto-save to logs_dir
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(orig_filename).stem)[:48] or "audio"
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    logs_dir.mkdir(parents=True, exist_ok=True)
    txt_path = logs_dir / f"file_{safe_stem}_{stamp}.txt"
    srt_path = logs_dir / f"file_{safe_stem}_{stamp}.srt"
    txt_path.write_text(txt_content, encoding="utf-8")
    srt_path.write_text(srt_content, encoding="utf-8")

    elapsed_s = round(time.perf_counter() - t0, 2)
    speed_x = round(duration_s / max(elapsed_s, 0.1), 1)
    _emit(
        "done",
        100.0,
        f"Hoàn tất trong {elapsed_s}s ({speed_x}× tốc độ thực)!",
        current_chunk=len(chunks),
        total_chunks=len(chunks),
        processed_audio_s=duration_s,
        duration_s=duration_s,
        eta_s=0.0,
        speed_x=speed_x,
        active=False,
    )

    return {
        "ok": True,
        "filename": orig_filename,
        "duration_s": duration_s,
        "elapsed_s": elapsed_s,
        "speed_x": speed_x,
        "asr_used": asr_used_label,
        "vad_used": vad_used,
        "normalized": normalize_audio,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "segment_count": len(segments_out),
        "segments": segments_out,
        "txt_file": str(txt_path),
        "srt_file": str(srt_path),
        "txt_content": txt_content,
        "srt_content": srt_content,
        "srt_target_content": srt_target_content,
        "srt_source_content": srt_source_content,
    }
