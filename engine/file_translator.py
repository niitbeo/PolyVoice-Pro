"""
Offline Batch Audio/Video File Translator (.mp3 / .wav / .m4a / .mp4 -> Bilingual .SRT & .TXT).

Uses local `ffmpeg` (or macOS built-in `afconvert` / Python `wave`) to extract 16 kHz mono PCM,
segments speech at natural pauses, runs offline ASR (`VietASR` with punctuation/ITN for Vietnamese,
`Zipformer-20M EN` for English), and translates with VOICERECORE's active offline MT engine
(`Hy-MT2-1.8B`, `VietAI EnViT5`, `NLLB-200-1.3B`).
"""

from __future__ import annotations

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

from session_logger import _fmt_srt_time

SAMPLE_RATE = 16000


class FileTranslateCancelled(RuntimeError):
    """Raised when the user cancels an active batch file translation job."""


def decode_media_to_pcm16(file_path: Path) -> np.ndarray:
    """Decode any audio/video file to 16 kHz mono int16 numpy array."""
    p = Path(file_path)
    # Fast path: native 16kHz mono 16-bit WAV
    if p.suffix.lower() == ".wav":
        try:
            with wave.open(str(p), "rb") as wf:
                if wf.getframerate() == SAMPLE_RATE and wf.getnchannels() == 1 and wf.getsampwidth() == 2:
                    raw = wf.readframes(wf.getnframes())
                    return np.frombuffer(raw, dtype=np.int16).copy()
        except Exception:
            pass

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        proc = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(p),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                "-f",
                "s16le",
                "pipe:1",
            ],
            capture_output=True,
            timeout=300,
        )
        if proc.returncode == 0 and proc.stdout:
            return np.frombuffer(proc.stdout, dtype=np.int16).copy()

    afconvert = shutil.which("afconvert")
    if afconvert:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_wav = Path(tf.name)
        try:
            proc = subprocess.run(
                [afconvert, "-f", "WAVE", "-d", f"LEI16@{SAMPLE_RATE}", "-c", "1", str(p), str(tmp_wav)],
                capture_output=True,
                timeout=180,
            )
            if proc.returncode == 0 and tmp_wav.exists():
                with wave.open(str(tmp_wav), "rb") as wf:
                    raw = wf.readframes(wf.getnframes())
                    return np.frombuffer(raw, dtype=np.int16).copy()
        finally:
            tmp_wav.unlink(missing_ok=True)

    raise RuntimeError(
        "Không thể giải mã định dạng file này. Hãy cài `ffmpeg` (hoặc tải lên file .WAV 16kHz mono)."
    )


def segment_pcm_by_pauses(
    pcm: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    min_seg_s: float = 3.5,
    target_seg_s: float = 11.0,
    max_seg_s: float = 16.0,
) -> list[tuple[float, float, np.ndarray]]:
    """
    Split long audio into natural speech chunks (3.5s–16s) at low-energy pauses so
    ASR punctuation & SRT timestamps align cleanly with human phrasing.
    """
    total_samples = len(pcm)
    if total_samples == 0:
        return []

    win = int(sample_rate * 0.1)  # 100ms frames
    n_frames = total_samples // win
    if n_frames <= int(max_seg_s * 10):
        return [(0.0, total_samples / sample_rate, pcm)]

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
            segments.append((s_idx / sample_rate, e_idx / sample_rate, pcm[s_idx:e_idx]))
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
        segments.append((s_idx / sample_rate, e_idx / sample_rate, pcm[s_idx:e_idx]))
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
) -> dict[str, Any]:
    """Full offline/hybrid pipeline: Media -> PCM -> ASR -> Glossary -> MT -> .TXT & .SRT."""
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

    _emit("decoding", 3.0, "Đang giải mã âm thanh 16kHz...")
    pcm = decode_media_to_pcm16(file_path)
    duration_s = round(len(pcm) / SAMPLE_RATE, 1)
    if duration_s < 0.3:
        raise RuntimeError("File âm thanh quá ngắn hoặc không có dữ liệu tiếng nói.")
    if cancel_event is not None and cancel_event.is_set():
        raise FileTranslateCancelled("Đã hủy dịch file.")

    chunks = segment_pcm_by_pauses(pcm)
    est_speed = 6.0 if mt_mode == "cloud-api" else 12.0
    est_eta_s = max(1.0, round(duration_s / est_speed, 1))
    _emit(
        "loading_models",
        8.0,
        f"Đã tách {len(chunks)} phân đoạn. Đang nạp mô hình AI...",
        total_chunks=len(chunks),
        duration_s=duration_s,
        eta_s=est_eta_s,
        eta_estimated=True,
    )

    use_vi_batch = source_lang == "vie_Latn"
    refiner = runtime.ensure_pass2_refiner() if use_vi_batch else None
    zf_vi = (
        runtime.ensure_onnx_asr_loaded("zipformer-30m-streaming")
        if use_vi_batch and runtime.model_manager.is_zipformer_streaming_ready()
        else None
    )
    en_moonshine = runtime.ensure_en_pass2_refiner() if source_lang == "eng_Latn" else None
    zf_en = (
        runtime.ensure_onnx_asr_loaded("zipformer-en-20m")
        if source_lang == "eng_Latn" and en_moonshine is None
        else None
    )
    if not use_vi_batch and en_moonshine is None and zf_en is None:
        raise RuntimeError("Dịch file offline hiện hỗ trợ file nói Tiếng Việt hoặc Tiếng Anh.")
    if cancel_event is not None and cancel_event.is_set():
        raise FileTranslateCancelled("Đã hủy dịch file.")

    protected_terms = runtime._get_glossary_hotwords()
    segments_out: list[dict[str, Any]] = []
    recent_ctx: list[tuple[str, str]] = []
    processed_audio_s = 0.0
    t_loop = time.perf_counter()

    _emit(
        "translating",
        12.0,
        f"Đang nhận dạng & dịch đoạn 1/{len(chunks)}...",
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

        if use_vi_batch and refiner is not None:
            p1_fallback = ""
            if zf_vi is not None and protected_terms:
                zf_vi.reset()
                p1_parts: list[str] = []
                for pos in range(0, len(seg_pcm), 1600):
                    o = zf_vi.accept_pcm(seg_pcm[pos : pos + 1600])
                    if o.kind == "final" and o.text:
                        p1_parts.append(o.text)
                fl = zf_vi.flush_final()
                if fl.kind == "final" and fl.text:
                    p1_parts.append(fl.text)
                p1_fallback = runtime.glossary.apply_asr_fixes(" ".join(p1_parts).strip())
            text, _ = refiner.refine(seg_pcm, p1_fallback, protected_terms=protected_terms)
        elif en_moonshine is not None:
            text, _ = en_moonshine.refine(seg_pcm, "")
        else:
            assert zf_en is not None
            zf_en.reset()
            parts: list[str] = []
            step = 1600
            for pos in range(0, len(seg_pcm), step):
                o = zf_en.accept_pcm(seg_pcm[pos : pos + step])
                if o.kind == "final" and o.text:
                    parts.append(o.text)
            fl = zf_en.flush_final()
            if fl.kind == "final" and fl.text:
                parts.append(fl.text)
            text = " ".join(parts).strip()

        text = runtime.glossary.apply_asr_fixes(text.strip())
        if text:
            tr = runtime.translate_text(
                text,
                source_lang,
                target_lang,
                mt_mode=mt_mode,
                beam_size=runtime.config.beam_size,
                llm_url=runtime.config.llm_url,
                llm_model=runtime.config.llm_model,
                llm_api_key=runtime.config.llm_api_key,
                context=recent_ctx[-2:],
            )
            recent_ctx.append((text, tr.translated_text))
            segments_out.append(
                {
                    "index": len(segments_out) + 1,
                    "start_s": round(start_s, 2),
                    "end_s": round(end_s, 2),
                    "start_srt": _fmt_srt_time(start_s),
                    "end_srt": _fmt_srt_time(end_s),
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
        "Đang xuất file biên bản .TXT & .SRT song ngữ...",
        current_chunk=len(chunks),
        total_chunks=len(chunks),
        processed_audio_s=duration_s,
        duration_s=duration_s,
        eta_s=0.5,
        speed_x=round(duration_s / max(time.perf_counter() - t_loop, 0.1), 1),
    )

    # Build .TXT and .SRT
    txt_lines = [
        f"=== PolyVoice Pro — Dịch File Offline: {orig_filename} ({duration_s}s) ===",
        f"Hướng dịch: {source_lang} -> {target_lang} | Tổng: {len(segments_out)} đoạn",
        "=" * 64,
        "",
    ]
    srt_lines: list[str] = []
    for seg in segments_out:
        txt_lines.append(f"[{seg['start_srt']} --> {seg['end_srt']}]")
        txt_lines.append(f"Gốc : {seg['source_text']}")
        txt_lines.append(f"Dịch: {seg['translated_text']}")
        txt_lines.append("")

        srt_lines.append(
            f"{seg['index']}\n{seg['start_srt']} --> {seg['end_srt']}\n"
            f"{seg['translated_text']}\n{seg['source_text']}\n"
        )

    txt_content = "\n".join(txt_lines)
    srt_content = "\n".join(srt_lines)

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
        "segment_count": len(segments_out),
        "segments": segments_out,
        "txt_file": str(txt_path),
        "srt_file": str(srt_path),
        "txt_content": txt_content,
        "srt_content": srt_content,
    }
