"""
Cross-platform real-time audio capture:
1. `MicrophoneStream`: Cross-platform mic capture (Windows WASAPI / macOS CoreAudio) via `sounddevice`.
2. `MacSystemAudioStream`: Native macOS System Audio capture (YouTube / Zoom / Meet) via
   `native-macos/voicerecore-macos-helper --mode system-pcm` (ScreenCaptureKit, zero BlackHole!).
"""

from __future__ import annotations

import platform
import queue
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_MS = 100
BLOCK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)
BYTES_PER_CHUNK = BLOCK_SIZE * 2  # int16 = 2 bytes


@dataclass
class AudioDevice:
    id: int
    name: str
    channels: int
    default_samplerate: float
    is_default: bool


def find_macos_helper() -> Optional[Path]:
    if platform.system() != "Darwin":
        return None
    here = Path(__file__).resolve().parent
    exe_dir = Path(sys.executable).resolve().parent
    candidates = [
        here.parent / "native-macos" / "voicerecore-macos-helper",
        here / "voicerecore-macos-helper",
        exe_dir / "voicerecore-macos-helper",
        exe_dir.parent / "voicerecore-macos-helper",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def list_input_devices() -> list[dict]:
    devices = sd.query_devices()
    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = -1

    result: list[dict] = []
    for idx, dev in enumerate(devices):
        max_in = int(dev.get("max_input_channels", 0))
        if max_in > 0:
            result.append(
                asdict(
                    AudioDevice(
                        id=idx,
                        name=str(dev.get("name", f"Microphone #{idx}")),
                        channels=max_in,
                        default_samplerate=float(dev.get("default_samplerate", 16000.0)),
                        is_default=(idx == default_in),
                    )
                )
            )
    return result


class MicrophoneStream:
    """Streams 16kHz mono int16 audio chunks into a bounded queue with live RMS metering."""

    def __init__(
        self,
        device_id: Optional[int] = None,
        sample_rate: int = SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        on_level: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.on_level = on_level

        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=64)
        self._stream: Optional[sd.InputStream] = None
        self._running = threading.Event()

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if not self._running.is_set():
            return
        mono = indata[:, 0] if indata.ndim > 1 else indata
        rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))
        level = min(1.0, rms * 6.5)
        if self.on_level is not None:
            try:
                self.on_level(level)
            except Exception:
                pass

        pcm_s16 = np.clip(mono * 32767.0, -32768.0, 32767.0).astype(np.int16)
        try:
            self._queue.put_nowait(pcm_s16)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(pcm_s16)
            except queue.Empty:
                pass

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype="float32",
            blocksize=self.block_size,
            device=self.device_id,
            callback=self._callback,
        )
        self._stream.start()

    def read_chunk(self, timeout: float = 0.25) -> Optional[np.ndarray]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._running.clear()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    @property
    def is_running(self) -> bool:
        return self._running.is_set()


class MacSystemAudioStream:
    """
    Captures clean system audio (YouTube, Zoom, Google Meet) on macOS via ScreenCaptureKit
    without requiring BlackHole or virtual audio drivers.
    """

    def __init__(
        self,
        helper_path: Path,
        on_level: Optional[Callable[[float], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.helper_path = Path(helper_path)
        self.on_level = on_level
        self.on_error = on_error
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=64)
        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._running = threading.Event()

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._proc = subprocess.Popen(
            [str(self.helper_path), "--mode", "system-pcm"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True, name="mac-sys-audio")
        self._reader_thread.start()

    def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        buf = bytearray()
        try:
            while self._running.is_set():
                data = self._proc.stdout.read(BYTES_PER_CHUNK - len(buf))
                if not data:
                    break
                buf.extend(data)
                if len(buf) >= BYTES_PER_CHUNK:
                    raw = bytes(buf[:BYTES_PER_CHUNK])
                    del buf[:BYTES_PER_CHUNK]
                    pcm_s16 = np.frombuffer(raw, dtype=np.int16).copy()
                    rms = float(np.sqrt(np.mean(np.square(pcm_s16.astype(np.float32) / 32768.0))))
                    if self.on_level is not None:
                        try:
                            self.on_level(min(1.0, rms * 6.5))
                        except Exception:
                            pass
                    try:
                        self._queue.put_nowait(pcm_s16)
                    except queue.Full:
                        try:
                            self._queue.get_nowait()
                            self._queue.put_nowait(pcm_s16)
                        except queue.Empty:
                            pass
        finally:
            if self._running.is_set() and self._proc is not None and self._proc.stderr is not None:
                err_txt = self._proc.stderr.read().decode("utf-8", errors="ignore").strip()
                if err_txt and self.on_error is not None:
                    self.on_error(
                        f"Chưa cấp quyền thu âm thanh hệ thống trên macOS (Cài đặt → Quyền riêng tư & Bảo mật → Ghi màn hình & Âm thanh hệ thống): {err_txt}"
                    )
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass

    def read_chunk(self, timeout: float = 0.25) -> Optional[np.ndarray]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._running.clear()
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=1.5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass


def has_windows_loopback() -> bool:
    if platform.system() != "Windows":
        return False
    try:
        import pyaudiowpatch  # type: ignore  # noqa: F401
        return True
    except Exception:
        pass
    try:
        for dev in sd.query_devices():
            name = str(dev.get("name", "")).lower()
            if int(dev.get("max_input_channels", 0)) > 0 and any(
                k in name for k in ("stereo mix", "loopback", "what u hear", "cable output")
            ):
                return True
    except Exception:
        pass
    return False


class WindowsLoopbackStream:
    """
    Cross-platform Windows System Audio capture (YouTube / Zoom / Meet):
    1) Prefers `pyaudiowpatch` WASAPI Loopback (zero virtual cable needed on Windows 10/11!).
    2) Falls back to Stereo Mix / Virtual Cable via `sounddevice` if `pyaudiowpatch` is absent.
    Resamples automatically to 16 kHz mono int16 chunks of 100ms.
    """

    def __init__(
        self,
        on_level: Optional[Callable[[float], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.on_level = on_level
        self.on_error = on_error
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=64)
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fallback_mic: Optional[MicrophoneStream] = None

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        try:
            import pyaudiowpatch as pyaudio  # type: ignore

            self._thread = threading.Thread(target=self._wasapi_loop, args=(pyaudio,), daemon=True, name="win-wasapi")
            self._thread.start()
            return
        except Exception:
            pass

        # Fallback: find Stereo Mix / Loopback device in sounddevice
        target_id = None
        for idx, dev in enumerate(sd.query_devices()):
            name = str(dev.get("name", "")).lower()
            if int(dev.get("max_input_channels", 0)) > 0 and any(
                k in name for k in ("stereo mix", "loopback", "what u hear", "cable output")
            ):
                target_id = idx
                break
        if target_id is None:
            self._running.clear()
            raise RuntimeError(
                "Trên Windows, hãy cài `pyaudiowpatch` (có sẵn trong bản .exe chính thức) hoặc bật Stereo Mix trong Windows Sound Settings."
            )
        self._fallback_mic = MicrophoneStream(device_id=target_id, on_level=self.on_level)
        self._fallback_mic.start()

    def _wasapi_loop(self, pyaudio) -> None:
        pa = None
        stream = None
        try:
            pa = pyaudio.PyAudio()
            wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_speakers = pa.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
            if not default_speakers.get("isLoopbackDevice", False):
                for lb in pa.get_loopback_device_info_generator():
                    if default_speakers["name"] in lb["name"]:
                        default_speakers = lb
                        break

            src_rate = int(default_speakers["defaultSampleRate"])
            src_ch = max(1, int(default_speakers["maxInputChannels"]))
            frames_per_100ms = max(160, int(src_rate * CHUNK_MS / 1000))
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=src_ch,
                rate=src_rate,
                frames_per_buffer=frames_per_100ms,
                input=True,
                input_device_index=default_speakers["index"],
            )
            while self._running.is_set():
                raw = stream.read(frames_per_100ms, exception_on_overflow=False)
                arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if src_ch > 1:
                    arr = arr.reshape(-1, src_ch).mean(axis=1)
                if src_rate != SAMPLE_RATE and len(arr) > 0:
                    x_old = np.linspace(0.0, 1.0, len(arr), endpoint=False)
                    x_new = np.linspace(0.0, 1.0, BLOCK_SIZE, endpoint=False)
                    arr = np.interp(x_new, x_old, arr).astype(np.float32)
                rms = float(np.sqrt(np.mean(np.square(arr), dtype=np.float64)))
                if self.on_level is not None:
                    try:
                        self.on_level(min(1.0, rms * 6.5))
                    except Exception:
                        pass
                pcm_s16 = np.clip(arr * 32767.0, -32768.0, 32767.0).astype(np.int16)
                try:
                    self._queue.put_nowait(pcm_s16)
                except queue.Full:
                    try:
                        self._queue.get_nowait()
                        self._queue.put_nowait(pcm_s16)
                    except queue.Empty:
                        pass
        except Exception as exc:
            if self._running.is_set() and self.on_error is not None:
                self.on_error(f"Lỗi thu âm WASAPI Loopback Windows: {exc}")
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            if pa is not None:
                try:
                    pa.terminate()
                except Exception:
                    pass
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass

    def read_chunk(self, timeout: float = 0.25) -> Optional[np.ndarray]:
        if self._fallback_mic is not None:
            return self._fallback_mic.read_chunk(timeout=timeout)
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._running.clear()
        if self._fallback_mic is not None:
            self._fallback_mic.stop()
            self._fallback_mic = None
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

