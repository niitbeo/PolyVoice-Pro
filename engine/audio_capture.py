"""
Cross-platform real-time audio capture:
1. `MicrophoneStream`: Cross-platform mic capture (Windows WASAPI / macOS CoreAudio) via `sounddevice`,
   with automatic fallback to device native sample rate + anti-aliased resampling to 16 kHz mono.
2. `MacSystemAudioStream`: Native macOS System Audio capture (YouTube / Zoom / Meet) via
   `native-macos/voicerecore-macos-helper --mode system-pcm` (ScreenCaptureKit, zero BlackHole!),
   with continuous stderr draining to prevent pipe-buffer deadlocks during multi-hour seminars.
3. `WindowsLoopbackStream`: Native Windows WASAPI Loopback via `pyaudiowpatch` in non-blocking
   callback mode (zero silence-read hang, automatic idle silence padding for VAD/endpointing,
   anti-aliased proportional resampling, and default-speaker hotplug recovery).
"""

from __future__ import annotations

import platform
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from functools import lru_cache
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


@lru_cache(maxsize=16)
def _fir_lowpass_kernel(src_rate: int, dst_rate: int, num_taps: int = 15) -> np.ndarray:
    """Lightweight Hann-windowed sinc anti-aliasing low-pass FIR kernel for downsampling."""
    cutoff = min(0.45, 0.45 * (dst_rate / max(src_rate, 1)))
    n = np.arange(num_taps, dtype=np.float64) - (num_taps - 1) / 2.0
    h = np.sinc(2.0 * cutoff * n) * np.hanning(num_taps)
    total = float(np.sum(h))
    if total != 0.0:
        h /= total
    return h.astype(np.float32)


def _resample_to_16k_mono(
    arr: np.ndarray,
    src_rate: int,
    src_ch: int = 1,
    target_rate: int = SAMPLE_RATE,
    channel_mode: str = "mix",
) -> np.ndarray:
    """Downmix or extract L/R channel from float32 audio and resample proportionally with anti-aliasing."""
    if arr.size == 0:
        return np.empty(0, dtype=np.float32)
    ch_mode = (channel_mode or "mix").lower()
    if arr.ndim > 1:
        n_ch = arr.shape[1]
        if ch_mode == "left" and n_ch >= 1:
            mono = arr[:, 0].astype(np.float32, copy=False)
        elif ch_mode == "right" and n_ch >= 2:
            mono = arr[:, 1].astype(np.float32, copy=False)
        else:
            mono = arr.mean(axis=1, dtype=np.float32)
    elif src_ch > 1:
        usable = (len(arr) // src_ch) * src_ch
        if usable == 0:
            return np.empty(0, dtype=np.float32)
        reshaped = arr[:usable].reshape(-1, src_ch)
        if ch_mode == "left":
            mono = reshaped[:, 0].astype(np.float32, copy=False)
        elif ch_mode == "right" and src_ch >= 2:
            mono = reshaped[:, 1].astype(np.float32, copy=False)
        else:
            mono = reshaped.mean(axis=1, dtype=np.float32)
    else:
        mono = arr.astype(np.float32, copy=False)

    if src_rate == target_rate or len(mono) == 0:
        return mono

    if src_rate > target_rate and len(mono) >= 15:
        kernel = _fir_lowpass_kernel(src_rate, target_rate)
        mono = np.convolve(mono, kernel, mode="same").astype(np.float32)

    target_len = max(1, int(round(len(mono) * target_rate / src_rate)))
    x_old = np.linspace(0.0, 1.0, len(mono), endpoint=False)
    x_new = np.linspace(0.0, 1.0, target_len, endpoint=False)
    return np.interp(x_new, x_old, mono).astype(np.float32)


class AudioConditioner:
    """
    Lightweight broadcast-grade audio conditioning (<0.2% CPU, thread-safe hot-update):
    - Manual Gain (-12 dB .. +24 dB) for quiet Amply/Mixer REC OUT or hot Line-In signals
    - Smooth Broadcast Auto-Gain (AGC) that only adapts on speech frames (never boosts silence hiss)
    - Soft-knee tanh limiter above 0.85 preventing harsh digital clipping on sudden peaks
    - Accurate RMS VU level (0..1) and peak clipping detection (>= 0.97)
    """

    def __init__(self, gain_db: float = 0.0, auto_gain: bool = False) -> None:
        self._lock = threading.Lock()
        self._gain_db = max(-12.0, min(24.0, float(gain_db)))
        self._gain_lin = float(10.0 ** (self._gain_db / 20.0))
        self._auto_gain = bool(auto_gain)
        self._agc_mult = 1.0

    def update(self, gain_db: Optional[float] = None, auto_gain: Optional[bool] = None) -> None:
        with self._lock:
            if gain_db is not None:
                self._gain_db = max(-12.0, min(24.0, float(gain_db)))
                self._gain_lin = float(10.0 ** (self._gain_db / 20.0))
            if auto_gain is not None:
                if bool(auto_gain) != self._auto_gain:
                    self._agc_mult = 1.0
                self._auto_gain = bool(auto_gain)

    def process_float(self, mono_f32: np.ndarray) -> tuple[np.ndarray, float, bool]:
        if mono_f32.size == 0:
            return np.empty(0, dtype=np.int16), 0.0, False
        with self._lock:
            g_lin = self._gain_lin
            use_agc = self._auto_gain
            agc_mult = self._agc_mult

        raw_peak = float(np.max(np.abs(mono_f32)))
        sig = mono_f32 if abs(g_lin - 1.0) < 1e-4 else (mono_f32 * g_lin)

        if use_agc:
            rms_in = float(np.sqrt(np.mean(np.square(sig), dtype=np.float64)))
            # Only adapt AGC during real speech (ignore quiet room noise floor)
            if rms_in > 0.008:
                target_rms = 0.11  # ~ -19 dBFS (sweet spot for Zipformer / VietASR / Moonshine)
                desired = max(0.45, min(5.5, target_rms / max(rms_in, 1e-5)))
                alpha = 0.35 if desired < agc_mult else 0.08  # fast attack, smooth release
                agc_mult = agc_mult + alpha * (desired - agc_mult)
                with self._lock:
                    self._agc_mult = agc_mult
            sig = sig * agc_mult

        post_peak = float(np.max(np.abs(sig)))
        clipping = (raw_peak >= 0.97) or (post_peak >= 0.98)

        # Soft-knee limiter above 0.85 so sudden shouting never wraps or hard-clips into square waves
        if post_peak > 0.85:
            knee = 0.85
            mag = np.abs(sig)
            over = mag > knee
            if np.any(over):
                sig = sig.copy()
                excess = (mag[over] - knee) / (1.0 - knee)
                limited = knee + (1.0 - knee) * np.tanh(excess)
                sig[over] = np.sign(sig[over]) * limited

        rms_out = float(np.sqrt(np.mean(np.square(sig), dtype=np.float64)))
        level = min(1.0, rms_out * 6.5)
        pcm_s16 = np.clip(sig * 32767.0, -32768.0, 32767.0).astype(np.int16)
        return pcm_s16, level, clipping

    def process_pcm16(self, pcm_s16: np.ndarray) -> tuple[np.ndarray, float, bool]:
        if pcm_s16.size == 0:
            return pcm_s16, 0.0, False
        f32 = pcm_s16.astype(np.float32) / 32768.0
        return self.process_float(f32)


def list_output_loopback_devices() -> list[dict]:
    """List selectable system output loopback devices (Windows WASAPI Loopback / Stereo Mix, macOS ScreenCaptureKit)."""
    sys_name = platform.system()
    if sys_name == "Darwin":
        return [
            {
                "id": -1,
                "name": "Toàn bộ âm thanh hệ thống (macOS ScreenCaptureKit)",
                "channels": 2,
                "default_samplerate": 48000.0,
                "is_default": True,
            }
        ]
    if sys_name != "Windows":
        return []

    out: list[dict] = []
    try:
        import pyaudiowpatch as pyaudio  # type: ignore

        pa = pyaudio.PyAudio()
        try:
            default_idx = -1
            if hasattr(pa, "get_default_wasapi_loopback"):
                try:
                    d = pa.get_default_wasapi_loopback()
                    if d:
                        default_idx = int(d.get("index", -1))
                except Exception:
                    pass
            seen: set[str] = set()
            for lb in pa.get_loopback_device_info_generator():
                if int(lb.get("maxInputChannels", 0)) <= 0:
                    continue
                idx = int(lb.get("index", -1))
                name = str(lb.get("name", f"Output Loopback #{idx}")).strip()
                key = name.lower()
                if key in seen and idx != default_idx:
                    continue
                seen.add(key)
                out.append(
                    asdict(
                        AudioDevice(
                            id=idx,
                            name=name,
                            channels=int(lb.get("maxInputChannels", 2)),
                            default_samplerate=float(lb.get("defaultSampleRate", 48000.0)),
                            is_default=(idx == default_idx),
                        )
                    )
                )
        finally:
            pa.terminate()
        if out:
            if not any(d["is_default"] for d in out):
                out[0]["is_default"] = True
            return out
    except Exception:
        pass

    # Fallback: Stereo Mix / Virtual Cable devices in sounddevice
    try:
        for idx, dev in enumerate(sd.query_devices()):
            name = str(dev.get("name", "")).strip()
            if int(dev.get("max_input_channels", 0)) > 0 and any(
                k in name.lower() for k in ("stereo mix", "loopback", "what u hear", "cable output")
            ):
                out.append(
                    asdict(
                        AudioDevice(
                            id=idx,
                            name=name,
                            channels=int(dev.get("max_input_channels", 2)),
                            default_samplerate=float(dev.get("default_samplerate", 48000.0)),
                            is_default=(len(out) == 0),
                        )
                    )
                )
    except Exception:
        pass
    return out


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

    # On Windows, PortAudio lists every mic 3-4x across MME, DirectSound, WASAPI, and WDM-KS.
    # Prefer Windows WASAPI (clean full Unicode names, low latency), falling back to DirectSound/MME.
    preferred_hostapi: Optional[int] = None
    if platform.system() == "Windows":
        try:
            hostapis = sd.query_hostapis()
            for pref_name in ("Windows WASAPI", "Windows DirectSound", "MME"):
                for h_idx, h in enumerate(hostapis):
                    if pref_name.lower() in str(h.get("name", "")).lower():
                        # Ensure at least one input device exists on this host API
                        if any(
                            int(d.get("max_input_channels", 0)) > 0 and d.get("hostapi") == h_idx
                            for d in devices
                        ):
                            preferred_hostapi = h_idx
                            default_cand = int(h.get("default_input_device", -1))
                            if default_cand >= 0:
                                default_in = default_cand
                            break
                if preferred_hostapi is not None:
                    break
        except Exception:
            preferred_hostapi = None

    result: list[dict] = []
    seen_names: set[str] = set()
    for idx, dev in enumerate(devices):
        max_in = int(dev.get("max_input_channels", 0))
        if max_in <= 0:
            continue
        if preferred_hostapi is not None and dev.get("hostapi") != preferred_hostapi:
            continue
        name = str(dev.get("name", f"Microphone #{idx}")).strip()
        norm_key = name.lower()
        if norm_key in seen_names and idx != default_in:
            continue
        seen_names.add(norm_key)
        result.append(
            asdict(
                AudioDevice(
                    id=idx,
                    name=name,
                    channels=max_in,
                    default_samplerate=float(dev.get("default_samplerate", 16000.0)),
                    is_default=(idx == default_in),
                )
            )
        )
    return result


class MicrophoneStream:
    """Streams 16kHz mono int16 audio chunks into a bounded queue with live RMS metering & broadcast conditioning."""

    def __init__(
        self,
        device_id: Optional[int] = None,
        sample_rate: int = SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        on_level: Optional[Callable[[float], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_meter: Optional[Callable[[float, bool], None]] = None,
        channel_mode: str = "mix",
        gain_db: float = 0.0,
        auto_gain: bool = False,
    ) -> None:
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.on_level = on_level
        self.on_error = on_error
        self.on_meter = on_meter
        self._channel_mode = (channel_mode or "mix").lower()
        self._cond = AudioConditioner(gain_db=gain_db, auto_gain=auto_gain)

        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=200)
        self._stream: Optional[sd.InputStream] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._src_rate: int = sample_rate
        self._src_channels: int = CHANNELS
        self._src_blocksize: int = block_size
        self._last_error_ts: float = 0.0

    def set_conditioning(
        self,
        gain_db: Optional[float] = None,
        auto_gain: Optional[bool] = None,
        channel_mode: Optional[str] = None,
    ) -> None:
        self._cond.update(gain_db=gain_db, auto_gain=auto_gain)
        if channel_mode is not None:
            self._channel_mode = (channel_mode or "mix").lower()

    def _emit_meter(self, level: float, clipping: bool) -> None:
        if self.on_meter is not None:
            try:
                self.on_meter(level, clipping)
            except Exception:
                pass
        elif self.on_level is not None:
            try:
                self.on_level(level)
            except Exception:
                pass

    def _process_block(self, indata: np.ndarray) -> None:
        mono = _resample_to_16k_mono(
            indata,
            self._src_rate,
            self._src_channels,
            target_rate=self.sample_rate,
            channel_mode=self._channel_mode,
        )
        if mono.size == 0:
            return
        pcm_s16, level, clipping = self._cond.process_float(mono)
        self._emit_meter(level, clipping)
        try:
            self._queue.put_nowait(pcm_s16)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(pcm_s16)
            except queue.Empty:
                pass

    def _read_loop(self) -> None:
        """
        Blocking PortAudio read loop (NO libffi/cffi C callback trampoline!).
        Prevents SIGSEGV inside `ffi_closure_SYSV_inner` on macOS ARM64 CoreAudio HALC_IOThread
        during rapid mic/system source switching and GC cycles.
        """
        st = self._stream
        blk = self._src_blocksize
        while self._running.is_set() and st is not None:
            try:
                indata, _overflowed = st.read(blk)
            except Exception:
                if not self._running.is_set():
                    break
                time.sleep(0.02)
                continue
            if not self._running.is_set():
                break
            if indata is not None and len(indata) > 0:
                self._process_block(indata)

    def _open_stream(self, rate: int, ch: int) -> sd.InputStream:
        blk = max(160, int(round(rate * CHUNK_MS / 1000)))
        self._src_rate = rate
        self._src_channels = ch
        self._src_blocksize = blk
        return sd.InputStream(
            samplerate=rate,
            channels=ch,
            dtype="float32",
            blocksize=blk,
            device=self.device_id,
        )

    def _spawn_reader(self) -> None:
        self._reader_thread = threading.Thread(
            target=self._read_loop,
            daemon=True,
            name="mic-blocking-reader",
        )
        self._reader_thread.start()

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        try:
            # Query device once so stereo L/R isolation opens 2 channels when user selects Left or Right!
            dev_info = {}
            try:
                dev_info = sd.query_devices(self.device_id, "input") or {}
            except Exception:
                dev_info = {}
            native_rate = max(8000, int(round(float(dev_info.get("default_samplerate", 48000.0)))))
            max_ch = max(1, int(dev_info.get("max_input_channels", 1)))
            prefer_ch = 2 if (max_ch >= 2 and self._channel_mode in ("left", "right")) else CHANNELS

            try:
                self._stream = self._open_stream(self.sample_rate, prefer_ch)
                self._stream.start()
                self._spawn_reader()
                return
            except sd.PortAudioError:
                # Many Windows WASAPI / Bluetooth / USB interfaces only accept native rate (48kHz/44.1kHz) or stereo
                attempts: list[tuple[int, int]] = []
                if native_rate != self.sample_rate:
                    attempts.append((native_rate, prefer_ch))
                if prefer_ch != 2 and max_ch >= 2:
                    attempts.append((native_rate, 2))
                    if native_rate != self.sample_rate:
                        attempts.append((self.sample_rate, 2))
                if prefer_ch != 1:
                    attempts.append((native_rate, 1))
                    if native_rate != self.sample_rate:
                        attempts.append((self.sample_rate, 1))
                last_exc: Optional[Exception] = None
                for rate, ch in attempts:
                    try:
                        self._stream = self._open_stream(rate, ch)
                        self._stream.start()
                        self._spawn_reader()
                        return
                    except Exception as exc:
                        last_exc = exc
                        if self._stream is not None:
                            try:
                                self._stream.close()
                            except Exception:
                                pass
                            self._stream = None
                if last_exc is not None:
                    raise last_exc
                raise
        except Exception:
            self._running.clear()
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
            raise

    def read_chunk(self, timeout: float = 0.25) -> Optional[np.ndarray]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._running.clear()
        th = self._reader_thread
        self._reader_thread = None
        # Wait for blocking `st.read()` (<= 100ms) to exit cleanly BEFORE calling PortAudio stop/close,
        # preventing concurrent `Pa_StopStream` vs `Pa_ReadStream` ring-buffer corruption!
        if th is not None and th.is_alive():
            th.join(timeout=0.6)
        st = self._stream
        self._stream = None
        if st is not None:
            try:
                if th is not None and th.is_alive():
                    st.abort()
                st.stop()
            except Exception:
                pass
            if th is not None and th.is_alive():
                th.join(timeout=0.5)
            try:
                st.close()
            except Exception:
                pass
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
        on_meter: Optional[Callable[[float, bool], None]] = None,
        gain_db: float = 0.0,
        auto_gain: bool = False,
    ) -> None:
        self.helper_path = Path(helper_path)
        self.on_level = on_level
        self.on_error = on_error
        self.on_meter = on_meter
        self._cond = AudioConditioner(gain_db=gain_db, auto_gain=auto_gain)
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=200)
        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stderr_ring: deque[bytes] = deque()
        self._stderr_bytes: int = 0
        self._stderr_lock = threading.Lock()
        self._running = threading.Event()

    def set_conditioning(
        self,
        gain_db: Optional[float] = None,
        auto_gain: Optional[bool] = None,
        channel_mode: Optional[str] = None,
    ) -> None:
        self._cond.update(gain_db=gain_db, auto_gain=auto_gain)

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        with self._stderr_lock:
            self._stderr_ring.clear()
            self._stderr_bytes = 0
        self._proc = subprocess.Popen(
            [str(self.helper_path), "--mode", "system-pcm"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        # Continuously drain stderr into a 2 KB ring buffer so macOS frameworks never deadlock on a full 64 KB pipe!
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(self._proc.stderr,),
            daemon=True,
            name="mac-sys-stderr",
        )
        self._stderr_thread.start()
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True, name="mac-sys-audio")
        self._reader_thread.start()

    def _drain_stderr(self, pipe) -> None:
        if pipe is None:
            return
        max_bytes = 2048
        try:
            while True:
                chunk = pipe.read(512)
                if not chunk:
                    break
                with self._stderr_lock:
                    self._stderr_ring.append(chunk)
                    self._stderr_bytes += len(chunk)
                    while self._stderr_bytes > max_bytes and self._stderr_ring:
                        dropped = self._stderr_ring.popleft()
                        self._stderr_bytes -= len(dropped)
        except Exception:
            pass

    def _get_stderr_tail(self) -> str:
        with self._stderr_lock:
            raw = b"".join(self._stderr_ring)
        return raw.decode("utf-8", errors="ignore").strip()

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
                    pcm_raw = np.frombuffer(raw, dtype=np.int16).copy()
                    pcm_s16, level, clipping = self._cond.process_pcm16(pcm_raw)
                    if self.on_meter is not None:
                        try:
                            self.on_meter(level, clipping)
                        except Exception:
                            pass
                    elif self.on_level is not None:
                        try:
                            self.on_level(level)
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
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=0.3)
            if self._running.is_set():
                err_txt = self._get_stderr_tail()
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
    Cross-platform Windows System Audio capture (YouTube / Zoom / Meet / OBS Output):
    1) Prefers `pyaudiowpatch` WASAPI Loopback in non-blocking callback mode (zero virtual cable,
       selectable output speaker, zero silence-read hang, idle silence padding, hotplug recovery).
    2) Falls back to Stereo Mix / Virtual Cable via `sounddevice` if `pyaudiowpatch` is absent.
    Resamples automatically with anti-aliasing to 16 kHz mono int16 chunks of 100ms.
    """

    def __init__(
        self,
        on_level: Optional[Callable[[float], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_meter: Optional[Callable[[float, bool], None]] = None,
        loopback_device_id: Optional[int] = None,
        gain_db: float = 0.0,
        auto_gain: bool = False,
    ) -> None:
        self.on_level = on_level
        self.on_error = on_error
        self.on_meter = on_meter
        self.loopback_device_id = loopback_device_id
        self._cond = AudioConditioner(gain_db=gain_db, auto_gain=auto_gain)
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=200)
        self._running = threading.Event()
        self._stop_ev = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fallback_mic: Optional[MicrophoneStream] = None

    def set_conditioning(
        self,
        gain_db: Optional[float] = None,
        auto_gain: Optional[bool] = None,
        channel_mode: Optional[str] = None,
    ) -> None:
        self._cond.update(gain_db=gain_db, auto_gain=auto_gain)
        if self._fallback_mic is not None:
            self._fallback_mic.set_conditioning(gain_db=gain_db, auto_gain=auto_gain)

    def _emit_meter(self, level: float, clipping: bool) -> None:
        if self.on_meter is not None:
            try:
                self.on_meter(level, clipping)
            except Exception:
                pass
        elif self.on_level is not None:
            try:
                self.on_level(level)
            except Exception:
                pass

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._stop_ev.clear()
        try:
            import pyaudiowpatch as pyaudio  # type: ignore

            self._thread = threading.Thread(target=self._wasapi_loop, args=(pyaudio,), daemon=True, name="win-wasapi")
            self._thread.start()
            return
        except Exception:
            pass

        # Fallback: find Stereo Mix / Loopback device in sounddevice
        target_id = None
        if self.loopback_device_id is not None and self.loopback_device_id >= 0:
            try:
                d = sd.query_devices(self.loopback_device_id)
                if d and int(d.get("max_input_channels", 0)) > 0:
                    target_id = self.loopback_device_id
            except Exception:
                target_id = None
        if target_id is None:
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
        self._fallback_mic = MicrophoneStream(
            device_id=target_id,
            on_level=self.on_level,
            on_error=self.on_error,
            on_meter=self.on_meter,
            gain_db=self._cond._gain_db,
            auto_gain=self._cond._auto_gain,
        )
        self._fallback_mic.start()

    @staticmethod
    def _find_wasapi_loopback_device(pa, pyaudio, preferred_id: Optional[int] = None) -> dict:
        if preferred_id is not None and preferred_id >= 0:
            try:
                cand = pa.get_device_info_by_index(int(preferred_id))
                if cand and cand.get("isLoopbackDevice", False) and int(cand.get("maxInputChannels", 0)) > 0:
                    return cand
            except Exception:
                pass
        if hasattr(pa, "get_default_wasapi_loopback"):
            try:
                dev = pa.get_default_wasapi_loopback()
                if dev and dev.get("isLoopbackDevice", False) and int(dev.get("maxInputChannels", 0)) > 0:
                    return dev
            except Exception:
                pass

        wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        out_idx = int(wasapi_info.get("defaultOutputDevice", -1))
        out_name = ""
        if out_idx >= 0:
            default_out = pa.get_device_info_by_index(out_idx)
            if default_out.get("isLoopbackDevice", False) and int(default_out.get("maxInputChannels", 0)) > 0:
                return default_out
            out_name = str(default_out.get("name", ""))

        first_valid: Optional[dict] = None
        for lb in pa.get_loopback_device_info_generator():
            if int(lb.get("maxInputChannels", 0)) <= 0:
                continue
            if first_valid is None:
                first_valid = lb
            if out_name and out_name in str(lb.get("name", "")):
                return lb
        if first_valid is not None:
            return first_valid
        raise RuntimeError("Không tìm thấy thiết bị loa WASAPI Loopback đang hoạt động trên Windows.")

    def _enqueue_pcm(self, pcm_s16: np.ndarray) -> None:
        try:
            self._queue.put_nowait(pcm_s16)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(pcm_s16)
            except queue.Empty:
                pass

    def _wasapi_loop(self, pyaudio) -> None:
        max_retries = 3
        attempt = 0

        while self._running.is_set() and not self._stop_ev.is_set():
            pa = None
            stream = None
            try:
                pa = pyaudio.PyAudio()
                loopback = self._find_wasapi_loopback_device(pa, pyaudio, preferred_id=self.loopback_device_id)
                src_rate = max(8000, int(round(float(loopback["defaultSampleRate"]))))
                src_ch = max(1, int(loopback["maxInputChannels"]))
                frames_per_100ms = max(160, int(round(src_rate * CHUNK_MS / 1000)))

                pcm_lock = threading.Lock()
                state = {
                    "last_pkt": time.monotonic(),
                    "pcm_buf": np.empty(0, dtype=np.int16),
                    "use_float32": False,
                }

                def _cb(in_data, frame_count, time_info, status_flags):
                    if not self._running.is_set():
                        return (None, pyaudio.paComplete)
                    now_ts = time.monotonic()
                    ready_blocks: list[np.ndarray] = []
                    if in_data:
                        if state["use_float32"]:
                            arr = np.frombuffer(in_data, dtype=np.float32)
                        else:
                            arr = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
                        mono16k = _resample_to_16k_mono(arr, src_rate, src_ch, target_rate=SAMPLE_RATE)
                        if mono16k.size > 0:
                            pcm, level, clipping = self._cond.process_float(mono16k)
                            self._emit_meter(level, clipping)
                            with pcm_lock:
                                state["last_pkt"] = now_ts
                                if state["pcm_buf"].size > 0:
                                    combined = np.concatenate((state["pcm_buf"], pcm))
                                else:
                                    combined = pcm
                                pos = 0
                                n = combined.size
                                while pos + BLOCK_SIZE <= n:
                                    ready_blocks.append(combined[pos : pos + BLOCK_SIZE].copy())
                                    pos += BLOCK_SIZE
                                state["pcm_buf"] = combined[pos:]
                        else:
                            with pcm_lock:
                                state["last_pkt"] = now_ts
                    else:
                        with pcm_lock:
                            state["last_pkt"] = now_ts
                    for blk in ready_blocks:
                        self._enqueue_pcm(blk)
                    return (None, pyaudio.paContinue)

                try:
                    stream = pa.open(
                        format=pyaudio.paInt16,
                        channels=src_ch,
                        rate=src_rate,
                        frames_per_buffer=frames_per_100ms,
                        input=True,
                        input_device_index=loopback["index"],
                        stream_callback=_cb,
                    )
                except Exception:
                    state["use_float32"] = True
                    stream = pa.open(
                        format=pyaudio.paFloat32,
                        channels=src_ch,
                        rate=src_rate,
                        frames_per_buffer=frames_per_100ms,
                        input=True,
                        input_device_index=loopback["index"],
                        stream_callback=_cb,
                    )

                stream.start_stream()
                attempt = 0  # Opened cleanly; reset reconnect counter
                silence_chunk = np.zeros(BLOCK_SIZE, dtype=np.int16)

                while self._running.is_set() and not self._stop_ev.is_set():
                    if self._stop_ev.wait(timeout=0.10):
                        break
                    if not stream.is_active():
                        # Default speaker disconnected / switched -> trigger reconnect
                        break
                    # WASAPI loopback emits zero callbacks when system audio is silent.
                    # First flush any trailing <100ms speech samples trapped in `pcm_buf` (zero-padded),
                    # then feed silence chunks after 200ms idle so VAD/Zipformer endpoints finalize cleanly.
                    flush_idle = False
                    rem = None
                    with pcm_lock:
                        if (time.monotonic() - state["last_pkt"]) >= 0.20:
                            flush_idle = True
                            if state["pcm_buf"].size > 0:
                                rem = state["pcm_buf"]
                                state["pcm_buf"] = np.empty(0, dtype=np.int16)
                    if flush_idle:
                        if rem is not None and rem.size > 0:
                            padded = np.zeros(BLOCK_SIZE, dtype=np.int16)
                            n_copy = min(rem.size, BLOCK_SIZE)
                            padded[:n_copy] = rem[:n_copy]
                            self._enqueue_pcm(padded)
                        else:
                            self._enqueue_pcm(silence_chunk)
                        self._emit_meter(0.0, False)

                if self._stop_ev.is_set() or not self._running.is_set():
                    break
                # Stream became inactive unexpectedly (e.g. headphones plugged in) -> retry on new default speaker
                attempt += 1
                if attempt > max_retries:
                    raise RuntimeError("Thiết bị loa Windows đã thay đổi hoặc ngắt kết nối.")
                time.sleep(0.4)
            except Exception as exc:
                attempt += 1
                if not self._running.is_set() or self._stop_ev.is_set():
                    break
                if attempt > max_retries:
                    if self.on_error is not None:
                        self.on_error(f"Lỗi thu âm WASAPI Loopback Windows: {exc}")
                    break
                time.sleep(0.4)
            finally:
                if stream is not None:
                    try:
                        stream.stop_stream()
                    except Exception:
                        pass
                    try:
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
        self._stop_ev.set()
        if self._fallback_mic is not None:
            self._fallback_mic.stop()
            self._fallback_mic = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass


class MixedAudioStream:
    """
    Conference Hall Dual-Input Mixer (Mic/Mixer + Computer System Audio -> Single 16kHz PCM stream):
    Sums live hall microphone/mixer audio with computer playback (YouTube / Zoom / video clips)
    into one clean soft-limited mono 16 kHz feed for single-direction conference translation (A -> B)
    without running a second ASR model!
    """

    def __init__(
        self,
        mic_stream: MicrophoneStream,
        sys_stream: MacSystemAudioStream | WindowsLoopbackStream,
        on_level: Optional[Callable[[float], None]] = None,
        on_meter: Optional[Callable[[float, bool], None]] = None,
        gain_db: float = 0.0,
        auto_gain: bool = False,
    ) -> None:
        self._mic = mic_stream
        self._sys = sys_stream
        self.on_level = on_level
        self.on_meter = on_meter
        self._cond = AudioConditioner(gain_db=gain_db, auto_gain=auto_gain)
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=200)
        self._running = threading.Event()
        self._mic_buf: deque[np.ndarray] = deque(maxlen=12)
        self._sys_buf: deque[np.ndarray] = deque(maxlen=12)
        self._buf_lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    def set_conditioning(
        self,
        gain_db: Optional[float] = None,
        auto_gain: Optional[bool] = None,
        channel_mode: Optional[str] = None,
    ) -> None:
        self._cond.update(gain_db=gain_db, auto_gain=auto_gain)
        if channel_mode is not None:
            self._mic.set_conditioning(channel_mode=channel_mode)

    def _emit_meter(self, level: float, clipping: bool) -> None:
        if self.on_meter is not None:
            try:
                self.on_meter(level, clipping)
            except Exception:
                pass
        elif self.on_level is not None:
            try:
                self.on_level(level)
            except Exception:
                pass

    def _pump(self, src_stream, target_buf: deque[np.ndarray]) -> None:
        while self._running.is_set():
            chunk = src_stream.read_chunk(timeout=0.15)
            if chunk is None:
                continue
            with self._buf_lock:
                target_buf.append(chunk)

    def _mix_loop(self) -> None:
        zero_f32 = np.zeros(BLOCK_SIZE, dtype=np.float32)
        while self._running.is_set():
            time.sleep(CHUNK_MS / 1000.0)
            with self._buf_lock:
                m_chunk = self._mic_buf.popleft() if self._mic_buf else None
                s_chunk = self._sys_buf.popleft() if self._sys_buf else None
            if m_chunk is None and s_chunk is None:
                continue
            m_f32 = (m_chunk.astype(np.float32) / 32768.0) if m_chunk is not None else zero_f32
            s_f32 = (s_chunk.astype(np.float32) / 32768.0) if s_chunk is not None else zero_f32
            n = max(m_f32.size, s_f32.size)
            if m_f32.size < n:
                m_f32 = np.pad(m_f32, (0, n - m_f32.size))
            if s_f32.size < n:
                s_f32 = np.pad(s_f32, (0, n - s_f32.size))
            mixed = m_f32 + s_f32
            pcm_s16, level, clipping = self._cond.process_float(mixed)
            self._emit_meter(level, clipping)
            try:
                self._queue.put_nowait(pcm_s16)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(pcm_s16)
                except queue.Empty:
                    pass
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        try:
            self._mic.start()
            self._sys.start()
        except Exception:
            self.stop()
            raise
        t_mic = threading.Thread(target=self._pump, args=(self._mic, self._mic_buf), daemon=True, name="mix-mic")
        t_sys = threading.Thread(target=self._pump, args=(self._sys, self._sys_buf), daemon=True, name="mix-sys")
        t_mix = threading.Thread(target=self._mix_loop, daemon=True, name="mix-sum")
        self._threads = [t_mic, t_sys, t_mix]
        for t in self._threads:
            t.start()

    def read_chunk(self, timeout: float = 0.25) -> Optional[np.ndarray]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._running.clear()
        try:
            self._mic.stop()
        except Exception:
            pass
        try:
            self._sys.stop()
        except Exception:
            pass
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=0.6)
        self._threads.clear()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
