"""
Build standalone VOICERECORE Python AI Engine sidecar for Windows (.exe) & macOS (.dmg).

Produces a self-contained PyInstaller `--onedir` bundle at:
    src-tauri/resources/engine-bin/voicerecore-engine/voicerecore-engine(.exe)

End users do NOT need Python, uv, or PyTorch installed — Tauri bundles this folder
inside the installer and `src-tauri/src/lib.rs` launches it automatically on startup.

Usage:
    # From repo root (after `cd engine && uv sync` and `uv pip install pyinstaller`):
    uv run --project engine python engine/build_sidecar.py
"""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE_DIR = ROOT / "engine"
OUT_ROOT = ROOT / "src-tauri" / "resources" / "engine-bin"
WORK_DIR = ENGINE_DIR / "build_sidecar_tmp"


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _unify_windows_onnxruntime(target_bundle: Path) -> None:
    """
    Prevent Windows DLL collision between `viet-asr` (ONNX Runtime 1.20.1, API v20)
    and `sherpa-onnx` (ONNX Runtime 1.28.2, API v28). Both ship a DLL named `onnxruntime.dll`.
    Because Windows can only load one `onnxruntime.dll` per process, unify on `sherpa-onnx`'s
    newer backwards-compatible DLL across `_internal/onnxruntime.dll` and
    `_internal/vietasr/_native/onnxruntime.dll`.
    """
    internal = target_bundle / "_internal"
    sherpa_ort = internal / "sherpa_onnx" / "lib" / "onnxruntime.dll"
    if not sherpa_ort.exists():
        print("[build_sidecar] WARNING: sherpa_onnx/lib/onnxruntime.dll not found; skipping ORT unification")
        return

    src_size = sherpa_ort.stat().st_size
    for dst in (
        internal / "onnxruntime.dll",
        internal / "vietasr" / "_native" / "onnxruntime.dll",
    ):
        if dst.exists():
            old_size = dst.stat().st_size
            if old_size != src_size:
                shutil.copy2(sherpa_ort, dst)
                print(
                    f"[build_sidecar] Unified {dst.relative_to(target_bundle)} "
                    f"({old_size:,} -> {src_size:,} bytes, sherpa-onnx ORT)"
                )


def _prune_bundle_bloat(target_bundle: Path, is_win: bool, is_mac: bool) -> None:
    """Safely prune non-runtime headers, static libs, and cross-OS PortAudio binaries."""
    internal = target_bundle / "_internal"
    pruned_bytes = 0

    def _rm(p: Path) -> None:
        nonlocal pruned_bytes
        if not p.exists():
            return
        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file():
                    try:
                        pruned_bytes += f.stat().st_size
                    except OSError:
                        pass
            shutil.rmtree(p, ignore_errors=True)
        else:
            try:
                pruned_bytes += p.stat().st_size
            except OSError:
                pass
            p.unlink(missing_ok=True)

    # C++ headers and static linker libs are never needed at runtime
    for rel in (
        "vietasr/_native/include",
        "vietasr/_native/lib",
        "sherpa_onnx/include",
    ):
        _rm(internal / rel)

    sherpa_lib = internal / "sherpa_onnx" / "lib"
    if sherpa_lib.exists():
        for f in sherpa_lib.glob("*.lib"):
            _rm(f)

    pa_dir = internal / "_sounddevice_data" / "portaudio-binaries"
    if pa_dir.exists():
        _rm(pa_dir / ".github")
        _rm(pa_dir / "README.md")
        if is_mac:
            for f in pa_dir.glob("*.dll"):
                _rm(f)
        elif is_win:
            for f in pa_dir.glob("*.dylib"):
                _rm(f)
            for pat in ("*32bit*.dll", "*arm64*.dll"):
                for f in pa_dir.glob(pat):
                    _rm(f)

    if pruned_bytes > 0:
        print(f"[build_sidecar] Pruned {pruned_bytes / (1024 * 1024):.1f} MB of non-runtime bloat")


def _is_portable_mac_binary(bin_path: Path) -> bool:
    """Return True if a macOS binary does not link /opt/homebrew or /usr/local dynamic libs."""
    try:
        out = subprocess.check_output(["otool", "-L", str(bin_path)], text=True)
        for line in out.splitlines()[1:]:
            dep = line.strip().split(" ", 1)[0]
            if dep.startswith(("/opt/homebrew/", "/usr/local/", "@rpath/")):
                return False
        return True
    except Exception:
        return False


def build() -> Path:
    if not _has_module("PyInstaller"):
        raise SystemExit(
            "PyInstaller chưa được cài trong venv engine.\n"
            "Chạy: uv pip install --python engine/.venv/bin/python pyinstaller  (macOS/Linux)\n"
            "hoặc: uv pip install --python engine/.venv/Scripts/python.exe pyinstaller pyaudiowpatch  (Windows)"
        )

    is_win = platform.system() == "Windows"
    is_mac = platform.system() == "Darwin"

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    target_bundle = OUT_ROOT / "voicerecore-engine"
    if target_bundle.exists():
        shutil.rmtree(target_bundle)
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)

    collect_pkgs = [
        "sherpa_onnx",
        "vietasr",
        "ctranslate2",
        "sounddevice",
        "sentencepiece",
        "segno",
        "multipart",
    ]
    if _has_module("certifi"):
        collect_pkgs.append("certifi")
    if _has_module("_sounddevice_data"):
        collect_pkgs.append("_sounddevice_data")
    if _has_module("imageio_ffmpeg"):
        collect_pkgs.append("imageio_ffmpeg")
    if is_win and _has_module("pyaudiowpatch"):
        collect_pkgs.append("pyaudiowpatch")

    hidden_imports = [
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        "fastapi",
        "starlette",
        "websockets",
        "httpx",
        "certifi",
        "huggingface_hub",
        "numpy",
        "pypdf",
        "python_multipart",
        "asr_engine",
        "audience_web",
        "audio_capture",
        "file_translator",
        "glossary",
        "languages",
        "model_manager",
        "quota_tracker",
        "session_logger",
        "slide_context",
        "translator",
    ]
    if _has_module("imageio_ffmpeg"):
        hidden_imports.append("imageio_ffmpeg")
    if is_win and _has_module("pyaudiowpatch"):
        hidden_imports.append("pyaudiowpatch")

    cmd: list[str] = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--name",
        "voicerecore-engine",
        "--distpath",
        str(OUT_ROOT),
        "--workpath",
        str(WORK_DIR),
        "--specpath",
        str(WORK_DIR),
        "--paths",
        str(ENGINE_DIR),
    ]
    if is_win:
        cmd.append("--noconsole")

    for pkg in collect_pkgs:
        cmd.extend(["--collect-all", pkg])
    for mod in hidden_imports:
        cmd.extend(["--hidden-import", mod])

    cmd.append(str(ENGINE_DIR / "server.py"))

    print("[build_sidecar] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ENGINE_DIR))

    if is_win:
        _unify_windows_onnxruntime(target_bundle)
    _prune_bundle_bloat(target_bundle, is_win=is_win, is_mac=is_mac)

    # Bundle macOS ScreenCaptureKit + Apple Neural Engine Swift helper if compiled
    if is_mac:
        mac_helper = ROOT / "native-macos" / "voicerecore-macos-helper"
        if mac_helper.exists():
            dst = target_bundle / "voicerecore-macos-helper"
            shutil.copy2(mac_helper, dst)
            dst.chmod(0o755)
            print(f"[build_sidecar] Bundled macOS native helper -> {dst}")

    # Bundle school logo next to sidecar so offline QR audience page displays crest logo
    school_logo = ENGINE_DIR / "school_logo.png"
    if school_logo.exists():
        shutil.copy2(school_logo, target_bundle / "school_logo.png")
        print(f"[build_sidecar] Bundled school logo -> {target_bundle / 'school_logo.png'}")

    # Bundle portable llama-server + runtime libs (Mac Metal / Win Vulkan+CPU) staged by scripts/fetch_llama_server.py
    llama_name = "llama-server.exe" if is_win else "llama-server"
    vendor_llama = ENGINE_DIR / "vendor" / "llama-bin"
    if (vendor_llama / llama_name).is_file():
        llama_dst_dir = target_bundle / "llama"
        if llama_dst_dir.exists():
            shutil.rmtree(llama_dst_dir)
        shutil.copytree(vendor_llama, llama_dst_dir)
        if not is_win:
            for p in llama_dst_dir.iterdir():
                if p.is_file() and (p.name == "llama-server" or p.name.endswith(".dylib")):
                    p.chmod(0o755)
        bundled_mb = sum(p.stat().st_size for p in llama_dst_dir.rglob("*") if p.is_file()) / 1048576.0
        print(f"[build_sidecar] Bundled portable llama-server runtime ({bundled_mb:.1f} MB) -> {llama_dst_dir}")
    else:
        # Fallback: copy llama-server from PATH only if self-contained (avoid broken Homebrew dylib links)
        llama_found = shutil.which("llama-server")
        if llama_found:
            llama_src = Path(llama_found).resolve()
            if llama_src.exists():
                if is_mac and not _is_portable_mac_binary(llama_src):
                    print(
                        f"[build_sidecar] Skipping non-portable Homebrew llama-server ({llama_src}) "
                        "to avoid broken /opt/homebrew dylib links on end-user Macs."
                    )
                else:
                    llama_dst = target_bundle / llama_name
                    shutil.copy2(llama_src, llama_dst)
                    if not is_win:
                        llama_dst.chmod(0o755)
                    print(f"[build_sidecar] Bundled llama-server -> {llama_dst}")

    exe_path = target_bundle / ("voicerecore-engine.exe" if is_win else "voicerecore-engine")
    if not exe_path.exists():
        raise RuntimeError(f"Expected sidecar binary not found: {exe_path}")

    print(f"[build_sidecar] SUCCESS: {exe_path}")
    return exe_path


if __name__ == "__main__":
    build()
