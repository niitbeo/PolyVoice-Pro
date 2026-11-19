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
    if _has_module("_sounddevice_data"):
        collect_pkgs.append("_sounddevice_data")
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
        "huggingface_hub",
        "numpy",
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
        "translator",
    ]
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

    # Copy llama-server next to sidecar if present on build machine (enables Hy-MT2-1.8B out of the box)
    llama_name = "llama-server.exe" if is_win else "llama-server"
    llama_found = shutil.which("llama-server")
    if llama_found:
        llama_src = Path(llama_found)
        if llama_src.exists():
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
