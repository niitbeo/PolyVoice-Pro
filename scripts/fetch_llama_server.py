#!/usr/bin/env python3
"""
Download & prune official portable `llama-server` binaries from `ggml-org/llama.cpp`
into `engine/vendor/llama-bin/` for bundling inside PolyVoice Pro installers:
- macOS arm64: Apple Silicon Metal GPU runtime (~30 MB uncompressed, ~11 MB gzipped, zero Homebrew links, zero symlinks)
- Windows x64: Vulkan GPU + AVX2/SSE4.2 CPU multi-ISA runtime (auto-accelerates Intel/AMD/NVIDIA GPU with zero-crash CPU fallback)

Usage:
    python3 scripts/fetch_llama_server.py
"""

from __future__ import annotations

import fnmatch
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

LLAMA_TAG = "b12556"
MAC_ARM64_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/llama-{LLAMA_TAG}-bin-macos-arm64.tar.gz"
WIN_X64_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/llama-{LLAMA_TAG}-bin-win-vulkan-x64.zip"

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "engine" / "vendor" / "llama-bin"

MAC_REQUIRED_NAMES = (
    "llama-server",
    "libllama-server-impl.dylib",
    "libllama-common.1.dylib",
    "libmtmd.1.dylib",
    "libllama.1.dylib",
    "libggml.1.dylib",
    "libggml-base.1.dylib",
    "libggml-cpu.1.dylib",
    "libggml-blas.1.dylib",
    "libggml-metal.1.dylib",
    "libggml-rpc.1.dylib",
    "LICENSE",
)

WIN_KEEP_PATTERNS = (
    "llama-server.exe",
    "llama-server-impl.dll",
    "llama-common.dll",
    "llama.dll",
    "mtmd.dll",
    "libomp.dll",
    "ggml.dll",
    "ggml-base.dll",
    "ggml-rpc.dll",
    "ggml-vulkan.dll",
    "ggml-cpu-*.dll",
    "LICENSE*",
)


def _download(url: str, dst: Path) -> None:
    print(f"[fetch_llama_server] Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "PolyVoicePro-CI/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, dst.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1024 * 1024)
    print(f"[fetch_llama_server] Downloaded {dst.stat().st_size / 1048576:.1f} MB")


def _stage_mac(tmp: Path) -> Path:
    archive = tmp / "llama-mac.tar.gz"
    _download(MAC_ARM64_URL, archive)
    extract_dir = tmp / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(extract_dir)

    # Locate folder containing `llama-server`
    found_bins = list(extract_dir.rglob("llama-server"))
    if not found_bins:
        raise RuntimeError("llama-server not found in macOS archive")
    src_dir = found_bins[0].parent

    # Discover any additional @rpath/*.dylib referenced by llama-server directly
    wanted = set(MAC_REQUIRED_NAMES)
    try:
        ot = subprocess.check_output(["otool", "-L", str(src_dir / "llama-server")], text=True)
        for line in ot.splitlines()[1:]:
            dep = line.strip().split(" ", 1)[0]
            if dep.startswith("@rpath/"):
                wanted.add(dep[len("@rpath/") :])
    except Exception:
        pass

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in sorted(wanted):
        src = src_dir / name
        if not src.exists():
            if name == "LICENSE":
                continue
            raise RuntimeError(f"Expected macOS llama runtime file missing: {src}")
        # Materialize real files (dereference symlinks) so Tauri bundler & codesign stay 100% symlink-safe
        real_src = src.resolve()
        dst = OUT_DIR / name
        shutil.copy2(real_src, dst)
        if name == "llama-server" or name.endswith(".dylib"):
            dst.chmod(0o755)

    return OUT_DIR / "llama-server"


def _stage_win(tmp: Path) -> Path:
    archive = tmp / "llama-win.zip"
    _download(WIN_X64_URL, archive)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            base = Path(info.filename).name
            if any(fnmatch.fnmatch(base, pat) for pat in WIN_KEEP_PATTERNS):
                dst = OUT_DIR / base
                with zf.open(info, "r") as src_f, dst.open("wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f, length=1024 * 1024)

    exe = OUT_DIR / "llama-server.exe"
    if not exe.exists():
        raise RuntimeError("llama-server.exe not found after extracting Windows archive")
    return exe


def main() -> None:
    sys_name = platform.system()
    arch = platform.machine().lower()

    with tempfile.TemporaryDirectory(prefix="pv_fetch_llama_") as td:
        tmp = Path(td)
        if sys_name == "Darwin" and arch == "arm64":
            bin_path = _stage_mac(tmp)
        elif sys_name == "Windows":
            bin_path = _stage_win(tmp)
        else:
            print(f"[fetch_llama_server] Skipping unsupported host platform: {sys_name} {arch}")
            return

    total_mb = sum(p.stat().st_size for p in OUT_DIR.rglob("*") if p.is_file()) / 1048576.0
    print(f"[fetch_llama_server] Staged {len(list(OUT_DIR.iterdir()))} files ({total_mb:.1f} MB) -> {OUT_DIR}")

    # Smoke-verify binary launches cleanly from isolated vendor directory
    res = subprocess.run(
        [str(bin_path), "--version"],
        cwd=str(OUT_DIR),
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    ver_out = (res.stdout or res.stderr or "").strip().splitlines()
    print("[fetch_llama_server] Verified:", " | ".join(ver_out[:2]))


if __name__ == "__main__":
    main()
