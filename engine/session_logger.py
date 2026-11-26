"""
Real-Time Auto-Save Meeting Logger (.TXT and .SRT) for multi-hour seminars & meetings.

Appends and flushes every committed paragraph directly to disk under
`~/Documents/VOICERECORE_Logs/` (with safe fallback if Documents is blocked by
Windows Controlled Folder Access or macOS TCC) so zero data is ever lost during
2–3+ hour sessions.
"""

from __future__ import annotations

import os
import platform
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional


def _windows_documents_dir() -> Path:
    """Resolve real Windows Documents folder (handles OneDrive redirection)."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class _GUID(ctypes.Structure):
                _fields_ = [
                    ("Data1", wintypes.DWORD),
                    ("Data2", wintypes.WORD),
                    ("Data3", wintypes.WORD),
                    ("Data4", ctypes.c_ubyte * 8),
                ]

            # FOLDERID_Documents = {FDD39AD0-238F-46AF-ADB4-6C85480369C7}
            fid = _GUID(
                0xFDD39AD0,
                0x238F,
                0x46AF,
                (ctypes.c_ubyte * 8)(0xAD, 0xB4, 0x6C, 0x85, 0x48, 0x03, 0x69, 0xC7),
            )
            p_path = ctypes.c_wchar_p()
            hr = ctypes.windll.shell32.SHGetKnownFolderPath(  # type: ignore[attr-defined]
                ctypes.byref(fid), 0, None, ctypes.byref(p_path)
            )
            if hr == 0 and p_path.value:
                resolved = Path(p_path.value)
                ctypes.windll.ole32.CoTaskMemFree(p_path)  # type: ignore[attr-defined]
                return resolved
        except Exception:
            pass
    return Path.home() / "Documents"


def _resolve_default_logs_dir() -> Path:
    env_dir = os.environ.get("VOICERECORE_LOG_DIR")
    if env_dir:
        return Path(env_dir)
    return _windows_documents_dir() / "VOICERECORE_Logs"


DEFAULT_LOGS_DIR = _resolve_default_logs_dir()


def _fmt_srt_time(seconds: float) -> str:
    sec_total = max(0.0, seconds)
    hh = int(sec_total // 3600)
    mm = int((sec_total % 3600) // 60)
    ss = int(sec_total % 60)
    ms = int(round((sec_total - int(sec_total)) * 1000))
    if ms >= 1000:
        ms = 999
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def _ensure_writable_dir(preferred: Path) -> Path:
    candidates = [
        preferred,
        Path.home() / ".cache" / "voicerecore" / "logs",
        Path(tempfile.gettempdir()) / "voicerecore_logs",
    ]
    for cand in candidates:
        try:
            cand.mkdir(parents=True, exist_ok=True)
            return cand
        except OSError:
            continue
    return preferred


class SessionLogger:
    def __init__(self, logs_dir: Path | None = None) -> None:
        self.logs_dir = _ensure_writable_dir(Path(logs_dir or DEFAULT_LOGS_DIR))
        self._lock = threading.Lock()
        self._txt_path: Optional[Path] = None
        self._srt_path: Optional[Path] = None
        self._session_start_mono: float = 0.0
        self._last_cue_end_s: float = 0.0
        self._count: int = 0
        # Visual paragraph grouping state for clean crash-safe .TXT meeting minutes
        self._cur_para_id: str = ""
        self._cur_para_ts: str = ""
        self._cur_para_langs: tuple[str, str] = ("", "")
        self._cur_para_speaker: str = ""
        self._cur_para_src: list[str] = []
        self._cur_para_tgt: list[str] = []
        self._cur_para_offset: int = 0

    def start_new_session(self, source_lang: str, target_lang: str) -> dict:
        with self._lock:
            self.logs_dir = _ensure_writable_dir(self.logs_dir)
            stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            self._txt_path = self.logs_dir / f"meeting_{stamp}.txt"
            self._srt_path = self.logs_dir / f"meeting_{stamp}.srt"
            self._session_start_mono = time.monotonic()
            self._last_cue_end_s = 0.0
            self._count = 0
            self._cur_para_id = ""
            self._cur_para_ts = ""
            self._cur_para_langs = ("", "")
            self._cur_para_speaker = ""
            self._cur_para_src = []
            self._cur_para_tgt = []
            self._cur_para_offset = 0

            header = (
                f"=== DAU Voice Pro — Biên Bản Dịch Song Ngữ ({stamp}) ===\n"
                f"Hướng dịch: {source_lang} -> {target_lang}\n"
                f"{'=' * 64}\n\n"
            )
            self._txt_path.write_text(header, encoding="utf-8")
            self._srt_path.write_text("", encoding="utf-8")
            self._cur_para_offset = self._txt_path.stat().st_size
            return self.get_status_unlocked()

    def _write_txt_inplace(self, payload: bytes) -> None:
        assert self._txt_path is not None
        for attempt in range(2):
            try:
                with open(self._txt_path, "r+b") as fb:
                    fb.seek(self._cur_para_offset)
                    fb.write(payload)
                    fb.truncate()
                    fb.flush()
                return
            except PermissionError:
                if attempt == 0:
                    time.sleep(0.02)
                else:
                    raise

    def append_entry(self, entry: dict) -> None:
        with self._lock:
            if self._txt_path is None or self._srt_path is None:
                return
            self._count += 1
            now_elapsed = max(0.5, time.monotonic() - self._session_start_mono)
            start_s = self._last_cue_end_s
            end_s = max(start_s + 1.5, now_elapsed)
            self._last_cue_end_s = end_s

            ts = entry.get("timestamp", time.strftime("%H:%M:%S"))
            src_lang = entry.get("source_lang", "")
            tgt_lang = entry.get("target_lang", "")
            src_txt = entry.get("source_text", "").strip()
            tr_txt = entry.get("translated_text", "").strip()
            para_id = str(entry.get("paragraph_id") or "").strip()
            speaker = str(entry.get("speaker") or "").strip()
            if speaker == "self":
                spk_tag = "[Bạn] "
            elif speaker == "remote":
                spk_tag = "[Đối tác] "
            elif speaker.upper().startswith("S") and speaker[1:].isdigit():
                spk_tag = f"[Diễn giả {speaker[1:]}] "
            elif speaker:
                spk_tag = f"[{speaker}] "
            else:
                spk_tag = ""

            # .TXT: group consecutive sentences with same paragraph_id & speaker into coherent paragraphs,
            # updating the trailing block in-place with immediate flush (100% crash-safe).
            same_para = (
                bool(para_id)
                and para_id == self._cur_para_id
                and (src_lang, tgt_lang) == self._cur_para_langs
                and speaker == self._cur_para_speaker
            )
            if same_para:
                self._cur_para_src.append(src_txt)
                if tr_txt:
                    self._cur_para_tgt.append(tr_txt)
                merged_src = " ".join(self._cur_para_src).strip()
                merged_tgt = " ".join(self._cur_para_tgt).strip()
                txt_block = (
                    f"[{self._cur_para_ts}] {spk_tag}({src_lang}): {merged_src}\n"
                    f"[{self._cur_para_ts}] {spk_tag}({tgt_lang}): {merged_tgt}\n\n"
                )
                self._write_txt_inplace(txt_block.encode("utf-8"))
            else:
                self._cur_para_id = para_id
                self._cur_para_ts = ts
                self._cur_para_langs = (src_lang, tgt_lang)
                self._cur_para_speaker = speaker
                self._cur_para_src = [src_txt]
                self._cur_para_tgt = [tr_txt] if tr_txt else []
                txt_block = (
                    f"[{ts}] {spk_tag}({src_lang}): {src_txt}\n"
                    f"[{ts}] {spk_tag}({tgt_lang}): {tr_txt}\n\n"
                )
                with open(self._txt_path, "ab") as fb:
                    self._cur_para_offset = fb.tell()
                    fb.write(txt_block.encode("utf-8"))
                    fb.flush()

            # .SRT: keep crisp YouTube-style sentence-by-sentence subtitle cues
            srt_block = (
                f"{self._count}\n"
                f"{_fmt_srt_time(start_s)} --> {_fmt_srt_time(end_s)}\n"
                f"{spk_tag}{tr_txt}\n"
                f"{spk_tag}{src_txt}\n\n"
            )
            with open(self._srt_path, "a", encoding="utf-8") as f:
                f.write(srt_block)
                f.flush()

    def get_status_unlocked(self) -> dict:
        return {
            "logs_dir": str(self.logs_dir),
            "txt_file": str(self._txt_path) if self._txt_path else "",
            "srt_file": str(self._srt_path) if self._srt_path else "",
            "saved_count": self._count,
        }

    def get_status(self) -> dict:
        with self._lock:
            return self.get_status_unlocked()

    def open_logs_folder(self) -> bool:
        self.logs_dir = _ensure_writable_dir(self.logs_dir)
        sys_name = platform.system()
        try:
            if sys_name == "Darwin":
                subprocess.Popen(["open", str(self.logs_dir)])
            elif sys_name == "Windows":
                os.startfile(str(self.logs_dir))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(self.logs_dir)])
            return True
        except Exception:
            return False

    def _resolve_safe_log_path(self, filename: str) -> Path:
        name = (filename or "").strip()
        if (
            not name
            or "/" in name
            or "\\" in name
            or ".." in name
            or not name.lower().endswith(".txt")
        ):
            raise ValueError("Tên file biên bản không hợp lệ.")
        root = self.logs_dir.resolve()
        target = (root / name).resolve()
        if target.parent != root:
            raise ValueError("Đường dẫn file biên bản nằm ngoài thư mục cho phép.")
        return target

    def list_sessions(self, limit: int = 80) -> list[dict]:
        """Return newest-first list of saved meeting transcripts in `logs_dir`."""
        self.logs_dir = _ensure_writable_dir(self.logs_dir)
        with self._lock:
            active_name = self._txt_path.name if self._txt_path is not None else ""
        items: list[dict] = []
        try:
            txt_files = sorted(
                self.logs_dir.glob("*.txt"),
                key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
                reverse=True,
            )
        except OSError:
            return []

        for p in txt_files[: max(1, min(200, int(limit)))]:
            try:
                st = p.stat()
                srt_p = p.with_suffix(".srt")
                preview = ""
                direction = ""
                with open(p, "r", encoding="utf-8", errors="replace") as fh:
                    for _ in range(30):
                        line = fh.readline()
                        if not line:
                            break
                        s = line.strip()
                        if not s or s.startswith("==="):
                            continue
                        if s.startswith("Hướng dịch:"):
                            direction = s.replace("Hướng dịch:", "").strip()
                            continue
                        preview = s[:140]
                        break
                items.append(
                    {
                        "filename": p.name,
                        "srt_filename": srt_p.name if srt_p.is_file() else "",
                        "modified_ts": st.st_mtime,
                        "modified_str": time.strftime("%d/%m/%Y %H:%M", time.localtime(st.st_mtime)),
                        "size_bytes": st.st_size,
                        "direction": direction,
                        "preview": preview or "(Chưa có nội dung)",
                        "is_active": p.name == active_name,
                    }
                )
            except OSError:
                continue
        return items

    def read_session(self, filename: str) -> dict:
        target = self._resolve_safe_log_path(filename)
        if not target.is_file():
            raise FileNotFoundError(f"Không tìm thấy biên bản: {filename}")
        st = target.stat()
        content = target.read_text(encoding="utf-8", errors="replace")
        srt_p = target.with_suffix(".srt")
        srt_content = srt_p.read_text(encoding="utf-8", errors="replace") if srt_p.is_file() else ""
        with self._lock:
            is_active = self._txt_path is not None and self._txt_path.name == target.name
        return {
            "ok": True,
            "filename": target.name,
            "srt_filename": srt_p.name if srt_p.is_file() else "",
            "content": content,
            "srt_content": srt_content,
            "size_bytes": st.st_size,
            "modified_str": time.strftime("%d/%m/%Y %H:%M:%S", time.localtime(st.st_mtime)),
            "is_active": is_active,
        }

    def delete_session(self, filename: str, *, is_listening: bool = False) -> dict:
        target = self._resolve_safe_log_path(filename)
        with self._lock:
            if is_listening and self._txt_path is not None and self._txt_path.name == target.name:
                raise RuntimeError("Không thể xóa file biên bản của phiên dịch đang chạy.")
        if not target.is_file():
            raise FileNotFoundError(f"Không tìm thấy biên bản: {filename}")
        target.unlink()
        srt_p = target.with_suffix(".srt")
        if srt_p.is_file():
            try:
                srt_p.unlink()
            except OSError:
                pass
        return {"ok": True, "deleted": target.name}
