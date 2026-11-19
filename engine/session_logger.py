"""
Real-Time Auto-Save Meeting Logger (.TXT and .SRT) for multi-hour seminars & meetings.

Appends and flushes every committed paragraph directly to disk under
`~/Documents/VOICERECORE_Logs/` so zero data is ever lost during 2–3+ hour sessions.
"""

from __future__ import annotations

import os
import platform
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

DEFAULT_LOGS_DIR = Path(
    os.environ.get(
        "VOICERECORE_LOG_DIR",
        Path.home() / "Documents" / "VOICERECORE_Logs",
    )
)


def _fmt_srt_time(seconds: float) -> str:
    sec_total = max(0.0, seconds)
    hh = int(sec_total // 3600)
    mm = int((sec_total % 3600) // 60)
    ss = int(sec_total % 60)
    ms = int(round((sec_total - int(sec_total)) * 1000))
    if ms >= 1000:
        ms = 999
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


class SessionLogger:
    def __init__(self, logs_dir: Path | None = None) -> None:
        self.logs_dir = Path(logs_dir or DEFAULT_LOGS_DIR)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
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
                f"=== PolyVoice Pro — Biên Bản Dịch Song Ngữ ({stamp}) ===\n"
                f"Hướng dịch: {source_lang} -> {target_lang}\n"
                f"{'=' * 64}\n\n"
            )
            self._txt_path.write_text(header, encoding="utf-8")
            self._srt_path.write_text("", encoding="utf-8")
            self._cur_para_offset = self._txt_path.stat().st_size
            return self.get_status_unlocked()

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
            spk_tag = "[Bạn] " if speaker == "self" else "[Đối tác] " if speaker == "remote" else ""

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
                with open(self._txt_path, "r+b") as fb:
                    fb.seek(self._cur_para_offset)
                    fb.write(txt_block.encode("utf-8"))
                    fb.truncate()
                    fb.flush()
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
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        sys_name = platform.system()
        try:
            if sys_name == "Darwin":
                subprocess.Popen(["open", str(self.logs_dir)])
            elif sys_name == "Windows":
                subprocess.Popen(["explorer", str(self.logs_dir)])
            else:
                subprocess.Popen(["xdg-open", str(self.logs_dir)])
            return True
        except Exception:
            return False
