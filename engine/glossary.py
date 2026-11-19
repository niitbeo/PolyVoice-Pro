"""
Custom Meeting Glossary & Protected-Term Engine for VOICERECORE.

Two layers of accuracy boosting (100% offline, zero latency overhead):
1. `asr_fixes`: Corrects known ASR mishearings BEFORE display and BEFORE translation
   (e.g. "mái chủ" -> "máy chủ", "coopfro" -> "Cloudflare", "c l i" -> "CLI").
2. `term_locks`: Protects proper names, brands, and fixed domain translations during MT
   using clean sentinel placeholders (e.g. "Nguyễn Lê Trường" -> stays "Nguyễn Lê Trường"
   instead of getting translated as "Nguyen Le School").
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

DEFAULT_GLOSSARY_PATH = Path(
    os.environ.get(
        "VOICERECORE_GLOSSARY_PATH",
        Path.home() / ".cache" / "voicerecore" / "glossary.json",
    )
)

PRESETS: dict[str, dict[str, list[dict[str, str]]]] = {
    "tech_ai": {
        "asr_fixes": [
            {"wrong": "mái chủ", "correct": "máy chủ"},
            {"wrong": "lập chìm", "correct": "lập trình"},
            {"wrong": "c l i", "correct": "CLI"},
            {"wrong": "m c p", "correct": "MCP"},
            {"wrong": "a p i", "correct": "API"},
            {"wrong": "canvast", "correct": "Canvas"},
            {"wrong": "nộp seat", "correct": "Obsidian"},
            {"wrong": "coopfro", "correct": "Cloudflare"},
            {"wrong": "clod", "correct": "Claude"},
            {"wrong": "ncip", "correct": "MCP"},
        ],
        "term_locks": [
            {"source": "MCP", "target": "MCP"},
            {"source": "API", "target": "API"},
            {"source": "CLI", "target": "CLI"},
            {"source": "Cloudflare", "target": "Cloudflare"},
            {"source": "Obsidian", "target": "Obsidian"},
            {"source": "Claude", "target": "Claude"},
        ],
    },
    "business_edu": {
        "asr_fixes": [
            {"wrong": "mc", "correct": "MC"},
            {"wrong": "hiệu trưởng chạy", "correct": "hiệu trưởng"},
            {"wrong": "k p i", "correct": "KPI"},
            {"wrong": "o k r", "correct": "OKR"},
        ],
        "term_locks": [
            {"source": "Giám đốc điều hành", "target": "CEO"},
            {"source": "Hiệu trưởng", "target": "Principal"},
            {"source": "KPI", "target": "KPI"},
            {"source": "OKR", "target": "OKR"},
        ],
    },
    "medical": {
        "asr_fixes": [
            {"wrong": "bệnh viện nhà nước", "correct": "bệnh viện công lập"},
        ],
        "term_locks": [
            {"source": "Giám đốc chuyên môn", "target": "Chief Medical Officer"},
            {"source": "Giám đốc bệnh viện", "target": "Hospital Director"},
        ],
    },
}

DEFAULT_DATA: dict[str, Any] = {
    "enabled": True,
    "asr_fixes": list(PRESETS["tech_ai"]["asr_fixes"]),
    "term_locks": [
        {"source": "MCP", "target": "MCP"},
        {"source": "API", "target": "API"},
        {"source": "CLI", "target": "CLI"},
        {"source": "Cloudflare", "target": "Cloudflare"},
        {"source": "Obsidian", "target": "Obsidian"},
        {"source": "Claude", "target": "Claude"},
    ],
}


def _word_boundary_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term.strip())
    # Unicode-aware boundary check (works cleanly with Vietnamese diacritics!)
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE | re.UNICODE)


class GlossaryManager:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or DEFAULT_GLOSSARY_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return {
                        "enabled": bool(raw.get("enabled", True)),
                        "asr_fixes": [
                            {"wrong": str(x.get("wrong", "")).strip(), "correct": str(x.get("correct", "")).strip()}
                            for x in raw.get("asr_fixes", [])
                            if str(x.get("wrong", "")).strip() and str(x.get("correct", "")).strip()
                        ],
                        "term_locks": [
                            {"source": str(x.get("source", "")).strip(), "target": str(x.get("target", "")).strip()}
                            for x in raw.get("term_locks", [])
                            if str(x.get("source", "")).strip() and str(x.get("target", "")).strip()
                        ],
                    }
            except Exception:
                pass
        data = json.loads(json.dumps(DEFAULT_DATA))
        self._save_unlocked(data)
        return data

    def _save_unlocked(self, data: dict[str, Any]) -> None:
        try:
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def get(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self._data["enabled"],
                "asr_fixes": list(self._data["asr_fixes"]),
                "term_locks": list(self._data["term_locks"]),
                "presets": list(PRESETS.keys()),
                "path": str(self.path),
            }

    def update(
        self,
        enabled: bool | None = None,
        asr_fixes: list[dict[str, str]] | None = None,
        term_locks: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if enabled is not None:
                self._data["enabled"] = bool(enabled)
            if asr_fixes is not None:
                self._data["asr_fixes"] = [
                    {"wrong": str(x.get("wrong", "")).strip(), "correct": str(x.get("correct", "")).strip()}
                    for x in asr_fixes
                    if str(x.get("wrong", "")).strip() and str(x.get("correct", "")).strip()
                ]
            if term_locks is not None:
                self._data["term_locks"] = [
                    {"source": str(x.get("source", "")).strip(), "target": str(x.get("target", "")).strip()}
                    for x in term_locks
                    if str(x.get("source", "")).strip() and str(x.get("target", "")).strip()
                ]
            self._save_unlocked(self._data)
        return self.get()

    def apply_preset(self, preset_name: str) -> dict[str, Any]:
        preset = PRESETS.get(preset_name)
        if not preset:
            raise ValueError(f"Unknown glossary preset: {preset_name}")
        with self._lock:
            existing_asr = {x["wrong"].lower() for x in self._data["asr_fixes"]}
            for item in preset["asr_fixes"]:
                if item["wrong"].lower() not in existing_asr:
                    self._data["asr_fixes"].append(dict(item))
                    existing_asr.add(item["wrong"].lower())

            existing_locks = {x["source"].lower() for x in self._data["term_locks"]}
            for item in preset["term_locks"]:
                if item["source"].lower() not in existing_locks:
                    self._data["term_locks"].append(dict(item))
                    existing_locks.add(item["source"].lower())

            self._save_unlocked(self._data)
        return self.get()

    def apply_asr_fixes(self, text: str) -> str:
        """Apply pre-MT ASR mishearing corrections (longest match first)."""
        if not text:
            return text
        with self._lock:
            if not self._data["enabled"] or not self._data["asr_fixes"]:
                return text
            rules = sorted(self._data["asr_fixes"], key=lambda r: len(r["wrong"]), reverse=True)

        out = text
        for rule in rules:
            pat = _word_boundary_pattern(rule["wrong"])
            out = pat.sub(rule["correct"], out)
        return out

    def protect_terms(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        """
        Replace locked source terms with opaque proper-noun tokens (`QTERM0`, `QTERM1`...)
        that neural MT models preserve verbatim, returning `(masked_text, restore_pairs)`.
        """
        if not text:
            return text, []
        with self._lock:
            if not self._data["enabled"] or not self._data["term_locks"]:
                return text, []
            rules = sorted(self._data["term_locks"], key=lambda r: len(r["source"]), reverse=True)

        masked = text
        restore_pairs: list[tuple[str, str]] = []
        idx = 0
        for rule in rules:
            pat = _word_boundary_pattern(rule["source"])
            if pat.search(masked):
                # Proper-noun-like Latin code token preserved cleanly by T5 & NLLB
                token = f"XTerm{chr(65 + (idx % 26))}{idx}"
                masked = pat.sub(token, masked)
                restore_pairs.append((token, rule["target"]))
                idx += 1
        return masked, restore_pairs

    @staticmethod
    def restore_terms(translated: str, restore_pairs: list[tuple[str, str]]) -> str:
        if not translated or not restore_pairs:
            return translated
        out = translated
        for token, target in restore_pairs:
            # Case-insensitive replacement in case MT lowercased the placeholder
            out = re.sub(re.escape(token), lambda _m, t=target: t, out, flags=re.IGNORECASE)
        return out
