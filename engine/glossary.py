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
import shutil
import tempfile
import threading
import time
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


# Canonical Vietnamese family surnames (~65 covering >99% of Vietnamese full names)
_VI_SURNAMES = frozenset({
    "nguyễn", "trần", "lê", "phạm", "hoàng", "huỳnh", "phan", "vũ", "võ", "đặng",
    "bùi", "đỗ", "hồ", "ngô", "dương", "lý", "trương", "đinh", "đoàn", "lâm",
    "trịnh", "đào", "cao", "tạ", "mai", "chu", "châu", "lưu", "lương", "thái",
    "tống", "phùng", "vương", "từ", "tô", "tăng", "quách", "lại", "kiều", "nghiêm",
    "thạch", "la", "hà", "trình", "khổng", "mạc", "doãn", "thân", "khúc", "triệu",
    "diệp", "hứa", "bạch", "nhan", "ninh", "khương", "lục", "nông", "vi", "bế",
    "âu", "đồng", "cù", "đàm", "tôn",
})

# Vietnamese common middle names used when recognizing lowercase names right after an honorific
_VI_COMMON_MIDDLE_NAMES = frozenset({
    "văn", "thị", "gia", "nam", "minh", "hoàng", "thanh", "quang", "đức", "hữu",
    "quốc", "xuân", "thu", "ngọc", "bảo", "tuấn", "anh", "duy", "hải", "hồng",
    "kim", "khánh", "phương", "thảo", "thùy", "thuỳ", "tiến", "trung", "trường", "việt",
    "công", "đình", "mạnh", "nhật", "nguyên", "tấn", "thành", "thế", "trọng", "vũ",
    "viết", "huy", "khải", "lâm", "long", "phúc", "sơn", "thái", "thắng", "thiên",
    "thiện", "toàn", "trí", "tùng", "vinh", "mỹ", "bích", "diễm", "hạnh", "hiền",
    "hoa", "hương", "lan", "linh", "mai", "nga", "nhi", "nhung", "oanh", "quỳnh",
    "tâm", "trang", "trinh", "tuyết", "vân", "yến", "bắc", "đông", "tây", "đăng",
    "giang", "hiếu", "hòa", "hoài", "hùng", "khoa", "kiên", "kiệt", "lê", "liên",
    "loan", "lực", "nghĩa", "nhân", "phước", "sang", "tài", "tân", "thịnh", "thuận",
    "thủy", "triều", "trúc", "tú", "tuệ", "uyên", "bá", "chí", "đắc", "hà", "như",
})

# Words that must NEVER be swallowed into an auto-detected Vietnamese person name
_VI_NON_NAME_WORDS = frozenset({
    "của", "là", "có", "và", "nhưng", "mà", "thì", "một", "hai", "ba", "bốn", "năm",
    "sáu", "bảy", "tám", "chín", "mười", "các", "những", "trong", "ngoài", "trên",
    "dưới", "với", "cho", "để", "từ", "về", "đến", "tới", "đã", "đang", "sẽ", "không",
    "chưa", "rồi", "cũng", "vẫn", "rất", "quá", "này", "đó", "đấy", "ấy", "kia",
    "nói", "bảo", "làm", "đi", "ra", "vào", "lên", "xuống", "xem", "biết", "thấy",
    "hỏi", "nghe", "gửi", "chia", "sẻ", "gặp", "đón", "chờ", "chở", "mua", "bán",
    "dùng", "giữ", "muốn", "phải", "cần", "được", "bị", "tại", "ở", "như", "vì",
    "nếu", "khi", "nên", "hay", "hoặc", "nữa", "thôi", "nhé", "nhá", "ạ", "đâu",
    "sao", "gì", "nào", "ai", "nước", "phố", "đường", "tỉnh", "thành", "quận",
    "huyện", "xã", "phường", "trường", "viện", "bộ", "sở", "ban", "ngành", "công",
    "ty", "tập", "đoàn", "khách", "hàng", "nghiệp", "doanh", "đồng", "tiền", "tháng",
    "ngày", "giờ", "phút", "giây", "đế", "vương", "nhà", "xe", "máy", "cửa", "lụa",
    "bia", "hơi", "quốc", "tế", "đội", "quân", "hệ", "thống", "chiến", "lược",
    "nhân", "dân", "hội", "nghị", "cuộc", "họp", "chuyện", "câu", "việc", "sự",
    "thực", "thế", "giới", "kinh", "lịch", "do", "tốc", "kỳ", "lỗi", "phòng",
    "tham", "dự", "phát", "biểu", "trình", "bày", "giới", "thiệu", "chia", "tạo",
})

# Vietnamese places, landmarks & brands prone to literal/Pinyin MT errors
# (Leaves Hà Nội -> Hanoi, Việt Nam -> Vietnam untouched so English stays natural)
_VI_PROTECTED_PLACES_AND_BRANDS: tuple[str, ...] = (
    "Hải Phòng", "Đà Nẵng", "Nha Trang", "Đà Lạt", "Vũng Tàu", "Cần Thơ",
    "Hạ Long", "Phú Quốc", "Bắc Ninh", "Bình Dương", "Đồng Nai", "Quảng Ninh",
    "Hải Dương", "Nghệ An", "Thanh Hóa", "Thái Nguyên", "Lào Cai", "Lạng Sơn",
    "Móng Cái", "Hàng Gai", "Kim Mã", "Tân Mỹ",
    "FPT", "VNPT", "Viettel", "Vingroup", "VinFast", "Metropole",
)

# Distinct non-Chinese human-name sentinels so Latin-script MT preserves "Chairman <Name>" / "Mr. <Name>" grammar!
_HUMAN_NAME_SENTINELS: tuple[str, ...] = (
    "Marek", "Davin", "Sorin", "Tarek", "Kaelen", "Bramen", "Joren", "Varek",
)

_VI_TITLE_PAT = (
    r"(?:phó\s+chủ\s+tịch|chủ\s+tịch|tổng\s+giám\s+đốc|phó\s+giám\s+đốc|giám\s+đốc|"
    r"phó\s+hiệu\s+trưởng|hiệu\s+trưởng|phó\s+giáo\s+sư|giáo\s+sư|tiến\s+sĩ|thạc\s+sĩ|"
    r"thủ\s+tướng|bộ\s+trưởng|thứ\s+trưởng|đồng\s+chí|đại\s+biểu|diễn\s+giả|"
    r"bác\s+sĩ|kỹ\s+sư|luật\s+sư|nhà\s+báo|ông|bà|anh|chị|thầy|cô|bác|chú|cậu|lão|ngài)"
)

_VI_SURNAME_ALT = "|".join(re.escape(s) for s in sorted(_VI_SURNAMES, key=len, reverse=True))
_VI_CAP_SURNAME_ALT = "|".join(
    re.escape(s[0].upper() + s[1:]) for s in sorted(_VI_SURNAMES, key=len, reverse=True)
)
_VI_CAP_SYL = r"[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ][a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]{1,7}"
_VI_ANY_SYL = r"[A-Za-zÀ-ỹà-ỹ]{1,8}"

# Pass A: Capitalized 3-4 syllable Vietnamese person name (e.g. "Trương Gia Bình", "Đặng Lê Nguyên Vũ")
_RE_CAP_VI_NAME = re.compile(
    rf"(?<!\w)((?:{_VI_CAP_SURNAME_ALT})\s+{_VI_CAP_SYL}\s+{_VI_CAP_SYL}(?:\s+{_VI_CAP_SYL})?)(?!\w)",
    re.UNICODE,
)

# Pass B: Title + lowercase/mixed 3-4 syllable Vietnamese person name (e.g. "anh trương gia bình", "ông hoàng nam tiến")
_RE_TITLE_VI_NAME = re.compile(
    rf"(?<!\w){_VI_TITLE_PAT}\s+((?:{_VI_SURNAME_ALT})\s+{_VI_ANY_SYL}\s+{_VI_ANY_SYL}(?:\s+{_VI_ANY_SYL})?)(?!\w)",
    re.IGNORECASE | re.UNICODE,
)


def _title_case_vi_name(raw: str) -> str:
    return " ".join((w[0].upper() + w[1:].lower()) if len(w) > 1 else w.upper() for w in raw.split())


def extract_vietnamese_proper_entities(text: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Automatically detect Vietnamese person names (preventing Sino-Vietnamese Pinyin
    hallucinations like 'Trương Gia Bình' -> 'Zhang Jingping' or 'Hoàng Nam Tiến' -> 'Huang Nam')
    and mistranslation-prone Vietnamese landmarks/brands ('Kim Mã', 'Tân Mỹ', 'Hải Phòng', 'FPT').

    Returns `(person_pairs, place_brand_pairs)` where each pair is `(source_span, canonical_name)`.
    """
    if not text:
        return [], []

    persons: list[tuple[str, str]] = []
    seen_spans: set[str] = set()

    def _validate_and_add(span: str, require_middle_hint: bool) -> None:
        toks = span.split()
        if len(toks) < 3 or len(toks) > 4:
            return
        if toks[0].lower() not in _VI_SURNAMES:
            return
        # Trim trailing stopword if a 4-word match accidentally grabbed a following verb ("trương gia bình nói")
        while len(toks) > 3 and toks[-1].lower() in _VI_NON_NAME_WORDS:
            toks.pop()
        if len(toks) < 3:
            return
        # Every syllable after surname must not be a common non-name word
        if any(t.lower() in _VI_NON_NAME_WORDS for t in toks[1:]):
            return
        # For lowercase/mixed matches after a title, keep strict 3-syllable name unless the 4th word is capitalized
        if require_middle_hint:
            if not (toks[1][0].isupper() or toks[1].lower() in _VI_COMMON_MIDDLE_NAMES):
                return
            if len(toks) == 4 and not toks[3][0].isupper():
                toks = toks[:3]
        matched = " ".join(toks)
        key = matched.lower()
        # Skip if already covered as a subset/superset of an earlier Pass-A capitalized match
        if key in seen_spans or any(key in s or s in key for s in seen_spans):
            return
        seen_spans.add(key)
        persons.append((matched, _title_case_vi_name(matched)))

    # Pass A: Capitalized names (check surname is also Title-Case so lowercase prose doesn't false-match)
    for m in _RE_CAP_VI_NAME.finditer(text):
        span = m.group(1)
        if span[0].isupper():
            _validate_and_add(span, require_middle_hint=False)

    # Pass B: Title + lowercase/mixed names (e.g. "anh trương gia bình", "ông hoàng nam tiến")
    for m in _RE_TITLE_VI_NAME.finditer(text):
        span = m.group(1)
        _validate_and_add(span, require_middle_hint=True)

    # Pass C: Protected Vietnamese landmarks, cities & enterprises
    places: list[tuple[str, str]] = []
    for canonical in _VI_PROTECTED_PLACES_AND_BRANDS:
        pat = _word_boundary_pattern(canonical)
        if pat.search(text) and canonical.lower() not in seen_spans:
            seen_spans.add(canonical.lower())
            places.append((canonical, canonical))

    # Longest spans first so overlapping names resolve cleanly
    persons.sort(key=lambda kv: len(kv[0]), reverse=True)
    places.sort(key=lambda kv: len(kv[0]), reverse=True)
    return persons, places


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
                # Preserve corrupted user glossary as a timestamped backup before resetting to defaults
                try:
                    backup = self.path.with_name(f"{self.path.stem}.corrupt.{int(time.time())}{self.path.suffix}")
                    shutil.copy2(self.path, backup)
                except Exception:
                    pass
        data = json.loads(json.dumps(DEFAULT_DATA))
        self._save_unlocked(data)
        return data

    def _save_unlocked(self, data: dict[str, Any]) -> None:
        tmp_path: Path | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(data, ensure_ascii=False, indent=2)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as tf:
                tmp_path = Path(tf.name)
                tf.write(payload)
                tf.flush()
                os.fsync(tf.fileno())
            os.replace(tmp_path, self.path)
        except Exception:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

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
            out = pat.sub(lambda _m, c=rule["correct"]: c, out)
        return out

    def get_effective_term_locks(
        self,
        text: str,
        source_lang: str = "vie_Latn",
        target_lang: str = "eng_Latn",
    ) -> list[dict[str, str]]:
        """
        Return user glossary term_locks merged with auto-detected Vietnamese proper names
        and landmarks for native LLM terminology intervention (Hy-MT2 / Gemini).
        """
        with self._lock:
            base = list(self._data["term_locks"]) if self._data.get("enabled", True) else []
        if source_lang != "vie_Latn" or target_lang == "vie_Latn" or not text:
            return base
        persons, places = extract_vietnamese_proper_entities(text)
        if not persons and not places:
            return base
        existing = {x["source"].lower() for x in base}
        merged = list(base)
        for src_span, canonical in [*persons, *places]:
            if src_span.lower() not in existing and canonical.lower() not in existing:
                merged.append({"source": src_span, "target": canonical})
                existing.add(src_span.lower())
                existing.add(canonical.lower())
        return merged

    def protect_terms(
        self,
        text: str,
        source_lang: str = "vie_Latn",
        target_lang: str = "eng_Latn",
    ) -> tuple[str, list[tuple[str, str]]]:
        """
        Replace locked glossary terms AND auto-detected Vietnamese proper names / landmarks
        with MT-safe placeholders, returning `(masked_text, restore_pairs)`:
        - Vietnamese person names -> Latin human-name sentinels (`Marek`, `Davin`...) for
          Latin targets (preserves natural 'Chairman Trương Gia Bình' / 'Mr. Hoàng Nam Tiến'
          grammar and 100% eliminates Chinese Pinyin hallucinations like 'Zhang Jingping'!),
          or opaque `XTermA0X` sentinels for non-Latin targets (Thai/Lao/Chinese/Japanese).
        - Glossary locks & places/brands -> opaque `XTermA0X` sentinels.
        """
        if not text:
            return text, []
        with self._lock:
            rules = (
                sorted(self._data["term_locks"], key=lambda r: len(r["source"]), reverse=True)
                if self._data.get("enabled", True) and self._data.get("term_locks")
                else []
            )

        masked = text
        restore_pairs: list[tuple[str, str]] = []
        idx = 0

        # 1) User-configured Glossary term locks first (highest precedence)
        for rule in rules:
            pat = _word_boundary_pattern(rule["source"])
            if pat.search(masked):
                token = f"XTerm{chr(65 + (idx % 26))}{idx}X"
                masked = pat.sub(lambda _m, tok=token: tok, masked)
                restore_pairs.append((token, rule["target"]))
                idx += 1

        # 2) Automatic Vietnamese Proper-Name & Landmark Shield (prevents Chinese Pinyin & literal errors)
        if source_lang == "vie_Latn" and target_lang != "vie_Latn":
            persons, places = extract_vietnamese_proper_entities(masked)
            is_latin_target = target_lang.endswith("_Latn")
            used_human: set[str] = set()

            for src_span, canonical in persons:
                pat = _word_boundary_pattern(src_span)
                if not pat.search(masked):
                    continue
                token = ""
                if is_latin_target:
                    for cand in _HUMAN_NAME_SENTINELS:
                        if cand not in used_human and not re.search(rf"(?<!\w){cand}(?!\w)", masked, re.IGNORECASE):
                            token = cand
                            used_human.add(cand)
                            break
                if not token:
                    token = f"XTerm{chr(65 + (idx % 26))}{idx}X"
                    idx += 1
                masked = pat.sub(lambda _m, tok=token: tok, masked)
                restore_pairs.append((token, canonical))

            for src_span, canonical in places:
                pat = _word_boundary_pattern(src_span)
                if pat.search(masked):
                    token = f"XTerm{chr(65 + (idx % 26))}{idx}X"
                    masked = pat.sub(lambda _m, tok=token: tok, masked)
                    restore_pairs.append((token, canonical))
                    idx += 1

        return masked, restore_pairs

    @staticmethod
    def restore_terms(translated: str, restore_pairs: list[tuple[str, str]]) -> str:
        if not translated or not restore_pairs:
            return translated
        out = translated
        for token, target in sorted(restore_pairs, key=lambda p: len(p[0]), reverse=True):
            # Word-boundary case-insensitive replacement prevents accidental substring collisions
            out = re.sub(rf"(?<!\w){re.escape(token)}(?!\w)", lambda _m, t=target: t, out, flags=re.IGNORECASE)
        return out
