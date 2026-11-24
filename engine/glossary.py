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
import unicodedata
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
            {"source": "Giám đốc điều hành", "target": "CEO", "target_lang": "eng_Latn"},
            {"source": "Hiệu trưởng", "target": "Principal", "target_lang": "eng_Latn"},
            {"source": "KPI", "target": "KPI"},
            {"source": "OKR", "target": "OKR"},
        ],
    },
    "medical": {
        "asr_fixes": [
            {"wrong": "bệnh viện nhà nước", "correct": "bệnh viện công lập"},
        ],
        "term_locks": [
            {"source": "Giám đốc chuyên môn", "target": "Chief Medical Officer", "target_lang": "eng_Latn"},
            {"source": "Giám đốc bệnh viện", "target": "Hospital Director", "target_lang": "eng_Latn"},
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
    "nhất", "hơn", "bằng", "gần", "xa", "sang", "giáo", "phật", "thần", "linh",
    "cung", "địa", "học", "thời", "trọng", "kính", "chúa", "tổ", "lão", "thượng",
})

# Known 2-word Vietnamese places / non-person phrases starting with a surname word
_VI_2WORD_NON_PERSON_PHRASES = frozenset({
    "hà nội", "hà tĩnh", "hà nam", "hà giang", "hà tây", "hà bắc",
    "thái bình", "thái nguyên", "thái lan", "thái dương", "thái cực", "thái thượng",
    "lâm đồng", "cao bằng", "cao nguyên", "cao cấp", "cao điểm", "cao trào", "cao tuổi",
    "châu á", "châu âu", "châu mỹ", "châu phi", "châu úc", "châu lục", "châu báu",
    "đồng tháp", "đồng nai", "đồng bằng", "đồng ý", "đồng hành", "đồng bộ", "đồng chí",
    "hồ gươm", "hồ tây", "hồ hoàn", "hồ sơ", "ninh bình", "ninh thuận",
    "lý tưởng", "lý thuyết", "lý lịch", "lý do", "lý tính", "lý luận",
    "tôn giáo", "tôn vinh", "tôn chỉ", "hoàng gia", "hoàng hôn", "hoàng hậu", "hoàng cung", "hoàng kim", "hoàng đế", "hoàng tử",
    "đào tạo", "lưu trữ", "lưu ý", "lưu hành", "lưu lượng", "tăng trưởng", "tăng cường",
    "phạm vi", "bùi ngùi", "trần gian", "trần trụi", "mai sau", "mai mối",
    "đoàn kết", "đoàn thể", "kiều bào", "kiều hối", "bạch cầu", "bạch kim", "bạch tuyết",
    "hứa hẹn", "nghiêm túc", "nghiêm trọng", "nghiêm ngặt", "thạch cao", "thạch anh",
    "diệp lục", "khúc xạ", "khúc quanh", "từ thiện", "từ điển", "từ chối", "từ bỏ",
    "đàm phán", "đàm đạo", "nông dân", "nông nghiệp", "nông thôn", "nông sản",
    "bế mạc", "bế tắc", "lương thực", "lương tâm", "lương thiện", "khổng lồ",
    "thân thiện", "thân thiết", "thân phận", "thân thể", "âu lạc", "âu châu",
    "vi mô", "vi sinh", "vi phạm", "triệu chứng", "triệu tập", "lục địa",
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
    r"thủ\s+tướng|bộ\s+trưởng|thứ\s+trưởng|đồng\s+chí|đại\s+biểu|diễn\s+giả|tác\s+giả|"
    r"bác\s+sĩ|kỹ\s+sư|luật\s+sư|nhà\s+báo|chú\s+bé|ông|bà|anh|chị|thầy|cô|bác|chú|cậu|lão|ngài)"
)

_VI_SURNAME_ALT = "|".join(re.escape(s) for s in sorted(_VI_SURNAMES, key=len, reverse=True))
_VI_CAP_SURNAME_ALT = "|".join(
    re.escape(s[0].upper() + s[1:]) for s in sorted(_VI_SURNAMES, key=len, reverse=True)
)
_VI_CAP_SYL = r"[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ][a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]{1,7}"
_VI_LOWER_SYL = r"[a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]{2,7}"
_VI_ANY_SYL = r"[A-Za-zÀ-ỹà-ỹ]{1,8}"

# Pass A1: Capitalized 3-4 syllable Vietnamese person name (e.g. "Trương Gia Bình", "Đặng Lê Nguyên Vũ")
_RE_CAP_VI_NAME = re.compile(
    rf"(?<!\w)((?:{_VI_CAP_SURNAME_ALT})\s+{_VI_CAP_SYL}\s+{_VI_CAP_SYL}(?:\s+{_VI_CAP_SYL})?)(?!\w)",
    re.UNICODE,
)

# Pass A2: Mixed-case 3-syllable person name where Pass-2 missed middle-word cap (e.g. "Ngô thừa Ân", "Trương gia Bình")
_RE_MIXED3_VI_NAME = re.compile(
    rf"(?<!\w)((?:{_VI_CAP_SURNAME_ALT})\s+{_VI_LOWER_SYL}\s+{_VI_CAP_SYL})(?!\w)",
    re.UNICODE,
)

# Pass A3: Standalone capitalized 2-syllable Vietnamese person name (e.g. "Phan Đăng", "Nguyễn Du", "Lê Lợi")
# Negative lookahead prevents slicing the first 2 words off a longer 3+ word capitalized phrase (e.g. "Tôn Ngộ Không")!
_RE_CAP2_VI_NAME = re.compile(
    rf"(?<!\w)((?:{_VI_CAP_SURNAME_ALT})\s+{_VI_CAP_SYL})(?!\w)(?!\s+{_VI_CAP_SYL})",
    re.UNICODE,
)

# Pass B1: Title + lowercase/mixed 3-4 syllable Vietnamese person name (e.g. "anh trương gia bình", "ông hoàng nam tiến")
_RE_TITLE_VI_NAME = re.compile(
    rf"(?<!\w){_VI_TITLE_PAT}\s+((?:{_VI_SURNAME_ALT})\s+{_VI_ANY_SYL}\s+{_VI_ANY_SYL}(?:\s+{_VI_ANY_SYL})?)(?!\w)",
    re.IGNORECASE | re.UNICODE,
)

# Pass B2: Title + 2-syllable Vietnamese person name (e.g. "diễn giả phan đăng", "anh phan đăng")
_RE_TITLE2_VI_NAME = re.compile(
    rf"(?<!\w){_VI_TITLE_PAT}\s+((?:{_VI_SURNAME_ALT})\s+{_VI_ANY_SYL})(?!\w)",
    re.IGNORECASE | re.UNICODE,
)


def _title_case_vi_name(raw: str) -> str:
    return " ".join((w[0].upper() + w[1:].lower()) if len(w) > 1 else w.upper() for w in raw.split())


def extract_vietnamese_proper_entities(
    text: str,
    session_persons: dict[str, str] | None = None,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Automatically detect Vietnamese person names (preventing Sino-Vietnamese Pinyin
    hallucinations like 'Trương Gia Bình' -> 'Zhang Jingping', 'Phan Đăng' -> 'Fang',
    or 'Ngô Thừa Ân' -> 'Wu Enlil / Wu Zetian') and mistranslation-prone Vietnamese
    landmarks/brands ('Kim Mã', 'Tân Mỹ', 'Hải Phòng', 'FPT').

    Returns `(person_pairs, place_brand_pairs)` where each pair is `(source_span, canonical_name)`.
    """
    if not text:
        return [], []

    persons: list[tuple[str, str]] = []
    seen_spans: set[str] = set()
    protected_place_lowers = {p.lower() for p in _VI_PROTECTED_PLACES_AND_BRANDS}

    def _validate_and_add(span: str, require_middle_hint: bool, min_words: int = 3) -> None:
        toks = span.split()
        if len(toks) < min_words or len(toks) > 4:
            return
        if toks[0].lower() not in _VI_SURNAMES:
            return
        # Trim trailing lowercase stopword/verb if Pass-B accidentally grabbed one ("trương gia bình nói")
        # Never slice a capitalized 4th word off a 4-word proper phrase!
        while len(toks) > 3 and not toks[-1][0].isupper() and toks[-1].lower() in _VI_NON_NAME_WORDS:
            toks.pop()
        if len(toks) < min_words:
            return
        # Every syllable after surname must not be a common non-name word
        if any(t.lower() in _VI_NON_NAME_WORDS for t in toks[1:]):
            return
        matched = " ".join(toks)
        key = matched.lower()
        if len(toks) == 2:
            if key in _VI_2WORD_NON_PERSON_PHRASES or key in protected_place_lowers:
                return
            if require_middle_hint and not (toks[1][0].isupper() or toks[1].lower() in _VI_COMMON_MIDDLE_NAMES):
                return
        elif require_middle_hint:
            # For lowercase/mixed 3-4 word matches after a title, require a common middle name or uppercase cue
            if not (toks[1][0].isupper() or toks[1].lower() in _VI_COMMON_MIDDLE_NAMES):
                return
            if len(toks) == 4 and not toks[3][0].isupper():
                toks = toks[:3]
                matched = " ".join(toks)
                key = matched.lower()
        # Skip if already covered as a subset/superset of an earlier longer match
        if key in seen_spans or any(key in s or s in key for s in seen_spans):
            return
        seen_spans.add(key)
        persons.append((matched, _title_case_vi_name(matched)))

    # Pass A1: Capitalized 3-4 syllable names (e.g. "Trương Gia Bình")
    for m in _RE_CAP_VI_NAME.finditer(text):
        span = m.group(1)
        if span[0].isupper():
            _validate_and_add(span, require_middle_hint=False, min_words=3)

    # Pass A2: Mixed-case 3-syllable names where middle syllable stayed lowercase ("Ngô thừa Ân")
    for m in _RE_MIXED3_VI_NAME.finditer(text):
        span = m.group(1)
        if span[0].isupper():
            _validate_and_add(span, require_middle_hint=False, min_words=3)

    # Pass B1: Title + lowercase/mixed 3-4 syllable names ("anh trương gia bình")
    for m in _RE_TITLE_VI_NAME.finditer(text):
        span = m.group(1)
        _validate_and_add(span, require_middle_hint=True, min_words=3)

    # Pass A3: Capitalized 2-syllable names ("Phan Đăng", "Nguyễn Du") — runs after 3-4 word passes
    for m in _RE_CAP2_VI_NAME.finditer(text):
        span = m.group(1)
        if span[0].isupper():
            _validate_and_add(span, require_middle_hint=False, min_words=2)

    # Pass B2: Title + 2-syllable names ("diễn giả phan đăng", "anh phan đăng")
    for m in _RE_TITLE2_VI_NAME.finditer(text):
        span = m.group(1)
        _validate_and_add(span, require_middle_hint=True, min_words=2)

    # Pass S: Remembered session person names (catches later lowercase recurrences like "Phan đăng" -> "Phan Đăng")
    if session_persons:
        for low_name, canonical in sorted(session_persons.items(), key=lambda kv: len(kv[0]), reverse=True):
            if low_name in seen_spans or any(low_name in s or s in low_name for s in seen_spans):
                continue
            pat = _word_boundary_pattern(canonical)
            m = pat.search(text)
            if m:
                seen_spans.add(low_name)
                persons.append((m.group(0), canonical))

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
        self._session_persons: dict[str, str] = {}

    def clear_session_entities(self) -> None:
        with self._lock:
            self._session_persons.clear()

    def _remember_persons_locked(self, persons: list[tuple[str, str]]) -> None:
        for _span, canonical in persons:
            key = canonical.lower()
            if key not in self._session_persons:
                if len(self._session_persons) >= 64:
                    # Drop oldest remembered name
                    self._session_persons.pop(next(iter(self._session_persons)))
                self._session_persons[key] = canonical

    def normalize_session_entities(self, text: str, source_lang: str = "vie_Latn") -> str:
        """
        Detect Vietnamese person names, remember them for the current session, and restore
        consistent Title-Case on mixed/lowercase recurrences (e.g. 'Phan đăng' -> 'Phan Đăng',
        'Ngô thừa Ân' -> 'Ngô Thừa Ân') before display and MT.
        """
        if not text or source_lang != "vie_Latn":
            return text
        with self._lock:
            snap = dict(self._session_persons)
        persons, _ = extract_vietnamese_proper_entities(text, session_persons=snap)
        if not persons:
            return text
        with self._lock:
            self._remember_persons_locked(persons)
        out = text
        for src_span, canonical in persons:
            if src_span != canonical:
                pat = _word_boundary_pattern(src_span)
                out = pat.sub(lambda _m, c=canonical: c, out)
        return out

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
                            (
                                {
                                    "source": str(x.get("source", "")).strip(),
                                    "target": str(x.get("target", "")).strip(),
                                    **(
                                        {"target_lang": str(x.get("target_lang", "")).strip()}
                                        if str(x.get("target_lang", "")).strip()
                                        else {}
                                    ),
                                }
                            )
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
                cleaned_locks: list[dict[str, str]] = []
                for x in term_locks:
                    src_s = str(x.get("source", "")).strip()
                    tgt_s = str(x.get("target", "")).strip()
                    if src_s and tgt_s:
                        entry = {"source": src_s, "target": tgt_s}
                        tl = str(x.get("target_lang", "")).strip()
                        if tl:
                            entry["target_lang"] = tl
                        cleaned_locks.append(entry)
                self._data["term_locks"] = cleaned_locks
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
        text = unicodedata.normalize("NFC", text)
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
            raw_base = list(self._data["term_locks"]) if self._data.get("enabled", True) else []
            session_snap = dict(self._session_persons)
        base = [
            r for r in raw_base
            if not r.get("target_lang") or r.get("target_lang") == target_lang
        ]
        if source_lang != "vie_Latn" or target_lang == "vie_Latn" or not text:
            return base
        persons, places = extract_vietnamese_proper_entities(text, session_persons=session_snap)
        if persons:
            with self._lock:
                self._remember_persons_locked(persons)
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
        extra_locks: list[dict[str, str]] | None = None,
    ) -> tuple[str, list[tuple[str, str]]]:
        """
        Replace locked glossary terms, Slide-RAG terms, AND auto-detected Vietnamese proper names /
        landmarks with MT-safe placeholders, returning `(masked_text, restore_pairs)`:
        - Vietnamese person names -> Latin human-name sentinels (`Marek`, `Davin`...) for
          Latin targets (preserves natural 'Chairman Trương Gia Bình' / 'Mr. Hoàng Nam Tiến'
          grammar and 100% eliminates Chinese Pinyin hallucinations like 'Zhang Jingping'!),
          or opaque `XTermA0X` sentinels for non-Latin targets (Thai/Lao/Chinese/Japanese).
        - Glossary / Slide-RAG locks & places/brands -> opaque `XTermA0X` sentinels.
        """
        if not text:
            return text, []
        text = unicodedata.normalize("NFC", text)
        with self._lock:
            manual_rules = (
                list(self._data["term_locks"])
                if self._data.get("enabled", True) and self._data.get("term_locks")
                else []
            )
            session_snap = dict(self._session_persons)

        # Merge manual Glossary locks (highest priority) + Slide-RAG matched locks, longest-match first
        combined_rules: list[dict[str, str]] = list(manual_rules)
        seen_src = {str(r.get("source", "")).lower() for r in combined_rules if r.get("source")}
        for item in extra_locks or []:
            src_s = str(item.get("source", "")).strip()
            tgt_s = str(item.get("target", "")).strip()
            if src_s and tgt_s and src_s.lower() not in seen_src:
                seen_src.add(src_s.lower())
                combined_rules.append({"source": src_s, "target": tgt_s, "target_lang": str(item.get("target_lang", ""))})
        rules = sorted(combined_rules, key=lambda r: len(r["source"]), reverse=True)

        masked = text
        restore_pairs: list[tuple[str, str]] = []
        idx = 0

        # 1) User-configured Glossary term locks first (highest precedence)
        for rule in rules:
            rule_tgt = rule.get("target_lang")
            if rule_tgt and rule_tgt != target_lang:
                continue
            pat = _word_boundary_pattern(rule["source"])
            if pat.search(masked):
                token = f"XTerm{chr(65 + (idx % 26))}{idx}X"
                masked = pat.sub(lambda _m, tok=token: tok, masked)
                restore_pairs.append((token, rule["target"]))
                idx += 1

        # 2) Automatic Vietnamese Proper-Name & Landmark Shield (prevents Chinese Pinyin & literal errors)
        if source_lang == "vie_Latn" and target_lang != "vie_Latn":
            persons, places = extract_vietnamese_proper_entities(masked, session_persons=session_snap)
            if persons:
                with self._lock:
                    self._remember_persons_locked(persons)
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
            # Opaque ASCII sentinels (`XTermA0X`) often attach directly to Hangul/CJK particles
            # (`XTermA0X는`, `XTermA0Xは`, `对于XTermC2X`) where Unicode `\w` matches non-ASCII scripts.
            # Use ASCII-only boundaries for `XTerm...X` so restoration succeeds 100% across all scripts!
            if token.startswith("XTerm") and token.endswith("X"):
                pat = rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])"
            else:
                pat = rf"(?<!\w){re.escape(token)}(?!\w)"
            out = re.sub(pat, lambda _m, t=target: t, out, flags=re.IGNORECASE)
        return out
