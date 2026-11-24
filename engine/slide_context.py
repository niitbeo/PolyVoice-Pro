"""
100% Offline Conference Slide / Document Context & Slide-Aware RAG (<1ms CPU)
for DAU Voice Pro (Windows x64 & macOS arm64).

Supports:
- `.pptx`: Native stdlib zipfile + XML parser (extracts Slide 1..N titles, body, tables & Speaker Notes)
- `.docx`: Native stdlib zipfile + XML parser (paragraphs, headings & tables)
- `.pdf` : Pure-Python `pypdf` page-by-page extraction (~300 KB, zero C deps)
- `.txt` / `.md`: UTF-8 heading/section chunking

Outputs:
- Compact conference topic summary (<= 18 words)
- Auto-extracted proper nouns, academic titles, acronyms & bilingual terms
  -> feeds BOTH ASR BPE Hotwords (Zipformer-30M) AND Hy-MT2 terminology injection
- Sub-millisecond Slide-Aware RAG (`retrieve_for_utterance`) tracking speaker progression
"""

from __future__ import annotations

import json
import math
import re
import threading
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class SlidePage:
    index: int
    title: str
    text: str
    notes: str = ""
    terms: list[str] = field(default_factory=list)


@dataclass
class ContextTerm:
    source: str
    target: str = ""  # empty means preserve proper noun / acronym verbatim
    enabled: bool = True
    origin: str = "slide"  # "slide" | "dau_preset" | "manual"


@dataclass
class SlideRagMatch:
    slide_index: int
    slide_title: str
    score: float
    domain_hint: str
    matched_terms: list[dict[str, str]]


# Official 1-click conference preset for Da Nang Architecture University (DAU)
DAU_CONFERENCE_TOPIC = "Hội thảo Khoa học Quốc tế — Trường Đại học Kiến trúc Đà Nẵng (DAU)"
DAU_CONFERENCE_TERMS: list[tuple[str, str]] = [
    ("Đại học Kiến trúc Đà Nẵng", "Da Nang Architecture University (DAU)"),
    ("Trường Đại học Kiến trúc Đà Nẵng", "Da Nang Architecture University (DAU)"),
    ("DAU", "DAU"),
    ("Ban Giám hiệu", "Board of Rectors"),
    ("Hiệu trưởng", "Rector"),
    ("Phó Hiệu trưởng", "Vice Rector"),
    ("Khoa Kiến trúc", "Faculty of Architecture"),
    ("Khoa Xây dựng", "Faculty of Civil Engineering"),
    ("Khoa Công nghệ Thông tin", "Faculty of Information Technology"),
    ("Khoa Mỹ thuật Công nghiệp", "Faculty of Industrial Fine Arts"),
    ("Khoa Quy hoạch Đô thị", "Faculty of Urban Planning"),
    ("Khoa Ngoại ngữ", "Faculty of Foreign Languages"),
    ("Đồ án tốt nghiệp", "Graduation Capstone Project"),
    ("Kiến trúc xanh", "Green Architecture"),
    ("Kiến trúc bền vững", "Sustainable Architecture"),
    ("Đô thị thông minh", "Smart City"),
    ("Quy hoạch đô thị", "Urban Planning"),
    ("Phát triển bền vững", "Sustainable Development"),
    ("Hội thảo Khoa học Quốc tế", "International Scientific Conference"),
    ("BIM", "BIM"),
    ("Net Zero", "Net Zero"),
    ("ESG", "ESG"),
]


_STOPWORDS: frozenset[str] = frozenset(
    {
        # Vietnamese common prose/function words
        "và", "của", "các", "những", "cho", "với", "trong", "trên", "tại", "từ", "đến",
        "được", "là", "có", "không", "một", "những", "này", "đó", "như", "khi", "để",
        "về", "theo", "hay", "hoặc", "nhưng", "mà", "thì", "lại", "cũng", "đã", "đang",
        "sẽ", "rất", "hơn", "nhất", "cần", "phải", "nên", "vì", "do", "bởi", "qua",
        "sau", "trước", "giữa", "bằng", "cùng", "ngoài", "vào", "ra", "lên", "xuống",
        "chúng", "ta", "tôi", "bạn", "họ", "mình", "người", "việc", "cách", "phần",
        "nội", "dung", "mục", "tiêu", "chính", "tổng", "quan", "giới", "thiệu", "kết",
        "luận", "cảm", "ơn", "quý", "vị", "đại", "biểu", "thưa", "kính", "chào",
        # English common stopwords
        "the", "and", "for", "with", "that", "this", "from", "into", "about", "your",
        "have", "will", "would", "could", "should", "they", "their", "them", "there",
        "what", "when", "where", "which", "while", "how", "who", "are", "was", "were",
        "been", "being", "has", "had", "not", "but", "can", "all", "our", "out", "you",
        "one", "two", "more", "most", "very", "just", "also", "than", "then", "now",
        "slide", "page", "section", "overview", "introduction", "conclusion", "thank",
        "thanks", "agenda", "summary", "outline", "next", "first", "second", "third",
    }
)

_RE_NON_WORD = re.compile(r"[^\w\u00C0-\u024F\u1E00-\u1EFF\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF\u0E00-\u0E7F\u1780-\u17FF]+")
_RE_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,8}(?:[-/][A-Z0-9]{1,5})?(?:\s+\d{2,4})?\b")
_RE_VI_TITLE_NAME = re.compile(
    r"\b(?:PGS\.?\s*TS\.?|GS\.?\s*TS\.?|TS\.|ThS\.|KTS\.|KS\.|Ông|Bà|Prof\.?|Dr\.?)\s+"
    r"[A-ZÀ-Ỵ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỵ][a-zà-ỹ]+){1,4}\b"
)
_RE_MULTI_CAP = re.compile(
    r"\b[A-ZÀ-Ỵ][A-Za-zÀ-ỹ0-9]{1,20}(?:\s+[A-ZÀ-Ỵ][A-Za-zÀ-ỹ0-9]{1,20}){1,4}\b"
)
_RE_BILINGUAL_PAREN = re.compile(
    r"([A-Za-zÀ-ỹ0-9][A-Za-zÀ-ỹ0-9\s\-/]{2,42}?)\s*\(\s*([A-Za-zÀ-ỹ0-9][A-Za-zÀ-ỹ0-9\s\-/]{1,42}?)\s*\)"
)
_RE_SLIDE_NUM = re.compile(r"slide(\d+)\.xml$", re.IGNORECASE)


def _strip_accents(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _tokenize_for_rag(text: str) -> set[str]:
    """Extract normalized word tokens + accent-folded tokens + CJK/Thai/Hangul 2-char bigrams."""
    raw = (text or "").strip()
    if not raw:
        return set()
    toks: set[str] = set()
    for part in _RE_NON_WORD.split(raw.lower()):
        if not part:
            continue
        # Non-spaced scripts: emit overlapping 2-char bigrams
        if any("\u3040" <= ch <= "\u9fff" or "\uac00" <= ch <= "\ud7af" or "\u0e00" <= ch <= "\u0e7f" for ch in part):
            if len(part) >= 2:
                for i in range(len(part) - 1):
                    toks.add(part[i : i + 2])
            continue
        if len(part) < 2 or part in _STOPWORDS:
            continue
        toks.add(part)
        folded = _strip_accents(part)
        if len(folded) >= 2 and folded != part and folded not in _STOPWORDS:
            toks.add(folded)
    return toks


def _clean_line(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", s or "").replace("\u00a0", " ").split()).strip()


def _has_vi_diacritics(s: str) -> bool:
    return any(
        ch in "ăâđêôơưĂÂĐÊÔƠƯàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ"
        "ÀÁẢÃẠẰẮẲẴẶẦẤẨẪẬÈÉẺẼẸỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌỒỐỔỖỘỜỚỞỠỢÙÚỦŨỤỪỨỬỮỰỲÝỶỸỴ"
        for ch in (s or "")
    )


def _infer_term_lang(s: str) -> str:
    """Infer target language code for directional bilingual pairs ('' = universal proper noun / acronym)."""
    txt = (s or "").strip()
    if not txt:
        return ""
    if _has_vi_diacritics(txt):
        return "vie_Latn"
    if any("\uac00" <= ch <= "\ud7af" for ch in txt):
        return "kor_Hang"
    if any("\u3040" <= ch <= "\u30ff" for ch in txt):
        return "jpn_Jpan"
    if any("\u4e00" <= ch <= "\u9fff" for ch in txt):
        return "zho_Hant"
    if any("\u0e00" <= ch <= "\u0e7f" for ch in txt):
        return "tha_Thai"
    # All-caps acronym / code (e.g. DAU, BIM, LOD 400, ESG) is language-neutral
    letters = [ch for ch in txt if ch.isalpha()]
    if letters and all(ch.isupper() for ch in letters):
        return ""
    # Pure ASCII phrase with lowercase letters (e.g. "Board of Rectors", "Glass Fiber", "Smart City")
    if all(ord(ch) < 128 for ch in txt) and any(ch.islower() for ch in txt):
        return "eng_Latn"
    return ""


def _term_word_match(term_low: str, term_fold: str, q_low: str, q_fold: str) -> bool:
    """Unicode word-boundary matching (prevents 'AI' matching 'đại học' or 'khoa' matching 'khoảng cách')."""
    if not term_low:
        return False
    # Non-spaced CJK/Thai/Khmer scripts can match directly by substring
    if any("\u3040" <= ch <= "\u9fff" or "\u0e00" <= ch <= "\u0e7f" or "\u1780" <= ch <= "\u17ff" for ch in term_low):
        return term_low in q_low
    pat_exact = rf"(?<!\w){re.escape(term_low)}(?!\w)"
    if re.search(pat_exact, q_low, re.UNICODE):
        return True
    # Accent-folded fallback ONLY for multi-word phrases (len >= 6) with word boundaries
    # so missing Vietnamese tone marks from ASR still lock multi-word terms with zero single-word collisions
    if " " in term_fold and len(term_fold) >= 6:
        pat_fold = rf"(?<!\w){re.escape(term_fold)}(?!\w)"
        if re.search(pat_fold, q_fold, re.UNICODE):
            return True
    return False


def _extract_pptx_pages(data: bytes) -> list[SlidePage]:
    import io

    pages: list[SlidePage] = []
    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
        names = zf.namelist()
        slide_entries: list[tuple[int, str]] = []
        for n in names:
            m = _RE_SLIDE_NUM.search(n)
            if m and n.startswith("ppt/slides/"):
                slide_entries.append((int(m.group(1)), n))
        slide_entries.sort(key=lambda x: x[0])

        def _read_xml_texts(xml_bytes: bytes) -> tuple[str, list[str]]:
            root = ET.fromstring(xml_bytes)
            title_cand = ""
            lines: list[str] = []
            # Walk shape trees so we can spot title placeholders cleanly
            for elem in root.iter():
                tag = elem.tag.rsplit("}", 1)[-1]
                if tag == "sp":
                    is_title = False
                    for ph in elem.iter():
                        if ph.tag.rsplit("}", 1)[-1] == "ph":
                            ph_type = (ph.attrib.get("type") or "").lower()
                            if ph_type in ("title", "ctrtitle", "subtitle"):
                                is_title = True
                    paras: list[str] = []
                    for p in elem.iter():
                        if p.tag.rsplit("}", 1)[-1] == "p":
                            run_txt = "".join(
                                (t.text or "")
                                for t in p.iter()
                                if t.tag.rsplit("}", 1)[-1] == "t"
                            )
                            cleaned = _clean_line(run_txt)
                            if cleaned:
                                paras.append(cleaned)
                    if paras:
                        if is_title and not title_cand:
                            title_cand = paras[0]
                        lines.extend(paras)
                elif tag == "tbl":
                    for tr in elem.iter():
                        if tr.tag.rsplit("}", 1)[-1] == "tr":
                            cells: list[str] = []
                            for tc in tr:
                                if tc.tag.rsplit("}", 1)[-1] == "tc":
                                    ct = _clean_line(
                                        " ".join(
                                            (t.text or "")
                                            for t in tc.iter()
                                            if t.tag.rsplit("}", 1)[-1] == "t"
                                        )
                                    )
                                    if ct:
                                        cells.append(ct)
                            if cells:
                                lines.append(" | ".join(cells))
            # Deduplicate consecutive identical lines
            dedup: list[str] = []
            seen: set[str] = set()
            for ln in lines:
                if ln not in seen:
                    seen.add(ln)
                    dedup.append(ln)
            if not title_cand and dedup:
                title_cand = dedup[0][:90]
            return title_cand, dedup

        for slide_no, slide_path in slide_entries:
            try:
                title, lines = _read_xml_texts(zf.read(slide_path))
            except Exception:
                continue

            notes_text = ""
            # Try direct slide rels first for exact notesSlide mapping, then fallback by index
            rels_path = f"ppt/slides/_rels/slide{slide_no}.xml.rels"
            notes_target = ""
            if rels_path in names:
                try:
                    rroot = ET.fromstring(zf.read(rels_path))
                    for rel in rroot:
                        tgt = rel.attrib.get("Target", "")
                        if "notesSlide" in tgt:
                            notes_target = "ppt/notesSlides/" + tgt.rsplit("/", 1)[-1]
                            break
                except Exception:
                    pass
            if not notes_target and f"ppt/notesSlides/notesSlide{slide_no}.xml" in names:
                notes_target = f"ppt/notesSlides/notesSlide{slide_no}.xml"
            if notes_target and notes_target in names:
                try:
                    _, nlines = _read_xml_texts(zf.read(notes_target))
                    # Filter standalone slide-number footers
                    nlines = [ln for ln in nlines if not ln.isdigit()]
                    notes_text = "\n".join(nlines)
                except Exception:
                    pass

            body = "\n".join(lines).strip()
            if body or notes_text:
                pages.append(
                    SlidePage(
                        index=len(pages) + 1,
                        title=(title or f"Slide {slide_no}")[:90],
                        text=body,
                        notes=notes_text,
                    )
                )
    return pages


def _extract_docx_pages(data: bytes) -> list[SlidePage]:
    import io

    pages: list[SlidePage] = []
    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
        if "word/document.xml" not in zf.namelist():
            return pages
        root = ET.fromstring(zf.read("word/document.xml"))

    sections: list[tuple[str, list[str]]] = []
    cur_title = ""
    cur_lines: list[str] = []
    cur_words = 0

    for elem in root.iter():
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag != "p":
            continue
        style_val = ""
        for pstyle in elem.iter():
            if pstyle.tag.rsplit("}", 1)[-1] == "pStyle":
                for k, v in pstyle.attrib.items():
                    if k.endswith("val"):
                        style_val = v
                        break
        txt = _clean_line(
            "".join((t.text or "") for t in elem.iter() if t.tag.rsplit("}", 1)[-1] == "t")
        )
        if not txt:
            continue
        is_heading = "heading" in style_val.lower() or "title" in style_val.lower()
        if (is_heading and cur_lines) or cur_words >= 130:
            sections.append((cur_title or (cur_lines[0][:85] if cur_lines else "Phần"), cur_lines))
            cur_title = txt[:85] if is_heading else ""
            cur_lines = [txt]
            cur_words = len(txt.split())
        else:
            if not cur_title:
                cur_title = txt[:85]
            cur_lines.append(txt)
            cur_words += len(txt.split())

    if cur_lines:
        sections.append((cur_title or cur_lines[0][:85], cur_lines))

    for idx, (title, lines) in enumerate(sections, 1):
        pages.append(SlidePage(index=idx, title=title, text="\n".join(lines)))
    return pages


def _extract_pdf_pages(data: bytes) -> list[SlidePage]:
    import io
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages: list[SlidePage] = []
    for i, page in enumerate(reader.pages, 1):
        raw = page.extract_text() or ""
        lines = [_clean_line(ln) for ln in raw.splitlines() if _clean_line(ln)]
        if not lines:
            continue
        title = lines[0][:90]
        pages.append(SlidePage(index=len(pages) + 1, title=title, text="\n".join(lines)))
    return pages


def _extract_text_pages(text: str) -> list[SlidePage]:
    raw = (text or "").replace("\r\n", "\n").strip()
    if not raw:
        return []
    # Split on explicit slide dividers (`---`, `===`, `# Heading`, `Slide N:`) or ~120-word chunks
    blocks = re.split(r"\n\s*(?:---+|\={3,}|(?=#{1,3}\s+)|(?=Slide\s+\d+[:.\s-]))", raw, flags=re.IGNORECASE)
    pages: list[SlidePage] = []
    for blk in blocks:
        lines = [_clean_line(ln.lstrip("#").strip()) for ln in blk.splitlines() if _clean_line(ln)]
        if not lines:
            continue
        # Sub-chunk overly long plain-text blocks so RAG stays granular
        chunk: list[str] = []
        wc = 0
        for ln in lines:
            chunk.append(ln)
            wc += len(ln.split())
            if wc >= 120:
                pages.append(SlidePage(index=len(pages) + 1, title=chunk[0][:90], text="\n".join(chunk)))
                chunk = []
                wc = 0
        if chunk:
            pages.append(SlidePage(index=len(pages) + 1, title=chunk[0][:90], text="\n".join(chunk)))
    return pages


_BILINGUAL_LEAD_TRIM_2: frozenset[str] = frozenset(
    {
        "sử dụng", "nghiên cứu", "ứng dụng", "phát triển", "quản lý", "giúp quản", "kết hợp",
        "tối ưu", "xây dựng", "thiết kế", "mô phỏng", "đánh giá", "phân tích", "triển khai",
        "bao gồm", "tích hợp", "áp dụng", "giới thiệu", "bàn về", "nói về", "giải pháp",
        "phối hợp", "cũng như", "như là", "dựa trên", "liên quan", "đối với",
    }
)
_BILINGUAL_LEAD_TRIM_1: frozenset[str] = frozenset(
    {
        "và", "về", "của", "bằng", "trong", "với", "là", "cho", "tại", "từ", "đến", "theo",
        "gồm", "như", "các", "những", "một", "giúp", "cùng", "qua", "hay", "hoặc", "vào",
        "the", "a", "an", "of", "in", "on", "to", "for", "with", "and", "or", "by", "using",
        "via", "from", "about", "including", "such", "as",
    }
)


def _trim_bilingual_left(left: str, right: str) -> str:
    """Trim leading prose verbs/prepositions before '(English Term)' so only the noun phrase is locked."""
    # Never trim after a comma/colon/semicolon — take only the final sub-clause right before '('
    s = re.split(r"[,;:—–\-•|]", left)[-1].strip()
    words = s.split()
    while len(words) > 2:
        w2 = f"{words[0].lower()} {words[1].lower()}"
        if w2 in _BILINGUAL_LEAD_TRIM_2:
            words = words[2:]
            continue
        if words[0].lower() in _BILINGUAL_LEAD_TRIM_1:
            words = words[1:]
            continue
        break
    if len(words) == 2 and words[0].lower() in _BILINGUAL_LEAD_TRIM_1:
        words = words[1:]
    # Cap phrase length reasonably relative to right-hand term so long clauses never get swallowed
    max_w = max(4, len(right.split()) + 3)
    if len(words) > max_w:
        words = words[-max_w:]
    return " ".join(words).strip()


def _extract_terms_from_pages(pages: list[SlidePage]) -> list[ContextTerm]:
    """
    Auto-extract acronyms, academic titles + proper names, bilingual parenthetical pairs,
    and recurring domain phrases from slides (100% offline heuristic, zero LLM cost).
    """
    full_text = "\n".join(f"{p.title}\n{p.text}\n{p.notes}" for p in pages)
    candidates: dict[str, ContextTerm] = {}
    freq: Counter[str] = Counter()

    def _add(src: str, tgt: str = "", boost: int = 1) -> None:
        s = _clean_line(src).strip(".,;:!?\"'()[]{}-–—")
        t = _clean_line(tgt).strip(".,;:!?\"'()[]{}-–—")
        if len(s) < 2 or len(s) > 48:
            return
        low = s.lower()
        if low in _STOPWORDS or s.isdigit():
            return
        # Skip generic ALL-CAPS Vietnamese slide headings ("NỘI DUNG", "GIỚI THIỆU", "KẾT LUẬN")
        if s.isupper() and len(s.split()) >= 2 and all(w.lower() in _STOPWORDS for w in s.split()):
            return
        key = low
        freq[key] += boost
        if key not in candidates:
            candidates[key] = ContextTerm(source=s, target=t, enabled=True, origin="slide")
        elif t and not candidates[key].target:
            candidates[key].target = t

    # 1) Bilingual parenthetical pairs e.g. "Sợi thủy tinh (Glass Fiber)" or "Smart City (Đô thị thông minh)"
    for m in _RE_BILINGUAL_PAREN.finditer(full_text):
        raw_left, right = _clean_line(m.group(1)), _clean_line(m.group(2))
        left = _trim_bilingual_left(raw_left, right)
        if len(left) < 2 or len(right) < 2:
            continue
        if len(left.split()) <= 6 and len(right.split()) <= 6:
            if right.isupper() and 2 <= len(right) <= 10:
                # Acronym expansion like "Đại học Kiến trúc Đà Nẵng (DAU)":
                # keep the acronym locked verbatim and keep the full phrase as ASR hotword (do NOT collapse full name to "DAU")
                _add(right, right, boost=4)
                _add(left, "", boost=2)
            else:
                _add(left, right, boost=3)

    # 2) Academic titles + names ("PGS.TS. Phạm Anh Tuấn", "Dr. John Smith")
    for m in _RE_VI_TITLE_NAME.finditer(full_text):
        _add(m.group(0), m.group(0), boost=5)

    # 3) Acronyms & model/standard codes ("DAU", "BIM", "LOD 400", "TSMC", "DRAM", "AI", "ESG")
    generic_caps = {"THE", "AND", "FOR", "WITH", "FROM", "THIS", "THAT", "ALL", "NEW", "TOP", "Slide"}
    for m in _RE_ACRONYM.finditer(full_text):
        tok = m.group(0).strip()
        if tok in generic_caps or len(tok) < 2:
            continue
        _add(tok, tok, boost=2)

    # 4) Multi-word capitalized proper nouns / domain phrases ("Inflection Pi", "Net Zero", "Smart City")
    for p in pages:
        page_terms: list[str] = []
        combined = f"{p.title}\n{p.text}\n{p.notes}"
        for m in _RE_MULTI_CAP.finditer(combined):
            phrase = _clean_line(m.group(0))
            words = phrase.split()
            if len(words) < 2 or len(words) > 5:
                continue
            if words[0].lower() in _STOPWORDS and words[-1].lower() in _STOPWORDS:
                continue
            if sum(1 for w in words if w.lower() not in _STOPWORDS) == 0:
                continue
            _add(phrase, "", boost=1)
            if phrase not in page_terms and len(page_terms) < 12:
                page_terms.append(phrase)
        for m in _RE_ACRONYM.finditer(combined):
            tok = m.group(0).strip()
            if tok not in generic_caps and tok not in page_terms and len(page_terms) < 15:
                page_terms.append(tok)
        p.terms = page_terms

    # Rank by priority/frequency and keep top 60 cleanest terms so hotwords stay fast & sharp
    ranked = sorted(candidates.items(), key=lambda kv: (-freq[kv[0]], -len(kv[1].source)))
    return [ct for _, ct in ranked[:60]]


def _derive_topic_summary(filename: str, pages: list[SlidePage]) -> str:
    parts: list[str] = []
    if pages:
        t1 = _clean_line(pages[0].title)
        if t1 and len(t1) >= 4:
            parts.append(t1)
        if len(pages) > 1:
            t2 = _clean_line(pages[1].title)
            if t2 and t2.lower() != t1.lower() and len(t2) >= 4:
                parts.append(t2)
    if not parts and filename:
        parts.append(Path(filename).stem.replace("_", " ").replace("-", " "))
    summary = " — ".join(parts)
    words = summary.split()
    if len(words) > 18:
        summary = " ".join(words[:18])
    return summary.strip()


class SlideContextManager:
    """Thread-safe 100% Offline Conference Slide/Document Context & Sub-Millisecond RAG."""

    def __init__(self, store_path: Path) -> None:
        self.store_path = Path(store_path)
        self._lock = threading.Lock()
        self.enabled: bool = True
        self.filename: str = ""
        self.topic: str = ""
        self.pages: list[SlidePage] = []
        self.terms: list[ContextTerm] = []
        self.active_slide_index: int = 0  # 1-based; 0 means none matched yet
        self._page_tokens: list[set[str]] = []
        self._idf: dict[str, float] = {}
        self._load()

    def _rebuild_index_locked(self) -> None:
        self._page_tokens = []
        df: Counter[str] = Counter()
        for p in self.pages:
            toks = _tokenize_for_rag(f"{p.title} {p.title} {p.text} {p.notes}")
            self._page_tokens.append(toks)
            for t in toks:
                df[t] += 1
        n = max(len(self.pages), 1)
        self._idf = {t: math.log((n + 1.0) / (c + 0.5)) + 1.0 for t, c in df.items()}

    def _load(self) -> None:
        if not self.store_path.is_file():
            return
        try:
            raw = json.loads(self.store_path.read_text(encoding="utf-8"))
            self.enabled = bool(raw.get("enabled", True))
            self.filename = str(raw.get("filename") or "")
            self.topic = str(raw.get("topic") or "")
            self.pages = [
                SlidePage(
                    index=int(p.get("index", i + 1)),
                    title=str(p.get("title") or ""),
                    text=str(p.get("text") or ""),
                    notes=str(p.get("notes") or ""),
                    terms=list(p.get("terms") or []),
                )
                for i, p in enumerate(raw.get("pages") or [])
            ]
            self.terms = [
                ContextTerm(
                    source=str(t.get("source") or ""),
                    target=str(t.get("target") or ""),
                    enabled=bool(t.get("enabled", True)),
                    origin=str(t.get("origin") or "slide"),
                )
                for t in (raw.get("terms") or [])
                if t.get("source")
            ]
            self.active_slide_index = int(raw.get("active_slide_index", 1 if self.pages else 0))
            self._rebuild_index_locked()
        except Exception:
            pass

    def _save_locked(self) -> None:
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "enabled": self.enabled,
                "filename": self.filename,
                "topic": self.topic,
                "active_slide_index": self.active_slide_index,
                "pages": [asdict(p) for p in self.pages],
                "terms": [asdict(t) for t in self.terms],
            }
            self.store_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def ingest_file(self, filename: str, data: bytes) -> dict[str, Any]:
        ext = Path(filename).suffix.lower()
        try:
            if ext == ".pptx":
                pages = _extract_pptx_pages(data)
            elif ext == ".docx":
                pages = _extract_docx_pages(data)
            elif ext == ".pdf":
                pages = _extract_pdf_pages(data)
            elif ext in (".txt", ".md", ".srt", ".csv"):
                pages = _extract_text_pages(data.decode("utf-8", errors="ignore"))
            else:
                raise ValueError(f"Định dạng '{ext}' chưa hỗ trợ. Hãy dùng .pptx, .pdf, .docx hoặc .txt.")
        except ValueError:
            raise
        except zipfile.BadZipFile as exc:
            raise ValueError(f"File '{Path(filename).name}' bị hỏng hoặc không đúng định dạng {ext.upper()}.") from exc
        except Exception as exc:
            raise ValueError(f"Không đọc được file '{Path(filename).name}' ({exc}). Hãy kiểm tra file hoặc xuất lại sang .pptx/.pdf/.txt.") from exc

        if not pages:
            raise ValueError("Không trích xuất được nội dung văn bản nào từ tài liệu (file có thể chỉ chứa ảnh chụp).")

        terms = _extract_terms_from_pages(pages)
        topic = _derive_topic_summary(filename, pages)

        with self._lock:
            self.enabled = True
            self.filename = Path(filename).name
            self.topic = topic
            self.pages = pages
            # Preserve any existing manual or DAU preset terms alongside newly extracted slide terms
            kept = [t for t in self.terms if t.origin in ("dau_preset", "manual")]
            seen = {t.source.lower() for t in kept}
            merged = list(kept)
            for t in terms:
                if t.source.lower() not in seen:
                    seen.add(t.source.lower())
                    merged.append(t)
            self.terms = merged[:80]
            self.active_slide_index = 1
            self._rebuild_index_locked()
            self._save_locked()
            return self.get_payload_locked()

    def ingest_text(
        self,
        text: Optional[str] = None,
        topic: Optional[str] = None,
        terms: Optional[list[dict[str, Any]]] = None,
        enabled: Optional[bool] = None,
        apply_dau_preset: bool = False,
        active_slide_index: Optional[int] = None,
    ) -> dict[str, Any]:
        with self._lock:
            if enabled is not None:
                self.enabled = bool(enabled)
            if text is not None:
                pages = _extract_text_pages(text)
                self.pages = pages
                self.filename = "Tài liệu dán trực tiếp" if pages else ""
                extracted = _extract_terms_from_pages(pages) if pages else []
                if not topic and pages:
                    self.topic = _derive_topic_summary("Hội nghị", pages)
                kept = [t for t in self.terms if t.origin in ("dau_preset", "manual")]
                seen = {t.source.lower() for t in kept}
                merged = list(kept)
                for t in extracted:
                    if t.source.lower() not in seen:
                        seen.add(t.source.lower())
                        merged.append(t)
                self.terms = merged[:80]
                self.active_slide_index = 1 if pages else 0
            if topic is not None:
                self.topic = _clean_line(topic)[:160]
            if terms is not None:
                parsed: list[ContextTerm] = []
                seen_t: set[str] = set()
                for item in terms:
                    src = _clean_line(str(item.get("source") or ""))
                    if not src or src.lower() in seen_t:
                        continue
                    seen_t.add(src.lower())
                    parsed.append(
                        ContextTerm(
                            source=src,
                            target=_clean_line(str(item.get("target") or "")),
                            enabled=bool(item.get("enabled", True)),
                            origin=str(item.get("origin") or "manual"),
                        )
                    )
                self.terms = parsed[:100]
            if apply_dau_preset:
                self.enabled = True
                if not self.topic:
                    self.topic = DAU_CONFERENCE_TOPIC
                seen_t = {t.source.lower() for t in self.terms}
                for src, tgt in DAU_CONFERENCE_TERMS:
                    if src.lower() not in seen_t:
                        seen_t.add(src.lower())
                        self.terms.append(ContextTerm(source=src, target=tgt, enabled=True, origin="dau_preset"))
            if active_slide_index is not None and 0 <= int(active_slide_index) <= len(self.pages):
                self.active_slide_index = int(active_slide_index)
            self._rebuild_index_locked()
            self._save_locked()
            return self.get_payload_locked()

    def clear(self) -> dict[str, Any]:
        with self._lock:
            self.filename = ""
            self.topic = ""
            self.pages = []
            self.terms = []
            self.active_slide_index = 0
            self._page_tokens = []
            self._idf = {}
            self._save_locked()
            return self.get_payload_locked()

    def get_payload(self) -> dict[str, Any]:
        with self._lock:
            return self.get_payload_locked()

    def get_payload_locked(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "filename": self.filename,
            "topic": self.topic,
            "slide_count": len(self.pages),
            "active_slide_index": self.active_slide_index,
            "active_slide_title": (
                self.pages[self.active_slide_index - 1].title
                if 1 <= self.active_slide_index <= len(self.pages)
                else ""
            ),
            "pages": [
                {
                    "index": p.index,
                    "title": p.title,
                    "preview": (p.text[:180] + "…") if len(p.text) > 180 else p.text,
                    "has_notes": bool(p.notes),
                    "terms": p.terms[:8],
                }
                for p in self.pages
            ],
            "terms": [asdict(t) for t in self.terms],
        }

    def get_hotwords(self) -> list[str]:
        """Return enabled Slide/Conference proper nouns & acronyms for ASR BPE hotword boosting."""
        with self._lock:
            if not self.enabled or not self.terms:
                return []
            out: list[str] = []
            seen: set[str] = set()
            for t in self.terms:
                if not t.enabled or not t.source:
                    continue
                low = t.source.lower()
                if low not in seen:
                    seen.add(low)
                    out.append(t.source)
                if t.target and t.target.lower() not in seen and len(t.target.split()) <= 4:
                    seen.add(t.target.lower())
                    out.append(t.target)
            return out[:48]

    def retrieve_for_utterance(
        self,
        utterance: str,
        src_lang: str = "vie_Latn",
        tgt_lang: str = "eng_Latn",
    ) -> Optional[SlideRagMatch]:
        """
        Sub-millisecond Slide-Aware RAG:
        1. Matches utterance against all slide pages (IDF overlap + progression bias).
        2. Selects up to 5 directly relevant terms from the active slide & session terms.
        3. Builds a compact <= 18-word domain hint for Hy-MT2 without risking context leaks.
        """
        clean = unicodedata.normalize("NFC", (utterance or "").strip())
        if not clean:
            return None

        with self._lock:
            if not self.enabled or (not self.pages and not self.topic and not self.terms):
                return None

            q_toks = _tokenize_for_rag(clean)
            q_low = clean.lower()
            q_fold = _strip_accents(q_low)

            best_idx = self.active_slide_index
            best_score = 0.0
            if self.pages and q_toks:
                cur = self.active_slide_index or 1
                for i, p_toks in enumerate(self._page_tokens):
                    shared = q_toks & p_toks
                    if not shared:
                        continue
                    base = sum(self._idf.get(t, 1.0) for t in shared) / max(len(q_toks), 1)
                    slide_no = i + 1
                    # Smooth progression prior: favor current slide and immediate next 2 slides,
                    # while keeping bonus modest so non-linear Q&A jumps back to any slide work reliably
                    if slide_no == cur:
                        bonus = 0.22
                    elif slide_no == cur + 1:
                        bonus = 0.18
                    elif slide_no == cur + 2:
                        bonus = 0.10
                    elif slide_no == cur - 1:
                        bonus = 0.06
                    else:
                        bonus = 0.0
                    score = base + bonus
                    if score > best_score:
                        best_score = score
                        best_idx = slide_no

                if best_score >= 0.45:
                    self.active_slide_index = best_idx

            active_page = (
                self.pages[self.active_slide_index - 1]
                if 1 <= self.active_slide_index <= len(self.pages)
                else None
            )

            # Collect ALL relevant MT term locks using word boundaries + strict target-language compatibility:
            # - Slide prose hotwords (`origin == "slide"` and `target == ""`) stay ASR/RAG-only so Title-Case
            #   Vietnamese slide headings are NEVER forced untranslated into English/Japanese/Korean output!
            # - Directional bilingual pairs (`Ban Giám hiệu -> Board of Rectors`) apply ONLY when the target
            #   language matches (`eng_Latn`), or in reverse when speaking `eng_Latn -> vie_Latn`.
            # - Universal acronyms, academic titles/names, and manual verbatim locks apply across ALL languages!
            raw_matches: list[dict[str, str]] = []
            for t in self.terms:
                if not t.enabled or not t.source:
                    continue
                eff_target = t.target or (t.source if t.origin == "manual" else "")
                if not eff_target:
                    continue

                s_low = t.source.lower()
                s_fold = _strip_accents(s_low)
                t_low = eff_target.lower()
                t_fold = _strip_accents(t_low)
                is_verbatim = s_low == t_low

                if is_verbatim:
                    if _term_word_match(s_low, s_fold, q_low, q_fold):
                        raw_matches.append({"source": t.source, "target": eff_target})
                    continue

                src_inferred = _infer_term_lang(t.source)
                tgt_inferred = _infer_term_lang(eff_target)

                # Forward: e.g. VI -> EN ("Ban Giám hiệu" -> "Board of Rectors")
                fwd_lang_ok = (not tgt_inferred) or (tgt_inferred == tgt_lang) or (
                    tgt_inferred == "zho_Hant" and tgt_lang.startswith("zho_")
                )
                if fwd_lang_ok and _term_word_match(s_low, s_fold, q_low, q_fold):
                    raw_matches.append({"source": t.source, "target": eff_target})
                    continue

                # Reverse: e.g. EN -> VI ("Smart City" -> "Đô thị thông minh")
                rev_lang_ok = (not src_inferred) or (src_inferred == tgt_lang) or (
                    src_inferred == "zho_Hant" and tgt_lang.startswith("zho_")
                )
                if rev_lang_ok and _term_word_match(t_low, t_fold, q_low, q_fold):
                    raw_matches.append({"source": eff_target, "target": t.source})

            # Rank longest phrase first and drop redundant sub-spans already covered by a longer match
            raw_matches.sort(key=lambda item: len(item["source"]), reverse=True)
            matched_terms: list[dict[str, str]] = []
            seen_src: list[str] = []
            for item in raw_matches:
                sl = item["source"].lower()
                if any(sl == prev or (len(sl) < len(prev) and sl in prev) for prev in seen_src):
                    continue
                seen_src.append(sl)
                matched_terms.append(item)
                if len(matched_terms) >= 8:
                    break

            # Build ultra-compact domain hint (<= 18 words) so Hy-MT2 stays fast & zero-leak
            hint_parts: list[str] = []
            if self.topic:
                hint_parts.append(self.topic)
            if active_page is not None and active_page.title:
                if active_page.title.lower() not in (self.topic or "").lower():
                    hint_parts.append(f"Slide {active_page.index}: {active_page.title}")
            hint_str = " | ".join(hint_parts)
            hint_words = hint_str.split()
            if len(hint_words) > 18:
                hint_str = " ".join(hint_words[:18])

            return SlideRagMatch(
                slide_index=active_page.index if active_page else 0,
                slide_title=active_page.title if active_page else "",
                score=round(best_score, 3),
                domain_hint=hint_str,
                matched_terms=matched_terms,
            )
