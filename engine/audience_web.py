"""
Self-contained Offline Web Views served directly by VOICERECORE Local Engine (v0.6):
1. `/audience` — Mobile-first LAN Audience View (Wordly / Google Live Translate style).
   Attendees scan the QR code over local Wi-Fi (ZERO Internet required!) and watch
   live subtitles on their own phones, even choosing their own preferred target language!
2. `/obs` — Transparent Lower-Third Browser Source for OBS Studio / vMix / Livestreaming.
"""

from __future__ import annotations

import base64
import socket
import sys
from pathlib import Path


def _load_school_logo_data_uri() -> str:
    here = Path(__file__).resolve().parent
    exe_dir = Path(sys.executable).resolve().parent
    for p in (
        here / "school_logo.png",
        here.parent / "public" / "school-logo.png",
        exe_dir / "school_logo.png",
    ):
        if p.exists() and p.is_file():
            try:
                b64 = base64.b64encode(p.read_bytes()).decode("ascii")
                return f"data:image/png;base64,{b64}"
            except Exception:
                pass
    return ""


_LOGO_DATA_URI = _load_school_logo_data_uri()
_LOGO_IMG_TAG = (
    f'<span class="crest"><img src="{_LOGO_DATA_URI}" alt="Logo" /></span>'
    if _LOGO_DATA_URI
    else ""
)


def get_local_lan_ip() -> str:
    """Detect the machine's active LAN IPv4 address for local Wi-Fi QR sharing."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
    except Exception:
        pass
    try:
        hostname_ip = socket.gethostbyname(socket.gethostname())
        if hostname_ip and not hostname_ip.startswith("127."):
            return hostname_ip
    except Exception:
        pass
    return "127.0.0.1"


AUDIENCE_HTML = (r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no" />
  <title>PolyVoice Pro — Phụ Đề Hội Nghị Trực Tiếp</title>
  <style>
    :root, [data-theme="dark"] {
      --bg: #0b0d14;
      --header-bg: rgba(16, 19, 29, 0.94);
      --card: #141824;
      --card-live: rgba(196, 0, 15, 0.16);
      --ctrl-bg: #1c2131;
      --border: rgba(148, 163, 184, 0.2);
      --text: #f8fafc;
      --src-text: #cbd5e1;
      --tgt-live: #fecaca;
      --muted: #94a3b8;
      --accent: #ef4444;
      --green: #34d399;
      color-scheme: dark;
    }
    [data-theme="light"] {
      --bg: #f4f4f6;
      --header-bg: rgba(255, 255, 255, 0.95);
      --card: #ffffff;
      --card-live: #fff5f5;
      --ctrl-bg: #f9fafb;
      --border: rgba(15, 23, 42, 0.11);
      --text: #111827;
      --src-text: #4b5563;
      --tgt-live: #b0000d;
      --muted: #6b7280;
      --accent: #c4000f;
      --green: #059669;
      color-scheme: light;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body {
      height: 100%; background: var(--bg); color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      -webkit-font-smoothing: antialiased;
      transition: background 0.2s ease, color 0.2s ease;
    }
    .app { display: flex; flex-direction: column; height: 100dvh; max-width: 960px; margin: 0 auto; }
    header {
      padding: 11px 14px; background: var(--header-bg);
      border-bottom: 1px solid var(--border);
      display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px;
      position: sticky; top: 0; z-index: 20;
      backdrop-filter: blur(12px);
    }
    .brand { display: flex; align-items: center; gap: 8px; font-weight: 800; font-size: 14px; letter-spacing: 0.02em; }
    .crest {
      width: 28px; height: 28px; border-radius: 7px; background: #fff;
      border: 1px solid rgba(196, 0, 15, 0.28); padding: 2px; display: inline-flex;
      align-items: center; justify-content: center; flex-shrink: 0;
    }
    .crest img { width: 100%; height: 100%; object-fit: contain; display: block; }
    .pro {
      padding: 1.5px 5px; border-radius: 4px; font-size: 9.5px; font-weight: 800;
      background: linear-gradient(135deg, #c4000f, #dc2626); color: #fff; letter-spacing: 0.06em;
    }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); }
    .dot.live { background: #c4000f; animation: pulse 1.2s infinite; }
    @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }
    .controls { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
    select, button {
      background: var(--ctrl-bg); color: var(--text); border: 1px solid var(--border);
      border-radius: 8px; padding: 6px 10px; font-size: 13px; font-weight: 600; cursor: pointer;
    }
    button.active { background: rgba(196, 0, 15, 0.14); border-color: var(--accent); color: var(--accent); }
    main {
      flex: 1; overflow-y: auto; padding: 14px; display: flex; flex-direction: column; gap: 12px;
      scroll-behavior: smooth;
    }
    .empty {
      margin: auto; text-align: center; color: var(--muted); font-size: 15px; line-height: 1.6; max-width: 420px;
    }
    .card {
      background: var(--card); border: 1px solid var(--border); border-radius: 14px;
      padding: 14px 16px; display: flex; flex-direction: column; gap: 8px;
      box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
    }
    .card.live { border-color: var(--accent); background: var(--card-live); }
    .meta { display: flex; align-items: center; justify-content: space-between; font-size: 11.5px; color: var(--muted); }
    .badge {
      padding: 2px 7px; border-radius: 999px; font-weight: 700; font-size: 11px;
      background: rgba(5, 150, 105, 0.14); color: var(--green);
    }
    .badge.remote { background: rgba(196, 0, 15, 0.14); color: var(--accent); }
    .src { color: var(--src-text); line-height: 1.45; }
    .tgt { color: var(--text); font-weight: 700; line-height: 1.45; }
    .card.live .tgt { color: var(--tgt-live); }
    .jump {
      position: fixed; bottom: 18px; left: 50%; transform: translateX(-50%);
      background: linear-gradient(135deg, #c4000f, #dc2626); color: #fff; border: none; border-radius: 999px;
      padding: 9px 18px; font-weight: 700; box-shadow: 0 8px 24px rgba(196, 0, 15, 0.42);
      display: none;
    }
    .jump.show { display: inline-flex; }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div class="brand">
        __LOGO_IMG__
        <span class="dot" id="statusDot"></span>
        <span>PolyVoice</span>
        <span class="pro">PRO</span>
      </div>
      <div class="controls">
        <select id="langSelect" title="Chọn ngôn ngữ phụ đề của bạn">
          <option value="follow">Theo màn hình chính</option>
          <option value="eng_Latn">English (Tiếng Anh)</option>
          <option value="tha_Thai">ไทย (Tiếng Thái)</option>
          <option value="lao_Laoo">ລາວ (Tiếng Lào)</option>
          <option value="khm_Khmr">ភាសាខ្មែរ (Khmer)</option>
          <option value="vie_Latn">Tiếng Việt</option>
          <option value="zho_Hans">简体中文 (Tiếng Trung)</option>
          <option value="jpn_Jpan">日本語 (Tiếng Nhật)</option>
          <option value="kor_Hang">한국어 (Tiếng Hàn)</option>
          <option value="fra_Latn">Français (Tiếng Pháp)</option>
          <option value="deu_Latn">Deutsch (Tiếng Đức)</option>
          <option value="spa_Latn">Español (Tây Ban Nha)</option>
          <option value="rus_Cyrl">Русский (Tiếng Nga)</option>
        </select>
        <button id="btnBilingual" class="active">Song ngữ</button>
        <button id="btnMinus">A-</button>
        <button id="btnPlus">A+</button>
        <button id="btnTheme" title="Đổi giao diện Sáng / Tối">☀️</button>
      </div>
    </header>
    <main id="feed">
      <div class="empty" id="emptyState">
        Đã kết nối tới phòng thuyết trình.<br/>
        Phụ đề trực tiếp sẽ tự động hiển thị tại đây ngay khi diễn giả bắt đầu nói...
      </div>
    </main>
    <button class="jump" id="jumpBtn">↓ Xuống câu mới nhất</button>
  </div>

  <script>
    const feed = document.getElementById("feed");
    const emptyState = document.getElementById("emptyState");
    const statusDot = document.getElementById("statusDot");
    const langSelect = document.getElementById("langSelect");
    const btnBilingual = document.getElementById("btnBilingual");
    const btnTheme = document.getElementById("btnTheme");
    const jumpBtn = document.getElementById("jumpBtn");

    function setAudTheme(t) {
      document.documentElement.setAttribute("data-theme", t);
      btnTheme.textContent = t === "light" ? "🌙" : "☀️";
      try { localStorage.setItem("voicerecore-aud-theme", t); } catch {}
    }
    setAudTheme((() => { try { return localStorage.getItem("voicerecore-aud-theme") || "dark"; } catch { return "dark"; } })());
    btnTheme.addEventListener("click", () => {
      const cur = document.documentElement.getAttribute("data-theme") || "dark";
      setAudTheme(cur === "dark" ? "light" : "dark");
    });

    let fontSize = 20;
    let showSource = true;
    let selectedLang = new URLSearchParams(location.search).get("lang") || "follow";
    if ([...langSelect.options].some(o => o.value === selectedLang)) langSelect.value = selectedLang;

    let entries = [];
    let partial = null;
    let pinnedBottom = true;
    let ws = null;

    function groupByParagraph(list) {
      const out = [];
      for (const item of list) {
        const prev = out[out.length - 1];
        const same = prev && item.paragraph_id && prev.paragraph_id === item.paragraph_id
          && prev.speaker === item.speaker && prev.target_lang === item.target_lang;
        if (same) {
          prev.source_text = (prev.source_text + " " + item.source_text).trim();
          prev.translated_text = (prev.translated_text + " " + (item.translated_text || "")).trim();
        } else {
          out.push({ ...item });
        }
      }
      return out;
    }

    function escapeHtml(s) {
      return (s || "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }

    function render() {
      const grouped = groupByParagraph(entries);
      if (grouped.length === 0 && !partial) {
        feed.innerHTML = "";
        feed.appendChild(emptyState);
        return;
      }
      let html = "";
      for (const item of grouped) {
        const spkBadge = item.speaker === "remote"
          ? `<span class="badge remote">Đối tác</span>`
          : item.speaker === "self" ? `<span class="badge">Diễn giả</span>` : "";
        html += `<div class="card">
          <div class="meta"><span>${escapeHtml(item.timestamp || "")}</span>${spkBadge}</div>
          ${showSource ? `<div class="src" style="font-size:${Math.max(13, fontSize - 5)}px">${escapeHtml(item.source_text)}</div>` : ""}
          <div class="tgt" style="font-size:${fontSize}px">${escapeHtml(item.translated_text || "—")}</div>
        </div>`;
      }
      if (partial) {
        html += `<div class="card live">
          <div class="meta"><span>● Đang nói...</span></div>
          ${showSource ? `<div class="src" style="font-size:${Math.max(13, fontSize - 5)}px;font-style:italic">${escapeHtml(partial.source_text)}</div>` : ""}
          <div class="tgt" style="font-size:${fontSize}px">${escapeHtml(partial.translated_text || "...")}</div>
        </div>`;
      }
      feed.innerHTML = html;
      if (pinnedBottom) feed.scrollTop = feed.scrollHeight;
    }

    feed.addEventListener("scroll", () => {
      const dist = feed.scrollHeight - feed.scrollTop - feed.clientHeight;
      pinnedBottom = dist < 72;
      jumpBtn.classList.toggle("show", !pinnedBottom);
    });
    jumpBtn.addEventListener("click", () => {
      pinnedBottom = true;
      jumpBtn.classList.remove("show");
      feed.scrollTop = feed.scrollHeight;
    });
    btnBilingual.addEventListener("click", () => {
      showSource = !showSource;
      btnBilingual.classList.toggle("active", showSource);
      render();
    });
    document.getElementById("btnMinus").addEventListener("click", () => {
      fontSize = Math.max(14, fontSize - 2); render();
    });
    document.getElementById("btnPlus").addEventListener("click", () => {
      fontSize = Math.min(40, fontSize + 2); render();
    });
    langSelect.addEventListener("change", () => {
      selectedLang = langSelect.value;
      entries = [];
      partial = null;
      render();
      if (ws) ws.close();
    });

    function connect() {
      const proto = location.protocol === "https:" ? "wss:" : "ws:";
      ws = new WebSocket(`${proto}//${location.host}/ws/audience?lang=${encodeURIComponent(selectedLang)}`);
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === "status") {
            statusDot.classList.toggle("live", Boolean(msg.data && msg.data.listening));
          } else if (msg.type === "history") {
            entries = (msg.data || []).slice(-120);
            render();
          } else if (msg.type === "partial") {
            const d = msg.data;
            partial = {
              id: d.id, source_text: d.source_text,
              translated_text: partial && partial.id === d.id ? partial.translated_text : ""
            };
            render();
          } else if (msg.type === "partial_mt") {
            const d = msg.data;
            if (partial && partial.id === d.id) {
              partial.translated_text = d.translated_text;
              render();
            }
          } else if (msg.type === "final") {
            const e = msg.data;
            if (partial && partial.id === e.id) partial = null;
            if (!entries.some(x => x.id === e.id)) {
              entries.push(e);
              if (entries.length > 150) entries = entries.slice(-150);
            }
            render();
          }
        } catch {}
      };
      ws.onclose = () => setTimeout(connect, 1200);
    }
    connect();
  </script>
</body>
</html>
""".replace("__LOGO_IMG__", _LOGO_IMG_TAG))


OBS_HTML = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <title>PolyVoice Pro — OBS Transparent Subtitles</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body {
      width: 100%; height: 100%; background: transparent !important; overflow: hidden;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .stage {
      width: 100%; height: 100%; display: flex; flex-direction: column;
      justify-content: flex-end; align-items: center; padding: 24px 40px;
    }
    .box {
      max-width: 92%; text-align: center;
      background: rgba(11, 13, 20, 0.84);
      border: 1px solid rgba(196, 0, 15, 0.5);
      border-radius: 14px; padding: 14px 24px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.55);
      transition: opacity 0.35s ease;
    }
    .box.hidden { opacity: 0; }
    .src { color: #cbd5e1; font-weight: 500; margin-bottom: 6px; text-shadow: 0 2px 6px rgba(0,0,0,0.85); }
    .tgt { color: #ffffff; font-weight: 800; line-height: 1.35; text-shadow: 0 2px 8px rgba(0,0,0,0.9); }
    .tgt.draft { color: #fecaca; }
  </style>
</head>
<body>
  <div class="stage">
    <div class="box hidden" id="box">
      <div class="src" id="src"></div>
      <div class="tgt" id="tgt"></div>
    </div>
  </div>
  <script>
    const qp = new URLSearchParams(location.search);
    const size = Math.max(18, Math.min(64, Number(qp.get("size") || 32)));
    const showSrc = qp.get("bilingual") !== "0";
    const lang = qp.get("lang") || "follow";
    const box = document.getElementById("box");
    const srcEl = document.getElementById("src");
    const tgtEl = document.getElementById("tgt");
    srcEl.style.fontSize = Math.round(size * 0.65) + "px";
    tgtEl.style.fontSize = size + "px";
    if (!showSrc) srcEl.style.display = "none";

    let hideTimer = null;
    let activeId = "";
    function show(src, tgt, isDraft) {
      if (!src && !tgt) return;
      srcEl.textContent = src || "";
      tgtEl.textContent = tgt || "...";
      tgtEl.classList.toggle("draft", Boolean(isDraft));
      box.classList.remove("hidden");
      if (hideTimer) clearTimeout(hideTimer);
      hideTimer = setTimeout(() => box.classList.add("hidden"), 12000);
    }
    function connect() {
      const proto = location.protocol === "https:" ? "wss:" : "ws:";
      const ws = new WebSocket(`${proto}//${location.host}/ws/audience?lang=${encodeURIComponent(lang)}`);
      ws.onmessage = (ev) => {
        try {
          const m = JSON.parse(ev.data);
          if (m.type === "partial") {
            activeId = m.data.id;
            show(m.data.source_text, tgtEl.textContent === "..." ? "..." : tgtEl.textContent, true);
          } else if (m.type === "partial_mt" && m.data.id === activeId) {
            show(srcEl.textContent, m.data.translated_text, true);
          } else if (m.type === "final") {
            activeId = "";
            show(m.data.source_text, m.data.translated_text, false);
          }
        } catch {}
      };
      ws.onclose = () => setTimeout(connect, 1200);
    }
    connect();
  </script>
</body>
</html>
"""
