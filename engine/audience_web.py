"""
Self-contained Offline Web Views served directly by PolyVoice Pro Local Engine (v0.8):
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
    f'<span class="crest"><img src="{_LOGO_DATA_URI}" alt="PolyVoice Pro" /></span>'
    if _LOGO_DATA_URI
    else ""
)
_FAVICON_TAG = (
    f'<link rel="icon" type="image/png" href="{_LOGO_DATA_URI}" />'
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


AUDIENCE_HTML = (
    r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no, viewport-fit=cover" />
  <title>PolyVoice Pro — Phụ Đề Hội Nghị Trực Tiếp</title>
  __FAVICON__
  <style>
    :root, [data-theme="dark"] {
      --bg: #0b0d14;
      --bg-glow: radial-gradient(circle at 50% 0%, rgba(196, 0, 15, 0.14), transparent 52%);
      --header-bg: rgba(15, 18, 28, 0.94);
      --card: rgba(21, 25, 38, 0.92);
      --card-live: rgba(196, 0, 15, 0.14);
      --ctrl-bg: #191e2e;
      --ctrl-hover: #23293e;
      --border: rgba(148, 163, 184, 0.16);
      --border-strong: rgba(239, 68, 68, 0.42);
      --text: #f8fafc;
      --text-1: #d1d5db;
      --src-text: #cbd5e1;
      --tgt-live: #fecaca;
      --muted: #94a3b8;
      --brand: #c4000f;
      --accent: #ef4444;
      --accent-text: #fca5a5;
      --accent-soft: rgba(196, 0, 15, 0.18);
      --green: #34d399;
      --green-text: #6ee7b7;
      --green-soft: rgba(52, 211, 153, 0.13);
      --green-border: rgba(52, 211, 153, 0.34);
      --amber-text: #fbbf24;
      --amber-soft: rgba(251, 191, 36, 0.13);
      --amber-border: rgba(251, 191, 36, 0.34);
      --shadow-card: 0 6px 24px rgba(0, 0, 0, 0.32);
      color-scheme: dark;
    }
    [data-theme="light"] {
      --bg: #f4f4f6;
      --bg-glow: radial-gradient(circle at 50% 0%, rgba(196, 0, 15, 0.07), transparent 50%);
      --header-bg: rgba(255, 255, 255, 0.95);
      --card: #ffffff;
      --card-live: #fff5f5;
      --ctrl-bg: #f5f6f8;
      --ctrl-hover: #eceef2;
      --border: rgba(15, 23, 42, 0.095);
      --border-strong: rgba(196, 0, 15, 0.36);
      --text: #111827;
      --text-1: #374151;
      --src-text: #4b5563;
      --tgt-live: #b0000d;
      --muted: #6b7280;
      --brand: #c4000f;
      --accent: #c4000f;
      --accent-text: #b0000d;
      --accent-soft: rgba(196, 0, 15, 0.085);
      --green: #059669;
      --green-text: #047857;
      --green-soft: rgba(5, 150, 105, 0.1);
      --green-border: rgba(5, 150, 105, 0.28);
      --amber-text: #b45309;
      --amber-soft: rgba(217, 119, 6, 0.1);
      --amber-border: rgba(217, 119, 6, 0.3);
      --shadow-card: 0 2px 12px rgba(15, 23, 42, 0.06);
      color-scheme: light;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
    html, body {
      height: 100%;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Inter, sans-serif;
      -webkit-font-smoothing: antialiased;
      transition: background-color 0.2s ease, color 0.2s ease;
    }
    .lucide {
      display: inline-block;
      vertical-align: middle;
      flex-shrink: 0;
      fill: none;
      stroke: currentColor;
      stroke-width: 2;
      stroke-linecap: round;
      stroke-linejoin: round;
    }
    .app {
      display: flex;
      flex-direction: column;
      height: 100dvh;
      max-width: 860px;
      margin: 0 auto;
      background: var(--bg-glow), var(--bg);
    }

    /* ---------- Two-Tier Conference Mobile Header ---------- */
    header {
      position: sticky;
      top: 0;
      z-index: 20;
      background: var(--header-bg);
      border-top: 3px solid var(--brand);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      padding: calc(10px + env(safe-area-inset-top, 0px)) 14px 10px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .hdr-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      min-width: 0;
    }
    .crest {
      width: 36px;
      height: 36px;
      border-radius: 10px;
      background: #ffffff;
      border: 1px solid rgba(196, 0, 15, 0.28);
      padding: 3.5px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
      box-shadow: 0 2px 10px rgba(196, 0, 15, 0.16);
    }
    .crest img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }
    .brand-meta {
      display: flex;
      flex-direction: column;
      line-height: 1.18;
      min-width: 0;
    }
    .brand-row {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-weight: 800;
      font-size: 15px;
      letter-spacing: 0.01em;
      color: var(--text);
      white-space: nowrap;
    }
    .pro {
      padding: 1.5px 5.5px;
      border-radius: 5px;
      font-size: 9.5px;
      font-weight: 800;
      background: linear-gradient(135deg, #c4000f, #dc2626);
      color: #ffffff;
      letter-spacing: 0.07em;
    }
    .brand-sub {
      font-size: 11px;
      color: var(--muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .hdr-right {
      display: flex;
      align-items: center;
      gap: 7px;
      flex-shrink: 0;
    }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      height: 30px;
      padding: 0 10px;
      border-radius: 999px;
      font-size: 11.5px;
      font-weight: 600;
      background: var(--green-soft);
      border: 1px solid var(--green-border);
      color: var(--green-text);
      white-space: nowrap;
    }
    .status-pill.live {
      background: var(--accent-soft);
      border-color: var(--border-strong);
      color: var(--accent-text);
    }
    .status-pill.warn {
      background: var(--amber-soft);
      border-color: var(--amber-border);
      color: var(--amber-text);
    }
    .dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: currentColor;
      flex-shrink: 0;
    }
    .status-pill.live .dot {
      animation: pulse 1.2s ease-in-out infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.35; transform: scale(1.25); }
    }
    .icon-btn {
      width: 32px;
      height: 32px;
      border-radius: 9px;
      background: var(--ctrl-bg);
      color: var(--text-1);
      border: 1px solid var(--border);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      flex-shrink: 0;
      transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease;
    }
    .icon-btn:hover {
      background: var(--ctrl-hover);
      color: var(--text);
      border-color: var(--border-strong);
    }

    /* ---------- Row 2: Clean Control Strip ---------- */
    .hdr-bar {
      display: flex;
      align-items: center;
      gap: 7px;
      width: 100%;
    }
    .lang-wrap {
      position: relative;
      flex: 1;
      min-width: 0;
      height: 34px;
      padding: 0 10px;
      background: var(--ctrl-bg);
      border: 1px solid var(--border);
      border-radius: 9px;
      display: flex;
      align-items: center;
      gap: 7px;
      cursor: pointer;
      transition: border-color 0.15s ease, background-color 0.15s ease;
    }
    .lang-wrap:hover, .lang-wrap:focus-within {
      background: var(--ctrl-hover);
      border-color: var(--border-strong);
    }
    .lang-icon-left {
      color: var(--accent-text);
      display: inline-flex;
      flex-shrink: 0;
      pointer-events: none;
    }
    .lang-label {
      flex: 1;
      min-width: 0;
      font-size: 12.5px;
      font-weight: 600;
      color: var(--text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      pointer-events: none;
    }
    .lang-icon-right {
      color: var(--muted);
      display: inline-flex;
      flex-shrink: 0;
      pointer-events: none;
    }
    .lang-wrap select {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      opacity: 0;
      cursor: pointer;
      font-size: 16px;
    }
    .bar-tools {
      display: flex;
      align-items: center;
      gap: 6px;
      flex-shrink: 0;
    }
    .ctrl-btn {
      height: 34px;
      padding: 0 10px;
      border-radius: 9px;
      background: var(--ctrl-bg);
      color: var(--text-1);
      border: 1px solid var(--border);
      font-size: 12.5px;
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 5px;
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.15s ease;
    }
    .ctrl-btn:hover {
      background: var(--ctrl-hover);
      color: var(--text);
    }
    .ctrl-btn.active {
      background: var(--accent-soft);
      border-color: var(--border-strong);
      color: var(--accent-text);
    }
    .font-stepper {
      display: inline-flex;
      align-items: center;
      height: 34px;
      background: var(--ctrl-bg);
      border: 1px solid var(--border);
      border-radius: 9px;
      overflow: hidden;
    }
    .step-btn {
      width: 30px;
      height: 100%;
      border: none;
      background: transparent;
      color: var(--text-1);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: background-color 0.15s ease, color 0.15s ease;
    }
    .step-btn:hover {
      background: var(--ctrl-hover);
      color: var(--text);
    }
    .step-label {
      padding: 0 6px;
      font-size: 11.5px;
      font-weight: 700;
      color: var(--muted);
      border-left: 1px solid var(--border);
      border-right: 1px solid var(--border);
      height: 100%;
      display: inline-flex;
      align-items: center;
      user-select: none;
    }

    /* ---------- Subtitle Feed & Empty State ---------- */
    main {
      flex: 1;
      overflow-y: auto;
      padding: 16px 14px calc(24px + env(safe-area-inset-bottom, 0px));
      display: flex;
      flex-direction: column;
      gap: 12px;
      scroll-behavior: smooth;
    }
    .empty-card {
      margin: auto;
      width: 100%;
      max-width: 352px;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 28px 22px;
      text-align: center;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 12px;
      box-shadow: var(--shadow-card);
    }
    .empty-medallion {
      width: 56px;
      height: 56px;
      border-radius: 16px;
      background: var(--accent-soft);
      border: 1px solid var(--border-strong);
      color: var(--accent-text);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 0 6px rgba(196, 0, 15, 0.06);
    }
    .empty-title {
      font-size: 15.5px;
      font-weight: 700;
      color: var(--text);
    }
    .empty-desc {
      font-size: 13px;
      line-height: 1.55;
      color: var(--muted);
    }
    .empty-chip {
      margin-top: 2px;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 11px;
      border-radius: 999px;
      font-size: 11.5px;
      font-weight: 600;
      color: var(--green-text);
      background: var(--green-soft);
      border: 1px solid var(--green-border);
    }

    /* ---------- Transcript Cards ---------- */
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 13px 15px;
      display: flex;
      flex-direction: column;
      gap: 7px;
      box-shadow: var(--shadow-card);
    }
    .card.live {
      border-color: var(--border-strong);
      border-left: 3px solid var(--brand);
      background: var(--card-live);
    }
    .meta {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      font-size: 11.5px;
      color: var(--muted);
    }
    .meta-live {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      color: var(--accent-text);
      font-weight: 700;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 2px 8px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 11px;
      background: var(--green-soft);
      border: 1px solid var(--green-border);
      color: var(--green-text);
    }
    .badge.remote {
      background: var(--accent-soft);
      border-color: var(--border-strong);
      color: var(--accent-text);
    }
    .src {
      color: var(--src-text);
      line-height: 1.48;
    }
    .tgt {
      color: var(--text);
      font-weight: 700;
      line-height: 1.45;
    }
    .card.live .tgt {
      color: var(--tgt-live);
    }
    .jump {
      position: fixed;
      bottom: calc(18px + env(safe-area-inset-bottom, 0px));
      left: 50%;
      transform: translateX(-50%);
      background: linear-gradient(135deg, #c4000f, #dc2626);
      color: #ffffff;
      border: none;
      border-radius: 999px;
      padding: 9px 16px;
      font-size: 13px;
      font-weight: 700;
      box-shadow: 0 8px 24px rgba(196, 0, 15, 0.42);
      display: none;
      align-items: center;
      gap: 6px;
      cursor: pointer;
    }
    .jump.show { display: inline-flex; }

    @media (max-width: 390px) {
      header { padding-left: 10px; padding-right: 10px; }
      .ctrl-btn span.btn-label { display: none; }
      .ctrl-btn { padding: 0 9px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div class="hdr-top">
        <div class="brand">
          __LOGO_IMG__
          <div class="brand-meta">
            <div class="brand-row">
              <span>PolyVoice</span>
              <span class="pro">PRO</span>
            </div>
            <div class="brand-sub">Phụ đề Hội nghị Trực tiếp</div>
          </div>
        </div>
        <div class="hdr-right">
          <div class="status-pill" id="statusPill">
            <span class="dot"></span>
            <span id="statusText">Sẵn sàng</span>
          </div>
          <button class="icon-btn" id="btnTheme" title="Đổi giao diện Sáng / Tối" aria-label="Đổi giao diện Sáng / Tối"></button>
        </div>
      </div>

      <div class="hdr-bar">
        <div class="lang-wrap">
          <span class="lang-icon-left" aria-hidden="true">
            <svg class="lucide" width="14" height="14" viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="10"/>
              <path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/>
              <path d="M2 12h20"/>
            </svg>
          </span>
          <span class="lang-label" id="langLabel">Theo màn hình chính</span>
          <span class="lang-icon-right" aria-hidden="true">
            <svg class="lucide" width="14" height="14" viewBox="0 0 24 24">
              <path d="m6 9 6 6 6-6"/>
            </svg>
          </span>
          <select id="langSelect" title="Chọn ngôn ngữ phụ đề của bạn" aria-label="Chọn ngôn ngữ phụ đề">
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
        </div>

        <div class="bar-tools">
          <button id="btnBilingual" class="ctrl-btn active" title="Bật / tắt hiển thị câu gốc song ngữ">
            <svg class="lucide" width="14" height="14" viewBox="0 0 24 24">
              <rect width="18" height="14" x="3" y="5" rx="2" ry="2"/>
              <path d="M7 15h4M15 15h2M7 11h2M13 11h4"/>
            </svg>
            <span class="btn-label">Song ngữ</span>
          </button>

          <div class="font-stepper" role="group" aria-label="Cỡ chữ phụ đề">
            <button id="btnMinus" class="step-btn" title="Giảm cỡ chữ" aria-label="Giảm cỡ chữ">
              <svg class="lucide" width="13" height="13" viewBox="0 0 24 24">
                <path d="M5 12h14"/>
              </svg>
            </button>
            <span class="step-label" id="fontLabel">Aa</span>
            <button id="btnPlus" class="step-btn" title="Tăng cỡ chữ" aria-label="Tăng cỡ chữ">
              <svg class="lucide" width="13" height="13" viewBox="0 0 24 24">
                <path d="M5 12h14"/><path d="M12 5v14"/>
              </svg>
            </button>
          </div>
        </div>
      </div>
    </header>

    <main id="feed">
      <div class="empty-card" id="emptyState">
        <div class="empty-medallion" aria-hidden="true">
          <svg class="lucide" width="26" height="26" viewBox="0 0 24 24">
            <path d="M2 10v3"/><path d="M6 6v11"/><path d="M10 3v18"/>
            <path d="M14 8v7"/><path d="M18 5v13"/><path d="M22 10v3"/>
          </svg>
        </div>
        <div class="empty-title">Đã kết nối phòng hội nghị</div>
        <div class="empty-desc">
          Phụ đề trực tiếp sẽ tự động hiển thị tại đây ngay khi diễn giả bắt đầu nói. Bạn có thể chọn ngôn ngữ riêng ở thanh trên.
        </div>
        <div class="empty-chip">
          <svg class="lucide" width="13" height="13" viewBox="0 0 24 24">
            <path d="M12 20h.01"/><path d="M2 8.82a15 15 0 0 1 20 0"/>
            <path d="M5 12.859a10 10 0 0 1 14 0"/><path d="M8.5 16.429a5 5 0 0 1 7 0"/>
          </svg>
          <span>Mạng nội bộ • Không cần Internet</span>
        </div>
      </div>
    </main>

    <button class="jump" id="jumpBtn">
      <svg class="lucide" width="14" height="14" viewBox="0 0 24 24">
        <path d="M12 5v14"/><path d="m19 12-7 7-7-7"/>
      </svg>
      <span>Xuống câu mới nhất</span>
    </button>
  </div>

  <script>
    const ICON_SUN = '<svg class="lucide" width="16" height="16" viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/></svg>';
    const ICON_MOON = '<svg class="lucide" width="15" height="15" viewBox="0 0 24 24"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>';
    const ICON_MIC = '<svg class="lucide" width="11" height="11" viewBox="0 0 24 24"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg>';
    const ICON_HEADPHONES = '<svg class="lucide" width="11" height="11" viewBox="0 0 24 24"><path d="M3 14h3a2 2 0 0 1 2 2v3a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-7a9 9 0 0 1 18 0v7a2 2 0 0 1-2 2h-1a2 2 0 0 1-2-2v-3a2 2 0 0 1 2-2h3"/></svg>';
    const ICON_RADIO = '<svg class="lucide" width="13" height="13" viewBox="0 0 24 24"><path d="M4.9 19.1C1 15.2 1 8.8 4.9 4.9"/><path d="M7.8 16.2c-2.3-2.3-2.3-6.1 0-8.5"/><circle cx="12" cy="12" r="2"/><path d="M16.2 7.8c2.3 2.3 2.3 6.1 0 8.5"/><path d="M19.1 4.9C23 8.8 23 15.1 19.1 19"/></svg>';

    const feed = document.getElementById("feed");
    const emptyState = document.getElementById("emptyState");
    const statusPill = document.getElementById("statusPill");
    const statusText = document.getElementById("statusText");
    const langSelect = document.getElementById("langSelect");
    const langLabel = document.getElementById("langLabel");
    const btnBilingual = document.getElementById("btnBilingual");
    const btnTheme = document.getElementById("btnTheme");
    const jumpBtn = document.getElementById("jumpBtn");

    function setAudTheme(t) {
      document.documentElement.setAttribute("data-theme", t);
      btnTheme.innerHTML = t === "light" ? ICON_MOON : ICON_SUN;
      try { localStorage.setItem("voicerecore-aud-theme", t); } catch {}
    }
    setAudTheme((() => { try { return localStorage.getItem("voicerecore-aud-theme") || "dark"; } catch { return "dark"; } })());
    btnTheme.addEventListener("click", () => {
      const cur = document.documentElement.getAttribute("data-theme") || "dark";
      setAudTheme(cur === "dark" ? "light" : "dark");
    });

    function setConnState(mode) {
      statusPill.classList.remove("live", "warn");
      if (mode === "live") {
        statusPill.classList.add("live");
        statusText.textContent = "Đang phát";
      } else if (mode === "warn") {
        statusPill.classList.add("warn");
        statusText.textContent = "Kết nối lại...";
      } else {
        statusText.textContent = "Sẵn sàng";
      }
    }

    let fontSize = 20;
    let showSource = true;
    let selectedLang = new URLSearchParams(location.search).get("lang") || "follow";
    if ([...langSelect.options].some(o => o.value === selectedLang)) langSelect.value = selectedLang;
    function syncLangLabel() {
      const opt = langSelect.options[langSelect.selectedIndex];
      if (langLabel && opt) langLabel.textContent = opt.textContent;
    }
    syncLangLabel();

    let entries = [];
    let partial = null;
    let pinnedBottom = true;
    let ws = null;
    let reconnectTimer = null;

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
          ? `<span class="badge remote">${ICON_HEADPHONES}<span>Đối tác</span></span>`
          : item.speaker === "self" ? `<span class="badge">${ICON_MIC}<span>Diễn giả</span></span>` : "";
        html += `<div class="card">
          <div class="meta"><span>${escapeHtml(item.timestamp || "")}</span>${spkBadge}</div>
          ${showSource ? `<div class="src" style="font-size:${Math.max(13, fontSize - 5)}px">${escapeHtml(item.source_text)}</div>` : ""}
          <div class="tgt" style="font-size:${fontSize}px">${escapeHtml(item.translated_text || "—")}</div>
        </div>`;
      }
      if (partial) {
        html += `<div class="card live">
          <div class="meta"><span class="meta-live">${ICON_RADIO}<span>Đang nói...</span></span></div>
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
      syncLangLabel();
      entries = [];
      partial = null;
      render();
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      if (ws) {
        ws.onclose = null;
        ws.close();
        ws = null;
      }
      connect();
    });

    function connect() {
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        ws.onclose = null;
        ws.close();
      }
      const proto = location.protocol === "https:" ? "wss:" : "ws:";
      const sock = new WebSocket(`${proto}//${location.host}/ws/audience?lang=${encodeURIComponent(selectedLang)}`);
      ws = sock;
      sock.onopen = () => {
        if (ws === sock) setConnState("ready");
      };
      sock.onmessage = (ev) => {
        if (ws !== sock) return;
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === "status") {
            setConnState(msg.data && msg.data.listening ? "live" : "ready");
          } else if (msg.type === "history") {
            entries = (msg.data || []).slice(-120);
            if (entries.length === 0) partial = null;
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
      sock.onclose = () => {
        if (ws !== sock) return;
        setConnState("warn");
        reconnectTimer = setTimeout(connect, 1200);
      };
    }
    connect();
  </script>
</body>
</html>
""".replace("__LOGO_IMG__", _LOGO_IMG_TAG).replace("__FAVICON__", _FAVICON_TAG)
)


OBS_HTML = (
    r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <title>PolyVoice Pro — OBS Transparent Subtitles</title>
  __FAVICON__
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
            const same = m.data.id === activeId;
            activeId = m.data.id;
            show(m.data.source_text, same ? tgtEl.textContent : "...", true);
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
""".replace("__FAVICON__", _FAVICON_TAG)
)
