"""
Self-contained Offline Web Views served directly by DAU Voice Pro Local Engine:
1. `/audience` — Mobile-first LAN Audience View (Wordly / Google Live Translate style).
   Attendees scan the QR code over local Wi-Fi (ZERO Internet required!) and watch
   live subtitles on their own phones, even choosing their own preferred target language!
2. `/obs` — Transparent Lower-Third Browser Source for OBS Studio / vMix / Livestreaming.
"""

from __future__ import annotations

import base64
import ipaddress
import os
import re
import socket
import subprocess
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
    f'<span class="crest"><img src="{_LOGO_DATA_URI}" alt="DAU Voice Pro" /></span>'
    if _LOGO_DATA_URI
    else ""
)
_FAVICON_TAG = (
    f'<link rel="icon" type="image/png" href="{_LOGO_DATA_URI}" />'
    if _LOGO_DATA_URI
    else ""
)


def _is_usable_lan_ipv4(ip: str) -> bool:
    try:
        addr = ipaddress.IPv4Address(ip)
    except Exception:
        return False
    return not (addr.is_loopback or addr.is_link_local or addr.is_unspecified or addr.is_multicast)


def _is_rfc1918_ipv4(ip: str) -> bool:
    try:
        addr = ipaddress.IPv4Address(ip)
    except Exception:
        return False
    return addr.is_private and not (addr.is_loopback or addr.is_link_local or addr.is_unspecified)


def get_local_lan_ip() -> str:
    """Detect the machine's active LAN IPv4 address for local Wi-Fi QR sharing (works 100% air-gapped offline)."""
    # 1) Query OS routing table via UDP connect (sends 0 packets); try Internet + RFC1918 LAN targets
    #    so air-gapped conference routers with no default WAN gateway still yield the Wi-Fi LAN IP!
    for target in (
        ("8.8.8.8", 80),
        ("10.255.255.255", 1),
        ("192.168.1.1", 80),
        ("192.168.0.1", 80),
        ("172.16.0.1", 80),
    ):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(target)
                ip = s.getsockname()[0]
                if _is_usable_lan_ipv4(ip):
                    return ip
        except Exception:
            continue

    # 2) Hostname / mDNS (.local on macOS) & adapter list resolution
    candidates: list[str] = []
    try:
        host = socket.gethostname()
        for h in (host, f"{host}.local"):
            try:
                for info in socket.getaddrinfo(h, None, socket.AF_INET):
                    ip = info[4][0]
                    if _is_usable_lan_ipv4(ip) and ip not in candidates:
                        candidates.append(ip)
            except Exception:
                pass
        try:
            for ip in socket.gethostbyname_ex(host)[2]:
                if _is_usable_lan_ipv4(ip) and ip not in candidates:
                    candidates.append(ip)
        except Exception:
            pass
    except Exception:
        pass

    # 3) Air-gapped interface scan fallback (macOS/Linux `ifconfig` / `ip -4 addr`)
    if not candidates and os.name != "nt":
        for cmd in (["ifconfig"], ["ip", "-4", "-o", "addr", "show"]):
            try:
                out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=1.5)
                for ip in re.findall(r"\binet\s+(\d+\.\d+\.\d+\.\d+)", out):
                    if _is_usable_lan_ipv4(ip) and ip not in candidates:
                        candidates.append(ip)
                if candidates:
                    break
            except Exception:
                continue

    for ip in candidates:
        if _is_rfc1918_ipv4(ip):
            return ip
    if candidates:
        return candidates[0]
    return "127.0.0.1"


AUDIENCE_HTML = (
    r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no, viewport-fit=cover" />
  <title>DAU Voice Pro — Phụ Đề Hội Nghị Trực Tiếp</title>
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
      background: transparent;
      border: none;
      padding: 0;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }
    .crest img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      filter: drop-shadow(0 1px 4px rgba(196, 0, 15, 0.28));
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
    .badge.spk2 {
      background: rgba(56, 189, 248, 0.14);
      border-color: rgba(56, 189, 248, 0.38);
      color: #38bdf8;
    }
    html[data-theme="light"] .badge.spk2 {
      background: rgba(2, 132, 199, 0.10);
      border-color: rgba(2, 132, 199, 0.28);
      color: #0369a1;
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
    .card.latest {
      border-color: var(--border-strong);
      border-left: 3.5px solid var(--brand);
    }
    .live-footer {
      flex-shrink: 0;
      border-top: 1px solid var(--border);
      background: var(--header-bg);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      padding: 9px 14px calc(10px + env(safe-area-inset-bottom, 0px));
      display: none;
      align-items: center;
      gap: 9px;
      min-height: 40px;
    }
    .live-footer.show { display: flex; }
    .live-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #ef4444;
      box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.22);
      flex-shrink: 0;
      animation: audPulse 1.4s ease-in-out infinite;
    }
    @keyframes audPulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.45; transform: scale(0.86); }
    }
    .live-src-text {
      font-size: 13px;
      color: var(--muted);
      font-style: italic;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      flex: 1;
      min-width: 0;
    }
    .freeze-overlay {
      position: fixed;
      inset: 0;
      z-index: 80;
      background: rgba(8, 10, 16, 0.94);
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      display: none;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 14px;
      padding: 28px;
      text-align: center;
    }
    .freeze-overlay.show { display: flex; }
    .pin-overlay {
      position: fixed;
      inset: 0;
      z-index: 95;
      background: rgba(8, 10, 16, 0.92);
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      display: none;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }
    .pin-overlay.show { display: flex; }
    .pin-card {
      width: min(360px, 100%);
      background: var(--card);
      border: 1px solid var(--border-strong);
      border-radius: 18px;
      padding: 24px 20px;
      text-align: center;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 12px;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.45);
    }
    .pin-icon {
      width: 48px;
      height: 48px;
      border-radius: 14px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      background: var(--accent-soft);
      border: 1px solid var(--border-strong);
      color: var(--accent-text);
    }
    .pin-title {
      font-size: 17px;
      font-weight: 800;
      color: var(--text);
    }
    .pin-sub {
      font-size: 13px;
      line-height: 1.5;
      color: var(--muted);
    }
    .pin-form {
      width: 100%;
      display: flex;
      gap: 8px;
      margin-top: 4px;
    }
    .pin-input {
      flex: 1;
      height: 44px;
      border-radius: 11px;
      border: 1px solid var(--border-strong);
      background: var(--bg);
      color: var(--text);
      font-size: 20px;
      font-weight: 800;
      letter-spacing: 0.35em;
      text-align: center;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      outline: none;
    }
    .pin-input:focus {
      border-color: var(--brand);
      box-shadow: 0 0 0 3px rgba(196, 0, 15, 0.22);
    }
    .pin-btn {
      height: 44px;
      padding: 0 18px;
      border: none;
      border-radius: 11px;
      background: linear-gradient(135deg, #c4000f, #dc2626);
      color: #ffffff;
      font-size: 13.5px;
      font-weight: 700;
      cursor: pointer;
      white-space: nowrap;
    }
    .pin-err {
      font-size: 12.5px;
      font-weight: 600;
      color: #f87171;
      min-height: 18px;
    }
    .freeze-title {
      font-size: 20px;
      font-weight: 800;
      color: #f8fafc;
      letter-spacing: -0.02em;
    }
    .freeze-msg {
      max-width: 420px;
      font-size: 15px;
      line-height: 1.55;
      color: #cbd5e1;
    }
    .jump {
      position: fixed;
      bottom: calc(62px + env(safe-area-inset-bottom, 0px));
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

    @media (max-width: 340px) {
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
              <span>DAU Voice</span>
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
            <option value="zho_Hant">繁體中文 (Đài Loan)</option>
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
          <span id="emptyChipText">Mạng nội bộ • Không cần Internet</span>
        </div>
      </div>
    </main>

    <div class="live-footer" id="liveFooter" aria-live="polite">
      <span class="live-dot" aria-hidden="true"></span>
      <span class="live-src-text" id="liveSrcText"></span>
    </div>

    <div class="freeze-overlay" id="freezeOverlay" role="status" aria-live="assertive">
      __LOGO_IMG__
      <div class="freeze-title">DAU Voice Pro</div>
      <div class="freeze-msg" id="freezeMsg">Tạm nghỉ — Phiên dịch hội nghị sẽ tiếp tục trong giây lát</div>
    </div>

    <div class="pin-overlay" id="pinOverlay" role="dialog" aria-modal="true" aria-labelledby="pinTitle">
      <div class="pin-card">
        <div class="pin-icon" aria-hidden="true">
          <svg class="lucide" width="22" height="22" viewBox="0 0 24 24">
            <rect width="18" height="11" x="3" y="11" rx="2" ry="2"/>
            <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
          </svg>
        </div>
        <div class="pin-title" id="pinTitle">Phòng hội nghị có mã PIN</div>
        <div class="pin-sub">
          Nhập mã PIN 4 số từ ban tổ chức hoặc quét lại mã QR mới nhất trên màn hình hội trường.
        </div>
        <form class="pin-form" id="pinForm">
          <input
            id="pinInput"
            class="pin-input"
            type="text"
            inputmode="numeric"
            pattern="[0-9]{4}"
            maxlength="4"
            autocomplete="one-time-code"
            placeholder="0000"
            aria-label="Mã PIN 4 số"
          />
          <button type="submit" class="pin-btn">Vào phòng</button>
        </form>
        <div class="pin-err" id="pinErr"></div>
      </div>
    </div>

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
    const liveFooter = document.getElementById("liveFooter");
    const liveSrcText = document.getElementById("liveSrcText");
    const freezeOverlay = document.getElementById("freezeOverlay");
    const freezeMsg = document.getElementById("freezeMsg");
    const emptyChipText = document.getElementById("emptyChipText");
    const pinOverlay = document.getElementById("pinOverlay");
    const pinForm = document.getElementById("pinForm");
    const pinInput = document.getElementById("pinInput");
    const pinErr = document.getElementById("pinErr");

    if (emptyChipText && location.protocol === "https:") {
      emptyChipText.textContent = "Kết nối HTTPS bảo mật • Hoạt động qua 4G/5G & Wi-Fi";
    }

    function setFrozen(frozen, msg) {
      if (msg) freezeMsg.textContent = msg;
      freezeOverlay.classList.toggle("show", Boolean(frozen));
    }

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
    const urlParams = new URLSearchParams(location.search);
    let selectedLang = urlParams.get("lang") || "follow";
    let roomPin = (urlParams.get("pin") || "").trim();
    if ([...langSelect.options].some(o => o.value === selectedLang)) {
      langSelect.value = selectedLang;
    } else {
      selectedLang = langSelect.value || "follow";
    }
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
    let heartbeatTimer = null;
    let pinBlocked = false;

    function showPinPrompt(errText) {
      pinBlocked = true;
      if (pinErr) pinErr.textContent = errText || "";
      if (pinOverlay) pinOverlay.classList.add("show");
      if (pinInput) {
        pinInput.value = "";
        setTimeout(() => pinInput.focus(), 60);
      }
    }
    function hidePinPrompt() {
      pinBlocked = false;
      if (pinErr) pinErr.textContent = "";
      if (pinOverlay) pinOverlay.classList.remove("show");
    }
    if (pinForm) {
      pinForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const val = (pinInput ? pinInput.value : "").replace(/\D/g, "").slice(0, 4);
        if (val.length !== 4) {
          if (pinErr) pinErr.textContent = "Vui lòng nhập đủ 4 chữ số.";
          return;
        }
        roomPin = val;
        hidePinPrompt();
        connect();
      });
    }

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
      return String(s ?? "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }

    function renderLiveBar() {
      const txt = (partial && partial.source_text ? partial.source_text : "").trim();
      if (txt) {
        liveSrcText.textContent = txt;
        liveFooter.classList.add("show");
      } else {
        liveFooter.classList.remove("show");
      }
    }

    function render() {
      renderLiveBar();
      const grouped = groupByParagraph(entries);
      if (grouped.length === 0) {
        feed.innerHTML = "";
        feed.appendChild(emptyState);
        return;
      }
      let html = "";
      for (let i = 0; i < grouped.length; i++) {
        const item = grouped[i];
        const isLatest = i === grouped.length - 1;
        const spkStr = String(item.speaker || "");
        let spkBadge = "";
        if (spkStr === "remote") {
          spkBadge = `<span class="badge remote">${ICON_HEADPHONES}<span>Đối tác</span></span>`;
        } else if (spkStr === "self") {
          spkBadge = `<span class="badge">${ICON_MIC}<span>Diễn giả</span></span>`;
        } else if (/^S\d+$/i.test(spkStr)) {
          const num = parseInt(spkStr.slice(1), 10) || 1;
          const cls = num % 2 === 0 ? "badge spk2" : "badge";
          spkBadge = `<span class="${cls}">${ICON_MIC}<span>Diễn giả ${num}</span></span>`;
        }
        html += `<div class="card${isLatest ? " latest" : ""}">
          <div class="meta"><span>${escapeHtml(item.timestamp || "")}</span>${spkBadge}</div>
          ${showSource ? `<div class="src" style="font-size:${Math.max(13, fontSize - 5)}px">${escapeHtml(item.source_text)}</div>` : ""}
          <div class="tgt" style="font-size:${fontSize}px">${escapeHtml(item.translated_text || "—")}</div>
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

    function stopHeartbeat() {
      if (heartbeatTimer) {
        clearInterval(heartbeatTimer);
        heartbeatTimer = null;
      }
    }

    function connect() {
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      stopHeartbeat();
      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        ws.onclose = null;
        ws.close();
      }
      const proto = location.protocol === "https:" ? "wss:" : "ws:";
      const qs = new URLSearchParams({ lang: selectedLang });
      if (roomPin) qs.set("pin", roomPin);
      const sock = new WebSocket(`${proto}//${location.host}/ws/audience?${qs.toString()}`);
      ws = sock;
      sock.onopen = () => {
        if (ws !== sock) return;
        setConnState("ready");
        // 25s heartbeat keeps Cloudflare Tunnel WebSocket alive well under its 100s idle timeout
        heartbeatTimer = setInterval(() => {
          if (ws === sock && sock.readyState === WebSocket.OPEN) {
            try { sock.send("ping"); } catch {}
          }
        }, 25000);
      };
      sock.onmessage = (ev) => {
        if (ws !== sock) return;
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === "pong") {
            return;
          } else if (msg.type === "pin_required") {
            const invalid = Boolean(msg.data && msg.data.invalid);
            showPinPrompt(invalid ? "Mã PIN chưa đúng hoặc đã thay đổi. Vui lòng nhập lại." : "");
            return;
          } else if (msg.type === "status") {
            hidePinPrompt();
            setConnState(msg.data && msg.data.listening ? "live" : "ready");
            if (msg.data && typeof msg.data.display_frozen === "boolean") {
              setFrozen(msg.data.display_frozen, msg.data.display_freeze_message);
            }
          } else if (msg.type === "display_freeze") {
            setFrozen(msg.data && msg.data.frozen, msg.data && msg.data.message);
          } else if (msg.type === "history") {
            entries = (msg.data || []).slice(-120);
            if (entries.length === 0) partial = null;
            render();
          } else if (msg.type === "history_patch" && msg.data && msg.data.entry) {
            const e = msg.data.entry;
            const removed = new Set((msg.data.removed_ids || []).map(String));
            if (removed.size > 0) {
              entries = entries.filter(x => !removed.has(String(x.id)));
            }
            const idx = entries.findIndex(x => String(x.id) === String(e.id));
            if (idx >= 0) {
              entries[idx] = { ...entries[idx], ...e };
            } else {
              entries.push(e);
              if (entries.length > 150) entries = entries.slice(-150);
            }
            render();
          } else if (msg.type === "partial") {
            const d = msg.data;
            partial = { id: d.id, source_text: d.source_text };
            renderLiveBar();
          } else if (msg.type === "final") {
            const e = msg.data;
            if (partial && partial.id === e.id) partial = null;
            const idx = entries.findIndex(x => String(x.id) === String(e.id));
            if (idx >= 0) {
              entries[idx] = { ...entries[idx], ...e };
            } else {
              entries.push(e);
              if (entries.length > 150) entries = entries.slice(-150);
            }
            render();
          }
        } catch {}
      };
      sock.onclose = (ev) => {
        if (ws !== sock) return;
        stopHeartbeat();
        if (ev && ev.code === 4403) {
          if (!pinBlocked) showPinPrompt("Phòng hội nghị yêu cầu mã PIN 4 số.");
          return;
        }
        if (pinBlocked) return;
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
  <title>DAU Voice Pro — OBS Transparent Subtitles</title>
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
    .prev {
      color: rgba(241, 245, 249, 0.78); font-weight: 600; line-height: 1.35;
      padding-bottom: 8px; margin-bottom: 8px;
      border-bottom: 1px dashed rgba(255, 255, 255, 0.18);
      text-shadow: 0 1px 5px rgba(0,0,0,0.8);
      display: none;
    }
    .prev.show { display: block; }
    .src { color: #cbd5e1; font-weight: 500; margin-bottom: 6px; text-shadow: 0 2px 6px rgba(0,0,0,0.85); }
    .tgt { color: #ffffff; font-weight: 800; line-height: 1.35; text-shadow: 0 2px 8px rgba(0,0,0,0.9); }
    .tgt.draft { color: #fecaca; }
    .box.frozen {
      border-color: rgba(251, 191, 36, 0.72);
      background: rgba(15, 23, 42, 0.92);
    }
    .box.frozen .tgt { color: #fde68a; }
  </style>
</head>
<body>
  <div class="stage">
    <div class="box hidden" id="box">
      <div class="prev" id="prev"></div>
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
    const prevEl = document.getElementById("prev");
    const srcEl = document.getElementById("src");
    const tgtEl = document.getElementById("tgt");
    prevEl.style.fontSize = Math.max(15, Math.round(size * 0.70)) + "px";
    srcEl.style.fontSize = Math.round(size * 0.62) + "px";
    tgtEl.style.fontSize = size + "px";
    if (!showSrc) srcEl.style.display = "none";

    let hideTimer = null;
    let activeId = "";
    let pendingSrc = "";
    let lastFinal = null;
    let prevFinal = null;
    let dwellHoldUntil = 0;
    let isFrozen = false;
    let freezeMessage = "Tạm nghỉ — Phiên dịch hội nghị sẽ tiếp tục trong giây lát";

    function render(prevTgt, src, tgt, isDraft) {
      if (isFrozen) return;
      if (!src && !tgt) return;
      if (prevTgt) {
        prevEl.textContent = prevTgt;
        prevEl.classList.add("show");
      } else {
        prevEl.textContent = "";
        prevEl.classList.remove("show");
      }
      srcEl.style.display = showSrc ? "" : "none";
      srcEl.textContent = src || "";
      tgtEl.textContent = tgt || "...";
      tgtEl.classList.toggle("draft", Boolean(isDraft));
      box.classList.remove("frozen", "hidden");
      if (hideTimer) clearTimeout(hideTimer);
      hideTimer = setTimeout(() => { if (!isFrozen) box.classList.add("hidden"); }, 18000);
    }

    function clearDisplay() {
      if (hideTimer) { clearTimeout(hideTimer); hideTimer = null; }
      prevEl.textContent = "";
      prevEl.classList.remove("show");
      srcEl.textContent = "";
      tgtEl.textContent = "";
      tgtEl.classList.remove("draft");
      box.classList.remove("frozen");
      box.classList.add("hidden");
    }

    function setFrozen(frozen, msg) {
      isFrozen = Boolean(frozen);
      if (msg) freezeMessage = String(msg);
      if (isFrozen) {
        if (hideTimer) { clearTimeout(hideTimer); hideTimer = null; }
        prevEl.textContent = "";
        prevEl.classList.remove("show");
        srcEl.style.display = "none";
        srcEl.textContent = "";
        tgtEl.textContent = freezeMessage;
        tgtEl.classList.remove("draft");
        box.classList.add("frozen");
        box.classList.remove("hidden");
      } else {
        box.classList.remove("frozen");
        if (lastFinal && (lastFinal.source_text || lastFinal.translated_text)) {
          render(prevFinal ? prevFinal.translated_text : "", lastFinal.source_text, lastFinal.translated_text, false);
        } else {
          clearDisplay();
        }
      }
    }

    function connect() {
      const proto = location.protocol === "https:" ? "wss:" : "ws:";
      const ws = new WebSocket(`${proto}//${location.host}/ws/audience?lang=${encodeURIComponent(lang)}`);
      ws.onmessage = (ev) => {
        try {
          const m = JSON.parse(ev.data);
          if (m.type === "status") {
            if (m.data && typeof m.data.display_frozen === "boolean") {
              setFrozen(m.data.display_frozen, m.data.display_freeze_message);
            }
          } else if (m.type === "display_freeze") {
            setFrozen(m.data && m.data.frozen, m.data && m.data.message);
          } else if (m.type === "history") {
            const list = Array.isArray(m.data) ? m.data : [];
            if (list.length === 0) {
              activeId = "";
              pendingSrc = "";
              lastFinal = null;
              prevFinal = null;
              if (!isFrozen) clearDisplay();
            } else {
              const tail = list.slice(-2);
              const prevItem = tail.length >= 2 ? tail[0] : null;
              const lastItem = tail[tail.length - 1];
              prevFinal = prevItem ? { id: String(prevItem.id || ""), source_text: prevItem.source_text || "", translated_text: prevItem.translated_text || "" } : null;
              lastFinal = { id: String(lastItem.id || ""), source_text: lastItem.source_text || "", translated_text: lastItem.translated_text || "" };
              if (!isFrozen) {
                render(prevFinal ? prevFinal.translated_text : "", lastFinal.source_text, lastFinal.translated_text, false);
              }
            }
          } else if (m.type === "history_patch" && m.data && m.data.entry) {
            const ent = m.data.entry;
            const entId = String(ent.id || "");
            const removed = new Set((m.data.removed_ids || []).map(String));
            let changed = false;
            if (lastFinal && (lastFinal.id === entId || removed.has(lastFinal.id))) {
              lastFinal = { id: entId, source_text: ent.source_text || "", translated_text: ent.translated_text || "" };
              if (prevFinal && (prevFinal.id === entId || removed.has(prevFinal.id))) prevFinal = null;
              changed = true;
            } else if (prevFinal && (prevFinal.id === entId || removed.has(prevFinal.id))) {
              prevFinal = { id: entId, source_text: ent.source_text || "", translated_text: ent.translated_text || "" };
              changed = true;
            }
            if (changed && !isFrozen && lastFinal) {
              render(prevFinal ? prevFinal.translated_text : "", lastFinal.source_text, lastFinal.translated_text, false);
            }
          } else if (m.type === "partial") {
            const same = m.data.id === activeId;
            activeId = m.data.id;
            pendingSrc = m.data.source_text || "";
            if (isFrozen) return;
            // Atomic hold: NEVER wipe out the previous sentence with "..." before new MT arrives!
            if (same && tgtEl.textContent && tgtEl.textContent !== "...") {
              render(lastFinal ? lastFinal.translated_text : "", pendingSrc, tgtEl.textContent, true);
            } else if (!lastFinal) {
              render("", pendingSrc, "...", true);
            }
          } else if (m.type === "partial_mt" && m.data.id === activeId) {
            if (isFrozen) return;
            const draftTgt = m.data.translated_text || "";
            const wc = draftTgt.trim().split(/\s+/).filter(Boolean).length;
            if (lastFinal && Date.now() < dwellHoldUntil && wc < 6) {
              return;
            }
            render(lastFinal ? lastFinal.translated_text : "", pendingSrc, draftTgt, true);
          } else if (m.type === "final") {
            const eid = String(m.data.id || "");
            const updated = { id: eid, source_text: m.data.source_text || "", translated_text: m.data.translated_text || "" };
            if (eid && lastFinal && lastFinal.id === eid) {
              lastFinal = updated;
            } else if (eid && prevFinal && prevFinal.id === eid) {
              prevFinal = updated;
            } else {
              activeId = "";
              pendingSrc = "";
              prevFinal = lastFinal;
              lastFinal = updated;
              const wc = lastFinal.translated_text.trim().split(/\s+/).filter(Boolean).length;
              dwellHoldUntil = Date.now() + Math.min(2400, Math.max(1200, wc * 85));
            }
            if (!isFrozen && lastFinal) {
              render(prevFinal ? prevFinal.translated_text : "", lastFinal.source_text, lastFinal.translated_text, false);
            }
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
