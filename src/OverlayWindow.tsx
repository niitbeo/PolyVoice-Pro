import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlignLeft,
  ArrowDown,
  ArrowDownToLine,
  ArrowUpToLine,
  BookOpenText,
  Columns2,
  Contrast,
  Eye,
  EyeOff,
  GripHorizontal,
  Languages,
  Layers,
  Maximize2,
  Mic,
  Minimize2,
  Minus,
  Monitor,
  MonitorPlay,
  Moon,
  MousePointerClick,
  PictureInPicture2,
  Pin,
  PinOff,
  Plus,
  Presentation,
  QrCode,
  Radio,
  Rows2,
  Sun,
  Trash2,
  Tv,
  Type,
  X,
} from "lucide-react";
import {
  AudienceInfo,
  ENGINE_HTTP,
  ENGINE_WS,
  EngineStatus,
  MonitorInfo,
  OVERLAY_LAYOUT_KEY,
  OverlayDockPayload,
  OverlayLayoutMode,
  ParagraphModeId,
  PartialEvent,
  THEME_STORAGE_KEY,
  ThemeMode,
  TranscriptEntry,
  applyTheme,
  formatSpeakerLabel,
  getInitialTheme,
  groupEntriesByParagraph,
  isTauriEnv,
  shortLangName,
} from "./types";

const OVERLAY_FT_DUAL_KEY = "polyvoice_ft_dual";
const OVERLAY_FT_COLOR_KEY = "polyvoice_ft_color";
const BAR_ALPHA_KEY = "polyvoice_bar_alpha";
const BAR_COLOR_KEY = "polyvoice_bar_color";
const BAR_SOURCE_KEY = "polyvoice_bar_source";
const BAR_FONT_KEY = "polyvoice_bar_font";
const BAR_ALPHA_PRESETS = [0.75, 0.9, 0, 0.55] as const;
const FULLTEXT_COLORS = [
  { id: "white", hex: "#ffffff", label: "Trắng tinh" },
  { id: "yellow", hex: "#fde047", label: "Vàng hội trường" },
  { id: "cyan", hex: "#7dd3fc", label: "Xanh sáng dịu" },
] as const;

function isInitialFloatingBarUrl(): boolean {
  if (typeof window === "undefined") return false;
  return window.location.hash.includes("bar=1") || window.location.search.includes("bar=1");
}

function getInitialBarAlpha(): number {
  try {
    const raw = localStorage.getItem(BAR_ALPHA_KEY);
    if (raw !== null) {
      const n = Number(raw);
      if (BAR_ALPHA_PRESETS.some((p) => Math.abs(p - n) < 0.01)) return n;
    }
  } catch {
    // ignore
  }
  return 0.75;
}

function getInitialBarColor(): "#ffffff" | "#fde047" {
  try {
    const saved = localStorage.getItem(BAR_COLOR_KEY);
    if (saved === "#fde047") return "#fde047";
  } catch {
    // ignore
  }
  return "#ffffff";
}

function getInitialBarShowSource(): boolean {
  try {
    const saved = localStorage.getItem(BAR_SOURCE_KEY);
    if (saved === "1" || saved === "true") return true;
  } catch {
    // ignore
  }
  return false;
}

function getInitialBarFontSize(): number {
  try {
    const raw = Number(localStorage.getItem(BAR_FONT_KEY));
    if (Number.isFinite(raw) && raw >= 16 && raw <= 56) return Math.round(raw);
  } catch {
    // ignore
  }
  return 26;
}

function getInitialOverlayLayout(): OverlayLayoutMode {
  try {
    const saved = localStorage.getItem(OVERLAY_LAYOUT_KEY);
    if (
      saved === "led-stage" ||
      saved === "full-text" ||
      saved === "stacked" ||
      saved === "youtube" ||
      saved === "target-only" ||
      saved === "split"
    ) {
      return saved;
    }
  } catch {
    // ignore
  }
  return "led-stage";
}

function getInitialFullTextDual(): boolean {
  try {
    const saved = localStorage.getItem(OVERLAY_FT_DUAL_KEY);
    if (saved === "0" || saved === "false") return false;
  } catch {
    // ignore
  }
  return true;
}

function getInitialFullTextColor(): string {
  try {
    const saved = localStorage.getItem(OVERLAY_FT_COLOR_KEY);
    if (saved && FULLTEXT_COLORS.some((c) => c.hex === saved)) return saved;
  } catch {
    // ignore
  }
  return "#ffffff";
}

function stepFontSize(current: number, deltaDir: 1 | -1): number {
  const step = current >= 80 ? 8 : current >= 48 ? 4 : 2;
  return Math.max(14, Math.min(140, current + deltaDir * step));
}

function renderLiveTarget(partial: PartialEvent) {
  if (partial.stable_text || partial.unstable_text) {
    return (
      <>
        {partial.stable_text && <span className="mt-stable">{partial.stable_text}</span>}
        {partial.stable_text && partial.unstable_text ? " " : null}
        {partial.unstable_text && <span className="mt-unstable">{partial.unstable_text}</span>}
        <span className="live-caret" aria-hidden="true" />
      </>
    );
  }
  if (partial.translated_text) {
    return (
      <>
        {partial.translated_text}
        <span className="live-caret" aria-hidden="true" />
      </>
    );
  }
  return (
    <span className="overlay-translating-dots" aria-label="Đang dịch trực tiếp">
      <span className="live-dots" aria-hidden="true">
        <span />
        <span />
        <span />
      </span>
    </span>
  );
}

const OVERLAY_LED_SOURCE_KEY = "polyvoice_led_show_source";

function getInitialLedShowSource(): boolean {
  try {
    const saved = localStorage.getItem(OVERLAY_LED_SOURCE_KEY);
    if (saved === "0" || saved === "false") return false;
  } catch {
    // ignore
  }
  return true;
}

function tailClipLiveText(raw: string, maxChars = 160): string {
  const s = (raw || "").trim();
  if (s.length <= maxChars) return s;
  const tail = s.slice(-maxChars);
  // Trim cleanly at word boundary for spaced scripts so the leading word isn't sliced in half
  const firstSpace = tail.indexOf(" ");
  if (firstSpace > 0 && firstSpace < 24) {
    return `… ${tail.slice(firstSpace + 1).trimStart()}`;
  }
  return `…${tail.trimStart()}`;
}

export default function OverlayWindow() {
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState<EngineStatus | null>(null);
  const [entries, setEntries] = useState<TranscriptEntry[]>([]);
  const [partial, setPartial] = useState<PartialEvent | null>(null);
  const [inflightPartial, setInflightPartial] = useState<PartialEvent | null>(null);
  // Adaptive minimum reading dwell hold for single/2-tier YouTube captions:
  // prevents a freshly finalized sentence from being bumped away < 1.2s after completion.
  const [ytDwellHoldUntil, setYtDwellHoldUntil] = useState(0);
  const [, setYtDwellTick] = useState(0);

  const [layoutMode, setLayoutMode] = useState<OverlayLayoutMode>(() => getInitialOverlayLayout());
  const [stageMode, setStageMode] = useState(false);
  const [controlsVisible, setControlsVisible] = useState(true);
  const [fontSize, setFontSize] = useState(32);
  const [opacity, setOpacity] = useState(0.96);
  const [theme, setTheme] = useState<ThemeMode>(() => getInitialTheme());
  const [alwaysOnTop, setAlwaysOnTop] = useState(true);
  const [filledMonitor, setFilledMonitor] = useState(false);
  const [userScrolledUp, setUserScrolledUp] = useState(false);
  const [displayFrozen, setDisplayFrozen] = useState(false);
  const [freezeMessage, setFreezeMessage] = useState(
    "Tạm nghỉ — Phiên dịch hội nghị sẽ tiếp tục trong giây lát"
  );
  const [showCornerQr, setShowCornerQr] = useState(false);
  const [showLedSource, setShowLedSource] = useState<boolean>(() => getInitialLedShowSource());
  const [fullTextDual, setFullTextDual] = useState<boolean>(() => getInitialFullTextDual());
  const [fullTextColor, setFullTextColor] = useState<string>(() => getInitialFullTextColor());
  const [ftSrcScrolledUp, setFtSrcScrolledUp] = useState(false);
  const [ftTgtScrolledUp, setFtTgtScrolledUp] = useState(false);
  const [detectedSourceLang, setDetectedSourceLang] = useState<string>("");
  const [detectedSourceLabel, setDetectedSourceLabel] = useState<string>("");
  const [audienceInfo, setAudienceInfo] = useState<AudienceInfo | null>(null);

  // Floating Transparent Glass Subtitle Bar mode (beneath Slides on Monitor #2 or over Video)
  const [floatingBarMode, setFloatingBarMode] = useState<boolean>(() => isInitialFloatingBarUrl());
  const [barAnchor, setBarAnchor] = useState<"bottom" | "top">("bottom");
  const [barMonitorIndex, setBarMonitorIndex] = useState<number>(0);
  const [barMonitorName, setBarMonitorName] = useState<string>("");
  const [barMonitors, setBarMonitors] = useState<MonitorInfo[]>([]);
  const [barClickThrough, setBarClickThrough] = useState<boolean>(false);
  const [barBgAlpha, setBarBgAlpha] = useState<number>(() => getInitialBarAlpha());
  const [barTextColor, setBarTextColor] = useState<"#ffffff" | "#fde047">(() => getInitialBarColor());
  const [barShowSource, setBarShowSource] = useState<boolean>(() => getInitialBarShowSource());
  const [barFontSize, setBarFontSize] = useState<number>(() => getInitialBarFontSize());
  const [barHovered, setBarHovered] = useState<boolean>(false);

  const isMacOs = useMemo(() => {
    if (status?.platform?.os) return status.platform.os === "Darwin";
    if (typeof navigator !== "undefined") {
      return /Mac|iPhone|iPad|iPod/i.test(navigator.platform || navigator.userAgent || "");
    }
    return false;
  }, [status?.platform?.os]);

  const wsRef = useRef<WebSocket | null>(null);
  const partialRef = useRef<PartialEvent | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const ledScrollRef = useRef<HTMLDivElement | null>(null);
  const ftSrcScrollRef = useRef<HTMLDivElement | null>(null);
  const ftTgtScrollRef = useRef<HTMLDivElement | null>(null);
  const ftSrcPinnedRef = useRef(true);
  const ftTgtPinnedRef = useRef(true);
  const pinnedToBottomRef = useRef(true);
  const scrollAnimRef = useRef<number | null>(null);
  const hideBarTimerRef = useRef<number | null>(null);
  const ytDwellTimerRef = useRef<number | null>(null);
  const floatingBarHideTimerRef = useRef<number | null>(null);
  const floatingControlsHoverRef = useRef(false);

  const fetchAudienceInfo = useCallback(() => {
    fetch(`${ENGINE_HTTP}/api/audience/info`)
      .then((r) => (r.ok ? r.json() : null))
      .then((info) => {
        if (info?.qr_svg) setAudienceInfo(info);
      })
      .catch(() => {});
  }, []);

  // Persist overlay layout & LED source-line preferences
  useEffect(() => {
    try {
      localStorage.setItem(OVERLAY_LAYOUT_KEY, layoutMode);
    } catch {
      // ignore
    }
  }, [layoutMode]);

  useEffect(() => {
    try {
      localStorage.setItem(OVERLAY_LED_SOURCE_KEY, showLedSource ? "1" : "0");
    } catch {
      // ignore
    }
  }, [showLedSource]);

  useEffect(() => {
    try {
      localStorage.setItem(OVERLAY_FT_DUAL_KEY, fullTextDual ? "1" : "0");
    } catch {
      // ignore
    }
  }, [fullTextDual]);

  useEffect(() => {
    try {
      localStorage.setItem(OVERLAY_FT_COLOR_KEY, fullTextColor);
    } catch {
      // ignore
    }
  }, [fullTextColor]);

  // Persist Floating Slide/Video Subtitle Bar preferences
  useEffect(() => {
    try {
      localStorage.setItem(BAR_ALPHA_KEY, String(barBgAlpha));
    } catch {
      // ignore
    }
  }, [barBgAlpha]);

  useEffect(() => {
    try {
      localStorage.setItem(BAR_COLOR_KEY, barTextColor);
    } catch {
      // ignore
    }
  }, [barTextColor]);

  useEffect(() => {
    try {
      localStorage.setItem(BAR_SOURCE_KEY, barShowSource ? "1" : "0");
    } catch {
      // ignore
    }
  }, [barShowSource]);

  useEffect(() => {
    try {
      localStorage.setItem(BAR_FONT_KEY, String(barFontSize));
    } catch {
      // ignore
    }
  }, [barFontSize]);

  const refreshBarMonitors = useCallback(async () => {
    if (!isTauriEnv()) return;
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      const list = await invoke<MonitorInfo[]>("list_monitors");
      if (Array.isArray(list)) setBarMonitors(list);
    } catch {
      // ignore
    }
  }, []);

  // Listen to Tauri dock-mode & click-through sync events (from Main App or Cmd/Ctrl+Shift+X hotkey)
  useEffect(() => {
    if (!isTauriEnv()) return;
    let cancelled = false;
    let unlistenDock: (() => void) | undefined;
    let unlistenCt: (() => void) | undefined;
    refreshBarMonitors();
    import("@tauri-apps/api/event")
      .then(({ listen }) => {
        listen<OverlayDockPayload>("overlay-dock-mode", (ev) => {
          const p = ev.payload;
          if (!p) return;
          setFloatingBarMode(Boolean(p.docked));
          if (p.anchor === "top" || p.anchor === "bottom") {
            setBarAnchor(p.anchor);
          }
          if (typeof p.monitor_index === "number") {
            setBarMonitorIndex(p.monitor_index);
          }
          if (typeof p.monitor_name === "string") {
            setBarMonitorName(p.monitor_name);
          }
          if (typeof p.click_through === "boolean") {
            setBarClickThrough(p.click_through);
          }
          if (p.docked) {
            void refreshBarMonitors();
          }
        })
          .then((fn) => {
            if (cancelled) fn();
            else unlistenDock = fn;
          })
          .catch(() => {});

        listen<boolean>("overlay-clickthrough-changed", (ev) => {
          setBarClickThrough(Boolean(ev.payload));
        })
          .then((fn) => {
            if (cancelled) fn();
            else unlistenCt = fn;
          })
          .catch(() => {});
      })
      .catch(() => {});

    return () => {
      cancelled = true;
      unlistenDock?.();
      unlistenCt?.();
    };
  }, [refreshBarMonitors]);

  // Re-pin & sync scroll positions cleanly whenever entering Full-Text or re-mounting the left column
  useEffect(() => {
    if (layoutMode !== "full-text") return;
    ftSrcPinnedRef.current = true;
    ftTgtPinnedRef.current = true;
    setFtSrcScrolledUp(false);
    setFtTgtScrolledUp(false);
    requestAnimationFrame(() => {
      if (ftSrcScrollRef.current) {
        ftSrcScrollRef.current.scrollTop = ftSrcScrollRef.current.scrollHeight;
      }
      if (ftTgtScrollRef.current) {
        ftTgtScrollRef.current.scrollTop = ftTgtScrollRef.current.scrollHeight;
      }
    });
  }, [layoutMode, fullTextDual]);

  // Independent Smart Scroll for Continuous Full-Text panes (only auto-scrolls when pinned near bottom)
  useEffect(() => {
    if (layoutMode !== "full-text") return;
    requestAnimationFrame(() => {
      const srcEl = ftSrcScrollRef.current;
      if (srcEl && ftSrcPinnedRef.current) {
        srcEl.scrollTop = srcEl.scrollHeight;
      }
      const tgtEl = ftTgtScrollRef.current;
      if (tgtEl && ftTgtPinnedRef.current) {
        tgtEl.scrollTop = tgtEl.scrollHeight;
      }
    });
  }, [entries, partial, fontSize, fullTextDual, layoutMode]);

  const handleFtScroll = (which: "src" | "tgt") => {
    const el = which === "src" ? ftSrcScrollRef.current : ftTgtScrollRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 100;
    if (which === "src") {
      ftSrcPinnedRef.current = nearBottom;
      setFtSrcScrolledUp(!nearBottom);
    } else {
      ftTgtPinnedRef.current = nearBottom;
      setFtTgtScrolledUp(!nearBottom);
    }
  };

  const scrollFullTextToBottom = () => {
    ftSrcPinnedRef.current = true;
    ftTgtPinnedRef.current = true;
    setFtSrcScrolledUp(false);
    setFtTgtScrolledUp(false);
    ftSrcScrollRef.current?.scrollTo({ top: ftSrcScrollRef.current.scrollHeight, behavior: "smooth" });
    ftTgtScrollRef.current?.scrollTo({ top: ftTgtScrollRef.current.scrollHeight, behavior: "smooth" });
  };

  // Keep LED stage sentences scrolled cleanly to the newest finalized row when text wraps tall
  useEffect(() => {
    if (layoutMode !== "led-stage") return;
    const node = ledScrollRef.current;
    if (!node) return;
    requestAnimationFrame(() => {
      node.scrollTop = node.scrollHeight;
    });
  }, [entries, fontSize, showLedSource, layoutMode]);

  // Sync theme with main window & allow independent Light/Dark switch for projector rooms
  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === THEME_STORAGE_KEY && (e.newValue === "light" || e.newValue === "dark")) {
        setTheme(e.newValue);
      }
    };
    const onFullscreenChange = () => {
      if (!isTauriEnv()) {
        const fs = Boolean(document.fullscreenElement);
        setFilledMonitor(fs);
        if (fs) setStageMode(true);
      }
    };
    window.addEventListener("storage", onStorage);
    document.addEventListener("fullscreenchange", onFullscreenChange);
    return () => {
      window.removeEventListener("storage", onStorage);
      document.removeEventListener("fullscreenchange", onFullscreenChange);
    };
  }, []);

  const autoHideActive = stageMode || (layoutMode === "full-text" && Boolean(status?.listening));

  // Auto-hide topbar cleanly when Stage Mode is active OR while live-translating in Full-Text Flow mode
  const revealControlsTemporarily = useCallback(() => {
    setControlsVisible(true);
    if (hideBarTimerRef.current) {
      window.clearTimeout(hideBarTimerRef.current);
      hideBarTimerRef.current = null;
    }
    if (autoHideActive) {
      hideBarTimerRef.current = window.setTimeout(() => {
        setControlsVisible(false);
      }, 3000);
    }
  }, [autoHideActive]);

  useEffect(() => {
    if (!autoHideActive) {
      if (hideBarTimerRef.current) {
        window.clearTimeout(hideBarTimerRef.current);
        hideBarTimerRef.current = null;
      }
      setControlsVisible(true);
      return;
    }
    revealControlsTemporarily();
    return () => {
      if (hideBarTimerRef.current) {
        window.clearTimeout(hideBarTimerRef.current);
        hideBarTimerRef.current = null;
      }
    };
  }, [autoHideActive, revealControlsTemporarily]);

  useEffect(() => {
    document.documentElement.classList.add("overlay-body");
    document.body.classList.add("overlay-body");
    let closed = false;
    let timer: number | undefined;

    const connect = () => {
      if (closed) return;
      const ws = new WebSocket(ENGINE_WS);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        fetch(`${ENGINE_HTTP}/api/health`)
          .then((r) => (r.ok ? r.json() : null))
          .then((d) => {
            if (d) {
              setStatus(d);
              if (typeof d.detected_source_lang === "string") setDetectedSourceLang(d.detected_source_lang);
              if (typeof d.display_frozen === "boolean") setDisplayFrozen(d.display_frozen);
              if (d.display_freeze_message) setFreezeMessage(d.display_freeze_message);
            }
          })
          .catch(() => {});
        fetchAudienceInfo();
      };

      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === "status") {
            setStatus(msg.data);
            if (typeof msg.data?.detected_source_lang === "string") {
              setDetectedSourceLang(msg.data.detected_source_lang);
              if (!msg.data.detected_source_lang) setDetectedSourceLabel("");
            }
            if (!msg.data?.listening) {
              partialRef.current = null;
              setPartial(null);
              setInflightPartial(null);
            }
            if (typeof msg.data?.display_frozen === "boolean") {
              setDisplayFrozen(msg.data.display_frozen);
            }
            if (msg.data?.display_freeze_message) {
              setFreezeMessage(msg.data.display_freeze_message);
            }
          } else if (msg.type === "detected_lang") {
            const code = String(msg.data?.detected_source_lang || "");
            const lbl = String(msg.data?.name_vi || code);
            setDetectedSourceLang(code);
            setDetectedSourceLabel(lbl);
          } else if (msg.type === "display_freeze") {
            setDisplayFrozen(Boolean(msg.data?.frozen));
            if (msg.data?.message) setFreezeMessage(msg.data.message);
          } else if (msg.type === "slide_match") {
            const idx = Number(msg.data?.slide_index || 0);
            const title = String(msg.data?.slide_title || "");
            setStatus((prev) =>
              prev && prev.slide_context
                ? {
                    ...prev,
                    slide_context: {
                      ...prev.slide_context,
                      active_slide_index: idx,
                      active_slide_title: title,
                    },
                  }
                : prev
            );
          } else if (msg.type === "partial") {
            const d = msg.data;
            const prev = partialRef.current;
            const same = Boolean(prev && prev.id === d.id && prev.target_lang === d.target_lang);
            if (
              !same &&
              prev &&
              prev.id !== d.id &&
              (prev.translated_text || prev.stable_text || prev.unstable_text)
            ) {
              setInflightPartial(prev);
            }
            const nextPartial: PartialEvent = {
              id: d.id,
              speaker: d.speaker,
              source_text: d.source_text,
              translated_text: same && prev ? prev.translated_text : "",
              stable_text: same && prev ? prev.stable_text : "",
              unstable_text: same && prev ? prev.unstable_text : "",
              source_lang: d.source_lang || "vie_Latn",
              target_lang: d.target_lang,
              asr_ms: d.asr_ms ?? 0,
              mt_ms: same && prev ? prev.mt_ms : 0,
            };
            partialRef.current = nextPartial;
            setPartial(nextPartial);
          } else if (msg.type === "partial_mt") {
            const d = msg.data;
            setInflightPartial((cur) => (cur && cur.id !== d.id ? null : cur));
            const prev = partialRef.current;
            if (prev && prev.id === d.id) {
              const updated: PartialEvent = {
                ...prev,
                translated_text: d.translated_text,
                stable_text: d.stable_text ?? "",
                unstable_text: d.unstable_text ?? "",
                mt_ms: d.mt_ms ?? 0,
                mt_engine: d.mt_engine,
              };
              partialRef.current = updated;
              setPartial(updated);
            }
          } else if (msg.type === "final") {
            const entry: TranscriptEntry = msg.data;
            if (partialRef.current && partialRef.current.id === entry.id) {
              partialRef.current = null;
              setPartial(null);
            }
            setInflightPartial((prev) => (prev && prev.id === entry.id ? null : prev));
            setEntries((prev) => {
              if (prev.some((x) => x.id === entry.id)) return prev;
              return [...prev, entry].slice(-200);
            });
            // Hold newly finalized sentence calmly in primary slot for 1.2s–2.4s (scaled by length)
            // before promoting it upward into the 2nd reading tier, so fast speech never feels rushed.
            const wc = (entry.translated_text || entry.source_text || "").trim().split(/\s+/).filter(Boolean).length;
            const holdMs = Math.min(2400, Math.max(1200, wc * 85));
            const until = Date.now() + holdMs;
            setYtDwellHoldUntil(until);
            if (ytDwellTimerRef.current) window.clearTimeout(ytDwellTimerRef.current);
            ytDwellTimerRef.current = window.setTimeout(() => {
              ytDwellTimerRef.current = null;
              setYtDwellTick((n) => n + 1);
            }, holdMs + 30);
          } else if (msg.type === "history" && Array.isArray(msg.data)) {
            setEntries(msg.data.slice(-200));
            if (msg.data.length === 0) {
              partialRef.current = null;
              setPartial(null);
              setInflightPartial(null);
            }
          } else if (msg.type === "audience_info" && msg.data?.qr_svg) {
            setAudienceInfo(msg.data);
          }
        } catch {
          // ignore
        }
      };

      ws.onclose = () => {
        setConnected(false);
        partialRef.current = null;
        setPartial(null);
        setInflightPartial(null);
        if (!closed) {
          timer = window.setTimeout(connect, 1200);
        }
      };
    };

    connect();
    return () => {
      closed = true;
      if (timer) clearTimeout(timer);
      if (ytDwellTimerRef.current) {
        window.clearTimeout(ytDwellTimerRef.current);
        ytDwellTimerRef.current = null;
      }
      if (floatingBarHideTimerRef.current) {
        window.clearTimeout(floatingBarHideTimerRef.current);
        floatingBarHideTimerRef.current = null;
      }
      wsRef.current?.close();
      document.documentElement.classList.remove("overlay-body");
      document.body.classList.remove("overlay-body");
    };
  }, [fetchAudienceInfo]);

  // Reliable auto-hide for Floating Bar micro-controls on pointer leave, blur, or idle
  const revealFloatingBarControls = useCallback(() => {
    setBarHovered(true);
    if (floatingBarHideTimerRef.current) {
      window.clearTimeout(floatingBarHideTimerRef.current);
      floatingBarHideTimerRef.current = null;
    }
    if (!floatingControlsHoverRef.current) {
      floatingBarHideTimerRef.current = window.setTimeout(() => {
        if (!floatingControlsHoverRef.current) {
          setBarHovered(false);
        }
        floatingBarHideTimerRef.current = null;
      }, 3200);
    }
  }, []);

  const hideFloatingBarControlsNow = useCallback(() => {
    floatingControlsHoverRef.current = false;
    if (floatingBarHideTimerRef.current) {
      window.clearTimeout(floatingBarHideTimerRef.current);
      floatingBarHideTimerRef.current = null;
    }
    setBarHovered(false);
  }, []);

  useEffect(() => {
    if (!floatingBarMode) return;
    const onDocLeave = () => hideFloatingBarControlsNow();
    const onBlur = () => hideFloatingBarControlsNow();
    document.addEventListener("mouseleave", onDocLeave);
    window.addEventListener("blur", onBlur);
    return () => {
      document.removeEventListener("mouseleave", onDocLeave);
      window.removeEventListener("blur", onBlur);
    };
  }, [floatingBarMode, hideFloatingBarControlsNow]);

  // Silky-smooth eased auto-scroll (prevents harsh instant jumps on conference LED screens)
  useEffect(() => {
    const el = scrollRef.current;
    if (!el || !pinnedToBottomRef.current) return;

    if (scrollAnimRef.current !== null) {
      cancelAnimationFrame(scrollAnimRef.current);
      scrollAnimRef.current = null;
    }

    const tick = () => {
      const node = scrollRef.current;
      if (!node || !pinnedToBottomRef.current) {
        scrollAnimRef.current = null;
        return;
      }
      const targetTop = Math.max(0, node.scrollHeight - node.clientHeight);
      const diff = targetTop - node.scrollTop;
      if (Math.abs(diff) <= 1.2) {
        node.scrollTop = targetTop;
        scrollAnimRef.current = null;
        return;
      }
      node.scrollTop += diff * 0.22;
      scrollAnimRef.current = requestAnimationFrame(tick);
    };
    scrollAnimRef.current = requestAnimationFrame(tick);

    return () => {
      if (scrollAnimRef.current !== null) {
        cancelAnimationFrame(scrollAnimRef.current);
        scrollAnimRef.current = null;
      }
    };
  }, [entries, partial, fontSize, layoutMode]);

  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    const atBottom = distanceFromBottom < 72;
    pinnedToBottomRef.current = atBottom;
    setUserScrolledUp(!atBottom);
  };

  const scrollToBottom = () => {
    const el = scrollRef.current;
    if (!el) return;
    pinnedToBottomRef.current = true;
    setUserScrolledUp(false);
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  };

  const handleTogglePin = async () => {
    const next = !alwaysOnTop;
    setAlwaysOnTop(next);
    if (isTauriEnv()) {
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        await invoke("set_overlay_always_on_top", { onTop: next });
      } catch {
        // ignore
      }
    }
  };

  const handleToggleFillMonitor = async () => {
    if (isTauriEnv()) {
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        const filled = await invoke<boolean>("toggle_overlay_fill_monitor");
        setFilledMonitor(filled);
        if (filled) setStageMode(true);
      } catch {
        // ignore
      }
    } else if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen?.().catch(() => {});
      setFilledMonitor(true);
      setStageMode(true);
    } else {
      document.exitFullscreen?.().catch(() => {});
      setFilledMonitor(false);
    }
  };

  const handleClose = async () => {
    if (isTauriEnv()) {
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        await invoke("hide_overlay");
      } catch {
        // ignore
      }
    } else {
      window.close();
    }
  };

  const handleEnterFloatingBar = async (
    targetMonitor?: number,
    targetAnchor: "bottom" | "top" = barAnchor
  ) => {
    if (isTauriEnv()) {
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        const payload = await invoke<OverlayDockPayload>("dock_overlay_subtitle_bar", {
          monitorIndex: targetMonitor ?? null,
          anchor: targetAnchor,
          heightPx: 176,
          clickThrough: barClickThrough,
        });
        setFloatingBarMode(true);
        setBarAnchor(payload.anchor === "top" ? "top" : "bottom");
        setBarMonitorIndex(payload.monitor_index);
        setBarMonitorName(payload.monitor_name);
        setBarClickThrough(payload.click_through);
        void refreshBarMonitors();
      } catch {
        setFloatingBarMode(true);
        setBarAnchor(targetAnchor);
      }
    } else {
      setFloatingBarMode(true);
      setBarAnchor(targetAnchor);
    }
  };

  const handleExitFloatingBar = async () => {
    if (isTauriEnv()) {
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        await invoke("undock_overlay_subtitle_bar");
      } catch {
        // ignore
      }
    }
    setFloatingBarMode(false);
    setBarClickThrough(false);
  };

  const handleBarToggleClickThrough = async () => {
    const next = !barClickThrough;
    setBarClickThrough(next);
    if (isTauriEnv()) {
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        await invoke("set_overlay_clickthrough", { ignore: next });
      } catch {
        setBarClickThrough(!next);
      }
    }
  };

  const handleCycleBarAlpha = () => {
    const idx = BAR_ALPHA_PRESETS.findIndex((p) => Math.abs(p - barBgAlpha) < 0.01);
    const next = BAR_ALPHA_PRESETS[(idx + 1) % BAR_ALPHA_PRESETS.length];
    setBarBgAlpha(next);
  };

  const handleBarDragMouseDown = async (e: React.MouseEvent<HTMLElement>) => {
    if (e.button !== 0 || barClickThrough) return;
    if ((e.target as HTMLElement).closest("button")) return;
    if (!isTauriEnv()) return;
    try {
      const { getCurrentWindow } = await import("@tauri-apps/api/window");
      await getCurrentWindow().startDragging();
    } catch {
      // fallback to data-tauri-drag-region
    }
  };

  const handleTitlebarMouseDown = async (e: React.MouseEvent<HTMLElement>) => {
    if (e.button !== 0) return;
    if ((e.target as HTMLElement).closest("button")) return;
    if (e.detail === 2) {
      handleToggleFillMonitor();
      return;
    }
    if (!isTauriEnv()) return;
    try {
      const { getCurrentWindow } = await import("@tauri-apps/api/window");
      await getCurrentWindow().startDragging();
    } catch {
      // fallback to native data-tauri-drag-region
    }
  };

  const currentParaMode: ParagraphModeId = status?.config?.paragraph_mode || "long";

  const cycleParagraphMode = () => {
    const next: ParagraphModeId =
      currentParaMode === "long" ? "medium" : currentParaMode === "medium" ? "short" : "long";
    fetch(`${ENGINE_HTTP}/api/session/config`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paragraph_mode: next }),
    }).catch(() => {});
  };

  const paraLabel =
    currentParaMode === "long" ? "Đoạn dài" : currentParaMode === "medium" ? "Đoạn vừa" : "Từng câu";

  const groupedEntries = useMemo(() => groupEntriesByParagraph(entries), [entries]);

  // Continuous Full-Text Flow blocks: merge consecutive entries seamlessly into running book-like
  // paragraphs, breaking ONLY when speaker or spoken language changes!
  const fullTextBlocks = useMemo(() => {
    const blocks: Array<{
      id: string;
      speaker: string;
      source_lang: string;
      target_lang: string;
      source_text: string;
      translated_text: string;
    }> = [];
    for (const item of entries) {
      const spk = (item.speaker || "").trim();
      const prev = blocks[blocks.length - 1];
      const sameFlow =
        prev &&
        prev.speaker === spk &&
        prev.source_lang === item.source_lang &&
        prev.target_lang === item.target_lang;
      if (sameFlow) {
        if (item.source_text) {
          prev.source_text = `${prev.source_text} ${item.source_text}`.replace(/\s{2,}/g, " ").trim();
        }
        if (item.translated_text) {
          prev.translated_text = `${prev.translated_text} ${item.translated_text}`.replace(/\s{2,}/g, " ").trim();
        }
      } else {
        blocks.push({
          id: item.id,
          speaker: spk,
          source_lang: item.source_lang,
          target_lang: item.target_lang,
          source_text: (item.source_text || "").trim(),
          translated_text: (item.translated_text || "").trim(),
        });
      }
    }
    return blocks;
  }, [entries]);

  const isAutoSrc = status?.config?.source_lang === "auto";
  const detectedNice = detectedSourceLabel || (detectedSourceLang ? shortLangName(detectedSourceLang) : "");
  const srcLabel = isAutoSrc
    ? detectedNice
      ? `Tự nhận (${detectedNice})`
      : "Tự nhận diện"
    : status?.source_lang_meta?.name_vi || "Tiếng Việt";
  const tgtLabel = status?.target_lang_meta?.name_vi || status?.config?.target_lang || "Tiếng Anh";

  const toggleDisplayFreeze = async () => {
    const prev = displayFrozen;
    const next = !prev;
    setDisplayFrozen(next);
    try {
      const res = await fetch(`${ENGINE_HTTP}/api/display/freeze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frozen: next }),
      });
      if (!res.ok) {
        setDisplayFrozen(prev);
      }
    } catch {
      setDisplayFrozen(prev);
    }
  };

  const cardBg =
    theme === "light"
      ? `rgba(255, 255, 255, ${opacity})`
      : layoutMode === "led-stage"
      ? `rgba(5, 7, 12, ${opacity})`
      : layoutMode === "full-text"
      ? `rgba(24, 26, 33, ${Math.min(opacity, 0.92)})`
      : `rgba(11, 14, 23, ${opacity})`;

  const isEmpty = entries.length === 0 && !partial;
  const srcStackedSize = Math.max(14, Math.round(fontSize * 0.66));
  const hideTopbar = autoHideActive && !controlsVisible;
  const ftTargetInk = theme === "light" ? "#0f172a" : fullTextColor;

  if (floatingBarMode) {
    const latestFinal = entries[entries.length - 1] || null;
    const rawActiveHasMt = Boolean(
      partial && (partial.translated_text || partial.stable_text || partial.unstable_text)
    );
    const activePartial: PartialEvent | null = rawActiveHasMt
      ? partial
      : inflightPartial && (!latestFinal || inflightPartial.id !== latestFinal.id)
        ? inflightPartial
        : partial;
    const activeHasMt = Boolean(
      activePartial &&
        (activePartial.translated_text || activePartial.stable_text || activePartial.unstable_text)
    );
    const partialWords = (activePartial?.translated_text || activePartial?.source_text || "")
      .trim()
      .split(/\s+/)
      .filter(Boolean).length;
    const holdFreshFinal = Boolean(
      latestFinal?.translated_text &&
        activePartial &&
        activePartial.id !== latestFinal.id &&
        Date.now() < ytDwellHoldUntil &&
        partialWords < 6
    );
    const showLivePrimary = Boolean(activeHasMt && activePartial && !holdFreshFinal);
    const barSourceText = showLivePrimary
      ? activePartial?.source_text || ""
      : latestFinal?.source_text || activePartial?.source_text || "";
    const isPureTransparent = barBgAlpha <= 0.01;
    const alphaLabel = isPureTransparent
      ? "Trong 0%"
      : `Mờ ${Math.round(barBgAlpha * 100)}%`;
    const ctShortcutHint = isMacOs ? "⌘⇧X" : "Ctrl+Shift+X";
    const barControlsOpen = barHovered && !barClickThrough;

    const controlsStrip = (
      <div
        className={`floating-bar-controls ${barControlsOpen ? "visible" : ""}`}
        data-tauri-drag-region
        onMouseDown={handleBarDragMouseDown}
        onMouseEnter={() => {
          floatingControlsHoverRef.current = true;
          if (floatingBarHideTimerRef.current) {
            window.clearTimeout(floatingBarHideTimerRef.current);
            floatingBarHideTimerRef.current = null;
          }
          setBarHovered(true);
        }}
        onMouseLeave={() => {
          floatingControlsHoverRef.current = false;
          revealFloatingBarControls();
        }}
        role="toolbar"
        aria-label="Điều khiển thanh phụ đề nổi Dưới Slide / Video"
      >
        <span
          className="floating-bar-grip"
          data-tauri-drag-region
          title="Kéo thả thanh phụ đề này lên bất kỳ vị trí nào trên Slide hoặc Video"
        >
          <GripHorizontal size={12.5} />
          <span className="floating-bar-tag">
            {barMonitorName ? barMonitorName : "Slide / Video"}
          </span>
        </span>

        {barMonitors.length > 1 && (
          <div className="floating-bar-seg" role="group" aria-label="Chọn màn hình">
            {barMonitors.map((m) => (
              <button
                key={m.index}
                type="button"
                className={`fb-btn ${barMonitorIndex === m.index ? "active" : ""}`}
                title={`Chuyển thanh phụ đề sang Màn ${m.index + 1} (${m.name} — ${m.width}×${m.height})`}
                onClick={() => void handleEnterFloatingBar(m.index, barAnchor)}
              >
                <Monitor size={11} /> Màn {m.index + 1}
              </button>
            ))}
          </div>
        )}

        <button
          type="button"
          className="fb-btn"
          title={barAnchor === "bottom" ? "Chuyển thanh phụ đề lên mép trên màn hình" : "Chuyển thanh phụ đề xuống mép dưới màn hình"}
          onClick={() =>
            void handleEnterFloatingBar(
              barMonitors.length > 0 ? barMonitorIndex : undefined,
              barAnchor === "bottom" ? "top" : "bottom"
            )
          }
        >
          {barAnchor === "bottom" ? <ArrowUpToLine size={11.5} /> : <ArrowDownToLine size={11.5} />}
          {barAnchor === "bottom" ? "Mép trên" : "Mép dưới"}
        </button>

        <button
          type="button"
          className={`fb-btn ${isPureTransparent ? "active" : ""}`}
          title="Đổi độ mờ nền kính (75% → 90% → Trong suốt 0% → 55%)"
          onClick={handleCycleBarAlpha}
        >
          <Contrast size={11.5} /> {alphaLabel}
        </button>

        <button
          type="button"
          className="fb-btn"
          title={barTextColor === "#ffffff" ? "Đổi chữ sang màu Vàng nổi bật" : "Đổi chữ sang màu Trắng tinh"}
          onClick={() => setBarTextColor((c) => (c === "#ffffff" ? "#fde047" : "#ffffff"))}
        >
          <span
            className="fb-color-swatch"
            style={{ backgroundColor: barTextColor }}
            aria-hidden="true"
          />
          {barTextColor === "#ffffff" ? "Trắng" : "Vàng"}
        </button>

        <button
          type="button"
          className={`fb-btn ${barShowSource ? "active" : ""}`}
          title={
            barShowSource
              ? "Đang hiện thêm dòng lời nói gốc nhỏ — bấm để ẩn cho gọn 1–2 dòng"
              : "Hiện thêm 1 dòng lời nói gốc nhỏ ngay trên bản dịch"
          }
          onClick={() => setBarShowSource((v) => !v)}
        >
          <Mic size={11} /> {barShowSource ? "Gốc: Bật" : "Gốc: Tắt"}
        </button>

        <div className="floating-bar-seg" role="group" aria-label="Cỡ chữ thanh phụ đề">
          <button
            type="button"
            className="fb-btn icon"
            title="Giảm cỡ chữ"
            onClick={() => setBarFontSize((s) => Math.max(16, s - 2))}
          >
            <Minus size={11} />
          </button>
          <span className="fb-size-val">{barFontSize}px</span>
          <button
            type="button"
            className="fb-btn icon"
            title="Tăng cỡ chữ"
            onClick={() => setBarFontSize((s) => Math.min(56, s + 2))}
          >
            <Plus size={11} />
          </button>
        </div>

        <button
          type="button"
          className={`fb-btn ${barClickThrough ? "active" : ""}`}
          title={`Bật Xuyên chuột để bấm Slide/Video bên dưới không bị vướng (Phím tắt toàn cục: ${ctShortcutHint})`}
          onClick={() => void handleBarToggleClickThrough()}
        >
          <MousePointerClick size={11.5} /> Xuyên chuột ({ctShortcutHint})
        </button>

        <button
          type="button"
          className="fb-btn accent"
          title="Mở rộng về cửa sổ Màn LED lớn (đầy đủ 6 chế độ hiển thị)"
          onClick={() => void handleExitFloatingBar()}
        >
          <Layers size={11.5} /> Màn LED
        </button>

        <button
          type="button"
          className="fb-btn icon close"
          title="Đóng thanh phụ đề nổi"
          onClick={() => void handleClose()}
        >
          <X size={12} />
        </button>
      </div>
    );

    return (
      <div
        className={`overlay-stage-root floating-bar-mode anchor-${barAnchor} ${
          barClickThrough ? "click-through" : ""
        }`}
        onMouseEnter={revealFloatingBarControls}
        onMouseMove={revealFloatingBarControls}
        onMouseLeave={hideFloatingBarControlsNow}
      >
        {barAnchor === "bottom" && controlsStrip}

        <div
          className={`floating-bar-capsule ${isPureTransparent ? "pure-transparent" : ""} ${
            showLivePrimary ? "live" : ""
          }`}
          data-tauri-drag-region
          onMouseDown={handleBarDragMouseDown}
          style={
            isPureTransparent
              ? undefined
              : { background: `rgba(8, 11, 18, ${barBgAlpha})` }
          }
        >
          {displayFrozen ? (
            <div
              className="floating-bar-tgt frozen"
              style={{ fontSize: Math.max(18, Math.round(barFontSize * 0.85)), color: barTextColor }}
            >
              DAU Voice Pro — {freezeMessage}
            </div>
          ) : isEmpty ? (
            <div
              className="floating-bar-placeholder"
              style={{ fontSize: Math.max(15, Math.round(barFontSize * 0.72)) }}
            >
              <span className={`dot ${connected && status?.listening ? "pulse" : ""}`} />
              <span>
                {!connected
                  ? "Đang kết nối DAU Voice Pro..."
                  : status?.listening
                  ? `Đang lắng nghe (${srcLabel} → ${tgtLabel})...`
                  : `Phụ đề nổi Dưới Slide / Video (${srcLabel} → ${tgtLabel}) — rê chuột để chỉnh hoặc kéo vị trí`}
              </span>
            </div>
          ) : (
            <>
              {barShowSource && barSourceText && (
                <div
                  className="floating-bar-src"
                  style={{ fontSize: Math.max(12, Math.round(barFontSize * 0.56)) }}
                >
                  {barSourceText}
                </div>
              )}
              <div
                className="floating-bar-tgt"
                style={{ fontSize: barFontSize, color: barTextColor }}
              >
                {showLivePrimary && activePartial ? (
                  renderLiveTarget(activePartial)
                ) : latestFinal?.translated_text ? (
                  <>
                    {latestFinal.translated_text}
                    {Boolean(activePartial) && <span className="live-caret" aria-hidden="true" />}
                  </>
                ) : activePartial ? (
                  <span className="overlay-translating-dots">
                    <span className="live-dots" aria-hidden="true">
                      <span />
                      <span />
                      <span />
                    </span>
                  </span>
                ) : (
                  "—"
                )}
              </div>
            </>
          )}
        </div>

        {barAnchor === "top" && controlsStrip}
      </div>
    );
  }

  return (
    <div
      className={`overlay-stage-root ${filledMonitor ? "filled" : ""} ${stageMode ? "stage-clean" : ""} ${
        layoutMode === "full-text" ? "full-text-mode" : ""
      }`}
      onMouseMove={(e) => {
        if (autoHideActive && e.clientY <= 72) {
          revealControlsTemporarily();
        }
      }}
    >
      <div className="overlay-stage-card" style={{ background: cardBg }}>
        {/* 2-Tier Floating Modal Chrome: Tier 1 = Dedicated Draggable Titlebar, Tier 2 = Compact Control Strip */}
        <div
          className={`overlay-chrome ${hideTopbar ? "auto-hidden" : ""}`}
          onMouseEnter={() => {
            if (autoHideActive) {
              if (hideBarTimerRef.current) {
                window.clearTimeout(hideBarTimerRef.current);
                hideBarTimerRef.current = null;
              }
              setControlsVisible(true);
            }
          }}
          onMouseLeave={() => {
            if (autoHideActive) revealControlsTemporarily();
          }}
        >
          {/* Tier 1: Dedicated Modal Titlebar — easy to grab & drag anywhere */}
          <header
            className="overlay-titlebar"
            data-tauri-drag-region
            onMouseDown={handleTitlebarMouseDown}
          >
            <div className="overlay-titlebar-left" data-tauri-drag-region>
              <div className="brand-crest sm" data-tauri-drag-region>
                <img src="/school-logo.png" alt="DAU Voice Pro" />
              </div>
              <span className="overlay-brand" data-tauri-drag-region>
                <span>DAU Voice</span>
                <span className="brand-pro-tag">PRO</span>
              </span>
              <span className="overlay-modal-sub" data-tauri-drag-region>
                Phụ đề nổi
              </span>
              <span className={`pill overlay-status-pill ${!connected ? "warn" : status?.listening ? "live" : "ok"}`}>
                <span className={`dot ${connected && status?.listening ? "pulse" : ""}`} />
                {!connected ? "Đang kết nối..." : status?.listening ? "Đang phát" : "Sẵn sàng"}
              </span>
            </div>

            <div className="overlay-drag-center" data-tauri-drag-region>
              <span className="overlay-drag-pill" data-tauri-drag-region>
                <GripHorizontal size={12} /> Kéo để di chuyển
              </span>
            </div>

            <div className="overlay-win-controls">
              <button
                className={`btn sm icon-only ${alwaysOnTop ? "info-active" : ""}`}
                title={alwaysOnTop ? "Đang ghim nổi trên cùng (bấm để tắt)" : "Ghim nổi trên cùng"}
                onClick={handleTogglePin}
              >
                {alwaysOnTop ? <Pin size={13} /> : <PinOff size={13} />}
              </button>
              <button
                className={`btn sm icon-only ${filledMonitor ? "info-active" : ""}`}
                title={filledMonitor ? "Thu gọn cửa sổ" : "Phóng đầy màn hình (nháy đúp thanh tiêu đề)"}
                onClick={handleToggleFillMonitor}
              >
                {filledMonitor ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
              </button>
              <button
                className="btn sm icon-only overlay-close-btn"
                title="Đóng màn hình phụ đề"
                onClick={handleClose}
              >
                <X size={13.5} />
              </button>
            </div>
          </header>

          {/* Tier 2: Compact Control Bar — wraps cleanly with zero overlap at any window size */}
          <div className="overlay-toolbar">
            <div className="overlay-toolbar-left">
              <div className="overlay-layout-seg" role="group" aria-label="Bố cục hiển thị">
                <button
                  className={`overlay-layout-btn ${layoutMode === "led-stage" ? "active" : ""}`}
                  onClick={() => setLayoutMode("led-stage")}
                  title="Sân khấu LED hội trường: 3–4 câu chốt đứng yên chữ cực lớn + thanh nghe sống mờ ở đáy (0% nhảy sửa chữ)"
                >
                  <MonitorPlay size={12} /> Sân khấu LED
                </button>
                <button
                  className={`overlay-layout-btn ${layoutMode === "full-text" ? "active" : ""}`}
                  onClick={() => setLayoutMode("full-text")}
                  title="Toàn văn bản liền mạch: cả bài nói trôi thành 1 trang văn bản sạch (1 cột hoặc 2 cột song song, không thẻ/timestamp)"
                >
                  <BookOpenText size={12} /> Toàn văn
                </button>
                <button
                  className={`overlay-layout-btn ${layoutMode === "stacked" ? "active" : ""}`}
                  onClick={() => setLayoutMode("stacked")}
                  title="Xếp tầng chuẩn hội trường: Gốc nhỏ ở trên • Bản dịch lớn ở dưới"
                >
                  <Rows2 size={12} /> Xếp tầng
                </button>
                <button
                  className={`overlay-layout-btn ${layoutMode === "youtube" ? "active" : ""}`}
                  onClick={() => setLayoutMode("youtube")}
                  title="Phụ đề gọn kiểu YouTube (1–2 dòng nổi nền tối bán trong suốt, đặt dưới Slide/Video)"
                >
                  <Tv size={12} /> YouTube
                </button>
                <button
                  className={`overlay-layout-btn ${layoutMode === "target-only" ? "active" : ""}`}
                  onClick={() => setLayoutMode("target-only")}
                  title="Chỉ hiển thị Bản Dịch cỡ lớn"
                >
                  <Type size={12} /> Chỉ dịch
                </button>
                <button
                  className={`overlay-layout-btn ${layoutMode === "split" ? "active" : ""}`}
                  onClick={() => setLayoutMode("split")}
                  title="Chia đôi 50/50 Trái–Phải để đối chiếu song ngữ"
                >
                  <Columns2 size={12} /> 50/50
                </button>
              </div>

              <span className="overlay-lang-chip">
                <Languages size={11.5} />
                <span>{layoutMode === "target-only" ? tgtLabel : `${srcLabel} → ${tgtLabel}`}</span>
              </span>
            </div>

            <div className="overlay-toolbar-right">
              <button
                className="btn sm overlay-ctrl-btn"
                title="Chế độ gom đoạn hiển thị (Đoạn dài / Đoạn vừa / Từng câu)"
                onClick={cycleParagraphMode}
              >
                <AlignLeft size={12} /> {paraLabel}
              </button>

              <div className="overlay-stepper" role="group" aria-label="Cỡ chữ phụ đề">
                <button
                  className="overlay-step-btn"
                  title="Giảm cỡ chữ"
                  onClick={() => setFontSize((s) => stepFontSize(s, -1))}
                >
                  <Minus size={12} />
                </button>
                <span className="overlay-size-tag">{fontSize}px</span>
                <button
                  className="overlay-step-btn"
                  title="Tăng cỡ chữ (tới 140px cho hội trường lớn)"
                  onClick={() => setFontSize((s) => stepFontSize(s, 1))}
                >
                  <Plus size={12} />
                </button>
              </div>

              <button
                className="btn sm overlay-ctrl-btn"
                title="Đổi độ mờ nền (78% / 94% / 100%)"
                onClick={() => setOpacity((o) => (o >= 0.99 ? 0.78 : o >= 0.9 ? 1.0 : 0.94))}
              >
                <Contrast size={12} /> {Math.round(opacity * 100)}%
              </button>

              {layoutMode === "led-stage" && (
                <button
                  className={`btn sm ${showLedSource ? "info-active" : "overlay-ctrl-btn"}`}
                  title={
                    showLedSource
                      ? "Đang hiển thị cả câu gốc nhỏ phía trên bản dịch LED — bấm để chỉ hiện bản dịch"
                      : "Bật hiển thị dòng lời nói gốc nhỏ ngay trên bản dịch LED"
                  }
                  onClick={() => setShowLedSource((v) => !v)}
                >
                  <Mic size={12} /> {showLedSource ? "Gốc: Bật" : "Gốc: Tắt"}
                </button>
              )}

              <button
                className={`btn sm ${showCornerQr ? "info-active" : "overlay-ctrl-btn"}`}
                title="Hiện/ẩn mã QR nhỏ góc màn LED cho đại biểu quét xem trên điện thoại"
                onClick={() => {
                  const next = !showCornerQr;
                  setShowCornerQr(next);
                  if (next && !audienceInfo?.qr_svg) {
                    fetchAudienceInfo();
                  }
                }}
              >
                <QrCode size={12.5} /> QR góc
              </button>

              <button
                className={`btn sm ${displayFrozen ? "active" : "overlay-ctrl-btn"}`}
                title={displayFrozen ? "Đang tạm ẩn màn LED & QR — bấm để phát lại" : "Tạm ẩn màn LED & QR sang màn chờ DAU (khi nghỉ giải lao / sự cố mic)"}
                onClick={() => void toggleDisplayFreeze()}
              >
                {displayFrozen ? <Eye size={12.5} /> : <EyeOff size={12.5} />}
                {displayFrozen ? "Phát lại LED" : "Tạm ẩn"}
              </button>

              <button
                className="btn sm overlay-ctrl-btn"
                title="Thu gọn thành thanh phụ đề kính mờ trong suốt đặt dưới Slide (Màn 2) hoặc nổi trên Video"
                onClick={() => void handleEnterFloatingBar()}
              >
                <PictureInPicture2 size={12.5} /> Dưới Slide / Video
              </button>

              <button
                className={`btn sm ${stageMode ? "info-active" : "overlay-ctrl-btn"}`}
                title={
                  stageMode
                    ? "Đang ở chế độ Trình chiếu sạch (tự ẩn thanh công cụ & thông số kỹ thuật) — bấm để tắt"
                    : "Bật chế độ Trình chiếu sạch (ẩn thanh nút khi rời chuột & ẩn thông số ms khỏi màn chiếu)"
                }
                onClick={() => setStageMode((v) => !v)}
              >
                <Presentation size={12.5} /> Trình chiếu
              </button>

              <button
                className="btn sm icon-only"
                title={theme === "dark" ? "Chuyển phụ đề sang nền Sáng (cho phòng chiếu sáng)" : "Chuyển phụ đề sang nền Tối"}
                onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
              >
                {theme === "dark" ? <Sun size={13} /> : <Moon size={13} />}
              </button>

              <button
                className="btn sm icon-only"
                title="Xóa chữ trên màn hình phụ đề"
                onClick={() => {
                  setEntries([]);
                  partialRef.current = null;
                  setPartial(null);
                  setInflightPartial(null);
                }}
              >
                <Trash2 size={13} />
              </button>
            </div>
          </div>
        </div>

        {/* Split Column Language Headers (only shown in 50/50 split layout) */}
        {layoutMode === "split" && (
          <div className={`overlay-col-headers ${stageMode ? "compact" : ""}`}>
            <div className="overlay-col-head left">
              <span className="col-badge src">
                <Mic size={11} /> NGÔN NGỮ GỐC
              </span>
              <span className="col-lang">{srcLabel}</span>
            </div>
            <div className="overlay-col-head right">
              <span className="col-badge tgt">
                <Languages size={11} /> NGÔN NGỮ DỊCH
              </span>
              <span className="col-lang">{tgtLabel}</span>
            </div>
          </div>
        )}

        {/* Scrollable Conference Subtitle Feed */}
        <div
          className={`overlay-feed ${layoutMode === "led-stage" ? "led-stage" : ""} ${
            layoutMode === "full-text" ? "full-text" : ""
          } ${showCornerQr && audienceInfo?.qr_svg && !displayFrozen ? "has-corner-qr" : ""}`}
          ref={scrollRef}
          onScroll={handleScroll}
        >
          {layoutMode === "full-text" ? (
            <div className={`overlay-fulltext-grid ${fullTextDual ? "dual" : "single"}`}>
              {fullTextDual && (
                <div
                  className="overlay-fulltext-pane src"
                  ref={ftSrcScrollRef}
                  onScroll={() => handleFtScroll("src")}
                  style={{ fontSize }}
                >
                  {isEmpty ? (
                    <div className="overlay-fulltext-placeholder">
                      {status?.listening ? "Đang lắng nghe diễn giả..." : `Đang chờ giọng nói (${srcLabel})...`}
                    </div>
                  ) : (
                    <>
                      {fullTextBlocks.map((blk, i) => {
                        const prevBlk = i > 0 ? fullTextBlocks[i - 1] : null;
                        const spkLabel = formatSpeakerLabel(blk.speaker);
                        const showSpk = Boolean(spkLabel) && (!prevBlk || prevBlk.speaker !== blk.speaker);
                        const showLang =
                          fullTextBlocks.length > 1 && (!prevBlk || prevBlk.source_lang !== blk.source_lang);
                        return (
                          <div className="overlay-fulltext-para" key={`ft-src-${blk.id}`}>
                            {(showSpk || showLang) && (
                              <span className="overlay-fulltext-inline-meta">
                                {showSpk && <span className="overlay-fulltext-speaker">{spkLabel}:</span>}
                                {showLang && (
                                  <span className="overlay-fulltext-lang">
                                    {shortLangName(blk.source_lang)}
                                  </span>
                                )}
                              </span>
                            )}
                            <span>{blk.source_text}</span>
                            {i === fullTextBlocks.length - 1 &&
                              partial?.source_text &&
                              (!partial.speaker || partial.speaker === blk.speaker) &&
                              partial.source_lang === blk.source_lang && (
                                <span className="overlay-fulltext-live-tail">
                                  {" "}
                                  {partial.source_text}
                                  <span className="live-caret src" aria-hidden="true" />
                                </span>
                              )}
                          </div>
                        );
                      })}
                      {partial?.source_text &&
                        (fullTextBlocks.length === 0 ||
                          (Boolean(partial.speaker) &&
                            partial.speaker !== fullTextBlocks[fullTextBlocks.length - 1].speaker) ||
                          partial.source_lang !== fullTextBlocks[fullTextBlocks.length - 1].source_lang) && (
                          <div className="overlay-fulltext-para">
                            {formatSpeakerLabel(partial.speaker) && (
                              <span className="overlay-fulltext-inline-meta">
                                <span className="overlay-fulltext-speaker">
                                  {formatSpeakerLabel(partial.speaker)}:
                                </span>
                              </span>
                            )}
                            <span className="overlay-fulltext-live-tail">
                              {partial.source_text}
                              <span className="live-caret src" aria-hidden="true" />
                            </span>
                          </div>
                        )}
                    </>
                  )}
                </div>
              )}

              <div
                className="overlay-fulltext-pane tgt"
                ref={ftTgtScrollRef}
                onScroll={() => handleFtScroll("tgt")}
                style={{ fontSize, color: ftTargetInk }}
              >
                {isEmpty ? (
                  <div className="overlay-fulltext-placeholder">
                    {status?.listening
                      ? "Bản dịch toàn văn sẽ trôi liền mạch tại đây..."
                      : `Chế độ Toàn văn (${tgtLabel}) — sẵn sàng phiên dịch`}
                  </div>
                ) : (
                  <>
                    {fullTextBlocks.map((blk, i) => {
                      const prevBlk = i > 0 ? fullTextBlocks[i - 1] : null;
                      const spkLabel = formatSpeakerLabel(blk.speaker);
                      const showSpk = Boolean(spkLabel) && (!prevBlk || prevBlk.speaker !== blk.speaker);
                      const showLang =
                        !fullTextDual &&
                        fullTextBlocks.length > 1 &&
                        (!prevBlk || prevBlk.source_lang !== blk.source_lang);
                      const liveTgtMatchesLast =
                        i === fullTextBlocks.length - 1 &&
                        Boolean(partial?.translated_text || partial?.stable_text || partial?.unstable_text) &&
                        (!partial?.speaker || partial.speaker === blk.speaker) &&
                        partial?.target_lang === blk.target_lang;
                      return (
                        <div className="overlay-fulltext-para" key={`ft-tgt-${blk.id}`}>
                          {(showSpk || showLang) && (
                            <span className="overlay-fulltext-inline-meta">
                              {showSpk && <span className="overlay-fulltext-speaker">{spkLabel}:</span>}
                              {showLang && (
                                <span className="overlay-fulltext-lang">
                                  {shortLangName(blk.source_lang)} → {shortLangName(blk.target_lang)}
                                </span>
                              )}
                            </span>
                          )}
                          <span>{blk.translated_text || "—"}</span>
                          {liveTgtMatchesLast && partial && (
                            <span className="overlay-fulltext-live-tail tgt">
                              {" "}
                              {renderLiveTarget(partial)}
                            </span>
                          )}
                        </div>
                      );
                    })}
                    {partial &&
                      Boolean(partial.translated_text || partial.stable_text || partial.unstable_text) &&
                      (fullTextBlocks.length === 0 ||
                        (Boolean(partial.speaker) &&
                          partial.speaker !== fullTextBlocks[fullTextBlocks.length - 1].speaker) ||
                        partial.target_lang !== fullTextBlocks[fullTextBlocks.length - 1].target_lang) && (
                        <div className="overlay-fulltext-para">
                          {formatSpeakerLabel(partial.speaker) && (
                            <span className="overlay-fulltext-inline-meta">
                              <span className="overlay-fulltext-speaker">
                                {formatSpeakerLabel(partial.speaker)}:
                              </span>
                            </span>
                          )}
                          <span className="overlay-fulltext-live-tail tgt">{renderLiveTarget(partial)}</span>
                        </div>
                      )}
                  </>
                )}
              </div>
            </div>
          ) : layoutMode === "led-stage" ? (
            <div className="overlay-led-container">
              <div className="overlay-led-sentences" ref={ledScrollRef}>
                {entries.length === 0 ? (
                  <div className="overlay-empty-single">
                    <div className="overlay-empty-card wide">
                      <div className="overlay-empty-icon info">
                        <MonitorPlay size={22} />
                      </div>
                      <div className="overlay-empty-title">
                        Sân khấu LED Hội trường{" "}
                        <span className="overlay-empty-pill info">{srcLabel} → {tgtLabel}</span>
                      </div>
                      <p className="overlay-empty-desc">
                        Bản dịch đứng yên khi chốt trọn câu (không nhảy sửa chữ), kèm dòng lời nói gốc và thanh nghe sống mở rộng ở đáy màn hình.
                      </p>
                    </div>
                  </div>
                ) : (
                  (() => {
                    const maxRows = showLedSource ? 3 : 4;
                    const visible = entries.slice(-maxRows);
                    const total = visible.length;
                    return visible.map((item, idx) => {
                      const distFromLatest = total - 1 - idx;
                      const tierClass =
                        distFromLatest === 0
                          ? "latest"
                          : distFromLatest === 1
                          ? "prev-1"
                          : distFromLatest === 2
                          ? "prev-2"
                          : "prev-3";
                      const tgtSize =
                        distFromLatest === 0 ? fontSize : Math.max(18, Math.round(fontSize * 0.82));
                      const srcSize = Math.max(14, Math.round(tgtSize * 0.52));
                      const showRowSource =
                        showLedSource &&
                        Boolean(item.source_text) &&
                        item.source_text.trim() !== (item.translated_text || "").trim();
                      return (
                        <div className={`overlay-led-item ${tierClass}`} key={item.id}>
                          {showRowSource && (
                            <div className="overlay-led-src" style={{ fontSize: srcSize }}>
                              {item.source_text}
                            </div>
                          )}
                          <div className="overlay-led-tgt" style={{ fontSize: tgtSize }}>
                            {item.translated_text || "—"}
                          </div>
                        </div>
                      );
                    });
                  })()
                )}
              </div>

              {/* Fixed Bottom Live Source Bar — wraps cleanly up to 2 lines & follows the latest spoken tail */}
              <div className={`overlay-led-livebar ${connected && (partial?.source_text || status?.listening) ? "active" : ""}`}>
                <span className={`overlay-led-dot ${connected && status?.listening ? "live" : ""}`} aria-hidden="true" />
                <span className="overlay-led-live-label">
                  {!connected
                    ? "Đang kết nối lại dịch vụ AI..."
                    : partial?.source_text
                    ? "Đang nghe:"
                    : status?.listening
                    ? "Đang lắng nghe diễn giả..."
                    : "Sẵn sàng hội trường"}
                </span>
                {connected && partial?.source_text && (
                  <span className="overlay-led-live-src">{tailClipLiveText(partial.source_text, 160)}</span>
                )}
                {isAutoSrc && detectedNice ? (
                  <span className="overlay-led-lang-tag">
                    Tự nhận: {detectedNice}
                  </span>
                ) : null}
                {status?.slide_context?.enabled && status?.slide_context?.active_slide_index ? (
                  <span className="overlay-led-slide-tag">
                    Slide #{status.slide_context.active_slide_index}
                  </span>
                ) : null}
              </div>
            </div>
          ) : isEmpty ? (
            layoutMode === "split" ? (
              <div className="overlay-empty-split">
                <div className="overlay-empty-col">
                  <div className="overlay-empty-card">
                    <div className="overlay-empty-icon">
                      <Mic size={19} />
                    </div>
                    <div className="overlay-empty-title">
                      Đang chờ giọng nói <span className="overlay-empty-pill">{srcLabel}</span>
                    </div>
                    <p className="overlay-empty-desc">
                      Lời nói gốc của diễn giả sẽ hiển thị theo thời gian thực tại cột này.
                    </p>
                  </div>
                </div>
                <div className="overlay-empty-col right">
                  <div className="overlay-empty-card">
                    <div className="overlay-empty-icon info">
                      <Languages size={19} />
                    </div>
                    <div className="overlay-empty-title">
                      Bản dịch song song <span className="overlay-empty-pill info">{tgtLabel}</span>
                    </div>
                    <p className="overlay-empty-desc">
                      Bản dịch khóa tiền tố ổn định sẽ xuất hiện đồng bộ tại đây và cuộn mượt lên trên.
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="overlay-empty-single">
                <div className="overlay-empty-card wide">
                  <div className="overlay-empty-icon info">
                    <Languages size={22} />
                  </div>
                  <div className="overlay-empty-title">
                    Sẵn sàng phụ đề trực tiếp{" "}
                    <span className="overlay-empty-pill info">
                      {layoutMode === "target-only" ? tgtLabel : `${srcLabel} → ${tgtLabel}`}
                    </span>
                  </div>
                  <p className="overlay-empty-desc">
                    {layoutMode === "target-only"
                      ? "Bản dịch cỡ lớn với cơ chế khóa tiền tố ổn định (không nhảy chữ) sẽ hiển thị gọn sạch tại đây."
                      : "Bố cục Xếp tầng hội trường: Lời gốc gọn nhẹ phía trên — Bản dịch nổi bật ngay phía dưới, dễ đọc từ hàng ghế cuối."}
                  </p>
                </div>
              </div>
            )
          ) : layoutMode === "youtube" ? (
            (() => {
              const latestFinal = entries[entries.length - 1] || null;
              const prevFinal = entries.length >= 2 ? entries[entries.length - 2] : null;
              const rawActiveHasMt = Boolean(
                partial && (partial.translated_text || partial.stable_text || partial.unstable_text)
              );
              const activePartial: PartialEvent | null = rawActiveHasMt
                ? partial
                : inflightPartial && (!latestFinal || inflightPartial.id !== latestFinal.id)
                  ? inflightPartial
                  : partial;
              const activeHasMt = Boolean(
                activePartial && (activePartial.translated_text || activePartial.stable_text || activePartial.unstable_text)
              );
              const partialWords = (activePartial?.translated_text || activePartial?.source_text || "")
                .trim()
                .split(/\s+/)
                .filter(Boolean).length;
              // Reading-comfort dwell hold: keep freshly finalized sentence bold in the primary slot
              // for 1.2s–2.4s while the next sentence only has a few words (< 6), then promote it gently
              // into the upper tier (`overlay-yt-prev`) so it stays readable for the ENTIRE next sentence!
              const holdFreshFinal = Boolean(
                latestFinal?.translated_text &&
                  activePartial &&
                  activePartial.id !== latestFinal.id &&
                  Date.now() < ytDwellHoldUntil &&
                  partialWords < 6
              );
              const showLivePrimary = Boolean(activeHasMt && activePartial && !holdFreshFinal);
              const ytPrevEntry: TranscriptEntry | null = showLivePrimary ? latestFinal : prevFinal;
              const ytSource = showLivePrimary
                ? activePartial?.source_text || ""
                : latestFinal?.source_text || activePartial?.source_text || "";
              return (
                <div className="overlay-yt-stage">
                  <div className={`overlay-yt-pill ${showLivePrimary ? "live" : ""}`}>
                    {ytPrevEntry && ytPrevEntry.translated_text && (
                      <div
                        className="overlay-yt-prev"
                        key={`yt-prev-${ytPrevEntry.id}`}
                        aria-label="Câu vừa hoàn tất"
                      >
                        <div
                          className="overlay-yt-prev-tgt"
                          style={{ fontSize: Math.max(14, Math.round(fontSize * 0.70)) }}
                        >
                          {ytPrevEntry.translated_text}
                        </div>
                      </div>
                    )}
                    <div className="overlay-yt-curr">
                      {ytSource && (
                        <div
                          className="overlay-yt-src"
                          style={{ fontSize: Math.max(13, Math.round(fontSize * 0.56)) }}
                        >
                          {ytSource}
                        </div>
                      )}
                      <div className="overlay-yt-tgt" style={{ fontSize }}>
                        {showLivePrimary && activePartial ? (
                          renderLiveTarget(activePartial)
                        ) : latestFinal?.translated_text ? (
                          <>
                            {latestFinal.translated_text}
                            {Boolean(activePartial) && <span className="live-caret" aria-hidden="true" />}
                          </>
                        ) : activePartial ? (
                          <span className="overlay-translating-dots">
                            <span className="live-dots" aria-hidden="true">
                              <span />
                              <span />
                              <span />
                            </span>
                          </span>
                        ) : (
                          "—"
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })()
          ) : layoutMode === "split" ? (
            <div className="overlay-feed-inner split">
              {groupedEntries.map((item) => (
                <div className="overlay-row" key={item.id}>
                  <div className="overlay-cell left">
                    {!stageMode && (
                      <div className="overlay-cell-meta">
                        {formatSpeakerLabel(item.speaker) && (
                          <span
                            className={`spk-badge ${
                              item.speaker === "self" ? "self" : item.speaker === "remote" ? "remote" : "diar"
                            }`}
                          >
                            {formatSpeakerLabel(item.speaker)}
                          </span>
                        )}
                        <span>{item.timestamp}</span>
                        {(item.sentence_count ?? 1) > 1 ? <span>• {item.sentence_count} câu</span> : null}
                      </div>
                    )}
                    <div className="overlay-cell-text src" style={{ fontSize }}>
                      {item.source_text}
                    </div>
                  </div>
                  <div className="overlay-cell right">
                    {!stageMode && (
                      <div className="overlay-cell-meta">
                        <span>{item.timestamp}</span>
                        {item.mt_ms > 0 ? <span>• {item.mt_ms}ms</span> : null}
                      </div>
                    )}
                    <div className="overlay-cell-text tgt" style={{ fontSize }}>
                      {item.translated_text || "—"}
                    </div>
                  </div>
                </div>
              ))}

              {partial && (
                <div className="overlay-row live">
                  <div className="overlay-cell left">
                    <div className="overlay-cell-meta live-tag">
                      <Radio size={12} /> Đang nói...
                    </div>
                    <div className="overlay-cell-text src draft" style={{ fontSize }}>
                      {partial.source_text}
                      <span className="live-caret src" aria-hidden="true" />
                    </div>
                  </div>
                  <div className="overlay-cell right">
                    <div className="overlay-cell-meta live-tag">
                      <Radio size={12} /> Đang dịch trực tiếp...
                    </div>
                    <div className="overlay-cell-text tgt draft" style={{ fontSize }}>
                      {renderLiveTarget(partial)}
                    </div>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className={`overlay-feed-inner ${layoutMode}`}>
              {groupedEntries.map((item, idx) => {
                const isRecent = idx >= groupedEntries.length - 2;
                return (
                  <div
                    className={`overlay-stage-block ${layoutMode} ${isRecent ? "recent" : "older"}`}
                    key={item.id}
                  >
                    {!stageMode && (
                      <div className="overlay-cell-meta">
                        {formatSpeakerLabel(item.speaker) && (
                          <span
                            className={`spk-badge ${
                              item.speaker === "self" ? "self" : item.speaker === "remote" ? "remote" : "diar"
                            }`}
                          >
                            {formatSpeakerLabel(item.speaker)}
                          </span>
                        )}
                        <span>{item.timestamp}</span>
                        {(item.sentence_count ?? 1) > 1 ? <span>• {item.sentence_count} câu</span> : null}
                        {item.mt_ms > 0 ? <span>• {item.mt_ms}ms</span> : null}
                      </div>
                    )}
                    {layoutMode === "stacked" && (
                      <div className="overlay-stage-src" style={{ fontSize: srcStackedSize }}>
                        {item.source_text}
                      </div>
                    )}
                    <div className="overlay-stage-tgt" style={{ fontSize }}>
                      {item.translated_text || "—"}
                    </div>
                  </div>
                );
              })}

              {partial &&
                (layoutMode !== "target-only" ||
                  Boolean(partial.translated_text || partial.stable_text || partial.unstable_text)) && (
                <div className={`overlay-stage-block live ${layoutMode}`}>
                  <div className="overlay-cell-meta live-tag">
                    <Radio size={12} /> {stageMode ? "LIVE" : "Đang nói & dịch trực tiếp..."}
                  </div>
                  {layoutMode === "stacked" && (
                    <div className="overlay-stage-src draft" style={{ fontSize: srcStackedSize }}>
                      {partial.source_text}
                      <span className="live-caret src" aria-hidden="true" />
                    </div>
                  )}
                  <div className="overlay-stage-tgt draft" style={{ fontSize }}>
                    {renderLiveTarget(partial)}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {((userScrolledUp && layoutMode !== "led-stage" && layoutMode !== "full-text") ||
          (layoutMode === "full-text" && (ftSrcScrolledUp || ftTgtScrolledUp))) && (
          <button
            className="overlay-jump-btn"
            onClick={layoutMode === "full-text" ? scrollFullTextToBottom : scrollToBottom}
          >
            <ArrowDown size={14} /> Xuống câu mới nhất
          </button>
        )}

        {/* Floating Bottom-Right Quick Controls for Continuous Full-Text Mode (A-/A+ up to 140px, 3 text colors, 1/2 col toggle) */}
        {layoutMode === "full-text" && !displayFrozen && (
          <div className="overlay-fulltext-pill" role="toolbar" aria-label="Điều khiển nhanh chế độ Toàn văn">
            <div className="overlay-ft-font">
              <button
                type="button"
                className="overlay-ft-btn"
                title="Giảm cỡ chữ"
                onClick={() => setFontSize((s) => stepFontSize(s, -1))}
              >
                A−
              </button>
              <span className="overlay-ft-size">{fontSize}</span>
              <button
                type="button"
                className="overlay-ft-btn"
                title="Tăng cỡ chữ (tới 140px)"
                onClick={() => setFontSize((s) => stepFontSize(s, 1))}
              >
                A+
              </button>
            </div>

            {theme === "dark" && (
              <div className="overlay-ft-colors" role="group" aria-label="Màu chữ bản dịch">
                {FULLTEXT_COLORS.map((c) => (
                  <button
                    key={c.id}
                    type="button"
                    className={`overlay-color-dot ${fullTextColor === c.hex ? "active" : ""}`}
                    style={{ backgroundColor: c.hex }}
                    title={`Màu chữ: ${c.label}`}
                    onClick={() => setFullTextColor(c.hex)}
                  />
                ))}
              </div>
            )}

            <button
              type="button"
              className={`overlay-ft-btn toggle ${fullTextDual ? "active" : ""}`}
              title={
                fullTextDual
                  ? "Đang hiện 2 cột Song ngữ (Gốc | Dịch) — bấm để chuyển sang 1 cột Chỉ dịch"
                  : "Đang hiện 1 cột Chỉ dịch — bấm để chia 2 cột Song ngữ (Gốc | Dịch)"
              }
              onClick={() => setFullTextDual((v) => !v)}
            >
              {fullTextDual ? <Columns2 size={14} /> : <AlignLeft size={14} />}
            </button>
          </div>
        )}

        {showCornerQr && audienceInfo?.qr_svg && !displayFrozen && (
          <div
            className={`overlay-corner-qr ${audienceInfo.active_mode === "public" ? "is-public" : ""}`}
            title={audienceInfo.audience_url}
          >
            <div
              className="overlay-corner-qr-box"
              dangerouslySetInnerHTML={{ __html: audienceInfo.qr_svg }}
            />
            <div className="overlay-corner-qr-caption">
              <strong>
                {audienceInfo.active_mode === "public" ? "Quét QR 4G / Wi-Fi" : "Quét QR phụ đề"}
              </strong>
              <span>
                {audienceInfo.pin_enabled && audienceInfo.room_pin
                  ? `PIN: ${audienceInfo.room_pin}`
                  : "Chọn ngôn ngữ riêng"}
              </span>
            </div>
          </div>
        )}

        {displayFrozen && (
          <div className="overlay-freeze-curtain" role="status" aria-live="assertive">
            <img src="/school-logo.png" alt="DAU Voice Pro" className="overlay-freeze-logo" />
            <div className="overlay-freeze-brand">
              DAU Voice <span className="brand-pro-tag">PRO</span>
            </div>
            <div className="overlay-freeze-msg">{freezeMessage}</div>
            <button
              className="btn primary sm overlay-unfreeze-btn"
              onClick={() => void toggleDisplayFreeze()}
              style={{ marginTop: 10 }}
            >
              <Eye size={13.5} /> Tiếp tục phát lên màn LED
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
