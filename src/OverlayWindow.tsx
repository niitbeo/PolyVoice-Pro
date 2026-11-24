import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlignLeft,
  ArrowDown,
  Columns2,
  Contrast,
  Eye,
  EyeOff,
  GripHorizontal,
  Languages,
  Maximize2,
  Mic,
  Minimize2,
  Minus,
  MonitorPlay,
  Moon,
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
  OVERLAY_LAYOUT_KEY,
  OverlayLayoutMode,
  ParagraphModeId,
  PartialEvent,
  THEME_STORAGE_KEY,
  ThemeMode,
  TranscriptEntry,
  applyTheme,
  getInitialTheme,
  groupEntriesByParagraph,
  isTauriEnv,
} from "./types";

function getInitialOverlayLayout(): OverlayLayoutMode {
  try {
    const saved = localStorage.getItem(OVERLAY_LAYOUT_KEY);
    if (
      saved === "led-stage" ||
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
  const [audienceInfo, setAudienceInfo] = useState<AudienceInfo | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const partialRef = useRef<PartialEvent | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const pinnedToBottomRef = useRef(true);
  const scrollAnimRef = useRef<number | null>(null);
  const hideBarTimerRef = useRef<number | null>(null);
  const ytDwellTimerRef = useRef<number | null>(null);

  const fetchAudienceInfo = useCallback(() => {
    fetch(`${ENGINE_HTTP}/api/audience/info`)
      .then((r) => (r.ok ? r.json() : null))
      .then((info) => {
        if (info?.qr_svg) setAudienceInfo(info);
      })
      .catch(() => {});
  }, []);

  // Persist overlay layout preference
  useEffect(() => {
    try {
      localStorage.setItem(OVERLAY_LAYOUT_KEY, layoutMode);
    } catch {
      // ignore
    }
  }, [layoutMode]);

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

  // Auto-hide topbar cleanly when Stage / Clean Presentation Mode is active
  const revealControlsTemporarily = useCallback(() => {
    setControlsVisible(true);
    if (hideBarTimerRef.current) {
      window.clearTimeout(hideBarTimerRef.current);
      hideBarTimerRef.current = null;
    }
    if (stageMode) {
      hideBarTimerRef.current = window.setTimeout(() => {
        setControlsVisible(false);
      }, 2600);
    }
  }, [stageMode]);

  useEffect(() => {
    if (!stageMode) {
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
  }, [stageMode, revealControlsTemporarily]);

  useEffect(() => {
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
      wsRef.current?.close();
      document.body.classList.remove("overlay-body");
    };
  }, [fetchAudienceInfo]);

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
        await invoke("toggle_overlay");
      } catch {
        // ignore
      }
    } else {
      window.close();
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

  const srcLabel = status?.source_lang_meta?.name_vi || "Tiếng Việt";
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
      : `rgba(11, 14, 23, ${opacity})`;

  const isEmpty = entries.length === 0 && !partial;
  const srcStackedSize = Math.max(14, Math.round(fontSize * 0.66));
  const hideTopbar = stageMode && !controlsVisible;

  return (
    <div
      className={`overlay-stage-root ${filledMonitor ? "filled" : ""} ${stageMode ? "stage-clean" : ""}`}
      onMouseMove={(e) => {
        if (stageMode && e.clientY <= 72) {
          revealControlsTemporarily();
        }
      }}
    >
      <div className="overlay-stage-card" style={{ background: cardBg }}>
        {/* 2-Tier Floating Modal Chrome: Tier 1 = Dedicated Draggable Titlebar, Tier 2 = Compact Control Strip */}
        <div
          className={`overlay-chrome ${hideTopbar ? "auto-hidden" : ""}`}
          onMouseEnter={() => {
            if (stageMode) {
              if (hideBarTimerRef.current) {
                window.clearTimeout(hideBarTimerRef.current);
                hideBarTimerRef.current = null;
              }
              setControlsVisible(true);
            }
          }}
          onMouseLeave={() => {
            if (stageMode) revealControlsTemporarily();
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
                  onClick={() => setFontSize((s) => Math.max(16, s - 2))}
                >
                  <Minus size={12} />
                </button>
                <span className="overlay-size-tag">{fontSize}px</span>
                <button
                  className="overlay-step-btn"
                  title="Tăng cỡ chữ"
                  onClick={() => setFontSize((s) => Math.min(56, s + 2))}
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
            showCornerQr && audienceInfo?.qr_svg && !displayFrozen ? "has-corner-qr" : ""
          }`}
          ref={scrollRef}
          onScroll={handleScroll}
        >
          {layoutMode === "led-stage" ? (
            <div className="overlay-led-container">
              <div className="overlay-led-sentences">
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
                        Bản dịch chỉ hiện khi chốt tròn câu (đứng yên 100%, không nhảy sửa chữ), lưu 3–4 câu gần nhất cỡ chữ lớn dễ đọc từ hàng ghế cuối.
                      </p>
                    </div>
                  </div>
                ) : (
                  (() => {
                    const visible = entries.slice(-4);
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
                      return (
                        <div className={`overlay-led-item ${tierClass}`} key={item.id}>
                          <div className="overlay-led-tgt" style={{ fontSize: distFromLatest === 0 ? fontSize : Math.max(18, Math.round(fontSize * 0.84)) }}>
                            {item.translated_text || "—"}
                          </div>
                        </div>
                      );
                    });
                  })()
                )}
              </div>

              {/* Fixed Bottom Live Source Bar — proves mic pickup live WITHOUT ever jittering translations above */}
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
                  <span className="overlay-led-live-src">{partial.source_text}</span>
                )}
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
                        {item.speaker === "self" && <span className="spk-badge self">Bạn</span>}
                        {item.speaker === "remote" && <span className="spk-badge remote">Đối tác</span>}
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
                        {item.speaker === "self" && <span className="spk-badge self">Bạn</span>}
                        {item.speaker === "remote" && <span className="spk-badge remote">Đối tác</span>}
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

        {userScrolledUp && layoutMode !== "led-stage" && (
          <button className="overlay-jump-btn" onClick={scrollToBottom}>
            <ArrowDown size={14} /> Xuống câu mới nhất
          </button>
        )}

        {showCornerQr && audienceInfo?.qr_svg && !displayFrozen && (
          <div className="overlay-corner-qr" title={audienceInfo.audience_url}>
            <div
              className="overlay-corner-qr-box"
              dangerouslySetInnerHTML={{ __html: audienceInfo.qr_svg }}
            />
            <div className="overlay-corner-qr-caption">
              <strong>Quét QR phụ đề</strong>
              <span>Chọn ngôn ngữ riêng</span>
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
