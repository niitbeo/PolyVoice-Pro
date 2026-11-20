import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlignLeft,
  ArrowDown,
  Columns2,
  Contrast,
  GripHorizontal,
  Languages,
  Maximize2,
  Mic,
  Minimize2,
  Minus,
  Moon,
  Pin,
  PinOff,
  Plus,
  Presentation,
  Radio,
  Rows2,
  Sun,
  Trash2,
  Type,
  X,
} from "lucide-react";
import {
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
    if (saved === "stacked" || saved === "target-only" || saved === "split") {
      return saved;
    }
  } catch {
    // ignore
  }
  return "stacked";
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
  return <span className="overlay-translating-dots">Đang dịch...</span>;
}

export default function OverlayWindow() {
  const [status, setStatus] = useState<EngineStatus | null>(null);
  const [entries, setEntries] = useState<TranscriptEntry[]>([]);
  const [partial, setPartial] = useState<PartialEvent | null>(null);

  const [layoutMode, setLayoutMode] = useState<OverlayLayoutMode>(() => getInitialOverlayLayout());
  const [stageMode, setStageMode] = useState(false);
  const [controlsVisible, setControlsVisible] = useState(true);
  const [fontSize, setFontSize] = useState(26);
  const [opacity, setOpacity] = useState(0.94);
  const [theme, setTheme] = useState<ThemeMode>(() => getInitialTheme());
  const [alwaysOnTop, setAlwaysOnTop] = useState(true);
  const [filledMonitor, setFilledMonitor] = useState(false);
  const [userScrolledUp, setUserScrolledUp] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const pinnedToBottomRef = useRef(true);
  const scrollAnimRef = useRef<number | null>(null);
  const hideBarTimerRef = useRef<number | null>(null);

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
        fetch(`${ENGINE_HTTP}/api/health`)
          .then((r) => (r.ok ? r.json() : null))
          .then((d) => {
            if (d) setStatus(d);
          })
          .catch(() => {});
      };

      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === "status") {
            setStatus(msg.data);
          } else if (msg.type === "partial") {
            const d = msg.data;
            setPartial((prev) => {
              const same = Boolean(prev && prev.id === d.id && prev.target_lang === d.target_lang);
              return {
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
            });
          } else if (msg.type === "partial_mt") {
            const d = msg.data;
            setPartial((prev) =>
              prev && prev.id === d.id
                ? {
                    ...prev,
                    translated_text: d.translated_text,
                    stable_text: d.stable_text ?? "",
                    unstable_text: d.unstable_text ?? "",
                    mt_ms: d.mt_ms ?? 0,
                    mt_engine: d.mt_engine,
                  }
                : prev
            );
          } else if (msg.type === "final") {
            const entry: TranscriptEntry = msg.data;
            setPartial((prev) => (prev && prev.id === entry.id ? null : prev));
            setEntries((prev) => {
              if (prev.some((x) => x.id === entry.id)) return prev;
              return [...prev, entry].slice(-200);
            });
          } else if (msg.type === "history" && Array.isArray(msg.data)) {
            setEntries(msg.data.slice(-200));
            if (msg.data.length === 0) setPartial(null);
          }
        } catch {
          // ignore
        }
      };

      ws.onclose = () => {
        if (!closed) {
          timer = window.setTimeout(connect, 1200);
        }
      };
    };

    connect();
    return () => {
      closed = true;
      if (timer) clearTimeout(timer);
      wsRef.current?.close();
      document.body.classList.remove("overlay-body");
    };
  }, []);

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

  const cardBg =
    theme === "light"
      ? `rgba(255, 255, 255, ${opacity})`
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
        {/* Top Draggable Header — Auto-hides in Stage Mode until pointer approaches top edge */}
        <header
          className={`overlay-topbar ${hideTopbar ? "auto-hidden" : ""}`}
          data-tauri-drag-region
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
          <div className="overlay-topbar-left" data-tauri-drag-region>
            <div className="brand-crest sm" data-tauri-drag-region>
              <img src="/school-logo.png" alt="PolyVoice Pro" />
            </div>
            <span className="overlay-brand" data-tauri-drag-region>
              <span>PolyVoice</span>
              <span className="brand-pro-tag">PRO</span>
            </span>
            <span className={`pill overlay-status-pill ${status?.listening ? "live" : "ok"}`}>
              <span className={`dot ${status?.listening ? "pulse" : ""}`} />
              {status?.listening ? "Đang phát" : "Sẵn sàng"}
            </span>
            <span className="overlay-drag-hint" data-tauri-drag-region>
              <GripHorizontal size={12} /> Kéo sang màn hình #2
            </span>
          </div>

          <div className="overlay-topbar-actions">
            {/* 3-Layout Presentation Switcher */}
            <div className="overlay-layout-seg" role="group" aria-label="Bố cục hiển thị">
              <button
                className={`overlay-layout-btn ${layoutMode === "stacked" ? "active" : ""}`}
                onClick={() => setLayoutMode("stacked")}
                title="Xếp tầng chuẩn hội trường: Gốc nhỏ ở trên ↑ • Bản dịch lớn ở dưới ↓"
              >
                <Rows2 size={12} /> Xếp tầng
              </button>
              <button
                className={`overlay-layout-btn ${layoutMode === "target-only" ? "active" : ""}`}
                onClick={() => setLayoutMode("target-only")}
                title="Chỉ hiển thị Bản Dịch cỡ lớn (tối ưu cho khán giả ngồi xa)"
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

            <span className="overlay-sep" />

            {/* Clean Stage Mode (auto-hides controls & hides technical metrics) */}
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
              className={`btn sm icon-only ${alwaysOnTop ? "info-active" : ""}`}
              title={alwaysOnTop ? "Đang ghim nổi trên cùng (bấm để tắt)" : "Ghim nổi trên cùng"}
              onClick={handleTogglePin}
            >
              {alwaysOnTop ? <Pin size={13} /> : <PinOff size={13} />}
            </button>

            <button
              className={`btn sm ${filledMonitor ? "info-active" : ""}`}
              title={filledMonitor ? "Thu về kích thước cửa sổ" : "Phóng đầy màn hình hiện tại (màn hình 1 hoặc 2)"}
              onClick={handleToggleFillMonitor}
            >
              {filledMonitor ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
              <span>{filledMonitor ? "Thu gọn" : "Đầy màn hình"}</span>
            </button>

            <span className="overlay-sep" />

            <button
              className="btn sm icon-only"
              title="Xóa chữ trên màn hình phụ đề"
              onClick={() => {
                setEntries([]);
                setPartial(null);
              }}
            >
              <Trash2 size={13} />
            </button>
            <button className="btn sm icon-only" title="Đóng màn hình phụ đề" onClick={handleClose}>
              <X size={13} />
            </button>
          </div>
        </header>

        {/* Sub-header language ribbon (hidden cleanly when stageMode auto-hides chrome in target-only/stacked) */}
        {layoutMode === "split" ? (
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
        ) : (
          !stageMode && (
            <div className="overlay-single-header">
              <span className="col-badge tgt">
                <Languages size={11} /> {layoutMode === "target-only" ? "CHỈ BẢN DỊCH" : "XẾP TẦNG HỘI TRƯỜNG"}
              </span>
              <span className="col-lang">
                {layoutMode === "target-only" ? tgtLabel : `${srcLabel} → ${tgtLabel}`}
              </span>
            </div>
          )
        )}

        {/* Scrollable Conference Subtitle Feed */}
        <div className="overlay-feed" ref={scrollRef} onScroll={handleScroll}>
          {isEmpty ? (
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

              {partial && (
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

        {userScrolledUp && (
          <button className="overlay-jump-btn" onClick={scrollToBottom}>
            <ArrowDown size={14} /> Xuống câu mới nhất
          </button>
        )}
      </div>
    </div>
  );
}
