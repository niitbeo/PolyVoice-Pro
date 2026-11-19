import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowDown,
  Maximize2,
  Minimize2,
  Minus,
  Moon,
  Move,
  Pin,
  PinOff,
  Plus,
  Sun,
  Trash2,
  X,
} from "lucide-react";
import {
  ENGINE_HTTP,
  ENGINE_WS,
  EngineStatus,
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

export default function OverlayWindow() {
  const [status, setStatus] = useState<EngineStatus | null>(null);
  const [entries, setEntries] = useState<TranscriptEntry[]>([]);
  const [partial, setPartial] = useState<PartialEvent | null>(null);

  const [fontSize, setFontSize] = useState(24);
  const [opacity, setOpacity] = useState(0.94);
  const [theme, setTheme] = useState<ThemeMode>(() => getInitialTheme());
  const [alwaysOnTop, setAlwaysOnTop] = useState(true);
  const [filledMonitor, setFilledMonitor] = useState(false);
  const [userScrolledUp, setUserScrolledUp] = useState(false);

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
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  const wsRef = useRef<WebSocket | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const pinnedToBottomRef = useRef(true);

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
            setPartial((prev) => ({
              id: d.id,
              source_text: d.source_text,
              translated_text:
                prev && prev.id === d.id && prev.target_lang === d.target_lang
                  ? prev.translated_text
                  : "",
              source_lang: d.source_lang || "vie_Latn",
              target_lang: d.target_lang,
              asr_ms: d.asr_ms ?? 0,
              mt_ms: prev && prev.id === d.id ? prev.mt_ms : 0,
            }));
          } else if (msg.type === "partial_mt") {
            const d = msg.data;
            setPartial((prev) =>
              prev && prev.id === d.id
                ? {
                    ...prev,
                    translated_text: d.translated_text,
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

  // Auto-scroll upward as new messages arrive when pinned to bottom
  useEffect(() => {
    const el = scrollRef.current;
    if (!el || !pinnedToBottomRef.current) return;
    el.scrollTop = el.scrollHeight;
  }, [entries, partial, fontSize]);

  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    const atBottom = distanceFromBottom < 64;
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
      } catch {
        // ignore
      }
    } else if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen?.().catch(() => {});
      setFilledMonitor(true);
    } else {
      document.exitFullscreen?.().catch(() => {});
      setFilledMonitor(false);
    }
  };

  const handleClose = async () => {
    if (isTauriEnv()) {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("toggle_overlay");
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
    currentParaMode === "long" ? "Đoạn: Dài" : currentParaMode === "medium" ? "Đoạn: Vừa" : "Đoạn: Từng câu";

  const groupedEntries = useMemo(() => groupEntriesByParagraph(entries), [entries]);

  const srcLabel = status?.source_lang_meta?.name_vi || "Tiếng Việt";
  const tgtLabel = status?.target_lang_meta?.name_vi || status?.config?.target_lang || "Tiếng Anh";

  const cardBg =
    theme === "light"
      ? `rgba(255, 255, 255, ${opacity})`
      : `rgba(10, 15, 28, ${opacity})`;

  return (
    <div className={`overlay-stage-root ${filledMonitor ? "filled" : ""}`}>
      <div className="overlay-stage-card" style={{ background: cardBg }}>
        {/* Top Draggable Header — Drag this bar to move onto 2nd monitor */}
        <header className="overlay-topbar" data-tauri-drag-region>
          <div className="overlay-topbar-left" data-tauri-drag-region>
            <div className="brand-crest sm" data-tauri-drag-region>
              <img src="/school-logo.png" alt="Logo trường" />
            </div>
            <span
              className={`dot ${status?.listening ? "pulse" : ""}`}
              style={{ color: status?.listening ? "var(--red)" : "var(--green)" }}
            />
            <span className="overlay-brand" data-tauri-drag-region>
              PolyVoice <span className="brand-pro-tag">PRO</span>
            </span>
            <span className="overlay-drag-hint" data-tauri-drag-region>
              <Move size={11} /> Kéo sang màn hình #2
            </span>
          </div>

          <div className="overlay-topbar-actions">
            <button
              className="btn sm active"
              title="Dịch chuẩn từng câu kiểu YouTube và gom hiển thị theo đoạn (Dài: ~3–4 câu / Vừa: ~2 câu / Từng câu)"
              onClick={cycleParagraphMode}
            >
              {paraLabel}
            </button>
            <button
              className="btn sm icon-only"
              title={theme === "dark" ? "Chuyển phụ đề sang nền Sáng (cho phòng chiếu sáng)" : "Chuyển phụ đề sang nền Tối"}
              onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
            >
              {theme === "dark" ? <Sun size={13} /> : <Moon size={13} />}
            </button>
            <button
              className={`btn sm ${alwaysOnTop ? "active" : ""}`}
              title={alwaysOnTop ? "Đang ghim nổi trên cùng (bấm để tắt)" : "Ghim nổi trên cùng"}
              onClick={handleTogglePin}
            >
              {alwaysOnTop ? <Pin size={13} /> : <PinOff size={13} />}
            </button>
            <button
              className="btn sm"
              title="Giảm cỡ chữ"
              onClick={() => setFontSize((s) => Math.max(14, s - 2))}
            >
              <Minus size={13} />
            </button>
            <span className="overlay-size-tag">{fontSize}px</span>
            <button
              className="btn sm"
              title="Tăng cỡ chữ"
              onClick={() => setFontSize((s) => Math.min(48, s + 2))}
            >
              <Plus size={13} />
            </button>
            <button
              className="btn sm"
              title="Đổi độ mờ nền (78% / 94% / 100% Đen đặc)"
              onClick={() => setOpacity((o) => (o >= 0.99 ? 0.78 : o >= 0.9 ? 1.0 : 0.94))}
            >
              {Math.round(opacity * 100)}%
            </button>
            <button
              className={`btn sm ${filledMonitor ? "active" : ""}`}
              title={filledMonitor ? "Thu về kích thước cửa sổ" : "Phóng đầy màn hình hiện tại (màn hình 1 hoặc 2)"}
              onClick={handleToggleFillMonitor}
            >
              {filledMonitor ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
              {filledMonitor ? "Thu gọn" : "Đầy màn hình"}
            </button>
            <button
              className="btn sm"
              title="Xóa chữ trên màn hình phụ đề"
              onClick={() => {
                setEntries([]);
                setPartial(null);
              }}
            >
              <Trash2 size={13} />
            </button>
            <button className="btn sm" title="Đóng màn hình phụ đề" onClick={handleClose}>
              <X size={13} />
            </button>
          </div>
        </header>

        {/* Fixed 2-Column Language Headers */}
        <div className="overlay-col-headers">
          <div className="overlay-col-head left">
            <span className="col-badge src">NGÔN NGỮ GỐC</span>
            <span className="col-lang">{srcLabel}</span>
          </div>
          <div className="overlay-col-head right">
            <span className="col-badge tgt">NGÔN NGỮ DỊCH</span>
            <span className="col-lang">{tgtLabel}</span>
          </div>
        </div>

        {/* Scrollable Paired Bilingual Conversation Feed (pushes upward as turns arrive) */}
        <div className="overlay-feed" ref={scrollRef} onScroll={handleScroll}>
          <div className="overlay-feed-inner">
            {entries.length === 0 && !partial && (
              <div className="overlay-empty-split">
                <div className="overlay-empty-col">
                  Đang chờ giọng nói <b>{srcLabel}</b>...
                </div>
                <div className="overlay-empty-col right">
                  Bản dịch <b>{tgtLabel}</b> sẽ hiển thị song song tại đây và tự đẩy dần lên trên theo cuộc trò chuyện.
                </div>
              </div>
            )}

            {groupedEntries.map((item) => (
              <div className="overlay-row" key={item.id}>
                <div className="overlay-cell left">
                  <div className="overlay-cell-meta">
                    {item.speaker === "self" && <span className="spk-badge self">Bạn</span>}
                    {item.speaker === "remote" && <span className="spk-badge remote">Đối tác</span>}
                    {item.timestamp}
                    {(item.sentence_count ?? 1) > 1 ? ` • ${item.sentence_count} câu` : ""}
                  </div>
                  <div className="overlay-cell-text src" style={{ fontSize }}>
                    {item.source_text}
                  </div>
                </div>
                <div className="overlay-cell right">
                  <div className="overlay-cell-meta">
                    {item.timestamp}
                    {item.mt_ms > 0 ? ` • ${item.mt_ms}ms` : ""}
                  </div>
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
                    <span className="dot pulse" /> Đang nói...
                  </div>
                  <div className="overlay-cell-text src draft" style={{ fontSize }}>
                    {partial.source_text}
                  </div>
                </div>
                <div className="overlay-cell right">
                  <div className="overlay-cell-meta live-tag">
                    <span className="dot pulse" /> Đang dịch trực tiếp...
                  </div>
                  <div className="overlay-cell-text tgt draft" style={{ fontSize }}>
                    {partial.translated_text || "..."}
                  </div>
                </div>
              </div>
            )}
          </div>
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
