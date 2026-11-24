import { useEffect, useRef, useState } from "react";
import {
  Check,
  FileText,
  GraduationCap,
  Layers,
  Loader2,
  Plus,
  Presentation,
  Sparkles,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import { ENGINE_HTTP, SlideContextPayload, SlideContextTerm } from "./types";

const MAX_UPLOAD_BYTES = 40 * 1024 * 1024; // 40 MB

async function parseJsonSafe<T>(res: Response): Promise<T | null> {
  try {
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

function formatSlideError(e: unknown, fallback: string): string {
  if (e instanceof Error) {
    const msg = e.message.trim();
    if (!msg || /failed to fetch|networkerror|load failed|unexpected token|json/i.test(msg)) {
      return fallback;
    }
    return msg;
  }
  return fallback;
}

interface SlideContextModalProps {
  open: boolean;
  onClose: () => void;
  context: SlideContextPayload | null;
  onUpdated: (next: SlideContextPayload) => void;
}

export function SlideContextModal({
  open,
  onClose,
  context,
  onUpdated,
}: SlideContextModalProps) {
  const [tab, setTab] = useState<"slides" | "terms" | "paste">("slides");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [topicDraft, setTopicDraft] = useState("");
  const [pasteDraft, setPasteDraft] = useState("");
  const [newSrc, setNewSrc] = useState("");
  const [newTgt, setNewTgt] = useState("");
  const [topicEditing, setTopicEditing] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (open) {
      setErr("");
      setTopicEditing(false);
      setDragOver(false);
    }
  }, [open]);

  if (!open) return null;

  const ctx = context;
  const pages = ctx?.pages || [];
  const terms = ctx?.terms || [];
  const enabled = ctx?.enabled ?? true;
  const activeSlideIdx = ctx?.active_slide_index ?? 0;

  const postTextUpdate = async (body: Record<string, unknown>): Promise<boolean> => {
    setBusy(true);
    setErr("");
    try {
      const res = await fetch(`${ENGINE_HTTP}/api/context/text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await parseJsonSafe<SlideContextPayload & { detail?: string }>(res);
      if (!res.ok || !data) {
        throw new Error(data?.detail || "Không thể cập nhật bối cảnh hội nghị");
      }
      onUpdated(data);
      return true;
    } catch (e) {
      setErr(formatSlideError(e, "Không thể kết nối tới dịch vụ AI để cập nhật bối cảnh."));
      return false;
    } finally {
      setBusy(false);
    }
  };

  const handleUploadFile = async (file: File) => {
    if (file.size > MAX_UPLOAD_BYTES) {
      setErr("Tệp vượt quá giới hạn 40 MB. Vui lòng nén hoặc tách nhỏ tài liệu trước khi nạp.");
      return;
    }
    setBusy(true);
    setErr("");
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${ENGINE_HTTP}/api/context/upload`, {
        method: "POST",
        body: fd,
      });
      const data = await parseJsonSafe<SlideContextPayload & { detail?: string }>(res);
      if (!res.ok || !data) {
        throw new Error(data?.detail || "Lỗi đọc tài liệu Slide");
      }
      onUpdated(data);
      setTab("slides");
    } catch (e) {
      setErr(formatSlideError(e, "Không thể kết nối tới dịch vụ AI hoặc không đọc được tệp tài liệu."));
    } finally {
      setBusy(false);
    }
  };

  const handleClear = async () => {
    setBusy(true);
    setErr("");
    try {
      const res = await fetch(`${ENGINE_HTTP}/api/context`, { method: "DELETE" });
      const data = await parseJsonSafe<SlideContextPayload & { detail?: string }>(res);
      if (!res.ok || !data) {
        throw new Error(data?.detail || "Không thể xóa bối cảnh phiên họp");
      }
      onUpdated(data);
      setPasteDraft("");
      setTopicDraft("");
      setTopicEditing(false);
    } catch (e) {
      setErr(formatSlideError(e, "Không thể kết nối tới dịch vụ AI để xóa bối cảnh."));
    } finally {
      setBusy(false);
    }
  };

  const saveTerms = (nextTerms: SlideContextTerm[]) => {
    void postTextUpdate({ terms: nextTerms });
  };

  const handleAddTerm = () => {
    if (busy) return;
    const s = newSrc.trim();
    if (!s) return;
    const next: SlideContextTerm[] = [
      { source: s, target: newTgt.trim() || s, enabled: true, origin: "manual" },
      ...terms.filter((t) => t.source.toLowerCase() !== s.toLowerCase()),
    ];
    setNewSrc("");
    setNewTgt("");
    saveTerms(next);
  };

  const handleSaveTopic = async () => {
    if (busy) return;
    const ok = await postTextUpdate({ topic: topicDraft });
    if (ok) setTopicEditing(false);
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal-card slide-modal-card"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Nạp Slide & Bối cảnh Hội nghị 100% Offline"
      >
        <div className="modal-head">
          <div className="row" style={{ gap: 10, minWidth: 0 }}>
            <div className="brand-crest sm">
              <img src="/school-logo.png" alt="DAU Voice Pro" />
            </div>
            <div style={{ minWidth: 0 }}>
              <div className="modal-title" style={{ display: "flex", alignItems: "center", gap: 7 }}>
                <span>Slide &amp; Bối cảnh Hội nghị</span>
                <span className="brand-pro-tag">RAG 100% OFFLINE</span>
              </div>
              <div className="modal-sub">
                Nạp .PPTX / .PDF / .DOCX trước giờ G — tự rút tên riêng vào ASR Hotwords &amp; đối chiếu từng trang Slide (&lt;1ms)
              </div>
            </div>
          </div>
          <button className="btn sm icon-only" onClick={onClose} title="Đóng">
            <X size={14} />
          </button>
        </div>

        <div
          className={`modal-body-pad slide-modal-body ${dragOver ? "drag-over" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            if (!dragOver) setDragOver(true);
          }}
          onDragLeave={(e) => {
            if (!e.currentTarget.contains(e.relatedTarget as Node | null)) {
              setDragOver(false);
            }
          }}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            const f = e.dataTransfer.files?.[0];
            if (f && !busy) void handleUploadFile(f);
          }}
        >
          {/* Quick Action Strip: Upload + DAU Preset + Enable Toggle + Clear */}
          <div className="slide-action-bar">
            <input
              ref={fileInputRef}
              type="file"
              accept=".pptx,.pdf,.docx,.txt,.md"
              style={{ display: "none" }}
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) void handleUploadFile(f);
                e.target.value = "";
              }}
            />
            <button
              className="btn primary sm"
              disabled={busy}
              onClick={() => fileInputRef.current?.click()}
            >
              {busy ? <Loader2 size={13} className="spin" /> : <Upload size={13} />}
              {pages.length > 0 ? "Thay Slide / Tài liệu mới" : "Nạp Slide / Tài liệu (.PPTX, .PDF, .DOCX)"}
            </button>

            <button
              className="btn sm"
              disabled={busy}
              onClick={() => void postTextUpdate({ apply_dau_preset: true })}
              title="Nạp 1 chạm 22 thuật ngữ & danh xưng chính thức Trường Đại học Kiến trúc Đà Nẵng (DAU)"
            >
              <GraduationCap size={13} /> Nạp chuẩn DAU
            </button>

            <div style={{ flex: 1 }} />

            <label className="slide-enable-toggle" title="Bật/tắt đối chiếu Slide RAG & Hotwords">
              <span>{enabled ? "Đang bật" : "Đang tắt"}</span>
              <span className="switch">
                <input
                  type="checkbox"
                  checked={enabled}
                  disabled={busy}
                  onChange={(e) => void postTextUpdate({ enabled: e.target.checked })}
                />
                <span className="switch-track" />
              </span>
            </label>

            {(pages.length > 0 || terms.length > 0 || ctx?.topic) && (
              <button
                className="btn sm"
                disabled={busy}
                onClick={handleClear}
                title="Xóa sạch Slide & thuật ngữ phiên họp sau khi kết thúc hội nghị"
              >
                <Trash2 size={12.5} /> Xóa phiên
              </button>
            )}
          </div>

          {err && <div className="slide-err-banner">{err}</div>}

          {/* Drag & Drop Zone: full when empty, compact strip when slides are loaded */}
          {pages.length === 0 && tab !== "paste" ? (
            <div
              className={`slide-dropzone ${dragOver ? "drag-over" : ""}`}
              onClick={() => !busy && fileInputRef.current?.click()}
            >
              <Presentation size={22} color="var(--accent-text)" />
              <div style={{ fontWeight: 700, fontSize: 13 }}>
                Kéo thả file Slide (.PPTX, .PDF, .DOCX, .TXT) vào đây trước khi hội nghị bắt đầu
              </div>
              <div style={{ fontSize: 11.5, color: "var(--text-2)" }}>
                Xử lý 100% nội bộ trên máy trong 2 giây (tối đa 40 MB) — đọc cả ghi chú diễn giả (Speaker Notes) để nghe &amp; dịch chuẩn tên riêng
              </div>
            </div>
          ) : (
            tab === "slides" && (
              <div
                className={`slide-dropzone compact ${dragOver ? "drag-over" : ""}`}
                onClick={() => !busy && fileInputRef.current?.click()}
              >
                <Upload size={14} color="var(--accent-text)" />
                <span>
                  {ctx?.filename ? (
                    <>
                      Đang nạp: <b>{ctx.filename}</b> — kéo thả hoặc bấm để thay Slide mới (≤ 40 MB)
                    </>
                  ) : (
                    "Kéo thả file .PPTX / .PDF / .DOCX mới vào bất kỳ đâu để thay Slide (≤ 40 MB)"
                  )}
                </span>
              </div>
            )
          )}

          {/* Active Topic Banner */}
          <div className="slide-topic-row">
            <span className="slide-topic-label">
              <Sparkles size={12} /> Chủ đề hội nghị:
            </span>
            {topicEditing ? (
              <div className="row" style={{ flex: 1, gap: 6 }}>
                <input
                  className="input"
                  style={{ flex: 1, padding: "4px 8px", fontSize: 12 }}
                  value={topicDraft}
                  autoFocus
                  disabled={busy}
                  onChange={(e) => setTopicDraft(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      void handleSaveTopic();
                    } else if (e.key === "Escape") {
                      e.preventDefault();
                      e.stopPropagation();
                      setTopicEditing(false);
                    }
                  }}
                  placeholder="VD: Hội thảo Quốc tế Kiến trúc Bền vững — Đại học Kiến trúc Đà Nẵng (DAU)"
                />
                <button
                  className="btn primary sm"
                  disabled={busy}
                  onClick={() => void handleSaveTopic()}
                >
                  <Check size={12} /> Lưu
                </button>
              </div>
            ) : (
              <button
                className="slide-topic-value"
                disabled={busy}
                onClick={() => {
                  setTopicDraft(ctx?.topic || "");
                  setTopicEditing(true);
                }}
                title="Bấm để sửa chủ đề hội nghị"
              >
                {ctx?.topic || "Chưa đặt chủ đề — bấm để nhập nhanh (≤ 18 từ)"}
              </button>
            )}
          </div>

          {/* Tabs */}
          <div className="segmented" style={{ marginBottom: 4 }}>
            <button
              className={`seg-btn ${tab === "slides" ? "active" : ""}`}
              onClick={() => setTab("slides")}
            >
              <Layers size={12} /> Trang Slide ({pages.length})
            </button>
            <button
              className={`seg-btn ${tab === "terms" ? "active" : ""}`}
              onClick={() => setTab("terms")}
            >
              <FileText size={12} /> Tên riêng &amp; Thuật ngữ ({terms.filter((t) => t.enabled).length}/{terms.length})
            </button>
            <button
              className={`seg-btn ${tab === "paste" ? "active" : ""}`}
              onClick={() => setTab("paste")}
            >
              <Plus size={12} /> Dán kịch bản / Chương trình
            </button>
          </div>

          {tab === "slides" && (
            <div className="slide-pages-list">
              {pages.length === 0 ? (
                <div className="slide-empty-note">
                  Chưa có trang Slide nào. Nạp file <b>.pptx / .pdf / .docx</b> hoặc bấm <b>Nạp chuẩn DAU</b> ở trên.
                </div>
              ) : (
                pages.map((p) => {
                  const isActive = p.index === activeSlideIdx;
                  return (
                    <button
                      key={p.index}
                      type="button"
                      disabled={busy}
                      className={`slide-page-card ${isActive ? "active" : ""}`}
                      onClick={() => void postTextUpdate({ active_slide_index: p.index })}
                      title="Bấm để ghim trang Slide này làm vị trí hiện tại của diễn giả"
                    >
                      <div className="slide-page-head">
                        <span className="slide-page-num">Slide #{p.index}</span>
                        <span className="slide-page-title">{p.title}</span>
                        {isActive && <span className="slide-page-live">Đang khớp</span>}
                      </div>
                      <div className="slide-page-preview">{p.preview}</div>
                      {p.terms && p.terms.length > 0 && (
                        <div className="slide-page-tags">
                          {p.terms.slice(0, 6).map((tg) => (
                            <span key={tg} className="slide-term-chip">
                              {tg}
                            </span>
                          ))}
                        </div>
                      )}
                    </button>
                  );
                })
              )}
            </div>
          )}

          {tab === "terms" && (
            <div className="slide-terms-wrap">
              <div className="row" style={{ gap: 6 }}>
                <input
                  className="input"
                  style={{ flex: 1 }}
                  value={newSrc}
                  disabled={busy}
                  onChange={(e) => setNewSrc(e.target.value)}
                  placeholder="Tên riêng / Thuật ngữ gốc (VD: PGS.TS. Phạm Anh Tuấn)"
                  onKeyDown={(e) => e.key === "Enter" && handleAddTerm()}
                />
                <input
                  className="input"
                  style={{ flex: 1 }}
                  value={newTgt}
                  disabled={busy}
                  onChange={(e) => setNewTgt(e.target.value)}
                  placeholder="Bản dịch khóa (để trống = giữ nguyên)"
                  onKeyDown={(e) => e.key === "Enter" && handleAddTerm()}
                />
                <button className="btn primary sm" disabled={busy || !newSrc.trim()} onClick={handleAddTerm}>
                  <Plus size={12.5} /> Thêm
                </button>
              </div>
              <div className="slide-terms-list">
                {terms.length === 0 ? (
                  <div className="slide-empty-note">
                    Chưa có thuật ngữ nào. Nạp file Slide hoặc bấm <b>Nạp chuẩn DAU</b> để tự tạo danh sách.
                  </div>
                ) : (
                  terms.map((t, idx) => (
                    <div key={`${t.source}-${idx}`} className={`slide-term-row ${t.enabled ? "" : "off"}`}>
                      <input
                        type="checkbox"
                        checked={t.enabled}
                        disabled={busy}
                        onChange={(e) => {
                          const next = terms.map((x, i) =>
                            i === idx ? { ...x, enabled: e.target.checked } : x
                          );
                          saveTerms(next);
                        }}
                      />
                      <span className="slide-term-src">{t.source}</span>
                      <span className="slide-term-sep">→</span>
                      <span className="slide-term-tgt">{t.target || t.source}</span>
                      <span className="slide-term-origin">
                        {t.origin === "dau_preset" ? "DAU" : t.origin === "manual" ? "Tay" : "Slide"}
                      </span>
                      <button
                        className="btn sm icon-only"
                        disabled={busy}
                        title="Xóa thuật ngữ"
                        onClick={() => saveTerms(terms.filter((_, i) => i !== idx))}
                      >
                        <Trash2 size={11.5} />
                      </button>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}

          {tab === "paste" && (
            <div className="slide-paste-wrap">
              <textarea
                className="input"
                rows={7}
                disabled={busy}
                style={{ resize: "vertical", lineHeight: 1.5 }}
                value={pasteDraft}
                onChange={(e) => setPasteDraft(e.target.value)}
                placeholder={
                  "Dán nhanh chương trình hội nghị, danh sách diễn giả hoặc nội dung bài phát biểu vào đây (tách slide bằng dòng '---')..."
                }
              />
              <div className="row" style={{ justifyContent: "flex-end", gap: 8 }}>
                <button
                  className="btn primary sm"
                  disabled={busy || !pasteDraft.trim()}
                  onClick={async () => {
                    const ok = await postTextUpdate({ text: pasteDraft });
                    if (ok) {
                      setPasteDraft("");
                      setTab("slides");
                    }
                  }}
                >
                  {busy ? <Loader2 size={12.5} className="spin" /> : <Check size={12.5} />}
                  Phân tích &amp; Nạp vào phiên họp (100% Offline)
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
