import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  ArrowLeftRight,
  ArrowUpCircle,
  BookOpen,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clock,
  Cloud,
  Copy,
  Cpu,
  Download,
  FileAudio,
  FileText,
  FolderOpen,
  Globe,
  Headphones,
  Layers,
  Loader2,
  Mic,
  MicOff,
  MonitorSpeaker,
  Moon,
  MousePointerClick,
  Play,
  Plus,
  QrCode,
  RefreshCw,
  Search,
  Send,
  Settings,
  Settings2,
  SlidersHorizontal,
  Smartphone,
  Sparkles,
  Sun,
  Trash2,
  Tv,
  Upload,
  Volume2,
  VolumeX,
  Wifi,
  X,
  Zap,
} from "lucide-react";
import {
  AsrEngineId,
  AudienceInfo,
  AudioDevice,
  AudioSourceId,
  CloudApiStatus,
  CloudProviderId,
  ENGINE_HTTP,
  ENGINE_WS,
  EngineStatus,
  FileTranslateProgress,
  FileTranslateResult,
  GlossaryData,
  InputChannelId,
  LanguageEntry,
  MeetingSummary,
  ModelState,
  ModelsStatus,
  MtModeId,
  ParagraphModeId,
  PartialEvent,
  SourceLanguageEntry,
  THEME_STORAGE_KEY,
  ThemeMode,
  TranscriptEntry,
  applyTheme,
  getInitialTheme,
  groupEntriesByParagraph,
  isTauriEnv,
  shortEngineName,
  shortLangName,
  speakOfflineTTS,
} from "./types";

const APP_VERSION = typeof __APP_VERSION__ !== "undefined" ? __APP_VERSION__ : "0.8.4";
const GITHUB_REPO = "niitbeo/PolyVoice-Pro";
const VU_BARS = 18;

const MODEL_DISPLAY_NAMES: Record<string, string> = {
  "zipformer-30m-streaming": "Nghe Tiếng Việt (+Hotwords)",
  "zipformer-en-20m": "Nghe Tiếng Anh (Nháp live)",
  "moonshine-en": "Nghe Tiếng Anh (Moonshine-v2)",
  "whisper-turbo": "Nghe Đa ngôn ngữ 99 NN (Whisper-Turbo)",
  envit5: "Dịch Việt ⇄ Anh (EnViT5)",
  "nllb-600m": "Dịch Lào & 202 NN (NLLB-600M)",
  "hy-mt2": "Dịch Thái / 38 NN (Hy-MT2)",
  "nllb-1.3b": "Dịch 202 NN (NLLB-1.3B)",
  "opus-vi-en": "Dịch Việt → Anh (Opus-MT)",
  "opus-en-vi": "Dịch Anh → Việt (Opus-MT)",
};

function applyModelStateUpdate(prev: ModelsStatus | null, prog: ModelState): ModelsStatus | null {
  if (!prev) return prev;
  if (prog.model_id === "hy-mt2") return { ...prev, mt_hymt2: prog };
  if (prog.model_id === "nllb-1.3b") return { ...prev, mt_nllb_13b: prog };
  if (prog.model_id === "nllb-600m") return { ...prev, mt_nllb_600m: prog };
  if (prog.model_id === "zipformer-30m-streaming")
    return { ...prev, asr_zipformer_streaming: prog };
  if (prog.model_id === "zipformer-en-20m")
    return { ...prev, asr_zipformer_en_20m: prog };
  if (prog.model_id === "moonshine-en")
    return { ...prev, asr_moonshine_en: prog };
  if (prog.model_id === "whisper-turbo")
    return { ...prev, asr_whisper_turbo: prog };
  if (prog.model_id === "opus-vi-en") return { ...prev, mt_opus_vi_en: prog };
  if (prog.model_id === "opus-en-vi") return { ...prev, mt_opus_en_vi: prog };
  if (prog.model_id === "envit5") return { ...prev, mt_envit5: prog };
  return prev;
}

function stripFileExtension(filename: string): string {
  return filename.replace(/\.[^./\\]+$/, "") || filename;
}

function makeFileProgKey(p: FileTranslateProgress): string {
  return `${p.job_id}|${p.stage}|${p.current_chunk}|${p.percent}|${p.eta_s ?? ""}`;
}

function compareSemver(a: string, b: string): number {
  const pa = a.replace(/^v/i, "").split(".").map((n) => parseInt(n, 10) || 0);
  const pb = b.replace(/^v/i, "").split(".").map((n) => parseInt(n, 10) || 0);
  for (let i = 0; i < 3; i++) {
    if ((pa[i] || 0) !== (pb[i] || 0)) return (pa[i] || 0) - (pb[i] || 0);
  }
  return 0;
}

function formatClockSec(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds || 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) {
    return `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
  }
  return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

function formatEtaVi(seconds: number | null | undefined, estimated?: boolean): string {
  if (seconds == null || !Number.isFinite(seconds)) return "Đang ước tính...";
  const s = Math.max(0, Math.round(seconds));
  if (s <= 4) return "< 5 giây";
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const prefix = estimated ? "Ước tính ~" : "Còn ~";
  if (h > 0) return m > 0 ? `${prefix}${h} giờ ${m} phút` : `${prefix}${h} giờ`;
  if (m > 0) return sec > 0 ? `${prefix}${m} phút ${sec} giây` : `${prefix}${m} phút`;
  return `${prefix}${sec} giây`;
}
const PINNED_CODES = [
  "vie_Latn",
  "eng_Latn",
  "tha_Thai",
  "lao_Laoo",
  "zho_Hans",
  "jpn_Jpan",
  "kor_Hang",
  "khm_Khmr",
];

export default function App() {
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState<EngineStatus | null>(null);
  const [models, setModels] = useState<ModelsStatus | null>(null);
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [loopbackDevices, setLoopbackDevices] = useState<AudioDevice[]>([]);
  const [refreshingDevices, setRefreshingDevices] = useState(false);
  const [languages, setLanguages] = useState<LanguageEntry[]>([]);
  const [sourceLanguages, setSourceLanguages] = useState<SourceLanguageEntry[]>([]);

  const [audioSource, setAudioSource] = useState<AudioSourceId>("mic");
  const [selectedDevice, setSelectedDevice] = useState<number | null>(null);
  const [selectedLoopback, setSelectedLoopback] = useState<number | null>(null);
  const [inputChannel, setInputChannel] = useState<InputChannelId>("mix");
  const [inputGainDb, setInputGainDb] = useState<number>(0);
  const [autoGain, setAutoGain] = useState<boolean>(false);
  const [showMixerPanel, setShowMixerPanel] = useState<boolean>(false);
  const [clipping, setClipping] = useState<boolean>(false);
  const clipTimerRef = useRef<number | undefined>(undefined);
  const [sourceLang, setSourceLang] = useState("vie_Latn");
  const [targetLang, setTargetLang] = useState("eng_Latn");
  const [langQuery, setLangQuery] = useState("");
  const [asrEngine, setAsrEngine] = useState<AsrEngineId>("auto");
  const [mtMode, setMtMode] = useState<MtModeId>("auto-offline");
  const [lastOfflineMtMode, setLastOfflineMtMode] = useState<MtModeId>("auto-offline");
  const [paragraphMode, setParagraphMode] = useState<ParagraphModeId>("long");
  const [pass2Refine, setPass2Refine] = useState(true);
  const [ecoMode, setEcoMode] = useState(false);
  const [beamSize, setBeamSize] = useState(1);
  const [translatePartials, setTranslatePartials] = useState(true);
  const [llmUrl, setLlmUrl] = useState("http://127.0.0.1:11434/v1");
  const [llmModel, setLlmModel] = useState("translategemma:4b");
  const [ttsEnabled, setTtsEnabled] = useState(false);
  const [overlayOpen, setOverlayOpen] = useState(false);
  const [clickThrough, setClickThrough] = useState(false);
  const [theme, setTheme] = useState<ThemeMode>(() => getInitialTheme());
  const [showAllLangs, setShowAllLangs] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

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
  const [glossaryOpen, setGlossaryOpen] = useState(false);
  const [glossary, setGlossary] = useState<GlossaryData | null>(null);
  const [newAsrWrong, setNewAsrWrong] = useState("");
  const [newAsrCorrect, setNewAsrCorrect] = useState("");
  const [newLockSource, setNewLockSource] = useState("");
  const [newLockTarget, setNewLockTarget] = useState("");
  const [savedLogCount, setSavedLogCount] = useState(0);

  // v0.6 Modals: QR Audience + OBS, Batch File Translator, AI Meeting Summary
  const [qrOpen, setQrOpen] = useState(false);
  const [audienceInfo, setAudienceInfo] = useState<AudienceInfo | null>(null);
  const [audienceCount, setAudienceCount] = useState(0);

  const [fileModalOpen, setFileModalOpen] = useState(false);
  const [fileBusy, setFileBusy] = useState(false);
  const [fileCancelling, setFileCancelling] = useState(false);
  const [fileProgress, setFileProgress] = useState<FileTranslateProgress | null>(null);
  const [fileTickNow, setFileTickNow] = useState(() => Date.now());
  const fileStartTsRef = useRef<number>(0);
  const fileProgUpdatedAtRef = useRef<number>(0);
  const lastFileProgKeyRef = useRef<string>("");
  const [fileResult, setFileResult] = useState<FileTranslateResult | null>(null);
  const [setupBannerDismissed, setSetupBannerDismissed] = useState(false);

  const [summaryOpen, setSummaryOpen] = useState(false);
  const [summaryBusy, setSummaryBusy] = useState(false);
  const [summaryData, setSummaryData] = useState<MeetingSummary | null>(null);

  // Official Cloud API Modal (Google Cloud / Microsoft Azure F0 / Google Gemini + $0 Quota Cap)
  const [cloudOpen, setCloudOpen] = useState(false);
  const [cloudStatus, setCloudStatus] = useState<CloudApiStatus | null>(null);
  const [cloudTab, setCloudTab] = useState<CloudProviderId>("azure");
  const [cloudKeyInput, setCloudKeyInput] = useState("");
  const [cloudRegion, setCloudRegion] = useState("southeastasia");
  const [cloudEndpoint, setCloudEndpoint] = useState("https://api.cognitive.microsofttranslator.com");
  const [cloudModel, setCloudModel] = useState("gemini-2.5-flash");
  const [cloudLimit, setCloudLimit] = useState(1950000);
  const [cloudHardCap, setCloudHardCap] = useState(true);
  const [cloudDraftOffline, setCloudDraftOffline] = useState(true);
  const [cloudSaving, setCloudSaving] = useState(false);
  const [cloudTestBusy, setCloudTestBusy] = useState(false);
  const [cloudTestResult, setCloudTestResult] = useState<{
    ok: boolean;
    message: string;
  } | null>(null);

  // Native 1-Click Auto-Updater State (Tauri v2 Plugin Updater + GitHub Releases)
  const [updateInfo, setUpdateInfo] = useState<{
    version: string;
    date?: string;
    body?: string;
    downloadUrl?: string;
    nativeUpdate?: {
      downloadAndInstall: (cb?: (ev: { event: string; data?: { contentLength?: number; chunkLength?: number } }) => void) => Promise<void>;
    };
  } | null>(null);
  const [updateModalOpen, setUpdateModalOpen] = useState(false);
  const [updateChecking, setUpdateChecking] = useState(false);
  const [updateCheckMsg, setUpdateCheckMsg] = useState("");
  const [updateInstalling, setUpdateInstalling] = useState(false);
  const [updateProgress, setUpdateProgress] = useState(0);
  const [updateProgressText, setUpdateProgressText] = useState("");

  const [level, setLevel] = useState(0);
  const [partial, setPartial] = useState<PartialEvent | null>(null);
  const [history, setHistory] = useState<TranscriptEntry[]>([]);
  const [errorMsg, setErrorMsg] = useState("");
  const [cloudNotice, setCloudNotice] = useState("");
  const [modelSuccessNotice, setModelSuccessNotice] = useState("");
  const [justDownloadedIds, setJustDownloadedIds] = useState<Record<string, boolean>>({});
  const [quickText, setQuickText] = useState("");
  const [quickBusy, setQuickBusy] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const ttsRef = useRef(ttsEnabled);
  ttsRef.current = ttsEnabled;
  const listeningRef = useRef(false);
  listeningRef.current = Boolean(status?.listening);
  const toggleListeningRef = useRef<() => void>(() => {});
  const copyTimerRef = useRef<number | undefined>(undefined);
  const llmDebounceRef = useRef<number | undefined>(undefined);
  const overlayPopupRef = useRef<Window | null>(null);
  const activeDownloadsRef = useRef<Set<string>>(new Set());
  const modelSuccessTimerRef = useRef<number | undefined>(undefined);
  const justDownloadedTimersRef = useRef<Record<string, number>>({});

  const isMacOs = useMemo(() => {
    if (status?.platform?.os) return status.platform.os === "Darwin";
    if (typeof navigator !== "undefined") {
      return /Mac|iPhone|iPad|iPod/i.test(navigator.platform || navigator.userAgent || "");
    }
    return false;
  }, [status?.platform?.os]);

  const syncStatusToLocalState = useCallback((h: EngineStatus, syncLlmInputs = false) => {
    setStatus(h);
    if (h.config?.source_lang) setSourceLang(h.config.source_lang);
    if (h.config?.target_lang) setTargetLang(h.config.target_lang);
    if (h.config?.audio_source) setAudioSource(h.config.audio_source);
    if (h.config?.input_channel) setInputChannel(h.config.input_channel);
    if (typeof h.config?.input_gain_db === "number") setInputGainDb(h.config.input_gain_db);
    if (typeof h.config?.auto_gain === "boolean") setAutoGain(h.config.auto_gain);
    if (typeof h.config?.loopback_device_id === "number") setSelectedLoopback(h.config.loopback_device_id);
    if (h.config?.asr_engine) setAsrEngine(h.config.asr_engine);
    if (h.config?.mt_mode) {
      setMtMode(h.config.mt_mode);
      if (h.config.mt_mode !== "cloud-api") {
        setLastOfflineMtMode(h.config.mt_mode);
      }
    }
    if (h.config?.paragraph_mode) setParagraphMode(h.config.paragraph_mode);
    if (typeof h.config?.pass2_refine === "boolean") setPass2Refine(h.config.pass2_refine);
    if (typeof h.config?.eco_mode === "boolean") setEcoMode(h.config.eco_mode);
    if (typeof h.config?.translate_partials === "boolean") {
      setTranslatePartials(h.config.translate_partials);
    }
    if (typeof h.config?.beam_size === "number") setBeamSize(h.config.beam_size);
    if (syncLlmInputs) {
      if (typeof h.config?.llm_url === "string") setLlmUrl(h.config.llm_url);
      if (typeof h.config?.llm_model === "string") setLlmModel(h.config.llm_model);
    }
    if (typeof h.logs?.saved_count === "number") setSavedLogCount(h.logs.saved_count);
    if (typeof h.audience?.connected_count === "number") setAudienceCount(h.audience.connected_count);
    if (typeof h.cloud_fallback_notice === "string") setCloudNotice(h.cloud_fallback_notice);
  }, []);

  const fetchInitialData = useCallback(async () => {
    try {
      const [hRes, dRes, lRes, slRes, mRes] = await Promise.all([
        fetch(`${ENGINE_HTTP}/api/health`),
        fetch(`${ENGINE_HTTP}/api/devices`),
        fetch(`${ENGINE_HTTP}/api/languages`),
        fetch(`${ENGINE_HTTP}/api/source-languages`),
        fetch(`${ENGINE_HTTP}/api/models/status`),
      ]);
      if (hRes.ok) {
        const h: EngineStatus = await hRes.json();
        syncStatusToLocalState(h, true);
      }
      if (dRes.ok) {
        const d = await dRes.json();
        const inList: AudioDevice[] = d.devices || [];
        const lbList: AudioDevice[] = d.loopback_devices || [];
        setDevices(inList);
        setLoopbackDevices(lbList);
        const def = inList.find((x: AudioDevice) => x.is_default);
        if (def) setSelectedDevice((prev) => (prev === null || !inList.some((i) => i.id === prev) ? def.id : prev));
        const defLb = lbList.find((x: AudioDevice) => x.is_default);
        if (defLb) setSelectedLoopback((prev) => (prev === null || !lbList.some((i) => i.id === prev) ? defLb.id : prev));
      }
      if (lRes.ok) {
        const l = await lRes.json();
        setLanguages(l.languages || []);
      }
      if (slRes.ok) {
        const sl = await slRes.json();
        setSourceLanguages(sl.languages || []);
      }
      if (mRes.ok) {
        setModels(await mRes.json());
      }
      setConnected(true);
    } catch {
      setConnected(false);
    }
  }, [syncStatusToLocalState]);

  // Connect WebSocket with auto-reconnect
  useEffect(() => {
    let closed = false;
    let retryTimer: number | undefined;
    let ws: WebSocket | null = null;

    const connect = () => {
      if (closed) return;
      ws = new WebSocket(ENGINE_WS);

      ws.onopen = () => {
        setConnected(true);
        setErrorMsg("");
        fetchInitialData();
      };

      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === "status") {
            syncStatusToLocalState(msg.data, false);
            fetch(`${ENGINE_HTTP}/api/models/status`)
              .then((r) => (r.ok ? r.json() : null))
              .then((d) => {
                if (d) setModels(d);
              })
              .catch(() => {});
          } else if (msg.type === "level") {
            setLevel(msg.data.level || 0);
            if (msg.data.clipping) {
              setClipping(true);
              if (clipTimerRef.current) clearTimeout(clipTimerRef.current);
              clipTimerRef.current = window.setTimeout(() => setClipping(false), 1400);
            }
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
                mt_engine: same && prev ? prev.mt_engine : undefined,
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
            setHistory((prev) => {
              if (prev.some((x) => x.id === entry.id)) return prev;
              return [entry, ...prev].slice(0, 200);
            });
            if (typeof entry.log_saved_count === "number") {
              setSavedLogCount(entry.log_saved_count);
            }
            if (ttsRef.current && entry.translated_text) {
              speakOfflineTTS(entry.translated_text, entry.target_lang);
            }
          } else if (msg.type === "history") {
            const list: TranscriptEntry[] = msg.data || [];
            setHistory([...list].reverse());
            if (list.length === 0) setPartial(null);
          } else if (msg.type === "audience_count") {
            const cnt = Number(msg.data?.connected_count || 0);
            setAudienceCount(cnt);
            setAudienceInfo((prev) =>
              prev
                ? {
                    ...prev,
                    connected_count: cnt,
                    active_languages: msg.data?.active_languages || [],
                  }
                : prev
            );
          } else if (msg.type === "model_progress") {
            const prog: ModelState = msg.data;
            if (prog.status === "downloading") {
              activeDownloadsRef.current.add(prog.model_id);
            } else if (prog.status === "ready") {
              if (activeDownloadsRef.current.has(prog.model_id)) {
                activeDownloadsRef.current.delete(prog.model_id);
                const label = MODEL_DISPLAY_NAMES[prog.model_id] || prog.model_id;
                setModelSuccessNotice(`Đã tải xong "${label}" thành công — sẵn sàng sử dụng!`);
                if (modelSuccessTimerRef.current) clearTimeout(modelSuccessTimerRef.current);
                modelSuccessTimerRef.current = window.setTimeout(() => {
                  setModelSuccessNotice("");
                }, 9000);
                setJustDownloadedIds((prev) => ({ ...prev, [prog.model_id]: true }));
                const oldT = justDownloadedTimersRef.current[prog.model_id];
                if (oldT) clearTimeout(oldT);
                justDownloadedTimersRef.current[prog.model_id] = window.setTimeout(() => {
                  setJustDownloadedIds((prev) => {
                    if (!prev[prog.model_id]) return prev;
                    const next = { ...prev };
                    delete next[prog.model_id];
                    return next;
                  });
                  delete justDownloadedTimersRef.current[prog.model_id];
                }, 9000);
              }
            } else if (prog.status === "error") {
              activeDownloadsRef.current.delete(prog.model_id);
            }
            setModels((prev) => applyModelStateUpdate(prev, prog));
          } else if (msg.type === "file_progress") {
            const fp: FileTranslateProgress = msg.data;
            lastFileProgKeyRef.current = makeFileProgKey(fp);
            fileProgUpdatedAtRef.current = Date.now();
            setFileProgress(fp);
            if (fp.active) {
              if (!fileStartTsRef.current) {
                fileStartTsRef.current = Date.now() - Math.round((fp.elapsed_s || 0) * 1000);
              }
              setFileBusy(true);
            } else {
              fileStartTsRef.current = 0;
              setFileBusy(false);
              setFileCancelling(false);
            }
          } else if (msg.type === "cloud_notice") {
            setCloudNotice(String(msg.data?.message || ""));
          } else if (msg.type === "error") {
            setErrorMsg(msg.data?.message || "Đã xảy ra lỗi không xác định");
          }
        } catch {
          // ignore
        }
      };

      ws.onclose = () => {
        setConnected(false);
        if (!closed) {
          retryTimer = window.setTimeout(connect, 1200);
        }
      };
    };

    connect();
    return () => {
      closed = true;
      if (retryTimer) clearTimeout(retryTimer);
      ws?.close();
    };
  }, [fetchInitialData, syncStatusToLocalState]);

  const handleRefreshDevices = useCallback(async () => {
    setRefreshingDevices(true);
    try {
      const res = await fetch(`${ENGINE_HTTP}/api/devices`);
      if (res.ok) {
        const d = await res.json();
        const inList: AudioDevice[] = d.devices || [];
        const lbList: AudioDevice[] = d.loopback_devices || [];
        setDevices(inList);
        setLoopbackDevices(lbList);
        const def = inList.find((x: AudioDevice) => x.is_default);
        if (def) setSelectedDevice((prev) => (prev === null || !inList.some((i) => i.id === prev) ? def.id : prev));
        const defLb = lbList.find((x: AudioDevice) => x.is_default);
        if (defLb) setSelectedLoopback((prev) => (prev === null || !lbList.some((i) => i.id === prev) ? defLb.id : prev));
      }
    } catch {
      // ignore
    } finally {
      window.setTimeout(() => setRefreshingDevices(false), 260);
    }
  }, []);

  const startListening = useCallback(async () => {
    setErrorMsg("");
    try {
      const res = await fetch(`${ENGINE_HTTP}/api/session/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          device_id: selectedDevice,
          loopback_device_id: selectedLoopback,
          audio_source: audioSource,
          input_channel: inputChannel,
          input_gain_db: inputGainDb,
          auto_gain: autoGain,
          source_lang: sourceLang,
          target_lang: targetLang,
          asr_engine: asrEngine,
          mt_mode: mtMode,
          paragraph_mode: paragraphMode,
          translate_partials: translatePartials,
          pass2_refine: pass2Refine,
          eco_mode: ecoMode,
          beam_size: beamSize,
          llm_url: llmUrl,
          llm_model: llmModel,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Không thể bắt đầu thu âm" }));
        setErrorMsg(err.detail || "Không thể bắt đầu thu âm");
      }
    } catch (e) {
      setErrorMsg(`Không kết nối được engine local: ${String(e)}`);
    }
  }, [
    selectedDevice,
    selectedLoopback,
    audioSource,
    inputChannel,
    inputGainDb,
    autoGain,
    sourceLang,
    targetLang,
    asrEngine,
    mtMode,
    paragraphMode,
    translatePartials,
    pass2Refine,
    ecoMode,
    beamSize,
    llmUrl,
    llmModel,
  ]);

  const stopListening = useCallback(async () => {
    try {
      await fetch(`${ENGINE_HTTP}/api/session/stop`, { method: "POST" });
    } catch {
      // ignore
    }
  }, []);

  const toggleListening = useCallback(() => {
    if (listeningRef.current) {
      stopListening();
    } else {
      startListening();
    }
  }, [startListening, stopListening]);
  toggleListeningRef.current = toggleListening;

  // Listen for Tauri global hotkey & overlay state sync (mounted once with cancellation guard)
  useEffect(() => {
    if (!isTauriEnv()) return;
    let cancelled = false;
    let unlistenToggle: (() => void) | undefined;
    let unlistenOverlay: (() => void) | undefined;
    import("@tauri-apps/api/event")
      .then(({ listen }) => {
        listen("toggle-listening", () => {
          toggleListeningRef.current();
        })
          .then((fn) => {
            if (cancelled) fn();
            else unlistenToggle = fn;
          })
          .catch(() => {});
        listen<boolean>("overlay-visibility", (ev) => {
          setOverlayOpen(Boolean(ev.payload));
        })
          .then((fn) => {
            if (cancelled) fn();
            else unlistenOverlay = fn;
          })
          .catch(() => {});
      })
      .catch(() => {});
    return () => {
      cancelled = true;
      unlistenToggle?.();
      unlistenOverlay?.();
    };
  }, []);

  // Keyboard shortcuts: Escape closes open modals (when not busy) + browser Cmd/Ctrl+Shift+Space fallback
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (glossaryOpen) setGlossaryOpen(false);
        else if (qrOpen) setQrOpen(false);
        else if (summaryOpen) setSummaryOpen(false);
        else if (cloudOpen) setCloudOpen(false);
        else if (fileModalOpen && !fileBusy) setFileModalOpen(false);
        else if (updateModalOpen && !updateInstalling) setUpdateModalOpen(false);
        return;
      }
      if (
        !isTauriEnv() &&
        connected &&
        e.shiftKey &&
        e.code === "Space" &&
        (isMacOs ? e.metaKey && !e.ctrlKey : e.ctrlKey && !e.metaKey)
      ) {
        e.preventDefault();
        toggleListeningRef.current();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [
    glossaryOpen,
    qrOpen,
    summaryOpen,
    cloudOpen,
    fileModalOpen,
    fileBusy,
    updateModalOpen,
    updateInstalling,
    connected,
    isMacOs,
  ]);

  useEffect(() => {
    return () => {
      if (copyTimerRef.current) clearTimeout(copyTimerRef.current);
      if (llmDebounceRef.current) clearTimeout(llmDebounceRef.current);
      if (modelSuccessTimerRef.current) clearTimeout(modelSuccessTimerRef.current);
      for (const tid of Object.values(justDownloadedTimersRef.current)) {
        clearTimeout(tid);
      }
      justDownloadedTimersRef.current = {};
    };
  }, []);

  const hotUpdateConfig = useCallback(async (patch: Record<string, unknown>) => {
    setErrorMsg("");
    try {
      const res = await fetch(`${ENGINE_HTTP}/api/session/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Không thể cập nhật cấu hình" }));
        setErrorMsg(err.detail || "Không thể cập nhật cấu hình");
      }
    } catch {
      // ignore
    }
  }, []);

  const scheduleLlmConfigUpdate = useCallback(
    (patch: Record<string, unknown>) => {
      if (llmDebounceRef.current) clearTimeout(llmDebounceRef.current);
      llmDebounceRef.current = window.setTimeout(() => {
        hotUpdateConfig(patch);
      }, 400);
    },
    [hotUpdateConfig]
  );

  const handleSelectSourceLang = (code: string) => {
    setSourceLang(code);
    // Auto-flip target to Vietnamese if user switches source away from Vietnamese and target was identical
    if (code === targetLang) {
      const nextTgt = code === "vie_Latn" ? "eng_Latn" : "vie_Latn";
      setTargetLang(nextTgt);
      hotUpdateConfig({ source_lang: code, target_lang: nextTgt });
    } else {
      hotUpdateConfig({ source_lang: code });
    }
  };

  const handleSelectTargetLang = (code: string) => {
    if (code === sourceLang) {
      const canSwap = sourceLanguages.some((s) => s.code === targetLang && s.code !== code);
      const nextSrc = canSwap ? targetLang : code === "vie_Latn" ? "eng_Latn" : "vie_Latn";
      setSourceLang(nextSrc);
      setTargetLang(code);
      hotUpdateConfig({ source_lang: nextSrc, target_lang: code });
      return;
    }
    setTargetLang(code);
    hotUpdateConfig({ target_lang: code });
  };

  const handleSwapLanguages = () => {
    const canSwapTargetAsSource = sourceLanguages.some((s) => s.code === targetLang);
    if (!canSwapTargetAsSource) {
      setErrorMsg("Ngôn ngữ đích hiện tại chưa hỗ trợ làm ngôn ngữ nói đầu vào.");
      return;
    }
    const nextSrc = targetLang;
    const nextTgt = sourceLang;
    setSourceLang(nextSrc);
    setTargetLang(nextTgt);
    hotUpdateConfig({ source_lang: nextSrc, target_lang: nextTgt });
  };

  const handleAudioSourceChange = (src: AudioSourceId) => {
    setAudioSource(src);
    hotUpdateConfig({ audio_source: src });
  };

  const handleOpenGlossary = async () => {
    setGlossaryOpen(true);
    try {
      const res = await fetch(`${ENGINE_HTTP}/api/glossary`);
      if (res.ok) {
        setGlossary(await res.json());
      }
    } catch (e) {
      setErrorMsg(`Không thể tải từ điển họp: ${String(e)}`);
    }
  };

  const updateGlossaryRemote = async (patch: Partial<Pick<GlossaryData, "enabled" | "asr_fixes" | "term_locks">>) => {
    try {
      const res = await fetch(`${ENGINE_HTTP}/api/glossary`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      });
      if (res.ok) {
        setGlossary(await res.json());
      }
    } catch (e) {
      setErrorMsg(`Không thể lưu từ điển họp: ${String(e)}`);
    }
  };

  const handleApplyPreset = async (presetName: string) => {
    try {
      const res = await fetch(`${ENGINE_HTTP}/api/glossary/preset/${presetName}`, { method: "POST" });
      if (res.ok) {
        setGlossary(await res.json());
      }
    } catch (e) {
      setErrorMsg(`Không thể nạp gói từ điển: ${String(e)}`);
    }
  };

  const handleAddAsrFix = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!glossary || !newAsrWrong.trim() || !newAsrCorrect.trim()) return;
    const next = [...glossary.asr_fixes, { wrong: newAsrWrong.trim(), correct: newAsrCorrect.trim() }];
    setNewAsrWrong("");
    setNewAsrCorrect("");
    await updateGlossaryRemote({ asr_fixes: next });
  };

  const handleRemoveAsrFix = async (idx: number) => {
    if (!glossary) return;
    const next = glossary.asr_fixes.filter((_, i) => i !== idx);
    await updateGlossaryRemote({ asr_fixes: next });
  };

  const handleAddTermLock = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!glossary || !newLockSource.trim()) return;
    const targetVal = newLockTarget.trim() || newLockSource.trim();
    const next = [...glossary.term_locks, { source: newLockSource.trim(), target: targetVal }];
    setNewLockSource("");
    setNewLockTarget("");
    await updateGlossaryRemote({ term_locks: next });
  };

  const handleRemoveTermLock = async (idx: number) => {
    if (!glossary) return;
    const next = glossary.term_locks.filter((_, i) => i !== idx);
    await updateGlossaryRemote({ term_locks: next });
  };

  const handleOpenLogsFolder = async () => {
    try {
      await fetch(`${ENGINE_HTTP}/api/logs/open`, { method: "POST" });
    } catch (e) {
      setErrorMsg(`Không thể mở thư mục biên bản: ${String(e)}`);
    }
  };

  const handleToggleOverlay = async () => {
    if (isTauriEnv()) {
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        const visible = await invoke<boolean>("toggle_overlay");
        setOverlayOpen(visible);
        if (!visible && clickThrough) {
          setClickThrough(false);
          await invoke("set_overlay_clickthrough", { ignore: false }).catch(() => {});
        }
      } catch (e) {
        setErrorMsg(`Không thể mở Overlay: ${String(e)}`);
      }
    } else {
      const existing = overlayPopupRef.current;
      if (existing && !existing.closed) {
        existing.close();
        overlayPopupRef.current = null;
        setOverlayOpen(false);
        return;
      }
      const win = window.open(
        `${window.location.origin}/#overlay`,
        "voicerecore-overlay",
        "width=1120,height=620,menubar=no,toolbar=no,location=no,status=no"
      );
      if (!win) {
        setErrorMsg("Trình duyệt đã chặn cửa sổ bật lên (popup). Hãy cho phép popup để mở Phụ đề nổi.");
        setOverlayOpen(false);
        return;
      }
      overlayPopupRef.current = win;
      setOverlayOpen(true);
      try {
        win.addEventListener("beforeunload", () => {
          overlayPopupRef.current = null;
          setOverlayOpen(false);
        });
      } catch {
        // ignore
      }
    }
  };

  const handleToggleClickThrough = async () => {
    const next = !clickThrough;
    setClickThrough(next);
    if (isTauriEnv()) {
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        await invoke("set_overlay_clickthrough", { ignore: next });
      } catch (e) {
        setClickThrough(!next);
        setErrorMsg(`Không thể bật chế độ bấm xuyên: ${String(e)}`);
      }
    }
  };

  const handleOpenQrModal = async () => {
    setQrOpen(true);
    try {
      const res = await fetch(`${ENGINE_HTTP}/api/audience/info`);
      if (res.ok) {
        const data: AudienceInfo = await res.json();
        setAudienceInfo(data);
        setAudienceCount(data.connected_count);
      }
    } catch (e) {
      setErrorMsg(`Không thể tạo mã QR phòng họp: ${String(e)}`);
    }
  };

  useEffect(() => {
    if (!fileBusy) return;
    const tick = window.setInterval(() => {
      setFileTickNow(Date.now());
    }, 1000);
    const poll = window.setInterval(async () => {
      try {
        const r = await fetch(`${ENGINE_HTTP}/api/files/progress`);
        if (r.ok) {
          const p: FileTranslateProgress = await r.json();
          if (p && p.active) {
            const key = makeFileProgKey(p);
            if (key !== lastFileProgKeyRef.current) {
              lastFileProgKeyRef.current = key;
              fileProgUpdatedAtRef.current = Date.now();
              setFileProgress(p);
            }
          }
        }
      } catch {
        // ignore fallback poll error
      }
    }, 2000);
    return () => {
      clearInterval(tick);
      clearInterval(poll);
    };
  }, [fileBusy]);

  const handleUploadMediaFile = async (file: File) => {
    const jobId = `job_${Date.now()}`;
    const now = Date.now();
    fileStartTsRef.current = now;
    fileProgUpdatedAtRef.current = now;
    setFileTickNow(now);
    setFileBusy(true);
    setFileCancelling(false);
    setFileResult(null);
    setErrorMsg("");
    const initProg: FileTranslateProgress = {
      job_id: jobId,
      filename: file.name,
      active: true,
      stage: "uploading",
      percent: 2,
      message: "Đang tải file lên engine nội bộ...",
      current_chunk: 0,
      total_chunks: 0,
      processed_audio_s: 0,
      duration_s: 0,
      elapsed_s: 0,
      eta_s: null,
      eta_estimated: true,
      speed_x: 0,
    };
    lastFileProgKeyRef.current = makeFileProgKey(initProg);
    setFileProgress(initProg);
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("source_lang", sourceLang);
      fd.append("target_lang", targetLang);
      fd.append("mt_mode", mtMode);
      fd.append("job_id", jobId);
      const res = await fetch(`${ENGINE_HTTP}/api/files/translate`, {
        method: "POST",
        body: fd,
      });
      const data = await res.json().catch(() => ({ detail: `Lỗi máy chủ (HTTP ${res.status})` }));
      if (!res.ok) {
        setErrorMsg(data.detail || "Không thể dịch file âm thanh/video");
      } else if (data.cancelled) {
        setFileProgress(null);
      } else {
        setFileResult(data);
      }
    } catch (e) {
      setErrorMsg(`Lỗi dịch file: ${String(e)}`);
    } finally {
      setFileBusy(false);
      setFileCancelling(false);
    }
  };

  const handleCancelFileTranslate = async () => {
    setFileCancelling(true);
    try {
      await fetch(`${ENGINE_HTTP}/api/files/cancel`, { method: "POST" });
    } catch {
      // ignore
    }
  };

  const handleDownloadBlob = (filename: string, content: string) => {
    const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleSummarizeMeeting = async () => {
    setSummaryOpen(true);
    setSummaryBusy(true);
    setErrorMsg("");
    try {
      const res = await fetch(`${ENGINE_HTTP}/api/session/summarize`, { method: "POST" });
      if (res.ok) {
        setSummaryData(await res.json());
      } else {
        setErrorMsg("Không thể tóm tắt cuộc họp");
      }
    } catch (e) {
      setErrorMsg(`Lỗi tóm tắt cuộc họp: ${String(e)}`);
    } finally {
      setSummaryBusy(false);
    }
  };

  const syncCloudFormForTab = (data: CloudApiStatus, pid: CloudProviderId) => {
    const p = data.providers[pid];
    setCloudTab(pid);
    setCloudKeyInput("");
    setCloudRegion(p?.region || "southeastasia");
    setCloudEndpoint(p?.endpoint || "https://api.cognitive.microsofttranslator.com");
    setCloudModel(p?.model || "gemini-2.5-flash");
    setCloudLimit(p?.monthly_limit || (pid === "azure" ? 1950000 : pid === "google" ? 480000 : 5000000));
    setCloudHardCap(Boolean(data.hard_cap_enabled));
    setCloudDraftOffline(Boolean(data.draft_offline));
    setCloudTestResult(null);
  };

  const handleSwitchCloudMode = (useCloud: boolean) => {
    if (useCloud) {
      if (!status?.cloud_api?.active_ready) {
        handleOpenCloudModal();
        return;
      }
      setMtMode("cloud-api");
      hotUpdateConfig({ mt_mode: "cloud-api" });
    } else {
      const targetOffline = lastOfflineMtMode === "cloud-api" ? "auto-offline" : lastOfflineMtMode;
      setMtMode(targetOffline);
      hotUpdateConfig({ mt_mode: targetOffline });
    }
  };

  const handleOpenCloudModal = async (preferredTab?: CloudProviderId) => {
    setCloudOpen(true);
    setCloudTestResult(null);
    try {
      const res = await fetch(`${ENGINE_HTTP}/api/cloud/config`);
      if (res.ok) {
        const data: CloudApiStatus = await res.json();
        setCloudStatus(data);
        syncCloudFormForTab(data, preferredTab || data.provider || "azure");
      }
    } catch (e) {
      setErrorMsg(`Không thể tải cấu hình Cloud API: ${String(e)}`);
    }
  };

  const handleSaveCloudConfig = async (activateMode = false) => {
    setCloudSaving(true);
    setCloudTestResult(null);
    try {
      const body: Record<string, unknown> = {
        provider: cloudTab,
        target_provider: cloudTab,
        monthly_limit: Math.max(1000, Number(cloudLimit) || 480000),
        hard_cap_enabled: cloudHardCap,
        draft_offline: cloudDraftOffline,
      };
      if (cloudKeyInput.trim()) {
        body.api_key = cloudKeyInput.trim();
      }
      if (cloudTab === "azure") {
        body.region = cloudRegion.trim();
        body.endpoint = cloudEndpoint.trim();
      }
      if (cloudTab === "gemini") {
        body.model = cloudModel.trim() || "gemini-2.5-flash";
      }
      const res = await fetch(`${ENGINE_HTTP}/api/cloud/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (res.ok) {
        const data: CloudApiStatus = await res.json();
        setCloudStatus(data);
        setCloudKeyInput("");
        if (activateMode) {
          if (!data.providers[cloudTab]?.has_key) {
            setCloudTestResult({
              ok: false,
              message: "Vui lòng nhập API Key hợp lệ trước khi bật chế độ Cloud API.",
            });
            return;
          }
          setMtMode("cloud-api");
          hotUpdateConfig({ mt_mode: "cloud-api" });
          setCloudOpen(false);
        } else {
          setCloudTestResult({ ok: true, message: "Đã lưu cấu hình API cục bộ an toàn trên máy." });
        }
      } else {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
        setCloudTestResult({ ok: false, message: `Không thể lưu cấu hình: ${err.detail || res.status}` });
      }
    } catch (e) {
      setCloudTestResult({ ok: false, message: `Lỗi lưu cấu hình: ${String(e)}` });
    } finally {
      setCloudSaving(false);
    }
  };

  const handleClearCloudKey = async () => {
    setCloudSaving(true);
    try {
      const res = await fetch(`${ENGINE_HTTP}/api/cloud/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_provider: cloudTab, api_key: "" }),
      });
      if (res.ok) {
        const data: CloudApiStatus = await res.json();
        setCloudStatus(data);
        setCloudKeyInput("");
        setCloudTestResult({ ok: true, message: "Đã xóa API Key đã lưu." });
      } else {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
        setCloudTestResult({ ok: false, message: `Không thể xóa API Key: ${err.detail || res.status}` });
      }
    } catch (e) {
      setCloudTestResult({ ok: false, message: `Lỗi xóa API Key: ${String(e)}` });
    } finally {
      setCloudSaving(false);
    }
  };

  const handleTestCloudApi = async () => {
    setCloudTestBusy(true);
    setCloudTestResult(null);
    try {
      const res = await fetch(`${ENGINE_HTTP}/api/cloud/test`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: cloudTab,
          api_key: cloudKeyInput.trim() || undefined,
          region: cloudTab === "azure" ? cloudRegion.trim() : undefined,
          endpoint: cloudTab === "azure" ? cloudEndpoint.trim() : undefined,
          model: cloudTab === "gemini" ? cloudModel.trim() : undefined,
          target_lang: targetLang === "vie_Latn" ? "eng_Latn" : targetLang,
        }),
      });
      const data = await res.json();
      if (data.ok) {
        if (data.cloud_api) setCloudStatus(data.cloud_api);
        setCloudTestResult({
          ok: true,
          message: `Kết nối thành công (${data.engine} • ${data.latency_ms}ms): "${data.translated_text}"`,
        });
      } else {
        setCloudTestResult({
          ok: false,
          message: `Kiểm tra thất bại: ${data.error || "Không xác định"}`,
        });
      }
    } catch (e) {
      setCloudTestResult({ ok: false, message: `Lỗi kết nối: ${String(e)}` });
    } finally {
      setCloudTestBusy(false);
    }
  };

  const handleResetCloudQuota = async () => {
    try {
      const res = await fetch(`${ENGINE_HTTP}/api/cloud/quota/reset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider: cloudTab }),
      });
      if (res.ok) {
        setCloudStatus(await res.json());
      }
    } catch {
      // ignore
    }
  };

  const checkForAppUpdate = useCallback(async (silent = true) => {
    if (!silent) {
      setUpdateChecking(true);
      setUpdateCheckMsg("Đang kiểm tra phiên bản mới...");
    }
    try {
      if (isTauriEnv()) {
        try {
          const { check } = await import("@tauri-apps/plugin-updater");
          const upd = await check();
          if (upd) {
            setUpdateInfo({
              version: upd.version,
              date: upd.date,
              body: upd.body,
              nativeUpdate: upd,
            });
            setUpdateModalOpen(true);
            setUpdateCheckMsg(`Có bản cập nhật mới v${upd.version}!`);
            return;
          }
          if (!silent) {
            setUpdateCheckMsg(`Bạn đang dùng phiên bản mới nhất (v${APP_VERSION}).`);
          }
          return;
        } catch {
          // Graceful fallback to GitHub Releases API (e.g. dev mode or before latest.json is live)
        }
      }

      const res = await fetch(`https://api.github.com/repos/${GITHUB_REPO}/releases/latest`, {
        headers: { Accept: "application/vnd.github+json" },
      });
      if (res.ok) {
        const rel = await res.json();
        const latestVer = String(rel.tag_name || "").replace(/^v/i, "");
        if (latestVer && compareSemver(latestVer, APP_VERSION) > 0) {
          setUpdateInfo({
            version: latestVer,
            date: rel.published_at,
            body: rel.body || "Bản cập nhật mới đã sẵn sàng trên GitHub Releases.",
            downloadUrl: rel.html_url || `https://github.com/${GITHUB_REPO}/releases/latest`,
          });
          setUpdateModalOpen(true);
          setUpdateCheckMsg(`Có bản cập nhật mới v${latestVer}!`);
        } else if (!silent) {
          setUpdateCheckMsg(`Bạn đang dùng phiên bản mới nhất (v${APP_VERSION}).`);
        }
      } else if (!silent) {
        setUpdateCheckMsg(`Bạn đang dùng phiên bản mới nhất (v${APP_VERSION}).`);
      }
    } catch {
      if (!silent) {
        setUpdateCheckMsg("Không thể kết nối GitHub để kiểm tra (đang ở chế độ Offline).");
      }
    } finally {
      if (!silent) setUpdateChecking(false);
    }
  }, []);

  // Silent background auto-update check 3 seconds after app launches
  useEffect(() => {
    const t = window.setTimeout(() => {
      checkForAppUpdate(true);
    }, 3000);
    return () => clearTimeout(t);
  }, [checkForAppUpdate]);

  const handleInstallUpdate = async () => {
    if (!updateInfo) return;
    if (!updateInfo.nativeUpdate) {
      if (updateInfo.downloadUrl) {
        window.open(updateInfo.downloadUrl, "_blank");
      }
      return;
    }
    setUpdateInstalling(true);
    setUpdateProgress(0);
    setUpdateProgressText("Đang dừng engine cục bộ & chuẩn bị tải bản cập nhật...");
    let totalBytes = 0;
    let downloadedBytes = 0;
    try {
      if (isTauriEnv()) {
        try {
          const { invoke } = await import("@tauri-apps/api/core");
          await invoke("stop_local_engine").catch(() => {});
        } catch {
          // ignore
        }
      }
      await updateInfo.nativeUpdate.downloadAndInstall((event) => {
        if (event.event === "Started") {
          totalBytes = event.data?.contentLength || 0;
          setUpdateProgress(3);
          setUpdateProgressText(
            totalBytes > 0
              ? `Đang tải bản cập nhật (${(totalBytes / 1048576).toFixed(1)} MB)...`
              : "Đang tải bản cập nhật..."
          );
        } else if (event.event === "Progress") {
          downloadedBytes += event.data?.chunkLength || 0;
          const pct = totalBytes > 0 ? Math.min(99, Math.round((downloadedBytes / totalBytes) * 100)) : 50;
          setUpdateProgress(pct);
          setUpdateProgressText(
            totalBytes > 0
              ? `Đang tải ${(downloadedBytes / 1048576).toFixed(1)} / ${(totalBytes / 1048576).toFixed(1)} MB (${pct}%)...`
              : `Đã tải ${(downloadedBytes / 1048576).toFixed(1)} MB...`
          );
        } else if (event.event === "Finished") {
          setUpdateProgress(100);
          setUpdateProgressText("Đã tải xong! Đang cài đặt và khởi động lại...");
        }
      });
      const { relaunch } = await import("@tauri-apps/plugin-process");
      await relaunch();
    } catch (e) {
      setUpdateInstalling(false);
      setUpdateProgressText("");
      setErrorMsg(`Lỗi cập nhật tự động: ${String(e)}`);
    }
  };

  const handleDownloadModel = async (
    modelId:
      | "hy-mt2"
      | "nllb-1.3b"
      | "nllb-600m"
      | "zipformer-30m-streaming"
      | "zipformer-en-20m"
      | "moonshine-en"
      | "whisper-turbo"
      | "opus-vi-en"
      | "opus-en-vi"
      | "envit5"
  ) => {
    setErrorMsg("");
    activeDownloadsRef.current.add(modelId);
    setJustDownloadedIds((prev) => {
      if (!prev[modelId]) return prev;
      const next = { ...prev };
      delete next[modelId];
      return next;
    });
    setModels((prev) =>
      applyModelStateUpdate(prev, {
        model_id: modelId,
        status: "downloading",
        progress: 0,
        downloaded_mb: 0,
        total_mb: 0,
        message: "Đang kết nối máy chủ tải...",
      })
    );
    try {
      const res = await fetch(`${ENGINE_HTTP}/api/models/download/${modelId}`, { method: "POST" });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
        const msg = err.detail || `HTTP ${res.status}`;
        activeDownloadsRef.current.delete(modelId);
        setModels((prev) =>
          applyModelStateUpdate(prev, {
            model_id: modelId,
            status: "error",
            progress: 0,
            downloaded_mb: 0,
            total_mb: 0,
            message: `Lỗi tải: ${msg}`,
          })
        );
        setErrorMsg(`Lỗi tải model ${modelId}: ${msg}`);
      }
    } catch (e) {
      activeDownloadsRef.current.delete(modelId);
      setModels((prev) =>
        applyModelStateUpdate(prev, {
          model_id: modelId,
          status: "error",
          progress: 0,
          downloaded_mb: 0,
          total_mb: 0,
          message: `Lỗi kết nối: ${String(e)}`,
        })
      );
      setErrorMsg(`Lỗi tải model: ${String(e)}`);
    }
  };

  const handleDownloadCoreModels = async () => {
    setErrorMsg("");
    setShowAdvanced(true);
    const missingIds: Array<"zipformer-30m-streaming" | "envit5" | "hy-mt2"> = [];
    if (models?.asr_zipformer_streaming?.status !== "ready" && models?.asr_zipformer_streaming?.status !== "downloading") {
      missingIds.push("zipformer-30m-streaming");
    }
    if (models?.mt_envit5?.status !== "ready" && models?.mt_envit5?.status !== "downloading") {
      missingIds.push("envit5");
    }
    if (models?.mt_hymt2?.status !== "ready" && models?.mt_hymt2?.status !== "downloading") {
      missingIds.push("hy-mt2");
    }
    for (const id of missingIds) {
      await handleDownloadModel(id);
    }
  };

  const handleDeleteModel = async (
    modelId: "hy-mt2" | "moonshine-en" | "zipformer-en-20m" | "nllb-600m" | "whisper-turbo"
  ) => {
    if (
      modelId === "hy-mt2" &&
      typeof window !== "undefined" &&
      !window.confirm("Xóa mô hình Hy-MT2 (~1.05 GB) khỏi ổ cứng? App sẽ tự dùng EnViT5 cho Việt ⇄ Anh và bạn có thể tải lại bất cứ lúc nào.")
    ) {
      return;
    }
    setErrorMsg("");
    try {
      const res = await fetch(`${ENGINE_HTTP}/api/models/${modelId}`, { method: "DELETE" });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Không thể xóa model" }));
        setErrorMsg(err.detail || "Không thể xóa model");
      } else {
        fetchInitialData();
      }
    } catch (e) {
      setErrorMsg(`Lỗi xóa model: ${String(e)}`);
    }
  };

  const handleDemo = async () => {
    setErrorMsg("");
    try {
      await fetch(`${ENGINE_HTTP}/api/demo/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_lang: sourceLang, target_lang: targetLang }),
      });
    } catch (e) {
      setErrorMsg(`Lỗi chạy thử: ${String(e)}`);
    }
  };

  const handleQuickTranslate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!quickText.trim()) return;
    setQuickBusy(true);
    setErrorMsg("");
    try {
      const res = await fetch(`${ENGINE_HTTP}/api/translate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: quickText.trim(),
          source_lang: sourceLang,
          target_lang: targetLang,
          mt_mode: mtMode,
          beam_size: beamSize,
          llm_url: llmUrl,
          llm_model: llmModel,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setErrorMsg(data.detail || "Không thể dịch văn bản");
      } else {
        const entry: TranscriptEntry = {
          id: `txt-${Date.now()}`,
          source_text: data.source_text,
          translated_text: data.translated_text,
          source_lang: data.source_lang,
          target_lang: data.target_lang,
          asr_ms: 0,
          mt_ms: data.latency_ms,
          mt_engine: data.engine,
          timestamp: new Date().toLocaleTimeString("vi-VN", { hour12: false }),
        };
        setHistory((prev) => [entry, ...prev]);
        setQuickText("");
        if (ttsEnabled && entry.translated_text) {
          speakOfflineTTS(entry.translated_text, entry.target_lang);
        }
      }
    } catch (err) {
      setErrorMsg(`Lỗi dịch: ${String(err)}`);
    } finally {
      setQuickBusy(false);
    }
  };

  const handleCopy = async (id: string, text: string) => {
    let ok = false;
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
        ok = true;
      }
    } catch {
      ok = false;
    }
    if (!ok) {
      try {
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        ok = document.execCommand("copy");
        document.body.removeChild(ta);
      } catch {
        ok = false;
      }
    }
    if (!ok) {
      setErrorMsg("Không thể sao chép vào bộ nhớ tạm trên trình duyệt này.");
      return;
    }
    setCopiedId(id);
    if (copyTimerRef.current) clearTimeout(copyTimerRef.current);
    copyTimerRef.current = window.setTimeout(() => setCopiedId(null), 1300);
  };

  const handleExport = (format: "txt" | "srt") => {
    if (history.length === 0) return;
    const chronological = [...history].reverse();
    let content = "";
    if (format === "txt") {
      // .TXT groups sentences cleanly by paragraph_id for readable meeting minutes
      const paragraphs = groupEntriesByParagraph(chronological);
      content = paragraphs
        .map(
          (h) =>
            `[${h.timestamp}] ${h.source_lang}: ${h.source_text}\n[${h.target_lang}] ${h.translated_text}\n`
        )
        .join("\n");
    } else {
      // .SRT keeps crisp YouTube-style sentence-by-sentence cues
      content = chronological
        .map((h, idx) => {
          const startSec = idx * 4;
          const endSec = startSec + 3;
          const fmt = (s: number) => {
            const hh = String(Math.floor(s / 3600)).padStart(2, "0");
            const mm = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
            const ss = String(s % 60).padStart(2, "0");
            return `${hh}:${mm}:${ss},000`;
          };
          return `${idx + 1}\n${fmt(startSec)} --> ${fmt(endSec)}\n${h.translated_text}\n${h.source_text}\n`;
        })
        .join("\n");
    }
    const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `polyvoice-pro-${sourceLang}-to-${targetLang}.${format}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const pinnedLanguages = useMemo(() => {
    const byCode = new Map(languages.map((l) => [l.code, l]));
    const picked = PINNED_CODES.map((c) => byCode.get(c)).filter((x): x is LanguageEntry => Boolean(x));
    if (picked.length > 0) return picked;
    return languages.slice(0, 8);
  }, [languages]);

  const langNameMap = useMemo(() => {
    const m = new Map<string, string>();
    for (const l of languages) m.set(l.code, l.name_vi);
    for (const s of sourceLanguages) if (!m.has(s.code)) m.set(s.code, s.name_vi);
    return m;
  }, [languages, sourceLanguages]);

  const coreModelRows = useMemo(
    () =>
      [
        ["Nghe Tiếng Việt (+Hotwords)", models?.asr_zipformer_streaming, "zipformer-30m-streaming", "49 MB"],
        ["Nghe Tiếng Anh (Nháp live)", models?.asr_zipformer_en_20m, "zipformer-en-20m", "130 MB"],
        ["Nghe Tiếng Anh (Chuẩn 0.6%)", models?.asr_moonshine_en, "moonshine-en", "135 MB"],
        ["Dịch Nháp Việt ⇄ Anh (EnViT5)", models?.mt_envit5, "envit5", "266 MB"],
        ["Dịch Chuẩn 38 NN (Hy-MT2)", models?.mt_hymt2, "hy-mt2", "1.05 GB"],
      ] as const,
    [models]
  );

  const optionalModelRows = useMemo(
    () =>
      [
        ["Nghe Đa ngôn ngữ 99 NN (Whisper-Turbo)", models?.asr_whisper_turbo, "whisper-turbo", "540 MB"],
        ["Dịch Tiếng Lào & 202 NN (NLLB-600M)", models?.mt_nllb_600m, "nllb-600m", "604 MB"],
      ] as const,
    [models]
  );

  const readyCoreCount = useMemo(
    () => coreModelRows.filter(([, st]) => st?.status === "ready").length,
    [coreModelRows]
  );

  const filteredLanguages = useMemo(() => {
    const q = langQuery.trim().toLowerCase();
    if (!q) return languages;
    return languages.filter(
      (l) =>
        l.name_vi.toLowerCase().includes(q) ||
        l.name_en.toLowerCase().includes(q) ||
        l.native.toLowerCase().includes(q) ||
        l.code.toLowerCase().includes(q)
    );
  }, [languages, langQuery]);

  const currentSrcObj = useMemo(
    () => sourceLanguages.find((l) => l.code === sourceLang),
    [sourceLanguages, sourceLang]
  );

  const currentTgtObj = useMemo(
    () => languages.find((l) => l.code === targetLang),
    [languages, targetLang]
  );

  const groupedHistory = useMemo(
    () => groupEntriesByParagraph([...history].reverse()).reverse(),
    [history]
  );

  const latestItem = history[0] || null;
  const isDraft = Boolean(partial);
  const partialHasMt = Boolean(
    partial && (partial.translated_text || partial.stable_text || partial.unstable_text)
  );
  // Atomic EN <-> VI pairing: hold BOTH previous source & translation during the ~150ms gap
  // before the next sentence's first live draft translation arrives (zero mismatched pairing & zero flash!)
  const holdPrevAtomic = Boolean(
    isDraft && !partialHasMt && translatePartials && latestItem?.translated_text
  );
  const spotSource = holdPrevAtomic
    ? latestItem?.source_text || ""
    : partial
    ? partial.source_text
    : latestItem?.source_text || "";
  const spotTarget = holdPrevAtomic
    ? ""
    : partial
    ? partial.translated_text
    : latestItem?.translated_text || "";
  const spotPendingDraft = isDraft && !partialHasMt;
  const spotAsrMs = partial ? partial.asr_ms : latestItem?.asr_ms ?? 0;
  const spotMtMs = partial ? partial.mt_ms : latestItem?.mt_ms ?? 0;
  const litBars = Math.round(level * VU_BARS);

  return (
    <div className="app-shell">
      {/* Single-Row Studio Top Bar */}
      <header className="topbar">
        <div className="topbar-left">
          <div className="brand">
            <div className="brand-crest">
              <img src="/school-logo.png" alt="Logo trường" />
            </div>
            <div className="brand-meta">
              <div className="brand-title">
                PolyVoice <span className="brand-pro-tag">PRO</span>
              </div>
              <div className="brand-sub">Phiên dịch Hội nghị • {languages.length || 202} Ngôn ngữ</div>
            </div>
          </div>

          {!connected ? (
            <span className="pill warn">
              <span className="dot pulse" /> Đang kết nối...
            </span>
          ) : status?.listening ? (
            <span
              className="pill live"
              title={`Engine đang chạy: ${status?.active_asr_label || "auto"}`}
            >
              <span className="dot pulse" />
              {audioSource === "mix"
                ? "Gộp hội trường"
                : audioSource === "dual"
                ? "Họp 2 chiều"
                : audioSource === "system"
                ? "Âm thanh máy"
                : "Đang lắng nghe"}
            </span>
          ) : (
            <span
              className="pill ok"
              title={`Sẵn sàng trên ${status?.platform?.os || ""} ${status?.platform?.arch || ""}`}
            >
              <span className="dot" /> Sẵn sàng
            </span>
          )}

          {updateInfo && (
            <button
              className="btn primary sm"
              onClick={() => setUpdateModalOpen(true)}
              title={`Có phiên bản mới PolyVoice Pro v${updateInfo.version} — bấm để cập nhật tự động`}
            >
              <ArrowUpCircle size={13} /> Cập nhật v{updateInfo.version}
            </button>
          )}
        </div>

        <div className="topbar-right">
          <button
            className={`btn ${overlayOpen ? "active" : ""}`}
            onClick={handleToggleOverlay}
            title="Mở cửa sổ phụ đề trong suốt chia đôi 50/50 (kéo sang màn hình thứ 2)"
          >
            <Layers size={13.5} /> Phụ đề nổi
          </button>

          {overlayOpen && isTauriEnv() && (
            <button
              className={`btn ${clickThrough ? "active" : ""}`}
              onClick={handleToggleClickThrough}
              title="Cho phép bấm chuột xuyên qua cửa sổ phụ đề để điều khiển slide PowerPoint bên dưới"
            >
              <MousePointerClick size={13.5} /> {clickThrough ? "Xuyên chuột: Bật" : "Xuyên chuột"}
            </button>
          )}

          <button
            className={`btn ${qrOpen || audienceCount > 0 ? "active" : ""}`}
            onClick={handleOpenQrModal}
            disabled={!connected}
            title="Phát mã QR Wi-Fi nội bộ để khán giả quét xem phụ đề đa ngôn ngữ trên điện thoại + link OBS"
          >
            <QrCode size={13.5} /> QR
            {audienceCount > 0 && <span className="btn-badge">{audienceCount}</span>}
          </button>

          <button
            className={`btn ${fileModalOpen ? "active" : ""}`}
            onClick={() => setFileModalOpen(true)}
            disabled={!connected}
            title="Kéo thả file ghi âm / video (.mp3, .wav, .mp4) dịch & tạo phụ đề .SRT song ngữ offline"
          >
            <FileAudio size={13.5} /> Dịch File
          </button>

          <button
            className={`btn ${glossaryOpen ? "active" : ""}`}
            onClick={handleOpenGlossary}
            disabled={!connected}
            title="Khóa tên riêng, thuật ngữ chuyên ngành và tự đồng bộ Hotwords vào ASR"
          >
            <BookOpen size={13.5} /> Từ điển
            {typeof status?.glossary_count === "number" && (
              <span className="btn-badge">{status.glossary_count}</span>
            )}
          </button>

          <span className="topbar-divider" />

          <button
            className={`btn icon-only ${ttsEnabled ? "active" : ""}`}
            onClick={() => setTtsEnabled((v) => !v)}
            title={ttsEnabled ? "Đang bật tự động đọc bản dịch (bấm để tắt)" : "Bật tự động đọc bản dịch (TTS offline)"}
          >
            {ttsEnabled ? <Volume2 size={14} /> : <VolumeX size={14} />}
          </button>

          <button
            className="btn icon-only"
            onClick={handleDemo}
            disabled={!connected}
            title="Chạy thử một câu mẫu streaming (gửi tới cả màn hình phụ & QR khán giả)"
          >
            <Play size={13.5} />
          </button>

          <button
            className="btn icon-only"
            onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
            title={theme === "dark" ? "Chuyển sang giao diện Sáng (Light Mode)" : "Chuyển sang giao diện Tối (Dark Mode)"}
          >
            {theme === "dark" ? <Sun size={14} /> : <Moon size={14} />}
          </button>
        </div>
      </header>

      {/* Main Split Workspace */}
      <div className="workspace">
        {/* Left Sidebar */}
        <aside className="sidebar">
          {/* Mic / System Audio Start-Stop Card */}
          <div className="card">
            <button
              className={`mic-cta ${status?.listening ? "stop" : "start"}`}
              onClick={toggleListening}
              disabled={!connected}
            >
              {status?.listening ? <MicOff size={17} /> : <Mic size={17} />}
              {status?.listening
                ? "Dừng dịch giọng nói"
                : `Bắt đầu dịch (${shortLangName(sourceLang, currentSrcObj?.name_vi)} → ${shortLangName(
                    targetLang,
                    currentTgtObj?.name_vi
                  )})`}
            </button>

            {/* Sleek Segmented Audio Source Switcher */}
            <div className="segmented" role="group" aria-label="Nguồn âm thanh">
              <button
                className={`seg-btn ${audioSource === "mic" ? "active" : ""}`}
                onClick={() => handleAudioSourceChange("mic")}
                title="Thu giọng nói từ Micro, Amply, Bàn Mixer hoặc USB Soundcard"
              >
                <Mic size={12} /> Micro/Mixer
              </button>
              <button
                className={`seg-btn ${audioSource === "system" ? "active" : ""}`}
                onClick={() => handleAudioSourceChange("system")}
                disabled={!(status?.system_audio_ready ?? status?.mac_helper_ready)}
                title={
                  (status?.system_audio_ready ?? status?.mac_helper_ready)
                    ? "Thu trực tiếp âm thanh máy tính: YouTube / Zoom / Meet / Livestream (macOS ScreenCaptureKit & Windows WASAPI Loopback)"
                    : "Cần macOS ScreenCaptureKit hoặc Windows WASAPI Loopback / Stereo Mix"
                }
              >
                <MonitorSpeaker size={12} /> Âm thanh máy
              </button>
              <button
                className={`seg-btn ${audioSource === "dual" || audioSource === "mix" ? "active" : ""}`}
                onClick={() => {
                  if (audioSource !== "dual" && audioSource !== "mix") {
                    handleAudioSourceChange("dual");
                  }
                }}
                disabled={!(status?.system_audio_ready ?? status?.mac_helper_ready)}
                title="Kết hợp đồng thời Micro/Mixer và Âm thanh máy (Họp 2 chiều A⇄B hoặc Gộp hội trường A→B)"
              >
                <Headphones size={12} /> 2 Nguồn
              </button>
            </div>

            {(audioSource === "dual" || audioSource === "mix") && (
              <div className="sub-segmented" role="group" aria-label="Chế độ kết hợp 2 nguồn">
                <button
                  className={`sub-seg-btn ${audioSource === "dual" ? "active" : ""}`}
                  onClick={() => handleAudioSourceChange("dual")}
                  title="Micro dịch xuôi (A→B), Loa Zoom/Meet dịch ngược (B→A)"
                >
                  Đàm thoại 2 chiều (A⇄B)
                </button>
                <button
                  className={`sub-seg-btn ${audioSource === "mix" ? "active" : ""}`}
                  onClick={() => handleAudioSourceChange("mix")}
                  title="Gộp tiếng Micro/Mixer + Video/Zoom trên máy vào chung 1 luồng dịch xuôi (A→B)"
                >
                  Gộp hội trường (A→B)
                </button>
              </div>
            )}

            <div className="vu-wrap">
              <div className="vu-row" aria-label="Âm lượng đầu vào">
                {Array.from({ length: VU_BARS }).map((_, idx) => {
                  const isLit = idx < litBars;
                  const isPeak = isLit && idx >= VU_BARS - 3;
                  return (
                    <div
                      key={idx}
                      className={`vu-bar ${isPeak ? "peak" : isLit ? "lit" : ""}`}
                    />
                  );
                })}
              </div>
              {clipping && (
                <div className="clip-alert">
                  <AlertTriangle size={11} /> Quá tải tiếng (Clip) — hãy giảm Gain hoặc bật Auto-Gain
                </div>
              )}
            </div>

            {audioSource !== "system" ? (
              <div className="row">
                <select
                  className="select"
                  value={selectedDevice ?? ""}
                  onChange={(e) => {
                    const id = e.target.value ? Number(e.target.value) : null;
                    setSelectedDevice(id);
                    if (status?.listening && id !== null) {
                      hotUpdateConfig({ device_id: id });
                    }
                  }}
                  title="Chọn Micro, Line-In, USB Soundcard hoặc Bàn Mixer"
                >
                  {devices.length === 0 && <option value="">Micro mặc định hệ thống</option>}
                  {devices.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.is_default ? "★ " : ""}
                      {d.name}
                    </option>
                  ))}
                </select>
                <button
                  className="btn sm icon-only"
                  onClick={handleRefreshDevices}
                  title="Làm mới danh sách thiết bị âm thanh cắm nóng"
                >
                  <RefreshCw size={12.5} className={refreshingDevices ? "spin" : ""} />
                </button>
                <button
                  className={`btn sm icon-only ${
                    showMixerPanel || inputGainDb !== 0 || autoGain || inputChannel !== "mix"
                      ? "info-active"
                      : ""
                  }`}
                  onClick={() => setShowMixerPanel((v) => !v)}
                  title="Chỉnh Gain dB, Auto-Gain chống vỡ tiếng & chọn kênh Trái/Phải cho Mixer/Soundcard"
                >
                  <SlidersHorizontal size={12.5} />
                </button>
              </div>
            ) : (
              <div className="row">
                {!isMacOs && loopbackDevices.length > 0 ? (
                  <select
                    className="select"
                    value={selectedLoopback ?? ""}
                    onChange={(e) => {
                      const id = e.target.value ? Number(e.target.value) : null;
                      setSelectedLoopback(id);
                      if (status?.listening && id !== null) {
                        hotUpdateConfig({ loopback_device_id: id });
                      }
                    }}
                    title="Chọn thiết bị Loa / Cổng âm thanh Windows cần bắt tiếng (WASAPI Loopback)"
                  >
                    {loopbackDevices.map((d) => (
                      <option key={d.id} value={d.id}>
                        {d.is_default ? "★ " : ""}
                        {d.name}
                      </option>
                    ))}
                  </select>
                ) : (
                  <div className="sys-audio-chip">
                    <MonitorSpeaker size={12} />
                    <span>{isMacOs ? "ScreenCaptureKit (Toàn máy)" : "WASAPI Loopback mặc định"}</span>
                  </div>
                )}
                <button
                  className="btn sm icon-only"
                  onClick={handleRefreshDevices}
                  title="Làm mới danh sách thiết bị âm thanh"
                >
                  <RefreshCw size={12.5} className={refreshingDevices ? "spin" : ""} />
                </button>
                <button
                  className={`btn sm icon-only ${
                    showMixerPanel || inputGainDb !== 0 || autoGain ? "info-active" : ""
                  }`}
                  onClick={() => setShowMixerPanel((v) => !v)}
                  title="Chỉnh Gain dB & Auto-Gain cho âm thanh hệ thống"
                >
                  <SlidersHorizontal size={12.5} />
                </button>
              </div>
            )}

            {!isMacOs && (audioSource === "dual" || audioSource === "mix") && loopbackDevices.length > 0 && (
              <div className="row">
                <select
                  className="select"
                  value={selectedLoopback ?? ""}
                  onChange={(e) => {
                    const id = e.target.value ? Number(e.target.value) : null;
                    setSelectedLoopback(id);
                    if (status?.listening && id !== null) {
                      hotUpdateConfig({ loopback_device_id: id });
                    }
                  }}
                  title="Chọn loa Windows để bắt âm thanh hệ thống (WASAPI Loopback)"
                >
                  {loopbackDevices.map((d) => (
                    <option key={d.id} value={d.id}>
                      Loa: {d.is_default ? "★ " : ""}
                      {d.name}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {showMixerPanel && (
              <div className="mixer-panel">
                <div className="mixer-row">
                  <span className="mixer-label">Gain</span>
                  <input
                    className="mixer-slider"
                    type="range"
                    min={-12}
                    max={24}
                    step={1}
                    value={inputGainDb}
                    onChange={(e) => {
                      const val = Number(e.target.value);
                      setInputGainDb(val);
                      hotUpdateConfig({ input_gain_db: val });
                    }}
                  />
                  <button
                    className="mixer-db-tag"
                    onClick={() => {
                      setInputGainDb(0);
                      hotUpdateConfig({ input_gain_db: 0 });
                    }}
                    title="Bấm để đặt lại 0 dB"
                  >
                    {inputGainDb > 0 ? `+${inputGainDb}` : inputGainDb} dB
                  </button>
                  <button
                    className={`btn sm ${autoGain ? "info-active" : ""}`}
                    style={{ padding: "3px 7px", fontSize: 10.5 }}
                    onClick={() => {
                      const next = !autoGain;
                      setAutoGain(next);
                      hotUpdateConfig({ auto_gain: next });
                    }}
                    title="Tự cân bằng âm lượng (AGC + chống vỡ tiếng Limiter) cho Amply/Mixer hội trường"
                  >
                    Auto
                  </button>
                </div>

                {audioSource !== "system" && (
                  <div className="mixer-row">
                    <span className="mixer-label">Kênh</span>
                    <div className="sub-segmented" style={{ flex: 1 }}>
                      {(
                        [
                          ["mix", "L+R (Trộn)"],
                          ["left", "Trái (Ch 1)"],
                          ["right", "Phải (Ch 2)"],
                        ] as const
                      ).map(([chId, lbl]) => (
                        <button
                          key={chId}
                          className={`sub-seg-btn ${inputChannel === chId ? "active" : ""}`}
                          onClick={() => {
                            setInputChannel(chId);
                            hotUpdateConfig({ input_channel: chId });
                          }}
                        >
                          {lbl}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            <div className="hint">
              <span>Phím tắt bật/tắt:</span>
              <span>
                <kbd>{isMacOs ? "⌘" : "Ctrl"}</kbd>+<kbd>Shift</kbd>+<kbd>Space</kbd>
              </span>
            </div>
          </div>

          {/* Bidirectional Source <-> Target Language Card */}
          <div className="card">
            <div className="card-title">
              <span className="row">
                <Globe size={12.5} /> Ngôn ngữ Nói ⇄ Dịch
              </span>
              <span style={{ color: "var(--accent-text)", fontWeight: 700, textTransform: "none", letterSpacing: 0 }}>
                → {currentTgtObj?.name_vi || targetLang}
              </span>
            </div>

            <div className="row">
              <select
                className="select"
                style={{ flex: 1 }}
                value={sourceLang}
                onChange={(e) => handleSelectSourceLang(e.target.value)}
                title="Ngôn ngữ người nói / video YouTube"
              >
                {sourceLanguages.map((sl) => (
                  <option key={sl.code} value={sl.code}>
                    Nói: {sl.name_vi}
                  </option>
                ))}
              </select>

              <button
                className="btn sm active icon-only"
                onClick={handleSwapLanguages}
                title="Đảo chiều ngôn ngữ (ví dụ: Việt ⇄ Anh)"
              >
                <ArrowLeftRight size={13} />
              </button>
            </div>

            {/* Clean 4x2 Uniform Grid of Top 8 Languages */}
            <div className="pin-grid">
              {pinnedLanguages.map((lang) => (
                <button
                  key={lang.code}
                  className={`lang-chip ${targetLang === lang.code ? "selected" : ""}`}
                  onClick={() => handleSelectTargetLang(lang.code)}
                  title={`${lang.name_vi} (${lang.native})`}
                >
                  {shortLangName(lang.code, lang.name_vi)}
                </button>
              ))}
            </div>

            {/* Expandable 202-Language Search & Picker */}
            <div className="row">
              <div style={{ position: "relative", flex: 1 }}>
                <Search
                  size={12.5}
                  style={{ position: "absolute", left: 9, top: "50%", transform: "translateY(-50%)", color: "var(--text-2)" }}
                />
                <input
                  className="input"
                  style={{ paddingLeft: 27 }}
                  placeholder={`Tìm trong ${languages.length || 202} ngôn ngữ...`}
                  value={langQuery}
                  onFocus={() => setShowAllLangs(true)}
                  onChange={(e) => {
                    setLangQuery(e.target.value);
                    setShowAllLangs(true);
                  }}
                />
              </div>
              <button
                className={`btn sm ${showAllLangs ? "active" : ""}`}
                onClick={() => setShowAllLangs((v) => !v)}
                title={showAllLangs ? "Thu gọn danh sách 202 ngôn ngữ" : "Mở rộng chọn trong 202 ngôn ngữ"}
              >
                {languages.length || 202}
                {showAllLangs ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
              </button>
            </div>

            {(showAllLangs || langQuery.trim().length > 0) && (
              <div className="lang-list">
                {filteredLanguages.map((lang) => (
                  <button
                    key={lang.code}
                    className={`lang-item ${targetLang === lang.code ? "selected" : ""}`}
                    onClick={() => {
                      handleSelectTargetLang(lang.code);
                      setLangQuery("");
                      setShowAllLangs(false);
                    }}
                  >
                    <span>
                      {lang.name_vi}{" "}
                      <span style={{ color: "var(--text-2)", fontSize: 11 }}>({lang.native})</span>
                    </span>
                    <span className="lang-code">{lang.code}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Speed / Engine Mode Card */}
          <div className="card">
            <div className="card-title">
              <span className="row">
                <Zap size={12.5} /> Engine &amp; Tối ưu Họp
              </span>
              <span
                style={{
                  fontSize: 10.5,
                  fontWeight: 700,
                  textTransform: "none",
                  letterSpacing: 0,
                  color: mtMode === "cloud-api" ? "var(--green-text)" : "var(--accent-text)",
                }}
              >
                {mtMode === "cloud-api"
                  ? "● Đang dùng Cloud"
                  : status?.hymt2_loaded && status?.hymt2_backend
                  ? `● Offline • ${status.hymt2_backend.toUpperCase()}`
                  : "● 100% Offline"}
              </span>
            </div>

            {/* 1-Click Offline <-> Cloud Mode Switch + Gear Icon for API Settings */}
            <div className="row" style={{ gap: 6 }}>
              <div className="segmented two-col" style={{ flex: 1 }} role="group" aria-label="Chế độ dịch Offline hoặc Cloud">
                <button
                  className={`seg-btn ${mtMode !== "cloud-api" ? "active" : ""}`}
                  onClick={() => handleSwitchCloudMode(false)}
                  title="Chạy dịch 100% Offline trên máy (0 đồng, không cần Internet)"
                >
                  <Cpu size={12} /> Offline
                </button>
                <button
                  className={`seg-btn ${mtMode === "cloud-api" ? "cloud-active" : ""}`}
                  onClick={() => handleSwitchCloudMode(true)}
                  disabled={!connected}
                  title={
                    status?.cloud_api?.active_ready
                      ? "Bật dịch Cloud API chính thức (Nháp vẫn chạy Offline để tiết kiệm hạn mức)"
                      : "Bấm để cấu hình API Key Google / Microsoft Azure / Gemini"
                  }
                >
                  <Cloud size={12} /> Cloud
                </button>
              </div>
              <button
                className={`btn sm icon-only ${cloudOpen ? "active" : ""}`}
                onClick={() => handleOpenCloudModal()}
                disabled={!connected}
                title="Cấu hình Cloud API Key (Microsoft Azure 2M Free / Google Cloud / Gemini)"
              >
                <Settings size={14} />
              </button>
            </div>

            {mtMode === "cloud-api" ? (() => {
              const ca = status?.cloud_api;
              const activeP = ca ? ca.providers[ca.provider] : null;
              const providerLabel =
                activeP?.id === "azure"
                  ? "Microsoft Azure"
                  : activeP?.id === "google"
                  ? "Google Cloud"
                  : activeP?.model || "Gemini";
              return (
                <div className="cloud-inline-strip" style={{ flexDirection: "column", alignItems: "stretch", gap: 4 }}>
                  <div className="row" style={{ justifyContent: "space-between" }}>
                    <span
                      style={{
                        color: activeP?.has_key && !activeP.cap_reached ? "var(--green-text)" : "var(--amber)",
                        fontWeight: 700,
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {activeP?.has_key
                        ? activeP.cap_reached
                          ? `⚠ Chạm hạn mức ${providerLabel}`
                          : `● ${providerLabel}`
                        : "⚠ Chưa nhập API Key"}
                    </span>
                    {activeP?.has_key && (
                      <span style={{ color: "var(--text-2)", fontFamily: "ui-monospace, monospace", fontSize: 10.5 }}>
                        {activeP.used_chars.toLocaleString("vi-VN")}/{activeP.monthly_limit.toLocaleString("vi-VN")}
                      </span>
                    )}
                  </div>
                  <div style={{ fontSize: 10.5, color: "var(--text-2)" }}>
                    {activeP?.has_key
                      ? "Nháp chạy Offline (0 tốn quota) • Hết hạn mức tự về Offline"
                      : "Bấm biểu tượng bánh răng ⚙ bên trên để dán API Key"}
                  </div>
                </div>
              );
            })() : (
              <label className="field-label">
                <span>Engine Dịch Offline (MT)</span>
                <select
                  className="select"
                  value={mtMode === "auto-offline" || mtMode === "light-offline" || mtMode === "local-llm" ? mtMode : "auto-offline"}
                  onChange={(e) => {
                    const mode = e.target.value as MtModeId;
                    setMtMode(mode);
                    if (mode !== "cloud-api") setLastOfflineMtMode(mode);
                    hotUpdateConfig({ mt_mode: mode });
                  }}
                >
                  <option value="auto-offline">
                    2 Tầng Chuẩn cao (Nháp EnViT5 + Chốt Hy-MT2 • 38 NN)
                  </option>
                  <option value="light-offline">
                    Siêu nhẹ &amp; Nhanh (Chỉ EnViT5 + NLLB-600M • 202 NN)
                  </option>
                  <option value="local-llm">Local LLM nâng cao (Ollama / llama.cpp)</option>
                </select>
              </label>
            )}

            <label className="field-label">
              <span>Nhận dạng Giọng nói (ASR)</span>
              <select
                className="select"
                value={asrEngine}
                onChange={(e) => {
                  const val = e.target.value as AsrEngineId;
                  setAsrEngine(val);
                  hotUpdateConfig({ asr_engine: val });
                }}
              >
                <option value="auto">Tự động tối ưu theo ngôn ngữ (Khuyên dùng)</option>
                <option value="moonshine-en" disabled={!status?.moonshine_en_ready}>
                  Moonshine-v2 Tiếng Anh (Streaming dấu câu chuẩn YouTube)
                </option>
                <option value="whisper-turbo" disabled={!status?.whisper_turbo_ready}>
                  Whisper-Turbo 99 NN (Thái, Nhật, Trung, Hàn, Pháp, Đức • Offline)
                </option>
                {status?.mac_helper_ready && (
                  <option value="apple-native">
                    Apple Neural Engine (macOS Native • ~1% CPU • 60 NN)
                  </option>
                )}
                <option value="zipformer-30m-streaming" disabled={!status?.zipformer_ready}>
                  Zipformer-30M Tiếng Việt (51MB • ~30ms)
                </option>
                <option value="zipformer-en-20m" disabled={!status?.zipformer_en_ready}>
                  Zipformer-20M Tiếng Anh (30MB • ~20ms)
                </option>
                <option value="viet-asr">VietASR Tiếng Việt (78MB • Dấu câu/ITN)</option>
              </select>
            </label>

            <label className="field-label">
              <span>Gom đoạn hiển thị</span>
              <select
                className="select"
                value={paragraphMode}
                onChange={(e) => {
                  const val = e.target.value as ParagraphModeId;
                  setParagraphMode(val);
                  hotUpdateConfig({ paragraph_mode: val });
                }}
              >
                <option value="long">Đoạn dài kiểu YouTube (3–4 câu/đoạn)</option>
                <option value="medium">Trung bình (~2 câu/đoạn)</option>
                <option value="short">Từng câu riêng lẻ (Phụ đề ngắn)</option>
              </select>
            </label>

            <label
              className="toggle-row"
              title="Chạy Pass-2 ở nền sau mỗi câu: VietASR ITN (+bảo vệ từ khóa Glossary) cho tiếng Việt & Moonshine-v2 (0.6% WER) cho tiếng Anh"
            >
              <span>Tinh chỉnh 2-Pass (Việt ITN &amp; Anh)</span>
              <span className="switch">
                <input
                  type="checkbox"
                  checked={pass2Refine}
                  onChange={(e) => {
                    setPass2Refine(e.target.checked);
                    hotUpdateConfig({ pass2_refine: e.target.checked });
                  }}
                />
                <span className="switch-track" />
              </span>
            </label>

            <label
              className="toggle-row"
              title="Giảm tần suất dịch nháp xuống 1 giây/lần giúp máy mát và ổn định suốt buổi hội thảo 2–3 tiếng"
            >
              <span>Họp dài — Eco (Mát CPU)</span>
              <span className="switch">
                <input
                  type="checkbox"
                  checked={ecoMode}
                  onChange={(e) => {
                    setEcoMode(e.target.checked);
                    hotUpdateConfig({ eco_mode: e.target.checked });
                  }}
                />
                <span className="switch-track" />
              </span>
            </label>
          </div>

          {/* Advanced & Models */}
          <div className="card">
            <div
              className="card-title"
              style={{ cursor: "pointer" }}
              onClick={() => setShowAdvanced((v) => !v)}
            >
              <span className="row">
                <Settings2 size={12.5} /> Mô hình Offline
              </span>
              <span className="row" style={{ gap: 4, textTransform: "none", letterSpacing: 0 }}>
                <span
                  style={{
                    color: readyCoreCount === 0 ? "var(--amber)" : "var(--green-text)",
                    fontWeight: 600,
                  }}
                >
                  {readyCoreCount}/{coreModelRows.length} cốt lõi
                </span>
                {showAdvanced ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
              </span>
            </div>

            {showAdvanced && (
              <>
                <label className="toggle-row">
                  <span>Dịch ngay khi đang nói</span>
                  <span className="switch">
                    <input
                      type="checkbox"
                      checked={translatePartials}
                      onChange={(e) => {
                        setTranslatePartials(e.target.checked);
                        hotUpdateConfig({ translate_partials: e.target.checked });
                      }}
                    />
                    <span className="switch-track" />
                  </span>
                </label>

                <label className="field-label">
                  <span>Beam giải mã Offline</span>
                  <select
                    className="select"
                    value={beamSize}
                    onChange={(e) => {
                      const val = Number(e.target.value);
                      setBeamSize(val);
                      hotUpdateConfig({ beam_size: val });
                    }}
                  >
                    <option value={1}>Beam = 1 (Nhanh nhất — Real-time)</option>
                    <option value={2}>Beam = 2 (Cân bằng)</option>
                    <option value={3}>Beam = 3 (Chính xác cao hơn)</option>
                  </select>
                </label>

                {mtMode === "local-llm" && (
                  <>
                    <input
                      className="input"
                      value={llmUrl}
                      onChange={(e) => {
                        const val = e.target.value;
                        setLlmUrl(val);
                        scheduleLlmConfigUpdate({ llm_url: val });
                      }}
                      placeholder="http://127.0.0.1:11434/v1"
                    />
                    <input
                      className="input"
                      value={llmModel}
                      onChange={(e) => {
                        const val = e.target.value;
                        setLlmModel(val);
                        scheduleLlmConfigUpdate({ llm_model: val });
                      }}
                      placeholder="translategemma:4b hoặc hy-mt:1.8b"
                    />
                  </>
                )}

                <div className="hint" style={{ marginTop: 2, fontWeight: 700, color: "var(--text-1)" }}>
                  <span>Gói Cốt lõi (~1.6 GB)</span>
                  <span style={{ color: "var(--text-2)", fontWeight: 500 }}>5 model chuẩn 2 tầng</span>
                </div>

                {coreModelRows.map(([label, st, id, sz]) => {
                  const isDownloading = st?.status === "downloading";
                  const isError = st?.status === "error";
                  const isJustDone = st?.status === "ready" && Boolean(justDownloadedIds[id]);
                  const pct = Math.max(0, Math.min(100, Math.round(st?.progress ?? 0)));
                  return (
                    <div className={`model-row ${isJustDone ? "success" : ""}`} key={id}>
                      <div className="model-top">
                        <span>{label}</span>
                        {st?.status === "ready" ? (
                          <div className="row" style={{ gap: 5 }}>
                            {isJustDone ? (
                              <span className="model-done-badge">
                                <CheckCircle2 size={12} /> Tải thành công
                              </span>
                            ) : (
                              <span style={{ color: "var(--green-text)", fontSize: 11, whiteSpace: "nowrap" }}>
                                ✓ {sz}
                              </span>
                            )}
                            {id === "hy-mt2" && (
                              <button
                                className="btn sm icon-only"
                                style={{ width: 22, height: 22, minWidth: 22, padding: 3 }}
                                onClick={() => handleDeleteModel("hy-mt2")}
                                title="Xóa Hy-MT2 (1.05 GB) nếu dùng máy cấu hình nhẹ — app tự chạy bằng EnViT5"
                              >
                                <Trash2 size={11} />
                              </button>
                            )}
                          </div>
                        ) : isDownloading ? (
                          <button className="btn sm downloading" disabled>
                            <Loader2 size={11.5} className="spin" /> {pct}%
                          </button>
                        ) : (
                          <button
                            className="btn primary sm"
                            onClick={() => handleDownloadModel(id)}
                          >
                            <Download size={11.5} /> {isError ? "Thử lại" : sz}
                          </button>
                        )}
                      </div>
                      {isDownloading && (
                        <>
                          <div className="progress-track">
                            <div className="progress-fill" style={{ width: `${Math.max(3, pct)}%` }} />
                          </div>
                          <div className="model-progress-meta">
                            <span>{st.message || "Đang tải..."}</span>
                            <span>
                              {st.total_mb > 0
                                ? `${st.downloaded_mb.toFixed(1)} / ${st.total_mb.toFixed(0)} MB`
                                : `${pct}%`}
                            </span>
                          </div>
                        </>
                      )}
                      {isError && st?.message && (
                        <div className="model-error-msg" title={st.message}>
                          {st.message}
                        </div>
                      )}
                    </div>
                  );
                })}

                <div className="hint" style={{ marginTop: 4, fontWeight: 700, color: "var(--text-1)" }}>
                  <span>Mở rộng Đa ngôn ngữ &amp; Tiếng Lào (Tùy chọn)</span>
                </div>

                {optionalModelRows.map(([label, st, id, sz]) => {
                  const isDownloading = st?.status === "downloading";
                  const isError = st?.status === "error";
                  const isJustDone = st?.status === "ready" && Boolean(justDownloadedIds[id]);
                  const pct = Math.max(0, Math.min(100, Math.round(st?.progress ?? 0)));
                  return (
                    <div className={`model-row ${isJustDone ? "success" : ""}`} key={id}>
                      <div className="model-top">
                        <span>{label}</span>
                        {st?.status === "ready" ? (
                          <div className="row" style={{ gap: 5 }}>
                            {isJustDone ? (
                              <span className="model-done-badge">
                                <CheckCircle2 size={12} /> Tải thành công
                              </span>
                            ) : (
                              <span style={{ color: "var(--green-text)", fontSize: 11, whiteSpace: "nowrap" }}>
                                ✓ {sz}
                              </span>
                            )}
                            <button
                              className="btn sm icon-only"
                              style={{ width: 22, height: 22, minWidth: 22, padding: 3 }}
                              onClick={() => handleDeleteModel(id)}
                              title={`Xóa ${label} (${sz}) để giải phóng dung lượng ổ cứng`}
                            >
                              <Trash2 size={11} />
                            </button>
                          </div>
                        ) : isDownloading ? (
                          <button className="btn sm downloading" disabled>
                            <Loader2 size={11.5} className="spin" /> {pct}%
                          </button>
                        ) : (
                          <button
                            className="btn primary sm"
                            onClick={() => handleDownloadModel(id)}
                          >
                            <Download size={11.5} /> {isError ? "Thử lại" : sz}
                          </button>
                        )}
                      </div>
                      {isDownloading && (
                        <>
                          <div className="progress-track">
                            <div className="progress-fill" style={{ width: `${Math.max(3, pct)}%` }} />
                          </div>
                          <div className="model-progress-meta">
                            <span>{st.message || "Đang tải..."}</span>
                            <span>
                              {st.total_mb > 0
                                ? `${st.downloaded_mb.toFixed(1)} / ${st.total_mb.toFixed(0)} MB`
                                : `${pct}%`}
                            </span>
                          </div>
                        </>
                      )}
                      {isError && st?.message && (
                        <div className="model-error-msg" title={st.message}>
                          {st.message}
                        </div>
                      )}
                    </div>
                  );
                })}

                {/* App Version & Manual Auto-Update Check */}
                <div className="update-footer-row">
                  <div className="row" style={{ justifyContent: "space-between", width: "100%" }}>
                    <span style={{ fontSize: 11, color: "var(--text-2)", fontWeight: 600 }}>
                      PolyVoice Pro v{APP_VERSION}
                    </span>
                    <button
                      className="btn sm"
                      style={{ padding: "3px 8px", fontSize: 10.5 }}
                      onClick={() => checkForAppUpdate(false)}
                      disabled={updateChecking}
                    >
                      <ArrowUpCircle size={11.5} /> {updateChecking ? "Đang kiểm tra..." : "Kiểm tra cập nhật"}
                    </button>
                  </div>
                  {updateCheckMsg && (
                    <div style={{ fontSize: 10.5, color: updateInfo ? "var(--accent-text)" : "var(--text-2)" }}>
                      {updateCheckMsg}
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        </aside>

        {/* Right Live Stage */}
        <main className="main-stage">
          {errorMsg && (
            <div className="banner-error">
              <span>{errorMsg}</span>
              <button className="btn sm" onClick={() => setErrorMsg("")}>
                Đóng
              </button>
            </div>
          )}

          {mtMode === "cloud-api" && cloudNotice && (
            <div className="banner-warn">
              <span className="row" style={{ gap: 7, minWidth: 0 }}>
                <AlertTriangle size={14} style={{ flexShrink: 0 }} />
                <span>{cloudNotice}</span>
              </span>
              <div className="row" style={{ gap: 6, flexShrink: 0 }}>
                <button className="btn sm" onClick={() => handleOpenCloudModal()}>
                  <Settings size={12} /> Đổi API
                </button>
                <button
                  className="btn sm icon-only"
                  onClick={() => setCloudNotice("")}
                  title="Ẩn thông báo"
                >
                  <X size={12} />
                </button>
              </div>
            </div>
          )}

          {modelSuccessNotice && (
            <div className="banner-success" role="status">
              <span className="row" style={{ gap: 7, minWidth: 0 }}>
                <CheckCircle2 size={15} style={{ flexShrink: 0 }} />
                <span>{modelSuccessNotice}</span>
              </span>
              <button
                className="btn sm icon-only"
                onClick={() => setModelSuccessNotice("")}
                title="Đóng thông báo"
              >
                <X size={12} />
              </button>
            </div>
          )}

          {connected &&
            models !== null &&
            !setupBannerDismissed &&
            mtMode !== "cloud-api" &&
            (models.mt_envit5?.status !== "ready" ||
              models.asr_zipformer_streaming?.status !== "ready") &&
            (() => {
              const coreDownloading =
                models.mt_envit5?.status === "downloading" ||
                models.asr_zipformer_streaming?.status === "downloading";
              const activeDl =
                models.mt_envit5?.status === "downloading"
                  ? models.mt_envit5
                  : models.asr_zipformer_streaming?.status === "downloading"
                  ? models.asr_zipformer_streaming
                  : null;
              const activePct = Math.max(0, Math.min(100, Math.round(activeDl?.progress ?? 0)));
              return (
                <div className="banner-setup">
                  <div className="row" style={{ gap: 8, minWidth: 0, flex: 1 }}>
                    {coreDownloading ? (
                      <Loader2 size={15} className="spin" style={{ flexShrink: 0 }} />
                    ) : (
                      <Download size={15} style={{ flexShrink: 0 }} />
                    )}
                    <div style={{ minWidth: 0, flex: 1 }}>
                      <div style={{ fontWeight: 700, fontSize: 12.5 }}>
                        {coreDownloading
                          ? `Đang tải bộ mô hình lõi Offline (${activePct}%)...`
                          : "Thiết lập lần đầu: Chưa tải bộ mô hình dịch Offline lõi (~315 MB)"}
                      </div>
                      <div style={{ fontSize: 11.5, color: "var(--text-2)", marginTop: 1 }}>
                        {coreDownloading
                          ? activeDl?.message || "Đang tải về thư mục mô hình cục bộ..."
                          : "Tải 1 lần (Zipformer-30M Việt + EnViT5 Việt ⇄ Anh) để dịch 100% không cần Internet."}
                      </div>
                      {coreDownloading && (
                        <div className="progress-track" style={{ marginTop: 6 }}>
                          <div className="progress-fill" style={{ width: `${Math.max(3, activePct)}%` }} />
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="row" style={{ gap: 6, flexShrink: 0 }}>
                    <button
                      className={`btn sm ${coreDownloading ? "downloading" : "primary"}`}
                      disabled={coreDownloading}
                      onClick={handleDownloadCoreModels}
                    >
                      {coreDownloading ? (
                        <>
                          <Loader2 size={12} className="spin" /> {activePct}%
                        </>
                      ) : (
                        <>
                          <Download size={12} /> Tải bộ mô hình lõi
                        </>
                      )}
                    </button>
                    <button
                      className="btn sm icon-only"
                      onClick={() => setSetupBannerDismissed(true)}
                      title="Ẩn gợi ý thiết lập"
                    >
                      <X size={12} />
                    </button>
                  </div>
                </div>
              );
            })()}

          {connected &&
            models !== null &&
            sourceLang !== "vie_Latn" &&
            sourceLang !== "eng_Latn" &&
            models.asr_whisper_turbo?.status !== "ready" &&
            (() => {
              const wt = models.asr_whisper_turbo;
              const isDl = wt?.status === "downloading";
              const pct = Math.max(0, Math.min(100, Math.round(wt?.progress ?? 0)));
              const srcName = currentSrcObj?.name_vi || sourceLang;
              return (
                <div className="banner-setup">
                  <div className="row" style={{ gap: 8, minWidth: 0, flex: 1 }}>
                    {isDl ? (
                      <Loader2 size={15} className="spin" style={{ flexShrink: 0 }} />
                    ) : (
                      <Download size={15} style={{ flexShrink: 0 }} />
                    )}
                    <div style={{ minWidth: 0, flex: 1 }}>
                      <div style={{ fontWeight: 700, fontSize: 12.5 }}>
                        {isDl
                          ? `Đang tải Whisper-Turbo 99 ngôn ngữ (${pct}%)...`
                          : `Nghe ${srcName} offline 100% trên cả Windows & macOS`}
                      </div>
                      <div style={{ fontSize: 11.5, color: "var(--text-2)", marginTop: 1 }}>
                        {isDl
                          ? wt?.message || "Đang tải Whisper Large-v3-Turbo + Silero VAD..."
                          : "Tải 1 lần Whisper-Turbo + Silero VAD (~540 MB) để nhận dạng 99 ngôn ngữ (Thái, Nhật, Trung, Hàn, Pháp, Đức...) kèm dấu câu."}
                      </div>
                      {isDl && (
                        <div className="progress-track" style={{ marginTop: 6 }}>
                          <div className="progress-fill" style={{ width: `${Math.max(3, pct)}%` }} />
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="row" style={{ gap: 6, flexShrink: 0 }}>
                    <button
                      className={`btn sm ${isDl ? "downloading" : "primary"}`}
                      disabled={isDl}
                      onClick={() => handleDownloadModel("whisper-turbo")}
                    >
                      {isDl ? (
                        <>
                          <Loader2 size={12} className="spin" /> {pct}%
                        </>
                      ) : (
                        <>
                          <Download size={12} /> Tải Whisper-Turbo (540 MB)
                        </>
                      )}
                    </button>
                  </div>
                </div>
              );
            })()}

          {/* Live Spotlight */}
          <section className={`spotlight ${status?.listening ? "live" : ""}`}>
            <div className="spotlight-head">
              <span className="row" style={{ gap: 7 }}>
                {isDraft && <span className="dot pulse" style={{ color: "var(--accent)" }} />}
                <span>
                  {isDraft
                    ? `Đang nghe (${shortLangName(sourceLang, currentSrcObj?.name_vi)}) & dịch trực tiếp...`
                    : `${currentSrcObj?.name_vi || sourceLang} → ${currentTgtObj?.name_vi || targetLang}`}
                </span>
              </span>
              <div className="metric-badges">
                {spotAsrMs > 0 && <span className="metric">ASR {Math.round(spotAsrMs)}ms</span>}
                {spotMtMs > 0 && <span className="metric">MT {Math.round(spotMtMs)}ms</span>}
                {spotTarget && (
                  <>
                    <button
                      className="btn sm icon-only"
                      onClick={() => speakOfflineTTS(spotTarget, targetLang)}
                      title="Đọc bản dịch"
                    >
                      <Volume2 size={12} />
                    </button>
                    <button
                      className="btn sm icon-only"
                      onClick={() => handleCopy("spot", spotTarget)}
                      title="Sao chép bản dịch"
                    >
                      {copiedId === "spot" ? <Check size={12} /> : <Copy size={12} />}
                    </button>
                  </>
                )}
              </div>
            </div>

            <div className={`spot-source ${isDraft ? "draft" : ""}`}>
              {spotSource ? (
                holdPrevAtomic ? (
                  <span className="spot-prev-hold">
                    {spotSource}
                    <span className="live-caret src" aria-hidden="true" />
                  </span>
                ) : (
                  <>
                    {spotSource}
                    {isDraft && <span className="live-caret src" aria-hidden="true" />}
                  </>
                )
              ) : (
                <span className="spot-empty">
                  Chọn <b>Ngôn ngữ nói</b> rồi bấm <b>Bắt đầu dịch</b> — hỗ trợ thu trực tiếp <b>Micro</b>, <b>YouTube</b> hoặc họp <b>2 Chiều</b>.
                </span>
              )}
            </div>

            <div className="spot-divider" />

            <div className={`spot-target ${isDraft ? "draft" : ""}`}>
              {spotTarget ? (
                isDraft && (partial?.stable_text || partial?.unstable_text) ? (
                  <>
                    {partial.stable_text && (
                      <span className="mt-stable">{partial.stable_text}</span>
                    )}
                    {partial.stable_text && partial.unstable_text ? " " : null}
                    {partial.unstable_text && (
                      <span className="mt-unstable">{partial.unstable_text}</span>
                    )}
                    <span className="live-caret" aria-hidden="true" />
                  </>
                ) : (
                  <>
                    {spotTarget}
                    {isDraft && <span className="live-caret" aria-hidden="true" />}
                  </>
                )
              ) : holdPrevAtomic && latestItem?.translated_text ? (
                <span className="spot-prev-hold">
                  {latestItem.translated_text}
                  <span className="live-caret" aria-hidden="true" />
                </span>
              ) : spotPendingDraft ? (
                <span className="spot-translating subtle" aria-label="Đang dịch trực tiếp">
                  <span className="live-dots" aria-hidden="true">
                    <span />
                    <span />
                    <span />
                  </span>
                </span>
              ) : (
                <span className="spot-empty">
                  Bản dịch thời gian thực sẽ xuất hiện tại đây...
                </span>
              )}
            </div>
          </section>

          {/* Quick Text Translation Test Bar */}
          <form className="quick-bar" onSubmit={handleQuickTranslate}>
            <input
              className="input"
              placeholder={`Gõ nhanh câu (${currentSrcObj?.name_vi || sourceLang}) để dịch thử sang ${
                currentTgtObj?.name_vi || targetLang
              }...`}
              value={quickText}
              onChange={(e) => setQuickText(e.target.value)}
            />
            <button
              className="btn primary"
              type="submit"
              disabled={!connected || quickBusy || !quickText.trim()}
            >
              <Send size={13.5} /> Dịch thử
            </button>
          </form>

          {/* History Feed */}
          <section className="history-wrap">
            <div className="history-head">
              <div className="history-head-left">
                <span className="history-title">Lịch sử hội thoại</span>
                <span className="history-count">
                  {groupedHistory.length} đoạn • {history.length} câu
                </span>
                {savedLogCount > 0 && (
                  <span className="autosave-badge" title={status?.logs?.txt_file || "Đã tự lưu xuống ổ đĩa"}>
                    ● Tự lưu {savedLogCount}
                  </span>
                )}
              </div>
              <div className="history-head-actions">
                <button
                  className="btn sm active"
                  onClick={handleSummarizeMeeting}
                  disabled={!connected || history.length === 0}
                  title="Tóm tắt ý chính & danh sách việc cần làm song ngữ (100% offline)"
                >
                  <Sparkles size={12} /> Tóm tắt
                </button>
                <button
                  className="btn sm"
                  onClick={handleOpenLogsFolder}
                  disabled={!connected}
                  title={`Mở thư mục biên bản tự động lưu (${status?.logs?.logs_dir || "~/Documents/VOICERECORE_Logs"})`}
                >
                  <FolderOpen size={12} /> Biên bản
                </button>
                <button
                  className="btn sm"
                  onClick={() => handleExport("txt")}
                  disabled={history.length === 0}
                  title="Xuất biên bản .TXT theo đoạn văn"
                >
                  <FileText size={12} /> .TXT
                </button>
                <button
                  className="btn sm"
                  onClick={() => handleExport("srt")}
                  disabled={history.length === 0}
                  title="Xuất phụ đề .SRT theo từng câu"
                >
                  <FileText size={12} /> .SRT
                </button>
                <button
                  className="btn sm icon-only"
                  onClick={() => {
                    setHistory([]);
                    fetch(`${ENGINE_HTTP}/api/history/clear`, { method: "POST" }).catch(() => {});
                  }}
                  disabled={history.length === 0}
                  title="Xóa lịch sử trên màn hình"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            </div>

            <div className="history-list">
              {groupedHistory.length === 0 ? (
                <div className="history-item" style={{ color: "var(--text-2)", fontSize: 13, display: "block", lineHeight: 1.5 }}>
                  Chưa có câu dịch nào. Hãy bấm <b>Bắt đầu dịch</b> (hoặc đảo chiều <b>⇄</b> để dịch YouTube Tiếng Anh → Tiếng Việt).
                </div>
              ) : (
                groupedHistory.map((item) => {
                  const srcShort = shortLangName(item.source_lang, langNameMap.get(item.source_lang));
                  const tgtShort = shortLangName(item.target_lang, langNameMap.get(item.target_lang));
                  const engShort = shortEngineName(item.mt_engine);
                  const techTooltip = [
                    `${item.source_lang} → ${item.target_lang}`,
                    item.mt_engine,
                    item.asr_ms > 0 ? `ASR ${item.asr_ms}ms` : "",
                    item.pass2_ms && item.pass2_ms > 0 ? `Pass-2 ${item.pass2_ms}ms` : "",
                    `MT ${item.mt_ms}ms`,
                  ]
                    .filter(Boolean)
                    .join(" • ");

                  return (
                    <div
                      className={`history-item ${
                        item.speaker === "self"
                          ? "spk-self"
                          : item.speaker === "remote"
                          ? "spk-remote"
                          : ""
                      }`}
                      key={item.id}
                    >
                      <div className="history-meta">
                        <span className="row" style={{ gap: 6, minWidth: 0 }} title={techTooltip}>
                          {item.speaker === "self" && <span className="spk-badge self">Bạn</span>}
                          {item.speaker === "remote" && <span className="spk-badge remote">Đối tác</span>}
                          <span>
                            {item.timestamp}
                            {(item.sentence_count ?? 1) > 1 ? ` • ${item.sentence_count} câu` : ""} • {srcShort} →{" "}
                            {tgtShort}
                            {engShort ? ` • ${engShort}` : ""} • {Math.round(item.mt_ms)}ms
                          </span>
                        </span>
                        <div className="row" style={{ gap: 4 }}>
                          <button
                            className="btn sm icon-only"
                            onClick={() => speakOfflineTTS(item.translated_text, item.target_lang)}
                            title="Đọc bản dịch"
                          >
                            <Volume2 size={11.5} />
                          </button>
                          <button
                            className="btn sm icon-only"
                            onClick={() => handleCopy(item.id, item.translated_text)}
                            title="Sao chép bản dịch"
                          >
                            {copiedId === item.id ? <Check size={11.5} /> : <Copy size={11.5} />}
                          </button>
                        </div>
                      </div>
                      <div className="history-vi">{item.source_text}</div>
                      <div className="history-tr">{item.translated_text || "—"}</div>
                    </div>
                  );
                })
              )}
            </div>
          </section>
        </main>
      </div>

      {/* Meeting Glossary Modal */}
      {glossaryOpen && (
        <div className="modal-backdrop" onClick={() => setGlossaryOpen(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <div className="row" style={{ gap: 8 }}>
                <BookOpen size={17} color="var(--accent)" />
                <div>
                  <div className="modal-title">Từ điển Họp &amp; Khóa Thuật Ngữ (100% Offline)</div>
                  <div className="modal-sub">
                    Sửa lỗi nghe nhầm ASR và bảo vệ tên riêng / thương hiệu không bị dịch sai nghĩa
                  </div>
                </div>
              </div>
              <div className="row">
                {glossary && (
                  <label className="row" style={{ fontSize: 12, fontWeight: 600, cursor: "pointer", marginRight: 8 }}>
                    <span>{glossary.enabled ? "Đang bật" : "Đang tắt"}</span>
                    <span className="switch">
                      <input
                        type="checkbox"
                        checked={glossary.enabled}
                        onChange={(e) => updateGlossaryRemote({ enabled: e.target.checked })}
                      />
                      <span className="switch-track" />
                    </span>
                  </label>
                )}
                <button className="btn sm" onClick={() => setGlossaryOpen(false)} title="Đóng">
                  <X size={14} />
                </button>
              </div>
            </div>

            {glossary ? (
              <>
                <div className="preset-bar">
                  <span style={{ fontSize: 12, color: "var(--text-2)" }}>Nạp nhanh bộ mẫu:</span>
                  <button className="btn sm" onClick={() => handleApplyPreset("tech_ai")}>
                    + Công nghệ / AI
                  </button>
                  <button className="btn sm" onClick={() => handleApplyPreset("business_edu")}>
                    + Doanh nghiệp / Giáo dục
                  </button>
                  <button className="btn sm" onClick={() => handleApplyPreset("medical")}>
                    + Y tế
                  </button>
                </div>

                <div className="glossary-grid">
                  {/* Column 1: Pre-MT ASR Fixes */}
                  <div className="glossary-col">
                    <div className="glossary-col-title">
                      1. Sửa lỗi nghe nhầm ASR ({glossary.asr_fixes.length})
                    </div>
                    <div className="glossary-col-hint">
                      Tự động sửa ngay trước khi hiển thị và trước khi dịch (VD: mái chủ → máy chủ)
                    </div>
                    <form className="rule-form" onSubmit={handleAddAsrFix}>
                      <input
                        className="input"
                        placeholder="Nghe nhầm (VD: coopfro)"
                        value={newAsrWrong}
                        onChange={(e) => setNewAsrWrong(e.target.value)}
                      />
                      <span className="rule-arrow">→</span>
                      <input
                        className="input"
                        placeholder="Từ đúng (VD: Cloudflare)"
                        value={newAsrCorrect}
                        onChange={(e) => setNewAsrCorrect(e.target.value)}
                      />
                      <button
                        className="btn primary sm"
                        type="submit"
                        disabled={!newAsrWrong.trim() || !newAsrCorrect.trim()}
                      >
                        <Plus size={13} />
                      </button>
                    </form>
                    <div className="rule-list">
                      {glossary.asr_fixes.map((rule, idx) => (
                        <div className="rule-row" key={`${rule.wrong}-${idx}`}>
                          <span className="rule-src">{rule.wrong}</span>
                          <span className="rule-arrow">→</span>
                          <span className="rule-tgt">{rule.correct}</span>
                          <button
                            className="btn sm"
                            onClick={() => handleRemoveAsrFix(idx)}
                            title="Xóa quy tắc"
                          >
                            <Trash2 size={11} />
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Column 2: Protected Proper Nouns & Locked Translations */}
                  <div className="glossary-col">
                    <div className="glossary-col-title">
                      2. Khóa tên riêng / Thuật ngữ ({glossary.term_locks.length})
                    </div>
                    <div className="glossary-col-hint">
                      Giữ nguyên tên người, công ty hoặc ép nghĩa cố định (VD: Nguyễn Lê Trường → giữ nguyên)
                    </div>
                    <form className="rule-form" onSubmit={handleAddTermLock}>
                      <input
                        className="input"
                        placeholder="Tên / Cụm gốc"
                        value={newLockSource}
                        onChange={(e) => setNewLockSource(e.target.value)}
                      />
                      <span className="rule-arrow">→</span>
                      <input
                        className="input"
                        placeholder="Bản dịch khóa (để trống = giữ nguyên)"
                        value={newLockTarget}
                        onChange={(e) => setNewLockTarget(e.target.value)}
                      />
                      <button
                        className="btn primary sm"
                        type="submit"
                        disabled={!newLockSource.trim()}
                      >
                        <Plus size={13} />
                      </button>
                    </form>
                    <div className="rule-list">
                      {glossary.term_locks.map((rule, idx) => (
                        <div className="rule-row" key={`${rule.source}-${idx}`}>
                          <span className="rule-src">{rule.source}</span>
                          <span className="rule-arrow">→</span>
                          <span className="rule-tgt">{rule.target}</span>
                          <button
                            className="btn sm"
                            onClick={() => handleRemoveTermLock(idx)}
                            title="Xóa quy tắc"
                          >
                            <Trash2 size={11} />
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="modal-foot">
                  <span>
                    Tự lưu: <code>{glossary.path}</code>
                    {typeof status?.hotwords_active === "number" && status.hotwords_active > 0
                      ? ` • ASR Hotwords: ${status.hotwords_active} từ khóa đang bật`
                      : ""}
                  </span>
                  <button className="btn primary sm" onClick={() => setGlossaryOpen(false)}>
                    Xong
                  </button>
                </div>
              </>
            ) : (
              <div style={{ padding: 24, textAlign: "center", color: "var(--text-2)" }}>
                Đang tải từ điển họp...
              </div>
            )}
          </div>
        </div>
      )}

      {/* QR Code LAN Audience & OBS Modal */}
      {qrOpen && (
        <div className="modal-backdrop" onClick={() => setQrOpen(false)}>
          <div className="modal-card qr-modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head qr-modal-head">
              <div className="row" style={{ gap: 12, minWidth: 0 }}>
                <div className="brand-crest qr-crest">
                  <img src="/school-logo.png" alt="PolyVoice Pro" />
                </div>
                <div style={{ minWidth: 0 }}>
                  <div className="modal-title qr-modal-title">
                    <span>Phát Phụ Đề Hội Trường</span>
                    <span className="brand-pro-tag">QR &amp; OBS</span>
                  </div>
                  <div className="modal-sub">
                    Chiếu mã QR để đại biểu quét xem phụ đề trực tiếp theo ngôn ngữ mong muốn
                  </div>
                </div>
              </div>
              <div className="row" style={{ gap: 8, flexShrink: 0 }}>
                <span className={`pill ${audienceCount > 0 ? "ok" : ""}`}>
                  <span className={`dot ${audienceCount > 0 ? "pulse" : ""}`} />
                  {audienceCount} thiết bị
                </span>
                <button
                  className="btn sm icon-only"
                  onClick={() => setQrOpen(false)}
                  title="Đóng"
                >
                  <X size={14} />
                </button>
              </div>
            </div>

            {audienceInfo ? (
              <div className="qr-grid">
                <div className="qr-box">
                  <div
                    className="qr-svg"
                    dangerouslySetInnerHTML={{ __html: audienceInfo.qr_svg }}
                  />
                  <div className="qr-scan-caption">
                    <div className="qr-scan-title">Quét bằng Camera điện thoại</div>
                    <div className="qr-scan-sub">Cùng Wi-Fi nội bộ • Không cần cài app</div>
                  </div>
                  <div className="qr-lang-strip">
                    {["Anh", "Thái", "Lào", "Trung", "+200"].map((lbl) => (
                      <span className="qr-lang-chip" key={lbl}>
                        {lbl}
                      </span>
                    ))}
                  </div>
                  {audienceInfo.active_languages.length > 0 && (
                    <div className="qr-active-langs">
                      Đang xem: {audienceInfo.active_languages.join(", ")}
                    </div>
                  )}
                </div>

                <div className="qr-details">
                  <div className="qr-link-card featured">
                    <div className="qr-link-head">
                      <span className="qr-step-num primary">1</span>
                      <Smartphone size={14} className="qr-link-icon primary" />
                      <span className="qr-link-title">Khán giả trên điện thoại</span>
                      <span className="qr-link-tag">Mạng LAN</span>
                    </div>
                    <div className="qr-url-row">
                      <input
                        className="input qr-url-input"
                        readOnly
                        value={audienceInfo.audience_url}
                        onFocus={(e) => e.currentTarget.select()}
                      />
                      <button
                        className="btn primary sm qr-copy-btn"
                        onClick={() => handleCopy("qr-aud", audienceInfo.audience_url)}
                      >
                        {copiedId === "qr-aud" ? (
                          <>
                            <Check size={12} /> Đã chép
                          </>
                        ) : (
                          <>
                            <Copy size={12} /> Chép
                          </>
                        )}
                      </button>
                    </div>
                    <div className="qr-link-desc">
                      Mỗi khán giả tự chọn ngôn ngữ phụ đề riêng ngay trên trình duyệt (VD: máy chiếu hiện Tiếng Anh, khách Thái/Lào chọn ngôn ngữ riêng).
                    </div>
                  </div>

                  <div className="qr-link-card">
                    <div className="qr-link-head">
                      <span className="qr-step-num">2</span>
                      <Tv size={14} className="qr-link-icon" />
                      <span className="qr-link-title">Phụ đề trong suốt cho OBS / Livestream</span>
                    </div>
                    <div className="qr-url-row">
                      <input
                        className="input qr-url-input"
                        readOnly
                        value={audienceInfo.obs_url}
                        onFocus={(e) => e.currentTarget.select()}
                      />
                      <button
                        className="btn sm qr-copy-btn"
                        onClick={() => handleCopy("qr-obs", audienceInfo.obs_url)}
                      >
                        {copiedId === "qr-obs" ? (
                          <>
                            <Check size={12} /> Đã chép
                          </>
                        ) : (
                          <>
                            <Copy size={12} /> Chép
                          </>
                        )}
                      </button>
                    </div>
                    <div className="qr-link-desc">
                      OBS Studio → thêm <b>Browser Source</b> (1920×260). Thêm{" "}
                      <code>?size=36&amp;bilingual=1</code> để tùy chỉnh cỡ chữ.
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div style={{ padding: 36, textAlign: "center", color: "var(--text-2)" }}>
                Đang tạo mã QR nội bộ...
              </div>
            )}

            <div className="modal-foot">
              <span className="qr-foot-note">
                <Wifi size={13} /> Chạy 100% qua mạng nội bộ (Wi-Fi / LAN), không gửi dữ liệu ra Internet.
              </span>
              <button className="btn sm" onClick={() => setQrOpen(false)}>
                Đóng
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Batch Audio/Video File Translator Modal */}
      {fileModalOpen && (
        <div className="modal-backdrop" onClick={() => !fileBusy && setFileModalOpen(false)}>
          <div className="modal-card" style={{ width: "min(820px, 100%)" }} onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <div className="row" style={{ gap: 8 }}>
                <FileAudio size={17} color="var(--accent)" />
                <div>
                  <div className="modal-title">Dịch File Ghi Âm &amp; Video Offline (.MP3 / .WAV / .M4A / .MP4)</div>
                  <div className="modal-sub">
                    Hướng dịch hiện tại: <b>{currentSrcObj?.name_vi || sourceLang} → {currentTgtObj?.name_vi || targetLang}</b> • Engine: <b>{mtMode}</b>
                  </div>
                </div>
              </div>
              <button className="btn sm icon-only" disabled={fileBusy} onClick={() => setFileModalOpen(false)} title="Đóng">
                <X size={14} />
              </button>
            </div>

            <div className="modal-body-pad">
              {fileBusy ? (
                (() => {
                  const pct = Math.max(2, Math.min(100, Math.round(fileProgress?.percent ?? 2)));
                  const elapsedClient = fileStartTsRef.current
                    ? Math.max(0, Math.floor((fileTickNow - fileStartTsRef.current) / 1000))
                    : Math.floor(fileProgress?.elapsed_s ?? 0);
                  const sinceUpdate = fileProgUpdatedAtRef.current
                    ? Math.max(0, (fileTickNow - fileProgUpdatedAtRef.current) / 1000)
                    : 0;
                  const liveEta =
                    fileProgress?.eta_s != null
                      ? Math.max(1, fileProgress.eta_s - sinceUpdate)
                      : null;
                  const hasDur = (fileProgress?.duration_s ?? 0) > 0;
                  const hasChunks = (fileProgress?.total_chunks ?? 0) > 0;
                  return (
                    <div className="file-progress-card">
                      <div className="file-prog-head">
                        <div className="row" style={{ gap: 10, minWidth: 0 }}>
                          <div className="file-prog-icon">
                            <Loader2 size={18} className="spin" />
                          </div>
                          <div style={{ minWidth: 0 }}>
                            <div className="file-prog-name">
                              {fileProgress?.filename || "Đang xử lý file..."}
                            </div>
                            <div className="file-prog-stage">
                              {fileCancelling
                                ? "Đang dừng tác vụ sau phân đoạn hiện tại..."
                                : fileProgress?.message || "Đang khởi tạo..."}
                            </div>
                          </div>
                        </div>
                        <div className="file-prog-pct">{pct}%</div>
                      </div>

                      <div className="file-prog-track" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
                        <div className="file-prog-fill" style={{ width: `${pct}%` }} />
                      </div>

                      <div className="file-stats-grid">
                        <div className="file-stat-box highlight">
                          <div className="file-stat-label">
                            <Clock size={12} /> Thời gian còn lại
                          </div>
                          <div className="file-stat-value accent">
                            {formatEtaVi(liveEta, fileProgress?.eta_estimated)}
                          </div>
                        </div>

                        <div className="file-stat-box">
                          <div className="file-stat-label">
                            <Layers size={12} /> Tiến độ phân đoạn
                          </div>
                          <div className="file-stat-value">
                            {hasChunks
                              ? `${fileProgress?.current_chunk ?? 0} / ${fileProgress?.total_chunks} đoạn`
                              : "Đang phân tách..."}
                          </div>
                          {hasDur && (
                            <div className="file-stat-sub">
                              {formatClockSec(fileProgress?.processed_audio_s ?? 0)} / {formatClockSec(fileProgress?.duration_s ?? 0)} âm thanh
                            </div>
                          )}
                        </div>

                        <div className="file-stat-box">
                          <div className="file-stat-label">
                            <Zap size={12} /> Tốc độ &amp; Đã chạy
                          </div>
                          <div className="file-stat-value">
                            {(fileProgress?.speed_x ?? 0) > 0
                              ? `${fileProgress?.speed_x}× thực tế`
                              : "Đang đo..."}
                          </div>
                          <div className="file-stat-sub">
                            Đã chạy: {formatClockSec(elapsedClient)}
                          </div>
                        </div>
                      </div>

                      <div className="file-prog-foot">
                        <span>Tự động lưu sẵn file .TXT &amp; .SRT song ngữ khi hoàn tất</span>
                        <button
                          className="btn sm"
                          disabled={fileCancelling}
                          onClick={handleCancelFileTranslate}
                        >
                          <X size={12} /> {fileCancelling ? "Đang hủy..." : "Hủy dịch"}
                        </button>
                      </div>
                    </div>
                  );
                })()
              ) : (
                <label
                  className="dropzone"
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => {
                    e.preventDefault();
                    const f = e.dataTransfer.files?.[0];
                    if (f && !fileBusy) handleUploadMediaFile(f);
                  }}
                >
                  <input
                    type="file"
                    accept="audio/*,video/*,.wav,.mp3,.m4a,.mp4,.mov,.webm,.ogg,.flac"
                    style={{ display: "none" }}
                    disabled={fileBusy}
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) handleUploadMediaFile(f);
                      e.target.value = "";
                    }}
                  />
                  <Upload size={22} color="var(--info)" />
                  <div style={{ fontWeight: 700, fontSize: 13.5 }}>
                    Bấm chọn file hoặc kéo thả file .MP3 / .WAV / .M4A / .MP4 vào đây
                  </div>
                  <div style={{ fontSize: 11.5, color: "var(--text-2)" }}>
                    Tự động lưu sẵn file .TXT &amp; .SRT song ngữ vào thư mục Biên bản
                  </div>
                </label>
              )}

              {fileResult && (
                <div className="summary-body" style={{ marginTop: 12 }}>
                  <div className="row" style={{ justifyContent: "space-between", flexWrap: "wrap" }}>
                    <span className="autosave-badge">
                      ✓ {fileResult.filename} ({fileResult.duration_s}s) • Xong trong {fileResult.elapsed_s}s ({fileResult.speed_x}×) • {fileResult.segment_count} đoạn
                    </span>
                    <div className="row">
                      <button
                        className="btn primary sm"
                        onClick={() =>
                          handleDownloadBlob(`${stripFileExtension(fileResult.filename)}.srt`, fileResult.srt_content)
                        }
                      >
                        <Download size={12} /> Tải .SRT
                      </button>
                      <button
                        className="btn sm"
                        onClick={() =>
                          handleDownloadBlob(`${stripFileExtension(fileResult.filename)}.txt`, fileResult.txt_content)
                        }
                      >
                        <FileText size={12} /> Tải .TXT
                      </button>
                    </div>
                  </div>
                  <div className="rule-list" style={{ maxHeight: 260, marginTop: 8 }}>
                    {fileResult.segments.map((seg) => (
                      <div className="history-item" key={seg.index} style={{ padding: "8px 10px" }}>
                        <div className="history-meta">
                          <span>#{seg.index} • {seg.start_srt} → {seg.end_srt} • {seg.mt_engine}</span>
                        </div>
                        <div className="history-vi" style={{ fontSize: 12.5 }}>{seg.source_text}</div>
                        <div className="history-tr" style={{ fontSize: 13.5 }}>{seg.translated_text}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="modal-foot">
              <span>{fileResult ? `Đã lưu tại: ${fileResult.srt_file}` : "Hỗ trợ tốt nhất cho file Tiếng Việt & Tiếng Anh"}</span>
              <button className="btn sm" disabled={fileBusy} onClick={() => setFileModalOpen(false)}>
                Đóng
              </button>
            </div>
          </div>
        </div>
      )}

      {/* AI Meeting Summary & Action Items Modal */}
      {summaryOpen && (
        <div className="modal-backdrop" onClick={() => setSummaryOpen(false)}>
          <div className="modal-card" style={{ width: "min(800px, 100%)" }} onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <div className="row" style={{ gap: 8 }}>
                <Sparkles size={17} color="var(--accent)" />
                <div>
                  <div className="modal-title">Tóm Tắt Cuộc Họp &amp; Việc Cần Làm (Song Ngữ Offline)</div>
                  <div className="modal-sub">
                    Trích xuất tự động ý chính, số liệu và mục hành động từ toàn bộ lịch sử phiên hiện tại
                  </div>
                </div>
              </div>
              <button className="btn sm" onClick={() => setSummaryOpen(false)}>
                <X size={14} />
              </button>
            </div>

            <div className="modal-body-pad" style={{ overflowY: "auto", maxHeight: 480 }}>
              {summaryBusy ? (
                <div style={{ padding: 28, textAlign: "center", color: "var(--text-2)" }}>
                  Đang tổng hợp biên bản song ngữ...
                </div>
              ) : summaryData ? (
                <div className="summary-body">
                  <div className="history-item">
                    <div className="glossary-col-title">Tổng quan phiên ({summaryData.time_range || "—"})</div>
                    <div className="history-vi">{summaryData.overview_src}</div>
                    <div className="history-tr" style={{ fontSize: 13.5 }}>{summaryData.overview_tgt}</div>
                  </div>

                  <div className="glossary-col-title" style={{ marginTop: 10 }}>
                    Điểm chính ({summaryData.key_points.length})
                  </div>
                  {summaryData.key_points.map((kp, i) => (
                    <div className="history-item" key={`kp-${i}`}>
                      <div className="history-meta"><span>[{kp.timestamp}]</span></div>
                      <div className="history-vi">{kp.source_text}</div>
                      <div className="history-tr" style={{ fontSize: 13.5 }}>{kp.translated_text}</div>
                    </div>
                  ))}

                  {summaryData.action_items.length > 0 && (
                    <>
                      <div className="glossary-col-title" style={{ marginTop: 10, color: "var(--green)" }}>
                        Việc cần làm / Số liệu quan trọng ({summaryData.action_items.length})
                      </div>
                      {summaryData.action_items.map((act, i) => (
                        <div className="history-item" key={`act-${i}`}>
                          <div className="history-meta"><span>[{act.timestamp}]</span></div>
                          <div className="history-vi">• {act.source_text}</div>
                          <div className="history-tr" style={{ fontSize: 13.5 }}>• {act.translated_text}</div>
                        </div>
                      ))}
                    </>
                  )}
                </div>
              ) : null}
            </div>

            <div className="modal-foot">
              <span>{summaryData ? `${summaryData.total_sentences} câu • ${summaryData.total_words} từ` : ""}</span>
              <div className="row">
                {summaryData && (
                  <button
                    className="btn sm"
                    onClick={() => {
                      const md = [
                        `# Tóm tắt cuộc họp (${summaryData.time_range || ""})`,
                        "",
                        summaryData.overview_src,
                        summaryData.overview_tgt,
                        "",
                        "## Điểm chính",
                        ...summaryData.key_points.map((k) => `- [${k.timestamp}] ${k.source_text}\n  -> ${k.translated_text}`),
                        "",
                        "## Việc cần làm / Số liệu",
                        ...summaryData.action_items.map((a) => `- [${a.timestamp}] ${a.source_text}\n  -> ${a.translated_text}`),
                      ].join("\n");
                      handleCopy("summary-md", md);
                    }}
                  >
                    {copiedId === "summary-md" ? <Check size={12} /> : <Copy size={12} />} Chép Markdown
                  </button>
                )}
                <button className="btn primary sm" onClick={() => setSummaryOpen(false)}>
                  Đóng
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
      {/* Official Cloud Translation API Modal (Google Cloud / Microsoft Azure F0 / Google Gemini) */}
      {cloudOpen && (
        <div className="modal-backdrop" onClick={() => setCloudOpen(false)}>
          <div className="modal-card" style={{ width: "min(760px, 100%)" }} onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <div className="row" style={{ gap: 8 }}>
                <Cloud size={17} color="var(--accent)" />
                <div>
                  <div className="modal-title">Tích Hợp Cloud Translation API Chính Thức (Hybrid Nháp Offline)</div>
                  <div className="modal-sub">
                    Chỉ gửi câu đã chốt lên Cloud API (tiết kiệm ~85% hạn mức) • Tự động rơi về Offline khi chạm ngưỡng miễn phí $0
                  </div>
                </div>
              </div>
              <button className="btn sm icon-only" onClick={() => setCloudOpen(false)} title="Đóng">
                <X size={14} />
              </button>
            </div>

            {cloudStatus ? (
              <div className="modal-body-pad" style={{ overflowY: "auto" }}>
                {/* Provider Tabs */}
                <div className="cloud-tabs">
                  {(
                    [
                      ["azure", "Microsoft Azure", "2.000.000 ký tự/tháng Free F0"],
                      ["google", "Google Cloud", "500.000 ký tự/tháng Free"],
                      ["gemini", "Google Gemini", "Siêu rẻ • Hiểu ngữ cảnh & Từ điển"],
                    ] as const
                  ).map(([pid, title, sub]) => {
                    const pState = cloudStatus.providers[pid];
                    return (
                      <button
                        key={pid}
                        className={`cloud-tab ${cloudTab === pid ? "active" : ""}`}
                        onClick={() => syncCloudFormForTab(cloudStatus, pid)}
                      >
                        <div className="cloud-tab-title">
                          <span>{title}</span>
                          {pState?.has_key && (
                            <span style={{ color: "var(--green-text)", fontSize: 10.5 }}>● Đã lưu Key</span>
                          )}
                        </div>
                        <span className="cloud-tab-sub">{sub}</span>
                      </button>
                    );
                  })}
                </div>

                {/* Selected Provider Config Panel */}
                {(() => {
                  const p = cloudStatus.providers[cloudTab];
                  return (
                    <div className="cloud-panel">
                      {/* Quota Meter */}
                      <div className="row" style={{ justifyContent: "space-between" }}>
                        <span style={{ fontSize: 12, fontWeight: 700, color: "var(--text-0)" }}>
                          Hạn mức tháng {cloudStatus.month}:{" "}
                          <span style={{ color: "var(--accent-text)" }}>
                            {p.used_chars.toLocaleString("vi-VN")} / {p.monthly_limit.toLocaleString("vi-VN")} ký tự
                          </span>{" "}
                          ({p.pct_used}%)
                        </span>
                        <button
                          className="btn sm"
                          style={{ padding: "3px 8px", fontSize: 11 }}
                          onClick={handleResetCloudQuota}
                          title="Đặt lại bộ đếm tháng về 0"
                        >
                          Đặt lại bộ đếm
                        </button>
                      </div>
                      <div className="progress-track">
                        <div className="progress-fill" style={{ width: `${Math.min(100, p.pct_used)}%` }} />
                      </div>

                      {/* API Key Input */}
                      <label className="field-label">
                        <span>
                          API Key {cloudTab === "azure" ? "(Azure Translator Key 1 / Key 2)" : cloudTab === "google" ? "(Google Cloud Translation API Key)" : "(Google AI Studio / Gemini API Key)"}
                        </span>
                        <div className="row">
                          <input
                            className="input"
                            type="password"
                            autoComplete="off"
                            value={cloudKeyInput}
                            onChange={(e) => setCloudKeyInput(e.target.value)}
                            placeholder={
                              p.has_key
                                ? `Đã lưu cục bộ (${p.key_preview}) — chỉ cần nhập nếu muốn đổi Key mới`
                                : "Dán API Key chính thức của bạn vào đây..."
                            }
                          />
                          {p.has_key && (
                            <button
                              className="btn sm"
                              onClick={handleClearCloudKey}
                              disabled={cloudSaving}
                              title="Xóa API Key đã lưu của nhà cung cấp này"
                            >
                              <Trash2 size={12} /> Xóa Key
                            </button>
                          )}
                        </div>
                      </label>

                      {cloudTab === "azure" && (
                        <div className="row" style={{ gap: 10 }}>
                          <label className="field-label" style={{ flex: 1 }}>
                            <span>Azure Region (VD: southeastasia, eastasia, hoặc để trống nếu Global)</span>
                            <input
                              className="input"
                              value={cloudRegion}
                              onChange={(e) => setCloudRegion(e.target.value)}
                              placeholder="southeastasia"
                            />
                          </label>
                          <label className="field-label" style={{ flex: 1 }}>
                            <span>Endpoint</span>
                            <input
                              className="input"
                              value={cloudEndpoint}
                              onChange={(e) => setCloudEndpoint(e.target.value)}
                              placeholder="https://api.cognitive.microsofttranslator.com"
                            />
                          </label>
                        </div>
                      )}

                      {cloudTab === "gemini" && (
                        <label className="field-label">
                          <span>Model Gemini</span>
                          <select
                            className="select"
                            value={cloudModel}
                            onChange={(e) => setCloudModel(e.target.value)}
                          >
                            <option value="gemini-2.5-flash">gemini-2.5-flash (Khuyên dùng • Nhanh ~0.9s, ổn định nhất)</option>
                            <option value="gemini-3.1-flash-lite">gemini-3.1-flash-lite (Flash Lite tiết kiệm nhất)</option>
                            <option value="gemini-3.5-flash-lite">gemini-3.5-flash-lite (Flash Lite thế hệ 3.5)</option>
                            <option value="gemini-3.8-flash">gemini-3.8-flash (Flash thế hệ 3.8 mới nhất)</option>
                          </select>
                        </label>
                      )}

                      <label className="field-label">
                        <span>
                          Ngưỡng tự khóa mỗi tháng (ký tự) —{" "}
                          {cloudTab === "azure"
                            ? "Gói F0 miễn phí 2.000.000/tháng (~44 giờ họp)"
                            : cloudTab === "google"
                            ? "Gói Free miễn phí 500.000/tháng (~11 giờ họp)"
                            : "Tự đặt giới hạn chi tiêu an toàn"}
                        </span>
                        <input
                          className="input"
                          type="number"
                          min={1000}
                          step={10000}
                          value={cloudLimit}
                          onChange={(e) => setCloudLimit(Number(e.target.value))}
                        />
                      </label>

                      <label className="toggle-row">
                        <span>Dịch nháp khi đang nói chạy 100% Offline (Tiết kiệm ~85% hạn mức API)</span>
                        <span className="switch">
                          <input
                            type="checkbox"
                            checked={cloudDraftOffline}
                            onChange={(e) => setCloudDraftOffline(e.target.checked)}
                          />
                          <span className="switch-track" />
                        </span>
                      </label>

                      <label className="toggle-row">
                        <span>Tự khóa ở hạn mức &amp; tự chuyển mượt về Offline SOTA (Không phát sinh phí)</span>
                        <span className="switch">
                          <input
                            type="checkbox"
                            checked={cloudHardCap}
                            onChange={(e) => setCloudHardCap(e.target.checked)}
                          />
                          <span className="switch-track" />
                        </span>
                      </label>
                    </div>
                  );
                })()}

                {cloudTestResult && (
                  <div className={`cloud-test-box ${cloudTestResult.ok ? "ok" : "err"}`}>
                    {cloudTestResult.message}
                  </div>
                )}
              </div>
            ) : (
              <div style={{ padding: 28, textAlign: "center", color: "var(--text-2)" }}>
                Đang tải cấu hình Cloud API...
              </div>
            )}

            <div className="modal-foot">
              <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                Key chỉ lưu cục bộ trên máy bạn (<code>cloud_config.json</code>), tự động rơi về Offline khi mất mạng.
              </span>
              <div className="row">
                <button
                  className="btn sm"
                  onClick={handleTestCloudApi}
                  disabled={cloudTestBusy || cloudSaving}
                >
                  <Zap size={12} /> {cloudTestBusy ? "Đang kiểm tra..." : "Kiểm tra API"}
                </button>
                <button
                  className="btn sm"
                  onClick={() => handleSaveCloudConfig(false)}
                  disabled={cloudSaving}
                >
                  Lưu
                </button>
                <button
                  className="btn primary sm"
                  onClick={() => handleSaveCloudConfig(true)}
                  disabled={cloudSaving}
                >
                  <Check size={12} /> Lưu &amp; Bật Cloud API
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Native 1-Click Auto-Update Modal */}
      {updateModalOpen && updateInfo && (
        <div className="modal-backdrop" onClick={() => !updateInstalling && setUpdateModalOpen(false)}>
          <div className="modal-card" style={{ width: "min(560px, 100%)" }} onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <div className="row" style={{ gap: 10 }}>
                <div className="brand-crest sm">
                  <img src="/school-logo.png" alt="Logo" />
                </div>
                <div>
                  <div className="modal-title">Có phiên bản mới — PolyVoice Pro v{updateInfo.version}</div>
                  <div className="modal-sub">
                    Phiên bản hiện tại: v{APP_VERSION} • Tự động nâng cấp giữ nguyên model offline &amp; từ điển
                  </div>
                </div>
              </div>
              <button
                className="btn sm icon-only"
                disabled={updateInstalling}
                onClick={() => setUpdateModalOpen(false)}
                title="Đóng"
              >
                <X size={14} />
              </button>
            </div>

            <div className="modal-body-pad">
              <div className="update-notes-box">
                <div style={{ fontSize: 12, fontWeight: 700, color: "var(--accent-text)", marginBottom: 6 }}>
                  Nội dung cập nhật (v{updateInfo.version})
                </div>
                <div style={{ fontSize: 12.5, color: "var(--text-1)", whiteSpace: "pre-wrap", lineHeight: 1.5 }}>
                  {updateInfo.body || "Cải thiện độ chính xác nhận dạng giọng nói, tốc độ dịch thuật và độ ổn định hệ thống."}
                </div>
              </div>

              {updateInstalling && (
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  <div className="hint">
                    <span>{updateProgressText}</span>
                    <span>{updateProgress}%</span>
                  </div>
                  <div className="progress-track">
                    <div className="progress-fill" style={{ width: `${updateProgress}%` }} />
                  </div>
                </div>
              )}
            </div>

            <div className="modal-foot">
              <span>Nguồn chính thức: github.com/{GITHUB_REPO}</span>
              <div className="row">
                <button
                  className="btn sm"
                  disabled={updateInstalling}
                  onClick={() => setUpdateModalOpen(false)}
                >
                  Để sau
                </button>
                <button
                  className="btn primary sm"
                  disabled={updateInstalling}
                  onClick={handleInstallUpdate}
                >
                  <ArrowUpCircle size={13} />{" "}
                  {updateInstalling
                    ? "Đang cập nhật..."
                    : updateInfo.nativeUpdate
                    ? "Cập nhật & Khởi động lại"
                    : "Mở trang tải bản mới"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
