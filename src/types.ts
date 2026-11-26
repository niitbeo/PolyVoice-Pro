declare global {
  const __APP_VERSION__: string;
}

export interface AudioDevice {
  id: number;
  name: string;
  channels: number;
  default_samplerate: number;
  is_default: boolean;
}

export interface LanguageEntry {
  code: string;
  name_vi: string;
  name_en: string;
  native: string;
  tts_voice: string;
  pinned: boolean;
}

export interface SourceLanguageEntry {
  code: string;
  locale: string;
  name_vi: string;
  name_en: string;
  native: string;
  offline_onnx: boolean;
  preferred_model?: string;
  preferred_model_mb?: number;
  fallback_model?: string;
  recommended_pack?: "core" | "asia" | "global";
  asr_tier?: "fast" | "universal" | "experimental";
  asr_note?: string;
}

export interface SourceReadiness {
  lang_code: string;
  ready: boolean;
  preferred_ready: boolean;
  fallback_ready: boolean;
  active_tier: "fast" | "universal" | "apple" | "experimental" | "missing";
  preferred_model: string;
  preferred_model_mb: number;
  fallback_model: string;
  recommended_pack: "core" | "asia" | "global";
  asr_tier: "fast" | "universal" | "experimental";
  note?: string;
}

export interface ModelState {
  model_id: string;
  status: "ready" | "missing" | "downloading" | "error" | "idle";
  progress: number;
  downloaded_mb: number;
  total_mb: number;
  message: string;
}

export interface ModelsStatus {
  models_dir: string;
  asr_zipformer_streaming: ModelState;
  asr_zipformer_multi?: ModelState;
  asr_moonshine_en?: ModelState;
  asr_moonshine_ja?: ModelState;
  asr_moonshine_ko?: ModelState;
  asr_parakeet_eu?: ModelState;
  asr_whisper_turbo?: ModelState;
  asr_vietasr: ModelState;
  mt_hymt2?: ModelState;
  mt_envit5: ModelState;
  mt_nllb_600m: ModelState;
}

export type AudioSourceId = "mic" | "system" | "dual" | "mix";
export type InputChannelId = "mix" | "left" | "right";
export type AsrEngineId =
  | "auto"
  | "soniox-cloud"
  | "hybrid-en"
  | "moonshine-en"
  | "moonshine-ja"
  | "moonshine-ko"
  | "parakeet-eu"
  | "apple-native"
  | "whisper-turbo"
  | "zipformer-30m-streaming"
  | "zipformer-multi"
  | "viet-asr";
export type MtModeId =
  | "auto-offline"
  | "light-offline"
  | "cloud-api"
  | "hy-mt2"
  | "nllb"
  | "fast-online"
  | "local-llm";

export type CloudProviderId = "soniox" | "azure" | "google" | "gemini";

export interface CloudProviderState {
  id: CloudProviderId;
  name: string;
  free_tier: number;
  unit?: "chars" | "seconds";
  has_key: boolean;
  key_preview: string;
  monthly_limit: number;
  used_chars: number;
  remaining_chars: number;
  pct_used: number;
  cap_reached: boolean;
  region: string;
  endpoint: string;
  model: string;
}

export interface CloudApiStatus {
  provider: CloudProviderId;
  month: string;
  hard_cap_enabled: boolean;
  draft_offline: boolean;
  active_ready: boolean;
  active_cap_reached: boolean;
  providers: Record<CloudProviderId, CloudProviderState>;
  config_path: string;
}
export type ParagraphModeId = "short" | "medium" | "long";

export interface AsrFixRule {
  wrong: string;
  correct: string;
}

export interface TermLockRule {
  source: string;
  target: string;
}

export interface GlossaryData {
  enabled: boolean;
  asr_fixes: AsrFixRule[];
  term_locks: TermLockRule[];
  presets: string[];
  path: string;
}

export interface LogsStatus {
  logs_dir: string;
  txt_file: string;
  srt_file: string;
  saved_count: number;
}

export interface SessionConfig {
  device_id: number | null;
  loopback_device_id?: number | null;
  audio_source: AudioSourceId;
  input_channel?: InputChannelId;
  input_gain_db?: number;
  auto_gain?: boolean;
  source_lang: string;
  target_lang: string;
  asr_engine: AsrEngineId;
  mt_mode: MtModeId;
  paragraph_mode: ParagraphModeId;
  translate_partials: boolean;
  pass2_refine: boolean;
  eco_mode: boolean;
  beam_size: number;
  llm_url: string;
  llm_model: string;
  llm_api_key: string;
}

export interface AudienceInfo {
  lan_ip: string;
  port: number;
  audience_url: string;
  obs_url: string;
  qr_svg: string;
  connected_count: number;
  active_languages: string[];
}

export interface MeetingSummary {
  total_sentences: number;
  total_words: number;
  time_range?: string;
  overview_src: string;
  overview_tgt: string;
  key_points: Array<{ timestamp: string; speaker?: string; source_text: string; translated_text: string }>;
  action_items: Array<{ timestamp: string; speaker?: string; source_text: string; translated_text: string }>;
}

export interface FileTranslateProgress {
  job_id: string;
  filename: string;
  active: boolean;
  stage: "uploading" | "decoding" | "loading_models" | "translating" | "saving" | "done" | "cancelled" | "error";
  percent: number;
  message: string;
  current_chunk: number;
  total_chunks: number;
  processed_audio_s: number;
  duration_s: number;
  elapsed_s: number;
  eta_s: number | null;
  eta_estimated?: boolean;
  speed_x: number;
}

export interface FileTranslateResult {
  ok: boolean;
  cancelled?: boolean;
  filename: string;
  duration_s: number;
  elapsed_s: number;
  speed_x: number;
  asr_used?: string;
  vad_used?: boolean;
  normalized?: boolean;
  source_lang?: string;
  target_lang?: string;
  segment_count: number;
  segments: Array<{
    index: number;
    start_srt: string;
    end_srt: string;
    source_text: string;
    translated_text: string;
    mt_engine: string;
  }>;
  txt_file: string;
  srt_file: string;
  txt_content: string;
  srt_content: string;
  srt_target_content?: string;
  srt_source_content?: string;
}

export interface SlidePageSummary {
  index: number;
  title: string;
  preview: string;
  has_notes?: boolean;
  terms?: string[];
}

export interface SlideContextTerm {
  source: string;
  target: string;
  enabled: boolean;
  origin?: "slide" | "dau_preset" | "manual";
}

export interface SlideContextSummary {
  enabled: boolean;
  filename: string;
  topic: string;
  slide_count: number;
  term_count: number;
  active_slide_index: number;
  active_slide_title: string;
}

export interface SlideContextPayload {
  enabled: boolean;
  filename: string;
  topic: string;
  slide_count: number;
  active_slide_index: number;
  active_slide_title: string;
  pages: SlidePageSummary[];
  terms: SlideContextTerm[];
}

export interface EngineStatus {
  listening: boolean;
  active_asr_label: string;
  remote_asr_label?: string;
  detected_source_lang?: string;
  asr_loaded: boolean;
  pass2_loaded?: boolean;
  hotwords_active?: number;
  zipformer_ready: boolean;
  zipformer_multi_ready?: boolean;
  moonshine_en_ready?: boolean;
  moonshine_ja_ready?: boolean;
  moonshine_ko_ready?: boolean;
  parakeet_eu_ready?: boolean;
  whisper_turbo_ready?: boolean;
  hymt2_ready?: boolean;
  hymt2_loaded?: boolean;
  hymt2_backend?: string;
  llama_server_ready?: boolean;
  envit5_ready: boolean;
  nllb_loaded: boolean;
  nllb_ready: boolean;
  mac_helper_ready: boolean;
  system_audio_ready?: boolean;
  apple_caps: {
    available: boolean;
    on_device_locales: string[];
    system_audio: boolean;
  };
  config: SessionConfig;
  source_lang_meta: SourceLanguageEntry | null;
  active_source_readiness?: SourceReadiness;
  target_lang_meta: LanguageEntry | null;
  logs?: LogsStatus;
  audience?: { connected_count: number; active_languages: string[] };
  glossary_enabled?: boolean;
  glossary_count?: number;
  slide_context?: SlideContextSummary;
  display_frozen?: boolean;
  display_freeze_message?: string;
  cloud_api?: CloudApiStatus;
  cloud_fallback_notice?: string;
  soniox_active?: boolean;
  envit5_device?: string;
  platform: {
    os: string;
    arch: string;
    python: string;
  };
  boot_error: string;
}

export type OverlayLayoutMode = "led-stage" | "stacked" | "youtube" | "target-only" | "split";
export const OVERLAY_LAYOUT_KEY = "polyvoice_overlay_layout";

export interface PartialEvent {
  id: string;
  speaker?: "self" | "remote" | "";
  source_text: string;
  translated_text: string;
  stable_text?: string;
  unstable_text?: string;
  source_lang: string;
  target_lang: string;
  asr_ms: number;
  mt_ms: number;
  mt_engine?: string;
}

export interface TranscriptEntry {
  id: string;
  paragraph_id?: string;
  item_ids?: string[];
  speaker?: "self" | "remote" | "";
  sentence_count?: number;
  source_text: string;
  translated_text: string;
  source_lang: string;
  target_lang: string;
  asr_ms: number;
  pass2_ms?: number;
  mt_ms: number;
  mt_engine: string;
  timestamp: string;
  log_saved_count?: number;
}

export type EvaluationIssueType = "ok" | "nuance" | "mt_inaccurate" | "asr_error";

export interface EvaluationItemResult {
  id: string;
  timestamp?: string;
  source_text: string;
  translated_text: string;
  source_lang: string;
  target_lang: string;
  score: number;
  issue_type: EvaluationIssueType;
  comment_vi: string;
  suggested_translation: string;
}

export interface GlossarySuggestion {
  wrong: string;
  correct: string;
}

export interface EvaluationReport {
  provider: string;
  mode: "semantic_audit" | "google_reference";
  overall_score: number;
  semantic_score: number;
  fluency_score: number;
  asr_score: number;
  verdict: string;
  summary_vi: string;
  items: EvaluationItemResult[];
  glossary_suggestions: GlossarySuggestion[];
  latency_ms: number;
  chars_used: number;
  cloud_api?: CloudApiStatus;
}

/**
 * YouTube-style Visual Paragraph Grouping:
 * Merges consecutive committed sentences that share the same `paragraph_id` & `speaker`
 * into one coherent multi-sentence paragraph for smooth reading on Monitor #2
 * and in History, while each sentence underneath was translated accurately on its own.
 */
export function groupEntriesByParagraph(chronological: TranscriptEntry[]): TranscriptEntry[] {
  const out: TranscriptEntry[] = [];
  for (const item of chronological) {
    const prev = out[out.length - 1];
    const canMerge =
      prev &&
      Boolean(item.paragraph_id) &&
      prev.paragraph_id === item.paragraph_id &&
      (prev.speaker || "") === (item.speaker || "") &&
      prev.source_lang === item.source_lang &&
      prev.target_lang === item.target_lang;

    if (canMerge) {
      prev.source_text = `${prev.source_text} ${item.source_text}`.replace(/\s{2,}/g, " ").trim();
      prev.translated_text = [prev.translated_text, item.translated_text]
        .filter(Boolean)
        .join(" ")
        .replace(/\s{2,}/g, " ")
        .trim();
      prev.asr_ms = item.asr_ms;
      prev.pass2_ms = item.pass2_ms;
      prev.mt_ms = item.mt_ms;
      prev.mt_engine = item.mt_engine;
      prev.sentence_count = (prev.sentence_count ?? 1) + 1;
      prev.item_ids = [...(prev.item_ids || []), item.id];
    } else {
      out.push({
        ...item,
        id: item.paragraph_id || item.id,
        item_ids: [item.id],
        sentence_count: 1,
      });
    }
  }
  return out;
}

const _ENGINE_PORT =
  (import.meta as unknown as { env?: Record<string, string | undefined> }).env?.VITE_ENGINE_PORT || "18765";
export const ENGINE_HTTP = `http://127.0.0.1:${_ENGINE_PORT}`;
export const ENGINE_WS = `ws://127.0.0.1:${_ENGINE_PORT}/ws`;

export type ThemeMode = "dark" | "light";
export const THEME_STORAGE_KEY = "voicerecore-theme";

export function getInitialTheme(): ThemeMode {
  if (typeof window === "undefined") return "dark";
  try {
    const saved = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    // ignore storage access issues
  }
  return "dark";
}

export function applyTheme(theme: ThemeMode): void {
  if (typeof document !== "undefined") {
    document.documentElement.setAttribute("data-theme", theme);
  }
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    // ignore
  }
}

const SHORT_LANG_LABELS: Record<string, string> = {
  vie_Latn: "Việt",
  eng_Latn: "Anh",
  tha_Thai: "Thái",
  lao_Laoo: "Lào",
  khm_Khmr: "Khmer",
  zho_Hans: "Trung",
  zho_Hant: "Đài Loan",
  jpn_Jpan: "Nhật",
  kor_Hang: "Hàn",
  fra_Latn: "Pháp",
  deu_Latn: "Đức",
  spa_Latn: "Tây Ban Nha",
  rus_Cyrl: "Nga",
  ind_Latn: "Indo",
  mya_Mymr: "Myanmar",
  auto: "Tự nhận",
};

export function shortLangName(code: string, fallback?: string): string {
  if (SHORT_LANG_LABELS[code]) return SHORT_LANG_LABELS[code];
  if (fallback) {
    return fallback.replace(/^Tiếng\s+/i, "").replace(/\s*\([^)]*\)/g, "").trim();
  }
  return code;
}

export function shortEngineName(engine: string): string {
  if (!engine) return "";
  if (engine === "soniox-cloud") return "Soniox Cloud";
  if (engine.startsWith("soniox:")) return `Soniox ${engine.slice(7).replace(/^stt-rt-/, "")}`;
  if (engine === "google-cloud") return "Google API";
  if (engine === "azure-translator") return "Azure API";
  if (engine.startsWith("gemini:")) return "Gemini API";
  if (engine.includes("envit5")) return "EnViT5";
  if (engine.includes("hy-mt2")) return "Hy-MT2";
  if (engine.includes("600m")) return "NLLB-600M";
  if (engine.includes("1.3b")) return "NLLB-1.3B";
  if (engine.includes("opus")) return "Opus";
  if (engine.includes("fast-online")) return "Online";
  if (engine.startsWith("llm:")) return engine.slice(4);
  return engine;
}

export function isTauriEnv(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

const NLLB_TO_BCP47: Record<string, string> = {
  eng_Latn: "en-US",
  vie_Latn: "vi-VN",
  zho_Hans: "zh-CN",
  zho_Hant: "zh-TW",
  jpn_Jpan: "ja-JP",
  kor_Hang: "ko-KR",
  fra_Latn: "fr-FR",
  deu_Latn: "de-DE",
  spa_Latn: "es-ES",
  rus_Cyrl: "ru-RU",
  tha_Thai: "th-TH",
  lao_Laoo: "lo-LA",
  khm_Khmr: "km-KH",
  ita_Latn: "it-IT",
  por_Latn: "pt-BR",
  ind_Latn: "id-ID",
  arb_Arab: "ar-SA",
  hin_Deva: "hi-IN",
  nld_Latn: "nl-NL",
  pol_Latn: "pl-PL",
  tur_Latn: "tr-TR",
  ukr_Cyrl: "uk-UA",
  ces_Latn: "cs-CZ",
  swe_Latn: "sv-SE",
};

export function speakOfflineTTS(text: string, nllbCode: string): void {
  if (!text.trim() || typeof window === "undefined" || !("speechSynthesis" in window)) {
    return;
  }
  try {
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(text);
    const bcp47 = NLLB_TO_BCP47[nllbCode] || nllbCode.slice(0, 2);
    utter.lang = bcp47;
    const voices = window.speechSynthesis.getVoices();
    const match =
      voices.find((v) => v.lang.toLowerCase() === bcp47.toLowerCase()) ||
      voices.find((v) => v.lang.toLowerCase().startsWith(bcp47.slice(0, 2).toLowerCase()));
    if (match) {
      utter.voice = match;
    }
    utter.rate = 1.05;
    window.speechSynthesis.speak(utter);
  } catch {
    // Ignore speech synthesis errors
  }
}
