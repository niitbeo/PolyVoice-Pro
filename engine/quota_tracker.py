"""
Local Cloud Translation API Config & Monthly Quota Guard for VOICERECORE.

Supports official APIs:
1. Google Cloud Translation API v2 (`google`) — 500,000 free chars/month (default safe cap: 480,000)
2. Microsoft Azure AI Translator F0 (`azure`) — 2,000,000 free chars/month (default safe cap: 1,950,000)
3. Google Gemini API (`gemini`) — ultra-low-cost context+glossary MT (default cap: 5,000,000)

API keys are stored ONLY on the local machine (`~/.cache/voicerecore/cloud_config.json`, 0600 on POSIX)
and are NEVER sent back in plaintext over HTTP GET responses (only masked previews).
When monthly usage reaches `hard_cap` (or network fails), VOICERECORE seamlessly falls back
to its 100% offline SOTA engines (`EnViT5` / `Hy-MT2-1.8B` / `NLLB-200-600M`).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

CACHE_DIR = Path(os.environ.get("VOICERECORE_CACHE_DIR", Path.home() / ".cache" / "voicerecore"))
DEFAULT_CONFIG_PATH = CACHE_DIR / "cloud_config.json"
DEFAULT_QUOTA_PATH = CACHE_DIR / "cloud_quota.json"

PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "soniox": {
        "name": "Soniox Real-Time (Nghe + Dịch 60 NN)",
        "free_tier": 0,
        "monthly_limit": 36_000,  # 10 hours of audio (36,000 seconds; ~$1.80)
        "api_key": "",
        "model": "stt-rt-v6",
        "unit": "seconds",
    },
    "google": {
        "name": "Google Cloud Translation",
        "free_tier": 500_000,
        "monthly_limit": 480_000,
        "api_key": "",
        "unit": "chars",
    },
    "azure": {
        "name": "Microsoft Azure Translator (Free F0)",
        "free_tier": 2_000_000,
        "monthly_limit": 1_950_000,
        "api_key": "",
        "region": "southeastasia",
        "endpoint": "https://api.cognitive.microsofttranslator.com",
        "unit": "chars",
    },
    "gemini": {
        "name": "Google Gemini API (Ngữ cảnh + Từ điển)",
        "free_tier": 0,
        "monthly_limit": 5_000_000,
        "api_key": "",
        "model": "gemini-2.5-flash",
        "unit": "chars",
    },
}


def _current_month() -> str:
    return time.strftime("%Y-%m")


def _mask_key(key: str) -> str:
    k = (key or "").strip()
    if not k:
        return ""
    if len(k) <= 8:
        return "•" * len(k)
    return f"{k[:4]}••••{k[-4:]}"


class CloudQuotaManager:
    def __init__(
        self,
        config_path: Path | None = None,
        quota_path: Path | None = None,
    ) -> None:
        self.config_path = Path(config_path or DEFAULT_CONFIG_PATH)
        self.quota_path = Path(quota_path or DEFAULT_QUOTA_PATH)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.quota_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._config: dict[str, Any] = self._load_config()
        self._quota: dict[str, Any] = self._load_quota()

    def _default_config(self) -> dict[str, Any]:
        return {
            "provider": "soniox",
            "hard_cap_enabled": True,
            "draft_offline": True,
            "providers": {
                k: dict(v) for k, v in PROVIDER_DEFAULTS.items()
            },
        }

    def _load_config(self) -> dict[str, Any]:
        base = self._default_config()
        if self.config_path.exists():
            try:
                raw = json.loads(self.config_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    if raw.get("provider") in PROVIDER_DEFAULTS:
                        base["provider"] = raw["provider"]
                    if isinstance(raw.get("hard_cap_enabled"), bool):
                        base["hard_cap_enabled"] = raw["hard_cap_enabled"]
                    if isinstance(raw.get("draft_offline"), bool):
                        base["draft_offline"] = raw["draft_offline"]
                    raw_provs = raw.get("providers") or {}
                    for pid in PROVIDER_DEFAULTS:
                        if isinstance(raw_provs.get(pid), dict):
                            base["providers"][pid].update(raw_provs[pid])
                    # Auto-migrate deprecated Gemini model names so saved user configs work immediately
                    if base["providers"]["gemini"].get("model") in (
                        "gemini-2.5-flash-lite",
                        "gemini-2.0-flash",
                        "gemini-1.5-flash",
                    ):
                        base["providers"]["gemini"]["model"] = "gemini-2.5-flash"
            except Exception:
                pass
        return base

    @staticmethod
    def _atomic_write_json(path: Path, data: dict[str, Any], *, private: bool = False) -> None:
        tmp_path: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(data, ensure_ascii=False, indent=2)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as tf:
                tmp_path = Path(tf.name)
                if private and os.name != "nt":
                    try:
                        os.chmod(tmp_path, 0o600)
                    except Exception:
                        pass
                tf.write(payload)
                tf.flush()
                os.fsync(tf.fileno())
            os.replace(tmp_path, path)
            if private and os.name != "nt":
                try:
                    os.chmod(path, 0o600)
                except Exception:
                    pass
        except Exception:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    def _save_config_unlocked(self) -> None:
        self._atomic_write_json(self.config_path, self._config, private=True)

    def _load_quota(self) -> dict[str, Any]:
        month = _current_month()
        default = {"month": month, "usage": {k: 0 for k in PROVIDER_DEFAULTS}}
        if self.quota_path.exists():
            try:
                raw = json.loads(self.quota_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and raw.get("month") == month:
                    usage = raw.get("usage") or {}
                    for k in PROVIDER_DEFAULTS:
                        default["usage"][k] = int(usage.get(k, 0))
                    return default
            except Exception:
                pass
        return default

    def _ensure_month_unlocked(self) -> None:
        m = _current_month()
        if self._quota.get("month") != m:
            self._quota = {"month": m, "usage": {k: 0 for k in PROVIDER_DEFAULTS}}
            self._save_quota_unlocked()

    def _save_quota_unlocked(self) -> None:
        self._atomic_write_json(self.quota_path, self._quota, private=False)

    def get_public_status(self) -> dict[str, Any]:
        """Return safe status for UI (keys are masked, never plaintext)."""
        with self._lock:
            self._ensure_month_unlocked()
            provs_out: dict[str, Any] = {}
            for pid, pcfg in self._config["providers"].items():
                used = int(self._quota["usage"].get(pid, 0))
                limit = max(1, int(pcfg.get("monthly_limit", PROVIDER_DEFAULTS[pid]["monthly_limit"])))
                key = str(pcfg.get("api_key", "")).strip()
                provs_out[pid] = {
                    "id": pid,
                    "name": PROVIDER_DEFAULTS[pid]["name"],
                    "free_tier": PROVIDER_DEFAULTS[pid]["free_tier"],
                    "unit": PROVIDER_DEFAULTS[pid].get("unit", "chars"),
                    "has_key": bool(key),
                    "key_preview": _mask_key(key),
                    "monthly_limit": limit,
                    "used_chars": used,
                    "remaining_chars": max(0, limit - used),
                    "pct_used": round(min(100.0, (used / limit) * 100.0), 1),
                    "cap_reached": bool(self._config["hard_cap_enabled"] and used >= limit),
                    "region": str(pcfg.get("region", "")),
                    "endpoint": str(pcfg.get("endpoint", "")),
                    "model": str(pcfg.get("model", "")),
                }
            active_pid = self._config["provider"]
            return {
                "provider": active_pid,
                "month": self._quota["month"],
                "hard_cap_enabled": bool(self._config["hard_cap_enabled"]),
                "draft_offline": bool(self._config["draft_offline"]),
                "active_ready": bool(provs_out[active_pid]["has_key"]),
                "active_cap_reached": bool(provs_out[active_pid]["cap_reached"]),
                "providers": provs_out,
                "config_path": str(self.config_path),
            }

    def get_credentials(self, provider: str | None = None) -> dict[str, Any]:
        """Internal backend accessor returning real credentials for making API calls."""
        with self._lock:
            self._ensure_month_unlocked()
            pid = provider if provider in PROVIDER_DEFAULTS else self._config["provider"]
            pcfg = dict(self._config["providers"][pid])
            pcfg["id"] = pid
            pcfg["used_chars"] = int(self._quota["usage"].get(pid, 0))
            pcfg["hard_cap_enabled"] = bool(self._config["hard_cap_enabled"])
            pcfg["draft_offline"] = bool(self._config["draft_offline"])
            return pcfg

    def update(self, patch: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._ensure_month_unlocked()
            if patch.get("provider") in PROVIDER_DEFAULTS:
                self._config["provider"] = patch["provider"]
            if isinstance(patch.get("hard_cap_enabled"), bool):
                self._config["hard_cap_enabled"] = patch["hard_cap_enabled"]
            if isinstance(patch.get("draft_offline"), bool):
                self._config["draft_offline"] = patch["draft_offline"]

            # Allow updating a specific provider's settings cleanly
            target_pid = patch.get("target_provider") or self._config["provider"]
            if target_pid in PROVIDER_DEFAULTS:
                pcfg = self._config["providers"][target_pid]
                if "api_key" in patch and patch["api_key"] is not None:
                    # Only overwrite when user actually submits a new string (empty string clears key)
                    pcfg["api_key"] = str(patch["api_key"]).strip()
                if "monthly_limit" in patch and patch["monthly_limit"] is not None:
                    min_limit = 60 if target_pid == "soniox" else 1000
                    pcfg["monthly_limit"] = max(min_limit, int(patch["monthly_limit"]))
                if "region" in patch and patch["region"] is not None:
                    pcfg["region"] = str(patch["region"]).strip()
                if "endpoint" in patch and patch["endpoint"] is not None:
                    ep = str(patch["endpoint"]).strip().rstrip("/")
                    if ep:
                        pcfg["endpoint"] = ep
                if "model" in patch and patch["model"] is not None:
                    m = str(patch["model"]).strip()
                    if m:
                        pcfg["model"] = m
            self._save_config_unlocked()
        return self.get_public_status()

    def can_spend(self, provider: str, n_chars: int) -> bool:
        with self._lock:
            self._ensure_month_unlocked()
            if provider not in PROVIDER_DEFAULTS:
                return False
            pcfg = self._config["providers"][provider]
            if not str(pcfg.get("api_key", "")).strip():
                return False
            if not self._config.get("hard_cap_enabled", True):
                return True
            used = int(self._quota["usage"].get(provider, 0))
            limit = max(1, int(pcfg.get("monthly_limit", PROVIDER_DEFAULTS[provider]["monthly_limit"])))
            return (used + max(0, int(n_chars))) <= limit

    def try_reserve(self, provider: str, n_chars: int) -> bool:
        """Atomically verify quota and reserve `n_chars` under a single lock (prevents concurrent TOCTOU overshoot)."""
        chars = max(0, int(n_chars))
        with self._lock:
            self._ensure_month_unlocked()
            if provider not in PROVIDER_DEFAULTS:
                return False
            pcfg = self._config["providers"][provider]
            if not str(pcfg.get("api_key", "")).strip():
                return False
            used = int(self._quota["usage"].get(provider, 0))
            if self._config.get("hard_cap_enabled", True):
                limit = max(1, int(pcfg.get("monthly_limit", PROVIDER_DEFAULTS[provider]["monthly_limit"])))
                if (used + chars) > limit:
                    return False
            if chars > 0:
                self._quota["usage"][provider] = used + chars
                self._save_quota_unlocked()
            return True

    def refund_usage(self, provider: str, n_chars: int) -> int:
        """Refund previously reserved chars (e.g. on cache hit or failed API call)."""
        chars = max(0, int(n_chars))
        with self._lock:
            self._ensure_month_unlocked()
            if provider not in PROVIDER_DEFAULTS or chars <= 0:
                return 0
            cur = int(self._quota["usage"].get(provider, 0))
            nxt = max(0, cur - chars)
            self._quota["usage"][provider] = nxt
            self._save_quota_unlocked()
            return nxt

    def record_usage(self, provider: str, n_chars: int) -> int:
        with self._lock:
            self._ensure_month_unlocked()
            if provider not in PROVIDER_DEFAULTS or n_chars <= 0:
                return 0
            cur = int(self._quota["usage"].get(provider, 0))
            nxt = cur + int(n_chars)
            self._quota["usage"][provider] = nxt
            self._save_quota_unlocked()
            return nxt

    def reset_usage(self, provider: str | None = None) -> dict[str, Any]:
        with self._lock:
            self._ensure_month_unlocked()
            if provider in PROVIDER_DEFAULTS:
                self._quota["usage"][provider] = 0
            else:
                for k in PROVIDER_DEFAULTS:
                    self._quota["usage"][k] = 0
            self._save_quota_unlocked()
        return self.get_public_status()
