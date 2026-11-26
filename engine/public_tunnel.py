"""
Isolated Public 4G/5G Conference QR Tunnel Manager (Cloudflare Tunnel / `cloudflared`).

Provides 100% free ($0), zero-server, zero-port-forwarding public HTTPS sharing
for DAU Voice Pro conference subtitles:
1. Quick Tunnel mode (`*.trycloudflare.com`): zero-account, 1-click ephemeral HTTPS URL.
2. Named Tunnel mode (optional custom domain, e.g. `https://live.dau.edu.vn`):
   permanent URL via Cloudflare Zero Trust Tunnel Token (saved locally with `0600`).

CRITICAL SECURITY ARCHITECTURE:
- Never points `cloudflared` at the main admin port (`18765`).
- Instead spins up a dedicated, strictly isolated FastAPI/Uvicorn ASGI server on
  `127.0.0.1:18768` (or a free loopback port in Quick mode) running on the SAME
  `asyncio` event loop as `EngineRuntime` (avoiding cross-loop WebSocket issues).
- The public ASGI app mounts ONLY:
    - `GET /` -> 302 redirect to `/audience`
    - `GET /audience` -> read-only audience HTML
    - `GET /favicon.ico` -> 204 No Content
    - `WS /ws/audience` -> read-only subtitle stream (with optional 4-digit PIN & client cap)
- ZERO `/api/*`, ZERO `/ws`, ZERO `/obs`, ZERO admin routes exist on the public port.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import json
import logging
import os
import platform
import re
import secrets
import shutil
import socket
import subprocess
import tarfile
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlencode, urlparse

import httpx
import segno
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from audience_web import AUDIENCE_HTML

logger = logging.getLogger("voicerecore.public_tunnel")

DEFAULT_PUBLIC_PORT = 18768
MAX_PUBLIC_AUDIENCE_CLIENTS = 200

_TRYCF_RE = re.compile(r"https://(?!api\.)([a-z0-9-]+)\.trycloudflare\.com", re.IGNORECASE)
_NAMED_READY_HINTS = (
    "registered tunnel connection",
    "connection registered",
)

_CLOUDFLARED_RELEASES_BASE = "https://github.com/cloudflare/cloudflared/releases/latest/download"


def _default_cache_dir() -> Path:
    return Path(os.environ.get("VOICERECORE_CACHE_DIR", str(Path.home() / ".cache" / "voicerecore")))


def _generate_pin() -> str:
    return f"{secrets.randbelow(10000):04d}"


def _mask_secret(val: str) -> str:
    s = (val or "").strip()
    if not s:
        return ""
    if len(s) <= 8:
        return "•" * len(s)
    return f"{s[:4]}••••{s[-4:]}"


def normalize_custom_domain(raw: str) -> str:
    """Normalize a user-provided custom domain into `https://host[:port]`."""
    s = (raw or "").strip()
    if not s:
        return ""
    if "://" not in s:
        s = f"https://{s}"
    parsed = urlparse(s)
    if parsed.scheme not in ("https", "http") or not parsed.netloc:
        raise ValueError("Tên miền không hợp lệ. Ví dụ đúng: https://live.dau.edu.vn")
    # Prefer https for Cloudflare tunnels unless explicitly localhost
    scheme = "https" if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1") else parsed.scheme
    return f"{scheme}://{parsed.netloc}".rstrip("/")


def get_cloudflared_download_spec() -> tuple[str, bool, str]:
    """Return `(download_url, is_tgz, binary_filename)` for the current OS/arch."""
    sys_name = platform.system().lower()
    machine = platform.machine().lower()
    is_arm = machine in ("arm64", "aarch64")

    if sys_name == "darwin":
        asset = "cloudflared-darwin-arm64.tgz" if is_arm else "cloudflared-darwin-amd64.tgz"
        return (f"{_CLOUDFLARED_RELEASES_BASE}/{asset}", True, "cloudflared")
    if sys_name == "windows":
        return (f"{_CLOUDFLARED_RELEASES_BASE}/cloudflared-windows-amd64.exe", False, "cloudflared.exe")
    if sys_name == "linux":
        asset = "cloudflared-linux-arm64" if is_arm else "cloudflared-linux-amd64"
        return (f"{_CLOUDFLARED_RELEASES_BASE}/{asset}", False, "cloudflared")
    raise RuntimeError(f"Hệ điều hành chưa hỗ trợ tải tự động cloudflared: {platform.system()} ({machine})")


def find_cloudflared_binary(cache_dir: Optional[Path] = None) -> Optional[Path]:
    """Locate an existing `cloudflared` executable (app cache -> PATH -> standard Homebrew dirs)."""
    cdir = cache_dir or _default_cache_dir()
    bin_name = "cloudflared.exe" if platform.system().lower() == "windows" else "cloudflared"

    cached = cdir / "bin" / bin_name
    if cached.is_file() and cached.stat().st_size > 100_000:
        return cached

    found = shutil.which("cloudflared")
    if found:
        p = Path(found)
        if p.is_file():
            return p

    if platform.system().lower() == "darwin":
        for extra in (
            Path("/opt/homebrew/bin/cloudflared"),
            Path("/usr/local/bin/cloudflared"),
        ):
            if extra.is_file():
                return extra

    return None


def download_cloudflared_binary(
    cache_dir: Optional[Path] = None,
    on_progress: Optional[Callable[[float], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Path:
    """Download official Cloudflare `cloudflared` release binary into `<cache_dir>/bin/`."""
    cdir = cache_dir or _default_cache_dir()
    bin_dir = cdir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    url, is_tgz, bin_name = get_cloudflared_download_spec()
    dest = bin_dir / bin_name

    tmp_download: Optional[Path] = None
    tmp_bin: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(dir=str(bin_dir), prefix=".cf_dl_", delete=False) as tf:
            tmp_download = Path(tf.name)
            with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, read=120.0)) as client:
                with client.stream("GET", url, headers={"User-Agent": "DAU-Voice-Pro/1.1"}) as resp:
                    resp.raise_for_status()
                    total = int(resp.headers.get("content-length") or 0)
                    downloaded = 0
                    last_emit = 0.0
                    if on_progress:
                        on_progress(1.0)
                    for chunk in resp.iter_bytes(chunk_size=131072):
                        if cancel_event is not None and cancel_event.is_set():
                            raise RuntimeError("Đã hủy tải cloudflared.")
                        if not chunk:
                            continue
                        tf.write(chunk)
                        downloaded += len(chunk)
                        if on_progress and total > 0:
                            now = time.monotonic()
                            if now - last_emit >= 0.18 or downloaded >= total:
                                pct = min(98.0, (downloaded / total) * 98.0)
                                on_progress(round(pct, 1))
                                last_emit = now

        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Đã hủy tải cloudflared.")

        if is_tgz:
            with tempfile.NamedTemporaryFile(dir=str(bin_dir), prefix=".cf_bin_", delete=False) as bf:
                tmp_bin = Path(bf.name)
                with tarfile.open(tmp_download, mode="r:gz") as tar:
                    target_member = None
                    for member in tar.getmembers():
                        if member.isfile() and Path(member.name).name == bin_name:
                            target_member = member
                            break
                    if target_member is None:
                        raise RuntimeError("Gói cloudflared tải về không chứa file thực thi hợp lệ.")
                    extracted = tar.extractfile(target_member)
                    if extracted is None:
                        raise RuntimeError("Không thể giải nén file thực thi cloudflared.")
                    shutil.copyfileobj(extracted, bf)
            final_tmp = tmp_bin
        else:
            final_tmp = tmp_download
            tmp_download = None

        if platform.system().lower() != "windows":
            os.chmod(final_tmp, 0o755)
        if platform.system().lower() == "darwin":
            try:
                subprocess.run(
                    ["xattr", "-d", "com.apple.quarantine", str(final_tmp)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=5,
                )
            except Exception:
                pass

        os.replace(final_tmp, dest)
        tmp_bin = None
        if on_progress:
            on_progress(100.0)
        return dest
    finally:
        for p in (tmp_download, tmp_bin):
            if p is not None:
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass


def _wait_dns_ready_via_doh(hostname: str, cancel_ev: threading.Event, max_wait_s: float = 8.0) -> bool:
    """Poll Cloudflare DoH (never poisons local OS NXDOMAIN cache!) until the fresh
    `*.trycloudflare.com` hostname has live DNS records on Cloudflare Edge."""
    deadline = time.monotonic() + max_wait_s
    # Brief initial pause before first lookup
    if cancel_ev.wait(1.0):
        return False
    while time.monotonic() < deadline and not cancel_ev.is_set():
        try:
            with httpx.Client(timeout=2.5) as hc:
                r = hc.get(
                    "https://1.1.1.1/dns-query",
                    params={"name": hostname, "type": "A"},
                    headers={"Accept": "application/dns-json"},
                )
                if r.status_code == 200:
                    data = r.json()
                    if data.get("Status") == 0 and data.get("Answer"):
                        return True
        except Exception:
            # If DoH is blocked by a local firewall, fall back to a short settle sleep
            if cancel_ev.wait(2.0):
                return False
            return True
        if cancel_ev.wait(0.8):
            return False
    return True


def _pick_loopback_port(preferred: int = DEFAULT_PUBLIC_PORT, strict: bool = False) -> int:
    """Return `preferred` if bindable on 127.0.0.1, otherwise an OS-assigned free loopback port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", preferred))
        return preferred
    except OSError as exc:
        if strict:
            raise RuntimeError(
                f"Cổng cô lập 127.0.0.1:{preferred} đang bị ứng dụng khác chiếm dụng. "
                "Vui lòng giải phóng cổng này trước khi bật Tên miền cố định."
            ) from exc
    finally:
        sock.close()

    sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock2.bind(("127.0.0.1", 0))
        return int(sock2.getsockname()[1])
    finally:
        sock2.close()


def build_isolated_public_app(
    runtime: Any,
    pin_verifier: Callable[[str], bool],
    is_pin_enabled: Callable[[], bool],
    max_clients: int = MAX_PUBLIC_AUDIENCE_CLIENTS,
) -> FastAPI:
    """Construct a strictly isolated ASGI app that exposes ONLY `/`, `/audience`, `/favicon.ico`, and `/ws/audience`.
    Zero `/api/*`, `/ws`, or `/obs` endpoints exist on this app."""
    pub_app = FastAPI(
        title="DAU Voice Pro Public Audience",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @pub_app.get("/")
    def pub_root(pin: str = ""):
        qs = f"?{urlencode({'pin': pin.strip()})}" if pin and pin.strip() else ""
        return RedirectResponse(url=f"/audience{qs}", status_code=302)

    @pub_app.get("/audience", response_class=HTMLResponse)
    def pub_audience():
        return HTMLResponse(
            AUDIENCE_HTML,
            headers={
                "Cache-Control": "no-store, max-age=0",
                "X-Robots-Tag": "noindex, nofollow",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @pub_app.get("/favicon.ico")
    def pub_favicon():
        return Response(status_code=204)

    @pub_app.websocket("/ws/audience")
    async def pub_ws_audience(ws: WebSocket, lang: str = "follow", pin: str = ""):
        await ws.accept()
        if is_pin_enabled() and not pin_verifier(pin):
            try:
                await ws.send_json(
                    {
                        "type": "pin_required",
                        "data": {"invalid": bool((pin or "").strip())},
                    }
                )
            except Exception:
                pass
            try:
                await ws.close(code=4403, reason="PIN required")
            except Exception:
                pass
            return

        if runtime.get_audience_summary().get("connected_count", 0) >= max_clients:
            try:
                await ws.send_json(
                    {
                        "type": "error",
                        "data": {"message": f"Phòng hội nghị đã đạt tối đa {max_clients} thiết bị cùng lúc."},
                    }
                )
            except Exception:
                pass
            try:
                await ws.close(code=4429, reason="Room capacity reached")
            except Exception:
                pass
            return

        try:
            await runtime.register_audience_ws(ws, lang=lang)
            while True:
                msg = await ws.receive_text()
                if msg == "ping":
                    await ws.send_json({"type": "pong"})
        except (WebSocketDisconnect, Exception):
            pass
        finally:
            await runtime.unregister_audience_ws(ws)

    return pub_app


class PublicTunnelManager:
    """Manages the isolated localhost audience port + `cloudflared` tunnel lifecycle."""

    def __init__(self, runtime: Any, cache_dir: Optional[Path] = None) -> None:
        self._runtime = runtime
        self._cache_dir = cache_dir or _default_cache_dir()
        self._config_path = self._cache_dir / "tunnel_config.json"

        self._lock = threading.RLock()
        self._generation: int = 0
        self._cancel_event: Optional[threading.Event] = None
        self._worker_thread: Optional[threading.Thread] = None

        self._status: str = "idle"  # idle | downloading | starting | ready | error
        self._mode: str = "quick"  # quick | named
        self._public_base_url: str = ""
        self._public_port: int = DEFAULT_PUBLIC_PORT
        self._error: str = ""
        self._status_message: str = ""
        self._download_percent: float = 0.0

        # Optional security & persistent named-tunnel configuration
        self._pin_enabled: bool = False
        self._room_pin: str = _generate_pin()
        self._named_domain: str = ""
        self._named_token: str = ""

        # Active runtime handles
        self._proc: Optional[subprocess.Popen] = None
        self._pub_server: Optional[uvicorn.Server] = None
        self._pub_future: Optional[concurrent.futures.Future] = None

        self._load_config()
        atexit.register(self.stop)

    def _load_config(self) -> None:
        if not self._config_path.is_file():
            return
        try:
            raw = json.loads(self._config_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._named_domain = str(raw.get("named_domain") or "").strip()
                self._named_token = str(raw.get("named_token") or "").strip()
                self._pin_enabled = bool(raw.get("pin_enabled", False))
                saved_pin = str(raw.get("room_pin") or "").strip()
                if re.fullmatch(r"\d{4}", saved_pin):
                    self._room_pin = saved_pin
        except Exception as exc:
            logger.warning("Failed to load tunnel_config.json: %s", exc)

    def _save_config(self) -> None:
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "named_domain": self._named_domain,
                "named_token": self._named_token,
                "pin_enabled": self._pin_enabled,
                "room_pin": self._room_pin,
            }
            tmp = self._config_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            if platform.system().lower() != "windows":
                os.chmod(tmp, 0o600)
            os.replace(tmp, self._config_path)
            if platform.system().lower() != "windows":
                os.chmod(self._config_path, 0o600)
        except Exception as exc:
            logger.warning("Failed to save tunnel_config.json: %s", exc)

    @property
    def pin_enabled(self) -> bool:
        with self._lock:
            return self._pin_enabled

    @property
    def room_pin(self) -> str:
        with self._lock:
            return self._room_pin

    def verify_pin(self, candidate: str) -> bool:
        with self._lock:
            if not self._pin_enabled:
                return True
            cand = (candidate or "").strip()
            if not cand:
                return False
            return secrets.compare_digest(cand, self._room_pin)

    def _build_audience_url_for_base(self, base_url: str, include_pin: bool = True) -> str:
        if not base_url:
            return ""
        url = f"{base_url.rstrip('/')}/audience"
        if include_pin and self._pin_enabled and self._room_pin:
            url = f"{url}?pin={self._room_pin}"
        return url

    def _notify_ui(self) -> None:
        try:
            info = self._runtime.get_audience_info()
            self._runtime.broadcast_threadsafe({"type": "audience_info", "data": info})
        except Exception:
            pass

    def get_public_status(self) -> dict[str, Any]:
        with self._lock:
            installed = find_cloudflared_binary(self._cache_dir) is not None
            ready = self._status == "ready" and bool(self._public_base_url)
            public_url = self._build_audience_url_for_base(self._public_base_url, include_pin=True) if ready else ""
            public_url_no_pin = (
                self._build_audience_url_for_base(self._public_base_url, include_pin=False) if ready else ""
            )
            public_qr_svg = ""
            if public_url:
                try:
                    public_qr_svg = segno.make(public_url).svg_inline(
                        omitsize=True, border=0, dark="#0f172a", light="#ffffff"
                    )
                except Exception:
                    public_qr_svg = ""
            return {
                "public_active": self._status in ("downloading", "starting", "ready"),
                "public_ready": ready,
                "public_status": self._status,
                "public_mode": self._mode,
                "public_port": self._public_port,
                "public_url": public_url,
                "public_url_display": public_url_no_pin,
                "public_qr_svg": public_qr_svg,
                "public_error": self._error,
                "public_status_message": self._status_message,
                "download_percent": self._download_percent,
                "cloudflared_installed": installed,
                "pin_enabled": self._pin_enabled,
                "room_pin": self._room_pin,
                "named_configured": bool(self._named_domain and self._named_token),
                "named_domain": self._named_domain,
                "named_token_masked": _mask_secret(self._named_token),
            }

    def update_pin(
        self,
        enabled: Optional[bool] = None,
        regenerate: bool = False,
        custom_pin: Optional[str] = None,
    ) -> dict[str, Any]:
        with self._lock:
            if enabled is not None:
                self._pin_enabled = bool(enabled)
            if custom_pin is not None and custom_pin.strip():
                pin_clean = custom_pin.strip()
                if not re.fullmatch(r"\d{4}", pin_clean):
                    raise ValueError("Mã PIN phải gồm đúng 4 chữ số (0000–9999).")
                self._room_pin = pin_clean
            elif regenerate or not re.fullmatch(r"\d{4}", self._room_pin):
                self._room_pin = _generate_pin()
            self._save_config()
        self._notify_ui()
        return self._runtime.get_audience_info()

    def clear_named_config(self) -> dict[str, Any]:
        with self._lock:
            self._named_domain = ""
            self._named_token = ""
            self._save_config()
        self._notify_ui()
        return self._runtime.get_audience_info()

    def start(
        self,
        mode: str = "quick",
        pin_enabled: Optional[bool] = None,
        custom_domain: Optional[str] = None,
        tunnel_token: Optional[str] = None,
    ) -> dict[str, Any]:
        norm_mode = "named" if (mode or "").strip().lower() == "named" else "quick"

        # Validate & persist optional parameters before stopping old tunnel
        with self._lock:
            if pin_enabled is not None:
                self._pin_enabled = bool(pin_enabled)
                if not re.fullmatch(r"\d{4}", self._room_pin):
                    self._room_pin = _generate_pin()

            eff_domain = self._named_domain
            eff_token = self._named_token
            if norm_mode == "named":
                if custom_domain is not None and custom_domain.strip():
                    eff_domain = normalize_custom_domain(custom_domain)
                if tunnel_token is not None and tunnel_token.strip():
                    eff_token = tunnel_token.strip()
                if not eff_domain:
                    raise ValueError("Vui lòng nhập tên miền cố định của trường (VD: https://live.dau.edu.vn).")
                if not eff_token:
                    raise ValueError("Vui lòng dán Cloudflare Tunnel Token để kết nối tên miền cố định.")
                self._named_domain = eff_domain
                self._named_token = eff_token
            self._save_config()

        # Tear down any previous run cleanly first
        self.stop(notify=False)

        with self._lock:
            self._generation += 1
            gen = self._generation
            cancel_ev = threading.Event()
            self._cancel_event = cancel_ev
            self._mode = norm_mode
            self._status = "starting"
            self._public_base_url = ""
            self._error = ""
            self._download_percent = 0.0
            self._status_message = (
                "Đang chuẩn bị đường hầm tên miền cố định..."
                if norm_mode == "named"
                else "Đang khởi tạo tên miền HTTPS miễn phí..."
            )
            t = threading.Thread(
                target=self._run_worker,
                args=(gen, norm_mode, eff_domain, eff_token, cancel_ev),
                daemon=True,
                name=f"public-tunnel-{gen}",
            )
            self._worker_thread = t
            t.start()

        self._notify_ui()
        return self._runtime.get_audience_info()

    def _start_isolated_server(self, port: int) -> tuple[uvicorn.Server, concurrent.futures.Future]:
        loop = self._runtime.loop
        if loop is None or loop.is_closed():
            raise RuntimeError("Event loop máy chủ chưa sẵn sàng.")

        pub_app = build_isolated_public_app(
            runtime=self._runtime,
            pin_verifier=self.verify_pin,
            is_pin_enabled=lambda: self.pin_enabled,
        )
        config = uvicorn.Config(
            pub_app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            ws_ping_interval=25.0,
            ws_ping_timeout=25.0,
        )
        server = uvicorn.Server(config)
        # Never override main process SIGINT/SIGTERM handlers
        server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        fut = asyncio.run_coroutine_threadsafe(server.serve(), loop)

        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            if server.started:
                return server, fut
            if fut.done():
                exc = fut.exception()
                raise RuntimeError(f"Không thể mở cổng cô lập 127.0.0.1:{port}: {exc}")
            time.sleep(0.05)

        server.should_exit = True
        raise RuntimeError(f"Quá thời gian chờ mở cổng cô lập 127.0.0.1:{port}.")

    def _stop_handles(
        self,
        proc: Optional[subprocess.Popen],
        pub_server: Optional[uvicorn.Server],
        pub_future: Optional[concurrent.futures.Future],
    ) -> None:
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2.5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=1.5)
            except Exception:
                pass
        if pub_server is not None:
            try:
                pub_server.should_exit = True
            except Exception:
                pass
        if pub_future is not None:
            try:
                pub_future.result(timeout=3.0)
            except Exception:
                pass

    def stop(self, notify: bool = True) -> dict[str, Any]:
        with self._lock:
            self._generation += 1
            if self._cancel_event is not None:
                self._cancel_event.set()
            proc = self._proc
            pub_server = self._pub_server
            pub_future = self._pub_future
            self._proc = None
            self._pub_server = None
            self._pub_future = None
            self._status = "idle"
            self._public_base_url = ""
            self._error = ""
            self._status_message = ""
            self._download_percent = 0.0

        self._stop_handles(proc, pub_server, pub_future)

        if notify:
            self._notify_ui()
        return self._runtime.get_audience_info()

    def _is_current(self, gen: int, cancel_ev: threading.Event) -> bool:
        with self._lock:
            return self._generation == gen and not cancel_ev.is_set()

    def _fail(self, gen: int, message: str) -> None:
        with self._lock:
            if self._generation != gen:
                return
            proc = self._proc
            pub_server = self._pub_server
            pub_future = self._pub_future
            self._proc = None
            self._pub_server = None
            self._pub_future = None
            self._status = "error"
            self._public_base_url = ""
            self._error = message
            self._status_message = ""
        self._stop_handles(proc, pub_server, pub_future)
        self._notify_ui()

    def _run_worker(
        self,
        gen: int,
        mode: str,
        named_domain: str,
        named_token: str,
        cancel_ev: threading.Event,
    ) -> None:
        proc: Optional[subprocess.Popen] = None
        pub_server: Optional[uvicorn.Server] = None
        pub_future: Optional[concurrent.futures.Future] = None
        try:
            # 1. Locate or download official `cloudflared`
            bin_path = find_cloudflared_binary(self._cache_dir)
            if bin_path is None:
                with self._lock:
                    if self._generation != gen:
                        return
                    self._status = "downloading"
                    self._download_percent = 1.0
                    self._status_message = "Đang tải công cụ Cloudflare Tunnel chính chủ (chỉ 1 lần, ~18 MB)..."
                self._notify_ui()

                def _on_prog(pct: float) -> None:
                    with self._lock:
                        if self._generation != gen:
                            return
                        self._download_percent = pct
                        self._status_message = f"Đang tải Cloudflare Tunnel... {pct:.0f}%"
                    self._notify_ui()

                bin_path = download_cloudflared_binary(
                    cache_dir=self._cache_dir,
                    on_progress=_on_prog,
                    cancel_event=cancel_ev,
                )

            if not self._is_current(gen, cancel_ev):
                return

            # 2. Spin up strictly isolated public-only ASGI server on 127.0.0.1:<public_port>
            with self._lock:
                if self._generation != gen:
                    return
                self._status = "starting"
                self._download_percent = 100.0
                self._status_message = "Đang mở cổng cô lập bảo mật & kết nối Cloudflare..."
            self._notify_ui()

            public_port = _pick_loopback_port(DEFAULT_PUBLIC_PORT, strict=(mode == "named"))
            pub_server, pub_future = self._start_isolated_server(public_port)

            with self._lock:
                if self._generation != gen or cancel_ev.is_set():
                    self._stop_handles(None, pub_server, pub_future)
                    return
                self._public_port = public_port
                self._pub_server = pub_server
                self._pub_future = pub_future

            # 3. Launch `cloudflared` outbound HTTP/2 tunnel (works reliably behind campus firewalls)
            if mode == "named":
                cmd = [
                    str(bin_path),
                    "tunnel",
                    "--no-autoupdate",
                    "--protocol",
                    "http2",
                    "run",
                    "--token",
                    named_token,
                ]
            else:
                cmd = [
                    str(bin_path),
                    "tunnel",
                    "--url",
                    f"http://127.0.0.1:{public_port}",
                    "--no-autoupdate",
                    "--protocol",
                    "http2",
                ]

            popen_kwargs: dict[str, Any] = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "bufsize": 1,
            }
            if platform.system().lower() == "windows":
                popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

            proc = subprocess.Popen(cmd, **popen_kwargs)
            with self._lock:
                if self._generation != gen or cancel_ev.is_set():
                    self._stop_handles(proc, pub_server, pub_future)
                    return
                self._proc = proc

            # 4. Read merged output stream to capture URL / readiness & monitor health
            ready_ev = threading.Event()
            recent_err_lines: list[str] = []
            pending_quick_url = ""
            tunnel_registered = False

            def _startup_watchdog() -> None:
                if not ready_ev.is_set() and self._is_current(gen, cancel_ev):
                    self._fail(
                        gen,
                        "Quá thời gian chờ kết nối Cloudflare (32 giây). Kiểm tra lại kết nối Internet hoặc tường lửa mạng.",
                    )

            watchdog = threading.Timer(32.0, _startup_watchdog)
            watchdog.daemon = True
            watchdog.start()

            def _finalize_ready(base_url: str, msg_text: str, check_dns: bool) -> None:
                if check_dns:
                    host = urlparse(base_url).hostname or ""
                    if host:
                        with self._lock:
                            if self._generation == gen:
                                self._status_message = "Đang kích hoạt DNS HTTPS toàn cầu..."
                        self._notify_ui()
                        _wait_dns_ready_via_doh(host, cancel_ev, max_wait_s=9.0)
                if not self._is_current(gen, cancel_ev):
                    return
                ready_ev.set()
                watchdog.cancel()
                with self._lock:
                    if self._generation != gen:
                        return
                    self._public_base_url = base_url
                    self._status = "ready"
                    self._error = ""
                    self._status_message = msg_text
                logger.info("Public Tunnel ready (%s): %s -> 127.0.0.1:%d", mode, base_url, public_port)
                self._notify_ui()

            try:
                assert proc.stdout is not None
                for raw_line in proc.stdout:
                    if not self._is_current(gen, cancel_ev):
                        break
                    line = raw_line.strip()
                    if not line:
                        continue
                    low = line.lower()
                    if "err" in low or "failed" in low or "unauthorized" in low or "invalid" in low:
                        # Redact token if ever echoed
                        safe_line = line.replace(named_token, "***") if named_token else line
                        recent_err_lines.append(safe_line)
                        if len(recent_err_lines) > 6:
                            recent_err_lines.pop(0)

                    if not ready_ev.is_set():
                        if mode == "quick":
                            if not pending_quick_url:
                                m = _TRYCF_RE.search(line)
                                if m:
                                    pending_quick_url = m.group(0).rstrip("/")
                                    with self._lock:
                                        if self._generation == gen:
                                            self._status_message = "Đã cấp tên miền — đang thiết lập kết nối Cloudflare..."
                                    self._notify_ui()
                            if any(hint in low for hint in _NAMED_READY_HINTS):
                                tunnel_registered = True
                            if pending_quick_url and tunnel_registered:
                                # Run DNS readiness check in background so we keep draining cloudflared stdout
                                target_url = pending_quick_url
                                pending_quick_url = ""
                                threading.Thread(
                                    target=_finalize_ready,
                                    args=(target_url, "Đang phát Công khai 4G/5G (HTTPS Miễn phí)", True),
                                    daemon=True,
                                    name=f"public-tunnel-dns-{gen}",
                                ).start()
                        else:
                            if any(hint in low for hint in _NAMED_READY_HINTS):
                                _finalize_ready(
                                    named_domain,
                                    f"Đang phát qua tên miền cố định ({named_domain})",
                                    False,
                                )
            finally:
                watchdog.cancel()

            is_ready = ready_ev.is_set()

            if not self._is_current(gen, cancel_ev):
                return

            rc = proc.wait(timeout=3.0) if proc.poll() is None else proc.returncode
            joined_err = " ".join(recent_err_lines).lower()
            if "token" in joined_err or "unauthorized" in joined_err or "401" in joined_err:
                msg = "Cloudflare Tunnel Token không hợp lệ hoặc đã hết hạn. Vui lòng kiểm tra lại Token."
            elif "no such host" in joined_err or "network is unreachable" in joined_err or "dial tcp" in joined_err:
                msg = "Không thể kết nối máy chủ Cloudflare. Vui lòng kiểm tra kết nối Internet của máy tính."
            elif not is_ready:
                msg = f"Không khởi tạo được đường hầm công khai (mã thoát {rc}). Kiểm tra Internet rồi thử lại."
            else:
                msg = f"Đường hầm công khai vừa bị ngắt kết nối (mã {rc}). Bấm Bật lại để khôi phục."
            self._fail(gen, msg)

        except Exception as exc:
            if not self._is_current(gen, cancel_ev):
                return
            err_txt = str(exc)
            if named_token:
                err_txt = err_txt.replace(named_token, "***")
            self._fail(gen, err_txt)
