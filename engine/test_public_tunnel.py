"""
Automated security & live E2E verification for Public 4G/5G Conference QR Tunnel.
Run: uv run --project engine python engine/test_public_tunnel.py
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import stat
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx
import websockets
from fastapi.testclient import TestClient

from public_tunnel import (
    PublicTunnelManager,
    build_isolated_public_app,
    find_cloudflared_binary,
    normalize_custom_domain,
)
import server as srv


class _MockRuntime:
    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self.loop = loop
        self.clients: dict = {}
        self.broadcasts: list[dict] = []

    def get_audience_summary(self) -> dict:
        return {"connected_count": len(self.clients), "active_languages": []}

    def get_audience_info(self) -> dict:
        return {"ok": True}

    def broadcast_threadsafe(self, payload: dict) -> None:
        self.broadcasts.append(payload)

    async def register_audience_ws(self, ws, lang: str = "follow") -> None:
        self.clients[ws] = lang
        await ws.send_json({"type": "status", "data": {"listening": False, "display_frozen": False}})

    async def unregister_audience_ws(self, ws) -> None:
        self.clients.pop(ws, None)


def test_domain_normalization_and_config_security() -> None:
    assert normalize_custom_domain("live.dau.edu.vn") == "https://live.dau.edu.vn"
    assert normalize_custom_domain("http://live.dau.edu.vn/") == "https://live.dau.edu.vn"
    assert normalize_custom_domain("https://sub.dau.edu.vn/audience") == "https://sub.dau.edu.vn"

    with tempfile.TemporaryDirectory() as td:
        cache = Path(td)
        rt = _MockRuntime()
        mgr = PublicTunnelManager(rt, cache_dir=cache)
        info = mgr.update_pin(enabled=True, custom_pin="4829")
        assert mgr.pin_enabled is True
        assert mgr.room_pin == "4829"
        assert mgr.verify_pin("4829") is True
        assert mgr.verify_pin("0000") is False
        assert mgr.verify_pin("") is False

        cfg_path = cache / "tunnel_config.json"
        assert cfg_path.is_file()
        if os.name != "nt":
            mode = stat.S_IMODE(cfg_path.stat().st_mode)
            assert mode == 0o600, f"Expected 0600, got {oct(mode)}"

        # Persist named token and verify it is never exposed raw in public status
        mgr._named_domain = "https://live.dau.edu.vn"
        mgr._named_token = "eyJhIjoiU0VDUkVUX1RPS0VOXzEyMzQ1Njc4OTAifQ=="
        mgr._save_config()
        st = mgr.get_public_status()
        assert st["named_configured"] is True
        assert st["named_domain"] == "https://live.dau.edu.vn"
        assert "SECRET_TOKEN" not in json.dumps(st)
        assert "••••" in st["named_token_masked"]
        mgr.stop(notify=False)
    print("[PASS] test_domain_normalization_and_config_security")


def test_isolated_public_app_routes_and_pin() -> None:
    rt = _MockRuntime()
    pin_state = {"enabled": False, "pin": "7391"}
    pub_app = build_isolated_public_app(
        runtime=rt,
        pin_verifier=lambda c: c.strip() == pin_state["pin"],
        is_pin_enabled=lambda: pin_state["enabled"],
    )
    client = TestClient(pub_app)

    # Root redirects to /audience (preserving PIN if given)
    r_root = client.get("/?pin=7391", follow_redirects=False)
    assert r_root.status_code == 302
    assert r_root.headers["location"] == "/audience?pin=7391"

    # /audience serves read-only HTML with noindex + heartbeat + PIN gate + S1/S2 badge support
    r_aud = client.get("/audience")
    assert r_aud.status_code == 200
    assert "noindex" in r_aud.headers.get("x-robots-tag", "")
    assert "pinOverlay" in r_aud.text
    assert 'sock.send("ping")' in r_aud.text
    assert "Diễn giả ${num}" in r_aud.text

    # /favicon.ico returns 204
    assert client.get("/favicon.ico").status_code == 204

    # ZERO admin routes exist on isolated public app!
    for forbidden_path in (
        "/api/health",
        "/api/cloud/config",
        "/api/logs/sessions",
        "/api/audience/info",
        "/api/shutdown",
        "/obs",
        "/docs",
        "/openapi.json",
    ):
        resp = client.get(forbidden_path)
        assert resp.status_code == 404, f"Public app leaked {forbidden_path} -> {resp.status_code}"
    assert client.post("/api/shutdown").status_code == 404

    # Admin /ws must NOT exist on isolated public app
    ws_failed = False
    try:
        with client.websocket_connect("/ws"):
            pass
    except Exception:
        ws_failed = True
    assert ws_failed, "Isolated public app unexpectedly accepted /ws"

    # WS /ws/audience works with PIN off + responds to heartbeat ping
    with client.websocket_connect("/ws/audience?lang=follow") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "status"
        ws.send_text("ping")
        pong = ws.receive_json()
        assert pong == {"type": "pong"}

    # Enable PIN: missing or wrong PIN receives pin_required + closes 4403
    pin_state["enabled"] = True
    with client.websocket_connect("/ws/audience?lang=follow") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "pin_required"
        assert msg["data"]["invalid"] is False

    with client.websocket_connect("/ws/audience?lang=follow&pin=0000") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "pin_required"
        assert msg["data"]["invalid"] is True

    with client.websocket_connect("/ws/audience?lang=follow&pin=7391") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "status"
        ws.send_text("ping")
        assert ws.receive_json() == {"type": "pong"}

    print("[PASS] test_isolated_public_app_routes_and_pin")


def test_hardened_loopback_guard_middleware() -> None:
    # Build a lightweight FastAPI app wrapped in the real LoopbackGuardMiddleware
    from fastapi import FastAPI

    mini = FastAPI()
    mini.add_middleware(srv.LoopbackGuardMiddleware)

    @mini.get("/api/cloud/config")
    def _cfg():
        return {"secret": "ok"}

    @mini.get("/audience")
    def _aud():
        return {"public": True}

    client = TestClient(mini)

    # Direct localhost works
    assert client.get("/api/cloud/config").status_code == 200

    # Any Cloudflare or reverse-proxy header MUST be blocked with 403 even from 127.0.0.1!
    for hdrs in (
        {"CF-Connecting-IP": "203.0.113.5"},
        {"CF-Ray": "8a1b2c3d4e5f-SGN"},
        {"X-Forwarded-For": "203.0.113.5"},
        {"X-Forwarded-Host": "demo.trycloudflare.com"},
        {"Forwarded": "for=203.0.113.5"},
        {"Host": "demo.trycloudflare.com"},
    ):
        r_admin = client.get("/api/cloud/config", headers=hdrs)
        assert r_admin.status_code == 403, f"Expected 403 for {hdrs}, got {r_admin.status_code}"
        r_pub = client.get("/audience", headers=hdrs)
        assert r_pub.status_code == 200, f"Expected 200 on /audience for {hdrs}, got {r_pub.status_code}"

    print("[PASS] test_hardened_loopback_guard_middleware")


def test_live_cloudflare_quick_tunnel() -> None:
    if find_cloudflared_binary() is None:
        print("[SKIP] test_live_cloudflare_quick_tunnel (cloudflared not preinstalled)")
        return

    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    with tempfile.TemporaryDirectory() as td:
        rt = _MockRuntime(loop=loop)
        mgr = PublicTunnelManager(rt, cache_dir=Path(td))
        try:
            mgr.start(mode="quick", pin_enabled=False)
            deadline = time.monotonic() + 35.0
            st = mgr.get_public_status()
            while time.monotonic() < deadline:
                st = mgr.get_public_status()
                if st["public_status"] in ("ready", "error"):
                    break
                time.sleep(0.25)

            assert st["public_status"] == "ready", f"Quick tunnel did not become ready: {st}"
            pub_url = st["public_url"]
            pub_base = pub_url.rsplit("/audience", 1)[0]
            pub_port = st["public_port"]
            assert pub_url.startswith("https://") and ".trycloudflare.com/audience" in pub_url
            assert "<svg" in st["public_qr_svg"]
            print(f"[INFO] Live Quick Tunnel URL: {pub_url} (isolated port 127.0.0.1:{pub_port})")

            # Give local ISP resolver a brief 1.5s margin after Cloudflare 1.1.1.1 DoH confirmed live
            time.sleep(1.5)
            http_ok = False
            last_exc = None
            for _ in range(12):
                try:
                    with httpx.Client(timeout=8.0) as hc:
                        r = hc.get(pub_url)
                        if r.status_code == 200 and "DAU Voice" in r.text:
                            # Also prove admin endpoint is 404 over the live public HTTPS URL!
                            r_adm = hc.get(f"{pub_base}/api/cloud/config")
                            assert r_adm.status_code == 404, f"Live public URL exposed /api/cloud/config: {r_adm.status_code}"
                            http_ok = True
                            break
                except Exception as exc:
                    last_exc = exc
                time.sleep(1.5)
            assert http_ok, f"Live HTTPS GET {pub_url} failed after DNS settle: {last_exc}"

            # Verify live WSS connection + ping/pong over Cloudflare HTTPS tunnel!
            ws_url = pub_base.replace("https://", "wss://") + "/ws/audience?lang=follow"

            async def _check_wss() -> None:
                async with websockets.connect(ws_url, open_timeout=10.0) as ws:
                    first = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                    assert first["type"] == "status"
                    await ws.send("ping")
                    pong = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                    assert pong == {"type": "pong"}

            fut = asyncio.run_coroutine_threadsafe(_check_wss(), loop)
            fut.result(timeout=15.0)
            print("[PASS] test_live_cloudflare_quick_tunnel (HTTPS 200 + /api 404 + WSS ping/pong)")

            # Stop tunnel and confirm clean release of isolated port & process
            mgr.stop(notify=False)
            assert mgr.get_public_status()["public_status"] == "idle"
            # Verify isolated port is released (bindable again)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", pub_port))
            finally:
                s.close()
            print("[PASS] test_live_cloudflare_quick_tunnel_cleanup")
        finally:
            mgr.stop(notify=False)
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(timeout=2.0)
            loop.close()


if __name__ == "__main__":
    test_domain_normalization_and_config_security()
    test_isolated_public_app_routes_and_pin()
    test_hardened_loopback_guard_middleware()
    test_live_cloudflare_quick_tunnel()
    print("ALL PUBLIC TUNNEL TESTS PASSED!")
