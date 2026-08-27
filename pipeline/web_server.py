from __future__ import annotations

import argparse
import hmac
import os
from http.cookies import SimpleCookie
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from .models import PipelineError
from .training_config import TrainingConfig
from .web_app import _page, _q
from .web_safety import FullHandler as SafetyHandler


class SecureHandler(SafetyHandler):
    """Final Web handler with optional token auth and override-safe config edits."""

    def do_GET(self) -> None:  # noqa: N802
        try:
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path == "/login":
                self._login_page()
                return
            if path != "/healthz" and not self._auth_ok():
                self._redirect("/login")
                return
            self._get()
        except (PipelineError, OSError, ValueError, KeyError) as exc:
            self._html(
                _page("错误", "<div class='hero'><h1>请求失败</h1></div>", error=str(exc)),
                status=400,
            )

    def do_POST(self) -> None:  # noqa: N802
        try:
            path = urlparse(self.path).path.rstrip("/") or "/"
            form = self._form()
            if path == "/login":
                self._login(form)
                return
            if not self._auth_ok():
                raise PipelineError("authentication required")
            origin = self.headers.get("Origin")
            host = self.headers.get("Host")
            if origin and host and urlparse(origin).netloc != host:
                raise PipelineError("cross-origin state change rejected")
            if form.get("_csrf", [""])[0] != self.app.csrf:
                raise PipelineError("CSRF validation failed; refresh the page and retry")
            self._post(form)
        except (PipelineError, OSError, ValueError, KeyError) as exc:
            self._html(
                _page("错误", "<div class='hero'><h1>操作失败</h1></div>", error=str(exc)),
                status=400,
            )

    def _auth_ok(self) -> bool:
        token = getattr(self.app, "auth_token", None)
        if not token:
            return True
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie") or "")
        except Exception:
            return False
        morsel = cookie.get("lora_web_token")
        supplied = unquote(morsel.value) if morsel else ""
        return hmac.compare_digest(supplied, str(token))

    def _login_page(self, *, error: str = "") -> None:
        if not getattr(self.app, "auth_token", None):
            self._redirect("/")
            return
        problem = f"<div class='error'>{error}</div>" if error else ""
        body = (
            "<div class='hero'><h1>LoRA Pipeline Web</h1>"
            "<div class='muted'>请输入 Web access token。</div></div>"
            f"{problem}<div class='panel' style='max-width:520px'>"
            "<form method='post' action='/login'>"
            "<label>Access token<input type='password' name='token' autofocus required></label>"
            "<div class='toolbar'><button class='good'>登录</button></div></form></div>"
        )
        self._html(_page("登录", body))

    def _login(self, form: dict[str, list[str]]) -> None:
        token = str(getattr(self.app, "auth_token", None) or "")
        supplied = form.get("token", [""])[0]
        if not token or not hmac.compare_digest(supplied, token):
            self._login_page(error="Access token 不正确")
            return
        self.send_response(303)
        self.send_header("Location", "/")
        self.send_header(
            "Set-Cookie",
            f"lora_web_token={quote(supplied, safe='')}; Path=/; HttpOnly; SameSite=Strict",
        )
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _config_save(self, name: str, form: dict[str, list[str]]) -> None:
        """Update core training fields without discarding non-Web overrides."""

        config = TrainingConfig.load(name, root=self.app.root)
        config.data["base"] = form["base"][0]
        config.data["trigger"] = form["trigger"][0].strip()
        config.data["strategy"] = form["strategy"][0]
        config.data["images_seen"] = int(form["images_seen"][0])

        overrides = dict(config.overrides)
        web_training = self._training_overrides(form).get("training")
        if web_training:
            overrides["training"] = web_training
        else:
            overrides.pop("training", None)
        config.data["overrides"] = overrides

        config.validate(require_enabled_base=True, root=self.app.root)
        config.save()
        self._redirect(f"/configs/{_q(name)}")


def make_server(
    host: str = "127.0.0.1",
    port: int = 7860,
    *,
    root: Path | None = None,
    auth_token: str | None = None,
) -> ThreadingHTTPServer:
    from .web_app import WebApplication

    app = WebApplication(root=root)
    app.auth_token = auth_token or None
    handler = type("BoundSecureHandler", (SecureHandler,), {"app": app})
    return ThreadingHTTPServer((host, int(port)), handler)


def serve(
    host: str = "127.0.0.1",
    port: int = 7860,
    *,
    allow_lan: bool = False,
    auth_token: str | None = None,
    unsafe_no_auth: bool = False,
    root: Path | None = None,
) -> None:
    loopback = host in {"127.0.0.1", "localhost", "::1"}
    if not loopback and not allow_lan:
        raise PipelineError("Refusing non-loopback bind without --allow-lan; prefer an SSH tunnel")
    if not loopback and not auth_token and not unsafe_no_auth:
        raise PipelineError(
            "LAN mode requires LORA_WEB_TOKEN/--token. "
            "Use --unsafe-no-auth only on an explicitly trusted private network."
        )

    server = make_server(host, port, root=root, auth_token=auth_token)
    print(f"LoRA Pipeline Web: http://{host}:{port}")
    if loopback:
        print(
            f"Remote browser: ssh -L {port}:127.0.0.1:{port} <nas>  "
            f"then open http://127.0.0.1:{port}"
        )
    elif auth_token:
        print("LAN mode enabled with access-token authentication.")
        print("Prefer HTTPS/reverse proxy if the LAN itself is not trusted.")
    else:
        print("WARNING: unauthenticated LAN mode was explicitly enabled.")
        print("Never expose this service directly to the public Internet.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="LoRA Pipeline NAS web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--allow-lan", action="store_true")
    parser.add_argument("--token", default=os.environ.get("LORA_WEB_TOKEN"))
    parser.add_argument(
        "--unsafe-no-auth",
        action="store_true",
        help="Allow explicit non-loopback binding without Web authentication.",
    )
    args = parser.parse_args(argv)
    serve(
        args.host,
        args.port,
        allow_lan=args.allow_lan,
        auth_token=args.token,
        unsafe_no_auth=args.unsafe_no_auth,
    )


if __name__ == "__main__":
    main()
