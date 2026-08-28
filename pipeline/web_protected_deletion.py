from __future__ import annotations

import argparse
import os
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from . import web_app as _web_app
from . import web_routes as _web_routes
from .dataset_workspace import DatasetWorkspace
from .lifecycle_guard import deletion_blockers, lifecycle_lock
from .models import PipelineError
from .resource_deletion import (
    delete_training_config,
    delete_training_project,
    delete_training_run,
    guarded_create_project_from_training_config,
)
from .service import load_project
from .web_app import WebApplication, _e, _q
from .web_outfit import OutfitHandler


# All UI training-workspace creation must participate in the same lifecycle lock as
# destructive operations. Keep the patch local to the final Web entry layer so older
# compatibility modules do not need to duplicate the policy.
_web_app.create_project_from_training_config = guarded_create_project_from_training_config
_web_routes.create_project_from_training_config = guarded_create_project_from_training_config

# Dataset/video jobs must become visible to the deletion guard atomically with their
# validation. web_metadata may have replaced this global with an enriched spawner, so
# wrap whatever implementation is active after importing OutfitHandler.
_original_spawn_job = _web_routes.spawn_job


def _guarded_spawn_job(kind, payload, *, root=None):
    resolved = (root or _web_app.repository_root()).resolve()
    with lifecycle_lock(resolved):
        dataset = str((payload or {}).get("dataset") or "")
        project = str((payload or {}).get("project") or "")
        if dataset:
            DatasetWorkspace.load(dataset, root=resolved)
        if project:
            load_project(project, root=resolved)
        return _original_spawn_job(kind, payload, root=resolved)


_web_routes.spawn_job = _guarded_spawn_job


class ProtectedDeletionHandler(OutfitHandler):
    """Final Web UI layer with protected deletion for Dataset/Config/Project resources."""

    def _post(self, form: dict[str, list[str]]) -> None:
        path = urlparse(self.path).path.rstrip("/")
        parts = path.split("/")
        if len(parts) == 4 and parts[1] == "configs" and parts[3] == "delete":
            name = unquote(parts[2])
            if form.get("confirm", [""])[0] != name:
                raise PipelineError("训练配置名称确认不匹配")
            delete_training_config(name, root=self.app.root)
            self._redirect("/configs")
            return
        if len(parts) == 5 and parts[1] == "results" and parts[4] == "delete":
            project_name = unquote(parts[2])
            run_id = unquote(parts[3])
            if form.get("confirm", [""])[0] != run_id:
                raise PipelineError("Run ID 确认不匹配")
            delete_training_run(project_name, run_id, root=self.app.root)
            self._redirect("/results")
            return
        if len(parts) == 4 and parts[1] == "status" and parts[3] == "delete":
            project_name = unquote(parts[2])
            if form.get("confirm", [""])[0] != project_name:
                raise PipelineError("训练工作区名称确认不匹配")
            delete_training_project(project_name, root=self.app.root)
            destination = form.get("return_to", ["/status"])[0]
            if destination not in {"/status", "/results"}:
                destination = "/status"
            self._redirect(destination)
            return
        super()._post(form)

    def _html(self, text: str, *, status: int = 200) -> None:
        if self.command == "GET" and status == 200:
            panel = self._protected_delete_panel()
            if panel:
                marker = "</div></body></html>"
                if marker in text:
                    text = text.replace(marker, panel + marker, 1)
        super()._html(text, status=status)

    def _protected_delete_panel(self) -> str:
        path = urlparse(self.path).path.rstrip("/")
        parts = path.split("/")

        if len(parts) == 3 and parts[1] == "datasets":
            name = unquote(parts[2])
            blockers = deletion_blockers("dataset", name, root=self.app.root)
            return self._blocker_notice(blockers) if blockers else ""

        if len(parts) == 3 and parts[1] == "configs":
            name = unquote(parts[2])
            blockers = deletion_blockers("training_config", name, root=self.app.root)
            return self._delete_panel(
                title="删除训练配置",
                resource_name=name,
                blockers=blockers,
                action=f"/configs/{_q(name)}/delete",
                button="永久删除训练配置",
                note="只删除可复用配置文件；已经冻结到历史 Run 的 Config snapshot 会保留。",
            )

        if len(parts) == 3 and parts[1] == "status":
            project_name = unquote(parts[2])
            blockers = deletion_blockers("project", project_name, root=self.app.root)
            return self._delete_panel(
                title="删除训练工作区",
                resource_name=project_name,
                blockers=blockers,
                action=f"/status/{_q(project_name)}/delete",
                button="永久删除训练工作区",
                note="删除整个 Project，包括所有 Run、权重、日志、缓存和评测产物；Dataset 与 Training Config 不受影响。",
                return_to="/status",
            )

        if len(parts) == 4 and parts[1] == "results":
            project_name = unquote(parts[2])
            run_id = unquote(parts[3])
            blockers = deletion_blockers(
                "run", f"{project_name}/{run_id}", root=self.app.root
            )
            return self._delete_panel(
                title="删除这个训练结果",
                resource_name=run_id,
                blockers=blockers,
                action=f"/results/{_q(project_name)}/{_q(run_id)}/delete",
                button="永久删除这个 Run",
                note="只删除这个 Run 的权重、日志、示例图、评测与 Run metadata；同一 Project 的其他 Run、Dataset 和 Training Config 保留。",
            )
        return ""

    def _blocker_notice(self, blockers: list[dict[str, str]]) -> str:
        rows = "".join(
            f"<li><span class='mono'>{_e(item['id'])}</span> · {_e(item['status'])} · {_e(item['reason'])}</li>"
            for item in blockers
        )
        return (
            "<div class='panel danger-zone' style='margin-top:18px'>"
            "<h3 class='warn'>当前禁止永久删除</h3>"
            "<p class='muted'>存在活动或待启动任务。可以继续编辑未来版本，但永久删除 Dataset / Source / 图片会被后端拒绝。</p>"
            f"<ul>{rows}</ul></div>"
        )

    def _delete_panel(
        self,
        *,
        title: str,
        resource_name: str,
        blockers: list[dict[str, str]],
        action: str,
        button: str,
        note: str,
        return_to: str | None = None,
    ) -> str:
        if blockers:
            rows = "".join(
                f"<li><span class='mono'>{_e(item['id'])}</span> · {_e(item['status'])} · {_e(item['reason'])}</li>"
                for item in blockers
            )
            return (
                "<div class='panel danger-zone' style='margin-top:18px'>"
                f"<h3 class='warn'>{_e(title)}：暂不可用</h3>"
                "<p class='muted'>活动/待启动引用必须先结束或取消：</p>"
                f"<ul>{rows}</ul></div>"
            )
        hidden_return = (
            f"<input type='hidden' name='return_to' value='{_e(return_to)}'>" if return_to else ""
        )
        return (
            "<div class='panel danger-zone' style='margin-top:18px'>"
            f"<h3 class='bad'>{_e(title)}</h3><p class='muted'>{_e(note)}</p>"
            f"<form method='post' action='{action}'>"
            f"<input type='hidden' name='_csrf' value='{self.app.csrf}'>{hidden_return}"
            f"<label>输入 <b>{_e(resource_name)}</b> 确认"
            f"<input name='confirm' autocomplete='off'></label>"
            f"<div class='toolbar'><button class='danger'>{_e(button)}</button></div></form></div>"
        )


def make_server(
    host: str = "127.0.0.1",
    port: int = 7860,
    *,
    root: Path | None = None,
    auth_token: str | None = None,
) -> ThreadingHTTPServer:
    app = WebApplication(root=root)
    app.auth_token = auth_token or None
    handler = type("BoundProtectedDeletionHandler", (ProtectedDeletionHandler,), {"app": app})
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
            "LAN mode requires LORA_WEB_TOKEN/--token. Use --unsafe-no-auth only on an explicitly trusted private network."
        )
    server = make_server(host, port, root=root, auth_token=auth_token)
    print(f"LoRA Pipeline Web: http://{host}:{port}")
    if loopback:
        print(
            f"Remote browser: ssh -L {port}:127.0.0.1:{port} <nas>  then open http://127.0.0.1:{port}"
        )
    elif auth_token:
        print("LAN mode enabled with access-token authentication.")
    else:
        print("WARNING: unauthenticated LAN mode was explicitly enabled.")
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
    parser.add_argument("--unsafe-no-auth", action="store_true")
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
