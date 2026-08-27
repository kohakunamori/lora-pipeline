from __future__ import annotations

import argparse
import os
from http.server import ThreadingHTTPServer
from pathlib import Path

from .config import load_base_registry
from .models import PipelineError
from .training_config import TrainingConfig, parse_anchor_tags
from .web_app import WebApplication, _e, _page, _q
from .web_metadata_batch import MetadataBatchHandler


class OutfitHandler(MetadataBatchHandler):
    """Full Web UI with explicit character-outfit training configuration."""

    def _configs(self) -> None:
        configs = self.app.configs()
        rows = []
        for config in configs:
            training = config.overrides.get("training", {})
            anchors = ", ".join(config.anchor_tags) if config.anchor_tags else "—"
            rows.append(
                f"<tr><td><a href='/configs/{_q(config.name)}'><b>{_e(config.name)}</b></a></td>"
                f"<td>{_e(config.target_type)}</td><td>{_e(anchors)}</td><td>{_e(config.base)}</td>"
                f"<td>{_e(config.strategy)}</td><td>{_e(training.get('network_dim', '默认'))}</td>"
                f"<td>{config.images_seen}</td></tr>"
            )
        bases = [
            (key, value)
            for key, value in load_base_registry(self.app.root).items()
            if value.enabled
        ]
        options = "".join(
            f"<option value='{_e(key)}'>{_e(key)} · {_e(value.name)}</option>"
            for key, value in bases
        )
        body = (
            "<div class='hero'><h1>训练配置</h1>"
            "<div class='muted'>Trigger 是唯一 LoRA 激活词；人物衣装使用独立人物锚点。</div></div>"
        )
        body += (
            "<table><tr><th>名称</th><th>训练目标</th><th>人物锚点</th><th>底模</th>"
            "<th>策略</th><th>Rank</th><th>images_seen</th></tr>"
            + (
                "".join(rows)
                or "<tr><td colspan=7 class=muted>暂无训练配置</td></tr>"
            )
            + "</table>"
        )
        body += f"""<div class="panel" style="margin-top:18px"><h3>创建训练配置</h3>
<form method="post" action="/configs/create"><input type="hidden" name="_csrf" value="{self.app.csrf}">
<div class="row">
<label>名称<input name="name" required></label>
<label>训练目标<select name="target_type"><option value="character">人物 character</option><option value="character_outfit">人物衣装 character_outfit</option><option value="style">风格 style</option></select></label>
<label>底模<select name="base">{options}</select></label>
<label>唯一 Trigger<input name="trigger" required placeholder="misuzu_nic26"></label>
<label>人物锚点（仅衣装，逗号分隔）<input name="anchor_tags" placeholder="hataya misuzu"></label>
<label>评测主体基础 Prompt<input name="subject_prompt" value="1girl"><span class="muted">衣装模式会自动附加人物锚点；不要填写 Trigger。</span></label>
<label>策略<select name="strategy"><option>quality</option><option>fast</option><option>cached</option></select></label>
<label>images_seen<input type="number" min="1" name="images_seen" value="1000"></label>
<label>Rank（留空=默认）<input type="number" min="1" name="network_dim"></label>
<label>Alpha（留空=默认）<input type="number" min="1" name="network_alpha"></label>
<label>UNet LR（留空=默认）<input name="unet_lr"></label>
</div><div class="toolbar"><button class="good">创建</button></div></form></div>"""
        self._html(_page("训练配置", body, active="configs"))

    def _config(self, name: str) -> None:
        config = TrainingConfig.load(name, root=self.app.root)
        training = config.overrides.get("training", {})
        bases = [
            (key, value)
            for key, value in load_base_registry(self.app.root).items()
            if value.enabled
        ]
        options = "".join(
            f"<option value='{_e(key)}' {'selected' if key == config.base else ''}>"
            f"{_e(key)} · {_e(value.name)}</option>"
            for key, value in bases
        )
        if config.concept_type == "character":
            target_options = (
                f"<option value='character' {'selected' if config.target_type == 'character' else ''}>人物 character</option>"
                f"<option value='character_outfit' {'selected' if config.target_type == 'character_outfit' else ''}>人物衣装 character_outfit</option>"
            )
        else:
            target_options = "<option value='style' selected>风格 style</option>"
        anchors = ", ".join(config.anchor_tags)
        subject = str(config.evaluation.get("subject_prompt", "1girl"))
        effective = str(config.effective_evaluation().get("subject_prompt", subject))
        body = f"""<div class="hero"><h1>{_e(name)}</h1>
<div class="muted">Config snapshot {_e(config.snapshot()['snapshot_hash'][:16])} · 实际评测主体：{_e(effective)}</div></div>
<div class="panel"><form method="post" action="/configs/{_q(name)}/save"><input type="hidden" name="_csrf" value="{self.app.csrf}">
<div class="row">
<label>训练目标<select name="target_type">{target_options}</select></label>
<label>底模<select name="base">{options}</select></label>
<label>唯一 Trigger<input name="trigger" value="{_e(config.trigger)}"></label>
<label>人物锚点（衣装模式）<input name="anchor_tags" value="{_e(anchors)}"></label>
<label>评测主体基础 Prompt<input name="subject_prompt" value="{_e(subject)}"><span class="muted">禁止包含 Trigger；衣装模式自动附加人物锚点。</span></label>
<label>策略<select name="strategy"><option {'selected' if config.strategy == 'quality' else ''}>quality</option><option {'selected' if config.strategy == 'fast' else ''}>fast</option><option {'selected' if config.strategy == 'cached' else ''}>cached</option></select></label>
<label>images_seen<input type="number" min="1" name="images_seen" value="{config.images_seen}"></label>
<label>Rank<input type="number" min="1" name="network_dim" value="{_e(training.get('network_dim',''))}"></label>
<label>Alpha<input type="number" min="1" name="network_alpha" value="{_e(training.get('network_alpha',''))}"></label>
<label>UNet LR<input name="unet_lr" value="{_e(training.get('unet_lr',''))}"></label>
</div><div class="toolbar"><button class="good">保存</button></div></form>
<p class="muted">character_outfit 会在清洗/生成 Caption 时固定加入 Trigger + 人物锚点，并使用衣装专用评测矩阵。</p></div>"""
        self._html(_page(name, body, active="configs"))

    def _config_create(self, form: dict[str, list[str]]) -> None:
        overrides = self._training_overrides(form)
        target_type = form.get("target_type", ["character"])[0]
        concept_type = "style" if target_type == "style" else "character"
        evaluation = {}
        if concept_type == "character":
            evaluation["subject_prompt"] = form.get("subject_prompt", ["1girl"])[0].strip()
        TrainingConfig.create(
            form["name"][0].strip(),
            concept_type=concept_type,
            target_type=target_type,
            base=form["base"][0],
            trigger=form["trigger"][0].strip(),
            anchor_tags=parse_anchor_tags(form.get("anchor_tags", [""])[0]),
            strategy=form.get("strategy", ["quality"])[0],
            images_seen=int(form.get("images_seen", ["1000"])[0]),
            overrides=overrides,
            evaluation=evaluation,
            root=self.app.root,
        )
        self._redirect(f"/configs/{_q(form['name'][0].strip())}")

    def _config_save(self, name: str, form: dict[str, list[str]]) -> None:
        config = TrainingConfig.load(name, root=self.app.root)
        target_type = form.get("target_type", [config.target_type])[0]
        if config.concept_type == "style" and target_type != "style":
            raise PipelineError("Style config cannot be converted to a character target in place")
        if config.concept_type == "character" and target_type not in {
            "character",
            "character_outfit",
        }:
            raise PipelineError("Character config target must be character or character_outfit")
        config.data["target_type"] = target_type
        config.data["base"] = form["base"][0]
        config.data["trigger"] = form["trigger"][0].strip()
        config.data["anchor_tags"] = (
            parse_anchor_tags(form.get("anchor_tags", [""])[0])
            if target_type == "character_outfit"
            else []
        )
        if config.concept_type == "character":
            config.evaluation["subject_prompt"] = form.get("subject_prompt", ["1girl"])[0].strip()
        config.data["strategy"] = form["strategy"][0]
        config.data["images_seen"] = int(form["images_seen"][0])
        config.data["overrides"] = self._training_overrides(form)
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
    app = WebApplication(root=root)
    app.auth_token = auth_token or None
    handler = type("BoundOutfitHandler", (OutfitHandler,), {"app": app})
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
