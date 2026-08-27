from __future__ import annotations

import argparse
import os
from http.server import ThreadingHTTPServer
from pathlib import Path

from .config import load_base_registry
from .models import PipelineError
from .training_config import TrainingConfig, parse_anchor_tags
from .training_parameters import (
    TRAINING_PARAMETER_SPECS,
    effective_training_settings,
    strategy_training_defaults,
    update_key_training_overrides,
)
from .web_app import WebApplication, _e, _page, _q
from .web_metadata_batch import MetadataBatchHandler


class OutfitHandler(MetadataBatchHandler):
    """Full Web UI with character-outfit targets and documented key parameter tuning."""

    def _configs(self) -> None:
        configs = self.app.configs()
        rows = []
        for config in configs:
            training = effective_training_settings(
                config.strategy, config.overrides, root=self.app.root
            )
            anchors = ", ".join(config.anchor_tags) if config.anchor_tags else "—"
            rows.append(
                f"<tr><td><a href='/configs/{_q(config.name)}'><b>{_e(config.name)}</b></a></td>"
                f"<td>{_e(config.target_type)}</td><td>{_e(anchors)}</td><td>{_e(config.base)}</td>"
                f"<td>{_e(config.strategy)}</td><td>{_e(training.get('network_dim', 16))}</td>"
                f"<td>{_e(training.get('batch_size', 1))}</td><td>{config.images_seen}</td></tr>"
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
            "<div class='muted'>先选 quality / fast / cached 预设；需要时只覆盖少数关键参数。Trigger 是唯一 LoRA 激活词，人物衣装使用独立人物锚点。</div></div>"
        )
        body += (
            "<table><tr><th>名称</th><th>训练目标</th><th>人物锚点</th><th>底模</th>"
            "<th>策略</th><th>Rank</th><th>Batch</th><th>images_seen</th></tr>"
            + (
                "".join(rows)
                or "<tr><td colspan=8 class=muted>暂无训练配置</td></tr>"
            )
            + "</table>"
        )
        body += f"""<div class="panel" style="margin-top:18px"><h3>创建训练配置</h3>
<form method="post" action="/configs/create"><input type="hidden" name="_csrf" value="{self.app.csrf}">
<div class="row">
<label>名称<input name="name" required></label>
<label>训练目标<select name="target_type"><option value="character">人物 character</option><option value="character_outfit">人物衣装 character_outfit</option><option value="style">风格 style</option></select></label>
<label>底模<select name="base">{options}</select></label>
<label>唯一 Trigger<input name="trigger" required placeholder="misuzu_nic26"><span class="muted">单一激活词，不要填逗号 Tag 列表。</span></label>
<label>人物锚点（仅衣装，逗号分隔）<input name="anchor_tags" placeholder="hataya misuzu"></label>
<label>评测主体基础 Prompt<input name="subject_prompt" value="1girl"><span class="muted">衣装模式会自动附加人物锚点；不要填写 Trigger。</span></label>
<label>训练策略<select name="strategy"><option>quality</option><option>fast</option><option>cached</option></select><span class="muted">下方高级参数留空时完全使用该预设。</span></label>
<label>images_seen<input type="number" min="1" name="images_seen" value="1000"><span class="muted">训练图片累计曝光预算；用于公平比较不同 Batch。</span></label>
<label>LoRA Rank<input type="number" min="1" name="network_dim" placeholder="留空=预设"><span class="muted">容量越大可学细节越多，也更占参数并增加过拟合风险。</span></label>
<label>LoRA Alpha<input type="number" min="1" name="network_alpha" placeholder="留空=预设"><span class="muted">与 Rank 一起决定 LoRA 更新缩放。</span></label>
<label>UNet LR<input name="unet_lr" placeholder="留空=预设，例如 1e-4"><span class="muted">过高易过拟合，过低可能学不住目标。</span></label>
<label>物理 Batch Size<input type="number" min="1" name="batch_size" placeholder="留空=预设"><span class="muted">不设人工上限；以实际 VRAM/OOM 为准。</span></label>
<label>梯度累积步数<input type="number" min="1" name="gradient_accumulation_steps" placeholder="留空=预设"><span class="muted">有效 Batch = 物理 Batch × 梯度累积。</span></label>
<label>Seed<input type="number" min="0" name="seed" placeholder="留空=预设"><span class="muted">用于复现实验；没有固定的“更好 Seed”。</span></label>
</div><div class="toolbar"><button class="good">创建</button></div></form></div>"""
        body += self._training_parameter_help_html()
        self._html(_page("训练配置", body, active="configs"))

    def _config(self, name: str) -> None:
        config = TrainingConfig.load(name, root=self.app.root)
        training = effective_training_settings(
            config.strategy, config.overrides, root=self.app.root
        )
        custom = config.overrides.get("training", {})
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
        effective_subject = str(config.effective_evaluation().get("subject_prompt", subject))
        effective_batch = int(training.get("batch_size", 1)) * int(
            training.get("gradient_accumulation_steps", 1)
        )
        body = f"""<div class="hero"><h1>{_e(name)}</h1>
<div class="muted">Config snapshot {_e(config.snapshot()['snapshot_hash'][:16])} · 实际评测主体：{_e(effective_subject)}</div></div>
<div class="grid">
<div class="card"><div class="muted">Rank / Alpha</div><b>{_e(training.get('network_dim',16))} / {_e(training.get('network_alpha',8))}</b></div>
<div class="card"><div class="muted">UNet LR</div><b>{_e(training.get('unet_lr',0.0001))}</b></div>
<div class="card"><div class="muted">物理 / 有效 Batch</div><b>{_e(training.get('batch_size',1))} / {effective_batch}</b></div>
<div class="card"><div class="muted">Seed</div><b>{_e(training.get('seed',42))}</b></div>
</div>
<div class="panel"><form method="post" action="/configs/{_q(name)}/save"><input type="hidden" name="_csrf" value="{self.app.csrf}">
<div class="row">
<label>训练目标<select name="target_type">{target_options}</select></label>
<label>底模<select name="base">{options}</select></label>
<label>唯一 Trigger<input name="trigger" value="{_e(config.trigger)}"></label>
<label>人物锚点（衣装模式）<input name="anchor_tags" value="{_e(anchors)}"></label>
<label>评测主体基础 Prompt<input name="subject_prompt" value="{_e(subject)}"><span class="muted">禁止包含 Trigger；衣装模式自动附加人物锚点。</span></label>
<label>策略<select name="strategy"><option {'selected' if config.strategy == 'quality' else ''}>quality</option><option {'selected' if config.strategy == 'fast' else ''}>fast</option><option {'selected' if config.strategy == 'cached' else ''}>cached</option></select><span class="muted">没有自定义的参数会跟随新策略预设。</span></label>
<label>images_seen<input type="number" min="1" name="images_seen" value="{config.images_seen}"></label>
<label>LoRA Rank<input type="number" min="1" name="network_dim" value="{_e(custom.get('network_dim',''))}" placeholder="留空=预设 {_e(strategy_training_defaults(config.strategy, root=self.app.root).get('network_dim',16))}"></label>
<label>LoRA Alpha<input type="number" min="1" name="network_alpha" value="{_e(custom.get('network_alpha',''))}" placeholder="留空=预设 {_e(strategy_training_defaults(config.strategy, root=self.app.root).get('network_alpha',8))}"></label>
<label>UNet LR<input name="unet_lr" value="{_e(custom.get('unet_lr',''))}" placeholder="留空=预设 {_e(strategy_training_defaults(config.strategy, root=self.app.root).get('unet_lr',0.0001))}"></label>
<label>物理 Batch Size<input type="number" min="1" name="batch_size" value="{_e(custom.get('batch_size',''))}" placeholder="留空=预设 {_e(strategy_training_defaults(config.strategy, root=self.app.root).get('batch_size',1))}"><span class="muted">无人工上限，以实测显存为准。</span></label>
<label>梯度累积步数<input type="number" min="1" name="gradient_accumulation_steps" value="{_e(custom.get('gradient_accumulation_steps',''))}" placeholder="留空=预设 {_e(strategy_training_defaults(config.strategy, root=self.app.root).get('gradient_accumulation_steps',1))}"></label>
<label>Seed<input type="number" min="0" name="seed" value="{_e(custom.get('seed',''))}" placeholder="留空=预设 {_e(strategy_training_defaults(config.strategy, root=self.app.root).get('seed',42))}"></label>
</div><div class="toolbar"><button class="good">保存</button></div></form>
<p class="muted">留空会恢复该参数的策略预设。character_outfit 会在 Caption 中固定加入 Trigger + 人物锚点，并使用衣装专用评测矩阵。</p></div>"""
        body += self._training_parameter_help_html(current_strategy=config.strategy)
        self._html(_page(name, body, active="configs"))

    def _config_create(self, form: dict[str, list[str]]) -> None:
        target_type = form.get("target_type", ["character"])[0]
        concept_type = "style" if target_type == "style" else "character"
        strategy = form.get("strategy", ["quality"])[0]
        overrides = self._training_overrides(form, strategy=strategy)
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
            strategy=strategy,
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
        strategy = form["strategy"][0]
        config.data["strategy"] = strategy
        config.data["images_seen"] = int(form["images_seen"][0])
        config.data["overrides"] = self._training_overrides(
            form,
            strategy=strategy,
            existing=config.overrides,
        )
        config.validate(require_enabled_base=True, root=self.app.root)
        config.save()
        self._redirect(f"/configs/{_q(name)}")

    def _training_overrides(
        self,
        form: dict[str, list[str]],
        *,
        strategy: str,
        existing: dict | None = None,
    ) -> dict:
        values: dict[str, int | float] = {}
        for key in (
            "network_dim",
            "network_alpha",
            "batch_size",
            "gradient_accumulation_steps",
            "seed",
        ):
            raw = form.get(key, [""])[0].strip()
            if raw:
                values[key] = int(raw)
        raw_lr = form.get("unet_lr", [""])[0].strip()
        if raw_lr:
            values["unet_lr"] = float(raw_lr)
        return update_key_training_overrides(
            existing or {},
            strategy=strategy,
            values=values,
            root=self.app.root,
        )

    def _training_parameter_help_html(self, *, current_strategy: str | None = None) -> str:
        strategy_defaults = {
            strategy: strategy_training_defaults(strategy, root=self.app.root)
            for strategy in ("quality", "fast", "cached")
        }
        rows: list[str] = []
        for spec in TRAINING_PARAMETER_SPECS:
            if spec.key == "images_seen":
                preset = "配置默认 1000（独立于策略）"
            elif current_strategy:
                preset = str(strategy_defaults[current_strategy].get(spec.key, "—"))
            else:
                preset = " / ".join(
                    f"{name}:{values.get(spec.key, '—')}"
                    for name, values in strategy_defaults.items()
                )
            rows.append(
                f"<tr><td><b>{_e(spec.label_zh)}</b><div class='mono muted'>{_e(spec.key)}</div></td>"
                f"<td>{_e(preset)}</td><td>{_e(spec.description_zh)}<br><span class='muted'>{_e(spec.recommendation_zh)}</span></td></tr>"
            )
        return (
            "<details class='panel' style='margin-top:18px' open>"
            "<summary><b>关键训练参数说明</b></summary>"
            "<p class='muted'>自定义参数只保存与策略预设不同的值。固定 images_seen 时，有效 Batch 越大，optimizer step 数越少，因此提高 Batch 并不是完全等价的纯加速。</p>"
            "<table><tr><th>参数</th><th>预设</th><th>作用与建议</th></tr>"
            + "".join(rows)
            + "</table></details>"
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
