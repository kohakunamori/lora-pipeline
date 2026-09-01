from __future__ import annotations

from urllib.parse import parse_qs, unquote, urlparse

from .models import PipelineError
from .target_training_advisor import target_training_advice
from .target_training_apply import apply_target_preferred_start
from .training_config import TrainingConfig
from .training_parameters import effective_training_settings
from .web_app import _e, _q
from .web_outfit import OutfitHandler


class TargetAdvisorHandler(OutfitHandler):
    """Web UI layer for explicit preview/apply target-aware training advice."""

    def _post(self, form: dict[str, list[str]]) -> None:
        path = urlparse(self.path).path.rstrip("/")
        parts = path.split("/")
        if len(parts) == 4 and parts[1] == "configs" and parts[3] == "advisor":
            name = unquote(parts[2])
            config = TrainingConfig.load(name, root=self.app.root)
            raw_count = form.get("image_count", [""])[0].strip()
            try:
                image_count = int(raw_count)
            except ValueError as exc:
                raise PipelineError("Advisor image count must be an integer >= 1") from exc
            if image_count < 1:
                raise PipelineError("Advisor image count must be an integer >= 1")

            current = effective_training_settings(
                config.strategy,
                config.overrides,
                root=self.app.root,
            )
            advice = target_training_advice(
                config.target_type,
                image_count=image_count,
                current_training=current,
                current_images_seen=config.images_seen,
            )
            images_seen, overrides = apply_target_preferred_start(
                strategy=config.strategy,
                overrides=config.overrides,
                current_training=current,
                advice=advice,
                root=self.app.root,
            )
            config.data["images_seen"] = images_seen
            config.data["overrides"] = overrides
            config.save()
            self._redirect(
                f"/configs/{_q(name)}?advisor_images={image_count}&advisor_applied=1"
            )
            return
        super()._post(form)

    def _html(self, text: str, *, status: int = 200) -> None:
        if self.command == "GET" and status == 200:
            panel = self._target_advisor_panel()
            if panel:
                marker = "</div></body></html>"
                if marker in text:
                    text = text.replace(marker, panel + marker, 1)
        super()._html(text, status=status)

    def _target_advisor_panel(self) -> str:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        parts = path.split("/")
        if len(parts) != 3 or parts[1] != "configs":
            return ""

        name = unquote(parts[2])
        try:
            config = TrainingConfig.load(name, root=self.app.root)
        except Exception:
            return ""

        query = parse_qs(parsed.query)
        raw_count = query.get("advisor_images", [""])[0].strip()
        applied = query.get("advisor_applied", [""])[0] == "1"
        default_count = raw_count or "40"
        panel = [
            "<div class='panel' style='margin-top:18px'>",
            "<h3>训练目标参数建议</h3>",
            "<p class='muted'>按人物 / 人物衣装 / 风格目标和预计训练图片数生成保守起点。预览不会修改配置；只有点击“应用首选起点”才写入 images_seen / Rank / Alpha / LR，Batch、梯度累积、Seed 与专家级 override 保持不变。</p>",
        ]
        if applied:
            panel.append("<p class='good'>已应用首选起点。建议仍是 heuristic，不代表质量真值。</p>")
        panel.append(
            f"<form method='get' action='/configs/{_q(name)}'>"
            f"<label>预计实际参与训练的图片数<input type='number' min='1' name='advisor_images' value='{_e(default_count)}'></label>"
            "<div class='toolbar'><button>预览目标建议</button></div></form>"
        )

        if raw_count:
            try:
                image_count = int(raw_count)
            except ValueError:
                image_count = 0
            if image_count < 1:
                panel.append("<p class='error'>预计训练图片数必须是大于等于 1 的整数。</p>")
            else:
                current = effective_training_settings(
                    config.strategy,
                    config.overrides,
                    root=self.app.root,
                )
                advice = target_training_advice(
                    config.target_type,
                    image_count=image_count,
                    current_training=current,
                    current_images_seen=config.images_seen,
                )
                panel.append(_advice_table_html(config, current, advice))
                warnings = list(advice.get("warnings", []))
                if warnings:
                    panel.append("<div class='muted'><b>风险提示</b><ul>")
                    panel.extend(f"<li>{_e(item)}</li>" for item in warnings)
                    panel.append("</ul></div>")
                panel.append(
                    "<p class='muted'>建议是保守起点，不是质量评分。实际数据偏置、过拟合和是否继续训练仍由 Preflight 与固定评测矩阵判断。</p>"
                )
                panel.append(
                    f"<form method='post' action='/configs/{_q(name)}/advisor'>"
                    f"<input type='hidden' name='_csrf' value='{_e(self.app.csrf)}'>"
                    f"<input type='hidden' name='image_count' value='{image_count}'>"
                    "<div class='toolbar'><button class='good'>应用首选起点（仅 images_seen / Rank / Alpha / LR）</button></div></form>"
                )

        panel.append("</div>")
        return "".join(panel)


def _advice_table_html(config, current: dict, advice: dict) -> str:
    recommended = advice["recommended"]
    preferred = advice["preferred_start"]
    rows = (
        ("images_seen", config.images_seen),
        ("network_dim", current.get("network_dim")),
        ("network_alpha", current.get("network_alpha")),
        ("unet_lr", current.get("unet_lr")),
    )
    output = [
        f"<p><b>目标：</b>{_e(config.target_type)} · <b>图片：</b>{advice['image_count']}</p>",
        "<table><tr><th>参数</th><th>当前</th><th>建议区间</th><th>首选起点</th></tr>",
    ]
    for key, current_value in rows:
        bounds = recommended[key]
        output.append(
            f"<tr><td>{_e(key)}</td><td>{_e(current_value)}</td>"
            f"<td>{_e(bounds['minimum'])} – {_e(bounds['maximum'])}</td>"
            f"<td><b>{_e(preferred[key])}</b></td></tr>"
        )
    output.append("</table>")
    return "".join(output)
