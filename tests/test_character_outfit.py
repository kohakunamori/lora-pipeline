from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from PIL import Image

from pipeline.dataset.caption_cleaner import clean_caption, parse_caption
from pipeline.dataset_workspace import DatasetWorkspace
from pipeline.evaluation.outfit import outfit_retention_proxy, outfit_trigger_leakage_proxy
from pipeline.evaluation.service import _build_cases
from pipeline.interactive_outfit import InteractiveWizard
from pipeline.materialization import run as materialize
from pipeline.materialization import caption
from pipeline.models import GeneratedImage, GenerationCase, PipelineError
from pipeline.prepared import load_current_generation
from pipeline.steps import preflight
from pipeline.training_config import TrainingConfig, create_project_from_training_config


def _repo_root(tmp_path: Path, monkeypatch) -> Path:
    source_root = Path(__file__).resolve().parents[1]
    root = tmp_path / "repo"
    root.mkdir()
    shutil.copytree(source_root / "profiles", root / "profiles")
    (root / "bases").mkdir()
    checkpoint = root / "base.safetensors"
    checkpoint.write_bytes(b"base")
    (root / "bases" / "registry.yaml").write_text(
        yaml.safe_dump(
            {
                "bases": {
                    "base": {
                        "name": "Base",
                        "path": str(checkpoint),
                        "family": "illustrious_sdxl",
                        "prediction_type": "epsilon",
                        "sha256": None,
                        "enabled": True,
                        "generation_defaults": {
                            "sampler": "euler_a",
                            "cfg": 4.5,
                            "steps": 2,
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LORA_PIPELINE_ROOT", str(root))
    return root


def _workspace(root: Path, *, caption_text: str | None = "portrait, smile") -> DatasetWorkspace:
    source = root / "source"
    source.mkdir(exist_ok=True)
    image = source / "a.png"
    Image.new("RGB", (768, 1024), "red").save(image)
    if caption_text is not None:
        image.with_suffix(".txt").write_text(caption_text + "\n", encoding="utf-8")
    workspace = DatasetWorkspace.create("demo", concept_type="character", root=root)
    workspace.add_source_from_directory(source, kind="image_directory", label="source")
    return workspace


def _outfit_config(root: Path, **kwargs) -> TrainingConfig:
    values = {
        "concept_type": "character",
        "target_type": "character_outfit",
        "base": "base",
        "trigger": "misuzu_nic26",
        "anchor_tags": ["hataya misuzu"],
        "evaluation": {"subject_prompt": "1girl"},
        "root": root,
    }
    values.update(kwargs)
    return TrainingConfig.create("outfit", **values)


def test_outfit_config_separates_trigger_anchor_and_runtime_concept(tmp_path, monkeypatch) -> None:
    root = _repo_root(tmp_path, monkeypatch)
    config = _outfit_config(root)

    assert config.concept_type == "character"
    assert config.target_type == "character_outfit"
    assert config.trigger == "misuzu_nic26"
    assert config.anchor_tags == ["hataya misuzu"]
    assert config.effective_evaluation()["subject_prompt"] == "1girl, hataya misuzu"

    runtime = config.runtime_overrides()
    assert runtime["caption"]["anchor_tags"] == ["hataya misuzu"]
    assert "different outfit" not in runtime["evaluation"]["prompts"]
    assert "dynamic pose" in runtime["evaluation"]["prompts"]

    workspace = _workspace(root)
    state = create_project_from_training_config(
        workspace, config, project_name="run-outfit", root=root
    )
    project = state.payload["project"]
    assert project["type"] == "character"
    assert project["training_target_type"] == "character_outfit"
    assert project["caption_anchor_tags"] == ["hataya misuzu"]
    assert project["evaluation"]["subject_prompt"] == "1girl, hataya misuzu"


def test_outfit_config_rejects_ambiguous_trigger_missing_anchor_and_eval_contamination(
    tmp_path, monkeypatch
) -> None:
    root = _repo_root(tmp_path, monkeypatch)
    with pytest.raises(PipelineError, match="cannot contain a comma"):
        TrainingConfig.create(
            "bad-trigger",
            concept_type="character",
            target_type="character_outfit",
            base="base",
            trigger="hataya misuzu, misuzu_nic26",
            anchor_tags=["hataya misuzu"],
            root=root,
        )
    with pytest.raises(PipelineError, match="requires at least one character anchor"):
        TrainingConfig.create(
            "no-anchor",
            concept_type="character",
            target_type="character_outfit",
            base="base",
            trigger="misuzu_nic26",
            root=root,
        )
    with pytest.raises(PipelineError, match="must not contain the LoRA trigger"):
        TrainingConfig.create(
            "bad-eval",
            concept_type="character",
            target_type="character_outfit",
            base="base",
            trigger="misuzu_nic26",
            anchor_tags=["hataya misuzu"],
            evaluation={"subject_prompt": "1girl, misuzu_nic26"},
            root=root,
        )


def test_caption_cleaner_pins_trigger_and_anchor_ahead_of_variable_tags() -> None:
    result = clean_caption(
        [
            "hataya_misuzu",
            "smile",
            "very detailed beach background with many objects",
            "swimsuit",
        ],
        trigger="misuzu_nic26",
        anchor_tags=["hataya misuzu"],
        max_token_length=9,
    )
    assert result.tags[:2] == ("misuzu_nic26", "hataya misuzu")
    assert result.tags.count("hataya misuzu") == 1
    assert result.tags.count("misuzu_nic26") == 1
    assert result.pruned


def test_caption_transform_injects_outfit_fixed_prefix(tmp_path, monkeypatch) -> None:
    root = _repo_root(tmp_path, monkeypatch)
    workspace = _workspace(root, caption_text="portrait, smile, swimsuit")
    config = _outfit_config(root)
    state = create_project_from_training_config(
        workspace, config, project_name="run-caption", root=root
    )

    caption.run(state, mode="existing_taglist_clean")
    generated = list((state.project_dir / "review" / "captions" / "generated").rglob("*.txt"))
    assert len(generated) == 1
    tags = parse_caption(generated[0].read_text(encoding="utf-8"))
    assert tags[:2] == ["misuzu_nic26", "hataya misuzu"]
    assert "swimsuit" in tags


def test_trigger_only_fallback_keeps_character_anchor(tmp_path, monkeypatch) -> None:
    root = _repo_root(tmp_path, monkeypatch)
    workspace = _workspace(root, caption_text=None)
    config = _outfit_config(root)
    state = create_project_from_training_config(
        workspace, config, project_name="run-fallback", root=root
    )

    materialize(state, allow_trigger_only=True)
    generation = load_current_generation(state.project_dir)
    record = generation.manifest["images"][0]
    text = (generation.root / record["caption"]).read_text(encoding="utf-8").strip()
    assert text == "misuzu_nic26, hataya misuzu"


def test_preflight_blocks_passthrough_that_omits_outfit_context(tmp_path, monkeypatch) -> None:
    root = _repo_root(tmp_path, monkeypatch)
    workspace = _workspace(root, caption_text="portrait, smile")
    config = _outfit_config(
        root,
        workflow={"caption_mode": "existing_passthrough"},
    )
    state = create_project_from_training_config(
        workspace, config, project_name="run-passthrough", root=root
    )

    materialize(state, caption_mode="existing_passthrough")
    with pytest.raises(PipelineError) as error:
        preflight.run(state, minimum_free_gib=0)
    message = str(error.value)
    assert "missing the LoRA trigger" in message
    assert "missing required character anchors" in message


def test_evaluation_case_builder_rejects_trigger_in_subject_prompt() -> None:
    with pytest.raises(PipelineError, match="must not contain the LoRA trigger"):
        _build_cases(
            [Path("candidate.safetensors")],
            strengths=[0.8],
            prompt_ids=["portrait"],
            trigger="misuzu_nic26",
            concept_type="character",
            subject_prompt="1girl, hataya misuzu, misuzu_nic26",
            seed=42,
        )


def _generated(contains_trigger: bool, *, prompt_id: str = "portrait") -> GeneratedImage:
    case = GenerationCase(
        case_id=f"case-{prompt_id}-{contains_trigger}",
        checkpoint=Path("candidate.safetensors"),
        checkpoint_label="candidate",
        strength=0.8,
        prompt_id=prompt_id,
        prompt=(
            f"misuzu_nic26, 1girl, hataya misuzu, {prompt_id}"
            if contains_trigger
            else f"1girl, hataya misuzu, {prompt_id}"
        ),
        negative_prompt="",
        seed=42,
        contains_trigger=contains_trigger,
    )
    return GeneratedImage(case=case, path=Path(f"{case.case_id}.png"))


def test_outfit_metrics_require_visual_review_and_measure_aligned_pair_coverage() -> None:
    generated = [
        _generated(True, prompt_id="portrait"),
        _generated(False, prompt_id="portrait"),
        _generated(True, prompt_id="full body"),
        _generated(False, prompt_id="full body"),
    ]
    retention = outfit_retention_proxy(generated)
    leakage = outfit_trigger_leakage_proxy(
        generated, anchor_tags=["hataya misuzu"]
    )

    assert retention["status"] == "manual_review_required"
    assert retention["positive_samples"] == 2
    assert leakage["status"] == "manual_review_required"
    assert leakage["aligned_on_off_pairs"] == 2
    assert leakage["pair_coverage_fraction"] == 1.0
    assert leakage["anchor_tags"] == ["hataya misuzu"]


def test_interactive_trigger_reprompts_immediately_on_comma(monkeypatch) -> None:
    wizard = InteractiveWizard()
    answers = iter(["hataya misuzu, misuzu_nic26", "misuzu_nic26"])
    monkeypatch.setattr(wizard, "_ask_text", lambda *args, **kwargs: next(answers))

    assert wizard._ask_training_trigger("demo") == "misuzu_nic26"
