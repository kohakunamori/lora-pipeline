from __future__ import annotations

import yaml

from pipeline.tokenizers import TokenCounts, count_project_tokens


class _Tokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(513))}


def test_project_token_count_uses_anima_qwen3(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    (root / "bases").mkdir(parents=True)
    base = root / "anima.safetensors"
    qwen3 = root / "qwen3.safetensors"
    vae = root / "vae.safetensors"
    for path in (base, qwen3, vae):
        path.write_bytes(b"fixture")
    (root / "bases" / "registry.yaml").write_text(
        yaml.safe_dump({"bases": {"anima": {
            "name": "Anima", "path": str(base), "family": "anima",
            "prediction_type": "rectified_flow", "enabled": True,
            "components": {"qwen3": {"path": str(qwen3)}, "vae": {"path": str(vae)}},
        }}}),
        encoding="utf-8",
    )
    monkeypatch.setattr("pipeline.tokenizers._load_qwen3_tokenizer", lambda path: _Tokenizer())

    counts = count_project_tokens("zz_style, portrait", {"base": "anima"}, root=root)

    assert counts == TokenCounts(
        clip_l=0, clip_g=0, exact=True, backend="transformers.Qwen3Tokenizer", qwen3=513
    )
    assert counts.maximum == 513
    assert counts.as_dict()["qwen3"] == 513
