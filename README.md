# LoRA Pipeline

A resumable command-line pipeline for Character and Style LoRA training on
Illustrious/SDXL checkpoints. Training is delegated to the pinned
[`sd-scripts`](https://github.com/kohya-ss/sd-scripts) backend; this repository
does not implement a custom SDXL training loop.

The default hardware profile targets a 16 GB NVIDIA V100 (`sm_70`) with FP16
and PyTorch SDPA. Hardware constraints, concept profiles, and training
strategies remain separate so a run can be reproduced and audited.

## What it provides

- Project-owned, immutable `raw/` datasets and resumable pipeline state.
- Separate Character and Style inspection, preparation, and evaluation paths.
- Interactive and non-interactive workflows on the same Python backend.
- `sd-scripts` TOML generation, dataset snapshots, checkpoint accounting, and
  GPU telemetry.
- Base checkpoint SHA256 verification and immutable run metadata.
- Fixed evaluation matrices, contact sheets, and HTML reports. Automatic
  signals are treated as review aids, not image-quality ground truth.
- Optional shell-free acquire/release commands for hosts that share a GPU.

The pipeline does not install or upgrade PyTorch, CUDA, NVIDIA drivers, or the
training backend. It also never deletes source datasets automatically.

## Requirements

- Linux and Python 3.11 or newer.
- An NVIDIA environment already validated for the target GPU. For the bundled
  V100 profile, `torch.cuda.get_arch_list()` must include `sm_70`.
- A local checkout of `sd-scripts` at commit
  `37a1cbbc5725ed2a3575506e7bd2001c9908ac92`.
- A local Illustrious/SDXL checkpoint that you are permitted to use.

Pinned Python packages are recorded in `environment/requirements.lock` and
`environment/constraints.txt`. Review them against your existing CUDA/PyTorch
environment before installing anything; the repository intentionally provides
no installer or automatic dependency upgrade path.

## Local configuration

Machine-specific paths are deliberately excluded from Git. Start from the
example environment record:

```bash
cp environment/environment-info.example.json environment/environment-info.json
```

Edit `python_path`, `sd_scripts_path`, driver/runtime versions, and validation
flags to match the host. The active `environment-info.json` is embedded into
each training run's metadata but is ignored by Git.

Register a local base checkpoint and persist its SHA256:

```bash
./lora base add example_base /models/example-base.safetensors
./lora base inspect example_base
./lora base list
```

This creates the ignored `bases/registry.yaml`; checkpoint paths and hashes are
therefore not published accidentally. `bases/registry.example.yaml` documents
the initial empty shape.

Run the environment checks before creating a project:

```bash
./lora doctor
./lora new
```

An equivalent non-interactive project creation looks like this:

```bash
./lora new \
  --name example-character \
  --concept character \
  --base example_base \
  --dataset /datasets/example-character \
  --trigger example_trigger \
  --strategy quality \
  --steps 1000 \
  --yes

./lora run example-character --caption-mode existing --yes
```

Use `--caption-mode generate` for the WD EVA02-Large Tagger v3 path, or
`existing` when every source image already has a same-stem `.txt` caption.
Model caches live under `.cache/` and are also ignored by Git.

## Resume and expert controls

Every completed or failed step is recorded atomically in `project.yaml`:

```bash
./lora open example-character
./lora run example-character --yes
./lora caption example-character --mode existing --force
./lora train example-character --dry-run
./lora train example-character --skip-preflight --yes
```

Optional workflow steps have explicit `--skip-*` switches. Inspection,
preparation, training configuration generation, and training are not generally
skippable. Preflight verifies the registered base SHA256, prepared images,
captions and token budget, the selected hardware envelope, disk space, and
output writability.

## Shared-GPU hook

By default, the public pipeline assumes the selected GPU is already available.
A host-specific reservation can be configured in the ignored
`environment/environment-info.json` file:

```json
{
  "gpu_lease": {
    "acquire_command": ["/usr/local/bin/gpu-lease", "acquire"],
    "release_command": ["/usr/local/bin/gpu-lease", "release"]
  }
}
```

Commands are executed directly as argument arrays, never through a shell. Both
commands are required. Release is attempted in the context-manager cleanup path
even when training fails. The hook is intentionally generic; queue policy and
GPU ownership remain responsibilities of the host.

## Outputs and evaluation

Each run keeps immutable evidence under
`projects/<name>/runs/<run-id>/`, including generated `train.toml` and
`dataset.toml`, environment snapshot, base SHA256, dataset and caption hashes,
command line, `sd-scripts` commit, logs, GPU telemetry, exposure accounting,
and candidate checkpoints.

Full evaluation can produce:

```text
best.safetensors
best.yaml
contact-sheet.jpg
report.html
metrics/evaluation.json
```

`best` is provisional until a human reviews the contact sheet. Identity,
leakage, and coverage signals are auxiliary and are never presented as an
objective quality score.

## Validation

The default suite excludes tests that require a real GPU:

```bash
python -m pytest
```

For an explicit `sd-scripts` V100 smoke, copy the example TOML files to their
ignored active names, replace the placeholder paths, ensure exclusive GPU
access, and run:

```bash
cp environment/smoke/train.example.toml environment/smoke/train.toml
cp environment/smoke/dataset-batch1.example.toml environment/smoke/dataset-batch1.toml
./environment/smoke/run_sd_scripts_smoke.sh 1
```

Batch 2 has a separate example dataset config. Validate it on your own host
before relying on that envelope.

## Configuration precedence

```text
hard safety constraints
  -> hardware profile
  -> concept profile
  -> training strategy
  -> project overrides
  -> explicit CLI overrides
```

Project `raw/` trees are immutable after creation; curation and exclusions
affect only manifests and `prepared/`. Base weights, datasets, generated LoRAs,
machine-specific configuration, and cached model files are not included in this
repository.
