# LoRA Pipeline

A resumable, interactive command-line pipeline for Character and Style LoRA
training on local Illustrious/SDXL checkpoints. Training is delegated to a
pinned [`sd-scripts`](https://github.com/kohya-ss/sd-scripts) checkout; this
repository does not implement a custom SDXL training loop.

The bundled hardware profile targets a 16 GB NVIDIA V100 (`sm_70`) with FP16,
PyTorch SDPA, 1024-area buckets, and profile-driven dataloader settings. The
pipeline does not install, upgrade, or calibrate PyTorch, CUDA, NVIDIA drivers,
or `sd-scripts`. Those are deployment responsibilities.

## Design guarantees

- Files under `projects/*/raw/` are never modified or deleted by the pipeline.
- Prepared datasets are immutable, content-addressed generations under
  `prepared/generations/<manifest-hash>/`.
- A completed step is reused only when its effective input fingerprint still
  matches. Input changes invalidate only dependent downstream steps.
- Character and Style are separate concept profiles. CCIP identity logic does
  not run for Style projects.
- Training budget is expressed canonically as `images_seen`, so Quality batch 1
  and Fast batch 2 are compared at equal image exposure rather than equal step
  count.
- Generated evaluation images have stable case IDs. They are never paired to
  prompts or metrics by sorted filename order.
- Evaluation never silently declares a winner. `best.safetensors` exists only
  after an explicit human `promote` action.
- Automatic identity, leakage, and coverage signals are review aids, not image
  quality ground truth.

## Requirements

- Linux and Python 3.11 or newer.
- A preconfigured NVIDIA environment. For the V100 profile,
  `torch.cuda.get_arch_list()` must contain `sm_70`.
- A local `sd-scripts` checkout pinned to the commit recorded in
  `environment/environment-info.json`.
- One or more local Illustrious/SDXL checkpoints that you are permitted to use.

Machine-specific configuration is intentionally excluded from Git. Copy the
example environment record only when provisioning the host:

```bash
cp environment/environment-info.example.json environment/environment-info.json
```

The runtime pipeline never edits this environment or installs packages.

## Register base checkpoints

```bash
./lora base add one_obsession_v19 /models/one-obsession-v19.safetensors
./lora base inspect one_obsession_v19
./lora base verify one_obsession_v19   # explicit full-file verification
./lora base list
```

The registry caches a full SHA256 together with a file stat signature. If the
checkpoint content changes, preflight blocks instead of silently accepting the
new file under the old base identity.

## Create a project

Interactive:

```bash
./lora new
```

Non-interactive:

```bash
./lora new \
  --name example-character \
  --concept character \
  --base one_obsession_v19 \
  --dataset /datasets/example-character \
  --trigger zz_example_character \
  --strategy quality \
  --images-seen 4000 \
  --yes
```

A project contains:

```text
projects/<name>/
├── raw/                         # immutable user input
├── validation/                  # optional holdout images; never trained
├── prepared/
│   ├── current.json
│   └── generations/<hash>/      # immutable image/caption generation
├── review/
├── cache/
├── runs/
└── project.yaml
```

## Caption modes

Caption handling is explicit:

```bash
./lora caption PROJECT --mode generate
./lora caption PROJECT --mode existing_passthrough
./lora caption PROJECT --mode existing_taglist_clean
./lora caption PROJECT --mode hybrid
./lora caption PROJECT --mode skip
```

- `generate`: tag with the configured anime tagger and apply concept policy.
- `existing_passthrough`: preserve existing `.txt` bytes through preparation.
- `existing_taglist_clean`: explicitly treat existing captions as Booru tag
  lists and normalize them.
- `hybrid`: keep existing content and add tagger suggestions; conflicts are
  written to review manifests.
- `skip`: use existing sidecars during preparation without running the caption
  step.

Raw tagger outputs are cached by image content plus backend/model configuration,
so adding or changing one image does not retag the entire dataset. Character tag
outputs are also used to flag possible mixed-character images for review.

Missing captions block preparation by default. Trigger-only fallback must be
explicit:

```bash
./lora prepare PROJECT --allow-trigger-only
```

This is recorded as a high-risk strategy in manifests and preflight output.

## Interactive and resumable workflow

```bash
./lora open PROJECT
```

The wizard offers Run, Review, or Skip for optional stages while calling the
same service functions as the non-interactive CLI. A typical explicit flow is:

```bash
./lora inspect PROJECT
./lora dedup PROJECT
./lora identity PROJECT                 # Character only
./lora caption PROJECT --mode generate
./lora review PROJECT
./lora prepare PROJECT
./lora preflight PROJECT
./lora train PROJECT
./lora evaluate PROJECT --stage screening
```

Or run the remaining stages:

```bash
./lora run PROJECT --caption-mode generate
```

Project state is written atomically. Interrupting a long training command records
`interrupted`; when `sd-scripts` saved state exists, resume the same run:

```bash
./lora train PROJECT --resume RUN_ID
```

`--force-step` reruns a step. `--break-lock` is deliberately separate and only
breaks a lock proven stale on the current host. A live process lock cannot be
overridden.

## Preflight

Preflight checks:

- base path, SHA256 identity, and stat cache;
- immutable prepared generation and selected image count;
- validation split isolation;
- missing/corrupt images and caption availability;
- CLIP-L and CLIP-G token counts using the SDXL tokenizers;
- configured bucket area versus the V100 envelope;
- resolved `images_seen`, optimizer steps, effective batch, and equivalent
  epochs;
- persistent and optional scratch storage writability/free space;
- unresolved duplicate and identity review items.

If tokenizer assets are not locally cached, preflight reports a clearly marked
heuristic fallback. Cache both SDXL tokenizers and rerun preflight before relying
on exact truncation checks.

## Training strategies and accounting

Hardware capability, concept policy, and training strategy remain separate:

```text
hardware profile × character/style profile × quality/fast/cached strategy
```

The canonical budget is image exposure:

```text
effective_batch = physical_batch × gradient_accumulation
optimizer_steps = ceil(target_images_seen / effective_batch)
actual_images_seen = optimizer_steps × effective_batch
```

Every run records physical/effective batch, optimizer steps, target and actual
images seen, equivalent epochs, dataset/caption hashes, base SHA256,
`sd-scripts` commit, generated TOML, CLI command, elapsed time, throughput,
peak VRAM, average GPU utilization, and storage paths.

Dataloader workers and an optional local scratch path are profile settings:

```yaml
data_loader:
  workers: 0
  persistent_workers: false
  cpu_threads_per_process: 1

storage:
  scratch_root: null
```

Set `scratch_root` only after validating a local SSD/NVMe path on the NAS. Logs
and checkpoints are synchronized back to persistent project storage.

## Evaluation and promotion

Evaluation is intentionally two-stage.

### 1. Screening

All candidate checkpoints are evaluated with a small canonical prompt matrix and
strengths `0.6`, `0.8`, and `1.0`, including trigger-on and trigger-off pairs:

```bash
./lora evaluate PROJECT --stage screening --run RUN_ID
```

### 2. Full finalist evaluation

Select one or two checkpoint finalists after reviewing screening sheets:

```bash
./lora evaluate PROJECT \
  --stage full \
  --run RUN_ID \
  --checkpoint candidate-000800 \
  --checkpoint candidate-001000
```

A full evaluation with more than two candidates is rejected unless finalists
are explicit. Each stage produces:

```text
samples/<stage>/generation-manifest.json
contact-sheets/<stage>/checkpoint-strength.jpg
contact-sheets/<stage>/prompt-checkpoint.jpg
contact-sheets/<stage>/trigger-leakage.jpg
metrics/evaluation-<stage>.json
report-<stage>.html
```

Character evaluation uses holdout images under `validation/` for CCIP identity
when available; otherwise the report clearly marks training-image fallback.
Style evaluation uses a cross-content matrix and independent warnings for
subject, portrait, background, multi-subject, and aspect-ratio bias.

Evaluation does **not** create `best.safetensors`. After manual review:

```bash
./lora promote PROJECT \
  --run RUN_ID \
  --checkpoint candidate-000800.safetensors \
  --strength 0.7
```

Promotion writes:

```text
best.safetensors
best.yaml
```

`best.yaml` records the manual selection, checkpoint SHA256, recommended
strength, base identity, training exposure, and evaluation evidence.

## Shared-GPU hook

A host can optionally provide shell-free reservation commands in the ignored
environment record:

```json
{
  "gpu_lease": {
    "acquire_command": ["/usr/local/bin/gpu-lease", "acquire"],
    "release_command": ["/usr/local/bin/gpu-lease", "release"]
  }
}
```

Both commands are required and are executed as argument arrays, never through a
shell.

## Validation

CPU tests and source compilation run in GitHub Actions:

```bash
python -m compileall -q pipeline tests
python -m pytest -q
```

GPU tests remain explicit because they require the configured V100 NAS:

```bash
cp environment/smoke/train.example.toml environment/smoke/train.toml
cp environment/smoke/dataset-batch1.example.toml environment/smoke/dataset-batch1.toml
./environment/smoke/run_sd_scripts_smoke.sh 1
```

The smoke test validates the deployed environment; it is not a per-dataset
calibration stage.
