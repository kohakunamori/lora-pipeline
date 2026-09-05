# LoRA Pipeline

A resumable, interactive pipeline for Character and Style LoRA training on local
Illustrious/SDXL checkpoints. Training is delegated to a pinned
[`sd-scripts`](https://github.com/kohya-ss/sd-scripts) checkout; this repository
does not implement a custom SDXL training loop.

The bundled hardware profile targets a 16 GB NVIDIA V100 (`sm_70`) with FP16,
PyTorch SDPA, 1024-area buckets, and profile-driven dataloader settings. The
pipeline does not install, upgrade, or calibrate PyTorch, CUDA, NVIDIA drivers,
or `sd-scripts`. Those are deployment responsibilities.

## Architecture

The workflow is deliberately split into three lifecycles:

```text
Dataset Workspace                Training Project                 Results
-----------------                ----------------                 -------
import / video ingest     ->      prepare (materialize)    ->      screening
inspect / audit                   preflight                       full evaluation
dedup / identity                 train                           human promote
tag / edit / exclude
        |
        +-- immutable Dataset snapshot + TrainingConfig snapshot
```

Dataset curation is not replayed as a Project training state machine. A Project
freezes the selected dataset/config inputs, materializes the effective training
captions and images, validates them, and trains. Evaluation is a repeatable
operation on a completed run and is not required for the training lifecycle to
be considered complete.

## Design guarantees

- Files under `projects/*/raw/` are never modified or deleted by the pipeline.
- Dataset inspection/curation belongs to `DatasetWorkspace`; frozen Projects do
  not normally rerun inspect, duplicate, identity, or review stages.
- Prepared datasets are immutable, content-addressed generations under
  `prepared/generations/<manifest-hash>/`.
- A completed training stage is reused only when its effective input fingerprint
  still matches. Input changes invalidate only dependent downstream stages.
- Character and Style are separate concept profiles. CCIP identity logic does
  not run for Style datasets.
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

## Dataset curation

The preferred workflow is to curate reusable data in a Dataset Workspace before
creating a training run. Import sources, audit images, resolve duplicates and
character identity issues where applicable, generate/edit captions, and exclude
bad samples there. Video extraction and identity selection also terminate in the
Dataset layer rather than becoming Project training stages.

A Dataset snapshot records the active image set, hashes, captions, inspection
metadata, and reusable curation analyses. Creating a Project from that Dataset
freezes the snapshot so later Dataset edits do not silently change an existing
training run.

Legacy direct Project creation remains available for compatibility:

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

For this path, image inspection is frozen when the Project raw snapshot is
created; it is no longer replayed by `./lora run`.

A project contains:

```text
projects/<name>/
├── raw/                         # immutable frozen input
├── validation/                  # optional holdout images; never trained
├── prepared/
│   ├── current.json
│   └── generations/<hash>/      # immutable effective image/caption generation
├── review/                      # compatibility/review artifacts
├── cache/
├── runs/
└── project.yaml
```

## Caption materialization

Caption generation and normalization are inputs to materialization, not a
separate Project lifecycle stage. Guided training supplies the selected caption
policy to `prepare`, which writes the effective captions into the immutable
prepared generation.

Supported policies remain:

- `generate`: tag with the configured anime tagger and apply concept policy.
- `existing_passthrough`: preserve existing `.txt` bytes.
- `existing_taglist_clean`: treat existing captions as Booru tag lists and
  normalize them.
- `hybrid`: keep existing content and add tagger suggestions; conflicts are
  recorded as review artifacts.
- `skip`: use raw sidecars without running a caption transform.

The explicit legacy utility remains available for diagnostics or migration:

```bash
./lora caption PROJECT --mode existing_taglist_clean
```

It is not a prerequisite state for `prepare`. `prepare` fingerprints the raw
images/captions and effective caption policy directly, so changing only the
legacy caption-step record does not invalidate an otherwise identical prepared
training set.

Raw tagger outputs are cached by image content plus backend/model configuration,
so adding or changing one image does not retag the entire dataset.

Missing captions block materialization by default. Trigger-only fallback must be
explicit:

```bash
./lora prepare PROJECT --allow-trigger-only
```

This is recorded as a high-risk strategy in manifests and preflight output.

## Training lifecycle

The normal Project training state machine is intentionally small:

```text
prepare -> preflight -> train
```

`prepare` is the current compatibility name for the materialization stage. It
combines the frozen Dataset snapshot with the TrainingConfig-specific trigger,
anchors and caption policy, then writes a content-addressed prepared generation.

Run all remaining training stages with:

```bash
./lora run PROJECT --caption-mode existing_taglist_clean
```

Or invoke the stages explicitly:

```bash
./lora prepare PROJECT
./lora preflight PROJECT
./lora train PROJECT
```

`inspect`, `dedup`, `identity`, `caption`, and `review` records may still exist
inside old Projects for migration compatibility, but normal run orchestration and
Project navigation do not replay them.

Project state is written atomically. Interrupting a long training command records
`interrupted`; when `sd-scripts` saved state exists, resume the same run:

```bash
./lora train PROJECT --resume RUN_ID
```

`--force-step` reruns a stage. `--break-lock` is deliberately separate and only
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
- inherited Dataset curation warnings and target-specific guardrails where
  available.

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

## Results: evaluation and promotion

Evaluation is intentionally outside the training state machine. A completed run
can be evaluated repeatedly without reopening or invalidating training.

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
