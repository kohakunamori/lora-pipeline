# LoRA Pipeline

A resumable, CLI/TUI-first pipeline for Character and Style LoRA training on
local Illustrious/SDXL checkpoints. Training is delegated to a pinned
[`sd-scripts`](https://github.com/kohya-ss/sd-scripts) checkout; this repository
does not implement a custom SDXL training loop.

The bundled hardware profile targets a 16 GB NVIDIA V100 (`sm_70`) with FP16,
PyTorch SDPA, 1024-area buckets, and profile-driven dataloader settings. The
pipeline does not install, upgrade, or calibrate PyTorch, CUDA, NVIDIA drivers,
or `sd-scripts`. Those are deployment responsibilities.

> The former Web UI, Web worker, and Web job surface has been removed and is no
> longer maintained. Supported interactive use is the terminal UI plus CLI/core
> APIs.

## Architecture

The workflow is deliberately split into three ownership boundaries:

```text
Dataset Workspace                Training Project                 Results
-----------------                ----------------                 -------
import / video ingest     ->      materialize              ->      screening
inspect / audit                   preflight                       full evaluation
dedup / identity                 train                           human promote
tag / edit / exclude
        |
        +-- immutable Dataset snapshot + TrainingConfig snapshot
```

Dataset curation is not replayed as a Project state machine. A Project freezes
the selected Dataset and TrainingConfig inputs, materializes the effective
training images/captions, validates them, and trains. Evaluation and promotion
are repeatable operations on completed Runs and are not required for the
training lifecycle to be complete.

The canonical Project state machine is exactly:

```text
materialize -> preflight -> train
```

Historical Project YAML may contain `inspect`, `dedup`, `identity`, `caption`,
`review`, `prepare`, or `evaluate` records. They are migrated into opaque
compatibility history when loaded and are never replayed by normal orchestration.
`prepare` remains only as a deprecated alias for `materialize` for old callers.

## Design guarantees

- Files under `projects/*/raw/` are immutable frozen inputs.
- Dataset inspection, dedup, identity analysis, tagging, editing, and exclusion
  belong to `DatasetWorkspace`, not Project stages.
- Fresh Dataset curation evidence may be copied into a Project snapshot for
  preflight/reporting, but it is never registered as a Project step.
- Materialized datasets are immutable, content-addressed generations under
  `prepared/generations/<manifest-hash>/`.
- A completed Project stage is reused only while its effective input fingerprint
  still matches. Input changes invalidate only dependent downstream stages.
- Character and Style are separate concept profiles. CCIP identity logic does
  not run for Style datasets.
- Training budget is expressed canonically as `images_seen`, so different batch
  strategies are compared at equal image exposure rather than equal optimizer
  step count.
- Evaluation is Run-scoped. Validation images and evaluation settings do not
  mutate or invalidate the Project training lifecycle.
- Generated evaluation images use stable case IDs and are never paired to
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

## Entry points

Interactive terminal UI:

```bash
./lora
```

CLI:

```bash
./lora --help
```

There is intentionally no Web UI entry point.

## Register base checkpoints

```bash
./lora base add one_obsession_v19 /models/one-obsession-v19.safetensors
./lora base inspect one_obsession_v19
./lora base verify one_obsession_v19
./lora base list
```

The registry caches a full SHA256 together with a file stat signature. If the
checkpoint content changes, preflight blocks instead of silently accepting the
new file under the old base identity.

## Dataset Workspace

The preferred workflow is to curate reusable data before creating a Training
Project. Import sources, audit images, resolve duplicates and character identity
issues where applicable, generate/edit captions, and exclude bad samples in the
Dataset Workspace. Video extraction and subject selection also terminate in the
Dataset layer.

A Dataset snapshot records the active image set, hashes, captions, inspection
metadata, and reusable curation analyses. Creating a Project freezes that
snapshot so later Dataset edits do not silently alter an existing run.

Fresh duplicate/identity analyses are frozen as `project.dataset_curation`
evidence. Their manifests can be copied under the Project `review/` directory so
preflight and reports can consume them, but no `steps.dedup` or `steps.identity`
record is created.

Legacy direct Project creation remains available:

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

For this path, inspection is frozen immediately beside the immutable `raw/`
snapshot and is not a Project stage.

A Project contains:

```text
projects/<name>/
├── raw/                         # immutable frozen input
├── validation/                  # optional holdout; never trained
├── dataset-manifest.json        # frozen inspection evidence
├── prepared/
│   ├── current.json
│   └── generations/<hash>/      # immutable effective training generation
├── review/                      # frozen Dataset/review evidence where available
├── cache/
├── runs/
└── project.yaml
```

## Materialization

Caption generation and normalization are input transforms of materialization,
not separate Project lifecycle stages. Materialization combines the frozen
Dataset snapshot with TrainingConfig-specific trigger/anchors/policy and writes a
content-addressed prepared generation.

Supported caption policies:

- `generate`: run the configured anime tagger and apply concept policy.
- `existing_passthrough`: preserve existing `.txt` bytes.
- `existing_taglist_clean`: normalize existing Booru-style tag lists.
- `hybrid`: preserve existing content and add useful tagger suggestions.
- `skip`: use raw sidecars without running a caption transform.

Raw tagger outputs are cached by image content plus backend/model configuration,
so changing one image does not force a complete retag.

Missing captions block materialization by default. Trigger-only fallback must be
explicit:

```bash
./lora materialize PROJECT --allow-trigger-only
```

The fallback is recorded in the prepared manifest and surfaced by preflight.

For old automation only, `prepare` is accepted as a deprecated alias of
`materialize`; new scripts should not use it.

## Training lifecycle

Run all remaining Project stages with:

```bash
./lora run PROJECT --caption-mode existing_taglist_clean
```

Or invoke canonical stages explicitly:

```bash
./lora materialize PROJECT
./lora preflight PROJECT
./lora train PROJECT
```

Project state is written atomically. Interrupting long training records the run
as interrupted; when `sd-scripts` saved state exists, resume it with:

```bash
./lora train PROJECT --resume RUN_ID
```

`--force-step` reruns a Project stage. `--break-lock` is deliberately separate
and only breaks a lock proven stale on the current host. A live process lock
cannot be overridden.

## Preflight

Preflight checks the frozen effective training inputs, including:

- base path, SHA256 identity, and stat cache;
- immutable materialized generation and selected image count;
- Dataset inspection and frozen curation evidence where available;
- validation split isolation from training;
- missing/corrupt images and caption availability;
- CLIP-L and CLIP-G token counts using SDXL tokenizers;
- configured bucket area versus the hardware envelope;
- resolved `images_seen`, optimizer steps, effective batch, and equivalent
  epochs;
- persistent and optional scratch storage writability/free space;
- target-specific guardrails.

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

Every Run records physical/effective batch, optimizer steps, target and actual
images seen, equivalent epochs, Dataset/caption hashes, base SHA256,
`sd-scripts` commit, generated TOML, CLI command, elapsed time, throughput,
peak VRAM, average GPU utilization, and storage paths.

Dataloader workers and optional local scratch storage are profile settings:

```yaml
data_loader:
  workers: 0
  persistent_workers: false
  cpu_threads_per_process: 1

storage:
  scratch_root: null
```

Set `scratch_root` only after validating a local SSD/NVMe path. Logs and
checkpoints are synchronized back to persistent Project storage.

## Results: evaluation and promotion

Results are Run-scoped and deliberately outside Project state.

### Screening

Evaluate candidate checkpoints with the small canonical matrix, including
trigger-on/off pairs:

```bash
./lora evaluate PROJECT --stage screening --run RUN_ID
```

### Full finalist evaluation

After screening, choose one or two finalists:

```bash
./lora evaluate PROJECT \
  --stage full \
  --run RUN_ID \
  --checkpoint candidate-000800 \
  --checkpoint candidate-001000
```

A full evaluation with more than two candidates is rejected unless finalists
are explicit. Results are stored beneath the Run, including:

```text
samples/<stage>/generation-manifest.json
contact-sheets/<stage>/checkpoint-strength.jpg
contact-sheets/<stage>/prompt-checkpoint.jpg
contact-sheets/<stage>/trigger-leakage.jpg
metrics/evaluation-<stage>.json
report-<stage>.html
```

Character evaluation uses holdout images under `validation/` for identity
comparison when available; otherwise reports clearly mark training-image
fallback. Style evaluation uses cross-content tests and independent Dataset-bias
warnings.

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

`best.yaml` records the human selection, checkpoint SHA256, recommended strength,
base identity, training exposure, and evaluation evidence.

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

Both commands are required and executed as argument arrays, never through a
shell.

## Validation

CPU tests and source compilation run in GitHub Actions:

```bash
python -m compileall -q pipeline tests
python -m pytest -q
```

GPU tests remain explicit because they require the configured V100 environment:

```bash
cp environment/smoke/train.example.toml environment/smoke/train.toml
cp environment/smoke/dataset-batch1.example.toml environment/smoke/dataset-batch1.toml
./environment/smoke/run_sd_scripts_smoke.sh 1
```

The smoke test validates the deployed environment; it is not a per-Dataset
calibration stage.
