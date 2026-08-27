# LoRA Pipeline Repository Rules

1. Do not implement a custom SDXL training loop.
2. Use the pinned `sd-scripts` checkout as the training backend.
3. Do not implement an installer or environment setup wizard.
4. Do not automatically upgrade PyTorch, CUDA, NVIDIA components, or `sd-scripts`.
5. Preserve V100 / `sm_70` compatibility and the validated FP16 + SDPA path.
6. Never modify, rename, replace, or delete files under `projects/*/raw/`.
7. Never automatically delete user data or historical runs.
8. Do not hard-code checkpoint-specific behavior into generic Illustrious code.
9. Character and Style are separate concept profiles; CCIP identity logic is Character-only.
10. Hardware constraints, concept policy, and training strategy must remain separate.
11. Every long-running step must be resumable or clearly marked non-resumable.
12. Every generated LoRA must retain base SHA256, dataset/caption hashes, profile values, and training exposure metadata.
13. Prefer simple Python modules over services, daemons, databases, and hidden global state.
14. Do not add Kubernetes, Docker orchestration, web servers, or databases unless explicitly requested.
15. Pin external training dependencies only after successful validation.
16. Before changing a validated environment dependency, document why the change is required and rerun CPU and V100 smoke tests.
17. Never treat automatic evaluation metrics as image-quality ground truth.
18. Keep interactive and non-interactive workflows on the same service backend.
19. Avoid speculative abstractions until a second implementation requires them.
20. Optimize for maintainability, reproducibility, and actual LoRA quality rather than architectural complexity.
21. A completed pipeline step may be reused only when its effective input fingerprint still matches.
22. Fingerprints must include only inputs that can change that step's output; evaluation-only settings must not invalidate training.
23. Prepared datasets are immutable, content-addressed generations. Do not overwrite a prior generation.
24. Run evidence snapshots must not contain symlinks to mutable project files.
25. `images_seen` is the canonical comparison budget. Do not compare different physical batches at equal optimizer steps.
26. Record physical batch, gradient accumulation, effective batch, optimizer steps, target/actual images seen, and equivalent epochs.
27. Validation images must never enter `dataset.toml` or a training snapshot.
28. Existing caption modes must be explicit. `existing_passthrough` must preserve bytes; cleaning requires an explicitly named clean/hybrid mode.
29. Missing captions block preparation unless trigger-only fallback is explicitly enabled and recorded.
30. Tagger caches must be keyed by image content and backend/model configuration, not filenames or mutable runtime counters.
31. Evaluation cases require stable IDs and an explicit generation manifest. Never pair images to prompts by sorted filename order.
32. Screening and full evaluation are separate stages; full evaluation should operate on one or two explicit finalists.
33. Evaluation must not create `best.safetensors`. Only an explicit human `promote` action may create promoted artifacts.
34. Promotion must record checkpoint SHA256, recommended strength, selection method, and evaluation evidence.
35. `--force-step` and `--break-lock` are separate safety controls. A live process lock cannot be overridden.
36. Base checkpoint hashes are identity assertions. Do not silently replace a registered digest when file content changes.
37. Storage optimization may use a validated scratch directory, but final logs, configs, checkpoints, and metadata must be synchronized to persistent storage.
38. New behavior requires regression tests before merge; GPU-dependent tests remain explicit and must not run in the default CPU suite.
