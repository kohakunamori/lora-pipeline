# LoRA Pipeline Repository Rules

1. Do not implement a custom SDXL training loop.
2. Use `sd-scripts` as the training backend.
3. Do not implement an installer or setup wizard.
4. Do not automatically upgrade PyTorch, CUDA, or NVIDIA components.
5. Preserve V100 / `sm_70` compatibility.
6. Do not modify files under `projects/*/raw/`.
7. Do not automatically delete user data.
8. Do not hard-code checkpoint-specific behavior into generic Illustrious code.
9. Character and Style are separate concept profiles.
10. Hardware constraints and training strategy must remain separate.
11. Every long-running step must be resumable.
12. Every generated LoRA must retain base model SHA256 and training metadata.
13. Prefer simple Python modules over services, daemons, or databases.
14. Do not add Kubernetes, Docker orchestration, web servers, or databases unless explicitly requested.
15. Pin external training dependencies after successful validation.
16. Before changing a validated environment dependency, explain why the change is required.
17. Do not treat automatic evaluation metrics as ground truth for image quality.
18. Keep interactive and non-interactive workflows on the same backend.
19. Avoid speculative abstractions until a second implementation requires them.
20. Optimize for maintainability and actual LoRA quality, not architectural complexity.
