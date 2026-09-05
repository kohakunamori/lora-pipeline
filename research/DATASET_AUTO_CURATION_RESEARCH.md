# Dataset Auto-Curation Research

This note records the evidence behind the DatasetWorkspace auto-curation design. The goal is not to maximize an arbitrary aesthetic score. For small Character / Character Outfit / Style LoRA datasets, useful low-frequency variation (rare poses, expressions, outfits, framing, source domains) can be more valuable than globally "pretty" images.

## Design rule

Signals are split into two classes:

1. **Deterministic reject / reversible safe exclusion**
   - corrupt image;
   - truncated file when a dedicated validator confirms it;
   - redundant exact byte-identical copy, while retaining one canonical copy.

2. **Review / ranking evidence only**
   - pHash or LPIPS near-duplicate groups;
   - generic or anime-specific aesthetic score;
   - monochrome classification;
   - comic / 3D / not-painting image-type classification;
   - zero or multiple detected anime heads;
   - relative blur, extreme exposure, low information;
   - watermark/text/layout/technical-defect caption tags;
   - CCIP identity outliers.

This asymmetry is intentional: model outputs and perceptual similarity are useful for prioritizing review, but they are not ground truth for whether a sample contributes useful concept coverage.

## Existing project baseline

Before this experiment, DatasetWorkspace already had:

- Pillow integrity / size / aspect-ratio audit;
- exact SHA duplicate suggestions;
- pHash near-duplicate clustering;
- CCIP character identity analysis;
- WD EVA02 Large auto-tagging with content-addressed cache;
- mutable reversible Dataset exclusions;
- immutable Dataset snapshots consumed by Training Projects;
- target-aware caption/semantic diversity diagnostics in Preflight.

The main missing layers were technical-quality ranking for still images, a second perceptual duplicate metric, anime-specific source/structure signals, and an orchestration policy that clearly separates safe automation from model-assisted review.

## Research findings

### 1. Deduplication matters, but exact hashes are not enough

SemDeDup uses pretrained visual embeddings to identify semantic duplicates rather than only exact duplicates, and reports that substantial redundant data can be removed while preserving performance and improving data efficiency:

- Abbas et al., *SemDeDup: Data-efficient learning at web-scale through semantic deduplication* — https://arxiv.org/abs/2303.09540
- Reference implementation — https://github.com/facebookresearch/SemDeDup

For this project, the useful takeaway is the hierarchy, not web-scale K-means infrastructure: exact SHA -> pHash -> learned perceptual/semantic similarity.

`dghs-imgutils` already provides LPIPS clustering for anime/image variants. Its documentation recommends a default LPIPS clustering threshold of `0.45`, so the first deep implementation uses this as a review default rather than introducing another model stack:

- https://dghs-imgutils.deepghs.org/main/api_doc/metrics/lpips.html

The current pHash layer remains valuable because it is cheap and deterministic enough for every Dataset. LPIPS is opt-in because it requires model inference.

### 2. Duplicate retention policy should preserve information

A duplicate detector still has to decide which member to keep. The old audit selected the lexicographically first exact duplicate. That can preserve the copy without a caption while excluding the captioned copy.

The new fast audit ranks exact byte-identical copies by:

1. caption availability;
2. pixel area;
3. file size;
4. deterministic key tie-break.

Because SHA-identical image files normally have identical pixels, caption availability is the most important practical discriminator.

LPIPS clusters are different: members may contain useful variation. They therefore only receive `recommended_keep` and `review_exclude_candidates`; they are never automatically excluded.

### 3. Anime-specific curation is already available in the project's dependency stack

The project already depends on `dghs-imgutils`. Version 0.19.0 exposes anime-specific models that are more appropriate than adding a generic photographic IQA stack:

- project / supported features — https://github.com/deepghs/imgutils
- anime image type (`3d`, `bangumi`, `comic`, `illustration`, `not_painting`) — https://dghs-imgutils.deepghs.org/v0.18.1/api_doc/validate/classify.html
- anime head detection — https://dghs-imgutils.deepghs.org/v0.15.0/api_doc/detect/head.html
- anime portrait framing — https://dghs-imgutils.deepghs.org/main/api_doc/validate/portrait.html
- monochrome detection — https://dghs-imgutils.deepghs.org/v0.15.0/api_doc/validate/monochrome.html
- truncated-file validation — https://dghs-imgutils.deepghs.org/v0.18.0/api_doc/validate/truncate.html
- Danbooru-derived anime aesthetic percentile — https://dghs-imgutils.deepghs.org/main/api_doc/metrics/dbaesthetic.html
- CCIP character similarity — https://dghs-imgutils.deepghs.org/main/api_doc/metrics/ccip.html

The imgutils README explicitly describes anime head detection as stable enough for automation while noting that person detection is still being iterated. For that reason this implementation uses **head count**, not person count, as the default structural signal.

### 4. waifuc's real anime pipelines validate the same ordering

DeepGHS' `waifuc` examples use a practical sequence including image-class filtering, duplicate filtering, face/head constraints, person splitting, CCIP, tagging and a second duplicate pass. The video tutorial explicitly recommends duplicate filtering before and after person extraction, one-face filtering, and minimum-size filtering:

- https://github.com/deepghs/waifuc
- https://deepghs.github.io/waifuc/main/tutorials/crawl_videos/index.html

This supports keeping multiple complementary duplicate stages rather than trying to replace pHash with one "perfect" detector.

### 5. Aesthetic score should not become an automatic deletion threshold

General web-dataset pipelines often use learned quality/alignment filters. Data Filtering Networks show that a model trained specifically to filter data can produce substantially better training sets than simply choosing a strong downstream classifier:

- Fang et al., *Data Filtering Networks* — https://arxiv.org/abs/2309.17425

Sieve similarly shows that a single CLIP alignment score can produce false positives/negatives and improves filtering by using captioning and semantic text similarity:

- Mahmoud et al., *Sieve: Multimodal Dataset Pruning using Image Captioning Models*, CVPR 2024 — https://openaccess.thecvf.com/content/CVPR2024/html/Mahmoud_Sieve_Multimodal_Dataset_Pruning_using_Image_Captioning_Models_CVPR_2024_paper.html

For a small personalization dataset the failure cost is higher: deleting the only rare full-body pose or outfit view can hurt disentanglement even if that image has a mediocre aesthetic score. Therefore anime aesthetic percentile is used only for review/ranking.

### 6. Dataset-relative technical quality is safer than a universal blur threshold

Absolute sharpness statistics vary strongly between line art, cel animation, screenshots, painted illustrations and soft-focus images. The fast audit therefore computes a bounded 512px proxy and records:

- luminance mean / standard deviation;
- grayscale entropy;
- dark / bright clipping fraction;
- transparent fraction;
- edge variance.

Possible blur is detected by `log(1 + edge_variance)` relative to the active dataset using median and MAD. It only becomes a review flag when there are enough samples. If MAD collapses to zero, no artificial global threshold is invented.

The project's video importer already follows the same general principle: it scores a local temporal window with FFmpeg `blurdetect` and selects the sharpest neighboring frame instead of trusting one fixed timestamp.

### 7. Caption contamination is also a data-quality signal

Diffusion memorization research identifies image duplication and text/caption duplication/specificity as important causes of copying:

- Somepalli et al., *Understanding and Mitigating Copying in Diffusion Models* — https://arxiv.org/abs/2305.20086
- Chen et al., *Towards Memorization-Free Diffusion Models*, CVPR 2024 — https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Towards_Memorization-Free_Diffusion_Models_CVPR_2024_paper.html

The optimizer therefore reports caption tags indicating likely dataset contamination or defects, including watermark/signature/text, blur/artifacts/lowres, comic/reference-sheet layouts and monochrome. These are review signals because the tags may be intentional or useful for a Style target.

### 8. Diversity must be protected while pruning

Semantic deduplication can accidentally remove underrepresented subgroups. FairDeDup explicitly studies this failure mode and changes the sample-retention strategy to better preserve underrepresented concepts:

- Slyman et al., *FairDeDup*, CVPR 2024 — https://openaccess.thecvf.com/content/CVPR2024/html/Slyman_FairDeDup_Detecting_and_Mitigating_Vision-Language_Fairness_Disparities_in_Semantic_Dataset_CVPR_2024_paper.html

The exact fairness setting is not the same as anime LoRA training, but the structural lesson transfers: pruning should not blindly collapse rare conditions. In this project, target-aware diagnostics already measure pose, expression, composition, background, lighting and outfit concentration. Future automatic subset selection should use those dimensions as preservation constraints.

## Implemented in `dataset_optimizer.py`

### Fast mode (default, no model download)

- corrupt detection;
- exact duplicate grouping and quality-aware canonical selection;
- bounded technical metrics;
- dataset-relative blur outlier review;
- extreme exposure / very-low-information review;
- caption contamination review;
- existing pHash analysis;
- existing CCIP analysis when applicable;
- JSON manifests under `review/optimization/`.

### Deep mode (`--deep`)

- imgutils truncated-file check;
- monochrome classification;
- anime image-type classification;
- Danbooru aesthetic percentile;
- anime portrait class;
- anime head count;
- LPIPS variant clustering.

Deep model outputs are advisory except for the deterministic truncated-file validator.

### Mutation policy

`--apply-safe` records reversible Dataset exclusions. It never deletes source files. Only records explicitly marked deterministic rejects are applied.

`--auto-tag` remains opt-in so a quality audit does not unexpectedly download/run a tagger or overwrite a manually curated caption strategy.

## Planned follow-up experiments

1. Calibrate fast technical distributions on real Character / Character Outfit / Style datasets instead of synthetic fixtures.
2. Measure pHash-vs-LPIPS overlap and false-positive rates on game screenshots, card art, live/video frames and official illustrations.
3. Add diversity-aware cluster retention so a near-duplicate group can preserve rare outfit/pose/expression states rather than simply choosing one representative.
4. Evaluate anime head area / framing distribution as an acquisition diagnostic (not a deletion rule).
5. Consider text-region detection (imgutils provides anime text detection) to estimate overlay coverage; do not auto-delete unless precision is demonstrated on project data.
6. Store tagger raw confidence distributions and revisit the current `0.35` general-tag threshold using target-specific validation rather than copying a model-card threshold blindly.
7. Compare before/after LoRA evaluation on fixed seeds and prompts so curation policy is judged by downstream identity/style fidelity and controllability, not by curator metrics alone.
