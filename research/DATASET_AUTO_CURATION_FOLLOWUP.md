# Dataset Auto-Curation Follow-up: acquisition, text overlays, and taggers

This note extends `DATASET_AUTO_CURATION_RESEARCH.md` with findings that arrived after the first optimizer implementation.

## 1. Pruning must have an inverse operation: acquisition guidance

A small LoRA dataset can be made worse by removing redundant-looking examples that are actually the only samples covering a rare pose, outfit, expression, background, or framing condition. The practical lesson from semantic-dedup work such as FairDeDup is that **which representative is retained** matters, not only whether two samples are similar.

The branch now implements `pipeline/dataset_acquisition.py` and the `lora-dataset-acquisition` command. It reuses the project's existing target-aware diagnostics instead of inventing a second diversity ontology:

- Character: pose, expression, composition, background, lighting, semantic outfit coverage.
- Character Outfit: the Character checks plus explicit full-body coverage.
- Style: dominant subject, portrait concentration, simple-background concentration, multi-subject coverage, and broad aspect-ratio diversity.

The report is written to `review/optimization/acquisition.json`. It never mutates the DatasetWorkspace. Missing captions are treated first as a **metadata gap**, not as proof that the visual dataset lacks diversity.

## 2. Text overlays should be detected and reviewed, not blindly deleted

T-MARS (*Improving Visual Representations by Circumventing Text Feature Learning*) observes that images containing caption-overlapping text can encourage OCR-like shortcuts. Crucially, the proposed method masks text and re-scores the image instead of discarding every image that contains text.

This is especially relevant to this project's likely sources:

- game screenshots with UI;
- dialogue / subtitle frames;
- manga or comic panels;
- cards with logos or overlaid typography;
- reposted art with signatures or watermarks.

`dghs-imgutils 0.19.0` already exposes `imgutils.ocr.detect_text_with_ocr`, returning text bounding boxes and confidences. A future deep audit should record text-box count and covered-area fraction. The first deployment should remain review-only; automatic masking/cropping needs source-specific validation because text may overlap important character details.

References:

- T-MARS: https://arxiv.org/abs/2307.03132
- imgutils OCR: https://dghs-imgutils.deepghs.org/main/api_doc/ocr/index.html

## 3. Illustration completeness is useful ranking evidence

`imgutils.validate.anime_completeness` classifies anime illustrations into `polished`, `rough`, and `monochrome`. This is a useful review/ranking signal for accidental sketches, unfinished frames, or rough source material.

It must **not** become a universal deletion rule: a Style LoRA may intentionally target sketch/rough art, and a rare rough image can still carry valuable identity/outfit coverage.

Reference:

- https://dghs-imgutils.deepghs.org/main/api_doc/validate/completeness.html

## 4. Do not replace the current WD threshold with a model-card number blindly

The current pipeline uses WD EVA02 Large v3 at general threshold `0.35`. The official model card reports a validation P=R operating point of `0.5296` and macro-F1 `0.4772`, but that operating point optimizes the model-card validation objective, not LoRA caption recall, identity disentanglement, or target-specific dataset diagnostics.

The newer PixAI Tagger v0.9 is explicitly recall-oriented, has roughly 13.5k Danbooru-style tags, and was trained on a newer 2025-01 Danbooru snapshot. It is a credible challenger for curation/caption assistance, but replacing the stable backend without a project-level A/B benchmark would exchange one uncalibrated policy for another.

Recommended experiment:

1. keep WD EVA02 Large v3 as stable backend;
2. store/raw-cache confidence distributions before thresholding when practical;
3. run PixAI v0.9 as a challenger on representative Character / Outfit / Style datasets;
4. compare precision/recall on manually reviewed tags that matter to the pipeline's diversity categories;
5. compare downstream LoRA fixed-seed evaluation, not only tagger F1;
6. choose thresholds separately for general descriptive tags and character suggestions if evidence supports it.

References:

- WD EVA02 Large v3 model card: https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3
- PixAI Tagger v0.9: https://huggingface.co/pixai-labs/pixai-tagger-v0.9

## 5. Current recommended execution order

```text
Dataset sources
    ↓
Fast audit
  integrity / exact SHA / technical metrics / caption risk
    ↓
--apply-safe (optional)
  reversible exclusions for deterministic rejects only
    ↓
pHash analysis
    ↓
Deep anime audit (optional)
  truncation / image type / aesthetic / framing / heads / monochrome
    ↓
LPIPS variant clustering (review only)
    ↓
CCIP identity analysis (Character)
    ↓
Tag/caption review
    ↓
Acquisition report
  what is underrepresented and should be added?
    ↓
Frozen Dataset snapshot
```

The key invariant is that curation should improve **information density without collapsing coverage**.
