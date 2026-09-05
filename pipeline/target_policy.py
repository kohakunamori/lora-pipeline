from __future__ import annotations

from .activation_recipe import attach_activation_recipe_snapshot
from .models import PipelineError


def attach_target_aware_dataset_semantics_snapshot(state, workspace):
    """Compatibility entry point for freezing the new ActivationRecipe.

    Legacy Dataset semantics remain disconnected from training runtime. The old
    call site is reused temporarily so TrainingConfig can freeze the lightweight
    character tag-group recipe without reintroducing semantic curation state.
    """

    state = attach_activation_recipe_snapshot(state, workspace)
    project = state.payload["project"]
    recipe = project.get("activation_recipe", {})
    groups = recipe.get("character_tags_groups", []) if isinstance(recipe, dict) else []
    if not groups:
        return state

    caption_mode = str(project.get("interactive_preferences", {}).get("caption_mode", "auto"))
    if caption_mode in {"existing_passthrough", "skip"}:
        raise PipelineError(
            "Character tag groups require a rewriteable caption mode so each image can learn "
            "its group_tag; use auto, generate, hybrid, or existing_taglist_clean."
        )

    overrides = project.setdefault("overrides", {})
    caption = overrides.setdefault("caption", {})
    trigger_policy = project.get("trigger_policy", {})
    protected = trigger_policy.get("protected_prefix", []) if isinstance(trigger_policy, dict) else []
    # Every active image is required to belong to exactly one group when groups are
    # enabled. Reserve one additional protected prefix slot for that image's group_tag.
    caption["keep_tokens"] = max(
        int(caption.get("keep_tokens", 1) or 1),
        max(1, len(protected)) + 1,
    )
    state.save()
    return state
