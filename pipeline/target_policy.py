from __future__ import annotations


def attach_target_aware_dataset_semantics_snapshot(state, workspace):
    """Legacy compatibility seam; Dataset semantics no longer affect training runtime.

    Dataset semantic metadata remains available to Dataset curation/UI code, but
    TrainingConfig no longer freezes or applies it to materialization, captions,
    fingerprints, or preflight. Remove this shim when the remaining call site in
    ``training_config`` is next rewritten.
    """

    del workspace
    return state
