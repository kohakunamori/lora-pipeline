from pipeline.interactive_materialization import InteractiveWizard
from pipeline.interactive_semantic_concepts import InteractiveWizard as BaseInteractiveWizard


def test_final_dataset_wizard_does_not_override_import_with_smart_crop() -> None:
    """Normal Dataset import must keep RAW sources unchanged.

    The final wizard intentionally inherits the ordinary import implementation;
    target-aware crop belongs to materialization and therefore runs once.
    """

    assert InteractiveWizard._import_image_directory is BaseInteractiveWizard._import_image_directory
