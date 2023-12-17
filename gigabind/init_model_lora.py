from gigabind.models import imagebind_model
from gigabind.models.imagebind_model import ModalityType, load_module
from gigabind.models import lora


def gigabind():
    # Model Configuration
    linear_probing = False
    device = "cuda:0"  # or "cuda:0" if torch.cuda.is_available()
    load_head_post_proc_finetuned = True

    # Ensure configuration is valid
    assert not (linear_probing and lora), (
        "Linear probing is a subset of lora training procedure for"
        " ImageBind. Cannot set both linear_probing=True and"
        " lora=True. "
    )

    # Factor adjustment based on lora settings
    if lora and not load_head_post_proc_finetuned:
        lora_factor = 12 / 0.07
    else:
        lora_factor = 1

    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    if lora:
        model.modality_trunks.update(
            lora.apply_lora_modality_trunks(
                model.modality_trunks,
                rank=4,
                # layer_idxs={ModalityType.TEXT: [0, 1, 2, 3, 4, 5, 6, 7, 8],
                #             ModalityType.VISION: [0, 1, 2, 3, 4, 5, 6, 7, 8]},
                modality_names=[
                    ModalityType.TEXT,
                    ModalityType.VISION,
                ],
            )
        )

        # Load lora params if found
        lora.load_lora_modality_trunks(
            model.modality_trunks,
            checkpoint_dir=".checkpoints/lora/550_epochs_lora",
            postfix="_dreambooth_last",
        )

        if load_head_post_proc_finetuned:
            # Load postprocessors & heads
            load_module(
                model.modality_postprocessors,
                module_name="postprocessors",
                checkpoint_dir=".checkpoints/lora/550_epochs_lora",
                postfix="_dreambooth_last",
            )
            load_module(
                model.modality_heads,
                module_name="heads",
                checkpoint_dir=".checkpoints/lora/550_epochs_lora",
                postfix="_dreambooth_last",
            )
    elif linear_probing:
        # Load heads
        load_module(
            model.modality_heads,
            module_name="heads",
            checkpoint_dir="./.checkpoints/lora/500_epochs_lp",
            postfix="_dreambooth_last",
        )

    model.eval()
    model.to(device)
