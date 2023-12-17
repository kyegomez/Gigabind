import torch
from torch import nn
import logging


from gigabind.models import imagebind_model
from gigabind.models.imagebind_model import ModalityType, load_module
from typing import Callable

# Setup Logging
logging.basicConfig(level=logging.INFO, force=True)


def detect_device():
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception as e:
        logging.error(e)
        device = "cpu"
    return device


class Gigabind(nn.Module):
    def __init__(
        self,
        lora=False,
        linear_probing=False,
        device: Callable = detect_device,
        load_head_post_proc_finetuned=True,
        *args,
        **kwargs
    ):
        super().__init__()
        self.lora = lora
        self.linear_probing = linear_probing
        self.device = device
        self.load_head_post_proc_finetuned = load_head_post_proc_finetuned

        # Ensure configuration is valid
        assert not (linear_probing and lora), (
            "Linear probing is a subset of lora training procedure for ImageBind. "
            "Cannot set both linear_probing=True and lora=True. "
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
                    modality_names=[ModalityType.TEXT, ModalityType.VISION],
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

        self.model.eval()
        self.model.to(device)

    def run(
        self, text: str = None, img: str = None, audio: str = None, *args, **kwargs
    ):
        """Run the model

        Args:
            text (str, optional): Text inputs. Defaults to None.
            img (str, optional): Img file path. Defaults to None.
            audio (str, optional): audio file paths. Defaults to None.

        Returns:
            embeddings: The computed embeddings for the inputs
        """
        try:
            # Prepare the inputs
            inputs = {}
            if text is not None:
                inputs["text"] = text
            if img is not None:
                inputs["img"] = img
            if audio is not None:
                inputs["audio"] = audio

            # Compute the embeddings
            embeddings = self.model(inputs)

            # Prepare the response
            response = {
                "embeddings": embeddings,  # The computed embeddings
                "modality_type": list(
                    inputs.keys()
                ),  # The types of modalities in the inputs
                "model_name": "gigabind-huge",  # The name of the model
                # Add any other information you want to include in the response
            }

            return response
        except Exception as e:
            logging.error(e)
            return e
