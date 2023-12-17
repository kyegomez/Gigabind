import torch
from torch import nn
import logging


from gigabind.models import imagebind_model
from gigabind.models.imagebind_model import ModalityType, load_module
from typing import Callable
from types import SimpleNamespace


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
    """Gigabind model

    Args:
        lora (bool, optional): Whether to use lora. Defaults to False.
        linear_probing (bool, optional): Whether to use linear probing. Defaults to False.
        device (Callable, optional): Device to run the model on. Defaults to detect_device.
        load_head_post_proc_finetuned (bool, optional): Whether to load the head and post processors. Defaults to True.

    Examples:
        >>> from gigabind import Gigabind
        >>> model = Gigabind()
        >>> model.run(text="Hello World!")
        {'embeddings': tensor([[-0.0000, -0.0000, -0.0000,  ..., -0.0000, -0.0000, -0.0000]]), 'modality_type': ['text'], 'model_name': 'gigabind-huge'}


    """

    def __init__(
        self,
        lora=False,
        linear_probing=False,
        device: Callable = "cuda:0",
        load_head_post_proc_finetuned=True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.lora = lora
        self.linear_probing = linear_probing
        self.device = device
        self.load_head_post_proc_finetuned = (
            load_head_post_proc_finetuned
        )

        # Ensure configuration is valid
        assert not (linear_probing and lora), (
            "Linear probing is a subset of lora training procedure"
            " for ImageBind. Cannot set both linear_probing=True and"
            " lora=True. "
        )

        # Factor adjustment based on lora settings
        if lora and not load_head_post_proc_finetuned:
            lora_factor = 12 / 0.07
        else:
            lora_factor = 1

        # Instantiate model
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        if lora:
            self.model.modality_trunks.update(
                lora.apply_lora_modality_trunks(
                    self.model.modality_trunks,
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
                self.model.modality_trunks,
                checkpoint_dir=".checkpoints/lora/550_epochs_lora",
                postfix="_dreambooth_last",
            )

            if load_head_post_proc_finetuned:
                # Load postprocessors & heads
                load_module(
                    self.model.modality_postprocessors,
                    module_name="postprocessors",
                    checkpoint_dir=(
                        ".checkpoints/lora/550_epochs_lora"
                    ),
                    postfix="_dreambooth_last",
                )
                load_module(
                    self.model.modality_heads,
                    module_name="heads",
                    checkpoint_dir=(
                        ".checkpoints/lora/550_epochs_lora"
                    ),
                    postfix="_dreambooth_last",
                )
        elif linear_probing:
            # Load heads
            load_module(
                self.model.modality_heads,
                module_name="heads",
                checkpoint_dir="./.checkpoints/lora/500_epochs_lp",
                postfix="_dreambooth_last",
            )

        self.model.eval()
        self.model.to(device)

    def run(
        self,
        text: str = None,
        img: str = None,
        audio: str = None,
        *args,
        **kwargs,
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
                inputs["vision"] = img
            if audio is not None:
                inputs["audio"] = audio

            with torch.no_grad():
                embeddings = self.model(inputs)

            # Prepare the response
            response = {
                "embeddings": embeddings,  # The computed embeddings
                "modality_type": list(
                    inputs.keys()
                ),  # The types of modalities in the inputs
                "model_name": (
                    "gigabind-huge"
                ),  # The name of the model
            }

            return response
        except Exception as e:
            logging.error(e)
            return e

    def print_modalities_available(self):
        """Prints the modalities available for the model"""
        ModalityType = SimpleNamespace(
            VISION="vision",
            TEXT="text",
            AUDIO="audio",
            THERMAL="thermal",
            DEPTH="depth",
            IMU="imu",
        )
        print("Modalities Available:")
        print(ModalityType.VISION)
        print(ModalityType.TEXT)
        print(ModalityType.AUDIO)
        print(ModalityType.THERMAL)
        print(ModalityType.DEPTH)
        print(ModalityType.IMU)

    def print_model(self):
        """Prints the model"""
        print(self.model)

    def print_model_parameters(self):
        """prints the model parameters"""
        print(self.model.parameters())
