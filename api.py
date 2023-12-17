import logging

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from gigabind import data
from gigabind.models import imagebind_model
from gigabind.models.imagebind_model import ModalityType, load_module
from gigabind.models import lora

# Setup Logging
logging.basicConfig(level=logging.INFO, force=True)

# Initialize FastAPI app
app = FastAPI()

# Model Configuration
lora = True
linear_probing = False
device = "cpu"  # or "cuda:0" if torch.cuda.is_available()
load_head_post_proc_finetuned = True

# Ensure configuration is valid
assert not (linear_probing and lora), (
    "Linear probing is a subset of LoRA training procedure for ImageBind. "
    "Cannot set both linear_probing=True and lora=True. "
)

# Factor adjustment based on LoRA settings
if lora and not load_head_post_proc_finetuned:
    lora_factor = 12 / 0.07
else:
    lora_factor = 1

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
if lora:
    model.modality_trunks.update(
        LoRA.apply_lora_modality_trunks(
            model.modality_trunks,
            rank=4,
            # layer_idxs={ModalityType.TEXT: [0, 1, 2, 3, 4, 5, 6, 7, 8],
            #             ModalityType.VISION: [0, 1, 2, 3, 4, 5, 6, 7, 8]},
            modality_names=[ModalityType.TEXT, ModalityType.VISION],
        )
    )

    # Load LoRA params if found
    LoRA.load_lora_modality_trunks(
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


class EmbeddingRequest(BaseModel):
    text_list: list
    image_paths: list
    audio_paths: list


@app.post("/embeddings/")
def get_embeddings(request: EmbeddingRequest):
    try:
        # Load data
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(request.text_list, device),
            ModalityType.VISION: data.load_and_transform_vision_data(
                request.image_paths, device, to_tensor=True
            ),
            ModalityType.AUDIO: data.load_and_transform_audio_data(
                request.audio_paths, device
            ),
        }

        with torch.no_grad():
            embeddings = model(inputs)

        # Process and return embeddings
        return {
            "vision_x_text": torch.softmax(
                embeddings[ModalityType.VISION]
                @ embeddings[ModalityType.TEXT].T
                * (lora_factor if lora else 1),
                dim=-1,
            ).tolist(),
            "audio_x_text": torch.softmax(
                embeddings[ModalityType.AUDIO]
                @ embeddings[ModalityType.TEXT].T
                * (lora_factor if lora else 1),
                dim=-1,
            ).tolist(),
            "vision_x_audio": torch.softmax(
                embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T,
                dim=-1,
            ).tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
