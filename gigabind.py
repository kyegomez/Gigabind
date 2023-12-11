from modal import Stub, Image
from pydantic import BaseModel, Field
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List, Optional
import logging
import torch
import data
import os
import modal

from models import imagebind_model
from models.imagebind_model import ModalityType, load_module
from models import lora as LoRA

app = FastAPI()

class InputData(BaseModel):
    text: Optional[List[str]] = Field(None)
    audio: Optional[UploadFile] = Field(None)
    vision: Optional[UploadFile] = Field(None)

@app.post("/process")
async def process_data(input_data: InputData = None, audio: UploadFile = File(None), vision: UploadFile = File(None)):
    # Load your model here (if not loaded)
    logging.basicConfig(level=logging.INFO, force=True)

    lora = True
    linear_probing = False
    device = "cpu"  # "cuda:0" if torch.cuda.is_available() else "cpu"
    load_head_post_proc_finetuned = True

    assert not (linear_probing and lora), (
        "Linear probing is a subset of LoRA training procedure for ImageBind. "
        "Cannot set both linear_probing=True and lora=True. "
    )

    if lora and not load_head_post_proc_finetuned:
        # Hack: adjust lora_factor to the `max batch size used during training / temperature` to compensate missing norm
        lora_factor = 12 / 0.07
    else:
        # This assumes proper loading of all params but results in shift from original dist in case of LoRA
        lora_factor = 1

    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    # Your existing code here

    # Prepare your data
    inputs = {}
    if input_data and input_data.text:
        inputs[ModalityType.TEXT] = data.load_and_transform_text(input_data.text, device)
    if audio:
        audio_filename = f"/tmp/{audio.filename}"
        with open(audio_filename, "wb") as buffer:
            buffer.write(await audio.read())
        audio_data = data.load_and_transform_audio_data([audio_filename], device)
        inputs[ModalityType.AUDIO] = audio_data
    if vision:
        vision_filename = f"/tmp/{vision.filename}"
        with open(vision_filename, "wb") as buffer:
            buffer.write(await vision.read())
        vision_data = data.load_and_transform_vision_data([vision_filename], device, to_tensor=True)
        inputs[ModalityType.VISION] = vision_data

    if not inputs:
        raise HTTPException(status_code=400, detail="No input data provided")

    # Then, you can process your data using your model and return the result
    with torch.no_grad():
        embeddings = model(inputs)

    result = {
        "Vision x Text": torch.softmax(
            embeddings[ModalityType.VISION]
            @ embeddings[ModalityType.TEXT].T
            * (lora_factor if lora else 1),
            dim=-1,
        ).tolist(),
        "Audio x Text": torch.softmax(
            embeddings[ModalityType.AUDIO]
            @ embeddings[ModalityType.TEXT].T
            * (lora_factor if lora else 1),
            dim=-1,
        ).tolist(),
        "Vision x Audio": torch.softmax(
            embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1
        ).tolist(),
    }

    return result
image =  modal.create_image('python:3.8')

stub = Stub('gigabind')
image = Image('python:3.8')
stub.image = image
