import logging

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from gigabind import data
from gigabind.models import imagebind_model, lora
from gigabind.models.imagebind_model import ModalityType, load_module
from gigabind.main import Gigabind

# Setup Logging
logging.basicConfig(level=logging.INFO, force=True)

# Initialize FastAPI app
app = FastAPI(debug=True)
gigabind = Gigabind()

class EmbeddingRequest(BaseModel):
    text_list: list
    image_paths: list
    audio_paths: list


@app.post("/embeddings/")
def get_embeddings(request: EmbeddingRequest):
    """Get embeddings from text, image, and audio inputs

    Args:
        request (EmbeddingRequest): _description_

    Raises:
        HTTPException: _description_
        HTTPException: _description_

    Returns:
        _type_: _description_
    """
    try:
        inputs = {}

        if request.text_list is not None:
            inputs['text'] = request.text_list

        if request.image_paths is not None:
            inputs['img'] = request.image_paths

        if request.audio_paths is not None:
            inputs['audio'] = request.audio_paths

        if not inputs:
            raise HTTPException(status_code=400, detail="No input provided")

        response = gigabind.run(**inputs)
        return response

    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
