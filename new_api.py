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
    text: str = None
    img: str = None
    audio: str = None

@app.post("/embeddings/")
def get_embeddings(request: EmbeddingRequest):
    try:
        inputs = {}

        if request.text is not None:
            inputs['text'] = request.text

        if request.img is not None:
            inputs['img'] = request.img

        if request.audio is not None:
            inputs['audio'] = request.audio

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
