# Transform Imagebind into an api using modal
import modal
import os
from typing import List

stub = modal.Stub()


@stub.function(gpu="any")
def get_imagebind(
    text: str = None,
    img: str = None,
    audio: str = None,
    list_texts: List[str] = None,
    list_img: List[str] = None,
    list_audio: List[str] = None
):
    pass