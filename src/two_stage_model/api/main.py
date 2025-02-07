from fastapi import FastAPI, UploadFile, File, Response, HTTPException
from starlette.responses import RedirectResponse, JSONResponse
import cv2
from io import BytesIO
import os
import numpy as np
from typing import List
import timm
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from inference import leakingClassifier, gen_heatmap_from_activations, overlay_heatmap
from utils import bytize_PIL, pil_2_cv2, cv2_2_pil
from diff import calculate_ssim

import config

app = FastAPI()

testTransform = T.Compose([T.Resize(size=(config.INPUT_SIZE, config.INPUT_SIZE)), 
                             T.ToTensor(),
                             T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                             ])

classifier = leakingClassifier(transform=testTransform)

class InvalidImageExtensionException(Exception):
    def __init__(self, allowedExtensions: List[str], currExtension: str):
        self.allowedExtensions = allowedExtensions
        self.currExtension = currExtension

@app.exception_handler(InvalidImageExtensionException)
async def invalid_image_extension_handler(request, exc: InvalidImageExtensionException):
    return JSONResponse(
        status_code=400,
        content={"message": f"Invalid image extension: {exc.currExtension}. Allowed extensions: {exc.allowedExtensions}"}
    )

@app.get("/")
def redirect_root():
    return RedirectResponse(url="/docs/")

@app.post("/binary_classification")
async def classify_leaking(files: List[UploadFile] = File(...)):
    """
    FastAPI endpoint for runway reconstruction.
    Inputs:
        image: the input data
    outputs: 
        result: the response containing processed result
    """
    validExtensions = {"jpg", "jpeg", "png"}
    for image in files:
        fileName = image.filename
        currExtension = fileName.split('.')[-1].lower()

        if currExtension not in validExtensions:
            raise InvalidImageExtensionException(validExtensions, currExtension)
    preImage, curImage = files
    preImageData = await preImage.read()
    curImageData = await curImage.read()
    preImageData = BytesIO(preImageData)
    preImageData.seek(0)
    curImageData = BytesIO(curImageData)
    curImageData.seek(0)
    preImagePIL = Image.open(preImageData)
    curImagePIL = Image.open(curImageData)
    _, prediction, diff = classifier.predict([preImagePIL, curImagePIL])
    base = pil_2_cv2(curImagePIL)
    base = cv2.resize(base, (640, 480))
    activations = classifier.get_intermediate_activations()[classifier.layer_name]
    heatmap = gen_heatmap_from_activations(activations)
    hmOverlay = overlay_heatmap(base, heatmap)
    H, W, _ = base.shape
    smallDiff = cv2.resize(diff, (W//2, H//2))
    smallHm = cv2.resize(hmOverlay, (W//2, H//2))
    ref = np.vstack((smallDiff, smallHm))
    print(base.shape, ref.shape)
    vis = np.hstack((base, ref))
    vis = cv2_2_pil(vis.astype("uint8"))




    result = Response(bytize_PIL(vis), headers = {"result": prediction}, media_type="image/jpeg")
    return result




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

