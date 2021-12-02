import os
import typing
from tempfile import NamedTemporaryFile

import cloudinary
import pydantic
import requests
from cloudinary.uploader import upload as cloudinary_upload
from fastapi import FastAPI, HTTPException

from app.settings import config
from lama import inference, output_dir

app = FastAPI()
cloudinary.config(
    cloud_name=config.CLOUDINARY_CLOUD_NAME,
    api_key=config.CLOUDINARY_API_KEY,
    api_secret=config.CLOUDINARY_API_SECRET,
    secure=True,
)


class InpaintingRequest(pydantic.BaseModel):
    image_url: str
    mask_url: str


@app.post("/inpainting", response_model=typing.Dict[str, str])
async def inpainting(request: InpaintingRequest):
    # Check the content type of the URL before downloading the content
    try:
        h = requests.head(request.image_url, allow_redirects=True)
    except Exception:
        raise HTTPException(400, detail="Invalid URL for image")
    if "image/png" not in h.headers["Content-Type"]:
        raise HTTPException(400, detail="Invalid image file type: expected png")
    try:
        h = requests.head(request.mask_url, allow_redirects=True)
    except Exception:
        raise HTTPException(400, detail="Invalid URL for video")
    if "image/png" not in h.headers["Content-Type"]:
        raise HTTPException(400, detail="Invalid image file type: expected png")

    # Download and write files to temporary directory
    resp = requests.get(request.image_url)
    with NamedTemporaryFile(delete=False, suffix=".png") as image_tmp:
        image_tmp.write(resp.content)

    resp = requests.get(request.mask_url)
    with NamedTemporaryFile(delete=False, suffix=".png") as mask_tmp:
        mask_tmp.write(resp.content)

    # Model inference
    output_filename = inference(image_tmp.name, mask_tmp.name)

    # Upload model output
    upload_resp = cloudinary_upload(
        os.path.join(output_dir, output_filename),
        folder="lama-outputs",
        resource_type="image",
    )

    return {"output_url": upload_resp["url"]}
