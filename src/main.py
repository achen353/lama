import os
import typing
from tempfile import NamedTemporaryFile

import cloudinary
import pydantic
import requests
from cloudinary.uploader import upload as cloudinary_upload
from fastapi import FastAPI, HTTPException

from app.settings import config
from inference import inference

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


async def inpainting(request: InpaintingRequest):
    # Check the content type of the URL before downloading the content
    try:
        h = requests.head(request.reference_image_url, allow_redirects=True)
    except Exception:
        raise HTTPException(400, detail="Invalid URL for image")
    if "image/jpeg" not in h.headers["Content-Type"]:
        raise HTTPException(400, detail="Invalid image file type: expected jpg/jpeg")
    try:
        h = requests.head(request.driving_video_url, allow_redirects=True)
    except Exception:
        raise HTTPException(400, detail="Invalid URL for video")
    if "image/jpeg" not in h.headers["Content-Type"]:
        raise HTTPException(400, detail="Invalid image file type: expected jpg/jpeg")

    # Download and write files to temporary directory
    resp = requests.get(request.reference_image_url)
    with NamedTemporaryFile(delete=False, suffix=".jpg") as image_tmp:
        image_tmp.write(resp.content)

    resp = requests.get(request.driving_video_url)
    with NamedTemporaryFile(delete=False, suffix=".mp4") as video_tmp:
        video_tmp.write(resp.content)

    # Model inference
    inference(image_tmp.name, video_tmp.name)

    # Upload model output
    upload_resp = cloudinary_upload(
        os.path.join(output_dir, "001.mp4"),
        folder="fewshot-vid2vid-outputs",
        resource_type="video",
    )

    return {"output_url": upload_resp["url"]}