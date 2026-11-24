"""Image decoding used by the copied formal evaluator."""

from io import BytesIO

from PIL import Image
from qwen_vl_utils import fetch_image


def process_image(image, image_patch_size=14):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if "bytes" in image:
        if "image" in image:
            raise ValueError("An image record cannot contain both bytes and image")
        image = dict(image)
        image["image"] = Image.open(BytesIO(image["bytes"]))
    return fetch_image(image, image_patch_size=image_patch_size)
