import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

MODEL_NAME = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def edit_image(init_image: Image.Image, prompt: str, strength=0.7, guidance_scale=8):
    """
    Edit an image using Stable Diffusion Img2Img.
    Returns a PIL Image.
    """
    result = pipe(prompt=prompt, image=init_image, strength=strength, guidance_scale=guidance_scale)
    return result.images[0] 