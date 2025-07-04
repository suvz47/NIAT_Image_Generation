import torch
from diffusers import StableDiffusionPipeline

MODEL_NAME = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_image(prompt, num_inference_steps=30, guidance_scale=7.5):
    """
    Generate an image from a prompt using Stable Diffusion.
    Returns a PIL Image.
    """
    result = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
    return result.images[0] 