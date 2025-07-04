import sys
import os
sys.path.append('src/llm')  # Add LLM module to path
from llm import generate_engineered_prompt

# --- Prompt Engineering for Image Editing ---
import configparser
from llama_cpp import Llama
from langchain_core.prompts import PromptTemplate

# Load LLM configuration for editing from config.ini
config = configparser.ConfigParser()
config.read('config.ini')

model_dir = config.get('LLM', 'model_dir_prompt')
n_ctx = config.getint('LLM', 'n_ctx')
n_batch = config.getint('LLM', 'n_batch')
max_tokens = config.getint('LLM', 'max_tokens')
temperature = config.getfloat('LLM', 'temperature')

# GGUF model for editing
model_files = [f for f in os.listdir(model_dir) if f.endswith('-00001-of-00002.gguf')]
assert model_files, f'No first split GGUF model found in {model_dir}/'
model_path = os.path.join(model_dir, model_files[0])

# Load the LLM using llama-cpp-python
llm = Llama(
    model_path=model_path,
    n_ctx=n_ctx,
    n_batch=n_batch,
    verbose=False
)
print(f'Loaded editing LLM model: {model_path}')

# Prompt template for editing (not generation)
prompt_template = PromptTemplate.from_template(
    """
You are an expert prompt engineer for AI image editing.

Your task is to rewrite the following user prompt to be more detailed, vivid, and creative, specifying style, lighting, composition, and any relevant details for an image-to-image editing model. The prompt should clearly describe how to transform the given image.

Output the improved prompt inside <improved_prompt> and </improved_prompt> tags, and output only ONE improved prompt. Do not repeat or generate multiple improved prompts.

Here is an example:
User prompt: make it look like a watercolor painting
<improved_prompt>Transform the image into a dreamy watercolor painting, with soft pastel colors, gentle brushstroke textures, subtle blending, and a light, airy atmosphere. The details should appear hand-painted, with delicate color transitions and a whimsical, artistic feel.</improved_prompt>

Now, here is the user prompt:
User prompt: {user_prompt}
<improved_prompt>
"""
)

def engineer_edit_prompt(user_prompt, max_tokens=None, temperature=None):
    if max_tokens is None:
        max_tokens = config.getint('LLM', 'max_tokens')
    if temperature is None:
        temperature = config.getfloat('LLM', 'temperature')
    prompt = (
        "<|im_start|>system\n"
        "You are a helpful assistant and an expert prompt engineer for AI image editing. "
        "Your task is to rewrite the user's prompt to be more detailed, vivid, and creative, specifying style, lighting, composition, and any relevant details for an image-to-image editing model. "
        "Output only one improved prompt."
        "<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    response = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )
    # Extract improved prompt from tags if present
    text = response["choices"][0]["text"].strip()
    if "<improved_prompt>" in text and "</improved_prompt>" in text:
        text = text.split("<improved_prompt>",1)[1].split("</improved_prompt>",1)[0].strip()
    return text

# --- Load Stable Diffusion pipeline for image-to-image editing ---
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import os

# Load the pre-trained Stable Diffusion model (optimized for Mac Metal)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("mps")  # Use Apple Silicon GPU acceleration

# Let user select the input image from available options
print("Available images in output directory:")
image_files = [f for f in os.listdir('output') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not image_files:
    raise FileNotFoundError("No image files found in output directory")

for i, filename in enumerate(image_files, 1):
    print(f"{i}. {filename}")

while True:
    try:
        choice = int(input(f"\nSelect image (1-{len(image_files)}): "))
        if 1 <= choice <= len(image_files):
            image_path = os.path.join('output', image_files[choice - 1])
            break
        else:
            print(f"Please enter a number between 1 and {len(image_files)}")
    except ValueError:
        print("Please enter a valid number")

# Load and preprocess the image
init_image = Image.open(image_path).convert("RGB").resize((512, 512))

# --- Prompt engineering section ---
# Ask the user for a prompt describing the desired transformation
user_prompt = input("Enter your image editing prompt (e.g., 'make it look like a watercolor painting'): ")

# Use the LLM to engineer a more detailed, creative prompt for image editing
try:
    engineered_prompt = engineer_edit_prompt(user_prompt)
    print(f"\nEngineered prompt for editing:\n{engineered_prompt}\n")
except Exception as e:
    print(f"[Warning] LLM prompt engineering failed: {e}\nUsing your original prompt.")
    engineered_prompt = user_prompt

# --- Image editing with Stable Diffusion ---
images = pipe(prompt=engineered_prompt, image=init_image, strength=0.7, guidance_scale=8).images

# Save the edited image in the output directory with a clear name
output_path = f"output/EDITED_{image_path.split('/')[-1]}"
images[0].save(output_path)
print(f"\nImage edited and saved as: {output_path}")