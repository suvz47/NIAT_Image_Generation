from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
from io import BytesIO
from PIL import Image

from src.llm.prompt_engineering import engineer_generation_prompt, engineer_editing_prompt
from src.image_gen.generate import generate_image
from src.image_edit.edit import edit_image

app = FastAPI()

# Allow CORS for Gradio UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def pil_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def base64_to_pil(data: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(data)))

@app.post("/generate")
async def generate(prompt: str = Form(...)):
    """
    Accepts a user prompt, engineers it, generates an image, and returns the image as base64.
    """
    engineered = engineer_generation_prompt(prompt)
    img = generate_image(engineered)
    img_b64 = pil_to_base64(img)
    return JSONResponse({"engineered_prompt": engineered, "image": img_b64})

@app.post("/edit")
async def edit(
    image: UploadFile = File(...),
    instruction: str = Form(...)
):
    """
    Accepts an uploaded image and edit instruction, engineers the instruction, edits the image, and returns the edited image as base64.
    """
    img = Image.open(image.file).convert("RGB")
    engineered = engineer_editing_prompt(instruction)
    edited = edit_image(img, engineered)
    img_b64 = pil_to_base64(edited)
    return JSONResponse({"engineered_edit": engineered, "image": img_b64})

@app.post("/edit-generated")
async def edit_generated(
    image_b64: str = Form(...),
    instruction: str = Form(...)
):
    """
    Accepts a base64 image (from previous generation) and edit instruction, engineers the instruction, edits the image, and returns the edited image as base64.
    """
    img = base64_to_pil(image_b64)
    engineered = engineer_editing_prompt(instruction)
    edited = edit_image(img, engineered)
    img_b64 = pil_to_base64(edited)
    return JSONResponse({"engineered_edit": engineered, "image": img_b64})