import gradio as gr
import requests
from PIL import Image
import io
import base64

API_URL = "http://localhost:8000"  # Change to your deployed FastAPI URL

def b64_to_image(b64str):
    return Image.open(io.BytesIO(base64.b64decode(b64str)))

def generate_image_workflow(prompt):
    resp = requests.post(f"{API_URL}/generate", data={"prompt": prompt})
    data = resp.json()
    img = b64_to_image(data["image"])
    return data["engineered_prompt"], img, img

def upload_image_workflow(image):
    # Just show the uploaded image as original and set state
    return image, image

def edit_image_workflow(image, instruction):
    if image is None:
        return "No image to edit!", None, None
    # Send image and instruction to /edit
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    files = {"image": ("image.png", buf, "image/png")}
    data = {"instruction": instruction}
    resp = requests.post(f"{API_URL}/edit", files=files, data=data)
    data = resp.json()
    img = b64_to_image(data["image"])
    return data["engineered_edit"], img, img

def edit_generated_image_workflow(image, instruction):
    if image is None:
        return "No image to edit!", None, None
    # Convert image to base64
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    data = {"image_b64": img_b64, "instruction": instruction}
    resp = requests.post(f"{API_URL}/edit-generated", data=data)
    data = resp.json()
    img = b64_to_image(data["image"])
    return data["engineered_edit"], img, img

def build_ui():
    with gr.Blocks(title="NIAT Demo: Prompt-to-Image & Editing") as demo:
        gr.Markdown("""
        # 🖼️ NIAT Demo: Prompt-to-Image & Image Editing
        <span style='font-size:1.1em;'>Generate an image from a prompt, or upload your own image to edit. Use natural language instructions for editing. All processing is AI-powered!</span>
        """)
        with gr.Row():
            with gr.Column():
                gr.Markdown("<div class='sketch-header'>1. Generate Image from Prompt</div>")
                prompt = gr.Textbox(label="Prompt", placeholder="A cat riding a bicycle in Paris", lines=7)
                gen_btn = gr.Button("Generate Image", variant="primary")
                engineered_prompt = gr.Textbox(label="Engineered Prompt (LLM output)", interactive=False, visible=True)
                gr.Markdown("<div class='sketch-header'>2. Upload Image to Edit</div>")
                upload = gr.Image(label="Upload Image", type="pil", height=80)
                upload_btn = gr.Button("Upload Image", variant="primary")
            with gr.Column():
                gr.Markdown("<div class='sketch-header'>3. View & Edit Image</div>")
                orig_image = gr.Image(label="Original Image", type="pil", interactive=False, height=180)
                edit_instruction = gr.Textbox(label="Edit Instructions", placeholder="Make the cat wear sunglasses", lines=2)
                edit_btn = gr.Button("Edit Image", variant="primary")
                engineered_edit = gr.Textbox(label="Engineered Edit Prompt (LLM output)", interactive=False, visible=True)
                edit_image_out = gr.Image(label="Edited Image", type="pil", interactive=False, height=180)
        state_image = gr.State()
        # Generate image workflow
        gen_btn.click(generate_image_workflow, inputs=prompt, outputs=[engineered_prompt, orig_image, state_image])
        # When user uploads an image, show it as original and set state
        upload_btn.click(upload_image_workflow, inputs=upload, outputs=[orig_image, state_image])
        # Edit image workflow (for both uploaded and generated images)
        edit_btn.click(edit_image_workflow, inputs=[state_image, edit_instruction], outputs=[engineered_edit, edit_image_out, orig_image])
    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch() 