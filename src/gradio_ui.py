import gradio as gr
import requests
from PIL import Image
import io
import base64
import configparser
import os

# Load API URL from config.ini
config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), "../config.ini"))
API_URL = config.get("API", "url") 

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
    with gr.Blocks(
        title="NIAT Demo: Prompt-to-Image & Editing",
        css="""
        @import url('https://fonts.googleapis.com/css2?family=Architects+Daughter&display=swap');
        html, body, .gr-block, .gr-box, .gr-button, .gr-textbox, .gr-image, .gr-markdown {
            font-family: 'Architects Daughter', 'Comic Sans MS', cursive, sans-serif !important;
        }
        .sketch-row {
            display: flex;
            flex-direction: row;
            align-items: stretch;
            gap: 32px;
            width: 100%;
        }
        .sketch-col {
            flex: 1 1 0;
            display: flex;
            flex-direction: column;
            justify-content: stretch;
        }
        .sketch-box {
            border: 2.5px dashed #2176ae;
            border-radius: 18px;
            background: #e3f6fd;
            box-shadow: 2px 2px 0 0 #2176ae, 4px 4px 0 0 #38b6ff;
            padding: 8px 8px 8px 8px;
            margin-bottom: 10px;
            flex: 0 0 auto;
        }
        .sketch-header {
            font-size:1.3em; font-weight:bold; margin-bottom:0.5em; color:#2176ae; letter-spacing:1px;
            text-shadow: 1px 1px 0 #38b6ff;
        }
        .sketch-or {
            text-align:center; font-weight:bold; color:#2176ae; margin: 1em 0; font-size:1.1em;
        }
        .gr-button {
            font-size:1.1em; border-radius:12px; border:2px solid #2176ae; background:#38b6ff;
            box-shadow: 1px 1px 0 0 #2176ae;
            font-family: 'Architects Daughter', 'Comic Sans MS', cursive, sans-serif !important;
            color: #fff;
        }
        .gr-button:hover {
            background:#2176ae; color:#fff;
        }
        .sketch-image {
            border:2.5px dashed #2176ae; border-radius:14px; background:#e3f6fd;
            box-shadow: 2px 2px 0 0 #2176ae;
            margin-bottom: 16px;
            min-height: 160px;
            height: 160px;
            width: 100%;
            object-fit: contain;
            flex: 1 1 auto;
        }
        .sketch-col .gr-image, .sketch-col .gr-textbox, .sketch-col .gr-button {
            margin-bottom: 18px !important;
        }
        .sketch-col .gr-image:last-child, .sketch-col .gr-textbox:last-child, .sketch-col .gr-button:last-child {
            margin-bottom: 0 !important;
        }
        .big-prompt-box textarea {
            min-height: 170px !important;
            font-size: 1.35em !important;
            padding: 8px 10px !important;
        }
        @media (max-width: 900px) {
            .sketch-row {
                flex-direction: column;
                gap: 16px;
            }
            .sketch-col {
                min-width: 0;
            }
            .sketch-image {
                min-height: 120px;
                height: 120px;
            }
            .big-prompt-box textarea {
                min-height: 100px !important;
                font-size: 1.1em !important;
            }
        }
        """
    ) as demo:
        gr.Markdown("""
        # 🖼️ NIAT Demo: Prompt-to-Image & Image Editing
        <span style='font-size:1.1em;'>Generate an image from a prompt, or upload your own image to edit. Use natural language instructions for editing. All processing is AI-powered!</span>
        """, elem_id="main-header")
        
        with gr.Row(elem_classes=["sketch-row"]):
            with gr.Column(scale=1, elem_classes=["sketch-col"]):
                gr.Markdown("<div class='sketch-header'>1. Generate Image from Prompt</div>")
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="A cat riding a bicycle in Paris",
                    lines=7,
                    info="Describe the image you want to generate.",
                    elem_classes=["sketch-box", "big-prompt-box"]
                )
                gen_btn = gr.Button("Generate Image", variant="primary")
                engineered_prompt = gr.Textbox(label="Engineered Prompt (LLM output)", interactive=False, visible=True, elem_classes=["sketch-box"])
                gr.Markdown("<div class='sketch-header'>2. Upload Image to Edit</div>")
                upload = gr.Image(label="Upload Image", type="pil", height=80, elem_classes=["sketch-image"])
                upload_btn = gr.Button("Upload Image", variant="primary")
                
            with gr.Column(scale=1, elem_classes=["sketch-col"]):
                gr.Markdown("<div class='sketch-header'>3. View & Edit Image</div>")
                orig_image = gr.Image(label="Original Image", type="pil", interactive=False, height=180, elem_classes=["sketch-image"])
                edit_instruction = gr.Textbox(label="Edit Instructions", placeholder="Make the cat wear sunglasses", lines=2, info="Describe how you want to edit the image.", elem_classes=["sketch-box"])
                edit_btn = gr.Button("Edit Image", variant="primary")
                engineered_edit = gr.Textbox(label="Engineered Edit Prompt (LLM output)", interactive=False, visible=True, elem_classes=["sketch-box"])
                edit_image_out = gr.Image(label="Edited Image", type="pil", interactive=False, height=180, elem_classes=["sketch-image"])
        
        # State to pass images between steps
        state_image = gr.State()

        # Generate image workflow
        gen_btn.click(generate_image_workflow, inputs=prompt, outputs=[engineered_prompt, orig_image, state_image])

        # When the user uploads an image, show it as original and set the state
        upload_btn.click(upload_image_workflow, inputs=upload, outputs=[orig_image, state_image])

        # Edit image workflow (for both uploaded and generated images)
        edit_btn.click(edit_image_workflow, inputs=[state_image, edit_instruction], outputs=[engineered_edit, edit_image_out, orig_image])
    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_port=8048, server_name="0.0.0.0", share=True) 