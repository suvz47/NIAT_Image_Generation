import gradio as gr
from PIL import Image
from src.llm.prompt_engineering import engineer_generation_prompt, engineer_editing_prompt

# Placeholder functions for backend logic

def engineer_prompt_llm(prompt):
    # Use the new generation prompt engineering function
    return engineer_generation_prompt(prompt)

def generate_image(engineered_prompt):
    # TODO: Connect to image generation model
    # Return a blank image for now
    img = Image.new('RGB', (512, 512), color='lightgray')
    return img

def engineer_edit_llm(image, edit_instruction):
    # Use the new editing prompt engineering function
    return engineer_editing_prompt(edit_instruction)

def edit_image(image, engineered_edit_prompt):
    # TODO: Connect to image editing model
    # For now, just invert the image as a placeholder
    return image.transpose(Image.FLIP_LEFT_RIGHT)

# Gradio UI
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
            min-height: 600px;
            height: 70vh;
        }
        .sketch-col {
            flex: 1 1 0;
            display: flex;
            flex-direction: column;
            justify-content: stretch;
            min-width: 320px;
            height: 100%;
        }
        .sketch-box {
            border: 2.5px dashed #2176ae;
            border-radius: 18px;
            background: #e3f6fd;
            box-shadow: 2px 2px 0 0 #2176ae, 4px 4px 0 0 #38b6ff;
            padding: 8px 8px 8px 8px;
            margin-bottom: 10px;
            min-height: 80px;
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
                engineered_prompt = gr.Textbox(label="Engineered Prompt (LLM output)", interactive=False, visible=False, elem_classes=["sketch-box"])
                gr.Markdown("<div class='sketch-header'>2. Upload Image to Edit</div>")
                upload = gr.Image(label="Upload Image", type="pil", height=80, elem_classes=["sketch-image"])
                upload_btn = gr.Button("Upload Image", variant="primary")
                
            with gr.Column(scale=1, elem_classes=["sketch-col"]):
                gr.Markdown("<div class='sketch-header'>3. View & Edit Image</div>")
                orig_image = gr.Image(label="Original Image", type="pil", interactive=False, height=180, elem_classes=["sketch-image"])
                edit_instruction = gr.Textbox(label="Edit Instructions", placeholder="Make the cat wear sunglasses", lines=2, info="Describe how you want to edit the image.", elem_classes=["sketch-box"])
                edit_btn = gr.Button("Edit Image", variant="primary")
                engineered_edit = gr.Textbox(label="Engineered Edit Prompt (LLM output)", interactive=False, visible=False, elem_classes=["sketch-box"])
                edit_image_out = gr.Image(label="Edited Image", type="pil", interactive=False, height=180, elem_classes=["sketch-image"])
        
        # State to pass images between steps
        state_image = gr.State()

        # Generate image workflow
        def handle_generate(prompt):
            eng = engineer_prompt_llm(prompt)
            img = generate_image(eng)
            return eng, img, img
        gen_btn.click(handle_generate, inputs=prompt, outputs=[engineered_prompt, orig_image, state_image])

        # When user uploads an image, show it as original and set state
        def handle_upload(img):
            return img, img
        upload_btn.click(handle_upload, inputs=upload, outputs=[orig_image, state_image])

        # Edit image workflow
        def handle_edit(image, instruction):
            if image is None:
                return "No image to edit!", None, None
            eng_edit = engineer_edit_llm(image, instruction)
            edited = edit_image(image, eng_edit)
            return eng_edit, edited, image
        edit_btn.click(handle_edit, inputs=[state_image, edit_instruction], outputs=[engineered_edit, edit_image_out, orig_image])

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
