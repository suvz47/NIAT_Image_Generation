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
    """Convert base64 string to PIL Image object"""
    return Image.open(io.BytesIO(base64.b64decode(b64str)))

def generate_image_workflow(prompt):
    """Send prompt to /generate endpoint and return engineered prompt + generated image"""
    resp = requests.post(f"{API_URL}/generate", data={"prompt": prompt})
    data = resp.json()
    img = b64_to_image(data["image"])
    return data["engineered_prompt"], img, img

def upload_image_workflow(image):
    """Handle uploaded image - display as original and store in state"""
    # Just show the uploaded image as original and set state
    return image, image

def edit_image_workflow(image, instruction):
    """Send image file + instruction to /edit endpoint for image editing"""
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
    """Send base64 image + instruction to /edit-generated endpoint"""
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
    """Build the main Gradio interface with custom CSS styling"""
    with gr.Blocks(
        title="NIAT Demo: Prompt-to-Image & Editing",  # Sets the browser tab title
        css="""
        /* Import Google Font for consistent typography across all components */
        @import url('https://fonts.googleapis.com/css2?family=Architects+Daughter&display=swap');
        
        /* Apply custom font family to all Gradio components for consistent styling */
        html, body, .gr-block, .gr-box, .gr-button, .gr-textbox, .gr-image, .gr-markdown {
            font-family: 'Architects Daughter', 'Comic Sans MS', cursive, sans-serif !important;
        }
        
        /* Main row container - creates horizontal layout for the two main columns */
        .sketch-row {
            display: flex;                    /* Use flexbox layout */
            flex-direction: row;              /* Arrange children horizontally */
            align-items: stretch;             /* Stretch children to same height */
            gap: 32px;                        /* Space between columns */
            width: 100%;                      /* Full width of container */
        }
        
        /* Individual column styling - each column takes equal space */
        .sketch-col {
            flex: 1 1 0;                      /* flex-grow: 1, flex-shrink: 1, flex-basis: 0 (equal width) */
            display: flex;                    /* Use flexbox for vertical stacking */
            flex-direction: column;           /* Stack children vertically */
            justify-content: stretch;         /* Stretch children to fill height */
        }
        
        /* Styled container boxes for text inputs and displays */
        .sketch-box {
            border: 2.5px dashed #2176ae;     /* Dashed border with blue color */
            border-radius: 18px;              /* Rounded corners */
            background: #e3f6fd;              /* Light blue background */
            box-shadow: 2px 2px 0 0 #2176ae, 4px 4px 0 0 #38b6ff;  /* Layered shadow effect */
            padding: 8px 8px 8px 8px;         /* Internal spacing */
            margin-bottom: 10px;              /* Space below each box */
            flex: 0 0 auto;                   /* Don't grow or shrink, use natural size */
        }
        
        /* Section headers with distinctive styling */
        .sketch-header {
            font-size:1.3em;                  /* Larger font size for emphasis */
            font-weight:bold;                 /* Bold text */
            margin-bottom:0.5em;              /* Space below header */
            color:#2176ae;                    /* Blue text color */
            letter-spacing:1px;               /* Space between letters */
            text-shadow: 1px 1px 0 #38b6ff;  /* Text shadow for depth */
        }
        
        /* Divider text styling */
        .sketch-or {
            text-align:center;                /* Center align text */
            font-weight:bold;                 /* Bold text */
            color:#2176ae;                    /* Blue color */
            margin: 1em 0;                    /* Vertical spacing */
            font-size:1.1em;                  /* Slightly larger font */
        }
        
        /* Button styling with custom appearance */
        .gr-button {
            font-size:1.1em;                  /* Larger font size */
            border-radius:12px;               /* Rounded corners */
            border:2px solid #2176ae;         /* Blue border */
            background:#38b6ff;               /* Light blue background */
            box-shadow: 1px 1px 0 0 #2176ae; /* Shadow for 3D effect */
            font-family: 'Architects Daughter', 'Comic Sans MS', cursive, sans-serif !important;  /* Custom font */
            color: #fff;                      /* White text */
        }
        
        /* Button hover state - darker background on mouse over */
        .gr-button:hover {
            background:#2176ae;               /* Darker blue background */
            color:#fff;                       /* White text */
        }
        
        /* Image container styling */
        .sketch-image {
            border:2.5px dashed #2176ae;      /* Dashed border */
            border-radius:14px;               /* Rounded corners */
            background:#e3f6fd;               /* Light blue background */
            box-shadow: 2px 2px 0 0 #2176ae; /* Shadow effect */
            margin-bottom: 16px;              /* Space below image */
            min-height: 160px;                /* Minimum height to prevent collapse */
            height: 160px;                    /* Fixed height for consistency */
            width: 100%;                      /* Full width of container */
            object-fit: contain;              /* Scale image to fit while maintaining aspect ratio */
            flex: 1 1 auto;                   /* Grow to fill available space */
        }
        
        /* Consistent spacing for all components in columns */
        .sketch-col .gr-image, .sketch-col .gr-textbox, .sketch-col .gr-button {
            margin-bottom: 18px !important;   /* Space between components */
        }
        
        /* Remove bottom margin from last component in each column */
        .sketch-col .gr-image:last-child, .sketch-col .gr-textbox:last-child, .sketch-col .gr-button:last-child {
            margin-bottom: 0 !important;      /* No bottom margin for last element */
        }
        
        /* Special styling for the main prompt textarea */
        .big-prompt-box textarea {
            min-height: 170px !important;     /* Taller textarea for longer prompts */
            font-size: 1.35em !important;     /* Larger font for better readability */
            padding: 8px 10px !important;     /* Internal spacing */
        }
        
        /* Responsive design for mobile devices */
        @media (max-width: 900px) {
            .sketch-row {
                flex-direction: column;        /* Stack columns vertically on small screens */
                gap: 16px;                     /* Reduced gap for mobile */
            }
            .sketch-col {
                min-width: 0;                  /* Allow columns to shrink below content width */
            }
            .sketch-image {
                min-height: 120px;             /* Smaller minimum height for mobile */
                height: 120px;                 /* Smaller fixed height for mobile */
            }
            .big-prompt-box textarea {
                min-height: 100px !important;  /* Smaller textarea height for mobile */
                font-size: 1.1em !important;   /* Smaller font for mobile */
            }
        }
        """
    ) as demo:
        # Main header with title and description
        # elem_id="main-header" allows targeting this specific markdown with CSS
        gr.Markdown("""
        # 🖼️ NIAT Demo: Prompt-to-Image & Image Editing
        <span style='font-size:1.1em;'>Generate an image from a prompt, or upload your own image to edit. Use natural language instructions for editing. All processing is AI-powered!</span>
        """, elem_id="main-header")
        
        # Two-column layout for the main interface
        # elem_classes=["sketch-row"] applies the flexbox row styling
        with gr.Row(elem_classes=["sketch-row"]):
            # Left column: Input controls
            # scale=1 gives equal width to both columns, elem_classes applies column styling
            with gr.Column(scale=1, elem_classes=["sketch-col"]):
                # Section 1: Image generation from prompt
                gr.Markdown("<div class='sketch-header'>1. Generate Image from Prompt</div>")
                
                # Text input for image generation prompt
                # lines=7 creates a multi-line textarea with 7 visible lines
                # placeholder shows example input to guide users
                # info provides additional help text below the input
                # elem_classes applies the sketch-box styling (border, background, etc.)
                prompt = gr.Textbox(
                    label="Prompt",                                    # Label displayed above the input
                    placeholder="A cat riding a bicycle in Paris",     # Example text shown when input is empty
                    lines=7,                                           # Creates a textarea with 7 visible lines
                    info="Describe the image you want to generate.",   # Help text displayed below the input
                    elem_classes=["sketch-box", "big-prompt-box"]      # CSS classes for styling
                )
                
                # Button to trigger image generation workflow
                # variant="primary" applies the primary button styling (blue background)
                gen_btn = gr.Button("Generate Image", variant="primary")
                
                # Display area for the LLM-engineered prompt (read-only)
                # interactive=False prevents user editing, visible=True shows the component
                engineered_prompt = gr.Textbox(
                    label="Engineered Prompt (LLM output)",    # Label for the display area
                    interactive=False,                         # User cannot edit this field
                    visible=True,                              # Component is visible (not hidden)
                    elem_classes=["sketch-box"]                # Apply sketch-box styling
                )
                
                # Section 2: Image upload
                gr.Markdown("<div class='sketch-header'>2. Upload Image to Edit</div>")
                
                # Image upload component - accepts PIL Image objects, fixed height of 80px
                # type="pil" specifies that the component works with PIL Image objects
                # height=80 sets a fixed height of 80 pixels for the upload area
                upload = gr.Image(
                    label="Upload Image",                      # Label above the upload area
                    type="pil",                                # Accept PIL Image objects (not file paths)
                    height=80,                                 # Fixed height in pixels
                    elem_classes=["sketch-image"]              # Apply image styling
                )
                
                # Button to process uploaded image
                upload_btn = gr.Button("Upload Image", variant="primary")
                
            # Right column: Image display and editing
            with gr.Column(scale=1, elem_classes=["sketch-col"]):
                # Section 3: Image viewing and editing
                gr.Markdown("<div class='sketch-header'>3. View & Edit Image</div>")
                
                # Display area for original image (generated or uploaded)
                # interactive=False prevents user editing, height=180px for larger display
                # This component shows the current image being worked with
                orig_image = gr.Image(
                    label="Original Image",                    # Label for the image display
                    type="pil",                                # Works with PIL Image objects
                    interactive=False,                         # User cannot edit the image
                    height=180,                                # Larger height for better visibility
                    elem_classes=["sketch-image"]              # Apply image styling
                )
                
                # Text input for editing instructions
                # lines=2 creates a smaller textarea for brief instructions
                # placeholder provides example of what to enter
                edit_instruction = gr.Textbox(
                    label="Edit Instructions",                 # Label above the input
                    placeholder="Make the cat wear sunglasses", # Example instruction
                    lines=2,                                   # Small textarea (2 lines)
                    info="Describe how you want to edit the image.", # Help text
                    elem_classes=["sketch-box"]                # Apply sketch-box styling
                )
                
                # Button to trigger image editing workflow
                edit_btn = gr.Button("Edit Image", variant="primary")
                
                # Display area for the LLM-engineered edit prompt (read-only)
                # Shows the AI's interpretation of the user's edit instruction
                engineered_edit = gr.Textbox(
                    label="Engineered Edit Prompt (LLM output)", # Label for the display
                    interactive=False,                          # Read-only field
                    visible=True,                               # Always visible
                    elem_classes=["sketch-box"]                 # Apply sketch-box styling
                )
                
                # Display area for the edited image result
                # Shows the final edited image after processing
                edit_image_out = gr.Image(
                    label="Edited Image",                       # Label for the result
                    type="pil",                                 # Works with PIL Image objects
                    interactive=False,                          # Display only, no editing
                    height=180,                                 # Same height as original for comparison
                    elem_classes=["sketch-image"]               # Apply image styling
                )
        
        # Hidden state component to pass images between workflow steps
        # This allows the edit workflow to access the current image (generated or uploaded)
        # gr.State() creates an invisible component that can store data
        state_image = gr.State()

        # Event handlers - connect UI components to backend functions
        
        # When generate button is clicked, call generate_image_workflow
        # inputs=prompt: takes text from the prompt textbox
        # outputs=[engineered_prompt, orig_image, state_image]: updates three components
        #   - engineered_prompt: shows the AI's processed prompt
        #   - orig_image: displays the generated image
        #   - state_image: stores the image for later editing
        gen_btn.click(generate_image_workflow, inputs=prompt, outputs=[engineered_prompt, orig_image, state_image])

        # When upload button is clicked, call upload_image_workflow
        # inputs=upload: takes the uploaded image file
        # outputs=[orig_image, state_image]: updates two components
        #   - orig_image: displays the uploaded image
        #   - state_image: stores the image for later editing
        upload_btn.click(upload_image_workflow, inputs=upload, outputs=[orig_image, state_image])

        # When edit button is clicked, call edit_image_workflow
        # inputs=[state_image, edit_instruction]: takes current image and edit text
        # outputs=[engineered_edit, edit_image_out, orig_image]: updates three components
        #   - engineered_edit: shows the AI's processed edit instruction
        #   - edit_image_out: displays the edited image result
        #   - orig_image: keeps original visible for comparison
        edit_btn.click(edit_image_workflow, inputs=[state_image, edit_instruction], outputs=[engineered_edit, edit_image_out, orig_image])
    return demo

if __name__ == "__main__":
    demo = build_ui()
    # Launch Gradio server on port 8048, accessible from any IP, with public sharing enabled
    # server_port=8048: sets the local port number
    # server_name="0.0.0.0": allows connections from any IP address (not just localhost)
    # share=True: creates a public URL for external access
    demo.launch(server_port=8048, server_name="0.0.0.0", share=True) 