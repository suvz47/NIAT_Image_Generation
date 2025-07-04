import os
from llama_cpp import Llama

# Load GGUF model from RunPod-mounted volume or local path
MODEL_DIR = os.environ.get("MODEL_DIR", "models/prompt_engineering")
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith("-00001-of-00002.gguf")]
if not model_files:
    raise FileNotFoundError(f"No GGUF model file ending in -00001-of-00002.gguf found in {MODEL_DIR}")
MODEL_PATH = os.path.join(MODEL_DIR, model_files[0])

# Initialize the model with GPU acceleration
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_batch=128,
    n_gpu_layers=-1,  # Use full GPU (adjust based on VRAM)
    verbose=False
)

# Prompt template
PROMPT_TEMPLATE = """
<|im_start|>system
You are a helpful assistant and an expert prompt engineer for AI image generation. 
Your task is to rewrite the user's prompt to be more detailed, vivid, and creative, specifying style, lighting, composition, and any relevant details for a text-to-image model. 
Output only one improved prompt.
<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""

def generate_engineered_prompt(user_prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
    prompt = PROMPT_TEMPLATE.format(user_prompt=user_prompt)
    response = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response["choices"][0]["text"].strip()