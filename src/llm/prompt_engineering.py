import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Download and load Qwen2.5-7B-Instruct from HuggingFace
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Prompt template for image generation/editing
PROMPT_TEMPLATE = (
    "You are an expert prompt engineer for AI image generation or editing. "
    "Rewrite the following user prompt to be more detailed, vivid, and creative, specifying style, lighting, composition, and any relevant details for a text-to-image or image-to-image model. "
    "Output the improved prompt inside <improved_prompt> and </improved_prompt> tags, and output only ONE improved prompt. "
    "User prompt: {user_prompt}\n<improved_prompt>"
)

def engineer_prompt(user_prompt, max_new_tokens=128, temperature=0.7):
    prompt = PROMPT_TEMPLATE.format(user_prompt=user_prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract improved prompt from tags
    if "<improved_prompt>" in decoded and "</improved_prompt>" in decoded:
        improved = decoded.split("<improved_prompt>",1)[1].split("</improved_prompt>",1)[0].strip()
        return improved
    # Fallback: return everything after the tag
    return decoded.split("<improved_prompt>")[-1].strip() 