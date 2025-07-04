import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Download and load Qwen2.5-7B-Instruct from HuggingFace
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

def engineer_prompt(user_prompt, max_new_tokens=512, temperature=0.7):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert prompt engineer for AI image generation or editing. "
                "Rewrite the following user prompt to be more detailed, vivid, and creative — "
                "specify style, lighting, composition, and relevant visual details. Output only the improved prompt."
            )
        },
        {"role": "user", "content": user_prompt}
    ]

    # Apply the chat template and tokenize
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    # Generate output
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Only decode the new tokens (assistant's part)
    generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response.strip()