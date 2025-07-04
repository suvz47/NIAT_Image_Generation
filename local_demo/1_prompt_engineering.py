"""
# Prompt Engineering Demo for Image Generation

This script demonstrates how to use a local LLM (in GGUF format) with llama-cpp-python and langchain to engineer better prompts for image generation. 
The user provides a simple prompt, and the LLM rewrites it to be more detailed and suitable for text-to-image models.
"""

# 1. Import Required Packages
import os
import sys
import configparser
from llama_cpp import Llama
from langchain_core.prompts import PromptTemplate

# 2. Load Configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Get LLM configuration
model_dir = config.get('LLM', 'model_dir_prompt')
n_ctx = config.getint('LLM', 'n_ctx')
n_batch = config.getint('LLM', 'n_batch')
max_tokens = config.getint('LLM', 'max_tokens')
temperature = config.getfloat('LLM', 'temperature')

# 3. Load the LLM Model (GGUF)
# Find the first split of the GGUF model
model_files = [f for f in os.listdir(model_dir) if f.endswith('-00001-of-00002.gguf')]
assert model_files, f'No first split GGUF model found in {model_dir}/'
model_path = os.path.join(model_dir, model_files[0])

# Load the LLM using llama-cpp-python
llm = Llama(
    model_path=model_path,
    n_ctx=n_ctx,      # Context window size
    n_batch=n_batch,  # Batch size for inference
    verbose=False     # Suppress verbose output
)
print(f'Loaded model: {model_path}')

# 4. Define the Prompt Engineering Chain with explicit tags, example, and clear instructions
prompt_template = PromptTemplate.from_template(
    """
You are an expert prompt engineer for AI image generation.

Your task is to rewrite the following user prompt to be more detailed, vivid, and creative, specifying style, lighting, composition, and any relevant details for a text-to-image model.

Output the improved prompt inside <improved_prompt> and </improved_prompt> tags, and output only ONE improved prompt. Do not repeat or generate multiple improved prompts.

Here is an example:
User prompt: a dog in a park
<improved_prompt>A photorealistic golden retriever joyfully running through a lush green park on a sunny afternoon, with soft sunlight filtering through tall trees, vibrant flowers in the background, 
and a blue sky overhead. The dog's fur glistens in the light, and its tongue is out in a playful expression.</improved_prompt>

Now, here is the user prompt:
User prompt: {user_prompt}
<improved_prompt>
"""
)

# 5. Create the Prompt Engineering Function (Stream, extract from tags)
def engineer_prompt_stream(user_prompt, max_tokens=None, temperature=None):
    # Use config values if not provided
    if max_tokens is None:
        max_tokens = config.getint('LLM', 'max_tokens')
    if temperature is None:
        temperature = config.getfloat('LLM', 'temperature')
    """
    Streams an improved image generation prompt using Qwen's chat format.
    The model is instructed to rewrite the user's prompt to be more detailed and creative.
    """
    # Qwen chat-style prompt with system and user roles
    prompt = (
        "<|im_start|>system\n"
        "You are a helpful assistant and an expert prompt engineer for AI image generation. "
        "Your task is to rewrite the user's prompt to be more detailed, vivid, and creative, specifying style, lighting, composition, and any relevant details for a text-to-image model. "
        "Output only one improved prompt."
        "<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    # Call the model and stream the output as it is generated
    stream = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True
    )
    for chunk in stream:
        text = chunk['choices'][0]['text']
        if text:
            sys.stdout.write(text)
            sys.stdout.flush()
    print()  # Print a newline after streaming is done

if __name__ == "__main__":
    print("\nPrompt Engineering for Image Generation (Terminal Mode)")
    print("-----------------------------------------------------")
    try:
        while True:
            # Prompt the user for an image description
            user_prompt = input("\nEnter a simple image prompt (Ctrl+C to exit): ")
            print("\nEngineered Prompt:\n------------------")
            engineer_prompt_stream(user_prompt)
    except KeyboardInterrupt:
        print("\nExiting. Thank you for using the prompt engineering demo!")
