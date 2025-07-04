#!/usr/bin/env python3
"""
Image Generation Demo using Stable Diffusion

This script demonstrates how to generate images using the Stable Diffusion model.
It allows users to input their own prompts and generates high-quality images.
"""

import torch
from diffusers import StableDiffusionPipeline
import os
from datetime import datetime
import configparser


def load_config():
    """
    Load configuration from config.ini file.
    
    Returns:
        configparser.ConfigParser: Configuration object
    """
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.ini')
    
    if os.path.exists(config_path):
        config.read(config_path)
        print("Configuration loaded from config.ini")
    else:
        print("config.ini not found")
    
    return config


def setup_pipeline(config):
    """
    Initialize and configure the Stable Diffusion pipeline.
    
    Args:
        config: Configuration object from config.ini
    
    Returns:
        StableDiffusionPipeline: Configured pipeline ready for image generation
    """
    print("Loading Stable Diffusion model...")
    
    # Get model paths from config
    local_model_path = config.get('Model', 'local_model_path')
    huggingface_model = config.get('Model', 'huggingface_model')
    
    # Check if local model exists
    if os.path.exists(local_model_path):
        print(f"Using local model from: {local_model_path}")
        # Load the model from local path
        pipe = StableDiffusionPipeline.from_pretrained(local_model_path)
    else:
        print("Local model not found, downloading from Hugging Face...")
        # Fallback to downloading from Hugging Face
        pipe = StableDiffusionPipeline.from_pretrained(huggingface_model)
    
    # Configure device (MPS for Mac M1/M2, CUDA for NVIDIA GPUs, CPU as fallback)
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA (NVIDIA GPU)")
    else:
        device = "cpu"
        print("Using CPU (slower generation)")
    
    pipe.to(device)
    return pipe


def get_user_prompt():
    """
    Get image generation prompt from user input.
    
    Returns:
        str: User-provided prompt for image generation
    """
    print("\n" + "="*50)
    print("IMAGE GENERATION DEMO")
    print("="*50)
    
    print("\nEnter your image description below.")
    print("Examples:")
    print("  - 'a cat wearing a hat'")
    print("  - 'a beautiful sunset over mountains'")
    print("  - 'a futuristic city with flying cars'")
    print("  - 'a cozy coffee shop interior'")
    
    while True:
        prompt = input("\nYour prompt: ").strip()
        if prompt:
            return prompt
        else:
            print("Please enter a valid prompt!")


def generate_image(pipe, prompt):
    """
    Generate an image using the provided prompt.
    
    Args:
        pipe: Stable Diffusion pipeline
        prompt (str): Text description for image generation
    
    Returns:
        PIL.Image: Generated image
    """
    print(f"\nGenerating image for: '{prompt}'")
    print("This may take a few moments...")
    
    # Generate the image
    result = pipe(prompt)
    image = result.images[0]
    
    return image


def save_image(image, prompt, config):
    """
    Save the generated image in the output folder with prompt name.
    
    Args:
        image: PIL Image to save
        prompt (str): Original prompt used for generation
        config: Configuration object from config.ini
    
    Returns:
        str: Path to the saved image file
    """
    # Get output directory from config
    output_dir = config.get('Output', 'output_dir')
    max_filename_length = config.getint('Output', 'max_filename_length')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a safe filename from the prompt
    safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_prompt = safe_prompt.replace(' ', '_')[:max_filename_length]  # Use config length
    
    # Add timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_prompt}_{timestamp}.png"
    
    # Full path to save the image
    filepath = os.path.join(output_dir, filename)
    
    # Save the image
    image.save(filepath)
    print(f"Image saved as: {filepath}")
    
    return filepath


def main():
    """
    Main function to orchestrate the image generation process.
    """
    try:
        # Load configuration
        config = load_config()
        
        # Setup the pipeline
        pipe = setup_pipeline(config)
        
        # Get user input
        prompt = get_user_prompt()
        
        # Generate the image
        image = generate_image(pipe, prompt)
        
        # Save the result
        filename = save_image(image, prompt, config)
        
        print("\n" + "="*50)
        print("Image generation completed successfully!")
        print(f"File: {filename}")
        print("="*50)
        
    except KeyboardInterrupt:
        print("\n\nGeneration cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Make sure you have enough memory and the model is properly loaded.")


if __name__ == "__main__":
    main()