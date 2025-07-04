from mflux import Flux1, Config, ModelConfig
import numpy as np
from PIL import Image


# Try different model configurations
print("Loading model...")

# Option 1: Try "dev" model with different settings
flux = Flux1.from_name(
   model_name="dev",  # Back to dev model
   quantize=None,     # No quantization
)

print("Model loaded successfully!")

# Generate an image with more conservative settings
print("Generating image...")
image = flux.generate_image(
   seed=123,  # Different seed
   prompt="cat wearing a hat",  # Very simple prompt
   config=Config(
      num_inference_steps=50,  # Moderate steps
      height=256,  # Smaller size
      width=256,
   )
)

print("Image generation completed!")
