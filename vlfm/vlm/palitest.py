import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from typing import Optional, Any

class VLMModel:
    """Vision-Language Model for indoor navigation."""

    def __init__(
        self,
        model_name: str = "google/paligemma-3b-mix-448",
        device: Optional[Any] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_name, token="")
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
            revision="bfloat16",
            token="",
        ).eval()
        self.model.to(device)
        self.device = device

    def process_input(self, image: np.ndarray, prompt: str) -> str:
        """
        Process the image and text prompt using the model.

        Args:
            image (numpy.ndarray): The input image as a numpy array.
            prompt (str): The text prompt for the model.

        Returns:
            str: The model's response.
        """
        # Convert the numpy array to a PIL image
        pil_image = Image.fromarray(image)

        # Prepare the input for the PaliGemma model
        inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt").to(self.device)

        # Generate the model's response
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=100)
            response = self.processor.decode(output[0], skip_special_tokens=True)

        return response

# Load your image
image_path = "_currentview.png"  # Replace with the path to your image
image = np.array(Image.open(image_path))

# Define your prompt
prompt = '''
            Which direction should the robot move to find a couch? Score each possible action from 0 to 1, where 1 means most likely to find the couch.  
Answer in this exact format:  
- Forward: [score]  
- Backward: [score]  
- Left: [score]  
- Right: [score]  
            '''

# Initialize the VLMModel
vlm_model = VLMModel()

# Process the input
response = vlm_model.process_input(image, prompt)

# Print the results
print("Model Response:", response)