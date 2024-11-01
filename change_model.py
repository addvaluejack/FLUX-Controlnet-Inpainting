from diffusers import ControlNetModel, UniPCMultistepScheduler, StableDiffusionControlNetInpaintPipeline
import torch
from diffusers.utils import load_image
import numpy as np

name = 'man_0'

init_image = load_image(f"input_images/{name}.jpg")
mask_image = load_image(f"mask_images/{name}.jpg")
control_image = load_image(f"control_images/{name}.jpg")

controlnet = ControlNetModel.from_pretrained(
    "fusing/stable-diffusion-v1-5-controlnet-openpose", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
     "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
pipe.enable_xformers_memory_efficient_attention()
pipe.to('cuda')

init_image = init_image.resize(control_image.size)
mask_image = mask_image.resize(control_image.size)

# generate image
output_image = pipe(
    "a walking white man",
    num_inference_steps=30,
    image=init_image,
    control_image=control_image,
    mask_image=mask_image
).images[0]

output_image.save(f"output_images/{name}.jpg")