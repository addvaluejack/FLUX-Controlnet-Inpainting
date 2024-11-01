import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetInpaintPipeline, FluxControlNetModel

base_model = '/home/jiangmuye/Models/FLUX.1-dev'
controlnet_model = 'InstantX/FLUX.1-dev-Controlnet-Union'

controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
pipe = FluxControlNetInpaintPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.bfloat16)
pipe.to("cuda")

name = 'man_0'
#name = 'woman_0'
init_image = load_image(f"input_images/{name}.jpg")
mask_image = load_image(f"mask_images/{name}.jpg")
control_image = load_image(f"control_images/{name}.jpg")
controlnet_conditioning_scale = 0.5
control_mode = 4

width, height = control_image.size

prompt = "The image depicts a scene that a short hair Chinese man is walking towards the camera."
#prompt = "The image is an illustration of a man standing in front of a palace."

for i in range(5):
    output_image = pipe(
        prompt,
        image=init_image,
        mask_image=mask_image,
        control_image=control_image,
        control_mode=control_mode,
        width=width,
        height=height,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_inference_steps=24, 
        guidance_scale=3.5,
    ).images[0]

    output_image.save(f"output_images/{name}_{i}.jpg")
