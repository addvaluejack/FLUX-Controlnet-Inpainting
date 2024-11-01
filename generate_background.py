import torch
from diffusers.utils import load_image, check_min_version
from controlnet_flux import FluxControlNetModel
from transformer_flux import FluxTransformer2DModel
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline

check_min_version("0.30.2")

# Set image path , mask path and prompt
item = 'foundation bottle'
image_path=f"input_images/{item}.jpg"
mask_path=f"mask_images/{item}.jpg"
prompts=[
    f"A {item} on the kitchen table.", 
    f"A lake surrounded by snow-capped mountains, with a {item} on a large rock in the lake.",
    f"A {item} on the sand beach.",
    f"A magnificent palace with a {item} placed on a table inside.",
    f"A bedroom with a {item} resting on silk sheets.",
    f"An autumn park with a {item} placed on a bench covered in fallen leaves.",
]

# Build pipeline
controlnet = FluxControlNetModel.from_pretrained("/home/jiangmuye/Models/FLUX.1-dev-Controlnet-Inpainting-Alpha", torch_dtype=torch.bfloat16)
transformer = FluxTransformer2DModel.from_pretrained(
        "/home/jiangmuye/Models/FLUX.1-dev", subfolder='transformer', torch_dytpe=torch.bfloat16
    )
pipe = FluxControlNetInpaintingPipeline.from_pretrained(
    "/home/jiangmuye/Models/FLUX.1-dev",
    controlnet=controlnet,
    transformer=transformer,
    torch_dtype=torch.bfloat16
).to("cuda:0")
pipe.transformer.to(torch.bfloat16)
pipe.controlnet.to(torch.bfloat16)

# Load image and mask
size = (768, 768)
image = load_image(image_path).convert("RGB").resize(size)
mask = load_image(mask_path).convert("RGB").resize(size)
#generator = torch.Generator(device="cuda").manual_seed(24)

# Inpaint
for i in range(len(prompts)):
    result = pipe(
        prompt=prompts[i],
        height=size[1],
        width=size[0],
        control_image=image,
        control_mask=mask,
        num_inference_steps=28,
        #generator=generator,
        controlnet_conditioning_scale=0.9,
        guidance_scale=3.5,
        negative_prompt="",
        true_guidance_scale=3.5
    ).images[0]

    result.save(f"output_images/{item}_{i}.jpg")


print("Successfully inpaint image")
