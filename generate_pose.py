from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image

name = 'woman_0'

init_image = load_image(f"input_images/{name}.jpg")

openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

control_image = openpose(init_image)
control_image = control_image.resize(init_image.size)
control_image.save(f"control_images/{name}.jpg")
