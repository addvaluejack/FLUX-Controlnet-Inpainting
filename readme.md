<div style="display: flex; justify-content: center; align-items: center;">
  <img src="images/alibaba.png" alt="alibaba" style="width: 20%; height: auto;">
  <img src="images/alimama.png" alt="alimama" style="width: 20%; height: auto;">
</div>

This repository provides a Inpainting ControlNet checkpoint for [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) model released by researchers from AlimamaCreative Team.

## Model Cards

* The model was trained on 12M laion2B and internal source images at resolution 768x768. The inference performs best at this size, with other sizes yielding suboptimal results.

* The recommended controlnet_conditioning_scale is 0.9 - 0.95.

* **Please note: This is only the alpha version during the training process. We will release an updated version when we feel ready.**

## Showcase

![flux1](images/flux1.jpg)
![flux2](images/flux2.jpg)
![flux3](images/flux3.jpg)

## Comparison with SDXL-Inpainting

Compared with [SDXL-Inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1)

From left to right: Input image | Masked image | SDXL inpainting | Ours

![0](images/0.jpg)
<small><i>*The image depicts a beautiful young woman sitting at a desk, reading a book. She has long, wavy brown hair and is wearing a grey shirt with a black cardigan. She is holding a pencil in her left hand and appears to be deep in thought. Surrounding her are numerous books, some stacked on the desk and others placed on a shelf behind her. A potted plant is also visible in the background, adding a touch of greenery to the scene. The image conveys a sense of serenity and intellectual pursuits.*</i></small>

![0](images/1.jpg)
<small><i>A woman with blonde hair is sitting on a table wearing a blue and white long dress. She is holding a green phone in her hand and appears to be taking a photo. There is a bag next to her on the table and a handbag beside her on the chair. The woman is looking at the phone with a smile on her face. The background includes a TV on the left wall and a couch on the right. A chair is also present in the scene.</i></small>

![0](images/2.jpg)
<small><i>The image is an illustration of a man standing in a cafe. He is wearing a white turtleneck, a camel-colored trench coat, and brown shoes. He is holding a cell phone and appears to be looking at it. There is a small table with a cup of coffee on it to his right. In the background, there is another man sitting at a table with a laptop. The man is wearing a black turtleneck and a tie. There are several cups and a cake on the table in the background. The man sitting at the table appears to be typing on the laptop.</i></small>

![0](images/3.jpg)
<small><i>The image depicts a scene from the anime series Dragon Ball Z, with the characters Goku, Naruto, and a child version of Gohan sharing a meal of ramen noodles. They are all sitting around a dining table, with Goku and Gohan on one side and Naruto on the other. They are all holding chopsticks and eating the noodles. The table is set with bowls of ramen, cups, and bowls of drinks. The arrangement of the characters and the food creates a sense of camaraderie and shared enjoyment of the meal.</i></small>

## Using with Diffusers


``` Shell
# install diffusers
pip install diffusers==0.30.2
# clone this repo
git clone https://github.com/alimama-creative/FLUX-Controlnet-Inpainting.git
# modify the image_path, mask_path, and prompt in main.py. Run：
python main.py
```

## LICENSE
Our weights fall under the [FLUX.1 [dev]](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md) Non-Commercial License.
