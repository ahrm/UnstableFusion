from lib2to3.pytree import NegatedPattern
from urllib import request
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionInpaintPipelineLegacy

from torch import autocast
import torch
from base64 import encodebytes, decodebytes

from flask import Flask, jsonify, request

try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    from flask_cloudflared import run_with_cloudflared

dummy_safety_checker = lambda images, **kwargs: (images, [False] * len(images))

class StableDiffusionHandler:
    def __init__(self, token=True):
        self.text2img = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
            revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=token).to("cuda")

        # self.text2img = StableDiffusionPipeline.from_pretrained(
        # "runwayml/stable-diffusion-v1-5",
        #     revision="fp16",
        #     torch_dtype=torch.float16,
        #     use_auth_token=token).to("cuda")

        # self.text2img.safety_checker = dummy_safety_checker

        self.inpainter = StableDiffusionInpaintPipelineLegacy(
            vae=self.text2img.vae,
            text_encoder=self.text2img.text_encoder,
            tokenizer=self.text2img.tokenizer,
            unet=self.text2img.unet,
            scheduler=self.text2img.scheduler,
            safety_checker=self.text2img.safety_checker,
            feature_extractor=self.text2img.feature_extractor
        ).to("cuda")

        self.img2img = StableDiffusionImg2ImgPipeline(
            unet=self.text2img.unet,
            scheduler=self.text2img.scheduler,
            vae=self.text2img.vae,
            text_encoder=self.text2img.text_encoder,
            tokenizer=self.text2img.tokenizer,
            safety_checker=self.text2img.safety_checker,
            feature_extractor=self.text2img.feature_extractor
        ).to("cuda")

        self.inpainter.safety_checker = dummy_safety_checker
        self.img2img.safety_checker = dummy_safety_checker
        self.text2img.safety_checker = dummy_safety_checker

        from gfpgan import GFPGANer

        self.gfpgan = GFPGANer(model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth", upscale=1, arch='clean',
                                          channel_multiplier=2, bg_upsampler=None)

    def get_generator(self, seed):
        if seed == -1:
            return None
        else:
            return torch.Generator("cuda").manual_seed(seed)

    def inpaint(self, prompt, image, mask, strength=0.75, steps=50, guidance_scale=7.5, seed=-1, callback=None, negative_prompt=None):
        print(f'Inpainting with strength {strength}, steps {steps}, guidance_scale {guidance_scale}, seed {seed}')
        image_ = Image.fromarray(image.astype(np.uint8)).resize((512, 512), resample=Image.Resampling.LANCZOS)
        mask_ = Image.fromarray(mask.astype(np.uint8)).resize((512, 512), resample=Image.Resampling.LANCZOS)

        with autocast("cuda"):
            im = self.inpainter(
                prompt=prompt,
                init_image=image_,
                mask_image=mask_,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=self.get_generator(seed),
                callback=callback,
                negative_prompt=negative_prompt
            )[0][0]

            cropped_faces, restored_faces, restored_img  = self.gfpgan.enhance(np.array(im)[:,:,::-1],
                                                                               has_aligned=False, only_center_face=False, paste_back=True, weight=0.5)
            gfpgan_sample = restored_img[:,:,::-1]
            im = Image.fromarray(gfpgan_sample)

            return im.resize((image.shape[1], image.shape[0]), resample=Image.Resampling.LANCZOS)

    def generate(self, prompt, width=512, height=512, strength=0.75, steps=50, guidance_scale=7.5,seed=-1, callback=None, negative_prompt=None):
        print(f'Generating with strength {strength}, steps {steps}, guidance_scale {guidance_scale}, seed {seed}')

        with autocast("cuda"):
            im = self.text2img(
                prompt=prompt,
                width=512,
                height=512,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                callback=callback,
                negative_prompt=negative_prompt,
                generator=self.get_generator(seed)
            )[0][0]

            cropped_faces, restored_faces, restored_img  = self.gfpgan.enhance(np.array(im)[:,:,::-1],
                                                                               has_aligned=False, only_center_face=False, paste_back=True, weight=0.5)
            gfpgan_sample = restored_img[:,:,::-1]
            im = Image.fromarray(gfpgan_sample)

            return im.resize((width, height), resample=Image.Resampling.LANCZOS)

    def reimagine(self, prompt, image, steps=50, guidance_scale=7.5, seed=-1, strength=0.75, callback=None, negative_prompt=None):

        print(f'Reimagining with strength {strength} steps {steps}, guidance_scale {guidance_scale}, seed {seed}')
        image_ = Image.fromarray(image.astype(np.uint8)).resize((512, 512), resample=Image.Resampling.LANCZOS)
        with autocast("cuda"):
            results = self.img2img(
                [prompt],
                init_image=image_,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=self.get_generator(seed),
                negative_prompt=negative_prompt,
                callback=callback
            )[0]

            im = results[0]

            cropped_faces, restored_faces, restored_img  = self.gfpgan.enhance(np.array(im)[:,:,::-1],
                                                                               has_aligned=False, only_center_face=False, paste_back=True, weight=0.5)
            gfpgan_sample = restored_img[:,:,::-1]
            im = Image.fromarray(gfpgan_sample)

            #print(len(results))
            im = results[0]
            return im.resize((image.shape[1], image.shape[0]), resample=Image.LANCZOS)

def run_app():
    app = Flask(__name__)
    if IN_COLAB:
        run_with_cloudflared(app)
    stable_diffusion_handler = StableDiffusionHandler()

    @app.route('/')
    def home():
        return "Game Over!"

    @app.route('/reimagine', methods=['POST'])
    def reimagine():
        # get request data
        data = request.get_json()
        prompt = data["prompt"]
        negative_prompt = data.get("negative_prompt", None)
        steps = data["steps"]
        guidance_scale = data["guidance_scale"]
        seed = data["seed"]
        strength = data["strength"]
        image = np.array(data['image'])

        generated = stable_diffusion_handler.reimagine(
            prompt,
            image,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            strength=strength,
            negative_prompt=negative_prompt)

        return jsonify({
            "status": "success",
             "image_size": generated.size,
             "image_mode": generated.mode,
             "image_data": encodebytes(generated.tobytes()).decode('ascii')
             })

    @app.route('/inpaint', methods=['POST'])
    def inpaint():
        # get request data
        data = request.get_json()
        prompt = data["prompt"]
        negative_prompt = data.get("negative_prompt", None)
        strength = data["strength"]
        steps = data["steps"]
        guidance_scale = data["guidance_scale"]
        seed = data["seed"]
        image = np.array(data['image'])
        mask = np.array(data['mask'])

        generated = stable_diffusion_handler.inpaint(
            prompt,
            image,
            mask,
            strength=strength,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            negative_prompt=negative_prompt)

        return jsonify({
            "status": "success",
             "image_size": generated.size,
             "image_mode": generated.mode,
             "image_data": encodebytes(generated.tobytes()).decode('ascii')
             })

    @app.route('/generate', methods=['POST'])
    def generate():
        # get request data
        data = request.get_json()
        prompt = data["prompt"]
        negative_prompt = data.get("negative_prompt", None)
        strength = data["strength"]
        steps = data["steps"]
        guidance_scale = data["guidance_scale"]
        seed = data["seed"]
        width = data["width"]
        height = data["height"]

        generated = stable_diffusion_handler.generate(
            prompt,
            strength=strength,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            width=width,
            height=height,
            negative_prompt=negative_prompt)

        return jsonify({
            "status": "success",
             "image_size": generated.size,
             "image_mode": generated.mode,
             "image_data": encodebytes(generated.tobytes()).decode('ascii')
             })


    app.run(debug=True)

if __name__ == '__main__':
    run_app()
