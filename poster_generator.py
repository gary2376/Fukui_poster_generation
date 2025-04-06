from PIL import Image, ImageOps
import torch
import random
import os

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLInpaintPipeline,
)
from controlnet_aux import ZoeDetector


def generate_final_poster_image(prompt, negative_prompt, original_image_path=None):
    """
    圖像生成主函數，輸入 prompt / negative_prompt，輸出 PIL.Image
    original_image_path 可選，如未提供則預設使用 combined.png
    """
    if original_image_path is None:
        original_image_path = "E:/python_project/contest/poster_fukui/code/full_function/data/temp/combined.png"

    # 讀取原始圖片
    original_image = Image.open(original_image_path).convert("RGBA")

    # 調整尺寸到 704x952（8 的倍數）
    canvas_size = (704, 952)
    resized_img = original_image.resize(canvas_size, Image.LANCZOS)
    white_bg_image = Image.new("RGBA", canvas_size, "white")
    white_bg_image.paste(resized_img, (0, 0), resized_img)

    # 建立 Zoe 深度圖
    zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators")
    image_zoe = zoe(white_bg_image.resize(canvas_size, Image.LANCZOS), detect_resolution=704, image_resolution=952)
    image_zoe = image_zoe.resize(canvas_size, Image.LANCZOS)

    # 載入 ControlNet 模型（延後載入以節省記憶體）
    controlnets = [
        ControlNetModel.from_pretrained(
            "destitech/controlnet-inpaint-dreamer-sdxl", torch_dtype=torch.float16, variant="fp16"
        ),
        ControlNetModel.from_pretrained("diffusers/controlnet-zoe-depth-sdxl-1.0", torch_dtype=torch.float16),
    ]

    # 載入 VAE
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")

    # 建立 SDXL ControlNet pipeline
    pipe_control = StableDiffusionXLControlNetPipeline.from_pretrained(
        "SG161222/RealVisXL_V4.0",
        torch_dtype=torch.float16,
        variant="fp16",
        controlnet=controlnets,
        vae=vae,
    ).to("cuda")

    # 第一階段生成 temp_image
    seed1 = random.randint(0, 2**32 - 1)
    generator1 = torch.Generator(device="cuda").manual_seed(seed1)
    temp_image = pipe_control(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=[white_bg_image, image_zoe],
        guidance_scale=8,
        num_inference_steps=30,
        generator=generator1,
        strength=0.3,
        controlnet_conditioning_scale=[0.6, 0.8],
        control_guidance_end=[0.9, 0.6],
        height=952,
        width=704,
    ).images[0]

    # 貼回原圖
    temp_image.paste(resized_img, (0, 0), resized_img)

    # 建立遮罩
    mask = Image.new("L", temp_image.size)
    mask.paste(resized_img.split()[3], (0, 0))
    mask = ImageOps.invert(mask)
    final_mask = mask.point(lambda p: p > 128 and 255)

    # 清除 control pipeline 釋放記憶體
    pipe_control = None
    torch.cuda.empty_cache()

    # 載入 SDXL Inpainting pipeline
    pipe_inpaint = StableDiffusionXLInpaintPipeline.from_pretrained(
        "OzzyGT/RealVisXL_V4.0_inpainting",
        torch_dtype=torch.float16,
        variant="fp16",
        vae=vae,
    ).to("cuda")

    # 模糊遮罩
    mask_blurred = pipe_inpaint.mask_processor.blur(final_mask, blur_factor=20)

    # 第二階段 outpaint
    seed2 = random.randint(0, 2**32 - 1)
    generator2 = torch.Generator(device="cuda").manual_seed(seed2)
    final_image = pipe_inpaint(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=temp_image,
        mask_image=mask_blurred,
        guidance_scale=8.0,
        strength=0.3,
        num_inference_steps=30,
        generator=generator2,
        height=952,
        width=704,
    ).images[0]

    # 最終合併貼回原圖區域
    final_image = Image.alpha_composite(final_image.convert("RGBA"), resized_img)

    return final_image
