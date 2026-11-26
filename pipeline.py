# Model loading and image/prompt encoding, adapted from BELM (github.com/zituitui/BELM).
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

SD15 = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SMOOTH = "shi-labs/smooth-diffusion-lora"


def load_pipeline(model=SMOOTH, device="cuda"):
    if model == SMOOTH:
        pipe = StableDiffusionPipeline.from_pretrained(
            SD15, torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False
        )
        lora = hf_hub_download(repo_id=SMOOTH, filename="smooth_lora.safetensors")
        pipe.unet.load_attn_procs(lora, use_safetensors=True)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model, torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False
        )
    pipe.scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        timestep_spacing="linspace",
        set_alpha_to_one=False,
    )
    pipe.unet.set_attn_processor(AttnProcessor2_0())
    return pipe.to(device)


def center_crop(im):
    width, height = im.size
    side = min(width, height)
    box = ((width - side) / 2, (height - side) / 2, (width + side) / 2, (height + side) / 2)
    return im.crop(box)


def load_image(path):
    return center_crop(Image.open(path).convert("RGB")).resize((512, 512))


def encode_image(pipe, image):
    x = transforms.PILToTensor()(image).to(pipe.device)
    x = x / 255.0
    x = (x - 0.5) / 0.5
    # samples from the VAE posterior, so this draws from the global torch generator
    latent = pipe.vae.encode(x.unsqueeze(0), return_dict=False)[0].sample()
    return latent * pipe.vae.config.scaling_factor


def decode_latent(pipe, latent):
    image = pipe.vae.decode(latent / pipe.vae.config.scaling_factor, return_dict=False)[0]
    image = torch.clamp(image * 0.5 + 0.5, 0, 1)
    return transforms.ToPILImage()(image[0])


def encode_prompt(pipe, prompt, negative_prompt=""):
    def embed(text):
        ids = pipe.tokenizer(
            text,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        return pipe.text_encoder(ids.to(pipe.device))[0]

    return torch.cat([embed(negative_prompt), embed(prompt)])
