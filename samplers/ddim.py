# Adapted from BELM (github.com/zituitui/BELM).
from samplers.common import ddim_step, predict_noise, schedule

NFE_PER_STEP = 1


def invert(
    pipe, x, prompt_embeds, steps, guidance_scale, freeze_step=0,
    guidance_type="classifier-free",
):
    # inversion always uses classifier-free guidance
    timesteps, alpha_bars = schedule(pipe, steps)
    for i in reversed(range(freeze_step, steps)):
        t = timesteps[i + 1] if i < steps - 1 else 1
        eps = predict_noise(pipe, x, t, prompt_embeds, guidance_scale)
        x = ddim_step(x, eps, alpha_bars[i + 1], alpha_bars[i])
    return x


def sample(
    pipe, x, prompt_embeds, steps, guidance_scale, freeze_step=0,
    guidance_type="classifier-free",
):
    timesteps, alpha_bars = schedule(pipe, steps)
    for i in range(freeze_step, steps):
        eps = predict_noise(pipe, x, timesteps[i], prompt_embeds, guidance_scale, guidance_type)
        x = ddim_step(x, eps, alpha_bars[i], alpha_bars[i + 1])
    return x
