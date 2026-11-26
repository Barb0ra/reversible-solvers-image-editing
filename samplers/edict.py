# EDICT (Wallace et al., 2023), adapted from BELM (github.com/zituitui/BELM).
from samplers.common import ddim_coefficients, ddim_step, predict_noise, schedule

NFE_PER_STEP = 2


def invert(
    pipe, x, prompt_embeds, steps, guidance_scale, freeze_step=0,
    guidance_type="classifier-free", p=0.93,
):
    # inversion always uses classifier-free guidance
    timesteps, alpha_bars = schedule(pipe, steps)
    y = x
    for i in reversed(range(freeze_step, steps)):
        t = timesteps[i]
        a, b = ddim_coefficients(alpha_bars[i], alpha_bars[i + 1])
        y_inter = (y - (1 - p) * x) / p
        x_inter = (x - (1 - p) * y_inter) / p
        eps = predict_noise(pipe, x_inter, t, prompt_embeds, guidance_scale)
        y = (y_inter - b * eps) / a
        eps = predict_noise(pipe, y, t, prompt_embeds, guidance_scale)
        x = (x_inter - b * eps) / a
    return x, y


def sample(
    pipe, state, prompt_embeds, steps, guidance_scale, freeze_step=0,
    guidance_type="classifier-free", p=0.93,
):
    x, y = state
    timesteps, alpha_bars = schedule(pipe, steps)
    for i in range(freeze_step, steps):
        t = timesteps[i]
        eps = predict_noise(pipe, y, t, prompt_embeds, guidance_scale, guidance_type)
        x_inter = ddim_step(x, eps, alpha_bars[i], alpha_bars[i + 1])
        eps = predict_noise(pipe, x_inter, t, prompt_embeds, guidance_scale, guidance_type)
        y_inter = ddim_step(y, eps, alpha_bars[i], alpha_bars[i + 1])
        x = p * x_inter + (1.0 - p) * y_inter
        y = p * y_inter + (1.0 - p) * x
    return x
