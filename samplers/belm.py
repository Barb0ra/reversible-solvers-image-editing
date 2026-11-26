# BELM (Wang et al., 2024), adapted from github.com/zituitui/BELM.
from samplers.common import alpha_sigma, ddim_step, predict_noise, schedule

NFE_PER_STEP = 1


def invert(
    pipe, x, prompt_embeds, steps, guidance_scale, freeze_step=0,
    guidance_type="classifier-free",
):
    # inversion always uses classifier-free guidance
    timesteps, alpha_bars = schedule(pipe, steps)
    x_prev = None
    for i in reversed(range(freeze_step, steps)):
        t = timesteps[i + 1] if i < steps - 1 else 0
        eps = predict_noise(pipe, x, t, prompt_embeds, guidance_scale)
        if i == steps - 1:
            x_next = ddim_step(x, eps, alpha_bars[i + 1], alpha_bars[i])
        else:
            # inverse of the step in sample(); i - 1 and i - 2 are the two less noisy timesteps
            alpha_i, sigma_i = alpha_sigma(alpha_bars[i])
            alpha_i_minus_1, sigma_i_minus_1 = alpha_sigma(alpha_bars[i + 1])
            alpha_i_minus_2, sigma_i_minus_2 = alpha_sigma(alpha_bars[i + 2])
            h_i = sigma_i / alpha_i - sigma_i_minus_1 / alpha_i_minus_1
            h_i_minus_1 = sigma_i_minus_1 / alpha_i_minus_1 - sigma_i_minus_2 / alpha_i_minus_2
            coef_x_i_minus_2 = (alpha_i / alpha_i_minus_2) * (h_i**2) / (h_i_minus_1**2)
            coef_x_i_minus_1 = (
                (alpha_i / alpha_i_minus_1) * (h_i_minus_1**2 - h_i**2) / (h_i_minus_1**2)
            )
            coef_eps = alpha_i * (h_i_minus_1 + h_i) * h_i / h_i_minus_1
            x_next = coef_x_i_minus_2 * x_prev + coef_x_i_minus_1 * x + coef_eps * eps
        x_prev, x = x, x_next
    return x, x_prev


def sample(
    pipe, state, prompt_embeds, steps, guidance_scale, freeze_step=0,
    guidance_type="classifier-free",
):
    # state holds the latents at the first two timesteps of generation
    x_prev, x = state
    timesteps, alpha_bars = schedule(pipe, steps)
    for i in range(freeze_step + 1, steps):
        eps = predict_noise(pipe, x, timesteps[i], prompt_embeds, guidance_scale, guidance_type)
        alpha_p, sigma_p = alpha_sigma(alpha_bars[i - 1])
        alpha_t, sigma_t = alpha_sigma(alpha_bars[i])
        alpha_s, sigma_s = alpha_sigma(alpha_bars[i + 1])
        # step sizes in sigma/alpha
        delta_1 = sigma_t / alpha_t - sigma_p / alpha_p
        delta_2 = sigma_s / alpha_s - sigma_t / alpha_t
        delta_3 = sigma_s / alpha_s - sigma_p / alpha_p
        coef_1 = delta_2 * delta_3 * alpha_s / delta_1
        coef_2 = (delta_2 / delta_1) ** 2 * (alpha_s / alpha_p)
        coef_3 = (delta_1 - delta_2) * delta_3 / (delta_1**2) * (alpha_s / alpha_t)
        x_prev, x = x, coef_1 * eps + coef_2 * x_prev + coef_3 * x
    return x
