# BDIA (Zhang et al., 2024), adapted from BELM (github.com/zituitui/BELM).
from samplers.common import alpha_sigma, ddim_coefficients, ddim_step, predict_noise, schedule

NFE_PER_STEP = 1


def invert(
    pipe, x, prompt_embeds, steps, guidance_scale, freeze_step=0,
    guidance_type="classifier-free", gamma=0.96,
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
            coef_x_i_minus_2 = 1 / gamma
            coef_x_i_minus_1 = alpha_i / alpha_i_minus_1 - (1 / gamma) * (
                alpha_i_minus_2 / alpha_i_minus_1
            )
            coef_eps = (sigma_i - (alpha_i / alpha_i_minus_1) * sigma_i_minus_1) - (1 / gamma) * (
                sigma_i_minus_2 - (alpha_i_minus_2 / alpha_i_minus_1) * sigma_i_minus_1
            )
            x_next = coef_x_i_minus_2 * x_prev + coef_x_i_minus_1 * x + coef_eps * eps
        x_prev, x = x, x_next
    return x, x_prev


def sample(
    pipe, state, prompt_embeds, steps, guidance_scale, freeze_step=0,
    guidance_type="classifier-free", gamma=0.96,
):
    # state holds the latents at the first two timesteps of generation
    x_prev, x = state
    timesteps, alpha_bars = schedule(pipe, steps)
    for i in range(freeze_step + 1, steps):
        eps = predict_noise(pipe, x, timesteps[i], prompt_embeds, guidance_scale, guidance_type)
        alpha_t, sigma_t = alpha_sigma(alpha_bars[i])
        alpha_p, sigma_p = alpha_sigma(alpha_bars[i - 1])
        coef_x, coef_eps = ddim_coefficients(alpha_bars[i], alpha_bars[i + 1])
        coef_x = coef_x - gamma * alpha_p / alpha_t
        coef_eps_2 = sigma_p - sigma_t * alpha_p / alpha_t
        coef_eps = coef_eps - gamma * coef_eps_2
        x_prev, x = x, gamma * x_prev + coef_x * x + coef_eps * eps
    return x
