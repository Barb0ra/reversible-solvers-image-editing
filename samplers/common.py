# Shared pieces of the discrete-time samplers, adapted from BELM (github.com/zituitui/BELM).
import torch


def schedule(pipe, steps):
    pipe.scheduler.set_timesteps(steps, device=pipe.device)
    timesteps = pipe.scheduler.timesteps
    # alpha_bar at each timestep, with alpha_bar = 1 appended for the clean image
    alpha_bars = [pipe.scheduler.alphas_cumprod[t].to(torch.float32) for t in timesteps]
    return timesteps, alpha_bars + [1]


def alpha_sigma(alpha_bar):
    return alpha_bar**0.5, (1 - alpha_bar) ** 0.5


def ddim_coefficients(alpha_bar_t, alpha_bar_s):
    # x_s = coef_x * x_t + coef_eps * eps
    alpha_t, sigma_t = alpha_sigma(alpha_bar_t)
    alpha_s, sigma_s = alpha_sigma(alpha_bar_s)
    coef_x = alpha_s / alpha_t
    return coef_x, sigma_s - sigma_t * coef_x


def ddim_step(x, eps, alpha_bar_t, alpha_bar_s):
    coef_x, coef_eps = ddim_coefficients(alpha_bar_t, alpha_bar_s)
    return coef_x * x + coef_eps * eps


def proxedit_noise_pred(eps_src, eps_tgt, guidance_scale, quantile=0.7):
    # proximal guidance: keep only the largest entries of the guidance direction
    delta = eps_tgt - eps_src
    B = delta.shape[0]
    tau = torch.quantile(delta.abs().reshape(B, -1), quantile, dim=1, keepdim=True)
    tau = tau.view(B, *([1] * (delta.dim() - 1)))
    mask = delta.abs() > tau
    return eps_src + guidance_scale * (delta * mask)


def predict_noise(pipe, x, t, prompt_embeds, guidance_scale, guidance_type="classifier-free"):
    noise_pred = pipe.unet(
        torch.cat([x] * 2), t, encoder_hidden_states=prompt_embeds, return_dict=False
    )[0]
    eps_uncond, eps_cond = noise_pred.chunk(2)
    if guidance_type == "proxedit":
        return proxedit_noise_pred(eps_uncond, eps_cond, guidance_scale)
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)
