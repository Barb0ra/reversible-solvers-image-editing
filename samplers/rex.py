# Rex (Blasingame and Liu, 2025) with an Euler or midpoint base method. The base method solves
# dz/dchi = eps(alpha z, t) in chi = sigma / alpha and z = x / alpha, on the continuous-time version
# of the scaled-linear schedule; x_hat is the second state of the McCallum-Foster scheme.
import torch

from samplers.common import predict_noise


def continuous_betas(scheduler):
    n = scheduler.config.num_train_timesteps
    return n * scheduler.config.beta_start, n * scheduler.config.beta_end


def alpha_sigma(t, scheduler):
    # sqrt(beta(t)) is linear in t
    beta0, beta1 = continuous_betas(scheduler)
    r = (beta0 * beta1) ** 0.5
    delta = beta1 - 2.0 * r + beta0
    log_alpha = -(delta / 6.0) * t**3 - ((r - beta0) / 2.0) * t**2 - (beta0 / 2.0) * t
    alpha = torch.exp(log_alpha)
    sigma = torch.sqrt(torch.clamp(1.0 - alpha * alpha, min=0.0))
    return alpha, sigma


def alpha_from_chi(chi):
    chi = torch.tensor(chi)
    return torch.rsqrt(1.0 + chi * chi)


def t_from_chi(chi, scheduler):
    # inverts chi(t) = sigma / alpha through the real root of a cubic in t
    beta0, beta1 = continuous_betas(scheduler)
    r = (beta0 * beta1) ** 0.5
    s = r - beta0
    delta = beta1 - 2.0 * r + beta0
    chi = torch.tensor(chi)
    log_term = torch.log1p(chi * chi)
    cube_root_arg = 2.0 * (s**3) - 3.0 * beta0 * delta * s - 3.0 * (delta**2) * log_term
    cube_root = torch.sign(cube_root_arg) * torch.abs(cube_root_arg).pow(1.0 / 3.0)
    t = (beta0 - r - cube_root) / delta
    return torch.clamp(t, 0.0, 1.0)


def chi_grid(pipe, steps):
    # steps + 1 grid points from the last timestep down to t = 0
    scheduler = pipe.scheduler
    scheduler.set_timesteps(steps + 1, device=pipe.device)
    timesteps = scheduler.timesteps.to(torch.float32)
    t = timesteps / float(scheduler.config.num_train_timesteps - 1)
    alpha, sigma = alpha_sigma(t, scheduler)
    return (sigma / alpha).tolist()


def noise_model(pipe, prompt_embeds, guidance_scale, guidance_type):
    def eps(z, chi):
        x = alpha_from_chi(chi).to(z.device) * z
        n = pipe.scheduler.config.num_train_timesteps
        t = t_from_chi(chi, pipe.scheduler).to(z.device) * float(n - 1)
        return predict_noise(pipe, x, t, prompt_embeds, guidance_scale, guidance_type)

    return eps


def psi(eps, x, chi, h, base):
    # increment of the base method in z over [chi, chi + h], starting from z = x / alpha(chi)
    z = x / alpha_from_chi(chi).to(x.device)
    k = eps(z, chi)
    if base == "midpoint":
        k = eps(z + 0.5 * h * k, chi + 0.5 * h)
    return h * k


def rex_step(eps, x, x_hat, chi, chi_next, coupling, base):
    h = chi_next - chi
    alpha = alpha_from_chi(chi).to(x.device)
    alpha_next = alpha_from_chi(chi_next).to(x.device)
    x_next = (alpha_next / alpha) * (coupling * x + (1.0 - coupling) * x_hat) + alpha_next * psi(
        eps, x_hat, chi, h, base
    )
    x_hat_next = (alpha_next / alpha) * x_hat - alpha_next * psi(eps, x_next, chi_next, -h, base)
    return x_next, x_hat_next


def rex_step_back(eps, x_next, x_hat_next, chi, chi_next, coupling, base):
    # exact inverse of rex_step
    h = chi_next - chi
    alpha = alpha_from_chi(chi).to(x_next.device)
    alpha_next = alpha_from_chi(chi_next).to(x_next.device)
    coupling_inv = 1.0 / coupling
    x_hat = (alpha / alpha_next) * x_hat_next + alpha * psi(eps, x_next, chi_next, -h, base)
    x = (
        (alpha / alpha_next) * (coupling_inv * x_next)
        + (1.0 - coupling_inv) * x_hat
        - alpha * coupling_inv * psi(eps, x_hat, chi, h, base)
    )
    return x, x_hat


def invert(
    pipe, x, prompt_embeds, steps, guidance_scale, freeze_step=0,
    guidance_type="classifier-free", base="euler", coupling=0.999,
):
    # inversion always uses classifier-free guidance
    eps = noise_model(pipe, prompt_embeds, guidance_scale, "classifier-free")
    chi = chi_grid(pipe, steps)
    x_hat = x
    for n in reversed(range(freeze_step, steps)):
        x, x_hat = rex_step_back(eps, x, x_hat, chi[n], chi[n + 1], coupling, base)
    return x, x_hat


def sample(
    pipe, state, prompt_embeds, steps, guidance_scale, freeze_step=0,
    guidance_type="classifier-free", base="euler", coupling=0.999,
):
    x, x_hat = state
    eps = noise_model(pipe, prompt_embeds, guidance_scale, guidance_type)
    chi = chi_grid(pipe, steps)
    for n in range(freeze_step, steps):
        x, x_hat = rex_step(eps, x, x_hat, chi[n], chi[n + 1], coupling, base)
    return x
