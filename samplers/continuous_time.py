# Shared pieces of the continuous-time samplers, adapted from DPM-Solver
# (github.com/LuChengTHU/dpm-solver).
import torch

from samplers.common import predict_noise


def interpolate_fn(x, xp, yp):
    # piecewise linear function through the keypoints (xp, yp), extended linearly outside them;
    # x has shape [N, C], xp and yp have shape [C, K]
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(
        y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)
    ).squeeze(2)
    return start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)


def expand_dims(v, dims):
    return v[(...,) + (None,) * (dims - 1)]


class NoiseScheduleVP:
    # The discrete schedule as a function of t in [1/N, 1]: log(alpha) is interpolated linearly
    # between the N training times t_n = n / N.
    def __init__(self, betas):
        log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
        self.T = 1.0
        self.log_alpha_array = self.numerical_clip_alpha(log_alphas).reshape((1, -1))
        self.log_alpha_array = self.log_alpha_array.to(dtype=torch.float32)
        self.total_N = self.log_alpha_array.shape[1]
        self.t_array = torch.linspace(0.0, 1.0, self.total_N + 1)[1:].reshape((1, -1))
        self.t_array = self.t_array.to(dtype=torch.float32)

    def numerical_clip_alpha(self, log_alphas, clipped_lambda=-5.1):
        log_sigmas = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_alphas))
        lambs = log_alphas - log_sigmas
        idx = torch.searchsorted(torch.flip(lambs, [0]), clipped_lambda)
        if idx > 0:
            log_alphas = log_alphas[:-idx]
        return log_alphas

    def marginal_log_mean_coeff(self, t):
        return interpolate_fn(
            t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)
        ).reshape((-1))

    def marginal_log_mean_coeff_derivative(self, t):
        xp = self.t_array.to(t.device).squeeze(0)
        yp = self.log_alpha_array.to(t.device).squeeze(0)
        j = torch.clamp(torch.searchsorted(xp, t, right=False) - 1, min=0, max=xp.shape[0] - 2)
        return (yp[j + 1] - yp[j]) / (xp[j + 1] - xp[j])

    def marginal_alpha(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        return torch.sqrt(1.0 - torch.exp(2.0 * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2.0 * lamb)
        t = interpolate_fn(
            log_alpha.reshape((-1, 1)),
            torch.flip(self.log_alpha_array.to(lamb.device), [1]),
            torch.flip(self.t_array.to(lamb.device), [1]),
        )
        return t.reshape((-1,))

    def marginal_sigma_bar(self, t):
        # sigma / alpha = exp(-lambda)
        return torch.exp(-self.marginal_lambda(t))

    def inverse_sigma_bar(self, sigma_bar):
        return self.inverse_lambda(-torch.log(sigma_bar))


def time_grid(ns, schedule, steps, freeze_step, inverting, device):
    # Times from 1/N to T for inversion and from T to 1/N for generation, uniform in logSNR,
    # in t or in sigma / alpha, without the first freeze_step steps of generation.
    t_start, t_end = (1.0 / ns.total_N, ns.T) if inverting else (ns.T, 1.0 / ns.total_N)
    if schedule == "logSNR":
        lambda_start = ns.marginal_lambda(torch.tensor(t_start).to(device))
        lambda_end = ns.marginal_lambda(torch.tensor(t_end).to(device))
        lambdas = torch.linspace(lambda_start.item(), lambda_end.item(), steps + 1).to(device)
        time_steps = ns.inverse_lambda(lambdas)
    elif schedule == "sigma_bar":
        sigma_bar_start = ns.marginal_sigma_bar(torch.tensor(t_start).to(device))
        sigma_bar_end = ns.marginal_sigma_bar(torch.tensor(t_end).to(device))
        sigma_bars = torch.linspace(sigma_bar_start.item(), sigma_bar_end.item(), steps + 1)
        time_steps = ns.inverse_sigma_bar(sigma_bars.to(device))
    else:
        time_steps = torch.linspace(t_start, t_end, steps + 1).to(device)
    if inverting:
        return time_steps[: steps + 1 - freeze_step]
    return time_steps[freeze_step:]


def noise_model(pipe, ns, prompt_embeds, guidance_scale, guidance_type):
    def eps(x, t):
        # the network takes the discrete time n = N * t - 1
        t_input = (t - 1.0 / ns.total_N) * 1000.0
        return predict_noise(pipe, x, t_input, prompt_embeds, guidance_scale, guidance_type)

    return eps


def data_prediction(ns, eps, x, t):
    noise = eps(x, t)
    alpha_t, sigma_t = ns.marginal_alpha(t), ns.marginal_std(t)
    return (x - expand_dims(sigma_t, x.dim()) * noise) / expand_dims(alpha_t, x.dim())
