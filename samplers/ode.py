# The probability flow ODE in the forms used by the reversible Heun and EES solvers:
#   black_box               x over lambda = log(alpha / sigma)
#   semilinear              y = x / alpha (noise prediction) or y = x / sigma (data prediction)
#                           over lambda, which removes the linear part of the drift
#   original_ode_black_box  x over t, noise prediction only
#   sigma_bar_semilinear    y = x / alpha over sigma / alpha, noise prediction only (the DDIM ODE)
from samplers.continuous_time import (
    NoiseScheduleVP,
    data_prediction,
    expand_dims,
    noise_model,
    time_grid,
)


def lambda_drift(ns, eps, x, t, prediction):
    # dx/dlambda
    if prediction == "noise":
        noise = eps(x, t)
        sigma = ns.marginal_std(t)
        return expand_dims(sigma**2, x.dim()) * x - expand_dims(sigma, x.dim()) * noise
    x0 = data_prediction(ns, eps, x, t)
    alpha = ns.marginal_alpha(t)
    return -expand_dims(alpha**2, x.dim()) * x + expand_dims(alpha, x.dim()) * x0


def right_hand_side(ns, eps, formulation, prediction):
    def black_box(x, lam):
        t = ns.inverse_lambda(lam.reshape(1)).expand(x.shape[0])
        return lambda_drift(ns, eps, x, t, prediction)

    def semilinear(y, lam):
        t = ns.inverse_lambda(lam.reshape(1)).expand(y.shape[0])
        alpha, sigma = ns.marginal_alpha(t), ns.marginal_std(t)
        if prediction == "noise":
            noise = eps(expand_dims(alpha, y.dim()) * y, t)
            return -expand_dims(sigma / alpha, y.dim()) * noise
        x0 = data_prediction(ns, eps, expand_dims(sigma, y.dim()) * y, t)
        return expand_dims(alpha / sigma, y.dim()) * x0

    def original_ode_black_box(x, t):
        t = t.reshape(1).expand(x.shape[0])
        sigma = ns.marginal_std(t)
        log_alpha_dot = ns.marginal_log_mean_coeff_derivative(t)
        noise = eps(x, t)
        return expand_dims(log_alpha_dot, x.dim()) * (
            x - noise * expand_dims(sigma.reciprocal(), x.dim())
        )

    def sigma_bar_semilinear(y, sigma_bar):
        t = ns.inverse_sigma_bar(sigma_bar.reshape(1)).expand(y.shape[0])
        alpha = ns.marginal_alpha(t)
        return eps(expand_dims(alpha, y.dim()) * y, t)

    forms = {
        "black_box": black_box,
        "semilinear": semilinear,
        "original_ode_black_box": original_ode_black_box,
        "sigma_bar_semilinear": sigma_bar_semilinear,
    }
    return forms[formulation]


def setup(
    pipe, x, prompt_embeds, steps, guidance_scale, freeze_step, guidance_type,
    prediction, formulation, schedule, inverting,
):
    # Returns the right-hand side f(y, s), the grid in s, the starting y and the map from y back
    # to x. Inversion runs from t = 1/N to t = 1, generation the other way.
    assert prediction == "noise" or formulation in ("black_box", "semilinear")
    ns = NoiseScheduleVP(pipe.scheduler.betas)
    eps = noise_model(pipe, ns, prompt_embeds, guidance_scale, guidance_type)
    time_steps = time_grid(ns, schedule, steps, freeze_step, inverting, x.device)
    f = right_hand_side(ns, eps, formulation, prediction)

    if formulation == "original_ode_black_box":
        grid = time_steps
    elif formulation == "sigma_bar_semilinear":
        grid = ns.marginal_sigma_bar(time_steps)
    else:
        grid = ns.marginal_lambda(time_steps)

    if formulation in ("black_box", "original_ode_black_box"):
        return f, grid, x, lambda y: y
    scale = ns.marginal_alpha if prediction == "noise" else ns.marginal_std
    scale_start, scale_end = scale(time_steps[0]), scale(time_steps[-1])
    y = x / expand_dims(scale_start, x.dim())
    return f, grid, y, lambda y: expand_dims(scale_end, y.dim()) * y
