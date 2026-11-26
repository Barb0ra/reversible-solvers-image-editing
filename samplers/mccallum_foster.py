# McCallum-Foster reversible scheme (McCallum and Foster, 2024) with an Euler or midpoint base
# method, on the black-box ODE for the data prediction in lambda = log(alpha / sigma), with steps
# uniform in t. Generation undoes the inversion pass step by step.
from samplers.continuous_time import NoiseScheduleVP, noise_model, time_grid
from samplers.ode import lambda_drift


def increments(base, ns, drift, t, lam):
    # Increments of the base method over the step from grid point a to b, driven by x_hat (added
    # to x) and by x (added to x_hat). scale multiplies the increment.
    if base == "euler":
        def from_x_hat(x_hat, a, b, scale=1.0):
            return scale * (lam[b] - lam[a]) * drift(x_hat, t[a].reshape(1))

        def from_x(x, a, b):
            return (lam[b] - lam[a]) * drift(x, t[b].reshape(1))

        return from_x_hat, from_x

    def midpoint(x, a, b, scale=1.0):
        h = lam[b] - lam[a]
        t_mid = ns.inverse_lambda((0.5 * (lam[a] + lam[b])).reshape(1))
        k = drift(x, ns.inverse_lambda(lam[a].reshape(1)))
        k = drift(x + 0.5 * h * k, t_mid)
        return scale * (h * k)

    def from_x(x, a, b):
        return -midpoint(x, b, a)

    return midpoint, from_x


def setup(
    pipe, x, prompt_embeds, steps, guidance_scale, freeze_step, guidance_type, base, inverting,
):
    ns = NoiseScheduleVP(pipe.scheduler.betas)
    eps = noise_model(pipe, ns, prompt_embeds, guidance_scale, guidance_type)
    t = time_grid(ns, "time_uniform", steps, freeze_step, inverting, x.device)

    def drift(x, t):
        return lambda_drift(ns, eps, x, t.expand(x.shape[0]), "data")

    from_x_hat, from_x = increments(base, ns, drift, t, ns.marginal_lambda(t))
    return from_x_hat, from_x, len(t) - 1


def invert(
    pipe, x, prompt_embeds, steps, guidance_scale, freeze_step=0,
    guidance_type="classifier-free", base="euler", coupling=0.999,
):
    # inversion always uses classifier-free guidance
    from_x_hat, from_x, n = setup(
        pipe, x, prompt_embeds, steps, guidance_scale, freeze_step, "classifier-free", base,
        inverting=True,
    )
    x_hat = x
    for i in range(n):
        x_next = coupling * x + (1.0 - coupling) * x_hat + from_x_hat(x_hat, i, i + 1)
        x_hat = x_hat + from_x(x_next, i, i + 1)
        x = x_next
    return x, x_hat


def sample(
    pipe, state, prompt_embeds, steps, guidance_scale, freeze_step=0,
    guidance_type="classifier-free", base="euler", coupling=0.999,
):
    x, x_hat = state
    from_x_hat, from_x, n = setup(
        pipe, x, prompt_embeds, steps, guidance_scale, freeze_step, guidance_type, base,
        inverting=False,
    )
    coupling_inv = 1.0 / coupling
    for i in range(n):
        x_hat = x_hat - from_x(x, i + 1, i)
        x = (
            coupling_inv * x
            + (1.0 - coupling_inv) * x_hat
            - from_x_hat(x_hat, i + 1, i, scale=coupling_inv)
        )
    return x
