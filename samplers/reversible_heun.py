# Reversible Heun (Kidger et al., 2021). It carries a second state y_hat and the last slope k, and
# generation undoes the inversion pass step by step.
from samplers.ode import setup

NFE_PER_STEP = 1


def heun_step(y, y_hat, k, s, s_next, f):
    h = s_next - s
    y_hat_next = 2.0 * y - y_hat + h * k
    k_next = f(y_hat_next, s_next)
    y_next = y + 0.5 * h * (k + k_next)
    return y_next, y_hat_next, k_next


def heun_step_back(y_next, y_hat_next, k_next, s, s_next, f):
    # exact inverse of heun_step
    h = s_next - s
    y_hat = 2.0 * y_next - y_hat_next - h * k_next
    k = f(y_hat, s)
    y = y_next - 0.5 * h * (k + k_next)
    return y, y_hat, k


def invert(
    pipe, x, prompt_embeds, steps, guidance_scale, freeze_step=0,
    guidance_type="classifier-free", prediction="data", formulation="black_box",
    schedule="time_uniform",
):
    f, grid, y, to_x = setup(
        pipe, x, prompt_embeds, steps, guidance_scale, freeze_step, guidance_type,
        prediction, formulation, schedule, inverting=True,
    )
    y_hat = y
    k = f(y_hat, grid[0])
    for i in range(len(grid) - 1):
        y, y_hat, k = heun_step(y, y_hat, k, grid[i], grid[i + 1], f)
    return to_x(y), y_hat, k


def sample(
    pipe, state, prompt_embeds, steps, guidance_scale, freeze_step=0,
    guidance_type="classifier-free", prediction="data", formulation="black_box",
    schedule="time_uniform",
):
    # y_hat and k stay in the variable the solver works in
    x, y_hat, k = state
    f, grid, y, to_x = setup(
        pipe, x, prompt_embeds, steps, guidance_scale, freeze_step, guidance_type,
        prediction, formulation, schedule, inverting=False,
    )
    for i in range(len(grid) - 1):
        y, y_hat, k = heun_step_back(y, y_hat, k, grid[i + 1], grid[i], f)
    return to_x(y)
