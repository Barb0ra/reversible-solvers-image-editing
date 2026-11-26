# Explicit and effectively symmetric Runge-Kutta methods EES(2,5) and EES(2,7) from Shmelev et al.
# (2025), "Explicit and Effectively Symmetric Runge-Kutta Methods". They are symmetric up to high
# order, so a step backwards nearly undoes a step forwards: inversion integrates the ODE towards
# t = 1 with the same method.
from samplers.ode import setup

# Butcher tableaux: the rows of the strictly lower triangular a, and the weights b.
# The nodes are c_i = sum(a_i).

# EES(2,5)
A25 = (
    (),
    (1.0 / 3,),
    (-5.0 / 48, 15.0 / 16),
)
B25 = (0.1, 0.5, 0.4)

# EES(2,7)
A27 = (
    (),
    (0.19526214587563498373277042526343397381010937487435064227444008733642250717,),
    (
        -0.1077411015677877062999296364912625800595970051926271636176383442503861467,
        0.67677669529663688110021109052621225982120898442211850914708496724884155980,
    ),
    (
        0.1729819437148737028347908631571798892416810829146278012536819167181,
        0.0497642243643495404995476631581165860974093190954603375419607844667,
        0.6796227589829592647992762610529865377558549105527365400671372551468,
    ),
)
B27 = (
    0.0540970937771939181139238448122075545922131695620825557478543418591,
    0.3153009687409353606286920517292641484692622768126391480840485527136,
    0.3918058124456121637721523103755848908155736608758348885042913162817,
    0.2387961250362585574852317930829434061229508927494434076638057891455,
)


def weighted_sum(weights, k):
    total = weights[0] * k[0]
    for w, k_j in zip(weights[1:], k[1:]):
        total = total + w * k_j
    return total


def rk_step(y, s, s_next, f, a, b):
    h = s_next - s
    k = [f(y, s)]
    for a_i in a[1:]:
        k.append(f(y + h * weighted_sum(a_i, k), s + h * sum(a_i)))
    return y + h * weighted_sum(b, k)


def solve(
    pipe, x, prompt_embeds, steps, guidance_scale, freeze_step, guidance_type,
    prediction, formulation, schedule, a, b, inverting,
):
    f, grid, y, to_x = setup(
        pipe, x, prompt_embeds, steps, guidance_scale, freeze_step, guidance_type,
        prediction, formulation, schedule, inverting,
    )
    for i in range(len(grid) - 1):
        y = rk_step(y, grid[i], grid[i + 1], f, a, b)
    return to_x(y)


def invert(
    pipe, x, prompt_embeds, steps, guidance_scale, freeze_step=0,
    guidance_type="classifier-free", prediction="data", formulation="semilinear",
    schedule="logSNR", a=A25, b=B25,
):
    return solve(
        pipe, x, prompt_embeds, steps, guidance_scale, freeze_step, guidance_type,
        prediction, formulation, schedule, a, b, inverting=True,
    )


def sample(
    pipe, x, prompt_embeds, steps, guidance_scale, freeze_step=0,
    guidance_type="classifier-free", prediction="data", formulation="semilinear",
    schedule="logSNR", a=A25, b=B25,
):
    return solve(
        pipe, x, prompt_embeds, steps, guidance_scale, freeze_step, guidance_type,
        prediction, formulation, schedule, a, b, inverting=False,
    )
