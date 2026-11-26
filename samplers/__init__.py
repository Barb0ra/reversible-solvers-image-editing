from functools import partial
from types import SimpleNamespace

from samplers import bdia, belm, ddim, edict, ees, mccallum_foster, reversible_heun, rex

# Each solver defines
#   invert(pipe, x, prompt_embeds, steps, guidance_scale, freeze_step, guidance_type, ...)
#     returning the inverted state,
#   sample(pipe, state, prompt_embeds, steps, guidance_scale, freeze_step, guidance_type, ...)
#     returning the generated latent, and
#   NFE_PER_STEP, the number of network evaluations per step.
# Solver-specific options (gamma for BDIA, p for EDICT, coupling for Rex and McCallum-Foster,
# prediction, formulation and schedule for EES and reversible Heun) are keyword arguments of both
# functions.


def variant(module, nfe_per_step, **options):
    # a solver module with some of its options fixed
    return SimpleNamespace(
        invert=partial(module.invert, **options),
        sample=partial(module.sample, **options),
        NFE_PER_STEP=nfe_per_step,
    )


SOLVERS = {
    "ddim": ddim,
    "edict": edict,
    "bdia": bdia,
    "belm": belm,
    "rex_euler": variant(rex, 2, base="euler"),
    "rex_midpoint": variant(rex, 4, base="midpoint"),
    "mccallum_foster_euler": variant(mccallum_foster, 2, base="euler"),
    "mccallum_foster_midpoint": variant(mccallum_foster, 4, base="midpoint"),
    "reversible_heun": reversible_heun,
    "ees25": variant(ees, 3, a=ees.A25, b=ees.B25),
    "ees27": variant(ees, 4, a=ees.A27, b=ees.B27),
}
