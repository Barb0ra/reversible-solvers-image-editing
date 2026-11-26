import argparse

import torch

from pipeline import SMOOTH, decode_latent, encode_image, encode_prompt, load_image, load_pipeline
from samplers import SOLVERS


@torch.no_grad()
def edit_latent(
    pipe,
    latent,
    source_prompt,
    target_prompt,
    solver="ees25",
    steps=48,
    guidance_scale=3.0,
    freeze_step=0,
    negative_prompt="",
    guidance_type="classifier-free",
    seed=1,
    **options,
):
    """Invert `latent` under the source prompt and regenerate it under the target prompt.

    The first `freeze_step` steps of generation are skipped: inversion stops that many steps
    early and generation starts from there. Returns the edited latent and the inverted state.
    """
    solver = SOLVERS[solver]
    kwargs = dict(freeze_step=freeze_step, guidance_type=guidance_type, **options)

    torch.manual_seed(seed)
    source_embeds = encode_prompt(pipe, source_prompt, negative_prompt)
    state = solver.invert(pipe, latent, source_embeds, steps, guidance_scale, **kwargs)

    torch.manual_seed(seed)
    target_embeds = encode_prompt(pipe, target_prompt, negative_prompt)
    edited = solver.sample(pipe, state, target_embeds, steps, guidance_scale, **kwargs)
    return edited, state


@torch.no_grad()
def edit_image(pipe, image_path, source_prompt, target_prompt, seed=1, **kwargs):
    torch.manual_seed(seed)
    latent = encode_image(pipe, load_image(image_path))
    edited, _ = edit_latent(pipe, latent, source_prompt, target_prompt, seed=seed, **kwargs)
    return decode_latent(pipe, edited)


def add_solver_arguments(parser):
    # shared by edit.py, benchmark.py and reconstruction.py
    parser.add_argument("--solver", default="ees25", choices=sorted(SOLVERS))
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--nfe", type=int, default=48, help="network evaluations per pass")
    budget.add_argument("--steps", type=int)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument(
        "--model",
        default=SMOOTH,
        help="shi-labs/smooth-diffusion-lora (default) or stable-diffusion-v1-5/stable-diffusion-v1-5",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--edict_p", type=float, help="EDICT mixing weight (default 0.93)")
    parser.add_argument("--bdia_gamma", type=float, help="BDIA gamma (default 0.96)")
    parser.add_argument(
        "--coupling", type=float, help="Rex and McCallum-Foster coupling (default 0.999)"
    )
    parser.add_argument(
        "--prediction", choices=["noise", "data"], help="EES and reversible Heun (default data)"
    )
    parser.add_argument(
        "--formulation",
        choices=["black_box", "semilinear", "original_ode_black_box", "sigma_bar_semilinear"],
        help="EES and reversible Heun (default semilinear for EES, black_box for reversible Heun)",
    )
    parser.add_argument(
        "--schedule",
        choices=["logSNR", "time_uniform", "sigma_bar"],
        help="EES and reversible Heun step spacing (default logSNR for EES, time_uniform for "
        "reversible Heun)",
    )


def num_steps(args):
    return args.steps or max(1, args.nfe // SOLVERS[args.solver].NFE_PER_STEP)


def solver_options(args):
    # only the options given, so that each solver keeps its own defaults
    options = {
        "p": args.edict_p,
        "gamma": args.bdia_gamma,
        "coupling": args.coupling,
        "prediction": args.prediction,
        "formulation": args.formulation,
        "schedule": args.schedule,
    }
    return {k: v for k, v in options.items() if v is not None}


def main():
    parser = argparse.ArgumentParser(description="Edit an image by inversion and regeneration.")
    parser.add_argument("image")
    parser.add_argument("source_prompt")
    parser.add_argument("target_prompt")
    parser.add_argument("--output", default="edited.png")
    add_solver_arguments(parser)
    parser.add_argument("--freeze_step", type=int, help="default: steps // 5")
    parser.add_argument(
        "--guidance_type", default="classifier-free", choices=["classifier-free", "proxedit"]
    )
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument(
        "--npi", action="store_true", help="negative-prompt inversion: source prompt as negative"
    )
    args = parser.parse_args()
    if args.guidance_type == "proxedit" and not args.npi:
        parser.error("--guidance_type proxedit needs --npi")

    steps = num_steps(args)
    freeze_step = steps // 5 if args.freeze_step is None else args.freeze_step

    pipe = load_pipeline(args.model, args.device)
    image = edit_image(
        pipe,
        args.image,
        args.source_prompt,
        args.target_prompt,
        solver=args.solver,
        steps=steps,
        guidance_scale=args.guidance_scale,
        freeze_step=freeze_step,
        negative_prompt=args.source_prompt if args.npi else args.negative_prompt,
        guidance_type=args.guidance_type,
        seed=args.seed,
        **solver_options(args),
    )
    image.save(args.output)


if __name__ == "__main__":
    main()
