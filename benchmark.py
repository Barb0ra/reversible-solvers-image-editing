import argparse
import os
import random

import numpy as np
import torch

from edit import add_solver_arguments, edit_image, num_steps, solver_options
from pie_bench import edits
from pipeline import SMOOTH, load_pipeline


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def default_output(args, steps):
    parts = [args.solver, f"steps{steps}", f"g{args.guidance_scale}"]
    parts += [f"{k}-{v}" for k, v in solver_options(args).items()]
    if args.model != SMOOTH:
        parts.append("sd15")
    if args.npi:
        parts.append("npi")
    if args.guidance_type == "proxedit":
        parts.append("proxedit")
    return os.path.join("results", f"{args.task}_edits", "_".join(parts))


def main():
    parser = argparse.ArgumentParser(description="Edit the PIE-Bench images with one solver.")
    parser.add_argument(
        "--task",
        default="small",
        choices=["small", "large"],
        help="small: the benchmark's prompt pairs, the first steps // 5 steps frozen; large: the "
        "next image's source prompt as target, no frozen steps",
    )
    parser.add_argument("--data", default="data/PIE-Bench_v1")
    parser.add_argument(
        "--categories",
        nargs="+",
        help="editing categories 0-9 (default: all for small edits, 0 for large edits)",
    )
    parser.add_argument("--output", help="default: results/<task>_edits/<configuration>")
    add_solver_arguments(parser)
    parser.add_argument(
        "--guidance_type",
        default="classifier-free",
        choices=["classifier-free", "proxedit"],
        help="proxedit (proximal guidance) also turns on --npi",
    )
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument(
        "--npi", action="store_true", help="negative-prompt inversion: source prompt as negative"
    )
    parser.add_argument("--limit", type=int, help="only the first LIMIT edits")
    parser.add_argument("--overwrite", action="store_true", help="redo existing images")
    args = parser.parse_args()
    args.npi = args.npi or args.guidance_type == "proxedit"

    steps = num_steps(args)
    freeze_step = steps // 5 if args.task == "small" else 0
    output = args.output or default_output(args, steps)
    todo = edits(args.data, args.task, args.categories)[: args.limit]

    pipe = load_pipeline(args.model, args.device)
    for edit in todo:
        path = os.path.join(output, "annotation_images", edit["image"])
        if os.path.exists(path) and not args.overwrite:
            continue
        setup_seed(args.seed)
        image = edit_image(
            pipe,
            os.path.join(args.data, "annotation_images", edit["image"]),
            edit["source"],
            edit["target"],
            solver=args.solver,
            steps=steps,
            guidance_scale=args.guidance_scale,
            freeze_step=freeze_step,
            negative_prompt=edit["source"] if args.npi else args.negative_prompt,
            guidance_type=args.guidance_type,
            seed=args.seed,
            **solver_options(args),
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image.save(path)
        print(path)


if __name__ == "__main__":
    main()
