import argparse
import csv
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch

from edit import add_solver_arguments, edit_latent, num_steps, solver_options
from pipeline import decode_latent, encode_image, load_image, load_pipeline


def coco_images(directory, captions_file, num_images):
    # the first num_images images in sorted order, each with its first caption
    with open(captions_file) as f:
        annotations = json.load(f)["annotations"]
    captions = defaultdict(list)
    for annotation in annotations:
        captions[annotation["image_id"]].append(annotation["caption"])

    paths = []
    for root, _, files in os.walk(directory):
        paths += [os.path.join(root, file) for file in files if file.endswith(".jpg")]
    paths = sorted(paths)[:num_images]
    return [(path, captions[int(re.search(r"(\d+)\.jpg$", path).group(1))][0]) for path in paths]


@torch.no_grad()
def reconstruct(pipe, path, caption, args):
    image = load_image(path)
    torch.manual_seed(args.seed)
    latent = encode_image(pipe, image)
    recon, _ = edit_latent(
        pipe,
        latent,
        caption,
        caption,
        solver=args.solver,
        steps=num_steps(args),
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        **solver_options(args),
    )
    recon_image = decode_latent(pipe, recon)

    original = np.array(image)
    latent_mse = np.mean((latent.cpu().numpy() - recon.cpu().numpy()) ** 2)
    # pixels in [0, 1]
    pixel_mse = np.mean((original / 255.0 - np.array(recon_image) / 255.0) ** 2)
    return latent_mse, pixel_mse, recon_image


def main():
    parser = argparse.ArgumentParser(
        description="Invert COCO images under their caption and regenerate them under the same "
        "caption; report the mean squared reconstruction error."
    )
    add_solver_arguments(parser)
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--data", default="data/coco2014")
    parser.add_argument(
        "--captions", default="data/coco2014/annotations/captions_val2014.json"
    )
    parser.add_argument("--output", default="results/reconstruction.csv", help="appended to")
    parser.add_argument("--image_dir", help="save the reconstructions here")
    args = parser.parse_args()

    pipe = load_pipeline(args.model, args.device)
    latent_mses, pixel_mses = [], []
    for path, caption in coco_images(args.data, args.captions, args.num_images):
        latent_mse, pixel_mse, recon_image = reconstruct(pipe, path, caption, args)
        print(os.path.basename(path), latent_mse, pixel_mse)
        latent_mses.append(latent_mse)
        pixel_mses.append(pixel_mse)
        if args.image_dir:
            os.makedirs(args.image_dir, exist_ok=True)
            recon_image.save(os.path.join(args.image_dir, os.path.basename(path)[:-4] + ".png"))

    options = " ".join(f"{k}={v}" for k, v in solver_options(args).items())
    row = [args.solver, options, args.model, num_steps(args), args.guidance_scale, len(pixel_mses)]
    row += [sum(latent_mses) / len(latent_mses), np.std(latent_mses)]
    row += [sum(pixel_mses) / len(pixel_mses), np.std(pixel_mses)]
    header = ["solver", "options", "model", "steps", "guidance_scale", "num_images"]
    header += ["latent_mse_mean", "latent_mse_std", "pixel_mse_mean", "pixel_mse_std"]
    new = not os.path.exists(args.output)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(header)
        writer.writerow(row)
    print(", ".join(f"{name}={value}" for name, value in zip(header, row)))


if __name__ == "__main__":
    main()
