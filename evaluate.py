# The metric definitions are from PnPInversion (github.com/cure-lab/PnPInversion),
# evaluation/evaluate.py.
import argparse
import os

import numpy as np
from PIL import Image

from metrics_calculator import MetricsCalculator
from pie_bench import edits, write_results

METRICS = {
    "small": [
        "structure_distance",
        "psnr_unedit_part",
        "lpips_unedit_part",
        "mse_unedit_part",
        "ssim_unedit_part",
        "clip_similarity_source_image",
        "clip_similarity_target_image",
        "clip_similarity_target_image_edit_part",
    ],
    # large edits change the whole image: CLIP similarity to the target prompt and LPIPS to the
    # source image
    "large": ["clip_similarity_target_image", "lpips"],
}


def mask_decode(encoded_mask, image_shape=[512, 512]):
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,))

    for i in range(0, len(encoded_mask), 2):
        splice_len = min(encoded_mask[i + 1], length - encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i] + j] = 1

    mask_array = mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0, :] = 1
    mask_array[-1, :] = 1
    mask_array[:, 0] = 1
    mask_array[:, -1] = 1

    return mask_array


def calculate_metric(
    metrics_calculator, metric, src_image, tgt_image, src_mask, tgt_mask, src_prompt, tgt_prompt
):
    if metric == "psnr":
        return metrics_calculator.calculate_psnr(src_image, tgt_image, None, None)
    if metric == "lpips":
        return metrics_calculator.calculate_lpips(src_image, tgt_image, None, None)
    if metric == "mse":
        return metrics_calculator.calculate_mse(src_image, tgt_image, None, None)
    if metric == "ssim":
        return metrics_calculator.calculate_ssim(src_image, tgt_image, None, None)
    if metric == "structure_distance":
        return metrics_calculator.calculate_structure_distance(src_image, tgt_image, None, None)
    if metric == "psnr_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_psnr(
                src_image, tgt_image, 1 - src_mask, 1 - tgt_mask
            )
    if metric == "lpips_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_lpips(
                src_image, tgt_image, 1 - src_mask, 1 - tgt_mask
            )
    if metric == "mse_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_mse(
                src_image, tgt_image, 1 - src_mask, 1 - tgt_mask
            )
    if metric == "ssim_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_ssim(
                src_image, tgt_image, 1 - src_mask, 1 - tgt_mask
            )
    if metric == "structure_distance_unedit_part":
        if (1 - src_mask).sum() == 0 or (1 - tgt_mask).sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_structure_distance(
                src_image, tgt_image, 1 - src_mask, 1 - tgt_mask
            )
    if metric == "psnr_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_psnr(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "lpips_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_lpips(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "mse_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_mse(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "ssim_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_ssim(src_image, tgt_image, src_mask, tgt_mask)
    if metric == "structure_distance_edit_part":
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_structure_distance(
                src_image, tgt_image, src_mask, tgt_mask
            )
    if metric == "clip_similarity_source_image":
        return metrics_calculator.calculate_clip_similarity(src_image, src_prompt, None)
    if metric == "clip_similarity_target_image":
        return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt, None)
    if metric == "clip_similarity_target_image_edit_part":
        if tgt_mask.sum() == 0:
            return "nan"
        else:
            return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt, tgt_mask)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a folder of edited PIE-Bench images.")
    parser.add_argument("folder", help="folder with the edited images under annotation_images/")
    parser.add_argument("--task", default="small", choices=["small", "large"])
    parser.add_argument("--data", default="data/PIE-Bench_v1")
    parser.add_argument(
        "--categories",
        nargs="+",
        help="editing categories 0-9 (default: all for small edits, 0 for large edits)",
    )
    parser.add_argument("--metrics", nargs="+", help="default: the metrics of the task")
    parser.add_argument(
        "--summary", help="CSV to append to (default: results/evaluation_<task>_edits.csv)"
    )
    parser.add_argument("--per_image", help="CSV to append one row per image to")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    summary = args.summary or f"results/evaluation_{args.task}_edits.csv"

    metrics = args.metrics or METRICS[args.task]
    metrics_calculator = MetricsCalculator(args.device, metrics)

    images, values, num_missing = [], {m: [] for m in metrics}, 0
    for edit in edits(args.data, args.task, args.categories):
        tgt_path = os.path.join(args.folder, "annotation_images", edit["image"])
        if not os.path.exists(tgt_path):
            num_missing += 1
            continue
        src_image = Image.open(os.path.join(args.data, "annotation_images", edit["image"]))
        tgt_image = Image.open(tgt_path)
        mask = mask_decode(edit["mask"])[:, :, np.newaxis].repeat(3, axis=2)

        images.append(edit["image"])
        for metric in metrics:
            value = calculate_metric(
                metrics_calculator,
                metric,
                src_image,
                tgt_image,
                mask,
                mask,
                edit["source"],
                edit["target"],
            )
            values[metric].append(float(value))

    write_results(args.folder, metrics, images, values, num_missing, summary, args.per_image)


if __name__ == "__main__":
    main()
