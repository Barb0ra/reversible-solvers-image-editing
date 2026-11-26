# PIE-Bench (Ju et al., 2024): the two editing tasks built from it and the result tables of the
# evaluation scripts.
import csv
import json
import os

import numpy as np

CATEGORIES = [str(i) for i in range(10)]


def clean(prompt):
    # the prompts mark the edited words with brackets
    return prompt.replace("[", "").replace("]", "")


def load_items(data, categories):
    with open(os.path.join(data, "mapping_file.json")) as f:
        mapping = json.load(f)
    return [item for item in mapping.values() if item["editing_type_id"] in categories]


def edits(data, task, categories=None):
    """The edits of a task as dicts with the image path (relative to annotation_images), the
    source and target prompts and the run-length encoded edit mask.

    Small edits are the benchmark's own prompt pairs, over all categories by default. Large edits
    sort the images by path and take the source prompt of the next image as the target, so the
    last image has no edit; by default they use category 0 only.
    """
    if task == "small":
        items = load_items(data, categories or CATEGORIES)
        targets = [clean(item["editing_prompt"]) for item in items]
    else:
        items = sorted(load_items(data, categories or ["0"]), key=lambda item: item["image_path"])
        targets = [clean(item["original_prompt"]) for item in items[1:]]
    return [
        dict(
            image=item["image_path"],
            source=clean(item["original_prompt"]),
            target=target,
            mask=item["mask"],
        )
        for item, target in zip(items, targets)
    ]


def append_rows(path, header, rows):
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(header)
        writer.writerows(rows)


def write_results(folder, metrics, images, values, num_missing, summary_path, per_image_path):
    # values[m] has one entry per image in images, nan where the metric is undefined
    means = [float(np.nanmean(np.array(values[m], dtype=np.float64))) for m in metrics]
    stds = [float(np.nanstd(np.array(values[m], dtype=np.float64))) for m in metrics]
    header = ["folder", "num_images", "num_missing"]
    header += [f"{m}_mean" for m in metrics] + [f"{m}_std" for m in metrics]
    append_rows(summary_path, header, [[folder, len(images), num_missing] + means + stds])
    for metric, mean in zip(metrics, means):
        print(f"{metric}: {mean}")

    if per_image_path:
        rows = [[folder, im] + [values[m][i] for m in metrics] for i, im in enumerate(images)]
        append_rows(per_image_path, ["folder", "image"] + metrics, rows)
