# PickScore (Kirstain et al., 2023) and ImageReward (Xu et al., 2023) of the edited images for
# their target prompts. Needs the packages in requirements-rewards.txt.
import argparse
import os

import ImageReward
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from pie_bench import edits, write_results

METRICS = ["pickscore_target_image", "imagereward_target_image"]


class RewardModels:
    def __init__(self, device, imagereward_root=None):
        self.device = device
        self.pick_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.pick_model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(device)
        self.imagereward_model = ImageReward.load(
            "ImageReward-v1.0", device=device, download_root=imagereward_root
        )

    @torch.no_grad()
    def pickscore(self, image, prompt):
        image_inputs = self.pick_processor(
            images=image, padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(self.device)
        text_inputs = self.pick_processor(
            text=prompt, padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(self.device)
        image_embs = self.pick_model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        text_embs = self.pick_model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        return (self.pick_model.logit_scale.exp() * (text_embs @ image_embs.T)[0]).item()

    @torch.no_grad()
    def imagereward(self, image, prompt):
        return float(self.imagereward_model.score(prompt, image))


def main():
    parser = argparse.ArgumentParser(description="Reward metrics of edited PIE-Bench images.")
    parser.add_argument("folder", help="folder with the edited images under annotation_images/")
    parser.add_argument("--task", default="small", choices=["small", "large"])
    parser.add_argument("--data", default="data/PIE-Bench_v1")
    parser.add_argument(
        "--categories",
        nargs="+",
        help="editing categories 0-9 (default: all for small edits, 0 for large edits)",
    )
    parser.add_argument(
        "--summary",
        help="CSV to append to (default: results/evaluation_rewards_<task>_edits.csv)",
    )
    parser.add_argument("--per_image", help="CSV to append one row per image to")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--imagereward_root", help="ImageReward checkpoint folder")
    args = parser.parse_args()
    summary = args.summary or f"results/evaluation_rewards_{args.task}_edits.csv"

    models = RewardModels(args.device, args.imagereward_root)
    images, values, num_missing = [], {m: [] for m in METRICS}, 0
    for edit in edits(args.data, args.task, args.categories):
        path = os.path.join(args.folder, "annotation_images", edit["image"])
        if not os.path.exists(path):
            num_missing += 1
            continue
        image = Image.open(path).convert("RGB")
        images.append(edit["image"])
        values["pickscore_target_image"].append(models.pickscore(image, edit["target"]))
        values["imagereward_target_image"].append(models.imagereward(image, edit["target"]))

    write_results(args.folder, METRICS, images, values, num_missing, summary, args.per_image)


if __name__ == "__main__":
    main()
