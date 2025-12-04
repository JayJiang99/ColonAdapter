from __future__ import absolute_import, division, print_function

"""
Simple folder-based inference script for the DUSt3R depth model.

This script:
- Loads the DUSt3R-based depth model exactly as in `evaluate_depth_colonaf.py`
- Takes a directory of images as input
- Forms consecutive image pairs (i, i+1)
- Runs the model to obtain per-pixel depths for the first image in each pair
- Optionally saves predicted depth maps and disparities as PNG/NumPy files

Example:
    python infer_depth_folder.py \\
        --image_dir /path/to/images \\
        --load_weights_folder /path/to/weights_XX \\
        --height 224 --width 224 \\
        --save_dir /path/to/save_preds
"""

import argparse
import math
import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from layers import disp_to_depth
from options import MonodepthOptions
from dust3r.model import AsymmetricCroCo3DStereo


cv2.setNumThreads(0)


def _load_image_paths(image_dir: str) -> List[str]:
    """Collect and sort image paths from a directory."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    image_paths: List[str] = []
    for name in sorted(os.listdir(image_dir)):
        _, ext = os.path.splitext(name)
        if ext.lower() in exts:
            image_paths.append(os.path.join(image_dir, name))
    if len(image_paths) < 2:
        raise ValueError(f"Need at least 2 images in {image_dir} to form pairs")
    return image_paths


def _build_pairs(image_paths: List[str]) -> List[Tuple[str, str]]:
    """Build consecutive image pairs (i, i+1)."""
    pairs: List[Tuple[str, str]] = []
    for idx in range(len(image_paths) - 1):
        pairs.append((image_paths[idx], image_paths[idx + 1]))
    return pairs


def _get_transform(height: int, width: int) -> transforms.Compose:
    """Return the same normalization as in `MonoDataset.to_norm`."""
    return transforms.Compose(
        [
            transforms.Resize((height, width), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def _load_and_preprocess_image(
    path: str,
    transform: transforms.Compose,
) -> torch.Tensor:
    """Load a single RGB image and apply resize + normalization."""
    with open(path, "rb") as file_obj:
        with Image.open(file_obj) as img:
            img = img.convert("RGB")
    tensor = transform(img)
    # Add batch dimension
    return tensor.unsqueeze(0)


def _init_model_from_options(opt: argparse.Namespace) -> AsymmetricCroCo3DStereo:
    """Instantiate DUSt3R model and load weights as in `evaluate_depth_colonaf.py`."""
    load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    if not os.path.isdir(load_weights_folder):
        raise FileNotFoundError(f"Cannot find weights folder: {load_weights_folder}")

    depth_model_path = os.path.join(load_weights_folder, "depth_model.pth")
    if not os.path.isfile(depth_model_path):
        raise FileNotFoundError(f"Cannot find depth model: {depth_model_path}")

    print(f"-> Loading DUSt3R weights from {depth_model_path}")
    state_dict = torch.load(depth_model_path, map_location="cpu")

    model = AsymmetricCroCo3DStereo(
        pos_embed="RoPE100",
        patch_embed_cls="PatchEmbedDust3R",
        img_size=(opt.height, opt.width),
        head_type="dpt",
        output_mode="pts3d",
        depth_mode=("exp", -math.inf, math.inf),
        conf_mode=("exp", 1, math.inf),
        enc_embed_dim=1024,
        enc_depth=24,
        enc_num_heads=16,
        dec_embed_dim=768,
        dec_depth=12,
        dec_num_heads=12,
        freeze="encoder",
        lora_rank=opt.lora_rank,
        lora_alpha=opt.lora_alpha,
        lora_dropout=opt.lora_dropout,
    )

    print(model.load_state_dict(state_dict, strict=True))
    del state_dict

    model.cuda()
    model.eval()
    return model


def run_inference_folder(
    image_dir: str,
    opt: argparse.Namespace,
    save_dir: str,
) -> None:
    """Run depth inference on a folder of images."""
    device = torch.device("cuda" if torch.cuda.is_available() and not opt.no_cuda else "cpu")

    image_paths = _load_image_paths(image_dir)
    pairs = _build_pairs(image_paths)
    print(f"-> Found {len(image_paths)} images, {len(pairs)} pairs")

    os.makedirs(save_dir, exist_ok=True)

    transform = _get_transform(opt.height, opt.width)
    model = _init_model_from_options(opt).to(device)

    all_depth_stats = []

    with torch.no_grad():
        for idx, (path1, path2) in enumerate(pairs):
            img1 = _load_and_preprocess_image(path1, transform).to(device)
            img2 = _load_and_preprocess_image(path2, transform).to(device)

            # Match the structure used in `evaluate_depth_colonaf.py`
            frame_id = torch.tensor([idx], dtype=torch.int64, device=device)
            instance0 = [[int(idx), int(idx) + 1]]
            instance1 = [[int(idx) + 1, int(idx)]]

            view1 = {
                "img": img1,
                "instance": instance0,
            }
            view2 = {
                "img": img2,
                "instance": instance1,
            }

            pred1, pred2 = model(view1, view2, return_pose=False)
            pts3d_1 = pred1["pts3d"].clone()  # B,H,W,3
            depth_1 = pts3d_1[..., 2]  # B,H,W

            # Convert to disparity-like map and depth in meters (same as evaluate_depth_colonaf.py)
            disp = 1.0 / depth_1
            disp, _ = disp_to_depth(disp, opt.min_depth, opt.max_depth)

            depth_np = depth_1.squeeze(0).cpu().numpy()
            disp_np = disp.squeeze(0).cpu().numpy()

            # Basic stats to monitor predictions
            min_d, max_d = float(depth_np.min()), float(depth_np.max())
            all_depth_stats.append((min_d, max_d))
            print(
                f"[{idx+1}/{len(pairs)}] "
                f"{os.path.basename(path1)} vs {os.path.basename(path2)} "
                f"depth range: ({min_d:.4f}, {max_d:.4f})"
            )

            # Save visualizations
            base_name1 = os.path.splitext(os.path.basename(path1))[0]
            depth_vis = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
            depth_vis = (depth_vis * 255.0).astype(np.uint8)
            depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

            cv2.imwrite(os.path.join(save_dir, f"{base_name1}_depth.png"), depth_vis_color)
            np.save(os.path.join(save_dir, f"{base_name1}_depth.npy"), depth_np)
            np.save(os.path.join(save_dir, f"{base_name1}_disp.npy"), disp_np)

    if all_depth_stats:
        mins, maxs = zip(*all_depth_stats)
        print(
            "Overall depth range across all pairs: "
            f"min={min(mins):.4f}, max={max(maxs):.4f}"
        )


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI for folder inference, reusing MonodepthOptions defaults when possible."""
    mono_opts = MonodepthOptions()
    base_parser = mono_opts.parser

    base_parser.description = "Folder-based depth inference with DUSt3R"

    base_parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing input images",
    )
    base_parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save predicted depth/disp outputs",
    )

    # We do not require eval_mono/eval_stereo for simple inference, but keep flags for compatibility.
    return base_parser


def main() -> None:
    parser = build_argparser()
    opt = parser.parse_args()

    run_inference_folder(
        image_dir=opt.image_dir,
        opt=opt,
        save_dir=opt.save_dir,
    )


if __name__ == "__main__":
    main()


