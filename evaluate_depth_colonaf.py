from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from main_utils import readlines
from options import MonodepthOptions
import datasets
import networks
import math

from dust3r.model import AsymmetricCroCo3DStereo
cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 150

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        depther_path = os.path.join(opt.load_weights_folder, "depth_model.pth")
        depther_dict = torch.load(depther_path)

        

        dataset = datasets.SyntheticColonDataset(opt.data_path, filenames,
                                                opt.height, opt.width,
                                                opt.frame_ids, 4, is_train=False)
        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)
        # Initialize model
        depther = AsymmetricCroCo3DStereo(
            pos_embed='RoPE100',
            patch_embed_cls='PatchEmbedDust3R',
            img_size=(opt.height, opt.width),
            head_type='dpt',
            output_mode='pts3d',
            depth_mode=('exp', -math.inf, math.inf),
            conf_mode=('exp', 1, math.inf),
            enc_embed_dim=1024,
            enc_depth=24,
            enc_num_heads=16,
            dec_embed_dim=768,
            dec_depth=12,
            dec_num_heads=12,
            freeze='encoder',
            lora_rank=opt.lora_rank,
            lora_alpha=opt.lora_alpha,
            lora_dropout=opt.lora_dropout

        )
        
        print('Loading pretrained: ', depther_path)
        print(depther.load_state_dict(depther_dict, strict=True))
        del depther_dict  # in case it occupies memory
        
    
        depther.cuda()
        depther.eval()


        

        

        pred_disps = []
        pred_depths = []

        

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color_norm", 0, 0)].cuda()
                input_color2 = data[("color_norm", 1, 0)].cuda()
                # print key of data
                
                # input_color = data[("color", 0, 0)]

                
                if ("instance", 0, 0) not in data:
                    frame_ids = data["frame_id"]
                    frame_ids = frame_ids.tolist() if isinstance(frame_ids, torch.Tensor) else frame_ids
                    data[("instance", 0, 0)] = [[fid, fid + 1] for fid in frame_ids]
                    data[("instance", 1, 0)] = [instance[::-1] for instance in data[("instance", 0, 0)]]

                # Get both views from the input
                view1 = {
                    'img': input_color,
                    'instance': data[("instance", 0, 0)]
                }
                view2 = {
                    'img': input_color2,
                    'instance': data[("instance", 1, 0)]
                }
        
                # Forward pass through Dust3R with both views, and get the transformation from view1 to view2
                pred1, pred2= depther(view1, view2, return_pose=False)
                pr_pts1 = pred1['pts3d'].clone()  # B,H,W,3
                pr_pts2 = pred2['pts3d_in_other_view'].clone()  # B,H,W,3
                pr_pts1_depth = pr_pts1[..., 2]
                # Convert outputs to depth maps
                output = {
                    ("disp", 0): 1.0/pr_pts1_depth,
                }
                
                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
     
                print(f"pred_disp shape: {pred_disp.shape}")
                # pred_disp = pred_disp.cpu()[:, 0].numpy()
                pred_disp = pred_disp.cpu().numpy()
                # pred_disp = pred_disp.numpy()
                print(f"min pred_disp: {np.min(pred_disp)}, max pred_disp: {np.max(pred_disp)}")
                

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)
                pred_depths.append(pr_pts1_depth.cpu().numpy())
        pred_disps = np.concatenate(pred_disps)
        pred_depths = np.concatenate(pred_depths)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]
        pred_disp = pred_disps[i]
        # print pred_disp shape
        
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        # pred_depth = 1/pred_disp
        pred_depth = pred_depths[i]
        pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))
        gt_depth_save = (gt_depth - np.min(gt_depth)) / (np.max(gt_depth) - np.min(gt_depth)) * 255
        pred_depth_save = (pred_depth - np.min(pred_depth)) / (np.max(pred_depth) - np.min(pred_depth)) * 255
        save_dir = opt.load_weights_folder + "/pred_depths"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(f"{save_dir}/pred_depth_{i}.png", pred_depth_save.astype(np.uint8))

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        
        print(f"min gt_depth: {np.min(gt_depth)}, max gt_depth: {np.max(gt_depth)}")
        print(f"min pred_depth: {np.min(pred_depth)}, max pred_depth: {np.max(pred_depth)}")
        

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        # print(compute_errors(gt_depth, pred_depth))

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())