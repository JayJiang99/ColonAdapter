from __future__ import absolute_import, division, print_function

import os
import time
import json
import logging
import math

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import datasets
import networks
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.geometry import normalize_pointcloud

from layers import *
from laplacian import LapLoss, LapLossConf
from main_utils import readlines, normalize_image, sec_to_hm_str

logger = logging.getLogger(__name__)


class Trainer:
    """End-to-end trainer for ColonAdapter combining DUSt3R with optical flow."""

    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []
        self.parameters_to_train_0 = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        # ---- Encoder ----
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.lap = LapLoss()
        self.lap_conf = LapLossConf()

        # ---- DUSt3R depth model with LoRA ----
        self.models["depth_model"] = AsymmetricCroCo3DStereo(
            pos_embed='RoPE100',
            patch_embed_cls='ManyAR_PatchEmbed',
            img_size=(self.opt.height, self.opt.width),
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
            lora_rank=self.opt.lora_rank,
            lora_alpha=self.opt.lora_alpha,
            lora_dropout=self.opt.lora_dropout
        )
        self.models["depth_model"].to(self.device)
        if self.opt.pretrained_path is not None:
            logger.info("Loading pretrained: %s", self.opt.pretrained_path)
            ckpt = torch.load(self.opt.pretrained_path, map_location=self.device, weights_only=False)
            loaded = self.models["depth_model"].load_state_dict(ckpt['model'], strict=False)
            logger.info("DUSt3R load result: %s", loaded)
            del ckpt
        self.parameters_to_train += list(
            filter(lambda p: p.requires_grad, self.models["depth_model"].parameters())
        )

        # ---- Position encoder/decoder (optical flow) ----
        self.models["position_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_images=2)
        self.models["position_encoder"].to(self.device)
        self.parameters_to_train_0 += list(self.models["position_encoder"].parameters())

        self.models["position"] = networks.PositionDecoder(
            self.models["position_encoder"].num_ch_enc, self.opt.scales)
        self.models["position"].to(self.device)
        self.parameters_to_train_0 += list(self.models["position"].parameters())

        # ---- Transform encoder/decoder (appearance flow) ----
        self.models["transform_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_images=2)
        self.models["transform_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["transform_encoder"].parameters())

        self.models["transform"] = networks.TransformDecoder(
            self.models["transform_encoder"].num_ch_enc, self.opt.scales)
        self.models["transform"].to(self.device)
        self.parameters_to_train += list(self.models["transform"].parameters())

        # ---- Pose network ----
        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)
                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        # ---- Optimizers ----
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.5)
        self.model_optimizer_0 = optim.Adam(self.parameters_to_train_0, 1e-4)
        self.model_lr_scheduler_0 = optim.lr_scheduler.StepLR(
            self.model_optimizer_0, self.opt.scheduler_step_size, 0.5)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        logger.info("Training model named: %s", self.opt.model_name)
        logger.info("Models saved to: %s", self.opt.log_dir)
        logger.info("Device: %s", self.device)

        # ---- Data ----
        datasets_dict = {"synthetic_colon": datasets.SyntheticColonDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        img_ext = '.png'
        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=1, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.spatial_transform = SpatialTransformer((self.opt.height, self.opt.width))
        self.spatial_transform.to(self.device)

        self.get_occu_mask_backward = get_occu_mask_backward((self.opt.height, self.opt.width))
        self.get_occu_mask_backward.to(self.device)

        self.get_occu_mask_bidirection = get_occu_mask_bidirection((self.opt.height, self.opt.width))
        self.get_occu_mask_bidirection.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        self.position_depth = {}

        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

            self.position_depth[scale] = optical_flow((h, w), self.opt.batch_size, h, w)
            self.position_depth[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        logger.info("Using split: %s", self.opt.split)
        logger.info("Training items: %d, Validation items: %d",
                    len(train_dataset), len(val_dataset))

        self.save_opts()
        self._log_params()

    def _log_params(self):
        total = 0
        trainable = 0
        for name, param in self.models["depth_model"].named_parameters():
            total += np.prod(param.size())
            if param.requires_grad:
                trainable += np.prod(param.size())
        logger.info("Total params: %d, Trainable: %d (%.1f%%)",
                    total, trainable, 100 * trainable / total)

    def set_train_0(self):
        """Enable gradients only for position models (Phase 1 training)."""
        for m in ["position_encoder", "position"]:
            for p in self.models[m].parameters():
                p.requires_grad = True
            self.models[m].train()

        for m in ["depth_model", "encoder", "pose_encoder", "pose",
                   "transform_encoder", "transform"]:
            for p in self.models[m].parameters():
                p.requires_grad = False
            self.models[m].eval()

    def set_train(self):
        """Enable gradients for all models except position (Phase 2 training)."""
        for m in ["position_encoder", "position"]:
            for p in self.models[m].parameters():
                p.requires_grad = False
            self.models[m].eval()

        for m in ["encoder", "depth_model", "pose_encoder", "pose",
                  "transform_encoder", "transform"]:
            for p in self.models[m].parameters():
                p.requires_grad = True
            self.models[m].train()

        if self.opt.lora_rank > 0:
            self.models["depth_model"].freeze_all_except_lora()
        else:
            self.models["depth_model"].set_freeze(self.models["depth_model"].freeze)

        self.models["encoder"].train()
        self.models["depth_model"].train()

    def set_eval(self):
        """Set all models to eval mode."""
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline."""
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()

    def run_epoch(self):
        """Run a single epoch of training."""
        logger.info("Epoch %d", self.epoch)

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()

            # Phase 1: train position encoder/decoder
            self.set_train_0()
            _, losses_0 = self.process_batch_0(inputs)
            self.model_optimizer_0.zero_grad()
            losses_0["loss"].backward()
            self.model_optimizer_0.step()

            # Phase 2: train all models
            self.set_train()
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            if batch_idx % self.opt.log_frequency == 0:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log("train", inputs, outputs, losses)

            self.step += 1

        self.model_lr_scheduler.step()
        self.model_lr_scheduler_0.step()

        if (self.epoch + 1) % self.opt.save_frequency == 0:
            self.save_model()

    def process_batch_0(self, inputs):
        """Phase 1: forward pass for position (optical flow) training."""
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        outputs = {}
        outputs.update(self.predict_poses_0(inputs))
        losses = self.compute_losses_0(inputs, outputs)

        return outputs, losses

    def predict_poses_0(self, inputs):
        """Predict optical flow between frames."""
        outputs = {}
        if self.num_pose_frames == 2:
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    inputs_all = [pose_feats[f_i], pose_feats[0]]
                    inputs_all_reverse = [pose_feats[0], pose_feats[f_i]]

                    position_inputs = self.models["position_encoder"](torch.cat(inputs_all, 1))
                    position_inputs_reverse = self.models["position_encoder"](torch.cat(inputs_all_reverse, 1))
                    outputs_0 = self.models["position"](position_inputs)
                    outputs_1 = self.models["position"](position_inputs_reverse)

                    for scale in self.opt.scales:
                        outputs[("position", scale, f_i)] = outputs_0[("position", scale)]
                        outputs[("position", "high", scale, f_i)] = F.interpolate(
                            outputs[("position", scale, f_i)],
                            [self.opt.height, self.opt.width],
                            mode="bilinear", align_corners=False)
                        outputs[("registration", scale, f_i)] = self.spatial_transform(
                            inputs[("color", f_i, 0)],
                            outputs[("position", "high", scale, f_i)])

                        outputs[("position_reverse", scale, f_i)] = outputs_1[("position", scale)]
                        outputs[("position_reverse", "high", scale, f_i)] = F.interpolate(
                            outputs[("position_reverse", scale, f_i)],
                            [self.opt.height, self.opt.width],
                            mode="bilinear", align_corners=False)
                        outputs[("occu_mask_backward", scale, f_i)], _ = self.get_occu_mask_backward(
                            outputs[("position_reverse", "high", scale, f_i)])
                        outputs[("occu_map_bidirection", scale, f_i)] = self.get_occu_mask_bidirection(
                            outputs[("position", "high", scale, f_i)],
                            outputs[("position_reverse", "high", scale, f_i)])

                    # Transform
                    transform_input = [outputs[("registration", 0, f_i)], inputs[("color", 0, 0)]]
                    transform_inputs = self.models["transform_encoder"](torch.cat(transform_input, 1))
                    outputs_2 = self.models["transform"](transform_inputs)

                    for scale in self.opt.scales:
                        outputs[("transform", scale, f_i)] = outputs_2[("transform", scale)]
                        outputs[("transform", "high", scale, f_i)] = F.interpolate(
                            outputs[("transform", scale, f_i)],
                            [self.opt.height, self.opt.width],
                            mode="bilinear", align_corners=False)
                        outputs[("refined", scale, f_i)] = (
                            outputs[("transform", "high", scale, f_i)] *
                            outputs[("occu_mask_backward", 0, f_i)].detach() +
                            inputs[("color", 0, 0)])
                        outputs[("refined", scale, f_i)] = torch.clamp(
                            outputs[("refined", scale, f_i)], min=0.0, max=1.0)

        return outputs

    def compute_losses_0(self, inputs, outputs):
        """Phase 1 losses: optical flow registration + smoothness."""
        losses = {}
        total_loss = 0.0

        for scale in self.opt.scales:
            loss = 0.0
            loss_smooth_registration = 0.0
            loss_registration = 0.0

            color = inputs[("color", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:
                occu_mask_backward = outputs[("occu_mask_backward", 0, frame_id)].detach()
                loss_smooth_registration += get_smooth_loss(
                    outputs[("position", scale, frame_id)], color)
                loss_registration += self.lap(
                    outputs[("registration", scale, frame_id)],
                    outputs[("refined", scale, frame_id)].detach(),
                    occu_mask_backward)

            loss += loss_registration / 2.0
            loss += self.opt.position_smoothness * (loss_smooth_registration / 2.0) / (2 ** scale)

            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def get_conf_log(self, x):
        return x, torch.log(x)

    def process_batch(self, inputs):
        """Phase 2: forward pass with DUSt3R depth + pose + transform."""
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        # Encoder features for pose network
        if self.opt.pose_model_type == "shared":
            all_color_aug = torch.cat([inputs["color_aug", i, 0] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]
            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]
        else:
            features = self.models["encoder"](inputs["color_aug", 0, 0])

        # Instance keys
        if ("instance", 0, 0) not in inputs:
            frame_ids = inputs["frame_id"]
            frame_ids = frame_ids.tolist() if isinstance(frame_ids, torch.Tensor) else frame_ids
            inputs[("instance", self.opt.frame_ids[0], 0)] = [
                [fid + self.opt.frame_ids[0]] for fid in frame_ids]
            inputs[("instance", self.opt.frame_ids[1], 0)] = [
                [fid + self.opt.frame_ids[1]] for fid in frame_ids]
            inputs[("instance", self.opt.frame_ids[2], 0)] = [
                [fid + self.opt.frame_ids[2]] for fid in frame_ids]

        # ---- DUSt3R forward pass ----
        view0 = {
            'img': inputs[("color_norm_aug", self.opt.frame_ids[0], 0)],
            'instance': inputs[("instance", self.opt.frame_ids[0], 0)]
        }
        view2 = {
            'img': inputs[("color_norm_aug", self.opt.frame_ids[2], 0)],
            'instance': inputs[("instance", self.opt.frame_ids[2], 0)]
        }

        pred00, pred02 = self.models["depth_model"](view0, view2, return_pose=False)
        pred22, pred20 = self.models["depth_model"](view2, view0, return_pose=False)

        pr_pts00 = pred00['pts3d'].clone()
        pr_pts02 = pred02['pts3d_in_other_view'].clone()
        pr_pts22 = pred22['pts3d'].clone()
        pr_pts20 = pred20['pts3d_in_other_view'].clone()

        batch_size = pr_pts22.shape[0]
        height, width = pr_pts22.shape[1], pr_pts22.shape[2]

        # Validity masks
        valid00 = (pr_pts00[..., 2] >= 0) & (pr_pts00[..., 2] <= 150)
        valid02 = (pr_pts02[..., 2] >= 0) & (pr_pts02[..., 2] <= 150)
        valid22 = (pr_pts22[..., 2] >= 0) & (pr_pts22[..., 2] <= 150)
        valid20 = (pr_pts20[..., 2] >= 0) & (pr_pts20[..., 2] <= 150)

        # Normalize point clouds
        pr_pts00, pr_pts02, norm_factor00 = normalize_pointcloud(
            pr_pts00, pr_pts02, 'avg_dis', valid00, valid02)
        pr_pts22 = pr_pts22 / norm_factor00
        pr_pts20 = pr_pts20 / norm_factor00

        # Confidence
        pred00['conf'][~valid00] = 1
        pred22['conf'][~valid22] = 1
        conf00, log_conf00 = self.get_conf_log(pred00['conf'].reshape(batch_size, height, width))
        conf22, log_conf22 = self.get_conf_log(pred22['conf'].reshape(batch_size, height, width))

        # Build outputs dict
        outputs = {
            ("pts3d", self.opt.frame_ids[0]): pr_pts00,
            ("pts3d", self.opt.frame_ids[2]): pr_pts22,
            ("depth", self.opt.frame_ids[0], 0): pr_pts00[..., 2].unsqueeze(1),
            ("depth", self.opt.frame_ids[2], 0): pr_pts22[..., 2].unsqueeze(1),
            ("conf", self.opt.frame_ids[0], 0): conf00.unsqueeze(1),
            ("conf", self.opt.frame_ids[2], 0): conf22.unsqueeze(1),
            ("log_conf", self.opt.frame_ids[0], 0): log_conf00.unsqueeze(1),
            ("log_conf", self.opt.frame_ids[2], 0): log_conf22.unsqueeze(1)
        }

        # Clamp depth
        outputs[("depth", self.opt.frame_ids[0], 0)] = torch.clamp(
            outputs[("depth", self.opt.frame_ids[0], 0)], min=0, max=250)
        outputs[("depth", self.opt.frame_ids[2], 0)] = torch.clamp(
            outputs[("depth", self.opt.frame_ids[2], 0)], min=0, max=250)

        # Multi-scale depth
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)
            outputs[("depth", self.opt.frame_ids[0], scale)] = F.interpolate(
                outputs[("depth", self.opt.frame_ids[0], 0)], [h, w],
                mode="bilinear", align_corners=True)
            outputs[("depth", self.opt.frame_ids[2], scale)] = F.interpolate(
                outputs[("depth", self.opt.frame_ids[2], 0)], [h, w],
                mode="bilinear", align_corners=True)

        # Pose predictions
        frame_ids1 = self.opt.frame_ids
        frame_ids2 = [self.opt.frame_ids[2], self.opt.frame_ids[1], self.opt.frame_ids[0]]

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features, outputs, frame_ids1, norm_factor00))
            outputs.update(self.predict_poses(inputs, features, outputs, frame_ids2, norm_factor00))

        # Predicted intrinsics at full resolution (scale 0)
        outputs[("pred_K", self.opt.frame_ids[0], 0)], outputs[("pred_inv_K", self.opt.frame_ids[0], 0)] = \
            self._get_intrinsics(pr_pts00, self.opt.width, self.opt.height)
        outputs[("pred_K", self.opt.frame_ids[2], 0)], outputs[("pred_inv_K", self.opt.frame_ids[2], 0)] = \
            self._get_intrinsics(pr_pts22, self.opt.width, self.opt.height)

        self.generate_images_pred(inputs, outputs, frame_ids1)
        self.generate_images_pred(inputs, outputs, frame_ids2)

        frame_ids_list = [frame_ids1, frame_ids2]
        losses = self.compute_losses(inputs, outputs, frame_ids_list)

        # ---- Pose loss (geometric consistency) ----
        pose_1_to_2 = outputs[("cam_T_cam", self.opt.frame_ids[0], self.opt.frame_ids[2])]
        pose_2_to_1 = outputs[("cam_T_cam", self.opt.frame_ids[2], self.opt.frame_ids[0])]
        ones = torch.ones_like(pr_pts22[..., :1])
        homogeneous_points_02 = torch.cat([pr_pts02, ones], dim=-1)
        homogeneous_points_20 = torch.cat([pr_pts20, ones], dim=-1)

        transformed_pr_pts22 = torch.bmm(
            homogeneous_points_02.reshape(batch_size, -1, 4),
            pose_1_to_2.transpose(1, 2))[..., :3]
        transformed_pr_pts22 = transformed_pr_pts22.reshape(
            batch_size, self.opt.height, self.opt.width, 3)

        transformed_pr_pts00 = torch.bmm(
            homogeneous_points_20.reshape(batch_size, -1, 4),
            pose_2_to_1.transpose(1, 2))[..., :3]
        transformed_pr_pts00 = transformed_pr_pts00.reshape(
            batch_size, self.opt.height, self.opt.width, 3)

        pose_loss = (torch.norm(transformed_pr_pts00 - pr_pts00, dim=-1) +
                     torch.norm(transformed_pr_pts22 - pr_pts22, dim=-1)).mean() / 2.0
        losses["pose_loss"] = pose_loss
        losses["loss"] = losses["loss"] + pose_loss

        # Focal loss
        pred_focal00 = outputs[("pred_K", self.opt.frame_ids[0], 0)][:, 0, 0]
        pred_focal22 = outputs[("pred_K", self.opt.frame_ids[2], 0)][:, 0, 0]
        focal_loss = torch.norm(
            (pred_focal00 - pred_focal22) / (pred_focal22 + pred_focal00), dim=-1).mean() / 2.0
        losses["focal_loss"] = focal_loss
        losses["loss"] = losses["loss"] + 0.1 * focal_loss

        # Pose consistency loss
        pose_2_to_1 = outputs[("cam_T_cam", self.opt.frame_ids[2], self.opt.frame_ids[1])]
        pose_2_to_0 = outputs[("cam_T_cam", self.opt.frame_ids[2], self.opt.frame_ids[0])]
        pose_0_to_1 = outputs[("cam_T_cam", self.opt.frame_ids[0], self.opt.frame_ids[1])]
        transformed_pose_2_to_1 = torch.bmm(pose_0_to_1, pose_2_to_0)
        pose_consisten_loss = torch.norm(
            transformed_pose_2_to_1 - pose_2_to_1, dim=-1).mean()
        losses["pose_consisten_loss"] = pose_consisten_loss

        # Geometry loss
        optical_flow_0_to_2 = outputs[("position", "high", 0, self.opt.frame_ids[0], self.opt.frame_ids[2])].detach()
        optical_flow_2_to_0 = outputs[("position", "high", 0, self.opt.frame_ids[2], self.opt.frame_ids[0])].detach()
        occu_mask_backward_02 = outputs[("occu_mask_backward", 0, self.opt.frame_ids[0], self.opt.frame_ids[2])].detach()
        occu_mask_backward_20 = outputs[("occu_mask_backward", 0, self.opt.frame_ids[2], self.opt.frame_ids[0])].detach()

        pr_pts00_orig = pr_pts00.detach()
        pr_pts22_orig = pr_pts22.detach()
        flow_matched_pts02 = self.spatial_transform(
            pr_pts00_orig.permute(0, 3, 1, 2), optical_flow_0_to_2)
        flow_matched_pts20 = self.spatial_transform(
            pr_pts22_orig.permute(0, 3, 1, 2), optical_flow_2_to_0)

        if occu_mask_backward_02.dim() == 3:
            occu_mask_backward_02 = occu_mask_backward_02.unsqueeze(1)
        if occu_mask_backward_20.dim() == 3:
            occu_mask_backward_20 = occu_mask_backward_20.unsqueeze(1)

        pts_diff02 = torch.abs(flow_matched_pts02 - pr_pts02.permute(0, 3, 1, 2))
        pts_diff20 = torch.abs(flow_matched_pts20 - pr_pts20.permute(0, 3, 1, 2))

        geometry_loss = (
            (pts_diff02 * occu_mask_backward_02).sum() / (occu_mask_backward_02.sum() + 1e-7) +
            (pts_diff20 * occu_mask_backward_20).sum() / (occu_mask_backward_20.sum() + 1e-7))
        losses["geometry_loss"] = geometry_loss
        losses["loss"] = losses["loss"] + 0.1 * geometry_loss

        return outputs, losses

    def predict_poses(self, inputs, features, disps, frame_ids, norm_factor):
        """Predict poses between frames (used in Phase 2)."""
        outputs = {}
        if self.num_pose_frames == 2:
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in frame_ids[1:]:
                if f_i != "s":
                    inputs_all = [pose_feats[f_i], pose_feats[frame_ids[0]]]
                    inputs_all_reverse = [pose_feats[frame_ids[0]], pose_feats[f_i]]

                    # Optical flow
                    position_inputs = self.models["position_encoder"](torch.cat(inputs_all, 1))
                    position_inputs_reverse = self.models["position_encoder"](torch.cat(inputs_all_reverse, 1))
                    outputs_0 = self.models["position"](position_inputs)
                    outputs_1 = self.models["position"](position_inputs_reverse)

                    for scale in self.opt.scales:
                        outputs[("position", scale, f_i, frame_ids[0])] = outputs_0[("position", scale)]
                        outputs[("position", "high", scale, f_i, frame_ids[0])] = F.interpolate(
                            outputs[("position", scale, f_i, frame_ids[0])],
                            [self.opt.height, self.opt.width],
                            mode="bilinear", align_corners=False)
                        outputs[("registration", scale, f_i, frame_ids[0])] = self.spatial_transform(
                            inputs[("color", f_i, 0)],
                            outputs[("position", "high", scale, f_i, frame_ids[0])])

                        outputs[("position_reverse", scale, f_i, frame_ids[0])] = outputs_1[("position", scale)]
                        outputs[("position_reverse", "high", scale, f_i, frame_ids[0])] = F.interpolate(
                            outputs[("position_reverse", scale, f_i, frame_ids[0])],
                            [self.opt.height, self.opt.width],
                            mode="bilinear", align_corners=False)
                        outputs[("occu_mask_backward", scale, f_i, frame_ids[0])], \
                            outputs[("occu_map_backward", scale, f_i, frame_ids[0])] = \
                            self.get_occu_mask_backward(
                                outputs[("position_reverse", "high", scale, f_i, frame_ids[0])])
                        outputs[("occu_map_bidirection", scale, f_i, frame_ids[0])] = \
                            self.get_occu_mask_bidirection(
                                outputs[("position", "high", scale, f_i, frame_ids[0])],
                                outputs[("position_reverse", "high", scale, f_i, frame_ids[0])])

                    # Appearance flow
                    transform_input = [
                        outputs[("registration", 0, f_i, frame_ids[0])],
                        inputs[("color", frame_ids[0], 0)]]
                    transform_inputs = self.models["transform_encoder"](torch.cat(transform_input, 1))
                    outputs_2 = self.models["transform"](transform_inputs)

                    for scale in self.opt.scales:
                        outputs[("transform", scale, f_i, frame_ids[0])] = outputs_2[("transform", scale)]
                        outputs[("transform", "high", scale, f_i, frame_ids[0])] = F.interpolate(
                            outputs[("transform", scale, f_i, frame_ids[0])],
                            [self.opt.height, self.opt.width],
                            mode="bilinear", align_corners=False)
                        outputs[("refined", scale, f_i, frame_ids[0])] = (
                            outputs[("transform", "high", scale, f_i, frame_ids[0])] *
                            outputs[("occu_mask_backward", 0, f_i, frame_ids[0])].detach() +
                            inputs[("color", frame_ids[0], 0)])
                        outputs[("refined", scale, f_i, frame_ids[0])] = torch.clamp(
                            outputs[("refined", scale, f_i, frame_ids[0])], min=0.0, max=1.0)

                    # Pose
                    pose_inputs = [self.models["pose_encoder"](torch.cat(inputs_all, 1))]
                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", frame_ids[0], f_i)] = axisangle
                    outputs[("translation", frame_ids[0], f_i)] = translation / norm_factor
                    outputs[("cam_T_cam", frame_ids[0], f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0])

        return outputs

    def generate_images_pred(self, inputs, outputs, frame_ids):
        """Generate warped color images via reprojection."""
        for scale in self.opt.scales:
            depth = outputs[("depth", frame_ids[0], scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                depth = F.interpolate(
                    depth, [self.opt.height, self.opt.width],
                    mode="bilinear", align_corners=True)
                source_scale = 0

            for i, frame_id in enumerate(frame_ids[1:]):
                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", frame_ids[0], frame_id)]

                if self.opt.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", frame_ids[0], frame_id)]
                    translation = outputs[("translation", frame_ids[0], frame_id)]
                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)
                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0],
                        frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, outputs[("pred_inv_K", frame_ids[0], source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, outputs[("pred_K", frame_ids[0], source_scale)], T)

                outputs[("sample", frame_ids[0], frame_id, scale)] = pix_coords
                outputs[("color", frame_ids[0], frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, 0)],
                    outputs[("sample", frame_ids[0], frame_id, scale)],
                    padding_mode="border")

                outputs[("position_depth", frame_ids[0], frame_id, scale)] = \
                    self.position_depth[source_scale](
                        cam_points, outputs[("pred_K", frame_ids[0], source_scale)], T)

    def _get_intrinsics(self, pts3d, W, H):
        """Estimate focal length from point cloud and build K matrix."""
        from dust3r.post_process import estimate_focal_knowing_depth
        device = pts3d.device
        pp = torch.tensor([W / 2, H / 2], device=device)
        focal = estimate_focal_knowing_depth(pts3d, pp, 'weiszfeld', 0., float('inf'))
        B = pts3d.shape[0]
        K = torch.zeros((B, 4, 4), device=device)
        for i in range(B):
            K[i, 0, 0] = K[i, 1, 1] = focal[i] + 1e-6
            K[i, :2, 2] = pp
            K[i, 2, 2] = 1.0
            K[i, 3, 3] = 1.0
        inv_K = torch.inverse(K)
        return K, inv_K

    def compute_reprojection_loss(self, pred, target):
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        return reprojection_loss

    def compute_reprojection_loss_conf(self, pred, target, conf, log_conf, alpha=0.2):
        abs_diff = torch.abs(target - pred)
        conf_abs_diff = abs_diff * conf - alpha * log_conf
        l1_loss = conf_abs_diff.mean(1, True)
        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            conf_ssim_loss = self.ssim(pred, target) * conf - alpha * log_conf
            ssim_loss = conf_ssim_loss.mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        return reprojection_loss

    def compute_losses(self, inputs, outputs, frame_ids_list):
        """Phase 2 losses: reprojection + transform + smoothness."""
        losses = {}
        total_loss = 0.0

        for scale in self.opt.scales:
            loss = 0.0
            loss_reprojection = 0.0
            loss_transform = 0.0
            loss_cvt = 0.0

            for frame_ids in frame_ids_list:
                disp = outputs[("depth", frame_ids[0], scale)]
                color = inputs[("color", frame_ids[0], scale)]

                for frame_id in frame_ids[1:]:
                    occu_mask_backward = outputs[
                        ("occu_mask_backward", 0, frame_id, frame_ids[0])].detach()

                    # Confidence-weighted reprojection
                    if frame_ids[0] == self.opt.frame_ids[0]:
                        loss_reprojection += 0.9 * ((
                            self.compute_reprojection_loss_conf(
                                outputs[("color", frame_ids[0], frame_id, scale)],
                                outputs[("refined", scale, frame_id, frame_ids[0])],
                                outputs[("conf", frame_ids[0], 0)],
                                outputs[("log_conf", frame_ids[0], 0)]) *
                            occu_mask_backward).sum() / occu_mask_backward.sum())
                    loss_reprojection += 0.1 * ((
                        self.compute_reprojection_loss(
                            outputs[("color", frame_ids[0], frame_id, scale)],
                            outputs[("refined", scale, frame_id, frame_ids[0])]) *
                        occu_mask_backward).sum() / occu_mask_backward.sum())

                    # Transform constraint
                    loss_transform += (
                        torch.abs(
                            outputs[("refined", scale, frame_id, frame_ids[0])] -
                            outputs[("registration", scale, frame_id, frame_ids[0])].detach()
                        ).mean(1, True) * occu_mask_backward).sum() / occu_mask_backward.sum()

                    # Brightness transform smoothness
                    loss_cvt += get_smooth_bright(
                        outputs[("transform", "high", scale, frame_id, frame_ids[0])],
                        inputs[("color", frame_ids[0], 0)],
                        outputs[("registration", scale, frame_id, frame_ids[0])].detach(),
                        occu_mask_backward)

                mean_disp = disp.mean(2, True).mean(3, True)
                norm_disp = disp / (mean_disp + 1e-7)
                smooth_loss = get_smooth_loss(norm_disp, color)

                loss += loss_reprojection / 2.0
                loss += self.opt.transform_constraint * (loss_transform / 2.0)
                loss += self.opt.transform_smoothness * (loss_cvt / 2.0)
                loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

                total_loss += loss
            losses["loss/{}".format(scale)] = loss
            losses["proj_loss/{}".format(scale)] = loss_reprojection
            losses["cvt_loss/{}".format(scale)] = loss_cvt
            losses["smooth_loss/{}".format(scale)] = smooth_loss
            losses["transform_loss/{}".format(scale)] = loss_transform

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def val(self):
        """Validate on a single minibatch."""
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)
            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement."""
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = (
            "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}"
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        )
        print(print_string.format(
            self.epoch, batch_idx, samples_per_sec, loss,
            sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write tensorboard events."""
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids[1:]:
                    for key in ["transform", "registration", "refined"]:
                        if key == "transform":
                            k = ("transform", "high", s, frame_id, self.opt.frame_ids[0])
                        elif key == "registration":
                            k = ("registration", s, frame_id, self.opt.frame_ids[0])
                        else:
                            k = ("refined", s, frame_id, self.opt.frame_ids[0])
                        if k in outputs:
                            writer.add_image(
                                "{}_{}_{}/{}".format(key, frame_id, s, j),
                                outputs[k][j].data, self.step)
                    if s == 0:
                        writer.add_image(
                            "occu_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs[("occu_mask_backward", s, frame_id, self.opt.frame_ids[0])][j].data,
                            self.step)

                writer.add_image(
                    "disp0_{}/{}".format(s, j),
                    normalize_image(outputs[("depth", self.opt.frame_ids[0], s)][j]), self.step)
                writer.add_image(
                    "disp1_{}/{}".format(s, j),
                    normalize_image(outputs[("depth", self.opt.frame_ids[2], s)][j]), self.step)

    def save_opts(self):
        """Save options to JSON."""
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()
        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights."""
        save_folder = os.path.join(
            self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)
        logger.info("Saved models to %s", save_folder)

    def load_model(self):
        """Load model weights from disk."""
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)
        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        logger.info("Loading model from %s", self.opt.load_weights_folder)

        for n in self.opt.models_to_load:
            logger.info("Loading %s weights...", n)
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path, map_location=self.device, weights_only=False)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        logger.info("Adam optimizer randomly initialized")
