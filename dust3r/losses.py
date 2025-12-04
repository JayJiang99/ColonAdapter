# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Implementation of DUSt3R training losses
# --------------------------------------------------------
from copy import copy, deepcopy
import torch
import torch.nn as nn

from dust3r.inference import get_pred_pts3d, find_opt_scaling, get_pred_pts3d_pose
from dust3r.utils.geometry import inv, geotrf, normalize_pointcloud
from dust3r.utils.geometry import get_joint_pointcloud_depth, get_joint_pointcloud_center_scale
from dust3r.utils.pose_error import get_rot_err, get_transl_ang_err, get_rot_err_torch, get_transl_ang_err_torch, pose_error_torch, pose_error_torch_euler
from dust3r.utils.device import to_numpy
def Sum(*losses_and_masks):
    loss, mask = losses_and_masks[0]
    if loss.ndim > 0:
        # we are actually returning the loss for every pixels
        return losses_and_masks
    else:
        # we are returning the global loss
        for loss2, mask2 in losses_and_masks[1:]:
            loss = loss + loss2
        return loss


class BaseCriterion(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction


class LLoss (BaseCriterion):
    """ L-norm loss
    """

    def forward(self, a, b):
        assert a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 3, f'Bad shape = {a.shape}'
        dist = self.distance(a, b)
        assert dist.ndim == a.ndim - 1  # one dimension less
        if self.reduction == 'none':
            return dist
        if self.reduction == 'sum':
            return dist.sum()
        if self.reduction == 'mean':
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f'bad {self.reduction=} mode')

    def distance(self, a, b):
        raise NotImplementedError()


class L21Loss (LLoss):
    """ Euclidean distance between 3d points  """

    def distance(self, a, b):
        return torch.norm(a - b, dim=-1)  # normalized L2 distance


L21 = L21Loss()


class Criterion (nn.Module):
    def __init__(self, criterion=None):
        super().__init__()
        assert isinstance(criterion, BaseCriterion), f'{criterion} is not a proper criterion!'
        self.criterion = copy(criterion)

    def get_name(self):
        return f'{type(self).__name__}({self.criterion})'

    def with_reduction(self, mode='none'):
        res = loss = deepcopy(self)
        while loss is not None:
            assert isinstance(loss, Criterion)
            loss.criterion.reduction = mode  # make it return the loss for each sample
            loss = loss._loss2  # we assume loss is a Multiloss
        return res


class MultiLoss (nn.Module):
    """ Easily combinable losses (also keep track of individual loss values):
        loss = MyLoss1() + 0.1*MyLoss2()
    Usage:
        Inherit from this class and override get_name() and compute_loss()
    """

    def __init__(self):
        super().__init__()
        self._alpha = 1
        self._loss2 = None

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

    def __mul__(self, alpha):
        assert isinstance(alpha, (int, float))
        res = copy(self)
        res._alpha = alpha
        return res
    __rmul__ = __mul__  # same

    def __add__(self, loss2):
        assert isinstance(loss2, MultiLoss)
        res = cur = copy(self)
        # find the end of the chain
        while cur._loss2 is not None:
            cur = cur._loss2
        cur._loss2 = loss2
        return res

    def __repr__(self):
        name = self.get_name()
        if self._alpha != 1:
            name = f'{self._alpha:g}*{name}'
        if self._loss2:
            name = f'{name} + {self._loss2}'
        return name

    def forward(self, *args, **kwargs):
        loss = self.compute_loss(*args, **kwargs)
        if isinstance(loss, tuple):
            loss, details = loss
        elif loss.ndim == 0:
            details = {self.get_name(): float(loss)}
        else:
            details = {}
        loss = loss * self._alpha

        if self._loss2:
            loss2, details2 = self._loss2(*args, **kwargs)
            loss = loss + loss2
            details |= details2

        return loss, details


class Regr3D (Criterion, MultiLoss):
    """ Ensure that all 3D points are correct.
        Asymmetric loss: view1 is supposed to be the anchor.

        P1 = RT1 @ D1, point clouds at world coord_frame
        P2 = RT2 @ D2
        loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1), I is identical matrix
        loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
              = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(self, criterion, norm_mode='avg_dis', gt_scale=False):
        super().__init__(criterion)
        self.norm_mode = norm_mode
        self.gt_scale = gt_scale

    def get_all_pts3d(self, gt1, gt2, pred1, pred2, dist_clip=None):
        # everything is normalized w.r.t. camera of view1
        in_camera1 = inv(gt1['camera_pose'])
        gt_pts1 = geotrf(in_camera1, gt1['pts3d'])  # B,H,W,3
        gt_pts2 = geotrf(in_camera1, gt2['pts3d'])  # B,H,W,3

        valid1 = gt1['valid_mask'].clone()
        valid2 = gt2['valid_mask'].clone()

        if dist_clip is not None:
            # points that are too far-away == invalid
            dis1 = gt_pts1.norm(dim=-1)  # (B, H, W)
            dis2 = gt_pts2.norm(dim=-1)  # (B, H, W)
            valid1 = valid1 & (dis1 <= dist_clip)
            valid2 = valid2 & (dis2 <= dist_clip)

        pr_pts1 = get_pred_pts3d(gt1, pred1, use_pose=False)
        pr_pts2 = get_pred_pts3d(gt2, pred2, use_pose=True)

        # normalize 3d points
        if self.norm_mode:
            pr_pts1, pr_pts2 = normalize_pointcloud(pr_pts1, pr_pts2, self.norm_mode, valid1, valid2)
        if self.norm_mode and not self.gt_scale:
            gt_pts1, gt_pts2 = normalize_pointcloud(gt_pts1, gt_pts2, self.norm_mode, valid1, valid2)

        return gt_pts1, gt_pts2, pr_pts1, pr_pts2, valid1, valid2, {}

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring = \
            self.get_all_pts3d(gt1, gt2, pred1, pred2, **kw)
        # loss on img1 side
        l1 = self.criterion(pred_pts1[mask1], gt_pts1[mask1])
        # loss on gt2 side
        l2 = self.criterion(pred_pts2[mask2], gt_pts2[mask2])
        self_name = type(self).__name__
        details = {self_name + '_pts3d_1': float(l1.mean()), self_name + '_pts3d_2': float(l2.mean())}
        return Sum((l1, mask1), (l2, mask2)), (details | monitoring)


class ConfLoss (MultiLoss):
    """ Weighted regression by learned confidence.
        Assuming the input pixel_loss is a pixel-level regression loss.

    Principle:
        high-confidence means high conf = 0.1 ==> conf_loss = x / 10 + alpha*log(10)
        low  confidence means low  conf = 10  ==> conf_loss = x * 10 - alpha*log(10) 

        alpha: hyperparameter
    """

    def __init__(self, pixel_loss, alpha=1):
        super().__init__()
        assert alpha > 0
        self.alpha = alpha
        self.pixel_loss = pixel_loss.with_reduction('none')

    def get_name(self):
        return f'ConfLoss({self.pixel_loss})'

    def get_conf_log(self, x):
        return x, torch.log(x)

    def compute_loss(self, gt1, gt2, pred1, pred2, res_pose1, **kw):
        # compute per-pixel loss
        ((loss1, msk1), (loss2, msk2)), details = self.pixel_loss(gt1, gt2, pred1, pred2, **kw)
        # check if any of the losses are nan, if it is loss1, print the shape of loss1
        if torch.isnan(loss1).any():
            print('nan in loss1', loss1.shape)
        if torch.isnan(loss2).any():
            print('nan in loss2', loss2.shape)
        # check if msk1 or msk2 is nan, if it is, print the shape of msk1 or msk2
        if torch.isnan(msk1).any():
            print('nan in msk1', msk1.shape)
        if torch.isnan(msk2).any():
            print('nan in msk2', msk2.shape)
        if loss1.numel() == 0:
            print('NO VALID POINTS in img1', force=True)
        if loss2.numel() == 0:
            print('NO VALID POINTS in img2', force=True)

        # weight by confidence
        conf1, log_conf1 = self.get_conf_log(pred1['conf'][msk1])
        conf2, log_conf2 = self.get_conf_log(pred2['conf'][msk2])
        
        conf_loss1 = loss1 * conf1 - self.alpha * log_conf1
        conf_loss2 = loss2 * conf2 - self.alpha * log_conf2
         # check if conf_loss1 or conf_loss2 is nan, if it is, print the shape of conf_loss1 or conf_loss2
        if torch.isnan(conf_loss1).any():
            print('nan in conf_loss1', conf_loss1.shape)
            # replace nan with zeros
            conf_loss1 = torch.nan_to_num(conf_loss1, nan=0.0)
        if torch.isnan(conf_loss2).any():
            print('nan in conf_loss2', conf_loss2.shape)
            # replace nan with zeros
            conf_loss2 = torch.nan_to_num(conf_loss2, nan=0.0)

        # average + nan protection (in case of no valid pixels at all)
        conf_loss1 = conf_loss1.mean() if conf_loss1.numel() > 0 else 0
        conf_loss2 = conf_loss2.mean() if conf_loss2.numel() > 0 else 0
        # conf_loss1 = abs(conf_loss1)
        # conf_loss2 = abs(conf_loss2)
        
        # # Compute pose loss
        gt_view1_to_view2 = torch.bmm(gt1['camera_pose'].inverse(), gt2['camera_pose'])
        # gt_view1_to_view2 = torch.bmm(gt2['camera_pose'].inverse(), gt1['camera_pose'])
        # Ensure res_pose is a tensor with proper shape
        pred_pose1 = res_pose1['pose']  
        # pred_pose2 = res_pose2['pose']
        
        # # Extract translation and rotation
        pre_transl1 = pred_pose1[..., :3, 3]
        # pre_transl2 = pred_pose2[..., :3, 3]
        # # Extract rotation matrices
        pred_rot1 = pred_pose1[..., :3, :3]
        # pred_rot2 = pred_pose2[..., :3, :3]
        
        pose_loss_dict1 = pose_error_torch_euler(pred_rot1, pre_transl1, gt_view1_to_view2)
        # pose_loss_dict2 = pose_error_torch(pred_rot2, pre_transl2, gt_view1_to_view2)
        #  set pose_loss_l1 as [t_err_l1, R_err_l1] which is 6 by 1 vector
        pose_loss_l1 = torch.cat([pose_loss_dict1['t_err_l1'], pose_loss_dict1['R_err_l1']], dim=-1)
        # pose_loss2 = (pose_loss_dict2['t_err_ang'] + pose_loss_dict2['R_err'])/180.0
        
        # pose_loss2 = pose_loss2.mean()
        # make pose_loss1 to have the same sign as (conf_loss1 + conf_loss2)
        # calculate confidence loss
        conf_pose, log_conf_pose = self.get_conf_log(res_pose1['conf'])
        # print('conf_pose.shape, log_conf_pose.shape')
        # print(conf_pose.shape, log_conf_pose.shape)
        conf_pose_loss = conf_pose * pose_loss_l1 - self.alpha * log_conf_pose
        conf_pose_loss = conf_pose_loss.mean() if conf_pose_loss.numel() > 0 else 0
        
        

        return conf_loss1 + conf_loss2+conf_pose_loss , dict(conf_loss_1=float(conf_loss1), conf_loss2=float(conf_loss2), pose_loss=float(conf_pose_loss), **details)


class Regr3D_ShiftInv (Regr3D):
    """ Same than Regr3D but invariant to depth shift.
    """

    def get_all_pts3d(self, gt1, gt2, pred1, pred2):
        # compute unnormalized points
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring = \
            super().get_all_pts3d(gt1, gt2, pred1, pred2)

        # compute median depth
        gt_z1, gt_z2 = gt_pts1[..., 2], gt_pts2[..., 2]
        pred_z1, pred_z2 = pred_pts1[..., 2], pred_pts2[..., 2]
        gt_shift_z = get_joint_pointcloud_depth(gt_z1, gt_z2, mask1, mask2)[:, None, None]
        pred_shift_z = get_joint_pointcloud_depth(pred_z1, pred_z2, mask1, mask2)[:, None, None]

        # subtract the median depth
        gt_z1 -= gt_shift_z
        gt_z2 -= gt_shift_z
        pred_z1 -= pred_shift_z
        pred_z2 -= pred_shift_z

        # monitoring = dict(monitoring, gt_shift_z=gt_shift_z.mean().detach(), pred_shift_z=pred_shift_z.mean().detach())
        return gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring


class Regr3D_ScaleInv (Regr3D):
    """ Same than Regr3D but invariant to depth shift.
        if gt_scale == True: enforce the prediction to take the same scale than GT
    """

    def get_all_pts3d(self, gt1, gt2, pred1, pred2):
        # compute depth-normalized points
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring = super().get_all_pts3d(gt1, gt2, pred1, pred2)

        # measure scene scale
        _, gt_scale = get_joint_pointcloud_center_scale(gt_pts1, gt_pts2, mask1, mask2)
        _, pred_scale = get_joint_pointcloud_center_scale(pred_pts1, pred_pts2, mask1, mask2)

        # prevent predictions to be in a ridiculous range
        pred_scale = pred_scale.clip(min=1e-3, max=1e3)

        # subtract the median depth
        if self.gt_scale:
            pred_pts1 *= gt_scale / pred_scale
            pred_pts2 *= gt_scale / pred_scale
            # monitoring = dict(monitoring, pred_scale=(pred_scale/gt_scale).mean())
        else:
            gt_pts1 /= gt_scale
            gt_pts2 /= gt_scale
            pred_pts1 /= pred_scale
            pred_pts2 /= pred_scale
            # monitoring = dict(monitoring, gt_scale=gt_scale.mean(), pred_scale=pred_scale.mean().detach())

        return gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring


class Regr3D_ScaleShiftInv (Regr3D_ScaleInv, Regr3D_ShiftInv):
    # calls Regr3D_ShiftInv first, then Regr3D_ScaleInv
    pass



def apply_log_to_norm(xyz):
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)
    xyz = xyz * torch.log1p(d)
    return xyz


class Regr3D_Metric (Regr3D):
    def __init__(self, criterion, norm_mode='avg_dis', gt_scale=False, opt_fit_gt=False,
                 sky_loss_value=2, max_metric_scale=False, loss_in_log=False):
        self.loss_in_log = loss_in_log
        if norm_mode.startswith('?'):
            # do no norm pts from metric scale datasets
            self.norm_all = False
            self.norm_mode = norm_mode[1:]
        else:
            self.norm_all = True
            self.norm_mode = norm_mode
        super().__init__(criterion, self.norm_mode, gt_scale)

        self.sky_loss_value = sky_loss_value
        self.max_metric_scale = max_metric_scale

    def get_all_pts3d(self, gt1, gt2, pred1, pred2, dist_clip=None):
        # everything is normalized w.r.t. camera of view1
        in_camera1 = inv(gt1['camera_pose'])
        gt_pts1 = geotrf(in_camera1, gt1['pts3d'])  # B,H,W,3
        gt_pts2 = geotrf(in_camera1, gt2['pts3d'])  # B,H,W,3

        valid1 = gt1['valid_mask'].clone()
        valid2 = gt2['valid_mask'].clone()

        if dist_clip is not None:
            # points that are too far-away == invalid
            dis1 = gt_pts1.norm(dim=-1)  # (B, H, W)
            dis2 = gt_pts2.norm(dim=-1)  # (B, H, W)
            valid1 = valid1 & (dis1 <= dist_clip)
            valid2 = valid2 & (dis2 <= dist_clip)

        if self.loss_in_log == 'before':
            # this only make sense when depth_mode == 'linear'
            gt_pts1 = apply_log_to_norm(gt_pts1)
            gt_pts2 = apply_log_to_norm(gt_pts2)

        pr_pts1 = get_pred_pts3d(gt1, pred1, use_pose=False).clone()
        pr_pts2 = get_pred_pts3d(gt2, pred2, use_pose=True).clone()
        # print if pred2 has camera_pose
        
        # check if pr_pts1 or pr_pts2 is nan, if it is, print the shape of pr_pts1 or pr_pts2
        if torch.isnan(pr_pts1).any():
            print('nan in pr_pts1', pr_pts1.shape)
            # replace nan with zeros
            pr_pts1 = torch.nan_to_num(pr_pts1, nan=0.0)
        if torch.isnan(pr_pts2).any():
            print('nan in pr_pts2', pr_pts2.shape)
            # replace nan with zeros
            pr_pts2 = torch.nan_to_num(pr_pts2, nan=0.0)
        if not self.norm_all:
            if self.max_metric_scale:
                B = valid1.shape[0]
                # valid1: B, H, W
                # torch.linalg.norm(gt_pts1, dim=-1) -> B, H, W
                # dist1_to_cam1 -> reshape to B, H*W
                dist1_to_cam1 = torch.where(valid1, torch.linalg.norm(gt_pts1, dim=-1), 0).view(B, -1)
                dist2_to_cam1 = torch.where(valid2, torch.linalg.norm(gt_pts2, dim=-1), 0).view(B, -1)

                # is_metric_scale: B
                # dist1_to_cam1.max(dim=-1).values -> B
                gt1['is_metric_scale'] = gt1['is_metric_scale'] \
                    & (dist1_to_cam1.max(dim=-1).values < self.max_metric_scale) \
                    & (dist2_to_cam1.max(dim=-1).values < self.max_metric_scale)
                gt2['is_metric_scale'] = gt1['is_metric_scale']

            mask = ~gt1['is_metric_scale']
        else:
            mask = torch.ones_like(gt1['is_metric_scale'])

        # normalize 3d points
        if self.norm_mode and mask.any():
            pr_pts1[mask], pr_pts2[mask] = normalize_pointcloud(pr_pts1[mask], pr_pts2[mask], self.norm_mode,
                                                                valid1[mask], valid2[mask])

        if self.norm_mode and not self.gt_scale:
            
            gt_pts1, gt_pts2, norm_factor = normalize_pointcloud(gt_pts1, gt_pts2, self.norm_mode,
                                                                 valid1, valid2, ret_factor=True)
            # apply the same normalization to prediction
            pr_pts1[~mask] = pr_pts1[~mask] / norm_factor[~mask]
            pr_pts2[~mask] = pr_pts2[~mask] / norm_factor[~mask]

        # return sky segmentation, making sure they don't include any labelled 3d points
        sky1 = gt1['sky_mask'] & (~valid1)
        sky2 = gt2['sky_mask'] & (~valid2)
        return gt_pts1, gt_pts2, pr_pts1, pr_pts2, valid1, valid2, sky1, sky2, {}

    def compute_loss(self, gt1, gt2, pred1, pred2,**kw):
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, sky1, sky2, monitoring = \
            self.get_all_pts3d(gt1, gt2, pred1, pred2,**kw)

        # check if anything from get_all_pts3d is nan, if it is, print the shape of the nan
        if torch.isnan(gt_pts1).any():
            print('nan in gt_pts1', gt_pts1.shape)
        if torch.isnan(gt_pts2).any():
            print('nan in gt_pts2', gt_pts2.shape)
        if torch.isnan(pred_pts1).any():
            print('nan in pred_pts1', pred_pts1.shape)
        if torch.isnan(pred_pts2).any():
            print('nan in pred_pts2', pred_pts2.shape)
        if self.sky_loss_value > 0:
            assert self.criterion.reduction == 'none', 'sky_loss_value should be 0 if no conf loss'
            # add the sky pixel as "valid" pixels...
            mask1 = mask1 | sky1
            mask2 = mask2 | sky2

        # loss on img1 side
        pred_pts1 = pred_pts1[mask1]
        gt_pts1 = gt_pts1[mask1]
        if self.loss_in_log and self.loss_in_log != 'before':
            # this only make sense when depth_mode == 'exp'
            pred_pts1 = apply_log_to_norm(pred_pts1)
            gt_pts1 = apply_log_to_norm(gt_pts1)
        l1 = self.criterion(pred_pts1, gt_pts1)
        # check if l1 is nan, if it is, print the shape of l1
        if torch.isnan(l1).any():
            print('nan in l1', l1.shape)

        # loss on gt2 side
        pred_pts2 = pred_pts2[mask2]
        gt_pts2 = gt_pts2[mask2]
        if self.loss_in_log and self.loss_in_log != 'before':
            pred_pts2 = apply_log_to_norm(pred_pts2)
            gt_pts2 = apply_log_to_norm(gt_pts2)
        l2 = self.criterion(pred_pts2, gt_pts2)
        # check if l2 is nan, if it is, print the shape of l2
        if torch.isnan(l2).any():
            print('nan in l2', l2.shape)

        if self.sky_loss_value > 0:
            assert self.criterion.reduction == 'none', 'sky_loss_value should be 0 if no conf loss'
            # ... but force the loss to be high there
            l1 = torch.where(sky1[mask1], self.sky_loss_value, l1)
            l2 = torch.where(sky2[mask2], self.sky_loss_value, l2)
        self_name = type(self).__name__
        details = {self_name + '_pts3d_1': float(l1.mean()), self_name + '_pts3d_2': float(l2.mean())}
        return Sum((l1, mask1), (l2, mask2)), (details | monitoring)


class Regr3D_ShiftInv_Metric (Regr3D_Metric):
    """ Same than Regr3D but invariant to depth shift.
    """

    def get_all_pts3d(self, gt1, gt2, pred1, pred2):
        # compute unnormalized points
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, sky1, sky2, monitoring = \
            super().get_all_pts3d(gt1, gt2, pred1, pred2)

        # compute median depth
        gt_z1, gt_z2 = gt_pts1[..., 2], gt_pts2[..., 2]
        pred_z1, pred_z2 = pred_pts1[..., 2], pred_pts2[..., 2]
        gt_shift_z = get_joint_pointcloud_depth(gt_z1, gt_z2, mask1, mask2)[:, None, None]
        pred_shift_z = get_joint_pointcloud_depth(pred_z1, pred_z2, mask1, mask2)[:, None, None]

        # subtract the median depth
        gt_z1 -= gt_shift_z
        gt_z2 -= gt_shift_z
        pred_z1 -= pred_shift_z
        pred_z2 -= pred_shift_z

        # monitoring = dict(monitoring, gt_shift_z=gt_shift_z.mean().detach(), pred_shift_z=pred_shift_z.mean().detach())
        return gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, sky1, sky2, monitoring


class Regr3D_ScaleInv_Metric (Regr3D_Metric):
    """ Same than Regr3D but invariant to depth scale.
        if gt_scale == True: enforce the prediction to take the same scale than GT
    """

    def get_all_pts3d(self, gt1, gt2, pred1, pred2):
        # compute depth-normalized points
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, sky1, sky2, monitoring = \
            super().get_all_pts3d(gt1, gt2, pred1, pred2)

        # measure scene scale
        _, gt_scale = get_joint_pointcloud_center_scale(gt_pts1, gt_pts2, mask1, mask2)
        _, pred_scale = get_joint_pointcloud_center_scale(pred_pts1, pred_pts2, mask1, mask2)

        # prevent predictions to be in a ridiculous range
        pred_scale = pred_scale.clip(min=1e-3, max=1e3)

        # subtract the median depth
        if self.gt_scale:
            pred_pts1 *= gt_scale / pred_scale
            pred_pts2 *= gt_scale / pred_scale
            # monitoring = dict(monitoring, pred_scale=(pred_scale/gt_scale).mean())
        else:
            gt_pts1 /= gt_scale
            gt_pts2 /= gt_scale
            pred_pts1 /= pred_scale
            pred_pts2 /= pred_scale
            # monitoring = dict(monitoring, gt_scale=gt_scale.mean(), pred_scale=pred_scale.mean().detach())

        return gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, sky1, sky2, monitoring


class Regr3D_ScaleShiftInv_Metric (Regr3D_ScaleInv_Metric, Regr3D_ShiftInv_Metric):
    # calls Regr3D_ShiftInv first, then Regr3D_ScaleInv
    pass

