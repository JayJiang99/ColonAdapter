# import torch  # useless
import numpy as np
import cv2
from pdb import set_trace as bb
import torch
import math

def get_rot_err(rot_a, rot_b):
    rot_err = rot_a.T.dot(rot_b)
    rot_err = cv2.Rodrigues(rot_err)[0]
    rot_err = np.reshape(rot_err, (1,3))
    rot_err = np.reshape(np.linalg.norm(rot_err, axis = 1), -1) / np.pi * 180/360.0
    return rot_err[0]

def get_transl_ang_err(dir_a, dir_b):
    dot_product = np.sum(dir_a * dir_b)
    cos_angle = dot_product / (np.linalg.norm(dir_a) * np.linalg.norm(dir_b))
    angle = np.arccos(cos_angle)
    err = np.degrees(angle)/360.0
    return err


def error_auc(rError, tErrors, thresholds):
    """
    Args:
        Error (list): [N,]
        tErrors (list): [N,]
        thresholds (list)
    """
    error_matrix = np.concatenate((rError[:, None], tErrors[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    errors = [0] + sorted(list(max_errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    # thresholds = [5, 10, 20, 30]
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}


# def calculate_auc(r_error, t_error, max_threshold=30):
#     """
#     Calculate the Area Under the Curve (AUC) for the given error arrays using PyTorch.

#     :param r_error: torch.Tensor representing R error values (Degree).
#     :param t_error: torch.Tensor representing T error values (Degree).
#     :param max_threshold: maximum threshold value for binning the histogram.
#     :return: cumulative sum of normalized histogram of maximum error values.
#     """
#     # Concatenate the error tensors along a new axis
#     error_matrix = torch.stack((r_error, t_error), dim=1)

#     # Compute the maximum error value for each pair
#     max_errors, _ = torch.max(error_matrix, dim=1)

#     # Define histogram bins
#     bins = torch.arange(max_threshold + 1)

#     # Calculate histogram of maximum error values
#     histogram = torch.histc(max_errors, bins=max_threshold + 1, min=0, max=max_threshold)

#     # Normalize the histogram
#     num_pairs = float(max_errors.size(0))
#     normalized_histogram = histogram / num_pairs

#     # Compute and return the cumulative sum of the normalized histogram
#     return torch.cumsum(normalized_histogram, dim=0).mean()


def calculate_auc_np(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays.

    :param r_error: numpy array representing R error values (Degree).
    :param t_error: numpy array representing T error values (Degree).
    :param max_threshold: maximum threshold value for binning the histogram.
    :return: cumulative sum of normalized histogram of maximum error values.
    """

    # Concatenate the error arrays along a new axis
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)

    # Compute the maximum error value for each pair
    max_errors = np.max(error_matrix, axis=1)

    # Define histogram bins
    bins = np.arange(max_threshold + 1)

    # Calculate histogram of maximum error values
    histogram, _ = np.histogram(max_errors, bins=bins)

    # Normalize the histogram
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs

    # Compute and return the cumulative sum of the normalized histogram
    return np.mean(np.cumsum(normalized_histogram))

def get_rot_err_torch(rot_a, rot_b):
    """PyTorch version of rotation error calculation with NaN handling
    Args:
        rot_a, rot_b: [..., 3, 3] rotation matrices
    Returns:
        rotation error in degrees normalized by 360
    """
    # Check for NaN in input and replace with identity matrix if needed
    if torch.isnan(rot_a).any():
        rot_a = torch.where(torch.isnan(rot_a), torch.eye(3, device=rot_a.device), rot_a)
    if torch.isnan(rot_b).any():
        rot_b = torch.where(torch.isnan(rot_b), torch.eye(3, device=rot_b.device), rot_b)
    
    # Compute relative rotation: rot_a.T @ rot_b
    rot_err = torch.bmm(rot_a.transpose(-2, -1), rot_b)
    
    # Convert to axis-angle representation
    cos_theta = (torch.diagonal(rot_err, dim1=-2, dim2=-1).sum(-1) - 1) / 2
    # Clamp to handle numerical errors
    cos_theta = torch.clamp(cos_theta, -1, 1)
    theta = torch.acos(cos_theta)
    
    # Handle any remaining NaN values
    theta = torch.nan_to_num(theta, nan=0.0)
    
    # Convert to degrees and normalize by 360
    return theta / torch.pi

def get_transl_ang_err_torch(dir_a, dir_b):
    """PyTorch version of translation direction error calculation with NaN handling
    Args:
        dir_a, dir_b: [..., 3] translation vectors
    Returns:
        angle error in degrees normalized by 360
    """
    # Add small epsilon to prevent division by zero and handle NaN
    eps = 1e-8
    
    # Replace NaN values with small valid vectors
    if torch.isnan(dir_a).any():
        dir_a = torch.where(torch.isnan(dir_a), torch.tensor([eps, 0, 0], device=dir_a.device), dir_a)
    if torch.isnan(dir_b).any():
        dir_b = torch.where(torch.isnan(dir_b), torch.tensor([eps, 0, 0], device=dir_b.device), dir_b)
    
    # Handle zero vectors
    dir_a_norm = dir_a / (torch.norm(dir_a, dim=-1, keepdim=True) + eps)
    dir_b_norm = dir_b / (torch.norm(dir_b, dim=-1, keepdim=True) + eps)
    
    # Compute angle between vectors
    cos_angle = torch.sum(dir_a_norm * dir_b_norm, dim=-1)
    cos_angle = torch.clamp(cos_angle, -1, 1)
    angle = torch.acos(cos_angle)
    
    # Handle any remaining NaN values
    angle = torch.nan_to_num(angle, nan=0.0)
    
    # Convert to degrees and normalize by 360
    return angle / torch.pi

def pose_error_torch(R, t, Tgt, reduce=None):
    """Compute angular, scale and euclidean error of translation vector (metric). Compute angular rotation error."""

    Rgt = Tgt[:, :3, :3]                  # [B, 3, 3]
    tgt = Tgt[:, :3, 3:].transpose(1, 2)  # [B, 1, 3]
    # print('Rgt.shape, tgt.shape, R.shape, t.shape')
    # print(Rgt.shape, tgt.shape, R.shape, t.shape)
     # convert rotation matrix to rotation vector as Rgt_vec
    Rgt_vec = torch.stack([torch.from_numpy(cv2.Rodrigues(r.detach().cpu().numpy())[0]) for r in Rgt]).to(Rgt.device)
    R_vec = torch.stack([torch.from_numpy(cv2.Rodrigues(r.detach().cpu().numpy())[0]) for r in R]).to(R.device)
    
    
    # compute rotation error as L1 distance
    R_err_l1 = torch.linalg.norm(R_vec - Rgt_vec, dim=-1)
    # compute the translation error as L1 distance
    # get the normalization scale of tgt.
    scale_tgt = torch.linalg.norm(tgt, dim=-1)
    t_err_l1 = torch.linalg.norm(t - tgt, dim=-1) / scale_tgt

    scale_t = torch.linalg.norm(t, dim=-1)
    scale_tgt = torch.linalg.norm(tgt, dim=-1)

    cosine = (t @ tgt.transpose(1, 2)).squeeze(-1) / (scale_t * scale_tgt + 1e-9)
    cosine = torch.clip(cosine, -1.0, 1.0)    # handle numerical errors
    t_ang_err = torch.rad2deg(torch.acos(cosine))
    t_ang_err = torch.minimum(t_ang_err, 180 - t_ang_err)

    t_scale_err = scale_t / scale_tgt
    t_scale_err_sym = torch.maximum(scale_t / scale_tgt, scale_tgt / scale_t)
    t_euclidean_err = torch.linalg.norm(t - tgt, dim=-1)

    residual = R.transpose(1, 2) @ Rgt
    trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
    cosine = (trace - 1) / 2
    cosine = torch.clip(cosine, -1., 1.)  # handle numerical errors
    R_err = torch.rad2deg(torch.acos(cosine))
    

    if reduce is None:
        def fn(x): return x
    elif reduce == 'mean':
        fn = torch.mean
    elif reduce == 'median':
        fn = torch.median

    t_ang_err = fn(t_ang_err)
    t_scale_err = fn(t_scale_err)
    t_euclidean_err = fn(t_euclidean_err)
    R_err = fn(R_err)

    errors = {'t_err_ang': t_ang_err,
              't_err_scale': t_scale_err,
              't_err_scale_sym': t_scale_err_sym,
              't_err_euc': t_euclidean_err,
              'R_err': R_err,
              'R_err_l1': R_err_l1,
              't_err_l1': t_err_l1}
    return errors


def rotation_matrix_to_euler_angles(rotation_matrices):
    """
    将旋转矩阵转换为欧拉角（ZYX顺序）。
    
    参数:
    rotation_matrices (torch.Tensor): 形状为 [B, 3, 3] 的旋转矩阵张量。
    
    返回:
    torch.Tensor: 形状为 [B, 3] 的欧拉角张量（单位为弧度）。
    """
    B = rotation_matrices.shape[0]
    euler_angles = torch.zeros(B, 3, device=rotation_matrices.device)
    
    for i in range(B):
        R = rotation_matrices[i]
        
        # 计算Y轴的旋转角度
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        
        euler_angles[i, 0] = x
        euler_angles[i, 1] = y
        euler_angles[i, 2] = z
    
    euler_angles = euler_angles.unsqueeze(-1)
    return euler_angles

def pose_error_torch_euler(R, t, Tgt, reduce=None):
    """Compute angular, scale and euclidean error of translation vector (metric). Compute angular rotation error."""
    Rgt = Tgt[:, :3, :3]                  # [B, 3, 3]
    tgt = Tgt[:, :3, 3:].transpose(1, 2)  # [B, 1, 3]
    
    Rgt_euler = rotation_matrix_to_euler_angles(Rgt)
    R_euler = rotation_matrix_to_euler_angles(R)
    # divide by PI to get normalize the euler angle to [0, 1]
    R_err_l1 = torch.linalg.norm(R_euler / math.pi - Rgt_euler / math.pi, dim=-1)
    # R_err_l1 = torch.linalg.norm(R_euler - Rgt_euler, dim=-1)
    
     # get the normalization scale of tgt.
    scale_tgt = torch.linalg.norm(tgt, dim=-1)
    t_err_l1 = torch.linalg.norm(t - tgt, dim=-1) / scale_tgt
    
    return {'R_err_l1': R_err_l1, 't_err_l1': t_err_l1}
    
    
    