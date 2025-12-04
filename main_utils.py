from __future__ import absolute_import, division, print_function
import os
import hashlib
import zipfile
from six.moves import urllib
import torch
import torch.nn.functional as F
from dust3r.post_process import estimate_focal_knowing_depth
def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def download_model_if_doesnt_exist(model_name):
    """If pretrained kitti model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
             "a964b8356e08a02d009609d9e3928f7c"),
        "stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
             "3dfb76bcff0786e4ec07ac00f658dd07"),
        "mono+stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
             "c024d69012485ed05d7eaa9617a96b81"),
        "mono_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
             "9c2f071e35027c895a4728358ffc913a"),
        "stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
             "41ec2de112905f85541ac33a854742d1"),
        "mono+stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
             "46c3b824f541d143a45c37df65fbab0a"),
        "mono_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
             "0ab0766efdfeea89a0d9ea8ba90e1e63"),
        "stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
             "afc2f2126d70cf3fdf26b550898b501a"),
        "mono+stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
             "cdc5fc9b23513c07d5b19235d9ef08f7"),
        }

    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = os.path.join("models", model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "encoder.pth")):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
            urllib.request.urlretrieve(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))

def view_synthesis(pts3d, pose_1_to_2, K, img):
        """
        Synthesize view using 3D points and pose transformation
        
        Args:
            pts3d: B, H, W, 3 - 3D points in source view coordinate frame
            pose_1_to_2: B, 4, 4 - Transformation from source to target view
            K: B, 3, 3 - Camera intrinsics
            img: B, 3, H, W - Source image
        Returns:
            B, 3, H, W - Synthesized image
        """
        B, H, W, _ = pts3d.shape
        device = pts3d.device
        
        # Transform points to target view coordinate frame
        ones = torch.ones_like(pts3d[...,:1])
        homogeneous_points = torch.cat([pts3d, ones], dim=-1)  # B,H,W,4
        
        # Apply pose transformation
        transformed_points = torch.bmm(
            homogeneous_points.reshape(B,-1,4), 
            pose_1_to_2.transpose(1,2)
        )  # B,H*W,4
        transformed_points = transformed_points[...,:3]  # B,H*W,3
        
        # Project transformed points
        projected = torch.bmm(K, transformed_points.transpose(1,2))  # B,3,H*W
        projected = projected.transpose(1,2)  # B,H*W,3
        
        # Get pixel coordinates
        pixel_coords = projected[...,:2] / (projected[...,2:] + self.eps)  # B,H*W,2
        pixel_coords = pixel_coords.reshape(B,H,W,2)  # B,H,W,2
        
        # Normalize coordinates to [-1,1]
        norm_coords = 2 * pixel_coords / torch.tensor([W-1, H-1], device=device) - 1
        
        # Sample from source image
        warped_img = F.grid_sample(
            img,
            norm_coords, 
            mode='bilinear',
            align_corners=True,
            padding_mode='zeros'
        )
        
        return warped_img

def get_intrinsics(pts3d,W,H, focal_mode='weiszfeld', min_focal=0., max_focal=float('inf')):
    device = pts3d.device
    pp = torch.tensor([W/2, H/2], device=device)
    focal = estimate_focal_knowing_depth(pts3d, pp, focal_mode, min_focal, max_focal)
    # print("focal", focal)
    B = pts3d.shape[0]
    K = torch.zeros((B, 4, 4), device=device)
    for i in range(B):
        K[i, 0, 0] = K[i, 1, 1] = focal[i]+1e-6
        K[i, :2, 2] = pp
        K[i, 2, 2] = 1.0
        K[i, 3, 3] = 1.0
    inv_K = torch.inverse(K)
    # print("K", K)
    # print("inv_K", inv_K)
    return K, inv_K


# def estimate_focal_knowing_depth(pts3d, pp, focal_mode='median', min_focal=0., max_focal=float('inf')):
#     """ CUDA优化版本焦距估计算法，支持以下特性：
#         1) 批量化张量运算（B同时处理多帧）
#         2) FP16/FP32混合精度兼容
#         3) 输入检查与设备自动匹配
#     """
#     assert pts3d.device.type == 'cuda', "Input must reside on CUDA device"
    
#     B, H, W, THREE = pts3d.shape
#     assert THREE == 3, "pts3d must have 3 channels in last dim"

#     # 生成像素网格(CUDA优化)
#     u, v = torch.meshgrid(
#         torch.arange(W, device=pts3d.device, dtype=torch.float32),
#         torch.arange(H, device=pts3d.device, dtype=torch.float32),
#         indexing='xy'
#     )  # (W,H), (W,H) → 转换为批处理兼容
#     # print the shape of u and v
#     print("u.shape", u.shape)
#     print("v.shape", v.shape)
#     pixels = torch.stack([u.T, v.T], dim=-1)  # (H,W,2)
#     pixels = pixels.view(1, H*W, 2) - pp.view(B, 1, 2)  # [B,HW,2]
    
#     # Reshape点云数据
#     pts3d_flat = pts3d.reshape(B, H*W, 3)  # (B, H*W, 3)
#     x, y, z = pts3d_flat.unbind(dim=-1)  # 三通道拆分
    
#     if focal_mode == 'median':
#         # CUDA加速中值计算方法 [^1]
#         fx_votes = (pixels[..., 0] * z) / x  # (B, HW)
#         fy_votes = (pixels[..., 1] * z) / y  
        
#         valid_mask = (~torch.isnan(fx_votes)) & (~torch.isnan(fy_votes))  # NaN过滤
#         fused_votes = torch.cat([fx_votes[valid_mask], fy_votes[valid_mask]], dim=0)
#         focal = torch.nanmedian(fused_votes.view(B, -1), dim=1).values  # (B,)
        
#     elif focal_mode == 'weiszfeld':
#         # Weiszfeld算法CUDA迭代优化 [^2]
#         xy_over_z = (pts3d_flat[..., :2] / pts3d_flat[..., 2].unsqueeze(-1))
#         xy_over_z = torch.nan_to_num(xy_over_z, nan=0.0, posinf=0.0, neginf=0.0)
        
#         # 闭式初始解 (CPU->CUDA保留梯度)
#         dot_xy_px = (xy_over_z * pixels).sum(dim=-1)  # (B, HW)
#         dot_xy_xy = (xy_over_z**2).sum(dim=-1)
#         focal = dot_xy_px.mean(dim=1) / (dot_xy_xy.mean(dim=1) + 1e-8)
        
#         # 迭代加权优化
#         for _ in range(10):
#             residual = pixels - focal.view(-1,1,1) * xy_over_z  # (B,HW,2)
#             dis = torch.norm(residual, p=2, dim=-1)            # 欧氏距离
#             w = torch.reciprocal(dis.clamp(min=1e-8))          # 计算权重
            
#             focal = (w * dot_xy_px).sum(dim=1) / ((w * dot_xy_xy).sum(dim=1) + 1e-8)  # 防止除零
            
#     else:
#         raise ValueError(f"Invalid focal_mode: '{focal_mode}'")

#     # 动态基础焦距计算 (原图尺寸无关)
#     focal_base = max(H, W) / (2 * torch.tan(torch.deg2rad(torch.tensor(60.0, device=pts3d.device))/2))
#     focal = torch.clamp(focal, min=min_focal*focal_base, max=max_focal*focal_base)
    
#     return focal