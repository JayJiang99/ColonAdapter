# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class
# --------------------------------------------------------
from copy import deepcopy
import torch
import os
from packaging import version
import huggingface_hub
import torch.nn as nn

from .utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape
from .heads import head_factory
from dust3r.patch_embed import get_patch_embed, ManyAR_PatchEmbed
# here is the path to third_party /home/zhiyijiang/monash/miccai2025/Endo3R/third_party
from third_party.raft import load_RAFT
from networks.resnet_encoder import ResnetEncoder

import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet  # noqa

inf = float('inf')

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), "Outdated huggingface_hub version, please reinstall requirements.txt"

def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")
class AsymmetricCroCo3DStereo (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/junyi/monst3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='encoder',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 use_pose_head=False,
                 lora_rank = 16,
                 lora_alpha = 1.0,
                 lora_dropout = 0.1,
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, use_pose_head, **croco_kwargs)
        self.set_freeze(freeze)
        if lora_rank > 0:
            self.enable_lora_finetuning(lora_rank, lora_alpha, lora_dropout)
        self._set_resnet_encoder(num_layers=18, pretrained=True)
        self.zero_convs = []
        for i in range(len(self.resnet_proj_layers) + 1):
            self.zero_convs.append(self.make_zero_conv(self.dec_embed_dim).cuda())
        self.zero_convs = nn.ModuleList(self.zero_convs)
        
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
            'encoder_and_decoder': [self.mask_token, self.patch_embed, self.enc_blocks, self.dec_blocks, self.dec_blocks2],
        }
        freeze_all_params(to_be_frozen[freeze])
        # print(f'Freezing {freeze} parameters')

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    # Add this class to the CroCoNet class
    def _set_resnet_encoder(self, num_layers=18, pretrained=True):
        """
        Initialize ResNet encoder for processing image pairs
        """
        self.resnet_encoder = ResnetEncoder(num_layers, pretrained)
    
        # Add projection layers to match dimensions with dec_blocks_pc
        self.resnet_proj_layers = nn.ModuleList()
    
        # Get the channel dimensions from ResNet
        resnet_channels = self.resnet_encoder.num_ch_enc
    
        # Create projection layers for each ResNet feature level
        for ch in resnet_channels:
            self.resnet_proj_layers.append(
                nn.Sequential(
                    nn.Conv2d(ch, self.dec_embed_dim, kernel_size=1),
                    nn.GroupNorm(1, self.dec_embed_dim),
                    nn.GELU()
                )
            )
        
        # Create spatial adapters for each ResNet level
        self.spatial_adapters = nn.ModuleList()
        for i in range(len(resnet_channels)):
            self.spatial_adapters.append(
                nn.Sequential(
                    nn.Conv2d(self.dec_embed_dim, self.dec_embed_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(1, self.dec_embed_dim),
                    nn.GELU(),
                    nn.AdaptiveAvgPool2d((14, 14))  # Adjust to match transformer's spatial dimensions (14x14)
                )
            )
    
        
        
    
    def make_zero_conv(self, channels):
        # return nn.Sequential(zero_module(conv_nd(1, channels, channels, 1, padding=0)))
        return nn.Sequential(conv_nd(1, channels, channels, 1, padding=0))
    # use normal conv instead of zero_module
    def make_normal_conv(self, channels):
        return nn.Sequential(conv_nd(1, channels, channels, 1, padding=0))

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, use_pose_head, patch_size, img_size,
                            **kw):
        if type(img_size) is int:
            img_size = (img_size, img_size)
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.use_pose_head = use_pose_head
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        
        # Only create pose head if use_pose_head is True
        if use_pose_head:
            self.downstream_pose_head1 = head_factory('pose', 'pose', self, has_conf=bool(conf_mode))
        
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)
        
        # pose head wrapper only if enabled
        if use_pose_head:
            self.head3 = transpose_to_landscape(self.downstream_pose_head1, activate=landscape_only)

    def process_resnet_features(self, view1, view2):
        """
        Process image pairs through ResNet encoder and prepare features
        for integration with dec_blocks_pc
        """
        # Get ResNet features for both images
        features1 = self.resnet_encoder(view1['img'])
        features2 = self.resnet_encoder(view2['img'])
        
        # Project features to match dec_embed_dim
        projected_features1 = []
        projected_features2 = []
    
        for i, (feat1, feat2) in enumerate(zip(features1, features2)):
            # Project features
            proj_feat1 = self.resnet_proj_layers[i](feat1)  # B, C, H, W
            proj_feat2 = self.resnet_proj_layers[i](feat2)  # B, C, H, W
            
            # Apply spatial adapter to match transformer dimensions
            proj_feat1 = self.spatial_adapters[i](proj_feat1)  # B, C, 14, 14
            proj_feat2 = self.spatial_adapters[i](proj_feat2)  # B, C, 14, 14
            
            # Reshape to match transformer format (B, N, C)
            B, C, H, W = proj_feat1.shape
            proj_feat1 = proj_feat1.flatten(2).transpose(1, 2)  # B, H*W, C
            proj_feat2 = proj_feat2.flatten(2).transpose(1, 2)  # B, H*W, C
            
            projected_features1.append(proj_feat1)
            projected_features2.append(proj_feat2)
    
        # Return the projected features
        return projected_features1, projected_features2

    
    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)
        # x (B, 576, 1024) pos (B, 576, 2); patch_size=16
        B,N,C = x.size()
        posvis = pos
        # add positional embedding without cls token
        assert self.enc_pos_embed is None
        # TODO: where to add mask for the patches
        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, posvis)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]

        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))

        # warning! maybe the images have different portrait/landscape orientations
        if is_symmetrized(view1, view2):
            # print("is_symmetrized")
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            # print("not is_symmetrized")
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2, resnet_features1, resnet_features2):
        final_output = [(f1, f2)]  # before projection
        original_D = f1.shape[-1]

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)
        
        # Add ResNet features - now shapes should match due to spatial adapters
        f1 = f1 + self.zero_convs[0](resnet_features1[0].transpose(-1, -2)).transpose(-1, -2)
        f2 = f2 + self.zero_convs[0](resnet_features2[0].transpose(-1, -2)).transpose(-1, -2)

        final_output.append((f1, f2))
        for i in range(len(self.dec_blocks)):
            blk1 = self.dec_blocks[i]
            blk2 = self.dec_blocks2[i]
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            
            # incorporate point maps
            if i < len(self.resnet_proj_layers):
                # Get the corresponding ResNet feature level (cycling through available levels)
                resnet_idx = min(i + 1, len(resnet_features1) - 1)
            
                # Apply zero_convs to incorporate ResNet features
                f1 = f1 + self.zero_convs[i+1](resnet_features1[resnet_idx].transpose(-1, -2)).transpose(-1, -2)
                f2 = f2 + self.zero_convs[i+1](resnet_features2[resnet_idx].transpose(-1, -2)).transpose(-1, -2)
            
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self, view1, view2, return_pose=False):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)
        resnet_features1, resnet_features2 = self.process_resnet_features(view1, view2)
        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2, resnet_features1, resnet_features2)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)
            
            # Only compute pose if use_pose_head is True and return_pose is True
            res_pose1 = None
            if self.use_pose_head and return_pose:
                res_pose1 = self._downstream_head(3, [tok.float() for tok in dec1], shape2)
        
        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        if self.use_pose_head and return_pose:
            return res1, res2, res_pose1
        else:
            return res1, res2
    
    def freeze_all_except_lora(self):
        
        # For parameter about resnet, set them to True
        for name, param in self.named_parameters():
            if 'resnet' in name:
                param.requires_grad = True
            if 'lora_' in name:
                param.requires_grad = True
            if 'zero_convs' in name:
                param.requires_grad = True
                
    def enable_lora_finetuning(self, rank=4, alpha=1.0, dropout=0.0):
        # """Enable LoRA finetuning by adding LoRA to attention layers and freezing other parameters"""
        # First freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        # self.set_freeze('encoder_and_decoder')
        # Add LoRA to encoder attention blocks
        for block in self.enc_blocks:
            block.attn.add_lora(rank, alpha, dropout)
            
        # Add LoRA to decoder attention blocks
        for block in self.dec_blocks:
            block.attn.add_lora(rank, alpha, dropout)
            block.cross_attn.add_lora(rank, alpha, dropout)
            
        # Add LoRA to second decoder attention blocks
        for block in self.dec_blocks2:
            block.attn.add_lora(rank, alpha, dropout)
            block.cross_attn.add_lora(rank, alpha, dropout)
            
        # Only enable training for LoRA parameters
        for name, param in self.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
            if 'resnet' in name:
                param.requires_grad = True
            if 'zero_convs' in name:
                param.requires_grad = True
                
        print(f"Enabled LoRA finetuning with rank={rank}, alpha={alpha}, dropout={dropout}")
        print("Number of trainable parameters:", sum(p.numel() for p in self.parameters() if p.requires_grad))
        # print the percentage of trainable parameters
        print("Percentage of trainable parameters:", sum(p.numel() for p in self.parameters() if p.requires_grad) / sum(p.numel() for p in self.parameters()) * 100)
