# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#  Edited: Yuekun Dai, Siyao Li
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import argparse
import math
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import torchvision
from copy import deepcopy
from torch import nn
from torch.nn import init
from torch_scatter import scatter as super_pixel_pooling

from basicsr.utils.registry import ARCH_REGISTRY
from raft.raft import RAFT


def flow_warp(x, flow, interpolation="bilinear", padding_mode="zeros", align_corners=True):
    """Warp an image or a feature map with optical flow.
    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.
    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f"The spatial sizes of input ({x.size()[-2:]}) and " f"flow ({flow.size()[1:3]}) are not the same.")
    _, _, h, w = x.size()
    # create mesh grid
    device = flow.device
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, device=device), torch.arange(0, w, device=device))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (w, h, 2)
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(x, grid_flow, mode=interpolation, padding_mode=padding_mode, align_corners=align_corners)
    return output


def MLP(channels: list, do_bn=True):
    """Multi-layer perceptron"""
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * width, one * height, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class CLIPEncoder(nn.Module):
    def __init__(self, freeze=True, resolution=None):
        super().__init__()
        clip_encoder, _, self.preprocess = open_clip.create_model_and_transforms("convnext_large_d_320")
        # We just assume preprocess is the same as our proprocess
        visual_model = clip_encoder.visual
        visual_model = list(visual_model.children())[0].children()
        visual_model0, visual_model1 = list(visual_model)[:-2]
        self.visual_encoder = torch.nn.Sequential(visual_model0, *list(visual_model1)[:-2])
        # For the visual encoder, [:-3] means [192,80,80] (default), [:-2] means [384,40,40] and [:-1] means [768,20,20]
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
        self.resolution = resolution

    def forward(self, x):
        if self.resolution:
            x = F.interpolate(x, self.resolution, mode="bilinear", align_corners=False)
        x = self.visual_encoder(x)
        return x.detach()


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet_offset(nn.Module):
    def __init__(self, ch_in, enc_dim):
        super(UNet_offset, self).__init__()
        # estimata the offset for the

        self.ch_in = ch_in

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=self.ch_in, ch_out=16)
        self.Conv2 = conv_block(ch_in=16, ch_out=32)
        self.Conv3 = conv_block(ch_in=32, ch_out=64)
        self.Conv4 = conv_block(ch_in=64, ch_out=128)

        self.Up4 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv4 = conv_block(ch_in=128, ch_out=64)

        self.Up3 = up_conv(ch_in=64, ch_out=32)
        self.Up_conv3 = conv_block(ch_in=64, ch_out=32)

        self.Up2 = up_conv(ch_in=32, ch_out=16)
        self.Up_conv2 = conv_block(ch_in=32, ch_out=16)

        self.Conv_1x1 = nn.Conv2d(16, enc_dim, kernel_size=1, stride=1, padding=0)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        init_type = "normal"
        gain = 0.02
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    def forward(self, x):

        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)  # 80 * 80

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)  # 40 * 40

        # decoding + concat path

        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        return d1


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, mask=False):
        super(DeformableConv2d, self).__init__()

        self.padding = padding
        self.mask = mask
        self.in_channels = in_channels

        self.channel_num = 3 * kernel_size * kernel_size if mask else 2 * kernel_size * kernel_size
        self.offset_unet = UNet_offset(in_channels, self.channel_num)

        # 수정된 부분: 입력 채널에 따른 조건부 처리
        if in_channels == 6:
            self.line_conv = nn.Conv2d(int(in_channels/2), int(out_channels/2), kernel_size, stride, padding=self.padding, bias=bias)
            self.color_conv = nn.Conv2d(int(in_channels/2), int(out_channels/2), kernel_size, stride, padding=self.padding, bias=bias)
        else:  # in_channels == 3
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=self.padding, bias=bias)

        self.out_conv = nn.Sequential(
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, use_offset=False):
        if self.in_channels == 6:
            # 6채널 입력 처리 (기존 로직)
            c, h, w = x.shape[1:]
            if c != 6:
                assert False, "input tensor should has 6 channels with 3 as line art and 3 as color."
            max_offset = max(h, w) / 4.0
            line_fea = x[:, 3:, :, :]
            color_fea = x[:, :3, :, :]

            if use_offset:
                out = self.offset_unet(x)
                offset = out
                mask = None
                offset = offset.clamp(-max_offset, max_offset)
                color_fea = torchvision.ops.deform_conv2d(
                    input=color_fea, offset=offset, weight=self.color_conv.weight, 
                    bias=self.color_conv.bias, padding=self.padding, mask=mask
                )
            else:
                color_fea = self.color_conv(color_fea)
            line_fea = self.line_conv(line_fea)
            x_out = torch.cat((color_fea, line_fea), dim=1)
            
        else:  # 3채널 입력 처리
            if use_offset:
                out = self.offset_unet(x)
                offset = out
                mask = None
                h, w = x.shape[2:]
                max_offset = max(h, w) / 4.0
                offset = offset.clamp(-max_offset, max_offset)
                x_out = torchvision.ops.deform_conv2d(
                    input=x, offset=offset, weight=self.conv.weight, 
                    bias=self.conv.bias, padding=self.padding, mask=mask
                )
            else:
                x_out = self.conv(x)

        x_out = self.out_conv(x_out)
        return x_out


class UNet(nn.Module):
    def __init__(self, enc_dim, ch_in=3, use_clip=False, clip_resolution=None):
        super(UNet, self).__init__()

        self.use_clip = use_clip
        self.ch_in = ch_in

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.DCN1 = DeformableConv2d(in_channels=self.ch_in, out_channels=64)
        # conv_block(ch_in=self.ch_in,ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)

        if use_clip:
            self.clip = CLIPEncoder(freeze=True, resolution=clip_resolution)
            self.fuse = nn.Sequential(
                nn.Conv2d(512 + 384, 512, kernel_size=3, padding=1, bias=False),
                nn.PReLU(),
            )

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, enc_dim, kernel_size=1, stride=1, padding=0)
        self.norm = nn.Sequential(nn.InstanceNorm2d(enc_dim))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        init_type = "normal"
        gain = 0.02
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    def forward(self, x, line=None, use_offset=False):
        if line is None:
            line = x

        x = (x - 0.5) * 2
        line = (line - 0.5) * 2  # To ensure the avg of the offset is 0
        # encoding path
        x1 = self.DCN1(x, use_offset)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)  # 80 * 80

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)  # 40 * 40

        if self.use_clip:
            x_clip_fea = self.clip(line)
            x_clip_fea_resized = F.interpolate(x_clip_fea, size=(x4.shape[2], x4.shape[3]), mode="bilinear", align_corners=False)
            x4 = self.fuse(torch.cat((x_clip_fea_resized, x4), dim=1))

        # decoding + concat path

        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.norm(d1)
        return d1


class SegmentDescriptor(nn.Module):
    """Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, enc_dim, ch_in=3, use_clip=False, encoder_resolution=None, clip_resolution=None):
        super().__init__()
        self.encoder_resolution = encoder_resolution
        self.encoder = UNet(enc_dim, ch_in, use_clip, clip_resolution)

    def forward(self, img, seg, line=None, use_offset=False):
        h, w = img.size()[-2:]
        if self.encoder_resolution:
            img = F.interpolate(img, self.encoder_resolution, mode="bilinear", align_corners=False)
            line = F.interpolate(line, self.encoder_resolution, mode="nearest") if line is not None else None
        x = self.encoder(img, line, use_offset)
        x = F.interpolate(x, (h, w), mode="bilinear", align_corners=False)
        n, c, nh, nw = x.size()
        return super_pixel_pooling(x.view(n, c, -1), seg.view(-1).long(), reduce="mean")
        # here return size is [1]xCx|Seg|


class KeypointEncoder(nn.Module):
    """Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([4] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts):
        inputs = kpts.transpose(1, 2)
        x = self.encoder(inputs)
        return x


def attention(query, key, value, mask=None):
    dim = query.shape[1]
    scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / dim**0.5
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum("bhnm,bdhm->bdhn", prob, value), prob


class MultiHeadedAttention(nn.Module):
    """Multi-head attention to increase model expressivitiy"""

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value, mask=None):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1) for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value, mask)
        self.prob.append(prob)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source, mask=None):
        message = self.attn(x, source, source, mask)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([AttentionalPropagation(feature_dim, 4) for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            layer.attn.prob = []
            if name == "cross":
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def transport(scores, alpha):
    """Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    # pad additional scores for unmatcheed (to -1)
    # alpha is the learned threshold
    couplings = torch.cat([torch.cat([scores, bins0], -1), torch.cat([bins1, alpha], -1)], 1)

    return couplings


@ARCH_REGISTRY.register()
class BasicPBC(nn.Module):
    """SuperGlue feature matching middle-end. A new hard-coded self-attention will be added to the transformer.
    This part is an AnT module with the hard coded transformer.

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

    This version also adds a hard-coded transformer.

    """

    def __init__(
        self,
        ch_in=6,
        descriptor_dim=128,
        keypoint_encoder=[32, 64, 128],
        GNN_layer_num=9,
        num_target_frames=12
    ):
        super().__init__()

        config = argparse.Namespace()
        config.ch_in = ch_in
        config.descriptor_dim = descriptor_dim
        config.keypoint_encoder = keypoint_encoder
        config.GNN_layers_num = GNN_layer_num
        config.GNN_layers = ["self", "cross"] * GNN_layer_num
        config.num_target_frames = num_target_frames

        self.config = config

        self.kenc = KeypointEncoder(self.config.descriptor_dim, self.config.keypoint_encoder)
        self.gnn = AttentionalGNN(self.config.descriptor_dim, self.config.GNN_layers)
        self.final_proj = nn.Conv1d(self.config.descriptor_dim, self.config.descriptor_dim, kernel_size=1, bias=True)

        # Add temporal encoding
        self.temporal_encoding = nn.Parameter(
            torch.zeros(self.config.num_target_frames, self.config.descriptor_dim)
        )
        self._init_temporal_encoding()

        bin_score = torch.nn.Parameter(torch.tensor(1.0))
        self.register_parameter("bin_score", bin_score)
        
        self.target_segment_desc = SegmentDescriptor(
            self.config.descriptor_dim, 
            3
        )
        self.ref_segment_desc = SegmentDescriptor(
            self.config.descriptor_dim, 
            6
        )

    def _init_temporal_encoding(self):
        position = torch.arange(self.config.num_target_frames).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.config.descriptor_dim, 2) * 
            (-math.log(10000.0) / self.config.descriptor_dim)
        )
        
        # Create a new tensor for the temporal encoding
        temporal_encoding = torch.zeros(self.config.num_target_frames, self.config.descriptor_dim)
        temporal_encoding[:, 0::2] = torch.sin(position * div_term)
        temporal_encoding[:, 1::2] = torch.cos(position * div_term)
        
        # Register the new tensor as a parameter
        self.temporal_encoding = nn.Parameter(temporal_encoding)

    def forward(self, data):
        """Run matching between one reference frame and multiple target frames"""
        
        # Process reference frame
        kpts_ref = data["keypoints_ref"].float()
        ref_img = torch.cat((data["recolorized_img"], data["line_ref"]), dim=1)
        desc_ref = self.ref_segment_desc(ref_img, data["segment_ref"])
        desc_ref = desc_ref[..., 1:]  # [B, C, N_ref]
        pos_ref = self.kenc(normalize_keypoints(kpts_ref, data["line_ref"].shape))
        desc_ref = desc_ref + pos_ref

        # Process all target frames together
        B = desc_ref.shape[0]
        all_desc = []
        
        for i in range(len(data["line_list"])):
            desc_i = self.target_segment_desc(data["line_list"][i], data["segment_list"][i])
            desc_i = desc_i[..., 1:]  # [B, C, N_i]
            pos_i = self.kenc(normalize_keypoints(data["keypoints_list"][i], data["line_list"][i].shape))
            desc_i = desc_i + pos_i
            
            # Add temporal encoding
            temporal_code = self.temporal_encoding[i:i+1]  # [1, D]
            temporal_code = temporal_code.unsqueeze(0).expand(B, -1, -1)  # [B, 1, D]
            temporal_code = temporal_code.transpose(1, 2)  # [B, D, 1]
            desc_i = desc_i + temporal_code.expand(-1, -1, desc_i.size(2))  # [B, D, N]
            
            all_desc.append(desc_i)

        # Concatenate all descriptors along the sequence dimension
        all_desc = torch.cat(all_desc, dim=2)  # [B, C, T*N]
        
        # Skip if no keypoints
        if all_desc.shape[2] < 2 or desc_ref.shape[2] < 2:
            print(f"No keypoints in {data['file_name'][0]}")
            return {
                "matches0": torch.full((0,), -1, dtype=torch.int, device=desc_ref.device),
                "matching_scores0": torch.zeros(0, device=desc_ref.device),
                "skip_train": True,
            }

        # Multi-layer Transformer network
        desc_src, desc_ref = self.gnn(all_desc, desc_ref)

        # Final MLP projection
        mdesc = self.final_proj(desc_src)  # [B, C, T*N]
        mdesc_ref = self.final_proj(desc_ref)  # [B, C, N_ref]

        # Compute matching scores
        scores = torch.einsum("bdn,bdm->bnm", mdesc, mdesc_ref)  # [B, T*N, N_ref]
        scores = scores / (self.config.descriptor_dim ** 0.5)

        # Run optimal transport
        scores = transport(scores, self.bin_score)  # [B, T*N, N_ref+1]

        # Process all matches
        all_matches = torch.cat([match for match in data["all_matches_list"]], dim=1) if "all_matches_list" in data else None
        all_matches_origin = all_matches.clone() if all_matches is not None else None

        if all_matches is not None:
            n = scores.shape[2] - 1  # N_ref
            all_matches[all_matches == -1] = n
            loss = nn.functional.cross_entropy(
                scores[:, :-1, :].reshape(-1, n + 1),
                all_matches.long().reshape(-1),
                reduction="mean"
            )

        scores = nn.functional.softmax(scores, dim=2)
        max0, max1 = scores[:, :-1, :].max(2), scores[:, :, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mscores0 = max0.values

        valid0 = indices0 < scores.shape[2] - 1
        valid1 = indices1 < scores.shape[1]
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
        
        if all_matches is None:
            return {
                "match_scores": scores[:, :-1, :][0],
                "matches0": indices0[0],  # use -1 for invalid match
                "matching_scores0": mscores0[0],
                "loss": -1,
                "skip_train": True,
                "accuracy": -1,
                "area_accuracy": -1,
                "valid_accuracy": -1,
                "invalid_accuracy": -1,
            }
        else:
            is_correct = all_matches_origin[0] == indices0[0]
            accuracy = (is_correct.sum() / len(all_matches_origin[0])).item()
            
            correct_indices = torch.arange(len(all_matches_origin[0]), device=is_correct.device)[is_correct]
            
            # Calculate total area accuracy across all frames
            total_area_correct = 0
            total_pixels = 0
            
            # 각 프레임별 세그먼트 수 계산
            segments_per_frame = [seg.max().item() for seg in data["segment_list"]]
            cumsum_segments = [0] + list(np.cumsum(segments_per_frame))
            
            for i, idx in enumerate(correct_indices):
                # 프레임 인덱스 찾기
                frame_idx = 0
                for j in range(len(cumsum_segments)-1):
                    if cumsum_segments[j] <= idx < cumsum_segments[j+1]:
                        frame_idx = j
                        break
                
                # 해당 프레임 내에서의 세그먼트 인덱스 계산
                segment_idx = idx - cumsum_segments[frame_idx]
                
                # 현재 프레임의 세그먼트 맵
                current_seg_map = data["segment_list"][frame_idx]
                
                # 해당 세그먼트의 픽셀 수 계산
                # segment_idx는 0부터 시작하므로 +1
                segment_mask = (current_seg_map == segment_idx + 1)
                total_area_correct += segment_mask.sum()
                total_pixels += current_seg_map.numel()  # 전체 픽셀 수
            
            area_accuracy = (total_area_correct / total_pixels).item()
            
            is_valid = all_matches_origin[0] != -1
            valid_accuracy = ((is_correct & is_valid).sum() / is_valid.sum()).item()
            invalid_accuracy = ((is_correct & ~is_valid).sum() / (~is_valid).sum()).item() if (~is_valid).sum() > 0 else None

            return {
                "match_scores": scores[:, :-1, :][0],
                "matches0": indices0[0],  # use -1 for invalid match
                "matching_scores0": mscores0[0],
                "loss": loss,
                "skip_train": False,
                "accuracy": accuracy,
                "area_accuracy": area_accuracy,
                "valid_accuracy": valid_accuracy,
                "invalid_accuracy": invalid_accuracy,
            }
