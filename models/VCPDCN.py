import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
import math
import torch.nn.functional as F
from einops import rearrange
from models.SwinTransformers import SwinTransformer
from models.pvtv2 import pvt_v2_b4

from models.encoders.dual_vmamba import vssm_small
from models.encoders.dual_vmamba import vssm_base
from models.MLA import MLA_head
from models.DeformableConv import deformable_LKA, deformable_LKA_Attention, DeformConv

from models.SwinTransformers import SwinTransformer

from loss import IOU, SSIM, SimMaxLoss, SimMinLoss, SimMaxLoss_Patch


def conv3x3_bn_relu(in_planes, out_planes, k=3, s=1, p=1, b=False):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=s, padding=p, bias=b),
        nn.BatchNorm2d(out_planes),
        nn.GELU(),
    )


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, groups=groups, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class SEModule(nn.Module):
    def __init__(self, dim, red=8, inner_act=nn.GELU, out_act=nn.Sigmoid):
        super().__init__()
        inner_dim = max(16, dim // red)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            inner_act(),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            out_act(),
        )

    def forward(self, x):
        x = x * self.proj(x)
        return x


class CBRConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CBRConv, self).__init__()
        self.SE = SEModule(dim=in_channel * 2)
        self.conv = nn.Conv2d(in_channel * 2, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.dilateconv = BasicConv2d(out_channel, out_channel, 3, padding=3, groups=out_channel, dilation=3)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x, y):
        out = self.SE(torch.cat((x, y), dim=1))
        out = self.bn(self.conv(out))
        return out + self.dilateconv(out)


# Global Contextual module
class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel // 4, in_channel // 4, 1),
            BasicConv2d(in_channel // 4, in_channel // 4, kernel_size=(1, 9), groups=in_channel // 4, padding=(0, 4)),
            BasicConv2d(in_channel // 4, in_channel // 4, kernel_size=(9, 1), groups=in_channel // 4, padding=(4, 0)),
            BasicConv2d(in_channel // 4, in_channel // 4, 3, padding=9, groups=in_channel // 4, dilation=9),
            SEModule(in_channel // 4)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel // 4, in_channel // 4, 1),
            BasicConv2d(in_channel // 4, in_channel // 4, kernel_size=(1, 3), groups=in_channel // 4, padding=(0, 1)),
            BasicConv2d(in_channel // 4, in_channel // 4, kernel_size=(3, 1), groups=in_channel // 4, padding=(1, 0)),
            BasicConv2d(in_channel // 4, in_channel // 4, 3, padding=3, groups=in_channel // 4, dilation=3),
            SEModule(in_channel // 4)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel // 4, in_channel // 4, 1),
            BasicConv2d(in_channel // 4, in_channel // 4, kernel_size=(1, 5), groups=in_channel // 4, padding=(0, 2)),
            BasicConv2d(in_channel // 4, in_channel // 4, kernel_size=(5, 1), groups=in_channel // 4, padding=(2, 0)),
            BasicConv2d(in_channel // 4, in_channel // 4, 3, padding=5, groups=in_channel // 4, dilation=5),
            SEModule(in_channel // 4)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel // 4, in_channel // 4, 1),
            BasicConv2d(in_channel // 4, in_channel // 4, kernel_size=(1, 7), groups=in_channel // 4, padding=(0, 3)),
            BasicConv2d(in_channel // 4, in_channel // 4, kernel_size=(7, 1), groups=in_channel // 4, padding=(3, 0)),
            BasicConv2d(in_channel // 4, in_channel // 4, 3, padding=7, groups=in_channel // 4, dilation=7),
            SEModule(in_channel // 4)
        )
        self.conv_cat = BasicConv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0, x1, x2, x3 = torch.chunk(x, 4, dim=1)
        x0 = self.branch0(x0)
        x1 = self.branch1(x1)
        x2 = self.branch2(x2)
        x3 = self.branch3(x3)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        return x

class ModalityFusion(nn.Module):
    def __init__(self, dim):
        super(ModalityFusion, self).__init__()

        self.attn = DeformConv(in_channels=dim, groups=dim)


        self.conv_fusion = nn.Sequential(
            nn.Conv2d(in_channels=dim * 3, out_channels=dim, kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb, depth, t0, t1, t2):
        b, c, h, w = rgb.shape
        t0_ = t0.unsqueeze(1)
        t1_ = t1.unsqueeze(-1)  # Batch, 1, channel
        t2_ = t2.unsqueeze(-1)  # Batch, channel, 1

        sim_rgb = t0_ @ t1_  # Batch, channel, 1
        sim_rgb = sim_rgb.view(b, 1, 1, 1)

        sim_depth = t0_ @ t2_  # Batch, channel, 1
        sim_depth = sim_depth.view(b, 1, 1, 1)



        rgbd = sim_rgb * rgb + sim_depth * depth
        rgb = (1-sim_rgb) * rgb
        depth = (1-sim_depth) * depth

        rgbd_fusion = self.conv_fusion(torch.cat([rgbd, rgb, depth], dim=1))
        
        out = self.attn(rgbd_fusion)

        return out

class ModalityInteraction(nn.Module):
    def __init__(self, dim):
        super(ModalityInteraction, self).__init__()

        self.mla_r = MLA_head(dim=dim)
        self.mla_d = MLA_head(dim=dim)

        self.mla_r_b = MLA_head(dim=dim)
        self.mla_d_b = MLA_head(dim=dim)

        self.conv_rgb = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1,
                      padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

        self.conv_depth = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1,
                      padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb, depth, tf1, tf2, tb1, tb2, temp=0.4):
        # t1: foreground rgbd token, Batch, channel
        # t2: background rgb token, Batch, channel
        # t3: background depth token, Batch, channel

        # Soft Normalization to combine foreground and background mask

        # RGB: Batch, channel, h, w
        b, c, h, w = rgb.shape

        # RGB branch
        tf1 = tf1.unsqueeze(1)  # Batch, 1, channel
        mask_fore_rgb = tf1 @ rgb.view(b, c, -1)
        mask_fore_rgb = mask_fore_rgb.view(b, 1, h, w)  # B, 1, H, W

        tb1 = tb1.unsqueeze(1)  # Batch, 1, channel
        mask_back_rgb = tb1 @ rgb.view(b, c, -1)  # B, 1, H, W
        mask_back_rgb = mask_back_rgb.view(b, 1, h, w)



        # Depth branch
        tf2 = tf2.unsqueeze(1)  # Batch, 1, channel
        mask_fore_depth = tf2 @ depth.view(b, c, -1)
        mask_fore_depth = mask_fore_depth.view(b, 1, h, w)  # B, 1, H, W

        tb2 = tb2.unsqueeze(1)  # Batch, 1, channel
        mask_back_depth = tb2 @ depth.view(b, c, -1)
        mask_back_depth = mask_back_depth.view(b, 1, h, w)  # B, 1, H, W

        rgb_f = self.mla_r(rgb, mask_fore_rgb, mask_fore_depth)
        rgb_b = self.mla_r_b(rgb, mask_back_rgb, mask_back_depth)
        rgb = self.conv_rgb(rgb + rgb_b + rgb_f)

        depth_f = self.mla_d(depth, mask_fore_depth, mask_fore_rgb)
        depth_b = self.mla_d_b(depth, mask_back_rgb, mask_back_depth)
        depth = self.conv_depth(depth + depth_f + depth_b)


        return rgb, depth

class ModalityAligh(nn.Module):
    def __init__(self, dim, out_act=nn.Sigmoid):
        super(ModalityAligh, self).__init__()
        self.proj_rgb = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=1, kernel_size=1, stride=1,
                      padding=0, bias=False),
            out_act(),
        )

        self.proj_d = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=1, kernel_size=1, stride=1,
                      padding=0, bias=False),
            out_act(),
        )

        self.conv_rgb = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1,
                      padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

        self.conv_d = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1,
                      padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

        self.avg = nn.AdaptiveAvgPool2d(1)

        self.pro_fore_r_1 = Mlp(dim)  # 1: modality-shared features
        self.pro_fore_r_2 = Mlp(dim)  # 2: modality-specific features

        self.pro_back_r = Mlp(dim)  # 1: modality-shared features

        self.pro_fore_d_1 = Mlp(dim)  # 1: modality-shared features
        self.pro_fore_d_2 = Mlp(dim)  # 2: modality-specific features

        self.pro_back_d = Mlp(dim)  # 1: modality-shared features

        # self.inter = Interaction(dim=dim)

        self.minloss = SimMinLoss(metric='cos').cuda()  # 拉远
        self.maxloss = SimMaxLoss_Patch(metric='cos', alpha=0.25).cuda()  # 拉近

    def forward(self, x, y):
        # RGB and Depth Feature Embedding
        mask_fore_r = self.proj_rgb(x)
        mask_back_r = 1 - mask_fore_r

        fea_fore_r = self.avg(x * mask_fore_r).squeeze(-1).squeeze(-1)  # Batch, channel
        fea_back_r = self.avg(x * mask_back_r).squeeze(-1).squeeze(-1)  # Batch, channel

        fea_fore_r_1 = self.pro_fore_r_1(fea_fore_r.unsqueeze(1)).squeeze(1)  # modality-shared features
        fea_fore_r_2 = self.pro_fore_r_2(fea_fore_r.unsqueeze(1)).squeeze(1)  # modality-specific features

        fea_back_r = self.pro_back_r(fea_back_r.unsqueeze(1)).squeeze(1)

        mask_fore_d = self.proj_d(y)
        mask_back_d = 1 - mask_fore_d

        fea_fore_d = self.avg(y * mask_fore_d).squeeze(-1).squeeze(-1)  # Batch, channel
        fea_back_d = self.avg(y * mask_back_d).squeeze(-1).squeeze(-1)  # Batch, channel

        fea_fore_d_1 = self.pro_fore_d_1(fea_fore_d.unsqueeze(1)).squeeze(1)  # modality-shared features
        fea_fore_d_2 = self.pro_fore_d_2(fea_fore_d.unsqueeze(1)).squeeze(1)  # modality-specific features

        fea_back_d = self.pro_back_d(fea_back_d.unsqueeze(1)).squeeze(1)

        # RGB and Depth feature interaction
        # x, y = self.inter(x, y, fea_fore_r_1, fea_back_r, fea_fore_d, fea_back_d)

        loss_1 = self.maxloss(fea_fore_r_1, fea_fore_d_1)  # modality-shared features loss
        loss_2 = self.minloss(fea_fore_r_2, fea_fore_d_2)  # modality-specific features loss
        loss_3 = self.maxloss(fea_back_r, fea_back_d)  # background features loss

        loss_4 = self.minloss(fea_fore_r_1, fea_back_r) + self.minloss(fea_fore_r_2,
                                                                       fea_back_r)  # RGB background-foreground loss
        loss_5 = self.minloss(fea_fore_d_1, fea_back_d) + self.minloss(fea_fore_d_2,
                                                                       fea_back_d)  # Depth background-foreground loss

        loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5

        token_r_list = [fea_fore_r_1, fea_fore_r_2, fea_back_r]
        token_d_list = [fea_fore_d_1, fea_fore_d_2, fea_back_d]

        return x, y, token_r_list, token_d_list, loss

class DCNLayer(nn.Module):
    def __init__(self, dim):
        super(DCNLayer, self).__init__()

        self.align = ModalityAligh(dim=dim)
        self.inter = ModalityInteraction(dim=dim)
        self.fuse = ModalityFusion(dim=dim)

    def forward(self, x, y):
        x, y, token_r_list, token_d_list, loss = self.align(x, y)
        token_fore_r_1, token_fore_r_2, token_back_r = token_r_list
        token_fore_d_1, token_fore_d_2, token_back_d = token_d_list

        x, y = self.inter(x, y, token_fore_r_1, token_fore_d_1, token_back_r, token_back_d)

        token_fore_1 = (token_fore_r_1 + token_fore_d_1) / 2
        rgbd = self.fuse(x, y, token_fore_1, token_fore_r_2, token_fore_d_2)

        return rgbd, x, y, loss

class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = super().forward(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x.contiguous()

class Decoder(nn.Module):
    def __init__(self, channels=[96, 192, 384, 768]):
        super(Decoder, self).__init__()

        self.channel = channels
        self.scale_trans2 = nn.Conv2d(in_channels=self.channel[1], out_channels=self.channel[0], kernel_size=1,
                                      stride=1,
                                      padding=0, bias=False)
        self.scale_trans3 = nn.Conv2d(in_channels=self.channel[2], out_channels=self.channel[1], kernel_size=1,
                                      stride=1,
                                      padding=0, bias=False)
        self.scale_trans4 = nn.Conv2d(in_channels=self.channel[3], out_channels=self.channel[2], kernel_size=1,
                                      stride=1,
                                      padding=0, bias=False)

        self.fdconv4 = GCM(in_channel=self.channel[3], out_channel=self.channel[3])
        self.fdconv3 = GCM(in_channel=self.channel[2], out_channel=self.channel[2])
        self.fdconv2 = GCM(in_channel=self.channel[1], out_channel=self.channel[1])
        self.fdconv1 = GCM(in_channel=self.channel[0], out_channel=self.channel[0])

        self.conv_pre = nn.Conv2d(in_channels=self.channel[0], out_channels=1, kernel_size=1, stride=1,
                                  padding=0, bias=False)

        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, fea1, fea2, fea3, fea4):
        fea4 = self.fdconv4(fea4)

        fea3 = self.scale_trans4(self.up2(fea4)) + fea3  # B, 384, 24, 24
        fea3 = self.fdconv3(fea3)

        fea2 = self.scale_trans3(self.up2(fea3)) + fea2  # B, 384, 24, 24
        fea2 = self.fdconv2(fea2)

        fea1 = self.scale_trans2(self.up2(fea2)) + fea1  # B, 384, 24, 24
        fea1 = self.fdconv1(fea1)

        mask = self.conv_pre(fea1)

        return self.up4(mask)

class VFMambaNet(nn.Module):
    def __init__(self, channels=[96, 192, 384, 768], embed_dim=384, type='VFMamba-S'):
        super(VFMambaNet, self).__init__()

        if type == 'VFMamba-S':
            self.rgbd_mobile = vssm_small()
            channels = [96, 192, 384, 768]
            embed_dim = 384
            self.channel = channels  # VMamba-Ting, -Samll 96, 192, 384, 768, : -Base 128, 256, 512, 1024
            self.embed_dim = embed_dim
            self.depth_layers = [2, 2, 2, 2]
        elif type == 'VFMamba-B':
            self.rgbd_mobile = vssm_base()
            channels = [128, 256, 512, 1024]
            embed_dim = 512
            self.channel = channels  # VMamba-Ting, -Samll 96, 192, 384, 768, : -Base 128, 256, 512, 1024
            self.embed_dim = embed_dim
            self.depth_layers = [2, 2, 2, 2]
        else:
            print("The VFMamba Type is wrong! Must is VFMamba-S or VFMamba-B")

        # # Swin Transformer as backbone
        # self.rgbd_mobile = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
        #                                    window_size=12)
        # pre_model = '/your weight path/swin_base_patch4_window12_384_22k.pth'
        # self.rgbd_mobile.load_state_dict(torch.load(pre_model)['model'], strict=False)
        # print(f"RGB +Depth SwinTransformer loading pre_model ${pre_model}")
        # self.channel = [128, 256, 512, 1024]
        # self.embed_dim = 512

        # # PVT v2 as backbone
        # self.rgbd_mobile = pvt_v2_b4()
        # path = '/media/gpu/dss/code/pre_model_pth/PVT-V2/pvt_v2_b4.pth'
        # save_model = torch.load(path)
        # model_dict = self.rgbd_mobile.state_dict()
        # state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        # model_dict.update(state_dict)
        # self.rgbd_mobile.load_state_dict(model_dict)
        # print("loading successful pvt_v2_b4.pth !")
        # self.channel = [64, 128, 320, 512]
        # self.embed_dim = 320

        self.relu = nn.ReLU()

        self.dcn4 = DCNLayer(dim=self.channel[3])
        self.dcn3 = DCNLayer(dim=self.channel[2])
        self.dcn2 = DCNLayer(dim=self.channel[1])
        self.dcn1 = DCNLayer(dim=self.channel[0])

        self.decoder_r = Decoder(channels=self.channel)
        self.decoder_d = Decoder(channels=self.channel)
        self.decoder_f = Decoder(channels=self.channel)

    def forward(self, x, d):
        rgb_list, depth_list = self.rgbd_mobile(x, d)
        rgb1, rgb2, rgb3, rgb4 = rgb_list
        d1, d2, d3, d4 = depth_list

        # # if backbone is Swin or PVT
        # rgb_list = self.rgbd_mobile(x)
        # rgb1, rgb2, rgb3, rgb4 = rgb_list
        # depth_list = self.rgbd_mobile(d)
        # d1, d2, d3, d4 = depth_list


        # 4-layer rgb and depth algin and fusion
        rgbd4, rgb4, d4, loss4 = self.dcn4(rgb4, d4)  # Alignment
        # 3-layer rgb and depth algin and fusion
        rgbd3, rgb3, d3, loss3 = self.dcn3(rgb3, d3)  # Alignment
        # 2-layer rgb and depth algin and fusion
        rgbd2, rgb2, d2, loss2 = self.dcn2(rgb2, d2)
        # 1-layer rgb and depth algin and fusion
        rgbd1, rgb1, d1, loss1 = self.dcn1(rgb1, d1)

        loss = loss1 + loss2 + loss3 + loss4

        mask_f = self.decoder_f(rgbd1, rgbd2, rgbd3, rgbd4)
        mask_r = self.decoder_r(rgbd1 + rgb_list[0], rgbd2 + rgb_list[1], rgbd3 + rgb_list[2], rgbd4 + rgb_list[3])
        mask_d = self.decoder_d(rgbd1 + depth_list[0], rgbd2 + depth_list[1], rgbd3 + depth_list[2],
                                rgbd4 + depth_list[3])

        return mask_f, mask_r, mask_d, loss



