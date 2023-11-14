"""
the final implement of Multi-Scale Fusion and Decomposition Network for Single Image Deraining
"""

import torch
import torch.nn as nn
from restormer_block import RestormerBlock


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):  # 不改变size的conv
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


def st_conv(in_channels, out_channels, kernel_size, bias=False, stride=2):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


##########################################################################
class depthwise_separable_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(depthwise_separable_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


##########################################################################
# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
# S2FB
class S2FB_2(nn.Module):
    def __init__(self, n_feat, reduction, bias, act):
        super(S2FB_2, self).__init__()
        self.DSC = depthwise_separable_conv(n_feat * 2, n_feat)
        self.CA_fea = CALayer(n_feat, reduction, bias=bias)

    def forward(self, x1, x2):
        FEA_1 = self.DSC(torch.cat((x1, x2), 1))
        res = self.CA_fea(FEA_1) + x1
        return res  


##########################################################################
# Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = [conv(n_feat, n_feat, kernel_size, bias=bias),
                        act,
                        conv(n_feat, n_feat, kernel_size, bias=bias)]

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


# Enhanced Channel Attention Block with DSC (ECAB with dsc)
class CAB_dsc(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB_dsc, self).__init__()
        modules_body = [depthwise_separable_conv(n_feat, n_feat),
                        act,
                        depthwise_separable_conv(n_feat, n_feat)]

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)
        self.S2FB2 = S2FB_2(n_feat, reduction, bias=bias, act=act)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res = self.S2FB2(res, x)
        # res += x
        return res


##########################################################################
# Spatial Attention Layer
class SALayer(nn.Module):
    def __init__(self, channel, reduction=4, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),  # // : 整除,向下取整
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv_du(x)
        return x * y


##########################################################################
# Long Feature Selection and Fusion Block (LFSFB)
class LFSFB(nn.Module):
    def __init__(self, n_feat, kernel_size, act, bias):
        super(LFSFB, self).__init__()
        self.FS = nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1, padding=0, bias=False)
        self.act1 = act
        self.FFU = nn.ConvTranspose2d(n_feat, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.act2 = act

    def forward(self, x1, x2):
        res = self.act1(self.FS(x1))
        res_out = self.act2(self.FFU(x2 + res))
        return res_out


##########################################################################
# Overlapped image patch embedding with 3x3 Conv
class PatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(PatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
        
        self.shallow_fea_B = depthwise_separable_conv(embed_dim, embed_dim)
        self.shallow_fea_R = depthwise_separable_conv(embed_dim, embed_dim)

    def forward(self, x):
        x_fea = self.proj(x)
        b_fea = self.shallow_fea_B(x_fea)
        r_fea = self.shallow_fea_R(x_fea)

        return [b_fea, r_fea]


##########################################################################
# Resizing modules
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, s_factor):
        super(DownSample, self).__init__()
        self.down_B = nn.Sequential(nn.Upsample(scale_factor=s_factor, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))
        self.down_R = nn.Sequential(nn.Upsample(scale_factor=s_factor, mode='bilinear', align_corners=False),
                                    nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x_B, x_R):
        return [self.down_B(x_B), self.down_R(x_R)]


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, s_factor):
        super(UpSample, self).__init__()
        self.up_B = nn.Sequential(nn.Upsample(scale_factor=s_factor, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        self.up_R = nn.Sequential(nn.Upsample(scale_factor=s_factor, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x_B, x_R):
        return [self.up_B(x_B), self.up_R(x_R)]


##########################################################################
# Reconstruction and Reproduction Block (RRB)
class RRB(nn.Module):
    def __init__(self, n_feat, kernel_size, act, bias):
        super(RRB, self).__init__()
        self.recon_B = conv(n_feat, 3, kernel_size, bias=bias)
        self.recon_R = conv(n_feat, 3, kernel_size, bias=bias)

    def forward(self, x):
        xB = x[0]
        xR = x[1]
        recon_B = self.recon_B(xB)
        recon_R = self.recon_R(xR)
        re_rain = recon_B + recon_R
        return [recon_B, re_rain, recon_R]


# Coupled Representation Block (CRB)
class CRB(nn.Module):
    def __init__(self, n_feat):
        super(CRB, self).__init__()

        # 设置可学习参数
        self.fuse_weight_BTOR = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_RTOB = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # 初始化
        self.fuse_weight_BTOR.data.fill_(0.2)
        self.fuse_weight_RTOB.data.fill_(0.2)

        self.conv_fuse_BTOR = nn.Sequential(nn.Conv2d(n_feat, n_feat, 1, padding=0, bias=False), nn.Sigmoid())
        self.conv_fuse_RTOB = nn.Sequential(nn.Conv2d(n_feat, n_feat, 1, padding=0, bias=False), nn.Sigmoid())

    def forward(self, xB_res, xR_res):
        res_BTOR = xB_res * self.conv_fuse_BTOR(xR_res) * self.fuse_weight_BTOR
        res_RTOB = xR_res * self.conv_fuse_RTOB(xB_res) * self.fuse_weight_RTOB

        xb = xB_res - res_BTOR + res_RTOB
        xr = xR_res - res_RTOB + res_BTOR
        
        return [xb, xr]
        # return [xb, xr, res_BTOR, res_RTOB]   


# FEB + CRB
class HRM(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_rb):
        super(HRM, self).__init__()

        heads = 2
        ffn_expansion_factor = 2.66
        LayerNorm_type = 'WithBias'  # Other option 'BiasFree'
        num_RB = num_rb  # number of Restormer Blocks
        self.CAB_r = CAB_dsc(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.CAB_b = CAB_dsc(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.down_R = st_conv(n_feat, n_feat, kernel_size, bias=bias)
        self.act1 = act
        modules_body = [
            RestormerBlock(dim=n_feat, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                           LayerNorm_type=LayerNorm_type) for _ in range(num_RB)]
        self.body = nn.Sequential(*modules_body)

        self.lfsfb = LFSFB(n_feat, kernel_size, act, bias)
        self.CRB = CRB(n_feat)

    def forward(self, x):
        xB = x[0]
        xR = x[1]
        res_down_R = self.act1(self.down_R(xR))
        res_R = self.body(res_down_R)
        xR_res = self.CAB_r(xR) + self.lfsfb(res_down_R, res_R)

        xB_res = self.CAB_b(xB)

        x = self.CRB(xB_res, xR_res)

        return x


##########################################################################
class MODEL(nn.Module):
    def __init__(self, in_c, out_c, n_feat, kernel_size, reduction, act, bias, num_tb):
        super(MODEL, self).__init__()

        # embedding
        self.patch_embed = PatchEmbed(in_c, n_feat)

        self.down0_1 = DownSample(n_feat, n_feat*2, 0.5)   # channel: 48, 96，192，384 # 2C H/2 W/2
        self.down0_2 = DownSample(n_feat, n_feat*4, 0.25)  # 4C H/4 W/4

        self.crb_0 = HRM(n_feat, kernel_size, reduction, act, bias, 8)
        self.crb_1 = HRM(n_feat * 2, kernel_size, reduction, act, bias, 4)
        self.crb_2 = HRM(n_feat * 4, kernel_size, reduction, act, bias, 4)
        self.crb_3 = nn.Sequential(*[HRM(n_feat, kernel_size, reduction, act, bias, 4) for _ in range(3)])

        self.up1_0 = UpSample(n_feat * 2, n_feat, 2)  # From Level 2 to Level 1
        self.up2_0 = UpSample(n_feat * 4, n_feat, 4)  # C H W

        self.point_conv_B = nn.Conv2d(n_feat * 3, n_feat, kernel_size=1)  # 调整通道数 C H W
        self.point_conv_R = nn.Conv2d(n_feat * 3, n_feat, kernel_size=1)

        self.rrb = RRB(n_feat, kernel_size, act, bias=bias)

    def forward(self, x):
        [B_fea, R_fea] = self.patch_embed(x)  # B C H W

        [out_B_0, out_R_0] = self.crb_0([B_fea, R_fea])

        [B_down_1, R_down_1] = self.down0_1(B_fea, R_fea)  # 2C H/2 W/2
        [out_B_1, out_R_1] = self.crb_1([B_down_1, R_down_1])
        [B_up1_0, R_up1_0] = self.up1_0(out_B_1, out_R_1)  # B C H W

        [B_down_2, R_down_2] = self.down0_2(B_fea, R_fea)  # B 4C H/4 W/4
        [out_B_2, out_R_2] = self.crb_2([B_down_2, R_down_2])
        [B_up2_0, R_up2_0] = self.up2_0(out_B_2, out_R_2)  # B C H W

        B_cat = torch.cat([out_B_0, B_up1_0, B_up2_0], 1)  # C + C + C
        R_cat = torch.cat([out_R_0, R_up1_0, R_up2_0], 1)

        B_fuse = self.point_conv_B(B_cat)  # 调整通道数 C H W
        R_fuse = self.point_conv_B(R_cat)

        [out_B_3, out_R_3] = self.crb_3([B_fuse, R_fuse])  # 进一步修正 C H W

        [img_B, img_R, streak] = self.rrb([out_B_3, out_R_3])

        return img_B, img_R, streak


##########################################################################
class HPCNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=48, kernel_size=3, reduction=4, num_tb=4, bias=False):
        super(HPCNet, self).__init__()

        act = nn.PReLU()
        self.model = MODEL(in_c, out_c, n_feat, kernel_size, reduction, act, bias, num_tb)

    def forward(self, x_img):  # b,c,h,w
        imitation, rain_out, streak = self.model(x_img)
        return [imitation, rain_out, streak]
