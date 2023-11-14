import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils.util as util
from SSIM import SSIM
from torchvision.models.vgg import vgg16


##########################################################################
# from IQA_pytorch import SSIM, MS_SSIM


class CharbonnierLoss1(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6, reduction='mean'):
        super(CharbonnierLoss1, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        diff = x - y
        if self.reduction == 'mean':
            loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        else:
            loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


class HuberLoss(nn.Module):
    """Huber Loss (L1)"""

    def __init__(self, delta=1e-2, reduction='mean'):
        super(HuberLoss, self).__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, x, y):
        abs_diff = torch.abs(x - y)
        q_term = torch.min(abs_diff, torch.full_like(abs_diff, self.delta))
        l_term = abs_diff - q_term
        if self.reduction == 'mean':
            loss = torch.mean(0.5 * q_term ** 2 + self.delta * l_term)
        else:
            loss = torch.sum(0.5 * q_term ** 2 + self.delta * l_term)
        return loss


class TVLoss(nn.Module):
    """Total Variation Loss"""

    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        return torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
               torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))


class GWLoss(nn.Module):
    """Gradient Weighted Loss"""

    def __init__(self, w=4, reduction='mean'):
        super(GWLoss, self).__init__()
        self.w = w
        self.reduction = reduction
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float)
        self.weight_x = nn.Parameter(data=sobel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=sobel_y, requires_grad=False)

    def forward(self, x1, x2):
        b, c, w, h = x1.shape
        weight_x = self.weight_x.expand(c, 1, 3, 3).type_as(x1)
        weight_y = self.weight_y.expand(c, 1, 3, 3).type_as(x1)
        Ix1 = F.conv2d(x1, weight_x, stride=1, padding=1, groups=c)
        Ix2 = F.conv2d(x2, weight_x, stride=1, padding=1, groups=c)
        Iy1 = F.conv2d(x1, weight_y, stride=1, padding=1, groups=c)
        Iy2 = F.conv2d(x2, weight_y, stride=1, padding=1, groups=c)
        dx = torch.abs(Ix1 - Ix2)
        dy = torch.abs(Iy1 - Iy2)
        # loss = torch.exp(2*(dx + dy)) * torch.abs(x1 - x2)
        loss = (1 + self.w * dx) * (1 + self.w * dy) * torch.abs(x1 - x2)
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return torch.sum(loss)


class StyleLoss(nn.Module):
    """Style Loss"""

    def __init__(self):
        super(StyleLoss, self).__init__()

    @staticmethod
    def gram_matrix(self, x):
        B, C, H, W = x.size()
        features = x.view(B * C, H * W)
        G = torch.mm(features, features.t())
        return G.div(B * C * H * W)

    def forward(self, input, target):
        G_i = self.gram_matrix(input)
        G_t = self.gram_matrix(target).detach()
        loss = F.mse_loss(G_i, G_t)
        return loss


class GANLoss(nn.Module):
    """GAN loss (vanilla | lsgan | wgan-gp)"""

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':
            def wgan_loss(input, target):
                # target1 is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    """Gradient Penalty Loss"""

    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1) ** 2).mean()
        return loss


class PyramidLoss(nn.Module):
    """Pyramid Loss"""

    def __init__(self, num_levels=3, pyr_mode='gau', loss_mode='l1', reduction='mean'):
        super(PyramidLoss, self).__init__()
        self.num_levels = num_levels
        self.pyr_mode = pyr_mode
        self.loss_mode = loss_mode
        assert self.pyr_mode == 'gau' or self.pyr_mode == 'lap'
        if self.loss_mode == 'l1':
            self.loss = nn.L1Loss(reduction=reduction)
        elif self.loss_mode == 'l2':
            self.loss = nn.MSELoss(reduction=reduction)
        elif self.loss_mode == 'hb':
            self.loss = HuberLoss(reduction=reduction)
        elif self.loss_mode == 'cb':
            self.loss = CharbonnierLoss1(reduction=reduction)
        else:
            raise ValueError()

    def forward(self, x, y):
        B, C, H, W = x.shape
        device = x.device
        gauss_kernel = util.gauss_kernel(size=5, device=device, channels=C)
        if self.pyr_mode == 'gau':
            pyr_x = util.gau_pyramid(img=x, kernel=gauss_kernel, max_levels=self.num_levels)
            pyr_y = util.gau_pyramid(img=y, kernel=gauss_kernel, max_levels=self.num_levels)
        else:
            pyr_x = util.lap_pyramid(img=x, kernel=gauss_kernel, max_levels=self.num_levels)
            pyr_y = util.lap_pyramid(img=y, kernel=gauss_kernel, max_levels=self.num_levels)
        loss = 0
        for i in range(self.num_levels):
            loss += self.loss(pyr_x[i], pyr_y[i])
        return loss


class LapPyrLoss(nn.Module):
    """Pyramid Loss"""

    def __init__(self, num_levels=3, lf_mode='ssim', hf_mode='cb', reduction='mean'):
        super(LapPyrLoss, self).__init__()
        self.num_levels = num_levels
        self.lf_mode = lf_mode
        self.hf_mode = hf_mode
        # if lf_mode == 'ssim':
        # #self.lf_loss = SSIM(channels=1)
        # self.lf_loss = SSIM(channels=1)
        # elif lf_mode == 'cb':
        # self.lf_loss = CharbonnierLoss1(reduction=reduction)
        # else:
        # raise ValueError()
        # if hf_mode == 'ssim':
        # #self.hf_loss = SSIM(channels=1)
        # self.hf_loss = SSIM(channels=1)
        # elif hf_mode == 'cb':
        # self.hf_loss = CharbonnierLoss1(reduction=reduction)
        # else:
        # raise ValueError()
        self.CharLoss = CharbonnierLoss1()

    def forward(self, x_img, y_img, x, y):
        B, C, H, W = x.shape
        device = x.device
        gauss_kernel = util.gauss_kernel(size=5, device=device, channels=C)
        pyr_x = util.laplacian_pyramid(img=x, kernel=gauss_kernel, max_levels=self.num_levels)
        pyr_y = util.laplacian_pyramid(img=y, kernel=gauss_kernel, max_levels=self.num_levels)
        # loss = self.lf_loss(pyr_x[-1], pyr_y[-1])
        loss = self.CharLoss(x_img, y_img)
        for i in range(self.num_levels - 1):
            # loss += self.hf_loss(pyr_x[i], pyr_y[i])
            loss += self.CharLoss(pyr_x[i], pyr_y[i])
        return loss


#########################################################################################
def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return [x_LL, x_HL, x_LH, x_HH]  # torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


# 使用哈尔 haar 小波变换来实现二维逆向离散小波
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width]) #[1, 12, 56, 56]
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width
    # print(out_batch, out_channel, out_height, out_width) #1 3 112 112
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    # print(x1.shape) #torch.Size([1, 3, 56, 56])
    # print(x2.shape) #torch.Size([1, 3, 56, 56])
    # print(x3.shape) #torch.Size([1, 3, 56, 56])
    # print(x4.shape) #torch.Size([1, 3, 56, 56])
    # h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h


# 二维离散小波
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)


# 逆向二维离散小波
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class CharbonnierLoss_dwt(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss_dwt, self).__init__()
        self.eps = eps
        self.target_down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)
        self.x_dwt = DWT()
        self.y_dwt = DWT()

    def forward(self, x, y):
        x_fea = self.x_dwt(x)
        y_fea = self.y_dwt(y)

        # _, _, x_kw, x_kh = x_fea[0].shape
        # _, _, y_kw, y_kh = y_fea[0].shape
        # if x_kw == y_kw:
        # diff = x_fea - y_fea
        # else:
        # diff = x_fea - self.target_down(y_fea)
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = np.sum(
            [torch.mean(torch.sqrt(((x_fea[j] - y_fea[j]) * (x_fea[j] - y_fea[j])) + (self.eps * self.eps))) for j in
             range(len(x_fea))])
        # loss = torch.mean(torch.sqrt(((x_fea[j]-y_fea[j]) * (x_fea[j]-y_fea[j])) + (self.eps*self.eps)))
        return loss


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.target_down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)

    def forward(self, x, y):
        _, _, x_kw, x_kh = x.shape
        _, _, y_kw, y_kh = y.shape
        if x_kw == y_kw:
            diff = x - y
        else:
            diff = x - self.target_down(y)
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class L1smooth(nn.Module):
    """L1smooth (L1)"""

    def __init__(self):
        super(L1smooth, self).__init__()
        self.target_down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)
        self.L1_smooth = torch.nn.SmoothL1Loss()

    def forward(self, x, y):
        _, _, x_kw, x_kh = x.shape
        _, _, y_kw, y_kh = y.shape
        if x_kw == y_kw:
            loss = self.L1_smooth(x, y)
        else:
            loss = self.L1_smooth(x, self.target_down(y))
        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()
        self.target_down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        _, _, x_kw, x_kh = x.shape
        _, _, y_kw, y_kh = y.shape
        if x_kw == y_kw:
            loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        else:
            loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(self.target_down(y)))
        # loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
        return k


class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features.cuda()

        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, x, y):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)

        g = self.to_relu_1_2(y)
        g_relu_1_2 = g
        g = self.to_relu_2_2(g)
        g_relu_2_2 = g
        g = self.to_relu_3_3(g)
        g_relu_3_3 = g
        g = self.to_relu_4_3(g)
        g_relu_4_3 = g
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        content_loss = self.mse_loss(
            h_relu_4_3, g_relu_4_3)
        return content_loss


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = vgg16(pretrained=True).features.cuda()
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h

        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3)
        return out
