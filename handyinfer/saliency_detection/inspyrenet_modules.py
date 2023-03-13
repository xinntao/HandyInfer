import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import List, Optional

# dilation and erosion functions are copied from
#  https://github.com/kornia/kornia/blob/master/kornia/morphology/morphology.py


def _neight2channels_like_kernel(kernel: torch.Tensor) -> torch.Tensor:
    h, w = kernel.size()
    kernel = torch.eye(h * w, dtype=kernel.dtype, device=kernel.device)
    return kernel.view(h * w, 1, h, w)


def dilation(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[List[int]] = None,
    border_type: str = 'geodesic',
    border_value: float = 0.0,
    max_val: float = 1e4,
    engine: str = 'unfold',
) -> torch.Tensor:
    r"""Return the dilated image applying the same kernel in each channel.

    .. image:: _static/img/dilation.png

    The kernel must have 2 dimensions.

    Args:
        tensor: Image with shape :math:`(B, C, H, W)`.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the center over which the operation is applied. Its shape is :math:`(k_x, k_y)`.
            For full structural elements use torch.ones_like(structural_element).
        structuring_element: Structuring element used for the grayscale dilation. It may be a non-flat
            structuring element.
        origin: Origin of the structuring element. Default: ``None`` and uses the center of
            the structuring element as origin (rounding towards zero).
        border_type: It determines how the image borders are handled, where ``border_value`` is the value
            when ``border_type`` is equal to ``constant``. Default: ``geodesic`` which ignores the values that are
            outside the image when applying the operation.
        border_value: Value to fill past edges of input if ``border_type`` is ``constant``.
        max_val: The value of the infinite elements in the kernel.
        engine: convolution is faster and less memory hungry, and unfold is more stable numerically

    Returns:
        Dilated image with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       morphology_101.html>`__.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(3, 3)
        >>> dilated_img = dilation(tensor, kernel)
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f'Input type is not a torch.Tensor. Got {type(tensor)}')

    if len(tensor.shape) != 4:
        raise ValueError(f'Input size must have 4 dimensions. Got {tensor.dim()}')

    if not isinstance(kernel, torch.Tensor):
        raise TypeError(f'Kernel type is not a torch.Tensor. Got {type(kernel)}')

    if len(kernel.shape) != 2:
        raise ValueError(f'Kernel size must have 2 dimensions. Got {kernel.dim()}')

    # origin
    se_h, se_w = kernel.shape
    if origin is None:
        origin = [se_h // 2, se_w // 2]

    # pad
    pad_e: List[int] = [origin[1], se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]
    if border_type == 'geodesic':
        border_value = -max_val
        border_type = 'constant'
    output: torch.Tensor = F.pad(tensor, pad_e, mode=border_type, value=border_value)

    # computation
    if structuring_element is None:
        neighborhood = torch.zeros_like(kernel)
        neighborhood[kernel == 0] = -max_val
    else:
        neighborhood = structuring_element.clone()
        neighborhood[kernel == 0] = -max_val

    if engine == 'unfold':
        output = output.unfold(2, se_h, 1).unfold(3, se_w, 1)
        output, _ = torch.max(output + neighborhood.flip((0, 1)), 4)
        output, _ = torch.max(output, 4)
    elif engine == 'convolution':
        B, C, H, W = tensor.size()
        h_pad, w_pad = output.shape[-2:]
        reshape_kernel = _neight2channels_like_kernel(kernel)
        output, _ = F.conv2d(
            output.view(B * C, 1, h_pad, w_pad), reshape_kernel, padding=0,
            bias=neighborhood.view(-1).flip(0)).max(dim=1)
        output = output.view(B, C, H, W)
    else:
        raise NotImplementedError(f"engine {engine} is unknown, use 'convolution' or 'unfold'")
    return output.view_as(tensor)


def erosion(
    tensor: torch.Tensor,
    kernel: torch.Tensor,
    structuring_element: Optional[torch.Tensor] = None,
    origin: Optional[List[int]] = None,
    border_type: str = 'geodesic',
    border_value: float = 0.0,
    max_val: float = 1e4,
    engine: str = 'unfold',
) -> torch.Tensor:
    r"""Return the eroded image applying the same kernel in each channel.

    .. image:: _static/img/erosion.png

    The kernel must have 2 dimensions.

    Args:
        tensor: Image with shape :math:`(B, C, H, W)`.
        kernel: Positions of non-infinite elements of a flat structuring element. Non-zero values give
            the set of neighbors of the center over which the operation is applied. Its shape is :math:`(k_x, k_y)`.
            For full structural elements use torch.ones_like(structural_element).
        structuring_element (torch.Tensor, optional): Structuring element used for the grayscale dilation.
            It may be a non-flat structuring element.
        origin: Origin of the structuring element. Default: ``None`` and uses the center of
            the structuring element as origin (rounding towards zero).
        border_type: It determines how the image borders are handled, where ``border_value`` is the value
            when ``border_type`` is equal to ``constant``. Default: ``geodesic`` which ignores the values that are
            outside the image when applying the operation.
        border_value: Value to fill past edges of input if border_type is ``constant``.
        max_val: The value of the infinite elements in the kernel.
        engine: ``convolution`` is faster and less memory hungry, and ``unfold`` is more stable numerically

    Returns:
        Eroded image with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       morphology_101.html>`__.

    Example:
        >>> tensor = torch.rand(1, 3, 5, 5)
        >>> kernel = torch.ones(5, 5)
        >>> output = erosion(tensor, kernel)
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f'Input type is not a torch.Tensor. Got {type(tensor)}')

    if len(tensor.shape) != 4:
        raise ValueError(f'Input size must have 4 dimensions. Got {tensor.dim()}')

    if not isinstance(kernel, torch.Tensor):
        raise TypeError(f'Kernel type is not a torch.Tensor. Got {type(kernel)}')

    if len(kernel.shape) != 2:
        raise ValueError(f'Kernel size must have 2 dimensions. Got {kernel.dim()}')

    # origin
    se_h, se_w = kernel.shape
    if origin is None:
        origin = [se_h // 2, se_w // 2]

    # pad
    pad_e: List[int] = [origin[1], se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]
    if border_type == 'geodesic':
        border_value = max_val
        border_type = 'constant'
    output: torch.Tensor = F.pad(tensor, pad_e, mode=border_type, value=border_value)

    # computation
    if structuring_element is None:
        neighborhood = torch.zeros_like(kernel)
        neighborhood[kernel == 0] = -max_val
    else:
        neighborhood = structuring_element.clone()
        neighborhood[kernel == 0] = -max_val

    if engine == 'unfold':
        output = output.unfold(2, se_h, 1).unfold(3, se_w, 1)
        output, _ = torch.min(output - neighborhood, 4)
        output, _ = torch.min(output, 4)
    elif engine == 'convolution':
        B, C, H, W = tensor.size()
        Hpad, Wpad = output.shape[-2:]
        reshape_kernel = _neight2channels_like_kernel(kernel)
        output, _ = F.conv2d(
            output.view(B * C, 1, Hpad, Wpad), reshape_kernel, padding=0, bias=-neighborhood.view(-1)).min(dim=1)
        output = output.view(B, C, H, W)
    else:
        raise NotImplementedError(f"engine {engine} is unknown, use 'convolution' or 'unfold'")

    return output


class SelfAttention(nn.Module):

    def __init__(self, in_channels, mode='hw', stage_size=None):
        super(SelfAttention, self).__init__()

        self.mode = mode

        self.query_conv = Conv2d(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.key_conv = Conv2d(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.value_conv = Conv2d(in_channels, in_channels, kernel_size=(1, 1))

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.stage_size = stage_size

    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).view(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.softmax(attention_map)
        projected_value = self.value_conv(x).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out


class Conv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 padding='same',
                 bias=False,
                 bn=True,
                 relu=False):
        super(Conv2d, self).__init__()
        if '__iter__' not in dir(kernel_size):
            kernel_size = (kernel_size, kernel_size)
        if '__iter__' not in dir(stride):
            stride = (stride, stride)
        if '__iter__' not in dir(dilation):
            dilation = (dilation, dilation)

        if padding == 'same':
            width_pad_size = kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1)
            height_pad_size = kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1)
        elif padding == 'valid':
            width_pad_size = 0
            height_pad_size = 0
        else:
            if '__iter__' in dir(padding):
                width_pad_size = padding[0] * 2
                height_pad_size = padding[1] * 2
            else:
                width_pad_size = padding * 2
                height_pad_size = padding * 2

        width_pad_size = width_pad_size // 2 + (width_pad_size % 2 - 1)
        height_pad_size = height_pad_size // 2 + (height_pad_size % 2 - 1)
        pad_size = (width_pad_size, height_pad_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_size, dilation, groups, bias=bias)
        self.reset_parameters()

        if bn is True:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        if relu is True:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight)


class PAA_kernel(nn.Module):

    def __init__(self, in_channel, out_channel, receptive_size, stage_size=None):
        super(PAA_kernel, self).__init__()
        self.conv0 = Conv2d(in_channel, out_channel, 1)
        self.conv1 = Conv2d(out_channel, out_channel, kernel_size=(1, receptive_size))
        self.conv2 = Conv2d(out_channel, out_channel, kernel_size=(receptive_size, 1))
        self.conv3 = Conv2d(out_channel, out_channel, 3, dilation=receptive_size)
        self.Hattn = SelfAttention(out_channel, 'h', stage_size[0] if stage_size is not None else None)
        self.Wattn = SelfAttention(out_channel, 'w', stage_size[1] if stage_size is not None else None)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        Hx = self.Hattn(x)
        Wx = self.Wattn(x)

        x = self.conv3(Hx + Wx)
        return x


class PAA_e(nn.Module):

    def __init__(self, in_channel, out_channel, base_size=None, stage=None):
        super(PAA_e, self).__init__()
        self.relu = nn.ReLU(True)
        if base_size is not None and stage is not None:
            self.stage_size = (base_size[0] // (2**stage), base_size[1] // (2**stage))
        else:
            self.stage_size = None

        self.branch0 = Conv2d(in_channel, out_channel, 1)
        self.branch1 = PAA_kernel(in_channel, out_channel, 3, self.stage_size)
        self.branch2 = PAA_kernel(in_channel, out_channel, 5, self.stage_size)
        self.branch3 = PAA_kernel(in_channel, out_channel, 7, self.stage_size)

        self.conv_cat = Conv2d(4 * out_channel, out_channel, 3)
        self.conv_res = Conv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))

        return x


class PAA_d(nn.Module):

    def __init__(self, in_channel, out_channel=1, depth=64, base_size=None, stage=None):
        super(PAA_d, self).__init__()
        self.conv1 = Conv2d(in_channel, depth, 3)
        self.conv2 = Conv2d(depth, depth, 3)
        self.conv3 = Conv2d(depth, depth, 3)
        self.conv4 = Conv2d(depth, depth, 3)
        self.conv5 = Conv2d(depth, out_channel, 3, bn=False)

        self.base_size = base_size
        self.stage = stage

        if base_size is not None and stage is not None:
            self.stage_size = (base_size[0] // (2**stage), base_size[1] // (2**stage))
        else:
            self.stage_size = [None, None]

        self.Hattn = SelfAttention(depth, 'h', self.stage_size[0])
        self.Wattn = SelfAttention(depth, 'w', self.stage_size[1])

        self.upsample = lambda img, size: F.interpolate(img, size=size, mode='bilinear', align_corners=True)

    def forward(self, fs):  # f3 f4 f5 -> f3 f2 f1
        fx = fs[0]
        for i in range(1, len(fs)):
            fs[i] = self.upsample(fs[i], fx.shape[-2:])
        fx = torch.cat(fs[::-1], dim=1)

        fx = self.conv1(fx)

        Hfx = self.Hattn(fx)
        Wfx = self.Wattn(fx)

        fx = self.conv2(Hfx + Wfx)
        fx = self.conv3(fx)
        fx = self.conv4(fx)
        out = self.conv5(fx)

        return fx, out


class SICA(nn.Module):

    def __init__(self, in_channel, out_channel=1, depth=64, base_size=None, stage=None, lmap_in=False):
        super(SICA, self).__init__()
        self.in_channel = in_channel
        self.depth = depth
        self.lmap_in = lmap_in
        if base_size is not None and stage is not None:
            self.stage_size = (base_size[0] // (2**stage), base_size[1] // (2**stage))
        else:
            self.stage_size = None

        self.conv_query = nn.Sequential(Conv2d(in_channel, depth, 3, relu=True), Conv2d(depth, depth, 3, relu=True))
        self.conv_key = nn.Sequential(Conv2d(in_channel, depth, 1, relu=True), Conv2d(depth, depth, 1, relu=True))
        self.conv_value = nn.Sequential(Conv2d(in_channel, depth, 1, relu=True), Conv2d(depth, depth, 1, relu=True))

        if self.lmap_in is True:
            self.ctx = 5
        else:
            self.ctx = 3

        self.conv_out1 = Conv2d(depth, depth, 3, relu=True)
        self.conv_out2 = Conv2d(in_channel + depth, depth, 3, relu=True)
        self.conv_out3 = Conv2d(depth, depth, 3, relu=True)
        self.conv_out4 = Conv2d(depth, out_channel, 1)

        self.threshold = Parameter(torch.tensor([0.5]))

        if self.lmap_in is True:
            self.lthreshold = Parameter(torch.tensor([0.5]))

    def forward(self, x, smap, lmap: Optional[torch.Tensor] = None):
        # assert not xor(self.lmap_in is True, lmap is not None)
        b, c, h, w = x.shape

        # compute class probability
        smap = F.interpolate(smap, size=x.shape[-2:], mode='bilinear', align_corners=False)
        smap = torch.sigmoid(smap)
        p = smap - self.threshold

        fg = torch.clip(p, 0, 1)  # foreground
        bg = torch.clip(-p, 0, 1)  # background
        cg = self.threshold - torch.abs(p)  # confusion area

        if self.lmap_in is True and lmap is not None:
            lmap = F.interpolate(lmap, size=x.shape[-2:], mode='bilinear', align_corners=False)
            lmap = torch.sigmoid(lmap)
            lp = lmap - self.lthreshold
            fp = torch.clip(lp, 0, 1)  # foreground
            bp = torch.clip(-lp, 0, 1)  # background

            prob = [fg, bg, cg, fp, bp]
        else:
            prob = [fg, bg, cg]

        prob = torch.cat(prob, dim=1)

        # reshape feature & prob
        if self.stage_size is not None:
            shape = self.stage_size
            shape_mul = self.stage_size[0] * self.stage_size[1]
        else:
            shape = (h, w)
            shape_mul = h * w

        f = F.interpolate(x, size=shape, mode='bilinear', align_corners=False).view(b, shape_mul, -1)
        prob = F.interpolate(prob, size=shape, mode='bilinear', align_corners=False).view(b, self.ctx, shape_mul)

        # compute context vector
        context = torch.bmm(prob, f).permute(0, 2, 1).unsqueeze(3)  # b, 3, c

        # k q v compute
        query = self.conv_query(x).view(b, self.depth, -1).permute(0, 2, 1)
        key = self.conv_key(context).view(b, self.depth, -1)
        value = self.conv_value(context).view(b, self.depth, -1).permute(0, 2, 1)

        # compute similarity map
        sim = torch.bmm(query, key)  # b, hw, c x b, c, 2
        sim = (self.depth**-.5) * sim
        sim = F.softmax(sim, dim=-1)

        # compute refined feature
        context = torch.bmm(sim, value).permute(0, 2, 1).contiguous().view(b, -1, h, w)
        context = self.conv_out1(context)

        x = torch.cat([x, context], dim=1)
        x = self.conv_out2(x)
        x = self.conv_out3(x)
        out = self.conv_out4(x)

        return x, out


class ImagePyramid:

    def __init__(self, ksize=7, sigma=1, channels=1):
        self.ksize = ksize
        self.sigma = sigma
        self.channels = channels

        k = cv2.getGaussianKernel(ksize, sigma)
        k = np.outer(k, k)
        k = torch.tensor(k).float()
        self.kernel = k.repeat(channels, 1, 1, 1)

    def cuda(self):
        self.kernel = self.kernel.cuda()
        return self

    def expand(self, x):
        z = torch.zeros_like(x)
        x = torch.cat([x, z, z, z], dim=1)
        x = F.pixel_shuffle(x, 2)
        x = F.pad(x, (self.ksize // 2, ) * 4, mode='reflect')
        x = F.conv2d(x, self.kernel * 4, groups=self.channels)
        return x

    def reduce(self, x):
        x = F.pad(x, (self.ksize // 2, ) * 4, mode='reflect')
        x = F.conv2d(x, self.kernel, groups=self.channels)
        x = x[:, :, ::2, ::2]
        return x

    def deconstruct(self, x):
        reduced_x = self.reduce(x)
        expanded_reduced_x = self.expand(reduced_x)

        if x.shape != expanded_reduced_x.shape:
            expanded_reduced_x = F.interpolate(expanded_reduced_x, x.shape[-2:])

        laplacian_x = x - expanded_reduced_x
        return reduced_x, laplacian_x

    def reconstruct(self, x, laplacian_x):
        expanded_x = self.expand(x)
        if laplacian_x.shape != expanded_x:
            laplacian_x = F.interpolate(laplacian_x, expanded_x.shape[-2:], mode='bilinear', align_corners=True)
        return expanded_x + laplacian_x


class Transition:

    def __init__(self, k=3):
        self.kernel = torch.tensor(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).float()

    def cuda(self):
        self.kernel = self.kernel.cuda()
        return self

    def __call__(self, x):
        x = torch.sigmoid(x)
        dx = dilation(x, self.kernel)
        ex = erosion(x, self.kernel)

        return ((dx - ex) > .5).float()
