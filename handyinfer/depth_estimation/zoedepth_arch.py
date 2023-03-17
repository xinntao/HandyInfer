# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from zoedepth.models.model_io import load_state_from_resource


def log_binom(n, k, eps=1e-7):
    """ log(nCk) using stirling approximation """
    n = n + eps
    k = k + eps
    return n * torch.log(n) - k * torch.log(k) - (n - k) * torch.log(n - k + eps)


class LogBinomial(nn.Module):

    def __init__(self, n_classes=256, act=torch.softmax):
        """Compute log binomial distribution for n_classes
        Args:
            n_classes (int, optional): number of output classes. Defaults to 256.
        """
        super().__init__()
        self.K = n_classes
        self.act = act
        self.register_buffer('k_idx', torch.arange(0, n_classes).view(1, -1, 1, 1))
        self.register_buffer('K_minus_1', torch.Tensor([self.K - 1]).view(1, -1, 1, 1))

    def forward(self, x, t=1., eps=1e-4):
        """Compute log binomial distribution for x
        Args:
            x (torch.Tensor - NCHW): probabilities
            t (float, torch.Tensor - NCHW, optional): Temperature of distribution. Defaults to 1..
            eps (float, optional): Small number for numerical stability. Defaults to 1e-4.
        Returns:
            torch.Tensor -NCHW: log binomial distribution logbinomial(p;t)
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)  # make it nchw

        one_minus_x = torch.clamp(1 - x, eps, 1)
        x = torch.clamp(x, eps, 1)
        y = log_binom(self.K_minus_1, self.k_idx) + self.k_idx * \
            torch.log(x) + (self.K - 1 - self.k_idx) * torch.log(one_minus_x)
        return self.act(y / t, dim=1)


class ConditionalLogBinomial(nn.Module):

    def __init__(self,
                 in_features,
                 condition_dim,
                 n_classes=256,
                 bottleneck_factor=2,
                 p_eps=1e-4,
                 max_temp=50,
                 min_temp=1e-7,
                 act=torch.softmax):
        """Conditional Log Binomial distribution

        Args:
            in_features (int): number of input channels in main feature
            condition_dim (int): number of input channels in condition feature
            n_classes (int, optional): Number of classes. Defaults to 256.
            bottleneck_factor (int, optional): Hidden dim factor. Defaults to 2.
            p_eps (float, optional): small eps value. Defaults to 1e-4.
            max_temp (float, optional): Maximum temperature of output distribution. Defaults to 50.
            min_temp (float, optional): Minimum temperature of output distribution. Defaults to 1e-7.
        """
        super().__init__()
        self.p_eps = p_eps
        self.max_temp = max_temp
        self.min_temp = min_temp
        self.log_binomial_transform = LogBinomial(n_classes, act=act)
        bottleneck = (in_features + condition_dim) // bottleneck_factor
        self.mlp = nn.Sequential(
            nn.Conv2d(in_features + condition_dim, bottleneck, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            # 2 for p linear norm, 2 for t linear norm
            nn.Conv2d(bottleneck, 2 + 2, kernel_size=1, stride=1, padding=0),
            nn.Softplus())

    def forward(self, x, cond):
        """Forward pass

        Args:
            x (torch.Tensor - NCHW): Main feature
            cond (torch.Tensor - NCHW): condition feature

        Returns:
            torch.Tensor: Output log binomial distribution
        """
        pt = self.mlp(torch.concat((x, cond), dim=1))
        p, t = pt[:, :2, ...], pt[:, 2:, ...]

        p = p + self.p_eps
        p = p[:, 0, ...] / (p[:, 0, ...] + p[:, 1, ...])

        t = t + self.p_eps
        t = t[:, 0, ...] / (t[:, 0, ...] + t[:, 1, ...])
        t = t.unsqueeze(1)
        t = (self.max_temp - self.min_temp) * t + self.min_temp

        return self.log_binomial_transform(p, t)


class SeedBinRegressorUnnormed(nn.Module):

    def __init__(self, in_features, n_bins=16, mlp_dim=256, min_depth=1e-3, max_depth=10):
        """Bin center regressor network. Bin centers are unbounded

        Args:
            in_features (int): input channels
            n_bins (int, optional): Number of bin centers. Defaults to 16.
            mlp_dim (int, optional): Hidden dimension. Defaults to 256.
            min_depth (float, optional): Not used. (for compatibility with SeedBinRegressor)
            max_depth (float, optional): Not used. (for compatibility with SeedBinRegressor)
        """
        super().__init__()
        self.version = '1_1'
        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0), nn.ReLU(inplace=True), nn.Conv2d(mlp_dim, n_bins, 1, 1, 0),
            nn.Softplus())

    def forward(self, x):
        """
        Returns tensor of bin_width vectors (centers). One vector b for every pixel
        """
        B_centers = self._net(x)
        return B_centers, B_centers


class Projector(nn.Module):

    def __init__(self, in_features, out_features, mlp_dim=128):
        """Projector MLP

        Args:
            in_features (int): input channels
            out_features (int): output channels
            mlp_dim (int, optional): hidden dimension. Defaults to 128.
        """
        super().__init__()

        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_dim, out_features, 1, 1, 0),
        )

    def forward(self, x):
        return self._net(x)


@torch.jit.script
def exp_attractor(dx, alpha: float = 300, gamma: int = 2):
    """Exponential attractor: dc = exp(-alpha*|dx|^gamma) * dx , where dx = a - c, a = attractor point, c = bin center,
     dc = shift in bin centermmary for exp_attractor
    Args:
        dx (torch.Tensor): The difference tensor dx = Ai - Cj, where Ai is the attractor point and Cj is the bin center.
        alpha (float, optional): Proportional Attractor strength. Determines the absolute strength. Lower alpha =
        greater attraction. Defaults to 300.
        gamma (int, optional): Exponential Attractor strength. Determines the "region of influence" and indirectly
        number of bin centers affected. Lower gamma = farther reach. Defaults to 2.
    Returns:
        torch.Tensor : Delta shifts - dc; New bin centers = Old bin centers + dc
    """
    return torch.exp(-alpha * (torch.abs(dx)**gamma)) * (dx)


@torch.jit.script
def inv_attractor(dx, alpha: float = 300, gamma: int = 2):
    """Inverse attractor: dc = dx / (1 + alpha*dx^gamma), where dx = a - c, a = attractor point, c = bin center,
    dc = shift in bin center
    This is the default one according to the accompanying paper.
    Args:
        dx (torch.Tensor): The difference tensor dx = Ai - Cj, where Ai is the attractor point and Cj is the bin center.
        alpha (float, optional): Proportional Attractor strength. Determines the absolute strength.
        Lower alpha = greater attraction. Defaults to 300.
        gamma (int, optional): Exponential Attractor strength. Determines the "region of influence" and indirectly
        number of bin centers affected. Lower gamma = farther reach. Defaults to 2.
    Returns:
        torch.Tensor: Delta shifts - dc; New bin centers = Old bin centers + dc
    """
    return dx.div(1 + alpha * dx.pow(gamma))


class AttractorLayerUnnormed(nn.Module):

    def __init__(self,
                 in_features,
                 n_bins,
                 n_attractors=16,
                 mlp_dim=128,
                 min_depth=1e-3,
                 max_depth=10,
                 alpha=300,
                 gamma=2,
                 kind='sum',
                 attractor_type='exp',
                 memory_efficient=False):
        """
        Attractor layer for bin centers. Bin centers are unbounded
        """
        super().__init__()

        self.n_attractors = n_attractors
        self.n_bins = n_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.alpha = alpha
        self.gamma = gamma
        self.kind = kind
        self.attractor_type = attractor_type
        self.memory_efficient = memory_efficient

        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0), nn.ReLU(inplace=True), nn.Conv2d(mlp_dim, n_attractors, 1, 1, 0),
            nn.Softplus())

    def forward(self, x, b_prev, prev_b_embedding=None, interpolate=True, is_for_query=False):
        """
        Args:
            x (torch.Tensor) : feature block; shape - n, c, h, w
            b_prev (torch.Tensor) : previous bin centers normed; shape - n, prev_nbins, h, w

        Returns:
            tuple(torch.Tensor,torch.Tensor) : new bin centers unbounded; shape - n, nbins, h, w. Two outputs just to
            keep the API consistent with the normed version
        """
        if prev_b_embedding is not None:
            if interpolate:
                prev_b_embedding = nn.functional.interpolate(
                    prev_b_embedding, x.shape[-2:], mode='bilinear', align_corners=True)
            x = x + prev_b_embedding

        A = self._net(x)
        n, c, h, w = A.shape

        b_prev = nn.functional.interpolate(b_prev, (h, w), mode='bilinear', align_corners=True)
        b_centers = b_prev

        if self.attractor_type == 'exp':
            dist = exp_attractor
        else:
            dist = inv_attractor

        if not self.memory_efficient:
            func = {'mean': torch.mean, 'sum': torch.sum}[self.kind]
            # .shape N, nbins, h, w
            delta_c = func(dist(A.unsqueeze(2) - b_centers.unsqueeze(1)), dim=1)
        else:
            delta_c = torch.zeros_like(b_centers, device=b_centers.device)
            for i in range(self.n_attractors):
                delta_c += dist(A[:, i, ...].unsqueeze(1) - b_centers)  # .shape N, nbins, h, w

            if self.kind == 'mean':
                delta_c = delta_c / self.n_attractors

        b_new_centers = b_centers + delta_c
        B_centers = b_new_centers

        return b_new_centers, B_centers


def percentile(input, percentiles):
    # source code is from https://github.com/aliutkus/torchpercentile
    """
    Find the percentiles of a tensor along the first dimension.
    """
    input_dtype = input.dtype
    input_shape = input.shape
    if not isinstance(percentiles, torch.Tensor):
        percentiles = torch.tensor(percentiles, dtype=torch.double)
    if not isinstance(percentiles, torch.Tensor):
        percentiles = torch.tensor(percentiles)
    input = input.double()
    percentiles = percentiles.to(input.device).double()
    input = input.view(input.shape[0], -1)
    in_sorted, in_argsort = torch.sort(input, dim=0)
    positions = percentiles * (input.shape[0] - 1) / 100
    floored = torch.floor(positions)
    ceiled = floored + 1
    ceiled[ceiled > input.shape[0] - 1] = input.shape[0] - 1
    weight_ceiled = positions - floored
    weight_floored = 1.0 - weight_ceiled
    d0 = in_sorted[floored.long(), :] * weight_floored[:, None]
    d1 = in_sorted[ceiled.long(), :] * weight_ceiled[:, None]
    result = (d0 + d1).view(-1, *input_shape[1:])

    return result.type(input_dtype)


class DepthModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.device = 'cpu'
        self.gray_r_map = torch.linspace(1, 0, 256)
        self.gray_r_map = self.gray_r_map.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3)

    def to(self, device) -> nn.Module:
        self.device = device
        self.gray_r_map = self.gray_r_map.to(device)
        return super().to(device)

    def forward(self, x, *args, **kwargs):
        raise NotImplementedError

    def _infer(self, x: torch.Tensor):
        """
        Inference interface for the model
        Args:
            x (torch.Tensor): input tensor of shape (b, c, h, w)
        Returns:
            torch.Tensor: output tensor of shape (b, 1, h, w)
        """
        return self(x)['metric_depth']

    def _infer_with_pad_aug(self,
                            x: torch.Tensor,
                            pad_input: bool = True,
                            fh: float = 3,
                            fw: float = 3,
                            upsampling_mode: str = 'bicubic',
                            padding_mode='reflect',
                            **kwargs) -> torch.Tensor:
        """
        Inference interface for the model with padding augmentation
        Padding augmentation fixes the boundary artifacts in the output depth map.
        Boundary artifacts are sometimes caused by the fact that the model is trained on NYU raw dataset
        which has a black or white border around the image.
        This augmentation pads the input image and crops the prediction back to the original size / view.

        Note: This augmentation is not required for the models trained with 'avoid_boundary'=True.
        Args:
            x (torch.Tensor): input tensor of shape (b, c, h, w)
            pad_input (bool, optional): whether to pad the input or not. Defaults to True.
            fh (float, optional): height padding factor. The padding is calculated as sqrt(h/2) * fh. Defaults to 3.
            fw (float, optional): width padding factor. The padding is calculated as sqrt(w/2) * fw. Defaults to 3.
            upsampling_mode (str, optional): upsampling mode. Defaults to 'bicubic'.
            padding_mode (str, optional): padding mode. Defaults to "reflect".
        Returns:
            torch.Tensor: output tensor of shape (b, 1, h, w)
        """
        # assert x is nchw and c = 3
        assert x.dim() == 4, 'x must be 4 dimensional, got {}'.format(x.dim())
        assert x.shape[1] == 3, 'x must have 3 channels, got {}'.format(x.shape[1])

        if pad_input:
            assert fh > 0 or fw > 0, 'atlease one of fh and fw must be greater than 0'
            pad_h = int(np.sqrt(x.shape[2] / 2) * fh)
            pad_w = int(np.sqrt(x.shape[3] / 2) * fw)
            padding = [pad_w, pad_w]
            if pad_h > 0:
                padding += [pad_h, pad_h]

            x = F.pad(x, padding, mode=padding_mode, **kwargs)
        out = self._infer(x)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=(x.shape[2], x.shape[3]), mode=upsampling_mode, align_corners=False)
        if pad_input:
            # crop to the original size, handling the case where pad_h and pad_w is 0
            if pad_h > 0:
                out = out[:, :, pad_h:-pad_h, :]
            if pad_w > 0:
                out = out[:, :, :, pad_w:-pad_w]
        return out

    def infer_with_flip_aug(self, x, pad_input: bool = True, **kwargs) -> torch.Tensor:
        """
        Inference interface for the model with horizontal flip augmentation
        Horizontal flip augmentation improves the accuracy of the model by averaging the output of the model
        with and without horizontal flip.
        Args:
            x (torch.Tensor): input tensor of shape (b, c, h, w)
            pad_input (bool, optional): whether to use padding augmentation. Defaults to True.
        Returns:
            torch.Tensor: output tensor of shape (b, 1, h, w)
        """
        # infer with horizontal flip and average
        out = self._infer_with_pad_aug(x, pad_input=pad_input, **kwargs)
        out_flip = self._infer_with_pad_aug(torch.flip(x, dims=[3]), pad_input=pad_input, **kwargs)
        out = (out + torch.flip(out_flip, dims=[3])) / 2
        return out

    def infer(self,
              x,
              pad_input: bool = True,
              with_flip_aug: bool = True,
              normalize: bool = True,
              dtype='float32',
              **kwargs) -> torch.Tensor:
        """
        Inference interface for the model
        Args:
            x (torch.Tensor): input tensor of shape (b, c, h, w)
            pad_input (bool, optional): whether to use padding augmentation. Defaults to True.
            with_flip_aug (bool, optional): whether to use horizontal flip augmentation. Defaults to True.
        Returns:
            torch.Tensor: output tensor of shape (b, 1, h, w)
        """
        if with_flip_aug:
            depth = self.infer_with_flip_aug(x, pad_input=pad_input, **kwargs)
        else:
            depth = self._infer_with_pad_aug(x, pad_input=pad_input, **kwargs)

        if normalize:
            depth = self.to_gray_r(depth, dtype=dtype)

        return depth

    def to_gray_r(self,
                  value,
                  vmin=None,
                  vmax=None,
                  invalid_val=-99,
                  invalid_mask=None,
                  background_color=128,
                  dtype='float32'):
        """Converts a depth map to a gray revers image.
        Args:
            value (torch.Tensor): Input depth map. Shape: (b, 1, H, W).
            All singular dimensions are squeezed
            vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used.
            Defaults to None.
            vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used.
            Defaults to None.
            invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'.
            Defaults to -99.
            invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
            background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels.
            Defaults to (128, 128, 128).
        Returns:
            tensor.Tensor, dtype - float32 if dtype == 'float32 or unit8: gray reverse depth map. shape (b, 1, H, W)
        """
        # Percentile can only process the first dimension
        # self.gray_r_map = self.gray_r_map.to(value.device)
        n, c, h, w = value.shape
        value = value.reshape(n, c, h * w).permute(2, 0, 1)

        if invalid_mask is None:
            invalid_mask = value == invalid_val
        mask = torch.logical_not(invalid_mask)

        # normaliza
        vmin_vmax = percentile(value[mask], [2, 85])
        vmin = vmin_vmax[0] if vmin is None else vmin
        vmax = vmin_vmax[1] if vmax is None else vmax

        value[:, vmin == vmax] = value[:, vmin == vmax] * 0.
        value[:, vmin != vmax] = (value[:, vmin != vmax] - vmin[vmin != vmax]) / (
            vmax[vmin != vmax] - vmin[vmin != vmax])

        value[invalid_mask] = torch.nan

        diff = torch.abs(self.gray_r_map - value)
        min_ids = torch.argmin(diff, dim=0)  # [h*w, n, c]

        min_ids[invalid_mask] = background_color
        min_ids = min_ids.reshape(h, w, n, c).permute(2, 3, 0, 1)

        if dtype == 'float32':
            min_ids = min_ids.type(value.dtype) / 255.0  # [0,1]

        return min_ids


class ZoeDepth(DepthModel):

    def __init__(self,
                 core,
                 n_bins=64,
                 bin_centers_type='softplus',
                 bin_embedding_dim=128,
                 min_depth=1e-3,
                 max_depth=10,
                 n_attractors=[16, 8, 4, 1],
                 attractor_alpha=1000,
                 attractor_gamma=2,
                 attractor_kind='mean',
                 attractor_type='inv',
                 min_temp=0.0212,
                 max_temp=50,
                 train_midas=False,
                 midas_lr_factor=10,
                 encoder_lr_factor=10,
                 pos_enc_lr_factor=10,
                 inverse_midas=False,
                 **kwargs):
        """ZoeDepth model. This is the version of ZoeDepth that has a single metric head
        Args:
            core (models.base_models.midas.MidasCore): The base midas model that is used for extraction of "relative"
            features
            n_bins (int, optional): Number of bin centers. Defaults to 64.
            bin_centers_type (str, optional): "normed" or "softplus". Activation type used for bin centers.
            For "normed" bin centers, linear normalization trick is applied. This results in bounded bin centers.
                                               For "softplus", softplus activation is used and thus are unbounded.
                                               Defaults to "softplus".
            bin_embedding_dim (int, optional): bin embedding dimension. Defaults to 128.
            min_depth (float, optional): Lower bound for normed bin centers. Defaults to 1e-3.
            max_depth (float, optional): Upper bound for normed bin centers. Defaults to 10.
            n_attractors (List[int], optional): Number of bin attractors at decoder layers. Defaults to [16, 8, 4, 1].
            attractor_alpha (int, optional): Proportional attractor strength. Refer to models.layers.attractor for
            more details. Defaults to 300.
            attractor_gamma (int, optional): Exponential attractor strength. Refer to models.layers.attractor for
            more details. Defaults to 2.
            attractor_kind (str, optional): Attraction aggregation "sum" or "mean". Defaults to 'sum'.
            attractor_type (str, optional): Type of attractor to use; "inv" (Inverse attractor) or "exp"
            (Exponential attractor). Defaults to 'exp'.
            min_temp (int, optional): Lower bound for temperature of output probability distribution. Defaults to 5.
            max_temp (int, optional): Upper bound for temperature of output probability distribution. Defaults to 50.
            train_midas (bool, optional): Whether to train "core", the base midas model. Defaults to True.
            midas_lr_factor (int, optional): Learning rate reduction factor for base midas model except its encoder
            and positional encodings. Defaults to 10.
            encoder_lr_factor (int, optional): Learning rate reduction factor for the encoder in midas model.
            Defaults to 10.
            pos_enc_lr_factor (int, optional): Learning rate reduction factor for positional encodings in
            the base midas model. Defaults to 10.
        """
        super().__init__()

        self.core = core
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.min_temp = min_temp
        self.bin_centers_type = bin_centers_type

        self.midas_lr_factor = midas_lr_factor
        self.encoder_lr_factor = encoder_lr_factor
        self.pos_enc_lr_factor = pos_enc_lr_factor
        self.train_midas = train_midas
        self.inverse_midas = inverse_midas

        if self.encoder_lr_factor <= 0:
            self.core.freeze_encoder(freeze_rel_pos=self.pos_enc_lr_factor <= 0)

        N_MIDAS_OUT = 32
        btlnck_features = self.core.output_channels[0]
        num_out_features = self.core.output_channels[1:]

        self.conv2 = nn.Conv2d(btlnck_features, btlnck_features, kernel_size=1, stride=1, padding=0)  # btlnck conv

        SeedBinRegressorLayer = SeedBinRegressorUnnormed
        Attractor = AttractorLayerUnnormed

        self.seed_bin_regressor = SeedBinRegressorLayer(
            btlnck_features, n_bins=n_bins, min_depth=min_depth, max_depth=max_depth)
        self.seed_projector = Projector(btlnck_features, bin_embedding_dim)
        self.projectors = nn.ModuleList([Projector(num_out, bin_embedding_dim) for num_out in num_out_features])
        self.attractors = nn.ModuleList([
            Attractor(
                bin_embedding_dim,
                n_bins,
                n_attractors=n_attractors[i],
                min_depth=min_depth,
                max_depth=max_depth,
                alpha=attractor_alpha,
                gamma=attractor_gamma,
                kind=attractor_kind,
                attractor_type=attractor_type) for i in range(len(num_out_features))
        ])

        last_in = N_MIDAS_OUT + 1  # +1 for relative depth

        # use log binomial instead of softmax
        self.conditional_log_binomial = ConditionalLogBinomial(
            last_in, bin_embedding_dim, n_classes=n_bins, min_temp=min_temp, max_temp=max_temp)

    def forward(self, x, return_final_centers=False, denorm=False, return_probs=False, **kwargs):
        """
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W)
            return_final_centers (bool, optional): Whether to return the final bin centers. Defaults to False.
            denorm (bool, optional): Whether to denormalize the input image. This reverses ImageNet normalization as
            midas normalization is different. Defaults to False.
            return_probs (bool, optional): Whether to return the output probability distribution. Defaults to False.

        Returns:
            dict: Dictionary containing the following keys:
                - rel_depth (torch.Tensor): Relative depth map of shape (B, H, W)
                - metric_depth (torch.Tensor): Metric depth map of shape (B, 1, H, W)
                - bin_centers (torch.Tensor): Bin centers of shape (B, n_bins).
                Present only if return_final_centers is True
                - probs (torch.Tensor): Output probability distribution of shape (B, n_bins, H, W).
                Present only if return_probs is True
        """
        b, c, h, w = x.shape
        # print("input shape ", x.shape)
        self.orig_input_width = w
        self.orig_input_height = h
        rel_depth, out = self.core(x, denorm=denorm, return_rel_depth=True)
        # print("output shapes", rel_depth.shape, out.shape)

        outconv_activation = out[0]
        btlnck = out[1]
        x_blocks = out[2:]

        x_d0 = self.conv2(btlnck)
        x = x_d0
        _, seed_b_centers = self.seed_bin_regressor(x)

        if self.bin_centers_type == 'normed' or self.bin_centers_type == 'hybrid2':
            b_prev = (seed_b_centers - self.min_depth) / \
                (self.max_depth - self.min_depth)
        else:
            b_prev = seed_b_centers

        prev_b_embedding = self.seed_projector(x)

        # unroll this loop for better performance
        for projector, attractor, x in zip(self.projectors, self.attractors, x_blocks):
            b_embedding = projector(x)
            b, b_centers = attractor(b_embedding, b_prev, prev_b_embedding, interpolate=True)
            b_prev = b.clone()
            prev_b_embedding = b_embedding.clone()

        last = outconv_activation

        if self.inverse_midas:
            # invert depth followed by normalization
            rel_depth = 1.0 / (rel_depth + 1e-6)
            rel_depth = (rel_depth - rel_depth.min()) / \
                (rel_depth.max() - rel_depth.min())
        # concat rel depth with last. First interpolate rel depth to last size
        rel_cond = rel_depth.unsqueeze(1)
        rel_cond = nn.functional.interpolate(rel_cond, size=last.shape[2:], mode='bilinear', align_corners=True)
        last = torch.cat([last, rel_cond], dim=1)

        b_embedding = nn.functional.interpolate(b_embedding, last.shape[-2:], mode='bilinear', align_corners=True)
        x = self.conditional_log_binomial(last, b_embedding)

        # Now depth value is Sum px * cx , where cx are bin_centers from the last bin tensor
        # print(x.shape, b_centers.shape)
        b_centers = nn.functional.interpolate(b_centers, x.shape[-2:], mode='bilinear', align_corners=True)
        out = torch.sum(x * b_centers, dim=1, keepdim=True)

        # Structure output dict
        output = dict(metric_depth=out)
        if return_final_centers or return_probs:
            output['bin_centers'] = b_centers

        if return_probs:
            output['probs'] = x

        return output
