import numpy as np
# from timm.models.layers import get_act_layer
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from timm.models.beit import gen_relative_position_index
from torch.utils.checkpoint import checkpoint
from typing import Optional


class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

        return x


class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

        # return out + x


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand:
            out_features = features // 2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {'scale_factor': 2}
        elif size is None:
            modifier = {'size': self.size}
        else:
            modifier = {'size': size}

        output = nn.functional.interpolate(output, **modifier, mode='bilinear', align_corners=self.align_corners)

        output = self.out_conv(output)

        return output


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ProjectReadout(nn.Module):

    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index:])
        features = torch.cat((x[:, self.start_index:], readout), -1)

        return self.project(features)


class Transpose(nn.Module):

    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x


activations = {}


def get_activation(name):

    def hook(model, input, output):
        activations[name] = output

    return hook


class Slice(nn.Module):

    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index:]


class AddReadout(nn.Module):

    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index:] + readout.unsqueeze(1)


def get_readout_oper(vit_features, features, use_readout, start_index=1):
    if use_readout == 'ignore':
        readout_oper = [Slice(start_index)] * len(features)
    elif use_readout == 'add':
        readout_oper = [AddReadout(start_index)] * len(features)
    elif use_readout == 'project':
        readout_oper = [ProjectReadout(vit_features, start_index) for out_feat in features]
    else:
        assert (False), "wrong operation for readout token, use_readout can be 'ignore', 'add', or 'project'"

    return readout_oper


def make_backbone_default(
    model,
    features=[96, 192, 384, 768],
    size=[384, 384],
    hooks=[2, 5, 8, 11],
    vit_features=768,
    use_readout='ignore',
    start_index=1,
    start_index_readout=1,
):
    pretrained = nn.Module()

    pretrained.model = model
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation('1'))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation('2'))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation('3'))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation('4'))

    pretrained.activations = activations

    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index_readout)

    # 32, 48, 136, 384
    pretrained.act_postprocess1 = nn.Sequential(
        readout_oper[0],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[0],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[0],
            out_channels=features[0],
            kernel_size=4,
            stride=4,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.act_postprocess2 = nn.Sequential(
        readout_oper[1],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[1],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[1],
            out_channels=features[1],
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.act_postprocess3 = nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[2],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
    )

    pretrained.act_postprocess4 = nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[3],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.Conv2d(
            in_channels=features[3],
            out_channels=features[3],
            kernel_size=3,
            stride=2,
            padding=1,
        ),
    )

    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [16, 16]

    return pretrained


def _get_rel_pos_bias(self, window_size):
    """
    Modification of timm.models.beit.py: Attention._get_rel_pos_bias to support arbitrary window sizes.
    """
    old_height = 2 * self.window_size[0] - 1
    old_width = 2 * self.window_size[1] - 1

    new_height = 2 * window_size[0] - 1
    new_width = 2 * window_size[1] - 1

    old_relative_position_bias_table = self.relative_position_bias_table

    old_num_relative_distance = self.num_relative_distance
    new_num_relative_distance = new_height * new_width + 3

    old_sub_table = old_relative_position_bias_table[:old_num_relative_distance - 3]

    old_sub_table = old_sub_table.reshape(1, old_width, old_height, -1).permute(0, 3, 1, 2)
    new_sub_table = F.interpolate(old_sub_table, size=(new_height, new_width), mode='bilinear')
    new_sub_table = new_sub_table.permute(0, 2, 3, 1).reshape(new_num_relative_distance - 3, -1)

    new_relative_position_bias_table = torch.cat(
        [new_sub_table, old_relative_position_bias_table[old_num_relative_distance - 3:]])

    key = str(window_size[1]) + ',' + str(window_size[0])
    if key not in self.relative_position_indices.keys():
        self.relative_position_indices[key] = gen_relative_position_index(window_size)

    relative_position_bias = new_relative_position_bias_table[self.relative_position_indices[key].view(-1)].view(
        window_size[0] * window_size[1] + 1, window_size[0] * window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    return relative_position_bias.unsqueeze(0)


def forward_adapted_unflatten(pretrained, x, function_name='forward_features'):
    b, c, h, w = x.shape

    exec(f'glob = pretrained.model.{function_name}(x)')

    layer_1 = pretrained.activations['1']
    layer_2 = pretrained.activations['2']
    layer_3 = pretrained.activations['3']
    layer_4 = pretrained.activations['4']

    layer_1 = pretrained.act_postprocess1[0:2](layer_1)
    layer_2 = pretrained.act_postprocess2[0:2](layer_2)
    layer_3 = pretrained.act_postprocess3[0:2](layer_3)
    layer_4 = pretrained.act_postprocess4[0:2](layer_4)

    unflatten = nn.Sequential(
        nn.Unflatten(
            2,
            torch.Size([
                h // pretrained.model.patch_size[1],
                w // pretrained.model.patch_size[0],
            ]),
        ))

    if layer_1.ndim == 3:
        layer_1 = unflatten(layer_1)
    if layer_2.ndim == 3:
        layer_2 = unflatten(layer_2)
    if layer_3.ndim == 3:
        layer_3 = unflatten(layer_3)
    if layer_4.ndim == 3:
        layer_4 = unflatten(layer_4)

    layer_1 = pretrained.act_postprocess1[3:len(pretrained.act_postprocess1)](layer_1)
    layer_2 = pretrained.act_postprocess2[3:len(pretrained.act_postprocess2)](layer_2)
    layer_3 = pretrained.act_postprocess3[3:len(pretrained.act_postprocess3)](layer_3)
    layer_4 = pretrained.act_postprocess4[3:len(pretrained.act_postprocess4)](layer_4)

    return layer_1, layer_2, layer_3, layer_4


def forward_beit(pretrained, x):
    return forward_adapted_unflatten(pretrained, x, 'forward_features')


def attention_forward(self, x, resolution, shared_rel_pos_bias: Optional[torch.Tensor] = None):
    """
    Modification of timm.models.beit.py: Attention.forward to support arbitrary window sizes.
    """
    B, N, C = x.shape

    qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

    q = q * self.scale
    attn = (q @ k.transpose(-2, -1))

    if self.relative_position_bias_table is not None:
        window_size = tuple(np.array(resolution) // 16)
        attn = attn + self._get_rel_pos_bias(window_size)
    if shared_rel_pos_bias is not None:
        attn = attn + shared_rel_pos_bias

    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def block_forward(self, x, resolution, shared_rel_pos_bias: Optional[torch.Tensor] = None):
    """
    Modification of timm.models.beit.py: Block.forward to support arbitrary window sizes.
    """
    if self.gamma_1 is None:
        x = x + self.drop_path(self.attn(self.norm1(x), resolution, shared_rel_pos_bias=shared_rel_pos_bias))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
    else:
        x = x + self.drop_path(
            self.gamma_1 * self.attn(self.norm1(x), resolution, shared_rel_pos_bias=shared_rel_pos_bias))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    return x


def beit_forward_features(self, x):
    """
    Modification of timm.models.beit.py: Beit.forward_features to support arbitrary window sizes.
    """
    resolution = x.shape[2:]

    x = self.patch_embed(x)
    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    if self.pos_embed is not None:
        x = x + self.pos_embed
    x = self.pos_drop(x)

    rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
    for blk in self.blocks:
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint(blk, x, shared_rel_pos_bias=rel_pos_bias)
        else:
            x = blk(x, resolution, shared_rel_pos_bias=rel_pos_bias)
    x = self.norm(x)
    return x


def _make_beit_backbone(
    model,
    features=[96, 192, 384, 768],
    size=[384, 384],
    hooks=[0, 4, 8, 11],
    vit_features=768,
    use_readout='ignore',
    start_index=1,
    start_index_readout=1,
):
    backbone = make_backbone_default(model, features, size, hooks, vit_features, use_readout, start_index,
                                     start_index_readout)

    backbone.model.patch_embed.forward = types.MethodType(patch_embed_forward, backbone.model.patch_embed)
    backbone.model.forward_features = types.MethodType(beit_forward_features, backbone.model)

    for block in backbone.model.blocks:
        attn = block.attn
        attn._get_rel_pos_bias = types.MethodType(_get_rel_pos_bias, attn)
        attn.forward = types.MethodType(attention_forward, attn)
        attn.relative_position_indices = {}

        block.forward = types.MethodType(block_forward, block)

    return backbone


def _make_pretrained_beitl16_384(pretrained, use_readout='ignore', hooks=None):
    model = timm.create_model('beit_large_patch16_384', pretrained=pretrained)

    hooks = [5, 11, 17, 23] if hooks is None else hooks
    return _make_beit_backbone(
        model,
        features=[256, 512, 1024, 1024],
        hooks=hooks,
        vit_features=1024,
        use_readout=use_readout,
    )


def patch_embed_forward(self, x):
    """
    Modification of timm.models.layers.patch_embed.py: PatchEmbed.forward to support arbitrary window sizes.
    """
    x = self.proj(x)
    if self.flatten:
        x = x.flatten(2).transpose(1, 2)
    x = self.norm(x)
    return x


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)

    return scratch


def _make_encoder(backbone,
                  features,
                  use_pretrained,
                  groups=1,
                  expand=False,
                  exportable=True,
                  hooks=None,
                  use_vit_only=False,
                  use_readout='ignore',
                  in_features=[96, 256, 512, 1024]):
    pretrained = _make_pretrained_beitl16_384(use_pretrained, hooks=hooks, use_readout=use_readout)
    scratch = _make_scratch([256, 512, 1024, 1024], features, groups=groups, expand=expand)  # BEiT_384-L (backbone)
    return pretrained, scratch


class DPT(nn.Module):

    def __init__(self,
                 head,
                 features=256,
                 backbone='beitl16_384',
                 readout='project',
                 channels_last=False,
                 use_bn=False,
                 **kwargs):

        super(DPT, self).__init__()

        self.channels_last = channels_last

        # For the Swin, Swin 2, LeViT and Next-ViT Transformers, the hierarchical architectures prevent setting the
        # hooks freely. Instead, the hooks have to be chosen according to the ranges specified in the comments.
        hooks = {
            'beitl16_384': [5, 11, 17, 23],
        }[backbone]

        in_features = None

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            False,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks,
            use_readout=readout,
            in_features=in_features,
        )

        self.number_layers = len(hooks) if hooks is not None else 4
        size_refinenet3 = None
        self.scratch.stem_transpose = None

        self.forward_transformer = forward_beit

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn, size_refinenet3)
        if self.number_layers >= 4:
            self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head

    def forward(self, x):

        if self.channels_last:
            x.contiguous(memory_format=torch.channels_last)

        layers = self.forward_transformer(self.pretrained, x)
        if self.number_layers == 3:
            layer_1, layer_2, layer_3 = layers
        else:
            layer_1, layer_2, layer_3, layer_4 = layers

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        if self.number_layers >= 4:
            layer_4_rn = self.scratch.layer4_rn(layer_4)

        if self.number_layers == 3:
            path_3 = self.scratch.refinenet3(layer_3_rn, size=layer_2_rn.shape[2:])
        else:
            path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
            path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        if self.scratch.stem_transpose is not None:
            path_1 = self.scratch.stem_transpose(path_1)

        out = self.scratch.output_conv(path_1)

        return out


class DPTDepthModel(DPT):

    def __init__(self, path=None, non_negative=True, **kwargs):
        features = kwargs['features'] if 'features' in kwargs else 256
        head_features_1 = kwargs['head_features_1'] if 'head_features_1' in kwargs else features
        head_features_2 = kwargs['head_features_2'] if 'head_features_2' in kwargs else 32
        kwargs.pop('head_features_1', None)
        kwargs.pop('head_features_2', None)

        head = nn.Sequential(
            nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        super().__init__(head, **kwargs)

    def forward(self, x):
        return super().forward(x).squeeze(dim=1)
