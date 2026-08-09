import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Type, Union


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first."""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class GRN(nn.Module):
    """ Global Response Normalization layer """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(0,2,3,4), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class MedNeXtBlock(nn.Module):
    """ MedNeXt Block """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 exp_r: int = 4,
                 kernel_size: int = 7,
                 stride: int = 1,
                 do_res: bool = True,
                 norm_type: str = 'group',
                 ):
        super().__init__()

        self.do_res = do_res
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        if norm_type == 'group':
            self.norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        elif norm_type == 'layer':
            self.norm = LayerNorm(normalized_shape=in_channels, data_format="channels_first")
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")

        self.pwconv1 = nn.Conv3d(in_channels, exp_r * in_channels, kernel_size=1, stride=1, padding=0)
        self.dwconv = nn.Conv3d(
            in_channels=exp_r * in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=exp_r * in_channels
        )
        self.grn = GRN(exp_r * in_channels) if exp_r * in_channels > 0 else nn.Identity()
        self.pwconv2 = nn.Conv3d(exp_r * in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.res_proj = None
        if stride != 1 or in_channels != out_channels:
            self.res_proj = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

        self.act = nn.GELU()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.dwconv(x)
        x = self.grn(x)
        x = self.pwconv2(x)

        if self.do_res:
            if self.res_proj is not None:
                residual = self.res_proj(residual)
            x = x + residual

        return x


class MedNeXt(nn.Module):
    """ MedNeXt encoder-decoder for medical image segmentation """
    def __init__(self,
                 in_channels: int = 1,
                 n_channels: int = 32,
                 n_classes: int = 2,
                 exp_r: int = 4,
                 kernel_size: int = 7,
                 deep_supervision: bool = False,
                 do_res: bool = True,
                 block_counts: List[int] = [2, 2, 3, 2],
                 ):
        super().__init__()

        self.depth = len(block_counts)
        self.deep_supervision = deep_supervision
        self.n_channels = n_channels

        self.stem = nn.Conv3d(in_channels, n_channels, kernel_size=1, stride=1, padding=0)

        self.enc_stages = []
        self.enc_downs = []
        current_channels = n_channels

        for i in range(self.depth):
            stage = []
            is_downsample = (i != 0)

            if is_downsample:
                out_channels = int(round(current_channels * 2))
                self.enc_downs.append(
                    nn.Conv3d(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        kernel_size=2, stride=2, padding=0
                    )
                )
                current_channels = out_channels
            else:
                out_channels = current_channels

            for j in range(block_counts[i]):
                stage.append(
                    MedNeXtBlock(
                        in_channels=current_channels,
                        out_channels=current_channels,
                        exp_r=exp_r,
                        kernel_size=kernel_size,
                        stride=1,
                        do_res=do_res,
                        norm_type='group'
                    )
                )

            self.enc_stages.append(nn.Sequential(*stage))

        self.enc_stages = nn.ModuleList(self.enc_stages)
        self.enc_downs = nn.ModuleList(self.enc_downs)

        self.dec_ups = []
        self.dec_stages = []

        for i in range(self.depth - 1, 0, -1):
            encoder_channels_at_i = n_channels * (2 ** i)
            encoder_channels_at_i_minus_1 = n_channels * (2 ** (i - 1))

            self.dec_ups.append(
                nn.ConvTranspose3d(
                    in_channels=encoder_channels_at_i,
                    out_channels=encoder_channels_at_i_minus_1,
                    kernel_size=2, stride=2, padding=0
                )
            )

            stage = []
            for j in range(block_counts[i]):
                in_ch = encoder_channels_at_i_minus_1 * 2 if j == 0 else encoder_channels_at_i_minus_1
                stage.append(
                    MedNeXtBlock(
                        in_channels=in_ch,
                        out_channels=encoder_channels_at_i_minus_1,
                        exp_r=exp_r,
                        kernel_size=kernel_size,
                        stride=1,
                        do_res=do_res,
                        norm_type='group'
                    )
                )
            self.dec_stages.append(nn.Sequential(*stage))

        self.dec_ups = nn.ModuleList(self.dec_ups)
        self.dec_stages = nn.ModuleList(self.dec_stages)
        self.out = nn.Conv3d(in_channels=n_channels, out_channels=n_classes, kernel_size=1, stride=1, padding=0)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_features: bool = False):
        x = self.stem(x)
        skips = []
        for i in range(self.depth):
            x = self.enc_stages[i](x)
            skips.append(x)
            if i < len(self.enc_downs):
                x = self.enc_downs[i](x)

        bottleneck = skips[-1]

        for i in range(len(self.dec_ups)):
            skip_idx = len(skips) - 2 - i
            skip = skips[skip_idx]
            x = self.dec_ups[i](x)
            x = torch.cat([x, skip], dim=1)
            x = self.dec_stages[i](x)

        logits = self.out(x)
        if return_features:
            return logits, bottleneck
        return logits


def MedNeXtB(in_channels: int = 1, n_channels: int = 32, n_classes: int = 2):
    return MedNeXt(
        in_channels=in_channels,
        n_channels=n_channels,
        n_classes=n_classes,
        exp_r=4,
        kernel_size=7,
        deep_supervision=False,
        do_res=True,
        block_counts=[2, 2, 3, 2]
    )
