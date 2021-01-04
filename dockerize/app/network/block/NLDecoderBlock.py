from collections import OrderedDict

import torch
from torch import nn
import numpy as np


def create_layer_blocks(block, in_dim, params):
    """
    all block must be
    block(in_dim, out_dim, *args, **kwags)
    """
    layers = []
    for param in params:
        layers.append(block(in_dim, *param))
        in_dim = param[0]
    return nn.Sequential(*layers)


class ConvTranspose2dOutputResize(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ConvTranspose2dOutputResize, self).__init__()
        self.output_size = None
        if "output_size" in kwargs:
            self.output_size = kwargs["output_size"]
            del kwargs["output_size"]

        self.deconv = nn.ConvTranspose2d(*args, **kwargs)

    def forward(self, x):
        x = self.deconv(x, output_size=self.output_size)
        return x


class DeconvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride, output_size=None):
        super(DeconvBlock, self).__init__()
        self.output_size = output_size
        if output_size is not None:
            self.output_size = np.concatenate([np.array([-1, -1]), self.output_size])

        layers = [
            ("deconv", ConvTranspose2dOutputResize(in_dim, out_dim, 3, stride=stride,
                                                   padding=1, output_size=output_size)),
            # ("bat", nn.BatchNorm2d(out_dim)),
            # ("bat", nn.GroupNorm(out_dim//2, out_dim)) if not last_layer else ("bat", nn.GroupNorm(1, out_dim)),
            ("bat", nn.GroupNorm(32, out_dim)),
            ("relu", nn.ReLU(inplace=True))
        ]
        self.in_dim = out_dim
        self.main = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.main(x)
        return x


class NLDecoderBlock(nn.Module):
    def __init__(self, in_dim, gf_dim, gfc_dim, tex_sz):
        super(NLDecoderBlock, self).__init__()

        self.gf_dim = gf_dim
        self.gfc_dim = gfc_dim

        self.s_h = int(tex_sz[0])
        self.s_w = int(tex_sz[1])
        self.s32_h = int(self.s_h/32)
        self.s32_w = int(self.s_w/32)
        self.out_dim = 3

        self.linear = nn.Linear(in_dim, self.gfc_dim * self.s32_h * self.s32_w)
        self.bn_relu = nn.Sequential(
            # nn.BatchNorm2d(self.gfc_dim),
            nn.GroupNorm(32, self.gfc_dim),
            nn.ReLU(inplace=True)
        )

        self.nl_decoder_block = self._make_decoder_layers(self.gfc_dim, [self.s32_h, self.s32_w])
        self.in_dim = self.gf_dim

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, self.gfc_dim, self.s32_h, self.s32_w)
        x = self.bn_relu(x)
        x = self.nl_decoder_block(x)
        return x

    def _make_decoder_layers(self, in_dim, in_sz):
        if not isinstance(in_sz, np.ndarray):
            in_sz = np.array(in_sz)
        layers = create_layer_blocks(DeconvBlock, in_dim, [
            [self.gf_dim * 5, 2, in_sz * 2],
            [self.gf_dim * 8, 1],

            [self.gf_dim * 8, 2, in_sz * 4],
            [self.gf_dim * 4, 1],
            [self.gf_dim * 6, 1],

            [self.gf_dim * 6, 2, in_sz * 8],
            [self.gf_dim * 3, 1],
            [self.gf_dim * 4, 1],

            [self.gf_dim * 4, 2, in_sz * 16],
            [self.gf_dim * 2, 1],
            [self.gf_dim * 2, 1],

            [self.gf_dim * 2, 2, in_sz * 32],
            [self.gf_dim * 1, 1]
        ])
        return layers


class NLDecoderTailBlock(nn.Module):
    """
    Create texture decoder network
    Output: uv_texture [N, tex_sz[0], tex_sz[1], self.c_dim]
    """

    def __init__(self, in_dim, out_dim, gf_dim, additional_layer=False, is_sigmoid=False):
        super(NLDecoderTailBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.is_sigmoid = is_sigmoid

        layers = []

        if additional_layer:
            layers.append(DeconvBlock(self.in_dim, gf_dim * 2, stride=1))
            self.in_dim = gf_dim * 2
            layers.append(DeconvBlock(gf_dim * 2, gf_dim * 2, stride=1))
            self.in_dim = gf_dim * 2

        layers.append(nn.ConvTranspose2d(self.in_dim, self.out_dim, 3, stride=1, padding=1))
        if self.is_sigmoid:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

        self.in_dim = self.out_dim

    def forward(self, x):
        return self.main(x)
