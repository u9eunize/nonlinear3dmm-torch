from collections import OrderedDict

import torch
from torch import nn
import numpy as np


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


class NLDecoderBlock(nn.Module):
    def __init__(self, in_dim, gf_dim, out_dim, in_sz):
        super(NLDecoderBlock, self).__init__()
        self.in_dim = in_dim

        if not isinstance(in_sz, np.ndarray):
            in_sz = np.array(in_sz)

        self.main = nn.Sequential(
            self._make_layer(gf_dim * 5, 2, in_sz * 2),
            self._make_layer(gf_dim * 8, 1),

            self._make_layer(gf_dim * 8, 2, in_sz * 4),
            self._make_layer(gf_dim * 4, 1),
            self._make_layer(gf_dim * 6, 1),

            self._make_layer(gf_dim * 6, 2, in_sz * 8),
            self._make_layer(gf_dim * 3, 1),
            self._make_layer(gf_dim * 4, 1),

            self._make_layer(gf_dim * 4, 2, in_sz * 16),
            self._make_layer(gf_dim * 2, 1),
            self._make_layer(gf_dim * 2, 1),

            self._make_layer(gf_dim * 2, 2, in_sz * 32),
            self._make_layer(gf_dim * 1, 1),

            nn.ConvTranspose2d(self.in_dim, out_dim, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

    def _make_layer(self, out_dim, stride, output_size=None):
        if output_size is not None:
            output_size = np.concatenate([np.array([-1, -1]), output_size])
        layers = [
            ("deconv", ConvTranspose2dOutputResize(self.in_dim, out_dim, 3, stride=stride,
                                                   padding=1, output_size=output_size)),
            ("bat", nn.BatchNorm2d(out_dim)),
            ("elu", nn.ELU(inplace=True))
        ]
        self.in_dim = out_dim
        return nn.Sequential(OrderedDict(layers))


class NLAlbedoDecoderBlock(nn.Module):
    """
    Create texture decoder network
    Output: uv_texture [N, tex_sz[0], tex_sz[1], self.c_dim]
    """

    def __init__(self, in_dim, gf_dim, tex_sz):
        super(NLAlbedoDecoderBlock, self).__init__()

        self.gf_dim = gf_dim

        self.s_h = int(tex_sz[0])
        self.s_w = int(tex_sz[1])
        self.s32_h = int(self.s_h/32)
        self.s32_w = int(self.s_w/32)

        self.linear = nn.Linear(in_dim, self.gf_dim * 10 * self.s32_h * self.s32_w)
        self.bn_elu = nn.Sequential(
            nn.BatchNorm2d(self.gf_dim * 10),
            nn.ELU(inplace=True)
        )

        self.nl_decoder_block = NLDecoderBlock(self.gf_dim * 10, self.gf_dim, 3, [self.s32_h, self.s32_w])
        self.in_dim = self.nl_decoder_block.in_dim

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, self.gf_dim * 10, self.s32_h, self.s32_w)
        x = self.bn_elu(x)
        x = self.nl_decoder_block(x)
        return x


class NLShapeDecoderBlock(nn.Module):

    def __init__(self, in_dim, gf_dim, gfc_dim, tex_sz):
        super(NLShapeDecoderBlock, self).__init__()

        self.gf_dim = gf_dim
        self.gfc_dim = gfc_dim

        self.s_h = int(tex_sz[0])
        self.s_w = int(tex_sz[1])
        self.s32_h = int(self.s_h / 32)
        self.s32_w = int(self.s_w / 32)

        # shape2d network
        self.linear = nn.Linear(in_dim, self.gfc_dim * self.s32_h * self.s32_w)
        self.bn_elu = nn.Sequential(
            nn.BatchNorm2d(self.gfc_dim),
            nn.ELU(inplace=True)
        )
        self.nl_decoder_block = NLDecoderBlock(self.gfc_dim, self.gf_dim, 3, [self.s32_h, self.s32_w])
        self.in_dim = self.nl_decoder_block.in_dim

    def forward(self, x):
        # shape 2d
        x = self.linear(x)
        x = x.view(-1, self.gfc_dim, self.s32_h, self.s32_w)
        x = self.bn_elu(x)
        shape2d = 2 * self.nl_decoder_block(x)

        return shape2d

"""

"""