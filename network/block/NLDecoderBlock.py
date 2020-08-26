from torch import nn


class NLDecoderBlock(nn.Module):
    def __init__(self, in_dim, gf_dim, out_dim):
        super(NLDecoderBlock, self).__init__()
        self.in_dim = in_dim

        self.main = nn.Sequential(
            self._make_layer(gf_dim * 5, 2),
            self._make_layer(gf_dim * 8, 1),

            self._make_layer(gf_dim * 8, 2),
            self._make_layer(gf_dim * 4, 1),
            self._make_layer(gf_dim * 6, 1),

            self._make_layer(gf_dim * 6, 2),
            self._make_layer(gf_dim * 3, 1),
            self._make_layer(gf_dim * 4, 1),

            self._make_layer(gf_dim * 4, 2),
            self._make_layer(gf_dim * 2, 1),
            self._make_layer(gf_dim * 2, 1),

            self._make_layer(gf_dim * 2, 2),
            self._make_layer(gf_dim * 1, 1),

            nn.ConvTranspose2d(self.in_dim, out_dim, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.main(x)
        return x

    def _make_layer(self, out_dim, stride):
        layers = [
            # original parameter: padding = same, now zeros
            nn.ConvTranspose2d(self.in_dim, out_dim, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ELU(inplace=True)
        ]
        self.in_dim = out_dim
        return nn.Sequential(*layers)


class NLAlbedoDecoderBlock(nn.Module):
    """
    Create texture decoder network
    Output: uv_texture [N, tex_sz[0], tex_sz[1], self.c_dim]
    """

    def __init__(self, in_dim, gf_dim, tex_sz):
        super(NLAlbedoDecoderBlock, self).__init__()

        self.df_dim = gf_dim

        self.s_h = int(tex_sz[0])
        self.s_w = int(tex_sz[1])
        self.s32_h = int(self.s_h/32)
        self.s32_w = int(self.s_w/32)

        self.linear = nn.Linear(in_dim, self.df_dim * 10 * self.s32_h * self.s32_w)
        self.bn_elu = nn.Sequential(
            nn.BatchNorm2d(self.df_dim * 10),
            nn.ELU(inplace=True)
        )

        self.nl_decoder_block = NLDecoderBlock(self.df_dim * 10, gf_dim, 3)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, self.df_dim * 10, self.s32_h, self.s32_w)
        x = self.bn_elu(x)
        x = self.nl_decoder_block(x)
        return x
