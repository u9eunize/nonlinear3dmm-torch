from torch import nn

from torch import nn



class Encoder(nn.Module):
    def __init__(self, in_dim, gf_dim, gfc_dim_m, gfc_dim_il, gfc_dim_shape, gfc_dim_tex, gfc_dim_exp, out_dim_trans, out_dim_rot, out_dim_il):
        super(Encoder, self).__init__()
        self.ngpu = 1
        self.in_dim = in_dim
        self.ngf = gf_dim

        self.main = nn.Sequential(
            self._make_layer(gf_dim * 1, 7, 2),
            self._make_layer(gf_dim * 2, 3, 1),

            self._make_layer(gf_dim * 2, 3, 2),
            self._make_layer(gf_dim * 2, 3, 1),
            self._make_layer(gf_dim * 4, 3, 1),

            self._make_layer(gf_dim * 4, 3, 2),
            self._make_layer(gf_dim * 3, 3, 1),
            self._make_layer(gf_dim * 6, 3, 1),

            self._make_layer(gf_dim * 6, 3, 2),
            self._make_layer(gf_dim * 4, 3, 1),
            self._make_layer(gf_dim * 8, 3, 1),

            self._make_layer(gf_dim * 8, 3, 2),
            self._make_layer(gf_dim * 5, 3, 1)
        )

        in_dim = gf_dim * 5
        self.trans = NLEmbeddingBlock(in_dim, gfc_dim_m, out_dim_trans)
        self.rot = NLEmbeddingBlock(in_dim, gfc_dim_m, out_dim_rot)
        self.il = NLEmbeddingBlock(in_dim, gfc_dim_il, out_dim_il)
        self.shape = NLEmbeddingBlock(in_dim, gfc_dim_shape)
        self.tex = NLEmbeddingBlock(in_dim, gfc_dim_tex)
        self.exp = NLEmbeddingBlock(in_dim, gfc_dim_exp)


    def forward(self, x, reg=False):
        if x.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            output = self.main(x)

        trans = self.trans(output)
        rot = self.rot(output)
        il = self.il(output)
        shape = self.shape(output)
        tex = self.tex(output)
        exp = self.exp(output)

        if reg:
           return trans, rot, il, shape, tex, exp, output
        return trans, rot, il, shape, tex, exp

    def _make_layer(self, out_dim, kernel_size, stride):
        layers = [
            # original parameter: padding = same, now zeros
            nn.Conv2d(self.in_dim, out_dim, kernel_size, stride=stride, padding=int((kernel_size - 1) / 2)),
            nn.BatchNorm2d(out_dim),
            # nn.GroupNorm(32, out_dim),

            nn.ReLU(inplace=False)  # inplace 옵션 주는 것은 의문
        ]
        self.in_dim = out_dim
        return nn.Sequential(*layers)


class NLEmbeddingBlock(nn.Module):
    def __init__(self, in_dim, out_dim, fc_dim=None):
        super(NLEmbeddingBlock, self).__init__()

        self.ngpu = 1
        self.num_flat_features = out_dim

        self.main = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.linear = None
        if fc_dim is not None:
            self.linear = nn.Linear(out_dim, fc_dim)

    def forward(self, x):
        if x.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
            output = output.view((-1, self.num_flat_features))
            if self.linear:
                output = nn.parallel.data_parallel(self.linear, output, range(self.ngpu))
        else:
            output = self.main(x)
            output = output.view((-1, output.size(1)))
            if self.linear:
                output = self.linear(output)
        return output


class NLEncoderBlock(nn.Module):
    def __init__(self, in_dim, gf_dim):
        super(NLEncoderBlock, self).__init__()
        self.ngpu = 1
        self.in_dim = in_dim
        self.ngf = gf_dim

        self.main = nn.Sequential(
            self._make_layer(gf_dim * 1, 7, 2),
            self._make_layer(gf_dim * 2, 3, 1),

            self._make_layer(gf_dim * 2, 3, 2),
            self._make_layer(gf_dim * 2, 3, 1),
            self._make_layer(gf_dim * 4, 3, 1),

            self._make_layer(gf_dim * 4, 3, 2),
            self._make_layer(gf_dim * 3, 3, 1),
            self._make_layer(gf_dim * 6, 3, 1),

            self._make_layer(gf_dim * 6, 3, 2),
            self._make_layer(gf_dim * 4, 3, 1),
            self._make_layer(gf_dim * 8, 3, 1),

            self._make_layer(gf_dim * 8, 3, 2),
            self._make_layer(gf_dim * 5, 3, 1)
        )

    def forward(self, x):
        if x.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            output = self.main(x)
        return output

    def _make_layer(self, out_dim, kernel_size, stride):
        layers = [
            # original parameter: padding = same, now zeros
            nn.Conv2d(self.in_dim, out_dim, kernel_size, stride=stride, padding=int((kernel_size - 1) / 2)),
            nn.BatchNorm2d(out_dim),
            # nn.GroupNorm(32, out_dim),
            nn.ReLU(inplace=True)  # inplace 옵션 주는 것은 의문
        ]
        self.in_dim = out_dim
        return nn.Sequential(*layers)
