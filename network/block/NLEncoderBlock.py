from torch import nn

from torch import nn


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
            nn.ELU(inplace=True)  # inplace 옵션 주는 것은 의문
        ]
        self.in_dim = out_dim
        return nn.Sequential(*layers)
