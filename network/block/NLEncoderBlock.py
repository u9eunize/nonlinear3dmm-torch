from torch import nn


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
