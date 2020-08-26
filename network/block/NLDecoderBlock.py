from torch import nn


class NLDecoderBlock(nn.Module):
    def __init__(self, in_dim, df_dim, out_dim):
        super(NLDecoderBlock, self).__init__()
        self.in_dim = in_dim

        self.main = nn.Sequential(
            self._make_layer(df_dim * 5, 2),
            self._make_layer(df_dim * 8, 1),

            self._make_layer(df_dim * 8, 2),
            self._make_layer(df_dim * 4, 1),
            self._make_layer(df_dim * 6, 1),

            self._make_layer(df_dim * 6, 2),
            self._make_layer(df_dim * 3, 1),
            self._make_layer(df_dim * 4, 1),

            self._make_layer(df_dim * 4, 2),
            self._make_layer(df_dim * 2, 1),
            self._make_layer(df_dim * 2, 1),

            self._make_layer(df_dim * 2, 2),
            self._make_layer(df_dim * 1, 1),

            nn.ConvTranspose2d(self.in_dim, out_dim, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        if x.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            output = self.main(x)
        return output

    def _make_layer(self, out_dim, stride):
        layers = [
            # original parameter: padding = same, now zeros
            nn.ConvTranspose2d(self.in_dim, out_dim, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ELU(inplace=True)  # inplace 옵션 주는 것은 의문
        ]
        self.in_dim = out_dim
        return nn.Sequential(*layers)
