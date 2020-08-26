from torch import nn


class NLEmbedingBlock(nn.Module):
    def __init__(self, in_dim, out_dim, fc_dim=None):
        super(NLEmbedingBlock, self).__init__()

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