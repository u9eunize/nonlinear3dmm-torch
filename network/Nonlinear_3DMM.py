import torch
from torch import nn

from network.block import *


class Nonlinear3DMM(nn.Module):
    def __init__(self, gf_dim=32, df_dim=32, gfc_dim=512, dfc_dim=512, nz=3, m_dim=8, il_dim=27, tex_sz=(192, 224)):
        super(Nonlinear3DMM, self).__init__()

        # naming from https://gist.github.com/EderSantana/9b0d5fb309d775b995d5236c32238349
        # TODO: gen(gf)->encoder(ef)
        self.gf_dim = gf_dim            # Dimension of encoder filters in first conv layer. [32]
        self.df_dim = df_dim            # Dimension of decoder filters in first conv layer. [32]
        self.gfc_dim = gfc_dim          # Dimension of gen encoder for for fully connected layer. [512]
        self.dfc_dim = gfc_dim          # Dimension of decoder units for fully connected layer. [512]

        self.nz = nz                    # number of color channels in the input images. For color images this is 3
        self.m_dim = m_dim              # Dimension of camera matrix latent vector [8]
        self.il_dim = il_dim            # Dimension of illumination latent vector [27]
        self.tex_sz = tex_sz            # Texture size

        # encoder
        self.nl_encoder = NLEncoderBlock(self.nz, self.gf_dim)
        self.in_dim = self.nl_encoder.in_dim

        # embedding each component (camera, illustration, shape, texture)
        self.lv_m_layer = NLEmbeddingBlock(self.in_dim, self.gfc_dim // 5, fc_dim=self.m_dim)
        self.lv_il_layer = NLEmbeddingBlock(self.in_dim, self.gfc_dim // 5, fc_dim=self.il_dim)
        self.lv_shape_layer = NLEmbeddingBlock(self.in_dim, self.gfc_dim // 2)
        self.lv_tex_layer = NLEmbeddingBlock(self.in_dim, self.gfc_dim // 2)

        #
        self.albedo_layer = NLAlbedoDecoderBlock(self.gfc_dim//2, self.gf_dim, self.tex_sz)


    def forward(self, x):
        x = self.nl_encoder(x)

        """
        lv_m = self.lv_m_layer(x)
        lv_il = self.lv_il_layer(x)
        lv_shape = self.lv_shape_layer(x)
        """
        lv_tex = self.lv_tex_layer(x)
        albedo = self.albedo_layer(lv_tex)

        return albedo


class Helper():
    def __init__(self, config):

        self.model = Nonlinear3DMM(config)
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def predict(self):
        pass


if __name__ == "__main__":
    import torch.optim as optim

    import torchvision
    import torchvision.transforms as transforms

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    import torch

    dataset = dset.ImageFolder(root="data/",
                               transform=transforms.Compose([
                                   transforms.CenterCrop(224),  # square를 한 후,
                                   transforms.ToTensor()  # Tensor로 바꾸고 (0~1로 자동으로 normalize)
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=2,
                                             shuffle=True)

    nl3dmm = Nonlinear3DMM()
    print(nl3dmm)

    optimizer = torch.optim.Adam(nl3dmm.parameters(), lr=0.0002, betas=(0.5, 0.999))
    lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss().to(device)

    num_epochs = 1
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            tex = nl3dmm(inputs)

            loss = criterion(tex, tex)
            loss.backward()
            optimizer.step()

            # Format batch
