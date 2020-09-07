from network.block import *
from rendering_ops import *
from ops import *


TRI_NUM = 105840
VERTEX_NUM = 53215
CONST_PIXELS_NUM = 20
MODEL_PATH = "./checkpoint"


class Nonlinear3DMM(nn.Module):
    def __init__(self, gf_dim=32, df_dim=32, gfc_dim=512, dfc_dim=512, nz=3, m_dim=8, il_dim=27,
                 tex_sz=(192, 224), img_sz=224):
        super(Nonlinear3DMM, self).__init__()
        dtype = torch.float

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
        self.img_sz = img_sz

        # Basis
        mu_shape, w_shape = load_Basel_basic('shape')
        mu_exp, w_exp = load_Basel_basic('exp')

        self.mean_shape = torch.tensor(mu_shape + mu_exp, dtype=dtype)
        self.std_shape = torch.tensor(np.tile(np.array([1e4, 1e4, 1e4]), VERTEX_NUM), dtype=dtype)
        # self.std_shape  = np.load('std_shape.npy')

        self.mean_m = torch.tensor(np.load('dataset/mean_m.npy'), dtype=dtype)
        self.std_m = torch.tensor(np.load('dataset/std_m.npy'), dtype=dtype)

        self.w_shape = torch.tensor(w_shape, dtype=dtype)
        self.w_exp = torch.tensor(w_exp, dtype=dtype)


        # encoder
        '''
        self.nl_encoder = NLEncoderBlock(self.nz, self.gf_dim)
        self.in_dim = self.nl_encoder.in_dim

        # embedding each component (camera, illustration, shape, texture)
        self.lv_m_layer = NLEmbeddingBlock(self.in_dim, self.gfc_dim // 5, fc_dim=self.m_dim)
        self.lv_il_layer = NLEmbeddingBlock(self.in_dim, self.gfc_dim // 5, fc_dim=self.il_dim)
        self.lv_shape_layer = NLEmbeddingBlock(self.in_dim, self.gfc_dim // 2)
        self.lv_tex_layer = NLEmbeddingBlock(self.in_dim, self.gfc_dim // 2)
        '''

        ###################################### encoder shasha
        self.nl_encoder = Encoder(self.nz, self.gf_dim, self.gfc_dim // 5, self.gfc_dim // 5, self.gfc_dim // 2, self.gfc_dim // 2, self.m_dim, self.il_dim)
        self.in_dim = self.nl_encoder.in_dim

        self.albedo_gen = NLAlbedoDecoderBlock(self.gfc_dim//2, self.gf_dim, self.tex_sz)
        self.shape_gen = NLShapeDecoderBlock(self.gfc_dim//2, self.gf_dim, self.gfc_dim, self.tex_sz)

    def forward(self, input_images):
        '''
        # encoding
        encoder_out = self.nl_encoder(input_images)

        # latent vector embedding m, il, shape, tex
        lv_m = self.lv_m_layer(encoder_out)
        lv_il = self.lv_il_layer(encoder_out)
        lv_shape = self.lv_shape_layer(encoder_out)
        lv_tex = self.lv_tex_layer(encoder_out)
        '''
        lv_m, lv_il, lv_shape, lv_tex = self.nl_encoder(input_images)

        # generate albedo and shape
        albedo = self.albedo_gen(lv_tex)
        shape2d = self.shape_gen(lv_shape)

        return lv_m, lv_il, lv_shape, lv_tex, albedo, shape2d

    def to(self, device, *args, **kwargs):
        ret = super(Nonlinear3DMM, self).to(device, *args, **kwargs)

        self.mean_shape = self.mean_shape.to(device)
        self.std_shape = self.std_shape.to(device)

        self.mean_m = self.mean_m.to(device)
        self.std_m = self.std_m.to(device)

        self.w_shape = self.w_shape.to(device)
        self.w_exp = self.w_exp.to(device)
        return ret

