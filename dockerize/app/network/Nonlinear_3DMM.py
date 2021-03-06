from network.block import *
from renderer.rendering_ops import *
import utils
from os.path import join


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

        # generate shape1d
        self.vt2pixel_u, self.vt2pixel_v = utils.load_3DMM_vt2pixel()

        self.vt2pixel_u = torch.tensor(self.vt2pixel_u[:-1], dtype=dtype)
        self.vt2pixel_v = torch.tensor(self.vt2pixel_v[:-1], dtype=dtype)

        # encoder
        self.nl_encoder = Encoder(self.nz, self.gf_dim, self.gfc_dim // 5, self.gfc_dim // 5, self.gfc_dim // 2,
                                  self.gfc_dim // 2, self.gfc_dim // 2, self.m_dim, self.il_dim)

        self.in_dim = self.nl_encoder.in_dim

        # decoder
        self.albedo_dec = NLDecoderBlock(self.gfc_dim // 2, self.gf_dim, self.gf_dim * 10, self.tex_sz)
        self.albedo_gen_base = NLDecoderTailBlock(self.gf_dim, self.nz, self.gf_dim, additional_layer=False)
        self.albedo_gen_comb = NLDecoderTailBlock(self.gf_dim, self.nz, self.gf_dim, additional_layer=False)

        self.shape_dec = NLDecoderBlock(self.gfc_dim // 2, self.gf_dim, self.gfc_dim, self.tex_sz)
        self.shape_gen_base = NLDecoderTailBlock(self.gf_dim, self.nz, self.gf_dim, additional_layer=False)
        self.shape_gen_comb = NLDecoderTailBlock(self.gf_dim, self.nz, self.gf_dim, additional_layer=False)

        self.exp_dec = NLDecoderBlock(self.gfc_dim // 2, self.gf_dim, self.gfc_dim, self.tex_sz)
        self.exp_gen_base = NLDecoderTailBlock(self.gf_dim, self.nz, self.gf_dim, additional_layer=False)
        self.exp_gen_comb = NLDecoderTailBlock(self.gf_dim, self.nz, self.gf_dim, additional_layer=False)

        # self.exp_dec = NLDecoderBlock(self.gfc_dim // 2, self.gf_dim, self.gfc_dim, self.tex_sz)
        # self.exp_gen = NLDecoderTailBlock(self.gf_dim, self.nz, self.gf_dim, additional_layer=False)

    def forward(self, input_images):
        batch_size = input_images.shape[0]
        vt2pixel_u = self.vt2pixel_u.view((1, 1, -1)).repeat(batch_size, 1, 1)
        vt2pixel_v = self.vt2pixel_v.view((1, 1, -1)).repeat(batch_size, 1, 1)

        lv_m, lv_il, lv_shape, lv_tex = self.nl_encoder(input_images)

        # albedo
        albedo_dec = self.albedo_dec(lv_tex)
        albedo_base = self.albedo_gen_base(albedo_dec)
        albedo_comb = self.albedo_gen_comb(albedo_dec)
        albedo_res = albedo_comb - albedo_base

        # shape
        shape_dec = self.shape_dec(lv_tex)
        shape_2d_base = self.shape_gen_base(shape_dec)
        shape_2d_comb = self.shape_gen_comb(shape_dec)

        shape_1d_base = self.make_1d(shape_2d_base, vt2pixel_u, vt2pixel_v)
        shape_1d_comb = self.make_1d(shape_2d_comb, vt2pixel_u, vt2pixel_v)

        shape_2d_res = shape_2d_base - shape_2d_comb
        shape_1d_res = shape_1d_comb - shape_1d_base

        # exp

        exp_dec = self.exp_dec(lv_tex)
        exp_2d_base = self.exp_gen_base(exp_dec)
        exp_2d_comb = self.exp_gen_comb(exp_dec)

        exp_1d_base = self.make_1d(exp_2d_base, vt2pixel_u, vt2pixel_v)
        exp_1d_comb = self.make_1d(exp_2d_comb, vt2pixel_u, vt2pixel_v)

        exp_2d_res = exp_2d_base - exp_2d_comb
        exp_1d_res = exp_1d_comb - exp_1d_base

        ret = dict(
            lv_m=lv_m,
            lv_il=lv_il,

            albedo_base=albedo_base,
            albedo_comb=albedo_comb,
            albedo_res=albedo_res,

            shape_2d_base=shape_2d_base,
            shape_2d_comb=shape_2d_comb,
            shape_1d_base=shape_1d_base,
            shape_1d_comb=shape_1d_comb,
            shape_2d_res=shape_2d_res,
            shape_1d_res=shape_1d_res,
        )
        ret.update(dict(
            exp_2d_base=exp_2d_base,
            exp_2d_comb=exp_2d_comb,
            exp_1d_base=exp_1d_base,
            exp_1d_comb=exp_1d_comb,
            exp_2d_res=exp_2d_res,
            exp_1d_res=exp_1d_res,
        ))

        return ret

    def make_1d(self, decoder_2d_result, vt2pixel_u, vt2pixel_v):
        batch_size = decoder_2d_result.shape[0]
        decoder_1d_result = bilinear_sampler_torch(decoder_2d_result, vt2pixel_u, vt2pixel_v)
        decoder_1d_result = decoder_1d_result.view(batch_size, -1)
        return decoder_1d_result

    def to(self, device, *args, **kwargs):
        ret = super(Nonlinear3DMM, self).to(device, *args, **kwargs)

        self.vt2pixel_u = self.vt2pixel_u.to(device)
        self.vt2pixel_v = self.vt2pixel_v.to(device)
        return ret

