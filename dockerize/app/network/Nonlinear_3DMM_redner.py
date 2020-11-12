from network.block import *
from renderer.rendering_ops import *
from os.path import join


class Nonlinear3DMM_redner(nn.Module):
    def __init__(self, gf_dim=32, df_dim=32, gfc_dim=512, dfc_dim=512, nz=3, trans_dim=3, rot_dim=3, il_dim=3,
                 tex_sz=(256, 256), img_sz=224):
        super(Nonlinear3DMM_redner, self).__init__()

        h, w = CFG.texture_size
        vt2pixel_u, vt2pixel_v = torch.split(torch.tensor(np.load('deep3d/BFM_uvmap.npy')), (1, 1), dim=-1)
        vt2pixel_v = torch.ones_like(vt2pixel_v) - vt2pixel_v
        self.vt2pixel_u, self.vt2pixel_v = vt2pixel_u * h, vt2pixel_v * w
        

        # naming from https://gist.github.com/EderSantana/9b0d5fb309d775b995d5236c32238349
        # TODO: gen(gf)->encoder(ef)
        self.gf_dim = gf_dim            # Dimension of encoder filters in first conv layer. [32]
        self.df_dim = df_dim            # Dimension of decoder filters in first conv layer. [32]
        self.gfc_dim = gfc_dim          # Dimension of gen encoder for for fully connected layer. [512]
        self.dfc_dim = gfc_dim          # Dimension of decoder units for fully connected layer. [512]

        self.nz = nz                    # number of color channels in the input images. For color images this is 3
        self.trans_dim = trans_dim      # Dimension of camera matrix latent vector [3]
        self.rot_dim = rot_dim          # Dimension of camera matrix latent vector [3]
        self.il_dim = il_dim            # Dimension of illumination latent vector [3]
        self.tex_sz = tex_sz            # Texture size
        self.img_sz = img_sz
        
        ###################################### encoder
        self.nl_encoder = Encoder(self.nz, self.gf_dim, self.gfc_dim // 5, self.gfc_dim // 5, self.gfc_dim // 2,
                                  self.gfc_dim // 2, self.gfc_dim // 2, self.m_dim, self.il_dim)

        self.in_dim = self.nl_encoder.in_dim

        self.albedo_dec = NLDecoderBlock(self.gfc_dim // 2, self.gf_dim, self.gf_dim * 10, self.tex_sz)
        self.albedo_gen_base = NLDecoderTailBlock(self.gf_dim, self.nz, self.gf_dim, additional_layer=True)
        self.albedo_gen_comb = NLDecoderTailBlock(self.gf_dim, self.nz, self.gf_dim, additional_layer=True)

        self.shape_dec = NLDecoderBlock(self.gfc_dim // 2, self.gf_dim, self.gfc_dim, self.tex_sz)
        self.shape_gen_base = NLDecoderTailBlock(self.gf_dim, self.nz, self.gf_dim, additional_layer=True)
        self.shape_gen_comb = NLDecoderTailBlock(self.gf_dim, self.nz, self.gf_dim, additional_layer=True)

    def forward(self, input_images):
        batch_size = input_images.shape[0]
        vt2pixel_u = self.vt2pixel_u.view((1, 1, -1)).repeat(batch_size, 1, 1)
        vt2pixel_v = self.vt2pixel_v.view((1, 1, -1)).repeat(batch_size, 1, 1)

        lv_trans, lv_rot, lv_il, lv_shape, lv_tex = self.nl_encoder(input_images)

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

        return dict(
            lv_trans=lv_trans,
            lv_rot=lv_rot,
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

    def make_1d(self, decoder_2d_result, vt2pixel_u, vt2pixel_v):
        batch_size = decoder_2d_result.shape[0]
        decoder_1d_result = bilinear_sampler_torch(decoder_2d_result, vt2pixel_u, vt2pixel_v)
        decoder_1d_result = decoder_1d_result.view(batch_size, -1)

        return decoder_1d_result

    def to(self, device, *args, **kwargs):
        ret = super(Nonlinear3DMM_redner, self).to(device, *args, **kwargs)
        self.vt2pixel = self.vt2pixel.to(device)

        return ret

