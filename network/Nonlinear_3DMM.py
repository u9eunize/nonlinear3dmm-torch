import time
import os
import re

from network.block import *
from rendering_ops import *
from ops import *
from torch.utils.data import DataLoader
from configure_dataset import NonlinearDataset
from datetime import datetime


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
        self.nl_encoder = Encoder_shasha(self.nz, self.gf_dim, self.gfc_dim // 5, self.gfc_dim // 5, self.gfc_dim // 2, self.gfc_dim // 2, self.m_dim, self.il_dim)
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


class Nonlinear3DMMHelper:

    def __init__(self, losses, device='cpu'):
        dtype = torch.float
        self.device = device

        # TODO parameterize
        self.tex_sz = (192, 224)
        self.img_sz = 224
        self.c_dim = 3
        self.landmark_num = 68
        self.losses = losses
        self.available_losses = list(filter(lambda a: a.endswith("_loss"), dir(self)))

        for loss_name in losses:
            assert loss_name + "_loss" in self.available_losses, loss_name + "_loss is not supported"

        self.shape_loss = "l2"
        self.tex_loss = "l1"

        self.nl_network = Nonlinear3DMM().to(self.device)

        self.uv_tri, self.uv_mask = load_3DMM_tri_2d(with_mask=True)
        self.uv_tri = torch.tensor(self.uv_tri).to(self.device)
        self.uv_mask = torch.tensor(self.uv_mask).to(self.device)

        # Basis
        mu_shape, w_shape = load_Basel_basic('shape')
        mu_exp, w_exp = load_Basel_basic('exp')

        self.mean_shape = torch.tensor(mu_shape + mu_exp, dtype=dtype).to(self.device)
        self.std_shape = torch.tensor(np.tile(np.array([1e4, 1e4, 1e4]), VERTEX_NUM), dtype=dtype).to(self.device)
        # self.std_shape  = np.load('std_shape.npy')

        self.mean_m = torch.tensor(np.load('dataset/mean_m.npy'), dtype=dtype).to(self.device)
        self.std_m = torch.tensor(np.load('dataset/std_m.npy'), dtype=dtype).to(self.device)

        self.w_shape = torch.tensor(w_shape, dtype=dtype).to(self.device)
        self.w_exp = torch.tensor(w_exp, dtype=dtype).to(self.device)

        # generate shape1d
        self.vt2pixel_u, self.vt2pixel_v = load_3DMM_vt2pixel()

        self.vt2pixel_u = torch.tensor(self.vt2pixel_u[:-1], dtype=dtype).to(self.device)
        self.vt2pixel_v = torch.tensor(self.vt2pixel_v[:-1], dtype=dtype).to(self.device)

    def eval(self):
        pass

    def predict(self):
        pass

    def train(self, num_epochs, batch_size, learning_rate, betas):
        nl3dmm = Nonlinear3DMM().to(self.device)

        nl3dmm, start_epoch = self.load(nl3dmm, MODEL_PATH)

        dataset = NonlinearDataset(phase='train')
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

        encoder_optimizer = torch.optim.Adam(nl3dmm.nl_encoder.parameters(), lr=learning_rate, betas=betas)
        global_optimizer = torch.optim.Adam(nl3dmm.parameters(), lr=learning_rate, betas=betas)

        for epoch in range(start_epoch, num_epochs):

            # For each batch in the dataloader
            for idx, samples in enumerate(dataloader, 0):
                loss, g_loss_wlandmark = self.train_step(
                    input_images=samples["image"].to(self.device),
                    input_masks=samples["mask_img"].to(self.device),
                    input_texture_labels=samples["texture"].to(self.device),
                    input_texture_masks=samples["mask"].to(self.device),
                    input_m_labels=samples["m_label"].to(self.device),
                    input_shape_labels=samples["shape_label"].to(self.device),
                    input_albedo_indexes=list(map(lambda a: a.to(self.device), samples["albedo_indices"]))
                )

                if idx % 2 == 0:
                    global_optimizer.zero_grad()
                    loss.backward()
                    global_optimizer.step()
                else:
                    encoder_optimizer.zero_grad()
                    g_loss_wlandmark.backward()
                    encoder_optimizer.step()

                if idx % 50 == 0:
                    print(datetime.now(), end=" ")
                    print(f"[{epoch}] {idx * batch_size}/{len(dataset)} ({(idx * batch_size)/len(dataset) * 100:.2f}%) "
                          f"g_loss: {loss:.6f}, "
                          f"landmark_loss: {g_loss_wlandmark:.6f}")

            self.save(nl3dmm, MODEL_PATH, epoch)

    def train_step(self, input_images, input_masks, input_texture_labels, input_texture_masks,
                   input_m_labels, input_shape_labels, input_albedo_indexes):
        """
        input_albedo_indexes = [x1,y1,x2,y2]
        """

        tic = time.time()
        batch_size = input_images.shape[0]
        # print(time.time() - tic, "start train_step")

        lv_m, lv_il, lv_shape, lv_tex, albedo, shape2d = self.nl_network(input_images)

        # print(time.time() - tic, "after nl_network")

        # calculate shape1d
        bat_sz = shape2d.shape[0]
        shape1d = bilinear_interpolate(shape2d, self.vt2pixel_u, self.vt2pixel_v)
        shape1d = shape1d.view(bat_sz, -1)
        # print(time.time() - tic, "after bilinear_interpolate")

        m_full = lv_m * self.std_m + self.mean_m
        shape_full = shape1d * self.std_shape + self.mean_shape

        shade = generate_shade(lv_il, lv_m, shape1d, self.tex_sz)
        tex = 2.0 * ((albedo + 1.0) / 2.0 * shade) - 1

        g_images, g_images_mask = warp_texture(tex, m_full, shape_full, output_size=self.img_sz)

        #  tf.multiply(input_masks_300W, tf.expand_dims(g_images_300W_mask, -1))
        g_images_mask = input_masks * g_images_mask.unsqueeze(1).repeat(1, 3, 1, 1)
        g_images = g_images * g_images_mask + input_images * (torch.ones_like(g_images_mask) - g_images_mask)

        # landmark
        m_full = lv_m * self.std_m + self.mean_m
        shape_full = shape1d * self.std_shape + self.mean_shape
        m_labels_full = input_m_labels * self.std_m + self.mean_m
        shape_labels_full = input_shape_labels * self.std_shape + self.mean_shape
        landmark_u, landmark_v = compute_landmarks(m_full, shape_full, output_size=self.img_sz)
        landmark_u_labels, landmark_v_labels = compute_landmarks(m_labels_full, shape_labels_full,
                                                                 output_size=self.img_sz)

        # print(time.time() - tic, "after renderer")

        # ---------------- Losses -------------------------
        # ready texture mask
        tex_vis_mask = (~input_texture_labels.eq((torch.ones_like(input_texture_labels) * -1))).float()
        tex_vis_mask = tex_vis_mask * input_texture_masks
        tex_ratio = torch.sum(tex_vis_mask) / (batch_size * self.tex_sz[0] * self.tex_sz[1] * self.c_dim)

        g_loss_shape = 10 * norm_loss(shape1d, input_shape_labels, loss_type=self.shape_loss)
        g_loss_m = 5 * norm_loss(lv_m, input_m_labels, loss_type="l2")

        # print(time.time() - tic, "after ready texture")

        g_loss = g_loss_shape + g_loss_m  # default loss

        kwargs = {
            "batch_size": batch_size,
            "landmark_u": landmark_u,
            "landmark_u_labels": landmark_u_labels,
            "landmark_v": landmark_v,
            "landmark_v_labels": landmark_v_labels,
            "shade": shade,
            "input_images": input_images,
            "g_images": g_images,
            "g_images_mask": g_images_mask,
            "input_texture_labels": input_texture_labels,
            "tex": tex,
            "tex_vis_mask": tex_vis_mask,
            "tex_ratio": tex_ratio,
            "shape2d": shape2d,
            "albedo": albedo,
            "input_albedo_indexes": input_albedo_indexes
        }
        if "reconstruction" not in self.losses and "texture" not in self.losses:
            self.losses.append("texture")

        g_loss_with_landmark = 0
        for loss_name in self.losses:
            loss_fn = self.__getattribute__(loss_name+"_loss")
            result = loss_fn(**kwargs)

            if loss_name == "landmark":
                g_loss_with_landmark = result
            else:
                g_loss += result

        g_loss_with_landmark = g_loss_with_landmark + g_loss
        return g_loss, g_loss_with_landmark

    def landmark_loss(self, batch_size, landmark_u, landmark_u_labels, landmark_v, landmark_v_labels, **kwargs):
        landmark_mse_mean = (
                torch.mean(norm_loss(landmark_u, landmark_u_labels, loss_type="l2", reduce_mean=False)) +
                torch.mean(norm_loss(landmark_v, landmark_v_labels, loss_type="l2", reduce_mean=False)))
        return landmark_mse_mean / self.landmark_num / batch_size / 50

    def batchwise_white_shading_loss(self, shade, **kwargs):
        uv_mask = self.uv_mask.unsqueeze(0).unsqueeze(0)
        mean_shade = torch.mean(shade * uv_mask, dim=[0, 2, 3]) * 16384 / 10379
        g_loss_white_shading = 10 * norm_loss(mean_shade, 0.99 * torch.ones(mean_shade.shape).float().to(self.device), loss_type="l2")
        return g_loss_white_shading

    def reconstruction_loss(self, batch_size, input_images, g_images, g_images_mask, **kwargs):
        g_loss_recon = 10 * (norm_loss(g_images, input_images, loss_type=self.tex_loss) /
                             (torch.sum(g_images_mask) / (batch_size * self.img_sz * self.img_sz)))
        return g_loss_recon

    def texture_loss(self, input_texture_labels, tex, tex_vis_mask, tex_ratio, **kwargs):
        g_loss_texture = norm_loss(tex, input_texture_labels, mask=tex_vis_mask,
                                   loss_type=self.tex_loss) / tex_ratio
        return g_loss_texture

    def smoothness_loss(self, shape2d, **kwargs):
        g_loss_smoothness = 1000 * norm_loss((shape2d[:, :, :-2, 1:-1] + shape2d[:, :, 2:, 1:-1] +
                                              shape2d[:, :, 1:-1, :-2] + shape2d[:, :, 1:-1, 2:]) / 4.0,
                                             shape2d[:, :, 1:-1, 1:-1], loss_type=self.shape_loss)
        return g_loss_smoothness

    def symmetry_loss(self, albedo, **kwargs):
        albedo_flip = torch.flip(albedo, dims=[3])
        flip_diff = torch.max(torch.abs(albedo - albedo_flip), torch.ones_like(albedo) * 0.05)
        g_loss_symmetry = norm_loss(flip_diff, torch.zeros_like(flip_diff), loss_type=self.tex_loss)
        return g_loss_symmetry

    def const_albedo_loss(self, albedo, input_albedo_indexes, **kwargs):

        albedo_1 = get_pixel_value(albedo, input_albedo_indexes[0], input_albedo_indexes[1])
        albedo_2 = get_pixel_value(albedo, input_albedo_indexes[2], input_albedo_indexes[3])
        diff = torch.max(torch.abs(albedo_1 - albedo_2), torch.ones_like(albedo_1) * 0.05)
        g_loss_albedo_const = 5 * norm_loss(diff, torch.zeros_like(diff), loss_type=self.tex_loss)
        return g_loss_albedo_const

    def const_local_albedo_loss(self, input_texture_labels, tex_vis_mask, albedo, **kwargs):
        chromaticity = (input_texture_labels + 1) / 2.0
        chromaticity = torch.div(chromaticity, torch.sum(chromaticity, dim=1, keepdim=True) + 1e-6)

        u_diff = -15 * torch.norm(chromaticity[:, :, :-1, :] - chromaticity[:, :, 1:, :], dim=1, keepdim=True)
        w_u = (torch.exp(u_diff) * tex_vis_mask[:, :, :-1, :]).detach()
        u_albedo_norm = norm_loss(albedo[:, :, :-1, :], albedo[:, :, 1:, :],
                                  loss_type="l2,1", p=0.8, reduce_mean=False) * w_u
        loss_local_albedo_u = torch.mean(u_albedo_norm) / torch.sum(w_u + 1e-6)

        v_diff = -15 * torch.norm(chromaticity[:, :, :, :-1] - chromaticity[:, :, :, 1:], dim=1, keepdim=True)
        w_v = (torch.exp(v_diff) * tex_vis_mask[:, :, :, :-1]).detach()
        v_albedo_norm = norm_loss(albedo[:, :, :, :-1], albedo[:, :, :, 1:],
                                  loss_type="l2,1", p=0.8, reduce_mean=False) * w_v
        loss_local_albedo_v = torch.mean(v_albedo_norm) / torch.sum(w_v + 1e-6)

        return (loss_local_albedo_u + loss_local_albedo_v) * 10

    def save(self, model, path, number):
        if not os.path.isdir(path):
            os.mkdir(path)
        if not os.path.isdir(self.get_checkpoint_dir(path, number)):
            os.mkdir(self.get_checkpoint_dir(path, number))
        torch.save(model.state_dict(), self.get_checkpoint_name(path, number))

    def load(self, model, path, number=-1):
        if number == -1:
            number = 0
            if not os.path.isdir(path):
                os.mkdir(path)
            for ckpt_name in os.listdir(path):
                number = max(number, max(map(int, re.findall(r"\d+", ckpt_name))))
        ckpt_name = self.get_checkpoint_name(path, number)

        if not os.path.isfile(ckpt_name):
            print(f"no checkpoint! path: {path}")
            return 0
        print(f"loading {ckpt_name}...")
        model.load_state_dict(torch.load(ckpt_name))
        model = model.train(True)
        print("DONE")
        return model, number + 1

    def get_checkpoint_dir(self, path, number):
        return os.path.join(path, f"ckpt_{number}")

    def get_checkpoint_name(self, path, number):
        return os.path.join(f"{self.get_checkpoint_dir(path, number)}", f"model_ckpt_{number}.pt")



if __name__ == "__main__":
    helper = Nonlinear3DMMHelper([
        'batchwise_white_shading',
        'const_albedo',
        'const_local_albedo',
        'landmark',
        'reconstruction',
        'smoothness',
        'symmetry'
    ], device='cuda' if torch.cuda.is_available() else 'cpu')
    helper.train(
        num_epochs=10,
        batch_size=20,
        learning_rate=0.0002,
        betas=(0.5, 0.999)
    )


