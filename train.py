from network.Nonlinear_3DMM import Nonlinear3DMM
from configure_dataset import *
from datetime import datetime
from ops import *
from rendering_ops import *
import time
import re
from torch.utils.tensorboard import SummaryWriter




MODEL_PATH = "./checkpoint"



class Nonlinear3DMMHelper:

    def __init__(self, losses, device='cpu'):
        dtype = torch.float
        self.device = device
        self.writer = SummaryWriter("runs/test_writer")

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

        # for log
        self.global_step = 0
        self.reconstruction_loss_input = None
        self.reconstruction_loss_generate = None
        self.texture_loss_input = None
        self.texture_loss_generate = None

    def eval(self):
        pass

    def predict(self):
        pass

    def train(self, num_epochs, batch_size, learning_rate, betas):
        nl3dmm = Nonlinear3DMM().to(self.device)

        nl3dmm, start_epoch = self.load(nl3dmm, MODEL_PATH)

        dataset = NonlinearDataset(phase='train')
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

        _, samples = next(enumerate(dataloader, 0))

        self.writer.add_graph(nl3dmm, samples["image"].to(self.device))

        encoder_optimizer = torch.optim.Adam(nl3dmm.nl_encoder.parameters(), lr=learning_rate, betas=betas)
        global_optimizer = torch.optim.Adam(nl3dmm.parameters(), lr=learning_rate, betas=betas)

        self.global_step = start_epoch * len(dataset)

        global_loss = 0
        global_loss_with_landmark = 0
        for epoch in range(start_epoch, num_epochs):
            # For each batch in the dataloader
            for idx, samples in enumerate(dataloader, 0):
                global_loss, global_loss_with_landmark = self.train_step(
                    input_images=samples["image"].to(self.device),
                    input_masks=samples["mask_img"].to(self.device),
                    input_texture_labels=samples["texture"].to(self.device),
                    input_texture_masks=samples["mask"].to(self.device),
                    input_m_labels=samples["m_label"].to(self.device),
                    input_shape_labels=samples["shape_label"].to(self.device),
                    input_albedo_indexes=list(map(lambda a: a.to(self.device), samples["albedo_indices"]))
                )

                self.writer.add_scalar("global_loss", global_loss, self.global_step)
                self.writer.add_scalar("global_loss_with_landmark", global_loss_with_landmark, self.global_step)
                self.writer.flush()

                if idx % 2 == 0:
                    global_optimizer.zero_grad()
                    global_loss.backward()
                    global_optimizer.step()
                else:
                    encoder_optimizer.zero_grad()
                    global_loss_with_landmark.backward()
                    encoder_optimizer.step()


                if idx % 10 == 0:
                    print(datetime.now(), end=" ")
                    print(f"[{epoch}] {idx * batch_size}/{len(dataset)} "
                          f"({(idx * batch_size)/(len(dataset)) * 100:.2f}%) "
                          f"g_loss: {global_loss:.6f}, "
                          f"landmark_loss: {global_loss_with_landmark:.6f}")

                self.global_step += 1

            self.writer.add_scalar("global_loss_per_epoch", global_loss, epoch)
            self.writer.add_scalar("global_landmark_loss_per_epoch", global_loss_with_landmark, epoch)
            if self.reconstruction_loss_generate is not None:
                self.writer.add_images("reconstruction_loss_input_per_epoch",
                                       self.reconstruction_loss_input, epoch)
                self.writer.add_images("reconstruction_loss_generate_per_epoch",
                                       self.reconstruction_loss_generate, epoch)
            if self.texture_loss_input is not None:
                self.writer.add_images("texture_loss_input_per_epoch",
                                       self.texture_loss_input, epoch)
                self.writer.add_images("texture_loss_generate_per_epoch",
                                       self.texture_loss_generate, epoch)

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
        vt2pixel_u = self.vt2pixel_u.view((1, 1, -1)).repeat(batch_size, 1, 1)
        vt2pixel_v = self.vt2pixel_v.view((1, 1, -1)).repeat(batch_size, 1, 1)

        shape1d = bilinear_sampler_torch(shape2d, vt2pixel_u, vt2pixel_v)
        shape1d = shape1d.view(bat_sz, -1)
        # print(time.time() - tic, "after bilinear_interpolate")

        m_full = lv_m * self.std_m + self.mean_m
        shape_full = shape1d * self.std_shape + self.mean_shape

        # shade = generate_shade_torch(lv_il, lv_m, shape1d, self.tex_sz)
        shade = generate_shade_torch(lv_il, m_full, shape1d, self.tex_sz)
        tex = 2.0 * ((albedo + 1.0) / 2.0 * shade) - 1

        g_images, g_images_mask = warp_texture_torch(tex, m_full, shape_full, output_size=self.img_sz)

        #  tf.multiply(input_masks_300W, tf.expand_dims(g_images_300W_mask, -1))
        g_images_mask = input_masks * g_images_mask.unsqueeze(1).repeat(1, 3, 1, 1)
        g_images = g_images * g_images_mask + input_images * (torch.ones_like(g_images_mask) - g_images_mask)

        # landmark
        m_full = lv_m * self.std_m + self.mean_m
        shape_full = shape1d * self.std_shape + self.mean_shape
        m_labels_full = input_m_labels * self.std_m + self.mean_m
        shape_labels_full = input_shape_labels * self.std_shape + self.mean_shape
        landmark_u, landmark_v = compute_landmarks_torch(m_full, shape_full, output_size=self.img_sz)
        landmark_u_labels, landmark_v_labels = compute_landmarks_torch(m_labels_full, shape_labels_full,
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

        self.writer.add_scalar("g_loss_shape", g_loss_shape, self.global_step)
        self.writer.add_scalar("g_loss_m", g_loss_m, self.global_step)

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
        landmark_loss = landmark_mse_mean / self.landmark_num / batch_size / 50

        self.writer.add_scalar("landmark_loss", landmark_loss, self.global_step)
        return landmark_loss

    def batchwise_white_shading_loss(self, shade, **kwargs):
        uv_mask = self.uv_mask.unsqueeze(0).unsqueeze(0)
        mean_shade = torch.mean(shade * uv_mask, dim=[0, 2, 3]) * 16384 / 10379
        g_loss_white_shading = 10 * norm_loss(mean_shade, 0.99 * torch.ones(mean_shade.shape).float().to(self.device), loss_type="l2")

        self.writer.add_scalar("g_loss_white_shading", g_loss_white_shading, self.global_step)
        return g_loss_white_shading

    def reconstruction_loss(self, batch_size, input_images, g_images, g_images_mask, **kwargs):
        g_loss_recon = 10 * (norm_loss(g_images, input_images, loss_type=self.tex_loss) /
                             (torch.sum(g_images_mask) / (batch_size * self.img_sz * self.img_sz)))

        self.reconstruction_loss_input = input_images
        self.reconstruction_loss_generate = g_images
        self.writer.add_scalar("reconstruction_loss", g_loss_recon, self.global_step)
        return g_loss_recon

    def texture_loss(self, input_texture_labels, tex, tex_vis_mask, tex_ratio, **kwargs):
        g_loss_texture = norm_loss(tex, input_texture_labels, mask=tex_vis_mask,
                                   loss_type=self.tex_loss) / tex_ratio

        self.writer.add_scalar("texture_loss", g_loss_texture, self.global_step)
        self.texture_loss_input = input_texture_labels * tex_vis_mask
        self.texture_loss_generate = tex * tex_vis_mask
        return g_loss_texture

    def smoothness_loss(self, shape2d, **kwargs):
        g_loss_smoothness = 10 * norm_loss((shape2d[:, :, :-2, 1:-1] + shape2d[:, :, 2:, 1:-1] +
                                            shape2d[:, :, 1:-1, :-2] + shape2d[:, :, 1:-1, 2:]) / 4.0,
                                            shape2d[:, :, 1:-1, 1:-1], loss_type=self.shape_loss)

        self.writer.add_scalar("g_loss_smoothness", g_loss_smoothness, self.global_step)
        return g_loss_smoothness

    def symmetry_loss(self, albedo, **kwargs):
        albedo_flip = torch.flip(albedo, dims=[3])
        flip_diff = torch.max(torch.abs(albedo - albedo_flip), torch.ones_like(albedo) * 0.05)
        g_loss_symmetry = norm_loss(flip_diff, torch.zeros_like(flip_diff), loss_type=self.tex_loss)

        self.writer.add_scalar("g_loss_symmetry", g_loss_symmetry, self.global_step)
        return g_loss_symmetry

    def const_albedo_loss(self, albedo, input_albedo_indexes, **kwargs):

        albedo_1 = get_pixel_value(albedo, input_albedo_indexes[0], input_albedo_indexes[1])
        albedo_2 = get_pixel_value(albedo, input_albedo_indexes[2], input_albedo_indexes[3])
        diff = torch.max(torch.abs(albedo_1 - albedo_2), torch.ones_like(albedo_1) * 0.05)
        g_loss_albedo_const = 5 * norm_loss(diff, torch.zeros_like(diff), loss_type=self.tex_loss)

        self.writer.add_scalar("g_loss_albedo_const", g_loss_albedo_const, self.global_step)
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
        loss_local_albedo = (loss_local_albedo_u + loss_local_albedo_v) * 10

        self.writer.add_scalar("loss_local_albedo", loss_local_albedo, self.global_step)
        return loss_local_albedo

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
            return model, 0
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
        'texture',
        'smoothness',
        'symmetry'
    ], device='cuda' if torch.cuda.is_available() else 'cpu')
    helper.train(
        num_epochs=50,
        batch_size=15,
        learning_rate=0.0002,
        betas=(0.5, 0.999)
    )
