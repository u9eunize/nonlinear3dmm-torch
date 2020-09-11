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

    def __init__(self, losses: list, device='cpu', name="test_writer", using_default_loss=True):
        dtype = torch.float
        self.device = device
        self.name = name
        self.writer = SummaryWriter("runs/" +  self.name)
        self.default_loss = ["shape", "m"]
        self.state_file_root_name = os.path.join(MODEL_PATH, self.name)

        # TODO parameterize
        self.tex_sz = (192, 224)
        self.img_sz = 224
        self.c_dim = 3
        self.landmark_num = 68
        self.losses = losses
        self.available_losses = list(filter(lambda a: a.endswith("_loss"), dir(self)))
        self.shape_loss_name = "l2"
        self.tex_loss_name = "l1"

        print("**** using ****")
        if using_default_loss:
            self.losses += self.default_loss

        if "reconstruction" not in self.losses and "texture" not in self.losses:
            self.losses.append("texture")

        for loss_name in losses:
            assert loss_name + "_loss" in self.available_losses, loss_name + "_loss is not supported"
            print(loss_name)

        self.net = Nonlinear3DMM()

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

        # for log
        self.global_step = 0
        self.reconstruction_loss_input = None
        self.reconstruction_loss_generate = None
        self.texture_loss_input = None
        self.texture_loss_generate = None
        self.images_input = None
        self.images_generate = None
        self.image_masks_input = None
        self.image_masks_generate = None

    def eval(self):
        pass

    def predict(self):
        pass

    def train(self, num_epochs, batch_size, learning_rate, betas, start_epoch=-1, step_log=100):
        dataset = NonlinearDataset(phase='train')
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=1)

        self.net.to(self.device)

        encoder_optimizer = torch.optim.Adam(self.net.nl_encoder.parameters(), lr=learning_rate, betas=betas)
        global_optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, betas=betas)

        self.net, global_optimizer, encoder_optimizer, start_epoch = self.load(
            self.net, global_optimizer, encoder_optimizer, path=self.state_file_root_name, start_epoch=start_epoch
        )

        self.net.to(self.device)

        _, samples = next(enumerate(dataloader, 0))
        self.writer.add_graph(self.net, samples["image"].to(self.device))

        self.global_step = start_epoch * len(dataloader)

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
                    # print([p.grad for p in self.nl_network.parameters()])
                    global_loss.backward()
                    global_optimizer.step()
                    # print([(np.max(p.cpu().detach().numpy())) for p in
                    #       self.net.nl_encoder.m.parameters()])
                    # print([((torch.max(p.grad).cpu().numpy().item(),
                    #          torch.min(p.grad).cpu().numpy().item()) if p.grad is not None else "None") for p in
                    #        self.nl_network.parameters()])

                else:
                    encoder_optimizer.zero_grad()
                    # print([p.grad for p in self.nl_network.parameters()])
                    global_loss_with_landmark.backward()
                    encoder_optimizer.step()
                    # print([((torch.max(p.grad).cpu().numpy().item(),
                    #          torch.min(p.grad).cpu().numpy().item()) if p.grad is not None else "None") for p in
                    #        self.nl_network.parameters()])

                print(datetime.now(), end=" ")
                print(f"[{epoch}, {idx:04d}] {idx * batch_size}/{len(dataset)} "
                      f"({(idx * batch_size)/(len(dataset)) * 100:.2f}%) "
                      f"g_loss: {global_loss:.6f}, "
                      f"landmark_loss: {global_loss_with_landmark:.6f}")

                if idx % step_log == 0:

                    #self.add_to_log("global_loss", global_loss)
                    #self.add_to_log("global_loss_with_landmark", global_loss_with_landmark)
                    if self.image_masks_input is not None:
                        self.writer.add_images("image_masks_input_step",
                                               self.image_masks_input, self.global_step)
                        self.writer.add_images("image_masks_generate_step",
                                               self.image_masks_generate.unsqueeze(1), self.global_step)
                    if self.images_input is not None:
                        self.writer.add_images("images_input_step",
                                               self.images_input, self.global_step)
                        self.writer.add_images("images_generate_step",
                                               self.images_generate, self.global_step)
                    if self.texture_loss_generate is not None:
                        self.writer.add_images("texture_loss_input_per_step",
                                               self.texture_loss_input, self.global_step)
                        self.writer.add_images("texture_loss_generate_per_step",
                                               self.texture_loss_generate, self.global_step)

                self.global_step += 1

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

            self.save(self.net, global_optimizer, encoder_optimizer, epoch, self.state_file_root_name)

    def renderer(self, lv_m, lv_il, albedo, shape2d, shape1d, inputs):
        batch_size = shape2d.shape[0]

        input_masks = inputs["input_masks"]
        input_images = inputs["input_images"]
        input_texture_masks = inputs["input_texture_masks"]
        input_texture_labels = inputs["input_texture_labels"]

        m_full = lv_m * self.std_m + self.mean_m
        shape_full = shape1d * self.std_shape + self.mean_shape

        shade = generate_shade_torch(lv_il, m_full, shape_full, self.tex_sz)
        tex = 2.0 * ((albedo + 1.0) / 2.0 * shade) - 1.0

        tex_vis_mask = (~input_texture_labels.eq((torch.ones_like(input_texture_labels) * -1))).float()
        tex_vis_mask = tex_vis_mask * input_texture_masks
        tex_ratio = torch.sum(tex_vis_mask) / (batch_size * self.tex_sz[0] * self.tex_sz[1] * self.c_dim)

        g_images, g_images_mask = warp_texture_torch(tex, m_full, shape_full, output_size=self.img_sz)

        self.image_masks_input = input_masks
        self.image_masks_generate = g_images_mask
        self.images_input = input_images
        self.images_generate = g_images

        g_images_mask = input_masks * g_images_mask.unsqueeze(1).repeat(1, 3, 1, 1)
        g_images = g_images * g_images_mask + input_images * (torch.ones_like(g_images_mask) - g_images_mask)

        param_dict = {
            "shade": shade,
            "tex": tex,
            "tex_vis_mask": tex_vis_mask,
            "tex_ratio": tex_ratio,

            "g_images": g_images,
            "g_images_mask": g_images_mask,
        }
        return param_dict

    def loss_calculation(self,  **kwargs):
        g_loss_with_landmark = 0
        g_loss = 0

        for loss_name in self.losses:
            loss_fn = self.__getattribute__(loss_name+"_loss")
            if not hasattr(loss_fn, '__call__'):
                continue
            result = loss_fn(**kwargs)

            if loss_name == "landmark":
                g_loss_with_landmark = result
            else:
                g_loss += result

        g_loss_with_landmark = g_loss_with_landmark + g_loss
        return g_loss, g_loss_with_landmark

    def train_step(self, **inputs):
        """
        input_albedo_indexes = [x1,y1,x2,y2]
        """
        input_images = inputs["input_images"]
        batch_size = input_images.shape[0]

        loss_param = {"batch_size": batch_size}

        lv_m, lv_il, lv_shape, lv_tex, albedo, shape2d, shape1d = self.net(input_images)
        renderer_dict = self.renderer(lv_m, lv_il, albedo, shape2d, shape1d, inputs)

        network_result = {
            "lv_m": lv_m,
            "lv_il": lv_il,
            "lv_shape": lv_shape,
            "lv_tex": lv_tex,
            "albedo": albedo,
            "shape2d": shape2d,
            "shape1d": shape1d
        }

        loss_param.update(inputs)
        loss_param.update(network_result)
        loss_param.update(renderer_dict)

        return self.loss_calculation(**loss_param)

    def landmark_calculation(self, mv, sv):
        m_full = mv * self.std_m + self.mean_m
        shape_full = sv * self.std_shape + self.mean_shape

        landmark_u, landmark_v = compute_landmarks_torch(m_full, shape_full, output_size=self.img_sz)
        return landmark_u, landmark_v

    def shape_loss(self, shape1d, input_shape_labels, **kwargs):
        g_loss_shape = 10 * norm_loss(shape1d, input_shape_labels, loss_type=self.shape_loss_name)
        self.writer.add_scalar("g_loss_shape", g_loss_shape, self.global_step)
        return g_loss_shape

    def m_loss(self, lv_m, input_m_labels, **kwargs):
        g_loss_m = 5 * norm_loss(lv_m, input_m_labels, loss_type="l1")
        self.writer.add_scalar("g_loss_m", g_loss_m, self.global_step)
        return g_loss_m

    def landmark_loss(self, batch_size, lv_m, shape1d, input_m_labels, input_shape_labels, **kwargs):
        landmark_u, landmark_v = self.landmark_calculation(lv_m, shape1d)
        landmark_u_labels, landmark_v_labels = self.landmark_calculation(input_m_labels, input_shape_labels)

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
        g_loss_recon = 10 * (norm_loss(g_images, input_images, loss_type=self.tex_loss_name) /
                             (torch.sum(g_images_mask) / (batch_size * self.img_sz * self.img_sz)))

        self.reconstruction_loss_input = input_images
        self.reconstruction_loss_generate = g_images
        self.writer.add_scalar("reconstruction_loss", g_loss_recon, self.global_step)
        return g_loss_recon

    def texture_loss(self, input_texture_labels, tex, tex_vis_mask, tex_ratio, **kwargs):
        g_loss_texture = 100 * norm_loss(tex, input_texture_labels, mask=tex_vis_mask,
                                         loss_type=self.tex_loss_name) / tex_ratio

        self.writer.add_scalar("texture_loss", g_loss_texture, self.global_step)
        self.texture_loss_input = input_texture_labels * tex_vis_mask
        self.texture_loss_generate = tex * tex_vis_mask
        return g_loss_texture

    def smoothness_loss(self, shape2d, **kwargs):
        g_loss_smoothness = 1000 * norm_loss((shape2d[:, :, :-2, 1:-1] + shape2d[:, :, 2:, 1:-1] +
                                              shape2d[:, :, 1:-1, :-2] + shape2d[:, :, 1:-1, 2:]) / 4.0,
                                              shape2d[:, :, 1:-1, 1:-1], loss_type=self.shape_loss_name)

        self.writer.add_scalar("g_loss_smoothness", g_loss_smoothness, self.global_step)
        return g_loss_smoothness

    def symmetry_loss(self, albedo, **kwargs):
        albedo_flip = torch.flip(albedo, dims=[3])
        flip_diff = torch.max(torch.abs(albedo - albedo_flip), torch.ones_like(albedo) * 0.05)
        g_loss_symmetry = norm_loss(flip_diff, torch.zeros_like(flip_diff), loss_type=self.tex_loss_name)

        self.writer.add_scalar("g_loss_symmetry", g_loss_symmetry, self.global_step)
        return g_loss_symmetry

    def const_albedo_loss(self, albedo, input_albedo_indexes, **kwargs):

        albedo_1 = get_pixel_value(albedo, input_albedo_indexes[0], input_albedo_indexes[1])
        albedo_2 = get_pixel_value(albedo, input_albedo_indexes[2], input_albedo_indexes[3])
        diff = torch.max(torch.abs(albedo_1 - albedo_2), torch.ones_like(albedo_1) * 0.05)
        g_loss_albedo_const = 5 * norm_loss(diff, torch.zeros_like(diff), loss_type=self.tex_loss_name)

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

    def save(self, model, global_optimizer, encoder_optimizer, epoch, path="checkpoint/"):
        dir_path = self.get_checkpoint_dir(path, epoch)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        torch.save({
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'global_optimizer': global_optimizer.state_dict(),
            'encoder_optimizer': encoder_optimizer.state_dict(),
        }, self.get_checkpoint_name(path, epoch))

    def load(self, model, global_optimizer=None, encoder_optimizer=None, start_epoch=-1, path="checkpoint/"):
        if start_epoch == -1:
            start_epoch = 0
            if not os.path.isdir(path):
                print(f"no checkpoint! path: {path}")
                return model, global_optimizer, encoder_optimizer, start_epoch
            for ckpt_dir_name in os.listdir(path):
                start_epoch = max(start_epoch, max(map(int, re.findall(r"\d+", ckpt_dir_name))))

        filename = self.get_checkpoint_name(path, start_epoch)
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location=self.device)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            global_optimizer.load_state_dict(checkpoint['global_optimizer'])
            encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(filename))

        return model, global_optimizer, encoder_optimizer, start_epoch

    def get_checkpoint_dir(self, path, number):
        return os.path.join(path, f"ckpt_{number}")

    def get_checkpoint_name(self, path, number):
        return os.path.join(f"{self.get_checkpoint_dir(path, number)}", f"model_ckpt_{number}.pt")


def pretrained_lr_test(lr, name=None, num_epochs=10, start_epoch=-1):
    pretrained_kwargs = {
        "losses": [
            'landmark',
            'batchwise_white_shading',
            'texture',
            'symmetry',
            'const_albedo',
            'smoothness'
        ],
        "device": 'cuda' if torch.cuda.is_available() else 'cpu',
        "name": "pretrain-tco-test-" + str(int(round(lr*10000))) if name is None else name
    }
    pretrained_helper = Nonlinear3DMMHelper(**pretrained_kwargs)
    pretrained_helper.train(
        num_epochs=num_epochs,
        batch_size=15,
        learning_rate=lr,
        betas=(0.5, 0.999),
        start_epoch=start_epoch
    )


if __name__ == "__main__":
    pretrained_lr_test(0.0002, num_epochs=50)
    # pretrained_lr_test(0.0002)
    # pretrained_lr_test(0.0003)
    # pretrained_lr_test(0.0005)
    # pretrained_lr_test(0.0008)
    # pretrained_lr_test(0.001)
