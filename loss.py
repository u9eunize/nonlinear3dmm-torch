import torch
import torch.nn.functional as F
from renderer.rendering_ops import *
import config
from os.path import join


def norm_loss(predictions, labels, mask=None, loss_type="l1", reduce_mean=True, p=1):
    """
    compatible with tf1
    detail tf: https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss
    detail torch l1: https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss
    detail torch l2: https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss
    """
    if mask is not None:
        predictions = predictions * mask
        labels = labels * mask

    if loss_type == "l1":
        loss = F.l1_loss(predictions, labels, reduction='sum')
    elif loss_type == "l2":
        loss = F.mse_loss(predictions, labels, reduction='sum') / 2

    elif loss_type == "l2,1":
        loss = torch.sqrt(torch.sum(torch.square(predictions - labels) + 1e-16, dim=1))
        if p != 1:
            loss = torch.pow(loss, p)
        return torch.sum(loss)
    else:
        loss = 0
        assert "available: 'l1', 'l2', 'l2,1'"

    if reduce_mean:
        numel = torch.prod(torch.tensor(predictions.shape)).float()
        loss = torch.div(loss, numel)

    return loss


class Loss:
    def __init__(self, loss_names):
        print("**** using ****")
        for loss_name in loss_names:
            print(loss_name)

        self.loss_names = loss_names
        self.shape_loss_name = "l2"
        self.tex_loss_name = "l1"
        self.landmark_num = 68
        self.img_sz = 224

        dtype = torch.float

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        mu_shape, w_shape = load_Basel_basic('shape')
        mu_exp, w_exp = load_Basel_basic('exp')

        self.mean_shape = torch.tensor(mu_shape + mu_exp, dtype=dtype)
        self.std_shape = torch.tensor(np.tile(np.array([1e4, 1e4, 1e4]), config.VERTEX_NUM), dtype=dtype)

        self.mean_m = torch.tensor(np.load(join(config.DATASET_PATH, 'mean_m.npy')), dtype=dtype)
        self.std_m = torch.tensor(np.load(join(config.DATASET_PATH, 'std_m.npy')), dtype=dtype)

        self.uv_tri, self.uv_mask = load_3DMM_tri_2d(with_mask=True)
        self.uv_tri = torch.tensor(self.uv_tri)
        self.uv_mask = torch.tensor(self.uv_mask)

        self.mean_shape = self.mean_shape.to(self.device)
        self.std_shape = self.std_shape.to(
            self.device)

        self.mean_m = self.mean_m.to(self.device)
        self.std_m = self.std_m.to(self.device)

        self.uv_tri = self.uv_tri.to(self.device)
        self.uv_mask = self.uv_mask.to(self.device)

    def __call__(self, input_images, g_images, input_masks, g_images_mask, **kwargs):
        landmark_loss = 0
        self.losses = {}

        if "landmark" in self.loss_names:
            landmark_loss = self.landmark_loss(**kwargs)
        if "batchwise_white_shading" in self.loss_names:
            self.losses['batchwise_white_shading_loss'] = self.batchwise_white_shading_loss(**kwargs)
        if "texture" in self.loss_names:
            self.losses['texture_loss'] = self.texture_loss(**kwargs)
        if "symmetry" in self.loss_names:
            self.losses['symmetry_loss'] = self.symmetry_loss(**kwargs)
        if "const_albedo" in self.loss_names:
            self.losses['const_albedo_loss'] = self.const_albedo_loss(**kwargs)
        if "smoothness" in self.loss_names:
            self.losses['smoothness_loss'] = self.smoothness_loss(**kwargs)
        if "shape" in self.loss_names:
            self.losses['shape_loss'] = self.shape_loss(**kwargs)
        if "m" in self.loss_names:
            self.losses['m_loss'] = self.m_loss(**kwargs)
        if "reconstruction" in self.loss_names:
            self.losses['reconstruction_loss'] = self.reconstruction_loss(**kwargs)
        if "const_local_albedo" in self.loss_names:
            self.losses['const_local_albedo_loss'] = self.const_local_albedo_loss(**kwargs)
        if "expression" in self.loss_names:
            self.losses['expression_loss'] = self.expression_loss(**kwargs)

        self.losses['g_loss'] = sum(self.losses.values())
        self.losses['landmark_loss'] = landmark_loss
        self.losses['g_loss_with_landmark'] = self.losses['landmark_loss'] + self.losses['g_loss']

        return self.losses['g_loss'], self.losses['g_loss_with_landmark']

    def expression_loss(self, exp, input_exp_labels, **kwargs):
        g_loss_exp = norm_loss(exp, input_exp_labels, loss_type=config.EXPRESSION_LOSS_TYPE)
        return config.EXPRESSION_LAMBDA * g_loss_exp

    def shape_loss(self, shape1d, input_shape_labels, **kwargs):
        g_loss_shape = norm_loss(shape1d, input_shape_labels, loss_type=config.SHAPE_LOSS_TYPE)
        return config.SHAPE_LAMBDA * g_loss_shape

    def m_loss(self, lv_m, input_m_labels, **kwargs):
        g_loss_m = norm_loss(lv_m, input_m_labels, loss_type=config.M_LOSS_TYPE)
        return config.M_LAMBDA * g_loss_m

    def landmark_loss(self, lv_m, shape1d, input_m_labels, input_shape_labels, **kwargs):
        batch_size = lv_m.shape[0]
        landmark_u, landmark_v = self.landmark_calculation(lv_m, shape1d)
        landmark_u_labels, landmark_v_labels = self.landmark_calculation(input_m_labels, input_shape_labels)

        u_loss = torch.mean(norm_loss(landmark_u, landmark_u_labels,
                                      loss_type=config.LANDMARK_LOSS_TYPE, reduce_mean=False))
        v_loss = torch.mean(norm_loss(landmark_v, landmark_v_labels,
                                      loss_type=config.LANDMARK_LOSS_TYPE, reduce_mean=False))
        landmark_mse_mean = u_loss + v_loss
        landmark_loss = landmark_mse_mean / self.landmark_num / batch_size

        return config.LANDMARK_LAMBDA * landmark_loss

    def batchwise_white_shading_loss(self, shade, **kwargs):
        uv_mask = self.uv_mask.unsqueeze(0).unsqueeze(0)
        mean_shade = torch.mean(shade * uv_mask, dim=[0, 2, 3]) * 16384 / 10379
        g_loss_white_shading = norm_loss(mean_shade,
                                         0.99 * torch.ones(mean_shade.shape).float().to(self.device),
                                         loss_type=config.BATCHWISE_WHITE_SHADING_LOSS_TYPE)

        return config.BATCHWISE_WHITE_SHADING_LAMBDA * g_loss_white_shading

    def reconstruction_loss(self, batch_size, input_images, g_images, g_images_mask, **kwargs):
        images_loss = norm_loss(g_images, input_images, loss_type=config.RECONSTRUCTION_LOSS_TYPE)
        mask_mean = torch.sum(g_images_mask) / (batch_size * self.img_sz * self.img_sz)
        g_loss_recon = images_loss / mask_mean
        return config.RECONSTRUCTION_LAMBDA * g_loss_recon

    def texture_loss(self, input_texture_labels, tex, tex_vis_mask, tex_ratio, **kwargs):
        g_loss_texture = config.TEXTURE_LAMBDA * norm_loss(tex, input_texture_labels, mask=tex_vis_mask,
                                                           loss_type=config.TEXTURE_LOSS_TYPE) / tex_ratio
        return g_loss_texture

    def smoothness_loss(self, shape2d, **kwargs):
        smoothness_sum = (shape2d[:, :, :-2, 1:-1] + shape2d[:, :, 2:, 1:-1] +
                          shape2d[:, :, 1:-1, :-2] + shape2d[:, :, 1:-1, 2:]) / 4.0
        g_loss_smoothness = norm_loss(smoothness_sum, shape2d[:, :, 1:-1, 1:-1],
                                      loss_type=config.SMOOTHNESS_LOSS_TYPE)
        return config.SMOOTHNESS_LAMBDA * g_loss_smoothness

    def symmetry_loss(self, albedo, **kwargs):
        albedo_flip = torch.flip(albedo, dims=[3])
        flip_diff = torch.max(torch.abs(albedo - albedo_flip), torch.ones_like(albedo) * 0.05)
        g_loss_symmetry = norm_loss(flip_diff, torch.zeros_like(flip_diff),
                                    loss_type=config.SYMMETRY_LOSS_TYPE)
        return config.SYMMETRY_LAMBDA * g_loss_symmetry

    def const_albedo_loss(self, albedo, input_albedo_indexes, **kwargs):
        albedo_1 = get_pixel_value_torch(albedo, input_albedo_indexes[0], input_albedo_indexes[1])
        albedo_2 = get_pixel_value_torch(albedo, input_albedo_indexes[2], input_albedo_indexes[3])
        diff = torch.max(torch.abs(albedo_1 - albedo_2), torch.ones_like(albedo_1) * 0.05)
        g_loss_albedo_const = norm_loss(diff, torch.zeros_like(diff),
                                        loss_type=config.CONST_ALBEDO_LOSS_TYPE)

        return config.CONST_ALBEDO_LAMBDA * g_loss_albedo_const

    def const_local_albedo_loss(self, input_texture_labels, tex_vis_mask, albedo, **kwargs):
        chromaticity = (input_texture_labels + 1) / 2.0
        chromaticity = torch.div(chromaticity, torch.sum(chromaticity, dim=1, keepdim=True) + 1e-6)

        u_diff = -15 * torch.norm(chromaticity[:, :, :-1, :] - chromaticity[:, :, 1:, :], dim=1, keepdim=True)
        w_u = (torch.exp(u_diff) * tex_vis_mask[:, :, :-1, :]).detach()
        u_albedo_norm = norm_loss(albedo[:, :, :-1, :], albedo[:, :, 1:, :],
                                  loss_type=config.CONST_LOCAL_ALBEDO_LOSS_TYPE, p=0.8, reduce_mean=False) * w_u
        loss_local_albedo_u = torch.mean(u_albedo_norm) / torch.sum(w_u + 1e-6)

        v_diff = -15 * torch.norm(chromaticity[:, :, :, :-1] - chromaticity[:, :, :, 1:], dim=1, keepdim=True)
        w_v = (torch.exp(v_diff) * tex_vis_mask[:, :, :, :-1]).detach()
        v_albedo_norm = norm_loss(albedo[:, :, :, :-1], albedo[:, :, :, 1:],
                                  loss_type=config.CONST_LOCAL_ALBEDO_LOSS_TYPE, p=0.8, reduce_mean=False) * w_v
        loss_local_albedo_v = torch.mean(v_albedo_norm) / torch.sum(w_v + 1e-6)
        loss_local_albedo = (loss_local_albedo_u + loss_local_albedo_v)

        return config.CONST_LOCAL_ALBEDO_LAMBDA * loss_local_albedo

    def landmark_calculation(self, mv, sv):
        m_full = mv * self.std_m + self.mean_m
        shape_full = sv * self.std_shape + self.mean_shape

        landmark_u, landmark_v = compute_landmarks_torch(m_full, shape_full)
        return landmark_u, landmark_v
