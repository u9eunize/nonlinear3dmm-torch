import torch
import torch.nn.functional as F
from renderer.rendering_ops_redner import *
from network.facenet import InceptionResnetV1

from settings import CFG
from os.path import join
from datetime import datetime

import numpy as np
import utils

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def norm_loss(predictions, labels, mask=None, loss_type="l1", reduce_mean=True, p=1.0):
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
    def __init__(self, loss_coefficients: dict, decay_per_epoch=None):
        print("**** using ****")
        if decay_per_epoch is None:
            decay_per_epoch = dict()

        self.loss_names = loss_coefficients.keys()
        self.is_perceptual = False

        for loss_name, loss_coefficient in loss_coefficients.items():
            print(loss_name, "coefficients:", loss_coefficient, end=" ")
            if loss_name not in decay_per_epoch:
                decay_per_epoch[loss_name] = 1.0
            else:
                print("decay per epoch:", decay_per_epoch[loss_name], end="")
            print("")
            if "perceptual" in loss_name:
                self.is_perceptual = True

        self.loss_coefficients = loss_coefficients
        self.decay_per_epoch = decay_per_epoch
        self.decay_step = CFG.start_loss_decay_step

        if self.is_perceptual:
            self.facenet = InceptionResnetV1('vggface2').eval().to(CFG.device)
            for layer_idx in CFG.perceptual_layer:
                self.facenet.__getattr__(layer_idx).register_forward_hook(get_activation(f'({layer_idx}) conv2d'))

        self.cache = dict()
        self.time_checker = dict()

    def __call__(self, **kwargs):
        self.losses = dict()
        self.generate_precalculation(**kwargs)

        for loss_name, loss_coefficient in self.loss_coefficients.items():
            self.time_start(loss_name)
            loss_fn_name = loss_name + "_loss"
            loss_fn = self.__getattribute__(loss_fn_name)
            decay_coefficient = self.decay_per_epoch[loss_name] ** self.decay_step
            self.losses[loss_fn_name] = (loss_coefficient * decay_coefficient) * loss_fn(**kwargs)
            self.time_end(loss_name)

        self.losses['g_loss'] = sum([v if "landmark" not in k else torch.tensor(0) for k, v in self.losses.items()])
        self.losses['landmark_loss'] = sum([v if "landmark" in k else torch.tensor(0) for k, v in self.losses.items()])
        self.losses['g_loss_with_landmark'] = self.losses['landmark_loss'] + self.losses['g_loss']

        self.release_precalculcation()
        return self.losses['g_loss'], self.losses['g_loss_with_landmark']

    def time_start(self, name):
        self.time_checker[name] = datetime.now()

    def time_end(self, name):
        self.time_checker[name] = (datetime.now() - self.time_checker[name]).total_seconds() * 1000

    def decay_coefficient(self):
        self.decay_step += 1

    def generate_precalculation(self, **kwargs):
        if self.is_perceptual:
            self.time_start("pre_rec")
            self.cache.update(self._perceptual_recon_precalculation(**kwargs))
            self.time_end("pre_rec")

    def release_precalculcation(self):
        self.cache = dict()
        pass

    def expression_loss(self, exp_1d, input_exp, **kwargs):
        g_loss_exp = norm_loss(exp_1d, input_exp, loss_type=CFG.expression_loss_type)
        return g_loss_exp

    def exp_regularization_loss(self, exp_1d, **kwargs):
        g_loss_exp_regularization = norm_loss(exp_1d, torch.zeros_like(exp_1d, device=CFG.device), loss_type=CFG.exp_regularization_loss_type)
        return g_loss_exp_regularization

    def shape_loss(self, shape_1d_base, input_shape, **kwargs):
        # shape_1d_base = shape_1d_base.view([-1, 3])
        # input_shape = input_shape.view([-1, 3])
        # g_loss_shape_x = norm_loss(shape_1d_base[:, 0], (input_shape[:, 0] + 0) * 10, loss_type=CFG.shape_loss_type)
        # g_loss_shape_y = norm_loss(shape_1d_base[:, 1], (input_shape[:, 1] + 0) * 10, loss_type=CFG.shape_loss_type)
        # g_loss_shape_z = norm_loss(shape_1d_base[:, 2], (input_shape[:, 2] + 0.75) * 10, loss_type=CFG.shape_loss_type)
        # g_loss_shape = g_loss_shape_x + g_loss_shape_y + g_loss_shape_z

        g_loss_shape = norm_loss(shape_1d_base, input_shape, loss_type=CFG.shape_loss_type)
        return g_loss_shape

    def shape_regularization_loss(self, shape_2d_base, **kwargs):
        g_loss_shape_regularization_x = norm_loss(shape_2d_base[:, 0, :, :], torch.zeros_like(shape_2d_base[:, 0, :, :], device=CFG.device), loss_type=CFG.shape_regularization_loss_type)
        g_loss_shape_regularization_y = norm_loss(shape_2d_base[:, 1, :, :], torch.zeros_like(shape_2d_base[:, 1, :, :], device=CFG.device), loss_type=CFG.shape_regularization_loss_type)
        g_loss_shape_regularization = g_loss_shape_regularization_x + g_loss_shape_regularization_y
        return g_loss_shape_regularization

    # def m_loss(self, lv_m, input_m_labels, **kwargs):
    #     g_loss_m = norm_loss(lv_m, input_m_labels, loss_type=CFG.m_loss_type)
    #     return g_loss_m

    def trans_loss(self, lv_trans, input_trans, **kwargs):
        g_loss_trans = norm_loss(lv_trans, input_trans, loss_type=CFG.m_loss_type)
        return g_loss_trans

    def angle_loss(self, lv_angle, input_angle, **kwargs):
        g_loss_angle = norm_loss(lv_angle, input_angle, loss_type=CFG.m_loss_type)
        return g_loss_angle

    def light_loss(self, lv_il, input_light, **kwargs):
        g_loss_light = norm_loss(lv_il, input_light, loss_type=CFG.m_loss_type)
        return g_loss_light

    # -------- landmark ---------
    # def landmark_calculation(self, mv, sv, ev):
    #     m_full = generate_full(mv, "m")
    #     shape_full = generate_full(sv, "shape")
    #     if ev is not None and CFG.using_expression:
    #         shape_full += generate_full(ev, "exp")
    #
    #     landmark_u, landmark_v = compute_landmarks_torch(m_full, shape_full)
    #     return landmark_u, landmark_v

    def _landmark_loss_calculation(self, shape_1d, exp_1d, lv_trans, lv_angle, input_shape, input_exp, input_trans, input_angle):
        batch_size = shape_1d.shape[0]
        shape_full = shape_1d + exp_1d + CFG.mean_shape
        shape_full = shape_full.view([batch_size, -1, 3])
        landmark_u, landmark_v = project_vertices(shape_full[:, CFG.landmark - 1], lv_trans, lv_angle)

        shape_full_label = input_shape + input_exp + CFG.mean_shape
        shape_full_label = shape_full_label.view([batch_size, -1, 3])
        landmark_u_label, landmark_v_label = project_vertices(shape_full_label[:, CFG.landmark - 1], input_trans, input_angle)

        u_loss = torch.mean(norm_loss(landmark_u, landmark_u_label,
                                      loss_type=CFG.landmark_loss_type, reduce_mean=False))
        v_loss = torch.mean(norm_loss(landmark_v, landmark_v_label,
                                      loss_type=CFG.landmark_loss_type, reduce_mean=False))
        landmark_mse_mean = u_loss + v_loss
        landmark_loss = landmark_mse_mean / CFG.landmark_num / batch_size / CFG.image_size

        return landmark_loss

    def base_landmark_loss(self, shape_1d_base, exp_1d, lv_trans, lv_angle,
                           input_shape, input_exp, input_trans, input_angle, **kwargs):
        return self._landmark_loss_calculation(shape_1d_base, exp_1d, lv_trans, lv_angle,
                                               input_shape, input_exp, input_trans, input_angle)

    def comb_landmark_loss(self, shape_1d_comb, exp_1d, lv_trans, lv_angle,
                           input_shape, input_exp, input_trans, input_angle, **kwargs):
        return self._landmark_loss_calculation(shape_1d_comb, exp_1d, lv_trans, lv_angle,
                                               input_shape, input_exp, input_trans, input_angle)

    def gt_landmark_loss(self, shape_1d_comb, exp_1d, lv_trans, lv_angle,
                         input_shape, input_exp, input_trans, input_angle, **kwargs):
        return self._landmark_loss_calculation(shape_1d_comb, exp_1d, input_trans, input_angle,
                                               input_shape, input_exp, input_trans, input_angle)

    # def batchwise_white_shading_loss(self, shade_base, **kwargs):
    #     uv_mask = self.uv_mask.unsqueeze(0).unsqueeze(0)
    #     mean_shade = torch.mean(shade_base * uv_mask, dim=[0, 2, 3]) * 16384 / 10379
    #     g_loss_white_shading = norm_loss(mean_shade,
    #                                      0.99 * torch.ones(mean_shade.shape).float().to(CFG.device),
    #                                      loss_type=CFG.batchwise_white_shading_loss_type)
    #     return g_loss_white_shading

    # --------- reconstrcution loss -------

    def _pixel_loss_calculation(self, g_images, input_images, g_images_mask):

        images_loss = norm_loss(g_images, input_images, loss_type="l2")
        mask_mean = torch.sum(g_images_mask) / (CFG.batch_size * CFG.image_size * CFG.image_size)
        g_loss_recon = images_loss / mask_mean
        return g_loss_recon

    def _perceptual_recon_precalculation(self, input_image, g_img_base, g_img_ac_sb, g_img_ab_sc,
                                         input_mask, **kwargs):
        idxes, fts = self._face_calculation_multiple(input_image * input_mask, g_img_base, g_img_ac_sb, g_img_ab_sc)

        return dict(
            pcpt_input_images=(idxes[0], fts[0]),
            pcpt_g_img_base=(idxes[1], fts[1]),
            pcpt_g_img_ac_sb=(idxes[2], fts[2]),
            pcpt_g_img_ab_sc=(idxes[3], fts[3]),
        )

    def _face_calculation(self, img):
        features = []
        idx_vec = self.facenet(img)
        for val in activation.values():
            features.append(val.clone())
        return idx_vec, features

    def _face_calculation_multiple(self, *imgs):
        bat_sz = imgs[0].shape[0]
        img_chunks = torch.cat(imgs, dim=0)
        features = [list() for _ in range(len(imgs))]

        idx_vecs = self.facenet(img_chunks.float())
        idx_vecs = torch.split(idx_vecs, bat_sz)
        for val in activation.values():
            result = torch.split(val.clone(), bat_sz)
            for i in range(len(imgs)):
                features[i].append(result[i])
        return idx_vecs, features

    def _perceptual_loss_calculation(self, name):
        i1_idx, i1_features = self.cache["pcpt_"+name]
        i2_idx, i2_features = self.cache["pcpt_input_images"]
        return self._perceptual_distance(i1_features, i2_features)

    def _perceptual_distance(self, features1, features2):
        perceptual_distance = 0
        for f1, f2 in zip(features1, features2):
            perceptual_distance += norm_loss(f1, f2, loss_type="l2") / len(features1)
        return perceptual_distance

    # def gradient_difference_loss(self, input_images, g_images, **kwargs):
    #     input_images_gradient_x = torch.abs(input_images[:, :, :-1, :] - input_images[:, :, 1:, :])
    #     input_images_gradient_y = torch.abs(input_images[:, :, :, :-1] - input_images[:, :, :, 1:])
    #
    #     g_images_gradient_x = torch.abs(g_images[:, :, :-1, :] - g_images[:, :, 1:, :])
    #     g_images_gradient_y = torch.abs(g_images[:, :, :, :-1] - g_images[:, :, :, 1:])
    #
    #     g_loss_gradient_difference = norm_loss(input_images_gradient_x, g_images_gradient_x, loss_type=CFG.gradient_difference_loss_type)\
    #                                  + norm_loss(input_images_gradient_y, g_images_gradient_y, loss_type=CFG.gradient_difference_loss_type)
    #
    #     return g_loss_gradient_difference

    def base_perceptual_recon_loss(self, g_img_base, **kwargs):
        return self._perceptual_loss_calculation("g_img_base")

    def mix_ac_sb_perceptual_recon_loss(self, g_img_ac_sb, **kwargs):
        return self._perceptual_loss_calculation( "g_img_ac_sb")

    def mix_ab_sc_perceptual_recon_loss(self, g_img_ab_sc, **kwargs):
        return self._perceptual_loss_calculation( "g_img_ab_sc")

    def comb_perceptual_recon_loss(self, g_img_comb, **kwargs):
        return self._perceptual_loss_calculation("g_img_comb")

    def base_pix_recon_loss(self, input_image, input_mask, g_img_base, g_mask_base, **kwargs):
        return self._pixel_loss_calculation(g_img_base, input_image, input_mask * g_mask_base)

    def mix_ac_sb_pix_recon_loss(self, input_image, input_mask, g_img_ac_sb, g_mask_ac_sb, **kwargs):
        return self._pixel_loss_calculation(g_img_ac_sb, input_image, input_mask * g_mask_ac_sb)

    def mix_ab_sc_pix_recon_loss(self, input_image, input_mask, g_img_ab_sc, g_mask_ab_sc, **kwargs):
        return self._pixel_loss_calculation(g_img_ab_sc, input_image, input_mask * g_mask_ab_sc)

    def comb_recon_pix_loss(self, input_image, input_mask, g_img_comb, g_mask_comb, **kwargs):
        return self._pixel_loss_calculation(g_img_comb, input_image, input_mask * g_mask_comb)

    # --------- texture ---------

    # def _texture_loss_calculation(self, tex, input_texture_labels, tex_vis_mask, tex_ratio):
    #     g_loss_texture = norm_loss(tex, input_texture_labels, mask=tex_vis_mask, loss_type=CFG.texture_loss_type)
    #     g_loss_texture = g_loss_texture / tex_ratio
    #     return g_loss_texture

    def _texture_loss_calculation(self, vcolor, input_vcolor):
        g_loss_vcolor = norm_loss(vcolor, input_vcolor, loss_type=CFG.texture_loss_type)
        return g_loss_vcolor

    def base_texture_loss(self, g_vcolor_base, input_vcolor, **kwargs):
        return self._texture_loss_calculation(g_vcolor_base, input_vcolor)

    def mix_ac_sb_texture_loss(self, g_vcolor_ac_sb, input_vcolor, **kwargs):
        return self._texture_loss_calculation(g_vcolor_ac_sb, input_vcolor)

    def mix_ab_sc_texture_loss(self, g_vcolor_ab_sc, input_vcolor, **kwargs):
        return self._texture_loss_calculation(g_vcolor_ab_sc, input_vcolor)

    def comb_texture_loss(self, g_vcolor_comb, input_vcolor, **kwargs):
        return self._texture_loss_calculation(g_vcolor_comb, input_vcolor)

    # --------- smoothness ---------

    def _smoothness_loss_calculation(self, shape_2d):
        smoothness_sum = (shape_2d[:, :, :-2, 1:-1] + shape_2d[:, :, 2:, 1:-1] +
                          shape_2d[:, :, 1:-1, :-2] + shape_2d[:, :, 1:-1, 2:]) / 4.0
        g_loss_smoothness = norm_loss(smoothness_sum, shape_2d[:, :, 1:-1, 1:-1],
                                      loss_type=CFG.smoothness_loss_type)
        return g_loss_smoothness

    def base_smoothness_loss(self, shape_2d_base, **kwargs):
        return self._smoothness_loss_calculation(shape_2d_base)

    def comb_smoothness_loss(self, shape_2d_comb, **kwargs):
        return self._smoothness_loss_calculation(shape_2d_comb)

    # def base_exp_smoothness_loss(self, exp_2d_base, **kwargs):
    #     return self._smoothness_loss_calculation(exp_2d_base)
    #
    # def comb_exp_smoothness_loss(self, exp_2d_comb, **kwargs):
    #     return self._smoothness_loss_calculation(exp_2d_comb)

    def exp_smoothness_loss(self, exp_2d, **kwargs):
        return self._smoothness_loss_calculation(exp_2d)

    def symmetry_loss(self, albedo_base, **kwargs):
        albedo_flip = torch.flip(albedo_base, dims=[3])
        flip_diff = torch.max(torch.abs(albedo_base - albedo_flip), torch.ones_like(albedo_base) * 0.05)
        g_loss_symmetry = norm_loss(flip_diff, torch.zeros_like(flip_diff),
                                    loss_type=CFG.symmetry_loss_type)
        return g_loss_symmetry

    def symmetry_shape_loss(self, shape_2d_base, **kwargs):
        shape_flip = torch.flip(shape_2d_base, dims=[3])

        shape_2d_base_x, _, shape_2d_base_z = torch.split(shape_2d_base, (1, 1, 1), dim=1)
        shape_flip_x, _, shape_flip_z = torch.split(shape_flip, (1, 1, 1), dim=1)

        flip_diff_x = torch.max(torch.abs(shape_2d_base_x + shape_flip_x), torch.ones_like(shape_2d_base_x) * 0.05)
        g_loss_symmetry_shape_x = norm_loss(flip_diff_x, torch.zeros_like(flip_diff_x),
                                            loss_type=CFG.symmetry_loss_type)

        # flip_diff_z = torch.max(torch.abs(shape_2d_base_z - shape_flip_z), torch.ones_like(shape_2d_base_z) * 0.05)
        # g_loss_symmetry_shape_z = norm_loss(flip_diff_z, torch.zeros_like(flip_diff_z),
        #                             loss_type=CFG.symmetry_loss_type)
        #
        # g_loss_symmetry_shape = g_loss_symmetry_shape_x + g_loss_symmetry_shape_z
        return g_loss_symmetry_shape_x

    def const_albedo_loss(self, albedo_base, input_albedo_indexes, **kwargs):
        albedo_1 = albedo_base[:, :, input_albedo_indexes[0], input_albedo_indexes[1]]
        albedo_2 = albedo_base[:, :, input_albedo_indexes[2], input_albedo_indexes[3]]
        diff = torch.max(torch.abs(albedo_1 - albedo_2), torch.ones_like(albedo_1) * 0.05)
        g_loss_albedo_const = norm_loss(diff, torch.zeros_like(diff),
                                        loss_type=CFG.const_albedo_loss_type)

        return g_loss_albedo_const
    #
    def const_local_albedo_loss(self, albedo_base, **kwargs):
        u_albedo_norm = norm_loss(albedo_base[:, :, :-1, :], albedo_base[:, :, 1:, :],
                                  loss_type=CFG.const_local_albedo_loss_type, p=0.8, reduce_mean=False)
        loss_local_albedo_u = torch.mean(u_albedo_norm)

        v_albedo_norm = norm_loss(albedo_base[:, :, :, :-1], albedo_base[:, :, :, 1:],
                                  loss_type=CFG.const_local_albedo_loss_type, p=0.8, reduce_mean=False)
        loss_local_albedo_v = torch.mean(v_albedo_norm)
        loss_local_albedo = (loss_local_albedo_u + loss_local_albedo_v)

        return loss_local_albedo
    #
    # def shade_mag_loss(self, shade_base, tex_base, **kwargs):
    #     return norm_loss(shade_base, torch.ones_like(shade_base),
    #                      mask=torch.gt(tex_base, 1).float(), loss_type=CFG.shade_mag_loss_type)

    def shape_residual_loss(self, shape_1d_res, **kwargs):
        return norm_loss(shape_1d_res, torch.zeros_like(shape_1d_res), loss_type=CFG.residual_loss_type)

    def albedo_residual_loss(self, albedo_res, **kwargs):
        return norm_loss(albedo_res, torch.zeros_like(albedo_res), loss_type=CFG.residual_loss_type)

    def exp_residual_loss(self, exp_1d_res, **kwargs):
        return norm_loss(exp_1d_res, torch.zeros_like(exp_1d_res), loss_type=CFG.residual_loss_type)


    # def residual_symmetry_loss(self, rotated_normal_2d, **kwargs):
    #     pass
