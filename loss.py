import torch
import torch.nn.functional as F
from renderer.rendering_ops import *
from network.facenet import InceptionResnetV1
import config
from os.path import join
from torchvision.models.vgg import vgg19_bn


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

activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach().cpu()
    return hook


class Loss:
    def __init__(self, loss_coefficients: dict):
        print("**** using ****")
        loss_names = loss_coefficients.keys()
        for loss_name, loss_coefficient in loss_coefficients.items():
            print(loss_name, "coefficients:", loss_coefficient)

        self.loss_names = loss_names
        self.loss_coefficients = loss_coefficients
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

        self.recognition_md = None

        # self.facenet = InceptionResnetV1('vggface2').eval().to(config.DEVICE)
        self.vgg19 = vgg19_bn(pretrained=True).eval().to(config.DEVICE)

        for layer_idx in config.CONTENT_LAYERS:
            self.vgg19.features[layer_idx].register_forward_hook(get_activation(f'({layer_idx}) conv2d'))

    def __call__(self, **kwargs):
        self.losses = dict()

        for loss_name, loss_coefficient in self.loss_coefficients.items():
            loss_fn_name = loss_name + "_loss"
            loss_fn = self.__getattribute__(loss_fn_name)
            self.losses[loss_fn_name] = loss_coefficient * loss_fn(**kwargs)

        self.losses['g_loss'] = sum([v if "landmark" not in k else 0 for k, v in self.losses.items()])
        self.losses['landmark_loss'] = sum([v if "landmark" in k else 0 for k, v in self.losses.items()])
        self.losses['g_loss_with_landmark'] = self.losses['landmark_loss'] + self.losses['g_loss']

        return self.losses['g_loss'], self.losses['g_loss_with_landmark']

    def identity_loss(self, input_images, g_images, g_images_random, **kwargs):
        input_vec = self.facenet(input_images).detach()
        generated_vec = self.facenet(g_images.float()).detach()
        random_generated_vec = self.facenet(g_images_random.float()).detach()
        g_loss_identity1 = norm_loss(input_vec, generated_vec, loss_type=config.IDENTITY_LOSS_TYPE)
        g_loss_identity2 = norm_loss(input_vec, random_generated_vec, loss_type=config.IDENTITY_LOSS_TYPE)
        return (1 - g_loss_identity1) + (1 - g_loss_identity2)

    def content_loss(self, input_images, g_images, **kwargs):
        input_features = self.facenet(input_images, return_features=True).detach()
        generated_features = self.facenet(g_images.float(), return_features=True).detach()
        g_loss_content = 0
        for i_feature, g_feature in zip(input_features, generated_features):
            _, C_j, H_j, W_j = i_feature.shape
            g_loss_content += torch.norm(i_feature - g_feature) / (C_j * H_j * W_j)
        return g_loss_content

    def gradient_difference_loss(self, input_images, g_images, **kwargs):
        input_images_gradient_x = torch.abs(input_images[:, :, :-1, :] - input_images[:, :, 1:, :])
        input_images_gradient_y = torch.abs(input_images[:, :, :, :-1] - input_images[:, :, :, 1:])

        g_images_gradient_x = torch.abs(g_images[:, :, :-1, :] - g_images[:, :, 1:, :])
        g_images_gradient_y = torch.abs(g_images[:, :, :, :-1] - g_images[:, :, :, 1:])

        g_loss_gradient_difference = norm_loss(input_images_gradient_x, g_images_gradient_x, loss_type=config.GRADIENT_DIFFERENCE_LOSS_TYPE)\
                                    + norm_loss(input_images_gradient_y, g_images_gradient_y, loss_type=config.GRADIENT_DIFFERENCE_LOSS_TYPE)

        return g_loss_gradient_difference

    def identity_loss_vgg19(self, input_images, g_images, g_images_random, **kwargs):
        with torch.no_grad():
            input_vec = self.vgg19(input_images).detach().cpu()
            self.activation_input = activation.copy()
            generated_vec = self.vgg19(g_images.float()).detach().cpu()
            self.activation_generated = activation.copy()
            random_generated_vec = self.vgg19(g_images_random.float()).detach().cpu()
        g_loss_identity1 = norm_loss(input_vec, generated_vec, loss_type=config.IDENTITY_LOSS_TYPE)
        g_loss_identity2 = norm_loss(input_vec, random_generated_vec, loss_type=config.IDENTITY_LOSS_TYPE)
        return g_loss_identity1 + g_loss_identity2

    def content_loss_vgg19(self, **kwargs):
        g_loss_content = 0
        for (i_key, i_val), (g_key, g_val) in zip(
                self.activation_input.items(),
                self.activation_generated.items()):
            _, C_j, H_j, W_j = i_val.shape
            # g_loss_content += norm_loss(i_val, g_val, loss_type=config.CONTENT_LOSS_TYPE) / (C_j * H_j * W_j) / len(config.CONTENT_LAYERS)
            g_loss_content += norm_loss(i_val, g_val, loss_type=config.CONTENT_LOSS_TYPE) / len(config.CONTENT_LAYERS)
        return g_loss_content

    def expression_loss(self, exp, input_exp_labels, **kwargs):
        g_loss_exp = norm_loss(exp, input_exp_labels, loss_type=config.EXPRESSION_LOSS_TYPE)
        return g_loss_exp

    def shape_loss(self, shape_1d_base, input_shape_labels, **kwargs):
        g_loss_shape = norm_loss(shape_1d_base, input_shape_labels, loss_type=config.SHAPE_LOSS_TYPE)
        return g_loss_shape

    def m_loss(self, lv_m, input_m_labels, **kwargs):
        g_loss_m = norm_loss(lv_m, input_m_labels, loss_type=config.M_LOSS_TYPE)
        return g_loss_m

    # -------- landmark ---------
    def _landmark_loss_calculation(self, lv_m, shape1d, input_m_labels, input_shape_labels):
        batch_size = lv_m.shape[0]
        landmark_u, landmark_v = self.landmark_calculation(lv_m, shape1d)
        landmark_u_labels, landmark_v_labels = self.landmark_calculation(input_m_labels, input_shape_labels)

        u_loss = torch.mean(norm_loss(landmark_u, landmark_u_labels,
                                      loss_type=config.LANDMARK_LOSS_TYPE, reduce_mean=False))
        v_loss = torch.mean(norm_loss(landmark_v, landmark_v_labels,
                                      loss_type=config.LANDMARK_LOSS_TYPE, reduce_mean=False))
        landmark_mse_mean = u_loss + v_loss
        landmark_loss = landmark_mse_mean / self.landmark_num / batch_size

        return landmark_loss

    def base_landmark_loss(self, lv_m, shape_1d_base, input_m_labels, input_shape_labels, **kwargs):
        return self._landmark_loss_calculation(lv_m, shape_1d_base, input_m_labels, input_shape_labels)

    def comb_landmark_loss(self, lv_m, shape_1d_comb, input_m_labels, input_shape_labels, **kwargs):
        return self._landmark_loss_calculation(lv_m, shape_1d_comb, input_m_labels, input_shape_labels)

    def gt_landmark_loss(self, shape_1d_comb, input_m_labels, input_shape_labels, **kwargs):
        return self._landmark_loss_calculation(input_m_labels, shape_1d_comb, input_m_labels, input_shape_labels)

    def batchwise_white_shading_loss(self, shade_base, **kwargs):
        uv_mask = self.uv_mask.unsqueeze(0).unsqueeze(0)
        mean_shade = torch.mean(shade_base * uv_mask, dim=[0, 2, 3]) * 16384 / 10379
        g_loss_white_shading = norm_loss(mean_shade,
                                         0.99 * torch.ones(mean_shade.shape).float().to(self.device),
                                         loss_type=config.BATCHWISE_WHITE_SHADING_LOSS_TYPE)
        return g_loss_white_shading

    # --------- reconstrcution loss -------

    def _reconstruction_loss_calculation(self, g_images, g_images_mask, input_images):
        batch_size = input_images.shape[0]
        images_loss = norm_loss(g_images, input_images, loss_type=config.RECONSTRUCTION_LOSS_TYPE)
        mask_mean = torch.sum(g_images_mask) / (batch_size * self.img_sz * self.img_sz)
        g_loss_recon = images_loss / mask_mean
        return g_loss_recon

    def base_recon_loss(self, g_img_base, g_img_mask_base, input_images, **kwargs):
        return self._reconstruction_loss_calculation(g_img_base, g_img_mask_base, input_images)

    def mix_ac_sb_recon_loss(self, g_img_ac_sb, g_img_mask_base, input_images, **kwargs):
        return self._reconstruction_loss_calculation(g_img_ac_sb, g_img_mask_base, input_images)

    def mix_ab_sc_recon_loss(self, g_img_ab_sc, g_img_mask_comb, input_images, **kwargs):
        return self._reconstruction_loss_calculation(g_img_ab_sc, g_img_mask_comb, input_images)

    def comb_recon_loss(self, g_img_comb, g_img_mask_comb, input_images, **kwargs):
        return self._reconstruction_loss_calculation(g_img_comb, g_img_mask_comb, input_images)

    # --------- texture ---------

    def _texture_loss_calculation(self, tex, input_texture_labels, tex_vis_mask, tex_ratio):
        g_loss_texture = norm_loss(tex, input_texture_labels, mask=tex_vis_mask,
                                   loss_type=config.TEXTURE_LOSS_TYPE)
        g_loss_texture = g_loss_texture / tex_ratio
        return g_loss_texture

    def base_texture_loss(self, tex_base, input_texture_labels, tex_vis_mask, tex_ratio, **kwargs):
        return self._texture_loss_calculation(tex_base, input_texture_labels, tex_vis_mask, tex_ratio)

    def mix_ac_sb_texture_loss(self, tex_mix_ac_sb, input_texture_labels, tex_vis_mask, tex_ratio, **kwargs):
        return self._texture_loss_calculation(tex_mix_ac_sb, input_texture_labels, tex_vis_mask, tex_ratio)

    def mix_ab_sc_texture_loss(self, tex_mix_ab_sc, input_texture_labels, tex_vis_mask, tex_ratio, **kwargs):
        return self._texture_loss_calculation(tex_mix_ab_sc, input_texture_labels, tex_vis_mask, tex_ratio)

    def comb_texture_loss(self, tex_comb, input_texture_labels, tex_vis_mask, tex_ratio, **kwargs):
        return self._texture_loss_calculation(tex_comb, input_texture_labels, tex_vis_mask, tex_ratio)

    # --------- smoothness ---------

    def _smoothness_loss_calculation(self, shape_2d, **kwargs):
        smoothness_sum = (shape_2d[:, :, :-2, 1:-1] + shape_2d[:, :, 2:, 1:-1] +
                          shape_2d[:, :, 1:-1, :-2] + shape_2d[:, :, 1:-1, 2:]) / 4.0
        g_loss_smoothness = norm_loss(smoothness_sum, shape_2d[:, :, 1:-1, 1:-1],
                                      loss_type=config.SMOOTHNESS_LOSS_TYPE)
        return g_loss_smoothness

    def base_smoothness_loss(self, shape_2d_base, **kwargs):
        return self._smoothness_loss_calculation(shape_2d_base)

    def comb_smoothness_loss(self, shape_2d_comb, **kwargs):
        return self._smoothness_loss_calculation(shape_2d_comb)

    def symmetry_loss(self, albedo_base, **kwargs):
        albedo_flip = torch.flip(albedo_base, dims=[3])
        flip_diff = torch.max(torch.abs(albedo_base - albedo_flip), torch.ones_like(albedo_base) * 0.05)
        g_loss_symmetry = norm_loss(flip_diff, torch.zeros_like(flip_diff),
                                    loss_type=config.SYMMETRY_LOSS_TYPE)
        return g_loss_symmetry

    def const_albedo_loss(self, albedo_base, input_albedo_indexes, **kwargs):
        albedo_1 = get_pixel_value_torch(albedo_base, input_albedo_indexes[0], input_albedo_indexes[1])
        albedo_2 = get_pixel_value_torch(albedo_base, input_albedo_indexes[2], input_albedo_indexes[3])
        diff = torch.max(torch.abs(albedo_1 - albedo_2), torch.ones_like(albedo_1) * 0.05)
        g_loss_albedo_const = norm_loss(diff, torch.zeros_like(diff),
                                        loss_type=config.CONST_ALBEDO_LOSS_TYPE)

        return g_loss_albedo_const

    def const_local_albedo_loss(self, albedo_base, input_texture_labels, tex_vis_mask, **kwargs):
        chromaticity = (input_texture_labels + 1) / 2.0
        chromaticity = torch.div(chromaticity, torch.sum(chromaticity, dim=1, keepdim=True) + 1e-6)

        u_diff = -15 * torch.norm(chromaticity[:, :, :-1, :] - chromaticity[:, :, 1:, :], dim=1, keepdim=True)
        w_u = (torch.exp(u_diff) * tex_vis_mask[:, :, :-1, :]).detach()
        u_albedo_norm = norm_loss(albedo_base[:, :, :-1, :], albedo_base[:, :, 1:, :],
                                  loss_type=config.CONST_LOCAL_ALBEDO_LOSS_TYPE, p=0.8, reduce_mean=False) * w_u
        loss_local_albedo_u = torch.mean(u_albedo_norm) / torch.sum(w_u + 1e-6)

        v_diff = -15 * torch.norm(chromaticity[:, :, :, :-1] - chromaticity[:, :, :, 1:], dim=1, keepdim=True)
        w_v = (torch.exp(v_diff) * tex_vis_mask[:, :, :, :-1]).detach()
        v_albedo_norm = norm_loss(albedo_base[:, :, :, :-1], albedo_base[:, :, :, 1:],
                                  loss_type=config.CONST_LOCAL_ALBEDO_LOSS_TYPE, p=0.8, reduce_mean=False) * w_v
        loss_local_albedo_v = torch.mean(v_albedo_norm) / torch.sum(w_v + 1e-6)
        loss_local_albedo = (loss_local_albedo_u + loss_local_albedo_v)

        return loss_local_albedo

    def landmark_calculation(self, mv, sv):
        m_full = mv * self.std_m + self.mean_m
        shape_full = sv * self.std_shape + self.mean_shape

        landmark_u, landmark_v = compute_landmarks_torch(m_full, shape_full)
        return landmark_u, landmark_v

    def shade_mag_loss(self, shade_base, tex_base, **kwargs):
        return norm_loss(shade_base, torch.ones_like(shade_base),
                         mask=torch.gt(tex_base, 1).float(), loss_type='l1')

    def shape_residual_loss(self, shape_1d_res, **kwargs):
        return norm_loss(shape_1d_res, torch.zeros_like(shape_1d_res), loss_type='l1')

    def albedo_residual_loss(self, albedo_res, **kwargs):
        return norm_loss(albedo_res, torch.zeros_like(albedo_res), loss_type='l1')


    # def residual_symmetry_loss(self, rotated_normal_2d, **kwargs):
    #     pass
