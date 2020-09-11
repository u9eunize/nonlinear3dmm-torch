import torch
import torch.nn.functional as F
from renderer.rendering_ops import *


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






class Loss():
	def __init__(self, losses):
		self.losses = losses
		self.shape_loss_name = "l2"
		self.tex_loss_name = "l1"
		self.landmark_num = 68
		self.img_sz = 224
		dtype = torch.float
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

		mu_shape, w_shape = load_Basel_basic('shape')
		mu_exp, w_exp = load_Basel_basic('exp')
		self.mean_shape = torch.tensor(mu_shape + mu_exp, dtype=dtype).to(self.device)
		self.std_shape = torch.tensor(np.tile(np.array([1e4, 1e4, 1e4]), VERTEX_NUM), dtype=dtype).to(self.device)
		# self.std_shape  = np.load('std_shape.npy')

		self.mean_m = torch.tensor(np.load('dataset/mean_m.npy'), dtype=dtype).to(self.device)
		self.std_m = torch.tensor(np.load('dataset/std_m.npy'), dtype=dtype).to(self.device)


		self.uv_tri, self.uv_mask = load_3DMM_tri_2d(with_mask=True)
		self.uv_tri = torch.tensor(self.uv_tri).to(self.device)
		self.uv_mask = torch.tensor(self.uv_mask).to(self.device)

	def __call__ ( self, **kwargs ):
		landmark_loss = 0
		g_loss = 0

		if "landmark" in self.losses: landmark_loss = self.landmark_loss(**kwargs)
		if "batchwise_white_shading" in self.losses: g_loss += self.batchwise_white_shading_loss(**kwargs)
		if "texture" in self.losses: g_loss += self.texture_loss(**kwargs)
		if "symmetry" in self.losses: g_loss += self.symmetry_loss(**kwargs)
		if "const_albedo" in self.losses: g_loss += self.const_albedo_loss(**kwargs)
		if "smoothness" in self.losses: g_loss += self.smoothness_loss(**kwargs)
		if "shape" in self.losses: g_loss += self.shape_loss(**kwargs)
		if "m" in self.losses: g_loss += self.m_loss(**kwargs)
		if "reconstruction" in self.losses: g_loss += self.reconstruction_loss(**kwargs)
		if "const_local_albedo" in self.losses: g_loss += self.const_local_albedo_loss(**kwargs)

		# for loss_name in self.losses:
		#     loss_fn = getattr(Loss, loss_name+"_loss")
		#     if not hasattr(loss_fn, '__call__'):
		#         continue
		#     result = loss_fn(**kwargs)
		#
		#     if loss_name == "landmark":
		#         g_loss_with_landmark = result
		#     else:
		#         g_loss += result

		g_loss_with_landmark = landmark_loss + g_loss
		return g_loss, g_loss_with_landmark

	def shape_loss ( self, shape1d, input_shape_labels, **kwargs ):
		g_loss_shape = 10 * norm_loss(shape1d, input_shape_labels, loss_type=self.shape_loss_name)
		# self.writer.add_scalar("g_loss_shape", g_loss_shape, self.global_step)
		return g_loss_shape


	def m_loss ( self, lv_m, input_m_labels, **kwargs ):
		g_loss_m = 5 * norm_loss(lv_m, input_m_labels, loss_type="l1")
		# self.writer.add_scalar("g_loss_m", g_loss_m, self.global_step)
		return g_loss_m


	def landmark_loss ( self, batch_size, lv_m, shape1d, input_m_labels, input_shape_labels, **kwargs ):
		landmark_u, landmark_v = self.landmark_calculation(lv_m, shape1d)
		landmark_u_labels, landmark_v_labels = self.landmark_calculation(input_m_labels, input_shape_labels)

		landmark_mse_mean = (
				torch.mean(norm_loss(landmark_u, landmark_u_labels, loss_type="l2", reduce_mean=False)) +
				torch.mean(norm_loss(landmark_v, landmark_v_labels, loss_type="l2", reduce_mean=False)))
		landmark_loss = landmark_mse_mean / self.landmark_num / batch_size / 50

		# self.writer.add_scalar("landmark_loss", landmark_loss, self.global_step)
		return landmark_loss


	def batchwise_white_shading_loss ( self, shade, **kwargs ):
		uv_mask = self.uv_mask.unsqueeze(0).unsqueeze(0)
		mean_shade = torch.mean(shade * uv_mask, dim=[0, 2, 3]) * 16384 / 10379
		g_loss_white_shading = 10 * norm_loss(mean_shade, 0.99 * torch.ones(mean_shade.shape).float().to(self.device),
		                                      loss_type="l2")

		# self.writer.add_scalar("g_loss_white_shading", g_loss_white_shading, self.global_step)
		return g_loss_white_shading


	def reconstruction_loss ( self, batch_size, input_images, g_images, g_images_mask, **kwargs ):
		g_loss_recon = 10 * (norm_loss(g_images, input_images, loss_type=self.tex_loss_name) /
		                     (torch.sum(g_images_mask) / (batch_size * self.img_sz * self.img_sz)))

		self.reconstruction_loss_input = input_images
		self.reconstruction_loss_generate = g_images
		# self.writer.add_scalar("reconstruction_loss", g_loss_recon, self.global_step)
		return g_loss_recon


	def texture_loss ( self, input_texture_labels, tex, tex_vis_mask, tex_ratio, **kwargs ):
		g_loss_texture = 100 * norm_loss(tex, input_texture_labels, mask=tex_vis_mask,
		                                 loss_type=self.tex_loss_name) / tex_ratio

		# self.writer.add_scalar("texture_loss", g_loss_texture, self.global_step)
		self.texture_loss_input = input_texture_labels * tex_vis_mask
		self.texture_loss_generate = tex * tex_vis_mask
		return g_loss_texture


	def smoothness_loss ( self, shape2d, **kwargs ):
		g_loss_smoothness = 1000 * norm_loss((shape2d[:, :, :-2, 1:-1] + shape2d[:, :, 2:, 1:-1] +
		                                      shape2d[:, :, 1:-1, :-2] + shape2d[:, :, 1:-1, 2:]) / 4.0,
		                                     shape2d[:, :, 1:-1, 1:-1], loss_type=self.shape_loss_name)

		# self.writer.add_scalar("g_loss_smoothness", g_loss_smoothness, self.global_step)
		return g_loss_smoothness


	def symmetry_loss ( self, albedo, **kwargs ):
		albedo_flip = torch.flip(albedo, dims=[3])
		flip_diff = torch.max(torch.abs(albedo - albedo_flip), torch.ones_like(albedo) * 0.05)
		g_loss_symmetry = norm_loss(flip_diff, torch.zeros_like(flip_diff), loss_type=self.tex_loss_name)

		# self.writer.add_scalar("g_loss_symmetry", g_loss_symmetry, self.global_step)
		return g_loss_symmetry


	def const_albedo_loss ( self, albedo, input_albedo_indexes, **kwargs ):
		albedo_1 = get_pixel_value(albedo, input_albedo_indexes[0], input_albedo_indexes[1])
		albedo_2 = get_pixel_value(albedo, input_albedo_indexes[2], input_albedo_indexes[3])
		diff = torch.max(torch.abs(albedo_1 - albedo_2), torch.ones_like(albedo_1) * 0.05)
		g_loss_albedo_const = 5 * norm_loss(diff, torch.zeros_like(diff), loss_type=self.tex_loss_name)

		# self.writer.add_scalar("g_loss_albedo_const", g_loss_albedo_const, self.global_step)
		return g_loss_albedo_const


	def const_local_albedo_loss ( self, input_texture_labels, tex_vis_mask, albedo, **kwargs ):
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

		# self.writer.add_scalar("loss_local_albedo", loss_local_albedo, self.global_step)
		return loss_local_albedo

	def landmark_calculation ( self, mv, sv ):
		m_full = mv * self.std_m + self.mean_m
		shape_full = sv * self.std_shape + self.mean_shape

		landmark_u, landmark_v = compute_landmarks_torch(m_full, shape_full, output_size=self.img_sz)
		return landmark_u, landmark_v