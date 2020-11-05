
from network.Nonlinear_3DMM import Nonlinear3DMM
from settings import CFG
from loss import Loss
from os.path import join, basename
from glob import glob
import torch
import torchvision.transforms.functional as F
import torchvision
from PIL import Image

from renderer.rendering_ops import *
from configure_dataset import NonlinearDataset
from settings import CFG
# d = NonlinearDataset(phase='test', frac=0.1)
# exp_random          = np.random.normal(0, 0.5, d.std_shape.shape)
# exp_random_tensor   = torch.tensor(exp_random)  # exp ~ N(0, 0.5)


# return images, losses, obj file, landmark output
dtype = torch.float32

mu_shape, w_shape = load_Basel_basic('shape')
mu_exp, w_exp = load_Basel_basic('exp')

mean_shape = torch.tensor(mu_shape + mu_exp, dtype=dtype).to(CFG.device)
std_shape = torch.tensor(np.tile(np.array([1e4, 1e4, 1e4]), CFG.vertex_num), dtype=dtype).to(CFG.device)

mean_m = torch.tensor(np.load(join(CFG.dataset_path, 'mean_m.npy')), dtype=dtype).to(CFG.device)
std_m = torch.tensor(np.load(join(CFG.dataset_path, 'std_m.npy')), dtype=dtype).to(CFG.device)


def save_to_obj(name, vertex, face):
	with open(name, 'w') as fd:
		for v1, v2, v3 in vertex:
			fd.write(f'v {v1:.3f} {v2:.3f} {v3:.3f}\n')

		fd.write("\n")
		for f1, f2, f3 in face:
			fd.write(f'f {f3 + 1} {f2 + 1} {f1 + 1}\n')

	print(name)


def rendering(input_images, infer):
	lv_m = infer["lv_m"]
	lv_il = infer["lv_il"]
	albedo_base = infer["albedo_base"]
	albedo_comb = infer["albedo_comb"]
	shape_1d_comb = infer["shape_1d_comb"]
	shape_1d_base = infer["shape_1d_base"]
	# exp = infer["exp"]

	m_full = generate_full(lv_m, std_m, mean_m)

	# shape_full_comb = generate_full((shape_1d_comb + exp), self.std_shape, self.mean_shape)
	# shape_full_base = generate_full((shape_1d_base + exp), self.std_shape, self.mean_shape)

	shape_full_base = generate_full(shape_1d_base, std_shape, mean_shape)
	shape_full_comb = generate_full(shape_1d_comb, std_shape, mean_shape)

	shade_base = generate_shade(lv_il, m_full, shape_full_base)
	shade_comb = generate_shade(lv_il, m_full, shape_full_comb)

	tex_base = generate_texture(albedo_base, shade_base)
	tex_mix_ac_sb = generate_texture(albedo_comb, shade_base)
	tex_mix_ab_sc = generate_texture(albedo_base, shade_comb)  # ab = albedo_base, sc = shape_comb
	tex_comb = generate_texture(albedo_comb, shade_comb)

	u_base, v_base, mask_base = warping_flow(m_full, shape_full_base)
	u_comb, v_comb, mask_comb = warping_flow(m_full, shape_full_comb)

	g_img_mask_base = mask_base.unsqueeze(1)
	g_img_mask_comb = mask_comb.unsqueeze(1)

	g_img_raw_base = rendering_wflow(tex_base, u_base, v_base)
	g_img_raw_ac_sb = rendering_wflow(tex_mix_ac_sb, u_base, v_base)
	g_img_raw_ab_sc = rendering_wflow(tex_mix_ab_sc, u_comb, v_comb)
	g_img_raw_comb = rendering_wflow(tex_comb, u_comb, v_comb)

	g_img_base = apply_mask(g_img_raw_base, g_img_mask_base, input_images)
	g_img_ac_sb = apply_mask(g_img_raw_ac_sb, g_img_mask_base, input_images)
	g_img_ab_sc = apply_mask(g_img_raw_ab_sc, g_img_mask_comb, input_images)
	g_img_comb = apply_mask(g_img_raw_comb, g_img_mask_comb, input_images)

	# ======= gt =======

	# for debugging
	g_img_shade_base = rendering_wflow(shade_base, u_base, v_base)
	g_img_shade_comb = rendering_wflow(shade_comb, u_comb, v_comb)

	return {
		"shade_base": shade_base.float(),
		"shade_comb": shade_comb.float(),

		"tex_base": tex_base.float(),
		"tex_mix_ac_sb": tex_mix_ac_sb.float(),
		"tex_mix_ab_sc": tex_mix_ab_sc.float(),
		"tex_comb": tex_comb.float(),

		"g_img_base": g_img_base.float(),
		"g_img_mask_base": g_img_mask_base.float(),

		"g_img_ac_sb": g_img_ac_sb.float(),
		"g_img_ab_sc": g_img_ab_sc.float(),

		"g_img_comb": g_img_comb.float(),
		"g_img_mask_comb": g_img_mask_comb.float(),

		# for debugging
		"g_img_raw_base": g_img_raw_base.float(),
		"g_img_raw_ac_sb": g_img_raw_ac_sb.float(),
		"g_img_raw_ab_sc": g_img_raw_ab_sc.float(),
		"g_img_raw_comb": g_img_raw_comb.float(),

		"g_img_shade_base": g_img_shade_base.float(),
		"g_img_shade_comb": g_img_shade_comb.float(),

		"shape_full_base": shape_full_base.float(),
		"shape_full_comb": shape_full_comb.float()
	}

def main():
	tri = load_3DMM_tri()
	face = np.transpose(tri)[:-1]

	losses = [
		'm',
		'shape',
		'landmark',
		'batchwise_white_shading',
		'texture',
		'symmetry',
		'const_albedo',
		'smoothness'
	]

	# define model and loss
	model = Nonlinear3DMM().to(CFG.device)
	model, _, _, _, _ = load_from_name(model)

	# load images for prediction
	fnames_raw = glob(join(CFG.prediction_src_path, "*"))
	total_len = len(fnames_raw)
	fnames = [fnames_raw[i:i + CFG.batch_size] for i in range(0, len(fnames_raw), CFG.batch_size)]

	# output image lists
	output = dict()

	for batch_fnames in fnames:
		input_images = []

		# resize and crop images
		for fname in batch_fnames:
			img = Image.open(fname)
			img = torchvision.transforms.functional.resize(img, CFG.image_size)
			img = torchvision.transforms.functional.center_crop(img, CFG.image_size)
			img = torchvision.transforms.functional.to_tensor(img)
			input_images.append(img)
			output[fname] = dict()
		input_images = torch.stack(input_images, dim=0).to(CFG.device)

		# forward network
		with torch.no_grad():
			infer = model(input_images)

			# exp = infer["exp"]
			# lv_m, lv_il, lv_shape,  albedo, exp =

		# make full
		result = rendering(input_images, infer)

		for idx, fname in enumerate(batch_fnames):
			output[fname]["input"] = input_images[idx]
			output[fname]["shade_base"] = result["shade_base"][idx]
			output[fname]["shade_comb"] = result["shade_comb"][idx]
			output[fname]["tex_base"] = result["tex_base"][idx]
			output[fname]["tex_mix_ac_sb"] = result["tex_mix_ac_sb"][idx]
			output[fname]["tex_mix_ab_sc"] = result["tex_mix_ab_sc"][idx]
			output[fname]["tex_comb"] = result["tex_comb"][idx]
			output[fname]["g_img_base"] = result["g_img_base"][idx]
			output[fname]["g_img_ac_sb"] = result["g_img_ac_sb"][idx]
			output[fname]["g_img_ab_sc"] = result["g_img_ab_sc"][idx]
			output[fname]["g_img_comb"] = result["g_img_comb"][idx]
			output[fname]["g_img_mask_base"] = result["g_img_mask_base"][idx]
			output[fname]["g_img_mask_comb"] = result["g_img_mask_comb"][idx]
			output[fname]["g_img_raw_base"] = result["g_img_raw_base"][idx]
			output[fname]["g_img_raw_ac_sb"] = result["g_img_raw_ac_sb"][idx]
			output[fname]["g_img_raw_ab_sc"] = result["g_img_raw_ab_sc"][idx]
			output[fname]["g_img_raw_comb"] = result["g_img_raw_comb"][idx]
			output[fname]["g_img_shade_base"] = result["g_img_shade_base"][idx]
			output[fname]["g_img_shade_comb"] = result["g_img_shade_comb"][idx]
			output[fname]["shape_full_base"] = result["shape_full_base"][idx]
			output[fname]["shape_full_comb"] = result["shape_full_comb"][idx]

			output[fname]["albedo_base"] = infer["albedo_base"][idx]
			output[fname]["albedo_comb"] = infer["albedo_comb"][idx]
			output[fname]["shape_1d_base"] = infer["shape_1d_base"][idx]
			output[fname]["shape_1d_comb"] = infer["shape_1d_comb"][idx]




	# save images
	if not os.path.isdir(CFG.prediction_dst_path):
		os.makedirs(CFG.prediction_dst_path)

	save_image = torchvision.utils.save_image
	for fname, out in output.items():
		name = join(CFG.prediction_dst_path, basename(fname).split('.')[0] + '_generated')
		save_image(torch.stack([out["shade_base"], out["shade_comb"], out["albedo_base"], out["albedo_comb"]]),
				   fp=name+"_shade.png")
		save_image(torch.stack([out["tex_base"], out["tex_mix_ac_sb"], out["tex_mix_ab_sc"], out["tex_comb"]]),
				   fp=name+"_tex.png")
		save_image(torch.stack([out["input"], out["g_img_base"], out["g_img_ac_sb"], out["g_img_ab_sc"], out["g_img_comb"]]),
				   fp=name+"_img_mask.png")
		save_image(torch.stack([out["input"], out["g_img_raw_base"], out["g_img_raw_ac_sb"], out["g_img_raw_ab_sc"], out["g_img_raw_comb"]]),
				   fp=name+"_img_raw.png")
		save_image(torch.stack([out["input"], out["g_img_shade_base"], out["g_img_shade_comb"]]),
				   fp=name+"_img_shade.png")

		save_to_obj(name+"_base.obj", out["shape_full_base"].view(-1, 3), face)
		save_to_obj(name+"_comb.obj", out["shape_full_comb"].view(-1, 3), face)

	# for o1, o2, o3, fname in zip(random_camera, random_il, random_exp, fnames_raw):
	# 	torchvision.utils.save_image(torch.stack([o1, o2, o3]), fp=join(CFG.prediction_dst_path, basename(fname).split('.')[0] + '_randomed.png'))


if __name__ == '__main__':
	main()


