
from network.Nonlinear_3DMM import Nonlinear3DMM
import config
from loss import Loss
from os.path import join, basename
from glob import glob
import torch
import torchvision.transforms.functional as F
import torchvision
from PIL import Image

from renderer.rendering_ops import *
from configure_dataset import NonlinearDataset

# d = NonlinearDataset(phase='test', frac=0.1)
# exp_random          = np.random.normal(0, 0.5, d.std_shape.shape)
# exp_random_tensor   = torch.tensor(exp_random)  # exp ~ N(0, 0.5)


# return images, losses, obj file, landmark output
dtype = torch.float32

mu_shape, w_shape = load_Basel_basic('shape')
mu_exp, w_exp = load_Basel_basic('exp')

mean_shape = torch.tensor(mu_shape + mu_exp, dtype=dtype).to(config.DEVICE)
std_shape = torch.tensor(np.tile(np.array([1e4, 1e4, 1e4]), config.VERTEX_NUM), dtype=dtype).to(config.DEVICE)

mean_m = torch.tensor(np.load(join(config.DATASET_PATH, 'mean_m.npy')), dtype=dtype).to(config.DEVICE)
std_m = torch.tensor(np.load(join(config.DATASET_PATH, 'std_m.npy')), dtype=dtype).to(config.DEVICE)


def save_to_obj(name, vertex, face):
	with open(name, 'w') as fd:
		result = ""
		for v1, v2, v3 in vertex:
			result += f'v {v1:.3f} {v2:.3f} {v3:.3f}\n'
		result += "\n"
		for f1, f2, f3 in face:
			result += f'f {f3 + 1} {f2 + 1} {f1 + 1}\n'
		fd.write(result)


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
	model = Nonlinear3DMM().to(config.DEVICE)
	model, _, _, _, _ = load(model)
	loss = Loss(losses)

	# load images for prediction
	fnames_raw = glob(join(config.PREDICTION_SRC_PATH, "*"))
	total_len = len(fnames_raw)
	fnames = [fnames_raw[i:i + config.BATCH_SIZE] for i in range(0, len(fnames_raw), config.BATCH_SIZE)]

	# output image lists
	output_gt = []
	output_images = []
	output_images_with_mask = []
	output_shapes = []

	for batch_fnames in fnames:
		input_images = []

		# resize and crop images
		for fname in batch_fnames:
			img = Image.open(fname)
			img = torchvision.transforms.functional.resize(img, config.IMAGE_SIZE)
			img = torchvision.transforms.functional.center_crop(img, config.IMAGE_SIZE)
			img = torchvision.transforms.functional.to_tensor(img)
			input_images.append(img)
		input_images = torch.stack(input_images, dim=0).to(config.DEVICE)

		# forward network
		with torch.no_grad():
			lv_m, lv_il, lv_shape, lv_tex, albedo, shape2d, shape1d, exp = model(input_images)

		# make full
		m_full = lv_m * std_m + mean_m
		shape_full = (shape1d + exp) * std_shape + mean_shape
		shade = generate_shade_torch(lv_il, m_full, shape_full)
		tex = 2.0 * ((albedo + 1.0) / 2.0 * shade) - 1.0

		g_images_raw, g_images_mask_raw = warp_texture_torch(tex, m_full, shape_full)
		g_images_mask_raw = g_images_mask_raw.unsqueeze(1).repeat(1, 3, 1, 1)
		g_images = g_images_raw * g_images_mask_raw + input_images * (torch.ones_like(g_images_mask_raw) - g_images_mask_raw)

		output_gt += input_images
		output_images += g_images_raw
		output_images_with_mask += g_images
		output_shapes += shape_full.view(shape_full.shape[0], -1, 3) / 10000

	# save images
	if not os.path.isdir(config.PREDICTION_DST_PATH):
		os.makedirs(config.PREDICTION_DST_PATH)
	for o1, o2, o3, fname, sh in zip(output_gt, output_images, output_images_with_mask, fnames_raw, output_shapes):
		name = join(config.PREDICTION_DST_PATH, basename(fname).split('.')[0] + '_generated')
		torchvision.utils.save_image(torch.stack([o1, o2, o3]), fp=name+".png")
		save_to_obj(name+".obj", sh, face)

	# for o1, o2, o3, fname in zip(random_camera, random_il, random_exp, fnames_raw):
	# 	torchvision.utils.save_image(torch.stack([o1, o2, o3]), fp=join(config.PREDICTION_DST_PATH, basename(fname).split('.')[0] + '_randomed.png'))


if __name__ == '__main__':
	main()


