import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from os.path import join, isdir, basename
from PIL import Image
from os import makedirs, cpu_count
from glob import glob
import random
import shutil
from utils import *
from settings import CFG
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm


class NonlinearDataset(Dataset):
	'''
		Nonlinear dataset load class
		it contains 2 functions,
			1. split raw data into train, test, and validation dataset and
			2. load each dataset item
	'''
	def __init__(self, phase, frac=1.0, dataset_dir=CFG.dataset_path):
		self.fdtype = np.float32
		self.frac = frac

		# initialize attributes
		self.dataset_dir = dataset_dir
		self.transform = transforms.Compose([
				transforms.ToTensor(),
		])

		# load mean and std shape
		self.mean_m = np.load(join(self.dataset_dir, 'mean_m.npy'))
		self.std_m = np.load(join(self.dataset_dir, 'std_m.npy'))
		self.const_alb_mask = load_const_alb_mask()

		mu_shape, w_shape = load_Basel_basic('shape')
		mu_exp, w_exp = load_Basel_basic('exp')

		self.std_shape = np.tile(np.array([1e4, 1e4, 1e4], dtype=self.fdtype), CFG.vertex_num)

		self.w_shape = w_shape
		self.w_exp = w_exp

		# split dataset into train, validation, test set with ratio 8:1:1
		if not self.is_splited():
			print('Dataset is not splited. ')
			self.split_dataset()

		self.load_dataset(phase)


	def __len__( self ):
		return int(self.image_filenames.shape[0] * self.frac)


	def __getitem__( self, idx ):
		# set random crop index
		ty = np.random.randint(0, 32 + 1)
		tx = np.random.randint(0, 32 + 1)

		# load image
		img_name    = self.image_filenames[idx]
		img         = Image.open(img_name)
		img         = transforms.functional.crop(img, ty, tx, CFG.image_size, CFG.image_size)
		img_tensor  = self.transform(img)

		# load mask
		mask_name   = self.mask_filenames[idx]
		mask        = Image.open(mask_name)
		mask_tensor = self.transform(mask)

		# load mask image
		mask_img_name   = self.mask_img_filenames[idx]
		mask_img        = Image.open(mask_img_name)
		mask_img        = transforms.functional.crop(mask_img, ty, tx, CFG.image_size, CFG.image_size)
		mask_img_tensor = self.transform(mask_img)

		# load texture
		texture_name    = self.texture_filenames[idx]
		texture         = Image.open(texture_name)
		texture_tensor  = self.transform(texture)

		# set label data
		delta_m = np.zeros(8, dtype=self.fdtype)
		delta_m[6] = np.divide(tx, self.std_m[6])
		delta_m[7] = np.divide(32 - ty, self.std_m[7])

		m_label = self.all_m[idx] - delta_m
		m_tensor = torch.tensor(m_label)

		batch_exp_para      = self.all_exp_para[idx, :]
		exp_label           = np.matmul(batch_exp_para, np.transpose(self.w_exp))
		exp_label           = np.divide(exp_label, self.std_shape)
		exp_tensor          = torch.tensor(exp_label)

		batch_shape_para    = self.all_shape_para[idx, :]
		shape_label         = np.matmul(batch_shape_para, np.transpose(self.w_shape))
		shape_label         = np.divide(shape_label, self.std_shape)
		shape_tensor        = torch.tensor(shape_label)



		# set random albedo indices
		indices1 = np.random.randint(low=0, high=self.const_alb_mask.shape[0], size=[CFG.const_pixels_num])
		indices2 = np.random.randint(low=0, high=self.const_alb_mask.shape[0], size=[CFG.const_pixels_num])

		albedo_indices_x1 = torch.tensor(np.reshape(self.const_alb_mask[indices1, 1], [CFG.const_pixels_num, 1]))
		albedo_indices_y1 = torch.tensor(np.reshape(self.const_alb_mask[indices1, 0], [CFG.const_pixels_num, 1]))
		albedo_indices_x2 = torch.tensor(np.reshape(self.const_alb_mask[indices2, 1], [CFG.const_pixels_num, 1]))
		albedo_indices_y2 = torch.tensor(np.reshape(self.const_alb_mask[indices2, 0], [CFG.const_pixels_num, 1]))

		sample = {
				'image_name'    : self.image_filenames[idx],
				'image'         : img_tensor,
				'mask'          : mask_tensor,
				'mask_img'      : mask_img_tensor,
				'texture'       : texture_tensor,

				'm_label'       : m_tensor,
				'exp_label'     : exp_tensor,
				'shape_label'   : shape_tensor,

				'albedo_indices': [
					albedo_indices_x1,
					albedo_indices_y1,
					albedo_indices_x2,
					albedo_indices_y2
				],
		}

		return sample


	def is_splited( self ):
		'''
			Check whether the dataset is splited before
			Returns
				splited: Bool (True if splited before, otherwise False)
		'''
		return (
			isdir(join(self.dataset_dir, 'train')) and
			isdir(join(self.dataset_dir, 'valid')) and
			isdir(join(self.dataset_dir, 'test'))
		)


	def split_dataset( self ):
		'''
			Split dataset if no pre-processing being conducted before
		'''
		print("Raw dataset is not manipulated. Spliting dataset...")
		datasets = ['AFW', 'AFW_Flip', 'HELEN', 'HELEN_Flip', 'IBUG', 'IBUG_Flip', 'LFPW', 'LFPW_Flip']

		# split the parameters
		idDim = 1
		mDim = idDim + 8
		poseDim = mDim + 7
		shapeDim = poseDim + 199
		expDim = shapeDim + 29
		texDim = expDim + 40
		ilDim = texDim + 10
		# colorDim  = ilDim + 7

		all_images = []
		all_paras = []

		# load all image names and corresponding parameters
		for dataset in datasets:
			with open(join(self.dataset_dir, 'filelist', f'{dataset}_filelist.txt'), 'r') as fd:
				images = [line.strip() for line in fd.readlines()]

			with open(join(self.dataset_dir, 'filelist', f'{dataset}_param.dat'), 'r') as fd:
				paras = np.fromfile(file=fd, dtype=np.float32)

			paras = paras.reshape((-1, ilDim)).astype(np.float32)
			all_images += [images]
			all_paras += [paras]

		# all_images = list(filter(lambda x: int(x.split('_')[-1].split('.')[0]) < 8 ))
		all_images = np.concatenate(all_images, axis=0)
		all_paras = np.concatenate(all_paras, axis=0)
		assert all_images.shape[0] == all_paras.shape[0], "Number of samples must be the same between images and paras"

		# random sample the files
		total_len = all_images.shape[0]
		random_indices = list(range(total_len))
		random.shuffle(random_indices)

		phases = [
				('train', int(0.9 * total_len)),
				('valid', int(0.95 * total_len)),
				('test', int(total_len))
		]

		# split dataset
		bef = 0
		for phase, last_idx in phases:
			print(f"Spliting {phase} dataset ...")
			# create directories
			for dataset in datasets:
				makedirs(join(self.dataset_dir, phase, dataset), exist_ok=True)

			image_paths = all_images[bef:last_idx]
			paras = all_paras[bef:last_idx]
			bef = last_idx

			tot = list(zip(image_paths, paras))
			paths_and_paras = sorted(tot, key=lambda a: a[0])

			param = []
			names = []

			# copy image and mask_img files, duplicate mask and texture files
			# for image_path, para in paths_and_paras:
			def _split(image_path, para):
				if int(image_path.split('_')[-1].split('.')[0]) > 7:
					return ("", None)

				# if idx % 100 == 0:
				# 	print("        Splitting {} dataset progress: {:.2f}% ({})".format(
				# 		phase,
				# 		idx / image_paths.shape[0] * 100,
				# 		basename(image_path)
				# 	))
				target_name = join(self.dataset_dir, phase, image_path.split('.')[0])

				# copy image and mask image files
				image = join(self.dataset_dir, 'image', image_path)
				mask_img = join(self.dataset_dir, 'mask_img', image_path)
				shutil.copy(image, join(target_name + '_image.png'))
				shutil.copy(mask_img, target_name + '_mask_img.png')

				# copy mask and texture files
				mask = join(self.dataset_dir, 'mask', self.image2texture_fn(image_path))
				texture = join(self.dataset_dir, 'texture', self.image2texture_fn(image_path))
				shutil.copy(mask, target_name + '_mask.png')
				shutil.copy(texture, target_name + '_texture.png')

				names.append(target_name + '_')
				param.append(para)

				return (target_name + '_', para)

			result = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(_split)(image_path, para) for image_path, para in tqdm(paths_and_paras))


			# 5. write params to the proper directory
			# tot_ = list(zip(names, param))
			tot_ = result
			tot_ = sorted(tot_, key=lambda a: a[0])
			tot_ = list(filter(lambda x: x[0] != "", tot_))
			param_ = []
			for _, para in tot_:
				param_.append(para)
			param_ = np.stack(param_)
			np.save(join(self.dataset_dir, phase, 'param'), param_)

		print("     Splited dataset!")

	def load_dataset ( self , phase ):
		'''
			Load dataset
			Parameters
				phase: 'train', 'test', or 'valid'
		'''
		# split parameters
		idDim = 1
		mDim = idDim + 8
		poseDim = mDim + 7
		shapeDim = poseDim + 199
		expDim = shapeDim + 29
		texDim = expDim + 40
		ilDim = texDim + 10
		# colorDim  = ilDim + 7

		# load dataset images by filtering
		all_files           = glob(join(self.dataset_dir, phase, "*", "*.png"))
		image_filenames     = list(filter(lambda x: '_image.png' in x, all_files))
		mask_filenames      = list(filter(lambda x: '_mask.png' in x, all_files))
		mask_img_filenames  = list(filter(lambda x: '_mask_img.png' in x, all_files))
		texture_filenames   = list(filter(lambda x: '_texture.png' in x, all_files))

		image_filenames.sort()
		mask_filenames.sort()
		mask_img_filenames.sort()
		texture_filenames.sort()

		# TODO filename validation check (same prefix)

		self.image_filenames    = np.array(image_filenames)
		self.mask_filenames     = np.array(mask_filenames)
		self.mask_img_filenames = np.array(mask_img_filenames)
		self.texture_filenames  = np.array(texture_filenames)

		# load data parameter
		all_paras = np.load(join(self.dataset_dir, phase, "param.npy"))
		assert(self.image_filenames.shape[0] == all_paras.shape[0])

		# Identity parameter
		self.pids_300W = all_paras[:, 0:idDim]

		# Projection matrix parameter
		all_m = all_paras[:, idDim:mDim]
		self.all_m = np.divide(np.subtract(all_m, self.mean_m), self.std_m)

		# Shape parameter
		all_shape_para = all_paras[:, poseDim:shapeDim]
		self.mean_shape_para = np.mean(all_shape_para, axis=0)
		self.std_shape_para = np.std(all_shape_para, axis=0)
		self.all_shape_para = all_shape_para  # np.divide(np.subtract(all_shape_para, self.mean_shape_para), self.std_shape_para)

		# Expression parameter
		all_exp_para = all_paras[:, shapeDim:expDim]
		self.mean_exp_para = np.mean(all_exp_para, axis=0)
		self.std_exp_para = np.std(all_exp_para, axis=0)
		self.all_exp_para = all_exp_para  # np.divide(np.subtract(all_exp_para, self.mean_exp_para), self.std_exp_para)

		# Texture parameter
		tex_para = all_paras[:, expDim:texDim]
		all_tex_para = np.concatenate(tex_para, axis=0)

		all_il_para = all_paras[:, texDim:ilDim]
		self.mean_il_para = np.mean(all_il_para, axis=0)
		self.std_il_para = np.std(all_il_para, axis=0)
		self.all_il_para = all_il_para


	def image2texture_fn ( self, image_fn ):
		'''
			Return dataset pair. mask and texture is shared to several images.
			Parameters
				image_fn:   string
			Returns
				image_fn:   corresponding image file name
		'''
		last = image_fn[-7:].find('_')
		if (last < 0):
			return image_fn
		else:
			return image_fn[:-7 + last] + '_0.png'


def main():
	# import time
	print(torch.cuda.is_available())
	dataset = NonlinearDataset(phase='train', frac=1.0)
	print(len(dataset))
	dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=1)
	# start = time.time()
	print(len(dataloader))
	# return
	for idx, samples in enumerate(dataloader):
		if idx > 2:
			break
		print(f'{idx/len(dataloader) * 100:.2f}% : {samples["image"][0]}')
		# print(time.time() - start)
		# start = time.time()


if __name__ == "__main__":
	main()


