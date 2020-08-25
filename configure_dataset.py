import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import _300W_LP_DIR
from os import cpu_count
from os.path import join
from PIL import Image
from os import makedirs
from glob import glob


VERTEX_NUM  = 53215
TRI_NUM     = 105840
N           = VERTEX_NUM * 3



class NonlinearDataset(Dataset):
	def __init__( self, dataset_dir=_300W_LP_DIR ):
		# initialize attributes
		self.dataset_dir = dataset_dir
		self.transform = transforms.Compose([
				transforms.ToTensor(),
		])

		# load mean and std shape
		self.mean_m = np.load(join(self.dataset_dir, 'mean_m.npy'))
		self.std_m = np.load(join(self.dataset_dir, 'std_m.npy'))

		# self.ty = 32 - np.random.random_integers(0, 32, 1)
		self.ty = 32 - np.random.randint(0, 32)
		# self.tx = np.random.random_integers(0, 32, 1)
		self.tx = 32 - np.random.randint(0, 32)

		self.delta_m = np.zeros(8)
		self.delta_m[6] = np.divide(self.ty, self.std_m[6])
		self.delta_m[7] = np.divide(32 - self.tx, self.std_m[7])

		mu_shape, w_shape = self.load_Basel_basic('shape')
		mu_exp, w_exp = self.load_Basel_basic('exp')

		self.mean_shape = mu_shape + mu_exp
		self.std_shape = np.tile(np.array([1e4, 1e4, 1e4]), VERTEX_NUM)
		# self.std_shape  = np.load('std_shape.npy')

		# self.mean_shape_tf = tf.constant(self.mean_shape, tf.float32)
		# self.std_shape_tf = tf.constant(self.std_shape, tf.float32)

		# self.mean_m_tf = tf.constant(self.mean_m, tf.float32)
		# self.std_m_tf = tf.constant(self.std_m, tf.float32)

		self.w_shape = w_shape
		self.w_exp = w_exp

		# load image and parameters
		self.load_300w_LP_dataset()


	def __len__( self ):
		return self.image_filenames.shape[0]


	def __getitem__( self, idx ):

		img_name = join(self.dataset_dir, 'image', self.image_filenames[idx])
		img = Image.open(img_name)
		img_tensor = self.transform(img)

		mask_img_name = join(self.dataset_dir, 'mask_img', self.image_filenames[idx])
		mask_img = Image.open(mask_img_name)
		mask_img_tensor = self.transform(mask_img)

		texture_name = join(self.dataset_dir, 'texture', self.image2texture_fn(self.image_filenames[idx]))
		texture = Image.open(texture_name)
		texture_tensor = self.transform(texture)

		mask_name = join(self.dataset_dir, 'mask', self.image2texture_fn(self.image_filenames[idx]))
		mask = Image.open(mask_name)
		mask_tensor = self.transform(mask)

		m_label = self.all_m[idx] - self.delta_m

		batch_shape_para = self.all_shape_para[idx, :]
		batch_exp_para = self.all_exp_para[idx, :]
		shape_label = np.divide( np.matmul(batch_shape_para, np.transpose(self.w_shape)) + np.matmul(batch_exp_para, np.transpose(self.w_exp)), self.std_shape)

		sample = {
				'image_name'    : self.image_filenames[idx],
				'image'         : img_tensor,
				'mask_img'      : mask_img_tensor,
				'texture'       : texture_tensor,
				'mask'          : mask_tensor,

				'm_label'       : m_label,
				'shape_label'   : shape_label,
				'height_offset' : self.tx,
				'width_offset'  : self.ty,
		}

		return sample


	def load_Basel_basic ( self, element, is_reduce=False ):
		with open(join(self.dataset_dir, '3DMM_definition', f'3DMM_{element}_basis.dat'), 'r') as fd:
			all_paras = np.fromfile(file=fd, dtype=np.float32)

		all_paras = np.transpose(all_paras.reshape((-1, N)).astype(np.float32))

		mu = all_paras[:, 0]
		w = all_paras[:, 1:]


		return mu, w


	def load_300w_LP_dataset ( self ):
		datasets = ['AFW', 'AFW_Flip', 'HELEN', 'HELEN_Flip', 'IBUG', 'IBUG_Flip', 'LFPW', 'LFPW_Flip']
		datasets = ['AFW']
		images, pid, m, pose, shape, exp, tex_para, il, tex, alb, mask = [[0] * len(datasets) for _ in range(11)]

		for idx, dataset in enumerate(datasets):
			with open(join(self.dataset_dir, 'filelist', f'{dataset}_filelist.txt'), 'r') as fd:
				all_images = [line.strip() for line in fd.readlines()]

			with open(join(self.dataset_dir, 'filelist', f'{dataset}_param.dat'), 'r') as fd:
				all_paras = np.fromfile(file=fd, dtype=np.float32)

			idDim = 1
			mDim = idDim + 8
			poseDim = mDim + 7
			shapeDim = poseDim + 199
			expDim = shapeDim + 29
			texDim = expDim + 40
			ilDim = texDim + 10
			# colorDim  = ilDim + 7

			all_paras = all_paras.reshape((-1, ilDim)).astype(np.float32)
			assert (len(all_images) == all_paras.shape[
				0]), "Number of samples must be the same between images and paras"

			images[idx], pid[idx], m[idx], pose[idx], shape[idx], exp[idx], tex_para[idx], il[idx] = \
				all_images, \
				all_paras[:, 0:idDim], \
				all_paras[:, idDim:mDim], \
				all_paras[:, mDim:poseDim], \
				all_paras[:, poseDim:shapeDim], \
				all_paras[:, shapeDim:expDim], \
				all_paras[:, expDim:texDim], \
				all_paras[:, texDim:ilDim]
			# color = all_paras[:,ilDim:colorDim]

		self.image_filenames = np.concatenate(images, axis=0)
		all_m = np.concatenate(m, axis=0)
		all_shape_para = np.concatenate(shape, axis=0)
		all_exp_para = np.concatenate(exp, axis=0)
		self.all_tex_para = np.concatenate(tex_para, axis=0)
		self.pids_300W = np.concatenate(pid, axis=0)
		# self.all_il       = np.concatenate(il, axis=0)

		self.all_m = np.divide(np.subtract(all_m, self.mean_m), self.std_m)

		self.mean_shape_para = np.mean(all_shape_para, axis=0)
		self.std_shape_para = np.std(all_shape_para, axis=0)
		self.all_shape_para = all_shape_para  # np.divide(np.subtract(all_shape_para, self.mean_shape_para), self.std_shape_para)

		self.mean_exp_para = np.mean(all_exp_para, axis=0)
		self.std_exp_para = np.std(all_exp_para, axis=0)
		self.all_exp_para = all_exp_para  # np.divide(np.subtract(all_exp_para, self.mean_exp_para), self.std_exp_para)


	def image2texture_fn ( self, image_fn ):
		last = image_fn[-7:].find('_')
		if (last < 0):
			return image_fn
		else:
			return image_fn[:-7 + last] + '_0.png'


def main():
	dataloader = DataLoader(NonlinearDataset(), batch_size=128, shuffle=True, num_workers=0)
	for idx, samples in enumerate(dataloader):
		print(f'{idx/len(dataloader) * 100:.2f}% : {samples["image_name"]}')
	return



if __name__ == "__main__":
	main()


