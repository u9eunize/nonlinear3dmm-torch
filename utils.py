"""
Notes: Many of .dat files are written using Matlab.
Hence, there are "-1" subtraction to Python 0-based indexing
"""
from __future__ import division
import math
import numpy as np
import config
import torch
import os
from torchvision.utils import save_image
from os import makedirs
from os.path import join
from glob import glob




def save ( model, global_optimizer, encoder_optimizer, epoch, path):
	dir_path = get_checkpoint_dir(path, epoch)
	makedirs(dir_path, exist_ok=True)

	torch.save({
			'epoch'            : epoch + 1,
			'state_dict'       : model.state_dict(),
			'global_optimizer' : global_optimizer.state_dict(),
			'encoder_optimizer': encoder_optimizer.state_dict(),
	}, get_checkpoint_name(path, epoch))


def load ( model, global_optimizer=None, encoder_optimizer=None, start_epoch=None):
	if not config.CHECKPOINT_PATH:
		return model, global_optimizer, encoder_optimizer, 0

	if start_epoch:
		ckpt_name =join(config.CHECKPOINT_DIR_PATH, config.CHECKPOINT_PATH, f'ckpt_{start_epoch}', f'model_ckpt_{start_epoch}.pt')
	else:
		ckpt_name = glob(join(config.CHECKPOINT_DIR_PATH, config.CHECKPOINT_PATH, 'ckpt_*', 'model_ckpt_*'))[-1]

	checkpoint = torch.load(ckpt_name, map_location=config.DEVICE)
	start_epoch = checkpoint['epoch']
	model.load_state_dict(checkpoint['state_dict'])
	if global_optimizer is not None:
		global_optimizer.load_state_dict(checkpoint['global_optimizer'])
	if encoder_optimizer is not None:
		encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
	print("=> loaded checkpoint '{}' (epoch {})".format(ckpt_name, checkpoint['epoch']))

	return model, global_optimizer, encoder_optimizer, start_epoch


def get_checkpoint_dir ( path, number ):
	return os.path.join(path, f"ckpt_{number}")


def get_checkpoint_name ( path, number ):
	return os.path.join(f"{get_checkpoint_dir(path, number)}", f"model_ckpt_{number}.pt")



def load_3DMM_tri ():
	# Triangle definition (i.e. from Basel model)

	# print ('Loading 3DMM tri ...')

	fd = open(config.DEFINITION_PATH + '3DMM_tri.dat')
	tri = np.fromfile(file=fd, dtype=np.int32)
	fd.close()
	# print tri

	tri = tri.reshape((3, -1)).astype(np.int32)
	tri = tri - 1
	tri = np.append(tri, [[config.VERTEX_NUM], [config.VERTEX_NUM], [config.VERTEX_NUM]], axis=1)

	# print('   DONE')
	return tri


def load_3DMM_vertex_tri ():
	# Vertex to triangle mapping (list of all trianlge containing the cureent vertex)

	# print('Loading 3DMM vertex tri ...')

	fd = open(config.DEFINITION_PATH + '3DMM_vertex_tri.dat')
	vertex_tri = np.fromfile(file=fd, dtype=np.int32)
	fd.close()

	vertex_tri = vertex_tri.reshape((8, -1)).astype(np.int32)
	# vertex_tri = np.append(vertex_tri, np.zeros([8,1]), 1)
	vertex_tri[vertex_tri == 0] = config.TRI_NUM + 1
	vertex_tri = vertex_tri - 1

	# print('    DONE')
	return vertex_tri


def load_3DMM_vt2pixel ():
	# Mapping in UV space

	fd = open(config.DEFINITION_PATH + 'vertices_2d_u.dat')
	vt2pixel_u = np.fromfile(file=fd, dtype=np.float32)
	vt2pixel_u = np.append(vt2pixel_u - 1, 0)
	fd.close()

	fd = open(config.DEFINITION_PATH + 'vertices_2d_v.dat')
	vt2pixel_v = np.fromfile(file=fd, dtype=np.float32)
	vt2pixel_v = np.append(vt2pixel_v - 1, 0)
	fd.close()

	return vt2pixel_u, vt2pixel_v


def load_3DMM_kpts ():
	# 68 keypoints indices

	# print('Loading 3DMM keypoints ...')

	fd = open(config.DEFINITION_PATH + '3DMM_keypoints.dat')
	kpts = np.fromfile(file=fd, dtype=np.int32)
	kpts = kpts.reshape((-1, 1))
	fd.close()

	return kpts - 1


def load_3DMM_tri_2d ( with_mask=False ):
	fd = open(config.DEFINITION_PATH + '3DMM_tri_2d.dat')
	tri_2d = np.fromfile(file=fd, dtype=np.int32)
	fd.close()

	tri_2d = tri_2d.reshape(192, 224)

	tri_mask = tri_2d != 0

	tri_2d[tri_2d == 0] = config.TRI_NUM + 1  # VERTEX_NUM + 1
	tri_2d = tri_2d - 1

	if with_mask:
		return tri_2d, tri_mask

	return tri_2d


def load_Basel_basic ( element, is_reduce=False ):
	fn = config.DEFINITION_PATH + '3DMM_' + element + '_basis.dat'
	# print('Loading ' + fn + ' ...')

	fd = open(fn)
	all_paras = np.fromfile(file=fd, dtype=np.float32)
	fd.close()

	all_paras = np.transpose(all_paras.reshape((-1, config.N)).astype(np.float32))

	mu = all_paras[:, 0]
	w = all_paras[:, 1:]

	# print('    DONE')

	return mu, w


def load_const_alb_mask ():
	fd = open(config.DEFINITION_PATH + '3DMM_const_alb_mask.dat')
	const_alb_mask = np.fromfile(file=fd, dtype=np.uint8)
	fd.close()
	const_alb_mask = const_alb_mask - 1
	const_alb_mask = const_alb_mask.reshape((-1, 2)).astype(np.uint8)

	return const_alb_mask


def load_3DMM_tri_2d_barycoord ():
	fd = open(config.DEFINITION_PATH + '3DMM_tri_2d_barycoord_reduce.dat')
	tri_2d_barycoord = np.fromfile(file=fd, dtype=np.float32)
	fd.close()

	tri_2d_barycoord = tri_2d_barycoord.reshape(192, 224, 3)

	return tri_2d_barycoord



def inverse_transform(images):
	return (images+1.)/2.


def merge(images, size):
	h, w = images.shape[1], images.shape[2]
	nn = images.shape[0]

	if size[1] < 0:
		size[1] = int(math.ceil(nn/size[0]))
	if size[0] < 0:
		size[0] = int(math.ceil(nn/size[1]))


	if (images.ndim == 4):
		img = np.zeros((h * size[0], w * size[1], 3))
		for idx, image in enumerate(images):
			i = idx % size[1]
			j = idx // size[1]
			img[j*h:j*h+h, i*w:i*w+w, :] = image
	else:
		img = images


	return img



def imsave ( images, size, path ):
	img = merge(images, size)
	img = torch.Tensor(img)
	img = img.permute((2, 0, 1))
	# plt.imshow(img)
	# plt.show()
	return save_image(img, path)


def save_images(images, size, image_path, inverse = True):
	if len(size) == 1:
		size= [size, -1]
	if size[1] == -1:
		size[1] = int(math.ceil(images.shape[0]/size[0]))
	if size[0] == -1:
		size[0] = int(math.ceil(images.shape[0]/size[1]))
	if (inverse):
		images = inverse_transform(images)

	return imsave(images, size, image_path)


