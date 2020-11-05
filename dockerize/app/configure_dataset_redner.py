from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from os.path import join, basename, exists
from PIL import Image

from settings import CFG
from utils import *
from renderer.rendering_ops_redner import Batch_Renderer
from joblib import Parallel, delayed
import multiprocessing


def split_name(fname):
	id, pi, theta, rho = [float(t) for t in basename(fname).split('.jpg')[0].split('_')]
	return int(id), theta, pi, rho

def translate_vertex(vertex, coeff_translate):
	translated = vertex + coeff_translate
	return translated

def rotate_vertex(vertex, coeff_rotate):
	coeff_x, coeff_y, coeff_z = coeff_rotate
	ro_x = torch.tensor(
		[[1, 0, 0], [0, math.cos(coeff_x), -math.sin(coeff_x)], [0, math.sin(coeff_x), math.cos(coeff_x)]])
	ro_y = torch.tensor(
		[[math.cos(coeff_y), 0, math.sin(coeff_y)], [0, 1, 0], [-math.sin(coeff_y), 0, math.cos(coeff_y)]])
	ro_z = torch.tensor(
		[[math.cos(coeff_z), -math.sin(coeff_z), 0], [math.sin(coeff_z), math.cos(coeff_z), 0], [0, 0, 1]])
	rotated = torch.mm(torch.mm(torch.mm (ro_x , ro_y) , ro_z), vertex.permute(1,0)).permute(1,0)
	return rotated


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

		# split dataset into train, validation, test set with ratio 8:1:1
		if not self.is_splited():
			print('Dataset is not splited. ')
			self.split_dataset()

		self.load_dataset(phase)
	
	def load_dataset ( self , phase ):
		'''
			Load dataset
			Parameters
				phase: 'train', 'test', or 'valid'
		'''
		# load dataset images by filtering
		txt = open(join(self.dataset_dir, f'{phase}.txt'), 'r')
		self.image_paths = [l.strip() for l in txt.readlines()]
		self.mask_paths = [
				join(self.dataset_dir, 'mask', basename(image_path).replace('.jpg', '_depth.jpg'))
				for image_path in self.image_paths
		]
		self.vertex_paths = {
				image_path : join(self.dataset_dir, 'vertex', '_'.join(basename(image_path).split('_')[:-2]) + '.npy')
				for image_path in self.image_paths
		}
		self.texture_paths = {
				image_path : join(self.dataset_dir, 'texture', '_'.join(basename(image_path).split('_')[:-2]) + '.jpg')
				for image_path in self.image_paths
		}
		self.params = torch.tensor(np.load(join(self.dataset_dir, f'{phase}_params.npy')), dtype=torch.float32)
		
		assert len(self.image_paths) == self.params.shape[0]

	def __len__( self ):
		return int(len(self.image_paths) * self.frac)


	def __getitem__( self, idx ):
		# set random crop index
		ty = np.random.randint(0, 32 + 1)
		tx = np.random.randint(0, 32 + 1)

		# load image
		img_name    = self.image_paths[idx]
		img         = Image.open(img_name)
		img         = transforms.functional.crop(img, ty, tx, CFG.image_size, CFG.image_size)
		img_tensor  = self.transform(img)
		
		# load texture
		tex_name    = self.texture_paths[img_name]
		tex         = Image.open(tex_name)
		tex_tensor  = self.transform(tex)
		
		# load camera parameters
		params = self.params[idx]
		camera, light, rotate, trans = torch.split(params, (3, 3, 3, 3), dim=-1)
		
		# read shape, color numpy file
		vertex_with_color = torch.tensor(np.load(self.vertex_paths[img_name]), dtype=torch.float32)
		vertex, color = torch.split(vertex_with_color, (3, 3), dim=-1)
		vertex = translate_vertex(vertex, trans)
		vertex = rotate_vertex(vertex, rotate)

		sample = {
				'image_name'    : img_name,
				'image'         : img_tensor,
				'vertex'        : vertex,
				'color'         : color / 255.0,
				'texture'       : tex_tensor,

				'camera'        : camera,
				'light'         : light,
		}

		return sample


	def is_splited( self ):
		'''
			Check whether the dataset is splited before
			Returns
				splited: Bool (True if splited before, otherwise False)
		'''
		return (
			exists(join(self.dataset_dir, 'train.txt')) and
			exists(join(self.dataset_dir, 'valid.txt')) and
			exists(join(self.dataset_dir, 'test.txt'))
		)


	def split_dataset( self ):
		'''
			Split dataset if no pre-processing being conducted before
		'''
		################### write train, valid, test txt file
		# all_images = sorted(glob(join(self.dataset_dir, 'image/*.jpg')))
		# total_len = len(all_images)
		# phases = [
		# 		('train', int(0.9 * total_len)),
		# 		('valid', int(0.95 * total_len)),
		# 		('test', int(total_len))
		# ]
		# bef = 0
		# for phase, last_idx in phases:
		# 	images = all_images[bef:last_idx]
		# 	images = [line + '\n' for line in images]
		# 	bef = last_idx
		# 	with open(join(self.dataset_dir, f'{phase}.txt'), 'w') as f:
		# 		f.writelines(images)

		################### write npy file from npy
		def color2uv ( vcolor, uv_mapper ):
			buffer = torch.zeros([3, 256, 256])

			color = torch.from_numpy(vcolor[:, 3:])
			color = torch.cat([color, torch.zeros([1, 3])], dim=0)

			u, v = torch.transpose(uv_mapper[:, :2], 0, 1)
			v1, v2, v3 = torch.transpose(uv_mapper[:, 2:], 0, 1)
			avg_color = torch.transpose((color[v1] + color[v2] + color[v3]) / 3, 0, 1)
			buffer[:, u, v] = avg_color

			return buffer

		def read_obj ( obj_name ):
			vertex_with_color = []
			with open(obj_name) as f:
				for line in f.readlines():
					splited = line.split(' ')
					if len(splited) == 1:
						break
					else:
						mode, x, y, z, r, g, b = splited
						vertex_with_color.append(np.array([float(x), float(y), float(z), float(r), float(g), float(b)]))

			vcolor = np.stack(vertex_with_color).astype(np.float32)
			np.save(join(self.dataset_dir, 'vertex', basename(obj_name[:-4])), vcolor)
			texture = color2uv(vcolor, uv_mapper)
			save_image(texture, join(self.dataset_dir, 'texture', basename(obj_name[:-4] + '.jpg')))

		uv_mapper_ = open(join(self.dataset_dir, 'bfm2009.idx'), 'r').readlines()
		uv_mapper = []
		for line in uv_mapper_:
			uv_mapper.append(np.array([int(x) for x in line.split(' ')]))

		uv_mapper = np.stack(uv_mapper)
		uv_mapper = torch.tensor(uv_mapper)

		indices_black = uv_mapper[:, 2:] == torch.tensor([0, 0, 0])
		uv_mapper[:, 2:][indices_black] = torch.max(uv_mapper) + 1

		obj_names = glob(join(self.dataset_dir, 'obj/*.obj'))
		Parallel(n_jobs=multiprocessing.cpu_count())(delayed(read_obj)(obj_name) for obj_name in tqdm(obj_names))
		# for obj_name in tqdm(obj_names):
		# 	read_obj(obj_name)

		################### concat parameters and rot and trans
		# camera_light = np.load(join(self.dataset_dir, 'parameter.npy'))
		# mat_names = sorted(glob(join(self.dataset_dir, 'mat/*.mat')))
		#
		# def concat(param, mat_name):
		# 	mat_file = io.loadmat(mat_name)
		# 	# rotation matrix
		# 	rotat = mat_file['coeff'][0][224:227]
		# 	trans = mat_file['coeff'][0][254:257]
		# 	return np.concatenate([param, rotat, trans], axis=-1)
		#
		# params = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(concat)(param, mat_name) for param, mat_name in tqdm(list(zip(camera_light, mat_names))))
		# params = np.stack(params)
		#
		# bef = 0
		# for phase, last_idx in phases:
		# 	np.save(join(self.dataset_dir, f'{phase}_params.npy'), params[bef:last_idx])
		# 	bef = last_idx
		
		
		
		
	
	
		

def main():
	dataset = NonlinearDataset(phase='train', frac=1.0)
	print(len(dataset))
	dataloader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=1)
	print(len(dataloader))
	
	for idx, samples in enumerate(dataloader):
		if idx > 2:
			break
		# print(f'{idx/len(dataloader) * 100:.2f}% : {samples["image"][0]}')
		renderer = Batch_Renderer()
		images, masks = renderer.render(
							   vertex_batch=samples['vertex'].cuda(),
		                       color_batch=samples['color'].cuda(),
		                       camera_batch=samples['camera'].cuda(),
		                       light_batch=samples['light'].cuda(),
		                       resolution=(CFG.image_size, CFG.image_size),
		                       print_timing=True)
		# print(time.time() - start)
		# start = time.time()


if __name__ == "__main__":
	main()


