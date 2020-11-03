from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from os.path import join, isdir, basename
from PIL import Image
from utils import *
from settings import CFG
from plyfile import PlyData
import multiprocessing
from joblib import Parallel, delayed
from renderer.rendering_ops_redner import render
import pyredner

def split_name(fname):
	id, pi, theta, rho = [float(t) for t in basename(fname).split('.jpg')[0].split('_')]
	return int(id), theta, pi, rho


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
		
		# read shape, color numpy file
		vertex_with_color = torch.tensor(np.load(self.vertices_filenames[idx]), dtype=torch.float32)
		vertex, color = torch.split(vertex_with_color, (3, 3), dim=-1)
		
		# load camera parameters
		camera = torch.tensor(self.all_cameras[idx], dtype=torch.float32)

		sample = {
				'image_name'    : self.image_filenames[idx],
				'image'         : img_tensor,
				'texture'       : color / 255.0,

				'm_label'       : camera,
				'shape_label'   : vertex,
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

		all_images = glob(join(self.dataset_dir, 'images/*.jpg'))
		
		train_images = list(filter(lambda x: int(basename(x).split('_')[0]) < 1800, all_images))
		valid_images = list(filter(lambda x: int(basename(x).split('_')[0]) >= 1800 and int(basename(x).split('_')[0]) < 1900, all_images))
		test_images = list(filter(lambda x: int(basename(x).split('_')[0]) >= 1900, all_images))
		
		for phase, images in zip(['train', 'valid', 'test'], [train_images, valid_images, test_images]):
			makedirs(join(self.dataset_dir, phase), exist_ok=True)
			
			def _split(image_name):
				target_name = join(self.dataset_dir, phase, basename(image_name))
				shutil.copy(image_name, target_name)
				
				id, theta, pi, rho = split_name(image_name)
				camera = np.stack([theta, pi, rho], axis=0)
				
				ply_name = join(self.dataset_dir, f'plys/{id}.ply')
				plydata = PlyData.read(ply_name)
				x, y, z = plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']
				r, g, b = plydata['vertex']['red'], plydata['vertex']['green'], plydata['vertex']['blue']
				vertex = np.stack([x, y, z], axis=1)
				color = np.stack([r, g, b], axis=1)
				
				vertex_with_color = np.concatenate([vertex, color], axis=-1)
				np.save(join(self.dataset_dir, phase, basename(image_name).split('.jpg')[0]), vertex_with_color)
				
				shutil.copy(image_name, target_name)
				
				return (target_name, camera)
			
			# for image_path in images:
			# 	_split(image_path)
			# 	print(image_path)
			result_ = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(_split)(image_path) for image_path in tqdm(images))
			
			result = sorted(result_, key=lambda a: a[0])
			cameras = []
			for _, camera in result:
				cameras.append(camera)
			cameras = np.stack(cameras, axis=0)
			np.save(join(self.dataset_dir, phase, 'cameras'), cameras)
			
		# save face indices
		ply_sample = PlyData.read(glob(join(self.dataset_dir, 'plys/1.ply'))[0])
		face = np.vstack(ply_sample['face']['vertex_indices'])
		np.save(join(self.dataset_dir, 'face'), face)
		
		print("     Splited dataset!")
	
	def load_dataset ( self , phase ):
		'''
			Load dataset
			Parameters
				phase: 'train', 'test', or 'valid'
		'''
		# load dataset images by filtering
		all_files           = glob(join(self.dataset_dir, phase, "*"))
		image_filenames     = list(filter(lambda x: '.jpg' in x, all_files))
		vertices_filenames      = list(filter(lambda x: '.npy' in x, all_files))

		image_filenames.sort()
		vertices_filenames.sort()

		for fname1, fname2 in zip(image_filenames, vertices_filenames):
			assert basename(fname1).split('.jpg')[0] == basename(fname2).split('.npy')[0]

		self.image_filenames    = np.array(image_filenames)
		self.vertices_filenames     = np.array(vertices_filenames)

		# load data parameter
		all_cameras = np.load(join(self.dataset_dir, phase, "cameras.npy"))
		assert(self.image_filenames.shape[0] == all_cameras.shape[0])
		self.all_cameras = all_cameras

def main():
	batch_size = 20
	dataset = NonlinearDataset(phase='train', frac=1.0)
	print(len(dataset))
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
	print(len(dataloader))
	
	
	face = torch.tensor(np.load('taehwan_dataset/face.npy'), device=pyredner.get_device())
	
	theta_light = math.pi * -0.5
	pi_light = 0
	rho_light = 100
	intensity = 5
	light_param = torch.tensor([theta_light, pi_light, rho_light, intensity])
	
	# render
	def batch_wise ( tensor, batch_size ):
		dim = len(tensor.shape)
		args = [batch_size] + [1] * dim
		return torch.unsqueeze(tensor, dim=0).repeat(*args)
	
	for idx, samples in enumerate(dataloader):
		if idx > 2:
			break
		# print(f'{idx/len(dataloader) * 100:.2f}% : {samples["image"][0]}')
		images, masks = render(samples['shape_label'].cuda(),
		                       samples['texture'].cuda(),
		                       samples['m_label'].cuda(),
		                       batch_wise(light_param, batch_size).cuda(),
		                       resolution=(224,224),
		                       print_timing=True)
		# print(time.time() - start)
		# start = time.time()


if __name__ == "__main__":
	main()


