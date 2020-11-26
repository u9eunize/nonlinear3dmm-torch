from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from os.path import join, basename, exists
from PIL import Image

from settings import CFG
from utils import *
from renderer.rendering_ops_redner import *
from joblib import Parallel, delayed
import multiprocessing
from scipy import io
from settings import init_3dmm_settings



class NonlinearDataset(Dataset):
	'''
		Nonlinear dataset load class
		it contains 2 functions,
			1. split raw data into train, test, and validation dataset and
			2. load each dataset item
	'''
	def __init__(self, phase, frac=1.0, dataset_dir=CFG.dataset_path):
		self.frac = frac

		# initialize attributes
		self.dataset_dir = dataset_dir
		self.transform = transforms.Compose([
			transforms.Resize((224, 224)),
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
		self.image_paths = self.image_paths[:int(len(self.image_paths) * self.frac)]
		self.mask_paths = [
				join(self.dataset_dir, 'mask', basename(image_path))
				for image_path in self.image_paths
		]
		self.vertex_paths = [
				join(self.dataset_dir, 'vertex', basename(image_path).replace('.jpg', '.npy'))
				for image_path in self.image_paths
		]
		self.params = torch.tensor(np.load(join(self.dataset_dir, 'parameter.npy')), dtype=torch.float32)

		print("Checking dataset validation")
		for img, mask, vertex in tqdm(list(zip(self.image_paths, self.mask_paths, self.vertex_paths))):
			assert basename(img)[:-4] == basename(mask)[:-4]
			assert basename(img)[:-4] == basename(vertex)[:-4]


	def __len__( self ):
		return len(self.image_paths)


	def __getitem__( self, idx ):
		# load image
		img_name    = self.image_paths[idx]
		img         = Image.open(img_name)
		img_tensor  = self.transform(img)
		
		# load mask
		mask_name = self.mask_paths[idx]
		mask = Image.open(mask_name)
		mask_tensor = self.transform(mask)
		
		# load camera parameters
		params = self.params[idx]
		angle, trans, light, exp = torch.split(params, (3, 3, 27, 64), dim=-1)
		exp = torch.einsum('ij,aj->ai', CFG.exBase_cpu, torch.unsqueeze(exp, 0))
		exp = exp.view((CFG.vertex_num, 3))
		
		# read shape, color numpy file
		vertex_with_color = torch.tensor(np.load(self.vertex_paths[idx]), dtype=torch.float32)
		vertex, vcolor = torch.split(vertex_with_color, (3, 3), dim=-1)
		# vertex, color = get_blender_vc(vertex, vcolor)
		vertex = vertex - torch.unsqueeze(trans, 0)
		vertex = torch.bmm(torch.unsqueeze(vertex, 0), Compute_rotation_matrix(torch.unsqueeze(-angle, 0), device='cpu'))
		vertex = torch.squeeze(vertex, 0)

		exp = exp.view(-1)
		shape = vertex.view(-1) - CFG.mean_shape_cpu - exp

		vcolor = vcolor - CFG.mean_tex_cpu

		sample = {
				'image_name'    : img_name,
				'image'         : img_tensor,
				'mask'          : mask_tensor,
				
				'trans'         : trans,
				'angle'         : angle,
				'light'         : light,
				'exp'           : exp,
				
				'shape'         : shape,
				'vcolor'        : vcolor,
				
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
		
		all_images = sorted(glob(join(self.dataset_dir, 'crop/*.jpg')))
		total_len = len(all_images)
		phases = [
				('train', int(0.9 * total_len)),
				('valid', int(0.95 * total_len)),
				('test', int(total_len))
		]
		
		################### write npy file from mat
		mat_names = sorted(glob(join(self.dataset_dir, 'mat/*.mat')))

		def read_mat ( mat_name ):
			# load translation and rotation coefficients
			mat_file = io.loadmat(mat_name)
			angle = mat_file['coeff'][0][224:227]
			trans = mat_file['coeff'][0][254:257]
			light = mat_file['coeff'][0][227:254]
			ex_coeff = mat_file['coeff'][0][80:144]  # expression
			return np.concatenate([angle, trans, light, ex_coeff])

		print("     Parsing mat files")
		parameters = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(read_mat)(mat_name) for mat_name in tqdm(mat_names))

		# parameters = []
		# for mat_name in tqdm(mat_names):
		# 	parameters.append(read_mat(mat_name))

		parameters = np.stack(parameters)
		np.save(join(CFG.dataset_path, 'parameter.npy'), parameters)
		
		################## write train, valid, test txt file
		bef = 0
		for phase, last_idx in phases:
			images = all_images[bef:last_idx]
			images = [line + '\n' for line in images]
			bef = last_idx
			with open(join(self.dataset_dir, f'{phase}.txt'), 'w') as f:
				f.writelines(images)
		
		

		
		
	
	
		

def main():
	init_3dmm_settings()
	dataset = NonlinearDataset(phase='train', frac=0.1)
	dataloader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=0)
	
	for idx, samples in enumerate(dataloader):
		shape = (samples['shape'] + samples['exp'] + CFG.mean_shape.cpu()).view([CFG.batch_size, -1, 3])
		images, masks, _ = renderer.render(
							   vertex_batch=shape.to(CFG.device),
		                       color_batch=(samples['vcolor'] + CFG.mean_tex.cpu()).to(CFG.device),
							   trans_batch=samples['trans'].to(CFG.device),
							   angle_batch=samples['angle'].to(CFG.device),
		                       light_batch=samples['light'].to(CFG.device),
		                       print_timing=True)
		image_ = torch.zeros_like(images[0])
		image_[:, :, 0] = images[0, :, :, 2]
		image_[:, :, 1] = images[0, :, :, 1]
		image_[:, :, 2] = images[0, :, :, 0]
		image = image_.cpu().detach().numpy()
		mask = masks[0].cpu().detach().numpy()
		image_name = samples['image_name'][0]
		image_label = samples['image'][0].permute(1, 2, 0).cpu().detach().numpy()

		continue





if __name__ == "__main__":
	main()


