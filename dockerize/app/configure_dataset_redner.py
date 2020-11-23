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


def split_name(fname):
	id, pi, theta, rho = [float(t) for t in basename(fname).split('.jpg')[0].split('_')]
	return int(id), theta, pi, rho

def idx(fname):
	return int(fname.split('_')[-1][:-4])

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
	rotated = torch.mm(torch.mm(torch.mm (ro_x , ro_y) , ro_z), vertex.permute(1,0))
	return rotated.permute(1,0)


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
				join(self.dataset_dir, 'mask', basename(image_path).replace('.jpg', '_mask.jpg'))
				for image_path in self.image_paths
		]
		self.vertex_paths = {
				image_path : join(self.dataset_dir, 'vertex', '_'.join(basename(image_path).split('_')[:-2]) + '.npy')
				for image_path in self.image_paths
		}
		self.params = torch.tensor(np.load(join(self.dataset_dir, f'parameter.npy')), dtype=torch.float32)

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
		param_idx = int(img_name.split('_')[-1][:-4])
		params = self.params[param_idx]
		trans, angle, light, exp = torch.split(params, (3, 3, 27, 64), dim=-1)
		exp = torch.einsum('ij,aj->ai', CFG.exBase, exp)
		
		# read shape, color numpy file
		vertex_with_color = torch.tensor(np.load(self.vertex_paths[img_name]), dtype=torch.float32)
		vertex, color = torch.split(vertex_with_color, (3, 3), dim=-1)
		# vertex, color = get_blender_vc(vertex, color)
		vertex = vertex - exp - CFG.mean_shape
		vertex = torch.bmm(vertex, Compute_rotation_matrix(-angle)) - trans
		vertex = vertex.view([-1])
		
		color = color - CFG.mean_tex / 255.0

		sample = {
				'image_name'    : img_name,
				'image'         : img_tensor * mask_tensor,
				'mask'          : mask_tensor,
				
				'trans'         : trans,
				'angle'         : angle,
				'light'         : light,
				'exp'           : exp,
				
				'vertex'        : vertex,
				'vcolor'        : color,
				
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
		
		all_images = sorted(glob(join(self.dataset_dir, 'image/*.jpg')), key=idx)
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
		
		print("     Splitting obj files")
		# parameters = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(read_mat)(mat_name) for mat_name in tqdm(mat_names))
		
		parameters = []
		for mat_name in tqdm(mat_names):
			parameters.append(read_mat(mat_name))
		
		parameters = np.stack(parameters)
		np.save(join(CFG.dataset_path, 'patameters.npy'), parameters)
		
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
	dataset = NonlinearDataset(phase='train', frac=1.0)
	print(len(dataset))
	dataloader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=0)
	print(len(dataloader))
	
	for idx, samples in enumerate(dataloader):
		images, masks, _ = renderer.render(
							   vertex_batch=(samples['vertex'] + CFG.mean_shape + samples['exp']).view([CFG.batch_size, -1, 3]).cuda(),
		                       color_batch=(samples['vcolor'] + CFG.mean_tex).cuda(),
							   trans_batch=samples['trans'].cuda(),
							   rotation_batch=samples['rotate'].cuda(),
		                       light_batch=samples['light'].cuda(),
		                       print_timing=True)
		image = images[0].cpu().detach().numpy()
		mask = masks[0].cpu().detach().numpy()
		image_name = samples['image_name'][0]
		image_label = samples['image'][0].permute(1, 2, 0).cpu().detach().numpy()
		break





if __name__ == "__main__":
	main()


