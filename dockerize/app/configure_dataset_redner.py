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
import matplotlib.pyplot as plt



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
		for img, mask in tqdm(list(zip(self.image_paths, self.mask_paths))):
			assert basename(img)[:-4] == basename(mask)[:-4]


	def __len__( self ):
		return len(self.image_paths)


	def __getitem__( self, idx ):
		# load image
		img_name    = self.image_paths[idx]
		img         = Image.open(img_name)
		b, g, r 	= img.split()
		img 		= Image.merge("RGB", (r, g, b))
		img_tensor  = self.transform(img)
		
		# load mask
		mask_name = self.mask_paths[idx]
		mask = Image.open(mask_name)
		mask_tensor = self.transform(mask)
		
		# load camera parameters
		params = self.params[idx]
		shape, exp, tex, angle, light, trans = torch.split(params, (80, 64, 80, 3, 27, 3), dim=-1)

		# shape = torch.einsum('ij,aj->ai', CFG.shapeBase_cpu, torch.unsqueeze(shape, 0))
		# shape = shape.view(-1)

		exp = torch.einsum('ij,aj->ai', CFG.exBase_cpu, torch.unsqueeze(exp, 0))
		exp = exp.view([-1, 3])[CFG.blender_to_deep_cpu].view(-1)

		# tex = torch.einsum('ij,aj->ai', CFG.texBase_cpu, torch.unsqueeze(tex, 0)) + CFG.mean_tex_cpu
		# tex = tex.view(-1) / 255.0
		
		# read shape, color numpy file
		vertex_with_color = torch.tensor(np.load(self.vertex_paths[idx]), dtype=torch.float32)[CFG.blender_to_deep_cpu]
		vertex, vcolor = torch.split(vertex_with_color, (3, 3), dim=-1)
		# vertex, color = get_blender_vc(vertex, vcolor)
		vertex = vertex - torch.unsqueeze(trans, 0)
		vertex = torch.bmm(torch.unsqueeze(vertex, 0), Compute_rotation_matrix(torch.unsqueeze(-angle, 0), device='cpu'))
		vertex = torch.squeeze(vertex, 0)
		exp = exp.view(-1)
		shape = vertex.view(-1) - CFG.mean_shape_cpu - exp

		# remove light effect and subtract mean tex
		vcolor = vcolor.view([-1, 3])
		vcolor = remove_light(vertex_batch=vertex.unsqueeze(0), color_batch=vcolor.unsqueeze(0), light_batch=light.unsqueeze(0), angle_batch=angle.unsqueeze(0), device="cpu")
		vcolor = vcolor.squeeze(0)
		vcolor -= CFG.mean_tex_cpu

		# set random albedo indices
		indices1 = np.random.randint(low=0, high=CFG.const_alb_mask.shape[0], size=[CFG.const_pixels_num])
		indices2 = np.random.randint(low=0, high=CFG.const_alb_mask.shape[0], size=[CFG.const_pixels_num])

		albedo_indices_x1 = CFG.const_alb_mask[indices1, 0].view([CFG.const_pixels_num, 1]).long()
		albedo_indices_y1 = CFG.const_alb_mask[indices1, 1].view([CFG.const_pixels_num, 1]).long()
		albedo_indices_x2 = CFG.const_alb_mask[indices2, 0].view([CFG.const_pixels_num, 1]).long()
		albedo_indices_y2 = CFG.const_alb_mask[indices2, 1].view([CFG.const_pixels_num, 1]).long()

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
			# shape 	= mat_file['coeff'][0][:80]
			# exp		= mat_file['coeff'][0][80:144]
			# tex 	= mat_file['coeff'][0][144:224]
			# angle 	= mat_file['coeff'][0][224:227]
			# light 	= mat_file['coeff'][0][227:254]
			# trans 	= mat_file['coeff'][0][254:257]

			return mat_file['coeff'][0]

		print("     Parsing mat files")
		parameters = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(read_mat)(mat_name) for mat_name in tqdm(mat_names))
		#
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
	batch_size = 4
	init_3dmm_settings()
	dataset = NonlinearDataset(phase='train', frac=0.1)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

	for idx, samples in enumerate(dataloader):
		shape = (samples['shape'] + samples['exp'] + CFG.mean_shape_cpu).view([batch_size, -1, 3])

		angle = samples['angle']
		trans = samples['trans']

		start = time()
		images, masks, _ = renderer.render(
							   vertex_batch=shape.to(CFG.device),
		                       color_batch=samples['vcolor'].to(CFG.device) + CFG.mean_tex,
							   trans_batch=trans.to(CFG.device),
							   angle_batch=angle.to(CFG.device),
		                       light_batch=samples['light'].to(CFG.device),
		                       print_timing=False)
		print(f'***** rendering time : {time() - start}')
		r, g, b = torch.split(images, (1, 1, 1), dim=3)
		images = torch.cat([b, g, r], dim=3)
		images = torch.cat([image for image in images], dim=1)

		image_labels = samples['image'].permute(0, 2, 3, 1).to(CFG.device)
		r, g, b = torch.split(image_labels, (1, 1, 1), dim=3)
		image_labels = torch.cat([b, g, r], dim=3)
		image_labels = torch.cat([image for image in image_labels], dim=1)

		masks = torch.cat([mask for mask in masks], dim=1)
		maskeds = images * masks + image_labels * (1 - masks)

		images = images.cpu().detach().numpy()
		image_labels = image_labels.cpu().detach().numpy()
		maskeds = maskeds.cpu().detach().numpy()

		continue





if __name__ == "__main__":
	main()


