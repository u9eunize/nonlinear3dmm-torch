from network.Nonlinear_3DMM_redner import Nonlinear3DMM_redner
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torchvision import transforms
from configure_dataset_redner import NonlinearDataset
from settings import *
import numpy as np
from PIL import Image
from glob import glob
from renderer.rendering_ops_redner import renderer
from utils import load
import face_alignment
from tqdm import tqdm
from skimage import io
from scipy.io import loadmat
from os.path import basename
import torch.nn.functional as F



def save_to_obj(name, vertex, face):
	with open(name, 'w') as fd:
		for v1, v2, v3 in vertex:
			fd.write(f'v {v1:.3f} {v2:.3f} {v3:.3f}\n')

		fd.write("\n")
		for f1, f2, f3 in face:
			fd.write(f'f {f1 + 1} {f2 + 1} {f3 + 1}\n')

	print(name)

def extract_landmark(fa, img_name):
	img = io.imread(img_name)
	# b, g, r = np.split(img, 3, axis=2)
	# img = np.concatenate([r, g, b], axis=2)
	try:
		preds = fa.get_landmarks(img)
		if preds is None:
			return
	except:
		return

	nose = preds[0][30]
	left_eye = (((preds[0][36][0] + preds[0][39][0]) / 2), (preds[0][36][1] + preds[0][39][1]) / 2)
	right_eye = ((preds[0][42][0] + preds[0][45][0]) / 2, (preds[0][42][1] + preds[0][45][1]) / 2)
	left_mouth = preds[0][48]
	right_mouth = preds[0][54]

	return np.array([left_eye, right_eye, nose, left_mouth, right_mouth])

def POS(xp,x):
	npts = xp.shape[1]

	A = np.zeros([2*npts,8])

	A[0:2*npts-1:2,0:3] = x.transpose()
	A[0:2*npts-1:2,3] = 1

	A[1:2*npts:2,4:7] = x.transpose()
	A[1:2*npts:2,7] = 1

	b = np.reshape(xp.transpose(),[2*npts,1])

	k,_,_,_ = np.linalg.lstsq(A,b,rcond=None)

	R1 = k[0:3]
	R2 = k[4:7]
	sTx = k[3]
	sTy = k[7]
	s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
	t = np.stack([sTx,sTy],axis = 0)

	return t,s

def process_img(img,lm,t,s,target_size = 224.):
	w0,h0 = img.size
	w = (w0/s*102).astype(np.int32)
	h = (h0/s*102).astype(np.int32)
	img = img.resize((w,h),resample = Image.BICUBIC)

	left = (w/2 - target_size/2 + float((t[0] - w0/2)*102/s)).astype(np.int32)
	right = left + target_size
	up = (h/2 - target_size/2 + float((h0/2 - t[1])*102/s)).astype(np.int32)
	below = up + target_size

	img = img.crop((left,up,right,below))
	img = np.array(img)
	# img = img[:,:,::-1] #RGBtoBGR
	img = np.expand_dims(img,0)
	lm = np.stack([lm[:,0] - t[0] + w0/2,lm[:,1] - t[1] + h0/2],axis = 1)/s*102
	lm = lm - np.reshape(np.array([(w/2 - target_size/2),(h/2-target_size/2)]),[1,2])

	return img,lm

def Preprocess(fa, file, lm3D):
	lm = extract_landmark(fa, file)
	if lm is None:
		return None, None, None
	img = Image.open(file)

	w0,h0 = img.size

	# change from image plane coordinates to 3D sapce coordinates(X-Y plane)
	lm = np.stack([lm[:,0],h0 - 1 - lm[:,1]], axis = 1)

	# calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
	t,s = POS(lm.transpose(),lm3D.transpose())

	# processing the image
	img_new,lm_new = process_img(img,lm,t,s)
	lm_new = np.stack([lm_new[:,0],223 - lm_new[:,1]], axis = 1)
	trans_params = np.array([w0,h0,102.0/s,t[0],t[1]])

	return img_new,lm_new,trans_params

def load_lm3d():

	Lm3D = loadmat('/dataset/BFM/similarity_Lm3D_all.mat')
	Lm3D = Lm3D['lm']

	# calculate 5 facial landmarks using 68 landmarks
	lm_idx = np.array([31,37,40,43,46,49,55]) - 1
	Lm3D = np.stack([Lm3D[lm_idx[0],:],np.mean(Lm3D[lm_idx[[1,2]],:],0),np.mean(Lm3D[lm_idx[[3,4]],:],0),Lm3D[lm_idx[5],:],Lm3D[lm_idx[6],:]], axis = 0)
	Lm3D = Lm3D[[1,2,0,3,4],:]

	return Lm3D

def load_img(img_path,lm_path):

	image = Image.open(img_path)
	lm = np.loadtxt(lm_path)

	return image,lm

def main():
	init_3dmm_settings()
	is_testset = False
	fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=CFG.device)
	lm3D = load_lm3d()

	with torch.no_grad():
		model = Nonlinear3DMM_redner().to(CFG.device).eval()
		model, _, _, start_epoch, start_step = load(model)

		if is_testset:	# test with testset
			dataset = NonlinearDataset(phase='test', frac=0.1)
			dataloader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=0)

		else:			# test with test image directory
			img_list = glob(join(CFG.prediction_src_path, "*.jpg"))
			img_list += glob(join(CFG.prediction_src_path, "*.jpeg"))
			img_list += glob(join(CFG.prediction_src_path, "*.png"))
			images = []
			name_list = []
			for file in img_list:
				input_img, _, _ = Preprocess(fa, file, lm3D)
				if input_img is None:
					print(f"No landmark in file : {file}")
					continue
				image_tensor = torch.tensor(input_img).permute(0, 3, 1, 2)
				images.append(F.interpolate(image_tensor, size=256))
				name_list.append(file)
			images = torch.cat(images, dim=0) / 255.0
			# images = images.permute(0, 3, 1, 2)
			# r, g, b = torch.split(images, (1, 1, 1), dim=1)
			# images = torch.cat([b, g, r], dim=1)

			dataloader = []
			for idx in range(0, images.shape[0], CFG.batch_size):
				start_idx = idx
				end_idx = min((idx + 1) * CFG.batch_size, images.shape[0])
				dataloader.append({'image': images[start_idx:end_idx], 'image_name': name_list[start_idx:end_idx]})

		for samples in dataloader:
			result = model(samples['image'].to(CFG.device))

			shape = result['shape_1d_base']
			albedo = result['albedo_1d_base']
			exp = result['exp_1d']
			trans = result['lv_trans']
			angle = result['lv_angle']
			light = result['lv_il']

			# shape = samples['shape'].to(CFG.device)
			# albedo = samples['vcolor'].to(CFG.device)
			# exp = samples['exp'].to(CFG.device)
			# trans = samples['trans'].to(CFG.device)
			# angle = samples['angle'].to(CFG.device)
			# light = samples['light'].to(CFG.device)

			shape = shape + CFG.mean_shape
			color = albedo + CFG.mean_tex

			images, masks, _ = renderer.render(
				vertex_batch=shape,
				color_batch=color,
				trans_batch=trans,
				angle_batch=angle,
				light_batch=light,
				print_timing=False)

			# for image, mask, image_label, image_name in zip(images, masks, samples['image'], samples['image_name']):
			# 	masked = image * mask + image_label.to(CFG.device).permute(1, 2 ,0) * (1 - mask)
			# 	save_image(masked.permute(2, 0, 1), join(CFG.prediction_dst_path, basename(image_name)))
			# 	save_to_obj(join(CFG.prediction_dst_path, basename(image_name).replace('.jpg', '.obj').replace('.jpeg', '.obj').replace('.png', '.obj')), vertex=(CFG.mean_shape + shape).view((-1, 3)), face=CFG.face)
			# 	pass

			# for degugginb
			images = torch.cat([image for image in images], dim=1)

			image_labels = samples['image'].permute(0, 2, 3, 1).to(CFG.device)
			image_labels = torch.cat([image for image in image_labels], dim=1)

			masks = torch.cat([mask for mask in masks], dim=1)
			maskeds = images * masks + image_labels * (1 - masks * 1)

			images = images.cpu().detach().numpy()[:,:,::-1]
			image_labels = image_labels.cpu().detach().numpy()[:,:,::-1]
			maskeds = maskeds.cpu().detach().numpy()[:,:,::-1]

			pass
	return




if __name__ == '__main__':
	main()


