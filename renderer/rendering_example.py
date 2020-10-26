from network.Nonlinear_3DMM import Nonlinear3DMM
from loss import Loss
from os.path import join, basename
from glob import glob
import torch
import torchvision.transforms.functional as F
import torchvision
from PIL import Image
import pyredner
import redner
from renderer.rendering_ops import *
from settings import CFG
from configure_dataset import NonlinearDataset


dtype = torch.float32
mu_shape, w_shape = load_Basel_basic('shape')
mu_exp, w_exp = load_Basel_basic('exp')
mean_shape = torch.tensor(mu_shape + mu_exp, dtype=dtype).to(CFG.device)
std_shape = torch.tensor(np.tile(np.array([1e4, 1e4, 1e4]), CFG.vertex_num), dtype=dtype).to(CFG.device)
mean_m = torch.tensor(np.load(join(CFG.dataset_path, 'mean_m.npy')), dtype=dtype).to(CFG.device)
std_m = torch.tensor(np.load(join(CFG.dataset_path, 'std_m.npy')), dtype=dtype).to(CFG.device)
def save_to_obj(name, vertex, face):
   with open(name, 'w') as fd:
      result = ""
      for v1, v2, v3 in vertex:
         result += f'v {v1:.3f} {v2:.3f} {v3:.3f}\n'
      result += "\n"
      for f1, f2, f3 in face:
         result += f'f {f3 + 1} {f2 + 1} {f1 + 1}\n'
      fd.write(result)
   print(name)
def compute_uvs(texture, m, shape):
   u, v = warp_texture_torch(texture, m, shape, with_uv=True)
   return torch.tensor(torch.cat([u[:-1], v[:-1]]).view(-1,2), dtype=torch.float32)
def main():
   tri = load_3DMM_tri()
   face = np.transpose(tri)[:-1]
   # checkerboard_texture = pyredner.imread('checkerboard.exr')
   # define model and loss
   model = Nonlinear3DMM().to(CFG.device)
   model, _, _, _, _ = load(model)
   # load images for prediction
   fnames_raw = glob(join(CFG.prediction_src_path, "*"))
   total_len = len(fnames_raw)
   fnames = [fnames_raw[i:i + CFG.batch_size] for i in range(0, len(fnames_raw), CFG.batch_size)]
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
         img = torchvision.transforms.functional.resize(img, CFG.image_size)
         img = torchvision.transforms.functional.center_crop(img, CFG.image_size)
         img = torchvision.transforms.functional.to_tensor(img)
         input_images.append(img)
      input_images = torch.stack(input_images, dim=0).to(CFG.device)
      # forward network
      with torch.no_grad():
         result = model(input_images)
         lv_m = result['lv_m']
         lv_il = result['lv_il']
         shape1d = result['shape_1d_comb']
         albedo = result['albedo_comb']

      vt2pixel_u, vt2pixel_v = load_FaceAlignment_vt2pixel(False)

      vt2pixel_u = torch.from_numpy(vt2pixel_u).cuda()[:-1]
      vt2pixel_v = torch.from_numpy(vt2pixel_v).cuda()[:-1]

      vt2pixel_u = torch.clamp(vt2pixel_u, 0, 192.0) / 192
      vt2pixel_v = torch.clamp(vt2pixel_v, 0, 224.0) / 224

      # vt2pixel_u -= torch.min(vt2pixel_u)
      # vt2pixel_u /= torch.max(vt2pixel_u)
      #
      # vt2pixel_v -= torch.min(vt2pixel_v)
      # vt2pixel_v /= torch.max(vt2pixel_v)

      uvs = torch.stack([vt2pixel_v, vt2pixel_u], dim=-1).float()


      # make full
      m_full = lv_m * std_m + mean_m
      shape_full = shape1d * std_shape + mean_shape
      shade = generate_shade(lv_il, m_full, shape_full)
      tex = 2.0 * ((albedo + 1.0) / 2.0 * shade) - 1.0
      g_images_raw, g_images_mask_raw = warp_texture_torch(tex, m_full, shape_full)
      g_images_mask_raw = g_images_mask_raw.unsqueeze(1).repeat(1, 3, 1, 1)
      g_images = g_images_raw * g_images_mask_raw + input_images * (torch.ones_like(g_images_mask_raw) - g_images_mask_raw)
      output_gt += input_images
      output_images += g_images_raw
      output_images_with_mask += g_images
      output_shapes += shape_full.view(shape_full.shape[0], -1, 3)
      for m , sh, a, t, n , s in zip(m_full, output_shapes, albedo, tex,fnames_raw, shade):
         # uv = torch.tensor(torch.cat([tu, tv]).view(-1,2), dtype=torch.float32)
         # # texture test
         # atext = pyredner.Texture(texels=a.permute(1,2,0), uv_scale = uvs)
         # ttext = pyredner.Texture(texels=t.permute(1, 2, 0), uv_scale=uvs)
         mat = pyredner.Material(diffuse_reflectance=t.permute(1,2,0), two_sided = True)
         mat_black = pyredner.Material( diffuse_reflectance=torch.tensor([1.0, 1.0, 1.0], device=pyredner.get_device()))
         materials = [mat, mat_black]
         # mat = pyredner.Material()
         pshape = pyredner.Shape(vertices=sh, indices=torch.tensor(face, dtype=torch.int32), material_id=0, uvs = uvs)
         # pshape.normals = -pyredner.compute_vertex_normal(pshape.vertices.cpu(), pshape.indices.cpu()) # 필
         pyredner.imwrite(t.permute(1,2,0).cpu(), join(CFG.prediction_dst_path, basename(n).split('.')[0] + '_img.jpg'))
         z_coord = 500000
         light_position = 150000.0
         light_vertices = torch.tensor([[-light_position, -light_position, z_coord], [light_position, -light_position, z_coord], [-light_position, light_position, z_coord], [light_position, light_position, z_coord]],
                                device=pyredner.get_device())
         light_indices = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int32, device=pyredner.get_device())
         shape_light = pyredner.Shape(light_vertices, light_indices, 1)
         camera0 = pyredner.automatic_camera_placement([pshape], resolution=(224, 224))
         t = camera0.position
         camera0.position = camera0.position - t - t
         # camera0.look_at = camera0.look_at - t - t
         # camera0.look_at = camera0.position - camera0.look_at
         camera = pyredner.Camera(position=camera0.position + torch.tensor([0, 0, 10]),
                            look_at=camera0.look_at + torch.tensor([0, 0, 0]),
                            up=camera0.up + torch.tensor([0, 0, 0]),
                            fov=camera0.fov,
                            resolution=camera0.resolution)
         light = pyredner.AreaLight(1, torch.tensor([5.0, 5.0, 5.0]), two_sided=True)
         scene = pyredner.Scene(camera, [pshape, shape_light], materials, [light])
         args = pyredner.RenderFunction.serialize_scene(scene=scene,num_samples=128,max_bounces=16)
         # Alias of the render function
         render = pyredner.RenderFunction.apply
         # render = pyredner.render_pathtracing(*args)
         img = render(0, *args)
         pyredner.imwrite(img.cpu(), join(CFG.prediction_dst_path, basename(n).split('.')[0] + '_redner.png'))
         # deffered test
         # uvs = compute_uvs()
         # ptext = pyredner.Texture(texels=a.permute(1, 2, 0))
         # mat = pyredner.Material(diffuse_reflectance=ptext)
         # objects = pyredner.Object(vertices = sh, indices=torch.tensor(face, dtype=torch.int32), material=mat, uvs= uvs)
         # objects.normals = pyredner.compute_vertex_normal(objects.vertices.cpu(), objects.indices.cpu())
         #
         # camera0 = pyredner.automatic_camera_placement([objects], resolution=(224, 224))
         # camera = pyredner.Camera(position=camera0.look_at + torch.tensor([0,  0, 35]),
         #                  look_at=camera0.look_at + torch.tensor([0,  0, 0]),
         #                  up=camera0.up+ torch.tensor([0,  0, 0]),
         #                  fov=camera0.fov,
         #                  resolution=camera0.resolution)
         # light = pyredner.PointLight(
         #     position=(camera0.position + torch.tensor((0.0, 0.0, 20.0))).to(pyredner.get_device()),
         #     intensity=torch.tensor((500.0, 500.0, 500.0), device=pyredner.get_device()))
         # envmap=pyredner.EnvironmentMap(a.permute(1, 2, 0))
         # scene = pyredner.pyredner.Scene(objects = [objects], camera = camera, envmap=envmap)
         # img = pyredner.render_deferred(scene, lights=[light])
         # save_images(torch.pow(img, 1.0/2.2).cpu(), [4, -1], basename(n).split('.')[0] + 'redner.jpg')
if __name__ == '__main__':
   main()