import pyredner
import torch
from plyfile import PlyData
import math
import numpy as np
from renderer.rendering_ops_redner import render



def main ():
    # read ply data and split vertex and color data
    # plydata = PlyData.read('1.ply')
    # fname = '1_0.2776017385292788_0.16254296733069473_70.79018321602928.jpg'
    plydata = PlyData.read('2.ply')
    fname = '2_0.15897386037903233_0.2043752904048205_58.26357616953039.jpg'
    # plydata = PlyData.read('3.ply')
    # fname = '3_5.584806478789926_0.013549064110521272_66.73159275208857.jpg'
    x, y, z = plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']
    r, g, b = plydata['vertex']['red'], plydata['vertex']['green'], plydata['vertex']['blue']
    vertex = np.stack([x, y, z], axis=1)
    color = np.stack([r, g, b], axis=1)
    face = np.vstack(plydata['face']['vertex_indices'])
    
    # define 3d model
    vertex = torch.tensor(vertex, device=pyredner.get_device())
    indices = torch.tensor(face, device=pyredner.get_device())
    color = torch.tensor(color, device=pyredner.get_device()) / 255.0

    # set camera parameters
    id, pi_camera, theta_camera, rho_camera = [float(t) for t in fname.split('.jpg')[0].split('_')]
    camera_param = torch.tensor([theta_camera, pi_camera, rho_camera], device=pyredner.get_device())
    
    # set light parameters
    theta_light = math.pi * -0.5
    pi_light = 0
    rho_light = 10
    intensity = 5000.0
    light_param = torch.tensor([theta_light, pi_light, rho_light, intensity], device=pyredner.get_device())

    # render
    def batch_wise(tensor, batch_size):
        dim = len(tensor.shape)
        args = [batch_size] + [1] * dim
        return torch.unsqueeze(tensor, dim=0).repeat(*args)
    
    batch_size = 2
    
    pyredner.set_print_timing(False)
    g_buffer = render(vertex_batch=batch_wise(vertex, batch_size),
                      indices_batch=batch_wise(indices, batch_size),
                      color_batch=batch_wise(color, batch_size),
                      camera_batch=batch_wise(camera_param, batch_size),
                      light_batch=batch_wise(light_param, batch_size),
                      resolution=(224,224),
                      print_timing=True)
    
    # write images
    for idx, img in enumerate(g_buffer):
        pyredner.imwrite(img.cpu(), 'taehwan_redner_{}.png'.format(idx))
    
    


if __name__ == '__main__':
    main()
