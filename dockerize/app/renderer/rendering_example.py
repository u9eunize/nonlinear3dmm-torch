import pyredner
import torch
from plyfile import PlyData
import math
import numpy as np
from renderer.rendering_ops_redner import Batch_Renderer
from utils import *
from os.path import basename



def main ():
    plydata = PlyData.read('renderer/0.ply')
    x, y, z = plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']
    r, g, b = plydata['vertex']['red'], plydata['vertex']['green'], plydata['vertex']['blue']
    vertex = np.stack([x, y, z], axis=1)
    color = np.stack([r, g, b], axis=1)
    face = np.vstack(plydata['face']['vertex_indices'])
    
    # define 3d model
    vertex = torch.tensor(vertex, device=pyredner.get_device())
    color = torch.tensor(color, device=pyredner.get_device()) / 255.0
    color = torch.ones_like(color, device=pyredner.get_device())

    # set camera parameters
    # id, pi_camera, theta_camera, rho_camera = [float(t) for t in basename(fname).split('.jpg')[0].split('_')]
    # camera_param = torch.tensor([theta_camera, pi_camera, rho_camera], device=pyredner.get_device())
    
    # set light parameters
    theta_light = math.pi * -0.5
    pi_light = 0
    rho_light = 20
    light_param = torch.tensor([theta_light, pi_light, rho_light], device=pyredner.get_device())
    
    # position parameters
    trans = torch.tensor([0.5, 0.5, 0.5], device=pyredner.get_device())
    rot = torch.tensor([0.0, 0.0, math.pi/2], device=pyredner.get_device())

    # render
    def batch_wise(tensor, batch_size):
        dim = len(tensor.shape)
        args = [batch_size] + [1] * dim
        return torch.unsqueeze(tensor, dim=0).repeat(*args)
    
    batch_size = 2
    
    pyredner.set_print_timing(False)
    renderer = Batch_Renderer()
    images, masks = renderer.render(
        vertex_batch=batch_wise(vertex, batch_size),
        color_batch=batch_wise(color, batch_size),
        trans_batch=batch_wise(trans, batch_size),
        rotation_batch=batch_wise(rot, batch_size),
        light_batch=batch_wise(light_param, batch_size),
        print_timing=True
    )
    
    masked = images * masks
    # write images
    for idx, img in enumerate(images):
        pyredner.imwrite(img.cpu(), 'renderer/rendering_result_{}.png'.format(idx))
    
    


if __name__ == '__main__':
    main()
