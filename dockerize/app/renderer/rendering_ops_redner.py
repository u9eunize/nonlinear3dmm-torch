import pyredner
import torch
import math
import redner
from typing import Optional, Tuple, Any
from utils import *
from settings import CFG
import numpy as np
from os.path import join



def sphere2xyz ( theta, pi, rho ):
    return torch.tensor(
        [rho * math.sin(theta) * math.cos(pi), rho * math.sin(theta) * math.sin(pi), rho * math.cos(theta)],
        device=pyredner.get_device())




class BatchRenderFunction(torch.autograd.Function):
    @staticmethod
    def forward ( ctx, seed, *args ):
        batch_dims = args[0]
        args_old_format = args[1:]
        chunk_len = int(len(args_old_format) / batch_dims)
        h, w = args_old_format[12]
        result = torch.zeros(batch_dims, h, w, 4, device=pyredner.get_device(), requires_grad=True)
        for k in range(0, batch_dims):
            sub_args = args_old_format[k * chunk_len:(k + 1) * chunk_len]
            result[k, :, :, :] = pyredner.RenderFunction.forward(ctx, seed, *sub_args)
        return result
    
    @staticmethod
    def backward ( ctx, grad_img ):
        # None gradient for seed and batch_dims
        ret_list = (None, None,)
        batch_dims = grad_img.shape[0]
        for k in range(0, batch_dims):
            # [1:] cuz original backward function returns None grad for seed input, but we manage that ourselves
            ret_list = ret_list + pyredner.RenderFunction.backward(ctx, grad_img[k, :, :, :])[1:]
        return ret_list




class Batch_Renderer():
    def __init__( self ):
        # declare constant values
        self.rho_divider = 18
        self.device = pyredner.get_device()
        self.mat = pyredner.Material(diffuse_reflectance=torch.zeros([1, 1, 3], device=self.device), use_vertex_color=True)
        self.args = {
                'num_samples'                : (64, 16),
                'max_bounces'                : 1,
                'channels'                   : [redner.channels.radiance, redner.channels.depth],
                'sampler_type'               : pyredner.sampler_type.sobol,
                'sample_pixel_center'        : False,
                'use_primary_edge_sampling'  : True,
                'use_secondary_edge_sampling': True,
                'device'                     : self.device
        }
        self.batch_render = BatchRenderFunction.apply
        self.envmap_texels = 0.5 * torch.ones([1, 1, 3], device=self.device, requires_grad=True)
        self.envmap = pyredner.EnvironmentMap(torch.abs(self.envmap_texels))
        self.indices = torch.tensor(np.load(join(CFG.dataset_path, 'face.npy')), dtype=torch.int32).to(CFG.device)
    

    def render ( self,
                 vertex_batch: Optional[torch.Tensor],
                 color_batch: Optional[torch.Tensor],
                 camera_batch: Optional[torch.Tensor],
                 light_batch: Optional[torch.Tensor],
                 resolution: Tuple[int, int] = (224, 224),
                 print_timing: Any = False,
                 uv_indices: Optional[torch.Tensor] = None,
                 texture: Optional[torch.Tensor] = None):
        """
            render batch 3D objects using path tracing mode(physically based mode)
    
            Args
            ====
            vertex: Optional[torch.Tensor]
                3D position of vertices
                float32 tensor with size [batch_size, num_vertices, 3]
            color: Optional[torch.Tensor]
                per-vertex color
                float32 tensor with size [batch_size, num_vertices, 3]
            camera_param: Optional[torch.Tensor]
                camera parameters defined by sphere coordinates.
                float32 tensor with size 3, [batch_size, theta, pi, rho]
            light_param: Optional[torch.Tensor]
                light parameters defined by sphere coordinates with intensity.
                float32 tensor with size 3, [batch_size, theta, pi, rho, intensity]
            resolution: Tuple[int, int]
                resolution parameter
                int32 tuple with size 2, [h, w]
    
            Returns
            =======
            torch.Tensor
                a camera that can see all the objects.
                float32 tensor with size [batch_size, h, w, 3]
        """
        pyredner.set_print_timing(print_timing)
        
        batch_size = vertex_batch.shape[0]
        scene_args = []
        
        for vertex, color, camera_param, light_param in zip(vertex_batch, color_batch, camera_batch, light_batch):
            # define face shape
            face = pyredner.Object(vertices=vertex, indices=self.indices, colors=color, material=self.mat)
            face.normals = pyredner.compute_vertex_normal(vertices=face.vertices, indices=face.indices)
            
            # camera parameters
            theta_camera, pi_camera, rho_camera = camera_param
            rho_camera /= self.rho_divider
            camera_position = sphere2xyz(theta_camera, pi_camera, rho_camera)
            camera_look_at = torch.tensor([0., 0., 0.], device=self.device)
            camera_up = torch.tensor([0., 1., 0.], device=self.device)
            camera_fov = torch.tensor([45.], device=self.device)
            camera = pyredner.Camera(position=camera_position,
                                     look_at=camera_look_at,
                                     up=camera_up,
                                     fov=camera_fov,
                                     resolution=resolution)
            
            # define light object
            theta_light, pi_light, rho_light = light_param
            light_position = sphere2xyz(theta_light, pi_light, rho_light)
            light = pyredner.generate_quad_light(position=light_position,
                                                 look_at=torch.zeros(3, device=self.device),
                                                 size=torch.tensor([0.1, 0.1], device=self.device),
                                                 intensity=torch.tensor([1.0, 1.0, 1.0], device=self.device) * 50000.0)
            
            # define scene parameter
            scene = pyredner.Scene(objects=[face, light], camera=camera, envmap=self.envmap)
            scene_args += pyredner.RenderFunction.serialize_scene(scene=scene, **self.args)
        
        scene_args = [batch_size] + scene_args
        g_buffer = self.batch_render(0, *scene_args)
        
        images, masks = torch.split(g_buffer, (3, 1), dim=-1)
        masks = masks > 0
        
        return images, masks


