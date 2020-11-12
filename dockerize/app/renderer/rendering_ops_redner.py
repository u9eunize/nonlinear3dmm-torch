import pyredner
import torch
import math
import redner
from typing import Optional, Tuple, Any
from utils import *
from settings import CFG
import numpy as np
from os.path import join
from renderer.rendering_ops import bilinear_sampler_torch







h, w = CFG.texture_size
vt2pixel_u, vt2pixel_v = torch.split(torch.tensor(np.load('deep3d/BFM_uvmap.npy')), (1, 1), dim=-1)
vt2pixel_v = torch.ones_like(vt2pixel_v) - vt2pixel_v
vt2pixel_u_, vt2pixel_v_ = vt2pixel_u * h, vt2pixel_v * w



def sphere2xyz ( sphere_coord ):
    theta, pi, rho = torch.split(sphere_coord, (1,1,1))
    return torch.tensor(
        [rho * math.sin(theta) * math.cos(pi), rho * math.sin(theta) * math.sin(pi), rho * math.cos(theta)],
        device=pyredner.get_device())

def translate_vertex(vertex, coeff_translate):
    translated = vertex + coeff_translate
    return translated

def rotate_vertex(vertex, coeff_rotate):
    coeff_x, coeff_y, coeff_z = coeff_rotate
    ro_x = torch.tensor(
        [[1, 0, 0], [0, math.cos(coeff_x), -math.sin(coeff_x)], [0, math.sin(coeff_x), math.cos(coeff_x)]], device=pyredner.get_device())
    ro_y = torch.tensor(
        [[math.cos(coeff_y), 0, math.sin(coeff_y)], [0, 1, 0], [-math.sin(coeff_y), 0, math.cos(coeff_y)]], device=pyredner.get_device())
    ro_z = torch.tensor(
        [[math.cos(coeff_z), -math.sin(coeff_z), 0], [math.sin(coeff_z), math.cos(coeff_z), 0], [0, 0, 1]], device=pyredner.get_device())
    rotated = torch.mm(torch.mm(torch.mm (ro_x , ro_y) , ro_z), vertex.permute(1,0))
    return rotated.permute(1,0)

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
    def __init__( self, resolution=(224,224) ):
        # declare constant values
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

        # camera parameters
        camera_position = torch.tensor([0., 0., 5.], dtype=torch.float32, device=self.device)
        camera_look_at = torch.tensor([0., 0., 0.], device=self.device)
        camera_up = torch.tensor([0., 1., 0.], device=self.device)
        camera_fov = torch.tensor([45.], device=self.device)
        self.camera = pyredner.Camera(position=camera_position,
                                 look_at=camera_look_at,
                                 up=camera_up,
                                 fov=camera_fov,
                                 resolution=resolution)
    

    def render ( self,
                 vertex_batch: Optional[torch.Tensor],
                 color_batch: Optional[torch.Tensor],
                 trans_batch: Optional[torch.Tensor],
                 rotation_batch: Optional[torch.Tensor],
                 light_batch: Optional[torch.Tensor],
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
        
        for vertex, color, trans, rot, light_param in zip(vertex_batch, color_batch, trans_batch, rotation_batch, light_batch):
            # rotation
            vertex = rotate_vertex(vertex, rot)
            
            # translation
            vertex = translate_vertex(vertex, trans)
            
            # define face shape
            face = pyredner.Object(vertices=vertex.contiguous(), indices=self.indices, colors=color, material=self.mat)
            face.normals = pyredner.compute_vertex_normal(vertices=face.vertices, indices=face.indices)
            
            # define light object
            light_position = sphere2xyz(light_param)
            light = pyredner.generate_quad_light(position=light_position,
                                                 look_at=torch.zeros(3, device=self.device),
                                                 size=torch.tensor([0.1, 0.1], device=self.device),
                                                 intensity=torch.tensor([1.0, 1.0, 1.0], device=self.device) * 50000.0)
            
            # define scene parameter
            scene = pyredner.Scene(objects=[face, light], camera=self.camera, envmap=self.envmap)
            scene_args += pyredner.RenderFunction.serialize_scene(scene=scene, **self.args)
        
        scene_args = [batch_size] + scene_args
        g_buffer = self.batch_render(0, *scene_args)
        
        images, masks = torch.split(g_buffer, (3, 1), dim=-1)
        masks = masks > 0
        
        return images, masks
    
    
    
renderer = Batch_Renderer()

def generate_full(vec, kind="shape"):
    assert kind in ["shape", "m", "exp"]
    std = getattr(CFG, f"std_{kind}")
    mean = getattr(CFG, f"mean_{kind}")

    return vec * std + mean


def generate_shade(lv_il, vertex):
    # render white image
    batch_size = lv_il.shape[0]
    images = renderer.render(
        vertex_batch=vertex,
        color_batch=torch.ones_like(vertex, device=CFG.device),
        trans_batch=torch.zeros([batch_size, 3], device=CFG.device),
        rotation_batch=torch.zeros([batch_size, 3], device=CFG.device),
        light_batch=lv_il
    )
    
    # by Jeong Woo Lee algorithm
    h, w = CFG.texture_size
    return torch.ones([batch_size, 3, h, w])


def generate_texture(albedo, shade, is_clamp=False):
    tex = 2.0 * ((albedo + 1.0) / 2.0 * shade) - 1.0
    
    if is_clamp:
        tex = torch.clamp(tex, 0, 1)
    return tex





def make_1d ( decoder_2d_result, vt2pixel_u, vt2pixel_v ):
    batch_size = decoder_2d_result.shape[0]
    decoder_1d_result = bilinear_sampler_torch(decoder_2d_result, vt2pixel_u, vt2pixel_v)
    decoder_1d_result = decoder_1d_result.view(batch_size, -1)
    
    return decoder_1d_result


def render_all(lv_trans, lv_rot, lv_il, albedo, shape_1d,
               input_mask=None, input_background=None, post_fix="",
               using_expression=CFG.using_expression,
               using_albedo_as_tex=CFG.using_albedo_as_tex):
    batch_size = lv_il.shape[0]
    shape_full = generate_full(shape_1d, "shape")

    shape_final = shape_full
    vertex = shape_final.view([batch_size, -1, 3])

    shade = generate_shade(lv_il, vertex)
    tex = generate_texture(albedo, shade)
    vt2pixel_u = vt2pixel_u_.view((1, 1, -1)).repeat(batch_size, 1, 1)
    vt2pixel_v = vt2pixel_v_.view((1, 1, -1)).repeat(batch_size, 1, 1)
    
    vcolor = make_1d(tex, vt2pixel_u, vt2pixel_v)

    images, masks = renderer.render(
        vertex_batch=vertex,
        color_batch=vcolor,
        trans_batch=lv_trans,
        rotation_batch=lv_rot,
        light_batch=lv_il
    )

    return {
        f"shade{post_fix}": shade,
        f"tex{post_fix}": tex,
        f"g_img_mask{post_fix}": masks,
        f"g_img{post_fix}": images * masks
    }

def render_mix(albedo_base, shade_base, shape_1d_base, albedo_comb, shade_comb, shape_1d_comb,
               lv_trans, lv_rot, lv_il,
               mask_base=None, mask_comb=None,
               input_mask=None, input_background=None,
               using_albedo_as_tex=CFG.using_albedo_as_tex,
               ):
    tex_mix_ac_sb = generate_texture(albedo_comb, shade_base)
    tex_mix_ab_sc = generate_texture(albedo_base, shade_comb)

    batch_size = lv_il.shape[0]
    shape_full_base = generate_full(shape_1d_base, "shape")
    vertex_base = shape_full_base.view([batch_size, -1, 3])
    shape_full_comb = generate_full(shape_1d_comb, "shape")
    vertex_comb = shape_full_comb.view([batch_size, -1, 3])
    

    vcolor_base = make_1d(tex_mix_ac_sb, vt2pixel_u, vt2pixel_v)
    vcolor_comb = make_1d(tex_mix_ab_sc, vt2pixel_u, vt2pixel_v)
    
    gen_img_ac_sb, gen_img_ac_sb_mask = renderer.render(
        vertex_batch=vertex_base,
        color_batch=vcolor_base,
        trans_batch=lv_trans,
        rotation_batch=lv_rot,
        light_batch=lv_il
    )

    gen_img_ab_sc, gen_img_ab_sc_mask = renderer.render(
        vertex_batch=vertex_comb,
        color_batch=vcolor_comb,
        trans_batch=lv_trans,
        rotation_batch=lv_rot,
        light_batch=lv_il
    )

    return {
        "tex_mix_ac_sb": tex_mix_ac_sb,
        "tex_mix_ab_sc": tex_mix_ab_sc,
        "g_img_ac_sb": gen_img_ac_sb * gen_img_ac_sb_mask,
        "g_img_ab_sc": gen_img_ab_sc * gen_img_ab_sc_mask,
    }


def generate_tex_mask(input_texture_labels, input_texture_masks):
    batch_size = input_texture_labels.shape[0]
    tex_vis_mask = (~input_texture_labels.eq((torch.ones_like(input_texture_labels) * -1))).float()
    tex_vis_mask = tex_vis_mask * input_texture_masks
    tex_ratio = torch.sum(tex_vis_mask) / (batch_size * CFG.texture_size[0] * CFG.texture_size[1] * CFG.c_dim)
    return {
        "tex_vis_mask": tex_vis_mask,
        "tex_ratio": tex_ratio,
    }