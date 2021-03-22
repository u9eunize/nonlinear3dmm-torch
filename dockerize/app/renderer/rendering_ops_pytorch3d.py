import pyredner
import torch
import math
import redner
from typing import Optional, Tuple, Any
from utils import *
from settings import CFG
import numpy as np
from os.path import join
import torch.nn.functional as F
from time import time
from joblib import Parallel, delayed
import multiprocessing

from pytorch3d.renderer import (
    look_at_view_transform, OrthographicCameras, PerspectiveCameras,
    RasterizationSettings, BlendParams,
    MeshRenderer, MeshRasterizer, HardPhongShader,
    Materials, DirectionalLights, TexturesVertex
)
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments
from pytorch3d.structures import Meshes


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

        # def _render_sample(idx):
        #     sub_args = args_old_format[idx * chunk_len:(idx + 1) * chunk_len]
        #     return pyredner.RenderFunction.forward(ctx, seed, *sub_args)
        # results = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(_render_sample)(idx) for idx in range(batch_dims))
        # result = torch.stack(results)

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


def remove_light(vertex_batch, color_batch, light_batch, angle_batch, device="cpu"):
    rotation = Compute_rotation_matrix(angle_batch, device=device)

    # compute face color
    face_norm = Compute_norm(vertex_batch, device=device)
    norm_r = torch.bmm(face_norm, rotation)
    colors = Illumination_block_inverse(color_batch, norm_r, light_batch, device=device)
    return colors

class Batch_Renderer():
    def __init__ ( self, resolution=(224, 224) ):
        # declare constant values
        self.device = pyredner.get_device()
        self.mat = pyredner.Material(diffuse_reflectance=torch.zeros([1, 1, 3], device=self.device),
                                     use_vertex_color=True)
        self.args = {
                'num_samples'                : (64, 0),
                'max_bounces'                : 1,
                'channels'                   : [redner.channels.radiance, redner.channels.depth],
                'sampler_type'               : pyredner.sampler_type.sobol,
                'sample_pixel_center'        : False,
                'use_primary_edge_sampling'  : False,
                'use_secondary_edge_sampling': False,
                'device'                     : self.device
        }
        self.batch_render = BatchRenderFunction.apply
        envmap_texels = 1.0 * torch.ones([1, 1, 3], requires_grad=True, device=pyredner.get_device())
        self.envmap = pyredner.EnvironmentMap(torch.abs(envmap_texels))
        
        # camera parameters
        camera_position = torch.tensor([0., 0., 10.], dtype=torch.float32, device=pyredner.get_device())
        camera_look_at = torch.tensor([0., 0., 0.], device=pyredner.get_device())
        camera_up = torch.tensor([0., 1., 0.], device=pyredner.get_device())
        camera_fov = torch.tensor([12.5936], device=pyredner.get_device())
        self.camera = pyredner.Camera(position=camera_position,
                                      look_at=camera_look_at,
                                      up=camera_up,
                                      fov=camera_fov,
                                      resolution=resolution)
    
    def render ( self,
                 vertex_batch: Optional[torch.Tensor],
                 color_batch: Optional[torch.Tensor],
                 trans_batch: Optional[torch.Tensor],
                 angle_batch: Optional[torch.Tensor],
                 light_batch: Optional[torch.Tensor],
                 print_timing: Any = False,
                 uv_indices: Optional[torch.Tensor] = None,
                 texture: Optional[torch.Tensor] = None ):
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
            trans: Optional[torch.Tensor]
                3D translation coordinates of vertices
                float32 tensor with size [batch_size, 3]
            angle: Optional[torch.Tensor]
                3D rotation angles of vertices
                float32 tensor with size [batch_size, 3]
            light: Optional[torch.Tensor]
                light parameters defined in phong shading.
                float32 tensor with size 27, [batch_size, 27]
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

        # compute vertex transformation
        rotation = Compute_rotation_matrix(angle_batch)
        vertices = torch.bmm(vertex_batch, rotation) + torch.unsqueeze(trans_batch, 1)
        
        # compute face color
        face_norm = Compute_norm(vertex_batch)
        norm_r = torch.bmm(face_norm, rotation)
        colors = Illumination_block(color_batch, norm_r, light_batch)
        # colors = Illumination_block_inverse(color_batch, norm_r, light_batch)

        scene_args = []
        for vertex, color in zip(vertices, colors):
            # define shape parameter
            shape_face = pyredner.Shape(vertices=vertex, indices=CFG.face, colors=color, material_id=0)
            shape_face.normals = pyredner.compute_vertex_normal(vertices=shape_face.vertices, indices=shape_face.indices)
            
            # define scene parameter
            scene = pyredner.Scene(camera=self.camera, shapes=[shape_face], materials=[self.mat], envmap=self.envmap)
            scene_args += pyredner.RenderFunction.serialize_scene(scene=scene, **self.args)
        
        scene_args = [batch_size] + scene_args

        g_buffer = self.batch_render(0, *scene_args)
        
        images, masks = torch.split(g_buffer, (3, 1), dim=-1)
        masks = masks > 0
        
        return images, masks, colors


class Batch_Renderer_pytorch3d():
    def __init__(self, resolution=224):
        materials = Materials(device=CFG.device)
        lights = DirectionalLights(ambient_color=((1.0, 1.0, 1.0),), device=CFG.device)

        R, T = look_at_view_transform(eye=torch.tensor([[0.0, 0.0, 10.0]]), device=CFG.device)
        cameras = PerspectiveCameras(focal_length=9.25, device=CFG.device, R=R, T=T)

        raster_settings = RasterizationSettings(image_size=resolution)
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

        # blend_params = BlendParams(1e-4, 1e-4, background_color=torch.zeros(3, dtype=torch.float32, device=CFG.device))
        shader = HardPhongShader(device=CFG.device, cameras=cameras, lights=lights, materials=materials)#blend_params=blend_params)

        self.renderer = MeshRendererWithFragments(rasterizer=rasterizer, shader=shader)

    def render(self,
               vertex_batch: Optional[torch.Tensor],
               color_batch: Optional[torch.Tensor],
               trans_batch: Optional[torch.Tensor],
               angle_batch: Optional[torch.Tensor],
               light_batch: Optional[torch.Tensor],
               landmark: Any = False,
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
            trans: Optional[torch.Tensor]
                3D translation coordinates of vertices
                float32 tensor with size [batch_size, 3]
            angle: Optional[torch.Tensor]
                3D rotation angles of vertices
                float32 tensor with size [batch_size, 3]
            light: Optional[torch.Tensor]
                light parameters defined in phong shading.
                float32 tensor with size 27, [batch_size, 27]
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

        # compute vertex transformation
        rotation = Compute_rotation_matrix(angle_batch.float())
        vertices = torch.bmm(vertex_batch.float(), rotation) + torch.unsqueeze(trans_batch.float(), 1)

        # compute face color
        face_norm = Compute_norm(vertex_batch.float())
        norm_r = torch.bmm(face_norm, rotation)
        colors = Illumination_block(color_batch.float(), norm_r, light_batch.float())

        meshes = Meshes(verts=[vertex for vertex in vertices], faces=[CFG.face for _ in range(batch_size)])
        meshes.textures = TexturesVertex(verts_features=colors)

        outputs, fragments = self.renderer(meshes)
        images, _ = torch.split(outputs, (3, 1), dim=3)
        masks = (fragments.zbuf > 0) * 1

        if landmark:
            landmark_u, landmark_v = project_vertices(vertex_batch[:, CFG.landmark], trans_batch, angle_batch)
            landmark_u = landmark_u.long()
            landmark_v = landmark_v.long()
            for image, u, v in zip(images, landmark_u, landmark_v):
                image[u, v, :] = 1

        return images, masks, colors


# renderer = Batch_Renderer()
renderer = Batch_Renderer_pytorch3d(resolution=256)



def Compute_rotation_matrix ( angles, device=CFG.device ):
    n_data = angles.shape[0]
    
    # compute rotation matrix for X-axis, Y-axis, Z-axis respectively
    rotation_X = torch.cat([torch.ones([n_data, 1]).to(device),
                            torch.zeros([n_data, 3]).to(device),
                            torch.cos(angles[:, 0]).view([n_data, 1]),
                            -torch.sin(angles[:, 0]).view([n_data, 1]),
                            torch.zeros([n_data, 1]).to(device),
                            torch.sin(angles[:, 0]).view([n_data, 1]),
                            torch.cos(angles[:, 0]).view([n_data, 1])],
                           dim=1
                           )
    
    rotation_Y = torch.cat([torch.cos(angles[:, 1]).view([n_data, 1]),
                            torch.zeros([n_data, 1]).to(device),
                            torch.sin(angles[:, 1]).view([n_data, 1]),
                            torch.zeros([n_data, 1]).to(device),
                            torch.ones([n_data, 1]).to(device),
                            torch.zeros([n_data, 1]).to(device),
                            -torch.sin(angles[:, 1]).view([n_data, 1]),
                            torch.zeros([n_data, 1]).to(device),
                            torch.cos(angles[:, 1]).view([n_data, 1])],
                           dim=1
                           )
    
    rotation_Z = torch.cat([torch.cos(angles[:, 2]).view([n_data, 1]),
                            -torch.sin(angles[:, 2]).view([n_data, 1]),
                            torch.zeros([n_data, 1]).to(device),
                            torch.sin(angles[:, 2]).view([n_data, 1]),
                            torch.cos(angles[:, 2]).view([n_data, 1]),
                            torch.zeros([n_data, 3]).to(device),
                            torch.ones([n_data, 1]).to(device)],
                           dim=1
                           )
    
    rotation_X = rotation_X.view([n_data, 3, 3])
    rotation_Y = rotation_Y.view([n_data, 3, 3])
    rotation_Z = rotation_Z.view([n_data, 3, 3])
    
    # R = RzRyRx
    rotation = torch.bmm(torch.bmm(rotation_Z, rotation_Y), rotation_X)
    
    # because our face shape is N*3, so compute the transpose of R, so that rotation shapes can be calculated as face_shape*R
    rotation = rotation.permute(0, 2, 1)
    
    return rotation



def Compute_norm ( face_shape, device=CFG.device ):
    shape = face_shape
    if device == "cpu":
        face_id = CFG.face_cpu
        point_id = CFG.point_buf_cpu
    else:
        face_id = CFG.face
        point_id = CFG.point_buf

    # compute normal for each face
    v1 = shape[:, face_id[:, 0].long()]
    v2 = shape[:, face_id[:, 1].long()]
    v3 = shape[:, face_id[:, 2].long()]
    e1 = v1 - v2
    e2 = v2 - v3
    face_norm = torch.cross(e1, e2)
    
    face_norm = F.normalize(face_norm, dim=2)
    face_norm = torch.cat([face_norm, torch.zeros([face_shape.shape[0], 1, 3]).to(device)], dim=1)
    
    # compute normal for each vertex using one-ring neighborhood
    v_norm = torch.squeeze(torch.sum(face_norm[:, point_id.long()], dim=2), dim=2)
    v_norm = F.normalize(v_norm, dim=2)
    
    return v_norm



def Illumination_block ( face_texture, norm_r, gamma ):
    batch_size = gamma.shape[0]
    n_point = norm_r.shape[1]
    gamma = gamma.view([batch_size, 3, 9])
    # set initial lighting with an ambient lighting
    init_lit = torch.tensor([0.8, 0, 0, 0, 0, 0, 0, 0, 0]).view([1, 1, 9]).to(CFG.device)
    gamma = gamma + init_lit
    
    # compute vertex color using SH function approximation
    a0 = torch.tensor(math.pi).to(CFG.device)
    a1 = torch.tensor(2 * math.pi / math.sqrt(3.0)).to(CFG.device)
    a2 = torch.tensor(2 * math.pi / math.sqrt(8.0)).to(CFG.device)
    c0 = torch.tensor(1 / math.sqrt(4 * math.pi)).to(CFG.device)
    c1 = torch.tensor(math.sqrt(3.0) / math.sqrt(4 * math.pi)).to(CFG.device)
    c2 = torch.tensor(3 * math.sqrt(5.0) / math.sqrt(12 * math.pi)).to(CFG.device)
    
    Y = torch.cat([(a0 * c0).view([1, 1, 1]).repeat(batch_size, n_point, 1),
                   torch.unsqueeze(-a1 * c1 * norm_r[:, :, 1], 2),
                   torch.unsqueeze(a1 * c1 * norm_r[:, :, 2], 2),
                   torch.unsqueeze(-a1 * c1 * norm_r[:, :, 0], 2),
                   torch.unsqueeze(a2 * c2 * norm_r[:, :, 0] * norm_r[:, :, 1], 2),
                   torch.unsqueeze(-a2 * c2 * norm_r[:, :, 1] * norm_r[:, :, 2], 2),
                   torch.unsqueeze(a2 * c2 * 0.5 / math.sqrt(3.0) * (3 * torch.square(norm_r[:, :, 2]) - 1), 2),
                   torch.unsqueeze(-a2 * c2 * norm_r[:, :, 0] * norm_r[:, :, 2], 2),
                   torch.unsqueeze(a2 * c2 * 0.5 * (torch.square(norm_r[:, :, 0]) - torch.square(norm_r[:, :, 1])), 2)],
                  dim=2)
    
    color_r = torch.squeeze(torch.bmm(Y, torch.unsqueeze(gamma[:, 0, :], dim=2)), dim=2)
    color_g = torch.squeeze(torch.bmm(Y, torch.unsqueeze(gamma[:, 1, :], dim=2)), dim=2)
    color_b = torch.squeeze(torch.bmm(Y, torch.unsqueeze(gamma[:, 2, :], dim=2)), dim=2)
    
    # [batchsize,N,3] vertex color in RGB order
    face_color = torch.stack(
        [color_r * face_texture[:, :, 0], color_g * face_texture[:, :, 1], color_b * face_texture[:, :, 2]], dim=2)
    
    return face_color


def Illumination_block_inverse(face_texture, norm_r, gamma, device=CFG.device):
    batch_size = gamma.shape[0]
    n_point = norm_r.shape[1]
    gamma = gamma.view([batch_size, 3, 9])
    # set initial lighting with an ambient lighting
    init_lit = torch.tensor([0.8, 0, 0, 0, 0, 0, 0, 0, 0]).view([1, 1, 9]).to(device)
    gamma = gamma + init_lit

    # compute vertex color using SH function approximation
    a0 = torch.tensor(math.pi).to(device)
    a1 = torch.tensor(2 * math.pi / math.sqrt(3.0)).to(device)
    a2 = torch.tensor(2 * math.pi / math.sqrt(8.0)).to(device)
    c0 = torch.tensor(1 / math.sqrt(4 * math.pi)).to(device)
    c1 = torch.tensor(math.sqrt(3.0) / math.sqrt(4 * math.pi)).to(device)
    c2 = torch.tensor(3 * math.sqrt(5.0) / math.sqrt(12 * math.pi)).to(device)

    Y = torch.cat([(a0 * c0).view([1, 1, 1]).repeat(batch_size, n_point, 1),
                   torch.unsqueeze(-a1 * c1 * norm_r[:, :, 1], 2),
                   torch.unsqueeze(a1 * c1 * norm_r[:, :, 2], 2),
                   torch.unsqueeze(-a1 * c1 * norm_r[:, :, 0], 2),
                   torch.unsqueeze(a2 * c2 * norm_r[:, :, 0] * norm_r[:, :, 1], 2),
                   torch.unsqueeze(-a2 * c2 * norm_r[:, :, 1] * norm_r[:, :, 2], 2),
                   torch.unsqueeze(a2 * c2 * 0.5 / math.sqrt(3.0) * (3 * torch.square(norm_r[:, :, 2]) - 1), 2),
                   torch.unsqueeze(-a2 * c2 * norm_r[:, :, 0] * norm_r[:, :, 2], 2),
                   torch.unsqueeze(a2 * c2 * 0.5 * (torch.square(norm_r[:, :, 0]) - torch.square(norm_r[:, :, 1])), 2)],
                  dim=2)

    color_r = torch.squeeze(torch.bmm(Y, torch.unsqueeze(gamma[:, 0, :], dim=2)), dim=2)
    color_g = torch.squeeze(torch.bmm(Y, torch.unsqueeze(gamma[:, 1, :], dim=2)), dim=2)
    color_b = torch.squeeze(torch.bmm(Y, torch.unsqueeze(gamma[:, 2, :], dim=2)), dim=2)

    # [batchsize,N,3] vertex color in RGB order
    face_color = torch.stack(
        [face_texture[:, :, 0] / color_r, face_texture[:, :, 1] / color_g, face_texture[:, :, 2] / color_b], dim=2)

    return face_color


def project_vertices(vertices, trans, angle):
    batch_size = vertices.shape[0]

    vertices = torch.bmm(vertices, Compute_rotation_matrix(angle)) + torch.unsqueeze(trans, 1)

    focal = 1100
    half_size = CFG.image_size / 2
    camera_pos = torch.tensor([0., 0., 10.]).view((1, 1, 3)).to(CFG.device)
    p_matrix = torch.tensor([
        [focal, 0,      half_size],
        [0,     focal,  half_size],
        [0,     0,      1],
    ]).to(CFG.device)
    p_matrix = torch.unsqueeze(p_matrix, 0).repeat(batch_size, 1, 1)

    reverse_z = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., -1.]]).to(CFG.device)
    reverse_z = torch.unsqueeze(reverse_z, 0).repeat(batch_size, 1, 1)
    shape = torch.bmm(vertices, reverse_z) + camera_pos

    aug_projection = torch.bmm(shape, torch.transpose(p_matrix, 1, 2))

    projection = aug_projection[:, :, 0:2] / aug_projection[:, :, 2].view((batch_size, aug_projection.shape[1], 1))

    u = CFG.image_size - projection[:, :, 1] - 1
    v = projection[:, :, 0]

    u = u.clamp(0, CFG.image_size - 1)
    v = v.clamp(0, CFG.image_size - 1)

    return u, v



def generate_full(vec, kind="shape"):
    assert kind in ["shape", "exp"]
    # std = getattr(CFG, f"std_{kind}")
    mean = getattr(CFG, f"mean_{kind}")

    # return vec * std + mean
    return vec + mean



def make_1d ( decoder_2d_result, vt2pixel_u, vt2pixel_v ):
    batch_size = decoder_2d_result.shape[0]
    decoder_1d_result = bilinear_sampler_torch(decoder_2d_result, vt2pixel_u, vt2pixel_v)
    decoder_1d_result = decoder_1d_result.view(batch_size, -1)
    
    return decoder_1d_result



def render_all(lv_trans, lv_angle, lv_il, vcolor, exp_1d, shape_1d, input_mask, input_background, landmark=False):
    shape_full = CFG.mean_shape + shape_1d + exp_1d
    vcolor_full = CFG.mean_tex + vcolor

    images, masks, vcolors = renderer.render(
        vertex_batch=shape_full,
        color_batch=vcolor_full,
        trans_batch=lv_trans,
        angle_batch=lv_angle,
        light_batch=lv_il,
        landmark=landmark
    )
    images = images.permute(0, 3, 1, 2)
    masks = masks.permute(0, 3, 1, 2)

    mask_combined = masks * input_mask

    return {
        "g_vcolor": vcolors,
        "g_mask": masks,
        "g_img": images * masks,
        # "g_img_bg": (images * mask_combined) + input_background * (1 - mask_combined),
        "g_img_bg": (images * masks) + input_background * (1 - masks * 1),
    }



def RGB2BGR(rgb):
    bgr = torch.zeros_like(rgb, device=CFG.device)
    bgr[:, 0, :, :] = rgb[:, 2, :, :]
    bgr[:, 1, :, :] = rgb[:, 1, :, :]
    bgr[:, 2, :, :] = rgb[:, 0, :, :]
    return bgr


def bilinear_sampler_torch ( img, x, y ):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - interpolated images according to grids. Same size as grid.
    """


    B, C, H, W = img.shape

    max_y = H - 1
    max_x = W - 1
    zero = 0

    x0 = torch.floor(x)
    x1 = x0 + 0
    y0 = torch.floor(y)
    y1 = y0 + 0

    # clip to range [0, H/W] to not violate img boundaries

    x0 = torch.clamp(x0, zero, max_x)
    x1 = torch.clamp(x1, zero, max_x)
    y0 = torch.clamp(y0, zero, max_y)
    y1 = torch.clamp(y1, zero, max_y)

    # get pixel value at corner coords

    Ia = get_pixel_value_torch(img, x0, y0)
    Ib = get_pixel_value_torch(img, x0, y1)
    Ic = get_pixel_value_torch(img, x1, y0)
    Id = get_pixel_value_torch(img, x1, y1)

    # recast as float for delta calculation

    x0 = x0.float()
    x1 = x1.float()
    y0 = y0.float()
    y1 = y1.float()

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition

    wa = torch.unsqueeze(wa, dim=3)
    wb = torch.unsqueeze(wb, dim=3)
    wc = torch.unsqueeze(wc, dim=3)
    wd = torch.unsqueeze(wd, dim=3)

    # compute output
    out = wa * Ia + wb * Ib + wc * Ic + wd * Id

    return out

def get_pixel_value_torch( img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W, )
    - y: flattened tensor of shape (B*H*W, )
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """

    batch_size, height, width = x.shape


    batch_idx = torch.arange(0, batch_size).view((batch_size, 1, 1)).type(torch.int64)
    b = batch_idx.repeat(1, height, width)

    value = img[b.long(), :, y.long(), x.long()]

    return value