# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
import os.path
import numpy as np
import os

# import tensorflow as tf
# from tensorflow.python.framework import ops
import torch
import torch.nn.functional as F
import ZBuffer_cuda
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from utils import *

# def get_shape(tensor):
#     static_shape = tensor.shape.as_list()
#     dynamic_shape = tf.unstack(tf.shape(tensor))
#     dims = [s[1] if s[0] is None else s[0]
#             for s in zip(static_shape, dynamic_shape)]
#     return dims

# _cuda_op_module_v2_sz224 = tf.load_op_library(os.path.join(tf.resource_loader.get_data_files_path(), 'TF_newop/cuda_op_kernel_v2_sz224.so'))
# zbuffer_tri_v2_sz224 = _cuda_op_module_v2_sz224.zbuffer_tri_v2_sz224
# ops.NotDifferentiable("ZbufferTriV2Sz224")
#
#
#
# def ZBuffer_Rendering_CUDA_op_v2_sz224(s2d, tri, vis):
#     tri_map, zbuffer = zbuffer_tri_v2_sz224(s2d, tri, vis)
#     return tri_map, zbuffer

def ZBuffer_Rendering_CUDA_op_v2_sz224_torch(s2d, tri, vis):
    # _cuda_op_module_v2_sz224 = tf.load_op_library(
    # os.path.join(tf.resource_loader.get_data_files_path(), 'TF_newop/cuda_op_kernel_v2_sz224.so'))
    # zbuffer_tri_v2_sz224 = _cuda_op_module_v2_sz224.zbuffer_tri_v2_sz224
    # tri_map, zbuffer = zbuffer_tri_v2_sz224(s2d, tri, vis)

    s2d = s2d.contiguous()
    # tri = tri.cuda()

    # s2d = s2d.double()
    # tri = tri.int()
    # vis = vis.double()


    map, mask = ZBuffer_cuda.forward(s2d, tri.int(), vis, TRI_NUM, VERTEX_NUM, OUTPUT_SIZE)

    return map, mask


# def warp_texture(texture, m, mshape, output_size=224):
#     '''
#         Render image
#         Parameters
#             texture:    [batch, 192, 224, 3]
#             m:          [batch, 8]
#             mshape:     [batch , VERTEXU_NUM * 3]
#         Returns
#             image: [batch, x, t, c]
#     '''
#     def flatten(x):
#         return tf.reshape(x, [-1])
#
#     n_size = get_shape(texture) #
#     n_size = n_size[0]
#
#     s = output_size
#
#     # Tri, tri2vt
#     tri = load_3DMM_tri()
#     vertex_tri = load_3DMM_vertex_tri()
#     vt2pixel_u, vt2pixel_v = load_3DMM_vt2pixel()
#
#     tri2vt1_const = tf.constant(tri[0,:], tf.int32)
#     tri2vt2_const = tf.constant(tri[1,:], tf.int32)
#     tri2vt3_const = tf.constant(tri[2,:], tf.int32)
#
#     tri_const = tf.constant(tri, tf.int32)
#     vertex_tri_const = tf.constant(vertex_tri, tf.int32)
#
#
#     #Vt2pix
#     vt2pixel_u_const = tf.constant(vt2pixel_u, tf.float32)
#     vt2pixel_v_const = tf.constant(vt2pixel_v, tf.float32)
#
#
#
#
#     # Convert projection matrix into 4x3 matrices
#     m = tf.reshape(m, [-1,4,2])                                                                              # batch_size x 4 x 2
#
#     m_row1 = tf.nn.l2_normalize(m[:,0:3,0], axis = 1)
#     m_row2 = tf.nn.l2_normalize(m[:,0:3,1], axis = 1)
#     m_row3 = tf.pad(tf.cross(m_row1, m_row2), [[0,0],[0,1]], mode='CONSTANT', constant_values=0)
#     m_row3 = tf.expand_dims(m_row3, axis=2)
#
#     m = tf.concat([m, m_row3], axis = 2)                                                                      # batch_size x 4 x 3
#
#
#
#
#     vertex3d = tf.reshape( mshape, shape = [n_size, -1, 3] )                                                  # batch_size x vertex_num x 3
#     vertex4d = tf.concat(axis = 2, values = [vertex3d, tf.ones(get_shape(vertex3d)[0:2] +[1], tf.float32)])   # batch_size x vertex_num x 4
#
#
#     vertex2d = tf.matmul(m, vertex4d, True, True)                                                             # batch_size x 3 x vertex_num
#     vertex2d = tf.transpose(vertex2d, perm=[0,2,1])                                                           # batch_size x vertex_num x 2
#
#
#
#     normal, normalf = compute_normal(vertex3d, tri_const, vertex_tri_const)                             # normal:  batch_size x vertex_num x 3  &  normalf: batch_size x tri_num x 3
#     normalf4d = tf.concat(axis = 2, values = [normalf, tf.ones(get_shape(normalf)[0:2] +[1], tf.float32)])    # batch_size x tri_num x 4
#
#
#     rotated_normalf = tf.matmul(m, normalf4d, True, True)                                                     # batch_size x 3 x tri_num
#     _, _, rotated_normalf_z = tf.split(axis=1, num_or_size_splits=3, value=rotated_normalf)                   # batch_size x 1 x tri_num
#
#     visible_tri = tf.greater(rotated_normalf_z, 0)
#
#
#     vertex2d_single    = tf.split(axis = 0, num_or_size_splits = n_size, value = vertex2d)
#     visible_tri_single = tf.split(axis = 0, num_or_size_splits = n_size, value = visible_tri)
#
#     pixel_u = []
#     pixel_v = []
#
#     masks = []
#
#     u, v = tf.meshgrid( tf.linspace(0.0, output_size-1.0, output_size), tf.linspace(0.0, output_size-1.0, output_size))
#     u = flatten(u)
#     v = flatten(v)
#
#
#     for i in range(n_size):
#         vertex2d_i = tf.squeeze(vertex2d_single[i], axis=0)                 # vertex_num x 3
#         visible_tri_i = tf.squeeze(visible_tri_single[i], axis=0)           # 1 x tri_num
#
#         [vertex2d_u, vertex2d_v, vertex2d_z]   = tf.split(axis=1, num_or_size_splits=3, value=vertex2d_i)
#         vertex2d_u = vertex2d_u - 1
#         vertex2d_v = s - vertex2d_v
#
#         vertex2d_i = tf.concat(axis=1, values=[vertex2d_v, vertex2d_u, vertex2d_z])
#         vertex2d_i = tf.transpose(vertex2d_i)
#
#         # Applying Z-buffer
#         tri_map_2d, mask_i = ZBuffer_Rendering_CUDA_op_v2_sz224(vertex2d_i, tri_const, visible_tri_i)
#
#         tri_map_2d_flat = tf.cast(tf.reshape(tri_map_2d, [-1]), 'int32')
#
#
#         # Calculate barycentric coefficient
#         vt1 = tf.gather( tri2vt1_const,  tri_map_2d_flat )
#         vt2 = tf.gather( tri2vt2_const,  tri_map_2d_flat )
#         vt3 = tf.gather( tri2vt3_const,  tri_map_2d_flat )
#
#
#         pixel1_uu = flatten(tf.gather( vertex2d_u,  vt1 ))
#         pixel2_uu = flatten(tf.gather( vertex2d_u,  vt2 ))
#         pixel3_uu = flatten(tf.gather( vertex2d_u,  vt3 ))
#
#         pixel1_vv = flatten(tf.gather( vertex2d_v,  vt1 ))
#         pixel2_vv = flatten(tf.gather( vertex2d_v,  vt2 ))
#         pixel3_vv = flatten(tf.gather( vertex2d_v,  vt3 ))
#         c1, c2, c3 = barycentric(pixel1_uu, pixel2_uu, pixel3_uu, pixel1_vv, pixel2_vv, pixel3_vv, u, v)
#
#
#         ##
#         pixel1_u = tf.gather( vt2pixel_u_const,  vt1 )
#         pixel2_u = tf.gather( vt2pixel_u_const,  vt2 )
#         pixel3_u = tf.gather( vt2pixel_u_const,  vt3 )
#
#         pixel1_v = tf.gather( vt2pixel_v_const,  vt1 )
#         pixel2_v = tf.gather( vt2pixel_v_const,  vt2 )
#         pixel3_v = tf.gather( vt2pixel_v_const,  vt3 )
#
#
#         pixel_u_i = tf.reshape(pixel1_u * c1 + pixel2_u * c2 + pixel3_u* c3, [output_size, output_size])
#         pixel_v_i = tf.reshape(pixel1_v * c1 + pixel2_v * c2 + pixel3_v* c3, [output_size, output_size])
#
#
#         pixel_u.append(pixel_u_i)
#         pixel_v.append(pixel_v_i)
#
#         masks.append(mask_i)
#
#     images = bilinear_sampler(texture, pixel_v, pixel_u)
#     masks = tf.stack(masks)
#
#     return images, masks



def warp_texture_torch ( texture, m, mshape, output_size=224 ):
    '''
        Render image
        Parameters
            texture:    [batch, 3, 192, 224]
            m:          [batch, 8]
            mshape:     [batch , VERTEXU_NUM * 3]
        Returns
            image: [batch, x, t, c]
    '''

    # texture = texture.permute((0, 2, 3, 1)) # [batch, 192, 224, 3]

    BATCH_SIZE = texture.shape[0]

    tri = torch.from_numpy(load_3DMM_tri()).cuda().long()  # [3, TRI_NUM + 1]
    vertex_tri = torch.from_numpy(load_3DMM_vertex_tri()).cuda()  # [8, VERTEX_NUM]
    vt2pixel_u, vt2pixel_v = [torch.from_numpy(vt2pixel).cuda() for vt2pixel in load_3DMM_vt2pixel()]  # [VERTEX_NUM + 1, ], [VERTEX_NUM + 1, ]


    m = m.view((BATCH_SIZE, 4, 2))
    m_row1 = F.normalize(m[:, 0:3, 0], dim=1)
    m_row2 = F.normalize(m[:, 0:3, 1], dim=1)
    m_row3 = torch.cross(m_row1, m_row2)
    m_row3 = F.pad(m_row3, (0, 1))
    m_row3 = torch.unsqueeze(m_row3, -1)
    m = torch.cat((m, m_row3), dim=2)

    vertex3d = mshape.view((BATCH_SIZE, -1, 3))
    vertex4d = F.pad(vertex3d, (0, 1), mode='constant', value=1)
    # vertex4d = torch.cat((vertex3d, torch.ones(texture_torch.shape[0:2] + [1])))

    # compare_numpy(vertex4d_, vertex4d)

    vertex2d = torch.matmul(vertex4d, m)
    # np.save('v2_t', vertex2d.cpu().numpy())
    # np.save('v4_t', vertex4d.cpu().numpy())
    # np.save('m_t', m.cpu().numpy())
    # vertex2d = torch.transpose(vertex2d, 1, 2)

    # compare_numpy(vertex2d_, vertex2d)

    normal, normalf = compute_normal_torch(vertex3d, tri, vertex_tri )  # normal:  batch_size x vertex_num x 3  &  normalf: batch_size x tri_num x 3
    # normal, normalf = _DEPRECATED_compute_normal_torch(vertex3d, tri, vertex_tri)
    # compare_numpy(normal_, normal)
    # compare_numpy(normalf_, normalf)

    # normalf4d_ = tf.concat(axis=2, values=[normalf_, tf.ones(get_shape(normalf_)[0:2] + [1], tf.float64)])  # batch_size x tri_num x 4
    normalf4d = F.pad(normalf, (0, 1), mode='constant', value=1).cuda()
    # print(normalf4d, m)
    # compare_numpy(normalf4d_, normalf4d)

    # rotated_normalf_ = tf.matmul(m_, normalf4d_, True, True)  # batch_size x 3 x tri_num
    # _, _, rotated_normalf_z_ = tf.split(axis=1, num_or_size_splits=3,
    #                                     value=rotated_normalf_)  # batch_size x 1 x tri_num
    # visible_tri_ = tf.greater(rotated_normalf_z_, 0)
    # vertex2d_single_ = tf.split(axis=0, num_or_size_splits=n_size_, value=vertex2d_)
    # visible_tri_single_ = tf.split(axis=0, num_or_size_splits=n_size_, value=visible_tri_)

    rotated_normalf = torch.matmul(normalf4d, m)
    rotated_normalf = torch.transpose(rotated_normalf, 1, 2)
    _, _, rotated_normalf_z = torch.split(rotated_normalf, (1, 1, 1), dim=1)
    visible_tri = torch.gt(rotated_normalf_z, 0)

    # compare_numpy(rotated_normalf_z_, rotated_normalf_z)

    # pixel_u_ = []
    # pixel_v_ = []
    #
    # masks_ = []
    #
    # u_, v_ = tf.meshgrid(tf.linspace(0.0, output_size - 1.0, output_size),
    #                      tf.linspace(0.0, output_size - 1.0, output_size))
    # u_ = flatten(u_)
    # v_ = flatten(v_)
    # u_ = tf.cast(u_, dtype=tf.float64)
    # v_ = tf.cast(v_, dtype=tf.float64)

    grid = torch.linspace(0, output_size - 1, output_size)
    u, v = torch.meshgrid(grid, grid)
    u = torch.transpose(u, 0, 1)
    v = torch.transpose(v, 0, 1)
    u = torch.flatten(u).cuda()
    v = torch.flatten(v).cuda()

    # ########################################################################################### begin of for
    # for i in range(n_size_):
    #     vertex2d_i_ = tf.squeeze(vertex2d_single_[i], axis=0)  # vertex_num x 3
    #     visible_tri_i_ = tf.squeeze(visible_tri_single_[i], axis=0)  # 1 x tri_num
    #
    #     # vertex2d_i = torch.squeeze(vertex2d, dim=1)
    #     # visible_tri_i = torch.squeeze(visible_tri, dim=1)
    #
    #     [vertex2d_u_, vertex2d_v_, vertex2d_z_] = tf.split(axis=1, num_or_size_splits=3, value=vertex2d_i_)
    #     vertex2d_u_ = vertex2d_u_ - 1
    #     vertex2d_v_ = output_size - vertex2d_v_
    #     vertex2d_i_ = tf.concat(axis=1, values=[vertex2d_v_, vertex2d_u_, vertex2d_z_])
    #     vertex2d_i_ = tf.transpose(vertex2d_i_)
    #
    #     # vertex2d_u, vertex2d_v, vertex2d_z = torch.split(vertex2d_i, (1, 1, 1), dim=2)
    #     # vertex2d_u = vertex2d_u - 1
    #     # vertex2d_v = output_size - vertex2d_v
    #     # vertex2d_i = torch.cat((vertex2d_v, vertex2d_u, vertex2d_z), dim=2)
    #     # vertex2d_i = torch.transpose(vertex2d_i, 1, 2)
    #
    #     # Applying Z-buffer
    #     tri_map_2d_, mask_i_ = ZBuffer_Rendering_CUDA_op_v2_sz224(vertex2d_i_, tri_const_, visible_tri_i_)
    #     tri_map_2d_flat_ = tf.cast(tf.reshape(tri_map_2d_, [-1]), 'int32')
    #
    #     # tri_map_2d_flat = tri_map_2d.view((BATCH_SIZE, -1))
    #
    #     # Calculate barycentric coefficient
    #     vt1_ = tf.gather(tri2vt1_const_, tri_map_2d_flat_)
    #     vt2_ = tf.gather(tri2vt2_const_, tri_map_2d_flat_)
    #     vt3_ = tf.gather(tri2vt3_const_, tri_map_2d_flat_)
    #
    #     # vt = torch.gather(torch.unsqueeze(torch.transpose(tri, 0, 1), 0).repeat(BATCH_SIZE, 1, 1), 1, torch.unsqueeze(tri_map_2d_flat, dim=-1).repeat(1, 1, 3))
    #
    #     # compare_numpy(vt1_, vt[0][:, 0])
    #     # compare_numpy(vt2_, vt[0][:, 1])
    #     # compare_numpy(vt3_, vt[0][:, 2])
    #
    #     pixel1_uu_ = flatten(tf.gather(vertex2d_u_, vt1_))
    #     pixel2_uu_ = flatten(tf.gather(vertex2d_u_, vt2_))
    #     pixel3_uu_ = flatten(tf.gather(vertex2d_u_, vt3_))
    #
    #     # pixel_uu = torch.gather(F.pad(vertex2d_u.repeat(1, 1, 3), (0, 0, 0, 1)), 1, vt)
    #
    #     # compare_numpy(pixel1_uu_, pixel_uu[0][:, 0])
    #     # compare_numpy(pixel2_uu_, pixel_uu[0][:, 1])
    #     # compare_numpy(pixel3_uu_, pixel_uu[0][:, 2])
    #
    #     pixel1_vv_ = flatten(tf.gather(vertex2d_v_, vt1_))
    #     pixel2_vv_ = flatten(tf.gather(vertex2d_v_, vt2_))
    #     pixel3_vv_ = flatten(tf.gather(vertex2d_v_, vt3_))
    #
    #     # pixel_vv = torch.gather(F.pad(vertex2d_v.repeat(1, 1, 3), (0, 0, 0, 1)), 1, vt)
    #
    #     # compare_numpy(pixel1_vv_, pixel_vv[0][:, 0])
    #     # compare_numpy(pixel2_vv_, pixel_vv[0][:, 1])
    #     # compare_numpy(pixel3_vv_, pixel_vv[0][:, 2])
    #
    #     c1_, c2_, c3_ = barycentric(pixel1_uu_, pixel2_uu_, pixel3_uu_, pixel1_vv_, pixel2_vv_, pixel3_vv_, u_, v_)
    #
    #     ##
    #     pixel1_u_ = tf.gather(vt2pixel_u_const_, vt1_)
    #     pixel2_u_ = tf.gather(vt2pixel_u_const_, vt2_)
    #     pixel3_u_ = tf.gather(vt2pixel_u_const_, vt3_)
    #
    #     # pixel_u = torch.gather(torch.unsqueeze(vt2pixel_u, -1).repeat(BATCH_SIZE, 1, 3), 1, vt)
    #
    #     # compare_numpy(pixel1_u_, pixel_u[0][:, 0])
    #     # compare_numpy(pixel2_u_, pixel_u[0][:, 1])
    #     # compare_numpy(pixel3_u_, pixel_u[0][:, 2])
    #
    #     pixel1_v_ = tf.gather(vt2pixel_v_const_, vt1_)
    #     pixel2_v_ = tf.gather(vt2pixel_v_const_, vt2_)
    #     pixel3_v_ = tf.gather(vt2pixel_v_const_, vt3_)
    #
    #     # pixel_v = torch.gather(torch.unsqueeze(vt2pixel_v, -1).repeat(BATCH_SIZE, 1, 3), 1, vt)
    #
    #     # compare_numpy(pixel1_v_, pixel_v[0][:, 0])
    #     # compare_numpy(pixel2_v_, pixel_v[0][:, 1])
    #     # compare_numpy(pixel3_v_, pixel_v[0][:, 2])
    #
    #     pixel_u_i_ = tf.reshape(pixel1_u_ * c1_ + pixel2_u_ * c2_ + pixel3_u_ * c3_, [output_size, output_size])
    #     pixel_v_i_ = tf.reshape(pixel1_v_ * c1_ + pixel2_v_ * c2_ + pixel3_v_ * c3_, [output_size, output_size])
    #
    #     # pixel_u = torch.sum((pixel_u * torch.cat((torch.unsqueeze(c1, -1), torch.unsqueeze(c2, -1), torch.unsqueeze(c3, -1)), -1)), dim=2).view((BATCH_SIZE, output_size, output_size))
    #     # pixel_v = torch.sum((pixel_v * torch.cat((torch.unsqueeze(c1, -1), torch.unsqueeze(c2, -1), torch.unsqueeze(c3, -1)), -1)), dim=2).view((BATCH_SIZE, output_size, output_size))
    #
    #     # compare_numpy(pixel_u_i_, pixel_u[0])
    #     # compare_numpy(pixel_v_i_, pixel_v[0])
    #
    #     pixel_u_.append(pixel_u_i_)
    #     pixel_v_.append(pixel_v_i_)
    #
    #     masks_.append(mask_i_)
    # ########################################################################################### end of for

    vertex2d_i = torch.squeeze(vertex2d, dim=1)
    visible_tri_i = torch.squeeze(visible_tri, dim=1)

    vertex2d_u, vertex2d_v, vertex2d_z = torch.split(vertex2d_i, (1, 1, 1), dim=2)
    vertex2d_u = vertex2d_u - 1
    vertex2d_v = output_size - vertex2d_v
    vertex2d_i = torch.cat((vertex2d_v, vertex2d_u, vertex2d_z), dim=2)
    vertex2d_i = torch.transpose(vertex2d_i, 1, 2)

    # np.save('i1_t', vertex2d_i.cpu().numpy())
    # np.save('i2_t', visible_tri.cpu().numpy())
    # vertex2d_i = torch.from_numpy(np.load('i1.npy')).cuda()
    tri_map_2d = []
    masks = []
    for i in range(BATCH_SIZE):
        tri_map_2d_i, masks_i = ZBuffer_Rendering_CUDA_op_v2_sz224_torch(vertex2d_i[i].float(), tri, visible_tri[i])
        tri_map_2d.append(tri_map_2d_i)
        masks.append(masks_i)

    tri_map_2d = torch.stack(tri_map_2d)
    masks = torch.stack(masks)

    # np.save('o1_t', tri_map_2d.cpu().numpy())
    # np.save('o2_t', masks.cpu().numpy())

    tri_map_2d_flat = tri_map_2d.view((BATCH_SIZE, -1))

    vt = torch.gather(torch.unsqueeze(torch.transpose(tri, 0, 1), 0).repeat(BATCH_SIZE, 1, 1), 1,
                      torch.unsqueeze(tri_map_2d_flat, dim=-1).repeat(1, 1, 3).long())

    pixel_uu = torch.gather(F.pad(vertex2d_u.repeat(1, 1, 3), (0, 0, 0, 1)), 1, vt)

    pixel_vv = torch.gather(F.pad(vertex2d_v.repeat(1, 1, 3), (0, 0, 0, 1)), 1, vt)

    c1, c2, c3 = barycentric_torch(pixel_uu, pixel_vv, u, v)

    pixel_u = torch.gather(torch.unsqueeze(vt2pixel_u, -1).repeat(BATCH_SIZE, 1, 3), 1, vt)
    pixel_v = torch.gather(torch.unsqueeze(vt2pixel_v, -1).repeat(BATCH_SIZE, 1, 3), 1, vt)

    # pixel_u = torch.sum(
    #     (pixel_u * torch.cat((torch.unsqueeze(c1, -1), torch.unsqueeze(c2, -1), torch.unsqueeze(c3, -1)), -1)),
    #     dim=2).view((BATCH_SIZE, output_size, output_size))
    # pixel_v = torch.sum(
    #     (pixel_v * torch.cat((torch.unsqueeze(c1, -1), torch.unsqueeze(c2, -1), torch.unsqueeze(c3, -1)), -1)),
    #     dim=2).view((BATCH_SIZE, output_size, output_size))

    pixel1_u, pixel2_u, pixel3_u = torch.split(pixel_u, (1, 1, 1), dim=-1)
    pixel1_v, pixel2_v, pixel3_v = torch.split(pixel_v, (1, 1, 1), dim=-1)

    pixel_u = torch.squeeze(pixel1_u) * c1 + torch.squeeze(pixel2_u) * c2 + torch.squeeze(pixel3_u) * c3
    pixel_v = torch.squeeze(pixel1_v) * c1 + torch.squeeze(pixel2_v) * c2 + torch.squeeze(pixel3_v) * c3

    pixel_u = pixel_u.view((BATCH_SIZE, output_size, output_size))
    pixel_v = pixel_v.view((BATCH_SIZE, output_size, output_size))

    # pixel_v_ = tf.stack(pixel_v_)
    # pixel_u_ = tf.stack(pixel_u_)
    #
    # compare_numpy(pixel_v_, pixel_v)
    # compare_numpy(pixel_u_, pixel_u)
    #
    # images_ = bilinear_sampler(texture_tf, pixel_v_, pixel_u_)
    # texture = texture.permute((0, 3, 1, 2))
    images = bilinear_sampler_torch(texture, pixel_v, pixel_u)
    # masks_ = tf.stack(masks_)

    # compare_numpy(images_, images)
    # compare_numpy(masks_, masks)

    # return images.permute((0, 3, 1, 2)), masks.permute((0, 3, 1, 2))
    return images.permute((0, 3, 1, 2)), masks



# def barycentric(pixel1_u, pixel2_u, pixel3_u, pixel1_v, pixel2_v, pixel3_v, u, v):
#
#     v0_u = pixel2_u - pixel1_u
#     v0_v = pixel2_v - pixel1_v
#
#     v1_u = pixel3_u - pixel1_u
#     v1_v = pixel3_v - pixel1_v
#
#     v2_u = u - pixel1_u
#     v2_v = v - pixel1_v
#
#     invDenom = 1.0/(v0_u * v1_v - v1_u * v0_v + 1e-6)
#     c2 = (v2_u * v1_v - v1_u * v2_v) * invDenom
#     c3 = (v0_u * v2_v - v2_u * v0_v) * invDenom
#     c1 = 1.0 - c2 - c3
#
#     return c1, c2, c3



def barycentric_torch( pixel_uu, pixel_vv, u, v):
    # v0_u_ = pixel2_u_ - pixel1_u_
	# v0_v_ = pixel2_v_ - pixel1_v_
    v0_u = pixel_uu[:, :, 1] - pixel_uu[:, :, 0]
    v0_v = pixel_vv[:, :, 1] - pixel_vv[:, :, 0]
    # v1_u_ = pixel3_u_ - pixel1_u_
    # v1_v_ = pixel3_v_ - pixel1_v_
    v1_u = pixel_uu[:, :, 2] - pixel_uu[:, :, 0]
    v1_v = pixel_vv[:, :, 2] - pixel_vv[:, :, 0]

    # v2_u_ = u_ - pixel1_u_
    # v2_v_ = v_ - pixel1_v_

    v2_u = u - pixel_uu[:, :, 0]
    v2_v = v - pixel_vv[:, :, 0]

    # invDenom_ = 1.0/(v0_u_ * v1_v_ - v1_u_ * v0_v_ + 1e-6)
    # c2_ = (v2_u_ * v1_v_ - v1_u_ * v2_v_) * invDenom_
    # c3_ = (v0_u_ * v2_v_ - v2_u_ * v0_v_) * invDenom_
    # c1_ = 1.0 - c2_ - c3_

    invDenom = (v0_u * v1_v - v1_u * v0_v + 1e-6)
    c2 = (v2_u * v1_v - v1_u * v2_v) / (invDenom)
    c3 = (v0_u * v2_v - v2_u * v0_v) / (invDenom)
    c1 = 1.0 - c2 - c3

    return c1, c2, c3



# def barycentric_alternative(pixel1_u, pixel2_u, pixel3_u, pixel1_v, pixel2_v, pixel3_v, u, v):
#     '''
#     More complicated version
#     '''
#     v0_u = pixel2_u - pixel1_u
#     v0_v = pixel2_v - pixel1_v
#
#     v1_u = pixel3_u - pixel1_u
#     v1_v = pixel3_v - pixel1_v
#
#     v2_u = u - pixel1_u
#     v2_v = v - pixel1_v
#
#     d00 = v0_u * v0_u + v0_v*v0_v
#     d01 = v0_u * v1_u + v0_v*v1_v
#     d11 = v1_u * v1_u + v1_v*v1_v
#     d20 = v2_u * v0_u + v2_v*v0_v
#     d21 = v2_u * v1_u + v2_v*v1_v
#
#     invDenom = 1.0 / (d00 * d11 - d01 * d01 + 1e-6)
#     c3 = (d11 * d20 - d01 * d21) * invDenom
#     c2 = (d00 * d21 - d01 * d20) * invDenom
#     c1 = 1.0 - c2 - c3
#
#     return c1, c2, c3
#
#
# def compute_normal(vertex, tri, vertex_tri):
#     # Unit normals to the faces
#     # Parameters:
#     #   vertex : batch_size x vertex_num x 3
#     #   tri : 3xtri_num
#     #   vertex_tri: T x vertex_num (T=8: maxium number of triangle each vertex can belong to)
#     # Output
#     #   normal:  batch_size x vertex_num x 3
#     #   normalf: batch_size x tri_num x 3
#
#     vt1_indices, vt2_indices, vt3_indices = tf.split(tri, num_or_size_splits = 3, axis = 0)
#
#
#     # Dimensions
#     batch_size = tf.shape(vertex)[0]
#     tri_num    = tf.shape(tri)[1]
#     vertex_num = tf.shape(vertex_tri)[1]
#     T = tf.shape(vertex_tri)[0]
#
#
#     # Create batch indices for tf.gather_nd
#     batch_idx = tf.range(0, batch_size)
#     batch_idx = tf.reshape(batch_idx, (batch_size, 1))
#     b = tf.tile(batch_idx, (1, tri_num))
#
#     k1 = tf.tile(vt1_indices, (batch_size, 1))
#     k2 = tf.tile(vt2_indices, (batch_size, 1))
#     k3 = tf.tile(vt3_indices, (batch_size, 1))
#
#     vt1_indices = tf.stack([b, k1], 2)
#     vt2_indices = tf.stack([b, k2], 2)
#     vt3_indices = tf.stack([b, k3], 2)
#
#
#     # Compute triangle normal using its vertices 3dlocation
#     vt1 = tf.gather_nd(vertex, vt1_indices)                     #batch_size x tri_num x 3
#     vt2 = tf.gather_nd(vertex, vt2_indices)
#     vt3 = tf.gather_nd(vertex, vt3_indices)
#
#
#     normalf = tf.cross(vt2 - vt1, vt3 - vt1)
#     normalf = tf.nn.l2_normalize(normalf, dim = 2)
#
#     mask = tf.expand_dims(tf.tile( tf.expand_dims(  tf.not_equal(vertex_tri, tri.shape[1] - 1), 2), multiples = [1, 1, 3]), 0)
#     mask = tf.cast( mask, vertex.dtype  )
#
#
#     # Compute vertices normal
#     vertex_tri = tf.reshape(vertex_tri, shape = [1, -1])
#
#     b = tf.tile(batch_idx, (1, T * vertex_num))
#     k = tf.tile(vertex_tri, (batch_size, 1))
#
#     indices = tf.stack([b, k], 2)
#
#
#     normal = tf.gather_nd(normalf, indices)
#     normal = tf.reshape(normal, shape = [-1, T, vertex_num, 3])
#
#     normal = tf.reduce_sum( tf.multiply( normal, mask ),  axis = 1)
#     normal = tf.nn.l2_normalize(normal, dim = 2)
#
#
#     # Enforce that the normal are outward
#
#     v = vertex - tf.reduce_mean(vertex,1, keepdims=True)
#     s = tf.reduce_sum( tf.multiply(v, normal), 1, keepdims=True )
#
#     count_s_greater_0 = tf.count_nonzero( tf.greater(s, 0), axis=0, keepdims=True )
#     count_s_less_0 = tf.count_nonzero( tf.less(s, 0), axis=0, keepdims=True )
#
#     sign = 2 * tf.cast(tf.greater(count_s_greater_0, count_s_less_0), tf.float32) - 1
#     normal = tf.multiply(normal, sign)
#     normalf = tf.multiply(normalf, sign)
#
#
#     return normal, normalf


def compute_normal_torch ( vertex, tri, vertex_tri ):
    # Unit normals to the faces
    # Parameters:
    #   vertex : batch_size x vertex_num x 3
    #   tri : 3xtri_num
    #   vertex_tri: T x vertex_num (T=8: maxium number of triangle each vertex can belong to)
    # Output
    #   normal:  batch_size x vertex_num x 3
    #   normalf: batch_size x tri_num x 3

    # vt1_indices_, vt2_indices_, vt3_indices_ = tf.split(tri_, num_or_size_splits=3, axis=0)

    # Dimensions
    # batch_size_ = tf.shape(vertex_)[0]
    # tri_num_ = tf.shape(tri_)[1]
    # vertex_num_ = tf.shape(vertex_tri_)[1]
    # T_ = tf.shape(vertex_tri_)[0]

    # Create batch indices for tf.gather_nd
    # batch_idx_ = tf.range(0, batch_size_)
    # batch_idx_ = tf.reshape(batch_idx_, (batch_size_, 1))
    # b_ = tf.tile(batch_idx_, (1, tri_num_))
    #
    # k1_ = tf.tile(vt1_indices_, (batch_size_, 1))
    # k2_ = tf.tile(vt2_indices_, (batch_size_, 1))
    # k3_ = tf.tile(vt3_indices_, (batch_size_, 1))
    #
    # vt1_indices_ = tf.stack([b_, k1_], 2)
    # vt2_indices_ = tf.stack([b_, k2_], 2)
    # vt3_indices_ = tf.stack([b_, k3_], 2)

    # Compute triangle normal using its vertices 3dlocation
    # vt1_ = tf.gather_nd(vertex_, vt1_indices_)  # batch_size x tri_num x 3
    # vt2_ = tf.gather_nd(vertex_, vt2_indices_)
    # vt3_ = tf.gather_nd(vertex_, vt3_indices_)
    #
    # normalf_ = tf.cross(vt2_ - vt1_, vt3_ - vt1_)
    # normalf_ = tf.nn.l2_normalize(normalf_, dim=2)

    BATCH_SIZE = vertex.shape[0]

    tri = torch.transpose(tri, 0, 1)
    tri = torch.unsqueeze(tri, 0)
    tri = torch.unsqueeze(tri, -1)
    tri = torch.cat(BATCH_SIZE * [tri])[:, :-1, :]

    vt1_indices = torch.cat(3 * [tri[:, :, 0]], -1).cuda()
    vt2_indices = torch.cat(3 * [tri[:, :, 1]], -1).cuda()
    vt3_indices = torch.cat(3 * [tri[:, :, 2]], -1).cuda()
    vt1 = torch.gather(vertex, 1, vt1_indices)
    vt2 = torch.gather(vertex, 1, vt2_indices)
    vt3 = torch.gather(vertex, 1, vt3_indices)
    zeros = torch.zeros((BATCH_SIZE, 1, 3)).cuda()
    vt1_padded = torch.cat((vt1, zeros), 1)
    vt2_padded = torch.cat((vt2, zeros), 1)
    vt3_padded = torch.cat((vt3, zeros), 1)

    normalf = torch.cross(vt2_padded - vt1_padded, vt3_padded - vt1_padded)
    normalf = F.normalize(normalf, dim=2)
    # normalf = torch.cat((normalf, zeros), 1)

    # compare_numpy(normalf_, normalf)

    # mask_ = tf.expand_dims(tf.tile(tf.expand_dims(tf.not_equal(vertex_tri_, tri_.shape[1] - 1), 2), multiples=[1, 1, 3]), 0)
    # mask_ = tf.cast(mask_, vertex_.dtype)

    # Compute vertices normal
    # vertex_tri_ = tf.reshape(vertex_tri_, shape=[1, -1])

    # b_ = tf.tile(batch_idx_, (1, T_ * vertex_num_))
    # k_ = tf.tile(vertex_tri_, (batch_size_, 1))
    #
    # indices_ = tf.stack([b_, k_], 2)
    #
    # normal_ = tf.gather_nd(normalf_, indices_)
    # normal_ = tf.reshape(normal_, shape=[-1, T_, vertex_num_, 3])
    #
    # normal_ = tf.reduce_sum(tf.multiply(normal_, mask_), axis=1)
    # normal_ = tf.nn.l2_normalize(normal_, dim=2)

    equal = vertex_tri != (tri.shape[1])
    expand = torch.unsqueeze(equal, 2)

    mask = expand.repeat(1, 1, 3).cuda()

    vertex_tri = vertex_tri.view((-1, 1))

    normal_indices = torch.unsqueeze(vertex_tri, 0)
    normal_indices = torch.cat(BATCH_SIZE * [normal_indices])
    normal_indices = torch.cat(3 * [normal_indices], -1).cuda()
    normal = torch.gather(normalf, 1, normal_indices.long())

    normal = normal.view((BATCH_SIZE, 8, -1, 3))
    multi = torch.mul(normal, mask)
    normal = torch.sum(multi, dim=1)

    # norm_sum = torch.norm(normal, dim=2, keepdim=True)
    # normal = torch.div(normal, norm_sum)

    normal = F.normalize(normal, dim=2)

    # Enforce that the normal are outward

    # v_ = vertex_ - tf.reduce_mean(vertex_, 1, keepdims=True)
    # s_ = tf.reduce_sum(tf.multiply(v_, normal_), 1, keepdims=True)

    mean = torch.mean(vertex, 1)
    mean = torch.unsqueeze(mean, 1)
    v = vertex - mean
    s = torch.sum(torch.mul(v, normal), 1, keepdim=True)

    # count_s_greater_0_ = tf.count_nonzero(tf.greater(s_, 0), axis=0, keepdims=True)
    # count_s_less_0_ = tf.count_nonzero(tf.less(s_, 0), axis=0, keepdims=True)

    count_s_greater_0 = torch.sum(1 * torch.gt(s, 0), 0, keepdim=True)
    count_s_less_0 = torch.sum(1 * torch.lt(s, 0), 0, keepdim=True)

    # compare_numpy(count_s_less_0_, count_s_less_0)

    # sign_ = 2 * tf.cast(tf.greater(count_s_greater_0_, count_s_less_0_), tf.float64) - 1
    # normal_ = tf.multiply(normal_, sign_)
    # normalf_ = tf.multiply(normalf_, sign_)

    sign = 2 * torch.gt(count_s_greater_0, count_s_less_0) - 1
    # sign = sign.repeat((1, VERTEX_NUM, 1))
    # sign = torch.unsqueeze(sign, 1)
    # sign = torch.unsqueeze(sign, -1)
    normal = torch.mul(normal, sign.repeat((1, VERTEX_NUM, 1)))
    normalf = torch.mul(normalf, sign.repeat((1, TRI_NUM + 1, 1)))

    return normal, normalf



# def compute_tri_normal(vertex,tri, vertex_tri):
#     # Unit normals to the faces
#     # vertex : 3xvertex_num
#     # tri : 3xtri_num
#
#     vertex = tf.transpose(vertex)
#
#     vt1_indices, vt2_indices, vt3_indices = tf.split(tf.transpose(tri), num_or_size_splits = 3, axis = 1)
#
#     vt1 = tf.gather_nd(vertex, vt1_indices)
#     vt2 = tf.gather_nd(vertex, vt2_indices)
#     vt3 = tf.gather_nd(vertex, vt3_indices)
#
#     normalf = tf.cross(vt2 - vt1, vt3 - vt1)
#     normalf = tf.nn.l2_normalize(normalf, dim = 1)
#
#     return normalf

# compute_normal2 = compute_tri_normal


# def compute_landmarks(m, shape, output_size=224):
#     # m: rotation matrix [batch_size x (4x2)]
#     # shape: 3d vertices location [batch_size x (vertex_num x 3)]
#
#     n_size = get_shape(m)
#     n_size = n_size[0]
#
#     s = output_size
#
#
#     # Tri, tri2vt
#     kpts = load_3DMM_kpts()
#     kpts_num = kpts.shape[0]
#
#     indices = np.zeros([n_size, kpts_num,2], np.int32)
#     for i in range(n_size):
#         indices[i,:,0] = i
#         indices[i,:,1:2] = kpts
#
#     indices = tf.constant(indices, tf.int32)
#
#     kpts_const = tf.constant(kpts, tf.int32)
#
#     vertex3d = tf.reshape( shape, shape = [n_size, -1, 3] )                                                   # batch_size x vertex_num x 3
#     vertex3d = tf.gather_nd(vertex3d, indices)        # Keypointd selection                                   # batch_size x kpts_num x 3
#     vertex4d = tf.concat(axis = 2, values = [vertex3d, tf.ones(get_shape(vertex3d)[0:2] +[1], tf.float32)])   # batch_size x kpts_num x 4
#
#     m = tf.reshape( m, shape = [n_size, 4, 2] )
#     vertex2d = tf.matmul(m, vertex4d, True, True)                                                             # batch_size x 2 x kpts_num
#     vertex2d = tf.transpose(vertex2d, perm=[0,2,1])                                                           # batch_size x kpts_num x 2
#
#     [vertex2d_u, vertex2d_v]  = tf.split(axis=2, num_or_size_splits=2, value=vertex2d)
#     vertex2d_u = vertex2d_u - 1
#     vertex2d_v = s - vertex2d_v
#
#     return vertex2d_u, vertex2d_v


def compute_landmarks_torch(m, shape, output_size):
    n_size = m.shape[0]
    s = output_size

    kpts = torch.from_numpy(load_3DMM_kpts())
    kpts_num = kpts.shape[0]
    indices = torch.zeros([n_size, kpts_num, 2]).int()
    for i in range(n_size):
        indices[i, :, 0] = i
        indices[i, :, 1:2] = kpts

    vertex3d = shape.view((n_size, -1, 3))
    indices1 = torch.arange(0, n_size).long()
    vertex3d = vertex3d[indices1, kpts.long(), :].permute((1, 0, 2))
    vertex4d = torch.cat((vertex3d, torch.ones((vertex3d.shape[0], vertex3d.shape[1], 1)).cuda()), dim=2)

    m = m.view((n_size, 4, 2))
    vertex2d = torch.matmul(vertex4d, m)

    [vertex2d_u, vertex2d_v] = torch.split(vertex2d, (1, 1), dim=2)
    vertex2d_u = vertex2d_u - 1
    vertex2d_v = s - vertex2d_v

    return vertex2d_u, vertex2d_v




# def rotate_shape(m, mshape, output_size = 224):
#
#     n_size = get_shape(m)
#     n_size = n_size[0]
#
#     m_single     = tf.split(axis = 0, num_or_size_splits = n_size, value = m)
#     shape_single = tf.split(axis = 0, num_or_size_splits = n_size, value = mshape)
#
#     vertex2ds = []
#
#     for i in range(n_size):
#
#         m_i = tf.transpose(tf.reshape(m_single[i], [4,2]))
#         m_i_row1 = tf.nn.l2_normalize(m_i[0,0:3], dim = 0)
#         m_i_row2 = tf.nn.l2_normalize(m_i[1,0:3], dim = 0)
#         m_i_row3 = tf.concat([tf.reshape(tf.cross(m_i_row1, m_i_row2), shape = [1, 3]), tf.zeros([1, 1])], axis = 1)
#
#         m_i = tf.concat([m_i, m_i_row3], axis = 0)
#
#         vertex3d_rs = tf.transpose(tf.reshape( shape_single[i], shape = [-1, 3] ))
#
#         vertex4d = tf.concat(axis = 0, values = [vertex3d_rs, tf.ones([1, get_shape(vertex3d_rs)[1]], tf.float32)])
#
#         vertex2d = tf.matmul(m_i, vertex4d, False, False)
#         vertex2d = tf.transpose(vertex2d)
#
#         [vertex2d_u, vertex2d_v, vertex2d_z]   = tf.split(axis=1, num_or_size_splits=3, value=vertex2d)
#         vertex2d_u = vertex2d_u - 1
#         vertex2d_v = output_size - vertex2d_v
#
#         vertex2d = tf.concat(axis=1, values=[vertex2d_v, vertex2d_u, vertex2d_z])
#         vertex2d = tf.transpose(vertex2d)
#
#         vertex2ds.append(vertex2d)
#
#     return tf.stack(vertex2ds)
#
#
# def get_pixel_value(img, x, y):
#     """
#     Utility function to get pixel value for coordinate
#     vectors x and y from a  4D tensor image.
#     Input
#     -----
#     - img: tensor of shape (B, H, W, C)
#     - x: flattened tensor of shape (B*H*W, )
#     - y: flattened tensor of shape (B*H*W, )
#     Returns
#     -------
#     - output: tensor of shape (B, H, W, C)
#     """
#     shape = tf.shape(x)
#     batch_size = shape[0]
#     height = shape[1]
#     width = shape[2]
#
#
#     batch_idx = tf.range(0, batch_size)
#     batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
#     b = tf.tile(batch_idx, (1, height, width))
#
#
#     indices = tf.stack([b, y, x], 3)
#
#     return tf.gather_nd(img, indices)


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
    # placeholder_x = np.random.randint(0, 224, (BATCH_SIZE, 224, 224))
    # placeholder_y = np.random.randint(0, 192, (BATCH_SIZE, 224, 224))
    #
    # x_ = tf.constant(placeholder_x)
    # y_ = tf.constant(placeholder_y)
    # x = torch.from_numpy(placeholder_x)
    # y = torch.from_numpy(placeholder_y)

    # shape_ = tf.shape(x_)
    # batch_size_ = shape_[0]
    # height_ = shape_[1]
    # width_ = shape_[2]

    batch_size, height, width = x.shape


    # batch_idx_ = tf.range(0, batch_size_)
    # batch_idx_ = tf.cast(tf.reshape(batch_idx_, (batch_size_, 1, 1)), dtype=tf.int64)
    # b_ = tf.tile(batch_idx_, (1, height_, width_))
    # indices_ = tf.stack([b_, y_, x_], 3)
    # value_ = tf.gather_nd(img_, indices_)
    # # sess.run(tf.gather_nd(tf.expand_dims(img_[0], 0), tf.expand_dims(indices_[0], 0))), sess.run(value_[0])

    batch_idx = torch.arange(0, batch_size).view((batch_size, 1, 1)).type(torch.int64)
    b = batch_idx.repeat(1, height, width)
    # indices = torch.stack((b, y, x), dim=-1)

    value = img[b.long(), :, y.long(), x.long()]

    return value



# def bilinear_sampler(img, x, y):
#     """
#     Performs bilinear sampling of the input images according to the
#     normalized coordinates provided by the sampling grid. Note that
#     the sampling is done identically for each channel of the input.
#     To test if the function works properly, output image should be
#     identical to input image when theta is initialized to identity
#     transform.
#     Input
#     -----
#     - img: batch of images in (B, H, W, C) layout.
#     - grid: x, y which is the output of affine_grid_generator.
#     Returns
#     -------
#     - interpolated images according to grids. Same size as grid.
#     """
#     # prepare useful params
#     B = tf.shape(img)[0]
#     H = tf.shape(img)[1]
#     W = tf.shape(img)[2]
#     C = tf.shape(img)[3]
#
#     max_y = tf.cast(H - 1, 'int32')
#     max_x = tf.cast(W - 1, 'int32')
#     zero = tf.zeros([], dtype='int32')
#
#     # cast indices as float32 (for rescaling)
#     x = tf.cast(x, 'float32')
#     y = tf.cast(y, 'float32')
#
#     # grab 4 nearest corner points for each (x_i, y_i)
#     # i.e. we need a rectangle around the point of interest
#     x0 = tf.cast(tf.floor(x), 'int32')
#     x1 = x0 + 1
#     y0 = tf.cast(tf.floor(y), 'int32')
#     y1 = y0 + 1
#
#     # clip to range [0, H/W] to not violate img boundaries
#     x0 = tf.clip_by_value(x0, zero, max_x)
#     x1 = tf.clip_by_value(x1, zero, max_x)
#     y0 = tf.clip_by_value(y0, zero, max_y)
#     y1 = tf.clip_by_value(y1, zero, max_y)
#
#     # get pixel value at corner coords
#     Ia = get_pixel_value(img, x0, y0)
#     Ib = get_pixel_value(img, x0, y1)
#     Ic = get_pixel_value(img, x1, y0)
#     Id = get_pixel_value(img, x1, y1)
#
#     # recast as float for delta calculation
#     x0 = tf.cast(x0, 'float32')
#     x1 = tf.cast(x1, 'float32')
#     y0 = tf.cast(y0, 'float32')
#     y1 = tf.cast(y1, 'float32')
#
#     # calculate deltas
#     wa = (x1-x) * (y1-y)
#     wb = (x1-x) * (y-y0)
#     wc = (x-x0) * (y1-y)
#     wd = (x-x0) * (y-y0)
#
#     # add dimension for addition
#     wa = tf.expand_dims(wa, axis=3)
#     wb = tf.expand_dims(wb, axis=3)
#     wc = tf.expand_dims(wc, axis=3)
#     wd = tf.expand_dims(wd, axis=3)
#
#     # compute output
#     out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
#     return out


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
    # prepare useful params
    # B_ = tf.shape(img_)[0]
    # H_ = tf.shape(img_)[1]
    # W_ = tf.shape(img_)[2]
    # C_ = tf.shape(img_)[3]

    # B, H, W, C = img.shape

    B, C, H, W = img.shape
    # max_y_ = tf.cast(H_ - 1, 'int64')
    # max_x_ = tf.cast(W_ - 1, 'int64')
    # zero_ = tf.zeros([], dtype='int64')

    max_y = H - 1
    max_x = W - 1
    zero = 0

    # cast indices as float32 (for rescaling)
    # x_ = tf.cast(x_, 'float64')
    # y_ = tf.cast(y_, 'float64')

    # grab 4 nearest corner points for each (x_i, y_i)
    # i.e. we need a rectangle around the point of interest
    # x0_ = tf.cast(tf.floor(x_), 'int64')
    # x1_ = x0_ + 1
    # y0_ = tf.cast(tf.floor(y_), 'int64')
    # y1_ = y0_ + 1

    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate img boundaries
    # x0_ = tf.clip_by_value(x0_, zero_, max_x_)
    # x1_ = tf.clip_by_value(x1_, zero_, max_x_)
    # y0_ = tf.clip_by_value(y0_, zero_, max_y_)
    # y1_ = tf.clip_by_value(y1_, zero_, max_y_)

    x0 = torch.clamp(x0, zero, max_x)
    x1 = torch.clamp(x1, zero, max_x)
    y0 = torch.clamp(y0, zero, max_y)
    y1 = torch.clamp(y1, zero, max_y)

    # get pixel value at corner coords
    # compare_numpy(x0_, x0)
    # compare_numpy(x1_, x1)
    # compare_numpy(y0_, y0)
    # compare_numpy(y1_, y1)

    # Ia_ = get_pixel_value(img_, x0_, y0_)
    # Ib_ = get_pixel_value(img_, x0_, y1_)
    # Ic_ = get_pixel_value(img_, x1_, y0_)
    # Id_ = get_pixel_value(img_, x1_, y1_)

    Ia = get_pixel_value_torch(img, x0, y0)
    Ib = get_pixel_value_torch(img, x0, y1)
    Ic = get_pixel_value_torch(img, x1, y0)
    Id = get_pixel_value_torch(img, x1, y1)

    # Ia = get_pixel_value_torch(img, x0, y0)
    # Ib = get_pixel_value_torch(img, x0, y1)
    # Ic = get_pixel_value_torch(img, x1, y0)
    # Id = get_pixel_value_torch(img, x1, y1)

    # recast as float for delta calculation
    # x0_ = tf.cast(x0_, 'float64')
    # x1_ = tf.cast(x1_, 'float64')
    # y0_ = tf.cast(y0_, 'float64')
    # y1_ = tf.cast(y1_, 'float64')

    x0 = x0.float()
    x1 = x1.float()
    y0 = y0.float()
    y1 = y1.float()

    # calculate deltas
    # wa_ = (x1_ - x_) * (y1_ - y_)
    # wb_ = (x1_ - x_) * (y_ - y0_)
    # wc_ = (x_ - x0_) * (y1_ - y_)
    # wd_ = (x_ - x0_) * (y_ - y0_)

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    # wa_ = tf.expand_dims(wa_, axis=3)
    # wb_ = tf.expand_dims(wb_, axis=3)
    # wc_ = tf.expand_dims(wc_, axis=3)
    # wd_ = tf.expand_dims(wd_, axis=3)

    wa = torch.unsqueeze(wa, dim=3)
    wb = torch.unsqueeze(wb, dim=3)
    wc = torch.unsqueeze(wc, dim=3)
    wd = torch.unsqueeze(wd, dim=3)

    # compute output
    # out_ = tf.add_n([wa_ * Ia_, wb_ * Ib_, wc_ * Ic_, wd_ * Id_])
    out = wa * Ia + wb * Ib + wc * Ic + wd * Id

    return out


# def generate_shade(il, m, mshape, texture_size = [192, 224], is_with_normal=False):
#     '''
#         빛, projection matric, shape로 texture를 입힐 shading map 렌더링
#
#         Parameters
#             il:             [batch, 27]
#             m:              [batch, 8]
#             mshape:         [batch, VERTEX_NUM * 3]
#             texture_size:   [192, 224]
#             is_with_normal: False
#         Returns
#             image: [batch, 192, 224, 3]
#     '''
#
#     n_size = get_shape(il)  # [batch, 27]
#     n_size = n_size[0]  # batch
#     ################################################################################################################
#     # load 3DMM files
#     tri = load_3DMM_tri()
#     vertex_tri = load_3DMM_vertex_tri()
#     vt2pixel_u, vt2pixel_v = load_3DMM_vt2pixel()
#     tri_2d = load_3DMM_tri_2d()
#     tri_2d_barycoord = load_3DMM_tri_2d_barycoord()
#
#     ################################################################################################################
#     # 삼각형의 각 vertex별로 나눈 것
#     tri_const     = tf.constant(tri, tf.int32)  # [3, TRI_NUM + 1]
#     tri2vt1_const = tf.constant(tri[0, :], tf.int32)  # [TRI_NUM + 1, ]
#     tri2vt2_const = tf.constant(tri[1, :], tf.int32)  # [TRI_NUM + 1, ]
#     tri2vt3_const = tf.constant(tri[2, :], tf.int32)  # [TRI_NUM + 1, ]
#
#     ################################################################################################################
#     # 텐서로...
#     vertex_tri_const    = tf.constant(vertex_tri, tf.int32)    # [8, VERTEX_NUM]
#     tri_2d_const        = tf.constant(tri_2d, tf.int32)    # [192, 224]
#     tri_2d_const_flat   = tf.reshape(tri_2d_const, shape=[-1,1])  # [192 * 224, ]
#
#     # 2d 이미지 픽셀에 해당하는 vertex들을 추출(vertex to pixel)
#     # 픽셀에 박힐 vertex만 추출한다는 거임!
#     vt1 = tf.gather( tri2vt1_const,  tri_2d_const_flat )    # [192 * 224, 1]
#     vt2 = tf.gather( tri2vt2_const,  tri_2d_const_flat )    # [192 * 224, 1]
#     vt3 = tf.gather( tri2vt3_const,  tri_2d_const_flat )    # [192 * 224, 1]
#
#     # 최종적으로 normal vector에 곱해져서 색을 정할 때 사용되는 상수    vt1_coeff = tf.reshape(tf.constant(tri_2d_barycoord[:,:,0], tf.float32), shape=[-1,1])  # [192 * 224, 1]
#     vt1_coeff = tf.reshape(tf.constant(tri_2d_barycoord[:, :, 0], tf.float32), shape=[-1, 1])  # [192 * 224, 1]
#     vt2_coeff = tf.reshape(tf.constant(tri_2d_barycoord[:, :, 1], tf.float32), shape=[-1, 1])  # [192 * 224, 1]
#     vt3_coeff = tf.reshape(tf.constant(tri_2d_barycoord[:, :, 2], tf.float32), shape=[-1, 1])  # [192 * 224, 1]
#
#     # 단순히 batch 단위로 나눈 것수
#     m_single     = tf.split(axis = 0, num_or_size_splits = n_size, value = m)   # [1, 8] x batch
#     shape_single = tf.split(axis = 0, num_or_size_splits = n_size, value = mshape)  # [1, VERTEX_NUM * 3] x batch
#
#     # normal vector들을 추출해보자
#     normalf_flats = []
#     for i in range(n_size):
#         # m은 projection parameter로 2개의 rotation 벡터를 포함
#         m_i = tf.transpose(tf.reshape(m_single[i], [4,2]))  # [4, 2]
#         # local space의 벡터
#         m_i_row1 = tf.nn.l2_normalize(m_i[0,0:3], dim = 0)  # [3, ]
#         # world space의 벡터
#         m_i_row2 = tf.nn.l2_normalize(m_i[1,0:3], dim = 0)  # [3, ]
#         # local space와 world space 모두 수직인 cross product 벡터
#         m_i_row3 = tf.cross(m_i_row1, m_i_row2) # [3, ]
#         # ??????????????????????????????????????????????????? 이게 뭘까??
#         m_i = tf.concat([ tf.expand_dims(m_i_row1, 0), tf.expand_dims(m_i_row2, 0), tf.expand_dims(m_i_row3, 0)], axis = 0) # [3, 3]
#
#         # 1개의 shape에 대하여 진행할거임
#         vertex3d_rs = tf.transpose(tf.reshape( shape_single[i], shape = [-1, 3] ))   # [3, VERTEX_NUM]
#
#         # vertex마다의 normal vector, triangle마다의 normal vector를 return
#         normal, normalf = _DEPRECATED_compute_normal(vertex3d_rs, tri_const, vertex_tri_const)  # [VERTEX, 3], [TRI_NUM, 3]
#
#         # normal vector를 projection에 맞게 rotation 시킴
#         normal = tf.transpose(normal)   # [3, VERTEX_NUM]
#         rotated_normal = tf.matmul(m_i, normal, False, False)   # [3, VERTEX_NUM]
#         rotated_normal = tf.transpose(rotated_normal)   # [VERTEX_NUM, 3]
#
#         # vertex에 해당하는 pixel 값들을 추출
#         normal_flat_vt1 = tf.gather_nd(rotated_normal, vt1) # [192 x 224, 3]
#         normal_flat_vt2 = tf.gather_nd(rotated_normal, vt2) # [192 x 224, 3]
#         normal_flat_vt3 = tf.gather_nd(rotated_normal, vt3) # [192 x 224, 3]
#
#         # barycentric coordinate로 변경
#         normalf_flat = normal_flat_vt1 * vt1_coeff + normal_flat_vt2 * vt2_coeff + normal_flat_vt3 * vt3_coeff    # [192 x 224, 3]
#         normalf_flats.append(normalf_flat)
#
#     # batch들을 합친 것
#     normalf_flats = tf.stack(normalf_flats) # [batch, 192 x 224, 3]
#
#     # 명암조절 map
#     shade = shading(il, normalf_flats)  # [batch, 192 x 224, 3]
#
#     if is_with_normal:
#         return tf.reshape(shade, shape = [-1, texture_size[0], texture_size[1], 3]), tf.reshape(normalf_flats, shape = [-1, texture_size[0], texture_size[1], 3]),
#
#     return tf.reshape(shade, shape = [-1, texture_size[0], texture_size[1], 3]) # [batch, 192, 224, 3]


def generate_shade_torch(il, m, mshape, texture_size = [192, 224], is_with_normal=False):
    '''
        빛, projection matric, shape로 texture를 입힐 shading map 렌더링

        Parameters
            il:             [batch, 27]
            m:              [batch, 8]
            mshape:         [batch, VERTEX_NUM * 3]
            texture_size:   [192, 224]
            is_with_normal: False
        Returns
            image: [batch, 192, 224, 3]
    '''

    BATCH_SIZE = il.shape[0]

    # load 3DMM files
    tri = torch.from_numpy(load_3DMM_tri())  # [3, TRI_NUM + 1]
    vertex_tri = torch.from_numpy(load_3DMM_vertex_tri())  # [8, VERTEX_NUM]
    vt2pixel_u, vt2pixel_v = [torch.from_numpy(vt2pixel) for vt2pixel in load_3DMM_vt2pixel()]  # [VERTEX_NUM + 1, ], [VERTEX_NUM + 1, ]
    tri_2d = torch.from_numpy(load_3DMM_tri_2d())  # [192, 224] (Fragment shader)
    tri_2d_barycoord = torch.from_numpy(load_3DMM_tri_2d_barycoord()).cuda()  # [192, 224, 3]

    # 삼각형의 각 vertex별로 나눈 것
    tri2vt1 = tri[0, :]
    tri2vt2 = tri[1, :]
    tri2vt3 = tri[2, :]

    # 텐서로...
    tri_2d_flat = tri_2d.view((1, -1))
    tri_2d_flat_concat = tri_2d_flat.repeat(3, 1)

    # 2d 이미지 픽셀에 해당하는 vertex들을 추출(vertex to pixel)
    # 픽셀에 박힐 vertex만 추출한다는 거임!
    vt = torch.gather(tri, 1, tri_2d_flat_concat.long()).cuda()

    # 최종적으로 normal vector에 곱해져서 색을 정할 때 사용되는 상수    vt1_coeff = tf.reshape(tf.constant(tri_2d_barycoord[:,:,0], tf.float32), shape=[-1,1])  # [192 * 224, 1]
    vt1_coeff = tri_2d_barycoord[:, :, 0].view((-1, 1))
    vt2_coeff = tri_2d_barycoord[:, :, 1].view((-1, 1))
    vt3_coeff = tri_2d_barycoord[:, :, 2].view((-1, 1))

    # 단순히 batch 단위로 나눈 것수
    m_i = m.view((BATCH_SIZE, 4, 2))[:, 0:3, :]
    # l2_sum = torch.norm(m_i, dim=(1), keepdim=True)
    # m_i_row = torch.div(m_i, l2_sum)
    m_i_row = F.normalize(m_i, dim=1)
    m_cross = torch.cross(m_i_row[:, :, 0], m_i_row[:, :, 1])
    m_cross = torch.unsqueeze(m_cross, -1)
    m_i = torch.cat((m_i_row, m_cross), dim=2)
    m_i = torch.transpose(m_i, 1, 2)

    vertex3d_rs = mshape.view((BATCH_SIZE, -1, 3))

    normal, normalf = compute_normal_torch(vertex3d_rs, tri, vertex_tri)

    normal = torch.transpose(normal, 1, 2)
    rotated_normal = torch.matmul(m_i, normal)
    rotated_normal = torch.transpose(rotated_normal, 1, 2)

    vt = torch.unsqueeze(vt, 0)
    vt = torch.unsqueeze(vt, -1)
    vt = torch.cat(BATCH_SIZE * [vt], 0)
    vt = torch.cat(3 * [vt], -1)
    vt1 = vt[:, 0, :]
    vt2 = vt[:, 1, :]
    vt3 = vt[:, 2, :]
    zeros = torch.zeros((BATCH_SIZE, 1, 3)).cuda()

    rotated_normal = torch.cat((rotated_normal, zeros), dim=1)
    normal_flat_vt1 = torch.gather(rotated_normal, 1, vt1)
    normal_flat_vt2 = torch.gather(rotated_normal, 1, vt2)
    normal_flat_vt3 = torch.gather(rotated_normal, 1, vt3)

    normalf_flat = normal_flat_vt1 * vt1_coeff + normal_flat_vt2 * vt2_coeff + normal_flat_vt3 * vt3_coeff


    # 명암조절 map
    shade = shading_torch(il, normalf_flat)
    shade = shade.view((-1, texture_size[0], texture_size[1], 3)) # [batch, 192, 224, 3]
    normalf_flat = normalf_flat.view((-1, texture_size[0], texture_size[1], 3))

    if is_with_normal:
        return shade, normalf_flat

    return shade.permute((0, 3, 1, 2))


# def shading(L, normal):
#     '''
#         빛과 triangle 각도에 따른 명암 계산
#
#         Parameters
#             L:      [batch, 27]
#             normal: [batch, 192 * 224, 3]
#         Returns
#             normal_map: [batch, 192, 224, 3]
#     '''
#
#     shape = normal.get_shape().as_list() # [batch, 192 * 224, 3]
#
#     normal_x, normal_y, normal_z = tf.split(tf.expand_dims(normal, -1), axis=2, num_or_size_splits=3) # [batch, 192 *224, 1, 1] x XYZ_space
#     pi = math.pi
#
#     # ㅁㄴㅇㄹ 몰라ㅏㅏ
#     sh    = [0]*9
#     sh[0] = 1/math.sqrt(4*pi) * tf.ones_like(normal_x) #
#     sh[1] = ((2*pi)/3)*(math.sqrt(3/(4*pi)))* normal_z
#     sh[2] = ((2*pi)/3)*(math.sqrt(3/(4*pi)))* normal_y
#     sh[3] = ((2*pi)/3)*(math.sqrt(3/(4*pi)))* normal_x
#     sh[4] = (pi/4)*(1/2)*(math.sqrt(5/(4*pi)))*(2*tf.square(normal_z)-tf.square(normal_x)-tf.square(normal_y))
#     sh[5] = (pi/4)*(3)  *(math.sqrt(5/(12*pi)))*(normal_y*normal_z)
#     sh[6] = (pi/4)*(3)  *(math.sqrt(5/(12*pi)))*(normal_x*normal_z)
#     sh[7] = (pi/4)*(3)  *(math.sqrt(5/(12*pi)))*(normal_x*normal_y)
#     sh[8] = (pi/4)*(3/2)*(math.sqrt(5/(12*pi)))*( tf.square(normal_x)-tf.square(normal_y))
#
#     sh = tf.concat(sh, axis=3) # [batch, 192 * 224, 1, 9]
#     print('sh.get_shape()')
#     print(sh.get_shape())
#
#     L1, L2, L3 = tf.split(L, num_or_size_splits = 3, axis=1)
#     L1 = tf.expand_dims(L1, 1)
#     L1 = tf.tile(L1, multiples=[1, shape[1], 1] )
#     L1 = tf.expand_dims(L1, -1)
#
#     L2 = tf.expand_dims(L2, 1)
#     L2 = tf.tile(L2, multiples=[1, shape[1], 1] )
#     L2 = tf.expand_dims(L2, -1)
#
#     L3 = tf.expand_dims(L3, 1)
#     L3 = tf.tile(L3, multiples=[1, shape[1], 1] )
#     L3 = tf.expand_dims(L3, -1)
#
#     print('L1.get_shape()')
#     print(L1.get_shape())
#
#     B1 = tf.matmul(sh, L1)
#     B2 = tf.matmul(sh, L2)
#     B3 = tf.matmul(sh, L3)
#
#     B = tf.squeeze(tf.concat([B1, B2, B3], axis = 2))
#
#     return B


def shading_torch ( L, normal):
    '''
		빛과 triangle 각도에 따른 명암 계산

		Parameters
			L:      [batch, 27]
			normal: [batch, 192 * 224, 3]
		Returns
			normal_map: [batch, 192, 224, 3]
	'''
    shape = normal.shape
    BATCH_SIZE = shape[0]

    normal = torch.unsqueeze(normal, -1)
    normal_x = normal[:, :, 0]
    normal_y = normal[:, :, 1]
    normal_z = normal[:, :, 2]

    pi = math.pi
    sh = torch.zeros(BATCH_SIZE, shape[1], 1, 9).cuda()
    sh[:, :, :, 0] = 1 / math.sqrt(4 * pi) * torch.ones_like(normal_x)  #
    sh[:, :, :, 1] = ((2 * pi) / 3) * (math.sqrt(3 / (4 * pi))) * normal_z
    sh[:, :, :, 2] = ((2 * pi) / 3) * (math.sqrt(3 / (4 * pi))) * normal_y
    sh[:, :, :, 3] = ((2 * pi) / 3) * (math.sqrt(3 / (4 * pi))) * normal_x
    sh[:, :, :, 4] = (pi / 4) * (1 / 2) * (math.sqrt(5 / (4 * pi))) * (2 * torch.pow(normal_z, 2) - torch.pow(normal_x, 2) - torch.pow(normal_y, 2))
    sh[:, :, :, 5] = (pi / 4) * (3) * (math.sqrt(5 / (12 * pi))) * (normal_y * normal_z)
    sh[:, :, :, 6] = (pi / 4) * (3) * (math.sqrt(5 / (12 * pi))) * (normal_x * normal_z)
    sh[:, :, :, 7] = (pi / 4) * (3) * (math.sqrt(5 / (12 * pi))) * (normal_x * normal_y)
    sh[:, :, :, 8] = (pi / 4) * (3 / 2) * (math.sqrt(5 / (12 * pi))) * (torch.pow(normal_x, 2) - torch.pow(normal_y, 2))

    L = torch.unsqueeze(L, 1)
    L = L.repeat(1, shape[1], 1)
    L = torch.unsqueeze(L, -1)

    L1, L2, L3 = torch.split(L, (9, 9, 9), dim=2)

    B1 = torch.matmul(sh, L1)
    B2 = torch.matmul(sh, L2)
    B3 = torch.matmul(sh, L3)
    B = torch.cat((B1, B2, B3), dim=-1)
    B = torch.squeeze(B, dim=2)

    return B



## _DEPRECATED_

# def _DEPRECATED_warp_texture(texture, m, mshape, output_size=224):
#     def flatten(x):
#         return tf.reshape(x, [-1])
#
#     n_size = get_shape(texture)
#     n_size = n_size[0]
#
#     s = output_size
#
#     # Tri, tri2vt
#     tri = load_3DMM_tri()
#     vertex_tri = load_3DMM_vertex_tri()
#     vt2pixel_u, vt2pixel_v = load_3DMM_vt2pixel()
#
#     tri2vt1_const = tf.constant(tri[0,:], tf.int32)
#     tri2vt2_const = tf.constant(tri[1,:], tf.int32)
#     tri2vt3_const = tf.constant(tri[2,:], tf.int32)
#
#     tri_const = tf.constant(tri, tf.int32)
#     vertex_tri_const = tf.constant(vertex_tri, tf.int32)
#
#
#     #Vt2pix
#     vt2pixel_u_const = tf.constant(vt2pixel_u, tf.float32)
#     vt2pixel_v_const = tf.constant(vt2pixel_v, tf.float32)
#
#
#     m_single     = tf.split(axis = 0, num_or_size_splits = n_size, value = m)
#     shape_single = tf.split(axis = 0, num_or_size_splits = n_size, value = mshape)
#
#     pixel_u = []
#     pixel_v = []
#
#     masks = []
#
#     u, v = tf.meshgrid( tf.linspace(0.0, output_size-1.0, output_size), tf.linspace(0.0, output_size-1.0, output_size))
#     u = flatten(u)
#     v = flatten(v)
#
#     for i in range(n_size):
#
#         m_i = tf.transpose(tf.reshape(m_single[i], [4,2]))
#         m_i_row1 = tf.nn.l2_normalize(m_i[0,0:3], dim = 0)
#         m_i_row2 = tf.nn.l2_normalize(m_i[1,0:3], dim = 0)
#         m_i_row3 = tf.concat([tf.reshape(tf.cross(m_i_row1, m_i_row2), shape = [1, 3]), tf.zeros([1, 1])], axis = 1)
#
#         m_i = tf.concat([m_i, m_i_row3], axis = 0)
#
#         # Compute 2d vertex
#         vertex3d_rs = tf.transpose(tf.reshape( shape_single[i], shape = [-1, 3] ))
#
#         normal, normalf = _DEPRECATED_compute_normal(vertex3d_rs,tri_const, vertex_tri_const)
#         normalf = tf.transpose(normalf)
#         normalf4d = tf.concat(axis=0, values=[normalf, tf.ones([1, normalf.get_shape()[-1]], tf.float32)])
#         rotated_normalf = tf.matmul(m_i, normalf4d, False, False)
#         _, _, rotated_normalf_z = tf.split(axis=0, num_or_size_splits=3, value=rotated_normalf)
#         visible_tri = tf.greater(rotated_normalf_z, 0)
#
#
#         vertex4d = tf.concat(axis = 0, values = [vertex3d_rs, tf.ones([1, vertex3d_rs.get_shape()[-1]], tf.float32)])
#
#         vertex2d = tf.matmul(m_i, vertex4d, False, False)
#         vertex2d = tf.transpose(vertex2d)
#
#         [vertex2d_u, vertex2d_v, vertex2d_z]   = tf.split(axis=1, num_or_size_splits=3, value=vertex2d)
#         vertex2d_u = vertex2d_u - 1
#         vertex2d_v = s - vertex2d_v
#
#         vertex2d = tf.concat(axis=1, values=[vertex2d_v, vertex2d_u, vertex2d_z])
#         vertex2d = tf.transpose(vertex2d)
#
#         # Applying Z-buffer
#         tri_map_2d, mask_i = ZBuffer_Rendering_CUDA_op_v2_sz224(vertex2d, tri_const, visible_tri)
#
#         tri_map_2d_flat = tf.cast(tf.reshape(tri_map_2d, [-1]), 'int32')
#
#
#         # Calculate barycentric coefficient
#
#         vt1 = tf.gather( tri2vt1_const,  tri_map_2d_flat )
#         vt2 = tf.gather( tri2vt2_const,  tri_map_2d_flat )
#         vt3 = tf.gather( tri2vt3_const,  tri_map_2d_flat )
#
#
#         pixel1_uu = flatten(tf.gather( vertex2d_u,  vt1 ))
#         pixel2_uu = flatten(tf.gather( vertex2d_u,  vt2 ))
#         pixel3_uu = flatten(tf.gather( vertex2d_u,  vt3 ))
#
#         pixel1_vv = flatten(tf.gather( vertex2d_v,  vt1 ))
#         pixel2_vv = flatten(tf.gather( vertex2d_v,  vt2 ))
#         pixel3_vv = flatten(tf.gather( vertex2d_v,  vt3 ))
#         c1, c2, c3 = barycentric(pixel1_uu, pixel2_uu, pixel3_uu, pixel1_vv, pixel2_vv, pixel3_vv, u, v)
#
#
#         ##
#         pixel1_u = tf.gather( vt2pixel_u_const,  vt1 )
#         pixel2_u = tf.gather( vt2pixel_u_const,  vt2 )
#         pixel3_u = tf.gather( vt2pixel_u_const,  vt3 )
#
#         pixel1_v = tf.gather( vt2pixel_v_const,  vt1 )
#         pixel2_v = tf.gather( vt2pixel_v_const,  vt2 )
#         pixel3_v = tf.gather( vt2pixel_v_const,  vt3 )
#
#
#         pixel_u_i = tf.reshape(pixel1_u * c1 + pixel2_u * c2 + pixel3_u* c3, [output_size, output_size])
#         pixel_v_i = tf.reshape(pixel1_v * c1 + pixel2_v * c2 + pixel3_v* c3, [output_size, output_size])
#
#
#         pixel_u.append(pixel_u_i)
#         pixel_v.append(pixel_v_i)
#
#         masks.append(mask_i)
#
#     images = bilinear_sampler(texture, pixel_v, pixel_u)
#     masks = tf.stack(masks)
#
#     return images, masks
#
# def _DEPRECATED_compute_landmarks(m, mshape, output_size=224):
#     # This is a deprecated version of compute landmarks which is not optimized
#
#     n_size = get_shape(m)
#     n_size = n_size[0]
#
#     s = output_size
#
#
#     # Tri, tri2vt
#     kpts = load_3DMM_kpts()
#
#     kpts_const = tf.constant(kpts, tf.int32)
#
#     m_single     = tf.split(axis = 0, num_or_size_splits = n_size, value = m)
#     shape_single = tf.split(axis = 0, num_or_size_splits = n_size, value = mshape)
#
#     landmarks_u = []
#     landmarks_v = []
#
#     for i in range(n_size):
#         # Compute 2d vertex
#         #vertex3d = tf.transpose(tf.reshape( mu_const + tf.matmul(w_shape_const, p_shape_single[i], False, True) + tf.matmul(w_exp_const, p_exp_single[i], False, True), shape = [-1, 3] ))
#
#         vertex3d_rs = tf.reshape( shape_single[i], shape = [-1, 3] )
#         vertex3d_rs = tf.transpose(tf.gather_nd(vertex3d_rs, kpts_const))
#         #print(get_shape(vertex3d_rs))
#         vertex4d = tf.concat(axis = 0, values = [vertex3d_rs, tf.ones([1, get_shape(vertex3d_rs)[1]], tf.float32)])
#
#         m_single_i = tf.transpose(tf.reshape(m_single[i], [4,2]))
#         vertex2d = tf.matmul(m_single_i, vertex4d, False, False)
#         vertex2d = tf.transpose(vertex2d)
#
#         [vertex2d_u, vertex2d_v]   = tf.split(axis=1, num_or_size_splits=2, value=vertex2d) #[vertex2d_u, vertex2d_v]   = tf.split(1, 2, vertex2d)
#         vertex2d_u = vertex2d_u - 1
#         vertex2d_v = s - vertex2d_v
#
#         landmarks_u.append(vertex2d_u)
#         landmarks_v.append(vertex2d_v)
#
#     return tf.stack(landmarks_u), tf.stack(landmarks_v)
#
#
# def _DEPRECATED_compute_normal ( vertex, tri, vertex_tri ):
#     # Unit normals to the faces
#     # vertex : 3xvertex_num
#     # tri : 3xtri_num
#
#     vertex = tf.transpose(vertex)
#
#     vt1_indices, vt2_indices, vt3_indices = tf.split(tf.transpose(tri), num_or_size_splits=3, axis=1)
#
#     vt1 = tf.gather_nd(vertex, vt1_indices)
#     # print('get_shape(vt1)')
#     # print(get_shape(vt1))
#     vt2 = tf.gather_nd(vertex, vt2_indices)
#     vt3 = tf.gather_nd(vertex, vt3_indices)
#
#     normalf = tf.cross(vt2 - vt1, vt3 - vt1)
#     normalf = tf.nn.l2_normalize(normalf, dim=1)
#
#     mask = tf.tile(tf.expand_dims(tf.not_equal(vertex_tri, tri.shape[1] - 1), 2), multiples=[1, 1, 3])
#     mask = tf.cast(mask, vertex.dtype)
#     vertex_tri = tf.reshape(vertex_tri, shape=[-1, 1])
#     normal = tf.reshape(tf.gather_nd(normalf, vertex_tri), shape=[8, -1, 3])
#
#     normal = tf.reduce_sum(tf.multiply(normal, mask), axis=0)
#     normal = tf.nn.l2_normalize(normal, dim=1)
#
#     # print('get_shape(normalf)')
#     # print(get_shape(normalf))
#
#     # print('get_shape(normal)')
#     # print(get_shape(normal))
#
#     # enforce that the normal are outward
#     v = vertex - tf.reduce_mean(vertex, 0)
#     s = tf.reduce_sum(tf.multiply(v, normal), 0)
#
#     count_s_greater_0 = tf.count_nonzero(tf.greater(s, 0))
#     count_s_less_0 = tf.count_nonzero(tf.less(s, 0))
#
#     sign = 2 * tf.cast(tf.greater(count_s_greater_0, count_s_less_0), tf.float32) - 1
#     normal = tf.multiply(normal, sign)
#     normalf = tf.multiply(normalf, sign)
#
#     return normal, normalf
#
# def _DEPRECATED_compute_normal_torch(vertex, tri, vertex_tri):
#     BATCH_SIZE = vertex.shape[0]
#
#
#     tri = torch.transpose(tri, 0, 1)
#     tri = torch.unsqueeze(tri, 0)
#     tri = torch.unsqueeze(tri, -1)
#     tri = torch.cat(BATCH_SIZE * [tri])[:, :-1, :]
#
#     vt1_indices = torch.cat(3 * [tri[:, :, 0]], -1)
#     vt2_indices = torch.cat(3 * [tri[:, :, 1]], -1)
#     vt3_indices = torch.cat(3 * [tri[:, :, 2]], -1)
#     vt1 = torch.gather(vertex, 1, vt1_indices)
#     vt2 = torch.gather(vertex, 1, vt2_indices)
#     vt3 = torch.gather(vertex, 1, vt3_indices)
#     zeros = torch.zeros((BATCH_SIZE, 1, 3), dtype=torch.double)
#     vt1_padded = torch.cat((vt1, zeros), 1)
#     vt2_padded = torch.cat((vt2, zeros), 1)
#     vt3_padded = torch.cat((vt3, zeros), 1)
#
#     normalf = torch.cross(vt2 - vt1, vt3 - vt1)
#     norm_sum = torch.norm(normalf, dim=2, keepdim=True)
#     normalf = torch.div(normalf, norm_sum)
#     normalf = torch.cat((normalf, zeros), 1)
#
#     equal = vertex_tri != (tri.shape[1])
#     expand = torch.unsqueeze(equal, 2)
#
#     mask = expand.repeat(1, 1, 3)
#
#     vertex_tri = vertex_tri.view((-1, 1))
#
#
#
#     normal_indices = torch.unsqueeze(vertex_tri, 0)
#     normal_indices = torch.cat(BATCH_SIZE * [normal_indices])
#     normal_indices = torch.cat(3 * [normal_indices], -1)
#     normal = torch.gather(normalf, 1, normal_indices.long())
#
#
#     normal = normal.view((BATCH_SIZE, 8, -1, 3))
#     multi = torch.mul(normal, mask)
#     normal = torch.sum(multi, dim=1)
#
#     norm_sum = torch.norm(normal, dim=2, keepdim=True)
#     normal = torch.div(normal, norm_sum)
#
#
#
#     mean = torch.mean(vertex, 1)
#     mean = torch.unsqueeze(mean, 1)
#     v = vertex - mean
#     s = torch.sum(torch.mul(v, normal), 1)
#
#     count_s_greater_0 = torch.sum(1 * torch.gt(s, 0), 1)
#     count_s_less_0 = torch.sum(1 * torch.lt(s, 0), 1)
#
#     sign = 2 * torch.gt(count_s_greater_0, count_s_less_0) - 1
#     sign = torch.unsqueeze(sign, 1)
#     sign = torch.unsqueeze(sign, -1)
#     normal = torch.mul(normal, sign)
#     normalf = torch.mul(normalf, sign)
#
#
#     return normal, normalf



'''
def unwarp_texture(image, m, mshape, output_size=224, is_reduce = False):
    #TO Do: correct the mask
    print("TODO: correct the mask in unwarp_texture(image, m, mshape, output_size=124, is_reduce = False)")


    n_size = get_shape(image)
    n_size = n_size[0]
    s = output_size   

     # Tri, tri2vt
    tri = load_3DMM_tri()
    vertex_tri = load_3DMM_vertex_tri()
    vt2pixel_u, vt2pixel_v = load_3DMM_vt2pixel()

        
    tri2vt1_const = tf.constant(tri[0,:], tf.int32)
    tri2vt2_const = tf.constant(tri[1,:], tf.int32)
    tri2vt3_const = tf.constant(tri[2,:], tf.int32)

    tri_const = tf.constant(tri, tf.int32)
    #tri_2d_const = tf.constant(tri_2d, tf.int32)
    vertex_tri_const = tf.constant(vertex_tri, tf.int32)


    #Vt2pix
    vt2pixel_u_const = tf.constant(vt2pixel_u, tf.float32)
    vt2pixel_v_const = tf.constant(vt2pixel_v, tf.float32)

    #indicies = np.zeros([s*s,2])
    #for i in range(s):
    #    for j in range(s):
    #        indicies[i*s+j ] = [i,j]

    #indicies_const = tf.constant(indicies, tf.float32)
    #[indicies_const_u, indicies_const_v] = tf.split(1, 2, indicies_const)

    ###########m = m * tf.constant(self.std_m) + tf.constant(self.mean_m)
    ###########mshape = mshape * tf.constant(self.std_shape) + tf.constant(self.mean_shape)

    m_single     = tf.split(axis = 0, num_or_size_splits = n_size, value = m)
    shape_single = tf.split(axis = 0, num_or_size_splits = n_size, value = mshape)
    
    pixel_u = []
    pixel_v = []

    masks = []
    for i in range(n_size):

        m_i = tf.transpose(tf.reshape(m_single[i], [4,2]))
        m_i_row1 = tf.nn.l2_normalize(m_i[0,0:3], dim = 0)
        m_i_row2 = tf.nn.l2_normalize(m_i[1,0:3], dim = 0)
        m_i_row3 = tf.concat([tf.reshape(tf.cross(m_i_row1, m_i_row2), shape = [1, 3]), tf.zeros([1, 1])], axis = 1)
                  
        m_i = tf.concat([m_i, m_i_row3], axis = 0)

        # Compute 2d vertex
        #vertex3d = tf.transpose(tf.reshape( mu_const + tf.matmul(w_shape_const, p_shape_single[i], False, True) + tf.matmul(w_exp_const, p_exp_single[i], False, True), shape = [-1, 3] ))

        vertex3d_rs = tf.transpose(tf.reshape( shape_single[i], shape = [-1, 3] ))

        normal, normalf = compute_normal(vertex3d_rs,tri_const, vertex_tri_const)
        normalf = tf.transpose(normalf)
        normalf4d = tf.concat(axis=0, values=[normalf, tf.ones([1, normalf.get_shape()[-1]], tf.float32)])
        rotated_normalf = tf.matmul(m_i, normalf4d, False, False)
        _, _, rotated_normalf_z = tf.split(axis=0, num_or_size_splits=3, value=rotated_normalf)
        visible_tri = tf.greater(rotated_normalf_z, 0)

        mask_i = tf.gather( tf.cast(visible_tri, dtype=tf.float32),  tri_2d_const ) 
        #print("get_shape(mask_i)")
        #print(get_shape(mask_i))


        vertex4d = tf.concat(axis = 0, values = [vertex3d_rs, tf.ones([1, vertex3d_rs.get_shape()[-1]], tf.float32)])
        
        vertex2d = tf.matmul(m_i, vertex4d, False, False)
        vertex2d = tf.transpose(vertex2d)
        
        [vertex2d_u, vertex2d_v, vertex2d_z]   = tf.split(axis=1, num_or_size_splits=3, value=vertex2d)
        vertex2d_u = tf.squeeze(vertex2d_u - 1)
        vertex2d_v = tf.squeeze(s - vertex2d_v)

        #vertex2d = tf.concat(axis=1, values=[vertex2d_v, vertex2d_u, vertex2d_z])
        #vertex2d = tf.transpose(vertex2d)

        #vertex2d_u = tf.transpose(vertex2d_u)
        #vertex2d_V = tf.transpose(vertex2d_v)



        vt1 = tf.gather( tri2vt1_const,  tri_2d_const ) 
        vt2 = tf.gather( tri2vt2_const,  tri_2d_const ) 
        vt3 = tf.gather( tri2vt3_const,  tri_2d_const )

        


        pixel1_u = tf.gather( vertex2d_u,  vt1 ) #tf.gather( vt2pixel_u_const,  vt1 ) 
        pixel2_u = tf.gather( vertex2d_u,  vt2 ) 
        pixel3_u = tf.gather( vertex2d_u,  vt3 )

        pixel1_v = tf.gather( vertex2d_v,  vt1 ) 
        pixel2_v = tf.gather( vertex2d_v,  vt2 ) 
        pixel3_v = tf.gather( vertex2d_v,  vt3 )

        pixel_u_i = tf.scalar_mul(scalar = 1.0/3.0, x = tf.add_n([pixel1_u, pixel2_u, pixel3_u]))
        pixel_v_i = tf.scalar_mul(scalar = 1.0/3.0, x = tf.add_n([pixel1_v, pixel2_v, pixel3_v]))

        pixel_u.append(pixel_u_i)
        pixel_v.append(pixel_v_i)

        masks.append(mask_i)


        
    texture = bilinear_sampler(image, pixel_u, pixel_v)
    masks = tf.stack(masks)

    return texture, masks
'''






def main():
    pass




if __name__ == '__main__':
    main()



