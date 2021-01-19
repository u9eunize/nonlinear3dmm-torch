import os
import torch
import torch.nn.functional as F
import ZBuffer_cuda

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import math
import utils
from settings import CFG

#
# def generate_full(vec, std, mean):
#     return vec * std + mean


def render_all(lv_trans, lv_rot, lv_il, albedo, shape_1d,
               input_mask=None, input_background=None, post_fix="",
               using_expression=CFG.using_expression,
               using_albedo_as_tex=CFG.using_albedo_as_tex):
    shape_full = generate_full(shape_1d, "shape")

    shape_final = shape_full

    shade = generate_shade(lv_il, m_full, shape_final)
    tex = albedo if using_albedo_as_tex else generate_texture(albedo, shade)

    u, v, mask = warping_flow(m_full, shape_final)
    mask = mask.unsqueeze(1)
    gen_img = rendering_wflow(tex, u, v)

    masked_img_dict = dict()
    if input_mask is not None:
        mask = mask * input_mask
        masked_img_dict[f"g_img_bg{post_fix}"] = apply_mask(gen_img, mask, input_background)

    return {
        f"shade{post_fix}": shade,
        f"tex{post_fix}": tex,
        f"u{post_fix}": u,
        f"v{post_fix}": v,
        f"g_img_mask{post_fix}": mask,
        f"g_img{post_fix}": gen_img,
        **masked_img_dict
    }


def render_mix(albedo_base, shade_base, albedo_comb, shade_comb,
               u_base, v_base, u_comb, v_comb, mask_base=None, mask_comb=None,
               input_mask=None, input_background=None,
               using_albedo_as_tex=CFG.using_albedo_as_tex):
    if using_albedo_as_tex:
        tex_mix_ac_sb = albedo_comb
        tex_mix_ab_sc = albedo_base
    else:
        tex_mix_ac_sb = generate_texture(albedo_comb, shade_base)
        tex_mix_ab_sc = generate_texture(albedo_base, shade_comb)

    gen_img_ac_sb = rendering_wflow(tex_mix_ac_sb, u_base, v_base)
    gen_img_ab_sc = rendering_wflow(tex_mix_ab_sc, u_comb, v_comb)

    masked_img_dict = dict()
    if mask_base is not None and mask_comb is not None:
        gen_img_bg_ac_sb = apply_mask(gen_img_ac_sb, mask_base * input_mask, input_background)
        gen_img_bg_ab_sc = apply_mask(gen_img_ab_sc, mask_comb * input_mask, input_background)
        masked_img_dict = {
            "g_img_bg_ac_sb": gen_img_bg_ac_sb,
            "g_img_bg_ab_sc": gen_img_bg_ab_sc,
        }

    return {
        "tex_mix_ac_sb": tex_mix_ac_sb,
        "tex_mix_ab_sc": tex_mix_ab_sc,
        "g_img_ac_sb": gen_img_ac_sb,
        "g_img_ab_sc": gen_img_ab_sc,
        **masked_img_dict
    }


def generate_full(vec, kind="shape"):
    assert kind in ["shape", "m", "exp"]
    std = getattr(CFG, f"std_{kind}")
    mean = getattr(CFG, f"mean_{kind}")

    return vec * std + mean


def minmax(vec):
    return torch.min(vec), torch.max(vec)


def generate_texture(albedo, shade, is_clamp=False):
    tex = 2.0 * ((albedo + 1.0) / 2.0 * shade) - 1.0

    if is_clamp:
        tex = torch.clamp(tex, 0, 1)
    return tex


def generate_tex_mask(input_texture_labels, input_texture_masks):
    batch_size = input_texture_labels.shape[0]
    tex_vis_mask = (~input_texture_labels.eq((torch.ones_like(input_texture_labels) * -1))).float()
    tex_vis_mask = tex_vis_mask * input_texture_masks
    tex_ratio = torch.sum(tex_vis_mask) / (batch_size * CFG.texture_size[0] * CFG.texture_size[1] * CFG.c_dim)
    return {
        "tex_vis_mask": tex_vis_mask,
        "tex_ratio": tex_ratio,
    }


def renderer(m_full, tex, shape_full, inputs, postfix=""):
    input_masks = inputs["input_masks"]
    input_images = inputs["input_images"]

    g_images_raw, g_images_mask_raw = warp_texture_torch(tex, m_full, shape_full)

    g_images_mask = input_masks * g_images_mask_raw.unsqueeze(1).repeat(1, 3, 1, 1)
    g_images = g_images_raw * g_images_mask + input_images * (torch.ones_like(g_images_mask) - g_images_mask)

    param_dict = {
        "g_images"+postfix: g_images,
        "g_images_mask"+postfix: g_images_mask,
        "g_images_raw"+postfix: g_images_raw,
        "g_images_mask_raw"+postfix: g_images_mask_raw.unsqueeze(1).repeat(1, 3, 1, 1),
    }
    return param_dict


def renderer_random(m_full, tex, shape_full, postfix=""):

    g_images, g_images_mask = warp_texture_torch(tex, m_full, shape_full)


    param_dict = {
        "g_images"+postfix: g_images,
        "g_images_mask"+postfix: g_images_mask.unsqueeze(1).repeat(1, 3, 1, 1),
    }
    return param_dict


def ZBuffer_Rendering_CUDA_op_v2_sz224_torch(s2d, tri, vis):

    s2d = s2d.contiguous()

    map, mask = ZBuffer_cuda.forward(s2d, tri.int(), vis, CFG.tri_num, CFG.vertex_num, CFG.image_size)

    return map, mask


def warping_flow(m, mshape, output_size=96, is_reduce=False):
    batch_size = mshape.shape[0]

    tri = utils.load_3DMM_tri(is_reduce)
    vertex_tri = utils.load_3DMM_vertex_tri(is_reduce)
    vt2pixel_u, vt2pixel_v = utils.load_FaceAlignment_vt2pixel(is_reduce)

    tri = torch.from_numpy(tri).cuda().long()  # [3, TRI_NUM + 1]
    vertex_tri = torch.from_numpy(vertex_tri).cuda()  # [8, VERTEX_NUM]
    vt2pixel_u = torch.from_numpy(vt2pixel_u).cuda()
    vt2pixel_v = torch.from_numpy(vt2pixel_v).cuda()

    m = m.view((batch_size, 4, 2))
    m_row1 = F.normalize(m[:, 0:3, 0], dim=1)
    m_row2 = F.normalize(m[:, 0:3, 1], dim=1)
    m_row3 = torch.cross(m_row1, m_row2)
    m_row3 = F.pad(m_row3, (0, 1))
    m_row3 = torch.unsqueeze(m_row3, -1)
    m = torch.cat((m, m_row3), dim=2)

    vertex3d = mshape.view((batch_size, -1, 3))
    vertex4d = F.pad(vertex3d, (0, 1), mode='constant', value=1)


    vertex2d = torch.matmul(vertex4d, m)

    normal, normalf = compute_normal_torch(vertex3d, tri, vertex_tri )

    normalf4d = F.pad(normalf, (0, 1), mode='constant', value=1).cuda()

    rotated_normalf = torch.matmul(normalf4d, m)
    rotated_normalf = torch.transpose(rotated_normalf, 1, 2)
    _, _, rotated_normalf_z = torch.split(rotated_normalf, (1, 1, 1), dim=1)
    visible_tri = torch.gt(rotated_normalf_z, 0)


    grid = torch.linspace(0, CFG.image_size - 1, CFG.image_size)
    u, v = torch.meshgrid(grid, grid)
    u = torch.transpose(u, 0, 1)
    v = torch.transpose(v, 0, 1)
    u = torch.flatten(u).cuda()
    v = torch.flatten(v).cuda()

    vertex2d_i = torch.squeeze(vertex2d, dim=1)
    visible_tri_i = torch.squeeze(visible_tri, dim=1)

    vertex2d_u, vertex2d_v, vertex2d_z = torch.split(vertex2d_i, (1, 1, 1), dim=2)
    vertex2d_u = vertex2d_u - 1
    vertex2d_v = CFG.image_size - vertex2d_v
    vertex2d_i = torch.cat((vertex2d_v, vertex2d_u, vertex2d_z), dim=2)
    vertex2d_i = torch.transpose(vertex2d_i, 1, 2)

    tri_map_2d = []
    masks = []
    for i in range(batch_size):
        tri_map_2d_i, masks_i = ZBuffer_Rendering_CUDA_op_v2_sz224_torch(vertex2d_i[i].float(), tri, visible_tri[i])
        tri_map_2d.append(tri_map_2d_i)
        masks.append(masks_i)

    tri_map_2d = torch.stack(tri_map_2d)
    masks = torch.stack(masks)


    tri_map_2d_flat = tri_map_2d.view((batch_size, -1))

    vt = torch.gather(torch.unsqueeze(torch.transpose(tri, 0, 1), 0).repeat(batch_size, 1, 1), 1,
                      torch.unsqueeze(tri_map_2d_flat, dim=-1).repeat(1, 1, 3).long())

    pixel_uu = torch.gather(F.pad(vertex2d_u.repeat(1, 1, 3), (0, 0, 0, 1)), 1, vt)

    pixel_vv = torch.gather(F.pad(vertex2d_v.repeat(1, 1, 3), (0, 0, 0, 1)), 1, vt)

    c1, c2, c3 = barycentric_torch(pixel_uu, pixel_vv, u, v)

    pixel_u = torch.gather(torch.unsqueeze(vt2pixel_u, -1).repeat(batch_size, 1, 3), 1, vt)
    pixel_v = torch.gather(torch.unsqueeze(vt2pixel_v, -1).repeat(batch_size, 1, 3), 1, vt)


    pixel1_u, pixel2_u, pixel3_u = torch.split(pixel_u, (1, 1, 1), dim=-1)
    pixel1_v, pixel2_v, pixel3_v = torch.split(pixel_v, (1, 1, 1), dim=-1)

    pixel_u = torch.squeeze(pixel1_u) * c1 + torch.squeeze(pixel2_u) * c2 + torch.squeeze(pixel3_u) * c3
    pixel_v = torch.squeeze(pixel1_v) * c1 + torch.squeeze(pixel2_v) * c2 + torch.squeeze(pixel3_v) * c3

    pixel_u = pixel_u.view((batch_size, CFG.image_size, CFG.image_size))
    pixel_v = pixel_v.view((batch_size, CFG.image_size, CFG.image_size))

    return pixel_u, pixel_v, masks


def rendering_wflow(texture, flow_u, flow_v):
    image = bilinear_sampler_torch(texture, flow_v, flow_u)
    image = image.permute((0, 3, 1, 2))
    return image


def apply_mask(image, mask, background=None):
    image = torch.mul(image, mask.float())

    if background is not None:
        image = image + torch.mul(background, 1 - mask.float())
    return image


def warp_texture_torch (texture, m, mshape, output_size=96, is_reduce=False):

    '''
        Render image
        Parameters
            texture:    [batch, 3, 192, 224]
            m:          [batch, 8]
            mshape:     [batch , VERTEXU_NUM * 3]
        Returns
            image: [batch, x, proxy.json, c]
    '''

    # texture = texture.permute((0, 2, 3, 1)) # [batch, 192, 224, 3]
    pixel_u, pixel_v, masks = warping_flow(m, mshape, output_size, is_reduce)

    images = bilinear_sampler_torch(texture, pixel_v, pixel_u)

    return images.permute((0, 3, 1, 2)), masks


def barycentric_torch( pixel_uu, pixel_vv, u, v):

    v0_u = pixel_uu[:, :, 1] - pixel_uu[:, :, 0]
    v0_v = pixel_vv[:, :, 1] - pixel_vv[:, :, 0]

    v1_u = pixel_uu[:, :, 2] - pixel_uu[:, :, 0]
    v1_v = pixel_vv[:, :, 2] - pixel_vv[:, :, 0]

    v2_u = u - pixel_uu[:, :, 0]
    v2_v = v - pixel_vv[:, :, 0]

    invDenom = (v0_u * v1_v - v1_u * v0_v + 1e-6)
    c2 = (v2_u * v1_v - v1_u * v2_v) / (invDenom)
    c3 = (v0_u * v2_v - v2_u * v0_v) / (invDenom)
    c1 = 1.0 - c2 - c3

    return c1, c2, c3



def compute_normal_torch ( vertex, tri, vertex_tri ):
    # Unit normals to the faces
    # Parameters:
    #   vertex : batch_size x vertex_num x 3
    #   tri : 3xtri_num
    #   vertex_tri: T x vertex_num (T=8: maxium number of triangle each vertex can belong to)
    # Output
    #   normal:  batch_size x vertex_num x 3
    #   normalf: batch_size x tri_num x 3
    batch_size = vertex.shape[0]

    tri = torch.transpose(tri, 0, 1)
    tri = torch.unsqueeze(tri, 0)
    tri = torch.unsqueeze(tri, -1)
    tri = torch.cat(batch_size * [tri])[:, :-1, :]

    vt1_indices = torch.cat(3 * [tri[:, :, 0]], -1).cuda()
    vt2_indices = torch.cat(3 * [tri[:, :, 1]], -1).cuda()
    vt3_indices = torch.cat(3 * [tri[:, :, 2]], -1).cuda()
    vt1 = torch.gather(vertex, 1, vt1_indices)
    vt2 = torch.gather(vertex, 1, vt2_indices)
    vt3 = torch.gather(vertex, 1, vt3_indices)
    zeros = torch.zeros((batch_size, 1, 3)).cuda()
    vt1_padded = torch.cat((vt1, zeros), 1)
    vt2_padded = torch.cat((vt2, zeros), 1)
    vt3_padded = torch.cat((vt3, zeros), 1)

    normalf = torch.cross(vt2_padded - vt1_padded, vt3_padded - vt1_padded)
    normalf = F.normalize(normalf, dim=2)


    equal = vertex_tri != (tri.shape[1])
    expand = torch.unsqueeze(equal, 2)

    mask = expand.repeat(1, 1, 3).cuda()

    vertex_tri = vertex_tri.view((-1, 1))

    normal_indices = torch.unsqueeze(vertex_tri, 0)
    normal_indices = torch.cat(batch_size * [normal_indices])
    normal_indices = torch.cat(3 * [normal_indices], -1).cuda()
    normal = torch.gather(normalf, 1, normal_indices.long())

    normal = normal.view((batch_size, 8, -1, 3))
    multi = torch.mul(normal, mask.float())
    normal = torch.sum(multi, dim=1)


    normal = F.normalize(normal, dim=2)

    # Enforce that the normal are outward

    mean = torch.mean(vertex, 1)
    mean = torch.unsqueeze(mean, 1)
    v = vertex - mean
    s = torch.sum(torch.mul(v, normal), 1, keepdim=True)


    count_s_greater_0 = torch.sum(1 * torch.gt(s, 0), 0, keepdim=True)
    count_s_less_0 = torch.sum(1 * torch.lt(s, 0), 0, keepdim=True)


    sign = 2 * torch.gt(count_s_greater_0, count_s_less_0).float() - 1

    normal = torch.mul(normal, sign.repeat((1, CFG.vertex_num, 1)))
    normalf = torch.mul(normalf, sign.repeat((1, CFG.tri_num + 1, 1)))

    return normal, normalf




def compute_landmarks_torch(m, shape):
    batch_size = m.shape[0]

    kpts = torch.from_numpy(utils.load_3DMM_kpts())
    kpts_num = kpts.shape[0]
    indices = torch.zeros([batch_size, kpts_num, 2]).int()
    for i in range(batch_size):
        indices[i, :, 0] = i
        indices[i, :, 1:2] = kpts

    vertex3d = shape.view((batch_size, -1, 3))
    indices1 = torch.arange(0, batch_size).long()
    vertex3d = vertex3d[indices1, kpts.long(), :].permute((1, 0, 2))
    vertex4d = torch.cat((vertex3d, torch.ones((vertex3d.shape[0], vertex3d.shape[1], 1)).cuda()), dim=2)

    m = m.view((batch_size, 4, 2))
    vertex2d = torch.matmul(vertex4d, m)

    [vertex2d_u, vertex2d_v] = torch.split(vertex2d, (1, 1), dim=2)
    vertex2d_u = vertex2d_u - 1
    vertex2d_v = CFG.image_size - vertex2d_v

    return vertex2d_u, vertex2d_v



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
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1

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


def generate_shade(il, m, mshape, is_with_normal=False, is_clamp=False):
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
    batch_size = il.shape[0]

    # load 3DMM files
    tri = torch.from_numpy(utils.load_3DMM_tri())  # [3, TRI_NUM + 1]
    vertex_tri = torch.from_numpy(utils.load_3DMM_vertex_tri())  # [8, VERTEX_NUM]
    vt2pixel_u, vt2pixel_v = [torch.from_numpy(vt2pixel) for vt2pixel in utils.load_3DMM_vt2pixel()]  # [VERTEX_NUM + 1, ], [VERTEX_NUM + 1, ]
    tri_2d = torch.from_numpy(utils.load_3DMM_tri_2d())  # [192, 224] (Fragment shader)
    tri_2d_barycoord = torch.from_numpy(utils.load_3DMM_tri_2d_barycoord()).cuda()  # [192, 224, 3]

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
    m_i = m.view((batch_size, 4, 2))[:, 0:3, :]
    # l2_sum = torch.norm(m_i, dim=(1), keepdim=True)
    # m_i_row = torch.div(m_i, l2_sum)
    m_i_row = F.normalize(m_i, dim=1)
    m_cross = torch.cross(m_i_row[:, :, 0], m_i_row[:, :, 1])
    m_cross = torch.unsqueeze(m_cross, -1)
    m_i = torch.cat((m_i_row, m_cross), dim=2)
    m_i = torch.transpose(m_i, 1, 2)

    vertex3d_rs = mshape.view((batch_size, -1, 3))

    normal, normalf = compute_normal_torch(vertex3d_rs, tri, vertex_tri)

    normal = torch.transpose(normal, 1, 2)
    rotated_normal = torch.matmul(m_i, normal)
    rotated_normal = torch.transpose(rotated_normal, 1, 2)

    vt = torch.unsqueeze(vt, 0)
    vt = torch.unsqueeze(vt, -1)
    vt = torch.cat(batch_size * [vt], 0)
    vt = torch.cat(3 * [vt], -1)
    vt1 = vt[:, 0, :]
    vt2 = vt[:, 1, :]
    vt3 = vt[:, 2, :]
    zeros = torch.zeros((batch_size, 1, 3)).cuda()

    rotated_normal = torch.cat((rotated_normal, zeros), dim=1)
    normal_flat_vt1 = torch.gather(rotated_normal, 1, vt1)
    normal_flat_vt2 = torch.gather(rotated_normal, 1, vt2)
    normal_flat_vt3 = torch.gather(rotated_normal, 1, vt3)

    normalf_flat = normal_flat_vt1 * vt1_coeff + normal_flat_vt2 * vt2_coeff + normal_flat_vt3 * vt3_coeff


    # 명암조절 map
    shade = shading_torch(il, normalf_flat)
    shade = shade.view((-1, CFG.texture_size[0], CFG.texture_size[1], 3)) # [batch, 192, 224, 3]
    normalf_flat = normalf_flat.view((-1, CFG.texture_size[0], CFG.texture_size[1], 3))

    if is_clamp:
        shade = torch.clamp(shade, -1, 1)

    if is_with_normal:
        return shade, normalf_flat

    return shade.permute((0, 3, 1, 2))



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
    batch_size = shape[0]

    normal = torch.unsqueeze(normal, -1)
    normal_x = normal[:, :, 0]
    normal_y = normal[:, :, 1]
    normal_z = normal[:, :, 2]

    pi = math.pi
    sh = torch.zeros(batch_size, shape[1], 1, 9).cuda()
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

    # gray = (L1 + L2 + L3) / 1
    # L1 = torch.ones_like(L1) * 10
    # L2 = torch.ones_like(L2) * 10
    # L3 = torch.ones_like(L3) * 10

    # alpha = 0.0
    # L1 = F.normalize(L1, dim=2) * alpha + 1 - alpha
    # L2 = F.normalize(L2, dim=2) * alpha + 1 - alpha
    # L3 = F.normalize(L3, dim=2) * alpha + 1 - alpha

    B1 = torch.matmul(sh, L1)
    B2 = torch.matmul(sh, L2)
    B3 = torch.matmul(sh, L3)
    B = torch.cat((B1, B2, B3), dim=-1)
    B = torch.squeeze(B, dim=2)

    return B

