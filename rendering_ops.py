import torch

from _3dmm_utils import *


def bilinear_interpolate(image, x, y):
    # 이 코드를 믿지 마시오... TODO 코드 검증

    B, C, H, W = image.shape

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()

    x1 = x0 + 1
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, W-1)
    x1 = torch.clamp(x1, 0, W-1)
    y0 = torch.clamp(y0, 0, H-1)
    y1 = torch.clamp(y1, 0, H-1)

    Ia = image[:, :, y0, x0].float().unsqueeze(3)
    Ib = image[:, :, y1, x0].float().unsqueeze(3)
    Ic = image[:, :, y0, x1].float().unsqueeze(3)
    Id = image[:, :, y1, x1].float().unsqueeze(3)

    x0 = x0.float()
    x1 = x1.float()
    y0 = y0.float()
    y1 = y1.float()

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    wa = wa.repeat(B, C, 1).unsqueeze(3).to(image.device)
    wb = wb.repeat(B, C, 1).unsqueeze(3).to(image.device)
    wc = wc.repeat(B, C, 1).unsqueeze(3).to(image.device)
    wd = wd.repeat(B, C, 1).unsqueeze(3).to(image.device)

    return sum([wa * Ia, wb * Ib, wc * Ic, wd * Id])


def generate_shade(il, m, mshape, texture_size, is_with_normal=False):

    return torch.rand([mshape.shape[0], 3, texture_size[0], texture_size[1]]).to("cuda")
    bat_sz = il.shape[0]

    tri = torch.tensor(load_3DMM_tri()).long()
    vertex_tri = torch.tensor(load_3DMM_vertex_tri()).long()
    # vt2pixel_u, vt2pixel_v = load_3DMM_vt2pixel()  # 사용 안함
    tri_2d = torch.tensor(load_3DMM_tri_2d()).long()
    tri_2d_barycoord = torch.tensor(load_3DMM_tri_2d_barycoord()).float()

    tri2vt1 = tri[0, :]
    tri2vt2 = tri[1, :]
    tri2vt3 = tri[2, :]

    tri_2d_flat = tri_2d.view(-1, 1)

    vt1 = tri2vt1[tri_2d_flat]
    vt2 = tri2vt2[tri_2d_flat]
    vt3 = tri2vt3[tri_2d_flat]

    vt1_coeff = tri_2d_barycoord[:, :, 0].view(-1, 1)
    vt2_coeff = tri_2d_barycoord[:, :, 1].view(-1, 1)
    vt3_coeff = tri_2d_barycoord[:, :, 2].view(-1, 1)

    m_single = torch.split(m, 1)
    shape_single = torch.split(mshape, 1)

    normalf_flats = []

    for i in range(bat_sz):
        m_i = torch.t(m_single[i].view(4, 2))

        # l2_normalize
        m_i_row1 = m_i[0, 0:3]
        m_i_row1_norm = torch.norm(m_i_row1, p=2, dim=0, keepdim=True)
        m_i_row1 = m_i_row1.div(m_i_row1_norm.expand_as(m_i_row1))

        m_i_row2 = m_i[1, 0:3]
        m_i_row2_norm = torch.norm(m_i[1, 0:3], p=2, dim=0, keepdim=True)
        m_i_row2 = m_i_row2.div(m_i_row2_norm.expand_as(m_i_row2))
        m_i_row3 = torch.cross(m_i_row1, m_i_row2)

        m_i = torch.cat([m_i_row1.unsqueeze(0), m_i_row2.unsqueeze(0), m_i_row3.unsqueeze(0)])

        vertex3d_rs = torch.t(shape_single[i].view(-1, 3))

    return np

def warp_texture(texture, m, mshape, output_size=224):
    images = texture
    masks = mshape
    return (torch.rand(images.shape[0], 3, output_size, output_size).to("cuda"),
            torch.rand(images.shape[0], output_size, output_size).to("cuda"))

kpts = load_3DMM_kpts()

def compute_landmarks(m, shape, output_size=224):
    # m: rotation matrix [batch_size x (4x2)]
    # shape: 3d vertices location [batch_size x (vertex_num x 3)]
    # validate OK

    n_size = m.shape[0]
    s = output_size

    kpts_num = kpts.shape[0]

    indices = np.zeros([n_size, kpts_num, 2], np.int32)

    for i in range(n_size):
        indices[i, :, 0] = i
        indices[i, :, 1:2] = kpts

    vertex3d = shape.view(n_size, -1, 3)
    vertex3d = vertex3d[:, kpts.reshape(-1), :]
    vertex4d = torch.cat((vertex3d, torch.ones(list(vertex3d.shape[0:2]) + [1]).float().to(vertex3d.device)), dim=2)

    m = m.view(n_size, 4, 2)
    vertex2d = torch.matmul(torch.transpose(m, 1, 2), torch.transpose(vertex4d, 1, 2))                                                             # batch_size x 2 x kpts_num
    vertex2d = torch.transpose(vertex2d, 1, 2)

    vertex2d_u = vertex2d[:, :, 0:1]
    vertex2d_u = vertex2d_u - 1

    vertex2d_v = vertex2d[:, :, 1:2]
    vertex2d_v = s - vertex2d_v

    return vertex2d_u, vertex2d_v

