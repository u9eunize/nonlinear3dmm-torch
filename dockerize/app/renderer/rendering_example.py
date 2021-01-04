import pyredner
import torch
import math
import numpy as np
from renderer.rendering_ops_redner import Batch_Renderer
from utils import *
from os.path import basename
from scipy.io import loadmat
from settings import *
from renderer.rendering_ops_redner import renderer



def main ():

    coeff = loadmat('/dataset/mat/006515.mat')['coeff'][0]
    coeff = torch.tensor(coeff)
    shape, exp, tex, angle, light, trans = torch.split(coeff, (80, 64, 80, 3, 27, 3), dim=-1)

    angle = angle.unsqueeze(0)
    light = light.unsqueeze(0)
    trans = trans.unsqueeze(0)

    shape = torch.einsum('ij,aj->ai', CFG.shapeBase_cpu, torch.unsqueeze(shape, 0))
    exp = torch.einsum('ij,aj->ai', CFG.exBase_cpu, torch.unsqueeze(exp, 0))
    tex = torch.einsum('ij,aj->ai', CFG.texBase_cpu, torch.unsqueeze(tex, 0)) / 255.0

    shape = (shape + exp + CFG.mean_shape_cpu).view([1, -1, 3])
    tex = (tex + CFG.mean_tex_cpu).view([1, -1, 3])

    images, masks, _ = renderer.render(
        vertex_batch=shape.to(CFG.device),
        color_batch=tex.to(CFG.device),
        trans_batch=trans.to(CFG.device),
        angle_batch=angle.to(CFG.device),
        light_batch=light.to(CFG.device),
        print_timing=True)

    image_ = torch.zeros_like(images[0])
    image_[:, :, 0] = images[0, :, :, 2]
    image_[:, :, 1] = images[0, :, :, 1]
    image_[:, :, 2] = images[0, :, :, 0]

    image = image_.cpu().detach().numpy()
    mask = masks[0].cpu().detach().numpy()

    return
    


if __name__ == '__main__':
    init_3dmm_settings()
    main()
