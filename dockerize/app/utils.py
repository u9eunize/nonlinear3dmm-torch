"""
Notes: Many of .dat files are written using Matlab.
Hence, there are "-1" subtraction to Python 0-based indexing
"""
from __future__ import division
import math
import json
import numpy as np
import torch
import os
import hashlib
import shutil
import sys
import tempfile
from os import makedirs
from os.path import join
from glob import glob
from urllib.request import urlopen, Request

from torchvision.utils import save_image
from torchvision.utils import make_grid

from settings import CFG, LOSSES

import importlib
importlib.import_module('settings', 'CFG')

try:
    from tqdm.auto import tqdm  # automatically select proper tqdm submodule if available
except ImportError:
    try:
        from tqdm import tqdm
    except ImportError:
        # fake tqdm if it's not installed
        class tqdm(object):  # type: ignore

            def __init__(self, total=None, disable=False,
                         unit=None, unit_scale=None, unit_divisor=None):
                self.total = total
                self.disable = disable
                self.n = 0
                # ignore unit, unit_scale, unit_divisor; they're just for real tqdm

            def update(self, n):
                if self.disable:
                    return

                self.n += n
                if self.total is None:
                    sys.stderr.write("\r{0:.1f} bytes".format(self.n))
                else:
                    sys.stderr.write("\r{0:.1f}%".format(100 * self.n / float(self.total)))
                sys.stderr.flush()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.disable:
                    return

                sys.stderr.write('\n')


def safe_json(data):
    if data is None:
        return True
    elif isinstance(data, (bool, int, float)):
        return True
    elif isinstance(data, (tuple, list)):
        return all(safe_json(x) for x in data)
    elif isinstance(data, dict):
        return all(isinstance(k, str) and safe_json(v) for k, v in data.items())
    return False


def save_configuration(path, name="configuration.json"):
    fp = open("/".join([path, CFG.name + "-" + name]), "w")
    config = dict()
    for key, value in CFG.__dict__.items():
        if safe_json(value):
            config[key] = value
    for key, value in LOSSES.items():
        if safe_json(value):
            config[key+"_loss"] = value
    json.dump(config, fp, indent=4)


def save(model, global_optimizer, encoder_optimizer, epoch, path, step):
    dir_path = get_checkpoint_dir(path, epoch)
    makedirs(dir_path, exist_ok=True)

    save_name = get_checkpoint_name(path, epoch, step)
    save_configuration(dir_path)

    print("=> save checkpoint '{} (epoch: {}, step: {})')".format(save_name, epoch, step))
    torch.save({
        'step': step,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'global_optimizer': global_optimizer.state_dict(),
        'encoder_optimizer': encoder_optimizer.state_dict(),
    }, save_name)
    print("=> saved checkpoint '{} (epoch: {}, step: {})')".format(save_name, epoch, step))


def load_from_name(model, global_optimizer=None, encoder_optimizer=None, ckpt_name=None):
    if ckpt_name is None:
        ckpt_name = join(CFG.checkpoint_root_path, CFG.checkpoint_regex)

    print("=> load checkpoint '{}'".format(ckpt_name))
    checkpoint = torch.load(ckpt_name, map_location=CFG.device)

    start_epoch = checkpoint['epoch']
    start_step = checkpoint['step'] if 'step' in checkpoint else 0

    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model")

    if global_optimizer is not None:
        global_optimizer.load_state_dict(checkpoint['global_optimizer'])
        print("=> loaded global_optimizer")
    if encoder_optimizer is not None:
        encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
        print("=> loaded encoder_optimizer")

    print("=> loaded checkpoint '{}' (epoch {}, step {})".format(ckpt_name, start_epoch, start_step))

    return model, global_optimizer, encoder_optimizer, start_epoch, start_step + 1


def load(model, global_optimizer=None, encoder_optimizer=None,
         start_epoch=CFG.checkpoint_epoch, start_step=CFG.checkpoint_step):
    if not CFG.checkpoint_name and not CFG.checkpoint_regex:
        return model, global_optimizer, encoder_optimizer, 0, 0

    if CFG.checkpoint_name:
        ckpt_name = join(CFG.checkpoint_root_path, CFG.checkpoint_name)
    else:
        start_epoch_str = f"{start_epoch:02d}" if start_epoch is not None else "*"
        start_step_str = f"{start_step:06d}.pt" if start_step is not None else "*"

        path = join(CFG.checkpoint_root_path, CFG.checkpoint_regex,
                    f'ckpt_{start_epoch_str}', f'model_ckpt_{start_epoch_str}_{start_step_str}')
        print(f"try load {path}")
        ckpt_name = glob(path)
        ckpt_name.sort()
        ckpt_name = ckpt_name[-1]

    return load_from_name(model, global_optimizer, encoder_optimizer, ckpt_name=ckpt_name)


def get_checkpoint_dir(path, epoch):
    return os.path.join(path, f"ckpt_{epoch:02d}")


def get_checkpoint_name(path, epoch, step):
    return os.path.join(f"{get_checkpoint_dir(path, epoch)}", f"model_ckpt_{epoch:02d}_{step:06d}.pt")


def grid_viewer(images, limit=8, split=False):
    if not isinstance(images, list):
        images = [images]

    reshape_imgs = []
    t_s = list(np.max(np.array([img.shape for img in images]), axis=0))
    for _, img in enumerate(images):
        reshape_image = torch.zeros(t_s).cuda().float()
        p_s = list(img.shape)
        reshape_image[:p_s[0], :p_s[1], :p_s[2], :p_s[3]] = img.float()
        reshape_image = torch.flip(reshape_image, dims=[1])
        if split:
            reshape_imgs += list(torch.split(reshape_image, 1, dim=1))
        else:
            reshape_imgs.append(reshape_image)

    image = make_grid(torch.cat(reshape_imgs, dim=0), nrow=t_s[0])
    return image.data.permute(1, 2, 0).cpu().numpy()


def load_3DMM_tri(is_reduce=False):
    # Triangle definition (i.e. from Basel model)

    # print ('Loading 3DMM tri ...')
    if not is_reduce:
        vertex_num = CFG.vertex_num
        postfix = ""
    else:
        vertex_num = CFG.vertex_num_reduce
        postfix = "_reduce"

    fd = open(CFG.definition_path + f'3DMM_tri{postfix}.dat')
    tri = np.fromfile(file=fd, dtype=np.int32)
    fd.close()
    # print tri

    tri = tri.reshape((3, -1)).astype(np.int32)
    tri = tri - 1
    tri = np.append(tri, [[vertex_num], [vertex_num], [vertex_num]], axis=1)

    # print('   DONE')
    return tri


def load_3DMM_vertex_tri(is_reduce=False):
    # Vertex to triangle mapping (list of all trianlge containing the cureent vertex)

    # print('Loading 3DMM vertex tri ...')

    if not is_reduce:
        post_fix = ""
        tri_num = CFG.tri_num
    else:
        post_fix = "_reduce"
        tri_num = CFG.tri_num_reduce

    fd = open(CFG.definition_path + f'3DMM_vertex_tri{post_fix}.dat')
    vertex_tri = np.fromfile(file=fd, dtype=np.int32)
    fd.close()

    vertex_tri = vertex_tri.reshape((8, -1)).astype(np.int32)
    # vertex_tri = np.append(vertex_tri, np.zeros([8,1]), 1)
    vertex_tri[vertex_tri == 0] = tri_num + 1
    vertex_tri = vertex_tri - 1

    # print('    DONE')
    return vertex_tri


def load_3DMM_vt2pixel():
    # Mapping in UV space

    fd = open(CFG.definition_path + 'vertices_2d_u.dat')
    vt2pixel_u = np.fromfile(file=fd, dtype=np.float32)
    vt2pixel_u = np.append(vt2pixel_u - 1, 0)
    fd.close()

    fd = open(CFG.definition_path + 'vertices_2d_v.dat')
    vt2pixel_v = np.fromfile(file=fd, dtype=np.float32)
    vt2pixel_v = np.append(vt2pixel_v - 1, 0)
    fd.close()

    return vt2pixel_u, vt2pixel_v


def load_3DMM_kpts():
    # 68 keypoints indices

    # print('Loading 3DMM keypoints ...')

    fd = open(CFG.definition_path + '3DMM_keypoints.dat')
    kpts = np.fromfile(file=fd, dtype=np.int32)
    kpts = kpts.reshape((-1, 1))
    fd.close()

    return kpts - 1


def load_3DMM_tri_2d(with_mask=False):
    fd = open(CFG.definition_path + '3DMM_tri_2d.dat')
    tri_2d = np.fromfile(file=fd, dtype=np.int32)
    fd.close()

    tri_2d = tri_2d.reshape(192, 224)

    tri_mask = tri_2d != 0

    tri_2d[tri_2d == 0] = CFG.tri_num + 1  # VERTEX_NUM + 1
    tri_2d = tri_2d - 1

    if with_mask:
        return tri_2d, tri_mask

    return tri_2d


def load_Basel_basic(element, is_reduce=False):
    fn = CFG.definition_path + '3DMM_' + element + '_basis.dat'
    # print('Loading ' + fn + ' ...')

    fd = open(fn)
    all_paras = np.fromfile(file=fd, dtype=np.float32)
    fd.close()

    all_paras = np.transpose(all_paras.reshape((-1, 53215)).astype(np.float32))

    mu = all_paras[:, 0]
    w = all_paras[:, 1:]

    # print('    DONE')

    return mu, w


def load_const_alb_mask():
    fd = open(CFG.definition_path + '3DMM_const_alb_mask.dat')
    const_alb_mask = np.fromfile(file=fd, dtype=np.uint8)
    fd.close()
    const_alb_mask = const_alb_mask - 1
    const_alb_mask = const_alb_mask.reshape((-1, 2)).astype(np.uint8)

    return const_alb_mask


def load_3DMM_tri_2d_barycoord():
    fd = open(CFG.definition_path + '3DMM_tri_2d_barycoord_reduce.dat')
    tri_2d_barycoord = np.fromfile(file=fd, dtype=np.float32)
    fd.close()

    tri_2d_barycoord = tri_2d_barycoord.reshape(192, 224, 3)

    return tri_2d_barycoord


def load_FaceAlignment_vt2pixel(is_reduce=False):
    post_fix = "_reduce" if is_reduce else ""
    fd = open(CFG.definition_path + 'vertices_2d_u' + post_fix + '.dat')
    vt2pixel_u = np.fromfile(file=fd, dtype=np.float32)
    vt2pixel_u = np.append(vt2pixel_u - 1, 0)
    fd.close()

    fd = open(CFG.definition_path + 'vertices_2d_v' + post_fix + '.dat')
    vt2pixel_v = np.fromfile(file=fd, dtype=np.float32)
    vt2pixel_v = np.append(vt2pixel_v - 1, 0) #vt2pixel_v[VERTEX_NUM] = 0
    fd.close()

    return vt2pixel_u, vt2pixel_v

def load_bfm2009_vt2pixel():
    fd = open(join(CFG.definition_path, 'bfm2009.idx'))
    vt2pixel = np.fromfile(file=fd, dtype=np.float32)
    fd.close()
    
    return vt2pixel

def load_bfm2009_pixel2vt():
    return None


def inverse_transform(images):
    return (images + 1.) / 2.


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    nn = images.shape[0]

    if size[1] < 0:
        size[1] = int(math.ceil(nn / size[0]))
    if size[0] < 0:
        size[0] = int(math.ceil(nn / size[1]))

    if (images.ndim == 4):
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
    else:
        img = images

    return img


def imsave(images, size, path):
    img = merge(images, size)
    img = torch.Tensor(img)
    img = img.permute((2, 0, 1))
    # plt.imshow(img)
    # plt.show()
    return save_image(img, path)


def save_images(images, size, image_path, inverse=True):
    if len(size) == 1:
        size = [size, -1]
    if size[1] == -1:
        size[1] = int(math.ceil(images.shape[0] / size[0]))
    if size[0] == -1:
        size[0] = int(math.ceil(images.shape[0] / size[1]))
    if (inverse):
        images = inverse_transform(images)

    return imsave(images, size, image_path)


def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    r"""Download object at the given URL to a local path.
    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, e.g. `/tmp/temporary_file`
        hash_prefix (string, optional): If not None, the SHA256 downloaded file should start with `hash_prefix`.
            Default: None
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True
    Example:
        >>> torch.hub.download_url_to_file('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth', '/tmp/temporary_file')
    """
    file_size = None
    # We use a different API for python2 since urllib(2) doesn'proxy.json recognize the CA
    # certificates in older Python
    req = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                   .format(hash_prefix, digest))
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)
