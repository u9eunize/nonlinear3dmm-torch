from utils import *
from rendering_ops import *
# import tensorflow as tf
import numpy as np
import torch
import ZBuffer_cuda
from utils import *


VERTEX_NUM = 53215

def main():

    batch_size = 16
    output_size = 224
    texture_size = [192, 224]
    mDim = 8
    vertexNum = VERTEX_NUM
    channel_num = 3
    tri_num = 105840
    vertex_num = 53215

    data = np.load('sample_data.npz')
    texture = torch.from_numpy(data['sample_texture']).cuda().float()
    shape = torch.from_numpy(data['sample_shape']).cuda().float()
    m = torch.from_numpy(data['sample_m']).cuda().float()
    # print(m.shape)
    texture = texture.permute((0, 3, 1, 2))
    images, foreground_mask = warp_texture_torch(texture, m, shape)
    images = images.cpu()
    images = images.permute((0, 2, 3, 1))
    foreground_mask = foreground_mask.cpu()
    save_images(images, [4, -1], './rendered_img.png')
    save_images(data['sample_texture'], [4, -1], './texture.png')


    # gpu_options = tf.GPUOptions(visible_device_list ="0", allow_growth = True)
    #
    # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)) as sess:
    #
    #     """ Graph """
    #     m_ph       = tf.placeholder(tf.float32, [batch_size, mDim])
    #     shape_ph   = tf.placeholder(tf.float32, [batch_size, vertexNum*3])
    #     texture_ph = tf.placeholder(tf.float32, [batch_size]+texture_size +[channel_num])
    #     images, foreground_mask = warp_texture(texture_ph, m_ph, shape_ph, output_size = output_size)
    #
    #     s_img  = sess.run( images, feed_dict={ texture_ph: data['sample_texture'], shape_ph:data['sample_shape'], m_ph:data['sample_m']})
    #
    #     save_images(s_img, [4, -1], './rendered_img.png')
    #     save_images(data['sample_texture'], [4, -1], './texture.png')
        
        

        
if __name__ == '__main__':
    main()
