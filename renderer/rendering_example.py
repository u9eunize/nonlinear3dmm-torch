from utils import *
from renderer.rendering_ops import *
# import tensorflow as tf
import numpy as np
import torch
import ZBuffer_cuda
from utils import *

def main():


    data = np.load('sample_data.npz')

    texture = torch.from_numpy(data['sample_texture']).double().cuda()
    shape = torch.from_numpy(data['sample_shape']).double().cuda()
    m = torch.from_numpy(data['sample_m']).double().cuda()
    # print(m.shape)
    texture = texture.permute((0, 3, 1, 2))
    images, foreground_mask = warp_texture_torch(texture, m, shape)
    images = images.cpu()
    images = images.permute((0, 2, 3, 1))
    foreground_mask = foreground_mask.cpu()
    save_images(images, [4, -1], './rendered_img.png')
    save_images(data['sample_texture'], [4, -1], './texture.png')

    # i1 = np.load('i1.npy')
    # i1_t = np.load('i1_t.npy')
    # i2 = np.load('i2.npy')
    # i2_t = np.load('i2_t.npy')
    # o1 = np.load('o1.npy')
    # o1_t = np.load('o1_t.npy')
    # o2 = np.load('o2.npy')
    # o2_t = np.load('o2_t.npy')
    # print("i1: ", np.sum(np.abs(i1 - i1_t)), i1.shape, i1_t.shape)
    # print("i2: ", np.sum(np.abs(i2 * 1 - i2_t * 1)), i2.shape, i2_t.shape)
    # print("o1: ", np.sum(np.abs(o1 - o1_t)), o1.shape, o1_t.shape)
    # print("o2: ", np.sum(np.abs(o2 - o2_t)), o2.shape, o2_t.shape)
    # pass
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
