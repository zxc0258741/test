 #-*- coding: UTF-8 -*- 
import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
import os
import time
i=0
IMAGE_W = 300 
IMAGE_H = 300 

VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
STYLE_STRENGTH = 500

STYLE_LAYERS=[('conv5_1',3.)]
MEAN_VALUES = np.array([123, 117, 104]).reshape((1,1,1,3))

  

def get_imlist(path):
  return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
imglist=get_imlist('/home/lcw/下載/tensorflow/pic')

def build_net(ntype, nin, nwb=None):
  if ntype == 'conv':
    return tf.nn.relu(tf.nn.conv2d(nin, nwb[0], strides=[1, 1, 1, 1], padding='SAME')+ nwb[1])
  elif ntype == 'pool':
    return tf.nn.avg_pool(nin, ksize=[1, 2, 2, 1],
                  strides=[1, 2, 2, 1], padding='SAME')

def get_weight_bias(vgg_layers, i,):
  weights = vgg_layers[i][0][0][0][0][0]
  weights = tf.constant(weights)
  bias = vgg_layers[i][0][0][0][0][1]
  bias = tf.constant(np.reshape(bias, (bias.size)))
  return weights, bias

def build_vgg19(path):
  net = {}
  vgg_rawnet = scipy.io.loadmat(path)
  vgg_layers = vgg_rawnet['layers'][0]
  net['input'] = tf.Variable(np.zeros((1, IMAGE_H, IMAGE_W, 3)).astype('float32'))
  net['conv1_1'] = build_net('conv',net['input'],get_weight_bias(vgg_layers,0))
  net['conv1_2'] = build_net('conv',net['conv1_1'],get_weight_bias(vgg_layers,2))
  net['pool1']   = build_net('pool',net['conv1_2'])
  net['conv2_1'] = build_net('conv',net['pool1'],get_weight_bias(vgg_layers,5))
  net['conv2_2'] = build_net('conv',net['conv2_1'],get_weight_bias(vgg_layers,7))
  net['pool2']   = build_net('pool',net['conv2_2'])
  net['conv3_1'] = build_net('conv',net['pool2'],get_weight_bias(vgg_layers,10))
  net['conv3_2'] = build_net('conv',net['conv3_1'],get_weight_bias(vgg_layers,12))
  net['conv3_3'] = build_net('conv',net['conv3_2'],get_weight_bias(vgg_layers,14))
  net['conv3_4'] = build_net('conv',net['conv3_3'],get_weight_bias(vgg_layers,16))
  net['pool3']   = build_net('pool',net['conv3_4'])
  net['conv4_1'] = build_net('conv',net['pool3'],get_weight_bias(vgg_layers,19))
  net['conv4_2'] = build_net('conv',net['conv4_1'],get_weight_bias(vgg_layers,21))
  net['conv4_3'] = build_net('conv',net['conv4_2'],get_weight_bias(vgg_layers,23))
  net['conv4_4'] = build_net('conv',net['conv4_3'],get_weight_bias(vgg_layers,25))
  net['pool4']   = build_net('pool',net['conv4_4'])
  net['conv5_1'] = build_net('conv',net['pool4'],get_weight_bias(vgg_layers,28))
  net['conv5_2'] = build_net('conv',net['conv5_1'],get_weight_bias(vgg_layers,30))
  net['conv5_3'] = build_net('conv',net['conv5_2'],get_weight_bias(vgg_layers,32))
  net['conv5_4'] = build_net('conv',net['conv5_3'],get_weight_bias(vgg_layers,34))
  net['pool5']   = build_net('pool',net['conv5_4'])
  return net

def gram_matrix(x, area, depth):
  x1 = tf.reshape(x,(area,depth))
  g = tf.matmul(tf.transpose(x1), x1)
  
  return g

def gram_matrix_val(x, area, depth):
  x1 = x.reshape(area,depth)
  g = np.dot(x1.T, x1)
  return g

def build_style(a, x,path):
  M = a.shape[1]*a.shape[2]
  N = a.shape[3]
  A = gram_matrix_val(a, M, N )
  G = gram_matrix(x, M, N )
  np.save(path,A)
  return G


def read_image(path):
  image = scipy.misc.imread(path)
  image = np.expand_dims(image,axis=2)
  image = image[np.newaxis,:IMAGE_H,:IMAGE_W,:] 
  image = image - MEAN_VALUES
  return image



def main():
  net = build_vgg19(VGG_MODEL)
  sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
  sess.run(tf.initialize_all_variables())
  for imgpath in imglist: 
    style_img = read_image(imgpath)
    sess.run([net['input'].assign(style_img)])
    style = sum(map(lambda l: build_style(sess.run(net[l[0]]) ,  net[l[0]],'/home/lcw/下載/tensorflow/npy/'+imgpath.split('/')[6].split('.')[0]), STYLE_LAYERS))
  

  

  
  
  

if __name__ == '__main__':
  main()
