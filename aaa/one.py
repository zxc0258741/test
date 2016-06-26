 #-*- coding: UTF-8 -*- 
import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
import os
import time
dic={}

def get_imlist(path):
  return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.npy')]
npy5list=get_imlist('/home/lcw/下載/tensorflow/cpool5/')

c5 = np.load('/home/lcw/下載/tensorflow/cpool5/Dali007.npy')


def build_style_loss(A, G):
  loss = np.sum((A-G)**2)
  #print loss
  #print  '---------'
  return loss

for npypath in npy5list:
  d = np.load(npypath)
  a=npypath.split('/')[6].split('.')[0]
  dic.update({
	a:build_style_loss(c5,d)
   })



dict= sorted(dic.iteritems(), key=lambda d:d[1], reverse = True)
for i,a in enumerate(dict):
  print a[0],a[1],abs(i-1400)

  

    
#build_style_loss(c,d)
