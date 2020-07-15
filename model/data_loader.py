import numpy as np
import cv2
import glob, os
import pickle as pkl
import random

def load_batch(batch,base_dir,train=True,size=[480,640]):
  images = np.empty((batch.shape[0],size[0],size[1]))
  accels = np.empty((batch.shape[0],size[0],size[1],3))
  tf = np.empty((batch.shape[0],size[0],size[1]))
  os.chdir(base_dir)
  augs = np.array(batch[:,2:],dtype=float)
  print(batch.shape[0])
  for i in range(batch.shape[0]):
    r = batch[i]
    a = augs[i]
    time = round(float(r[1]),2)
    image = cv2.imread(r[0] + "depth_{}.png".format(time),0)
    value = np.load(r[0] + "indices_{}.npy".format(time),0)
    with open(r[0] + '{}.pkl'.format(time), 'rb') as f:
      u = pkl._Unpickler(f)
      u.encoding = 'latin1'
      p = u.load()
      
    value = dict_to_accel(p,value)
    
    if train:
      if a[0] != 0 or a[1] != 0:
        images[i] = np.roll(image,shift=(int(a[0]),int(a[1])),axis=(0,1))
        accels[i] = np.roll(value,shift=(int(a[0]),int(a[1])),axis=(0,1))
        
      elif a[2] != 0 or a[3] != 0:
        if a[2] != 0 and a[3] !=0:
          flip_i = np.flip(image,axis=0)
          value_i = np.flip(value,axis=0)
          images[i] = np.flip(flip_i,axis=1)
          accels[i] = np.flip(value_i,axis=1)
        elif a[2] != 0:
          images[i] = np.flip(image,axis=0)
          accels[i] = np.flip(value,axis=0)
        else:
          images[i] = np.flip(image,axis=1)
          accels[i] = np.flip(value,axis=1)
          
      elif a[4] != 0:
        images[i] = image+np.random.normal(0,a[4],image.shape)
        accels[i] = value
    
      else:
        images[i] = image
        accels[i] = value
        
    else:
      images[i] = image
      accels[i] = value
    
    tf[i] = accels[i,:,:,0]>-1
  
  return images, accels, tf
  
def dict_to_accel(dic, values):
  accels = np.empty((values.shape[0],values.shape[1],3))
  indices = np.argwhere(values[:,:,0]>-1)
  for i in indices:
    tup = tuple((i[0],i[1]))
    ac = list(dic[tup])
    accels[i[0],i[1]] = random.choice(ac)
  return accels
