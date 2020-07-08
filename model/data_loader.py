import numpy as np
import cv2
import glob, os
import cv2
import pickle as pkl

def load_inputs(locs,base_dir):
  images = []
  vectors = []
  curr_dir = ""
  os.chdir(base_dir)
  dirs = os.listdir('.')
  """
  for r in locs:
    if r[0] + "/" + r[1] != curr_dir:
      os.chdir(base_dir)
      os.chdir(r[0] + "/" + r[1])
  """
  for d in dirs:
    if os.path.isdir(d):
      os.chdir(d + '/run1/')
      print(d)
      for f in os.listdir('.'):
        if f.endswith('.pkl'):
          name = f[:-4]
          images.append(cv2.imread("depth_{}.png".format(name),0))
          with open("{}.pkl".format(name),'rb') as f:
            vectors.append(pkl.load(f,encoding='bytes'))
    os.chdir('../../')
  return np.array(images), vectors
    
