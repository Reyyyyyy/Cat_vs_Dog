import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

dogs_dir = os.listdir(r'D:\kaggle\data\Dog')
cats_dir = os.listdir(r'D:\kaggle\data\Cat')

dog_shapes = []
cat_shapes = []

os.chdir(r'D:\kaggle\data\Dog')
for each in dogs_dir:
    img = np.array(Image.open(each))
    if img.shape not in dog_shapes:
        dog_shapes.append(img.shape)
        print('已知狗的图片分辨率有%d种'%(len(dog_shapes)))
    if len(dog_shapes) == 10:
        print('分辨率不少于10种')
        break

print('\n')

os.chdir(r'D:\kaggle\data\Cat')
for each in cats_dir:
    img = np.array(Image.open(each))
    if img.shape not in cat_shapes:
        cat_shapes.append(img.shape)
        print('已知猫的图片分辨率有%d种'%(len(cat_shapes)))
    if len(cat_shapes) == 10:
        print('分辨率不少于10种')
        break
