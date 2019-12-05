import csv
import os
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

train_imgs = os.listdir(r'C:\Users\tensorflow\Desktop\迁移学习\train')
test_imgs = os.listdir(r'C:\Users\tensorflow\Desktop\迁移学习\test')

train_batches = []
train_labels = []

test_batches = []

for ps,img in enumerate(train_imgs):
    img_name = r'C:\Users\tensorflow\Desktop\迁移学习\train\\'+img
    species = img.split('.')[0]
    idx = int(img.split('.')[1])
    image = Image.open(img_name)
    resized_img = image.resize((224,224),Image.ANTIALIAS)
    img_array = np.array(resized_img)
    
    train_batches.append(img_array)
    if species =='cat':
        train_labels.append(np.array([1,0]))
    if species =='dog':
        train_labels.append(np.array([1,0]))
    print('train:',ps+1)
    
for ps,img in enumerate(test_imgs):
    img_name = r'C:\Users\tensorflow\Desktop\迁移学习\test\\'+img
    image = Image.open(img_name)
    resized_img = image.resize((224,224),Image.ANTIALIAS)
    img_array = np.array(resized_img)
    
    test_batches.append(img_array)
    print('test:',ps+1)                   
    
train = shuffle(train_batches,train_labels)
#把dataset以npy格式保存
os.chdir(r'C:\Users\tensorflow\Desktop\迁移学习\dateset')
np.save('train_batches.npy',train[0])
np.save('train_labels.npy',train[1])
np.save('test_batches.npy',test_batches)

