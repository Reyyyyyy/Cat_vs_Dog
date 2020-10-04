import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings("error", category=UserWarning)#捕获警告

dogs_dir = os.listdir(r'D:\kaggle\data\Dog')
cats_dir = os.listdir(r'D:\kaggle\data\Cat')

#训练集 11961:111935      测试集 498:500       狗：猫
#猫的标签[1,0]  狗的标签[0,1]
train_batches = []
train_labels  = []
test_batches  = []
test_labels   = []

os.chdir(r'D:\kaggle\data\Dog')
for idx,each in enumerate(dogs_dir):
    try:
        img = Image.open(each)
        img = np.array(img.resize((224,224),Image.ANTIALIAS))
    except:
        plt.imshow(img)
        plt.show()
    if img.shape !=(224,224,3):
        continue
    if idx >= 12000:
        test_batches.append(img)
        test_labels.append(np.array([0,1]))
        if (idx+1) %100 == 0:
            print('狗图片的测试集已完成:{:.2f}%'.format((len(test_batches)/500)*100))
    else:    
        train_batches.append(img)
        train_labels.append(np.array([0,1]))
        if (idx+1) %100 == 0:
            print('狗图片的训练集已完成:{:.2f}%'.format((len(train_batches)/12000)*100))

print('\n')

os.chdir(r'D:\kaggle\data\Cat')
for idx,each in enumerate(cats_dir):
    try:
        img = Image.open(each)
        img = np.array(img.resize((224,224),Image.ANTIALIAS))
    except:
        plt.imshow(img)
        plt.show()
    if img.shape !=(224,224,3):
        continue
    if idx >= 12000:
        test_batches.append(img)
        test_labels.append(np.array([1,0]))
        if (idx+1) %100 == 0: 
            print('猫图片的测试集已完成:{:.2f}%'.format(((len(test_batches)-500)/500)*100))
    else:    
        train_batches.append(img)
        train_labels.append(np.array([1,0]))
        if (idx+1) %100 == 0:
            print('猫图片的训练集已完成:{:.2f}%'.format(((len(train_batches)-12000)/12000)*100))

#打乱数据
train = shuffle(train_batches,train_labels)
test  = shuffle(test_batches,test_labels)
#把dataset以npy格式保存
os.chdir(r'D:\kaggle')
np.save('train_batches.npy',train[0])
np.save('train_labels.npy',train[1])
np.save('test_batches.npy',test[0])
np.save('test_labels.npy',test[1])

              
