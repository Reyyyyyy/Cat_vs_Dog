import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten ,Dense ,Dropout ,GlobalAveragePooling2D
from keras.models import Model,load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.optimizers import Adagrad
import keras

#Loss =  0.0495    Train accuracy =  0.9835    Test accuracy = 0.988

#采用增强的数据进行训练
train_datagen = ImageDataGenerator(rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   rescale = 1/255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   fill_mode = 'nearest')
train = train_datagen.flow_from_directory(r'D:\kaggle\data\for_enhanced_data\train',
                                    target_size=(299,299),
                                    batch_size=96,
                                    class_mode='binary')
#创造测试集
test_batches = (np.load(r'D:\kaggle\test_batches.npy',allow_pickle=True))/255
test_labels = np.load(r'D:\kaggle\test_labels.npy',allow_pickle=True)
#建模(迁移学习InceptionV3)
base_model = InceptionV3(weights=None,include_top=False)
base_model.load_weights(r'D:\kaggle\inceptionV3\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
# 增加新的输出层
global_pooling = GlobalAveragePooling2D()(base_model.output) # GlobalAveragePooling2D 将 MxNxC 的张量转换成 1xC 张量，C是通道数
fc = Dense(512,activation='relu')(global_pooling)
drop = Dropout(rate=0.8)(fc)
output = Dense(1,activation='sigmoid')(drop)
model = Model(inputs=base_model.input,outputs=output)

#编译模型(两步，第一步只训练新层，基础模型冻结；第二步训练基础模型的部分层以及新层，其余层冻结。
def setup_to_transfer_learning(model,base_model):#base_model
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

def setup_to_fine_tuning(model,base_model):
    FROZEN_LAYER = 16 
    for layer in base_model.layers[:FROZEN_LAYER]:
        layer.trainable = False
    for layer in base_model.layers[FROZEN_LAYER:]:
        layer.trainable = True
    model.compile(optimizer=Adagrad(lr=0.0001),loss='binary_crossentropy',metrics=['accuracy'])

#训练和评估模型
def test_model(model,test_batches,test_labels,nums=500):
    preds = (model.predict(test_batches[:nums])).flatten() > 0.5
    labels = np.argmax(test_labels[:nums],axis=1)
    return np.sum(preds == labels)/nums
    
setup_to_transfer_learning(model,base_model)
model.fit_generator(train,
                    samples_per_epoch = 3000,
                    nb_epoch = 3,
                    #validation_data = validation_generator,
                    #nb_val_samples=400)
                    )
print('Test Accuracy = ',test_model(model,test_batches,test_labels,nums=500))
setup_to_fine_tuning(model,base_model)
model.fit_generator(train,
                    samples_per_epoch = 3000,
                    nb_epoch = 3,
                    #validation_data = validation_generator,
                    #nb_val_samples=400)
                    )
print('Test Accuracy = ',test_model(model,test_batches,test_labels,nums=500))
'''
#保存模型
model.save_weights("weights")
with open("model.json", "w") as f:
    f.write(model.to_json())
'''
input()
