import numpy as np

vgg16 = (np.load('vgg16.npy',allow_pickle=True,encoding='bytes')).tolist()
for name,params in vgg16.items():
    weight = params[0]
    bias = params[1]
    print('name:',str(name).replace('b','').replace('\'',''),\
          '  weights\'shape:',str(weight.shape),\
          '  bias\'shape',str(bias.shape))
