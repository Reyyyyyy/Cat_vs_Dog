import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.utils import shuffle

#超参数
lr = 0.0001
batch_size = 24
epochs = 3
view_1 = 5
view_2 = 5
view_3 = 5
view_4 = 5
view_5 = 5
num_filter_1 = 32
num_filter_2 = 64
num_filter_3 = 64
num_filter_4 = 64
num_filter_5 = 64
fc_neuron_num = 1024
use_dropout = True
dropout = 0.5#每个元素被保留的概率
keep_prob = tf.placeholder(tf.float32)

def get_data():
    x_train = np.load(r'C:\Users\tensorflow\Desktop\迁移学习\dateset\train_batches.npy',allow_pickle=True)
    y_train = np.load(r'C:\Users\tensorflow\Desktop\迁移学习\dateset\train_labels.npy',allow_pickle=True)
    return x_train,y_train
    
def conv2d(x,w,b,strides=1):
    x = tf.nn.conv2d(x,w,strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)

    return tf.nn.relu(x)

def maxpool2d(x,k=2):
    x = tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,k,k,1],padding='SAME')
    return x

def conv_net(x,weights,biases,use_dropout):
    
    conv_1 = conv2d(x,weights['wc1'],biases['bc1'])
    conv_1 = maxpool2d(conv_1)
    
    conv_2 = conv2d(conv_1,weights['wc2'],biases['bc2'])
    
    conv_3 = conv2d(conv_2,weights['wc3'],biases['bc3'])
    conv_3 = maxpool2d(conv_3)
    
    conv_4 = conv2d(conv_3,weights['wc4'],biases['bc4'])
    conv_4 = maxpool2d(conv_4)

    conv_5 = conv2d(conv_4,weights['wc5'],biases['bc5'])
    conv_5 = maxpool2d(conv_5)
    
    flatten = tf.reshape(conv_5,[-1,14*14*num_filter_5])

    fc = tf.nn.relu(tf.matmul(flatten,weights['wf'])+biases['bf'])

    if use_dropout:
        fc = tf.nn.dropout(fc,keep_prob)

    out = tf.matmul(fc,weights['out']) + biases['out']

    return out

weights={'wc1':tf.Variable(tf.random.truncated_normal([view_1,view_1,3,num_filter_1],stddev=0.02)),
         'wc2':tf.Variable(tf.random.truncated_normal([view_2,view_2,num_filter_1,num_filter_2],stddev=0.02)/np.sqrt(num_filter_1/2)),
         'wc3':tf.Variable(tf.random.truncated_normal([view_3,view_3,num_filter_2,num_filter_3],stddev=0.02)/np.sqrt(num_filter_2/2)),
         'wc4':tf.Variable(tf.random.truncated_normal([view_4,view_4,num_filter_3,num_filter_4],stddev=0.05)/np.sqrt(num_filter_3/2)),
         'wc5':tf.Variable(tf.random.truncated_normal([view_5,view_5,num_filter_4,num_filter_5],stddev=0.05)/np.sqrt(num_filter_4/2)),
         'wf':tf.Variable(tf.random.truncated_normal([14*14*num_filter_5,fc_neuron_num],stddev=0.04)/np.sqrt(num_filter_3/2)),
         'out':tf.Variable(tf.random.truncated_normal([fc_neuron_num,2],stddev=1/192)/np.sqrt(192/2))
         }

biases={'bc1':tf.Variable(tf.zeros([num_filter_1])),
        'bc2':tf.Variable(tf.zeros([num_filter_2])),
        'bc3':tf.Variable(tf.zeros([num_filter_3])),
        'bc4':tf.Variable(tf.zeros([num_filter_4])),
        'bc5':tf.Variable(tf.zeros([num_filter_5])),
        'bf':tf.Variable(tf.zeros([fc_neuron_num])),
        'out':tf.Variable(tf.zeros([2])),
        }

x = tf.placeholder(tf.float32,[None,224,224,3])
y = tf.placeholder(tf.float32,[None,2])

pred = conv_net(x,weights,biases,use_dropout)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,axis=1),tf.argmax(y,axis=1)),dtype=tf.float32))
init = tf.global_variables_initializer()

if __name__=='__main__':  
    xs,ys = get_data()#变量名千万不能和占位符重名
    train_x = xs[0:20000]
    train_y = ys[0:20000]
    test_x = xs[20000:25000]
    test_y = ys[20000:25000]
    with tf.Session() as sess:
        sess.run(init)
        #Train
        steps = int(epochs*20000/batch_size)
        for step in range(steps):
            pointer = step*batch_size%20000
            x_batch = train_x[pointer:pointer+batch_size]
            y_batch = train_y[pointer:pointer+batch_size]

            sess.run(optimizer,feed_dict={x:x_batch,y:y_batch,keep_prob:dropout})
            loss = sess.run(cost,feed_dict={x:x_batch,y:y_batch,keep_prob:dropout})
            acc = sess.run(accuracy,feed_dict={x:x_batch,y:y_batch,keep_prob:1.0})
            print('loss:',loss)
            print('accuracy:',acc,'\n')
            
        #Evaluate
        avg_acc_test = 0
        for i in range(50):
            acc_test = sess.run(accuracy,feed_dict={x:test_x[i*100:(i+1)*100],y:test_y[i*100:(i+1)*100],keep_prob:1.0})
            print('Test accuracy: ',acc_test)
            avg_acc_test += acc_test

        avg_acc_test = avg_acc_test/50
        print('Done! Average accuracy of test data is: ',avg_acc_test)
        #Save
        saver.save(sess,'my_model.ckpt')
