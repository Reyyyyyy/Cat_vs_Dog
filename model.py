import tensorflow as tf
import numpy as np

#引入vgg16
vgg16 = (np.load('vgg16.npy',allow_pickle=True,encoding='bytes')).tolist()

#超参数
lr=0.001
batch_size=10#batch_size不能太大，不然内存会挤爆，程序强制重启
epochs=3

def get_train_batch():
    x_train = np.load(r'C:\Users\tensorflow\Desktop\迁移学习\dateset\train_batches.npy',allow_pickle=True)
    y_train = np.load(r'C:\Users\tensorflow\Desktop\迁移学习\dateset\train_labels.npy',allow_pickle=True)
    return x_train,y_train

def get_test_batch():
    return np.load(r'C:\Users\tensorflow\Desktop\迁移学习\dateset\test_batches.npy',allow_pickle = True)
    #缺少测试集标签

def conv2d(x,w,b,strides=1):
    x = tf.nn.conv2d(x,w,strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

def maxpool2d(x,strides=2):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,strides,strides,1],padding='SAME')

def VGG16(x):
    conv1_1 = conv2d(x,vgg16[b'conv1_1'][0],vgg16[b'conv1_1'][1])
    conv1_2 = conv2d(conv1_1,vgg16[b'conv1_2'][0],vgg16[b'conv1_2'][1])
    mp1 = maxpool2d(conv1_2)
    conv2_1 = conv2d(mp1,vgg16[b'conv2_1'][0],vgg16[b'conv2_1'][1])
    conv2_2 = conv2d(conv2_1,vgg16[b'conv2_2'][0],vgg16[b'conv2_2'][1])
    mp2 = maxpool2d(conv2_2)
    conv3_1 = conv2d(mp2,vgg16[b'conv3_1'][0],vgg16[b'conv3_1'][1])
    conv3_2 = conv2d(conv3_1,vgg16[b'conv3_2'][0],vgg16[b'conv3_2'][1])
    conv3_3 = conv2d(conv3_2,vgg16[b'conv3_3'][0],vgg16[b'conv3_3'][1])
    mp3 = maxpool2d(conv3_3)
    conv4_1 = conv2d(mp3,vgg16[b'conv4_1'][0],vgg16[b'conv4_1'][1])
    conv4_2 = conv2d(conv4_1,vgg16[b'conv4_2'][0],vgg16[b'conv4_2'][1])
    conv4_3 = conv2d(conv4_2,vgg16[b'conv4_3'][0],vgg16[b'conv4_3'][1])
    mp4 = maxpool2d(conv4_3)
    conv5_1 = conv2d(mp4,vgg16[b'conv5_1'][0],vgg16[b'conv5_1'][1])
    conv5_2 = conv2d(conv5_1,vgg16[b'conv5_2'][0],vgg16[b'conv5_2'][1])
    conv5_3 = conv2d(conv5_2,vgg16[b'conv5_3'][0],vgg16[b'conv5_3'][1])
    mp5 = maxpool2d(conv5_3)
    xed = mp5
    return xed
    
def Tail(x,weights,biases):
    xed = VGG16(x)
    flatten = tf.reshape(xed,[-1,7*7*512])
    fc1 = tf.nn.relu(tf.matmul(flatten,weights['wf1'])+biases['bf1'])
    fc2 = tf.nn.relu(tf.matmul(fc1,weights['wf2'])+biases['bf2'])
    out = tf.matmul(fc2,weights['out']) + biases['out']
    return out

weights={'wf1':tf.Variable(tf.random.truncated_normal([7*7*512,512],stddev=0.04)),\
         'wf2':tf.Variable(tf.random.truncated_normal([512,1024],stddev=0.04))/16,\
         'out':tf.Variable(tf.random.truncated_normal([1024,2],stddev=0.04))/np.sqrt(512)}

biases={'bf1':tf.zeros([512]),\
        'bf2':tf.zeros([1024]),\
        'out':tf.zeros([2])}

x = tf.placeholder(tf.float32,[None,224,224,3])
y = tf.placeholder(tf.float32,[None,2])

pred = Tail(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,axis=1),tf.argmax(y,axis=1)),dtype=tf.float32))
init = tf.global_variables_initializer()

if __name__=='__main__':  
    xs,ys = get_train_batch()#变量名千万不能和占位符重名
    train_x = xs[0:20000]
    train_y = ys[0:20000]
    test_x = xs[20000:25000]
    test_y = ys[20000:25000]
    with tf.Session() as sess:
        sess.run(init)
        #Train
        for step in range(int(epochs*25000/batch_size)):
            pointer = step*batch_size%25000
            x_batch = train_x[pointer:pointer+batch_size]
            y_batch = train_y[pointer:pointer+batch_size]

            sess.run(optimizer,feed_dict={x:x_batch,y:y_batch})
            loss = sess.run(cost,feed_dict={x:x_batch,y:y_batch})
            acc = sess.run(accuracy,feed_dict={x:x_batch,y:y_batch})
            print('loss:',loss)
            print('accuracy:',acc,'\n')
        #Evaluate
        avg_acc_test = 0
        for i in range(int(test_batches[0]/100)):
            acc_test = sess.run(accuracy,feed_dict={x:test_x,y:test_y})
            print('Test accuracy: ',acc_test)
            avg_acc_test += acc_test

        avg_acc_test = avg_acc_test/50
        print('Done! Average accuracy of test data is: ',avg_acc_test)

    #保存模型变量，注意json不接受numpy的array,要变成list
    '''
    with open ('weights.json','w') as f:
        ws = {}
        for name,w in weights.items():
            ws[name] = sess.run(w).tolist()
        json.dump(ws,f)

    with open ('biases.json','w') as f:
        bs = {}
        for name,b in biases.items():
            bs[name] = sess.run(b).tolist()
        json.dump(bs,f)
    '''
