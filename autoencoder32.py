# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.contrib.slim as slim
learning_rate=0.005
training_epoch=1#500
display_step=2
INPUT_IMG_WIDE,INPUT_IMG_HEIGHT,INPUT_IMG_CHANNEL=128,128,3
LATENT_IMG_WIDE,LATENT_IMG_HEIGHT,LATENT_IMG_CHANNEL=32,32,32
batch_size=5
EPS = 10e-5
total_batch=200
nr_steps=20
D=1
e_sigma=0.25
pred_init=0.1 
theta_init=0.1
pixel_prior = {
        'p': 0.0,               # probability of success for pixel prior Bernoulli
        'mu': 0.0,              # mean of pixel prior Gaussian
        'sigma': 0.25           # std of pixel prior Gaussian
    }
savedir="/home/1_experiment/1-pretrain_model/savemodel32/auto"
summary_dir="/home/1_experiment/1-pretrain_model/logs"

#读取数据
img_path="/home/1-pretrain_model/output.tfrecords10"
img_path = tf.convert_to_tensor(img_path,dtype=tf.string)
data_queue=tf.train.string_input_producer(string_tensor=tf.train.match_filenames_once(img_path),num_epochs=5000,shuffle=True)
img_path2="/home/1-pretrain_model/output.tfrecords6"
img_path2 = tf.convert_to_tensor(img_path2,dtype=tf.string) 
data_queue2=tf.train.string_input_producer(string_tensor=tf.train.match_filenames_once(img_path2),num_epochs=500,shuffle=True)	
#设置占位符
x=tf.placeholder(tf.float32,shape=[None,None,None,3],name='input_images')
#keep_prob = tf.placeholder(tf.float32,name='keep_prob')#dropout的概率，可调整，防止过拟合

#创建变量
weights = {
    'conv1': tf.Variable(tf.truncated_normal([5, 5, INPUT_IMG_CHANNEL, 64],stddev=0.01)),
    'conv2': tf.Variable(tf.random_normal([5, 5, 64, 128],stddev=0.01)),
    'conv3': tf.Variable(tf.random_normal([3, 3, 128, 128],stddev=0.01)),
    'conv4': tf.Variable(tf.random_normal([3, 3, 128, 128],stddev=0.01)), 
    'conv5': tf.Variable(tf.random_normal([3, 3, 128, 128],stddev=0.01)),
    'conv6': tf.Variable(tf.random_normal([3, 3, 128, 128],stddev=0.01)),
    'conv7': tf.Variable(tf.random_normal([3, 3, 128, 128],stddev=0.01)),
    'conv8': tf.Variable(tf.random_normal([3, 3, 128, 128],stddev=0.01)),
    'conv9': tf.Variable(tf.random_normal([3, 3, 128, 32],stddev=0.01)),
    'conv10':tf.Variable(tf.random_normal([3, 3, 32, 64],stddev=0.01)),
    #'trans_conv11':tf.Variable(tf.random_normal([2, 2, 128, 64],stddev=0.01)),
    'conv12':tf.Variable(tf.random_normal([3, 3, 128, 128],stddev=0.01)),
    'conv13':tf.Variable(tf.random_normal([3, 3, 128, 128],stddev=0.01)),
    'conv14':tf.Variable(tf.random_normal([3, 3, 128, 128],stddev=0.01)),
    'conv15':tf.Variable(tf.random_normal([3, 3, 128, 128],stddev=0.01)),
    'conv16':tf.Variable(tf.random_normal([3, 3, 128, 128],stddev=0.01)),
    'conv17':tf.Variable(tf.random_normal([3, 3, 128, 128],stddev=0.01)),
    'conv18':tf.Variable(tf.random_normal([3, 3, 128, 32],stddev=0.01)),
    #'trans_conv19':tf.Variable(tf.random_normal([2, 2, 256, 32],stddev=0.01)),
    'conv20':tf.Variable(tf.random_normal([3, 3, 256, 16],stddev=0.01)),
    'conv21':tf.Variable(tf.random_normal([3, 3, 16, 3],stddev=0.01))
}


#创建模型


def read_image_batch(file_queue, batch_size):
    reader=tf.TFRecordReader()
    _,image=reader.read(file_queue)
    features = tf.parse_single_example(
            image,
            features={
                    'image_raw':tf.FixedLenFeature([],tf.string)
                    })
    img=tf.decode_raw(features['image_raw'],tf.uint8)
    return img,features

def copy_and_crop_and_merge(result_from_contract_layer,result_from_upsampling):
    result_from_contract_layer_crop=result_from_contract_layer
    return tf.concat(values=[result_from_contract_layer_crop,result_from_upsampling],axis=-1)
    
def batch_norm(x, eps=EPS,affine=True, name='BatchNorm2d'):
    mean_this_batch, variance_this_batch = tf.nn.moments(x, list(range(len(x.shape) - 1)), name='moments')
    mean = tf.identity(mean_this_batch)
    variance = tf.identity(variance_this_batch)
    normed = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=None, scale=None,  variance_epsilon=eps)
    return normed

        
def conv2d(X, W):
    conv=tf.nn.conv2d(X, W, strides=[1, 1, 1, 1],padding='SAME') 
    normed_batch=tf.contrib.layers.batch_norm(conv,decay=0.9)
    return normed_batch

def conv2d2(X,W):
    conv=tf.nn.conv2d(X, W, strides=[1, 2, 2, 1],padding='SAME') 
    normed_batch=tf.contrib.layers.batch_norm(conv,decay=0.9)
    return normed_batch
    
'''def trans_conv2d(X, W,output_shape):
    transconv=tf.nn.conv2d_transpose(X, W,output_shape,strides=[1, 2, 2, 1],padding='VALID')
    normed_batch=tf.contrib.layers.batch_norm(transconv,decay=0.9)
    return normed_batch'''
def trans_conv2d(X, filters, kernel_size):
    transconv=tf.layers.conv2d_transpose(X, filters, kernel_size,strides=(2, 2),padding='valid')
    normed_batch=tf.contrib.layers.batch_norm(transconv,decay=0.9)
    return normed_batch
def encoder(X):
    normed_batch1=X
    h_conv1 = tf.nn.leaky_relu(conv2d2(normed_batch1, weights['conv1']))
    h_conv2 = tf.nn.leaky_relu(conv2d2(h_conv1,weights['conv2']))
    h_conv3 = tf.nn.leaky_relu(conv2d(h_conv2,weights['conv3']))
    h_conv4 = conv2d(h_conv3,weights['conv4'])
    h_merge1= h_conv4+h_conv2
    h_conv5 = tf.nn.leaky_relu(conv2d(h_merge1,weights['conv5']))
    h_conv6 = conv2d(h_conv5,weights['conv6'])
    h_merge2= h_conv6+h_merge1
    h_conv7 = tf.nn.leaky_relu(conv2d(h_merge2,weights['conv7']))
    h_conv8 = conv2d(h_conv7,weights['conv8'])
    h_merge3= h_conv8+h_merge2
    h_middle= tf.tanh(conv2d(h_merge3,weights['conv9']))
    return h_middle

def decoder(Y):
    h_conv10 = tf.nn.leaky_relu(conv2d(Y,weights['conv10']))
    h_transconv11 = trans_conv2d(h_conv10,filters=128,kernel_size=(2,2))
    h_conv12 = tf.nn.leaky_relu(conv2d(h_transconv11,weights['conv12']))
    h_conv13 = conv2d(h_conv12,weights['conv13'])
    h_merge4 = h_transconv11+h_conv13
    h_conv14 = tf.nn.leaky_relu(conv2d(h_merge4,weights['conv14']))
    h_conv15 = conv2d(h_conv14,weights['conv15'])
    h_merge5 = h_conv15+h_merge4
    h_conv16 = tf.nn.leaky_relu(conv2d(h_merge5,weights['conv16']))
    h_conv17 = conv2d(h_conv16,weights['conv17'])
    h_merge6 = h_conv17+h_merge5
    h_conv18 = tf.nn.leaky_relu(conv2d(h_merge6,weights['conv18']))
    h_transconv19 = trans_conv2d(h_conv18,filters=256,kernel_size=(2,2))
    h_conv20 = tf.nn.leaky_relu(conv2d(h_transconv19,weights['conv20']))
    h_conv21 = tf.tanh(conv2d(h_conv20,weights['conv21']))
    return h_conv21


#输出的节点
middle_out = encoder(x)
prediction = decoder(middle_out)
loss=tf.reduce_mean(tf.pow(x-prediction,2))
merged_summary_op=tf.summary.scalar('loss_function',loss)
#创建反向传播算法
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(loss) 
#训练模型  
img,features=read_image_batch(data_queue,batch_size)
img=tf.image.convert_image_dtype(img,tf.float32)
img=tf.reshape(img,[INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL])
train_image= tf.train.shuffle_batch(tensors=[img],batch_size=5,capacity=5000,min_after_dequeue=5)
#测试数据
img_test,features_test=read_image_batch(data_queue2,batch_size)
img_test=tf.image.convert_image_dtype(img_test,tf.float32)
img_test=tf.reshape(img_test,[INPUT_IMG_WIDE,INPUT_IMG_HEIGHT,INPUT_IMG_CHANNEL])
test_image= tf.train.shuffle_batch(tensors=[img_test],batch_size=5,capacity=500,min_after_dequeue=5)
#prediction = tf.image.convert_image_dtype(prediction,dtype=tf.uint8)  
print(features)
print(tf.shape(img))  
print(train_image) 
saver=tf.train.Saver(max_to_keep=1)     #生成saver                      
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
    with tf.device("/gpu:0"):        
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        summary_writer=tf.summary.FileWriter(summary_dir,sess.graph)
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)#启动线程
        try:
            for epoch in range(training_epoch):
            #while not coord.should_stop():
                for i in range(total_batch):
                    img_numpy = sess.run(train_image)
                    _,c=sess.run([optimizer,loss],feed_dict={x:img_numpy})
                    reconstruction_train = sess.run(prediction, feed_dict={x: img_numpy})
                    code_train=sess.run(middle_out,feed_dict={x:img_numpy})
                    if epoch % display_step == 0:
                        print("Epoch:",'%04d' % (epoch+1),"cost=","{:.9f}".format(c))
                    saver.save(sess,savedir+"encoder32.cpkt",global_step=epoch)     #保存模型
                    summary=sess.run(merged_summary_op,feed_dict={x:img_numpy});
                summary_writer.add_summary(summary,epoch);
            img_test=sess.run(test_image)
            reconstruction_test = sess.run(prediction, feed_dict={x: img_test})
            encoder =sess.run(middle_out,feed_dict={x:img_test})
            show_num=5
            f, a = plt.subplots(2, 5, figsize=(5, 2))
            for i in range(show_num):
                a[0][i].imshow(img_test[i])
                a[1][i].imshow(reconstruction_test[i])
            plt.imshow(reconstruction_train[1])
            plt.imshow(encoder[1,:,:,-1])
            plt.show()
        except tf.errors.OutOfRangeError:
            print("Done training -- epoch limit reached")
        finally:
            coord.request_stop()
        coord.join(threads)    
summary_writer.close()        
