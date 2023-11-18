# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#from soft_n_cut_loss import soft_n_cut_loss
import scipy.misc
import torch
from sklearn.utils.linear_assignment_ import linear_assignment
import datetime
import cv2
from sklearn.cluster import KMeans
from metrics import Results
import sklearn.metrics
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
learning_rate=0.0001
training_epoch=1
display_step=1
INPUT_IMG_WIDE,INPUT_IMG_HEIGHT,INPUT_IMG_CHANNEL=128,128,3
LATENT_IMG_WIDE,LATENT_IMG_HEIGHT,LATENT_IMG_CHANNEL=64,64,99
maxIter=50
batch_size=1
batch_size1=5
EPS = 1e-6
#total_batch=1000
total_batch1=20
nr_steps=3
K=3#components的数目
#K1=3
minlabels=2
D=1
e_sigma=0.25
pred_init=0.0
theta_init=0.0
momentum=0.9
c_parameter=0.05
savedir="/home/savemodel/1"
summary_dir="/home/logs"
#读取training数据
#savepath2="/home/1_experiment/different_noise/0.2/img"
#savepath3="/home/1_experiment/different_noise/0.2/pred"
img_path1="/home/tfrecoderscocofew"
img_path1 = tf.convert_to_tensor(img_path1,dtype=tf.string)
data_queue1=tf.train.string_input_producer(string_tensor=tf.train.match_filenames_once(img_path1),num_epochs=1000,shuffle=False)
#restore model
def read_image_batch(file_queue, batch_size1):
    reader=tf.TFRecordReader()
    _,image=reader.read(file_queue)
    features = tf.parse_single_example(
            image,
            features={
                    'image_raw':tf.FixedLenFeature([],tf.string),
                    'label':tf.FixedLenFeature([],tf.string),
                    'mask':tf.FixedLenFeature([],tf.string)
                    })
    img=tf.decode_raw(features['image_raw'],tf.uint8)
    label=tf.decode_raw(features['label'],tf.uint8)
    mask=tf.decode_raw(features['mask'],tf.uint8)
    return img,label,mask

#em algorithm
class EMCell():
    def __init__(self,a,distribution):
        self.a=a#[w,h,c]
        self.distribution=distribution
     
    def init_state(self,batch_size,k,dtype):
        #2.initial prediction
        with tf.name_scope('pred_init'):
            pred_shape=tf.stack([batch_size,k,1,LATENT_IMG_CHANNEL])  #[B,K,1,C]
            #pred =tf.Variable(tf.abs(tf.random_normal(shape=pred_shape, dtype=dtype)))
            pred=tf.Variable(tf.truncated_normal(pred_shape, mean=0.0, stddev=tf.sqrt(2/k), dtype=tf.float32))
            pred=pred/(EPS+tf.norm(pred,axis=3,keep_dims=True))#l2 norm
            #pred1=tf.reduce_mean(X,axis=2,keep_dims=True)#[B,1,1,C]
            #pred=tf.tile(pred1,multiples=[1,K,1,1]) #[B,K,1,C]
            
            
        with tf.name_scope('pis_init'):
            shape = tf.stack([batch_size,k,1,1]) #[B,K,1,1]
            pis=tf.Variable(tf.random_uniform(shape,0,1,dtype=dtype))
            
        with tf.name_scope('cov'):
            #shape=tf.stack([batch_size,k,1,LATENT_IMG_CHANNEL])#[B,K,1,C,C]
            #cov=tf.Variable(tf.matrix_diag(tf.ones(shape,dtype=dtype)))
            cov=tf.Variable(tf.eye(LATENT_IMG_CHANNEL,batch_shape=[batch_size,k,1])*0.1)
            #cov=tfp.stats.covariance(X,X,sample_axis=2,event_axis=-1)
            #cov=tf.expand_dims(cov,1)#[B,1,1,C,C]
            #cov=tf.tile(cov,multiples=[1,K,1,1,1])#[B,K,1,C,C]
           
        with tf.name_scope('gamma_init'):           
            shape = tf.stack([batch_size,k,self.a[0]*self.a[1],1]) #[b,k,W*H,1]
            gamma=tf.abs(tf.random_normal(shape,dtype=dtype))
            gamma /=tf.reduce_sum(gamma,1,keep_dims=True) #在K上加和
            # init with all 1 if K = 1
            if k == 1:
                gamma = tf.ones_like(gamma)
            gamma=tf.Variable(gamma)
       
        return pis,pred,cov,gamma
    def logdet(self,a):
        res=tf.py_func(lambda a:np.linalg.slogdet(a)[1],
                          [a],tf.float32)
        return res
    def MultivariatieNormalDis(self,pred,cov,inputs,k):
        regular_factor=tf.matrix_diag(tf.ones([batch_size,k,LATENT_IMG_CHANNEL]))*0.03
        cov1=tf.squeeze(cov,2)+regular_factor#[B,K,C,C]
        distance=inputs-pred#[B,K,W*H,C]
        cov_inverse=tf.py_func(np.linalg.inv, [cov1], tf.float32)#[B,K,C,C]        
        distance1=tf.matmul(distance,cov_inverse)#[B,K,W*H,C]
        distance2=tf.multiply(distance1,distance)
        m_distance=tf.reduce_sum(distance2,3,keep_dims=True)#[B,K,W*H,1]
        log_det_cov0=self.logdet(cov1)
        log_det_cov=tf.reshape(log_det_cov0,(batch_size,k,1,1))
        return -0.5*(m_distance+log_det_cov+(LATENT_IMG_CHANNEL*tf.log(2*np.pi)))#(B,K,W*H,1)
   
        
    def compute_em_probabilities(self,pred,data,cov,pis,k):
        with tf.name_scope('em_loss_{}'.format(self.distribution)):
            if self.distribution =='gaussian':                
                probability=self.MultivariatieNormalDis(pred,cov,data,k)
                probabilitys=tf.exp(probability)
            else:
                raise ValueError(
                        'Unknown distribution_type:"{}"'.format(self.distribution))
            
            return probabilitys #[B,K,w*h,1]
    
    #E-step    
    def e_step(self,pred,pis,data,cov,k):
        with tf.name_scope('e_step'):
            probs=self.compute_em_probabilities(pred,data,cov,pis,k)#(B,K,W*H,1)
            gamma=pis*probs / (tf.reduce_sum(pis*probs,1,keep_dims=True))#在K上加和
            return gamma  #[B,K,W*H,1]
    def update_pis(self,gamma):        
        #pis
        pis_update=tf.reduce_sum(gamma,axis=2,keep_dims=True)/(LATENT_IMG_WIDE*LATENT_IMG_HEIGHT) #[B,K,1,1]
        return pis_update#[B,K,1,1]
            
    def update_pred(self,inputs,gamma,k): 
        mu_update=tf.reduce_sum(gamma*inputs,axis=2,keep_dims=True)/(tf.reduce_sum(gamma,axis=2,keep_dims=True))
        mu_update=mu_update/(EPS+tf.norm(mu_update,axis=3,keep_dims=True))#l2 norm
        return mu_update
    
    def update_cov(self,inputs,gamma,pred,k):
        gamma1=tf.expand_dims(gamma,4) #(B,K,W*H,1,1)
        a=tf.expand_dims(inputs-pred,4) #(B,K,W*H,C,1)
        a1=tf.matmul(a,tf.transpose(a,[0,1,2,4,3]))  #(B,K,W*H,C,C)
        covs=tf.reduce_sum(gamma1*a1,axis=2,keep_dims=True)
        cov_update=covs/(tf.reduce_sum(gamma1,axis=2,keep_dims=True))
        return cov_update #(B,K,1,C,C)

    
    def m_step(self,inputs,pred,pis,cov,k):
        gamma_update=self.e_step(pred,pis,inputs,cov,k) #[B,K,W*H,1]
        pis_update=self.update_pis(gamma_update)
        pred_update=self.update_pred(inputs,gamma_update,k)
        cov_update=self.update_cov(inputs,gamma_update,pred_update,k)
        return pis_update,pred_update,cov_update,gamma_update
    
         
    def __call__(self,data,state,k,scope=None):
        input_data=data
        pis_old,pred_old,cov_old,gamma_old=state        
        pis,pred,cov,gamma=self.m_step(input_data,pred_old,pis_old,cov_old,k)
        outputs=(pis,pred,cov,gamma)
        return outputs,outputs

def MultivariatieNormalDis(pred,cov,inputs,k):
        cov1=tf.squeeze(cov,2)+tf.matrix_diag(tf.ones([batch_size,k,LATENT_IMG_CHANNEL]))*0.05#[B,K,C,C]
        #cov1=tf.matrix_diag(tf.ones([batch_size,k,LATENT_IMG_CHANNEL]))
        distance=inputs-pred#[B,K,W*H,C]
        cov_inverse=tf.matrix_inverse(cov1)
        distance1=tf.matmul(distance,cov_inverse)#[B,K,W*H,C]
        distance2=tf.multiply(distance1,distance)
        m_distance=tf.reduce_sum(distance2,3,keep_dims=True)
        log_det_cov0=tf.linalg.logdet(cov1)     
        log_det_cov=tf.reshape(log_det_cov0,(batch_size,k,1,1))
        return -0.5*(m_distance+log_det_cov+(LATENT_IMG_CHANNEL*tf.log(2*np.pi)))#(B,K,W*H,1)
        
def compute_log_likelihood(mu,cov,pis,inputs,k,EPS):
    with tf.name_scope('Q_function'):
        probs=tf.exp(MultivariatieNormalDis(mu,cov,inputs,k))
        probability=tf.reduce_sum(pis*probs,1,keep_dims=True)#[B,1,W*H,1]
        log_likelihood=tf.reduce_mean(tf.log(probability+EPS))
        return -log_likelihood
def kmeans_init(img, k):
    kmeans_model = KMeans(init='k-means++', n_clusters=k, n_init=10, max_iter=300, n_jobs=15,
                              random_state=42)
    labels=kmeans_model.fit_predict(img)
    centroids = kmeans_model.cluster_centers_.T
    means = centroids / np.sqrt(np.diag(np.matmul(centroids.T, centroids)))
    means=np.transpose(means,(1,0))#(K,C)
    try:
        means = np.array(means)
        cov = np.array([np.cov(img[labels == i].T) for i in range(k)], dtype=np.float32)
        ids = set(labels)
        pis = np.array([np.sum([labels == i]) / len(labels) for i in ids], dtype=np.float32)
    except Exception as ex:
        pass
    return means,cov,pis#,labels
def KMeans_init(x,k):
    x1=tf.reshape(x,(LATENT_IMG_WIDE*LATENT_IMG_HEIGHT,LATENT_IMG_CHANNEL))#[W*H,C]
    center,cov,pis=tf.py_func(kmeans_init,[x1,k],[tf.float32,tf.float32,tf.float32])
    mu=tf.reshape(center,(batch_size,k,1,LATENT_IMG_CHANNEL))#[1,K,1,C]
    sigma=tf.reshape(cov,(batch_size,k,1,LATENT_IMG_CHANNEL,LATENT_IMG_CHANNEL))#[B,K,1,C,C]
    p=tf.reshape(pis,(batch_size,k,1,1))#[B,K,1,1]    
    return p,mu,sigma
def moving_average(x,momentum):   
    with tf.name_scope('MovingAverage'):
        x1=tf.reduce_mean(x,axis=0,keep_dims=True)
        x2=momentum*x+(1-momentum)*x1
        return x2    
def static_em_iterations(inputs,k,nr_steps,batch_size):
    W, H, C =LATENT_IMG_WIDE,LATENT_IMG_HEIGHT,LATENT_IMG_CHANNEL
    em_cell=EMCell([W,H,C],distribution='gaussian')     
    with tf.name_scope('initial_state'):
        hidden_state_0 = em_cell.init_state(batch_size,k,dtype=tf.float32)
        pis_0,pred_0,cov_0,gamma_0=hidden_state_0
        pis_1,pred_1,cov_1=KMeans_init(inputs[0],k)
        hidden_state=pis_1,pred_1,cov_1,gamma_0
        #hidden_state=KMeans_init(inputs[0],K)
    #build static iteration
    outputs = [hidden_state]
    em_losses=[]
    with tf.variable_scope('EM') as varscope:       
        for t in range(nr_steps):
            varscope.reuse_variables() if t > 0 else None #share weights across time
            with tf.name_scope('step_{}'.format(t)):
                input_data=inputs[t]
                hidden_state,output=em_cell(input_data,hidden_state,k)
                pis,pred,cov,gamma=output
                em_loss=compute_log_likelihood(pred,cov,pis,input_data,k,EPS=1e-6)
            em_losses.append(em_loss)
            outputs.append(output)           
    with tf.name_scope('collect_outputs'):
        pises,preds,covs,gammas0= zip(*outputs)
        pises = tf.stack(pises)                 # (T,B,K,1,1)
        preds = tf.stack(preds)                 # (T,B,K,1,C)
        covs = tf.stack(covs)                   # (T,B,K,1,C,C)
        gammas0 = tf.stack(gammas0)               # (T, B,K, W*H,1)
    with tf.name_scope('em_loss'):        
        EM_loss=tf.reduce_sum(tf.stack(em_losses))
    with tf.name_scope('likelihood'):
        likelihood=-em_loss
    return EM_loss, covs, preds, pises,gammas0,likelihood

def correct(g,k):
    g=tf.squeeze(g,4)#[T,B,K,W*H]
    g1=tf.transpose(g[nr_steps],(0,2,1))#[B,W*H,K]
    g2=tf.reshape(g1,[batch_size,LATENT_IMG_WIDE,LATENT_IMG_HEIGHT,k])
    h=tf.image.resize_images(g2,[128,128],method=0)
    return h


def entropy_loss_batch(p,k):
    p1=tf.reshape(p,(batch_size,INPUT_IMG_WIDE*INPUT_IMG_HEIGHT,k))
    p2=tf.reduce_mean(p1,axis=1,keep_dims=True)#(B,1,K)
    p3=p2*tf.log(p2)#(B,1,K)    
    L=tf.reduce_sum(p3,axis=2,keep_dims=True)#(B,1,1)
    return L

def hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
  assert (isinstance(flat_preds, torch.Tensor) and
          isinstance(flat_targets, torch.Tensor))
  num_samples = flat_targets.shape[0]

  assert (preds_k == targets_k)  # one to one
  num_k = preds_k
  num_correct = np.zeros((num_k, num_k))

  for c1 in range(num_k):
    for c2 in range(num_k):
      # elementwise, so each sample contributes once
      votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
      num_correct[c1, c2] = votes

  # num_correct is small
  match = linear_assignment(num_samples - num_correct)

  # return as list of tuples, out_c to gt_c
  res = []
  for out_c, gt_c in match:
    res.append((out_c, gt_c))

  return res

def select_label(label,mask):
    mask1=mask.astype(np.bool)#(W,H)
   # mask1=mask.astype(np.uint8)
    label2=torch.from_numpy(label)
    mask2=torch.from_numpy(mask1)
    new_label=label2.masked_select(mask2)#(num_unmasked)
    return new_label#(num_unmasked)
def select_pred(out,mask):
    mask1=mask.astype(np.bool)#(W,H)
    out1,mask2=torch.from_numpy(np.asarray(out)),torch.from_numpy(mask1)
    out2=out1.masked_select(mask2)
    return out2 #(num_unmasked)
def acc(y_true,y_pred,k):
    max_num_samples=INPUT_IMG_WIDE*INPUT_IMG_HEIGHT
    preds_all = torch.zeros(max_num_samples, dtype=torch.int32)
    targets_all = torch.zeros(max_num_samples, dtype=torch.int32)
    num_unmasked=y_true.shape[0]
    preds_all[0:num_unmasked]=y_pred#(num_unmasked)
    targets_all[0:num_unmasked]=y_true#(num_unmasked)
    preds_all=preds_all[:num_unmasked]
    targets_all=targets_all[:num_unmasked]
    match=hungarian_match(preds_all,targets_all,k,k)
    reordered_preds = torch.zeros(num_unmasked, dtype=preds_all.dtype)
    preds=reordered_preds.numpy()
    targets=targets_all.numpy()
    for pred_i,target_i in match:
        selected=(preds_all==pred_i)
        preds[selected]=target_i 
    if float(preds.shape[0])==0:
    #if len(np.unique(preds))<=2:
        acc=0.5
    else:
        acc=int((preds==targets).sum())/float(preds.shape[0])
    return acc
def per_img_acc(label,pred,mask,k):
    unmasked_label=select_label(label,mask)#(num_unmasked)
    unmasked_pred=select_pred(pred,mask)#(num_unmasked)
    acc1=acc(unmasked_label,unmasked_pred,k)
    return acc1
def batch_acc(target,pred,mask,k):
    acc_batch=0
    for i in range(batch_size1):
        acc=per_img_acc(target[i][:,:,0],pred[i],mask[i],k)
        acc_batch+=acc
    return acc_batch
def select_unmask(label,pred,mask):
    unmasked_label=select_label(label,mask)#(num_unmasked)
    unmasked_pred=select_pred(pred,mask)#(num_unmasked)
    #print('unmasked_label_list:',np.unique(unmasked_label))#[0,1,2]
    #print('unmasked_pred_list:',np.unique(unmasked_pred))#[0,1,2]
    max_num_samples=INPUT_IMG_WIDE*INPUT_IMG_HEIGHT
    preds_all = torch.zeros(max_num_samples, dtype=torch.int32)
    targets_all = torch.zeros(max_num_samples, dtype=torch.int32)
    num_unmasked=unmasked_label.shape[0]
    preds_all[0:num_unmasked]=unmasked_pred#(num_unmasked)
    targets_all[0:num_unmasked]=unmasked_label#(num_unmasked)
    preds_all=preds_all[:num_unmasked].numpy()
    targets_all=targets_all[:num_unmasked].numpy()
    return targets_all,preds_all
# Compute confusion matrix
def confusion_matrix(act_labels, pred_labels):
    uniqueLabels = list(set(act_labels))
    clusters = list(set(pred_labels))
    cm = [[0 for i in range(len(clusters))] for i in range(len(uniqueLabels))]
    for i, act_label in enumerate(uniqueLabels):
        for j, pred_label in enumerate(pred_labels):
            if act_labels[j] == act_label:
                cm[i][pred_label] = cm[i][pred_label] + 1
    return cm
def cmat_to_psuedo_y_true_and_y_pred(y_true,y_pred):
  """
  Convert a confusion matrix to psuedo y_true and y_pred
  :param cmat: Confusion matrix
  :return: psuedo y_true and y_pred
  """
  print('y_true_list:',np.unique(y_true))
  print('y_pred_list:',np.unique(y_pred))
  cmat=confusion_matrix(y_true,y_pred)
  y_true = []
  y_pred = []
  for true_class, row in enumerate(cmat):
    for pred_class, elm in enumerate(row):
      y_true.extend([true_class] * elm)
      y_pred.extend([pred_class] * elm)
  return y_true, y_pred
def gaussian_noise(image,mean,var):
    #image=np.array(image/255,dtype=float)
    noise=np.random.normal(mean,var,image.shape)
    out=image+noise
    if out.min()<0:
        low_clip=-1
    else:
        low_clip=0
    out=np.clip(out,low_clip,1.0)
    #out=np.uint8(out*255)
    return out
def gaussian_noise_batch(test_img_numpy,mean,var):
    test_img_all=[]
    for i in range(batch_size1):
        test_img=gaussian_noise(test_img_numpy[i],mean,var)
        test_img_all.append(test_img)
    test_img_all1=np.stack(test_img_all)#(B,W,H,C)
    return test_img_all1
def corrupt_mask(test_img_numpy,mask):
    test_img_all=[]
    for i in range(batch_size1):
        test_img=test_img_numpy[i]*mask
        test_img_all.append(test_img)
    test_img_all1=np.stack(test_img_all)
    return test_img_all1
saver=tf.train.import_meta_graph('/home/savemodel/autoencoder32.cpkt-499.meta')
middle_out=tf.get_default_graph().get_tensor_by_name("Tanh:0")#(5,32,32,32)
prediction=tf.get_default_graph().get_tensor_by_name("Tanh_1:0")
h_conv1=tf.get_default_graph().get_tensor_by_name("LeakyRelu:0")#(5,64,64,64)
h_conv1_norm=h_conv1/(EPS+tf.norm(h_conv1,axis=3,keep_dims=True))
x=tf.get_default_graph().get_tensor_by_name("input_images:0")  
middle_up=tf.image.resize_images(middle_out,[64,64],method=0)#(5,64,64,32)
middle_up_norm=middle_up/(EPS+tf.norm(middle_up,axis=3,keep_dims=True))
x_reshape=tf.image.resize_images(x,[64,64],method=0)#(5,64,64,3)
x_reshape_norm=x_reshape/(EPS+tf.norm(x_reshape,axis=3,keep_dims=True))
code=tf.concat([middle_up_norm,h_conv1_norm,x_reshape_norm],axis=3)     #(5,64,64,99) 
w=tf.shape(code)
middle=tf.reshape(code,tf.stack([w[0],w[1]*w[2],w[3]]))#[B,W*H,64]
middle_0=tf.expand_dims(middle,1)#[B,1,W*H,C]
middle1=tf.expand_dims(middle_0,0)
middle2=tf.tile(middle1,multiples=[nr_steps+1,1,1,1,1])#[T+1,B,1,W*H,C]
reconstruction_loss=tf.reduce_mean(tf.pow(x-prediction,2))
x1=tf.reduce_mean(x,3)#[B,W,H]
x2=tf.reshape(x1,[-1,INPUT_IMG_WIDE*INPUT_IMG_HEIGHT])#[-1,128*128]

#train
def Per_image_train(middle2,k,sess,batch_acc_sum1,Ari,Nmi,Ri,Vi,time):
    L=[]
    test_img_numpy,test_label_numpy,test_mask_numpy = sess.run([test_image,test_labels,test_masks]) 
    print("test_mask_numpy:",(test_mask_numpy.min(),test_mask_numpy.max()))
    #Add gaussian noise to input images
    #print('test_img_numpy_shape:',test_img_numpy.shape)#(5,128,128,3)
    #print('test_img_numpy_range:',(test_img_numpy.min(),test_img_numpy.max()))#(0,1)
    #test_img_numpy=gaussian_noise_batch(test_img_numpy,mean=0,var=0.45)
    #test_img_numpy=corrupt_mask(test_img_numpy,mask)
    results = Results()
    results.initialization()  
    for i in range(batch_size1):
        EM_loss, covs, preds, pises,gammas,likelihood=static_em_iterations(middle2[:,i,:,:,:],k,nr_steps,batch_size)
        gammas1=correct(gammas,K)#[1,W,H,K]
        entropy_loss=entropy_loss_batch(gammas1,K)  
        loss=EM_loss+(0.1*entropy_loss)+(0.5*reconstruction_loss)
        optimizer=tf.train.AdamOptimizer(learning_rate,name='adam').minimize(loss)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        starttime = datetime.datetime.now()   
        for j in range(maxIter):
            _,e,gammas_test=sess.run([optimizer,loss,gammas1],feed_dict={x:test_img_numpy})
            if epoch % display_step == 0:
                print("Iteration:",'%04d' % (j+1),"Batch:",'%03d' %(i+1),"Loss=","{:.6f}".format(np.mean(e)))
        endtime = datetime.datetime.now()
        time_per=(endtime-starttime).seconds
        time+=time_per
        print ("Per_image_time:",(endtime - starttime).seconds)
        L1=np.argmax(gammas_test,axis=3)#(1,W,H)
        nlabels=len(np.unique(L1))
        per_acc=per_img_acc(test_label_numpy[i][:,:,0],L1[0],test_mask_numpy[i],k)
        unmasked_label,unmasked_pred=select_unmask(test_label_numpy[i][:,:,0],L1[0],test_mask_numpy[i])
        ari=adjusted_rand_score(unmasked_label,unmasked_pred)
        nmi=normalized_mutual_info_score(unmasked_label,unmasked_pred)
        ri, vi = results.update(L1[0],test_label_numpy[i][:,:,0])
        if nlabels<minlabels:
            print ("nLabels", nlabels, "reached minLabels", minlabels, ".")
            break
        print('per_RI is: ', ri)
        print('per_VI is: ', vi)
        batch_acc_sum1+=per_acc
        Ari+=ari
        Nmi+=nmi
        Ri+=ri
        Vi+=vi
       
        label_colours = np.random.randint(255,size=(100,3))
        im_target_rgb = np.array([label_colours[ c % len(np.unique(L1)) ] for c in L1])
        l=np.reshape(im_target_rgb,(128,128,3))
        L.append(l)#(B,W,H,3)      
    L2=np.stack(L)#(B,W,H,3)
    return batch_acc_sum1,L2,test_label_numpy,test_img_numpy,Ari,Nmi,Ri,Vi,time

#test
img1,label1,mask1=read_image_batch(data_queue1,batch_size1)
img1=tf.image.convert_image_dtype(img1,tf.float32)
img1=tf.reshape(img1,[INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL])
label1=tf.reshape(label1,[INPUT_IMG_WIDE, INPUT_IMG_HEIGHT,INPUT_IMG_CHANNEL])
mask1=tf.reshape(mask1,[INPUT_IMG_WIDE, INPUT_IMG_HEIGHT])
test_image,test_labels,test_masks= tf.train.batch(tensors=[img1,label1,mask1],batch_size=5,capacity=5000) 
config = tf.ConfigProto(allow_soft_placement=True)    
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth=True          
with tf.Session(config=config) as sess:
    with tf.device("/gpu:0"):
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)  
        saver.restore(sess,"/home/savemodel/autoencoder32.cpkt-499")
        summary_writer=tf.summary.FileWriter(summary_dir,sess.graph)
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)#启动线程               
        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)
        initialize_op = tf.variables_initializer(uninitialized_vars)
        sess.run(initialize_op)
        try:
            for epoch in range(training_epoch):
                test_acc=0
                test_ri=0
                test_vi=0
                test_sc=0
                L_total=[]
                I_total=[]
                batch_acc_sum=0
                Ari=0
                Nmi=0
                Ri=0
                Vi=0
                time=0
                for m in range(total_batch1):
                    batch_acc_sum,L1,T1,I1,Ari,Nmi,Ri,Vi,time=Per_image_train(middle2,K,sess,batch_acc_sum,Ari,Nmi,Ri,Vi,time)
                    L_total.append(L1)
                    I_total.append(I1)
                    print("Num_Batch:",'%03d' %(m+1))
                L_total1=np.stack(L_total)#(total_batch,B,W,H,3)
                I_total1=np.stack(I_total)
            L_all=np.reshape(L_total1,(total_batch1*batch_size1,128,128,3))
            I_all=np.reshape(I_total1,(total_batch1*batch_size1,128,128,3))
            #for i in range(total_batch1*batch_size1):
                #cv2.imwrite(savepath3+"/{}.png".format(i),L_all[i])
                #cv2.imwrite(savepath2+"/{}.jpg".format(i),I_all[i]*255)
            print("done!")
            print("test_accuracy=","{:.4f}".format(batch_acc_sum/(batch_size1*total_batch1)))
            print("test_ARI=","{:.4f}".format(Ari/(batch_size1*total_batch1)))
            print("test_NMI=","{:.4f}".format(Nmi/(batch_size1*total_batch1)))
            print("test_PRI=","{:.4f}".format(Ri/(batch_size1*total_batch1)))
            print("test_VI=","{:.4f}".format(Vi/(batch_size1*total_batch1)))

            #l=T1.astype(np.uint8)
            #show_num=5
            #f, a = plt.subplots(3, 5, figsize=(5, 3))
            #for i in range(show_num):
                #a[0][i].imshow(I1[i])
                #a[1][i].imshow(L1[i])
                #a[2][i].imshow(l[i][:,:,0])
            #plt.show()
        except tf.errors.OutOfRangeError:
            print("Done training -- epoch limit reached")
        finally:
            coord.request_stop()
coord.join(threads)    
summary_writer.close()   

    

