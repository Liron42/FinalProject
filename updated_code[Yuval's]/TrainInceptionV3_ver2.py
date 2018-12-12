# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 20:56:00 2017

@author: yuval
"""

import tensorflow as tf
from TransferLearning import my_inception_v3
import numpy as np

train=False
# train parameters
epoch=10;batch_size=32; num_classes=2; layer_name='Mixed_7b'
height_width=299; model_name='inception_mamograms'
# Early Stopping parameters
check_interval = 100
max_checks_without_progress =20
# number of folds
k_fold=5

# data path 
data_dir='~/fold'
# path for the inceptiovV3 checkpoint
checkpoint_inception='~inception_v3.ckpt'

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
config = tf.ConfigProto()
config.gpu_options.allow_growth =True    
# layers size
nfs=96; fs=3;
n_hidden=32

for fold in range(k_fold):
    reset_graph()
    # data path for train/test in each fold
    dataset0=data_dir+ str(fold) +'/train0'
    dataset1=data_dir+ str(fold) +'/train1'
    test0=data_dir+ str(fold) +'/test0'
    test1=data_dir+ str(fold) +'/test1'
    #checkpoints path 
    checkpoint_dir=data_dir+ str(fold) +'/checkpoints' +' layer_name_'+ str(layer_name) + '_n' + str(n_hidden)+'_nfs'+str(nfs)+'_fs'+str(fs)

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        net = my_inception_v3(sess,checkpoint_inception=checkpoint_inception,
                                      checkpoint_dir=checkpoint_dir,layer_name=layer_name,
                                      n_hidden=n_hidden,nfs=nfs,fs=fs,height_width=height_width)
        if train:
                net.train(dataset0,dataset1,epoch,batch_size,check_interval,max_checks_without_progress)
                results=net.print_test_accuracy(test0,test1,batch_size,ROC=True,Confusion_Matrix=True)
        else:
          if not net.load(checkpoint_dir)[0]:
            raise Exception("[!] Train a model first, then run test mode")
          results=net.print_test_accuracy(test0,test1,batch_size,ROC=True,Confusion_Matrix=False,saveAUC=True)
        X = tf.placeholder(tf.float32, shape=[None, self.height_width,
                                                   self.height_width, self.channels], name="X")
        Y = tf.placeholder(tf.int64, shape=[None, self.num_classes], name='y_true')
        is_traing = tf.placeholder(tf.bool,name='is_traing')
        value_feed_dict = {X,Y,is_traing}
        tf_cnnvis.activation_visualization(net, value_feed_dict, input_tensor=None, layers='c', path_logdir='./Log', path_outdir='./Output')


                   


      

            


