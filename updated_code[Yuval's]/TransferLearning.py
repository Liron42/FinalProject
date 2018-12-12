# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 15:49:32 2017

@author: yuval
"""
from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import fully_connected,flatten
import tensorflow as tf
import os
from glob import glob
import scipy.misc
import numpy as np
import time
from six.moves import xrange
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import metrics

class my_inception_v3(object):

    def __init__(self, sess,checkpoint_inception,checkpoint_dir,layer_name,n_hidden
                 ,nfs,fs,use_pooling=True,use_BN=True,channels=3,num_classes=2,
                 height_width=299,model_name='inception_mamograms'):
        self.sess = sess
        self.num_classes=num_classes
        self.channels=channels
        self.layer_name=layer_name
        self.height_width=height_width
        self.model_name=model_name
        self.checkpoint_inception=checkpoint_inception
        self.checkpoint_dir=checkpoint_dir
        self.n_hidden=n_hidden
        self.nfs=nfs
        self.fs=fs
        self.use_pooling=use_pooling
        self.use_BN=use_BN
        self.build_model()
        
    def build_model(self):
         with tf.name_scope('inputs'):
             self.X = tf.placeholder(tf.float32, shape=[None, self.height_width,
                                                   self.height_width, self.channels], name="X")
             self.y_true = tf.placeholder(tf.int64, shape=[None, self.num_classes], name='y_true')
             y_true_cls = tf.argmax(self.y_true, dimension=1)
             self.is_traing=tf.placeholder(tf.bool,name='is_traing')
         # Normalizing the images
         self.X = tf.map_fn(lambda img: tf.image.per_image_standardization (img), self.X)
           
         with tf.name_scope('data_augmentation'):    
             self.X = tf.map_fn(lambda img: tf.image.random_brightness(img,max_delta=20/255), self.X)
             self.X = tf.map_fn(lambda img: tf.image.random_contrast(img, lower=0.5, upper=2), self.X)
         # inception layer
         if self.layer_name=='PreLogits': # Running on the entire network, remove only the last layer
             with slim.arg_scope(inception.inception_v3_arg_scope()):
                _, end_points = inception.inception_v3(self.X, num_classes=1001, is_training=is_traing)
                end_transfer_net=tf.squeeze(end_points[self.layer_name], axis=[1, 2])
         else:
             with slim.arg_scope(inception.inception_v3_arg_scope()):
                end_transfer_net,_= inception.inception_v3_base(self.X,
                         final_endpoint=self.layer_name,
                         min_depth=16,
                         depth_multiplier=1.0,
                         scope=None)     
                
         tf.summary.histogram(self.layer_name, end_transfer_net)
         self.inception_saver = tf.train.Saver()

         if self.layer_name!='PreLogits':
             # extra CNN for Identifying features focused on mammograms
             self.conv_net,self.w1 = self.conv2d(end_transfer_net,self.nfs, k_h=self.fs, k_w=self.fs, 
                                d_h=1, d_w=1,stddev=0.02, name='conv',padding='VALID')
             tf.summary.histogram('conv',self.conv_net) 
             tf.summary.histogram('Wight_conv', self.w1)       
                          
             x_maxpool = tf.nn.max_pool(value=self.conv_net,
                               ksize=[1, self.conv_net.get_shape()[1],self.conv_net.get_shape()[1], 1],
                               strides=[1, 2, 2, 1],padding='VALID')
             end_transfer_net=tf.nn.relu(x_maxpool)   
             self.flatt=flatten(end_transfer_net,scope='flatten')
             tf.summary.histogram('flatten', self.flatt)
         else:
             self.flatt=end_transfer_net
             tf.summary.histogram('flatten', self.flatt)
             
         self.x_hidden = fully_connected(self.flatt, self.n_hidden,
                                             scope='hidden')
         tf.summary.histogram('hidden', self.x_hidden)
                  
         self.logits = fully_connected(self.x_hidden, self.num_classes,
                                             activation_fn=None,scope='output')
         tf.summary.histogram('logits', self.logits)
         
         with tf.name_scope('predicted'):
            self.y_pred = tf.nn.softmax(self.logits)
            self.y_pred_cls = tf.argmax(self.y_pred, dimension=1)
    
         with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                        labels=self.y_true,name='xentropy')
            self.loss = tf.reduce_mean(cross_entropy,name='xentropy_mean')
            tf.summary.scalar('loss',self.loss)
            
         with tf.name_scope('performance'): 
            correct_prediction = tf.equal(self.y_pred_cls, y_true_cls)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
         self.merged=tf.summary.merge_all() # merge the data for the tensorboard
         self.saver = tf.train.Saver(max_to_keep=1)
         
         
    def train(self,dataset0,dataset1,epoch,batch_size,check_interval,max_checks_without_progress):
        
        # Early Stopping parameters
        best_loss = np.infty
        checks_since_last_progress = 0
        best_model_params = None 

        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5)
            # train only the extra CNN
            mammograms_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='conv|hidden|output')
            training_op = optimizer.minimize(self.loss, var_list=mammograms_vars)


            tf.global_variables_initializer().run()
            
            counter = 1 # counter for the Iterations
            start_time = time.time()
            # load the checkpints
            could_load, checkpoint_counter=self.load(self.checkpoint_dir)
            if could_load:
                    print(" [*] Load SUCCESS")
                    counter=checkpoint_counter
            else:
                    print(" [!] Load failed...")
                    self.inception_saver.restore(self.sess,self.checkpoint_inception)
                                                  
            writer = tf.summary.FileWriter(self.checkpoint_dir, self.sess.graph) # for the tensorboard                
            # Preparation of the data
            # dataset0:path for the "Health" images, dataset1:path for the images with tumor
            data0 = glob(os.path.join(dataset0,'*.jpg'))
            data1= glob(os.path.join(dataset1,'*.jpg'))
            # 0 for "Health", 1 for sick
            TRAIN_labels=np.concatenate([np.zeros(len(data0)),np.ones(len(data1))],axis=0)
            DATA=data0+data1
            for epoch in xrange(epoch):
                # shupple the data evey epoch and organization in batchs
                data, train_labels = shuffle(DATA, TRAIN_labels)
                batch_idxs = len(data)// batch_size
                
                
                for idx in xrange(0, batch_idxs):
                        batch_files = data[idx*batch_size:(idx+1)*batch_size]
                        batch = [self.Get_image(batch_file)
                                         for batch_file in batch_files]
                        # Normalizing the images to range 0-1 (like the in the original inceptionV3)
                        batch_images = np.array(batch).astype(np.float32)/255
                        
                        y=train_labels[idx*batch_size:(idx+1)*batch_size]
                        y_labels=np.zeros([len(y),self.num_classes])
                        y_labels[y==0,0]=1
                        y_labels[y==1,1]=1
                        # train the networkd
                        _,err=self.sess.run([training_op,self.loss],
                                        feed_dict={self.X: batch_images ,self.y_true:y_labels,self.is_traing:False})

                        counter += 1
                        
                        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f" \
                              % (epoch, idx, batch_idxs,
                                 time.time() - start_time, err))
                        # Summary of results in check point
                        if np.mod(counter, check_interval) == 0:
                            acc,summary=self.sess.run([self.accuracy,self.merged],
                                                          feed_dict={self.X: batch_images ,
                                                                     self.y_true:y_labels,self.is_traing:False})
                            msg = " Training Accuracy: {1:>6.1%}"
                            print(msg.format(counter+1,acc))
#                            err_summary=self.loss_summary.eval(feed_dict={self.X: batch_images , self.y_true:y_labels,self.is_traing:False})
                            writer.add_summary(summary, counter)
                            if err < best_loss:
                                best_loss = err
                                checks_since_last_progress = 0
                                best_model_params = self.get_model_params()
                            else:
                                checks_since_last_progress += 1
                                
                        if np.mod(counter, check_interval*2) == 0:
                            self.save(counter)
                if checks_since_last_progress > max_checks_without_progress:
                    print("Early stopping!")
                    break
                
            if best_model_params:
                self.restore_model_params(best_model_params)
            else :
                self.save(counter)
                
            writer.close()

    def save(self, step):
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

            self.saver.save(self.sess,
                        os.path.join(self.checkpoint_dir, self.model_name),
                        global_step=step)

    def return_feed_dict(self,batch_size,dataset0,dataset1):
        data0 = glob(os.path.join(dataset0,'*.jpg'))
        data1= glob(os.path.join(dataset1,'*.jpg'))
        TRAIN_labels=np.concatenate([np.zeros(len(data0)),np.ones(len(data1))],axis=0)
        data,train_labels = shuffle((data0+data1),TRAIN_labels)
        y=train_labels[0:batch_size]
        y_labels=np.zeros([len(y),self.num_classes])
        y_labels[y==0,0]=1
        y_labels[y==1,1]=1
        batch_files = data[0:batch_size]
        batch = [self.Get_image(batch_file)for batch_file in batch_files]
        # Normalizing the images to range 0-1 (like the in the original inceptionV3)
        batch_images = np.array(batch).astype(np.float32)/255
        feed_dict = {self.X: batch_images,
                     self.y_true: y_labels, self.is_traing: False}
        return feed_dict

    def load(self, checkpoint_dir):
            import re
            print(" [*] Reading checkpoints...")
    
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            print(ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
                print(" [*] Success to read {}".format(ckpt_name))
                return True, counter 
            else:
                print(" [*] Failed to find a checkpoint")
                return False, 0   
                
    def Get_image(self,image_path):
           return scipy.misc.imread(image_path, mode='RGB') 
       
    
    def get_model_params(self):
        gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

    def restore_model_params(self,model_params):
        gvar_names = list(model_params.keys())
        assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                      for gvar_name in gvar_names}
        init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
        feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
        tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

    
    def conv2d(self,input_, output_dim,
           k_h=5, k_w=5, d_h=1, d_w=1, stddev=0.02,
           name="conv2d",padding='VALID'):
       with tf.variable_scope(name):
           w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            
           conv = tf.nn.conv2d(input_ , w, strides=[1, d_h, d_w, 1], padding=padding)
   
           biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
           conv = tf.nn.bias_add(conv, biases)
    
       return conv, w
        

    def print_test_accuracy(self,test0,test1,batch_size,ROC=True,Confusion_Matrix=True,saveAUC=True):
            # Preparation of the data
            # dataset0:path for the "Health" images, dataset1:path for the images with tumor
            data0 = glob(os.path.join(test0,'*.jpg'))
            #print(data0)
            data1= glob(os.path.join(test1,'*.jpg'))
            TRAIN_labels=np.concatenate([np.zeros(len(data0)),np.ones(len(data1))],axis=0)
            DATA=data0+data1
            data, train_labels = shuffle(DATA, TRAIN_labels)
            batch_idxs = len(data)// batch_size
            batch_end = len(data) % batch_size
            cls_score=np.zeros(shape=[len(data),2], dtype=np.float32)
            cls_pred=np.zeros(shape=[len(data)], dtype=np.int)
            if batch_end>1:
                batch_idxs+=1
                ind=1
            else:
                ind=0
                
            acc=np.zeros(shape=[batch_idxs], dtype=np.float32) 
            #print(2)
            #print(batch_idxs) 
            for idx in xrange(0, batch_idxs):
                if idx==batch_idxs-1 and ind:
                   Start= (idx)*batch_size
                   End=Start+batch_end
                else:
                   Start=idx*batch_size
                   End=(idx+1)*batch_size
                #print(1)   
                batch_files = data[Start:End]
                batch = [self.Get_image(batch_file)
                                 for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)/255
                #print(batch)
                #print(batch_files)
                #print(batch_images)
                Train_labels=train_labels[Start:End]
                
                y_labels=np.zeros([len(Train_labels),self.num_classes])
                y_labels[Train_labels==0,0]=1
                y_labels[Train_labels==1,1]=1
            

                acc[idx],cls_score[Start:End,:],cls_pred[Start:End]=self.sess.run([self.accuracy,self.y_pred,self.y_pred_cls],
                       feed_dict={self.X: batch_images , self.y_true:y_labels,self.is_traing:False})

                
            # Summary of training results in dataframe format                
            d={'image name': data,'true label': train_labels,
               'predicted label': cls_pred,'score 0':cls_score[:,0],'score 1':cls_score[:,1]}
            df = pd.DataFrame(d)
            print(df)
        
            
            msg = " test Accuracy: {0:>6.1%}"
            print(msg.format(np.mean(acc)))


            if ROC:
                 test_cls = train_labels
                 scores=cls_score[:,1]
                 fpr, tpr, thresholds = metrics.roc_curve(test_cls, scores, pos_label=1)
                 print(fpr)
                 print(tpr)
                 AUC=metrics.auc(fpr,tpr)
                 if saveAUC:
                     plt.plot(fpr,tpr)
                     msg= "AUC on Test-Set: {} "
                     plt.title(msg.format(AUC))
                     plt.savefig(self.checkpoint_dir + '/AUC.jpg')
                     plt.close()
                 else:
                     return AUC
                     
            if Confusion_Matrix:
                self.plot_confusion_matrix(cls_pred,train_labels)
            return df
                
    def plot_confusion_matrix(self,cls_pred,train_labels):        

            # Get the confusion matrix using sklearn.
            cm = confusion_matrix(y_true=train_labels,
                                  y_pred=cls_pred)
            # Print the confusion matrix as text.
            print(cm)

            # Plot the confusion matrix as an image.
            plt.matshow(cm)
        
            # Make various adjustments to the plot.
            plt.colorbar()
            tick_marks = np.arange(self.num_classes)
            plt.xticks(tick_marks, range(self.num_classes))
            plt.yticks(tick_marks, range(self.num_classes))
            plt.xlabel('Predicted')
            plt.ylabel('True')
        
            plt.show()
            
            

