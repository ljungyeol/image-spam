
# coding: utf-8

# In[1]:


import tensorflow as tf
import flags
import image
import sys
import datetime
import numpy as np

FLAGS = tf.app.flags.FLAGS

def weight_variable(shape) :
    init = tf.truncated_normal(shape, stddev=1)
    return tf.Variable(init)

def bias_variable(shape) :
    init = tf.constant(1,shape=shape)
    return tf.Variable(init)

def conv2d(x, W) :
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool(x, h, w) :
    return tf.nn.max_pool(x, ksize=[1,h,w,1],strides=[1,h,w,1], padding='SAME')


    


#이미지 불러오기
print("1. train 2. eval")
val = input()

if val is '1':
    images, labels = image.get_data('train',FLAGS.batch_size)
else:
    images, labels = image.get_data('eval', FLAGS.batch_size)



#신경망구성
pool_size = FLAGS.pool_size
kernel_size = FLAGS.kernel_size
channel = FLAGS.channel
img_size = FLAGS.img_size

# 첫번째 합성곱
w1 = weight_variable([kernel_size,kernel_size,channel,img_size])
b1 = bias_variable([img_size])
L1 = tf.nn.relu(conv2d(images,w1)+ tf.cast(b1, tf.float32))
L1 = max_pool(L1,pool_size,pool_size)

prev_img_size = img_size
img_size *= 2

# 두번째 합성곱
w2 = weight_variable([kernel_size,kernel_size,prev_img_size,img_size])
b2 = bias_variable([img_size])
L2 = tf.nn.relu(conv2d(L1,w2)+tf.cast(b2, tf.float32))
L2 = max_pool(L2,pool_size,pool_size)

prev_img_size = img_size
img_size *= 2

# 세번째 합성곱
w3 = weight_variable([kernel_size,kernel_size,prev_img_size,img_size])
b3 = bias_variable([img_size])
L3 = tf.nn.relu(conv2d(L2,w3)+tf.cast(b3, tf.float32))
L3 = max_pool(L3,pool_size,pool_size)

fc_size = FLAGS.fc_size



# full_connected, 
w_fc1 = weight_variable([16*16*128,fc_size])
b_fc1 = bias_variable([fc_size])
x_fc1 = tf.reshape(L3,[-1, 16*16*128])
L_fc1 = tf.nn.relu(tf.matmul(x_fc1, w_fc1) +tf.cast(b_fc1, tf.float32))

keep_prob = tf.placeholder(tf.float32)
L_fc1 = tf.nn.dropout(L_fc1,keep_prob)

w_fc2 = weight_variable([fc_size,2])
b_fc2 = bias_variable([2])

#hypothesis output
y_conv = tf.matmul(L_fc1,w_fc2) +tf.cast(b_fc2, tf.float32) 

#cross entropy 정의
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
                               (logits=y_conv, labels=labels))

#train step 정의
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#tensorboard 기록하기 위한 선언
cost_sum = tf.summary.scalar("cost", cross_entropy)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True



if tf.gfile.Exists(FLAGS.checkpoint_dir) == False:
    tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

if val is '1':
    with tf.Session(config=config) as sess:
      
        saver = tf.train.Saver()
      
        #if tf.gfile.Exists(FLAGS.checkpoint_dir + '/model.ckpt'):
        #    saver.restore(sess, FLAGS.checkpoint_dir + '/model.ckpt')
            
        #else:
        init = tf.global_variables_initializer()
        sess.run(init)

        #saver.restore(sess, FLAGS.checkpoint_dir + '/model.ckpt')
        
        writer = tf.summary.FileWriter("./logs/cost_log")
        writer.add_graph(sess.graph) 
        merge_sum = tf.summary.merge_all()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        
        for step in range(FLAGS.max_steps):

            #sess.run(train_step, feed_dict={keep_prob: 0.7})
            summary, _ = sess.run([merge_sum,train_step], feed_dict={keep_prob: 0.7})
            writer.add_summary(summary, global_step=step)
            print (step, sess.run(cross_entropy, feed_dict={keep_prob: 1.0}))

            if step % 100 == 0 or (step + 1) == FLAGS.max_steps:
                saver.save(sess, FLAGS.checkpoint_dir + '/model.ckpt')
        print("Train is complete!")
        coord.request_stop()
        coord.join(threads)
        
        
else:
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.checkpoint_dir + '/model.ckpt')
        
        delta = datetime.timedelta()
        max_steps = 14
        true_count = 0.
        pre_total_count = 0.
        pre_true_count = 0.
        rec_total_count = 0.
        rec_true_count = 0.
        total_sample_count = max_steps * FLAGS.batch_size
        
        top_k_op = tf.nn.in_top_k(y_conv, labels, 1)
        #acc = tf.metrics.accuracy(labels=labels, predictions=y_conv)
        #recall = tf.metrics.recall(
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        for i in range(0, max_steps):
            start = datetime.datetime.now()
            predictions = sess.run(top_k_op, feed_dict={keep_prob: 1.0})
            #원래 값이 spam인것 구하기
            pre = sess.run(labels)
            for j in range(0, max_steps):
                if pre[j] == 1:
                    rec_total_count +=1
                    rec_true_count += predictions[j]
                    pre_total_count += predictions[j]
                    pre_true_count += predictions[j]
                elif pre[j] == 0:
                    if predictions[j] == False:
                        pre_total_count += 1
                    
                    
            #print(sess.run(acc, feed_dict = {keep_prob: 1.0}))
            print(predictions)
            true_count += np.sum(predictions)
            delta += datetime.datetime.now() - start
        coord.request_stop()
        coord.join(threads)
    print ('total sample count: %d' % total_sample_count)
    #print ('total pre count: %d' %pre_total_count)
    #print ('total rec count: %d' %rec_total_count)
    #print ('true count: %d' % true_count)
    #print ('true pre count: %d' % pre_true_count)
    #print ('true rec count: %d' % rec_true_count)   
    
    print ('recall : %f' % (rec_true_count / rec_total_count))
    print ('accuracy : %f' % (true_count / total_sample_count))
    print ('precision : %f' % (pre_true_count / pre_total_count))
    print ('evaluation time: %f seconds' % ((delta.seconds + delta.microseconds / 1E6) / max_steps)) 

