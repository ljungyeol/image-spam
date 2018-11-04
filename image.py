
# coding: utf-8

# In[1]:


import tensorflow as tf
import os
import random
import numpy as np
import flags

import matplotlib
import matplotlib.pyplot as plt


FLAGS = tf.app.flags.FLAGS
           
            
def get_filename(data_set):
    labels = []
    filename_set = []
    
    with open(FLAGS.data_dir + '/label.txt') as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(',')]
            labels += inner_list
        
        for i, label in enumerate(labels):
            list = os.listdir(FLAGS.data_dir + '/' + data_set + '/' + label)
            for filename in list:
                filename_set.append([i,FLAGS.data_dir + '/' + data_set + '/' + label + '/'
                                   +filename])
        random.shuffle(filename_set)
        return filename_set
    
def read_jpeg(filename):
    read_file = tf.read_file(filename)
    decoded_image = tf.image.decode_jpeg(read_file, channels=FLAGS.channel)
    resized_image = tf.image.resize_images(decoded_image, [FLAGS.img_h,FLAGS.img_w])
    resized_image = tf.cast(resized_image, tf.uint8)    
    return resized_image


def convert_images(sess, data_set):
    filename_set = get_filename(data_set)
    with open('./data/'  + data_set + '_data.bin', 'wb') as f:
        for i in range(0, len(filename_set)):
            read_image = read_jpeg(filename_set[i][1])
            try:
                image = sess.run(read_image)
            except Exception as e:
                print(e.message) 
                continue
            print(i, filename_set[i][0], image.shape)
            f.write(chr(filename_set[i][0]).encode())
            f.write(image.data)

            
            
            
# train, eval 에서 이미지 다시 불러오기            
def get_data(data_set, batch_size):
    if data_set is 'train':
        return distorted_inputs(batch_size)
    else:
        return eval_inputs(batch_size)
    
    
def distorted_inputs(batch_size):
    image, label = read_raw_images('train')
    reshaped_image = tf.cast(image,tf.float32)
    
    distorted_image = tf.random_crop(reshaped_image, [FLAGS.height,FLAGS.width,FLAGS.channel])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    float_image = tf.image.per_image_standardization(distorted_image)
        
    return generate_image_and_label_batch(float_image, label, 100, batch_size, shuffle=True)
                                          
def eval_inputs(batch_size):
    image, label = read_raw_images('eval')
    reshaped_image = tf.cast(image, tf.float32)

    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, FLAGS.height, FLAGS.width)
    return generate_image_and_label_batch(resized_image, label, 10, batch_size, shuffle=True)
                                          
                                          
def read_raw_images(data_set):
    filename = ['./data/'  + data_set + '_data.bin']
    filename_queue = tf.train.string_input_producer(filename)
    size = (FLAGS.img_h) * (FLAGS.img_w) * FLAGS.channel
    record_bytes = size + 1
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)

    label = tf.cast(tf.slice(record_bytes, [0],[1]), tf.int32)
    image = tf.reshape(tf.slice(record_bytes,[1],[size]),
                       [FLAGS.channel,FLAGS.img_h,FLAGS.img_w])
    image_ = tf.transpose(image, [1,2,0])


    return image_, label

def generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    num_preprocess_threads = 4
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples * batch_size)

    return images, tf.reshape(label_batch, [batch_size])

def main(argv = None):
    with tf.Session() as sess:
        convert_images(sess, 'train')     
        convert_images(sess, 'eval')        

if __name__ == '__main__' :
    tf.app.run()
    

