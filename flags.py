
# coding: utf-8

# In[1]:


import tensorflow as tf

#flag 지정
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('f', '', '')
#트레이닝에 사용할 이미지 개수

#원본 이미지 크기
tf.app.flags.DEFINE_integer('img_h', 144, '')
tf.app.flags.DEFINE_integer('img_w', 144, '')

#resize 이미지 크기
tf.app.flags.DEFINE_integer('height', 128, '')
tf.app.flags.DEFINE_integer('width', 128, '')

#이미지 채널의 크기, 흑백1, RGB3
tf.app.flags.DEFINE_integer('channel', 3, '')

#최종 분류 개수, spam or ham
tf.app.flags.DEFINE_integer('class', 2, '')

#conv layer 개수
#tf.app.flags.DEFINE_integer('conv', 4, '')

#conv kernel_size
tf.app.flags.DEFINE_integer('kernel_size', 5, '')

#max pool size, 2*2
tf.app.flags.DEFINE_integer('pool_size', 2, '')

#layer 통과하는 image 개수
tf.app.flags.DEFINE_integer('img_size', 32, '')

#full connected layer 개수
#tf.app.flags.DEFINE_integer('fc', 2, '')

#fc input size
tf.app.flags.DEFINE_integer('fc_size', 512, '')

#학습 횟수
tf.app.flags.DEFINE_integer('max_steps', 10000, '')

#learning_rate
tf.app.flags.DEFINE_float('learning_rate', 0.001, '')

#트레이닝에 사용할 이미지 개수
tf.app.flags.DEFINE_integer('batch_size', 50, '')

#image 데이터 위치
tf.app.flags.DEFINE_string('data_dir', './data', '')
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint', '')


# In[4]:




