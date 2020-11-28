# Copyright 2020
# 
# Yaojie Liu, Joel Stehouwer, Xiaoming Liu, Michigan State University
# 
# All Rights Reserved.
# 
# This research is based upon work supported by the Office of the Director of 
# National Intelligence (ODNI), Intelligence Advanced Research Projects Activity
# (IARPA), via IARPA R&D Contract No. 2017-17020200004. The views and 
# conclusions contained herein are those of the authors and should not be 
# interpreted as necessarily representing the official policies or endorsements,
# either expressed or implied, of the ODNI, IARPA, or the U.S. Government. The 
# U.S. Government is authorized to reproduce and distribute reprints for 
# Governmental purposes not withstanding any copyright annotation thereon. 
# ==============================================================================
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from model.utils import Conv, Upsample, Downsample
  
def Gen(x, training_nn, scope):
  nlayers = [16, 64, 64, 96, ]

  x  = tf.concat([x,tf.image.rgb_to_yuv(x)], axis=3)
  x0 = Conv(x,  nlayers[1], scope+'/conv0', training_nn)
  # Block 1
  x1 = Conv(x0, nlayers[2], scope+'/conv1', training_nn)
  x1 = Conv(x1, nlayers[3], scope+'/conv2', training_nn)
  x1 = Downsample(x1, nlayers[2], scope+'/conv3', training_nn)
  # Block 2
  x2 = Conv(x1, nlayers[2], scope+'/conv4', training_nn)
  x2 = Conv(x2, nlayers[3], scope+'/conv5', training_nn)
  x2 = Downsample(x2, nlayers[2], scope+'/conv6', training_nn)
  # Block 3
  x3 = Conv(x2, nlayers[2], scope+'/conv7', training_nn)
  x3 = Conv(x3, nlayers[3], scope+'/conv8', training_nn)
  x3 = Downsample(x3, nlayers[2], scope+'/conv9', training_nn)
  # Decoder
  u1 = Upsample(x3, nlayers[1], scope+'/up1', training_nn)
  u2 = Upsample(tf.concat([u1, x2], 3), nlayers[1], scope+'/up2', training_nn)
  u3 = Upsample(tf.concat([u2, x1], 3), nlayers[1], scope+'/up3', training_nn)
  n1 = tf.nn.tanh(Conv(Conv(u1, nlayers[0], scope+'/n1', training_nn), 6, scope+'/nn1', training_nn, act=False, norm=False))
  n2 = tf.nn.tanh(Conv(Conv(u2, nlayers[0], scope+'/n2', training_nn), 3, scope+'/nn2', training_nn, act=False, norm=False))
  n3 = tf.nn.tanh(Conv(Conv(u3, nlayers[0], scope+'/n3', training_nn), 3, scope+'/nn3', training_nn, act=False, norm=False))

  s = tf.reduce_mean(n1[:,:,:,3:6], axis=[1,2], keepdims=True)
  b = tf.reduce_mean(n1[:,:,:,:3], axis=[1,2], keepdims=True)
  C = tf.nn.avg_pool(n2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
  T = n3

  # ESR
  map1 = tf.image.resize_images(x1,[32,32])
  map2 = tf.image.resize_images(x2,[32,32])
  map3 = tf.image.resize_images(x3,[32,32])
  maps = tf.concat([map1, map2, map3],3)
  x4 = Conv(maps, nlayers[2], scope+'/conv10', training_nn, apply_dropout=True)
  x4 = Conv(x4, nlayers[1], scope+'/conv11', training_nn, apply_dropout=True)
  x5 = Conv(x4, 1, scope+'/conv12', training_nn, act=False, norm=False) 

  return x5, s, b, C, T

def Disc_s(x, training_nn, scope):
  nlayers = [16, 32, 64, 96, ]
  x  = tf.concat([x,tf.image.rgb_to_yuv(x)], axis=3)
  # Block 1
  #x1 = Conv(x, nlayers[1], scope+'/conv1', training_nn)
  x1 = Downsample(x, nlayers[1], scope+'/conv2', training_nn)
  # Block 2
  #x2 = Conv(x1, nlayers[2], scope+'/conv3', training_nn)
  x2 = Downsample(x1, nlayers[2], scope+'/conv4', training_nn)
  # Block 3
  #x3 = Conv(x2, nlayers[2], scope+'/conv5', training_nn)
  x3 = Downsample(x2, nlayers[3], scope+'/conv6', training_nn)
  # Block 4
  x4 = Conv(x3, nlayers[3], scope+'/conv7', training_nn)
  x4l = Conv(x4, 1, scope+'/conv8', training_nn, act=False, norm=False) 
  x4s = Conv(x4, 1, scope+'/conv9', training_nn, act=False, norm=False) 

  return x4l, x4s

def Disc(x, training_nn, scope):
  nlayers = [16, 32, 64, 96, ]
  x  = tf.concat([x,tf.image.rgb_to_yuv(x)], axis=3)
  # Block 1
  x1 = Conv(x, nlayers[1], scope+'/conv1', training_nn)
  x1 = Downsample(x1, nlayers[1], scope+'/conv2', training_nn)
  # Block 2
  x2 = Conv(x1, nlayers[2], scope+'/conv3', training_nn)
  x2 = Downsample(x2, nlayers[2], scope+'/conv4', training_nn)
  # Block 3
  x3 = Conv(x2, nlayers[2], scope+'/conv5', training_nn)
  x3 = Downsample(x3, nlayers[3], scope+'/conv6', training_nn)
  # Block 4
  x4 = Conv(x3, nlayers[3], scope+'/conv7', training_nn)
  x4l = Conv(x4, 1, scope+'/conv8', training_nn, act=False, norm=False) 
  x4s = Conv(x4, 1, scope+'/conv9', training_nn, act=False, norm=False) 

  return x4l, x4s

def get_train_op(sum_loss, global_step, config, scope_name):
  # Variables that affect learning rate.
  decay_steps = config.NUM_EPOCHS_PER_DECAY * config.STEPS_PER_EPOCH

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(config.LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  config.LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)

  # Generate moving averages of all losses and associated summaries.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  loss_averages_op = loss_averages.apply([sum_loss])
  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.AdamOptimizer(lr)
    grads = opt.compute_gradients(sum_loss, var_list=tf.trainable_variables(scope=scope_name))
  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Track the moving averages of all trainable variables.
  with tf.name_scope('TRAIN-'+scope_name) as scope:
    variable_averages = tf.train.ExponentialMovingAverage(config.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables(scope=scope_name))

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train-'+scope_name)

  return train_op
