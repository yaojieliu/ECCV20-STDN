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
import tensorflow as tf

def l1_loss(x, y, mask=None):
    xshape = x.shape
    if mask is not None:
        loss = tf.math.reduce_mean(tf.reshape(tf.abs(x-y), [xshape[0], -1]), axis=1, keepdims=True)
        loss = tf.math.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8)
    else:
        loss = tf.math.reduce_mean(tf.abs(x-y))
    return loss

def l2_loss(x, y, mask=None):
    xshape = x.shape
    if mask is not None:
        loss = tf.math.reduce_mean(tf.reshape(tf.square(x-y), [xshape[0], -1]), axis=1, keepdims=True)
        loss = tf.math.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8)
    else:
        loss = tf.math.reduce_mean(tf.square(x-y))
    return loss