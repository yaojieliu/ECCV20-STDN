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
import numpy as np
import cv2
import time
from model.dataset import Dataset
from model.config  import Config
from model.model   import Gen, Disc, Disc_s, get_train_op
from model.utils   import Error, plotResults
from model.loss    import l1_loss, l2_loss
from model.warp    import warping

def _step(config, data_batch, training_nn):
  global_step = tf.train.get_or_create_global_step()
  bsize = config.BATCH_SIZE
  imsize = config.IMAGE_SIZE
  im2size = 160
  im3size = 40
  # Get images and labels.
  img, reg = data_batch.nextit
  img  = tf .transpose(img, perm=[1, 0, 2, 3, 4])
  img  = tf.reshape(img, [bsize*2, imsize, imsize, 3])
  img2 = tf.image.resize_images(img, [im2size, im2size])
  img3 = tf.image.resize_images(img, [im3size, im3size])
  reg  = tf.reshape(reg, [bsize, imsize, imsize, 3])

  ################################### STEP 1 ##################################################################
  M, s, b, C, T = Gen(img, training_nn=training_nn, scope='STDN')

  ################################### STEP 2 ##################################################################
  recon1 = (1-s)*img - b - tf.image.resize_images(C, [imsize, imsize]) - T
  trace = img - recon1
  trace_warp = warping(trace[bsize:,...], reg, imsize)
  synth1 = img[:bsize,...]+ trace_warp
  img_d1 = tf.concat([img, recon1[bsize:,...], synth1], 0)
  d1l, d1s = Disc_s(img_d1, training_nn=training_nn, scope='Disc/d1')

  recon2 = tf.image.resize_images(recon1, [im2size, im2size])
  synth2 = tf.image.resize_images(synth1, [im2size, im2size])
  img_d2 = tf.concat([img2, recon2[bsize:,...], synth2], 0)
  d2l, d2s = Disc_s(img_d2, training_nn=training_nn, scope='Disc/d2')

  recon3 = tf.image.resize_images(recon1, [im3size, im3size])
  synth3 = tf.image.resize_images(synth1, [im3size, im3size])
  img_d3 = tf.concat([img3, recon3[bsize:,...], synth3], 0)
  d3l, d3s = Disc_s(img_d3, training_nn=training_nn, scope='Disc/d3')

  ################################### STEP 3 ##################################################################
  s_hard = s * tf.random.uniform([bsize*2, 1, 1, 1], minval=0.1, maxval=0.8)
  b_hard = b * tf.random.uniform([bsize*2, 1, 1, 1], minval=0.1, maxval=0.8)
  C_hard = C * tf.random.uniform([bsize*2, 1, 1, 1], minval=0.1, maxval=0.8)
  T_hard = T * tf.random.uniform([bsize*2, 1, 1, 1], minval=0.1, maxval=0.8)
  recon_hard1 = (1-s_hard)*img - b - tf.image.resize_images(C, [imsize, imsize]) - T
  recon_hard2 = (1-s)*img - b_hard - tf.image.resize_images(C, [imsize, imsize]) - T
  recon_hard3 = (1-s)*img - b - tf.image.resize_images(C_hard, [imsize, imsize]) - T
  recon_hard4 = (1-s)*img - b - tf.image.resize_images(C, [imsize, imsize]) - T_hard
  recon_hard_s1 = tf.cond(tf.greater(tf.random.uniform([1],0,1)[0],0.5),lambda: recon_hard1, lambda: recon_hard2)
  recon_hard_s2 = tf.cond(tf.greater(tf.random.uniform([1],0,1)[0],0.5),lambda: recon_hard3, lambda: recon_hard4)
  recon_hard = tf.cond(tf.greater(tf.random.uniform([1],0,1)[0],0.5),lambda: recon_hard_s1, lambda: recon_hard_s2)
  img_a1 = tf.stop_gradient(tf.concat([img[:bsize,...], recon_hard[bsize:,...]],axis=0))
  img_a2 = tf.stop_gradient(tf.concat([img[:bsize,...], synth1],axis=0))
  dec = tf.greater(tf.random.uniform([1],0,1)[0],0.5)
  img_a = tf.cond(dec,lambda: img_a1, lambda: img_a2)
  M_a, s_a, b_a, C_a, T_a = Gen(img_a, training_nn=training_nn, scope='STDN')
  traces_a = s_a*img + b_a + tf.image.resize_images(C_a, [imsize, imsize]) + T_a

  ################################### Losses ##################################################################
  d1_rl, _, d1_sl, _ = tf.split(d1l, 4)
  d2_rl, _, d2_sl, _ = tf.split(d2l, 4)
  d3_rl, _, d3_sl, _ = tf.split(d3l, 4)
  _, d1_rs, _, d1_ss = tf.split(d1s, 4)
  _, d2_rs, _, d2_ss = tf.split(d2s, 4)
  _, d3_rs, _, d3_ss = tf.split(d3s, 4)
  # loss for step 1.
  M_li, M_sp = tf.split(M, 2, 0)
  esr_loss = l1_loss(M_li,-1) + l1_loss(M_sp,1)
  gan_loss = l2_loss(d1_sl,1) + l2_loss(d2_sl,1) + l2_loss(d3_sl,1) +\
             l2_loss(d1_ss,1) + l2_loss(d2_ss,1) + l2_loss(d3_ss,1)
  reg_loss_li = l2_loss(s[:bsize,...],0) + l2_loss(b[:bsize,...],0) + l2_loss(C[:bsize,...],0) + l2_loss(T[:bsize,...],0)
  reg_loss_sp = l2_loss(s[bsize:,...],0) + l2_loss(b[bsize:,...],0) + l2_loss(C[bsize:,...],0) + l2_loss(T[bsize:,...],0)
  reg_loss = reg_loss_li*10 + reg_loss_sp*1e-4
  g_loss = esr_loss*50 + gan_loss + reg_loss

  # loss for step2
  d_loss =(l2_loss(d1_rl,1) + l2_loss(d2_rl,1) + l2_loss(d3_rl,1) +\
           l2_loss(d1_rs,1) + l2_loss(d2_rs,1) + l2_loss(d3_rs,1) +\
           l2_loss(d1_sl,0) + l2_loss(d2_sl,0) + l2_loss(d3_sl,0) +\
           l2_loss(d1_ss,0) + l2_loss(d2_ss,0) + l2_loss(d3_ss,0)) / 4

  # loss for step3.
  esr_loss_a = l1_loss(M_a[:bsize,...],-1) + l1_loss(M_a[bsize:,...],1)
  pixel_loss = l1_loss(traces_a[:bsize,...], tf.stop_gradient(trace_warp))
  a_loss_1 = esr_loss_a*5 + pixel_loss*0.0 #  #
  a_loss_2 = esr_loss_a*5 + pixel_loss*0.1 #  #
  a_loss = tf.cond(dec,lambda: a_loss_1, lambda: a_loss_2)
  
  if training_nn:
    g_op = get_train_op(g_loss+a_loss, global_step, config, "STDN")
    d_op = get_train_op(d_loss, global_step, config, "Disc")
  else:
    g_op = None
    d_op = None

  # log info
  losses = [g_loss, d_loss, a_loss]
  fig = [img, (M+1)/2, s*5, b*5, C*5, T*5, recon1, img_a]
  fig = plotResults(fig)

  return losses, g_op, d_op, fig


def main(argv=None):
  # Configurations
  config = Config(gpu='1',
                  root_dir='./data/train/',
                  root_dir_val='./data/val/',
                  mode='training')

  # Create data feeding pipeline.
  dataset_train = Dataset(config, 'train')
  dataset_val   = Dataset(config, 'val')

  # Train Graph
  losses, g_op, d_op, fig   = _step(config, dataset_train, training_nn=True)
  losses_val, _, _, fig_val = _step(config, dataset_val,   training_nn=False)

  # Add ops to save and restore all the variables.
  saver = tf.train.Saver(max_to_keep=50,)
  with tf.Session(config=config.GPU_CONFIG) as sess:
    # Restore the model
    ckpt = tf.train.get_checkpoint_state(config.LOG_DIR)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      last_epoch = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      print('**********************************************************')
      print('Restore from Epoch '+str(last_epoch))
      print('**********************************************************')
    else:
      init = tf.initializers.global_variables()
      last_epoch = 0
      sess.run(init)
      print('**********************************************************')
      print('Train from scratch.')
      print('**********************************************************')

    avg_loss = Error()
    print_list = {}
    for epoch in range(int(last_epoch), config.MAX_EPOCH):
      start = time.time()
      # Train one epoch
      for step in range(config.STEPS_PER_EPOCH):
        if step%config.G_D_RATIO ==0:
          _losses = sess.run(losses+[g_op, d_op, fig])
        else:
          _losses = sess.run(losses+[g_op, fig])

        # Logging
        print_list['g_loss'] = _losses[0]
        print_list['d_loss'] = _losses[1]
        print_list['a_loss'] = _losses[2]
        display_list = ['Epoch '+str(epoch+1)+'-'+str(step+1)+'/'+ str(config.STEPS_PER_EPOCH)+':'] +\
                       [avg_loss(x) for x in print_list.items()]                  
        print(*display_list+['          '], end='\r')
        # Visualization
        if step%config.LOG_FR_TRAIN ==0:
          fname = config.LOG_DIR+'/Epoch-'+str(epoch+1)+'-'+str(step+1)+'.png'
          cv2.imwrite(fname, _losses[-1])

      # Model saving
      saver.save(sess, config.LOG_DIR+'/ckpt', global_step=epoch+1)
      print('\n', end='\r')

      # Validate one epoch
      for step in range(config.STEPS_PER_EPOCH_VAL):
        _losses = sess.run(losses_val+[fig_val])

        # Logging
        print_list['g_loss'] = _losses[0]
        print_list['d_loss'] = _losses[1]
        print_list['a_loss'] = _losses[2]
        display_list = ['Epoch '+str(epoch+1)+'-Val-'+str(step+1)+'/'+ str(config.STEPS_PER_EPOCH_VAL)+':'] +\
                       [avg_loss(x, val=1) for x in print_list.items()]
        print(*display_list+['          '], end='\r')
        # Visualization
        if step%config.LOG_FR_TEST ==0:
            fname = config.LOG_DIR+'/Epoch-'+str(epoch+1)+'-Val-'+str(step+1)+'.png'
            cv2.imwrite(fname, _losses[-1])

      # time of one epoch
      print('\n    Time taken for epoch {} is {:3g} sec'.format(epoch + 1, time.time() - start))
      avg_loss.reset()

if __name__ == '__main__':
  tf.app.run()
