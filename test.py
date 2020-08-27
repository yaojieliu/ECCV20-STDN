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
import time
from model.dataset import Dataset
from model.config import Config
from model.model import Gen

def _step(config, data_batch, training_nn):
  global_step = tf.train.get_or_create_global_step()
  bsize = config.BATCH_SIZE
  imsize = config.IMAGE_SIZE

  # Get images and labels for CNN.
  img, im_name = data_batch.nextit
  img = tf.reshape(img, [bsize, imsize, imsize, 3])

  # Forward the Generator
  M, s, b, C, T = Gen(img, training_nn=training_nn, scope='STDN')
  M = tf.reduce_mean(M, axis=[1,2,3])
  s = tf.reduce_mean(s, axis=[1,2,3])
  b = tf.reduce_mean(b, axis=[1,2,3])
  C = tf.reduce_mean(C, axis=[1,2,3])
  T = tf.reduce_mean(T, axis=[1,2,3])
  
  return M, s, b, C, T, im_name

def main(argv=None): 
  # Configurations
  config = Config(gpu='1',
                  root_dir='./data/test/',
                  root_dir_val=None,
                  mode='testing')
  config.BATCH_SIZE = 1

  # Get images and labels.
  dataset_test = Dataset(config, 'test')

  # Train
  _M, _s, _b, _C, _T, _imname = _step(config, dataset_test, False)

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

    step_per_epoch = int(len(dataset_test.name_list) / config.BATCH_SIZE)
    with open(config.LOG_DIR + '/test/score.txt', 'w') as f:
      for step in range(step_per_epoch):
        M, s, b, C, T, imname = sess.run([_M, _s, _b, _C, _T, _imname])
        # save the score
        for i in range(config.BATCH_SIZE):
            _name = imname[i].decode('UTF-8')
            _line = _name + ',' + str("{0:.3f}".format(M[i])) + ','\
                                + str("{0:.3f}".format(s[i])) + ','\
                                + str("{0:.3f}".format(b[i])) + ','\
                                + str("{0:.3f}".format(C[i])) + ','\
                                + str("{0:.3f}".format(T[i]))
            f.write(_line + '\n')  
            print(str(step+1)+'/'+str(step_per_epoch)+':'+_line, end='\r')  
    print("\n")

if __name__ == '__main__':
  tf.app.run()
