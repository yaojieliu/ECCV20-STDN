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
import numpy as np
import random
import glob
import cv2
from PIL import Image 
from model.warp import generate_offset_map
autotune = -1

class Dataset():
    def __init__(self, config, train_mode):
        self.config = config
        if self.config.MODE == 'training':
            self.input_tensors = self.inputs_for_training(train_mode)
        elif self.config.MODE == 'testing':
            self.input_tensors, self.name_list = self.inputs_for_testing()
        self.nextit = self.input_tensors.make_one_shot_iterator().get_next()

    def inputs_for_training(self, train_mode):
        if train_mode == 'train':
            data_dir_li = self.config.LI_DATA_DIR
            data_dir_sp = self.config.SP_DATA_DIR
        elif train_mode == 'val':
            data_dir_li = self.config.LI_DATA_DIR_VAL
            data_dir_sp = self.config.SP_DATA_DIR_VAL

        li_data_samples = []
        for _dir in data_dir_li:
            _list = glob.glob(_dir)
            li_data_samples += _list
        sp_data_samples = []
        for _dir in data_dir_sp:
            _list = glob.glob(_dir)
            sp_data_samples += _list

        # make live/spoof sample lists equal
        li_len = len(li_data_samples)
        sp_len = len(sp_data_samples)
        if li_len<sp_len:
            while len(li_data_samples)<sp_len:
                li_data_samples += random.sample(li_data_samples, len(li_data_samples))
            li_data_samples = li_data_samples[:sp_len]
        elif li_len>sp_len:
            while len(sp_data_samples)<li_len:
                sp_data_samples += random.sample(sp_data_samples, len(sp_data_samples))
            sp_data_samples = sp_data_samples[:li_len]
        shuffle_buffer_size = len(li_data_samples)
        
        dataset = tf.data.Dataset.from_tensor_slices((li_data_samples, sp_data_samples))
        dataset = dataset.shuffle(shuffle_buffer_size).repeat(-1)
        if train_mode == 'train':
            dataset = dataset.map(map_func=self.parse_fn, num_parallel_calls=autotune)
            dataset = dataset.batch(batch_size=self.config.BATCH_SIZE).prefetch(buffer_size=autotune)
        else:
            dataset = dataset.map(map_func=self.parse_fn_val, num_parallel_calls=autotune)
            dataset = dataset.batch(batch_size=self.config.BATCH_SIZE).prefetch(buffer_size=autotune)
        return dataset

    def inputs_for_testing(self):
        data_dir =  self.config.LI_DATA_DIR + self.config.SP_DATA_DIR
        data_samples = []
        for _dir in data_dir:
            _list = sorted(glob.glob(_dir))
            data_samples += _list

        def list_extend(vd_list):
            new_list = []
            for _file in vd_list:
                meta = glob.glob(_file+'/*.png')
                new_list += meta
            return new_list
        data_samples = list_extend(data_samples)  
        dataset = tf.data.Dataset.from_tensor_slices((data_samples))
        dataset = dataset.map(map_func=self.parse_fn_test, num_parallel_calls=autotune)
        dataset = dataset.batch(batch_size=self.config.BATCH_SIZE).prefetch(buffer_size=autotune)
        return dataset, data_samples

    def parse_fn(self, file1, file2):
        config = self.config
        imsize = config.IMAGE_SIZE
        lm_reverse_list = np.array([17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,
                           27,26,25,24,23,22,21,20,19,18,
                           28,29,30,31,36,35,34,33,32,
                           46,45,44,43,48,47,40,39,38,37,42,41,
                           55,54,53,52,51,50,49,60,59,58,57,56,65,64,63,62,61,68,67,66],np.int32) -1

        def _parse_function(_file1, _file2):
            # live
            _file1 = _file1.decode('UTF-8')
            meta = glob.glob(_file1+'/*.png')
            try:
                fr = meta[random.randint(0, len(meta) - 1)]
            except:
                print(_file1, len(meta))
            im_name = fr
            lm_name = fr[:-3] + 'npy'
            image = Image.open(im_name)
            width, height = image.size
            image_li = image.resize((imsize,imsize))
            image_li = np.array(image_li,np.float32)
            lm_li = np.load(lm_name) / width
            if np.random.rand() > 0.5:
                image_li = cv2.flip(image_li, 1)
                lm_li[:,0] = 1 - lm_li[:,0]
                lm_li = lm_li[lm_reverse_list,:]

            # spoof
            _file2 = _file2.decode('UTF-8')
            meta = glob.glob(_file2+'/*.png')
            try:
                fr = meta[random.randint(0, len(meta) - 1)]
            except:
                print(_file2, len(meta))
            im_name = fr
            lm_name = fr[:-3] + 'npy'
            image = Image.open(im_name)
            width, height = image.size
            image_sp = image.resize((imsize,imsize))
            image_sp = np.array(image_sp,np.float32)
            lm_sp = np.load(lm_name) / width
            if np.random.rand() > 0.5:
                image_sp = cv2.flip(image_sp, 1)
                lm_sp[:,0] = 1 - lm_sp[:,0]
                lm_sp = lm_sp[lm_reverse_list,:]

            # offset map
            reg_map_sp = generate_offset_map(lm_sp, lm_li)

            return np.array(image_li,np.float32)/255, np.array(image_sp,np.float32)/255, reg_map_sp.astype(np.float32)

        image_li, image_sp, reg_map_sp = tf.py_func(_parse_function, [file1, file2], [tf.float32, tf.float32, tf.float32])
        image_li   = tf.ensure_shape(image_li,   [imsize, imsize, 3])
        image_sp   = tf.ensure_shape(image_sp,   [imsize, imsize, 3])
        reg_map_sp = tf.ensure_shape(reg_map_sp, [imsize, imsize, 3])
        # data augmentation
        image      = tf.stack([tf.image.random_brightness(image_li, 0.25), tf.image.random_brightness(image_sp, 0.25)], axis=0)
        return image, reg_map_sp


    def parse_fn_val(self, file1, file2):
        config = self.config
        imsize = config.IMAGE_SIZE
        lm_reverse_list = np.array([17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,
                                    27,26,25,24,23,22,21,20,19,18,
                                    28,29,30,31,36,35,34,33,32,
                                    46,45,44,43,48,47,40,39,38,37,42,41,
                                    55,54,53,52,51,50,49,60,59,58,57,56,65,64,63,62,61,68,67,66],np.int32) -1

        def _parse_function(_file1, _file2):
            # live
            _file1 = _file1.decode('UTF-8')
            meta = glob.glob(_file1+'/*.png')
            try:
                fr = meta[random.randint(0, len(meta) - 1)]
            except:
                print(_file1, len(meta))
                input()
            im_name = fr
            lm_name = fr[:-3] + 'npy'
            image = Image.open(im_name)
            width, height = image.size
            image_li = image.resize((imsize,imsize))
            image_li = np.array(image_li,np.float32)
            lm_li = np.load(lm_name) / width
            if np.random.rand() > 0.5:
                image_li = cv2.flip(image_li, 1)
                lm_li[:,0] = 1 - lm_li[:,0]
                lm_li = lm_li[lm_reverse_list,:]

            # spoof
            _file2 = _file2.decode('UTF-8')
            meta = glob.glob(_file2+'/*.png')
            try:
                fr = meta[random.randint(0, len(meta) - 1)]
            except:
                print(_file2, len(meta))
                input()
            im_name = fr
            lm_name = fr[:-3] + 'npy'
            image = Image.open(im_name)
            width, height = image.size
            image_sp = image.resize((imsize,imsize))
            image_sp = np.array(image_sp,np.float32)
            lm_sp = np.load(lm_name) / width
            if np.random.rand() > 0.5:
                image_sp = cv2.flip(image_sp, 1)
                lm_sp[:,0] = 1 - lm_sp[:,0]
                lm_sp = lm_sp[lm_reverse_list,:]

            # offset map
            reg_map_sp = generate_offset_map(lm_sp, lm_li)

            return np.array(image_li,np.float32)/255, np.array(image_sp,np.float32)/255, reg_map_sp.astype(np.float32)

        image_li, image_sp, reg_map_sp = tf.py_func(_parse_function, [file1, file2], [tf.float32, tf.float32, tf.float32])
        image_li   = tf.ensure_shape(image_li,   [imsize, imsize, 3])
        image_sp   = tf.ensure_shape(image_sp,   [imsize, imsize, 3])
        reg_map_sp = tf.ensure_shape(reg_map_sp, [imsize, imsize, 3])
        # data augmentation
        image      = tf.stack([image_li, image_sp], axis=0)
        return image, reg_map_sp

    def parse_fn_test(self, file):
        config = self.config
        imsize = config.IMAGE_SIZE

        def _parse_function(_file):
            _file = _file.decode('UTF-8')
            image_list = []
            im_name = _file
            image = Image.open(im_name)
            image = image.resize((imsize, imsize))
            return np.array(image,np.float32)/255, im_name

        image, im_name = tf.py_func(_parse_function, [file], [tf.float32, tf.string])
        image   = tf.ensure_shape(image, [config.IMAGE_SIZE, config.IMAGE_SIZE, 3])
        return image, im_name
