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
import os
import tensorflow as tf

# Base Configuration Class
class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # GPU Usage
    GPU_USAGE = '0'

    # Log and Model Storage Default
    LOG_DIR = './log/STDN'
    LOG_DEVICE_PLACEMENT = False

    # Input Data Meta
    IMAGE_SIZE = 256
    MAP_SIZE = 32

    # Training Meta
    BATCH_SIZE = 5
    G_D_RATIO = 2
    LEARNING_RATE = 6e-5
    LEARNING_RATE_DECAY_FACTOR = 0.9
    LEARNING_MOMENTUM = 0.999
    MAX_EPOCH = 50
    MOVING_AVERAGE_DECAY = 0.9999
    NUM_EPOCHS_PER_DECAY = 10.0 
    STEPS_PER_EPOCH = 2000
    STEPS_PER_EPOCH_VAL = 500
    LOG_FR_TRAIN = int(STEPS_PER_EPOCH / 10)
    LOG_FR_TEST  = int(STEPS_PER_EPOCH_VAL / 10)

    def __init__(self, gpu, root_dir, root_dir_val, mode):
        """Set values of computed attributes."""
        self.MODE = mode
        self.GPU_USAGE = gpu
        self.GPU_OPTIONS = tf.GPUOptions(per_process_gpu_memory_fraction=1, visible_device_list =self.GPU_USAGE, allow_growth = True)
        self.GPU_CONFIG = tf.ConfigProto(log_device_placement=self.LOG_DEVICE_PLACEMENT, gpu_options = self.GPU_OPTIONS)
        self.LI_DATA_DIR = [root_dir+'live/*']
        self.SP_DATA_DIR = [root_dir+'spoof/*']
        if root_dir_val:
            self.LI_DATA_DIR_VAL = [root_dir_val+'live/*']
            self.SP_DATA_DIR_VAL = [root_dir_val+'spoof/*']
        self.compile()

    def compile(self):
        if not os.path.isdir(self.LOG_DIR):
            os.mkdir(self.LOG_DIR)
        if not os.path.isdir(self.LOG_DIR+'/test'):
            os.mkdir(self.LOG_DIR+'/test')
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)) and a[0].isupper():
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
