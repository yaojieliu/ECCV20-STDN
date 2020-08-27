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
import matplotlib.tri as mtri

def tf_flatten(a):
    """Flatten tensor"""
    return tf.reshape(a, [-1])

def tf_repeat(a, repeats, axis=0):
    assert len(a.get_shape()) == 1
    a = tf.expand_dims(a, -1)
    a = tf.tile(a, [1, repeats])
    return tf.reshape(a, [-1])

def tf_repeat_2d(a, repeats):
    assert len(a.get_shape()) == 2
    a = tf.expand_dims(a, 0)
    a = tf.tile(a, [repeats, 1, 1])
    return a

def warping(x, offsets, imsize):
    bsize = x.shape[0]
    xsize = x.shape[1]
    offsets = offsets * imsize
    offsets = tf.reshape(offsets[:,:,:,0:2], (bsize, -1, 2)) # do not need z information

    # first build the grid for target face coordinates
    t_coords = tf.meshgrid(tf.range(xsize), tf.range(xsize), indexing='ij')
    t_coords = tf.stack(t_coords, axis=-1)
    t_coords = tf.cast(t_coords, tf.float32)
    t_coords = tf.reshape(t_coords, (-1, 2))
    t_coords = tf_repeat_2d(t_coords, bsize)
    # find the coordinates in the source image to copy pixels
    s_coords = t_coords + offsets
    s_coords = tf.clip_by_value(s_coords, 0, tf.cast(xsize-1, tf.float32))

    n_coords = s_coords.shape[1]
    idx = tf_repeat(tf.range(bsize), n_coords)

    def _gather_pixel(_x, coords):
        coords = tf.cast(coords, tf.int32)
        xcoords = tf.reshape(coords[..., 0], [-1])
        ycoords = tf.reshape(coords[..., 1], [-1])
        ind = tf.stack([idx, xcoords, ycoords], axis=-1)

        _y = tf.gather_nd(_x, ind)
        _y = tf.reshape(_y, (bsize, n_coords, _x.shape[3]))
        return _y

    # solve fractional coordinates via bilinear interpolation
    s_coords_lu = tf.floor(s_coords)
    s_coords_rb = tf.ceil(s_coords)
    s_coords_lb = tf.stack([s_coords_lu[..., 0], s_coords_rb[..., 1]], axis=-1)
    s_coords_ru = tf.stack([s_coords_rb[..., 0], s_coords_lu[..., 1]], axis=-1)
    _x_lu = _gather_pixel(x, s_coords_lu)
    _x_rb = _gather_pixel(x, s_coords_rb)
    _x_lb = _gather_pixel(x, s_coords_lb)
    _x_ru = _gather_pixel(x, s_coords_ru)
    # bilinear interpolation
    s_coords_fraction = s_coords - tf.cast(s_coords_lu, tf.float32)
    s_coords_fraction_x = s_coords_fraction[..., 0]
    s_coords_fraction_y = s_coords_fraction[..., 1]
    _xs, _ys = s_coords_fraction_x.shape
    s_coords_fraction_x = tf.reshape(s_coords_fraction_x, [_xs, _ys, 1])
    s_coords_fraction_y = tf.reshape(s_coords_fraction_y, [_xs, _ys, 1])
    _x_u = _x_lu + (_x_ru - _x_lu) * s_coords_fraction_x
    _x_b = _x_lb + (_x_rb - _x_lb) * s_coords_fraction_x
    warped_x = _x_u + (_x_b - _x_u) * s_coords_fraction_y
    warped_x = tf.reshape(warped_x, (bsize, xsize, xsize, -1))

    return warped_x

def generate_offset_map(source, target):
    anchor_pts = [[0,0],[0,256],[256,0],[256,256],
                  [0,128],[128,0],[256,128],[128,256],
                  [0,64],[0,192],[256,64],[256,192],
                  [64,0],[192,0],[64,256],[192,256]] 
    anchor_pts = np.asarray(anchor_pts)/ 256
    xi, yi = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))
    _source = np.concatenate([source, anchor_pts], axis=0).astype(np.float32)
    _target = np.concatenate([target, anchor_pts], axis=0).astype(np.float32)
    _offset = _source - _target

    # interp2d
    _triang  = mtri.Triangulation(_target[:,0], _target[:,1])
    _interpx = mtri.LinearTriInterpolator(_triang, _offset[:,0])
    _interpy = mtri.LinearTriInterpolator(_triang, _offset[:,1])
    _offsetmapx = _interpx(xi, yi)
    _offsetmapy = _interpy(xi, yi)

    offsetmap = np.stack([_offsetmapy, _offsetmapx, _offsetmapx*0], axis=2)
    return offsetmap
