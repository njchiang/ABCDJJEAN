"""
General python utils
Mostly file input/output using Python

main author: Jeff
"""

import os, sys
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.io import loadmat
from scipy.io import savemat
from nilearn import image
from nilearn import masking


#######################################
# File I/O
#######################################
# JSON I/O
def load_json(*path, logger=None):
    """
    Load JSON config file
    :param path: path to json
    :param logger: logger instance
    :return: dict with JSON contents
    """
    with open(os.path.join(*path), "r") as f:
        d = json.load(f)
    return d


# Matlab I/O
def save_mat_data(fn, **kwargs):
    """
    :param fn: path to file
    :param logger: logger file
    :param kwargs: any key value pairs-- keys will become
    fieldnames of the struct with value.
    :return: None: write a mat file
    """
    savemat(fn, kwargs)
    return kwargs


def load_mat_data(fn):
    """
    Loads matlab file (just a wrapper)
    :param args: path to file
    :param logger: logger instance or none
    :return: dict
    """
    return loadmat(os.path.join(fn))


# Nilearn
def load_img(bs):
    """
    Simple wrapper for nilearn load_img to load NIFTI images
    :param path: path to subject directory
    :param logger: logfile ID
    :return: Nifti1Image
    """
    return image.load_img(bs, dtype=np.float64)


def concat_imgs(imgs):
    """
    Simple wrapper for concat_imgs
    :param imgs: list of nifti images or filenames
    :param logger: logfile ID
    :return: Nifti1Image
    """
    return image.concat_imgs(imgs, dtype=np.float64)


def index_img(imgs, index):
    """
    Simple wrapper for index_img
    :param imgs:
    :param index:
    :param logger:
    :return:
    """
    return image.index_img(imgs, index)


def load_labels(lp, **pdargs):
    """
    Simple wrapper using Pandas to load label files
    :param args: path to file directory
    :param logger: logfile ID
    :param pdargs: pandas read_csv args
    :return: pandas DataFrame with labels
    """
    return pd.read_csv(lp, **pdargs)


#######################################
# Image processing
#######################################
def mask_img(im, mask=None):
    """
    Wrapper for apply_mask (adds logging)
    :param im: image
    :param mask: mask. if none, will estimate mask to generate 2d
    :param logger: logger ID
    :return: masked image
    """
    # solved as of nilearn 2.3
    # if isinstance(im, str):
    return masking.apply_mask(im, mask, dtype=np.float64)
    # else:
    #     write_to_logger("Masking file")
    #     return masking._apply_mask_fmri(im, mask, dtype=np.float64)


def data_to_img(d, img, copy_header=False):
    """
    Wrapper for new_image_like
    :param img: Image with header you want to add
    :param d: data
    :param copy_header: Boolean
    :param logger: logger instance
    :return: Image file
    """
    return image.new_img_like(image.mean_img(img), d, copy_header=copy_header)


def center_img(img):
    """

    :param img:
    :param logger:
    :return:
    """
    return image.math_img("img - np.mean(img, axis=-1, keepdims=True)",
                          img=img)


def unmask_img(d, mask):
    """
    Unmasks matrix d according to mask
    :param d: numpy array (2D)
    :param mask: mask
    :param logger: logger instance
    :return: image file
    """
    return masking.unmask(d, mask)


def clip_brain(mask):
    """
    Finds the min and max indices on each axis of the mask
    :param mask: 3d image
    :return: indices (list of tuples)
    """
    return [(d.min(), d.max()) for d in np.where(mask > 0)]
#######################################
# Computation
#######################################
