# -*- coding: utf-8 -*-
"""
Oct 29 

-----------------------------------------
Unsupervised_Image_Classification

-----------------------------------------



--------------
Additional notes:

To install Tensor flow on GPU:
    https://www.tensorflow.org/install/pip#windows-native_1
    https://www.youtube.com/watch?v=yLVFwAaFACk&t=457s
    The Youtube video is a little outdated but I found it useful
    Numpy needs to be downgraded to a previous version


"""

import numpy as np
import pandas as pd
import os#
import glob#
import matplotlib.pyplot as plt
#from PIL import Image
import skimage as ski
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import scipy
import imageio.v3 as iio



import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

"""
-------------------------------------------------------------------------------
 For this project we are going to use a mix of Object Oriented Programing and
Function Oriented Programing.

 Personaly I like to make either behavior-oriented or data-oriented classes

"""
   
    
    
# Let's keep it simple and create a function
def applyThresholdMean(images_pd_Series, col_name):
    tresh = [ski.filters.threshold_mean(images_pd_Series[im]) \
             for im in range(len(images_pd_Series))] 
        
    temp = [images_pd_Series[im] > tresh[im] \
                     for im in range(len(tresh))]
        
    return pd.Series(data = temp, name = col_name)











