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

def displayImage(image, title = 'An Image', cmap = 'gray'):
    plt.imshow(image, cmap = cmap)
    plt.title(title)
    plt.show()
    plt.clf()


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

class ImportingAndPreparation:
    def __init__(self, folder_absolute_path):
        self.folder_absolute_path = folder_absolute_path
       
    def extractCaracter(self, character = 'A'):
        fullpath = self.folder_absolute_path + character + "/*.png"
        all_character_path = glob.glob(fullpath)
        im = [iio.imread(im) for im in all_character_path]
        char = [character] * len(im)
        output = pd.DataFrame({'images':im,
                               'character':char})
        return output
        
    def resizeAllImages(self, list_of_images, output_shape):
        temp = [ski.transform.resize(im, output_shape = output_shape)\
                   for im in list_of_images]
        return temp
    
    
    



