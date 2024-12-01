"""
Nov 7 

Unsupervised_Image_Classification

-----------------------------------------
Unsupervised_Image_Classification

-----------------------------------------

1. Access the data



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
import time


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from applyThresholdMean import applyThresholdMean
from ImportingAndPreparation import ImportingAndPreparation
from Vis import Vis



"""
-------------------------------------------------------------------------------
"""

"""
     Data importation and Cleaning
"""




path = "C:/Users/morai/Python_Project/Unsupervised_Image_Classification/standard OCR dataset/archive/data/training_data/"
#character = "0"


A = ImportingAndPreparation(path)
images_A = A.extractCaracter("A")


[Vis.displayImage(images_A.images[im]) for im in range(0, 500, 100)]
# Not all images have the same definition...

# Let's dig deeper
X_resolution = [images_A.images[im].shape[0] for im in range(len(images_A))]
Y_resolution = [images_A.images[im].shape[1] for im in range(len(images_A))]

# Let's visualize the distribution with a histogram
plt.hist(X_resolution, bins = 50)
plt.show()
plt.clf()

plt.hist2d(X_resolution, Y_resolution)
plt.title("2D histogram of X and Y's resolutions")
plt.xlabel('Distribution of X_resolution')
plt.ylabel('Distribution of Y_resolution')
plt.show()
plt.clf()
# okay, let's re-size all the images in to 40 by 40 squares

images_A.images = A.resizeAllImages(images_A.images, (40, 40))

[Vis.displayImage(images_A.images[im]) for im in range(0, 500, 100)]
# Looking good

# Let's bring in other characters
X = ImportingAndPreparation(path)
images_X = X.extractCaracter('X')
[Vis.displayImage(images_X.images[im]) for im in range(0, 500, 100)]
# Just like the images of A, they need to be resized

images_X.images = X.resizeAllImages(images_X.images, (40, 40))

[Vis.displayImage(images_X.images[im]) for im in range(0, 500, 100)]
# looking good =)

M = ImportingAndPreparation(path)
images_M = M.extractCaracter('M')
images_M.images = M.resizeAllImages(images_M.images, (40, 40))

# Let's dedete what we don't need anymore
del X_resolution, Y_resolution, A, M, X

#
# We are now done with the data cleaning. Let's get the data ready 
# for annalysis
#

"""
     Data Pre-Processing
"""

all_images_original = pd.concat([images_A, images_M, images_X])

all_images_original = all_images_original.sample(frac = 1,
                         random_state = 4,
                         ignore_index  =True)

all_images = all_images_original.copy()
# must add .copy() becasue in python variables are pointers, not buckets.

[Vis.displayImage(all_images.images[im], title = "All Images: " ) \
 for im in range(0, 1700, 300)]
# Looking good

# let's do some data pre-processing
# From experience algorithms work best on data containing the bare minimum of 
# information to make a decition. 

# let's start with a few images, including one of each character
sample = all_images.images[0:3]

# Apply a handful of thresholding filters to se what works best
[ski.filters.try_all_threshold(sample[im]) for im in range(len(sample))]
# Mean and Yen deliver shte best results
# Let's compare their runing time

# threshodl_mean running time
start_mean = time.time()
tresh = [ski.filters.threshold_mean(all_images.images[im]) \
         for im in range(len(all_images))] 
    
temp = [all_images.images[im] > tresh[im] \
                 for im in range(len(tresh))]
    
all_images2 = pd.Series({"images": temp})
end_mean = time.time()
time_mean = end_mean - start_mean
print("Applying treshold_mean on all 1719 images took {} second \n"\
      .format(time_mean))

del tresh, temp, all_images2


# threshodl_Yen running time
start_mean = time.time()
tresh = [ski.filters.threshold_yen(all_images.images[im]) \
         for im in range(len(all_images))] 
    
temp = [all_images.images[im] > tresh[im] \
                 for im in range(len(tresh))]
    
all_images2 = pd.Series({"images": temp})
end_mean = time.time()
time_yen = end_mean - start_mean
print("Applying treshold_mean on all 1719 images took {} second \n"\
      .format(time_yen))

del tresh, temp, all_images2

print(" Threshold_mean is more than {} faster than treshold_yen for \n\
      very similar results, we are going wtih threhold_mean"\
      .format(time_yen // time_mean))
    
del start_mean, end_mean, time_mean, time_yen
#
# Verdict: threshold_mean is quicker
# Let's add it to Homemade_Functions as a stand alone function
# rather that within a class
#

temp = applyThresholdMean(all_images.images, 'images')


[Vis.displayImage(all_images.images[im], title = "Clean images: " ) \
 for im in range(0, 1700, 300)]

[Vis.displayImage(temp[im], title = "PreProcessed images: " ) \
 for im in range(0, 1700, 300)]
# Nice this is working fine !

all_images.images = temp

del sample, temp

X = all_images.images.values
X.shape
np.stack(X, axis = 0).shape
X = np.stack(X, axis = 0)
X.shape




#
# We are now done with data cleaning and pre-processing
#

"""
     Data Modeling
"""

#
# k-means
# Let's start with kmeans and see if it can detect the 3 different letters
#

from sklearn.cluster import KMeans

# 3 clusteres because of three characters
kmean = KMeans(n_clusters= 3\
               ,random_state = 0\
               )

X_predict = kmean.fit_predict(X.reshape(1719,-1))

X_transform = kmean.fit_transform(X.reshape(1719,-1))

X_transform = pd.DataFrame(X_transform\
                           , columns= ('Cluster1', 'Cluster2', 'Cluster3'))
# Add the 'y' column
X_transform['character'] = all_images.character


# Let's plot the results
#3D scatter plot
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'

fig = px.scatter_3d(pd.DataFrame(X_transform)\
                    , x='Cluster1'\
                    , y='Cluster2'\
                    , z='Cluster3'\
                    ,color='character'\
                    , hover_name = list(X_transform.index)
                    )
fig.update_traces(marker=dict(size=5,
                              line=dict(width=2,
                                        color='DarkSlateGrey'))\
            #      ,selector=dict(mode='markers')\
                  )
fig.show()

# Let's have la closser look at some specific images
# Let's start with the one that the algorithm classified without hesitation
# aka the one a the outer edge of the clusters

# index of images at the outer edge of clusters
x_obvious = [173,510,873, 1401, 1491, 989]
[Vis.displayImage(all_images.images[im]\
                      , title = 'Obvious Classification - id: {}'.format(im))\
 for im in x_obvious]
    
# Let's have a look at the center of the plot where all 3 clusters bleed off
# eachothers 
# sus for susbicious
x_sus = [501, 1098, 622, 125, 1504, 505]
[Vis.displayImage(all_images.images[im]\
                      , title = 'Suspisous Classification - id: {}'.format(im))\
 for im in x_sus]
# It looks like if the letters are in itatlic or more artistic the algorithm
# is confused. This can easily be solved with data augmentation techniques

# Lastly, let's have a look the the letters that the algorithm got wrong
x_wrong = [1368, 706, 1219, 822, 180, 620, 906, 635, 134 ]
[Vis.displayImage(all_images.images[im]\
                      , title = 'Wrong Classification - id: {}'.format(im))\
 for im in x_wrong]
# As expected those images are even more stylised/artistic font.
# Except for id: 620, this looks more like a chinese character that an X...
# Let's dig deeper
Vis.displayImage(all_images.images[620]\
                     , title = "Wrong Classification - id 620")
Vis.displayImage(all_images_original.images[620]\
                     , title = "Wrong Class Original Im - id 620")
# It is fair to say that id: 620 is an outlier. 
   
        

#
# This is looking good! 
# but there is a limitation as we can only visualize 3 or less characters
# The solution would be tu run a dimention reduction algorithm like PCA
# to set the number of dimentions to 3 regarless of the number of
# character been tested.
#
# Also we can work on precision as there is some over-lapping
#
# Let's work on decresing the cluster's overlap before add more letters
#























"""

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters= 2, random_state=1)
test = kmeans.fit(sample[0].reshape(-1, 1))
labels = test.labels_
new_im = labels.reshape(40, 40)
Vis.displayImage(new_im, title = "image 1 after kmeans")
# This is over klill, the images a re already quite simple.
# Just converting the images into binary images should be enough

from sklearn.preprocessing import StandardScaler



test = StandardScaler(X = sample[0])
test = test.round()
Vis.displayImage(test, "sample[0] binary")


sample[0].mean()
test = [0 if p < sample[0].mean() else 1 for p in sample[0].reshape(1, -1)]



#
# This good for now, we are done with the data prepping.
# It is time to move on to the data analysis! Yay! 
#

"""

























"""
-------------------------------------------------------------------------------
OOP Training




student = Human(12, "Albert", "York")
print(student.age)

student.gotOlder(5)
print(student.age)



#a_women = Acquaintance('French', 'Brunette')

a_woman = Acquaintance(20, 'Barbara', 'stressand')
a_woman.askAboutTheirDay()

an_other_woman = Acquaintance(25, 'Barbara', "Sting", 'Amrican', 'Pink')

an_other_woman.askAboutTheirDay()
an_other_woman.fname
an_other_woman.gotOlder(2)
an_other_woman.age


"""

























