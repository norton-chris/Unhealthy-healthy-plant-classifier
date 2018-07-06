
# coding: utf-8

# In[ ]:


import numpy as np
import keras
import cv2 as cv2
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from skimage.viewer.plugins import measure
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from skimage.measure import regionprops
import matplotlib.patches as mpatches
from skimage.morphology import label
from skimage import *
#get_ipython().magic('matplotlib inline')


# In[2]:


train_path = 'Training'
test_path = 'Test'


# In[3]:


train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224),classes=['Healthy_Leaves','Unhealthy_Leaves'], batch_size=10)
test_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224),classes=['Healthy_Leaves','Unhealthy_Leaves'], batch_size=10)


# In[ ]:


img = cv2.imread('Training/Healthy_Leaves/ny1053-01-1.jpg',0)
edges = cv2.Canny(img,200,200)

laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y


plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()


# In[ ]:


img = cv2.imread('Training/Healthy_Leaves/ny1053-01-1.jpg',0)
img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
ret, labels = cv2.connectedComponents(img)

# Map component labels to hue val
label_hue = np.uint8(179*labels/np.max(labels))
blank_ch = 255*np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

# cvt to BGR for display
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

# set bg label to black
labeled_img[label_hue==0] = 0

cv2.imshow('labeled.png', labeled_img)
#cv2.waitKey()


# In[ ]:


#axes = plt.subplots(ncols=1, nrows=1)
#ax = axes.flat
label_image = label(sobelx)
# for region in regionprops(label_image):
#     # Draw rectangle around segmented coins.
#     minr, minc, maxr, maxc = region.bbox
#     rect = mpatches.Rectangle((minc, minr),
#                               maxc - minc,
#                               maxr - minr,
#                               fill=False,
#                               edgecolor='red',
#                               linewidth=2)
    #ax0.add_patch(rect)

#plt.tight_layout()
# plt.imshow(label_image)
# roi = regionprops(label_image)
# plt.show()
plt.imshow(label_image)
blobs = measure.label(label_image, connectivity=1)
props = measure.regionprops(blobs)
roi = regionprops(props)
cv2.imshow(roi)
cv2.rectangle(labeled_img)

