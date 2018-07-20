
import numpy as np
import os

from datashape import json
from tensorflow import keras
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
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from skimage.measure import regionprops
import matplotlib.patches as mpatches
from skimage.morphology import label
from skimage import *
from scipy import misc, ndimage
from keras.callbacks import TensorBoard
import sys


# In[2]:


train_path = 'Training'
test_path = 'Test'
valid_path = 'Valid'


# In[3]:


train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224),classes=['Healthy_Leaves','Unhealthy_Leaves'], batch_size=20)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224),classes=['Healthy_Leaves','Unhealthy_Leaves'], batch_size=20)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224),classes=['Healthy_Leaves','Unhealthy_Leaves'], batch_size=5)


# In[4]:


def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label') 
    plt.xlabel('Predicted label')

def replace_image(image, image_to_be_replaced_path, filename):
    cv2.imwrite(os.path.join(path , filename), image)
model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)),
    Flatten(),
    Dense(2,activation='softmax')
])

vgg16_model = keras.applications.vgg16.VGG16()


model = Sequential()
for layer in vgg16_model.layers:
    model.add(layer)


model.layers.pop()



for layer in model.layers:
    layer.trainable = False



model.add(Dense(2, activation='softmax'))



model.compile(Adam(lr=.0001),loss='categorical_crossentropy',metrics=['accuracy'])

tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
model.fit_generator(train_batches, steps_per_epoch=4,
                   validation_data=valid_batches, validation_steps=4,epochs=1,verbose=2, callbacks=[tensorBoardCallback])



def read_in():
    lines = input()
    return lines

def analyzePhoto():
    directory = read_in()

    image = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224),classes=[directory], batch_size=20)
    classify_image = model.predict_generator(image,steps=1,verbose=0)
    return classify_image

while True:
    print(analyzePhoto())