#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import autokeras as ak

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# In[4]:


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("Number of training examples:", len(x_train))
print("Number of test examples:", len(x_test))


# In[5]:


y_train = y_train.flatten()
y_test = y_test.flatten()


# In[6]:


plt.imshow(x_train[5])
print(y_train[5])


# In[9]:


# Initialize the image classifier.
clf = ak.ImageClassifier()
# Feed the image classifier with training data.
clf.fit(x_train, y_train)


# In[10]:


best_model = clf.export_model()
print(best_model.evaluate(x_test, y_test))

