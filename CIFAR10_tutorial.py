#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# In[2]:


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("Number of training examples:", len(x_train))
print("Number of test examples:", len(x_test))


# In[3]:


y_train = y_train.flatten()
y_test = y_test.flatten()


# In[4]:


plt.imshow(x_train[5])
print(y_train[5])


# In[5]:


batch_size = 32
img_height = 32
img_width = 32


# In[6]:


normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)


# In[7]:


num_classes = 10


# In[8]:


data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)


# In[9]:


plt.figure(figsize=(10, 10))
images = x_train
for i in range(9):
    augmented_images = data_augmentation(images)[0]
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images.numpy().astype("uint8"))
    plt.axis("off")


# In[10]:


model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])


# In[11]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[12]:


# Save weights to be able to reset later
model.save_weights('model.h5')
model.summary()


# In[13]:


epochs = 15
history = model.fit(
  x_train, y_train,
  validation_data=(x_test, y_test),
  epochs=epochs
)


# In[14]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[17]:


# Load initialized weights
model.load_weights('model.h5')

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(
  x_train, y_train,
  validation_data=(x_test, y_test),
  epochs=1000,
  callbacks=[callback]
)


# In[16]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

