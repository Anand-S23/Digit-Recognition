#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Activation
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD


# In[2]:


# create the training & test sets, skipping the header row with [1:]
train = pd.read_csv("train.csv")
print(train.shape)
train.head()


# In[3]:


test= pd.read_csv("test.csv")
print(test.shape)
test.head()


# In[4]:


x_train = (train.iloc[:,1:])# all pixel values
y_train = train.iloc[:,0] # only labels i.e targets digits
x_test = test


# In[5]:


x_train


# In[6]:


y_train


# In[7]:


x_train=x_train/255.
x_test=x_test/255.


# In[8]:


x_train=x_train.values.reshape(-1,28,28,1)
x_test=x_test.values.reshape(-1,28,28,1)

x_train.shape


# In[9]:


y_train_vectors=to_categorical(y_train)


# In[10]:


x_train, x_val, y_train, y_val= train_test_split(x_train, y_train_vectors, test_size=0.2, random_state=2)


# In[11]:


print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)


# In[12]:


model=Sequential()
model.add( Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='valid', activation=None, use_bias=False, input_shape=(x_train.shape[1:])) )
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add( Conv2D(filters=48, kernel_size=(3,3), strides=(1,1), padding='valid', activation=None, use_bias=False) )
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add( Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='valid', activation=None, use_bias=False) )
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add( Conv2D(filters=80, kernel_size=(3,3), strides=(1,1), padding='valid', activation=None, use_bias=False) )
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add( Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), padding='valid', activation=None, use_bias=False) )
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add( Conv2D(filters=112, kernel_size=(3,3), strides=(1,1), padding='valid', activation=None, use_bias=False) )
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add( Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='valid', activation=None, use_bias=False) )
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add( Conv2D(filters=144, kernel_size=(3,3), strides=(1,1), padding='valid', activation=None, use_bias=False) )
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add( Conv2D(filters=160, kernel_size=(3,3), strides=(1,1), padding='valid', activation=None, use_bias=False) )
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add( Conv2D(filters=176, kernel_size=(3,3), strides=(1,1), padding='valid', activation=None, use_bias=False) )
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(units=10))
model.add(BatchNormalization())
model.add(Activation('softmax'))


# In[13]:


sgd = SGD(learning_rate=0.05)

model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])


# In[14]:


train_datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=False,
                             rotation_range=10,
                             zoom_range=0.1,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=False,
                             vertical_flip=False
                            )

train_generator = train_datagen.flow(x_train, y_train,
                                     batch_size=120,
                                     shuffle=True)

val_datagen = ImageDataGenerator()
val_generator = val_datagen.flow(x_val, y_val,
                                 batch_size=120,
                                 shuffle=True)


# In[15]:


trained_model = model.fit(train_generator,validation_data=(val_generator), epochs=1, batch_size=50)


# In[16]:


predictions_final = model.predict(x_test)

print(predictions_final)

predictions_final = np.argmax(predictions_final, axis=1)

print(predictions_final)


# In[ ]:




