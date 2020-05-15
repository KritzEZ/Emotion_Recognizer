#!/usr/bin/env python
# coding: utf-8

# # Emotion Analyzer Training Model

# In this program we are creating a model, training the model and saving the strusture and data set weights to use for the video analyzer. 

# In[13]:


import pandas as pd
import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split

df=pd.read_csv('fer2013.csv')


# Splitting the single pixel values of each picture and apending to array (creating a 2D array)
pixels = []
for index, row in df.iterrows():
    val=row['pixels'].split(" ")
    pixels.append(np.array(val, 'float32'))

#Splitting the training and testing data (10% is testing data)
x_train, x_test, y_train, y_test = train_test_split(pixels, df['emotion'], test_size=0.1)

x_train = np.array(x_train,'float32')
x_test = np.array(x_test, 'float32')

y_train = keras.utils.to_categorical(y_train, 7)
y_test = keras.utils.to_categorical(y_test, 7)

#Normalizing the data values
x_train = (x_train - np.mean(x_train, axis=0))/np.std(x_train, axis=0)
x_test = (x_test - np.mean(x_test, axis=0))/np.std(x_test, axis=0)

#Reshaping the size of the array
x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)


#1st convolution layer
model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1), activation="relu"))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))

#2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))

#3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(Flatten())

#fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(7, activation="softmax"))

#Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


#Training the model with 200 iterations, and allows data to shuffle order
model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=200,
    validation_data=(x_test, y_test),
    shuffle=True
)

#Saving the model strusture and weights to files for use in video analyzing
model_structure = model.to_json()
f = Path("model_structure.json")
f.write_text(model_structure)

model.save_weights("model_weights.h5")


# ### End Note:
# 
# After the program is trained, it is able to achieve about 90% accuracy. Although this is good, it can be imporved by trying different types of data rescaling techniques like rescaling to -1 to 1 scale or standardizing. We can also imporve the results by tuning the learning rate, the droupout rate, early stopping or trying different activiation functions.
# 
