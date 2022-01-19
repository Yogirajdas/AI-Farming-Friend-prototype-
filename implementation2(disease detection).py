import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
#(this data is obtained directly from kaggle to google colab)
dir = '/content/gdrive/MyDrive/Kaggle/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)'
image_size=224

train_gen = keras.preprocessing.image.ImageDataGenerator()
test_gen = keras.preprocessing.image.ImageDataGenerator()

train_gen = train_gen.flow_from_directory("/content/gdrive/MyDrive/Kaggle/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train", target_size=(image_size,image_size), class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', follow_links=False, subset=None, interpolation='nearest')
test_gen = test_gen.flow_from_directory("/content/gdrive/MyDrive/Kaggle/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid", target_size=(image_size,image_size), class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', follow_links=False, subset=None, interpolation='nearest')

Lables = train_gen.class_indices
print(Lables)
type(Lables)

#sorting labels properly.
Lables.keys()
labels_list = list(Lables.keys())
labels_list

details_list = []
for i in labels_list:
  details = i.split("___")
  details_list.append(details)
print(details_list)

print(details_list[0][1])

copy_list = details_list.copy()
print(copy_list)

count=0
for i in range(len(copy_list)) :
  if copy_list[i][1] == 'healthy':
    count = count+1
    ##copy_list[i][1].replace('healthy', '')
    copy_list[i].remove('healthy')
print(copy_list)

import os
N = train_gen.labels.shape[0]
positive_frequencies = []
negative_frequencies = []
for i in range(38):
  if details_list[i][1]!='healthy':
    list = os.listdir('/content/gdrive/MyDrive/Kaggle/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/' + str(labels_list[i])) # dir is your directory path
    number_files = len(list)
    positive_frequencies.append(number_files/N)
    negative_frequencies.append(1 - number_files/N)

print('positive_frequencies = ', positive_frequencies)
print('negative_frequencies = ', negative_frequencies)
len(positive_frequencies)

pos_weights = negative_frequencies
neg_weights = positive_frequencies

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, AveragePooling2D, Activation, BatchNormalization, ZeroPadding2D

from tensorflow import keras
import keras.backend as K

base_model = keras.applications.DenseNet121(
    include_top=True, weights=None, input_tensor=None,
    input_shape=None, pooling=None, classes=1000
)

base_model.summary()

def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
   
    
    def weighted_loss(y_true, y_pred):
    
        
        # initialize loss to zero
        loss = 0.0
        
       

        for i in range(len(pos_weights)):
            # for each class, add average weighted loss for that class
            loss_pos = -1 * K.mean(pos_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon))
            loss_neg = -1 * K.mean(neg_weights[i] * (1 - y_true[:, i]) * K.log(1 - y_pred[:, i] + epsilon))
            loss += loss_pos + loss_neg
        
        return loss
    
        
    return weighted_loss

base_model.trainable = False

# Create new model on top
inputs = keras.Input(shape=(224, 224, 3))
x1 = base_model(inputs, training=False)


x = keras.layers.Dropout(0.2)(x1)  
outputs = keras.layers.Dense(38, activation="sigmoid")(x)

model = keras.Model(inputs, outputs, name='leaf_disease_model')
model.summary()

model.compile(optimizer=keras.optimizers.Adam(),loss=get_weighted_loss(pos_weights, neg_weights), metrics=[keras.metrics.CategoricalAccuracy()])

from keras.callbacks import EarlyStopping

# EarlyStopping callback.
early_stop = EarlyStopping(monitor='val_loss', 
                           patience=3, 
                           verbose=1)

callbacks_list = [early_stop]

history = model.fit(train_gen,
                        steps_per_epoch=300,  
                        validation_data=test_gen,
                        epochs=5,
                        validation_steps=300,
                        callbacks=callbacks_list)

