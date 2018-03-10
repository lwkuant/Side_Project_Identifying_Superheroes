# -*- coding: utf-8 -*-
"""
Identifying Superheroes from Product Images
https://www.crowdanalytix.com/contests/identifying-superheroes-from-product-images
"""

import re
from itertools import chain
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import os
os.chdir('D:/Dataset/Competition_Identifying_Superheroes_from_Product_Images')

import shutil

from collections import Counter

seed = 1000

from skimage.io import imread

from time import time

import pickle

### Set up the Important Variables
height = 200
width = 200
num_class = 12
num_train = 5000
num_val = 433
train_dir = 'D:/Dataset/Competition_Identifying_Superheroes_from_Product_Images/data_train'
val_dir = 'D:/Dataset/Competition_Identifying_Superheroes_from_Product_Images/data_valid'



hero_names = os.listdir('CAX_Superhero_Train')
hero_names = [re.match('^\w.+', x).group(0) for x in hero_names if re.match('^\w+', x) != None]

hero_id = list(range(len(hero_names)))

hero_dict = {hero_names[i]: hero_id[i] for i in range(len(hero_names))}

image_name = [[i for i in os.listdir('D:/Dataset/Competition_Identifying_Superheroes_from_Product_Images/CAX_Superhero_Train/' + j) if i != '.DS_Store'] for j in hero_names]
image_name = list(chain(*image_name))
image_label = [[i]*(len(os.listdir('D:/Dataset/Competition_Identifying_Superheroes_from_Product_Images/CAX_Superhero_Train/' + i)) - 1) for i in hero_names]
image_label = list(chain(*image_label))

hero_df = pd.DataFrame(np.c_[image_name, image_label])
hero_df.columns = ['Image_name', 'Image_label']
hero_df['Image_class'] = hero_df['Image_label'].apply(lambda x: hero_dict[x])

hero_df.shape

hero_df['Image_label'].value_counts()

hero_df_train, hero_df_valid = train_test_split(hero_df, test_size = 433, stratify = hero_df['Image_class'],
                                                shuffle = True, random_state = seed)

os.mkdir('./data_train')
os.mkdir('./data_valid')

for i in range(len(set(hero_df['Image_class']))):
    os.mkdir('./data_train/'+str(i))
    os.mkdir('./data_valid/'+str(i))
    
for i in range(hero_df_train.shape[0]):
    shutil.copy(os.path.join('D:/Dataset/Competition_Identifying_Superheroes_from_Product_Images/CAX_Superhero_Train',  hero_df_train['Image_label'].values[i], hero_df_train['Image_name'].values[i]),
                os.path.join('D:/Dataset/Competition_Identifying_Superheroes_from_Product_Images/data_train', str(hero_df_train['Image_class'].values[i])))
    print(i)
    
for i in range(hero_df_valid.shape[0]):
    shutil.copy(os.path.join('D:/Dataset/Competition_Identifying_Superheroes_from_Product_Images/CAX_Superhero_Train',  hero_df_valid['Image_label'].values[i], hero_df_valid['Image_name'].values[i]),
                os.path.join('D:/Dataset/Competition_Identifying_Superheroes_from_Product_Images/data_valid', str(hero_df_valid['Image_class'].values[i])))
    print(i)
    

pic_shape = [imread('D:/Dataset/Competition_Identifying_Superheroes_from_Product_Images/CAX_Superhero_Train/' + y + '/' + x).shape \
             for y in hero_names for x in os.listdir('D:/Dataset/Competition_Identifying_Superheroes_from_Product_Images/CAX_Superhero_Train/' + y) if x != '.DS_Store']

for file in os.listdir('D:/Dataset/Competition_Identifying_Superheroes_from_Product_Images/CAX_Superhero_Test/'):
    if file != '.DS_Store':
        pic_shape.append(imread('D:/Dataset/Competition_Identifying_Superheroes_from_Product_Images/CAX_Superhero_Test/' + file).shape)

Counter(pic_shape).most_common(5)


### InceptionResNetV2
np.random.seed(seed)
batch_size = 1
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionResNetV2

conv_base = InceptionResNetV2(weights='imagenet',
                  include_top=False,
                  input_shape=(height, width, 3))

conv_base.summary() 
final_layer_shape = [4, 4, 1536]

datagen = ImageDataGenerator(rescale=1./255)

def extract_features(directory, sample_count, batch_size):
    features = np.zeros(shape=([sample_count] + final_layer_shape))
    labels = np.zeros(shape=(sample_count, num_class))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='categorical')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        print(i)
        if i * batch_size >= sample_count:
            break
    return features, labels

start_time = time()
train_features, train_labels = extract_features(train_dir, num_train, batch_size) # 711.58s
print(time() - start_time)

start_time = time()
validation_features, validation_labels = extract_features(val_dir, num_val, 1) # 61.51415181159973s
print(time() - start_time)

with open('./train_features_InceptionResNetV2', 'wb') as fp:
    pickle.dump(train_features, fp)
with open('./train_labels_InceptionResNetV2', 'wb') as fp:
    pickle.dump(train_labels, fp)
with open('./validation_features_InceptionResNetV2', 'wb') as fp:
    pickle.dump(validation_features, fp)
with open('./validation_InceptionResNetV2', 'wb') as fp:
    pickle.dump(validation_labels, fp)    

with open('./train_features_InceptionResNetV2', 'rb') as fp:
    train_features_InceptionResNetV2 = pickle.load(fp)
with open('./train_labels_InceptionResNetV2', 'rb') as fp:
    train_labels = pickle.load(fp)
with open('./validation_features_InceptionResNetV2', 'rb') as fp:
    validation_features_InceptionResNetV2 = pickle.load(fp)
with open('./validation_InceptionResNetV2', 'rb') as fp:
    validation_labels = pickle.load(fp)   

train_feature_InceptionResNetV2 = np.reshape(train_feature_InceptionResNetV2, (num_train, np.prod(final_layer_shape)))
validation_features_InceptionResNetV2 = np.reshape(validation_features_InceptionResNetV2, (num_val, np.prod(final_layer_shape)))



### 
np.random.seed(seed)
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=np.prod(final_layer_shape)))
model.add(layers.Dense(num_class, activation='softmax'))
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
    loss='categorical_crossentropy',
    metrics=['acc'])
history = model.fit(train_features, train_labels,
    epochs=50,
    batch_size=20,
    validation_data=(validation_features, validation_labels))

import matplotlib.pyplot as plt
import seaborn as sns
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

sns.set_style("darkgrid")
plt.figure(figsize=[20, 15])
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.plot(epochs, acc, 'ko', label='Training Accuracy')
plt.plot(epochs, val_acc, 'k', label='Validation Accuracy')
plt.yticks(np.linspace(0, 1, 6))
plt.xticks(np.linspace(0, 50, 6))
plt.title('Training and Validation Accuracy', fontsize=25)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.legend(loc=4, prop={'size': 15})
plt.savefig('Accuracy_InceptionResNetV2_1_layer_256.png')

sns.set_style("darkgrid")
plt.figure(figsize=[20, 15])
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.plot(epochs, loss, 'ko', label='Training Loss')
plt.plot(epochs, val_loss, 'k', label='Validation Loss')
plt.title('Training and Validation Loss', fontsize=25)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.legend(loc=0, prop={'size': 15})
plt.savefig('Loss_InceptionResNetV2_1_layer_256.png')


### InceptionV3
np.random.seed(seed)
batch_size = 1
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3

conv_base = InceptionV3(weights='imagenet',
                  include_top=False,
                  input_shape=(height, width, 3))

conv_base.summary() 
final_layer_shape = [4, 4, 2048]

datagen = ImageDataGenerator(rescale=1./255)

def extract_features(directory, sample_count, batch_size):
    features = np.zeros(shape=([sample_count] + final_layer_shape))
    labels = np.zeros(shape=(sample_count, num_class))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='categorical')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        print(i)
        if i * batch_size >= sample_count:
            break
    return features, labels

start_time = time()
train_features, train_labels = extract_features(train_dir, num_train, batch_size) # 711.58s
print(time() - start_time)

start_time = time()
validation_features, validation_labels = extract_features(val_dir, num_val, 1) # 61.51415181159973s
print(time() - start_time)

with open('./train_features_InceptionV3', 'wb') as fp:
    pickle.dump(train_features, fp)
with open('./train_labels_InceptionV3', 'wb') as fp:
    pickle.dump(train_labels, fp)
with open('./validation_features_InceptionV3', 'wb') as fp:
    pickle.dump(validation_features, fp)
with open('./validation_InceptionV3', 'wb') as fp:
    pickle.dump(validation_labels, fp)    

with open('./train_features_InceptionV3', 'rb') as fp:
    train_feature_InceptionV3 = pickle.load(fp)
with open('./train_labels_InceptionV3', 'rb') as fp:
    train_labels = pickle.load(fp)
with open('./validation_features_InceptionV3', 'rb') as fp:
    validation_features_InceptionV3 = pickle.load(fp)
with open('./validation_InceptionV3', 'rb') as fp:
    validation_labels = pickle.load(fp)   

train_feature_InceptionV3 = np.reshape(train_feature_InceptionV3, (num_train, np.prod(final_layer_shape)))
validation_features_InceptionV3 = np.reshape(validation_features_InceptionV3, (num_val, np.prod(final_layer_shape)))


### 
np.random.seed(seed)
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=np.prod(final_layer_shape)))
model.add(layers.Dense(num_class, activation='softmax'))
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
    loss='categorical_crossentropy',
    metrics=['acc'])
history = model.fit(train_feature_InceptionV3, train_labels,
    epochs=50,
    batch_size=20,
    validation_data=(validation_features_InceptionV3, validation_labels))

import matplotlib.pyplot as plt
import seaborn as sns
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

sns.set_style("darkgrid")
plt.figure(figsize=[20, 15])
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.plot(epochs, acc, 'ko', label='Training Accuracy')
plt.plot(epochs, val_acc, 'k', label='Validation Accuracy')
plt.yticks(np.linspace(0, 1, 6))
plt.xticks(np.linspace(0, 50, 6))
plt.title('Training and Validation Accuracy', fontsize=25)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.legend(loc=4, prop={'size': 15})
plt.savefig('Accuracy_InceptionV3_1_layer_256.png')

sns.set_style("darkgrid")
plt.figure(figsize=[20, 15])
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.plot(epochs, loss, 'ko', label='Training Loss')
plt.plot(epochs, val_loss, 'k', label='Validation Loss')
plt.title('Training and Validation Loss', fontsize=25)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.legend(loc=0, prop={'size': 15})
plt.savefig('Loss_InceptionV3_1_layer_256.png')


### InceptionV3 + InceptionResNetV2
with open('./train_features_InceptionResNetV2', 'rb') as fp:
    train_features_InceptionResNetV2 = pickle.load(fp)
with open('./train_labels_InceptionResNetV2', 'rb') as fp:
    train_labels = pickle.load(fp)
with open('./validation_features_InceptionResNetV2', 'rb') as fp:
    validation_features_InceptionResNetV2 = pickle.load(fp)
with open('./validation_InceptionResNetV2', 'rb') as fp:
    validation_labels = pickle.load(fp)   
    
with open('./train_features_InceptionV3', 'rb') as fp:
    train_features_InceptionV3 = pickle.load(fp)
with open('./train_labels_InceptionV3', 'rb') as fp:
    train_labels = pickle.load(fp)
with open('./validation_features_InceptionV3', 'rb') as fp:
    validation_features_InceptionV3 = pickle.load(fp)
with open('./validation_InceptionV3', 'rb') as fp:
    validation_labels = pickle.load(fp)     
    
train_features_InceptionResNetV2 = np.reshape(train_features_InceptionResNetV2, (num_train, np.prod([4, 4, 1536])))
validation_features_InceptionResNetV2 = np.reshape(validation_features_InceptionResNetV2, (num_val, np.prod([4, 4, 1536])))
    
train_features_InceptionV3 = np.reshape(train_features_InceptionV3, (num_train, np.prod([4, 4, 2048])))
validation_features_InceptionV3 = np.reshape(validation_features_InceptionV3, (num_val, np.prod([4, 4, 2048])))

train_features = np.c_[train_features_InceptionResNetV2, train_features_InceptionV3]
validation_features = np.c_[validation_features_InceptionResNetV2, validation_features_InceptionV3]

np.random.seed(seed)
batch_size = 1
from keras import models
from keras import layers
from keras import optimizers
np.random.seed(seed)
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=np.prod(train_features.shape[1])))
model.add(layers.Dense(num_class, activation='softmax'))
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
    loss='categorical_crossentropy',
    metrics=['acc'])
history = model.fit(train_features, train_labels,
    epochs=50,
    batch_size=20,
    validation_data=(validation_features, validation_labels))


import matplotlib.pyplot as plt
import seaborn as sns
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

sns.set_style("darkgrid")
plt.figure(figsize=[20, 15])
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.plot(epochs, acc, 'ko', label='Training Accuracy')
plt.plot(epochs, val_acc, 'k', label='Validation Accuracy')
plt.yticks(np.linspace(0, 1, 6))
plt.xticks(np.linspace(0, 50, 6))
plt.title('Training and Validation Accuracy', fontsize=25)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.legend(loc=4, prop={'size': 15})
plt.savefig('Accuracy_InceptionResNetV2+InceptionV3_1_layer_256.png')

sns.set_style("darkgrid")
plt.figure(figsize=[20, 15])
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.plot(epochs, loss, 'ko', label='Training Loss')
plt.plot(epochs, val_loss, 'k', label='Validation Loss')
plt.title('Training and Validation Loss', fontsize=25)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.legend(loc=0, prop={'size': 15})
plt.savefig('Loss_InceptionResNetV2+InceptionV3_1_layer_256.png')