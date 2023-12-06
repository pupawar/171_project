 # -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 20:57:15 2023

@author: Hongyu Jiang
All the training process are recorded here instead of on the notebook
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import re
import matplotlib.patches as patches
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import tensorflow as tf
from sklearn.model_selection import train_test_split
from IPython.display import display
from PIL import Image
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add
from tensorflow.keras.layers import Input, LeakyReLU, Concatenate, UpSampling2D, ZeroPadding2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import pickle

#Here we define a residual block to make it more convient to implement darknet53
def residual_block(x, num_filters, filter_size):
    # Shortcut connection
    shortcut = x

    # First convolutional layer (1x1 filter to reduce the number of channels)
    x = Conv2D(num_filters, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second convolutional layer (3x3 filter)
    x = Conv2D(num_filters * 2, (filter_size, filter_size), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Adding the shortcut to the output
    x = Add()([x, shortcut])
    
    return x

def darknet53(input_layer):
    
    x = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    
    for i in range(1):
        x = residual_block(x, 32, 3)
    
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    
    for i in range(2):
        x = residual_block(x, 64, 3)
    
    x = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    
    for i in range(8):
        x = residual_block(x, 128, 3)
    route_1 = x
    
    x = layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    
    for i in range(8):
        x = residual_block(x, 256, 3)
    route_2 = x
    
    x = layers.Conv2D(1024, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    
    for i in range(4):
        x = residual_block(x, 512, 3)
    
    return route_1, route_2, x#The reason why we do this is because YOLOv3 extracts features in three different sizes


def YOLOv3(input_layer, num_classes=15):
    route_1, route_2, conv = darknet53(input_layer)

    #large size objects
    conv = Conv2D(512, (1, 1), strides=(1, 1), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU(alpha=0.1)(conv)

    conv = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU(alpha=0.1)(conv)

    conv = Conv2D(512, (1, 1), strides=(1, 1), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU(alpha=0.1)(conv)

    conv = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU(alpha=0.1)(conv)

    conv = Conv2D(512, (1, 1), strides=(1, 1), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU(alpha=0.1)(conv)

    conv_lobj_branch = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv)
    conv_lobj_branch = BatchNormalization()(conv_lobj_branch)
    conv_lobj_branch = LeakyReLU(alpha=0.1)(conv_lobj_branch)

    # Output for large-size objects
    conv_lbbox = Conv2D(3 * (num_classes + 5), (1, 1), strides=(1, 1), padding='same', use_bias=True)(conv_lobj_branch)
    
    
    #mid size objects
    conv = Conv2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU(alpha=0.1)(conv)

    # Upsampling and concatenation
    conv = UpSampling2D(size=(2, 2))(conv)
    conv = Concatenate()([conv, route_2])

    conv = Conv2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU(alpha=0.1)(conv)

    conv = Conv2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU(alpha=0.1)(conv)

    conv = Conv2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU(alpha=0.1)(conv)

    conv = Conv2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU(alpha=0.1)(conv)

    conv = Conv2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU(alpha=0.1)(conv)

    conv_mobj_branch = Conv2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv)
    conv_mobj_branch = BatchNormalization()(conv_mobj_branch)
    conv_mobj_branch = LeakyReLU(alpha=0.1)(conv_mobj_branch)

    # Output for mid-size objects
    conv_mbbox = Conv2D(3 * (num_classes + 5), (1, 1), strides=(1, 1), padding='same', use_bias=True)(conv_mobj_branch)
    
    
    #tiny size objects
    conv = UpSampling2D(size=(2, 2))(conv)
    conv = Concatenate()([conv, route_1])

    conv = Conv2D(128, (1, 1), strides=(1, 1), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU(alpha=0.1)(conv)

    conv = Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU(alpha=0.1)(conv)

    conv = Conv2D(128, (1, 1), strides=(1, 1), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU(alpha=0.1)(conv)

    conv = Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU(alpha=0.1)(conv)

    conv = Conv2D(128, (1, 1), strides=(1, 1), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv)
    conv = BatchNormalization()(conv)
    conv = LeakyReLU(alpha=0.1)(conv)

    conv_sobj_branch = Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0005))(conv)
    conv_sobj_branch = BatchNormalization()(conv_sobj_branch)
    conv_sobj_branch = LeakyReLU(alpha=0.1)(conv_sobj_branch)

    # Output for tiny-size objects
    conv_sbbox = Conv2D(3 * (num_classes + 5), (1, 1), strides=(1, 1), padding='same', use_bias=True)(conv_sobj_branch)
    
    conv_sbbox = Conv2D(30, (1, 1), strides=(1, 1), padding='same', activation='linear')(conv_sbbox)
    conv_mbbox = Conv2D(30, (1, 1), strides=(1, 1), padding='same', activation='linear')(conv_mbbox)
    conv_lbbox = Conv2D(30, (1, 1), strides=(1, 1), padding='same', activation='linear')(conv_lbbox)
    
    model = Model(inputs = input_layer, outputs = [conv_sbbox, conv_mbbox, conv_lbbox])
    return model


model = YOLOv3(Input(shape =(96, 96, 3)))

output1 = model.layers[-3].output  # 12*12*30
output2 = model.layers[-2].output  # 6*6*30
output3 = model.layers[-1].output  # 3*3*30

# Apply global average pooling to each output
pooled_output1 = GlobalAveragePooling2D()(output1)
pooled_output2 = GlobalAveragePooling2D()(output2)
pooled_output3 = GlobalAveragePooling2D()(output3)

# Concatenate the pooled outputs
concatenated_outputs = Concatenate()([pooled_output1, pooled_output2, pooled_output3])

# Fully connected layer to get the keypoints
keypoints = Dense(30, activation='linear')(concatenated_outputs)

# Create the new model
custom_model = Model(inputs=model.input, outputs=keypoints)

# Compile and train the model
custom_model.compile(optimizer = Adam(learning_rate = 0.001), loss='mean_squared_error', metrics=['mae', 'acc'])

# Print model summary
#custom_model.summary()

def sort_pattern(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def sort_pattern(filename):
    # Extract numerical part of the filename for sorting, assuming the format 'name_number.jpg'
    num_part = ''.join(filter(str.isdigit, filename))
    return int(num_part) if num_part.isdigit() else float('inf')  # Return a large number if no digits found

def load_images(path):
    images = []
    file_list = os.listdir(path)
    sorted_files = sorted(file_list, key=sort_pattern)

    for filename in sorted_files:
        if filename.endswith(".jpg"):
            img_path = os.path.join(path, filename)
            # Open the image and convert it to grayscale
            img = Image.open(img_path).convert('L')
            # Convert the grayscale image to RGB
            img_rgb = img.convert('RGB')
            # Append the RGB image as a NumPy array to the images list
            images.append(np.array(img_rgb))
            img.close()

    return images

def show_keypoints(images, keypoints, n_rows=3, n_cols=7):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows), dpi=100)

    for i, ax in enumerate(axes.flatten()):
        img_array = np.array(images[i], dtype=np.uint8)  # Convert list of images to NumPy array
        img_array = img_array.squeeze()  # Remove singleton dimensions
        img = Image.fromarray(img_array)  # Convert NumPy array to PIL Image
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Sample #{i}')

        for j in range(0, len(keypoints[i]), 2):
            x = keypoints[i][j]
            y = keypoints[i][j + 1]
            ax.plot(x, y, 'ro', markersize=2)

    plt.tight_layout()
    plt.show()

train_data = pd.read_csv('training.csv')
path = r"images\train_images"
images = load_images(path)
#show_keypoints(images[:21], train_data.values[:21])

train_data.isna().mean().round(4) * 100
train_data.fillna(train_data.describe().T['50%'], inplace=True)
train_data.sample(5).T

x_train = np.array(images).reshape(-1, 96, 96, 3).astype('float64')
y_train = train_data.values.astype('float64')

# Reduce learning rate for improving convergence, and early stopping
# Dynamically adjust learning rate
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='min', baseline=None)
reduce_LR = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5, min_lr=1e-15, mode='min', verbose=1)

callback = tf.keras.callbacks.TerminateOnNaN()
history = custom_model.fit(x_train, y_train, validation_split=0.2, epochs = 100, callbacks=[reduce_LR, early_stopping], batch_size=32)

# Save the model
custom_model.save('YOLOv3.h5')
with open('History-YOLOv3.pkl', 'wb') as f:
    pickle.dump(history.history, f)
    
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.close()

# Plot and save the accuracy graph
plt.plot(history.history['acc'], label='Training Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')
plt.close()

# Plot and save the learning rate graph
plt.plot(history.history['lr'], label='Learning Rate')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.legend()
plt.savefig('learning_rate_plot.png')
plt.close()

# Plot and save the mean squared error graph
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error (MAE)')
plt.legend()
plt.savefig('mae_plot.png')
plt.close()
