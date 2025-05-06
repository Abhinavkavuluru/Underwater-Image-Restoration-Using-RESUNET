import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, LSTM, TimeDistributed, Conv2DTranspose, Dropout, Concatenate, concatenate, UpSampling2D, BatchNormalization, Activation, Add, Reshape, Multiply, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt

# Constants
IMG_SIZE = (256, 256)
batch_size = 32
epochs = 50
steps_per_epoch = 30  # Define steps per epoch

# Load images
def load_images(directory):
    images = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0 
        images.append(img_array)
    return np.array(images)

# Load datasets
degraded_images = load_images('input')
ground_truth_images = load_images('GT')

# Split datasets
train_degraded, remaining_degraded, train_gt, remaining_gt = train_test_split(degraded_images, ground_truth_images, test_size=0.2, random_state=42)
test_degraded, val_degraded, test_gt, val_gt = train_test_split(remaining_degraded, remaining_gt, test_size=0.5, random_state=42)

X_train = train_degraded
y_train = train_gt
X_val = val_degraded
y_val = val_gt

# Defining SiLU activation function
def silu(x):
    return x * tf.math.sigmoid(x)

def conv_block(inputs, filters, kernel_size=(3, 3), padding='same', strides=1):
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(inputs)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    # ECA module
    squeeze = GlobalAveragePooling2D()(conv)
    reshape = Reshape((1, 1, filters))(squeeze)
    conv_weights = Conv2D(filters // 8, (1, 1), padding='same', activation=silu)(reshape)
    conv_weights = Conv2D(filters, (1, 1), padding='same', activation='sigmoid')(conv_weights)
    scaled_features = Multiply()([conv, conv_weights])

    return scaled_features

def res_block(inputs, filters, kernel_size=(3, 3), padding='same', strides=1):
    conv1 = conv_block(inputs, filters, kernel_size, padding, strides)
    conv2 = conv_block(conv1, filters, kernel_size, padding)
    shortcut = Conv2D(filters, (1, 1), padding=padding, strides=strides)(inputs)
    shortcut = BatchNormalization()(shortcut)
    output = Add()([conv2, shortcut])
    output = Activation('relu')(output)
    return output

def res_unetsilu(input_shape=(256, 256, 3), base_filters=32):
    inputs = Input(shape=input_shape)

    # Encoder Part
    conv1 = conv_block(inputs, base_filters)
    pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = res_block(pool1, base_filters * 2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    conv3 = res_block(pool2, base_filters * 4)
    pool3 = MaxPooling2D((2, 2))(conv3)
    conv4 = res_block(pool3, base_filters * 8)
    pool4 = MaxPooling2D((2, 2))(conv4)

    # Bottleneck
    conv5 = res_block(pool4, base_filters * 16)

    # Decoder Part
    up6 = Conv2D(base_filters * 8, (2, 2), padding='same')(UpSampling2D(size=(2, 2))(conv5))
    up6 = Concatenate(axis=3)([conv4, up6])
    conv6 = res_block(up6, base_filters * 8)
    up7 = Conv2D(base_filters * 4, (2, 2), padding='same')(UpSampling2D(size=(2, 2))(conv6))
    up7 = Concatenate(axis=3)([conv3, up7])
    conv7 = res_block(up7, base_filters * 4)
    up8 = Conv2D(base_filters * 2, (2, 2), padding='same')(UpSampling2D(size=(2, 2))(conv7))
    up8 = Concatenate(axis=3)([conv2, up8])
    conv8 = res_block(up8, base_filters * 2)
    up9 = Conv2D(base_filters, (2, 2), padding='same')(UpSampling2D(size=(2, 2))(conv8))
    up9 = Concatenate(axis=3)([conv1, up9])
    conv9 = res_block(up9, base_filters)

    output = Conv2D(3, (1, 1), activation='sigmoid')(conv9)

    model1 = Model(inputs=inputs, outputs=output)
    return model1

model1 = res_unetsilu()
model1.compile(optimizer=Adam(), loss=MeanSquaredError())
model1.summary()

history = model1.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), steps_per_epoch=steps_per_epoch)

# Save the model
model1.save('m2.keras')

test_loss = model1.evaluate(test_degraded, test_gt)
print("Test Loss:", test_loss)

predictions = model1.predict(test_degraded)

import numpy as np

def calculate_psnr(gt_images, pred_images, max_val=255):
    mse = np.mean(np.square(gt_images - pred_images))
    psnr = 10 * np.log10((max_val ** 2) / mse)
    return psnr

# Assuming predictions and test_gt are already defined
psnr = calculate_psnr(test_gt, predictions)
print("PSNR is:", psnr)

from skimage.metrics import structural_similarity as ssim

def calculate_ssim(gt_images, pred_images, data_range=255, win_size=3):
    ssim_values = []
    for i in range(len(gt_images)):
        score, _ = ssim(gt_images[i], pred_images[i], win_size=win_size, full=True, data_range=data_range)
        ssim_values.append(score)
    return np.mean(ssim_values)

# Assuming predictions and test_gt are already defined
ssim_score = calculate_ssim(test_gt, predictions)
print("SSIM:", ssim_score)
