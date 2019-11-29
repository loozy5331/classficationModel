import os

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from classification_models.keras import Classifiers


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# load models
MODEL_SAVE_FOLDER_PATH = './model/'
if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
  os.mkdir(MODEL_SAVE_FOLDER_PATH)

model_path = MODEL_SAVE_FOLDER_PATH + 'mobilenet-' + '{epoch:02d}-{val_loss:.4f}.hdf5'

cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
cb_early_stopping = EarlyStopping(monitor='val_loss', patience=10)

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

valid_datagen = ImageDataGenerator(rescale = 1./255)

# prepare your data
train_generator = train_datagen.flow_from_directory('images/text-align/train/',
                                                 target_size = (331, 331),
                                                 batch_size = 10,
                                                 class_mode = 'categorical')


valid_generator =valid_datagen.flow_from_directory('images/text-align/valid/',
                                            target_size = (331, 331),
                                            batch_size = 10,
                                            class_mode = 'categorical')


# build model
MobileNetv2, preprocess_input = Classifiers.get('mobilenetv2')
n_classes = list(os.listdir("images/text-align/train/"))

base_model = MobileNetv2(input_shape=(331,331,3), weights='imagenet', include_top=False)
x = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(len(n_classes), activation='softmax')(x)
model = keras.models.Model(inputs=[base_model.input], outputs=[output])


# # Compiling the CNN
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# train
history = model.fit_generator(train_generator,
                         steps_per_epoch = 300,
                         epochs = 3,
                         validation_data = valid_generator,
                         validation_steps = 2000,
                         callbacks=[cb_checkpoint, cb_early_stopping])

