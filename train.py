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

class TrainModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.MODEL_SAVE_FOLDER_PATH = f'./model/{model_name}'
        if not os.path.exists(self.MODEL_SAVE_FOLDER_PATH):
            os.mkdir(self.MODEL_SAVE_FOLDER_PATH)
        self.model_path = self.MODEL_SAVE_FOLDER_PATH + 'mobilenetv2-' + '{epoch:02d}-{val_loss:.4f}.hdf5'
        self.n_classes = len(os.listdir(f"images/{self.model_name}/train/"))

    def run(self):
        cb_checkpoint = ModelCheckpoint(filepath=self.model_path, monitor='val_loss', verbose=1, save_best_only=True)
        cb_early_stopping = EarlyStopping(monitor='val_loss', patience=10)

        train_datagen = ImageDataGenerator(rescale = 1./255,
                                          shear_range = 0.2,
                                          zoom_range = 0.2,
                                          horizontal_flip = True)

        valid_datagen = ImageDataGenerator(rescale = 1./255)

        # prepare your data
        train_generator = train_datagen.flow_from_directory(f'images/{self.model_name}/train/',
                                                        target_size = (331, 331),
                                                        batch_size = 10,
                                                        class_mode = 'categorical')


        valid_generator =valid_datagen.flow_from_directory(f'images/{self.model_name}/valid/',
                                                    target_size = (331, 331),
                                                    batch_size = 10,
                                                    class_mode = 'categorical')

        # build model
        MobileNetv2, preprocess_input = Classifiers.get('mobilenetv2')

        base_model = MobileNetv2(input_shape=(331,331,3), weights='imagenet', include_top=False)
        x = keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = keras.layers.Dense(self.n_classes, activation='softmax')(x)
        model = keras.models.Model(inputs=[base_model.input], outputs=[output])


        # # Compiling the CNN
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

        # train
        history = model.fit_generator(train_generator,
                                steps_per_epoch = 30000,
                                epochs = 1000,
                                validation_data = valid_generator,
                                validation_steps = 2000,
                                callbacks=[cb_checkpoint, cb_early_stopping])

if __name__=="__main__":
    train_model = TrainModel("text-align")
    train_model.run()


