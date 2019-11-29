import keras
from keras.models import load_model
from PIL import Image
import numpy as np
import os
import time

if __name__ == "__main__":
    image = np.array(Image.open("images/text-align/train/right/3PMMT_04_장비관리__text-align__2901072.png")).reshape(-1, 331, 331, 3)
    model = load_model("model/mobilenet-02-1.7277.hdf5")
    pas = time.time()
    predict = model.predict(image) 
    classes = os.listdir()
    print(predict)
    fu = time.time()
    print(f"time: {fu-pas}")
    
