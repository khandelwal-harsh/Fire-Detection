import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import cv2
import os


class FireDetection:
    def __init__(self):
        print("[INFO] Initialized the Fire Detection Model")
        print(os.listdir())
        self.model = load_model('firenet_v2.h5')

    def predict(self, img, min_score):
        """
        Input : img (str, io.BytesIO, np.ndarray): image path or image
                min_score (float)                : confidence threshold
        --------------------------------------------------------------------
        Output: output tuple                     : (score, class_name)
                img (np.ndarray)                 : RGB image
        """
        width = 224
        height = 224

        if type(img) is not np.ndarray:
            img = Image.open(img)
            img = np.array(img)


        x = cv2.resize(img, (width, height))

        x = x.astype("float") / 255.0
        x = img_to_array(x)
        x = np.expand_dims(x, axis=0)

        pred = self.model.predict(x)

        output = []
        if pred[0][0] >= min_score:
            label = 'Fire'
            output.append((pred[0][0], label))
        else:
            label = 'No-Fire'
            output.append((pred[0][1],label))

        return output, img



