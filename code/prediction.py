
import tensorflow as tf
import numpy as np
import pickle as pk
from PIL import Image

class cnn_images_10:
    
    #función para inicializar las variables requeridas para el modelo
    def __init__(self, ruta_modelo, ruta_translate, ruta_class):
        #cargamos el modelo entrenado
        self.model = tf.keras.models.load_model(ruta_modelo)
        #cargamos el nombre de las clases
        self.class_names = pk.load(open(ruta_class, "rb"))
        #cargamos el translate.py para traducir las clases de las imágenes
        self.translate = pk.load(open(ruta_translate, "rb"))
    
    #función para dada una ruta de imagen, predecir a que categoría corresponde
    def predict(self, ruta_imagen):
        image = Image.open(ruta_imagen).resize((100,100))
        img_procesada = np.asarray(image).reshape((1,100,100,3))
        prediction = self.model(img_procesada)
        indice = np.argmax(prediction)
        y_predict = self.translate[self.class_names[indice]]
        return y_predict
