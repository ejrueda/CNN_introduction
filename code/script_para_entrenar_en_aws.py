
import tensorflow as tf
import os
import numpy as np
from datetime import datetime

batch_size = 64
img_height = 100
img_width = 100
data_root='./Images'
print("datos de entrenamiento")
data_train = tf.keras.preprocessing.image_dataset_from_directory(str(data_root),
                                                               validation_split=None,
                                                               subset="training",
                                                               seed=123,
                                                               image_size=(img_height, img_width),
                                                               batch_size=batch_size,
                                                               label_mode='categorical')

#definimos el model_augmentation que transforma las imágenes de entrada
model_augmentation = tf.keras.Sequential(name="augmentation_model")
model_augmentation.add(tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"))
model_augmentation.add(tf.keras.layers.experimental.preprocessing.RandomRotation(.1, fill_mode="constant"))
model_augmentation.add(tf.keras.layers.experimental.preprocessing.RandomZoom(.2, fill_mode="constant"))
#mapeamos los datos de entrada
map_data_train = data_train.map(lambda x, y: (model_augmentation(x), y))
#cargamos el modelo entrenado
modelo_entrenado = tf.keras.models.load_model("./models_finalversionmap15.0version145x145")
#entrenamos el modelo
hist_model = modelo_entrenado.fit(map_data_train, epochs=10, validation_split=.1)
#guardamos un json que contiene el historial de entrenamiento
open("./resultados_entrenamiento.json", "w").write(hist_model)
#gurdamos el modelo entrenado
#por ahora no sobreescribimos el modelo, lo guardamos con el mes y el dia
fecha = datetime.now()
modelo_entrenado.save("./modelo_reentrenado_"+str(fecha.month)+"_"+str(fecha.day))
#generamos el modelo en formato tensorflowlite a partir del modelo reentrenado
converter = tf.lite.TFLiteConverter.from_keras_model(modelo_entrenado)
tflite = converter.convert()
#guardamos el modelo de tensorflowlite (tambien por mes y dia)
open("../model_tflite_"+str(fecha.month)+"_"+str(fecha.day)+".tflite", "wb").write(tflite)
