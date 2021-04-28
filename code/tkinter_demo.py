
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os, shutil
import numpy as np
import tensorflow as tf
import cv2
import time

class Root(tk.Tk):
    
    def __init__(self):
        super(Root,self).__init__() #se hereda de la clase Tk
        self.fileName = ""
        self.title("Delivery robot Neural Network")
        self.minsize(350,500)
        self.labelFrame = tk.LabelFrame(self,text="Open an image")
        self.labelFrame.grid(column=0,row=1,padx= 175, pady= 20)
        self.btton()
        #label frame para cuando se pinta la imágen
        self.lf_2 = tk.LabelFrame(self, text="Input image")
        self.lf_2.grid(column=0,row=3)
        #label frame para cuando se pinta la predicción
        self.lf_3 = tk.LabelFrame(self, text="Prediction")
        self.lf_3.grid(column=0,row=5)
        self.count_initial_predict = 0
        self.v_class = tk.StringVar()
        self.v_proba = tk.StringVar()
        self.iconphoto(False, tk.PhotoImage(file='../images/robot.png'))
        
        ## cargamos el modelo de tensorflow lite
        self.interpreter = tf.lite.Interpreter(model_path="../model_tflite.tflite")
        self.interpreter.allocate_tensors()
        #detalles de los tensores de entrada y salida
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        #tamaño de entrada
        self.input_shape = self.input_details[0]['shape']
        
        #self.take_photo_and_predict()
        
    def btton(self):
        self.button = tk.Button(self.labelFrame, text="Select a photo", command=self.fileDialog)
        self.button.grid(column=0,row=1)
        
    def fileDialog(self):
        self.fileName = filedialog.askopenfilename(initialdir = "../data/archive/", title="Select A File",
                                                   filetypes=(("jpeg","*.jpg"),("png","*.png")))
        #self.label = tk.Label(self.labelFrame, text="")
        #self.label.grid(column =0,row = 2)
        #self.label.configure(text = "Selected image: " + self.fileName.split("/")[-1])
        
        #abrimos y dibujamos la imagen
        load = Image.open(self.fileName).resize((self.input_shape[1],self.input_shape[2]), Image.BICUBIC)
        render = ImageTk.PhotoImage(load)
        img = tk.Label(self.lf_2, image=render)
        img.image = render
        img.grid(column=0,row=3)
        #pintamos el botón de predecir
        pre_btn = tk.Button(self, text="Predict",command=lambda: self.predict(np.array(load,
                                                                                       dtype=np.float32)))
        pre_btn.grid(column=0, row=4, pady=15)
        
    def predict(self, image):
        input_image = np.expand_dims(image, axis=0)
        #predicción
        self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
        #se invoca el interprete para que haga la predicción
        self.interpreter.invoke()
        #se obtiene el tensor de salida del modelo
        output_tensor = self.interpreter.get_tensor(self.output_details[0]['index'])
        idx = np.argmax(output_tensor)
        proba = output_tensor[0][idx]
        #idx = np.random.choice(range(image.shape[0]))
        #proba = np.random.choice(range(101))
        if self.count_initial_predict == 0:
            self.v_class.set("Class: " + str(idx))
            self.v_proba.set("Probability: " + str(proba) + "%")
            #texto de predicción
            lb_class = tk.Label(self.lf_3, textvariable=self.v_class)
            lb_class.grid(column=0,row=5)
            lb_proba = tk.Label(self.lf_3, textvariable=self.v_proba)
            lb_proba.grid(column=0,row=6)
            self.count_initial_predict += 1
        else:
            self.v_class.set("Class: " + str(idx))
            self.v_proba.set("Probability: " + str(proba) + "%")
        
        
    def take_photo_and_predict(self):
        #se accede a la camara mediante opencv y te toma una captura
        cam = cv2.VideoCapture(0)   # 0 -> index of camera
        flag_photo, img = cam.read()
        if flag_photo: #si se captura la foto sin errores
            cv2.namedWindow("cam-test", cv2.WINDOW_AUTOSIZE)
            #imshow("cam-test",img)
            #waitKey(0)
            cv2.destroyWindow("cam-test")
            #cv2.imwrite("testfilename"+str(i)+".jpg",img) #save image
            cam.release()
            #se visualiza la imágen
            render = ImageTk.PhotoImage(img)
            img = tk.Label(self.lf_2, image=render)
            img.image = render
            img.grid(column=0,row=3)
        
    
if __name__ == '__main__':
    
    root = Root()
    root.mainloop()
