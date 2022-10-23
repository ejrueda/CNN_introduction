
import paramiko
from scp import SCPClient
from datetime import datetime
import time
import os

def create_SSH_Client(host, port, user, password_name, password_path):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port, user, password_name, key_filename=password_path)
    return client

ip_servidor = "3.142.240.64"
ruta_clave = "../../../proyectos_ai/delivery_robot/Deliveryrobotpassword.pem"

while True:
    fecha = datetime.now()
    if fecha.hour == 15:
        #print("Entrando")
        #se cargan las imaǵenes al servidor
        ssh = create_SSH_Client(ip_servidor, 22, "ubuntu","Deliveryrobotpassword.pem", ruta_clave)
        scp = SCPClient(ssh.get_transport())
        scp.put('../Images', recursive=True,
                remote_path='/home/ubuntu/delivery_robot_project/static/')
        #se cierra la conexión al terminar
        ssh.close()
        #se ejecuta el script en aws que permite reentrenar el modelo de tensorflow
        # y genera un nuevo modelo en tensorflowlite
        os.system('ssh -i '+ruta_clave+' ubuntu@'+ip_servidor+' "sudo bash /home/ubuntu/bash_entrenar.sh"')
        
        #Después de entrenador y generado el tflite en el servidor, nos traemos el .tflite para que
        #funcione en la interfaz de tkinter
        os.system('scp -i '+ruta_clave+' ubuntu@'+ip_servidor+':~/delivery_robot_project/static/model_tflite_'+str(fecha.month)+'_'+str(fecha.day)+'.tflite'+' ../data/')
        #se eliminan las imágenes del servidor
        break
        time.sleep(60*60) #se para la ejecución por una hora
