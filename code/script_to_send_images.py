
import paramiko
from scp import SCPClient
import datetime
import time

def create_SSH_Client(host, port, user, password_name, password_path):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port, user, password_name, key_filename=password_path)
    return client

while True:
    hora = datetime.datetime.now().hour
    if hora == 24:
        #se cargan las imaǵenes al servidor
        ssh = create_SSH_Client("18.116.37.95", 22, "ubuntu","Deliveryrobotpassword.pem",
                     "../../../proyectos_ai/delivery_robot/Deliveryrobotpassword.pem")
        scp = SCPClient(ssh.get_transport())
        scp.put('../data/tf_lite_images', recursive=True,
                remote_path='/home/ubuntu/delivery_robot_project/static/')
        #se cierra la conexión al terminar
        ssh.close()
    time.sleep(60*60) #se para la ejecución por una hora
