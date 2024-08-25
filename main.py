import sys
import os
import numpy as np
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QLibraryInfo
from simple_pyspin import Camera
import cv2 
# chao
sys.path.append('../') #
from stokeslib.polarization_full_dec_array import polarization_full_dec_array
from stokeslib.calcular_stokes import calcular_stokes
from stokeslib.calcular_mueller_canal_inv import calcular_mueller_canal_inv 
from stokeslib.acoplar_mueller import acoplar_mueller 
from stokeslib.mueller_mean import mueller_mean
from stokeslib.normalizar_mueller import normalizar_mueller
from camaralib.digitalizar import digitalizar
from raspberrylib.runcmd import runcmd
from stokeslib.calcular_propiedades import calcular_aolp
from camaralib.guardar_mueller import guardar_mueller_canal

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)

# Configuración inicial cámara
exposure_time = 50000
N = 1
decimador = 1

# Matrices de Stokes
dimx = 2448
dimy = 2048
S_in_stat_inv = np.zeros((dimy//2,dimx//2,3,3))[::decimador,::decimador]
S_out_stat = S_in_stat_inv.copy()

class Ui(QMainWindow):
    def __init__(self, cam):
        super(Ui, self).__init__()

        #Objeto cámara
        self.cam = cam

        # Carga GUI diseñado
        loadUi('gui/gui.ui',self)      
            
        #Inicializa Cámara
        self.start_cam(self)

        #Configura Cámara
        self.config_cam(self, exposure_time, decimador)

        #Muestra imagen
        self.start_recording(self) 

        #Espera teclas movimiento
        self.move_cam(self)

        #Espera boton captura
        self.capture_listen(self)

        #Muestra GUI
        self.show()

    def start_cam(self, label):

        #Exposicion
        self.cam.ExposureAuto = 'Off'
    	
        #Formato
        self.cam.PixelFormat = "BayerRG8"
        
        #Stokes de entrada
        self.S_in_stat_inv = S_in_stat_inv

        #Inicia cámara
        self.cam.start()

    def config_cam(self, label, exposure_time, decimador):
        
        #Tiempo de exposicion
        self.cam.ExposureTime = exposure_time # microseconds

        #Número de promedios
        self.decimador = decimador

    def start_recording(self, label):   	
        timer = QtCore.QTimer(self)
        
        #Conexión
        timer.timeout.connect(self.update_image)
        timer.start(0)
        self.update_image()    

    def update_image(self):
        #Captura imagen
        img = self.cam.get_array()

        #Medibles
        I90, I45, I135, I0 = polarization_full_dec_array(img)

        #Stokes
        S_out_stat[:,:,0,:], S_out_stat[:,:,1,:], S_out_stat[:,:,2,:] = calcular_stokes(I90, I45, I135, I0, decimador = self.decimador)

        ### Calculo de Mueller ###
        self.M_shot = calcular_mueller_canal_inv(self.S_in_stat_inv,S_out_stat)
        M_show = cv2.cvtColor(cv2.applyColorMap(digitalizar(acoplar_mueller(self.M_shot[::4,::4,:,:]), 'M8'), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        print('M', mueller_mean(self.M_shot))
        print('aolp '+ str(calcular_aolp(S_out_stat[0,0,0,1], S_out_stat[0,0,0,2])*180/np.pi) +' '+ str(calcular_aolp(S_out_stat[0,0,1,1], S_out_stat[0,0,1,2])*180/np.pi) + ' ' + str(calcular_aolp(S_out_stat[0,0,2,1], S_out_stat[0,0,2,2])*180/np.pi))
        
        #Formato Array to PixMap
        h, w, _ = M_show.shape
        S0QIMG = QImage(M_show, w, h, 3*w, QImage.Format_RGB888)
        pixmap = QPixmap(S0QIMG)

        #Plot
        self.S0.setPixmap(pixmap)

    def move_up(self):
        self.S_in_stat_inv = np.linalg.pinv(S_out_stat.copy())

    def move_down(self):
        guardar_mueller_canal(self.M_shot, 'mueller', 'mueller_show')
        cv2.imwrite('mueller/mueller.png', digitalizar(acoplar_mueller(self.M_shot), 'M16'))

    def move_left(self):
        with open('S_in_inv.npy', 'wb') as f:
            np.save(f, self.S_in_stat_inv)

    def move_right(self):
        with open('S_in_inv.npy', 'rb') as f:
            self.S_in_stat_inv = np.load(f)[::decimador,::decimador,:,:]
   
    def move_cam(self, label):
        up_btn = self.up_btn
        up_btn.clicked.connect(self.move_up)
        
        dwn_btn = self.dwn_btn
        dwn_btn.clicked.connect(self.move_down)
        
        left_btn = self.left_btn
        left_btn.clicked.connect(self.move_left)
        
        right_btn = self.right_btn
        right_btn.clicked.connect(self.move_right)
    
    def auto_capture(self):
        runcmd("cd ../; python3 capturar_muestra.py", verbose=True)

    def capture_listen(self, label):
        capture_btn = self.capture_btn
        capture_btn.clicked.connect(self.auto_capture)

def main(cam):
    app = QApplication(sys.argv)
    instance = Ui(cam)
    app.exec_()

if __name__ == "__main__":

    #Inicia GUI
    with Camera() as cam:
        main(cam)      
    
    #Detener cámara
    cam.stop()  

    #Salir
    sys.exit()
