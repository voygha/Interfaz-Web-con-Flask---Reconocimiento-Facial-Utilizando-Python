from flask import Flask
from flask import render_template, request,redirect,url_for,flash
from flaskext.mysql import MySQL
import smtplib
from decouple import  config
import cv2
import os

#se crea la aplicacion
app= Flask(__name__)

@app.route('/')
def index():
    return render_template('alumnos/index.html')
#AQUI TERMINA INDEX

#AQUI ES PARA AGREGAR UN NUEVO CLIENTE
@app.route('/create')
def create():
   return render_template('alumnos/create.html')
#AQUI TERMINA

#EDITAR REGISTROS
@app.route('/indicaciones')
def indicaciones():  
    return render_template('alumnos/indicaciones.html')
#TERMINA EDITAR REGISTRO


@app.route('/activar')
def activar():  
    return render_template('alumnos/activar.html')

#EDITAR REGISTROS
@app.route('/verificar')
def verificar():
    
	dataPath = 'C:/Users/LUISINPERRIN/Desktop/Prototipointerfaz/Data' #Cambia a la ruta donde hayas almacenado Data
	imagePaths = os.listdir(dataPath)
	print('imagePaths=',imagePaths)

	face_recognizer = cv2.face.EigenFaceRecognizer_create()
	#face_recognizer = cv2.face.FisherFaceRecognizer_create()
	#face_recognizer = cv2.face.LBPHFaceRecognizer_create()

	# Leyendo el modelo
	face_recognizer.read('modeloEigenFace.xml')
	#face_recognizer.read('modeloFisherFace.xml')
	#face_recognizer.read('modeloLBPHFace.xml')

	#cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

	#cap = cv2.VideoCapture('imagenes y Videos de Prueba/prueba6.mp4')

	#cap = cv2.VideoCapture('imagenes y Videos de Prueba/Luis.mp4')

	cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #captureDevice = camera
	#cap = cv2.VideoCapture(1)

	faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

	while True:
		
		ret,frame = cap.read()
		if ret == False: break
		

		#if cv2.waitKey(1) & 0xFF == ord('q'):
			#break 

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		auxFrame = gray.copy()

		faces = faceClassif.detectMultiScale(gray,1.3,5)

		for (x,y,w,h) in faces:
			rostro = auxFrame[y:y+h,x:x+w]
			rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
			result = face_recognizer.predict(rostro)

			cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)


			
			# FisherFace
			if result[1] < 10000:
				cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
				#return render_template('/alumnos/verificar.html')
			else:
				cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
			
			# LBPHFace
			#if result[1] < 70:
			#	cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
			#	cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
			#else:
			#	cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
			#	cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
			
		cv2.imshow('frame',frame)
		k = cv2.waitKey(1)
		if k == 27:
			break

	cap.release()
	cv2.destroyAllWindows()










    
#TERMINA EDITAR REGISTRO

if __name__ == '__main__':
    app.run(debug=True)
