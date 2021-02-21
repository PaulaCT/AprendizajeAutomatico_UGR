# -*- coding: utf-8 -*-

#############################
#####     LIBRERIAS     #####
#############################
import numpy as np
import matplotlib.pyplot as plot


#-------------------------------------------------------------------------------#
#---------------------- Ejercicio sobre regresión lineal -----------------------#
#-------------------------------------------------------------------------------#


eleccion=input('Escriba 0 para mostrar graficas adicionales: ')
if eleccion=='0':
    extenso=True
else:
    extenso=False

#---------------------------#
#-------- FUNCIONES --------#
#---------------------------#


# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(1)
			else:
				y.append(-1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))		
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	return x, y

# Función para calcular el error
def Err(x,y,w):
    error=0
    tamanio=0
    while tamanio<len(x):
        error+=(np.dot(np.transpose(w),x[tamanio])-y[tamanio])**2
        #error+=(np.dot(w,x[tamanio])-y[tamanio])**2
        tamanio+=1
    media=error/len(x)
    return media

# Derivada de la función Err
def dErr(x,y,w):
    error=0
    tamanio=0
    while tamanio<len(x):
        error+=(np.dot(np.transpose(w),x[tamanio])-y[tamanio])*x[tamanio]
        #error+=(np.dot(w,x[tamanio])-y[tamanio])*x[tamanio]
        tamanio+=1
    media=2*error/len(x)
    return media

# Gradiente Descendente Estocastico
def sgd(x,y,w):	
    #w=np.zeros([3],np.float64)
    num_iteracion=0
    valor_buscado=1e-20    
    max_iteraciones=500
    learning_rate=0.01
    while np.all(Err(x,y,w)>valor_buscado) and num_iteracion<max_iteraciones: 
        x_minibatch,y_minibatch=minibatch(x_train,y_train,32)
        w=w-learning_rate*dErr(x_minibatch,y_minibatch,w)
        num_iteracion+=1 
    return w   

# Selecciona una muestra
def minibatch(x,y,tam_minibatch):
    i=np.random.choice(y.size,tam_minibatch,replace=False)
    x_minibatch=x[i,:]
    y_minibatch=y[i]
    return x_minibatch,y_minibatch

# Algoritmo pseudoinversa	
def pseudoinverse(x, y):
	return np.dot(np.linalg.pinv(x),y)
    
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

# Dibuja la regresión lineal
def regressLin(x_train,_x,_y,leyenda,titulo):
    if leyenda=='SGD':
        color_r='mediumvioletred'
    else:
        color_r='hotpink'
    plot.scatter(x_train[:,1],x_train[:,2],c=y_train,cmap='tab20')
    plot.plot(_x,_y,linewidth=2,color=color_r,label=leyenda)
    plot.legend(loc=4)
    plot.title(titulo)
    plot.show()
    
# Función f
def f(u,v):
    return np.sign((u-0.2)**2+v**2-0.6)

# Cambia aleatoriamente el signo de las etiquetas
def ruido(etiquetas):
    aux=np.random.choice(etiquetas.size,size=(int(etiquetas.size*0.1)))
    etiquetas[aux]*=-1
    return etiquetas

# Repite el experimento 'repeticiones' veces
def experimento(muestra,repeticiones):
    ein=0
    eout=0
    for i in range(repeticiones):
        muestra=simula_unif(1000,2,1)
        muestra_ini=simula_unif(1000,2,1)
        etiquetas_ini=f(muestra_ini[:,0],muestra_ini[:,1])
        etiquetas=f(muestra[:,0],muestra[:,1])
        etiquetas=ruido(etiquetas)
        if repeticiones==1:
            positivo=muestra[etiquetas>=0]
            negativo=muestra[etiquetas<0]
            plot.scatter(positivo[:,0], positivo[:,1],c='lightsalmon',label='positivo')
            plot.scatter(negativo[:,0], negativo[:,1],c='palegreen',label='negativo')
            plot.legend(loc=2)
            plot.title('Muestra de entrenamiento con ruido')
            plot.show()
            input("\n--- Pulsar tecla para continuar ---\n")
        w=np.zeros([3],np.float64)
        datos=np.c_[np.zeros((1000, 1),np.float64),muestra]
        datos_ini=np.c_[np.zeros((1000, 1),np.float64),muestra_ini]
        w=sgd(datos,etiquetas,w)
        ein+=Err(datos_ini,etiquetas_ini,w)
        eout+=Err(datos,etiquetas,w)
    ein=ein/repeticiones
    eout=eout/repeticiones
    return ein,eout


# Lectura de los datos de entrenamiento
x_train,y_train=readData('datos/X_train.npy','datos/y_train.npy')
# Lectura de los datos para el test
x_test,y_test=readData('datos/X_test.npy','datos/y_test.npy')

print ('\t\t ----- EJERCICIO SOBRE REGRESION LINEAL -----\n')
print ('EJERCICIO 1\n')
# Gradiente descendente estocastico

w=np.zeros([3], np.float64)
w=sgd(x_train,y_train,w)

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x_train,y_train,w))
print ("Eout: ", Err(x_test,y_test,w))



_x=np.linspace(0,1,y_train.size)
_y=(-w[0]-w[1]*_x)/w[2]
regressLin(x_train,_x,_y,'SGD','Regresion lineal con SGD')


input("\n--- Pulsar tecla para continuar ---\n")

# Algoritmo Pseudoinversa

w=pseudoinverse(x_train,y_train)

print ('\nBondad del resultado para el algoritmo de la pseudoinversa:\n')
print ("Ein: ", Err(x_train,y_train,w))
print ("Eout: ", Err(x_test,y_test,w))

_x_p=np.linspace(0,1,y_train.size)
_y_p=(-w[0]-w[1]*_x_p)/w[2]
regressLin(x_train,_x_p,_y_p,'pseudoinversa','Regresion lineal con pseudoinversa')

if extenso:
    input("\n--- Pulsar tecla para continuar ---\n")
    plot.scatter(x_train[:,1],x_train[:,2],c=y_train,cmap='tab20')
    plot.plot(_x,_y,linewidth=2,color='mediumvioletred',label='SGD')
    plot.plot(_x_p,_y_p,linewidth=2,color='hotpink',label='pseudoinversa')
    plot.legend(loc=4)
    plot.title('Comparacion de regresiones lineales')
    plot.show()
    
input("\n--- Pulsar tecla para continuar ---\n")


#------------------------------Ejercicio 2 -------------------------------------#


# EXPERIMENTO	
# a) Muestra de entrenamiento N = 1000, cuadrado [-1,1]x[-1,1]	
print ('EJERCICIO 2\n')
print ('Muestra N = 1000, cuadrado [-1,1]x[-1,1]')
muestra=simula_unif(1000,2,1)

plot.scatter(muestra[:,0], muestra[:,1],c='lightsalmon')
plot.title('Muestra de entrenamiento')
plot.show()

input("\n--- Pulsar tecla para continuar ---\n")

ein1,eout1=experimento(muestra,1)
print ('Errores Ein tras 1rep del experimento:\n')
print ("Ein: ", ein1)

# -------------------------------------------------------------------


# d) Ejecutar el experimento 1000 veces
ein_media,eout_media=experimento(muestra,1000) 
print ('Errores Ein y Eout medios tras 1000reps del experimento:\n')
print ("Ein media: ", ein_media)
print ("Eout media: ", eout_media)

