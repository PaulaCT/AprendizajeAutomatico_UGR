# -*- coding: utf-8 -*-

#--------------------------------------------------------------------------------#
#------------------------------ LIBRERIAS --------------------------------------#
#-------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plot

#-------------------------------------------------------------------------------#
#------------------------------ FUNCIONES --------------------------------------#
#-------------------------------------------------------------------------------#

# Función simula_unif genera una lista de N vectores de dimensión dim con números 
# aleatorios uniformes dentro del intervalo rango

def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

# Función simula_recta genera de forma aleatoria los parámetros, v=(a, b) de una 
# recta, y=ax+b, que corta al cuadrado [−50, 50] × [−50, 50].

def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    a = (y2-y1)/(x2-x1) 
    b = y1 - a*x1
    return a, b

#Función ajustar_PLA

def ajustar_PLA(datos, label, max_iter, inicio):
    contador = 0
    aux = inicio
    while contador < max_iter:
        for i in range(len(datos[:,0])):
            if np.sign(np.dot(np.transpose(inicio), datos[i])) != label[i]:
                inicio = inicio + label[i] * datos[i]
        if np.array_equal(inicio,aux):
            break
        aux = inicio
        contador += 1
    return inicio, contador
    
# Función f devuelve el signo de la distancia de cada punto a la recta simulada 
# con simula_recta
    
def f(x,y,a,b):
    return np.sign(y-a*x-b)

# Función aniadir_ruido selecciona aleatoriamente un 10% de las etiquetas positivas
# y un 10% de las negativas y le cambia el signo

def aniadir_ruido(etiquetas):
    etiquetas_ruido=etiquetas
    positivas= np.where (etiquetas>=0)
    negativas = np.where (etiquetas<0)
    positivas = np.array(positivas)
    negativas = np.array(negativas)
    selec_positivas = np.random.choice(positivas.size, size=int(positivas.size*0.1))
    selec_negativas = np.random.choice(negativas.size, size=int(negativas.size*0.1))
    selec_positivas = np.array(selec_positivas)
    selec_negativas = np.array(selec_negativas)
    etiquetas_ruido[selec_positivas]=-1
    etiquetas_ruido[selec_negativas]=1
    return etiquetas_ruido
    
# Función nube_puntos dibuja una gráfica con la nube de puntos de salida clasificados
# o no, junto a una recta divisoria que pasa por dos de dichos puntos

def nube_seleccion(x, titulo,punto1, punto2, etiquetas, etiquetados, f):    
    a = (punto2[1]-punto1[1])/(punto2[0]-punto1[0]) 
    b = punto1[1]-a*punto1[0]
    _y = a*x +b
    plot.axis([0, 2, 0, 2])
    if etiquetados==False:
        plot.scatter(x[:, 0], x[:, 1],color='orchid')
    else:
        x_positivo=x[etiquetas>=0]
        x_negativo=x[etiquetas<0]
        plot.scatter(x_positivo[:,0], x_positivo[:,1], color='orchid', label='positivo') 
        plot.scatter(x_negativo[:,0], x_negativo[:,1], color='turquoise', label='negativo')
    plot.scatter(punto1[0], punto1[1],color='midnightblue',label='seleccionados')
    plot.scatter(punto2[0], punto2[1],color='midnightblue')
    plot.plot(x,_y,color='darkgreen',label='recta')
    plot.title(titulo)
    plot.legend(loc=4)
    plot.show() 
    if (f==False):
        input("\n--- Pulsar tecla para continuar ---\n")

# Función sgd_rl implementa regresión lineal con gradiente descendente estocástico
        
def sgd_rl(x,etiquetas,w):
    minima = 0.01
    learning_rate = 0.01
    num_datos = x.shape[0]
    diferencia = np.inf
    while np.all(diferencia > minima):
        desordenar = np.random.permutation(num_datos)
        x_d = x[desordenar]
        y_d = etiquetas[desordenar]
        aux = w
        gradiente=0
        for xi, y in zip(x_d,y_d):
            gradiente += sigm(xi,y,w)
        gradiente = -gradiente/num_datos
        w = w - learning_rate*gradiente
        diferencia = np.linalg.norm(aux-w)
    return w

# Función sigm

def sigm(x,y,w):
    return np.exp(y*np.dot(np.transpose(w),x))/(1+np.exp(y*np.dot(np.transpose(w),x.reshape(-1))))

# Función err

def err(x,y,w):
    y = y.reshape(-1,1)
    w = w.reshape(-1,1)
    e = np.log(1+np.exp(-y*np.transpose(w)*x))
    return np.mean(e)

# Función eleccion_spyder permite elegir mostrar o no gráficas adicionales desde
# el IDE Spyder (Python 2.7)
    
def eleccion_spyder():
    print ('Escriba 0 para mostrar graficas adicionales: ')
    eleccion = input()
    if eleccion==0:
        return True
    else:
        return False
    
# Función eleccion_terminal permite elegir mostrar o no gráficas adicionales desde
# la terminal o versiones distintas del IDE Spyder
        
def eleccion_terminal():
    eleccion=input('Escriba 0 para mostrar graficas adicionales: ')
    if eleccion=='0':
        return True
    else:
        return False 

#-------------------------------------------------------------------------------#
#----------------- Ejercicio sobre modelos lineales ----------------------------#
#-------------------------------------------------------------------------------#


# Fijamos la semilla
        
np.random.seed(10)

# Mostrar gráficas adicionales; descomentar en caso de no mostrarse  

#extenso=eleccion_spyder()
extenso=eleccion_terminal()

#-------------------------------------------------------------------------------#
#----------------------- Apartado 1 --------------------------------------------#
#-------------------------------------------------------------------------------#

print ('\nEJERCICIO 1\n')

# Ejecutar el algoritmo PLA con los datos simulados en el apartados 2a del ejercicio 1

x = simula_unif(100, 2, [-50,50])
a, b = simula_recta([-50,50])
etiquetas=f(x[:,0],x[:,1],a,b)
x = np.c_[np.ones((100, 1), np.float64), x]

# Inicializar el algoritmo,anotar el número medio de iteraciones necesarias para 
# converger y valorar el resultado relacionando el punto de inicio con el número 
# de iteraciones con: 

# a) el vector cero 

inicio = np.zeros(3)
w, iteracion = ajustar_PLA(x, etiquetas, 5000, inicio)
print ('------------------------------ Sin ruido ----------------------------- \n')
print ('\t Punto inicial \t\t\t\t\t\t\t\t Iteracion')
print ('0\t [', inicio[0], inicio[1], inicio[2],']\t\t\t\t\t\t\t', iteracion)
media=iteracion

#b) vectores de números aleatorios en [0, 1] (10 veces). 

vectores = simula_unif(10, 3, [0,1])
ix=1
for i in vectores:
    w, iteracion =  ajustar_PLA(x, etiquetas, 5000, i)
    media+=iteracion
    
    if ix==5 or ix==8:
        print (ix, '\t [', i[0], i[1], i[2],'] \t', iteracion)
    else:
        print (ix, '\t [', i[0], i[1], i[2],'] \t\t', iteracion)
    ix += 1

print ("Nº medio de iteraciones: ", media/11)
input("\n--- Pulsar tecla para continuar ---\n")

# Repetir con los datos simulados en el apartado 2b del ejercicio 1 ¿Observa algún 
# comportamiento diferente?

print ('\n\n------------------------------ Con ruido ----------------------------- \n')

etiquetas_ruido = aniadir_ruido(etiquetas)
w, iteracion = ajustar_PLA(x, etiquetas, 5000, inicio)
print ('\t Punto inicial \t\t\t\t\t\t\t\t Iteracion')
print ('0\t [', inicio[0], inicio[1], inicio[2],']\t\t\t\t\t\t\t', iteracion)
media=iteracion
ix=1
for i in vectores:
    w, iteracion =  ajustar_PLA(x, etiquetas, 5000, i)
    media+=iteracion
    if ix==8 or ix==5:
        print (ix, '\t [', i[0], i[1], i[2],'] \t', iteracion)
    else:
        print (ix, '\t [', i[0], i[1], i[2],'] \t\t', iteracion)
    ix += 1
    
print ("Nº medio de iteraciones: ", media/11)
    
print ('\n\nAl añadirle ruido a las etiquetas, el número máximo de iteraciones asignado (5000) no')
print ('resulta suficiente como para que el algoritmo converja')
input("\n--- Pulsar tecla para continuar ---\n")

#-------------------------------------------------------------------------------#
#----------------------- Apartado 2 --------------------------------------------#
#-------------------------------------------------------------------------------#

print ('\nEJERCICIO 2\n')

# Generar una muestra de 100 puntos de dimensión 2 para [0,2]×[0,2]. Aplicar la función
# f para clasificar los puntos en función a una recta que se generará seleccionando
# aleatoriamente 2 puntos de la muestra.

d = simula_unif(100, 2, [0,2])
punto1 = d[np.random.choice(99)]
punto2 = d[np.random.choice(99)]

a = (punto2[1]-punto1[1])/(punto2[0]-punto1[0]) 
b = punto1[1]-a*punto1[0]
etiquetas=f(d[:,0],d[:,1],a,b)

if extenso:
    nube_seleccion(d,'Conjunto de datos d junto a recta generada a partir de puntos',
                   punto1,punto2,etiquetas,False,False)
    
nube_seleccion(d,'Conjunto de datos d etiquetados junto a recta',
               punto1,punto2,etiquetas,True,True)

# Implementar Regresión Logística(RL) con Gradiente Descendente Estocástico
# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar
# Eout usando para ello un número suficientemente grande de nuevas muestras

_x = np.linspace(0, 2, 100)
_y = a * _x + b
d = np.c_[np.ones(d.shape[0]), d]
w = np.zeros(len(d[0]))
w_regresion = sgd_rl(d,etiquetas,w)
print('Error de regresion logistica en la muestra: ',err(d,etiquetas,w))
input("\n--- Pulsar tecla para continuar ---\n")


n_d = simula_unif(1500, 2, [0,2])
punto1 = n_d[np.random.choice(1499)]
punto2 = n_d[np.random.choice(1499)]
a = (punto2[1]-punto1[1])/(punto2[0]-punto1[0]) 
b = punto1[1]-a*punto1[0]
n_etiquetas=f(n_d[:,0],n_d[:,1],a,b)
if extenso:
    nube_seleccion(n_d,'Conjunto de datos n_d junto a recta generada a partir de puntos',
                   punto1,punto2,n_etiquetas,False,False)
    
nube_seleccion(n_d,'Conjunto de datos n_d etiquetados junto a recta',
               punto1,punto2,n_etiquetas,True,True)

n_d = np.c_[np.ones(1500), n_d]
n_w = sgd_rl(n_d, n_etiquetas, w)

print('Error de regresion logistica en la nueva muestra:', err(n_d, n_etiquetas, n_w))

