# -*- coding: utf-8 -*-

#############################
#####     LIBRERIAS     #####
#############################
import numpy as np
import matplotlib.pyplot as plot


#-------------------------------------------------------------------------------#
#------------- Ejercicio sobre la búsqueda iterativa de óptimos ----------------#
#-------------------------------------------------------------------------------#


eleccion=input('Escriba 0 para mostrar graficas adicionales: ')
if eleccion=='0':
    extenso=True
else:
    extenso=False

#---------------------------#
#-------- FUNCIONES --------#
#---------------------------#

# Función E
def E(u,v): 
    return ((u*np.exp(v))-2*v*np.exp(-u))**2

# Derivada parcial de E respecto de u
def Eu(u,v):
    return  2*np.exp(-2*u)*(u*np.exp(u+v)-2*v)*(np.exp(u+v)+2*v)

# Derivada parcial de E respecto de v
def Ev(u,v):
	return 2*np.exp(-2*u)*(u*np.exp(u+v)-2)*(u*np.exp(u+v)-2*v)
#
	
# Gradiente de E
def gradE(u,v):
	return np.array([Eu(u,v), Ev(u,v)], dtype=np.float64)

# Gradiente descendiente para E
def gdE(u,v,learning_rate,valor_buscado,max_iteraciones):
    num_iteracion=0
    valores=[]
    iteraciones=[]
    w=np.array([u,v])
    while np.all(E(w[0],w[1])>valor_buscado) and num_iteracion<=max_iteraciones:
        valores.append(E(w[0],w[1]))
        iteraciones.append(num_iteracion)
        w=w-learning_rate*gradE(w[0],w[1])
        num_iteracion+=1
    return w[0],w[1],num_iteracion,valores,iteraciones

# Función para dibujar entorno de E
def entornoE(_x,_y,_z, u, v, etiqueta, posicion): 
    plot.contourf(_x, _y, _z, 100)
    plot.title('Entorno de la funcion E')
    plot.colorbar()
    plot.plot(u,v,'o',c='deeppink', label=etiqueta)
    plot.legend(loc=posicion)
    plot.show()
    
#Función f
def f(x,y):
    return (x-2)**2+2*(y+2)**2+2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

#Derivada parcial de f respecto de x
def fx(x,y):
    return  2*(2*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)+x-2)

#Derivada parcial de f respecto de y
def fy(x,y):
    return  4*(np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)+y+2)

# Gradiente de f
def gradf(x,y):
	return np.array([fx(x,y), fy(x,y)], dtype=np.float64)

# Gradiente descendiente para f
def gdf(x,y,learning_rate,valor_buscado,max_iteraciones):
    num_iteracion=0
    valores=[]
    iteraciones=[]
    w=np.array([x,y])
    while np.all(f(w[0],w[1])>valor_buscado) and num_iteracion<=max_iteraciones:
        valores.append(f(w[0],w[1]))
        iteraciones.append(num_iteracion)
        w=w-learning_rate*gradf(w[0],w[1])
        num_iteracion+=1
    return w[0],w[1],num_iteracion,valores,iteraciones

# Función para dibujar entorno de f
def entornof(_x,_y,_z, x, y, etiqueta, posicion): 
    plot.contourf(_x, _y, _z, 1000)
    plot.title('Entorno de la funcion f')
    plot.colorbar()
    plot.plot(x,y,'o',c='deeppink', label=etiqueta)
    plot.legend(loc=posicion)
    plot.show()


print ('\n\t\t ----- GRADIENTE DESCENDENTE -----')
print ('\nEJERCICIO 2\n')
res = 100
_x = np.linspace(0, 1.5, res)
_y = np.linspace(0, 1.5, res)
_z = np.zeros((res, res))
for ix, x in enumerate(_x):
    for iy, y in enumerate(_y):
        _z[iy, ix] = E(x,y)
        
if extenso:
    entornoE(_x,_y,_z,1,1,'punto inicial',4)
    input("\n--- Pulsar tecla para continuar ---\n") 

learning_rate=0.1
max_iteraciones=10000000
valor_buscado=1e-14

u_sol,v_sol,num_iteracion,valores,iteraciones= gdE(1,1,learning_rate,valor_buscado,max_iteraciones)

# b) ¿Cuántas iteraciones tarda el algoritmo en obtener por primera vez un valor de E(u, v)
# inferior a 10−14 . (Usar flotantes de 64 bits)
print ('Numero de iteraciones: ', num_iteracion)

# c) ¿En qué coordenadas (u, v) se alcanzó por primera vez un valor igual o menor a 10−14
# en el apartado anterior.
print ('Coordenadas obtenidas: (',u_sol, ', ',v_sol,')')

if extenso:
    entornoE(_x,_y,_z,u_sol,v_sol,'punto obtenido',4)
    input("\n--- Pulsar tecla para continuar ---\n") 
if extenso:
    plot.plot(iteraciones,valores)
    plot.xlabel("Iteracion")
    plot.ylabel("E(x,y)")
    plot.title('Descenso en E')
    plot.show()
    input("\n--- Pulsar tecla para continuar ---\n")

print ('\nEJERCICIO 3\n')

_x = np.linspace(-3, 3, res)
_y = np.linspace(-3, 3, res)
for ix, x in enumerate(_x):
    for iy, y in enumerate(_y):
        _z[iy, ix] = f(x,y)
 
if extenso:
    entornof(_x,_y,_z,1,-1,'punto inicial',1)
    input("\n--- Pulsar tecla para continuar ---\n")

learning_rate=0.01
max_iteraciones=50
valor_buscado=1e-20
#valor_buscado=-1

x_sol,y_sol,num_iteracion,valores,iteraciones= gdf(1,-1,learning_rate,valor_buscado,max_iteraciones)

# a) Usar gradiente descendente para minimizar la función f, con punto inicial (1,1)
# tasa de aprendizaje 0.01 y max 50 iteraciones. Repetir con tasa de aprend. 0.1
print ('\nLearning rate igual a 0.01')
print ('Valor obtenido: ',f(x_sol,v_sol))

if extenso:
    entornof(_x,_y,_z,x_sol,y_sol,'punto obtenido lr=0.01',1)
    input("\n--- Pulsar tecla para continuar ---\n") 

plot.plot(iteraciones,valores)
plot.xlabel("Iteracion")
plot.ylabel("f(x,y)")
plot.title('Descenso en f con learning rate 0,01')
plot.show()
input("\n--- Pulsar tecla para continuar ---\n") 

learning_rate=0.1
x_sol,y_sol,num_iteracion,valores,iteraciones= gdf(1,-1,learning_rate,valor_buscado,max_iteraciones)

print ('\nLearning rate igual a 0.1')
print ('Valor obtenido: ',f(x_sol,v_sol))

if extenso:
    entornof(_x,_y,_z,x_sol,y_sol,'punto obtenido lr=0.1',1)
    input("\n--- Pulsar tecla para continuar ---\n") 

plot.plot(iteraciones,valores)
plot.xlabel("Iteracion")
plot.ylabel("f(x,y)")
plot.title('Descenso en f con learning rate 0,1')
plot.show()
input("\n--- Pulsar tecla para continuar ---\n") 

puntos_x=np.array([2.1,3,1.5,1])
puntos_y=np.array([-2.1,-3,1.5,-1])

# b) Obtener el valor minimo y los valores de (x,y) con los
# puntos de inicio siguientes: (2,1, −2,1), (3, −3),(1,5, 1,5),(1, −1). Generar una
# tabla con los valores obtenidos
learning_rate=0.01
valor_buscado=-100
print ('\nLearning rate igual a 0.01\n')
print ('\nPunto inicial\tValor obtenido\t\tCoord X\t\t\tCoord Y\n' )
for i, x in enumerate(puntos_x):
    x_sol,y_sol,num_ite,valores,iteraciones= gdf(puntos_x[i],puntos_y[i],learning_rate,valor_buscado,max_iteraciones)
    print ('(',puntos_x[i],',',puntos_y[i],'):\t',f(x_sol,y_sol),'\t',x_sol,'\t',y_sol)
learning_rate=0.1
print ('\nLearning rate igual a 0.1\n')
print ('\nPunto inicial\tValor obtenido\t\tCoord X\t\t\tCoord Y\n' )
for i, x in enumerate(puntos_x):
    x_sol,y_sol,num_ite,valores,iteraciones= gdf(puntos_x[i],puntos_y[i],learning_rate,valor_buscado,max_iteraciones)
    print ('(',puntos_x[i],',',puntos_y[i],'):\t',f(x_sol,y_sol),'\t',x_sol,'\t',y_sol)






