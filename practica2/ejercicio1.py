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

# Función simula_gaus genera una lista de N vectores de dimensión dim con números 
# aleatorios extraídos  de una distribución Gaussiana de media 0 y varianza dada 
# por la posición del vector sigma

def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
        
    return out

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

# Función nube_puntos dibuja una gráfica con la nube de puntos de salida generados
# con cierta distribución (apartado 1)

def nube_puntos(x,titulo):
    if titulo=='Distribucion gaussiana':
        color_r='seagreen'
    else:
        color_r='indianred'
    plot.scatter(x[:,0], x[:,1], color=color_r) #seagreen
    plot.title(titulo)
    plot.show() 
    input("\n--- Pulsar tecla para continuar ---\n")
    
# Función f devuelve el signo de la distancia de cada punto a la recta simulada 
# con simula_recta
    
def f(x,y,a,b):
    return np.sign(y-a*x-b)

# Función nube_puntos dibuja una gráfica con la nube de puntos de salida generados
# con cierta distribución etiquetados junto a la recta empleada

def nube_puntos_recta(etiquetas,a,b,x,titulo):
    x_positivo=x[etiquetas>=0]
    x_negativo=x[etiquetas<0]
    _x = np.linspace(-50, 50, etiquetas.size)
    _y = a*_x +b
    plot.scatter(x_positivo[:,0], x_positivo[:,1], color='indianred', label='positivo') 
    plot.scatter(x_negativo[:,0], x_negativo[:,1], color='darkorange', label='negativo')
    plot.plot(_x,_y,linewidth=3,color='maroon',label='recta')
    plot.legend(loc=4)
    plot.title(titulo)
    plot.show() 
    input("\n--- Pulsar tecla para continuar ---\n")

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


# Función f1

def f1(x):
    return (x[:, 0]-10)**2+(x[:, 1]-20)**2-400

# Función f2

def f2(x):
    return 0.5*(x[:, 0]+10)**2+(x[:, 1]-20)**2-400

# Función f3

def f3(x):
    return 0.5*(x[:, 0]-10)**2-(x[:, 1]+20)**2-400

# Función f4

def f4(x):
    return x[:, 1]-20*(x[:, 0]**2)-5*x[:, 0]+3

# Función grafica_no_lineal dibuja una gráfica con  la nube de puntos etiquetados
# por la función 'funcion' junto a la misma

def grafica_no_lineal(x, etiquetas, funcion, titulo):
    if funcion==f1:
        colorp='crimson'
        colorn='teal'
    elif funcion==f2:
        colorp='springgreen'
        colorn='goldenrod'
    elif funcion==f3:
        colorp='indigo'
        colorn='yellow'
    else:
        colorp='chartreuse'
        colorn='hotpink'
    color_l='black'
    min_x = x.min(axis=0)
    max_x = x.max(axis=0)
    x_positivo=x[etiquetas>=0]
    x_negativo=x[etiquetas<0]
    plot.scatter(x_positivo[:,0], x_positivo[:,1], color=colorp, label='positivo') 
    plot.scatter(x_negativo[:,0], x_negativo[:,1], color=colorn, label='negativo')
    _x, _y = np.meshgrid(np.linspace(round(min(min_x)), round(max(max_x)), x.shape[0]),
                         np.linspace(round(min(min_x)), round(max(max_x)), x.shape[0]))
    posiciones = np.vstack([_x.ravel(), _y.ravel()])
    plot.contour(_x, _y, funcion(posiciones.T).reshape(x.shape[0], x.shape[0]), [0], colors=color_l)
    plot.legend(loc=4)
    plot.title(titulo)
    plot.show()
    input("\n--- Pulsar tecla para continuar ---\n")

# Función calcular_error devuelve el número de valores de etiquetas y de nuevas_etiquetas
# que sean iguales 

def calcular_error(etiquetas, nuevas_etiquetas):
    aux = dualizar(etiquetas) + dualizar(nuevas_etiquetas)
    error = np.count_nonzero(aux)
    return error, aux

# Función dualizar devuelve un array a partir de etiquetas en el cual por cada valor
# positivo de etiquetas se almacena un 1 y por cada valor negativo se almacena un -1

def dualizar(etiquetas):
    positivas= np.where (etiquetas>0)
    negativas = np.where (etiquetas<0)
    etiquetas_dual = etiquetas
    etiquetas_dual[positivas]=1
    etiquetas_dual[negativas]=-1
    return etiquetas_dual

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

# Función nube_puntos_aciertos dibuja una gráfica que representa el porcentaje de
# aciertos de una función dada

def nube_puntos_aciertos(x,etiquetas,titulo, f):
    aciertos = x[etiquetas!=0]
    fallos = x[etiquetas==0]
    plot.scatter(aciertos[:,0], aciertos[:,1], color='seagreen', label='aciertos') 
    plot.scatter(fallos[:,0], fallos[:,1], color='firebrick', label='fallos')
    plot.legend(loc=4)
    plot.title(titulo)
    plot.show() 
    if f==False:
        input("\n--- Pulsar tecla para continuar ---\n")
    

#-------------------------------------------------------------------------------#
#------------- Ejercicio sobre la complejidad de H y el ruido ------------------#
#-------------------------------------------------------------------------------#


# Fijamos la semilla
np.random.seed(1)

# Mostrar gráficas adicionales; descomentar en caso de no mostrarse  
extenso=eleccion_spyder()
#extenso=eleccion_terminal()

#-------------------------------------------------------------------------------#
#----------------------- Apartado 1 --------------------------------------------#
#-------------------------------------------------------------------------------#

print ('\nEJERCICIO 1\n')

# Dibujar una gráfica con la nube de puntos de salida correspondiente.

# a) Considere N = 50, dim = 2, rango = [−50, +50] con simula_unif (N, dim, rango).

x_u = simula_unif(50, 2, [-50,50])
nube_puntos(x_u,'Distribucion uniforme')

# b) Considere N = 50, dim = 2 y sigma = [5, 7] con simula_gaus(N, dim, sigma).

x_g = simula_gaus(50, 2, np.array([5,7]))
nube_puntos(x_g, 'Distribucion gaussiana')

#-------------------------------------------------------------------------------#
#----------------------- Apartado 2 --------------------------------------------#
#-------------------------------------------------------------------------------#

print ('\nEJERCICIO 2\n')

# Con ayuda de la función simula_unif (100, 2, [−50, 50]) generar una muestra de
# puntos 2D a los que vamos añadir una etiqueta usando el signo de la función 
# f(x, y)=y−ax−b, es decir el signo de la distancia de cada punto a la recta simulada
# con simula_recta().

x_u = simula_unif(100, 2, [-50,50])
a, b = simula_recta([-50,50])
etiquetas=f(x_u[:,0],x_u[:,1],a,b)
    
# a) Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta,
# junto con la recta usada para ello. 

if extenso:
    nube_puntos(x_u,'Nube de puntos sin etiquetar (distribucion uniforme)')
nube_puntos_recta(etiquetas,a,b,x_u,'Nube de puntos etiquetados junto a recta')

# b) Modifique de forma aleatoria un 10% etiquetas positivas y otro 10% de negativas 
# y guarde los puntos con sus nuevas etiquetas. Dibuje de nuevo la gráfica anterior.

etiquetas_ruido = aniadir_ruido(etiquetas)
nube_puntos_recta(etiquetas_ruido,a,b,x_u,'Nube de puntos etiquetados con ruido junto a recta')

# c) Supongamos ahora que las funciones f1, f2, f3 y f4 definen la frontera de clasificación
# de los puntos de la muestra en lugar de una recta. Visualizar el etiquetado generado en 2b 
# junto con cada una de las gráficas de cada una de las funciones. 

etiquetas=f(x_u[:,0],x_u[:,1],a,b)

print ('\nPrimera funcion: \n')
if extenso:
    grafica_no_lineal(x_u, etiquetas, f1, 'Nube de puntos etiquetados sin ruido (f1)')
grafica_no_lineal(x_u, etiquetas_ruido, f1, 'Nube de puntos etiquetados con ruido (f1)')

print ('\nSegunda funcion: \n')
if extenso:
    grafica_no_lineal(x_u, etiquetas, f2, 'Nube de puntos etiquetados sin ruido (f2)')
grafica_no_lineal(x_u, etiquetas_ruido, f2, 'Nube de puntos etiquetados con ruido (f2)')

print ('\nTercera funcion: \n')
if extenso:
    grafica_no_lineal(x_u, etiquetas, f3, 'Nube de puntos etiquetados sin ruido (f3)')
grafica_no_lineal(x_u, etiquetas_ruido, f3, 'Nube de puntos etiquetados con ruido (f3)')

print ('\nCuarta funcion: \n')
if extenso:
    grafica_no_lineal(x_u, etiquetas, f4, 'Nube de puntos etiquetados sin ruido (f4)')
grafica_no_lineal(x_u, etiquetas_ruido, f4, 'Nube de puntos etiquetados con ruido(f4)')

print ('\n')

# Comparar las regiones positivas y negativas de estas nuevas funciones con las obtenidas
# en el caso de la recta. ¿Son estas funciones más complejas mejores clasificadores 
# que la función lineal? Observe las gráficas y diga que consecuencias extrae sobre 
# la influencia del proceso de modificación de etiquetas en el proceso de aprendizaje.

etiquetas1 = f1(x_u)
etiquetas2 = f2(x_u)
etiquetas3 = f3(x_u)
etiquetas4 = f4(x_u)

if extenso:    
    porcentaje1, aux1 = calcular_error(etiquetas, etiquetas1)
    porcentaje2, aux2 = calcular_error(etiquetas, etiquetas2)
    porcentaje3, aux3 = calcular_error(etiquetas, etiquetas3)
    porcentaje4, aux4 = calcular_error(etiquetas, etiquetas4)
    print ('---------------------- Sin ruido --------------------- \n')
    print ('Porcentaje de aciertos de la funcion 1:\t ',porcentaje1,'%')
    print ('Porcentaje de aciertos de la funcion 2:\t ',porcentaje2,'%')
    print ('Porcentaje de aciertos de la funcion 3:\t ',porcentaje3,'%')
    print ('Porcentaje de aciertos de la funcion 4:\t ',porcentaje4,'%')
    print ('\n')
    print ('---------------------- Con ruido --------------------- \n')
porcentaje1, aux1 = calcular_error(etiquetas_ruido, etiquetas1)
porcentaje2, aux2 = calcular_error(etiquetas_ruido, etiquetas2)
porcentaje3, aux3 = calcular_error(etiquetas_ruido, etiquetas3)
porcentaje4, aux4 = calcular_error(etiquetas_ruido, etiquetas4)
print ('Porcentaje de aciertos de la funcion 1:\t ',porcentaje1,'%')
print ('Porcentaje de aciertos de la funcion 2:\t ',porcentaje2,'%')
print ('Porcentaje de aciertos de la funcion 3:\t ',porcentaje3,'%')
print ('Porcentaje de aciertos de la funcion 4:\t ',porcentaje4,'%')
if extenso:    
    nube_puntos_aciertos(x_u,aux1,'Porcentaje de aciertos de f1',False)
    nube_puntos_aciertos(x_u,aux2,'Porcentaje de aciertos de f2',False)
    nube_puntos_aciertos(x_u,aux3,'Porcentaje de aciertos de f3',False)
    nube_puntos_aciertos(x_u,aux4,'Porcentaje de aciertos de f4',True)
    
print ('\nLa muestra está clasificada de manera lineal, por lo que ninguna función será ')
print ('mejor que la lineal; conociendo eso, la primera función (circunferencia) es la ')
print ('más representativa.\n')


