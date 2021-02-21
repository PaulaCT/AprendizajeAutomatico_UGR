#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#---------------------------#
#-------- LIBRERIAS --------#
#---------------------------#

import numpy as np
import matplotlib.pylab as plt
import sklearn.metrics as mt
import sklearn.linear_model as lm
import seaborn as sns
import pandas as pn
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from warnings import simplefilter



#---------------------------#
#-------- FUNCIONES --------#
#---------------------------#

# Lectura

def read(train, test):
	# Lectura de ficheros
    data_train = np.loadtxt(train,delimiter=",")
    data_test = np.loadtxt(test,delimiter=",")
    # Separar x e y (train)
    x_train = data_train[:,:-1]
    y_train = data_train[:,-1]
    # Separar x e y (test)
    x_test = data_test[:,:-1]
    y_test = data_test[:,-1]
    # Mostrar información
    print ('Tamanio train: ', len(y_train), '\nTamanio test: ' , len(y_test))
    return x_train, y_train, x_test, y_test

# Gráficas

def dibujarDistribucion(nombre,conjunto):
    if nombre == "Cantidad de instancias por digito (train)":
        color = "GnBu"
    else:
        color = "PRGn"
    n, b, px = plt.hist(conjunto, edgecolor='white', linewidth=1.2)
    plt.title(nombre)
    plt.xlabel("Digito")
    plt.ylabel("Num instancias")
    cm = plt.cm.get_cmap(color)
    bc = 0.5 * (b[:-1] + b[1:])
    col = bc - min(bc)
    col /= max(col)
    for c, p in zip(col, px):
        plt.setp(p, 'facecolor', cm(c))
    plt.show()
    input("\n--- Pulsar tecla para continuar ---\n")

def dibujarDigitos(x_train, y_train):   
    for i in range(10):    
        plot = plt.subplot(2,5,i+1)
        seleccionados=x_train[y_train==i]
        plot.imshow(np.split(seleccionados[3],8),cmap='inferno_r')#'coolwarm')
        plot.set_xticks(())
        plot.set_yticks(())    
        plot.set_title('Digito '+str(i))
    plt.show()
    input("\n--- Pulsar tecla para continuar ---\n")

def dibujarNube(x,y):
    plt.figure()
    pca = PCA(n_components=2)
    proj = pca.fit_transform(x)
    plt.scatter(proj[:,0],proj[:,1],c=y_train,cmap="Set3")
    plt.colorbar()
    plt.title("Nube de puntos clasificados por digito")
    plt.show()
    input("\n--- Pulsar tecla para continuar ---\n")

def dibujarMatrizConfusion(predicciones,y,contraste,nombre):
    matriz_confusion = mt.confusion_matrix(y, predicciones)
    sns.heatmap(matriz_confusion,cmap='RdPu', annot=True, fmt=".1f")
    plt.xlabel('Prediccion')
    plt.ylabel('Real')
    plt.title('Matriz de Confusion - '+nombre)
    plt.show()
    input("\n--- Pulsar tecla para continuar ---\n")
    if contraste == 'True':
        plt.imshow(matriz_confusion,cmap='PRGn')
        plt.colorbar()
        plt.xlabel('Prediccion')
        plt.ylabel('Real')
        plt.title('Matriz de Confusion - Contraste - '+nombre)
        plt.show()
        input("\n--- Pulsar tecla para continuar ---\n")

def dibujarComparaciones(resultados):
    filas = ['Score','Accuracy','ClasificationError','CohensKappa','F1Score']
    columnas = ['LR-Lasso','LR-Ridge','PLA-Lasso','PLA-Ridge','SGD-Lasso','SGD-Ridge']
    dataframe = pn.DataFrame(resultados, index=filas, columns=columnas)
    sns.heatmap(dataframe, annot=True, fmt=".4f",linewidth=0.5, cmap='nipy_spectral')
    plt.xticks(rotation=90)
    plt.show()
    
# Preprocesar datos

def preprocesar(x):
    sin_var = VarianceThreshold(0.075)
    x = sin_var.fit_transform(x)
    norm = MinMaxScaler()
    x = norm.fit_transform(x)
    return x

# Ajustar

def ajuste(x_train_n,y_train,x_test_n,y_test,funcion,reg):
    # Hiperparámetros
    if funcion==lm.LogisticRegression:
        if reg=='l1':
            hparam = [{'penalty':[reg],'C':[1,10,100,1000,10000], 'tol':[1e-3,1e-4], 'solver':['liblinear'], 'multi_class':['auto']}]
        else:
            hparam = [{'penalty':[reg],'C':[1,10,100,1000,10000], 'tol':[1e-3,1e-4], 'solver':['newton-cg'], 'multi_class':['auto']}]
    if funcion==lm.Perceptron:
        hparam = [{'penalty':[reg],'alpha':[0.1,0.001,0.0001,0.00001],'tol':[1e-3,1e-4]}]
    if funcion==lm.SGDClassifier:
        hparam = [{'loss': ['log'], 'penalty':[reg], 'n_iter':[1,10,100,1000]}]
    modelo = funcion(random_state=1,max_iter=1000)
    modelo = GridSearchCV(modelo, hparam, cv=5, scoring='accuracy')    
    modelo.fit(x_train_n,y_train)
    infoModelo(modelo)
    modelo = funcion(**modelo.best_params_)
    modelo.fit(x_train_n,y_train)
    predictions = modelo.predict(x_test_n)
    score = modelo.score(x_test_n,y_test)
    return predictions, score
    

def errorClasificacion(accuracy):
    return (1.0-accuracy)

def infoModelo(modelo):
    print("Mejores parametros: ", modelo.best_params_)
    print("Validacion cruzada: ", modelo.best_score_)
    print("Ein :", 1-modelo.best_score_)

def bondadClasificacion(predicciones,valor,y_test,nombre,i,s,a,ce,ck,f1):
    s[i]=valor
    a[i]=mt.accuracy_score(y_test, predicciones)
    ce[i]=errorClasificacion(a[i])
    ck[i]=mt.cohen_kappa_score(y_test, predicciones)
    f1[i]=mt.f1_score(y_test, predicciones,average='weighted')
    print('Hit score: ', valor)
    print('Exactitud: ', a[i])
    print('Error en clasificacion: ', ce[i])
    print('Cohen\'s Kappa: ', ck[i])
    print('F1 (weighted): ', f1[i])
    #print('Otras metricas: ')
    #print(mt.classification_report(y_test, predicciones))
    print('Matriz de confusion: ')
    if extenso:
        dibujarMatrizConfusion(predicciones,y_test,'True',nombre)
    else:
        dibujarMatrizConfusion(predicciones,y_test,'False',nombre)

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
#------------------------- Problema de clasificación ---------------------------#
#-------------------------------------------------------------------------------#

# Fijamos la semilla     
np.random.seed(10)

# Mostrar gráficas adicionales; descomentar en caso de no mostrarse  
extenso=eleccion_spyder()
#extenso=eleccion_terminal()

# Ignorar 'Future Warnings'
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

#-------------------------------------------------------------------------------#
#----------------------- Apartado 1 --------------------------------------------#
#-------------------------------------------------------------------------------#

# Lectura de datos

x_train, y_train, x_test, y_test = read("datos/optdigits.tra","datos/optdigits.tes")
# Gráficas adicionales

if extenso:
    # Distribución de valores de las etiquetas
    dibujarDistribucion("Cantidad de instancias por digito (train)",y_train)
    dibujarDistribucion("Cantidad de instancias por digito (test)",y_test)   
    # Ejemplo de dígitos
    dibujarDigitos(x_train,y_train)
    
#-------------------------------------------------------------------------------#
#----------------------- Apartado 3 --------------------------------------------#
#-------------------------------------------------------------------------------#

# Fijar conjuntos

x_train_n = preprocesar(x_train)
x_test_n = preprocesar(x_test)
if extenso:
    dibujarNube(x_train_n, y_train)
    
print "Preprocesado Columnas Filas Ejemplo_valores"
print "Antes       ",len(x_train),"   ",len(x_train[0]),"   ",x_train[1,3]
print "Despues     ",len(x_train_n),"   ",len(x_train_n[0]),"   ",x_train_n[1,3]


#-------------------------------------------------------------------------------#
#----------------------- Apartado 10 -------------------------------------------#
#-------------------------------------------------------------------------------#

lasso = 'l1'
ridge = 'l2'

score = np.zeros(6)
accuracy = np.zeros(6)
clasification_error = np.zeros(6)
cohens_kappa = np.zeros(6)
f1_score = np.zeros(6)

print('\n\n---- Regresion Logistica - Regularizacion Lasso ----\n')
predicciones, valor = ajuste(x_train_n,y_train,x_test_n,y_test,lm.LogisticRegression, lasso)
bondadClasificacion(predicciones, valor,y_test,'RL-Lasso',0,score,accuracy,clasification_error,
                    cohens_kappa,f1_score)

print('---- Regresion Logistica - Regularizacion Ridge ----\n')
predicciones, valor = ajuste(x_train_n,y_train,x_test_n,y_test,lm.LogisticRegression, ridge)
bondadClasificacion(predicciones, valor,y_test,'RL-Ridge',1,score,accuracy,clasification_error,
                    cohens_kappa,f1_score)

print('---- Perceptron - Regularizacion Lasso ----\n')
predicciones, valor = ajuste(x_train_n,y_train,x_test_n,y_test,lm.Perceptron, lasso)
bondadClasificacion(predicciones, valor,y_test,'PLA-Lasso',2,score,accuracy,clasification_error,
                    cohens_kappa,f1_score)

print('---- Perceptron - Regularizacion Ridge ----\n')
predicciones, valor = ajuste(x_train_n,y_train,x_test_n,y_test,lm.Perceptron, ridge)
bondadClasificacion(predicciones, valor,y_test,'PLA-Ridge',3,score,accuracy,clasification_error,
                    cohens_kappa,f1_score)

print('---- SGD - Regularizacion Lasso ----\n')
predicciones, valor = ajuste(x_train_n,y_train,x_test_n,y_test,lm.SGDClassifier, lasso)
bondadClasificacion(predicciones, valor,y_test,'SGD-Lasso',4,score,accuracy,clasification_error,
                    cohens_kappa,f1_score)

print('---- SGD - Regularizacion Ridge ----\n')
predicciones, valor = ajuste(x_train_n,y_train,x_test_n,y_test,lm.SGDClassifier, ridge)
bondadClasificacion(predicciones, valor,y_test,'SGD-Ridge',5,score,accuracy,clasification_error,
                    cohens_kappa,f1_score)

if extenso:
    resultados = ([score,accuracy,clasification_error,cohens_kappa,f1_score])
    dibujarComparaciones(resultados)