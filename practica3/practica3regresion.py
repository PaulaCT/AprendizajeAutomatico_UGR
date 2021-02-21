#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#---------------------------#
#-------- LIBRERIAS --------#
#---------------------------#

import numpy as np
import pandas as pn
import matplotlib.pylab as plt
import sklearn.metrics as mt
import sklearn.linear_model as lm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from warnings import simplefilter



#---------------------------#
#-------- FUNCIONES --------#
#---------------------------#

# Lectura

def read(crimen):
	# Lectura de ficheros
    cabeceras = ['state','county','community','communityname','fold','population',
                 'householdsize','racepctblack','racePctWhite','racePctAsian','racePctHisp',
                 'agePct12t21','agePct12t29','agePct16t24','agePct65up','numbUrban',
                 'pctUrban','medIncome','pctWWage','pctWFarmSelf','pctWInvInc','pctWSocSec',
                 'pctWPubAsst', 'pctWRetire','medFamInc','perCapInc','whitePerCap',
                 'blackPerCap','indianPerCap','AsianPerCap','OtherPerCap','HispPerCap',
                 'NumUnderPov','PctPopUnderPov','PctLess9thGrade','PctNotHSGrad','PctBSorMore',
                 'PctUnemployed','PctEmploy','PctEmplManu','PctEmplProfServ','PctOccupManu',
                 'PctOccupMgmtProf','MalePctDivorce','MalePctNevMarr','FemalePctDiv',
                 'TotalPctDiv','PersPerFam','PctFam2Par','PctKids2Par','PctYoungKids2Par',
                 'PctTeen2Par','PctWorkMomYoungKids','PctWorkMom','NumIlleg','PctIlleg',
                 'NumImmig','PctImmigRecent','PctImmigRec5','PctImmigRec8','PctImmigRec10',
                 'PctRecentImmig','PctRecImmig5','PctRecImmig8','PctRecImmig10',
                 'PctSpeakEnglOnly','PctNotSpeakEnglWell','PctLargHouseFam','PctLargHouseOccup',
                 'PersPerOccupHous','PersPerOwnOccHous','PersPerRentOccHous','PctPersOwnOccup',
                 'PctPersDenseHous','PctHousLess3BR','MedNumBR','HousVacant','PctHousOccup',
                 'PctHousOwnOcc','PctVacantBoarded','PctVacMore6Mos','MedYrHousBuilt',
                 'PctHousNoPhone','PctWOFullPlumb','OwnOccLowQuart','OwnOccMedVal',
                 'OwnOccHiQuart','RentLowQ','RentMedian','RentHighQ','MedRent','MedRentPctHousInc',
                 'MedOwnCostPctInc','MedOwnCostPctIncNoMtg','NumInShelters','NumStreet',
                 'PctForeignBorn','PctBornSameState','PctSameHouse85','PctSameCity85',
                 'PctSameState85','LemasSwornFT','LemasSwFTPerPop','LemasSwFTFieldOps',
                 'LemasSwFTFieldPerPop','LemasTotalReq','LemasTotReqPerPop','PolicReqPerOffic',
                 'PolicPerPop','RacialMatchCommPol','PctPolicWhite','PctPolicBlack','PctPolicHisp',
                 'PctPolicAsian','PctPolicMinor','OfficAssgnDrugUnits','NumKindsDrugsSeiz',
                 'PolicAveOTWorked','LandArea','PopDens','PctUsePubTrans','PolicCars',
                 'PolicOperBudg','LemasPctPolicOnPatr','LemasGangUnitDeploy','LemasPctOfficDrugUn',
                 'PolicBudgPerPop','ViolentCrimesPerPop' ]    
    data = pn.read_csv(crimen, names = cabeceras)
    
    # Eliminación de datos incompletos
    incompletas = ['LemasSwornFT','LemasSwFTPerPop','LemasSwFTFieldOps',
             'LemasSwFTFieldPerPop','LemasTotalReq','LemasTotReqPerPop','PolicReqPerOffic',
             'PolicPerPop','RacialMatchCommPol','PctPolicWhite','PctPolicBlack','PctPolicHisp',
             'PctPolicAsian','PctPolicMinor','OfficAssgnDrugUnits','NumKindsDrugsSeiz',
             'PolicAveOTWorked','PolicCars','PolicOperBudg','LemasPctPolicOnPatr',
             'LemasGangUnitDeploy','PolicBudgPerPop']
    discretas = ['state','county','community','communityname','fold']
    data.loc[130,'OtherPerCap'] = 0.28
    data = data.drop(incompletas,1)
    data = data.drop(discretas,1)
        
    data = data.to_numpy(dtype='float32')
    # Separar x e y 
    x = data[:,:-1]
    y = data[:,-1]
    # Mostrar información
    print ('Tamanio muestra: ', len(y))
    return x, y    

# Gráficas

def dibujarDistribucion(nombre,conjunto):
    if nombre == "Frecuencia en porcentajes - crimen violento/poblacion (train)":
        color = "GnBu"
    else:
        color = "PRGn"
    n, b, px = plt.hist(conjunto, edgecolor='white', linewidth=1.2)
    plt.title(nombre)
    plt.xlabel("Porcentaje")
    plt.ylabel("Frecuencia")
    cm = plt.cm.get_cmap(color)
    bc = 0.5 * (b[:-1] + b[1:])
    col = bc - min(bc)
    col /= max(col)
    for c, p in zip(col, px):
        plt.setp(p, 'facecolor', cm(c))
    plt.show()
    input("\n--- Pulsar tecla para continuar ---\n")

def dibujarComparaciones(resultados):
    filas = ['Score', 'R2Score', 'MeanSquaredError','MedianAbsoluteError','ExplainedVarianceScore']
    columnas = ['LinearRegresion','SGD-Lasso','SGD-Ridge','Ridge']
    dataframe = pn.DataFrame(resultados, index=filas, columns=columnas)
    sns.heatmap(dataframe, annot=True, fmt=".4f",linewidth=0.5, cmap='nipy_spectral')
    plt.xticks(rotation=90)
    plt.show()
   
def dibujarNubeAciertos(model,y_test,y_pred,x_test, nombre):
    etiquetas = diferencias(y_test,y_pred)
    plt.scatter(y_test, x_test[:,0], c=etiquetas, cmap='RdGy_r')
    plt.colorbar()
    plt.title('Diferencias entre datos reales y predichos - '+nombre)
    plt.ylabel('ViolentCrimesPerPop (real)')
    plt.xlabel('Population')
    plt.show() 
 
def dibujarMatrizConfusion(predicciones,y,contraste):
    matriz_confusion = mt.confusion_matrix(y, predicciones)
    sns.heatmap(matriz_confusion,cmap='RdPu', annot=True, fmt=".1f")
    plt.xlabel('Prediccion')
    plt.ylabel('Real')
    plt.title('Matriz de Confusion')
    plt.show()
    input("\n--- Pulsar tecla para continuar ---\n")
    if contraste == 'True':
        plt.imshow(matriz_confusion,cmap='PRGn')
        plt.colorbar()
        plt.xlabel('Prediccion')
        plt.ylabel('Real')
        plt.title('Matriz de Confusion')
        plt.show()
        input("\n--- Pulsar tecla para continuar ---\n")

# Preprocesar datos

def preprocesar(x):
    normal = Normalizer().fit(x)
    x = normal.transform(x) 
    scal = MinMaxScaler()
    x = scal.fit_transform(x)
    return x

# Ajustar

def ajuste(x_train,y_train,x_test,y_test,funcion):
    if funcion==lm.LinearRegression:
        hparam = [{'fit_intercept':[True],'normalize':[True,False]}]  
        modelo = funcion()
        modelo = GridSearchCV(modelo, hparam, cv=5, iid=False)
    if funcion==lm.Ridge:
        hparam = [{'alpha':[0.1,0.05,0.01,0.005,0.001],'solver':['cholesky','sag'],'tol':[1e-3,1e-4]}]    
        modelo = funcion(max_iter=100000)
        modelo = GridSearchCV(modelo,hparam,cv=5,scoring='r2') 
    modelo.fit(x_train,y_train)
    infoModelo(modelo)
    modelo = funcion(**modelo.best_params_)
    modelo.fit(x_train,y_train)
    predictions = modelo.predict(x_test)
    score = modelo.score(x_test,y_test)
    return modelo, predictions, score

def ajusteSGD(x_train,y_train,x_test,y_test,reg):
    hparam = [{'loss': ['epsilon_insensitive'], 'penalty':[reg], 'n_iter':[1,10,100,1000]}]
    modelo = lm.SGDRegressor(random_state=1,max_iter=1000)
    modelo = GridSearchCV(modelo, hparam, cv=5, iid=False)    
    modelo.fit(x_train,y_train)
    infoModelo(modelo)
    modelo = lm.SGDRegressor(**modelo.best_params_)
    modelo.fit(x_train,y_train)
    predictions = modelo.predict(x_test)
    score = modelo.score(x_test,y_test)
    return modelo, predictions, score

def infoModelo(modelo):
    print"Mejores parametros: ", modelo.best_params_
    print"Validacion cruzada: ", modelo.best_score_
    print"Ein :", 1-modelo.best_score_

def bondadRegresion(predicciones,valor,y_test,i,s,rs,mse,mae,evs):
    print'Hit score: ', valor
    s[i]=valor
    rs[i]=mt.r2_score(y_test, predicciones)
    mse[i]=mt.mean_squared_error(y_test, predicciones)
    mae[i]=mt.median_absolute_error(y_test, predicciones)
    evs[i]=mt.explained_variance_score(y_test, predicciones)
    print'Valor r2: ',rs[i]
    print'Valor de la varianza explicada', evs[i]
    print'Error de los minimos cuadrados: ', mse[i]
    print'Error absoluto medio: ', mae[i]
    

def diferencias(real,prediccion):
    aux = real - prediccion
    aux = abs(aux)
    return aux

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
#--------------------------- Problema de regresión -----------------------------#
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

x, y = read("datos/communities.data")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print 'Tamanio train: ', len(y_train), '\nTamanio test: ' , len(y_test)

if extenso:
    # Distribución de valores de y
    dibujarDistribucion("Frecuencia en porcentajes - crimen violento/poblacion (train)",y_train)
    dibujarDistribucion("Frecuencia en porcentajes - crimen violento/poblacion (test)",y_test) 
    
#-------------------------------------------------------------------------------#
#----------------------- Apartado 3 --------------------------------------------#
#-------------------------------------------------------------------------------#

# Fijar conjuntos

x_train_n = preprocesar(x_train)
x_test_n = preprocesar(x_test)

print "Preprocesado Columnas Filas Ejemplo_valores"
print "Antes       ",len(x_train),"   ",len(x_train[0]),"   ",x_train[1,3]
print "Despues     ",len(x_train_n),"   ",len(x_train_n[0]),"   ",x_train_n[1,3]

#-------------------------------------------------------------------------------#
#----------------------- Apartado 10 -------------------------------------------#
#-------------------------------------------------------------------------------#

lasso = 'l1'
ridge = 'l2'

score = np.zeros(4)
r2_score = np.zeros(4)
mean_squared_error = np.zeros(4)
median_absolute_error = np.zeros(4)
explained_variance_score = np.zeros(4)

print('\n\n---- Regresion Lineal ----\n')
model, predicciones, valor = ajuste(x_train_n,y_train,x_test_n,y_test,lm.LinearRegression)
bondadRegresion(predicciones, valor,y_test,0,score,r2_score,mean_squared_error,
                median_absolute_error,explained_variance_score)
dibujarNubeAciertos(model,y_test,predicciones,x_test,'RL')
input("\n--- Pulsar tecla para continuar ---\n")

print('---- SGD - Regularizacion Lasso ----\n')
model, predicciones, valor = ajusteSGD(x_train_n,y_train,x_test_n,y_test,lasso)
bondadRegresion(predicciones, valor,y_test,1,score,r2_score,mean_squared_error,
                median_absolute_error,explained_variance_score)
dibujarNubeAciertos(model,y_test,predicciones,x_test,'SGD-Lasso')
input("\n--- Pulsar tecla para continuar ---\n")

print('---- SGD Regularizacion Ridge ----\n')
model, predicciones, valor = ajusteSGD(x_train_n,y_train,x_test_n,y_test,ridge)
bondadRegresion(predicciones, valor,y_test,2,score,r2_score,mean_squared_error,
                median_absolute_error,explained_variance_score)
dibujarNubeAciertos(model,y_test,predicciones,x_test,'SGD-Ridge')
input("\n--- Pulsar tecla para continuar ---\n")

print('\n---- Ridge ----\n')
model, predicciones, valor = ajuste(x_train_n,y_train,x_test_n,y_test,lm.Ridge)
bondadRegresion(predicciones, valor,y_test,3,score,r2_score,mean_squared_error,
                median_absolute_error,explained_variance_score)
dibujarNubeAciertos(model,y_test,predicciones,x_test,'Ridge')
input("\n--- Pulsar tecla para continuar ---\n")

if extenso:
    resultados = ([score,r2_score,mean_squared_error,median_absolute_error,explained_variance_score])
    dibujarComparaciones(resultados)
