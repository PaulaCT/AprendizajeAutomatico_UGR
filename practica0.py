# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plot

"""
Ejercicio 1
"""

matriz = load_iris()

a=matriz.data
b=matriz.target_names

x1=a[0:49,-2:]
x2=a[50:99,-2:]
x3=a[100:149,-2:]

plot.scatter(x1[:,0],x1[:,1],c=[[1,0,0]],label=b[0])
plot.scatter(x2[:,0],x2[:,1],c=[[0,1,0]],label=b[1])
plot.scatter(x3[:,0],x3[:,1],c=[[0,0,1]],label=b[2])
plot.title('Iris database')
plot.legend(loc=2)
plot.show()

input('Pulse enter para pasar al ejercicio 2')
"""
Ejercicio 2
"""

x12=np.random.permutation(x1)
x22=np.random.permutation(x2)
x32=np.random.permutation(x3)
training=np.array(np.concatenate((x12[:40,:],x22[:40,:],x32[:40,:]),axis=0,out=None))
test=np.array(np.concatenate((x12[-10:,:],x22[-10:,:],x32[-10:,:]),axis=0,out=None))

plot.scatter(training[:,0],training[:,1],c=[[1,1,0]],label="training")
plot.scatter(test[:,0],test[:,1],c=[[0,1,1]],label="test")
plot.legend(loc=2)
plot.show()
print("Tamaño training: ",training.shape[0], ". Tamaño test: ",test.shape[0])

input('Pulse enter para pasar al ejercicio 3')
"""
Ejercicio 3
"""

n=np.linspace(0,2*np.pi,100)
seno=np.sin(n)
coseno=np.cos(n)
suma=np.sin(n)+np.cos(n)

plot.plot(n,seno,"k--",label="Sin(x)")
plot.plot(n,coseno,"b--",label="Cos(x)")
plot.plot(n,suma,"r--",label="Sin(x)+Cos(x)")
plot.legend(loc=2)
plot.show()
#n,a b-- ,r r--