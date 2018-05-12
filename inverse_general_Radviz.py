# -*- coding: utf-8 -*-
"""
Created on Sun May 21 20:29:00 2017

@author: VTRAN
"""
#import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import requests
from io import StringIO
import numpy as np
import pandas as pd
from pandas.tools.plotting import radviz
from pandas.tools.plotting import parallel_coordinates
from pandas.tools.plotting import scatter_matrix

import time
from sklearn.neighbors import NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import sklearn
from scipy.optimize import differential_evolution
from sklearn.metrics import silhouette_score as silhouette

from datasets import *
'''
uci_data=['opt.tes', 'yeast', 'wpdc', 'auto', 'wdbc', 'olive',
'ecoli', 'y14c', 'pen.tes', 'opt.tra', 'bcw', 'pen.tra', 'wine', 'iris']
'''
#(x,y)=DATA['iris']
#(x,y)=DATA['wine']
#(x,y)=DATA['y14c']
#(x,y)=DATA['ecoli']
#(x,y)=DATA['olive']
#(x,y)=DATA['auto']


(x,y)=DATA['bcw']
#(x,y)=DATA['wdbc']
#(x,y)=DATA['wpdc'] 
#(x,y)=DATA['yeast']
#(x,y)=DATA['pen.tra']
#(x,y)=DATA['pen.tes']
#(x,y)=DATA['opt.tra']
#(x,y)=DATA['opt.tes']

#from readdata import *
'''
istar_data=['All_reduced.DATA',  'dermatology.data', 'elephant.DATA', 'ETHZ.data', 'fiber-notnorm.data', 
'freeFoto.DATA', 'mammal.data', 'Mice.data', 'movementLibras.data', 'optdigits.DATA', 
'PHYSHING.DATA', 'QSAR.data', 'satimage.data', 'segment.data', 'SegmentationNormcols.DATA', 
'shapes.data', 'SpamBase.data', 'texture.data', 'twonorm.data', 'VEHICLE.DATA', 
'wdbc_std.data','ionosphere.txt', 'primary-tumor.txt', 'SONAR.txt', 'spectfheart.txt']
'''
#(x,y)=DT[2]



def normalize_cols(m):
    col_max=m.max(axis=0)
    col_min=m.min(axis=0)
    return (m-col_min)/(col_max-col_min)


def NearestCentroidClassifier(x,y):
	#'''
	#clf=NearestCentroid()
	clf=LDA()
	clf.fit(x,y)
	ac=clf.score(x,y)
	#'''
	#ac=silhouette(x, y, metric='sqeuclidean')
	return ac


	
def StarCoordinates(m):
	theta=np.pi*2/m
	v=np.zeros((m,2))
	for i in range(m):
		v[i,0]=np.cos(i*theta)
		v[i,1]=np.sin(i*theta)
	return v

X=np.nan_to_num(normalize_cols(x))
(n,m)=X.shape

# circular Radviz
def cradviz(x,alpha):
	#alpha
	m=len(x)
	theta=np.zeros(m)
	
	for i in range(m):
		theta[i]=(1-alpha[i])*x[i]+alpha[i]*(1-x[i])
		
	#anchor=np.array([np.cos(theta),np.sin(theta)])
	#p=np.dot(anchor,x)/np.sum(x)
	anchor=StarCoordinates(m)# (m,2) matrix
	
	p=np.dot(theta,anchor)/np.sum(theta)
	
	return p


def circleradviz(X,a):
	#p=sum 
	(n,m)=X.shape
	Y=np.zeros((n,2))
	for i in range(n):
		Y[i]=cradviz(X[i],a)
	return Y
	
#Differential evolution
bounds = [(0.0,1.0)]*(m)

def func(alpha):
	#m=np.int(len(alpha))
	v=alpha#.reshape((m,2))
	#v=x
	#Y=np.dot(X,v)
	Y=circleradviz(X,v)
	
	ac=1.0-NearestCentroidClassifier(Y,y)
	#ac=1.0-silhouette(Y,y)
	return ac
	
'''The differential evolution strategy to use. Should be one of:

‘best1bin’
‘best1exp’
‘rand1exp’
‘randtobest1exp’
‘best2exp’
‘rand2exp’
‘randtobest1bin’
‘best2bin’
‘rand2bin’
‘rand1bin’
'''

#result = differential_evolution(func, bounds,strategy='best1bin',maxiter=50,popsize=30)

#NP = 75, CR = 0.8803, F = 0.4717

#'''
result = differential_evolution(func, bounds,strategy='rand1bin',
	maxiter=50,popsize=15,mutation=0.4717,recombination=0.8803)

print("General Inverse Radviz: LDA Classifier: ")
print(1.0-result.fun)

x_optimal=result.x
#'''

'''
#original position 
theta=np.zeros(m)
for i in range(m):
	theta[i]=2*i*np.pi/m
	
x_optimal=theta
'''

V=x_optimal#.reshape((m,2))
#Y=np.dot(X,V_optimal)
Y=circleradviz(X,V)
'''
idx=np.argsort(V)


'''
print("optimal: ",V)

classes=np.unique(y)
colors=matplotlib.pyplot.cm.gist_rainbow(np.linspace(0,1,len(classes)))
#colors=matplotlib.pyplot.cm.rainbow(np.linspace(0,1,len(classes)))
cm = matplotlib.colors.ListedColormap(colors)
axiscolors=matplotlib.pyplot.cm.rainbow(np.linspace(0,1,m))

plt.figure(figsize=(6,6))
#draw anchor segment point

#pm=np.argsort(V)

#'''
anchor=StarCoordinates(m)

for i in range(m):
	al = V#np.linspace(V[i,0],V[i,1], 30)
	#plt.scatter(anchor[i,0],anchor[i,1],color=axiscolors[i],size=al[i]*10,alpha=0.95)
	plt.plot([0,anchor[i,0]],[0,anchor[i,1]],color=axiscolors[i],alpha=0.5)
	plt.scatter(al[i]*anchor[i,0],al[i]*anchor[i,1],color=axiscolors[i],alpha=0.95)
	plt.text((1+0.05)*anchor[i,0],(1+0.05)*anchor[i,1],str(i+1),color="black")
	
#'''
	
#Anchor=np.array([np.cos(t),np.sin(t)])
#plt.scatter(Anchor[0,:],Anchor[1,:],color="red",s=30*V_optimal)

#draw unit circle
t = np.linspace(0,2*np.pi, 100)
plt.plot(np.cos(t),np.sin(t),alpha=0.95)

plt.scatter(Y[:,0],Y[:,1],c=y,cmap=cm,marker='.',s=10)

plt.show()
	



