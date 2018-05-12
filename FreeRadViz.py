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
#import skflow
import sklearn
from scipy.optimize import differential_evolution
from sklearn.metrics import silhouette_score as silhouette
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from datasets import *
'''
uci_data=['opt.tes', 'yeast', 'wpdc', 'auto', 'wdbc', 'olive',
'ecoli', 'y14c', 'pen.tes', 'opt.tra', 'bcw', 'pen.tra', 'wine', 'iris']
'''
#(x,y)=DATA['iris']
(x,y)=DATA['wine']
#(x,y)=DATA['y14c']
#(x,y)=DATA['ecoli']
#(x,y)=DATA['olive']
#(x,y)=DATA['auto']
#(x,y)=DATA['bcw']
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
    return ((m-col_min)/(col_max-col_min))
	#return (-1+2*(m-col_min)/(col_max-col_min))


def NearestCentroidClassifier(x,y):
	#clf=NearestCentroid()
	clf=LDA()
	clf.fit(x,y)
	ac=clf.score(x,y)
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
def flsradviz(x,alpha):
	#alpha
	m=len(x)
	dpoint=np.zeros((m,2))
	
	for i in range(m):
		dpoint[i,0]=alpha[i,0]*x[i]+alpha[i,2]*(1-x[i])
		dpoint[i,1]=alpha[i,1]*x[i]+alpha[i,3]*(1-x[i])
		
	#anchor=np.array([np.cos(theta),np.sin(theta)])
	#p=np.dot(anchor,x)/np.sum(x)
	p=np.dot(x,dpoint)/np.sum(x)
	return p


def freelinesegmentradviz(X,a):
	#p=sum 
	(n,m)=X.shape
	Y=np.zeros((n,2))
	for i in range(n):
		Y[i]=flsradviz(X[i],a)
	return Y
	
#Differential evolution
bounds = [(-1.0,1.0)]*(4*m)
def func(alpha):
	m=np.int(len(alpha)/4)
	v=alpha.reshape((m,4))
	#v=x
	#Y=np.dot(X,v)
	Y=freelinesegmentradviz(X,v)
	
	ac=1.0-NearestCentroidClassifier(Y,y)
	#ac=1.0-silhouette(Y,y)
	return ac
	

result = differential_evolution(func, bounds,strategy='rand1bin',
		maxiter=50,popsize=15,mutation=0.4717,recombination=0.8803)
print("General Radviz: Nearest Centroid Classifier: ")
print(1.0-result.fun)

x_optimal=result.x
V=x_optimal.reshape((m,4))
#Y=np.dot(X,V_optimal)
Y=freelinesegmentradviz(X,V)

#print("optimal: ",V)

classes=np.unique(y)
colors=matplotlib.pyplot.cm.rainbow(np.linspace(0,1,len(classes)))
cm = matplotlib.colors.ListedColormap(colors)
axiscolors=matplotlib.pyplot.cm.rainbow(np.linspace(0,1,m))

plt.figure(figsize=(6,6))

#'''
#draw anchor segment point
for i in range(m):
	#t = np.linspace(V[i,0],V[i,1], 30)
	#quiver
	#plt.plot((1.0+i*0.02)*np.cos(t),(1.0+i*0.02)*np.sin(t),color="red",alpha=0.5)
	#plt.scatter(V[i,0],V[i,2],color="blue",s=15)
	#plt.scatter(V[i,1],V[i,3],color="blue",alpha=0.0)
	#plt.plot([V[i,0],V[i,2]],[V[i,1],V[i,3]],color="blue")
	#plt.quiver(V[i,0],V[i,2],V[i,1]-V[i,0],V[i,3]-V[i,2],color="blue",angles='xy', scale_units = 'xy', scale = 0.5,)
	plt.arrow(V[i,0],V[i,2],V[i,1]-V[i,0],V[i,3]-V[i,2],color=axiscolors[i],head_width=0.02, head_length=0.02)
	plt.text((V[i,0]*0+V[i,1]*2)/2,(V[i,2]*0+V[i,3]*2)/2,str(i),color="black")
#'''
	
#Anchor=np.array([np.cos(t),np.sin(t)])
#plt.scatter(Anchor[0,:],Anchor[1,:],color="red",s=30*V_optimal)

#draw unit circle
#t = np.linspace(0,2*np.pi, 100)
#plt.plot(np.cos(t),np.sin(t),alpha=0.1)

#plt.arrow(-1.0,0.0,2.0,0,color="black",head_width=0.02, head_length=0.02)
#plt.arrow(0.0,-1.0,0.0,2.0,color="black",head_width=0.02, head_length=0.02)

plt.scatter(Y[:,0],Y[:,1],c=y,cmap=cm,marker='.',s=10)
#plt.xlim(-1.1, 1.1)
#plt.ylim(-1.1, 1.1)

plt.show()
	



