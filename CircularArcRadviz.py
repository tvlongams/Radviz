# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 13:52:29 2017

@author: VTRAN
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import requests
from io import StringIO
import numpy as np
import pandas as pd
from pandas.tools.plotting import radviz
from pandas.tools.plotting import parallel_coordinates
from pandas.tools.plotting import scatter_matrix
from sklearn.decomposition import PCA, FastICA
import matplotlib.animation as animation
import matplotlib.animation as manimation
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid
#import skflow
import sklearn

sess=tf.Session()

#step data
from datasets import *
'''
uci_data=['opt.tes', 'yeast', 'wpdc', 'auto', 'wdbc', 'olive',
'ecoli', 'y14c', 'pen.tes', 'opt.tra', 'bcw', 'pen.tra', 'wine', 'iris']
'''
#(x_vals,target)=DATA['wine']
#(x_vals,target)=DATA['iris']
#(x_vals,target)=DATA['y14c']
#(x_vals,target)=DATA['ecoli']
#(x_vals,target)=DATA['olive']
#(x_vals,target)=DATA['auto']
#(x_vals,target)=DATA['bcw']
#(x_vals,target)=DATA['wdbc']
#(x_vals,target)=DATA['wpdc'] 
(x_vals,target)=DATA['yeast']
#(x_vals,target)=DATA['pen.tra']
#(x_vals,target)=DATA['pen.tes']
#(x_vals,target)=DATA['opt.tra']
#(x_vals,target)=DATA['opt.tes']

'''
from readdata import *
istar_data=['All_reduced.DATA',  'dermatology.data', 'elephant.DATA', 'ETHZ.data', 'fiber-notnorm.data', 
'freeFoto.DATA', 'mammal.data', 'Mice.data', 'movementLibras.data', 'optdigits.DATA', 
'PHYSHING.DATA', 'QSAR.data', 'satimage.data', 'segment.data', 'SegmentationNormcols.DATA', 
'shapes.data', 'SpamBase.data', 'texture.data', 'twonorm.data', 'VEHICLE.DATA', 
'wdbc_std.data','ionosphere.txt', 'primary-tumor.txt', 'SONAR.txt', 'spectfheart.txt']
(x_vals,target)=DT[2]
#'''
def normalize_cols(m):
    col_max=m.max(axis=0)
    col_min=m.min(axis=0)
    #return 2*(m-col_min)/(col_max-col_min)-1
    return (m-col_min)/(col_max-col_min)

x_vals=np.nan_to_num(normalize_cols(x_vals))

def convert_to_frame(x,y):
    values=pd.DataFrame(x)
    labels=pd.DataFrame({'Labels':y})
    dat = pd.concat([values, labels], axis=1)
    return dat

def convert_class_to_vector(i,n,m=None):
    c=np.zeros(n)
    if m:
        c[i-m]=1.0
    else:
        c[i]=1.0
    return c

K=len(np.unique(target))
y_vals=np.zeros((len(target),K)) # y.shape =(150,3)
for i in range(len(target)):
    y_vals[i]=convert_class_to_vector(int(target[i]),K,1)

    
#step 3
seed=3
tf.set_random_seed(seed)
np.random.seed(seed)
batch_size=128

# creator 4 layers
n0=x_vals.shape[1] # input
n1=2 # hidden 1
n2=K # hidden 2


#print(n0)
lrate=0.01
#step 4 creating data
train_indices=np.random.choice(len(x_vals),round(len(x_vals)*1),replace=False)
#test_indices=np.array(list(set(range(len(x_vals)))-set(train_indices)))

## training data
train_data=x_vals[train_indices]
train_target=y_vals[train_indices]
## testing data
#test_data=x_vals[test_indices]
#test_target=y_vals[test_indices]

## input data set
x_data=tf.placeholder(shape=[None,n0],dtype=tf.float32) ## (None,4) matrix
y_target=tf.placeholder(shape=[None,K],dtype=tf.float32) ## (None,3) matrix


classes=np.unique(target)
colors=matplotlib.pyplot.cm.rainbow(np.linspace(0,1,len(classes)))
cm = matplotlib.colors.ListedColormap(colors)


'''layer circular arc radviz
 W[i]=[A,B] a line segment A=W[i,0], B=W[i,1]'''

W=tf.Variable(tf.random_uniform(shape=[2,n0],minval=0.0,maxval=2*3.1459))


def general_sc(x,W):
	x_1=tf.convert_to_tensor(x,dtype=tf.float32)
	x_0=tf.subtract(tf.ones(tf.shape(x_1)),x_1)# x_0=1-x_1
	
	W0=tf.reshape(tf.tile(W[0,:],[batch_size]),(batch_size,n0))
	W1=tf.reshape(tf.tile(W[1,:],[batch_size]),(batch_size,n0))

	A=tf.multiply(x_0,W0)
	B=tf.multiply(x_1,W1)

	Alpha=tf.add(A,B)
	Acos=tf.cos(Alpha)
	Asin=tf.sin(Alpha)
	
	V1=tf.reduce_sum(tf.multiply(x_1,Acos),axis=1)
	V2=tf.reduce_sum(tf.multiply(x_1,Asin),axis=1)
	V=tf.stack([V1,V2],axis=1)
	s1=tf.reduce_sum(x_1,axis=1)
	P=V/tf.reshape(s1,(-1,1))
	return P


layer1=general_sc(x_data,W)


# layer for classifier
def fully_connected(input_layer,weight,bias):
    layer=tf.add(tf.matmul(input_layer,weight),bias)
    return tf.nn.sigmoid(layer)

W2=tf.Variable(tf.random_normal(shape=[n1,n2]))
B2=tf.Variable(tf.random_normal(shape=[n2]))
f_out=fully_connected(layer1,W2,B2)

Y = tf.nn.softmax(f_out)


## define loss function

#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
#cross_entropy = tf.reduce_mean(cross_entropy)*100

cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=f_out, labels=y_target)
loss=tf.reduce_mean(cross_entropy)
#loss=tf.reduce_mean(tf.square(y_target-f_out))

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(y_target, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

opt=tf.train.AdamOptimizer(lrate)
train_step=opt.minimize(loss)
#init=tf.initialize_all_variables()
init=tf.global_variables_initializer()
sess.run(init)

train_acc=[]
test_acc=[]
loss_vec=[]


def binary(n):
    if n==1:
        b=np.array([[0],[1]])
    elif n==2:
        b=np.array([[0,0],[0,1],[1,0],[1,1]])
    else:
        a1=binary(n-1)
        m=len(a1)
        a2=a1
        c1=np.concatenate((a1,np.zeros((m,1))),axis=1)
        c2=np.concatenate((a2,np.ones((m,1))),axis=1)
        b=np.concatenate((c1, c2), axis=0)
    return b

#training
for i in range(200):
	for j in range(10):
		rand_index=np.random.choice(len(train_data),size=batch_size)
		x_train=train_data[rand_index]
		y_train=train_target[rand_index]
				
		sess.run(train_step,feed_dict={x_data:x_train,y_target:y_train})
		temp_loss=sess.run(loss,feed_dict={x_data:x_train,y_target:y_train})
		loss_vec.append(temp_loss)

#optimal visualization
batch_size=len(x_vals)

def proj_gsc(x,W):
	x_1=tf.convert_to_tensor(x,dtype=tf.float32)
	x_0=tf.subtract(tf.ones(tf.shape(x_1)),x_1)# x_0=1-x_1
	
	W0=tf.reshape(tf.tile(W[0,:],[batch_size]),(batch_size,n0))
	W1=tf.reshape(tf.tile(W[1,:],[batch_size]),(batch_size,n0))

	A=tf.multiply(x_0,W0)
	B=tf.multiply(x_1,W1)

	Alpha=tf.add(A,B)
	Acos=tf.cos(Alpha)
	Asin=tf.sin(Alpha)
	
	V1=tf.reduce_sum(tf.multiply(x_1,Acos),axis=1)
	V2=tf.reduce_sum(tf.multiply(x_1,Asin),axis=1)
	V=tf.stack([V1,V2],axis=1)
	s1=tf.reduce_sum(x_1,axis=1)
	P=V/tf.reshape(s1,(-1,1))
	return P

V=sess.run(W)
X1=proj_gsc(x_vals,W)
X2=sess.run(X1)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
axes.set_xlim([-1.2,1.2])
axes.set_ylim([-1.2,1.2])

train_indices=range(len(x_vals))
axes.scatter(X2[:,0],X2[:,1],c=target[train_indices],cmap=cm,marker='.')
axes.set_title("general star coordinates")

t=np.linspace(0,2*np.pi,100)
xt=np.cos(t)
yt=np.sin(t)
axes.plot(xt,yt,color='black')

m=n0
axiscolors=matplotlib.pyplot.cm.gist_rainbow(np.linspace(0,1,m))

for i in range(m):
	t = np.linspace(V[0,i],V[1,i], 100)
	plt.plot((1.02+i*0.02)*np.cos(t),(1.02+i*0.02)*np.sin(t),color=axiscolors[i],alpha=1.0)
	#plt.text((1.0+i*0.02)*np.cos(t[15]),(1.0+i*0.02)*np.cos(t[15]),str(i),color="black")

plt.show()

'''	
FFMpegWriter = animation.writers['ffmpeg']
metadata = dict(title='Star Visualization', artist='SC',
                comment='SC')
writer = FFMpegWriter(fps=15, metadata=metadata)

#plt.ion()
#fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

with writer.saving(fig, "GeneralSCs.mp4", 100):
	for i in range(200):
	#def animate(i):
		for j in range(10):
			rand_index=np.random.choice(len(train_data),size=batch_size)
			x_train=train_data[rand_index]
			y_train=train_target[rand_index]
				
			sess.run(train_step,feed_dict={x_data:x_train,y_target:y_train})
			temp_loss=sess.run(loss,feed_dict={x_data:x_train,y_target:y_train})
			loss_vec.append(temp_loss)

		Y1=sess.run(layer1,feed_dict={x_data:train_data})
		V=sess.run(W)
		axes.clear()
		X1=Y1
		axes.scatter(X1[:,0],X1[:,1],c=target[train_indices],cmap=cm,marker='.')
		axes.set_title("general star coordinates")

		plt.pause(0.001)
		writer.grab_frame()	

plt.close()
'''
