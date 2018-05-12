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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

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
#(x_vals,target)=DATA['iris']
#(x_vals,target)=DATA['wine']
#(x_vals,target)=DATA['y14c']
#(x_vals,target)=DATA['ecoli']
#(x_vals,target)=DATA['olive']
#(x_vals,target)=DATA['auto']
#(x_vals,target)=DATA['bcw']
#(x_vals,target)=DATA['wdbc']
#(x_vals,target)=DATA['wpdc'] 
#(x_vals,target)=DATA['yeast']
#(x_vals,target)=DATA['pen.tra']
#(x_vals,target)=DATA['pen.tes']
(x_vals,target)=DATA['opt.tra']
#(x_vals,target)=DATA['opt.tes']




def PCA_plot(x,xt=None,k=2):
	pca = PCA(n_components=k)
	pca.fit(x)
	X = pca.transform(x)
	if xt==None:
		return X
	else:
		Xt=pca.transform(xt)
		return (X,Xt)

def LDA_plot(x,y,xt=None,k=2):
	lda=LinearDiscriminantAnalysis(n_components=k)
	lda.fit(x,y)
	X=lda.transform(x)
	if xt==None:
		return X
	else:
		Xt=lda.transform(xt)
		return (X,Xt)


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

    
	
def StarCoordinates(m):
	theta=np.pi*2/m
	v=np.zeros((m,2))
	for i in range(m):
		v[i,0]=np.cos(i*theta)
		v[i,1]=np.sin(i*theta)
	return v
	
#step 3
seed=3
tf.set_random_seed(seed)
np.random.seed(seed)
batch_size=128

# creator 4 layers
n0=x_vals.shape[1] # input
n1=2 # hidden 1
n2=K # hidden 2
#n3=20 # hidden 3
#n4=K # output

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


##define multilayer neural neywork
def fully_connected(input_layer,weight,bias):
    layer=tf.add(tf.matmul(input_layer,weight),bias)
    return tf.nn.sigmoid(layer)


#W1=tf.Variable(tf.random_normal(shape=[n0,n1]))

W1=tf.Variable(tf.random_uniform(shape=[n0,n1],minval=-1.0,maxval=1.0))

#W1=tf.convert_to_tensor(StarCoordinates(n0))


B1=tf.Variable(tf.random_normal(shape=[n1]))
#layer1=tf.matmul(x_data,W1)
layer1=tf.add(tf.matmul(x_data,W1),B1)

W2=tf.Variable(tf.random_normal(shape=[n1,n2]))
B2=tf.Variable(tf.random_normal(shape=[n2]))
f_out=fully_connected(layer1,W2,B2)

Y = tf.nn.softmax(f_out)
'''
B1=tf.Variable(tf.random_normal(shape=[n1]))
layer1=fully_connected(x_data,W1,B1)

W2=tf.Variable(tf.random_normal(shape=[n1,n2]))
B2=tf.Variable(tf.random_normal(shape=[n2]))
layer2=fully_connected(layer1,W2,B2)

W3=tf.Variable(tf.random_normal(shape=[n2,n3]))
B3=tf.Variable(tf.random_normal(shape=[n3]))
layer3=fully_connected(layer2,W3,B3)

W4=tf.Variable(tf.random_normal(shape=[n3,n4]))
B4=tf.Variable(tf.random_normal(shape=[n4]))
f_out=fully_connected(layer3,W4,B4)

Y = tf.nn.softmax(f_out)
'''
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

def NearestCentroidClassifier(x,y):
	#clf=NearestCentroid()
	clf=LDA()
	clf.fit(x,y)
	ac=clf.score(x,y)
	return ac


	

def vertexradvizprojection(x,p):
	m=len(p)
	v=StarCoordinates(m)
	ix=x
	n=len(x)
	for i in range(n):
		for j in range(m):
			if p[j]==1:
				ix[i,j]=1-x[i,j]
		ix[i]=ix[i]/np.sum(ix[i]) 
	px=np.dot(ix,v)
	return px
	
def optimalvertexradviz(x,y):    
    m=len(x[0])
    b=np.zeros(m)
    px=vertexradvizprojection(x,b)
    return px
	

#(max_quality,vertex,Y)=optimalvertexradviz(X,y)
	
FFMpegWriter = animation.writers['ffmpeg']
metadata = dict(title='Star Visualization', artist='SC',
                comment='SC')
writer = FFMpegWriter(fps=15, metadata=metadata)

#plt.ion()
#fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

with writer.saving(fig, "Star Coordinates.mp4", 100):
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
		Y4=sess.run(Y,feed_dict={x_data:train_data})
		
		V=sess.run(W1)
		axes.clear()
		#axes.set_xlim([-8,8])
		#axes.set_ylim([-8,8])		
		#for i in range(n0):
		#    axes.plot([np.zeros(2),V[i]],color="red")	
		for i in range(n0):
			axes.plot([0.0,V[i,0]],[0.0,V[i,1]] ,'b-',lw=1)	
		
		#print(V[0].shape)
		
		#axes[1].clear()
			
		#parallel_coordinates(dat1,'Labels',colormap='rainbow',ax=axes[1,0])
		#X1,Xt1=PCA_plot(Y1,Ytest1)
		X1=Y1
		axes.scatter(X1[:,0],X1[:,1],c=target[train_indices],cmap=cm,marker='.')
		#axes[1,0].scatter(Xt1[:,0],Xt1[:,1],c=target[test_indices],cmap=cm,marker='x')
		

			
		axes.set_title("star coordinates")
		#axes[1,0].legend().set_visible(False)
			
		#axes[1].set_title("parallel coordinates layer")
		#dat4=convert_to_frame(Y4,target[train_indices])	
		#axes[1].plot(loss_vec,'b-')
		#parallel_coordinates(dat4,'Labels',colormap='rainbow',ax=axes[1])
		#X2=optimalvertexradviz(Y4,target[train_indices])
		#axes[1].scatter(X2[:,0],X2[:,1],c=target[train_indices],cmap=cm,marker='.')
		#axes[1].legend().set_visible(False)
		
		#writer.grab_frame()   
		#plt.draw()
		#plt.show()
		plt.pause(0.001)
		writer.grab_frame()	
    

print("LDA: ",NearestCentroidClassifier(X1,target[train_indices]))

plt.show()

'''
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
    Y4=sess.run(Y,feed_dict={x_data:train_data})
    
    V=sess.run(W1)
    axes.clear()	
    #for i in range(n0):
    #    axes.plot([np.zeros(2),V[i]],color="red")	
    for i in range(n0):
        axes.plot([0.0,V[i,0]],[0.0,V[i,1]] ,'b-',lw=1)	
	
    #print(V[0].shape)
	
    #axes[1].clear()
        
    #parallel_coordinates(dat1,'Labels',colormap='rainbow',ax=axes[1,0])
    #X1,Xt1=PCA_plot(Y1,Ytest1)
    X1=Y1
    axes.scatter(X1[:,0],X1[:,1],c=target[train_indices],cmap=cm,marker='.')
    #axes[1,0].scatter(Xt1[:,0],Xt1[:,1],c=target[test_indices],cmap=cm,marker='x')
	

		
    axes.set_title("star coordinates")
    #axes[1,0].legend().set_visible(False)
        
    #axes[1].set_title("parallel coordinates layer")
    #dat4=convert_to_frame(Y4,target[train_indices])	
    #axes[1].plot(loss_vec,'b-')
    #parallel_coordinates(dat4,'Labels',colormap='rainbow',ax=axes[1])
    #X2=optimalvertexradviz(Y4,target[train_indices])
    #axes[1].scatter(X2[:,0],X2[:,1],c=target[train_indices],cmap=cm,marker='.')
    #axes[1].legend().set_visible(False)
	
    #writer.grab_frame()   
    #plt.draw()
    #plt.show()
    plt.pause(0.001) 
	'''










