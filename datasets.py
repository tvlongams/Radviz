import numpy as np
import pandas as pd

def ReadData():
    DATA={}
    #iris
    data=np.loadtxt("Datasets//iris.dat")
    x=data[:,1:5] # x_vals.shape=(150,4)
    y=data[:,0].astype(int)
    DATA['iris']=(x,y)
    #wine
    data=np.loadtxt("Datasets//wine.data",delimiter=",")
    x=data[:,1:13] # x_vals.shape=(178,12)
    y=data[:,0]
    DATA['wine']=(x,y)
    #olive
    data=pd.read_table("Datasets//olive.txt",delimiter='\t',header=None)
    x=np.array(data.loc[:,1:8]).astype(float)
    y=np.array(data[0]).astype(int)
    DATA['olive']=(x,y)
    #y14c
    data=pd.read_table("Datasets//y14c.txt",delimiter='\t',header=None)
    x=np.array(data.loc[:,1:10]).astype(float)
    y=np.array(data[11]).astype(int)
    DATA['y14c']=(x,y)
    #ecoli
    data=pd.read_table("Datasets//ecoli.txt",sep='\t',header=None)
    x=np.array(data.loc[:,1:7]).astype(float)
    y=pd.factorize(data[8])[0]
    DATA['ecoli']=(x,y)
    #auto-mpg
    data=pd.read_table("Datasets//auto-mpg.data",delimiter='\t',header=None)
    x=np.array(data.loc[:,0:6]).astype(float)
    y=np.array(data[7]).astype(int)
    DATA['auto']=(x,y)
    #breast-cancer-wisconsin.data
    data=pd.read_table("Datasets//breast-cancer-wisconsin.data",delimiter=',',header=None)
    x=np.array(data.loc[:,1:9]).astype(float)
    y=pd.factorize(data.loc[:,10])[0]
    DATA['bcw']=(x,y)
    #wdbc
    data=pd.read_table("Datasets//wdbc.data",delimiter=',',header=None)
    x=data.loc[:,2:]
    #y=wdbc.loc[:,1]
    y=pd.factorize(data.loc[:,1])[0]
    DATA['wdbc']=(x,y)
    #wpbc
    data=pd.read_table("Datasets//wpbc.data",delimiter=',',header=None)
    x=data.loc[:,2:]
    y=pd.factorize(data.loc[:,1])[0]
    DATA['wpdc']=(x,y)
    #yeast
    data=pd.read_table("Datasets//yeast.txt",delimiter='\t',header=None)
    x=data.loc[:,1:8]
    y=pd.factorize(data.loc[:,9])[0]
    DATA['yeast']=(x,y)
    #pendigits_16.train
    data=np.loadtxt("Datasets//pendigits_16.tra",delimiter=',')
    x=data[:,0:16]# (7494,16)
    y=data[:,-1].astype(int)
    DATA['pen.tra']=(x,y)
    #pendigits_16.test
    data=np.loadtxt("Datasets//pendigits_16.tes",delimiter=',')
    x=data[:,0:16]# (3498,16)
    y=data[:,-1].astype(int)
    DATA['pen.tes']=(x,y)
    #optdigits_64.train
    data=np.loadtxt("Datasets//optdigits_64.tra",delimiter=',')
    x=data[:,0:64]# (3823,64)
    y=data[:,-1].astype(int)
    DATA['opt.tra']=(x,y)
    #optdigits_64.test
    data=np.loadtxt("Datasets//optdigits_64.tes",delimiter=',')
    x=data[:,0:64]# (1797,64)
    y=data[:,-1].astype(int)
    DATA['opt.tes']=(x,y)
    return DATA

DATA=ReadData()
uci_data=['opt.tes', 'yeast', 'wpdc', 'auto', 'wdbc', 'olive',
'ecoli', 'y14c', 'pen.tes', 'opt.tra', 'bcw', 'pen.tra', 'wine', 'iris']