import pandas as pd 
# list of 26 data sets DT26: baseball.data, no information of classess

istar_data=['All_reduced.DATA',  'dermatology.data', 'elephant.DATA', 'ETHZ.data', 'fiber-notnorm.data', 
'freeFoto.DATA', 'mammal.data', 'Mice.data', 'movementLibras.data', 'optdigits.DATA', 
'PHYSHING.DATA', 'QSAR.data', 'satimage.data', 'segment.data', 'SegmentationNormcols.DATA', 
'shapes.data', 'SpamBase.data', 'texture.data', 'twonorm.data', 'VEHICLE.DATA', 
'wdbc_std.data','ionosphere.txt', 'primary-tumor.txt', 'SONAR.txt', 'spectfheart.txt']
#,'basketball.data']

def readfile(fname):
    data=pd.read_table(fname,sep=';',header=None,skiprows=lambda x: x<=3)
    with open(fname) as f:
        att=f.readlines()[3]
    coindex=att.split('\n')[0].split(';')# columns index
    c=data.columns.values
    x=data[c[1:len(c)-1]]
    x.columns=coindex
    y=data[c[len(c)-1]]
    return (x,y)

def ReadData(listdata):
    DATA={}
    for i in range(len(listdata)):
        fname="Datasets/"+istar_data[i]
        x,y=readfile(fname)
        DATA[i]=(x,y)
    return DATA

DT=ReadData(istar_data)