# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:55:30 2025

@author: Meovv Van

email: 1259053332@qq.com

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import pickle
import cmaps
from PIL import Image
import package.CalModule as cl
import package.PlotModule as pm
from scipy import interpolate
Income_Group = pd.read_csv('../res/SEdata/Income_Group.csv')

FatData = pd.read_csv('../res/SEdata/FAT.csv')

old = pd.read_excel('../res/dataprced/Oldpeople.xlsx')
smoke = pd.read_csv('../res/SEdata/share-of-adults-who-smoke.csv')


LID = pd.read_csv('../res/NCDs/Location_ID.csv')
CID = pd.read_excel('../res/other/countriesID.xls') 
CID2 = pd.read_excel('../res/other/C_Code.xlsx') 
mergeID = pd.merge(LID, CID,left_on ='location_name',right_on ='FCNAME',how = 'inner')  
Income_Group = pd.read_csv('../res/SEdata/Income_Group.csv')
mergeID = pd.merge(LID, CID,left_on ='location_name',right_on ='FCNAME',how = 'inner')  
MID = pd.merge(mergeID,Income_Group,left_on ='SOC',right_on ='Country Code',how = 'left')


smoke = pd.read_csv('../res/SEdata/share-of-adults-who-smoke.csv')

c = smoke['Code'].unique()
c = c[c==c]
years = [2000,2005,2010,2015,2018,2019,2020]
X = np.arange(2000, 2021)
Ds = []

for i in range(len(c)):
    ds = smoke[smoke['Code']==c[i]]['Prevalence of current tobacco use (% of adults)'].values
    
    f = interpolate.interp1d(years, ds)
    d2 = f(X)
    Ds.append(list(d2))
Ds = np.asarray(Ds)

data = pd.DataFrame(Ds,columns=X)
data.insert(0,'Code',c)

DC = []




Ds2 = []
Dc2 = []
for i in MID['SOC'].values:
    if i in data['Code'].values:
        print(i)
        Ds2.append(data.loc[data['Code']==i].values[0][1:].T)
       
    else:
        Ds2.append(np.nanmean(data.values[:,1:],axis=0))
       

for i in MID['SOC'].values:
    if i in old['Country Code'].values:
        print(i)
        
        Dc2.append(np.squeeze(old.loc[old['Country Code']==i].values[:,2:]))
    else:
       
        Dc2.append(np.nanmean(old.values[:,2:],axis=0))

data2 = pd.DataFrame(Ds2,columns=X)
data2.insert(0,'Code',MID['SOC'].values)

data3 = pd.DataFrame(Dc2,columns=np.arange(2001, 2021))
data3.insert(0,'Code',MID['SOC'].values)

F = FatData[(FatData['DIM_SEX']=='TOTAL' )&(FatData['DIM_TIME']>=2000)&(FatData['DIM_TIME']<=2020)]
A = CID2[['official_name_en','ISO3166-1-Alpha-3']]
B = pd.DataFrame({'B':np.array(FatData['GEO_NAME_SHORT'].unique())})
C = pd.merge(A,B,left_on ='official_name_en',right_on ='B',how = 'left')
D = pd.merge(C,MID,left_on ='ISO3166-1-Alpha-3',right_on ='SOC',how = 'right')


Dx = []
Fc = F['GEO_NAME_SHORT'].unique()
for i in range(len(Fc)):
    d = F[F['GEO_NAME_SHORT']==Fc[i]].sort_values(by='DIM_TIME')['RATE_PER_100_N'].values
    
    Dx.append(list(d))
Dx = np.asarray(Dx)
Dx = pd.DataFrame(Dx,columns=X)
Dx.insert(0,'GEO_NAME_SHORT',Fc)
Dx2 = []

for i in D['official_name_en'].values:
    if i in Dx['GEO_NAME_SHORT'].values:
        print(i)
        Dx2.append(Dx.loc[Dx['GEO_NAME_SHORT']==i].values[0][1:])
    else:
        Dx2.append(np.nanmean(Dx.values[:,1:],axis=0))


Dx2 = pd.DataFrame(Dx2,columns=X)

Dx2.insert(0,'Code',D['FCNAME'].values)
indx = Dx2['Code'].values
data2.index = indx
data3.index = indx
Dx2.index = indx
data2 = data2.drop(columns=['Code',2000]).T
data3 = data3.drop(columns=['Code']).T
data2.index = np.arange(0,20)
data3.index = np.arange(0,20)
Dx2 = Dx2.drop(columns=['Code',2000]).T
Dx2.index = np.arange(0,20)


OPdata = pd.read_csv('../res/SEdata/Occupational PM etc.csv')
OPdata = OPdata[['location_id', 'location_name','metric_id', 'metric_name', 'year', 'val', 'upper',
'lower']]
a3 = {}
for c in range(len(mergeID)):
    cname = mergeID['location_name'][c]
    fid  =  mergeID['FID']
    da = np.array(OPdata[(OPdata['location_name']==cname)
                   &(OPdata['metric_id']==2)
                   ].sort_values('year',axis=0)['val'][1:-1])
    
    da[np.isnan(da)] = np.nanmax(da)
    a3[cname] = da
a3 = pd.DataFrame(a3)


V_Factor = {}
V_Factor['v1'] = data2 # smoke
V_Factor['v2'] = Dx2   # Fat
V_Factor['v3'] = a3   # 职业暴露
V_Factor['v4'] = data3   # old

with open('../res/dataprced/V_Factor.pkl', 'wb') as f:
 	pickle.dump(V_Factor, f)

