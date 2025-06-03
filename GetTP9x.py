# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:55:24 2024

@author: Van Meovv

E-mails:1259053332@qq.com

github: https://github.com/QAQMeow
"""


import time
import netCDF4 as nc
import numpy as np
import joblib
import pickle


with open('../res/other/Mask.pkl', 'rb') as f:  # 读取pickle文件
    Maskdata = joblib.load(f)
    f.close()
Mask = Maskdata['mask']

Dir1 = ['../res/']
Dir2 = ['climate/']

# def getMonthlyData(Data,t):
#     YM = [] 
#     for k in range(len(t)):
#         YM.append([t[k].year,t[k].month])
#     YM = np.array(YM)
    
#     MD = []
#     for m in range(1,13):
#         #yd = np.nansum(Data[TT[:,0]==y,:,:],axis=0)
#         MD.append(np.nansum(Data[YM[:,1]==m,:,:],axis=0))
#     MD = np.array(MD)
#     return MD

def fliplrdata(data):
    d = data.copy()
    pp = data[:,:,720:].copy()
    pp2 = data[:,:,:720].copy()
    d[:,:,:720] = pp
    d[:,:,720:] = pp2
    
    return d


Years = np.array(list(range(1991,2021)))
T90 = []
T95 = []
T99 = []

for d1 in Dir1:
    for d2 in Dir2:
        D = []
        Tt = []
        for y in Years:   
            td1 = d1+d2+'TMAX/tmax_'+str(y)+'.nc'
            tdf1 = nc.Dataset(td1)
            tvd1 = tdf1.variables['tem'][:,:,:]
            time = tdf1.variables['time']
            t1 = nc.num2date(time,time.units)
            A = tvd1
            T = t1
            if y==1991:
                td0 = d1+d2+'TMAX/tmax_'+str(y-1)+'.nc'
                tdf0 = nc.Dataset(td0)
                tvd0 = tdf0.variables['tem'][-7:,:,:]
                time0 = tdf0.variables['time']
                t0 = nc.num2date(time0,time0.units)
                A = np.concatenate((tvd0,tvd1),axis=0,out=None)
                T = np.concatenate((t0[-7:],t1),axis=0,out=None)
                tdf0.close()
                
            if y==2020:    
                td2 = d1+d2+'TMAX/tmax_'+str(y+1)+'.nc'   
                tdf2 = nc.Dataset(td2)
                tvd2 = tdf0.variables['tem'][:7,:,:]
                time2 = tdf2.variables['time']
                t2 = nc.num2date(time2,time2.units)
                A = np.concatenate((tvd1,tvd2),axis=0,out=None)
                T = np.concatenate((t1,t2[:7]),axis=0,out=None)
                tdf2.close()
           
            tdf1.close()
            
            
            if y == 1991:
                D = A
                Tt = T
            else:
                D = np.concatenate((D,A),axis=0,out=None)
                Tt = np.concatenate((Tt,T),axis=0,out=None)
            print(y)

            

YMD = [] 
for k in range(len(Tt)):
    YMD.append([Tt[k].year,Tt[k].month,Tt[k].day])
YMD = np.array(YMD)
TA = YMD[7:7+365]
T90 = []
T95 = []
T99 = []


for i in range(len(TA)):
    m = TA[i,1]
    d = TA[i,2]
    Dx = []
    for y in range(1991,2021):
        indx = np.intersect1d(np.intersect1d(np.where(YMD[:,0]==y),np.where(YMD[:,1]==m)),np.where(YMD[:,2]==d))[0]
        data = D[indx-7:indx+7,:,:]
        
        if y ==1991:
            
            Dx = data
        else:
            Dx =  np.concatenate((Dx,data),axis=0,out=None)
    
        print([y,i])
    T90.append(np.percentile(Dx, 90, axis=0, method='linear'))       
    T95.append(np.percentile(Dx, 95, axis=0, method='linear'))
    T99.append(np.percentile(Dx, 99, axis=0, method='linear'))    

    
T90 = np.array(T90)
T95 = np.array(T95)
T99 = np.array(T99)    
    
T9x = {}
T9x['T90'] = T90
T9x['T95'] = T95
T9x['T99'] = T99               
with open('../res/other/TP9x_15.pkl', 'wb') as f:
 	pickle.dump(T9x, f)         
            
            
            
            
            
            
            
            