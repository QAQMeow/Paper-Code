# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 14:05:55 2024

@author: olcxy
"""
import os
import netCDF4 as nc
import numpy as np
import h5netcdf
import joblib
import datetime

N = 90
a = 0.98
with open('../res/other/phatFisk'+str(N)+'.pkl', 'rb') as f:  # 读取pickle文件

    Phat = joblib.load(f)
    f.close()

with open('../res/other/Mask.pkl', 'rb') as f:  # 读取pickle文件
    Maskdata = joblib.load(f)
    
    f.close()
mask = Maskdata['mask']
mlon = Maskdata['lon']
mlat = Maskdata['lat']
LON = Maskdata['LON']
LAT = Maskdata['LAT']


with open('../res/other/Dx'+str(N)+'.pkl', 'rb') as f:  # 读取pickle文件
    Dx = joblib.load(f)
    f.close()


# with open('../Data/PRE025/pre30.pkl', 'rb') as f:  # 读取pickle文件
#     Pre = np.float16(joblib.load(f))
#     f.close()

# with open('../Data/PET025/pet30.pkl', 'rb') as f:  # 读取pickle文件
#     PET = np.float16(joblib.load(f))
#     f.close()
    
    
def getsapei(a,N,alpha,beta,gamm,x):
    sapei = np.zeros_like(x)
    x = np.array(x[N:])
    
    Fx = 1/(1+np.power(alpha/(x-gamm),beta))
    P = 1-Fx
    result = np.zeros(len(P))
    W= np.zeros(len(P))
    W[P<=0.5] = np.sqrt(-2*np.log(P[P<=0.5]))
    W[P>0.5] = np.sqrt(-2*np.log(1-P[P>0.5]))
    
    WW = np.power(W,2)
    WWW = np.power(W,3)
    C=[2.515517,0.802853,0.010328]
    d=[1.432788,0.189269,0.001308]
    result= W - (C[0] + C[1]*W + C[2]*WW) / (1 + d[0]*W + d[1]*WW + d[2]*WWW)
    
    result[P >0.5] = -result[P >0.5]
    sapei[N:] = result
    return sapei 

Phat = Phat[:,mask==1]




SAPEI = np.zeros_like(Dx).astype(np.float16)
# for i in range(len(PET[1])):
#     D = Pre - PET
#     A = np.zeros(N+1)
#     sapei = np.zeros_like(Pre)
#     for i in range(N+1):
#         A[i] = np.power(a,N-i)
    
#     x = []
#     for i in range(len(D)-N):
#         d = np.sum(A*D[i:i+N+1],axis=0)
#         x.append(d)
#     del D    
#     SAPEI[:,i] = getsapei(a,N,Phat[0,i],Phat[1,i],Phat[2,i],x)
#     #print(i)
#     if i%10000==0:
#         print(i)

for i in range(len(Dx[1])):
    x = Dx[:,i]
    SAPEI[:,i] = getsapei(a,N,Phat[0,i],Phat[1,i],Phat[2,i],x)
    #print(i)
    if i%10000==0:
        print(i)


d1 = 'G:'
d2 = 'SCI'
d3 = 'res/climate/sapei'


time_series = [datetime.datetime(1991,1,1) + datetime.timedelta(days=i) for i in range(10958)]

TY = []
for i in time_series:
    TY.append(i.year)
TY = np.array(TY)
  
TM = []
for i in time_series:
    TM.append(i.month) 
TM = np.array(TM)

for y in np.arange(1991,2021):
     
    petdata = nc.Dataset('../res/climate/pet/pet_'+str(y)+'.nc')
    
    
    s1 = SAPEI[TY==y,:]   
    sapei = []
    for i in range(len(s1)):
        m = mask.copy()
        m[m==1] = s1[i]
        sapei.append(m)
    sapei = np.array(sapei)
    if not os.path.isfile(d1+'/'+d2+'/'+d3+'/'+'sapei'+'_'+str(y)+'.nc'):

        NewData = nc.Dataset(d1+'/'+d2+'/'+d3+'/'+'sapei'+str(N)+'_'+
                             str(y)+'.nc', 'w', format='NETCDF4')
        NewData.description = d3+' '+'sapei'
    
        time = NewData.createDimension('time', None)
        lat = NewData.createDimension('lat', 600)
        lon = NewData.createDimension('lon', 1440)
    
        times = NewData.createVariable("time", "f8", ("time",))
        times.units = petdata.variables['time'].units
        times.axis = petdata.variables['time'].axis
        times.calendar = petdata.variables['time'].calendar
        times[:] = petdata.variables['time'][:]
    
        latitudes = NewData.createVariable("lat", "f8", ("lat",))
        latitudes.units = petdata.variables['lat'].units
        latitudes.axis = petdata.variables['lat'].axis
        latitudes[:] = petdata.variables['lat'][:]
    
        longitudes = NewData.createVariable("lon", "f4", ("lon",))
        longitudes.units = petdata.variables['lon'].units
        longitudes.axis = petdata.variables['lon'].axis
        longitudes[:] = petdata.variables['lon'][:]
    
        Gdata = NewData.createVariable('sapei', "f4", ("time", "lat", "lon"), fill_value=-9999, zlib=True,
                                       least_significant_digit=3)
        Gdata.units = ' '
        Gdata.standard_name = 'sapei'
        Gdata.missing_value = -9999
        Gdata[:, :, :] = sapei
    
        NewData.close()
    print(y)
 
 
 
 
 
 