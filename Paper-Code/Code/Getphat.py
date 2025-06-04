# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 21:25:54 2024

@author: olcxy
"""
import os
import numpy as np
import joblib
import netCDF4 as nc
import threading 
from FiskFit2 import FiskFit2
import datetime


with open('../res/other/Mask.pkl', 'rb') as f:  # 读取pickle文件
    Maskdata = joblib.load(f)
    
    f.close()
mask = Maskdata['mask']
mlon = Maskdata['lon']
mlat = Maskdata['lat']
LON = Maskdata['LON']
LAT = Maskdata['LAT']

Pre = []
PET = []
for y in range(1991,2021):
    predata = nc.Dataset('../res/climate/pre/pre_'+str(y)+'.nc')
    pr = np.array(predata.variables['pre'])
    petdata = nc.Dataset('../res/climate/pet/pet_'+str(y)+'.nc')
    pet = np.array(petdata.variables['pet'])
    if y == 1991:
        Pre = pr[:,mask==1]
        PET = pet[:,mask==1]
    else:
        Pre = np.concatenate(((Pre,pr[:,mask==1])),axis=0,out=None)
        PET = np.concatenate(((PET,pet[:,mask==1])),axis=0,out=None)
    print(y)
    predata.close()
    petdata.close()
#D:Pre-PET    

D = Pre-PET
# with open('../Data/PRE025/pre.pkl', 'wb') as f: 
#     joblib.dump(Pre,f)
#     f.close()
# with open('../Data/PET025/pet.pkl', 'wb') as f:
#     joblib.dump(PET,f)
#     f.close()
    
del Pre,PET


def getsapei(alpha,beta,gamm,x,N):
    
    sapei = np.zeros(len(x)+N)
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



N = 90
a = 0.98
A = np.float32(np.zeros([N+1,np.shape(D)[1]]))
for i in range(N+1):
    A[i,:] = np.power(a,N-i)
    
Dx = np.float32(np.zeros(np.shape(D)))

def process(N,Dr,Dm,B):
    for i in range(len(Dr)-N):
        Dm[i+N,:] = np.nansum(np.float32(Dr[i:i+N+1,:])*B,axis = 0)
        #D[:N,:,:] = 0;
           
s = np.array([ 0,0,0,0,0,0,0],dtype = int)
e = np.array([2000,4000,6000,8000,10000,12000,len(D)],dtype = int)
for i in range(len(e)-1):
    s[i+1] = e[i]-N
T = {}
for i in range(7):
    T[i] = threading.Thread(target=process,args=(N,D[s[i]:e[i],:],Dx[s[i]:e[i],:],A))
for i in range(7):
    T[i].start()
for i in range(7):
    T[i].join()

with open('../res/other/Dx'+str(N)+'.pkl', 'wb') as f:  # 读取pickle文件
    joblib.dump(Dx,f)
    f.close()
    
 


# with open('../Data/Dx.pkl', 'rb') as f:  # 读取pickle文件
#     Dx = joblib.load(f)
    
#     f.close()


Sa = np.shape(Dx)
SAPEI = np.float32(np.zeros_like(Dx))

Ph = np.zeros([3,Sa[1]])

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'));   
for i in range(Sa[1]):
    [alpha,beta,gamm] = FiskFit2(Dx[N:,i])
    Ph[:,i] = [alpha,beta,gamm]
    #SAPEI[:,i] = getsapei(alpha, beta, gamm, Dx[N:,i], N)  
    if i%10000==0:
        print(i)
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'));

'''   
Db = Dx[N:,:].copy()
del Dx

print('Dx ready')
#[alpha,beta,gamm] = FiskFit2(np.nanmean(np.float32(Db),axis=1))
#[alpha,beta,gamm] = np.load(wr+'abg.npy')




def process2(S,ph,Dr):
    
    for i in range(np.shape(Dr)[1]):
        if not np.isnan(Dr[1,i]):
            [alpha,beta,gamm] = FiskFit2(Dr[:,i])
            Ph[:,i] = [alpha,beta,gamm]
            S[:,i] = getsapei(alpha, beta, gamm, Dr[:,i], N)
s1 = np.array([ 0,0,0,0,0,0,0,0,0,0,0,0,0],dtype = int)
e1 = np.array([20000,40000,60000,800000,100000,120000,140000,160000,180000,200000,220000,240000,Sa[1]],dtype = int)

T = {}
for i in range(13):
    T[i] = threading.Thread(target=process2,args=(SAPEI[:,s1[i]:e1[i]],Ph[:,s1[i]:e1[i]],Db[:,s1[i]:e1[i]]))
for i in range(13):
    T[i].start()
for i in range(13):
    T[i].join()
'''

phat = np.zeros([3,600,1440])*np.nan
for i in range(3):
    phat[i,mask==1] = Ph[i]
    
   
    

with open('../res/other/phatFisk'+str(N)+'.pkl', 'wb') as f:  # 读取pickle文件

    joblib.dump(phat,f)
    f.close()



'''
print('SAPEI ready')
time_series = [datetime.datetime(1981,1,1) + datetime.timedelta(days=i) for i in range(14610)]

TY = []
for i in time_series:
    TY.append(i.year)
TY = np.array(TY)
  
TM = []
for i in time_series:
    TM.append(i.month)
TM = np.array(TM)
d1 = 'G:'
d2 = 'SCIGlobalData'
d3 = 'Data/SAPEI'

for y in np.arange(1981,2021):
    maskdata = nc.Dataset('G:/SCIGlobalData/pr_2015.nc')
    petdata = nc.Dataset('H:'+'/'+'SCIData'+'/'+'PET'+'/'+str(y)+'_daily_pet.hdf')
    
    
    s1 = SAPEI[TY==y,:]   
    sapei = []
    for i in range(len(s1)):
        m = mask.copy()
        m[m==1] = s1[i]
        sapei.append(m)
    sapei = np.array(sapei)
    if not os.path.isfile(d1+'/'+d2+'/'+d3+'/'+'sapei'+str(N)+'_'+str(y)+'.nc'):

        NewData = nc.Dataset(d1+'/'+d2+'/'+d3+'/'+'sapei_' +
                             str(y)+'.nc', 'w', format='NETCDF4')
        NewData.description = d3+' '+'sapei'
    
        time = NewData.createDimension('time', None)
        lat = NewData.createDimension('lat', 600)
        lon = NewData.createDimension('lon', 1440)
    
        times = NewData.createVariable("time", "f8", ("time",))
        times.units = petdata.variables['time'].units
        times.axis = maskdata.variables['time'].axis
        times.calendar = petdata.variables['time'].calendar
        times[:] = petdata.variables['time'][:]
    
        latitudes = NewData.createVariable("lat", "f8", ("lat",))
        latitudes.units = maskdata.variables['lat'].units
        latitudes.axis = maskdata.variables['lat'].axis
        latitudes[:] = maskdata.variables['lat'][:]
    
        longitudes = NewData.createVariable("lon", "f4", ("lon",))
        longitudes.units = maskdata.variables['lon'].units
        longitudes.axis = maskdata.variables['lon'].axis
        longitudes[:] = maskdata.variables['lon'][:]
    
        Gdata = NewData.createVariable('sapei', "f4", ("time", "lat", "lon"), fill_value=-9999, zlib=True,
                                       least_significant_digit=3)
        Gdata.units = ' '
        Gdata.standard_name = 'sapei'
        Gdata.missing_value = -9999
        Gdata[:, :, :] = sapei
    
        NewData.close()
    print(y)
'''


