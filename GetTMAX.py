# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:05:20 2024

@author: olcxy
"""

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
import xarray as xr
import pandas as pd
import h5netcdf
import joblib
import pickle
import cmaps
import cartopy.crs as ccrs
from mpl_toolkits.basemap import Basemap
import datetime
import pymannkendall as mk
import scipy.stats as st
import os
import netCDF4 as nc
maskdata = nc.Dataset('G:/SCIGlobalData/pr_2015.nc')
mask  = np.array( maskdata.variables['pr'][1,:,:])
mask[mask>1e15] = np.nan
mask[mask==mask] = 1
mlon = maskdata.variables['lon'][:]
mlat = maskdata.variables['lat'][:]
LON,LAT = np.meshgrid(mlon,mlat)



time = [datetime.datetime(2021,1,1) + datetime.timedelta(days=i) for i in range(365)]

TY = []
for i in time:
    TY.append(i.year)
TY = np.array(TY)
  
TM = []
for i in time:
    TM.append(i.month)
TM = np.array(TM)

d1 = 'G:/SCI/res/climate'
d2 = 'OriginalData/Temperature'
d3 = 'TMAX'
Years  = np.arange(2021,2022 )  

d4 = []
for y in Years:
    petdata = h5netcdf.File('G:/SCI/res/climate/OriginalData/PET/'+str(y)+'_daily_pet.nc')
    lon = petdata.variables['longitude'][:]
    lat = petdata.variables['latitude'][:]
    TIME = petdata.variables['time']
    time = nc.num2date(TIME,TIME.attrs['units'])
   
    tmax = []
    for i in range(len(time)):
        temdata = h5netcdf.File('G:/SCI/res/climate/OriginalData/Temperature/TMAX/Temperature-Air-2m-Max-24h_C3S-glob-agric_AgERA5_'
                             +time[i].strftime()[:4]+time[i].strftime()[5:7]+time[i].strftime()[8:10]+'_final-v1.1.nc')
        tem =  np.squeeze(temdata.variables['Temperature_Air_2m_Max_24h'][:])
        temdata.close()
        tem1 = tem[:,:1800]
        tem2 = tem[:,1800:]
        tem3 = tem.copy()
        tem3[:,:1800] = tem2
        tem3[:,1800:] = tem1
               
        temd = xr.DataArray(
            data=tem3,
            dims=("lat", "lon"),
            coords={"lat": lat[:], "lon": lon[:]+180},
        )
        
        
        dtem = np.array(temd.interp(lon=mlon, lat=mlat, method="nearest"))
        tmax.append(dtem)
        
      
        d4.append(dtem[mask==mask])
    
    if not os.path.isfile(d1+'/'+d2+'/'+d3+'/'+'tmax'+'_'+str(y)+'.nc'):

        tmax = np.array(tmax,dtype = np.float32)    
        tmax[tmax<0] = -9999
        NewData = nc.Dataset(d1+'/'+d2+'/'+d3+'/'+'tmax_'+str(y)+'.nc','w',format = 'NETCDF4')
        NewData.description = d3+' '+'sapei'
        
        time = NewData.createDimension('time', None)
        lat = NewData.createDimension('lat', 600)
        lon = NewData.createDimension('lon', 1440)
        
        times = NewData.createVariable("time","f8",("time",))
        times.units = petdata.variables['time'].attrs['units']
        times.axis =maskdata.variables['time'].axis
        times.calendar = petdata.variables['time'].attrs['calendar']
        times[:] = petdata.variables['time'][:]
        
        latitudes = NewData.createVariable("lat","f8",("lat",))
        latitudes.units =  maskdata.variables['lat'].units
        latitudes.axis = maskdata.variables['lat'].axis
        latitudes[:] = maskdata.variables['lat'][:]
        
        longitudes = NewData.createVariable("lon","f4",("lon",))
        longitudes.units = maskdata.variables['lon'].units
        longitudes.axis = maskdata.variables['lon'].axis
        longitudes[:] = maskdata.variables['lon'][:]
        
        Gdata = NewData.createVariable('tmax',"f4",("time","lat","lon"),fill_value =1e+20,zlib=True,
                                       least_significant_digit=3)
        Gdata.units = ' '
        Gdata.standard_name = 'tmax'
        Gdata.missing_value = -9999
        Gdata[:,:,:] = tmax
        
        NewData.close()
        petdata.close()    
       
    print(y)


   
    
    
    
    