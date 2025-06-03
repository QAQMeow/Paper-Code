# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 11:49:38 2025

@author: Meovv Van

email: 1259053332@qq.com

"""
import joblib
import pickle
import numpy as np
import pandas as pd
import netCDF4 as nc
from PIL import Image
import package.CalModule as cl
 


ncds508 = pd.read_csv('../res/dataprced/NCDs-508-22.csv')
mor508 = ncds508[ncds508['measure_id']==1]
prev508 = ncds508[ncds508['measure_id']==5]

with open('../res/other/Mask.pkl', 'rb') as f:  # 读取pickle文件
    Maskdata = joblib.load(f)
    f.close()
Mask = Maskdata['mask']

lat = Maskdata['lat']
lon = Maskdata['lon']

Mask  = cl.fliplrMap(Mask)

with open('../res/other/TP9x_15.pkl', 'rb') as f:  # 读取pickle文件
    TP9x = joblib.load(f)
    f.close()
T90 = TP9x['T90'] 
T95 = TP9x['T95'] 
T99 = TP9x['T99'] 

Pop = []
CDHDs = []
CDHAs = []
CDHIs = []
DF = []
HF = []
DI = []
HI = []
HD = []
DD = []
EF = []

def getEI(di,hi):
    HI = 1e-10+(hi-32-273.15)/(40-32)
    DI = 1e-10+(di-(-1))/(-2-(-1))
    
    EI = HI+DI
    return EI,HI,DI


for y in range(2001,2021):
    
    fr2 = '../res/SEdata/GlobPop/025/GlobalPop_'+str(y)+'.tif'
    tif2 = np.array(Image.open(fr2))
    tif2[tif2<0] = 0
    pop = np.flipud(tif2)
    Pop.append(pop)
        
    sapei_dir =  '../res/climate/sapei2/sapei_'+str(y)+'.nc'
    data_s = nc.Dataset(sapei_dir)
    sapei = data_s.variables['sapei'][:]
    ti = data_s.variables['time']
    t = nc.num2date(ti,ti.units)
    d_days = sapei.copy()
    di = sapei.copy() 
    di[di>-1] = np.nan
    d_days[d_days>-1] = 0
    d_days[d_days<=-1] = 1
    d_days[d_days!=1] = 0
    
    tmax_dir =  '../res/climate/tmax/tmax_'+str(y)+'.nc'
    data_t = nc.Dataset(tmax_dir)
    tmax = data_t.variables['tem'][:]
  
    tth = T90
    if len(tmax) ==366:
        tth =  np.concatenate((tth,tth[np.newaxis,-1]),axis=0,out=None)
    
    h_days = tmax.copy()
    hi = tmax.copy()
    hi[hi<32+273.15] = np.nan
    h_days[h_days<32+273.15] = np.nan
    h_days = h_days-tth
    h_days[h_days<0] = np.nan
    h_days[~np.isnan(h_days)] = 1
    h_days[h_days!=1] = 0
     
     
    
    cdh_days = d_days*h_days
    
    
    cdh_days[np.isnan(cdh_days)] = 0
    ef = np.zeros([600,1440])*np.nan
    hf = ef.copy()
    df = ef.copy()
    M1 = cl.fliplrMap(Mask)
    for i in range(600):
        for j in range(1440):
            if M1[i,j]==1:
                cdh_days[:,i,j],ef[i,j] = cl.getEvents(cdh_days[:,i,j], 3)
                h_days[:,i,j],hf[i,j] = cl.getEvents(h_days[:,i,j], 3)
                d_days[:,i,j],df[i,j] = cl.getEvents(d_days[:,i,j], 3)
    
    #tm[cdhds!=1] = np.nan
    #ints[cdhds!=1] = np.nan
    # hi = hi*cdh_days
    # di = di*cdh_days
    ei,hi,di = getEI(di,hi)
    ei = ei*cdh_days
    hi = hi*h_days
    di = di*d_days
    Ycdh_days = np.nansum(cdh_days,axis=0)
    Ycdh_days = cl.fliplrMap(Ycdh_days)
    Ycdh_i = np.nansum(ei,axis=0)
    
    
    cdh_area = Ycdh_days.copy()
    cdh_area[cdh_area>0] = 1
    CDHDs.append(Ycdh_days)
    CDHAs.append(cdh_area)
    CDHIs.append(cl.fliplrMap(Ycdh_i))
    DF.append(cl.fliplrMap(df))
    HF.append(cl.fliplrMap(hf))
    DI.append(cl.fliplrMap(np.nansum(di,axis=0)))
    HI.append(cl.fliplrMap(np.nansum(hi,axis=0)))
    HD.append(cl.fliplrMap(np.nansum(h_days,axis=0)))
    DD.append(cl.fliplrMap(np.nansum(d_days,axis=0)))
    EF.append(cl.fliplrMap(ef))
    print(y)

CDHEs_data  = {}
CDHEs_data['CDHE_frequence'] = np.array(EF)
CDHEs_data['CDHE_days']  = np.array(CDHDs)
CDHEs_data['CDHE_area'] = np.array(CDHAs)   
CDHEs_data['CDHE_intensity'] = np.array(CDHIs)  
CDHEs_data['HE_frequence'] = np.array(HF)
CDHEs_data['DE_frequence'] = np.array(DF)
CDHEs_data['HE_days'] = np.array(HD)
CDHEs_data['DE_days'] = np.array(DD)
CDHEs_data['HE_intensity'] = np.array(HI)
CDHEs_data['DE_intensity'] = np.array(DI)
CDHEs_data['Pop'] = np.array(Pop)
#CDHEs_data = pd.DataFrame(CDHEs_data)
with open('../res/dataprced/CDHEs_data.pkl', 'wb') as f:
	pickle.dump(CDHEs_data, f)


# ED = {}
# EP = {}
# EA = {}
# Pp = {}
# for c in range(len(mergeID)):
#     cname = mergeID['location_name'][c]
#     fid  =  mergeID['FID'][c]
#     frac = prev508[(prev508['location_name']==cname)].sort_values('year',axis=0)['val'][1:-1]
#     ced = []
#     cep = []
#     cea = []
#     pp = []
#     for i in range(20):
#         cdhd = CDHDs[i]
#         cdha = CDHAs[i]
#         pop = Pop[i]
    
#         c1 = np.float32(countries.copy())
#         c1[c1!=fid] = np.nan
#         c1[c1>0] = 1
#         cp = pop*c1*frac.values[i]
        
#         ep = np.nansum(cdha*cp)
#         ed = np.nansum(cdhd*cp)/ep
#         ea = np.nansum(c1*gridsArea*cdha)
#         ced.append(ed)
#         cep.append(ep)
#         cea.append(ea)
#         pp.append(np.nansum(cp))
#     ED[cname] = np.array(ced)
#     EP[cname] = np.array(cep)
#     EA[cname] = np.array(cea)
#     Pp[cname] = np.array(pp)
#     print(c)
    
# ED = pd.DataFrame(ED)        
# EP = pd.DataFrame(EP) 
# EA = pd.DataFrame(EA) 
# Pp = pd.DataFrame(Pp) 