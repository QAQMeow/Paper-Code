# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 11:22:43 2025

@author: Meovv Van

email: 1259053332@qq.com

"""
import joblib
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import package.CalModule as cl

ncds508 = pd.read_csv('../res/dataprced/NCDs-508-22.csv')
mor508 = ncds508[ncds508['measure_id']==1]
prev508 = ncds508[ncds508['measure_id']==5]
inc508 = ncds508[ncds508['measure_id']==6]
LID = pd.read_csv('../res/NCDs/Location_ID.csv')
CID = pd.read_excel('../res/other/countriesID.xls')

mergeID = pd.merge(LID, CID,left_on ='location_name',right_on ='FCNAME',how = 'inner')

fr ='../res/map/World_countries.tif'
tif =  Image.open(fr )
countries = np.flipud(tif)
fr1 ='../res/map/GlobalAREA.tif'
gridsArea =  np.array(Image.open(fr1))

with open('../res/dataprced/CDHEs_data.pkl', 'rb') as f:  # 读取pickle文件
    CDHEs_data = joblib.load(f)
    f.close()

CDHAs = CDHEs_data['CDHE_area']
Pop = CDHEs_data['Pop']

Pop508 = {}
EP = {}
D508 = {}
Dr = {}
E6 = {} 
E7 = {}
for c in range(len(mergeID)):
    cname = mergeID['location_name'][c]
    fid  =  mergeID['FID'][c]
    frac = prev508[(prev508['location_name']==cname)&(prev508['metric_id']==2)&(prev508['age_id']==22)].sort_values('year',axis=0)['val'][1:-1]
    d508 = mor508[(mor508['location_name']==cname)&(mor508['metric_id']==1)&(mor508['age_id']==22)].sort_values('year',axis=0)['val'][1:-1]
    p508_2 = prev508[(prev508['location_name']==cname)&(prev508['metric_id']==1)&(prev508['age_id']==22)].sort_values('year',axis=0)['val'][1:-1]
    p508 = []
    Ep = []
    Ecp = []
    e6 = np.array(frac)
    e7 = np.array(inc508[(inc508['location_name']==cname)&(inc508['metric_id']==2)&(inc508['age_id']==22)].sort_values('year',axis=0)['val'][1:-1])
    for i in range(20):
        
        cdha = CDHAs[i].copy() #暴露面积
        pop = Pop[i]  #人口格网
        c1 = np.float32(countries.copy())
        c1[c1!=fid] = np.nan
        c1[c1>0] = 1  #国家
        cdha[c1!=1] = np.nan
        popc = pop*c1
        
        cp = popc*frac.values[i] #患病人口格网
        epg = cdha*popc*frac.values[i] #暴露患病格网
        ppr = np.nansum(cp) #患病人口格网总数
        
        ep = np.nansum(cdha*popc)/np.nansum(popc)
        ecp =  np.nansum(epg)
        p508.append(ppr)
        Ecp.append(ecp)
        Ep.append(ep)
        
    #p508 = np.array(p508)
    p508 = frac.values
    Ecp = np.array(Ecp)
    Ep1 = frac.values*np.array(Ep)
    #Pop508[cname] = p508_2.values
    Pop508[cname] =p508_2.values
    EP[cname] = Ep1
    D508[cname] = np.array(d508)
    Dr[cname] = d508.values/p508_2.values
    E6[cname] = e6
    E7[cname] = e7
    print(c)
    
Pop508 = pd.DataFrame(Pop508)
EP = pd.DataFrame(EP)
D508 = pd.DataFrame(D508)
Dr = pd.DataFrame(Dr)
#E_f = EP/Pop508
E_f = 100*EP#/Pop508
E_Factor = {}
E_Factor['e1'] = E_f
E_Factor['e2'] = EP
E_Factor['e3'] = Pop508
E_Factor['e4'] = Dr
E_Factor['e5'] = D508
E_Factor['e6'] = pd.DataFrame(E6)
E_Factor['e7'] = pd.DataFrame(E7)
with open('../res/dataprced/E_Factor.pkl', 'wb') as f:
 	pickle.dump(E_Factor, f)