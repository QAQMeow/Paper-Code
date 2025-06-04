# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 16:48:10 2025

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

LID = pd.read_csv('../res/NCDs/Location_ID.csv')
CID = pd.read_excel('../res/other/countriesID.xls')

mergeID = pd.merge(LID, CID,left_on ='location_name',right_on ='FCNAME',how = 'inner')

fr ='../res/map/World_countries.tif'
tif =  Image.open(fr )
countries = np.flipud(tif)
fr1 ='../res/map/GlobalAREA.tif'
gridsArea =  np.array(Image.open(fr1))

# with open('../res/dataprced/CDHEs_data.pkl', 'rb') as f:  # 读取pickle文件
#     CDHEs_data = joblib.load(f)
#     f.close()
with open('../res/dataprced/CDHEs_data.pkl', 'rb') as f:  # 读取pickle文件
    CDHEs_data = joblib.load(f)
    f.close()

CDHFs = CDHEs_data['CDHE_frequence']
CDHDs = CDHEs_data['CDHE_days']
CDHAs = CDHEs_data['CDHE_area']
CDHIs = CDHEs_data['CDHE_intensity']
Pop = CDHEs_data['Pop']



ED = {}
EF = {}
EI = {}


for c in range(len(mergeID)):
    cname = mergeID['location_name'][c]
    fid  =  mergeID['FID'][c]
    frac = prev508[(prev508['location_name']==cname)
                   &(prev508['metric_id']==2)
                   &(prev508['age_id']==22)
                   ].sort_values('year',axis=0)['val'][1:-1]
    ced = []
    cef = []
    cei = []
     
    for i in range(20):
        cdhd = CDHDs[i]
        cdha = CDHAs[i]
        cdhf = CDHFs[i]
        cdhi = CDHIs[i]
        pop = Pop[i]
    
        c1 = np.float32(countries.copy())
        c1[c1!=fid] = np.nan
        c1[c1>0] = 1
        cp = pop*c1*frac.values[i]
        
        ep = np.nansum(cdha*cp)
        ed = np.nansum(cdhd*cp)/ep
        
        ef = np.nansum(cdhf*cp)/ep
        ei = np.nansum(cdhi*cp)/ep
        # cp = c1
        # ed = np.nansum(cdhd*cp)/np.nansum(cp)
        # ef = np.nansum(cdhf*cp)/np.nansum(cp)
        ced.append(ed)
        cei.append(ei)
        cef.append(ef)
         
         
         
    ced = np.array(ced)    
    ced[np.isnan(ced)] = 0
    cef = np.array(cef) 
    cef[np.isnan(cef)] = 0
    cei = np.array(cei) 
    cei[np.isnan(cei)] = 0
    ED[cname] = ced
    EF[cname] = cef
    EI[cname] = cei
    print(c)
    
ED = pd.DataFrame(ED)        
EF = pd.DataFrame(EF) 
EI = pd.DataFrame(EI) 
H_Factor = {}
H_Factor['h1'] = ED  #历时
H_Factor['h2'] = EF  #频次
H_Factor['h3'] = EI  #强度
with open('../res/dataprced/H_Factor.pkl', 'wb') as f:
 	pickle.dump(H_Factor, f)
     

