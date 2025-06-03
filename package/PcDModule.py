# -*- coding: utf-8 -*-
 
"""
Created on Wed Jan  1 19:01:58 2025

@author: Meovv Van

email: 1259053332@qq.com

"""
 

def selectNCDsdata(data,measure_id = 0, metric_id = 0,cause_id = 0,location_id = 0,age_id=0,year=0):
    '''  
    measure_id
    1 死亡 5 患病 6 发病
    metric_id
    1 数量 2 百分比 3 率
    cause_id
    558 精神障碍       508 慢性呼吸系统疾病
    542 神经系统疾病   974 糖尿病和肾病
    526 消化系统疾病   491 心血管疾病
    410 肿瘤
    age_id
     1: <5        6: 5-9       7: 10-14     8: 15-19     9: 20-24   
    10: 25-29    11: 30-34    12: 35-39    13: 40-44    14: 45-49   
    15: 50-54    16: 55-59    17: 60-64    18: 65-69    19: 70-75
    
    22: all      23: 5-14     24: 15-49    26: >70      39: 0-14
   284: 20-54   420: <70
   
    '''
    if measure_id == 0:
        measure_id = data['measure_id']
    if metric_id == 0:
        metric_id = data['metric_id']
    if cause_id == 0:
        cause_id = data['cause_id']
    if location_id == 0:
        location_id = data['location_id']
    if age_id == 0:
        age_id = data['age_id']
    if year == 0:
        year = data['year']
        
    
    s_d = data[(data['measure_id']==measure_id)
               &(data['metric_id']==metric_id) 
               & (data['cause_id']==cause_id) 
               & (data['location_id']==location_id)
               & (data['age_id']==age_id)
               & (data['year']==year)
               ]
   
    return s_d


def getAgegroup(data,cname,aid):
    A = data[(data['location_name']==cname)&(data['metric_id']==1)&(data['age_id']==aid)]['val'].values
    B = data[(data['location_name']==cname)&(data['metric_id']==2)&(data['age_id']==aid)]['val'].values
    C = data[(data['location_name']==cname)&(data['metric_id']==1)&(data['age_id']==22)]['val'].values
    D = data[(data['location_name']==cname)&(data['metric_id']==2)&(data['age_id']==22)]['val'].values
    agegroup = (A/B)/(C/D)
    
    return agegroup


import numpy as np
import pandas as pd
def getEdata(data,CIDdata,va):
    Da = {}
    dd = []
    for i in range(len(CIDdata)):
        soc = CIDdata['SOC'][i]
        da = data[(data['Country code']==soc)&(data['Variable']==va)].sort_values('Year',axis=0)['Value']
        if len(da)<20:
            da = np.pad(da,(20-len(da),0),'constant',constant_values=(0,0))
        dd.append(da)
    dd = np.array(dd)        
    Da['SOC'] = CIDdata['SOC']
    Da['location_name'] = CIDdata['location_name']
    for i in range(20):
        Da[str(i+2000)] = dd[:,i]
    Da = pd.DataFrame(Da)
    return Da
    
    
def getAPEdata(data,CIDdata):
    Da = {}
    dd = []
    for i in range(len(CIDdata)):
        soc = CIDdata['SOC'][i]
        da = data[(data['Code']==soc)].sort_values('Year',axis=0)['Air pollution emissions']
        if len(da)<20:
            da = np.pad(da,(20-len(da),0),'constant',constant_values=(0,0))
        dd.append(da)
    dd = np.array(dd)        
    Da['SOC'] = CIDdata['SOC']
    Da['location_name'] = CIDdata['location_name']
    for i in range(20):
        Da[str(i+2000)] = dd[:,i]
    Da = pd.DataFrame(Da)
    return Da
    

def getApGdata(APE,GDP,CIDdata):
    Da = {}
    
    dd = []
    dg = []
    for i in range(len(CIDdata)):
        soc = CIDdata['SOC'][i]
        da = APE[(APE['Code']==soc)].sort_values('Year',axis=0)['Air pollution emissions'].values
        
        if len(da)<20 and len(da)>0:
            da = np.pad(da,(0,20-len(da)),'constant',constant_values=np.nanmax(da))
        if len(da)== 0:
            da = np.zeros(20)
        if soc in GDP['Country Code'].values:
            gdp = GDP[GDP['Country Code']==soc].values[0][3:].astype(np.float32)
        else:
            gdp == np.nan*da
        gdp[np.isnan(gdp)] = np.nanmax(gdp)
        da = da/gdp
        da[np.isnan(da)] = np.nanmean(da)
        dd.append(da)
        dg.append(gdp)
    dd = np.array(dd) 
    dg = np.array(dg)       
    Da['SOC'] = CIDdata['SOC']
    Da['location_name'] = CIDdata['location_name']
    Da2 = Da.copy()
    for i in range(20):
        Da[str(i+2000)] = dd[:,i]
        Da2[str(i+2000)] = dg[:,i]
    Da = pd.DataFrame(Da)
    Da2 = pd.DataFrame(Da2)
    return Da,Da2

def getSAEdata(APE ,CIDdata):
    Da = {}
    
    dd = []
    dg = []
    for i in range(len(CIDdata)):
        soc = CIDdata['SOC'][i]
        da = APE[(APE['Code']==soc)].sort_values('Year',axis=0)['Access to electricity (% of population)'].values
        
        if len(da)<20 and len(da)>0:
            da = np.pad(da,(20-len(da),0),'constant',constant_values=np.nanmin(da))
        if len(da)== 0:
            da = np.zeros(20)*np.nan
         
        
        da[np.isnan(da)] = np.nanmean(da)
        dd.append(da)
       
    dd = np.array(dd) 
   
    Da['SOC'] = CIDdata['SOC']
    Da['location_name'] = CIDdata['location_name']
    
    for i in range(20):
        Da[str(i+2000)] = dd[:,i]
        
    Da = pd.DataFrame(Da)
   
    return Da 


def getEIdata(APE,LA,CIDdata):
    Da = {}
    
    dd = []
    dg = []
    for i in range(len(CIDdata)):
        soc = CIDdata['SOC'][i]
        da = APE[(APE['Code']==soc)].sort_values('Year',axis=0)['Air pollution emissions'].values
        
        if len(da)<20 and len(da)>0:
            da = np.pad(da,(0,20-len(da)),'constant',constant_values=np.nanmax(da))
        if len(da)== 0:
            da = np.zeros(20)
        if soc in LA['Country Code'].values:
            la = LA[LA['Country Code']=='CHN']['Land area(km2)'].values[0]
        else:
            la == 1
         
        da = da/la
        da[np.isnan(da)] = np.nanmean(da)
        dd.append(da)
        
    dd = np.array(dd) 
    
    Da['SOC'] = CIDdata['SOC']
    Da['location_name'] = CIDdata['location_name']
 
    for i in range(20):
        Da[str(i+2000)] = dd[:,i]
 
    Da = pd.DataFrame(Da)
  
    return Da 
