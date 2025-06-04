# -*- coding: utf-8 -*-
 
"""
Created on Wed Jan  1 19:01:58 2025

@author: Meovv Van

email: 1259053332@qq.com

"""

import numpy as np

def fliplrMap(data):
    '''
    change  central meridian 0 to 180 

    Parameters                       
    -------      
    data : numpy.array,MxN          
           Array with central meridian 0    
    
    Returns
    -------
    d : numpy.array,MxN
        Array with central meridian 180
     
    '''
    d = data.copy()
    S  = np.shape(d)
    m = int(S[1]/2)
    pp = data[:,m:].copy()
    pp2 = data[:,:m].copy()
    d[:,:m] = pp
    d[:,m:] = pp2
    return d

def getEvents(data,th):
    '''
    The event consists of a series of minor events occurring on consecutive days

    Parameters
    ----------
    data : list or numpy.array,int,,Nx1, a sequence only contains 0 and 1,
           0 means that the minor event did not occur, 1 is the opposite
         
    th : int
         Minimum number of consecutive days to be recognized as an event
         

    Returns
    -------
    x : a sequence that only contains 0 and 1,and only contains events which consecutive days >= th
        0 means that the minor event did not occur, 1 is the opposite
        
    f : frequence of events(consecutive days >= th) in data

    '''
    x = np.zeros_like(data)
    d1 = np.append(data,0)
    d1 = np.insert(d1,0,0)
    b = d1[:-1]-d1[1:]

    s = np.where(b==-1)[0]
    t = np.where(b==1)[0]
    c = (t-s)
    e = np.where(c>=th)[0]
    for i in range(len(e)):
        n = e[i]
        x[s[n]:t[n]] = 1
    f = len(e)
    return x,f



def classDa(r,le):
    r2 = r.copy()
    if le == 5:
        lv = np.array([5,5,25,75,95])
    if le == 7:
        lv = np.array([12.5,12.5,25,37.5,62.5,75,87.5])
    l = 1    
    r2[r<=np.nanpercentile(r,lv[0])] = 1
         
    for i in range(1,le):
        l+=1
        r2[r>np.nanpercentile(r,lv[i])] = l
        
        
    return r2


def Normalize(data,dirc=1):
    '''
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    dirc : 0 or 1
    The default is 1,max-min Normalize. 0 is min-max Normalize

    Returns
    -------
    Nd : data were Normalized

    '''
    maxd = np.nanmax(data)
    mind = np.nanmin(data) 
    
    if dirc == 1:
        Nd = (data-mind)/(maxd-mind+1e-15)
    else:
        Nd = (maxd-data)/(maxd-mind+1e-15)
    
    
    Nd[Nd!=Nd] = 0
    return Nd + 1e-15

def getbd(X,db):
    from sklearn.neighbors import KernelDensity
    CDF = 10
    X=np.array(X)
    X[X!=X] = 0
    while (np.abs(np.nanmax(CDF)-1)>1e-15):
        kde = KernelDensity(kernel='gaussian',bandwidth=len(X)/db).fit(X.reshape(-1, 1))
        pdf = np.exp(kde.score_samples(np.sort(X).reshape(-1, 1)))
        CDF = np.cumsum(pdf)
        if np.nanmax(CDF)>1:
            db = db-2*np.abs(np.nanmax(CDF)-1)
        else:db = db+np.abs(np.nanmax(CDF)-1)/2
        #print([db,np.abs(np.nanmax(CDF)-1),np.nanmax(CDF)])
    return len(X)/db


def Normalize_c(Xd,dirc =1):
    '''
    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    dirc : 0 or 1
    The default is 1,max-min Normalize. 0 is max-min Normalize

    Returns
    -------
    Nd : data were Normalized

    '''
    import pandas as pd
    from sklearn.neighbors import KernelDensity
    Xd=np.array(Xd)
    X = Xd[Xd==Xd]
    D = pd.DataFrame({'index':range(len(X)),'data':X})
    kde = KernelDensity(kernel='gaussian',algorithm ='kd_tree',bandwidth=getbd(X,2)).fit(X.reshape(-1, 1))
    pdf = np.exp(kde.score_samples(np.sort(X).reshape(-1, 1)))
    CDF = np.cumsum(pdf)
    D2 =  D.sort_values('data')
    D2.insert(D2.shape[1], 'pdf', value=pdf)
    D2.insert(D2.shape[1], 'cdf', value=CDF)
    
    nX = Xd.copy()
    nX[nX==nX] = D2.sort_values('index')['cdf']
    if dirc==1:
        nX[nX==nX] = D2.sort_values('index')['cdf']
        return  nX
    else:
        nX[nX==nX] = 1-D2.sort_values('index')['cdf']
        return  nX
    


def Normalize_z(Xd):
    '''
    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    dirc : 0 or 1
    The default is 1,max-min Normalize. 0 is max-min Normalize

    Returns
    -------
    Nd : data were Normalized

    '''
    import numpy as np
    
    X_std = (np.array(Xd)-np.mean(Xd))/(np.std(Xd)+1e-16)
    X_std[X_std!=X_std] = 0
    return X_std
    
    
def getW(X):
    X = np.float32(X)
    if X.shape[1] >1:
        sigma = np.nanstd(X , axis=0)
        corr = np.corrcoef(X .T)
        C = sigma * np.nansum(1 - np.abs(corr), axis=0)
        weights = C / np.nansum(C)
        weights[np.isnan(weights)] = 1/X.shape[1]
    else:
        weights =  1
    
    return weights    


def getPCAW(data):
    from sklearn.decomposition import PCA    
    data = data.dropna(axis=1)
    data = np.array(data).T
    pca = PCA(n_components= 1)
    pca.fit(data)
    return pca.explained_variance_ratio_[0]

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

def calRF(w,Data):
    
    return  np.nansum(Data*w,axis=1)

def getHEVAR(H,E,V,A,m=1):
    
    import pandas as pd
    Income_Group = pd.read_csv('../res/SEdata/Income_Group.csv')
    LID = pd.read_csv('../res/NCDs/Location_ID.csv')
    CID = pd.read_excel('../res/other/countriesID.xls')
    mergeID = pd.merge(LID, CID,left_on ='location_name',right_on ='FCNAME',how = 'inner')  
    MID = pd.merge(mergeID,Income_Group,left_on ='SOC',right_on ='Country Code',how = 'left')
    
    H1 = Normalize(H['h1'])
    H2 = Normalize(H['h2'])
    H3 = Normalize(H['h3'])
    
    E1 = Normalize(E['e1'])
    
    V1 = Normalize(V['v1'])
    V2 = Normalize(V['v2'])
    V3 = Normalize(V['v3'])
    V4 = Normalize(V['v4'])
    
    A1 = Normalize(A['a1'])
    A2 = Normalize(A['a2'],0)
    A3 = Normalize(A['a3'])
    A4 = Normalize(A['a4'],0)
    RF = {}
    W = {}
    
    # w_h = np.array([getPCAW(H1),getPCAW(H2),getPCAW(H3)])
    # w_e = getPCAW(E1)
    # w_v = np.array([getPCAW(V1),getPCAW(V2),getPCAW(V3),getPCAW(V4)])
    # w_a = np.array([getPCAW(A1),getPCAW(A2),getPCAW(A3),getPCAW(A4)])
  
    
    hf = pd.DataFrame({})
    ef = pd.DataFrame({})
    vf = pd.DataFrame({})
    af = pd.DataFrame({})
    r = pd.DataFrame({})
    for cname in mergeID['location_name']:
        # w_h  = getW(np.array([H1[cname],H2[cname],H3[cname]]).T)
        # w_e  = getW(np.array([E1[cname]]).T)
        # w_v  = getW(np.array([V1[cname],V2[cname],V3[cname],V4[cname]]).T)
        # w_a  = getW(np.array([A1[cname],A2[cname],A3[cname],A4[cname]]).T)
        w_h = np.array([1,1,1])/3
        w_e = np.array([1])
        w_v = np.array([1,1,1,1])/4
        w_a = np.array([1,1,1,1])/4
        
        #wh[cname],we[cname],wa[cname],wv[cname] = w_h ,w_e ,w_v ,w_a 
        W[cname] =  {'wh':w_h,'we':w_e ,'wv':w_v,'wa':w_a}
       
    for cname in mergeID['location_name']: 
        
        hf[cname] = calRF(W[cname]['wh'],np.array([H1[cname],H2[cname],H3[cname]]).T)
        ef[cname] = calRF(W[cname]['we'],np.array([E1[cname]]).T)
        vf[cname] = calRF(W[cname]['wv'],np.array([V1[cname],V2[cname],V3[cname],V4[cname]]).T)
        af[cname] = calRF(W[cname]['wa'],np.array([A1[cname],A2[cname],A3[cname],A4[cname]]).T)
    
    # hf = Normalize(hf)
    # ef = Normalize(ef)
    # af = Normalize(af)
    # vf = Normalize(vf)
    for cname in mergeID['location_name']:    
        if m == 1:
            r[cname] =  (hf[cname].values*ef[cname].values*vf[cname].values)/(af[cname].values)
        if m == 2:
            r[cname] =  (hf[cname].values+ef[cname].values+vf[cname].values)/(af[cname].values)
        if m == 3:
            r[cname] =  (hf[cname].values+ef[cname].values+vf[cname].values)-af[cname].values
        if m == 4:
            r[cname] = (hf[cname].values*ef[cname].values*vf[cname].values)-af[cname].values
        #w_r =  getW(np.array([hf[cname],ef[cname],vf[cname],af[cname]]).T)
        #r[cname] = np.power(hf[cname].values,w_r[0])*np.power(ef[cname].values,w_r[1])*np.power(vf[cname].values,w_r[2])/np.power(af[cname].values,w_r[3])
        #r[cname] = np.power(hf[cname].values,0.25)*np.power(ef[cname].values,0.25)*np.power(vf[cname].values,0.25)*np.power(1-af[cname].values,0.25)
        
    
 
    
    RF['h'] = hf
    RF['e'] = ef
    RF['v'] = vf
    RF['a'] = af
    
    return r,RF,W    
    
    

    
    
    
    
    
    
    