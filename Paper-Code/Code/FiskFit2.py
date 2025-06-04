# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 19:33:25 2023

@author: olcxy
"""

def FiskFit2(data):
    import numpy as np
    from math import gamma
    #由于原始数据序列中可能存在负值,所以采用了3 参数的 log-logistic 概率分布。
    #详见GBT 20481-2017《气象干旱等级》中SPEI计算方法log-logistic 概率分布计算部分
    N = len(data)
   #计算w0,w1,w2
   
    #F = np.zeros(N)
    rk = np.array(range(1,N+1))
    r = np.array([np.sort(data),rk[np.argsort(data)],rk]).T
    r2 = r[np.argsort(r[:,1])]
    F = (r2[:,2]-0.35)/N
    

    w0 = np.mean(np.power(1-F,0)*data)
    w1 = np.mean(np.power(1-F,1)*data)
    w2 = np.mean(np.power(1-F,2)*data)
#计算参数
    beta=(2*w1-w0)/(6*w1-w0-6*w2);
    
    g1=gamma(1+1/beta);
    g2=gamma(1-1/beta);
    
    alpha=(w0-2*w1)*beta/(g1*g2);
    
    gamm=w0-alpha*g1*g2;
    
    #logLogisticCDF = 1/(1+np.power(alpha/(data-gamm),beta))
    return alpha,beta,gamm



