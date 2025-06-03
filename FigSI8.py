# -*- coding: utf-8 -*-
"""
Created on Sat May 24 12:26:36 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""

 
 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import pickle
import cmaps
from PIL import Image
import package.CalModule as cl
import package.PlotModule as pm
from matplotlib import rcParams

with open('../res/dataprced/H_Factor.pkl', 'rb') as f:  # 读取pickle文件
    H = joblib.load(f)
    f.close()
with open('../res/dataprced/E_Factor.pkl', 'rb') as f:  # 读取pickle文件
    E = joblib.load(f)
    f.close()
with open('../res/dataprced/V_Factor.pkl', 'rb') as f:  # 读取pickle文件
    V = joblib.load(f)
    f.close()
with open('../res/dataprced/A_Factor.pkl', 'rb') as f:  # 读取pickle文件
    A = joblib.load(f)
    f.close()

Income_Group = pd.read_csv('../res/SEdata/Income_Group.csv')


LID = pd.read_csv('../res/NCDs/Location_ID.csv')
CID = pd.read_excel('../res/other/countriesID.xls') 
mergeID = pd.merge(LID, CID,left_on ='location_name',right_on ='FCNAME',how = 'inner')  
MID = pd.merge(mergeID,Income_Group,left_on ='SOC',right_on ='Country Code',how = 'left')


R_C = {}
W = {}

# def getW(X):
#     X = np.float32(X)
#     sigma = np.nanstd(X , axis=0)
#     corr = np.corrcoef(X .T)
#     C = sigma * np.nansum(1 - np.abs(corr), axis=0)
#     weights = C / np.nansum(C)
#     return weights

def getW(data):
    from sklearn.decomposition import PCA
    data = np.array(data).T
    pca = PCA(n_components= 1)
    pca.fit(data)
    return pca.explained_variance_ratio_[0]

Income_Group = pd.read_csv('../res/SEdata/Income_Group.csv')

mergeID = pd.merge(LID, CID,left_on ='location_name',right_on ='FCNAME',how = 'inner')  
MID = pd.merge(mergeID,Income_Group,left_on ='SOC',right_on ='Country Code',how = 'left')


cg = ['高收入国家','中高等收入国家','中低等收入国家',  '低收入国家', ]
lab = ['HICs','UMICs','LMICs','LICs']
colors = ['#8983BF','#82B0D2','#FFBE7A','#FF8884']
colors = ['#C72228','#F98F34','#6B98C4','#0C4E9B']
colors = cmaps. MPL_YlGnBu._colors[np.int16(np.linspace(20,110,4))]
colors =  [cl.rgb_to_hex([140,42,115]),cl.rgb_to_hex([223,74,104]),cl.rgb_to_hex([252,154,107]),cl.rgb_to_hex([255,183,3])]
lab2 = ['duration','frequency','intensity']

def example_plot2(ax,RData,tit,ylab,i,j,yl):
    from scipy.stats import sem
    X = np.arange(2001,2021)
    Y2 = []
    yr = []
    
    dd  = RData[RData['Income_Group']==cg[i]]
    A = dd.values[:,2:]
    A = A.astype(np.float32)
    yerr = sem(A,axis=0)
     
    mx = np.nanmean(A,axis=0)
   
    import statsmodels.api as sm
    X1 = sm.add_constant(X)
    res = sm.OLS(mx, X1).fit()
    slope = np.round(res.params[1],5)
    pval = res.pvalues[1]
    pred_ols = res.get_prediction()
    iv_l = pred_ols.summary_frame()["obs_ci_lower"]
    iv_u = pred_ols.summary_frame()["obs_ci_upper"]
    ax.fill_between(X,iv_l,iv_u,alpha = 0.15,color = colors[i],edgecolor = 'none')
     
    #ax.plot(X,mx,marker='.',linestyle='--',color = colors[i],label = lab[i])
        #ax.scatter(X,Y2[g,:],marker='.',color = colors[g],label = lab[g])
    ax.errorbar(X,mx,yerr, markersize=1,fmt='o', linewidth=0.4, capsize=0.2,
               color = colors[i],label = lab[i])
    
    ax.plot(X, res.fittedvalues,linewidth=0.8,markersize = 1, label="Global trend",color = colors[i])
    #ax.boxplot(A)
    ax.set_xlim([2001,2020])
    ax.set_ylim(yl)
    #ax.legend(facecolor='none',edgecolor='none',fontsize = 8,ncol=1,loc = 'upper left')
    ax.set_xticks(ticks= [2001,2010,2020] ,labels = [2001,2010,2020] ,fontsize=5)
    ax.tick_params(axis='both', direction='in')
    plt.setp(ax.get_yticklabels(), fontsize=5)
    if j==0:
        ax.set_title(tit,fontsize = 5)
        
    if i==0:
        ax.set_ylabel(ylab,fontsize = 5)
    ax.text(0.05, 0.85, 'slope = '+str(slope), transform=ax.transAxes)
    
    if pval<0.05:
        ax.text(0.05, 0.7, 'p <0.05 ', transform=ax.transAxes)
    #print(res.pvalues)
    lw  = .6
    ax.spines['top'].set_linewidth(lw)
    ax.spines['bottom'].set_linewidth(lw)
    ax.spines['left'].set_linewidth(lw)
    ax.spines['right'].set_linewidth(lw)


def plot5(ax,RData,yl,i,j,colors):
    from scipy.stats import sem
    import statsmodels.api as sm
    X = np.arange(2001,2021)
    Y2 = []
    yr = []
    
    dd  = RData
    A = dd.values[:,2:]
    A = A.astype(np.float32)
    yerr = sem(A,axis=0)
     
    mx = np.nanmean(A,axis=0)
   
   
    X1 = sm.add_constant(X)
    res = sm.OLS(mx, X1).fit()
    slope = np.round(res.params[1],5)
    pval = res.pvalues[1]
    pred_ols = res.get_prediction()
    iv_l = pred_ols.summary_frame()["obs_ci_lower"]
    iv_u = pred_ols.summary_frame()["obs_ci_upper"]
    ax.fill_between(X,iv_l,iv_u,alpha = 0.15,color = '#9C9C9C',edgecolor = 'none')
     
    #ax.plot(X,mx,marker='.',linestyle='--',color = colors[i],label = lab[i])
        #ax.scatter(X,Y2[g,:],marker='.',color = colors[g],label = lab[g])
    ax.errorbar(X,mx,yerr, markersize=1,fmt='o', linewidth=0.4, capsize=0.2,
               color = '#9C9C9C')
    
    ax.plot(X, res.fittedvalues,linewidth=0.8,markersize = 1, label="Global trend",color = 'black')
    #ax.boxplot(A)
    ax.set_xlim([2001,2020])
    ax.set_ylim(yl)
    #ax.legend(facecolor='none',edgecolor='none',fontsize = 8,ncol=1,loc = 'upper left')
    ax.set_xticks(ticks= [2001,2010,2020] ,labels = [2001,2010,2020] ,fontsize=5)
    ax.tick_params(axis='both', direction='in')
    plt.setp(ax.get_yticklabels(), fontsize=5)
    if j==0:
        ax.set_title('Global',fontsize = 5)
        
    
    ax.text(0.05, 0.85, 'slope = '+str(slope), transform=ax.transAxes)
    
    if pval<0.05:
        ax.text(0.05, 0.7, 'p <0.05 ', transform=ax.transAxes)
    #print(res.pvalues)
    lw  = .6
    ax.spines['top'].set_linewidth(lw)
    ax.spines['bottom'].set_linewidth(lw)
    ax.spines['left'].set_linewidth(lw)
    ax.spines['right'].set_linewidth(lw)

AData = [H['h1'].T,H['h2'].T,H['h3'].T]
RD =[]
for j in range(3):
    RData = {}
    RData['location_name'] = MID['location_name']
    RData['Income_Group'] = MID['Income_Group']
    for i in list(range(2001,2021)):
        RData[str(i)] =AData[j][i-2001].values
    
    RData = pd.DataFrame(RData)
    RD.append(RData)


width_cm =12 # 设置图形宽度
height_cm = 7# 设置图形高度

# 将宽度和高度转换为英寸
width_inch = width_cm / 2.54
height_inch = height_cm / 2.54

fig, axs = plt.subplots(ncols=5, nrows=3, figsize=(width_inch , height_inch ),
                      layout="constrained",dpi = 300)
config = {
            "font.family": 'serif',
            "font.size": 5,# 相当于小四大小
            "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            "font.serif": ['Arial'],#宋体
            'axes.unicode_minus': False # 处理负号，即-号
         }
rcParams.update(config)  
fig.patch.set_facecolor('#FFFFFF')
YL = np.array([[-0.01,20],[-0.01,3],[0,20]],)
for col in range(5):
    for row in range(3):
        if col <4:
            example_plot2(axs[row, col],RD[row],lab[col],lab2[row],col,row,YL[row])
        else:
            plot5(axs[row,col],RD[row],YL[row],col,row,colors)
plt.rc('axes',linewidth = 0.5 )
#fig.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.08,wspace=0.002,hspace=0.05)

 
# Data = V['v2'].T
# Data.insert(0,'Group',MID['Income_Group'].values)
# Data.insert(0,'Code',MID['FCNAME'].values)

# for i in range(4):
#     d = Data[Data['Group']==cg[i]].values[:,2:]
#     dm = np.nanmean(d,axis=0)
#     plt.plot(dm,label = lab[i])
# plt.legend()
