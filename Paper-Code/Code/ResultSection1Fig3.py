# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 01:58:51 2025

@author: Meovv Van

email: 1259053332@qq.com

"""

 
 

import joblib
import pickle
import cmaps
import numpy as np
import pandas as pd
from PIL import Image
import package.CalModule as cl
import package.PlotModule as pm
from matplotlib import rcParams
import matplotlib.pyplot as plt
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
    
LID = pd.read_csv('../res/NCDs/Location_ID.csv')
CID = pd.read_excel('../res/other/countriesID.xls')
R_C = {}
W = {}

def getW(X):
    X = np.float32(X)
    sigma = np.nanstd(X , axis=0)
    corr = np.corrcoef(X .T)
    C = sigma * np.nansum(1 - np.abs(corr), axis=0)
    weights = C / np.nansum(C)
    return weights

# def getW(data):
#     from sklearn.decomposition import PCA
#     data = np.array(data).T
#     pca = PCA(n_components= 1)
#     pca.fit(data)
#     return pca.explained_variance_ratio_[0]

Income_Group = pd.read_csv('../res/SEdata/Income_Group.csv')

mergeID = pd.merge(LID, CID,left_on ='location_name',right_on ='FCNAME',how = 'inner')  
MID = pd.merge(mergeID,Income_Group,left_on ='SOC',right_on ='Country Code',how = 'left')
r,RF,W  = cl.getHEVAR(H,E,V,A,1)
R_C = pd.DataFrame(r).T
R_C.columns = list(range(0,20,1))
# import seaborn as sns 
# import matplotlib.pyplot as plt
# fig ,ax = plt.subplots(figsize=(10,8),dpi=200); sns.kdeplot(data = R_C, 
#     common_norm=False, palette="viridis",
#    fill= True,alpha=0.1, linewidth=1, );
# ax.legend(np.flipud(np.arange(2001,2021)),ncol=4,loc="upper right",fontsize = 12, frameon=False);
# ax.set_xlabel('Risk value');ax.set_xlim([0,1])
     


r = np.nanmean(np.array(R_C.values).astype(np.float32),axis=1)
r2 = cl.classDa(r,7)
r[np.isinf(r)]= np.nan
r2[np.isnan(r)] = np.nan
Ax = pd.DataFrame({'NAME':RF['a'].keys(),
                   'ENAME':MID['FENAME'],
                   'Income_Group':MID['Income_Group'],
                  'H':np.nanmean(RF['h'],axis=0),
                  'E':np.nanmean(RF['e'],axis=0),
                  'V':np.nanmean(RF['v'],axis=0),
                  'A':np.nanmean(np.array(RF['a'].values).astype(np.float32),axis=0),
                  'R':np.nanmean(np.array(R_C.values).astype(np.float32),axis=1),})
 
CEH = pd.DataFrame({"NAME":mergeID["NAME"],"CNAME":MID['location_name'],'GP':MID['Income_Group'],'E': Ax['H']})
#pm.PLotGlobal(CEH,'Global Hazard',0,0.45,'Hazard value', cmaps.MPL_OrRd)
CEE = pd.DataFrame({"NAME":mergeID["NAME"],"CNAME":MID['location_name'],'GP':MID['Income_Group'],'E': Ax['E']})
#pm.PLotGlobal(CEE,'Global Exposure',0,0.65,'Exposure value', cmaps.MPL_YlGnBu)
CEV = pd.DataFrame({"NAME":mergeID["NAME"],"CNAME":MID['location_name'],'GP':MID['Income_Group'],'E': Ax['V']})
#pm.PLotGlobal(CEV,'Global Vulnerability',0,0.87,'Vulnerability value', cmaps.MPL_PuRd)
CEA = pd.DataFrame({"NAME":mergeID["NAME"],"CNAME":MID['location_name'],'GP':MID['Income_Group'],'E': Ax['A']})
#pm.PLotGlobal(CEA,'Global Adaptability',0,0.92,'Adaptability value', cmaps.WhiteBlue)
CER = pd.DataFrame({"NAME":mergeID["NAME"],"CNAME":MID['location_name'],'GP':MID['Income_Group'],'E': r})
#pm.PLotGlobal(CER,'Global Risk',0,0.45,'Risk value', cmaps.MPL_BuPu_r+cmaps.MPL_OrRd)
 
CEXX = pd.DataFrame({"NAME":mergeID["NAME"],'E': np.nanmean(V['v1'],axis=0)})


def PLotGlobal(ax,CE,y,dmin,dmax,lab,c=cmaps.NCV_jaisnd):
    '''
    

    Parameters
    ----------
    CE : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    c : TYPE, optional
        DESCRIPTION. The default is cmaps.NCV_jaisnd.

    Returns
    -------
    None.

    '''
    #c = cmaps.WhiteYellowOrangeRed
    #c = cmaps.MPL_GnBu
    #c = cmaps.MPL_BuPu
    #c = cmaps.MPL_RdPu
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import Normalize
    from matplotlib import rcParams
    import matplotlib as mpl
    from mpl_toolkits.basemap import Basemap
    config = {
                "font.family": 'serif',
                "font.size": 5,# 相当于小四大小
                "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
                "font.serif": ['Arial'],#宋体
                'axes.unicode_minus': False # 处理负号，即-号
             }
    rcParams.update(config)  

    
    
    #plt.title(y)
    #m = Basemap(projection='robin',resolution='l',lon_0=0)
    m = Basemap(ax = ax,projection = 'cyl',resolution='l',lon_0=0,llcrnrlon = -180, llcrnrlat = -60, urcrnrlon = 180, urcrnrlat = 90)
    m.fillcontinents(color = '#FFFFFF')


    #m.drawcoastlines() 
    m.readshapefile('../res/map/World_countries', 'World_countries',drawbounds=True,linewidth = 0.25)
    #m.drawcountries()  
    #cbar = plt.colorbar(cs,location = 'bottom',pad=0.02,fraction=0.05)   
    #plt.title(i)
    df_poly = pd.DataFrame({
            'shapes': [Polygon(np.array(shape)) for shape in m.World_countries],
            'area': [area['NAME'] for area in m.World_countries_info]
        })

    colrnum = pd.DataFrame({'area':CE['NAME'],'L':CE['E']})
    df_poly2 = df_poly.merge(colrnum)
    norm = Normalize(vmin=dmin,vmax = dmax)
    pc = PatchCollection(df_poly2.shapes, zorder=2)
    pc.set_facecolor(c(norm(df_poly2['L'].fillna(0).values)))
    ax.add_collection(pc)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.text(0,1,lab,transform=ax.transAxes,fontsize = 7,fontname = 'Arial')
    im2 = mpl.cm.ScalarMappable(norm=norm, cmap=c)
    cbar = plt.colorbar(im2, ax=ax, orientation='horizontal',
    extend='max',

    #label='Exposure of population to dry hot days(million)',
    #label='Population-weighted dry hot exposure days',
    label=lab,
    
    location = 'bottom',pad=0.01,fraction=0.025,anchor=(0.1,0.2)
)
    cbar.outline.set_linewidth(0.5)
    cbar.set_label(y, labelpad = -20,fontsize = 6)
    #plt.savefig('../Figure/losses/'+SSPs[i]+term[j]+' economic productivity losses'+'.png', bbox_inches='tight',dpi = 300)
    #plt.show()


from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib import rcParams
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Polygon as SPoly
config = {
            "font.family": 'serif',
            "font.size": 5,# 相当于小四大小
            "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            "font.serif": ['Arial'],#宋体
            'axes.unicode_minus': False # 处理负号，即-号
         }
rcParams.update(config)  

width_cm = 18 # 设置图形宽度
height_cm = 10 # 设置图形高度

# 将宽度和高度转换为英寸
width_inch = width_cm / 2.54
height_inch = height_cm / 2.54
# 使用inch指定图形大小
fig,axs = plt.subplots(nrows=2, ncols=2,figsize=(width_inch, height_inch),dpi=300)
fig.patch.set_facecolor('#FFFFFF')
 
PLotGlobal(axs[0,0],CEH,'Hazard',0,.3,'A', cmaps.MPL_OrRd)
 
PLotGlobal(axs[0,1],CEE,'Exposure',0,0.4,'B', cmaps.MPL_YlGnBu)
 
PLotGlobal(axs[1,0],CEV,'Vulnerability',0.58,2,'C', cmaps.MPL_PuRd)
 
PLotGlobal(axs[1,1],CEA,'Adaptability',0.35,3.6,'D', cmaps.WhiteBlue)
 
fig.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.08,wspace=0.005,hspace=0.05)


plt.savefig('../res/Fig/fig_f/fig4.png', dpi=300)
#plt.savefig('../res/Fig/fig_f/fig4.pdf', dpi=300)

# fig,axs = plt.subplots(nrows=2, ncols=2,figsize=(width_inch, height_inch),dpi=300)
# fig.patch.set_facecolor('#FFFFFF')
# PLotGlobal(axs[0,0],CEH,'Hazard',0,.25,'Hazard value', cmaps.MPL_OrRd)
 
# PLotGlobal(axs[0,1],CEE,'Exposure',0,.65,'Exposure value', cmaps.MPL_YlGnBu)
 
# PLotGlobal(axs[1,0],CEV,'Vulnerability',0,0.8,'Vulnerability value', cmaps.MPL_PuRd)
 
# PLotGlobal(axs[1,1],CEA,'Adaptability',0,1,'Adaptability value', cmaps.WhiteBlue)
# fig.subplots_adjust(left=0,right=1,top=1,bottom=0,
#                     wspace=0.02,hspace=0.02)



def rgb_to_hex(rgb):
    # 使用format格式化字符串，将RGB三个分量转换为两位十六进制数，并拼接成完整的十六进制颜色字符串
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

# CER = pd.DataFrame({"NAME":mergeID["NAME"],'E': r})
# # colors =[]
# # c1 = cmaps.MPL_GnBu_r._colors[np.int16(np.linspace(0,127,4))]
# # for i in range(4):
# #     colors.append(rgb_to_hex(np.int16(c1[i]*255)))
# #colors = cmaps.MPL_YlGnBu._colors[np.int16(np.linspace(0,127,4))]
# colors = ['#8983BF','#82B0D2','#FFBE7A','#FF8884']
# cg = ['高收入国家','中高等收入国家','中低等收入国家',  '低收入国家', ]
# lab = ['High Income','Upper-middle Income','Lower-middle Income','Low Income']
# def pl(ax,data,CE,colors,tit):


#     ET  = pd.DataFrame(data.T);ET["NAME"] = mergeID["NAME"].values;ET['Income_Group'] = MID['Income_Group'].values
#     ET['SOC'] = mergeID['SOC'].values
#     cl = ['#FFFFFF']
#     cl  = np.array(cl*len(ET['Income_Group']))
#     for i in range(4):
#         cl[ET['Income_Group'].values==cg[i]] = colors [i]
#     ET['colors'] = cl        
#     ET2 = pd.merge(ET,CE,left_on ='NAME',right_on ='NAME')
#     ETSR = ET2.sort_values('E',ascending=False)        
#     dd = ETSR.iloc[:20,:].sort_values('E',ascending=True)        
    

#     bplot = ax.boxplot(dd.iloc[:,:20].T,vert = False, patch_artist=True,  # fill with color
#                        tick_labels =dd.iloc[:,22] ,showbox = True,boxprops = dict(linestyle='--', linewidth=2, color='#000000'),
# whiskerprops = dict(linestyle='--', linewidth=2, color='#000000'),capprops = dict(linewidth=2, color='#000000'))
#     cols = dd.iloc[:,23].astype(str).values
#     for patch, color in zip(bplot['boxes'], cols):
#         patch.set_facecolor(color)

#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.spines['bottom'].set_linewidth(2)
#     ax.spines['left'].set_linewidth(2)
#     ax.tick_params(axis='both',labelsize=25)
#     ax.set_title(tit)
#     #ax.spines['left'].set_visible(False)
#     #ax.spines['bottom'].set_visible(False)
    
#     #plt.show()

# fig,axs = plt.subplots(nrows=1, ncols=5,figsize=(18,20),dpi=300)   
# from matplotlib import rcParams
# config = {
#             "font.family": 'serif',
#             "font.size": 25,# 相当于小四大小
#             "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
#             "font.serif": ['Times New Roman'],#宋体
#             'axes.unicode_minus': False # 处理负号，即-号
#          }
# rcParams.update(config)

# pl(axs[0],h,CEH,colors,'Hazard')
# pl(axs[1],e,CEE,colors,'Exposure')
# pl(axs[2],v,CEV,colors,'Vulnerability')
# pl(axs[3],a,CEA,colors,'Adaptability')

# pl(axs[4],R_C.T,CER,colors,'Risk')
# testx = np.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]])
# ax4 = plt.axes([1.25, 0.6, 0, 0])
# bp = ax4.boxplot(testx.T,showbox = True, patch_artist=True,);
# for patch, color in zip(bp['boxes'], colors):
#     patch.set_facecolor(color)
# lg = ax4.legend(lab,frameon=True,fontsize = 20)
# lg.get_frame().set_linewidth(2.5)
# lg.get_frame().set_edgecolor('black')
# plt.axis('off')

# fig.subplots_adjust(left=0,right=1,top=1,bottom=0,
#                     wspace=0.5,hspace=0.1)
# plt.show()

