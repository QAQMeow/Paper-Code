# -*- coding: utf-8 -*-
"""
Created on Sat May 24 03:59:39 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import pickle
import cmaps
import numpy as np
import pandas as pd
from PIL import Image
import package.CalModule as cl
import package.PlotModule as pm
from matplotlib import rcParams
from scipy import stats
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
    
    location = 'bottom',pad=0.01,fraction=0.025,anchor=(0.5,0.3)
)
    cbar.outline.set_linewidth(0.5)
    cbar.set_label(y, labelpad = -20,fontsize = 6)
    #plt.savefig('../Figure/losses/'+SSPs[i]+term[j]+' economic productivity losses'+'.png', bbox_inches='tight',dpi = 300)
    #plt.show()
    
    

    
CEV1 = pd.DataFrame({"NAME":mergeID["NAME"],"CNAME":MID['location_name'],'GP':MID['Income_Group'],'E': np.nanmean(V['v1'],axis=0)})
#pm.PLotGlobal(CEH,'Global Hazard',0,0.45,'Hazard value', cmaps.MPL_OrRd)
CEV2 = pd.DataFrame({"NAME":mergeID["NAME"],"CNAME":MID['location_name'],'GP':MID['Income_Group'],'E': np.nanmean(V['v2'],axis=0)})
#pm.PLotGlobal(CEE,'Global Exposure',0,0.65,'Exposure value', cmaps.MPL_YlGnBu)
CEV3 = pd.DataFrame({"NAME":mergeID["NAME"],"CNAME":MID['location_name'],'GP':MID['Income_Group'],'E': 100*np.nanmean(V['v3'],axis=0)})
CEV4 = pd.DataFrame({"NAME":mergeID["NAME"],"CNAME":MID['location_name'],'GP':MID['Income_Group'],'E': np.nanmean(V['v4'],axis=0)})
    
    
width_cm = 18 # 设置图形宽度
height_cm =9 # 设置图形高度

# 将宽度和高度转换为英寸
width_inch = width_cm / 2.54
height_inch = height_cm / 2.54   
    
    
fig,axs = plt.subplots(nrows=2, ncols=2,figsize=(width_inch, height_inch),dpi=300)
fig.patch.set_facecolor('#FFFFFF')
 
PLotGlobal(axs[0,0],CEV1,'tobacco use(%)',0,56,'A', cmaps. MPL_RdPu)
 
PLotGlobal(axs[0,1],CEV2,'obesity(%)',0,72,'B', cmaps. MPL_RdPu)
 
PLotGlobal(axs[1,0],CEV3,'occupational pollution(%)',0.,25,'C', cmaps. MPL_RdPu)
PLotGlobal(axs[1,1],CEV4,'old population(%)',0.,30,'D', cmaps. MPL_RdPu)
 
fig.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.08,wspace=0.005,hspace=0.1)

plt.savefig('../res/Fig/fig_f/SI_fig6.png', dpi=300)