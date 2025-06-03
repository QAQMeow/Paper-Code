# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 11:17:21 2025

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
cg = ['高收入国家','中高等收入国家','中低等收入国家',  '低收入国家', ]
lab = ['High Income','Upper-middle Income','Lower-middle Income','Low Income']
lab = ['HICs','UMICs','LMICs','LICs']
mergeID = pd.merge(LID, CID,left_on ='location_name',right_on ='FCNAME',how = 'inner')  
MID = pd.merge(mergeID,Income_Group,left_on ='SOC',right_on ='Country Code',how = 'left')
R1,RF,W = cl.getHEVAR(H,E,V,A,1)
 
R2,RF,W = cl.getHEVAR(H,E,V,A,2)
 
R3,RF,W = cl.getHEVAR(H,E,V,A,3)
 
R4,RF,W = cl.getHEVAR(H,E,V,A,4)
 
r1 = pd.DataFrame(R1).T.astype(np.float32)
r1.columns = list(range(0,20,1))
r2 = pd.DataFrame(R2).T.astype(np.float32)
r2.columns = list(range(0,20,1))
r3 = pd.DataFrame(R3).T.astype(np.float32)
r3.columns = list(range(0,20,1))
r4 = pd.DataFrame(R4).T.astype(np.float32)
r4.columns = list(range(0,20,1))
Ax = pd.DataFrame({'NAME1':RF['a'].keys(),
                   'NAME':MID['NAME'],
                  'H':np.nanmean(RF['h'],axis=0),
                  'E':np.nanmean(RF['e'],axis=0),
                  'V':np.nanmean(RF['v'],axis=0),
                  'A':np.nanmean(np.array(RF['a'].values).astype(np.float32),axis=0),
                  'R':np.nanmean(np.array(r1.values).astype(np.float32),axis=1),})

# import seaborn as sns 
# import matplotlib.pyplot as plt
# fig ,ax = plt.subplots(figsize=(10,8),dpi=200); sns.kdeplot(data = R_C, 
#     common_norm=False, palette="viridis",
#    fill= True,alpha=0.1, linewidth=1, );
# ax.legend(np.flipud(np.arange(2001,2021)),ncol=4,loc="upper right",fontsize = 12, frameon=False);
# ax.set_xlabel('Risk value');ax.set_xlim([0,1])
     


r1x = stats.zscore(np.nanmean(r1.values ,axis=1))
r2x = stats.zscore(np.nanmean(r2.values ,axis=1))
r3x = stats.zscore(np.nanmean(r3.values ,axis=1))
r4x = stats.zscore(np.nanmean(r4.values ,axis=1))
rx = r1x#(r1x+r2x+r3x+r4x)/4
CER1 = pd.DataFrame({"NAME":mergeID["NAME"],"CNAME":mergeID["location_name"],'E': r1x})
CER2 = pd.DataFrame({"NAME":mergeID["NAME"],"CNAME":mergeID["location_name"],'E': r2x})
CER3 = pd.DataFrame({"NAME":mergeID["NAME"],"CNAME":mergeID["location_name"],'E': r3x})
CER4 = pd.DataFrame({"NAME":mergeID["NAME"],"CNAME":mergeID["location_name"],'E': r4x})
CERA = pd.DataFrame({"NAME":mergeID["NAME"],"CNAME":mergeID["location_name"],'E': rx,'GP':MID['Income_Group']})


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
                "font.size": 7,# 相当于小四大小
                "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
                "font.serif": ['Arial'],#宋体
                'axes.unicode_minus': False # 处理负号，即-号
             }
    rcParams.update(config)  

    #ax.set_title(lab)
    ax.text(0,1,lab,transform=ax.transAxes,fontsize = 7,fontname = 'Arial')
    #m = Basemap(projection='robin',resolution='l',lon_0=0)
    m = Basemap(ax = ax,projection = 'cyl',resolution='l',lon_0=0,llcrnrlon = -180, llcrnrlat = -60, urcrnrlon = 180, urcrnrlat = 90)
    m.fillcontinents(color = '#FFFFFF')


    #m.drawcoastlines() 
    m.readshapefile('../res/map/World_countries', 'World_countries',drawbounds=True,linewidth=  0.25)
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

    im2 = mpl.cm.ScalarMappable(norm=norm, cmap=c)
    cbar = plt.colorbar(im2, ax=ax, orientation='horizontal',
    extend='both',

    #label='Exposure of population to dry hot days(million)',
    #label='Population-weighted dry hot exposure days',
    label=lab,
    
    location = 'bottom',pad=0.01,fraction=0.025,anchor=(0.5,0.3)
)
    cbar.outline.set_linewidth(0.25)
    cbar.set_label(y, labelpad = -25)
    #plt.savefig('../Figure/losses/'+SSPs[i]+term[j]+' economic productivity losses'+'.png', bbox_inches='tight',dpi = 300)
    #plt.show()




from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib import rcParams
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Polygon as SPoly


width_cm = 18 # 设置图形宽度
height_cm = 9 # 设置图形高度

# 将宽度和高度转换为英寸
width_inch = width_cm / 2.54
height_inch = height_cm / 2.54
# 使用inch指定图形大小
fig,axs = plt.subplots(nrows=2, ncols=2,figsize=(width_inch, height_inch),dpi=300)
fig.patch.set_facecolor('#FFFFFF')
 

PLotGlobal(axs[0,0],CER1,'Global Risk',-1.5,1.5,'A. R = (H·E·V)/A', cmaps.MPL_summer+ cmaps.MPL_YlOrRd)
PLotGlobal(axs[0,1],CER2,'Global Risk',-1.5,1.5,'B. R = (H+E+V)/A',cmaps.MPL_summer+ cmaps.MPL_YlOrRd)
PLotGlobal(axs[1,0],CER3,'Global Risk',-1.5,1.5,'C. R = H+E+V-A', cmaps.MPL_summer+ cmaps.MPL_YlOrRd)
PLotGlobal(axs[1,1],CER4,'Global Risk',-1.5,1.5,'D. R = H·E·V-A', cmaps.MPL_summer+ cmaps.MPL_YlOrRd)

fig.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.01,wspace=0.005,hspace=0.02)


width_cm = 18# 设置图形宽度
height_cm = 8 # 设置图形高度

# 将宽度和高度转换为英寸
width_inch = width_cm / 2.54
height_inch = height_cm / 2.54
# 使用inch指定图形大小
fig,ax = plt.subplots(nrows=1, ncols=1,figsize=(width_inch, height_inch),dpi=300)
fig.patch.set_facecolor('#FFFFFF')
 

PLotGlobal(ax ,CERA,'Global Risk',-1.5,1.5,' ', cmaps.MPL_summer+ cmaps.MPL_YlOrRd)
 

fig.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.08,wspace=0.005,hspace=0.02)

plt.savefig('../res/Fig/fig_f/fig5.png', dpi=300)
#plt.savefig('../res/Fig/fig_f/fig5.pdf', dpi=300)

# 
# def rgb_to_hex(rgb):
#     # 使用format格式化字符串，将RGB三个分量转换为两位十六进制数，并拼接成完整的十六进制颜色字符串
#     return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

def Hex_to_RGB(hex,alpha):
    r = int(hex[1:3],16)
    g = int(hex[3:5],16)
    b = int(hex[5:7], 16)
    #rgb = str(r)+','+str(g)+','+str(b)
    rgb = [r/255,g/255,b/255,alpha]
    
    return rgb


def plot1(ax,data,lab,c,fl):
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import Normalize
    from matplotlib import rcParams
    import matplotlib as mpl
    from mpl_toolkits.basemap import Basemap
    from shapely.geometry import Polygon as SPoly
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle,Rectangle
    colors2 =  ['#ECF4E5','#B3D5B2','#7EBAB5','#387FAB']
    config = {
                "font.family": 'serif',
                "font.size":5,# 相当于小四大小
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
    m.readshapefile('../res/map/GlobalArea', 'GA',drawbounds=True,linewidth=0.25,color = '#808080')
    m.readshapefile('../res/map/World_countries', 'World_countries',drawbounds=False)
    #m.drawcountries()  
    #cbar = plt.colorbar(cs,location = 'bottom',pad=0.02,fraction=0.05)   
    #plt.title(i)
    df_poly = pd.DataFrame({
            'shapes': [Polygon(np.array(shape)) for shape in m.World_countries],
            'shapes2': [SPoly(np.array(shape)) for shape in m.World_countries],
            'len': [len(np.array(shape)) for shape in m.World_countries],
            'area': [area['NAME'] for area in m.World_countries_info]
        })

    for d in range(len(data)):
        colrnum = pd.DataFrame({'area':data[d]['NAME']})
        df_poly2 = df_poly.merge(colrnum)
        pc = PatchCollection(df_poly2.shapes, zorder=2,edgecolor='#000000',linewidth=0.25)
        pc.set_facecolor(c[d])

        ax.add_collection(pc)

    ax.text(0,1,fl,transform=ax.transAxes,fontsize = 7,fontname = 'Arial')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    for d in range(len(data)):     
        ax.add_patch(Rectangle((-27+d*27+2, -50),  # 圆心
                    25,8,  # 半径
                    facecolor=c[d],edgecolor = '#000000',linewidth = 0.25
                    ))
        ax.text(-27+d*25+3*d+2,-58,lab[d],fontsize = 5,color = '#000000')
        


def plot2(ax,data,lab,c,fl):
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import Normalize
    from matplotlib import rcParams
    import matplotlib as mpl
    from mpl_toolkits.basemap import Basemap
    from shapely.geometry import Polygon as SPoly
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle,Rectangle
    colors2 =  ['#ECF4E5','#B3D5B2','#7EBAB5','#387FAB']
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
    m.readshapefile('../res/map/GlobalArea', 'GA',drawbounds=True,linewidth=0.25,color = '#808080')
    m.readshapefile('../res/map/World_countries', 'World_countries',drawbounds=False)
    #m.drawcountries()  
    #cbar = plt.colorbar(cs,location = 'bottom',pad=0.02,fraction=0.05)   
    #plt.title(i)
    df_poly = pd.DataFrame({
            'shapes': [Polygon(np.array(shape)) for shape in m.World_countries],
            'shapes2': [SPoly(np.array(shape)) for shape in m.World_countries],
            'len': [len(np.array(shape)) for shape in m.World_countries],
            'area': [area['NAME'] for area in m.World_countries_info]
        })

    for d in range(len(data)):
        colrnum = pd.DataFrame({'area':data[d]['NAME']})
        df_poly2 = df_poly.merge(colrnum)
        pc = PatchCollection(df_poly2.shapes, zorder=2,edgecolor='#000000',linewidth=0.25)
        pc.set_facecolor(c[d])

        ax.add_collection(pc)

    ax.text(0,1,fl,transform=ax.transAxes,fontsize = 7,fontname = 'Arial')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    for d in range(len(data)):     
        ax.add_patch(Rectangle((-27+d*27+2, -50),  # 圆心
                    25,8,  # 半径
                    facecolor=c[d],edgecolor = '#000000',linewidth = 0.25
                    ))
        ax.text(-25+d*25+3*d+8-d,-58,lab[d],fontsize = 5,color = '#000000')



def plot3(ax,data,lab,c,fl):
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import Normalize
    from matplotlib import rcParams
    import matplotlib as mpl
    from mpl_toolkits.basemap import Basemap
    from shapely.geometry import Polygon as SPoly
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle,Rectangle
    colors2 =  ['#ECF4E5','#B3D5B2','#7EBAB5','#387FAB']
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
    m.readshapefile('../res/map/GlobalArea', 'GA',drawbounds=True,linewidth=0.25,color = '#808080')
    m.readshapefile('../res/map/World_countries', 'World_countries',drawbounds=False)
    #m.drawcountries()  
    #cbar = plt.colorbar(cs,location = 'bottom',pad=0.02,fraction=0.05)   
    #plt.title(i)
    df_poly = pd.DataFrame({
            'shapes': [Polygon(np.array(shape)) for shape in m.World_countries],
            'shapes2': [SPoly(np.array(shape)) for shape in m.World_countries],
            'len': [len(np.array(shape)) for shape in m.World_countries],
            'area': [area['NAME'] for area in m.World_countries_info]
        })

    for d in range(len(data)):
        colrnum = pd.DataFrame({'area':data[d]['NAME']})
        df_poly2 = df_poly.merge(colrnum)
        pc = PatchCollection(df_poly2.shapes, zorder=2,edgecolor='#000000',linewidth=0.25)
        pc.set_facecolor(c[d])

        ax.add_collection(pc)

    ax.text(0,1,fl,transform=ax.transAxes,fontsize = 7,fontname = 'Arial')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    for d in range(len(data)):     
        ax.add_patch(Rectangle((-27+d*27+2, -50),  # 圆心
                    25,8,  # 半径
                    facecolor=c[d],edgecolor = '#000000',linewidth = 0.25
                    ))
        ax.text(-25+d*25+3*d+12-d,-58,lab[d],fontsize = 5,color = '#000000')

hp60 = np.percentile(Ax['H'],60)
ep60 = np.percentile(Ax['E'],60)
vp60 = np.percentile(Ax['V'],60)
ap40 = np.percentile(Ax['A'],40)
rp60 = np.percentile(Ax['R'],60)

#H-E-V-A
A1 = Ax[(Ax['H']>=hp60)&(Ax['E']>=ep60)&(Ax['V']>=vp60)&(Ax['A']<=ap40)&(Ax['R']>=rp60)]


#H-E-V
A2 = Ax[(Ax['H']>=hp60)&(Ax['E']>=ep60)&(Ax['V']>=vp60)&(Ax['R']>=rp60)]
idx = A1.index
for i in idx:
    A2=A2.drop(index=i)
    print(idx)
#H-E-A
A3 = Ax[(Ax['H']>=hp60)&(Ax['E']>=ep60)&(Ax['A']<=ap40)&(Ax['R']>=rp60)]
idx = A1.index
for i in idx:
    A3=A3.drop(index=i)
    print(idx)
#H-V-A
A4 = Ax[(Ax['H']>=hp60)&(Ax['V']>=vp60)&(Ax['A']<=ap40)&(Ax['R']>=rp60)]
idx = A1.index
for i in idx:
    A4=A4.drop(index=i)
    print(idx)
#E-V-A
A5 = Ax[(Ax['E']>=ep60)&(Ax['V']>=vp60)&(Ax['A']<=ap40)&(Ax['R']>=rp60)]
idx = A1.index
for i in idx:
    A5=A5.drop(index=i)
    print(idx)

#H-E
A6 = Ax[(Ax['H']>=hp60)&(Ax['E']>=ep60)&(Ax['R']>=rp60)]
idx = np.unique(np.concatenate([A1.index, A2.index, A3.index]))
 
for i in idx:
    A6=A6.drop(index=i)
    print(idx)
    
#H-V
A7 = Ax[(Ax['H']>=hp60)&(Ax['V']>=vp60)&(Ax['R']>=rp60)]
idx = np.unique(np.concatenate([A1.index, A2.index,A4.index]))
for i in idx:
    A7=A7.drop(index=i)
    print(idx)
    
#H-A
A8 = Ax[(Ax['H']>=hp60)&(Ax['A']<=ap40)&(Ax['R']>=rp60)]
idx = np.unique(np.concatenate([A1.index, A3.index, A4.index]))
for i in idx:
    A8=A8.drop(index=i)
    print(idx)
    
#E-V
A9 = Ax[(Ax['E']>=ep60)&(Ax['V']>=vp60)&(Ax['R']>=rp60)]
idx = np.unique(np.concatenate([A1.index,A2.index, A5.index,]))
for i in idx:
    A9=A9.drop(index=i)
    print(idx)
    
#E-A
A10 = Ax[(Ax['E']>=ep60)&(Ax['A']<=ap40)&(Ax['R']>=rp60)]
idx =np.unique(np.concatenate([A1.index,A3.index, A5.index,]))
for i in idx:
    A10=A10.drop(index=i)
    print(idx)

#V-A
A11 = Ax[(Ax['V']>=vp60)&(Ax['A']<=ap40)&(Ax['R']>=rp60)]
idx = np.unique(np.concatenate([A1.index, A4.index,A5.index]))
for i in idx:
    A11=A11.drop(index=i)
    print(idx)




#H
A12 = Ax[(Ax['H']>=hp60)&(Ax['R']>=rp60)]
idx = np.unique(np.concatenate([A1.index, A2.index, A3.index, A4.index, A6.index, A7.index, A8.index]))
for i in idx:
    A12=A12.drop(index=i)
    print(idx)

#E
A13 = Ax[(Ax['E']>=ep60)&(Ax['R']>=rp60)]
idx = np.unique(np.concatenate([A1.index, A2.index, A3.index, A5.index, A6.index, A9.index,A10.index]))
for i in idx:
    A13=A13.drop(index=i)
    print(idx)

#V
A14 = Ax[(Ax['V']>=vp60)&(Ax['R']>=rp60)]
idx = np.unique(np.concatenate([A1.index, A2.index, A4.index, A5.index, A7.index, A9.index,A11.index]))
for i in idx:
    A14=A14.drop(index=i)
    print(idx)
    
#A
A15 = Ax[(Ax['A']<=ap40)&(Ax['R']>=rp60)]
idx = np.unique(np.concatenate([A1.index, A3.index, A4.index, A5.index, A8.index, A10.index,A11.index]))
for i in idx:
    A15=A15.drop(index=i)
    print(idx)

d0 = [A1]
l0 = ['H-E-V-A']
d1 = [A2,A3,A4,A5]
l1 = ['H-E-V','H-E-A','H-V-A','E-V-A']
d2 = [A6,A7,A8,A9,A10,A11]
l2 = ['H-E','H-V','H-A','E-V','E-A','V-A']
d3 = [A12,A13,A14,A15]
l3 =['H','E','V','A']
config = {
                "font.family": 'serif',
                "font.size": 5,# 相当于小四大小
                "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
                "font.serif": ['Arial'],#宋体
                'axes.unicode_minus': False # 处理负号，即-号
             }
rcParams.update(config)  

width_cm = 18 # 设置图形宽度
height_cm = 9 # 设置图形高度

# 将宽度和高度转换为英寸
width_inch = width_cm / 2.54
height_inch = height_cm / 2.54
# 使用inch指定图形大小
fig,axs = plt.subplots(nrows=2, ncols=2,figsize=(width_inch, height_inch),dpi=300)
fig.patch.set_facecolor('#FFFFFF')
plot1(axs[0,0],d0,l0,['#C6DCB9'],'A')

plot1(axs[0,1],d1,l1,['#B6E2DC','#FBE3CD','#C6C3E1','#FAA09C'],'B')

plot2(axs[1,0],d2,l2,['#8DECF5','#E9E4AF','#CEBAF0','#C6DCB9','#F9C89B','#90D0C2'],'C')

plot3(axs[1,1],d3,l3,['#DCA0DD','#A38277','#F1B543','#6E86A5'],'D')
 
 
fig.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.01,wspace=0.005,hspace=0.02)

plt.savefig('../res/Fig/fig_f/fig6.png', dpi=300)
#plt.savefig('../res/Fig/fig_f/fig6.pdf', dpi=300)