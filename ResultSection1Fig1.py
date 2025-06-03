# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 15:40:25 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tifffile as tf
import joblib
import cmaps
from PIL import Image
from matplotlib.colors import ListedColormap
import package.CalModule as cl
import pymannkendall as mk
with open('../res/other/Mask.pkl', 'rb') as f:  # 读取pickle文件
    Maskdata = joblib.load(f)
    f.close()
Mask = Maskdata['mask']
lat = Maskdata['lat']
lon = Maskdata['lon']
Mask2 = cl.fliplrMap(Mask)

with open('../res/dataprced/CDHEs_data.pkl', 'rb') as f:  # 读取pickle文件
    CDHEs_data = joblib.load(f)
    f.close()


CDHAs = CDHEs_data['CDHE_area']
CDHDs = CDHEs_data['CDHE_days']
CDHFs = CDHEs_data['CDHE_frequence']
CDHIs = CDHEs_data['CDHE_intensity']
 

CFm = np.nanmean(CDHFs ,axis=0)
CFm = CFm*Mask2
CDm = np.nanmean(CDHDs ,axis=0)
CDm = CDm*Mask2
CIm = np.nanmean(CDHIs ,axis=0)
CIm = CIm*Mask2


Years = np.arange(2001,2021)
def getSP(data,Mask):
    Slopes = np.nan*np.zeros_like(Mask)
    Pva = np.nan*np.zeros_like(Mask)
    for i in range(600):
        for j in range(1440):
            if(Mask[i,j]==1):
                x = data[:,i,j]
                res = mk.original_test(x)
                Slopes[i,j] = res.slope
                Pva[i,j] = res.p
    return Slopes,Pva
 
sD,pD =  getSP(CDHDs, Mask2)
sF,pF =  getSP(CDHFs, Mask2)
sI,pI =  getSP(CDHIs, Mask2)



def plot1(fi,ax,p1,lab,lon,lat,c):

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


    ##ax.set_title(lab1[i]+' '+lab2[j])
    #m = Basemap(projection='robin',resolution='l',lon_0=0)
    m = Basemap(ax = ax,projection = 'cyl',resolution='l',lon_0=0,llcrnrlon = -180, llcrnrlat = -60, urcrnrlon = 180, urcrnrlat = 90)
    m.fillcontinents(color = '#FFFFFF')



    #m.drawcoastlines() 
    m.readshapefile('../res/map/GlobalArea', 'World_countries',drawbounds=True,linewidth=0.25)
    LON,LAT = np.meshgrid(lon,lat)
    Lon = LON-180 
    xi, yi = m(Lon, LAT)
    FL = ['A','C','E']
    ax.text(0,1,FL[fi],transform=ax.transAxes,fontsize = 7,fontname = 'Arial')
    cs = m.pcolormesh(xi, yi, p1, cmap=c)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    from matplotlib.colors import Normalize
    norm = Normalize(vmin=np.nanmin(p1),vmax = np.nanmax(p1))
    im2 = mpl.cm.ScalarMappable(norm=norm, cmap=c)
    cbar = plt.colorbar(im2, ax=ax, orientation='horizontal',
    extend='max',

    location = 'bottom',pad=0.01,fraction=0.03,anchor=(0.55,-1)
    )
    cbar.outline.set_linewidth(0.5)
    cbar.set_label(lab, labelpad = -21,fontsize = 5)



def plot2(fi,ax,s,p2,lab,lon,lat,c):

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

    m = Basemap(ax = ax,projection = 'cyl',resolution='l',lon_0=0,llcrnrlon = -180, llcrnrlat = -60, urcrnrlon = 180, urcrnrlat = 90)
    m.fillcontinents(color = '#FFFFFF')



    #m.drawcoastlines() 
    m.readshapefile('../res/map/GlobalArea', 'World_countries',drawbounds=True,linewidth=0.25)
    LON,LAT = np.meshgrid(lon,lat)
    Lon = LON-180 
    xi, yi = m(Lon, LAT)

    FL = ['B','D','F']
    ax.text(0,1,FL[fi],transform=ax.transAxes,fontsize = 7,fontname = 'Arial')
    
    cs = m.pcolormesh(xi, yi, s,cmap=c,vmin=np.nanmin(s),vmax = -np.nanmin(s))
    p2[p2>0.05] = np.nan
    p22 = p2.copy()

    p22[s<=0] = np.nan
    p23 = p2.copy()
    p23[s>0] = np.nan

    ax.contourf(xi, yi,p22,10, hatches=['////'],colors='none')
    ax.contourf(xi, yi,p23,10, hatches=['\\\\'],colors='none')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    from matplotlib.colors import Normalize
    norm = Normalize(vmin=np.nanmin(s),vmax = -np.nanmin(s))
    im2 = mpl.cm.ScalarMappable(norm=norm, cmap=c)
    cbar = plt.colorbar(im2, ax=ax, orientation='horizontal',
    extend='both',

    location = 'bottom',pad=0.01,fraction=0.03,anchor=(0.55,-1)
    )
    cbar.outline.set_linewidth(0.5)

    cbar.set_label(lab, labelpad = -21,fontsize = 5)




width_cm = 12 # 设置图形宽度
height_cm = 10 # 设置图形高度

# 将宽度和高度转换为英寸
width_inch = width_cm / 2.54
height_inch = height_cm / 2.54
# 使用inch指定图形大小
fig,ax = plt.subplots(nrows=3, ncols=2,figsize=(width_inch, height_inch),dpi = 300)
fig.patch.set_facecolor('#FFFFFF')


plot1(0,ax[0,0],CDm,'Duration(days)',lon,lat,c = cmaps.MPL_gnuplot2_r[:100])
plot2(0,ax[0,1],sD,pD,'Slope',lon,lat,c = cmaps.MPL_RdBu_r)

plot1(1,ax[1,0],CFm,'Frequency',lon,lat,c = cmaps.MPL_gnuplot2_r[:100])
plot2(1,ax[1,1],sF,pF,'Slope',lon,lat,c = cmaps.MPL_RdBu_r)

plot1(2,ax[2,0],CIm,'Intensity',lon,lat,c = cmaps.MPL_gnuplot2_r[:100])
plot2(2,ax[2,1],sI,pI,'Slope',lon,lat,c = cmaps.MPL_RdBu_r)
fig.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.05,wspace=0.02,hspace=0.1)
    
plt.savefig('../res/Fig/fig_f/fig2.png', dpi=300)
#plt.savefig('../res/Fig/fig_f/fig2.eps', dpi=300)
#plt.savefig('../res/Fig/fig_f/fig2.pdf', dpi=300)   