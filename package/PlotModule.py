# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 19:01:58 2025

@author: Meovv Van

email: 1259053332@qq.com

"""
import cmaps
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def PLotGlobal(CE,y,dmin,dmax,lab,c=cmaps.NCV_jaisnd):
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
                "font.size": 15,# 相当于小四大小
                "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
                "font.serif": ['Times New Roman'],#宋体
                'axes.unicode_minus': False # 处理负号，即-号
             }
    rcParams.update(config)  

    width_cm = 38 # 设置图形宽度
    height_cm = 18 # 设置图形高度

    # 将宽度和高度转换为英寸
    width_inch = width_cm / 2.54
    height_inch = height_cm / 2.54

    # 使用inch指定图形大小
    #fig,axs = plt.subplots(nrows=2, ncols=2,figsize=(width_inch, height_inch),dpi=300)


    fig ,ax = plt.subplots(figsize=(width_inch, height_inch),dpi=300)
    
    
    plt.title(y)
    #m = Basemap(projection='robin',resolution='l',lon_0=0)
    m = Basemap(projection = 'cyl',resolution='l',lon_0=0,llcrnrlon = -180, llcrnrlat = -60, urcrnrlon = 180, urcrnrlat = 90)
    m.fillcontinents(color = '#FFFFFF')


    #m.drawcoastlines() 
    m.readshapefile('../res/map/World_countries', 'World_countries',drawbounds=True)
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
    #extend='max',

    #label='Exposure of population to dry hot days(million)',
    #label='Population-weighted dry hot exposure days',
    label=lab,
    location = 'bottom',pad=0.01,fraction=0.03,anchor=(0.0,0.1)
)
    cbar.outline.set_linewidth(1.0)
    cbar.set_label(y, labelpad = 0.1)
    #plt.savefig('../Figure/losses/'+SSPs[i]+term[j]+' economic productivity losses'+'.png', bbox_inches='tight',dpi = 300)
    plt.show()
    
    
    
    
    
    
def PLot1(p1,p2,p3,p4,y,c,mlon,mlat,a,b):
    #c = cmaps.NCV_jaisnd
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import Normalize
    from matplotlib import rcParams
    from mpl_toolkits.basemap import Basemap
    config = {
                "font.family": 'serif',
                "font.size": 15,# 相当于小四大小
                "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
                "font.serif": ['Times New Roman'],#宋体
                'axes.unicode_minus': False # 处理负号，即-号
             }
    rcParams.update(config)  

    width_cm = 28 # 设置图形宽度
    height_cm = 18 # 设置图形高度

    # 将宽度和高度转换为英寸
    width_inch = width_cm / 2.54
    height_inch = height_cm / 2.54

    # 使用inch指定图形大小
    #fig,axs = plt.subplots(nrows=2, ncols=2,figsize=(width_inch, height_inch),dpi=300)


    fig ,ax = plt.subplots(figsize=(width_inch, height_inch),dpi=300)
    plt.title(y)
    #m = Basemap(projection='robin',resolution='l',lon_0=0)
    m = Basemap(projection = 'cyl',resolution='l',lon_0=0,llcrnrlon = -180, llcrnrlat = -60, urcrnrlon = 180, urcrnrlat = 90)
    m.fillcontinents(color = '#FFFFFF')

    LON,LAT = np.meshgrid(mlon,mlat)
    Lon = LON-180 
    xi, yi = m(Lon, LAT)
    cs = m.pcolormesh(xi, yi, p1, cmap=c)
    ax.contourf(xi, yi,p3,10, hatches=['////'],colors='none')
    ax.contourf(xi, yi,p4,10, hatches=['\\\\'],colors='none')
    ax.contourf(xi, yi,p2,10, hatches=['xxx'],colors='none')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    #m.drawparallels([-60,-45,0,45,90], labels=[1,0,0,0], linewidth=0.01,fontsize=10)
    #m.drawmeridians(np.arange(-180., 181., 40.), labels=[0,0,0,1], linewidth=0.01,fontsize=10)
    #m.drawcoastlines() 
    m.readshapefile('../res/map/GlobalArea', 'World_countries',drawbounds=True)
    #m.drawcountries()  
    #cbar = plt.colorbar(cs,location = 'bottom',pad=0.02,fraction=0.05)   
    left, bottom, width, height = 0.18, 0.32, 0.1, 0.15
    a = np.array(a)
    b = np.array(b)
    C = np.arange(1,26).reshape(5,5)
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.pcolor(C,cmap=c)
    ax2.set_xticks([1,2,3,4],b[[1,2,3,4]])
    ax2.set_yticks([1,2,3,4],a[[1,2,3,4]])
    ax2.set_xlabel('HDs slope(days/a)',labelpad = 0.08)
    ax2.set_ylabel('DDs slope(days/a)',labelpad = 0.08)



   # plt.savefig('../Figure/test/CDHDs_'+y+'.png', bbox_inches='tight',dpi = 300)

    plt.show()   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    