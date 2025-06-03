# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 18:28:48 2025

@author: Meovv Van

@mails : 1259053332@qq.com
"""

import joblib
import pickle
import cmaps
import numpy as np
import pandas as pd
from PIL import Image
import package.CalModule as cl
import package.PlotModule as pm
import matplotlib.pyplot as plt
import pymannkendall as mk

with open('../res/dataprced/E_Factor.pkl', 'rb') as f:  # 读取pickle文件
    E = joblib.load(f)
    f.close()
    
LID = pd.read_csv('../res/NCDs/Location_ID.csv')
CID = pd.read_excel('../res/other/countriesID.xls')

 
cg = ['高收入国家','中高等收入国家','中低等收入国家',  '低收入国家', ]
lab = ['High Income','Upper-middle Income','Lower-middle Income','Low Income']
lab = ['HICs','UMICs','LMICs','LICs']
LID = pd.read_csv('../res/NCDs/Location_ID.csv')
CID = pd.read_excel('../res/other/countriesID.xls') 
 
Income_Group = pd.read_csv('../res/SEdata/Income_Group.csv')
colors = ['#C72228','#F98F34','#6B98C4','#8887CB']
colors =  [cl.rgb_to_hex([140,42,115]),cl.rgb_to_hex([223,74,104]),cl.rgb_to_hex([252,154,107]),cl.rgb_to_hex([255,183,3])]
colors2 =  ['#ECF4E5','#B3D5B2','#7EBAB5','#387FAB']
cg = ['高收入国家','中高等收入国家','中低等收入国家',  '低收入国家', ]
mergeID = pd.merge(LID, CID,left_on ='location_name',right_on ='FCNAME',how = 'inner')  
MID = pd.merge(mergeID,Income_Group,left_on ='SOC',right_on ='Country Code',how = 'left')
CL = np.zeros_like(MID['Income_Group'].values)
for i in range(4):
    CL[MID['Income_Group'].values==cg[i]] = i


def Hex_to_RGB(hex,alpha):
    r = int(hex[1:3],16)
    g = int(hex[3:5],16)
    b = int(hex[5:7], 16)
    #rgb = str(r)+','+str(g)+','+str(b)
    rgb = [r/255,g/255,b/255,alpha]
    
    return rgb

def getSP(data):
    slopes = []
    pva =  []
    for i in data.keys():
       
                x = data[i]
                res = mk.original_test(x)
                slopes.append(res.slope)
                pva.append(res.p)
    return pd.DataFrame({'C': data.keys(),'S':np.array(slopes),'P':np.array(pva)})

SP3 = getSP(E['e6'])
SP3['NAME'] = MID['NAME']
SP4 = getSP(E['e4'])
SP4['NAME'] = MID['NAME']

def PLotGlobal1(ax,CE):
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
    from shapely.geometry import Polygon as SPoly
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle,Rectangle
    config = {
                "font.family": 'serif',
                "font.size": 6,# 相当于小四大小
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
    m.readshapefile('../res/map/GlobalArea', 'GA',drawbounds=True,linewidth=0.3)
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

    Cen = []
    for shape in df_poly['shapes2']:
        center = shape.centroid
        Cen.append([center.x,center.y])
    Cen = np.array(Cen)
    df_poly['centerx']  = Cen[:,0]
    df_poly['centery']  = Cen[:,1]
    colrnum = pd.DataFrame({'area':CE['NAME'],'L':CE['E'],'L2':CE['E2'],'GP':CE['GP']})
    df_poly2 = df_poly.merge(colrnum)

    cx = []
    cy = []
    s = []
    s1 = []
    gp = []
    for i in df_poly2['area'].unique():
       A = df_poly2[(df_poly2['area']==i)]
       B = A[A['len']==np.nanmax(A['len'].values)]
       cx.append(B['centerx'].values[0])
       cy.append(B['centery'].values[0])
       s1.append(B['L'].values[0])
       s.append(B['L2'].values[0])
       gp.append(B['GP'].values[0]) 
    s = np.array(s)
    s1 = np.array(s1)
    cx = np.array(cx)
    cy = np.array(cy)
    gp = np.array(gp)
    
    for ci in range(4):
        
        pc = PatchCollection(df_poly2[df_poly2['GP'] == cg[ci]].shapes, zorder=1,edgecolor='#FFFFFF',linewidth=0.25)
        pc.set_facecolor(colors2[ci])
        ax.add_collection(pc)
        
    c_x = cx 
    c_y = cy 
    s_ = s 
    s_1 = s1 
    
    ax.scatter(c_x,c_y,s=30*np.array(s_)/5e6, facecolor=Hex_to_RGB(colors[1],0.4),linestyles = ['--'],edgecolor = '#000000',linewidth = 0.4)
    dy = (30*np.array(s_)/5e6-30*np.array(s_1)/5e6)
    s2 = cl.Normalize(30*np.array(s_)/5e6+30*np.array(s_1)/5e6)
    ax.scatter(c_x,c_y-dy/(100),s=30*np.array(s_1)/5e6 ,facecolor=Hex_to_RGB(colors[1],0.7),linestyles = ['-'],edgecolor = '#000000' ,linewidth =0.6)
    
      #dff
         
        
        
    ax.scatter(-150,-20,s=20*np.nansum(s)/5e6, facecolor=[1,1,1,1],linestyles = ['--'],edgecolor = '#000000',linewidth =0.6)
    dy = (20*np.nansum(s)/5e6-20*np.nansum(s1)/5e6)
   
    ax.scatter(-150,-20-3,s=20*np.nansum(s1)/5e6 ,facecolor=[1,1,1,1],linestyles = ['-'],edgecolor = '#000000' ,linewidth =0.6)
    #pc = PatchCollection(df_poly2.shapes, zorder=2)
    #pc.set_facecolor(c(norm(df_poly2['L'].fillna(0).values)))
    #ax.add_collection(pc)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
   
    ax.plot([-135,-135+10],[8,8],linestyle = '--',color = '#000000',linewidth =0.6)
    ax.text(-135+12,5,'2020  '+str(np.round(np.nansum(s)/1e6,2)),fontsize = 5,color = '#000000')
    ax.plot([-130,-130+10],[-5,-5],linestyle = '-',color = '#000000',linewidth =0.6)
    ax.text(-130+12,-5,'2001  '+str(np.round(np.nansum(s1)/1e6,2)),fontsize = 5,color = '#000000')
    #ax.legend(custom_lines, ['2001  '+str(np.round(np.nansum(s1)/1e6,2))+'billion',],fontsize =12,ncol = 1,edgecolor='none',bbox_to_anchor=(0.235, 0.42, 0.1, 0.02),facecolor ='none')  
    #ax.legend(custom_lines2, ['2020  '+str(np.round(np.nansum(s)/1e6,2))+'billion',],fontsize =12,ncol = 1,edgecolor='none',bbox_to_anchor=(0.22, 0.46, 0.1, 0.02),facecolor ='none')  
    
    
    ax.add_patch(Rectangle((-27, -60),  # 圆心
                25,8,  # 半径
                facecolor=Hex_to_RGB(colors2[0],0.7),edgecolor = '#000000',linewidth = 0.4
                ))
    ax.text(-20,-58,lab[0],fontsize = 5,color = '#000000')
    ax.add_patch(Rectangle((0, -60),  # 圆心
                 25,8,  # 半径
                 facecolor=Hex_to_RGB(colors2[1],0.7),edgecolor = '#000000',linewidth = 0.4
                ))
    ax.text(2,-58,lab[1],fontsize = 5,color = '#555555')
    ax.add_patch(Rectangle((27, -60),  # 圆心
                 25,8,  # 半径
                 facecolor=Hex_to_RGB(colors2[2],0.7),edgecolor = '#000000',linewidth = 0.4
                ))
    ax.text(31,-58,lab[2],fontsize = 5,color = '#555555')
    ax.add_patch(Rectangle((54, -60),  # 圆心
                25,8,  # 半径
                facecolor=Hex_to_RGB(colors2[3],0.7),edgecolor = '#000000',linewidth = 0.4
                ))
    ax.text(60,-58,lab[3],fontsize = 5,color = '#FFFFFF')
    ax.text(-27,-50,'Income Group',fontsize = 6,color = '#000000')
    
    ax.text(-170,-60,'Global number of patients',fontsize = 6)
    ax.text(-170,-66,'unit. million',fontsize = 6)
    ax.text(0,1,'A',transform=ax.transAxes,fontsize = 7,fontname = 'Arial')

def PLotGlobal2(ax,CE):
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
    from shapely.geometry import Polygon as SPoly
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle,Rectangle
    config = {
                "font.family": 'serif',
                "font.size": 6,# 相当于小四大小
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
    m.readshapefile('../res/map/GlobalArea', 'GA',drawbounds=True,linewidth=0.25)
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

    Cen = []
    for shape in df_poly['shapes2']:
        center = shape.centroid
        Cen.append([center.x,center.y])
    Cen = np.array(Cen)
    df_poly['centerx']  = Cen[:,0]
    df_poly['centery']  = Cen[:,1]
    colrnum = pd.DataFrame({'area':CE['NAME'],'L':CE['E'],'L2':CE['E2'],'GP':CE['GP']})
    df_poly2 = df_poly.merge(colrnum)

    cx = []
    cy = []
    s = []
    s1 = []
    gp = []
    for i in df_poly2['area'].unique():
       A = df_poly2[(df_poly2['area']==i)]
       B = A[A['len']==np.nanmax(A['len'].values)]
       cx.append(B['centerx'].values[0])
       cy.append(B['centery'].values[0])
       s1.append(B['L'].values[0])
       s.append(B['L2'].values[0])
       gp.append(B['GP'].values[0]) 
    s = np.array(s)
    s1 = np.array(s1)
    cx = np.array(cx)
    cy = np.array(cy)
    gp = np.array(gp)
    
    for ci in range(4):
        
        pc = PatchCollection(df_poly2[df_poly2['GP'] == cg[ci]].shapes, zorder=1,edgecolor='#FFFFFF',linewidth=0.25)
        pc.set_facecolor(colors2[ci])
         
        ax.add_collection(pc)
        
  
    c_x = cx 
    c_y = cy 
    s_ = s 
    s_1 = s1 
    
    ax.scatter(c_x,c_y,s=30*np.array(s_)/1e5, facecolor=Hex_to_RGB(colors[0],0.4),linestyles = ['--'],edgecolor = '#000000',linewidth = 0.6)
    dy = (30*np.array(s_)/1e5-30*np.array(s_1)/1e5)
    s2 = cl.Normalize(30*np.array(s_)/1e5+30*np.array(s_1)/1e5)
    ax.scatter(c_x,c_y-dy/(100),s=30*np.array(s_1)/1e5 ,facecolor=Hex_to_RGB(colors[0],0.7),linestyles = ['-'],edgecolor = '#000000' ,linewidth =0.6)
    
      #dff
         
        
        
    ax.scatter(-150,-20,s=20*np.nansum(s)/1e5, facecolor=[1,1,1,1],linestyles = ['--'],edgecolor = '#000000',linewidth =0.6)
    dy = (20*np.nansum(s)/1e5-20*np.nansum(s1)/1e5)
   
    ax.scatter(-150,-20-3,s=20*np.nansum(s1)/1e5 ,facecolor=[1,1,1,1],linestyles = ['-'],edgecolor = '#000000' ,linewidth =0.6)
    #pc = PatchCollection(df_poly2.shapes, zorder=2)
    #pc.set_facecolor(c(norm(df_poly2['L'].fillna(0).values)))
    #ax.add_collection(pc)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
   
    ax.plot([-145,-145+10],[0,0],linestyle = '--',color = '#000000',linewidth =0.6)
    ax.text(-145+12,0,'2020  '+str(np.round(np.nansum(s)/1e6,2)),fontsize =5,color = '#000000')
    ax.plot([-140,-140+10],[-7,-7],linestyle = '-',color = '#000000',linewidth =0.6)
    ax.text(-140+12,-8,'2001  '+str(np.round(np.nansum(s1)/1e6,2)),fontsize = 5,color = '#000000')
              
    ax.add_patch(Rectangle((-27, -60),  # 圆心
                 25,8,  # 半径
                 facecolor=Hex_to_RGB(colors2[0],0.7),edgecolor = '#000000',linewidth = 0.4
                 ))
    ax.text(-20,-58,lab[0],fontsize = 5,color = '#000000')
    ax.add_patch(Rectangle((0, -60),  # 圆心
                 25,8,  # 半径
                 facecolor=Hex_to_RGB(colors2[1],0.7),edgecolor = '#000000',linewidth = 0.4
                ))
    ax.text(2,-58,lab[1],fontsize = 5,color = '#555555')
    ax.add_patch(Rectangle((27, -60),  # 圆心
                 25,8,  # 半径
                 facecolor=Hex_to_RGB(colors2[2],0.7),edgecolor = '#000000',linewidth = 0.4
                ))
    ax.text(31,-58,lab[2],fontsize = 5,color = '#555555')
    ax.add_patch(Rectangle((54, -60),  # 圆心
                25,8,  # 半径
                facecolor=Hex_to_RGB(colors2[3],0.7),edgecolor = '#000000',linewidth = 0.4
                ))
    ax.text(60,-58,lab[3],fontsize = 5,color = '#FFFFFF')
    ax.text(-27,-50,'Income Group',fontsize = 6,color = '#000000')
    

    ax.text(-170,-55,'Global number of deaths',fontsize = 6,)
    ax.text(-170,-62,'unit. million',fontsize = 6,)
    ax.text(0,1,'B',transform=ax.transAxes,fontsize = 7,fontname = 'Arial')


def PLotGlobal3(ax,CE,SP):
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
    from shapely.geometry import Polygon as SPoly
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle,Rectangle
    config = {
                "font.family": 'serif',
                "font.size": 6,# 相当于小四大小
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
    m.readshapefile('../res/map/GlobalArea', 'GA',drawbounds=True,linewidth=0.25)
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

    Cen = []
    for shape in df_poly['shapes2']:
        center = shape.centroid
        Cen.append([center.x,center.y])
    Cen = np.array(Cen)
    df_poly['centerx']  = Cen[:,0]
    df_poly['centery']  = Cen[:,1]
    colrnum = pd.DataFrame({'area':CE['NAME'],'L':CE['E'],'GP':CE['GP']})
    df_poly2 = df_poly.merge(colrnum)

    cx = []
    cy = []
    cn = []
    s1 = []
    gp = []
    for i in df_poly2['area'].unique():
       A = df_poly2[(df_poly2['area']==i)]
       B = A[A['len']==np.nanmax(A['len'].values)]
       cx.append(B['centerx'].values[0])
       cy.append(B['centery'].values[0])
       s1.append(B['L'].values[0])
       cn.append(i)
       gp.append(B['GP'].values[0]) 
  
    s1 = np.array(s1)
    cx = np.array(cx)
    cy = np.array(cy)
    gp = np.array(gp)
    cn = np.array(cn)
    for ci in range(4):
        
        pc = PatchCollection(df_poly2[df_poly2['GP'] == cg[ci]].shapes, zorder=1,edgecolor='#FFFFFF',linewidth=0.25)
        pc.set_facecolor(colors2[ci])
         
        ax.add_collection(pc)
    
    SU = SP[(SP['P']<0.05)&(SP['S']>0)]
    SD = SP[(SP['P']<0.05)&(SP['S']<0)]
    new_list = [x for x in cn if x not in SU['NAME'].values]
    new_list2 = [x for x in new_list if x not in SD['NAME'].values]
    for i in SU['NAME'].values:
        c_x = cx[cn==i]
        c_y = cy[cn==i]
       
        s_1 = s1[cn==i]
        
        ax.scatter(c_x,c_y,s=5e2*np.array(s_1), facecolor='#EE0000',linestyles = ['-'],edgecolor = '#000000',linewidth = 0.6,alpha = 0.5)
      
    
    for i in SD['NAME'].values:
        c_x = cx[cn==i]
        c_y = cy[cn==i] 
       
        s_1 = s1[cn==i]
    
        ax.scatter(c_x,c_y,s=5e2*np.array(s_1), facecolor='#00BFFF',linestyles = ['-'],edgecolor = '#000000',linewidth =  0.6,alpha = 0.5)
    
    for i in new_list2:
        c_x = cx[cn==i]
        c_y = cy[cn==i]
       
        s_1 = s1[cn==i]
    
        ax.scatter(c_x,c_y,s=5e2*np.array(s_1), facecolor='#CCCCCC',linestyles = ['-'],edgecolor = '#000000',linewidth = 0.6,alpha = 0.5)
       
        
   
    #pc = PatchCollection(df_poly2.shapes, zorder=2)
    #pc.set_facecolor(c(norm(df_poly2['L'].fillna(0).values)))
    #ax.add_collection(pc)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
   
           
    ax.add_patch(Rectangle((-27, -60),  # 圆心
                 25,8,  # 半径
                 facecolor=Hex_to_RGB(colors2[0],0.7),edgecolor = '#000000',linewidth = 0.4
                 ))
    ax.text(-20,-58,lab[0],fontsize = 5,color = '#000000')
    ax.add_patch(Rectangle((0, -60),  # 圆心
                 25,8,  # 半径
                 facecolor=Hex_to_RGB(colors2[1],0.7),edgecolor = '#000000',linewidth = 0.4
                ))
    ax.text(2,-58,lab[1],fontsize = 5,color = '#555555')
    ax.add_patch(Rectangle((27, -60),  # 圆心
                 25,8,  # 半径
                 facecolor=Hex_to_RGB(colors2[2],0.7),edgecolor = '#000000',linewidth = 0.4
                ))
    ax.text(31,-58,lab[2],fontsize = 5,color = '#555555')
    ax.add_patch(Rectangle((54, -60),  # 圆心
                25,8,  # 半径
                facecolor=Hex_to_RGB(colors2[3],0.7),edgecolor = '#000000',linewidth = 0.4
                ))
    ax.text(60,-58,lab[3],fontsize = 5,color = '#FFFFFF')
    ax.text(-27,-50,'Income Group',fontsize = 6,color = '#000000')
    ax.text(0,1,'C',transform=ax.transAxes,fontsize = 7,fontname = 'Arial')

def PLotGlobal4(ax,CE,SP):
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
    from shapely.geometry import Polygon as SPoly
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle,Rectangle
    config = {
                "font.family": 'serif',
                "font.size": 7,# 相当于小四大小
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
    m.readshapefile('../res/map/GlobalArea', 'GA',drawbounds=True,linewidth=0.25)
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

    Cen = []
    for shape in df_poly['shapes2']:
        center = shape.centroid
        Cen.append([center.x,center.y])
    Cen = np.array(Cen)
    df_poly['centerx']  = Cen[:,0]
    df_poly['centery']  = Cen[:,1]
    colrnum = pd.DataFrame({'area':CE['NAME'],'L':CE['E'],'GP':CE['GP']})
    df_poly2 = df_poly.merge(colrnum)

    cx = []
    cy = []
    cn = []
    s1 = []
    gp = []
    for i in df_poly2['area'].unique():
       A = df_poly2[(df_poly2['area']==i)]
       B = A[A['len']==np.nanmax(A['len'].values)]
       cx.append(B['centerx'].values[0])
       cy.append(B['centery'].values[0])
       s1.append(B['L'].values[0])
       cn.append(i)
       gp.append(B['GP'].values[0]) 
  
    s1 = np.array(s1)
    cx = np.array(cx)
    cy = np.array(cy)
    gp = np.array(gp)
    cn = np.array(cn)
    for ci in range(4):
        
        pc = PatchCollection(df_poly2[df_poly2['GP'] == cg[ci]].shapes, zorder=1,edgecolor='#FFFFFF',linewidth=0.25)
        pc.set_facecolor(colors2[ci])
         
        ax.add_collection(pc)
        
    SU = SP[(SP['P']<0.05)&(SP['S']>0)]
    SD = SP[(SP['P']<0.05)&(SP['S']<0)]
    new_list = [x for x in cn if x not in SU['NAME'].values]
    new_list2 = [x for x in new_list if x not in SD['NAME'].values]
    for i in SU['NAME'].values:
        c_x = cx[cn==i]
        c_y = cy[cn==i]
       
        s_1 = s1[cn==i]
        
        ax.scatter(c_x,c_y,s=2e3*np.array(s_1), facecolor='#EE0000',linestyles = ['-'],edgecolor = '#000000',linewidth = 0.6,alpha = 0.5)
      
    
    for i in SD['NAME'].values:
        c_x = cx[cn==i]
        c_y = cy[cn==i] 
       
        s_1 = s1[cn==i]
    
        ax.scatter(c_x,c_y,s=2e3*np.array(s_1), facecolor='#00BFFF',linestyles = ['-'],edgecolor = '#000000',linewidth =  0.6,alpha = 0.5)
    
    for i in new_list2:
        c_x = cx[cn==i]
        c_y = cy[cn==i]
       
        s_1 = s1[cn==i]
    
        ax.scatter(c_x,c_y,s=2e3*np.array(s_1), facecolor='#CCCCCC',linestyles = ['-'],edgecolor = '#000000',linewidth = 0.6,alpha = 0.5)
       
        
   
    #pc = PatchCollection(df_poly2.shapes, zorder=2)
    #pc.set_facecolor(c(norm(df_poly2['L'].fillna(0).values)))
    #ax.add_collection(pc)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
   
           
    ax.add_patch(Rectangle((-27, -60),  # 圆心
                 25,8,  # 半径
                 facecolor=Hex_to_RGB(colors2[0],0.7),edgecolor = '#000000',linewidth = 0.4
                 ))
    ax.text(-20,-58,lab[0],fontsize = 5,color = '#000000')
    ax.add_patch(Rectangle((0, -60),  # 圆心
                 25,8,  # 半径
                 facecolor=Hex_to_RGB(colors2[1],0.7),edgecolor = '#000000',linewidth = 0.4
                ))
    ax.text(2,-58,lab[1],fontsize = 5,color = '#555555')
    ax.add_patch(Rectangle((27, -60),  # 圆心
                 25,8,  # 半径
                 facecolor=Hex_to_RGB(colors2[2],0.7),edgecolor = '#000000',linewidth = 0.4
                ))
    ax.text(31,-58,lab[2],fontsize = 5,color = '#555555')
    ax.add_patch(Rectangle((54, -60),  # 圆心
                25,8,  # 半径
                facecolor=Hex_to_RGB(colors2[3],0.7),edgecolor = '#000000',linewidth = 0.4
                ))
    ax.text(60,-58,lab[3],fontsize = 5,color = '#FFFFFF')
    ax.text(-27,-50,'Income Group',fontsize = 6,color = '#000000')
    ax.text(0,1,'D',transform=ax.transAxes,fontsize = 7,fontname = 'Arial')

CEX = pd.DataFrame({"NAME":mergeID["NAME"],'GP':MID["Income_Group"],'E':E['e3'].values[0,:],'E2':E['e3'].values[19,:]})

CEX2 = pd.DataFrame({"NAME":mergeID["NAME"],'GP':MID["Income_Group"],'E':E['e5'].values[0,:],'E2':E['e5'].values[19,:]})

CEX3 = pd.DataFrame({"NAME":mergeID["NAME"],'GP':MID["Income_Group"],'E':np.nanmean(E['e6'].values,axis=0),})
CEX4 = pd.DataFrame({"NAME":mergeID["NAME"],'GP':MID["Income_Group"],'E':np.nanmean(E['e4'].values,axis=0),})

width_cm = 18# 设置图形宽度
height_cm = 9 # 设置图形高度

# 将宽度和高度转换为英寸
width_inch = width_cm / 2.54
height_inch = height_cm / 2.54
# 使用inch指定图形大小
fig,ax = plt.subplots(nrows=2, ncols=2,figsize=(width_inch, height_inch),dpi=400)
fig.patch.set_facecolor('#FFFFFF')
 
PLotGlobal1(ax[0,0],CEX,)
PLotGlobal2(ax[1,0],CEX2)
PLotGlobal3(ax[0,1],CEX3,SP3)
PLotGlobal4(ax[1,1],CEX4,SP4)

fig.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.01,wspace=0.005,hspace=0.02)




plt.savefig('../res/Fig/fig_f/fig3.png', dpi=300)
#plt.savefig('../res/Fig/fig_f/fig3.pdf', dpi=300)


