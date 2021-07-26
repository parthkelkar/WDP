# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 14:06:28 2021

@author: kelka
"""

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import iris
import iris.analysis.cartography
import iris.analysis.stats
import iris.plot as iplt
import iris.quickplot as qplt
import cf_units
import cartopy.io.shapereader as shpreader
import dask.array as da
import shapely.vectorized as shp_vect
from iris.analysis import Aggregator
from iris.util import rolling_window
# import mask_removal as  m
import time

# 


def coorelation(path1,filename1,filename2,filename3,model):   
    
    
    
    path='D:/project/Birmingham-gpp-data/'+path1+'/'    
    pr = iris.load_cube(path+filename1)
    tas = iris.load_cube(path+filename2)
    gpp = iris.load_cube(path+filename3)
    
    
    tcoord = tas.coord('time')
    tcoord.units = cf_units.Unit(tcoord.units.origin, calendar='365_day')
    tcoord = pr.coord('time')
    tcoord.units = cf_units.Unit(tcoord.units.origin, calendar='365_day')
    tcoord = gpp.coord('time')
    tcoord.units = cf_units.Unit(tcoord.units.origin, calendar='365_day')
    
    
    tas.convert_units('celsius')
    pr.convert_units('kg m-2 days-1')
    gpp.convert_units('kg m-2 days-1')
    
    
    gpp.coord('latitude').points=pr.coord('latitude').points
    gpp.coord('latitude').bounds=pr.coord('latitude').bounds
    gpp.coord('longitude').points=pr.coord('longitude').points
    gpp.coord('longitude').bounds=pr.coord('longitude').bounds
        
        
        
    cubea=iris.analysis.stats.pearsonr(pr,tas,['time'])
    cubeb=iris.analysis.stats.pearsonr(gpp,pr,['time'])
    cubec=iris.analysis.stats.pearsonr(tas,gpp,['time'])
        
    
    qplt.contourf(cubea)
    plt.title('pr vs tas')
    plt.show()
    # plt.savefig('D:/project/Birmingham-gpp-data/plots/'+
    #             path1+'_'+model+'_pr_TAS'+time.strftime("%Y%m%d-%H%M%S"))
    plt.close('all')
    
    
    
    qplt.contourf(cubeb)
    plt.title('pr vs gpp')
    plt.show()
    # plt.savefig('D:/project/Birmingham-gpp-data/plots/'+
    #             path1+'_'+model+'_pr_GPP'+time.strftime("%Y%m%d-%H%M%S"))
    
    plt.close('all')
    
    
    qplt.contourf(cubec)
    plt.title('gpp vs tas')
    plt.show()
    # plt.savefig('D:/project/Birmingham-gpp-data/plots/'+
    #             path1+'_'+model+'_GPP_TAS'+time.strftime("%Y%m%d-%H%M%S"))
  
    plt.close('all')





path1 = 'JJA'
model='BCC_ESM1'
filename1='CMIP6_BCC-ESM1_Amon_piControl_r1i1p1f1_pr_1850-2300.nc'
filename2='CMIP6_BCC-ESM1_Amon_piControl_r1i1p1f1_tas_1850-2300.nc'
filename3='CMIP6_BCC-ESM1_Lmon_piControl_r1i1p1f1_gpp_1850-2300.nc'
coorelation(path1,filename1,filename2,filename3,model)

# model='IPSL_CM6A'
# filename1='CMIP6_IPSL-CM6A-LR_Amon_piControl_r1i1p1f1_pr_1850-3049.nc'
# filename2='CMIP6_IPSL-CM6A-LR_Amon_piControl_r1i1p1f1_tas_1850-3049.nc'
# filename3='CMIP6_IPSL-CM6A-LR_Lmon_piControl_r1i1p1f1_gpp_1850-3049.nc'
# coorelation(path1,filename1,filename2,filename3,model)

# model='UKESM1'
# filename1='CMIP6_UKESM1-0-LL_Amon_piControl_r1i1p1f2_pr_1960-3059.nc'
# filename2='CMIP6_UKESM1-0-LL_Amon_piControl_r1i1p1f2_tas_1960-3059.nc'
# filename3='CMIP6_UKESM1-0-LL_Lmon_piControl_r1i1p1f2_gpp_1960-3059.nc'
# coorelation(path1,filename1,filename2,filename3,model)