#!/usr/bin/env python

#     |------------------------------------------------------------|
#     | Author: Diego Bruciaferri                                  |
#     | Date and place: 07-09-2021, Met Office, UK                 |
#     |------------------------------------------------------------|


import os
from os.path import join, isfile, basename, splitext
import glob
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
from xnemogcm import open_domain_cfg
import cartopy.crs as ccrs
import cmocean
from utils import plot_hpge

# ==============================================================================
# Input parameters

# 1. INPUT FILES

vcoord = 'r02-r02'
DOMCFG_file = '/data/users/dbruciaf/SE-NEMO/se-orca025/MEs_300_1650/ant/r02-r02/domain_cfg_r02-r02.nc'
HPGE_dir = '/data/users/dbruciaf/SE-NEMO/se-orca025/MEs_300_1650/ant/'
HPGE_file = '/maximum_hpge.nc'


# 3. PLOT
#lon0 = -178.
#lon1 =  178.
#lat0 =  -78.8
#lat1 =   88.
proj =  None #ccrs.Robinson() #ccrs.Mercator() #ccrs.Robinson()

# ==============================================================================

# Loading domain geometry
ds_dom  = open_domain_cfg(files=[DOMCFG_file])
for i in ['bathymetry','bathy_meter']:
    for dim in ['x','y']:
        ds_dom[i] = ds_dom[i].rename({dim: dim+"_c"})

ds_hpge  = xr.open_dataset(HPGE_dir + vcoord + HPGE_file)

# Extracting only the part of the domain we need
#ds_dom  = ds_dom.isel(x_c=slice(880,1200),x_f=slice(880,1200),y_c=slice(880,1140),y_f=slice(880,1140))
#ds_hpge =  ds_hpge.isel(x=slice(880,1200),y=slice(880,1140))


# Plotting BATHYMETRY ----------------------------------------------------------

#bathy = ds_dom["bathymetry"]
bathy = ds_dom["bathymetry"]#.isel(x_c=slice(1, None), y_c=slice(1, None))
varss = list(ds_hpge.keys())

for env in range(len(varss)):

    # GLOBAL SHELVES

    fig_name = 'GLO_' + varss[env] + '_' + vcoord + '.png'
    fig_path = "./"
    lon = None #ds_dom["glamf"]
    lat = None #ds_dom["gphif"]
    var = ds_hpge[varss[env]] 
    colmap = 'hot_r' #cmocean.cm.ice
    vmin = 0.0
    vmax = 0.1
    cbar_extend = 'max' #"max"
    cbar_label = "HPG errors [$m\;s^{-1}$]"
    cbar_hor = 'horizontal'
    map_lims = [0, 1441, 390, 1206]
    cn_lev = [300.] # None 

    plot_hpge(fig_name, fig_path, lon, lat, var, proj, colmap, 
              vmin, vmax, cbar_extend, cbar_label, cbar_hor, map_lims, bathy, cn_lev)

    # ANTARCTIC SHELF

    fig_name = 'ANT_' + varss[env] + '_' + vcoord + '.png'
    fig_path = "./"
    lon = None #ds_dom["glamf"]
    lat = None #ds_dom["gphif"]
    var = ds_hpge[varss[env]]
    colmap = 'hot_r' #cmocean.cm.ice
    vmin = 0.0
    vmax = 0.1
    cbar_extend = 'max' #"max"
    cbar_label = "HPG errors [$m\;s^{-1}$]"
    cbar_hor = 'horizontal'
    map_lims = [0, 1441, 0, 390]
    cn_lev = [800.] # None 

    plot_hpge(fig_name, fig_path, lon, lat, var, proj, colmap,
              vmin, vmax, cbar_extend, cbar_label, cbar_hor, map_lims, bathy, cn_lev) 
