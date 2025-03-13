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

vcoord = 'MEs_450-800_3200'#_r007_r004v2'
DOMCFG_file = '/data/users/dbruciaf/SE-NEMO/se-orca025/MEs_450-800_3200/domain_cfg_r018-01-01_glo-r018-01_ant_opt_v3.nc'
HPGE_dir = '/data/users/dbruciaf/SE-NEMO/se-orca025/'
HPGE_file_ant = '/maximum_hpge_3env_800_015-010_ant.nc'
HPGE_file_glo = '/maximum_hpge_4env_450_015-010-010_glo.nc'

#vcoord = 'hsz_51_ztap'
#DOMCFG_file = '/data/users/dbruciaf/SE-NEMO/se-orca025/hsz_39_ztap/domain_cfg_39_ztaper_match.nc'
#HPGE_dir = '/data/users/dbruciaf/SE-NEMO/se-orca025/'
#HPGE_file = '/maximum_hpge.nc'

# 3. PLOT
proj =  None #ccrs.Robinson() #ccrs.Mercator() #ccrs.Robinson()

# ==============================================================================

# Loading domain geometry
ds_dom  = open_domain_cfg(files=[DOMCFG_file])
for i in ['bathymetry','bathy_meter']:
    for dim in ['x','y']:
        ds_dom[i] = ds_dom[i].rename({dim: dim+"_c"})


# GLOBAL SHELVES
ds_hpge  = xr.open_dataset(HPGE_dir + vcoord + HPGE_file_glo)

bathy = ds_dom["bathymetry"]#.isel(x_c=slice(1, None), y_c=slice(1, None))
varss = list(ds_hpge.keys())

for env in range(len(varss)):

    fig_name = 'GLO_' + varss[env] + '_' + vcoord + '.png'
    fig_path = "./"
    lon = None #ds_dom["glamf"]
    lat = None #ds_dom["gphif"]
    var = ds_hpge[varss[env]] 
    colmap = 'hot_r' #cmocean.cm.ice
    vmin = 0.0
    vmax = 0.05
    cbar_extend = 'max' #"max"
    cbar_label = "HPG errors [$m\;s^{-1}$]"
    cbar_hor = 'horizontal'
    #map_lims = [0, 1441, 390, 1206]
    map_lims = [0, 1441, 0, 1206]
    cn_lev = [450.,1500.,2600.] # None 

    plot_hpge(fig_name, fig_path, lon, lat, var, proj, colmap, 
              vmin, vmax, cbar_extend, cbar_label, cbar_hor, map_lims, bathy, cn_lev)

# ANTARCTIC SHELF
ds_hpge  = xr.open_dataset(HPGE_dir + vcoord + HPGE_file_ant)
bathy = ds_dom["bathymetry"]#.isel(x_c=slice(1, None), y_c=slice(1, None))
varss = list(ds_hpge.keys())

for env in range(len(varss)):

    fig_name = 'ANT_' + varss[env] + '_' + vcoord + '.png'
    fig_path = "./"
    lon = None #ds_dom["glamf"]
    lat = None #ds_dom["gphif"]
    var = ds_hpge[varss[env]]
    colmap = 'hot_r' #cmocean.cm.ice
    vmin = 0.0
    vmax = 0.05
    cbar_extend = 'max' #"max"
    cbar_label = "HPG errors [$m\;s^{-1}$]"
    cbar_hor = 'horizontal'
    map_lims = [0, 1441, 0, 390]
    cn_lev = [800.] # None 

    plot_hpge(fig_name, fig_path, lon, lat, var, proj, colmap,
              vmin, vmax, cbar_extend, cbar_label, cbar_hor, map_lims, bathy, cn_lev) 
