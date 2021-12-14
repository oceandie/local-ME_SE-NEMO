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
import cartopy.crs as ccrs
import cmocean
from utils import plot_hpge

# ==============================================================================
# Input parameters

# 1. INPUT FILES

BATHY_file = '/data/users/dbruciaf/SE-NEMO/se-orca025/MEs_300_1650/ant/r015-r010_r007_r004v2/bathymetry.MEs_4env_800_015-010_007_004v2_ant_maxdep_2600.0.nc'

# 3. PLOT
#lon0 = -178.
#lon1 =  178.
#lat0 =  -78.8
#lat1 =   88.
proj =  None #ccrs.Robinson() #ccrs.Mercator() #ccrs.Robinson()

# ==============================================================================

# Loading domain geometry
ds  = xr.open_dataset(BATHY_file)

# Plotting BATHYMETRY ----------------------------------------------------------

#bathy = ds_dom["bathymetry"]
env = ds["hbatt_2"]#.isel(x_c=slice(1, None), y_c=slice(1, None))

fig_name = 'envelope_2.png'
fig_path = "./"
lon = None #ds_dom["glamf"]
lat = None #ds_dom["gphif"]
colmap = "jet" #cmocean.cm.ice
vmin = 350.
vmax = 2500.
cbar_extend = 'max' #"max"
cbar_label = "Depth [$m$]"
cbar_hor = 'horizontal'
map_lims = [0, 1441, 0, 1206]
cn_lev = None 

plot_hpge(fig_name, fig_path, lon, lat, env, proj, colmap, 
          vmin, vmax, cbar_extend, cbar_label, cbar_hor, map_lims, env, cn_lev)

env = ds["msk_pge2"]#.isel(x_c=slice(1, None), y_c=slice(1, None))

fig_name = 'pge2.png'
fig_path = "./"
lon = None #ds_dom["glamf"]
lat = None #ds_dom["gphif"]
colmap = "jet" #cmocean.cm.ice
vmin = 1.
vmax = 2.
cbar_extend = 'max' #"max"
cbar_label = ""
cbar_hor = 'horizontal'
map_lims = [0, 1441, 0, 1206]
cn_lev = None

plot_hpge(fig_name, fig_path, lon, lat, env, proj, colmap,
          vmin, vmax, cbar_extend, cbar_label, cbar_hor, map_lims, env, cn_lev)

