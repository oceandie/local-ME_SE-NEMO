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

BATHY_file1 = '/data/users/dbruciaf/SE-NEMO/se-orca025/MEs_300_1650/ant/r015-r010/bathymetry.MEs_4env_800_015-010_ant_maxdep_2600.0.nc'

BATHY_file2 = '/data/users/dbruciaf/SE-NEMO/se-orca025/MEs_300_1650/ant/r015-r010_r007_r004v2/bathymetry.MEs_4env_800_015-010_007_004v2_ant_maxdep_2600.0.nc'

# 3. PLOT
proj =  None #ccrs.Robinson() #ccrs.Mercator() #ccrs.Robinson()

# ==============================================================================

# Loading domain geometry
ds1 = xr.open_dataset(BATHY_file1)
ds2 = xr.open_dataset(BATHY_file2)

# Plotting BATHYMETRY ----------------------------------------------------------

env1 = ds1["hbatt_2"]#.isel(x_c=slice(1, None), y_c=slice(1, None))
env2 = ds2["hbatt_2"]
env = env1-env2

fig_name = 'diff_envelope_2.png'
fig_path = "./"
lon = None #ds_dom["glamf"]
lat = None #ds_dom["gphif"]
colmap = "jet" #cmocean.cm.ice
vmin = -np.nanmax(np.absolute(env))
vmax =  np.nanmax(np.absolute(env))
cbar_extend = 'max' #"max"
cbar_label = "Depth [$m$]"
cbar_hor = 'horizontal'
map_lims = [0, 1441, 0, 1206]
cn_lev = None 

plot_hpge(fig_name, fig_path, lon, lat, env, proj, colmap, 
          vmin, vmax, cbar_extend, cbar_label, cbar_hor, map_lims, env, cn_lev)

