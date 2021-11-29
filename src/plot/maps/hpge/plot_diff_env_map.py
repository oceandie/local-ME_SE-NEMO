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
from utils import plot_hpge, plot_env

# ==============================================================================
# Input parameters

# 1. INPUT FILES

vcoord = 'r12_r12_vs_r12_r12-r075-r040_v3'
ENV_file1 = '/data/users/dbruciaf/OVF/MEs_GO8/env4.v2.maxdep/2800/r12_r12/bathymetry.MEs_4env_2800_r12_r12_maxdep_2800.0.nc'
ENV_file2 = '/data/users/dbruciaf/OVF/MEs_GO8/env4.v2.maxdep/2800/r12_r12-r075-r040_v3/bathymetry.MEs_4env_2800_r12_r12-r075-r040_v3_maxdep_2800.0.nc'

# 3. PLOT
lon0 = -45.
lon1 =  5.0
lat0 =  53.
lat1 =  72.
proj = ccrs.Mercator() #ccrs.Robinson()

# ==============================================================================

ds_env1 = xr.open_dataset(ENV_file1)
ds_env2 = xr.open_dataset(ENV_file2)

#ds_env = xr.open_dataset(ENV_file)
ds_env = ds_env1 - ds_env2

# Extracting only the part of the domain we need

ds_env =  ds_env.isel(x=slice(880,1200),y=slice(880,1140))
ds_env1 = ds_env1.isel(x=slice(880,1200),y=slice(880,1140))

# Plotting envelope ----------------------------------------------------------

env = "hbatt_3"

fig_name = env + '_' + vcoord + '.png'
fig_path = "./"
lon = ds_env["nav_lon"]
lat = ds_env["nav_lat"]
var = ds_env[env] 
colmap = cmocean.cm.deep
#colmap = ''
#vmin = 150.0
#vmax = 2800
vmin = -840
vmax = 0
cbar_extend = 'max' #"max"
cbar_label = "Depth [$m$]"
cbar_hor = 'vertical'
map_lims = [lon0, lon1, lat0, lat1]
bathy = ds_env1.Bathymetry
cn_lev = [0., 250., 500., 1000., 1500, 2000., 3000., 4000.]

plot_hpge(fig_name, fig_path, lon, lat, var, proj, colmap, 
          vmin, vmax, cbar_extend, cbar_label, cbar_hor, map_lims, bathy, cn_lev)

 
