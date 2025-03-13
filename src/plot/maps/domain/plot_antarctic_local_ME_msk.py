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
import matplotlib.colors as colors
import xarray as xr
import cartopy.crs as ccrs
import cmocean
from utils import plot_map

# ==============================================================================
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

# ==============================================================================
# Input parameters

# 1. INPUT FILES

BATHY_file = '/data/users/dbruciaf/SE-NEMO/se-orca025/MEs_300_1650/ant/r015-r010_r007_r004v2/bathymetry.MEs_4env_800_015-010_007_004v2_ant_maxdep_2600.0.nc'

proj =  None #ccrs.Robinson() #ccrs.Mercator() #ccrs.Robinson()

# ==============================================================================

# Loading domain geometry
ds  = xr.open_dataset(BATHY_file)

# Plotting BATHYMETRY ----------------------------------------------------------

bathy  = ds["Bathymetry"]
so_msk = ds["s2z_msk"]
bathy = xr.where(so_msk>0, bathy, bathy.where(bathy==0))

cmap = plt.get_cmap('hot')
new_cmap = truncate_colormap(cmap, 0., 0.9)

fig_name = 'antarctic_local-ME.png'
fig_path = "./"
lon = None 
lat = None
colmap = new_cmap #cmocean.cm.ice
vmin = 0.
vmax = 2.
cbar_extend = 'neither' #"max"
cbar_label = "Depth [$m$]"
cbar_hor = 'vertical'
map_lims = [0, 1441, 0, 1206]
cn_lev = None #[800., 2600.] 

plot_map(fig_name, fig_path, lon, lat, so_msk, proj, colmap, 
         vmin, vmax, cbar_extend, cbar_label, cbar_hor, map_lims, bathy, cn_lev)

