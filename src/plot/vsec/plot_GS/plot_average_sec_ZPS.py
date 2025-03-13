#!/usr/bin/env python

import os
import sys
import subprocess
import numpy as np
import xarray
from xnemogcm import open_domain_cfg, open_nemo
from plot_section import mpl_sec_loop
from utils import compute_masks
import cmocean
import matplotlib.colors as colors
import gsw as gsw
import nsv

# ========================================================================
# INPUT PARAMETERS

# 1. Input files

DOMCFG_zps = '/data/users/dbruciaf/SE-NEMO/se-orca025/se-nemo-domain_cfg/domain_cfg_zps.nc'
U_zps = '/data/users/dbruciaf/SE-NEMO/se-orca025/final_15-11-2024/se-nemo_ZPS_average_1991-2019_grid_U.nc'
V_zps = '/data/users/dbruciaf/SE-NEMO/se-orca025/final_15-11-2024/se-nemo_ZPS_average_1991-2019_grid_V.nc'

# 2. ANALYSIS
# Ezer 2016 sec
#sec_lon1 = [-77., -60.]
#sec_lat1 = [ 35.,  35.]
sec_lon1 = [-77., -70.]
sec_lat1 = [ 35.,  35.]
# North Cape Hatteras
sec_lon2 = [-78., -60.]
sec_lat2 = [ 38.5,  38.5]
# Schoonover thesis Fig1.3 left side
sec_lon3 = [-78., -60.]
sec_lat3 = [ 34.,  34.]

sec_I_indx_1b_L  = [sec_lon1] #, sec_lon2] #, sec_lon3]
sec_J_indx_1b_L  = [sec_lat1] #, sec_lat2] #, sec_lat3]
coord_type_1b_L  = "dist"
rbat2_fill_1b_L  = "false"
xlim_1b_L        = [0., 500.] #"maxmin" #[0., 1200.]
ylim_1b_L        = [0., 3000.] # 5500.
vlevel_1b_L      = 'Z_ps'
xgrid_1b_L       = "false"

# ========================================================================

# Loading domain geometry
ds_dom  = open_domain_cfg(files=[DOMCFG_zps])
for i in ['bathy_metry']:
    for dim in ['x','y']:
        ds_dom[i] = ds_dom[i].rename({dim: dim+"_c"})

# Computing masks
ds_dom = compute_masks(ds_dom, merge=True)

# Loading NEMO files
#ds_U = open_nemo(ds_dom, files=[U_zps])
ds_V = open_nemo(ds_dom, files=[V_zps])

# Extracting only the part of the domain we need
ds_dom = ds_dom.isel(x_c=slice(730,2010),x_f=slice(730,2010),
                     y_c=slice(760,2500),y_f=slice(760,2500))
#ds_U =  ds_U.isel(x_f=slice(730,2010),y_c=slice(760,2500))
ds_V =  ds_V.isel(x_c=slice(730,2010),y_f=slice(760,2500))

tlon2 = ds_dom["glamt"].values
tlat2 = ds_dom["gphit"].values
e3t_3 = ds_dom["e3t_0"].values
e3w_3 = ds_dom["e3w_0"].values
tmsk3 = ds_dom["tmask"].values
bathy = ds_dom["bathy_metry"].values

nk = e3t_3.shape[0]
nj = e3t_3.shape[1]
ni = e3t_3.shape[2]

tlon3 = np.repeat(tlon2[np.newaxis, :, :], nk, axis=0)
tlat3 = np.repeat(tlat2[np.newaxis, :, :], nk, axis=0)

tdep3 = np.zeros(shape=(nk,nj,ni))
wdep3 = np.zeros(shape=(nk,nj,ni))
wdep3[0,:,:] = 0.
tdep3[0,:,:] = 0.5 * e3w_3[0,:,:]
for k in range(1, nk):
    wdep3[k,:,:] = wdep3[k-1,:,:] + e3t_3[k-1,:,:]
    tdep3[k,:,:] = tdep3[k-1,:,:] + e3w_3[k,:,:]

# Computing speed
#da_u = ds_U['uo']
da_v = ds_V['vo']
#da_u = da_u.rolling({'x_f':2}).mean().fillna(0.)
da_v = da_v.rolling({'y_f':2}).mean().fillna(0.)

proj = []

# CURRENT PLOTS

#CMAP = colors.ListedColormap(['purple','indigo','darkblue',
#                              'dodgerblue','deepskyblue','lightskyblue',
#                              'mediumspringgreen','lime','greenyellow','yellow','gold','orange',
#                              'darkorange','orangered','red','firebrick','darkred','gray'])

var_strng  = "Current"
unit_strng = "[$m\;s^{-1}$]"
date       = ""
timeres_dm = "1d"
timestep   = [0]
PlotType   = "contourf"
colmap     = cmocean.cm.curl
cn_level   = np.arange(-0.50, 0.55, 0.05)
cn_line    = "true"
mbat_ln    = "false"
mbat_fill  = "true"
varlim     = [-0.5, 0.5]
check      = 'false'
check_val  = 'false'

mpl_sec_loop('Current', '.png', var_strng, unit_strng, date, timeres_dm, timestep, PlotType,
              sec_I_indx_1b_L, sec_J_indx_1b_L, tlon3, tlat3, tdep3, wdep3, tmsk3, da_v.values, proj,
              coord_type_1b_L, vlevel_1b_L, bathy, [], rbat2_fill_1b_L, mbat_ln, mbat_fill,
              xlim_1b_L, ylim_1b_L, varlim, check, check_val, xgrid_1b_L, 
              colmap=colmap, cn_level=cn_level, cn_line=cn_line, cn_label="false")
