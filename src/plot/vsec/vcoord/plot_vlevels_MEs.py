#!/usr/bin/env python

import os
import sys
import subprocess
import numpy as np
import xarray as xr
from xnemogcm import open_domain_cfg
from plot_section import mpl_sec_loop
from utils import compute_masks

# ========================================================================
# INPUT PARAMETERS

# 1. Input files

DOMCFG_MEs = '/data/users/dbruciaf/SE-NEMO/se-orca025/se-nemo-domain_cfg/domain_cfg_MEs.nc'
LOCAL_area = '/data/users/dbruciaf/SE-NEMO/se-orca025/MEs_450-800_3200/bathymetry.loc_area.dep3200_polglo_sig3_itr1.MEs_4env_450_018-010-010_opt_v2_glo.nc'
#LOCAL_area = '/data/users/dbruciaf/SE-NEMO/se-orca025/MEs_450-800_3200/bathymetry.loc_area.dep3200_polant_sig3_itr1.MEs_3env_800_018-010_opt_v3_ant.nc'

# 2. ANALYSIS cross-sections

# ANTARCTICA
#sec_lon1 = [830, 963]
#sec_lat1 = [341, 334]
sec_lon1 = [-59.13, -30.91]
sec_lat1 = [-77.16, -67.18]
#sec_lon2 = [911, 1005]
#sec_lat2 = [207, 329]
#sec_lon3 = [434, 433]
#sec_lat3 = [160, 305]
#sec_lon4 = [833, 807]
#sec_lat4 = [256, 324]

# GLOBAL
sec_lon2 = [ -5.29, -19.56] # NW E shelf
sec_lat2 = [ 50.24,  50.32]
sec_lon3 = [121.85, 105.57] # Australia
sec_lat3 = [-20.29, -17.57]
sec_lon4 = [120.53, 134.99] # China
sec_lat4 = [ 36.97,  24.12]
#sec_lon5 = [819, 882]
#sec_lat5 = [824, 795]
#sec_lon6 = [778, 789]
#sec_lat6 = [811, 770]
#sec_lon7 = [929, 954]
#sec_lat7 = [1039, 971]
#sec_lon8 = [1061, 1123]
#sec_lat8 = [1097, 1114]
#sec_lon9 = [1123, 1101]
#sec_lat9 = [918, 888]
#sec_lon10 = [1163, 1129]
#sec_lat10 = [942, 1006]


#sec_I_indx_1b_L  = [sec_lon1]#, sec_lon2, sec_lon3, sec_lon4] 
#sec_J_indx_1b_L  = [sec_lat1]#, sec_lat2, sec_lat3, sec_lat4] 
sec_I_indx_1b_L  = [sec_lon2, sec_lon3, sec_lon4] #, sec_lon6, sec_lon7, sec_lon8, sec_lon9, sec_lon10]
sec_J_indx_1b_L  = [sec_lat2, sec_lat3, sec_lat4] #, sec_lat6, sec_lat7, sec_lat8, sec_lat9, sec_lat10]
coord_type_1b_L  = "dist"
rbat2_fill_1b_L  = "false"
xlim_1b_L        = "maxmin"    # [0., 1600.]
ylim_1b_L        = [0., 5900.] # "maxmin"
vlevel_1b_L      = 'MES'
xgrid_1b_L       = "false"

# 3. INDEXES specifying a cut of the domain
#    needed to seep up plots
i1 = 0
i2 = -1 #1440
j1 = 0
j2 = -1 #470

# ========================================================================
# Reading local-area mask
msk_mes = None
ds_msk = xr.open_dataset(LOCAL_area)
ds_msk = ds_msk.isel(x=slice(i1,i2),y=slice(j1,j2))
if "s2z_msk" in ds_msk.variables:
   msk_mes = ds_msk["s2z_msk"].values
   #msk_mes[msk_mes>0] = 1
hbatt = []
nenv = 1
while nenv > 0:
  name_env = "hbatt_"+str(nenv)
  if name_env in ds_msk.variables:
      hbatt.append(ds_msk[name_env].values)
      nenv+=1
  else:
      nenv=0
del ds_msk

if msk_mes is not None:
   for env in hbatt:
       env[msk_mes < 2] = np.nan
msk_mes[msk_mes>0] = 1

# Loading domain geometry
ds_dom  = open_domain_cfg(files=[DOMCFG_MEs])
for i in ['bathymetry','bathy_meter']:
    for dim in ['x','y']:
        ds_dom[i] = ds_dom[i].rename({dim: dim+"_c"})

# Computing masks
ds_dom = compute_masks(ds_dom, merge=True)

# Extracting only the part of the domain we need
ds_dom = ds_dom.isel(x_c=slice(i1,i2),x_f=slice(i1,i2),
                     y_c=slice(j1,j2),y_f=slice(j1,j2))

tlon2 = ds_dom["glamt"].values
tlat2 = ds_dom["gphit"].values
e3t_3 = ds_dom["e3t_0"].values
e3w_3 = ds_dom["e3w_0"].values
tmsk3 = ds_dom["tmask"].values
bathy = ds_dom["bathymetry"].values

nk = e3t_3.shape[0]
nj = e3t_3.shape[1]
ni = e3t_3.shape[2]

tlon3 = np.repeat(tlon2[np.newaxis, :, :], nk, axis=0)
tlat3 = np.repeat(tlat2[np.newaxis, :, :], nk, axis=0)

# Computing model levels' depth
tdep3 = np.zeros(shape=(nk,nj,ni))
wdep3 = np.zeros(shape=(nk,nj,ni))
wdep3[0,:,:] = 0.
tdep3[0,:,:] = 0.5 * e3w_3[0,:,:]
for k in range(1, nk):
    wdep3[k,:,:] = wdep3[k-1,:,:] + e3t_3[k-1,:,:]
    tdep3[k,:,:] = tdep3[k-1,:,:] + e3w_3[k,:,:]

proj = []

# PLOTTING VERTICAL DOMAIN

var_strng  = ""
unit_strng = ""
date       = ""
timeres_dm = ""
timestep   = []
PlotType   = ""
var4       = []
#hbatt      = []
mbat_ln    = "false"
mbat_fill  = "true"
varlim     = "no"
check      = 'false'
check_val  = 'false'


mpl_sec_loop('ORCA025-locMEs mesh', '.png', var_strng, unit_strng, date, timeres_dm, timestep, PlotType,
              sec_I_indx_1b_L, sec_J_indx_1b_L, tlon3, tlat3, tdep3, wdep3, tmsk3, var4, proj,
              coord_type_1b_L, vlevel_1b_L, bathy, hbatt, rbat2_fill_1b_L, mbat_ln, mbat_fill,
              xlim_1b_L, ylim_1b_L, varlim, check, check_val, xgrid_1b_L, msk_mes=msk_mes)


