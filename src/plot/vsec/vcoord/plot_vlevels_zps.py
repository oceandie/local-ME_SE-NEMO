#!/usr/bin/env python

import os
import sys
import subprocess
import numpy as np
import xarray
from xnemogcm import open_domain_cfg
from plot_section import mpl_sec_loop
from utils import compute_masks

# ========================================================================
# INPUT PARAMETERS

# 1. Input files

DOMCFG_zps = '/data/users/dbruciaf/OVF/zps_GO8/real/domain_cfg_zps.nc'

# 2. ANALYSIS cross-sections

# Iceland-Faroe Ridge
sec_lon1 = [ 0.34072625, -3.56557722,-18.569585  ,-26.42872351, -30.314948]
sec_lat1 = [68.26346438, 65.49039963, 60.79252542, 56.24488972,  52.858934]

# Denmark Strait
sec_lon2 = [-10.84451672, -25.30818606, -35.61730763, -44.081319]
sec_lat2 = [ 71.98049514,  66.73449533,  61.88833838,  56.000932]
sec_lon3 = [-33.4446, -24.0055]
sec_lat3 = [ 67.6902,  65.5927]
sec_lon4 = [-29.8692, -24.0002]
sec_lat4 = [ 68.4979,  65.8794]

# Pierre-like section
sec_lon5 = [-34.8650, -35.6696, -6.094913]
sec_lat5 = [ 62.6496,  66.1957, 56.283842]

# Mattia-like section
sec_lon6 = [-27.70520729, -28.07997397, -28.45969688, -28.85371389,
            -29.20340174, -29.45701538, -29.81805224, -30.2004988 ,
            -30.62412978, -31.0539745 , -31.48476823, -31.90946898,
            -32.32042909, -32.68453171, -32.99092802, -33.17180383,
            -33.33201739, -33.5831037 , -33.94562868, -34.26654917,
            -34.54165955, -34.86276303, -35.1888309 , -35.49957294,
            -35.76996098, -36.09467135, -36.44359909, -36.80504419,
            -37.09818747]
sec_lat6 = [66.03902481, 65.94587003, 65.85452313, 65.77525271, 65.66862967,
            65.52229975, 65.42612706, 65.34899946, 65.31453947, 65.30626256,
            65.31424516, 65.28674186, 65.2342894 , 65.14096325, 65.01632524,
            64.85612429, 64.69301328, 64.54988437, 64.46799526, 64.35972094,
            64.2274042 , 64.11548858, 64.0059749 , 63.88921957, 63.75946712,
            63.65173953, 63.56142445, 63.48258579, 63.35948396]

sec_I_indx_1b_L  = [sec_lon1, sec_lon2, sec_lon3, sec_lon4, sec_lon5, sec_lon6]
sec_J_indx_1b_L  = [sec_lat1, sec_lat2, sec_lat3, sec_lat4, sec_lat5, sec_lat6]
coord_type_1b_L  = "dist"
rbat2_fill_1b_L  = "false"
xlim_1b_L        = "maxmin"    # [0., 1600.]
ylim_1b_L        = [0., 3500.] # "maxmin"
vlevel_1b_L      = 'Z_ps'
xgrid_1b_L       = "false"

# 3. INDEXES specifying a cut of the domain
#    needed to seep up plots
i1 = 880
i2 = 1150
j1 = 880
j2 = 1140

# ========================================================================

# Loading domain geometry
ds_dom  = open_domain_cfg(files=[DOMCFG_zps])
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
hbatt      = []
mbat_ln    = "false"
mbat_fill  = "true"
varlim     = "no"
check      = 'true'
check_val  = 'false'

mpl_sec_loop('ORCA025-zps mesh', '.png', var_strng, unit_strng, date, timeres_dm, timestep, PlotType,
              sec_I_indx_1b_L, sec_J_indx_1b_L, tlon3, tlat3, tdep3, wdep3, tmsk3, var4, proj,
              coord_type_1b_L, vlevel_1b_L, bathy, hbatt, rbat2_fill_1b_L, mbat_ln, mbat_fill,
              xlim_1b_L, ylim_1b_L, varlim, check, check_val, xgrid_1b_L)

