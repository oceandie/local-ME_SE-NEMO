#!/usr/bin/env python

import os
import sys
import subprocess
import numpy as np
import xarray as xr
from xnemogcm import open_domain_cfg, open_nemo
from plot_section import mpl_sec_loop
from utils import compute_masks
from matplotlib import pyplot as plt
import matplotlib.colors as colors

# ========================================================================
# INPUT PARAMETERS

# 1. Input files

DOMCFG_MEs = '/data/users/dbruciaf/OVF/szT_GO8/ideal/domain_cfg_szt_p1_L48_mod_ideal.nc'
LOCAL_area = '/data/users/dbruciaf/OVF/MEs_GO8/env4.v2.maxdep/2800/r12_r12-r075-r040_v3/bathymetry.loc_area.dep2800_novf_sig1_stn9_itr1.MEs_4env_2800_r12_r12-r075-r040_v3.nc'
inp_dir = '/data/users/dbruciaf/OVF/outputs/ideal/lin_str_ldf/'
TRACER_list = [inp_dir + '/szt/nemo_cg602o_1d_19760101-19760201_grid_T.nc',
               inp_dir + '/szt/nemo_cg602o_1d_19760201-19760301_grid_T.nc',
               inp_dir + '/szt/nemo_cg602o_1d_19760301-19760401_grid_T.nc']

# 2. ANALYSIS cross-sections

tra_lim = 0.1 # minimum passive tracer [] -> to identify dense plume 
sec_lon1 = [-24.6384, -27.8618, -31.868 , -33.3767]
sec_lat1 = [ 66.73  ,  65.7171,  64.7748,  62.1913]

sec_I_indx_1b_L  = [sec_lon1]
sec_J_indx_1b_L  = [sec_lat1]
coord_type_1b_L  = "dist"
rbat2_fill_1b_L  = "false"
xlim_1b_L        = [100., 750.]
ylim_1b_L        = [0., 3000.]
vlevel_1b_L      = 'SZT'
xgrid_1b_L       = "false"
first_zlv        = 49

var_strng  = "Tracer"
unit_strng = "[g/kg]"
date       = ""
timeres_dm = "1d"
timestep   = range(89)
PlotType   = "pcolor"
cn_level   = [0.1,0.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.,5.5,6.,6.5,7.,8.,9.,10]
cn_line    = "false"
mbat_ln    = "false"
mbat_fill  = "true"
varlim     = [0.1,10]
check      = 'true'
check_val  = 'false'

cmap = plt.get_cmap('jet')
CMAP = colors.ListedColormap(cmap(np.linspace(0,1.,len(cn_level))))

# 3. INDEXES specifying a cut of the domain
#    needed to seep up plots
i1 = 880
i2 = 1150
j1 = 880
j2 = 1140

# ========================================================================
# Reading local-MEs mask
msk_mes = None
ds_msk = xr.open_dataset(LOCAL_area)
ds_msk = ds_msk.isel(x=slice(i1,i2),y=slice(j1,j2))
if "s2z_msk" in ds_msk.variables:
   msk_mes = ds_msk["s2z_msk"].values
   msk_mes[msk_mes>0] = 1
del ds_msk

# Loading domain geometry
ds_dom  = open_domain_cfg(files=[DOMCFG_MEs])
for i in ['bathymetry','bathy_meter']:
    for dim in ['x','y']:
        ds_dom[i] = ds_dom[i].rename({dim: dim+"_c"})

# Computing masks
ds_dom = compute_masks(ds_dom, merge=True)

# Loading NEMO files
ds_T = open_nemo(ds_dom, files=TRACER_list)

# Extracting only the part of the domain we need
ds_dom = ds_dom.isel(x_c=slice(i1,i2),x_f=slice(i1,i2),
                     y_c=slice(j1,j2),y_f=slice(j1,j2))
ds_T =  ds_T.isel(x_c=slice(i1,i2),y_c=slice(j1,j2))

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

da_tra = ds_T['so_seos'] - 20. # environment [] of the passive tracer
da_tra = da_tra.where(da_tra > tra_lim)
tra4 = da_tra.values

proj = []

mpl_sec_loop('Passive tracer conc.', '.png', var_strng, unit_strng, date, timeres_dm, timestep, PlotType,
              sec_I_indx_1b_L, sec_J_indx_1b_L, tlon3, tlat3, tdep3, wdep3, tmsk3, tra4, proj,
              coord_type_1b_L, vlevel_1b_L, bathy, [], rbat2_fill_1b_L, mbat_ln, mbat_fill,
              xlim_1b_L, ylim_1b_L, varlim, check, check_val, xgrid_1b_L, 
              colmap=CMAP, cn_level=cn_level, cn_line=cn_line, msk_mes=msk_mes, first_zlv=first_zlv)
