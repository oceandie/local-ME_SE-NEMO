#!/usr/bin/env pycnd2

import os
import subprocess
import numpy as np
import netCDF4 as nc4
from mpl_toolkits.basemap import Basemap
import NEMOpy.tools.common      as com
import NEMOpy.tools.math        as mat
import NEMOpy.domain.hgrid      as hgrd
import NEMOpy.iom.namelist      as nml
import NEMOpy.iom.iom           as iom
import NEMOpy.iom.initcond      as ini
import NEMOpy.plot.plot_map     as pmap
import plot_section as psec
import NEMOpy.plot.plot_profile as ppro

#sec_I_indx_1b_L  = [-1]*len(range(0,1200,100)) + range(0,1400,100)
#sec_J_indx_1b_L  = range(0,1200,100) + len(range(0,1400,100))*[-1]
sec_I_indx_1b_L  = [-1,-1, -1]
sec_J_indx_1b_L  = [300,400,500]
coord_type_1b_L  = "dist"
rbat2_fill_1b_L  = "false"
xlim_1b_L        = "maxmin" #[16000., 19000.]
ylim_1b_L        = [0., 6210.] #[0., 3500.] #5900.] #[0, 5920]
vlevel_1b_L      = 'Z_ps'
xgrid_1b_L       = "false"

#========================================================================
# 1. READING BATHY_METER and MESH_MASK FILE AND DELETING TIME DIMENSION
#========================================================================

batyfile = "/data/users/dbruciaf/OVF/GO8-eORCA025_domain/bathymetry.eORCA025-GO6.nocanyon_byhand.nc"
meshfile = "/data/users/dbruciaf/OVF/zps_GO8/real/domain_cfg_zps.nc"

print batyfile
f_bat = nc4.Dataset(batyfile,"r")
bathy = np.array(f_bat.variables["Bathymetry"])[:,:]
varNames = f_bat.variables.keys()
hbatt = []
if "hbatt" in varNames:
   hbatt.append(np.array(f_bat.variables["hbatt"])[:,:])
if "hbatt_1" in varNames:
   hbatt.append(np.array(f_bat.variables["hbatt_1"])[:,:])
   nenv = 2
   while nenv != 0:
         name = 'hbatt_' + str(nenv)
         if name in varNames:
            nenv = nenv + 1
            hbatt.append(np.array(f_bat.variables[name])[:,:])
         else:
            nenv = 0
f_bat.close()

f_bat = nc4.Dataset(meshfile,"r")
tlon2 = np.array(f_bat.variables["glamt"])[0,:,:]
tlat2 = np.array(f_bat.variables["gphit"])[0,:,:]
e1t_2 = np.array(f_bat.variables["e1t"])[0,:,:]
e2t_2 = np.array(f_bat.variables["e2t"])[0,:,:]
e3t_3 = np.array(f_bat.variables["e3t_0"])[0,:,:,:]
e3w_3 = np.array(f_bat.variables["e3w_0"])[0,:,:,:]
tmsk2 = np.array(f_bat.variables["top_level"])[0,:,:]
tlev  = np.array(f_bat.variables["bottom_level"])[0,:,:]
f_bat.close()

nk = e3t_3.shape[0]
nj = e3t_3.shape[1]
ni = e3t_3.shape[2]

tlon3 = np.repeat(tlon2[np.newaxis, :, :], nk, axis=0)
tlat3 = np.repeat(tlat2[np.newaxis, :, :], nk, axis=0)
e2t_3 = np.repeat(e2t_2[np.newaxis, :, :], nk, axis=0)
e1t_3 = np.repeat(e1t_2[np.newaxis, :, :], nk, axis=0)

tdep3 = np.zeros(shape=(nk,nj,ni))
wdep3 = np.zeros(shape=(nk,nj,ni))
wdep3[0,:,:] = 0.
tdep3[0,:,:] = 0.5 * e3w_3[0,:,:]
for k in range(1, nk):
    wdep3[k,:,:] = wdep3[k-1,:,:] + e3t_3[k-1,:,:]
    tdep3[k,:,:] = tdep3[k-1,:,:] + e3w_3[k,:,:]

tmsk3 = np.zeros(shape=(nk,nj,ni))
tmsk3[0,:,:] = tmsk2
for j in range(nj):
    for i in range(ni):
        if tmsk3[0,j,i] == 1:
           nl = tlev[j,i]
           tmsk3[0:nl,j,i] = 1


print " Computing MERCATOR projected coordinates ..."
print ""
lower_lft_corner_lon  = np.nanmin(tlon3)# - 0.1
lower_lft_corner_lat  = np.nanmin(tlat3)# - 0.1
upper_rgt_corner_lon  = np.nanmax(tlon3)# + 0.1
upper_rgt_corner_lat  = np.nanmax(tlat3)# + 0.1

PROJ = Basemap(llcrnrlon=lower_lft_corner_lon, llcrnrlat=lower_lft_corner_lat,
               urcrnrlon=upper_rgt_corner_lon, urcrnrlat=upper_rgt_corner_lat,
               projection='merc',resolution='h')

proj = PROJ 

#========================================================================
# 3. PLOTTING VERTICAL DOMAIN
#========================================================================

print "-------------------------------------------"
print " *** PLOTTING VERTICAL DOMAIN SECTIONS *** "
print "-------------------------------------------"

var_strng  = ""
unit_strng = ""
date       = ""
timeres_dm = ""
timestep   = []
PlotType   = ""
#tlon3      = nemo_grid.tlon
#tlat3      = nemo_grid.tlat
#tdep3      = nemo_grid.tdep3
#wdep3      = nemo_grid.wdep3
#tmsk3      = nemo_grid.tmsk
var4       = []
mbat_ln    = "false"
mbat_fill  = "true"
varlim     = "no"
check      = 'true'
check_val  = 'false'

psec.mpl_sec_loop('eORCA025-zps mesh', '.png', var_strng, unit_strng, date, timeres_dm, timestep, PlotType,
                  sec_I_indx_1b_L, sec_J_indx_1b_L, tlon3, tlat3, tdep3, wdep3, tmsk3, var4, proj,
                  coord_type_1b_L, vlevel_1b_L, bathy, hbatt, rbat2_fill_1b_L, mbat_ln, mbat_fill,
                  xlim_1b_L, ylim_1b_L, varlim, check, check_val, xgrid_1b_L)


