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

# Nordic ovf
sec_lon1 = [ 0.34072625, -3.56557722,-18.569585  ,-26.42872351, -30.314948]
sec_lat1 = [68.26346438, 65.49039963, 60.79252542, 56.24488972,  52.858934]
sec_lon2 = [-10.84451672, -25.30818606, -35.61730763, -44.081319]
sec_lat2 = [ 71.98049514,  66.73449533,  61.88833838,  56.000932]

# Gulf Stream
sec_lon3 = [-82.56, -66.82]
sec_lat3 = [ 32.82,  26.69]
#sec_lon4 = [-76.32, -68.90]
#sec_lat4 = [ 35.88,  34.40]

# Florida-Carribean Sea
sec_lon4 = [-80.70, -79.45, -76.67]
sec_lat4 = [ 25.29,  16.80,   8.55]

# Gulf of Mexico
sec_lon5 = [-92.74, -90.01]
sec_lat5 = [ 30.10,  21.07]

# North Sea
sec_lon6 = [ 5.62, -0.02]
sec_lat6 = [53.23, 63.33]

# NWS
sec_lon7 = [-5.56, -11.65]
sec_lat7 = [50.28,  45.46]
sec_lon8 = [-10.32, -22.39]
sec_lat8 = [ 52.42,  49.71]

# Norwegian Sea
sec_lon9 = [-20.68, 14.30]
sec_lat9 = [ 73.98, 67.15]
sec_lon10 = [-19.10, 15.53]
sec_lat10 = [ 77.25, 77.25]

# Labrador Sea
sec_lon11 = [-63.05, -51.09]
sec_lat11 = [ 66.11,  58.38]
sec_lon12 = [-63.05, -47.75]
sec_lat12 = [ 57.73,  61.71]

# Canada Atlantic Ocean
sec_lon13 = [-65.02, -35.98]
sec_lat13 = [ 48.84,  44.45]

# Bering Sea
sec_lon14 = [-173.96, -173.39]
sec_lat14 = [  64.97,   39.06]

# North Pacific Ocean - Alaska
sec_lon15 = [-148.78, -141.75]
sec_lat15 = [  61.34,   54.65]

# North Pacific Ocean - Alaska

#sec_I_indx_1b_L  = [-1]*len(range(0,1200,100)) + range(0,1400,100)
#sec_J_indx_1b_L  = range(0,1200,100) + len(range(0,1400,100))*[-1]
sec_I_indx_1b_L  = [sec_lon1, sec_lon2, sec_lon3, sec_lon4,sec_lon5, sec_lon6, sec_lon7, sec_lon8, sec_lon9, sec_lon10, sec_lon11, sec_lon12, sec_lon13, sec_lon14, sec_lon15]
sec_J_indx_1b_L  = [sec_lat1, sec_lat2, sec_lat3, sec_lat4,sec_lat5, sec_lat6, sec_lat7, sec_lat8, sec_lat9, sec_lat10, sec_lat11, sec_lat12, sec_lat13, sec_lat14, sec_lat15]
coord_type_1b_L  = "dist"
rbat2_fill_1b_L  = "false"
xlim_1b_L        = "maxmin" #[16000., 19000.]
ylim_1b_L        = [0, 6000] #[0, 3500] #5900 #3500] # 800
vlevel_1b_L      = 'MES'
xgrid_1b_L       = "false"

#========================================================================
# 1. READING BATHY_METER and MESH_MASK FILE AND DELETING TIME DIMENSION
#========================================================================
#batyfile = "/data/users/dbruciaf/SE-NEMO/se-orca025/MEs_250_1500/bathymetry.MEs_4env_250_maxdep_1500.0.nc"
#meshfile = "/data/users/dbruciaf/SE-NEMO/se-orca025/MEs_250_1500/mesh_mask_36-14-7-18.nc"
batyfile = "/data/users/dbruciaf/SE-NEMO/se-orca025/MEs_300_1650/bathymetry.MEs_4env_300_glo_maxdep_1650.0.nc"
meshfile = "/data/users/dbruciaf/SE-NEMO/se-orca025/MEs_300_1650/mesh_mask.nc"

print batyfile
f_bat = nc4.Dataset(batyfile,"r")
bathy = np.array(f_bat.variables["Bathymetry"])[:,:]
varNames = f_bat.variables.keys()
hbatt = []
#if "hbatt" in varNames:
#   hbatt.append(np.array(f_bat.variables["hbatt"])[:,:])
#if "hbatt_1" in varNames:
#   hbatt.append(np.array(f_bat.variables["hbatt_1"])[:,:])
#   nenv = 2
#   while nenv != 0:
#         name = 'hbatt_' + str(nenv)
#         if name in varNames:
#            nenv = nenv + 1
#            hbatt.append(np.array(f_bat.variables[name])[:,:])
#         else:
#            nenv = 0
f_bat.close()

#nemo_grid = iom.read_nemo_mesh(meshfile,coorfile)
nemo_grid = iom.read_nemo_mesh(meshfile)

print " Computing MERCATOR projected coordinates ..."
print ""
#lower_lft_corner_lon  = np.nanmin(nemo_grid.tlon) - 0.1
lower_lft_corner_lon  = np.nanmin(nemo_grid.tlon)
lower_lft_corner_lat  = np.nanmin(nemo_grid.tlat)
upper_rgt_corner_lon  = np.nanmax(nemo_grid.tlon)
upper_rgt_corner_lat  = np.nanmax(nemo_grid.tlat)

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
tlon3      = nemo_grid.tlon
tlat3      = nemo_grid.tlat
tdep3      = nemo_grid.tdep3
wdep3      = nemo_grid.wdep3
tmsk3      = nemo_grid.tmsk
var4       = []
mbat_ln    = "false"
mbat_fill  = "true"
varlim     = "no"
check      = 'true'
check_val  = 'false'

psec.mpl_sec_loop('eORCA025-MEs mesh', '.png', var_strng, unit_strng, date, timeres_dm, timestep, PlotType,
                  sec_I_indx_1b_L, sec_J_indx_1b_L, tlon3, tlat3, tdep3, wdep3, tmsk3, var4, proj,
                  coord_type_1b_L, vlevel_1b_L, bathy, hbatt, rbat2_fill_1b_L, mbat_ln, mbat_fill,
                  xlim_1b_L, ylim_1b_L, varlim, check, check_val, xgrid_1b_L)


