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
from xnemogcm import open_domain_cfg, open_nemo
import cartopy.crs as ccrs
import cmocean
import gsw as gsw
from utils import plot_bot_plume, compute_masks, e3_to_dep 

# ------------------------------------------------------------------------------------------
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
# -------------------------------------------------------------------------------------------
class TwoInnerPointsNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, low=None, up=None, clip=False):
        self.low = low
        self.up = up
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.low, self.up, self.vmax], [0, 0.25, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
# ==============================================================================

# Input parameters

# 1. INPUT FILES

DOMCFG_zps = '/data/users/dbruciaf/SE-NEMO/se-orca025/se-nemo-domain_cfg/domain_cfg_zps.nc'
DOMCFG_MEs = '/data/users/dbruciaf/SE-NEMO/se-orca025/se-nemo-domain_cfg/domain_cfg_MEs.nc'

# 3. PLOT
lon0 = -100.
lon1 = -5 #-40.
lat0 =  20.
lat1 =  61.
proj = ccrs.Mercator() #ccrs.Robinson()

# COLORBAR
cmap    = plt.get_cmap('afmhot_r')
newcmap = truncate_colormap(cmap, 0.0, 0.8)
col_dp  = newcmap(np.linspace(0,1.,128))
newcmap = truncate_colormap(cmocean.cm.ice, 0.1, 1.0)
#col_sh  = cmocean.cm.ice(np.linspace(0,1.,128))
col_sh  = newcmap(np.linspace(0,1.,128))
col     = list(zip(np.linspace(0,0.5,128),col_sh))
col    += list(zip(np.linspace(0.5,1.0,128),col_dp))
CMAP   = colors.LinearSegmentedColormap.from_list('mycmap', col)
norm = TwoInnerPointsNormalize(vmin=0, vmax=0.5, low=0.1, up=0.2)

# ==============================================================================

# Loading domain geometry
ds_dom_zps  = open_domain_cfg(files=[DOMCFG_zps])
for i in ['bathy_metry']:
    for dim in ['x','y']:
        ds_dom_zps[i] = ds_dom_zps[i].rename({dim: dim+"_c"})

ds_dom_MEs  = open_domain_cfg(files=[DOMCFG_MEs])
for i in ['bathymetry','bathy_meter']:
    for dim in ['x','y']:
        ds_dom_MEs[i] = ds_dom_MEs[i].rename({dim: dim+"_c"})

U_ZPS_EN = '/data/users/dbruciaf/EN4/interp_GO8_zps/geo_adj/nemo_cv530o_10d_19760322-19760401_grid_U.nc'
U_ZPS_NT = '/data/users/dbruciaf/SE-NEMO/se-orca025/GS1p0_notide/SENEMO_1991-2019_average_grid_U.nc'
U_ZPS_TD = '/data/users/dbruciaf/SE-NEMO/se-orca025/GS1p1_tide/SENEMO_1991-2019_average_grid_U.nc'
U_MES_TD = '/data/users/dbruciaf/SE-NEMO/se-orca025/GS1p2_full/SENEMO_1991-2019_average_grid_U.nc'
V_ZPS_EN = '/data/users/dbruciaf/EN4/interp_GO8_zps/geo_adj/nemo_cv530o_10d_19760322-19760401_grid_V.nc'
V_ZPS_NT = '/data/users/dbruciaf/SE-NEMO/se-orca025/GS1p0_notide/SENEMO_1991-2019_average_grid_V.nc'
V_ZPS_TD = '/data/users/dbruciaf/SE-NEMO/se-orca025/GS1p1_tide/SENEMO_1991-2019_average_grid_V.nc'
V_MES_TD = '/data/users/dbruciaf/SE-NEMO/se-orca025/GS1p2_full/SENEMO_1991-2019_average_grid_V.nc'

# Loading NEMO files
ds_U_zps_en = open_nemo(ds_dom_zps, files=[U_ZPS_EN])
ds_U_zps_nt = open_nemo(ds_dom_zps, files=[U_ZPS_NT])
ds_U_zps_td = open_nemo(ds_dom_zps, files=[U_ZPS_TD])
ds_U_MEs_td = open_nemo(ds_dom_MEs, files=[U_MES_TD])
ds_V_zps_en = open_nemo(ds_dom_zps, files=[V_ZPS_EN])
ds_V_zps_nt = open_nemo(ds_dom_zps, files=[V_ZPS_NT])
ds_V_zps_td = open_nemo(ds_dom_zps, files=[V_ZPS_TD])
ds_V_MEs_td = open_nemo(ds_dom_MEs, files=[V_MES_TD])

# Extracting only the part of the domain we need
ds_U_zps_en =  ds_U_zps_en.isel(x_f=slice(730,2010),y_c=slice(760,2500))
ds_U_zps_nt =  ds_U_zps_nt.isel(x_f=slice(730,2010),y_c=slice(760,2500))
ds_U_zps_td =  ds_U_zps_td.isel(x_f=slice(730,2010),y_c=slice(760,2500))
ds_U_MEs_td =  ds_U_MEs_td.isel(x_f=slice(730,2010),y_c=slice(760,2500))
ds_V_zps_en =  ds_V_zps_en.isel(x_c=slice(730,2010),y_f=slice(760,2500))
ds_V_zps_nt =  ds_V_zps_nt.isel(x_c=slice(730,2010),y_f=slice(760,2500))
ds_V_zps_td =  ds_V_zps_td.isel(x_c=slice(730,2010),y_f=slice(760,2500))
ds_V_MEs_td =  ds_V_MEs_td.isel(x_c=slice(730,2010),y_f=slice(760,2500))

# Extracting only the part of the domain we need
ds_dom_zps = ds_dom_zps.isel(x_c=slice(730,2010),x_f=slice(730,2010),
                             y_c=slice(760,2500),y_f=slice(760,2500))
ds_dom_MEs = ds_dom_MEs.isel(x_c=slice(730,2010),x_f=slice(730,2010),
                             y_c=slice(760,2500),y_f=slice(760,2500))

ds_dom_zps = compute_masks(ds_dom_zps, merge=True)
ds_dom_MEs = compute_masks(ds_dom_MEs, merge=True)

# Computing model levels depth
gdepw_zps, gdept_zps = e3_to_dep(ds_dom_zps.e3w_0, ds_dom_zps.e3t_0)
gdepw_MEs, gdept_MEs = e3_to_dep(ds_dom_MEs.e3w_0, ds_dom_MEs.e3t_0)
e1t_zps = ds_dom_zps["e1t"]
e2t_zps = ds_dom_zps["e2t"]
e3t_zps = ds_dom_zps["e3t_0"]
e1t_MEs = ds_dom_MEs["e1t"]
e2t_MEs = ds_dom_MEs["e2t"]
e3t_MEs = ds_dom_MEs["e3t_0"]

# Identifying model cells shallower than threshold
lim = 200. # m

U_zps_en = ds_U_zps_en['uo']
U_zps_en = U_zps_en.rename({'x_f':'x_c'})
U_zps_en = U_zps_en.assign_coords({"x_c": e1t_zps['x_c'].data})

U_zps_nt = ds_U_zps_nt['uo']
U_zps_nt = U_zps_nt.rename({'x_f':'x_c'})
U_zps_nt = U_zps_nt.assign_coords({"x_c": e1t_zps['x_c'].data})

U_zps_td = ds_U_zps_td['uo']
U_zps_td = U_zps_td.rename({'x_f':'x_c'})
U_zps_td = U_zps_td.assign_coords({"x_c": e1t_zps['x_c'].data})

U_MEs_td = ds_U_MEs_td['uo']
U_MEs_td = U_MEs_td.rename({'x_f':'x_c'})
U_MEs_td = U_MEs_td.assign_coords({"x_c": e1t_MEs['x_c'].data})

V_zps_en = ds_V_zps_en['vo']
V_zps_en = V_zps_en.rename({'y_f':'y_c'})
V_zps_en = V_zps_en.assign_coords({"y_c": e1t_zps['y_c'].data})

V_zps_nt = ds_V_zps_nt['vo']
V_zps_nt = V_zps_nt.rename({'y_f':'y_c'})
V_zps_nt = V_zps_nt.assign_coords({"y_c": e1t_zps['y_c'].data})

V_zps_td = ds_V_zps_td['vo']
V_zps_td = V_zps_td.rename({'y_f':'y_c'})
V_zps_td = V_zps_td.assign_coords({"y_c": e1t_zps['y_c'].data})

V_MEs_td = ds_V_MEs_td['vo']
V_MEs_td = V_MEs_td.rename({'y_f':'y_c'})
V_MEs_td = V_MEs_td.assign_coords({"y_c": e1t_MEs['y_c'].data})

U_zps_en = U_zps_en.where(gdept_zps<=lim)
U_zps_nt = U_zps_nt.where(gdept_zps<=lim)
U_zps_td = U_zps_td.where(gdept_zps<=lim)
U_MEs_td = U_MEs_td.where(gdept_MEs<=lim)
V_zps_en = V_zps_en.where(gdept_zps<=lim)
V_zps_nt = V_zps_nt.where(gdept_zps<=lim)
V_zps_td = V_zps_td.where(gdept_zps<=lim)
V_MEs_td = V_MEs_td.where(gdept_MEs<=lim)

e1t_zps = e1t_zps.where(gdept_zps<=lim)
e2t_zps = e2t_zps.where(gdept_zps<=lim)
e3t_zps = e3t_zps.where(gdept_zps<=lim)
e1t_MEs = e1t_MEs.where(gdept_MEs<=lim)
e2t_MEs = e2t_MEs.where(gdept_MEs<=lim)
e3t_MEs = e3t_MEs.where(gdept_MEs<=lim)

# COMPUTING TRANSPORT

# 1) zps_en
U_tra_zps_en = (U_zps_en * e2t_zps * e3t_zps).sum(dim='z_c', skipna=True) * 1e-6 # in Sv
V_tra_zps_en = (V_zps_en * e1t_zps * e3t_zps).sum(dim='z_c', skipna=True) * 1e-6 # in Sv
C_zps_en = np.sqrt(U_tra_zps_en**2 + V_tra_zps_en**2).squeeze()
print(np.nanmax(C_zps_en)) 

# 2) zps_nt
U_tra_zps_nt = (U_zps_nt * e2t_zps * e3t_zps).sum(dim='z_c', skipna=True) * 1e-6 # in Sv
V_tra_zps_nt = (V_zps_nt * e1t_zps * e3t_zps).sum(dim='z_c', skipna=True) * 1e-6 # in Sv
C_zps_nt = np.sqrt(U_tra_zps_nt**2 + V_tra_zps_nt**2).squeeze()

# 3) zps_td
U_tra_zps_td = (U_zps_td * e2t_zps * e3t_zps).sum(dim='z_c', skipna=True) * 1e-6 # in Sv
V_tra_zps_td = (V_zps_td * e1t_zps * e3t_zps).sum(dim='z_c', skipna=True) * 1e-6 # in Sv
C_zps_td = np.sqrt(U_tra_zps_td**2 + V_tra_zps_td**2).squeeze()

# 4) MEs
U_tra_MEs_td = (U_MEs_td * e2t_MEs * e3t_MEs).sum(dim='z_c', skipna=True) * 1e-6 # in Sv
V_tra_MEs_td = (V_MEs_td * e1t_MEs * e3t_MEs).sum(dim='z_c', skipna=True) * 1e-6 # in Sv
C_MEs_td = np.sqrt(U_tra_MEs_td**2 + V_tra_MEs_td**2).squeeze()

print(' computing transport:  done')

# PLOT =============================================================================
fig_path = "./"
colmap = 'afmhot' #'nipy_spectral'# 'RdBu_r'
cbar_extend = "both"
cn_lev = [500.]#, 1500, 4000.]
cbar_hor = 'horizontal'
map_lims = [lon0, lon1, lat0, lat1]

# 1. Plotting zps
bathy = ds_dom_zps["bathy_metry"]
lon = ds_dom_zps["glamf"]
lat = ds_dom_zps["gphif"]

vmin =  0.0
vmax =  1.5
vstp =  0.1
cbar_label = "Vol. transport upper" + str(lim) + "[$Sv$]"
vcor = 'zps_en4'

print('  T')
fig_name = "vol_tra_"+vcor+'_'+str(lim)+'.png'

plot_bot_plume(fig_name, fig_path, lon, lat, C_zps_en, proj,
               colmap, vmin, vmax, vstp, cbar_extend, cbar_label,
               cbar_hor, map_lims, bathy=bathy, cn_lev=cn_lev, land='gray')

# 2. Plotting zps
bathy = ds_dom_zps["bathy_metry"]
lon = ds_dom_zps["glamf"]
lat = ds_dom_zps["gphif"]
vcor = 'zps_notide'

print('  T')
fig_name = "vol_tra_"+vcor+'_'+str(lim)+'.png'
           
plot_bot_plume(fig_name, fig_path, lon, lat, C_zps_nt, proj,
               colmap, vmin, vmax, vstp, cbar_extend, cbar_label, 
               cbar_hor, map_lims, bathy=bathy, cn_lev=cn_lev, land='gray')
 

# 3. Plotting zps
bathy = ds_dom_zps["bathy_metry"]
lon = ds_dom_zps["glamf"]
lat = ds_dom_zps["gphif"]
vcor = 'zps_tide'

print('  T')
fig_name = "vol_tra_"+vcor+'_'+str(lim)+'.png'
           
plot_bot_plume(fig_name, fig_path, lon, lat, C_zps_td, proj,
               colmap, vmin, vmax, vstp, cbar_extend, cbar_label, 
               cbar_hor, map_lims, bathy=bathy, cn_lev=cn_lev, land='gray')


# 4. Plotting MEs 
bathy = ds_dom_MEs["bathymetry"]
lon = ds_dom_MEs["glamf"]
lat = ds_dom_MEs["gphif"]
vcor = 'MEs_tide'

print('  T')
fig_name = "vol_tra_"+vcor+'_'+str(lim)+'.png'

plot_bot_plume(fig_name, fig_path, lon, lat, C_MEs_td, proj,
               colmap, vmin, vmax, vstp, cbar_extend, cbar_label,
               cbar_hor, map_lims, bathy=bathy, cn_lev=cn_lev, land='gray')


