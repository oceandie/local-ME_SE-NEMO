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
from utils import plot_bot_plume
import scipy.interpolate as interpolate

# ==============================================================================

# Input parameters

# 1. INPUT FILES

DOMCFG_zps = '/data/users/dbruciaf/GS/orca025/zps_GO8/domain_cfg_zps_new_bathy.nc'
DOMCFG_MEs = '/data/users/dbruciaf/GS/orca025/MEs_GO8/4env_MEs_000_015_020/domain_cfg_000-015-020_v1.nc'

TOBSdir = '/data/users/dbruciaf/NOAA_WOA18/2005-2017/temperature/0.25'
SOBSdir = '/data/users/dbruciaf/NOAA_WOA18/2005-2017/salinity/0.25'
Tzpsdir = '/data/users/dbruciaf/GS/orca025/outputs/JRA_real/zps_u-co007/'
Tmesdir = '/data/users/dbruciaf/GS/orca025/outputs/JRA_real/MEs_u-co024/'

# 3. PLOT
#lon0 = -82.
#lon1 = -60.
#lat0 =  25.
#lat1 =  45.
lon0 = -82.
lon1 = -42.
lat0 =  24
lat1 =  47.5
proj = ccrs.Mercator() #ccrs.Robinson()

fig_path = "./"
colmap = 'RdBu_r'
cbar_extend = "both"
cn_lev = [500., 1000.]
cbar_hor = 'horizontal'
map_lims = [lon0, lon1, lat0, lat1]
Tmin = -5.0
Tmax = 5.2
Tstp = 0.2
Smin = -1.
Smax = 1.05
Sstp = 0.05

# ==============================================================================

# Loading domain geometry
DS_zps  = open_domain_cfg(files=[DOMCFG_zps])
for i in ['bathymetry','bathy_meter']:
    for dim in ['x','y']:
        DS_zps[i] = DS_zps[i].rename({dim: dim+"_c"})

DS_MEs  = open_domain_cfg(files=[DOMCFG_MEs])
for i in ['bathymetry','bathy_meter']:
    for dim in ['x','y']:
        DS_MEs[i] = DS_MEs[i].rename({dim: dim+"_c"})

# ==============================================================================

y_str = 'bottom_biases_average_2009-2013.png'
 
# -------------------------
# 1. NOAA WOA observations    
Tfile = TOBSdir + '/woa18_A5B7_t00_04.nc'
Sfile = SOBSdir + '/woa18_A5B7_s00_04.nc'

# Loading NOAA files
ds_obs_T = xr.open_dataset(Tfile, decode_times=False)
ds_obs_S = xr.open_dataset(Sfile, decode_times=False)

ds_obs_T = ds_obs_T.squeeze()
ds_obs_S = ds_obs_S.squeeze()
   
# Extracting only the part of the domain we need
ds_obs_T = ds_obs_T.isel(lon=slice(300,900), lat=slice(350,790))
ds_obs_S = ds_obs_S.isel(lon=slice(300,900), lat=slice(350,790))

# Computing bottom level
Tan = ds_obs_T.t_an.values
nk = Tan.shape[0]
nj = Tan.shape[1]
ni = Tan.shape[2]
bottom_level = np.zeros((nj,ni), dtype=int)
for j in range(nj):
    for i in range(ni):
        dry_lev = np.isnan(Tan[:,j,i]).sum()
        if dry_lev < nk:
           if dry_lev == 0:
              bottom_level[j,i] = nk
           else:
              for k in range(nk):
                  if np.isnan(Tan[k,j,i]):
                     bottom_level[j,i] = k
                     break

ds_obs_T['bottom_level'] = xr.DataArray(bottom_level, 
                                        coords=(ds_obs_T.lat, ds_obs_T.lon), 
                                        dims=('lat','lon')
                                       )
ds_obs_S['bottom_level'] = xr.DataArray(bottom_level,
                                        coords=(ds_obs_S.lat, ds_obs_S.lon), 
                                        dims=('lat','lon')
                                       )
lev = ds_obs_T['bottom_level'].load()-1
lev = lev.where(lev>0,0) # we removenegative indexes
depth = ds_obs_T['depth'].load()
daTan = ds_obs_T['t_an']
daSan = ds_obs_S['s_an']
# Computing potential temperature
daSA = gsw.SA_from_SP(daSan, depth, ds_obs_T.lon, ds_obs_T.lat)
daTP = gsw.pt0_from_t(daSA, daTan, depth)
# Extracting values at the bottom
daSan = daSan.assign_coords(depth=range(nk))
daTP = daTP.assign_coords(depth=range(nk))
da_obs_T_bot = daTP.isel(depth=lev)
da_obs_S_bot = daSan.isel(depth=lev)

# -------------------------
# 2. MODELS
T_zps = Tzpsdir + '/nemo_co007o_1y_average_2009-2013_grid_T.nc'
T_MEs = Tmesdir + '/nemo_co024o_1y_average_2009-2013_grid_T.nc'

# Loading NEMO files
ds_T_zps = open_nemo(DS_zps, files=[T_zps])
ds_T_MEs = open_nemo(DS_MEs, files=[T_MEs])

# Extracting only the part of the domain we need
ds_dom_zps = DS_zps.isel(x_c=slice(700,1150),x_f=slice(700,1150),
                         y_c=slice(700,1100),y_f=slice(700,1100))
ds_dom_MEs = DS_MEs.isel(x_c=slice(700,1150),x_f=slice(700,1150),
                         y_c=slice(700,1100),y_f=slice(700,1100))

ds_T_zps =  ds_T_zps.isel(x_c=slice(700,1150),y_c=slice(700,1100))
ds_T_MEs =  ds_T_MEs.isel(x_c=slice(700,1150),y_c=slice(700,1100))

# Computing model T-levels depth
e3w_3_zps = ds_dom_zps["e3w_0"].values
nk = e3w_3_zps.shape[0]
nj = e3w_3_zps.shape[1]
ni = e3w_3_zps.shape[2]
dep3_zps = np.zeros(shape=(nk,nj,ni))
dep3_zps[0,:,:] = 0.5 * e3w_3_zps[0,:,:]
for k in range(1, nk):
    dep3_zps[k,:,:] = dep3_zps[k-1,:,:] + e3w_3_zps[k,:,:]

e3w_3_MEs = ds_dom_MEs["e3w_0"].values
dep3_MEs = np.zeros(shape=(nk,nj,ni))
dep3_MEs[0,:,:] = 0.5 * e3w_3_MEs[0,:,:]
for k in range(1, nk):
    dep3_MEs[k,:,:] = dep3_MEs[k-1,:,:] + e3w_3_MEs[k,:,:]

dep4_zps = np.repeat(dep3_zps[np.newaxis, :, :, :], 1, axis=0)
dep4_MEs = np.repeat(dep3_MEs[np.newaxis, :, :, :], 1, axis=0)
ds_T_zps["Tdepth"] = xr.DataArray(dep4_zps,
                                  coords=ds_T_zps["thetao_con"].coords,
                                  dims=ds_T_zps["thetao_con"].dims
                                  )
ds_T_MEs["Tdepth"] = xr.DataArray(dep4_MEs,
                                  coords=ds_T_MEs["thetao_con"].coords,
                                  dims=ds_T_MEs["thetao_con"].dims
                                  )

    
bathy = ds_dom_zps["bathymetry"]
lon = ds_dom_zps["glamf"]
lat = ds_dom_zps["gphif"]

# A. Interpolating obs to orca025 grid
da_obs_T_bot_int = da_obs_T_bot.interp(lat=ds_dom_zps.gphit, 
                                       lon=ds_dom_zps.glamt,
                                       method='linear')
da_obs_S_bot_int = da_obs_S_bot.interp(lat=ds_dom_zps.gphit, 
                                       lon=ds_dom_zps.glamt,
                                       method='linear')

# B. zps biases
lev = ds_dom_zps['bottom_level'].load()-1 # because indexes are in Fortran convention
lev = lev.where(lev>0,0) # we removenegative indexes
CT_zps_bot = ds_T_zps['thetao_con'].isel(z_c=lev)[0,:,:].squeeze()
AS_zps_bot = ds_T_zps['so_abs'].isel(z_c=lev)[0,:,:].squeeze()
# Computing potential temperature
PT_zps_bot = gsw.conversions.pt_from_CT(AS_zps_bot.values, CT_zps_bot.values)
# Computing practical salinity
lon2 = ds_dom_zps["glamt"].values
lat2 = ds_dom_zps["gphit"].values
dep_zps_bot = ds_T_zps["Tdepth"].isel(z_c=lev)[0,:,:].squeeze().values
prs = gsw.p_from_z(-dep_zps_bot, lat2)
PS_zps_bot = gsw.SP_from_SA(AS_zps_bot.values, prs, lon2, lat2)

zps_var_T = xr.DataArray(PT_zps_bot,
                         coords=CT_zps_bot.coords,
                         dims=CT_zps_bot.dims
                         )
zps_var_S = xr.DataArray(PS_zps_bot,
                         coords=AS_zps_bot.coords,
                         dims=AS_zps_bot.dims
                         )


print('  T')
Tdiff = zps_var_T - da_obs_T_bot_int
Tdiff = Tdiff.where(bathy<=500.)
cbar_label = "Bottom ocean T zps - NOAA [C$^{\circ}$]"
fig_name = "zps_gs_JRA_bot_T_diff_"+y_str
plot_bot_plume(fig_name, fig_path, lon, lat, Tdiff, proj,
               colmap, Tmin, Tmax, Tstp, cbar_extend, cbar_label,
               cbar_hor, map_lims, bathy, lon, lat, cn_lev)  
    

print('  S')
Tdiff = zps_var_S - da_obs_S_bot_int
Tdiff = Tdiff.where(bathy<=500.)
cbar_label = "Bottom ocean S zps - NOAA [PSU]"
fig_name = "zps_gs_JRA_bot_S_diff_"+y_str
plot_bot_plume(fig_name, fig_path, lon, lat, Tdiff, proj,
               colmap, Smin, Smax, Sstp, cbar_extend, cbar_label,
               cbar_hor, map_lims, bathy, lon, lat, cn_lev)


# B. mes biases
lev = ds_dom_MEs['bottom_level'].load()-1 # because indexes are in Fortran convention
lev = lev.where(lev>0,0) # we removenegative indexes
CT_mes_bot = ds_T_MEs['thetao_con'].isel(z_c=lev)[0,:,:].squeeze()
AS_mes_bot = ds_T_MEs['so_abs'].isel(z_c=lev)[0,:,:].squeeze()
# Computing potential temperature
PT_mes_bot = gsw.conversions.pt_from_CT(AS_mes_bot.values, CT_mes_bot.values)
# Computing practical salinity
lon2 = ds_dom_MEs["glamt"].values
lat2 = ds_dom_MEs["gphit"].values
dep_mes_bot = ds_T_MEs["Tdepth"].isel(z_c=lev)[0,:,:].squeeze().values
prs = gsw.p_from_z(-dep_mes_bot, lat2)
PS_mes_bot = gsw.SP_from_SA(AS_mes_bot.values, prs, lon2, lat2)

mes_var_T = xr.DataArray(PT_mes_bot,
                         coords=CT_mes_bot.coords,
                         dims=CT_mes_bot.dims
                         )
mes_var_S = xr.DataArray(PS_mes_bot,
                         coords=AS_mes_bot.coords,
                         dims=AS_mes_bot.dims
                         )


print('  T')
Tdiff = mes_var_T - da_obs_T_bot_int
Tdiff = Tdiff.where(bathy<=500.)
cbar_label = "Bottom ocean T MEs - NOAA [C$^{\circ}$]"
fig_name = "mes_gs_JRA_bot_T_diff_"+y_str
plot_bot_plume(fig_name, fig_path, lon, lat, Tdiff, proj,
               colmap, Tmin, Tmax, Tstp, cbar_extend, cbar_label,
               cbar_hor, map_lims, bathy, lon, lat, cn_lev)

print('  S')
Tdiff = mes_var_S - da_obs_S_bot_int
Tdiff = Tdiff.where(bathy<=500.)
cbar_label = "Bottom ocean S MEs - NOAA [PSU]"
fig_name = "mes_gs_JRA_bot_S_diff_"+y_str
plot_bot_plume(fig_name, fig_path, lon, lat, Tdiff, proj,
               colmap, Smin, Smax, Sstp, cbar_extend, cbar_label,
               cbar_hor, map_lims, bathy, lon, lat, cn_lev)
