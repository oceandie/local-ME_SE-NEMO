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

U_ZPS_EN = '/data/users/dbruciaf/EN4/interp_GO8_zps/geo_adj/nemo_cv662o_1d_19760302-19760312_grid_U.nc'
U_ZPS_NT = '/data/users/dbruciaf/SE-NEMO/se-orca025/GS1p0_notide/SENEMO_1991-2019_average_grid_U.nc'
U_ZPS_TD = '/data/users/dbruciaf/SE-NEMO/se-orca025/GS1p1_tide/SENEMO_1991-2019_average_grid_U.nc'
U_MES_TD = '/data/users/dbruciaf/SE-NEMO/se-orca025/GS1p2_full/SENEMO_1991-2019_average_grid_U.nc'
V_ZPS_EN = '/data/users/dbruciaf/EN4/interp_GO8_zps/geo_adj/nemo_cv662o_1d_19760302-19760312_grid_V.nc'
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

U_zps_en = ds_U_zps_en['uo']
U_zps_nt = ds_U_zps_nt['uo']
U_zps_td = ds_U_zps_td['uo']
U_MEs_td = ds_U_MEs_td['uo']
V_zps_en = ds_V_zps_en['vo']
V_zps_nt = ds_V_zps_nt['vo']
V_zps_td = ds_V_zps_td['vo']
V_MEs_td = ds_V_MEs_td['vo']

# COMPUTING CURRENT SPEED AT SURFACE

# 1) zps_en
Uzps = U_zps_en.squeeze().mean(axis=0).squeeze()[0,:,:]
Vzps = V_zps_en.squeeze().mean(axis=0).squeeze()[0,:,:]
uzps = Uzps.rolling({'x_f':2}).mean().fillna(0.)
vzps = Vzps.rolling({'y_f':2}).mean().fillna(0.)
czps = np.sqrt(np.power(uzps.data,2) + np.power(vzps.data,2))
C_zps_en = xr.DataArray(data = czps,
                        dims = ["y_c","x_c"],
                        coords=dict(x_c=(["x_c"], vzps.x_c.data),
                                    y_c=(["y_c"], uzps.y_c.data)),
                        attrs=dict(description="Current speed",
                                   units="m/s")
                       )

# 2) zps_nt
Uzps = U_zps_nt.squeeze()[0,:,:]
Vzps = V_zps_nt.squeeze()[0,:,:]
uzps = Uzps.rolling({'x_f':2}).mean().fillna(0.)
vzps = Vzps.rolling({'y_f':2}).mean().fillna(0.)
czps = np.sqrt(np.power(uzps.data,2) + np.power(vzps.data,2))
C_zps_nt = xr.DataArray(data = czps,
                        dims = ["y_c","x_c"],
                        coords=dict(x_c=(["x_c"], vzps.x_c.data),
                                    y_c=(["y_c"], uzps.y_c.data)),
                        attrs=dict(description="Current speed",
                                   units="m/s")
                       )
# 3) zps_td
Uzps = U_zps_td.squeeze()[0,:,:]
Vzps = V_zps_td.squeeze()[0,:,:]
uzps = Uzps.rolling({'x_f':2}).mean().fillna(0.)
vzps = Vzps.rolling({'y_f':2}).mean().fillna(0.)
czps = np.sqrt(np.power(uzps.data,2) + np.power(vzps.data,2))
C_zps_td = xr.DataArray(data = czps,
                        dims = ["y_c","x_c"],
                        coords=dict(x_c=(["x_c"], vzps.x_c.data),
                                    y_c=(["y_c"], uzps.y_c.data)),
                        attrs=dict(description="Current speed",
                                   units="m/s")
                       )

# 4) MEs
Umes = U_MEs_td.squeeze()[0,:,:]
Vmes = V_MEs_td.squeeze()[0,:,:]
umes = Umes.rolling({'x_f':2}).mean().fillna(0.)
vmes = Vmes.rolling({'y_f':2}).mean().fillna(0.)
cmes = np.sqrt(np.power(umes.data,2) + np.power(vmes.data,2))
C_MEs_td = xr.DataArray(data = cmes,
                        dims = ["y_c","x_c"],
                        coords=dict(x_c=(["x_c"], vmes.x_c.data),
                                    y_c=(["y_c"], umes.y_c.data)),
                        attrs=dict(description="Current speed",
                                   units="m/s")
                       )

print(' computing speed:  done')

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
vmax =  0.7
vstp =  0.05
cbar_label = "Sea Surf. Current Speed [$m/s$]"
#vcor = 'zps'
vcor = 'zps_en4'

print('  T')
fig_name = "cur_"+vcor+'.png'

plot_bot_plume(fig_name, fig_path, lon, lat, C_zps_en, proj,
               colmap, vmin, vmax, vstp, cbar_extend, cbar_label,
               cbar_hor, map_lims, bathy=bathy, cn_lev=cn_lev, land='gray')

# 2. Plotting zps
bathy = ds_dom_zps["bathy_metry"]
lon = ds_dom_zps["glamf"]
lat = ds_dom_zps["gphif"]

vmin =  0.0
vmax =  0.7
vstp =  0.05
cbar_label = "Sea Surf. Current Speed [$m/s$]"
#vcor = 'zps'
vcor = 'zps_notide'

print('  T')
fig_name = "cur_"+vcor+'.png'
           
plot_bot_plume(fig_name, fig_path, lon, lat, C_zps_nt, proj,
               colmap, vmin, vmax, vstp, cbar_extend, cbar_label, 
               cbar_hor, map_lims, bathy=bathy, cn_lev=cn_lev, land='gray')
 

# 3. Plotting zps
bathy = ds_dom_zps["bathy_metry"]
lon = ds_dom_zps["glamf"]
lat = ds_dom_zps["gphif"]

vmin =  0.0
vmax =  0.7
vstp =  0.05
cbar_label = "Sea Surf. Current Speed [$m/s$]"
#vcor = 'zps'
vcor = 'zps_tide'

print('  T')
fig_name = "cur_"+vcor+'.png'
           
plot_bot_plume(fig_name, fig_path, lon, lat, C_zps_td, proj,
               colmap, vmin, vmax, vstp, cbar_extend, cbar_label, 
               cbar_hor, map_lims, bathy=bathy, cn_lev=cn_lev, land='gray')


# 4. Plotting MEs 
bathy = ds_dom_MEs["bathymetry"]
lon = ds_dom_MEs["glamf"]
lat = ds_dom_MEs["gphif"]

vmin =  0.0
vmax =  0.7
vstp =  0.05
cbar_label = "Sea Surf. Current Speed [$m/s$]"
vcor = 'MEs_tide'

print('  T')
fig_name = "cur_"+vcor+".png"

plot_bot_plume(fig_name, fig_path, lon, lat, C_MEs_td, proj,
               colmap, vmin, vmax, vstp, cbar_extend, cbar_label,
               cbar_hor, map_lims, bathy=bathy, cn_lev=cn_lev, land='gray')


