import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib as mpl   
import matplotlib.lines as mlines
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import cartopy.mpl.ticker as ctk
import cartopy.feature as feature

# Input settings
# a) T files
woa_T = "/data/users/dbruciaf/NOAA_WOA18/2005-2017/temperature/0.25/woa18_A5B7_t00_04.nc"
cnt_T = "/data/users/dbruciaf/SE-NEMO/se-orca025/final_15-11-2024/se-nemo_ZPS_average_1991-2019_grid_T.nc"
tid_T = "/data/users/dbruciaf/SE-NEMO/se-orca025/final_15-11-2024/se-nemo_ZPS_TIDE_average_1991-2019_grid_T.nc"
mes_T = '/data/users/dbruciaf/SE-NEMO/se-orca025/final_15-11-2024/se-nemo_MES_TIDE_average_1991-2019_thetao_con_conservative_remap.nc'

# b) Level at 200m
lev200 = [24, 29, 29, 29] 

# c) GOSI10p0 grid coordinates
domf_0 = '/data/users/dbruciaf/OVF/GOSI9-eORCA025/domcfg_eORCA025_v3.nc'

# d) PLOT limits
lon0 = -85
lon1 = -8.
lat0 = 20.
lat1 = 50.
map_lims = [lon0, lon1, lat0, lat1]
proj = ccrs.Mercator()

###################################################

# Reading GOSI10p0 grid coordinates
ds_d0 = xr.open_dataset(domf_0)
ds_d0 = ds_d0.rename_dims({'y':'j','x':'i'})
nav_lon0 = ds_d0.nav_lon
nav_lat0 = ds_d0.nav_lat
bathy = ds_d0.bathy_metry.squeeze()
del ds_d0

# WOA T
ds_woa = xr.open_dataset(woa_T, decode_times=False)
ds_woa = ds_woa.rename_dims({'lat':'j','lon':'i','depth':'k'})
woa15  = ds_woa['t_an'].squeeze().isel({'i':slice(310,730),'j':slice(420,640),'k':lev200[0]})
woa15['lon'] = woa15.lon + 360.
woa15 = woa15.where(woa15.lat>30.)

# BATHY
bathy = bathy.isel({'i':slice(700,1120), 'j':slice(750,1020)})
nav_lon = nav_lon0.isel({'i':slice(700,1120), 'j':slice(750,1020)})
nav_lat = nav_lat0.isel({'i':slice(700,1120), 'j':slice(750,1020)})


# GS1p0_notide
ds_cnt = xr.open_dataset(cnt_T)
ds_cnt = ds_cnt.rename_dims({'y':'j','x':'i','deptht':'k'})
ds_cnt = ds_cnt.assign_coords(nav_lon=nav_lon0, nav_lat=nav_lat0)
ds_cnt.coords["i"] = range(ds_cnt.dims["i"])
ds_cnt.coords["j"] = range(ds_cnt.dims["j"])
cntT = ds_cnt['thetao_con'].squeeze()
# Get rid of discontinuity on lon grid 
# (from https://gist.github.com/willirath/fbfd21f90d7f2a466a9e47041a0cee64)
# see also https://docs.xarray.dev/en/stable/examples/multidimensional-coords.html
after_discont = ~(cntT.coords["nav_lon"].diff("i", label="upper") > 0).cumprod("i").astype(bool)
cntT.coords["nav_lon"] = (cntT.coords["nav_lon"] + 360 * after_discont)

cntT  = cntT.isel(i=slice(1, -1), j=slice(None, -1))
cnt15 = cntT.isel({'i':slice(700,1120), 'j':slice(750,1020), 'k':lev200[1]})
cnt15 = cnt15.where(cnt15.nav_lat>30.)
land  = cntT.isel({'i':slice(700,1120), 'j':slice(750,1020), 'k':0})

# GS1p1_tide
ds_tid = xr.open_dataset(tid_T)
ds_tid = ds_tid.rename_dims({'y':'j','x':'i','deptht':'k'})
ds_tid = ds_tid.assign_coords(nav_lon=nav_lon0, nav_lat=nav_lat0)
ds_tid.coords["i"] = range(ds_tid.dims["i"])
ds_tid.coords["j"] = range(ds_tid.dims["j"])
tidT = ds_tid['thetao_con'].squeeze()
# Get rid of discontinuity on lon grid 
# (from https://gist.github.com/willirath/fbfd21f90d7f2a466a9e47041a0cee64)
# see also https://docs.xarray.dev/en/stable/examples/multidimensional-coords.html
after_discont = ~(tidT.coords["nav_lon"].diff("i", label="upper") > 0).cumprod("i").astype(bool)
tidT.coords["nav_lon"] = (tidT.coords["nav_lon"] + 360 * after_discont)

tidT  = tidT.isel(i=slice(1, -1), j=slice(None, -1))
tid15 = tidT.isel({'i':slice(700,1120), 'j':slice(750,1020), 'k':lev200[2]})
tid15 = tid15.where(tid15.nav_lat>30.)

# GS1p2_full
ds_mes = xr.open_dataset(mes_T)
ds_mes = ds_mes.drop_vars(['gphit','glamt','nav_lat','nav_lon','gdept_1d','gdept_0','gdept_0_target'])
ds_mes = ds_mes.rename_dims({'y_c':'j','x_c':'i','z_c':'k'})
ds_mes = ds_mes.rename({'y_c':'j','x_c':'i','z_c':'k'})
ds_mes = ds_mes.assign_coords(nav_lon=nav_lon0, nav_lat=nav_lat0)
mesT = ds_mes['__xarray_dataarray_variable__'].squeeze()
# Get rid of discontinuity on lon grid 
# (from https://gist.github.com/willirath/fbfd21f90d7f2a466a9e47041a0cee64)
# see also https://docs.xarray.dev/en/stable/examples/multidimensional-coords.html
after_discont = ~(mesT.coords["nav_lon"].diff("i", label="upper") > 0).cumprod("i").astype(bool)
mesT.coords["nav_lon"] = (mesT.coords["nav_lon"] + 360 * after_discont)

mesT  = mesT.isel(i=slice(1, -1), j=slice(None, -1))
mes15 = mesT.isel({'i':slice(700,1120), 'j':slice(750,1020), 'k':lev200[3]})
mes15 = mes15.where(mes15.nav_lat>30.)

# PLOTTING # ----------------------------------------------------------------------

# Model land
land = land.fillna(-999)
land = land.where(land==-999)

# figaspect(0.5) makes the figure twice as wide as it is tall. 
# Then the *1.5 increases the size of the figure.
fig = plt.figure(figsize=plt.figaspect(0.5)*4.)
ax = fig.add_subplot((111), projection=proj)

transform = ccrs.PlateCarree()

gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
gl.xlines = False
gl.ylines = False
gl.top_labels = False
gl.bottom_labels = True
gl.right_labels = True
gl.left_labels = False
#gl.xformatter = LONGITUDE_FORMATTER
#gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 40, 'color': 'k'}
gl.ylabel_style = {'size': 40, 'color': 'k'}
gl.rotate_labels=False
gl.xlocator=ctk.LongitudeLocator(6)
gl.ylocator=ctk.LatitudeLocator(4)
gl.xformatter=ctk.LongitudeFormatter(zero_direction_label=False)
gl.yformatter=ctk.LatitudeFormatter()


# LAND
#ax.contourf(land.nav_lon, land.nav_lat, land, colors='gray', transform = transform)
ax.contourf(land.nav_lon, land.nav_lat, land, colors='gray', transform = transform)

lev = [500., 1000., 2000., 3000., 4000., 5000.]
ax.contour(nav_lon, nav_lat, bathy, levels=lev, colors='k', linewidths=1, transform = transform)
ax.contour(nav_lon, nav_lat, bathy, levels=[0], colors='k', linewidths=3, transform = transform)

# WOA
p = ax.contour(woa15.lon, woa15.lat, woa15, levels=[15], colors='k', linewidths=8, transform=transform)

# GS1p0_notide
ax.contour(cnt15.nav_lon, cnt15.nav_lat, cnt15, levels=[15], colors='magenta', linewidths=8, transform=transform)

# GS1p1_tide
ax.contour(tid15.nav_lon, tid15.nav_lat, tid15, levels=[15], colors='deepskyblue', linewidths=8, transform=transform)

# GS1p2_full
ax.contour(mes15.nav_lon, mes15.nav_lat, mes15, levels=[15], colors='limegreen', linewidths=8, transform=transform)

# Ezer 2016 section
ax.plot([-77., -70.], [ 35.,  35.], 'k', linestyle='--', linewidth=5, transform=transform)

# North Cape Hatteras
#ax.plot([-78., -60.], [ 38.5,  38.5], 'k', linestyle='--', linewidth=5, transform=transform)

ax.set_extent(map_lims, crs=ccrs.PlateCarree())

# Legend
obs  = mlines.Line2D([], [], linewidth=6, color='black', marker=None,
                    label='WOA18')
mod1 = mlines.Line2D([], [], linewidth=6, color='magenta', marker=None,
                    label='ZPS')
mod2 = mlines.Line2D([], [], linewidth=6, color='deepskyblue', marker=None,
                    label='ZPS_TIDE')
mod3 = mlines.Line2D([], [], linewidth=6, color='limegreen', marker=None,
                    label='MES_TIDE')

plt.legend(handles=[obs,mod1, mod2, mod3], fontsize="40", loc="lower center", fancybox=True, framealpha=1.)



plt.savefig("gulf_stream_15deg_senemo.png", bbox_inches="tight")
print("done")
plt.close()

