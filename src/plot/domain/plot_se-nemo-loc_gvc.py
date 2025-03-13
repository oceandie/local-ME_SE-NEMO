#!/usr/bin/env python

import numpy as np
from iris import load
from iris.analysis.cartography import project
from iris.coords import AuxCoord
from iris import quickplot as qplt
from iris.util import squeeze, mask_cube
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as feature
import cf_units

config = 'orca025'

if config == 'orca025':
   glo = "/data/users/dbruciaf/SE-NEMO/se-orca025/MEs_450-800_3200/bathymetry.loc_area.dep3200_polglo_sig3_itr1.nc"
   ant = "/data/users/dbruciaf/SE-NEMO/se-orca025/MEs_450-800_3200/bathymetry.loc_area.dep3200_polant_sig3_itr1.nc"
   cor = "/data/users/dbruciaf/SE-NEMO/se-orca025/se-nemo-domain_cfg/coordinates.nc"

# Dealing with coordinates

my_cor_list = load(cor)
LAT = my_cor_list.extract_cube("gphit")
LON = my_cor_list.extract_cube("glamt")

my_glo_list = load(glo)
s2z_msk_glo = my_glo_list.extract_cube("s2z_msk")
bathy       = my_glo_list.extract_cube("Bathymetry")

lat = s2z_msk_glo.aux_coords[0]
lat_kwargs = lat.metadata._asdict()
lat_kwargs['standard_name'] = 'latitude'
lat_kwargs['units'] = cf_units.Unit("degrees")
lon = s2z_msk_glo.aux_coords[1]
lon_kwargs = lon.metadata._asdict()
lon_kwargs['standard_name'] = 'longitude'
lon_kwargs['units'] = cf_units.Unit("degrees")

lat = AuxCoord(LAT.core_data(), **lat_kwargs)
lon = AuxCoord(LON.core_data(), **lon_kwargs)

s2z_msk_glo.remove_coord('nav_lat')
s2z_msk_glo.remove_coord('nav_lon')
s2z_msk_glo.add_aux_coord(lat, [0, 1])
s2z_msk_glo.add_aux_coord(lon, [0, 1])

bathy.remove_coord('nav_lat')
bathy.remove_coord('nav_lon')
bathy.add_aux_coord(lat, [0, 1])
bathy.add_aux_coord(lon, [0, 1])

land = bathy.copy()

my_ant_list = load(ant)
s2z_msk_ant = my_ant_list.extract_cube("s2z_msk")
s2z_msk_ant.remove_coord('nav_lat')
s2z_msk_ant.remove_coord('nav_lon')
s2z_msk_ant.add_aux_coord(lat, [0, 1])
s2z_msk_ant.add_aux_coord(lon, [0, 1])

s2z_msk_glo = mask_cube(s2z_msk_glo, s2z_msk_glo.data==0) 
s2z_msk_ant = mask_cube(s2z_msk_ant, s2z_msk_ant.data==0)
land = mask_cube(land, land.data!=0)

# Project the data - seems to be necessary before plotting ORCA data.
#  https://scitools-iris.readthedocs.io/en/stable/generated/gallery/oceanography/plot_orca_projection.html
projected_cube, _ = project(s2z_msk_glo, ccrs.PlateCarree()) #, nx=1920, ny=1080)
qplt.pcolormesh(projected_cube, cmap='cool_r') #, norm=colors.LogNorm(vmin=1e-3, vmax=1e-1))
cbar = plt.colorbar()
cbar.remove()

projected_cube, _ = project(s2z_msk_ant, ccrs.PlateCarree())
qplt.pcolormesh(projected_cube, cmap='cool_r')
cbar = plt.colorbar()
cbar.remove()

projected_cube, _ = project(land, ccrs.PlateCarree())
qplt.pcolormesh(projected_cube, cmap='binary')
cbar = plt.colorbar()
cbar.remove()

projected_cube, _ = project(bathy, ccrs.PlateCarree())
qplt.contour(projected_cube, [3200.], linewidths=0.3, colors='k')

# Sections
col='gold'
# 1) Antarctica
sec_lon = [-59.13, -30.91]
sec_lat = [-77.16, -67.18]
#plt.plot(sec_lon, sec_lat, 'o-', color='black'  , markersize=1, transform=ccrs.PlateCarree())
plt.plot(sec_lon, sec_lat, '-', color='black', linewidth=2.5, transform=ccrs.PlateCarree())
plt.plot(sec_lon, sec_lat, '-', color=col, linewidth=1.5, transform=ccrs.PlateCarree())

# 2) NW European shelf
sec_lon = [ -5.29, -19.56]
sec_lat = [ 50.24,  50.32]
plt.plot(sec_lon, sec_lat, '-', color='black', linewidth=2.5, transform=ccrs.PlateCarree())
plt.plot(sec_lon, sec_lat, '-', color=col, linewidth=1.5, transform=ccrs.PlateCarree())

# 3) Australian shelf
sec_lon = [121.85, 105.57]
sec_lat = [-20.29, -17.57]
plt.plot(sec_lon, sec_lat, '-', color='black', linewidth=2.5, transform=ccrs.PlateCarree())
plt.plot(sec_lon, sec_lat, '-', color=col, linewidth=1.5, transform=ccrs.PlateCarree())

# 4) Chinese shelf
sec_lon = [120.53, 134.99]
sec_lat = [ 36.97,  24.12]
plt.plot(sec_lon, sec_lat, '-', color='black', linewidth=2.5, transform=ccrs.PlateCarree())
plt.plot(sec_lon, sec_lat, '-', color=col, linewidth=1.5, transform=ccrs.PlateCarree())

plt.title("")

plt.gca().coastlines(linewidth=0.5)
plt.gca().add_feature(feature.LAND, color='gray', edgecolor='black', zorder=1)
plt.savefig("se_nemo_loc-GVC.png", dpi=500, bbox_inches="tight", pad_inches=0.1)


