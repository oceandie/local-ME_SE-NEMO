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
   glo = "/data/users/dbruciaf/SE-NEMO/se-orca025/MEs_450-800_3200/maximum_hpge_MEs_4env_450_018-010-010_glo_opt_v2.nc"
   ant = "/data/users/dbruciaf/SE-NEMO/se-orca025/MEs_450-800_3200/maximum_hpge_MEs_3env_800_018-010_ant_opt_v3.nc"
   cor = "/data/users/dbruciaf/SE-NEMO/se-orca025/se-nemo-domain_cfg/coordinates.nc"

# Dealing with coordinates

my_cor_list = load(cor)
LAT = my_cor_list.extract_cube("gphit")
LON = my_cor_list.extract_cube("glamt")

nj = LAT.shape[0]
ni = LAT.shape[1]

my_glo_list = load(glo)
hpge1_glo = my_glo_list.extract_cube("max_hpge_1")
hpge2_glo = my_glo_list.extract_cube("max_hpge_2")
hpge3_glo = my_glo_list.extract_cube("max_hpge_3")
hpge_glo = hpge1_glo.copy()

my_ant_list = load(ant)
hpge1_ant = my_ant_list.extract_cube("max_hpge_1")
hpge2_ant = my_ant_list.extract_cube("max_hpge_2")
hpge_ant = hpge1_ant.copy()

hpge_glo.data = np.maximum(hpge1_glo.data,np.maximum(hpge2_glo.data,hpge3_glo.data))
hpge_ant.data = np.maximum(hpge1_glo.data,hpge2_glo.data)

#for j in range(nj):
#    for i in range(ni):
#        array1 = [hpge1_glo[j,i].data,
#                  hpge2_glo[j,i].data,
#                  hpge3_glo[j,i].data,
#                 ]
#        hpge_glo[j,i].data = np.amax(array1)
#
#        array2 = [hpge1_ant[j,i].data,
#                  hpge2_ant[j,i].data,
#                 ]
#        hpge_ant[j,i].data = np.amax(array2)

#lat = hpge1_glo.aux_coords[0]
lat_kwargs = {} #lat.metadata._asdict()
lat_kwargs['standard_name'] = 'latitude'
lat_kwargs['units'] = cf_units.Unit("degrees")
#lon = hpge1_glo.aux_coords[1]
lon_kwargs = {} #lon.metadata._asdict()
lon_kwargs['standard_name'] = 'longitude'
lon_kwargs['units'] = cf_units.Unit("degrees")

lat = AuxCoord(LAT.core_data(), **lat_kwargs)
lon = AuxCoord(LON.core_data(), **lon_kwargs)

#hpge_glo.remove_coord('nav_lat')
#hpge_glo.remove_coord('nav_lon')
hpge_glo.add_aux_coord(lat, [0, 1])
hpge_glo.add_aux_coord(lon, [0, 1])

#hpge_ant.remove_coord('nav_lat')
#hpge_ant.remove_coord('nav_lon')
hpge_ant.add_aux_coord(lat, [0, 1])
hpge_ant.add_aux_coord(lon, [0, 1])


#s2z_msk_glo = mask_cube(s2z_msk_glo, s2z_msk_glo.data==0) 
#s2z_msk_ant = mask_cube(s2z_msk_ant, s2z_msk_ant.data==0)
#land = mask_cube(land, land.data!=0)

# Project the data - seems to be necessary before plotting ORCA data.
#  https://scitools-iris.readthedocs.io/en/stable/generated/gallery/oceanography/plot_orca_projection.html
#projected_cube, _ = project(hpge_glo, ccrs.PlateCarree()) #, nx=1920, ny=1080)
projected_cube, _ = project(hpge_glo, ccrs.Robinson()) #, nx=1920, ny=1080)
qplt.pcolormesh(projected_cube, cmap='hot', vmin=0.0, vmax=0.05)
cbar = plt.colorbar()
cbar.remove()

#projected_cube, _ = project(hpge_ant, ccrs.PlateCarree())
projected_cube, _ = project(hpge_ant, ccrs.Robinson())
qplt.pcolormesh(projected_cube, cmap='hot', vmin=0.0, vmax=0.05)
cbar = plt.colorbar()
cbar.remove()

plt.title("")

plt.gca().coastlines(linewidth=0.5)
plt.gca().add_feature(feature.LAND, color='gray', edgecolor='black', zorder=1)
plt.savefig("hpge.png", dpi=500, bbox_inches="tight", pad_inches=0.1)


