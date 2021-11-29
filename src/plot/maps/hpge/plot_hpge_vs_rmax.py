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
import xarray as xr

# ==============================================================================
# Input files
# ==============================================================================

# Path of the file containing max HPGE spurious currents velocity field
#HPGEfile = '/scratch/dbruciaf/OVF_HPGE/MEs_2800_r12_r12/maximum_hpge.nc'
#RMAXfile = '/data/users/dbruciaf/OVF/MEs_GO8/env4.v2.maxdep/2800/r12_r12/bathymetry.MEs_4env_2800_r12_r12_maxdep_2800.0.nc'
#HPGEfile = '/scratch/dbruciaf/OVF_HPGE/MEs_2800_r12_r12-r04/maximum_hpge.nc'
#RMAXfile = '/data/users/dbruciaf/OVF/MEs_GO8/env4.v2.maxdep/2800/r12_r12-r04/bathymetry.MEs_4env_2800_r12_r12-r04_maxdep_2800.0.nc'
#HPGEfile = '/scratch/dbruciaf/OVF_HPGE/MEs_2800_r12_r12-r075-r060/maximum_hpge.nc'
#RMAXfile = '/data/users/dbruciaf/OVF/MEs_GO8/env4.v2.maxdep/2800/r12_r12-r075-r060/bathymetry.MEs_4env_2800_r12_r12-r075-r060_maxdep_2800.0.nc'
HPGEfile = '/data/users/dbruciaf/OVF/MEs_GO8/env4.v2.maxdep/2800/r12_r12-r075-r040_v3/maximum_hpge.nc'
RMAXfile = '/data/users/dbruciaf/OVF/MEs_GO8/env4.v2.maxdep/2800/r12_r12-r075-r040_v3/bathymetry.MEs_4env_2800_r12_r12-r075-r040_v3_maxdep_2800.0.nc'

maxvel = 0.05
#maxrmx = 0.075
#maxrmx = 0.060
maxrmx = 0.050
color = ['red', 'blue']
label = ['env-2','env-3'] 

ds_hpge = xr.open_dataset(HPGEfile)
ds_rmax = xr.open_dataset(RMAXfile)
ds_hpge = ds_hpge.where(ds_rmax.s2z_msk!=0)
ds_rmax = ds_rmax.where(ds_rmax.s2z_msk!=0)

varsH = list(ds_hpge.keys())
varsR = []

if len(varsH) == 2: 
   varsR.append('rmax0_2')
   varsR.append('rmax0_3')
elif len(varsH) == 1:
   varsR.append('rmax0_1')
else:
   print('ERROR: this conifg does not exist yet')

for env in range(len(varsH)):
    fig, ax = plt.subplots(ncols=1, nrows=1)
    hpge = ds_hpge[varsH[env]]
    rmax = ds_rmax[varsR[env]]
    ax.scatter(hpge.data.flatten(),rmax.data.flatten(),s=20,color=color[env], label=label[env])
    ax.plot([maxvel,maxvel],[0.,rmax.max()],'k')
    ax.plot([0.,hpge.max()],[maxrmx,maxrmx],'k')
    ax.set_xlabel('max hpge [$m\;s^{-1}$]')
    ax.set_ylabel('Rmax')
    plt.legend(loc='upper left', numpoints=1, ncol=2, fontsize=8, bbox_to_anchor=(0, 0))
    plt.show()

