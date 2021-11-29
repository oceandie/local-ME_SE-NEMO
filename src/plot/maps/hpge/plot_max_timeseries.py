#!/usr/bin/env python

#     |------------------------------------------------------------|
#     | This module creates a 2D field of maximum spurious current |
#     | in the vertical and in time after an HPGE test.            |
#     | The resulting file can be used then to optimise the rmax   |
#     | of Multi-Envelope vertical grids.                          |
#     |                                                            |
#     | Author: Diego Bruciaferri                                  |
#     | Date and place: 07-09-2021, Met Office, UK                 |
#     |------------------------------------------------------------|


import os
from os.path import join, isfile, basename, splitext
import glob
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import xarray as xr

# ==============================================================================
# Input files
# ==============================================================================

# Folder path containing HPGE spurious currents velocity files 
HPGEdir = '/scratch/dbruciaf/OVF_HPGE/zps'
#HPGEdir = '/scratch/dbruciaf/OVF_HPGE/MEs_2800_r075_r12'
#HPGEdir = '/scratch/dbruciaf/OVF_HPGE/MEs_2800_r15_r15'

# Name of the zonal and meridional velocity variables
Uvar = 'uo'
Vvar = 'vo'
# Name of the variable to chunk with dask and size of chunks
chunk_var = 'time_counter'
chunk_size = 1

# ==============================================================================
# LOOP

Ufiles = sorted(glob.glob(HPGEdir+'/*grid-U.nc'))
Vfiles = sorted(glob.glob(HPGEdir+'/*grid-V.nc'))

for F in range(len(Ufiles)):

    print(Ufiles[F])

    ds_U = xr.open_dataset(Ufiles[F], chunks={chunk_var:chunk_size})
    U4   = ds_U[Uvar]
    ds_V = xr.open_dataset(Vfiles[F], chunks={chunk_var:chunk_size})
    V4   = ds_V[Vvar]

    # rename some dimensions
    U4 = U4.rename({U4.dims[0]: 't', U4.dims[1]: 'k'})
    V4 = V4.rename({V4.dims[0]: 't', V4.dims[1]: 'k'})

    # interpolating from U,V to T
    U = U4.rolling({'x':2}).mean().fillna(0.)
    V = V4.rolling({'y':2}).mean().fillna(0.) 
    
    hpge = np.sqrt(np.power(U,2) + np.power(V,2))

    max_hpge = hpge.max(dim=('k','y','x'))

    cm = 1/2.54  # centimeters in inches
    fig = plt.figure(figsize=(20*cm,7*cm), dpi=100)
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    ax = fig.add_subplot(spec[:1])

    ax.plot(max_hpge.data)
    ax.set_xlabel('days')
    ax.set_ylabel('max HPG error [$m\;s^{-1}$]')
    name = HPGEdir + '/max_hpge_timeseries.png'
    plt.savefig(name)
