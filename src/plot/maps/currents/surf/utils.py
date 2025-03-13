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
import matplotlib.gridspec as gridspec
import xarray as xr
from xnemogcm import open_domain_cfg, open_nemo
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import cartopy.feature as feature
import cmocean

def plot_bot_plume(fig_name, fig_path, lon, lat, var, proj, colmap, vmin, vmax, vstp, cbar_extend, cbar_label, cbar_hor, map_lims, bathy=None, lon_bat=None, lat_bat=None, cn_lev=None, ucur=None, vcur=None, land=None):

    fig = plt.figure(figsize=(30,20), dpi=200)
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

    ax = fig.add_subplot(spec[:1], projection=proj)
    ax.coastlines()
    if land is not None:
       ax.add_feature(feature.LAND, color=land,edgecolor=land,zorder=1)
    else:
       ax.add_feature(feature.LAND, color='black',edgecolor='black',zorder=1)
    #ax.gridlines()

    if isinstance(vstp, list):
       CN_LEV = vstp
    else:
       CN_LEV = np.arange(vmin, vmax, vstp)

    # Drawing settings
    cmap = colmap #cmocean.cm.deep
    transform = ccrs.PlateCarree()
    pcol_kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax, extend=cbar_extend, transform=transform)
    #pcol_kwargs = dict(cmap=cmap, extend=cbar_extend, transform=transform)
    #cbar_kwargs = dict(ticks=CN_LEV,orientation=cbar_hor)
    cbar_kwargs = dict(orientation=cbar_hor)

    # Grid settings
    gl_kwargs = dict()
    gl = ax.gridlines(**gl_kwargs)
    gl.xlines = True
    gl.ylines = True
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 30, 'color': 'k'}
    gl.ylabel_style = {'size': 30, 'color': 'k'}

    # Plotting
    var = var.where(var != 0, np.nan)
    pcol = ax.contourf(lon, lat, var, levels=CN_LEV, **pcol_kwargs)
    #pc   = ax.contour(lon, lat, var, levels=CN_LEV,colors = 'k', vmin=vmin, vmax=vmax, extend=cbar_extend, transform=transform)

    if bathy is not None:
       if lon_bat is not None and lat_bat is not None:
          bcon = ax.contour(lon_bat, lat_bat, bathy, levels=cn_lev, colors='k', transform=transform)
          #bcon = ax.contour(lon_bat, lat_bat, bathy, levels=[2800.], colors='r', linewidths=5.0, transform=transform)
       else:
          bcon = ax.contour(lon, lat, bathy, levels=cn_lev, colors='w', transform=transform)
          #bcon = ax.contour(lon, lat, bathy, levels=[2800.], colors='r', linewidths=5.0, transform=transform)

    if ucur is not None and vcur is not None:
    
       stp = 4
       ucur = ucur.where(var>0.02)
       vcur = vcur.where(var>0.02)
       QV = ax.quiver(lon.values[::stp,::stp], lat.values[::stp,::stp], \
                      ucur.values[::stp,::stp], vcur.values[::stp,::stp], \
                      transform=transform, \
                      color='deepskyblue', \
                      units='xy',\
                      angles='xy', \
                      scale=0.2,
                      scale_units='inches')
       plt.quiverkey(QV, 0.4, 0.3, 0.1, "0.1 m/s", labelpos = "S", coordinates='figure', fontproperties={'size': 25})
    
    cb = plt.colorbar(pcol, **cbar_kwargs)
    cb.set_label(label=cbar_label,size=40)
    cb.ax.tick_params(labelsize=30) 
    ax.set_extent(map_lims)
    print(f"Saving {fig_path}", end=": ")
    plt.savefig(fig_path+fig_name, bbox_inches="tight")
    print("done")
    plt.close()

def e3_to_dep(e3W, e3T):

    gdepT = xr.full_like(e3T, None, dtype=np.double).rename('gdepT')
    gdepW = xr.full_like(e3W, None, dtype=np.double).rename('gdepW')

    gdepW[{"z_f":0}] = 0.0
    gdepT[{"z_c":0}] = 0.5 * e3W[{"z_f":0}]
    for k in range(1, e3W.sizes["z_f"]):
        gdepW[{"z_f":k}] = gdepW[{"z_f":k-1}] + e3T[{"z_c":k-1}]
        gdepT[{"z_c":k}] = gdepT[{"z_c":k-1}] + e3W[{"z_f":k}]

    return tuple([gdepW, gdepT])

def compute_masks(ds_domain, merge=False):
    """
    Compute masks from domain_cfg Dataset.
    If merge=True, merge with the input dataset.
    Parameters
    ----------
    ds_domain: xr.Dataset
        domain_cfg datatset
    add: bool
        if True, merge with ds_domain
    Returns
    -------
    ds_mask: xr.Dataset
        dataset with masks
    """

    # Extract variables
    k = ds_domain["z_c"] + 1
    top_level = ds_domain["top_level"]
    bottom_level = ds_domain["bottom_level"]

    # Page 27 NEMO book.
    # I think there's a typo though.
    # It should be:
    #                  | 0 if k < top_level(i, j)
    # tmask(i, j, k) = | 1 if top_level(i, j) ≤ k ≤ bottom_level(i, j)
    #                  | 0 if k > bottom_level(i, j)
    tmask = xr.where(np.logical_or(k < top_level, k > bottom_level), 0, np.nan)
    tmask = xr.where(np.logical_and(bottom_level >= k, top_level <= k), 1, tmask)
    tmask = tmask.rename("tmask")

    tmask = tmask.transpose("z_c","y_c","x_c")

    # Need to shift and replace last row/colum with tmask
    # umask(i, j, k) = tmask(i, j, k) ∗ tmask(i + 1, j, k)
    umask = tmask.rolling(x_c=2).prod().shift(x_c=-1)
    umask = umask.where(umask.notnull(), tmask)
    umask = umask.rename("umask")

    # vmask(i, j, k) = tmask(i, j, k) ∗ tmask(i, j + 1, k)
    vmask = tmask.rolling(y_c=2).prod().shift(y_c=-1)
    vmask = vmask.where(vmask.notnull(), tmask)
    vmask = vmask.rename("vmask")

    # Return
    masks = xr.merge([tmask, umask, vmask])
    if merge:
        return xr.merge([ds_domain, masks])
    else:
        return masks

