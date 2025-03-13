#!/usr/bin/env python

#     |------------------------------------------------------------|
#     | This module creates general envelope surfaces to be        |
#     | used to generate a Localised Multi-Envelope s-coordinate   |
#     | vertical grid.                                             |
#     |                                                            |
#     |                                                            |
#     |                      -o#&&*''''?d:>b\_                     |
#     |                 _o/"`''  '',, dMF9MMMMMHo_                 |
#     |              .o&#'        `"MbHMMMMMMMMMMMHo.              |
#     |            .o"" '         vodM*$&&HMMMMMMMMMM?.            |
#     |           ,'              $M&ood,~'`(&##MMMMMMH\           |
#     |          /               ,MMMMMMM#b?#bobMMMMHMMML          |
#     |         &              ?MMMMMMMMMMMMMMMMM7MMM$R*Hk         |
#     |        ?$.            :MMMMMMMMMMMMMMMMMMM/HMMM|`*L        |
#     |       |               |MMMMMMMMMMMMMMMMMMMMbMH'   T,       |
#     |       $H#:            `*MMMMMMMMMMMMMMMMMMMMb#}'  `?       |
#     |       ]MMH#             ""*""""*#MMMMMMMMMMMMM'    -       |
#     |       MMMMMb_                   |MMMMMMMMMMMP'     :       |
#     |       HMMMMMMMHo                 `MMMMMMMMMT       .       |
#     |       ?MMMMMMMMP                  9MMMMMMMM}       -       |
#     |       -?MMMMMMM                  |MMMMMMMMM?,d-    '       |
#     |        :|MMMMMM-                 `MMMMMMMT .M|.   :        |
#     |         .9MMM[                    &MMMMM*' `'    .         |
#     |          :9MMk                    `MMM#"        -          |
#     |            &M}                     `          .-           |
#     |             `&.                             .              |
#     |               `~,   .                     ./               |
#     |                   . _                  .-                  |
#     |                     '`--._,dd###pp=""'                     |
#     |                                                            |
#     |                                                            |
#     | Author: Diego Bruciaferri                                  |
#     | Date and place: 24-06-2021, Met Office, UK                 |
#     |------------------------------------------------------------|


import sys
import os
from os.path import isfile, basename, splitext
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
from scipy.ndimage import gaussian_filter

from lib import *

# ==============================================================================
# 1. Checking for input files
# ==============================================================================

msg_info('GRID ENVELOPES GENERATOR', main=True)

# Checking input file for envelopes
if len(sys.argv) != 2:
   raise TypeError(
         'You need to specify the absolute path of the envelopes input file'
   )

# Reading envelopes infos
env_file = sys.argv[1]
msg_info('Reading envelopes parameters ....')
envInfo = read_envInfo(env_file)

# Reading bathymetry and horizontal grid
msg_info('Reading bathymetry data ... ')
ds_bathy = xr.open_dataset(envInfo.bathyFile).squeeze()
ds_bathy = ds_bathy.set_coords(["nav_lon","nav_lat"])
bathy = ds_bathy["Bathymetry"].squeeze()

msg_info('Reading horizontal grid data ... ')
ds_grid = xr.open_dataset(envInfo.hgridFile) 
glamt = ds_grid["glamt"]
gphit = ds_grid["gphit"]

ds_env = ds_bathy.copy()

# Defining local variables -----------------------------------------------------

num_env = len(envInfo.e_min_ofs)
ni      = glamt.shape[1]
nj      = glamt.shape[0]

e_min_ofs = envInfo.e_min_ofs
e_max_dep = envInfo.e_max_dep
e_loc_vel = envInfo.e_loc_vel
e_loc_var = envInfo.e_loc_var
e_loc_vmx = envInfo.e_loc_vmx
e_loc_rmx = envInfo.e_loc_rmx
e_loc_hal = envInfo.e_loc_hal
e_glo_rmx = envInfo.e_glo_rmx

e_loc_rgn = envInfo.e_loc_rgn
e_loc_mes = envInfo.e_loc_mes
s2z_sigma = envInfo.s2z_sigma
s2z_itera = envInfo.s2z_itera

# Computing LSM
lsm = xr.where(bathy > 0, 1, 0) 

#--------------------------------------------------------------------------------------
# Cretaing envelopes
for env in range(num_env):

    msg = 'ENVELOPE ' + str(env) + ': min_ofs = '+ str(e_min_ofs[env]) + \
          ', max_dep = ' + str(e_max_dep[env]) + ', glo_rmx = ' + str(e_glo_rmx[env])
    msg_info(msg, main=True)

    env_bathy = bathy.copy()

    # =============================================================
    # 1. GEOMETRY of the envelope
    #
    # e_min_ofs[env] > 0     : offset to be added to surf to comupute 
    #                          minimum depth of envelope env
    # e_min_ofs[env] = 'flat': flat envelope with constant depth of
    #                          e_max_dep[env] m
    # e_max_dep[env] = 'max' : e_max_dep[env] = np.amax(bathy)

    msg = '1. Computing the geometry of the envelope'
    msg_info(msg)
    
    if env == 0:
       # For the first envelope, the upper envelope 
       # is the ocean surface at rest
       surf = xr.full_like(bathy,0.,dtype=np.double)
    else:
       # For deeper envelopes, the upper 
       # envelope is the previous envelope
       surf = ds_env["hbatt_"+str(env)]

    if isinstance(e_max_dep[env], str) and e_max_dep[env] == "max": 
       e_max_dep[env] = np.nanmax(bathy)

    # CASE A: flat envelope
    if isinstance(e_min_ofs[env], str) and e_min_ofs[env] == "flat":                     
       msg = 'Generating a flat envelope'
       msg_info(msg)
       hbatt = xr.full_like(bathy,e_max_dep[env],dtype=np.double)
    # CASE B: general envelope
    else:
       msg = 'Generating a general envelope'
       msg_info(msg)
       hbatt = calc_zenv(env_bathy, surf, e_min_ofs[env], e_max_dep[env])  
 
    # MEs-coordinates are tapered in vicinity of the Equator
    if (np.nanmax(gphit) * np.nanmin(gphit)) < 0:
       ztaper = np.exp( -(gphit/8.)**2. )
       hbatt  = np.nanmax(hbatt) * ztaper + hbatt * (1. - ztaper)   

    hbatt.plot()
    plt.show()
    # =============================================================
    # 2. SMOOTHING the envelope
  
    msg = '2. Smoothing the envelope'
    msg_info(msg)  

    # Computing then MB06 Slope Parameter for the raw envelope
    rmax_raw = np.amax(calc_rmax(hbatt)*lsm).values
    msg = '   Slope parameter of raw envelope: rmax = ' + str(rmax_raw)
    msg_info(msg)

    if len(e_loc_vel[env]) == 0:
       hbatt_smt = hbatt.copy()
    else:
       # LOCAL SMOOTHING
       env_tmp = hbatt.copy()
       # 1. Generate 2D map for target rmax
       msk_pge = env_tmp.copy()*0.
       trg_pge = env_tmp.copy()*0.
       for m in range(len(e_loc_vel[env])):

           hal = e_loc_hal[env][m]
              
           filename = e_loc_vel[env][m]
           varname  = e_loc_var[env][m]
           ds_vel = xr.open_dataset(filename)
           hpge = ds_vel[varname].data

           nj = hpge.shape[0]
           ni = hpge.shape[1]
           for j in range(nj):
               for i in range(ni):
                   max_hpge = hpge[j,i]
                   if max_hpge >= e_loc_vmx[env][m]:
                      trg_pge[j-hal:j+hal+1,i-hal:i+hal+1] = msk_pge[j,i] + 1
         
           msk_pge = trg_pge.copy()

       msk_pge.plot.pcolormesh(add_colorbar=True, add_labels=True, \
                                   cbar_kwargs=dict(pad=0.15, shrink=1, \
                                   label='TOTAL HPGE smoothing mask'))
       plt.show()

       msg = '   Total number of points with HPGE > ' + str(e_loc_vmx[env][m]) + ' m/s: ' + str(np.nansum(msk_pge.where(msk_pge>0.)))
       msg_info(msg,)

       # 2. Local smoothing with Martinho & Batteen 2006                   
       for m in range(len(e_loc_rmx[env])):

           r0x = e_loc_rmx[env][m]
           hal = e_loc_hal[env][m]
           msg = ("   Local smoothing of areas of the raw" +\
                  " envelope where HPGE > " + str(e_loc_vmx[env][m]) + " m/s\n"+\
                  "   Envelope  : " + str(env) + "\n"+\
                  "   Local rmax: " + str(r0x) + "\n"+\
                  "   Local halo: " + str(hal) + " cells")
           msg_info(msg)

           msk_smth = msk_pge.where(msk_pge==(m+1)) 

           msg = '   Total number of points where we target rmax is ' + str(r0x) + ': ' + str(np.nansum(msk_smth)) 
           msg_info(msg,)

           msk_smth.plot.pcolormesh(add_colorbar=True, add_labels=True, \
                                   cbar_kwargs=dict(pad=0.15, shrink=1, \
                                   label='RMAX-HPGE smoothing mask'))
           plt.show()

           # smoothing with Martinho & Batteen 2006 
           hbatt_smt = smooth_MB06(env_tmp, r0x)

           # applying smoothing only where HPGE are large
           WRK = hbatt_smt.data
           TMP = env_tmp.data
           WRK[np.isnan(msk_smth)] = TMP[np.isnan(msk_smth)]
           hbatt_smt.data = WRK
           env_tmp = hbatt_smt.copy()

       ds_env["msk_pge"+str(env+1)] = msk_pge

    # GLOBAL SMOOTHING
    if e_glo_rmx[env] > 0:

       msg = ('   Global smoothing of the raw envelope')
       msg_info(msg)

       #da_wrk = hbatt_smt.copy()
       #env_wrk = np.copy(hbatt_smt.data)

       if env == 0:
          # for the first envelope, we follow NEMO approach
          # set first land point adjacent to a wet cell to
          # min_dep as this needs to be included in smoothing
          cst_lsm = lsm.rolling({dim: 3 for dim in lsm.dims}, min_periods=2).sum()
          cst_lsm = cst_lsm.shift({dim: -1 for dim in lsm.dims})
          cst_lsm = (cst_lsm > 0) & (lsm == 0)
          da_wrk = hbatt_smt.where(cst_lsm == 0, e_min_ofs[env])
       else:
          da_wrk = hbatt_smt.copy()

       # smoothing with Martinho & Batteen 2006 
       hbatt_smt = smooth_MB06(da_wrk, e_glo_rmx[env])


    # Computing then MB06 Slope Parameter for the smoothed envelope
    rmax0_smt = calc_rmax(hbatt_smt)*lsm
    rmax0_smt.plot.pcolormesh(add_colorbar=True, add_labels=True, \
                              cbar_kwargs=dict(pad=0.15, shrink=1, \
                              label='Slope parameter'))
    plt.show()
    rmax_smt = np.amax(rmax0_smt).data
    msg = '   Slope parameter of smoothed envelope: rmax = ' + str(rmax_smt)
    msg_info(msg)

    # Saving envelope DataArray
    ds_env["hbatt_"+str(env+1)] = hbatt_smt
    ds_env["rmax0_"+str(env+1)] = rmax0_smt

# -------------------------------------------------------------------------------------
# Setting a localised MEs-coord. system if required
if envInfo.zgridFile != "no":

   msg = 'SETTING A LOCALISED MEs-coordinates system'
   msg_info(msg, main=True)

   # Read distribution of levels of global z-coord. grid
   dsz = xr.open_dataset(envInfo.zgridFile)

   # Computing vertical levels depth
   # We use e3{t,z}_1d to avoid z-partial steps
   e3T = dsz.e3t_1d.broadcast_like(dsz.e3t_0).squeeze()
   e3W = dsz.e3w_1d.broadcast_like(dsz.e3w_0).squeeze()
   gdepw_0, gdept_0 = e3_to_dep(e3W, e3T)

   # Create input mask identifying zone
   # where we want local MEs-coordinates
   # 0 = z-zone, 1 = MEs-zone
   msk_s = generate_mes_area(e_loc_rgn, bathy, e_max_dep[e_loc_mes])
   ds_env["mes_area"] = msk_s
   msk_s.plot()
   plt.show()

   # Creating the transition zone
   msg = '   Using envelope ' + str(num_env-1) + ' to compute the transition zone'
   msg_info(msg)

   env = ds_env["hbatt_"+str(num_env)]
   lev = gdepw_0[{"z":-1}]
   zenv = env.where(msk_s==1,lev)
   
   # 2. Gaussian filtering for identifying 
   #    limits of transition zone
   wrk = zenv.copy()
   for nit in range(s2z_itera):
      zenv_gauss = gaussian_filter(wrk, s2z_sigma)
      zenv_gauss = zenv.where(msk_s==1,zenv_gauss)
      wrk = zenv_gauss.copy()

   # 3. Mask identifying zones
   #    z      = 0
   #    s to z = 1
   #    s      = 2
   msk_zones = xr.full_like(zenv,None,dtype=np.double)
   msk = np.zeros(shape=msk_zones.values.shape)

   diff = zenv_gauss - zenv
   msk[diff.values!=0] = 1
   msk[msk_s==1] = 2
   msk_zones.data = msk
   ds_env["s2z_msk"] = msk_zones

   # 4. Weights to generate transitioning envelope
   msg = '   Computing weights ...'
   msg_info(msg)
   weights = weighting_dist(msk_zones, a=1.7)
   da_wgt = xr.full_like(zenv,None,dtype=np.double)
   da_wgt.data = weights
   ds_env["s2z_wgt"] = da_wgt

   # 5. Creating transitioning envelopes
   env = ds_env["hbatt_"+str(num_env)]
   lev = gdepw_0[{"z":-1}]
   wrk = xr.full_like(env,None,dtype=np.double)
   wrk.data = weights * env.data + (1. - weights) * lev.data
   ds_env["hbatt_"+str(num_env)].data = wrk.data

# -------------------------------------------------------------------------------------   
# Writing the bathy_meter.nc file

msg = 'WRITING the bathy_meter.nc FILE'
msg_info(msg)

out_name = splitext(basename(env_file))[0] + '_maxdep_' + str(e_max_dep[e_loc_mes])

out_file = "bathymetry." + out_name + ".nc"

ds_env.to_netcdf(out_file)

