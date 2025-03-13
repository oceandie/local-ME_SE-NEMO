import numpy as np
import xarray as xr
import xgcm as xg
import xnemogcm as xn #import open_domain_cfg, open_nemo
import matplotlib.pyplot as plt

# Loading data and preparing the xarray dataset¶

dir_dom_src = "/data/users/dbruciaf/SE-NEMO/se-orca025/se-nemo-domain_cfg/"
lst_dom_src = [dir_dom_src + "mesh_mask_MEs.nc"]
dir_out     = "/data/users/dbruciaf/SE-NEMO/se-orca025/final_15-11-2024/"
lst_out     = [dir_out + "se-nemo_MES_TIDE_average_1991-2019_grid_T.nc"]
ds_src = xn.open_nemo_and_domain_cfg(nemo_files=lst_out, domcfg_files=lst_dom_src).chunk({'z_f': -1, 'z_c':-1})
ds_src = ds_src.squeeze().drop_vars(['t','time_centered','time_centered_bounds','t_bounds']).fillna(0)

dir_dom_dst = "/data/users/dbruciaf/OVF/GOSI9-eORCA025/"
lst_dom_dst = ["mesh_mask.nc"]
ds_dst = xn.open_domain_cfg(datadir=dir_dom_dst, files=lst_dom_dst).chunk({'z_f': -1, 'z_c':-1})

# We load the mask for the localisation area

loc_msk = "/data/users/dbruciaf/SE-NEMO/se-orca025/MEs_450-800_3200/bathymetry.loc_area.dep3200_polglo_sig3_itr1.MEs_4env_450_018-010-010_opt_v2_glo.nc"
ds_loc  = xr.open_dataset(loc_msk)
ds_loc  = ds_loc.rename_dims({'x':'x_c','y':'y_c'})
da_msk  = ds_loc.s2z_msk
da_msk  = da_msk.where(da_msk==0,1)
del ds_loc

# We include the relevant variables in the ds_src dataset

ds_src['loc_msk']        = da_msk
ds_src['gdepw_0_target'] = ds_dst.gdepw_0
ds_src['gdept_0_target'] = ds_dst.gdept_0
ds_src['e3t_0_target']   = ds_dst.e3t_0
ds_src['tmask_target']   = ds_dst.tmask
ds_src = ds_src.set_coords(['gdepw_0_target','gdept_0_target'])
del ds_dst

# We need to have an extensive variable for the conservative interpolation: 
# let's use a proxy of the heat content (T * e3t).
ds_src['Hc'] = (ds_src.thetao_con * ds_src.e3t_0)
ds_src.Hc.attrs = {
    'standard_name':'heat content',
    'units':'K m'
}

# We remove the deepest T point, which is always land in NEMO: 
# needed for the vertical interpolation algorithm.
ds_src = ds_src.isel({'z_c':slice(None,-1)})

# We create the xgcm grid object
grd_src = xg.Grid(ds_src, metrics=xn.get_metrics(ds_src), periodic=False)

# Conservative remapping

# Defining the target depth array abd cleaning obsolete coordinates. 
varW = ['x_c','y_c','gphit','glamt','gdepw_1d','gdepw_0','nav_lat','nav_lon','gdepw_0_target']
target_values = ds_src.gdepw_0_target.isel(x_c=990,y_c=830).drop(varW)
varT = ['x_c','y_c','gphit','glamt','gdept_1d','gdept_0','nav_lat','nav_lon','gdept_0_target']
target_e3t    = ds_src.e3t_0_target.isel(x_c=1050,y_c=1000).drop(varT)
#target_e3t

# Conservative remapping from MEs to z vertical coordinates
transformed_cons = grd_src.transform(
    ds_src.Hc,
    'Z',
    target_values,
    method='conservative',
    target_data=ds_src.gdepw_0
).compute()

transformed_cons = transformed_cons.rename({'z_f':'z_c'}).assign_coords({'z_c':ds_src.z_c})
#transformed_cons.coords

# Recomputing temperature from the heat content defined on the new zps target grid
transformed_t_cons = transformed_cons / target_e3t
transformed_t_cons.compute()

# Applying land-sea masks
#transformed_t_cons.gdept_1d.data = ds_src.gdept_0_target.isel(x_c=1050,y_c=1000).drop(varT).data
transformed_t_cons = transformed_t_cons.where(ds_src.tmask_target == 1)
ds_src['thetao_con']  = ds_src.thetao_con.where(ds_src.tmask == 1)

transformed_t_cons.to_netcdf("se-nemo_GS1p2_full_average_1991-2019_thetao_con_conservative_remap.nc")

