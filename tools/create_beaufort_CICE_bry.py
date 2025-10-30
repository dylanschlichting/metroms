# Have I ever told you the definition of insanity???
# To do - update time, documentation, and general code quality
import os
import glob
import shutil
import calendar
import xesmf as xe
import netCDF4
import numpy as np
import datetime as dt
import xarray as xr
from collections import OrderedDict as odict
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

def align_times(dsr, dsr_temp, dsf, Tair, convert_hourly_to_daily=True):
    """
    Aligns all times to the most restrictive overlapping range
    and converts Tair to daily if needed.

    Parameters
    ----------
    dsr : xarray.Dataset
        ROMS dataset with 'ocean_time'
    dsr_temp : xarray.Dataset
        ROMS temp dataset with 'temp_time'
    dsf : xarray.Dataset
        Forcing dataset with 'tair_time'
    Tair : xarray.DataArray
        Hourly Tair from forcing dataset
    convert_hourly_to_daily : bool
        Whether to average Tair to daily

    Returns
    -------
    ocean_time_aligned : np.ndarray
    temp_time_aligned : np.ndarray
    Tair_daily : xarray.DataArray
    """
    # Extract time axes in seconds
    ocean_time_sec = dsr['ocean_time'].values
    temp_time_sec = dsr_temp['temp_time'].values
    tair_time_sec = dsf['tair_time'].values

    # Determine overlapping range
    t_start = max(ocean_time_sec.min(), temp_time_sec.min(), tair_time_sec.min())
    t_end   = min(ocean_time_sec.max(), temp_time_sec.max(), tair_time_sec.max())

    # Subset ocean and temp datasets
    ocean_mask = (ocean_time_sec >= t_start) & (ocean_time_sec <= t_end)
    ocean_time_aligned = ocean_time_sec[ocean_mask]

    temp_mask = (temp_time_sec >= t_start) & (temp_time_sec <= t_end)
    temp_time_aligned = temp_time_sec[temp_mask]

    # Subset Tair
    tair_mask = (tair_time_sec >= t_start) & (tair_time_sec <= t_end)
    Tair_sel = Tair[tair_mask]
    tair_sel_sec = tair_time_sec[tair_mask]

    # Convert hourly → daily if requested
    if convert_hourly_to_daily:
        n_hours = len(Tair_sel)
        n_full_days = n_hours // 24
        remainder = n_hours % 24

        Tair_np = Tair_sel.values

        if remainder != 0:
            # pad last partial day with NaNs to make a full day
            pad_width = ((0, 24 - remainder),) + ((0, 0),) * (Tair_np.ndim - 1)
            Tair_np = np.pad(Tair_np, pad_width, mode='constant', constant_values=np.nan)
            n_full_days += 1

        # Reshape and take daily mean (ignores NaNs in last partial day)
        Tair_daily_np = np.nanmean(
            Tair_np.reshape(n_full_days, 24, *Tair_np.shape[1:]), axis=1
        )

        # Time vector for daily Tair
        t_tair_daily = tair_sel_sec[::24][:n_full_days]

        Tair_daily = xr.DataArray(
            Tair_daily_np,
            dims=['tair_time', 'lat', 'lon'],
            coords={'tair_time': t_tair_daily,
                    'lat': Tair.lat,
                    'lon': Tair.lon},
        )

        # Copy original time attributes
        Tair_daily['tair_time'].attrs = dsf['tair_time'].attrs.copy()

    else:
        Tair_daily = Tair_sel

    return ocean_time_aligned, temp_time_aligned, Tair_daily

# ROMS boundary forcing files - we need this for the following variables:
# Sea ice area fraction, thickness, and ice velocities 
roms_path = '/pscratch/sd/b/bundzis/Beaufort_ROMS_2020_dvd_myroms_ice_scratch/Forcing_files/Bryclm/Attempt001/ice_bry_2019_2024_001_nonegthick.nc'
dsr = xr.open_dataset(roms_path)
# Open bry file for temperature and select the top vertical level, need this for interpolation later
path = '/pscratch/sd/b/bundzis/Beaufort_ROMS_2020_dvd_myroms_ice_scratch/Forcing_files/Bryclm/Attempt001/temp_bry_2019_2024_20vert_001_short.nc'
dsr_temp = xr.open_dataset(path).isel(s_rho=-1)

path = '/pscratch/sd/b/bundzis/Beaufort_ROMS_2020_dvd_myroms_ice_scratch/Forcing_files/Bryclm/Attempt001/salt_bry_2019_2024_20vert_001_short.nc'
dsr_salt = xr.open_dataset(path).isel(s_rho=-1)


# Open forcing file. We need this atmospheric temperature to fill in ice surface temperature
# and internal ice temperature later.
path = '/pscratch/sd/b/bundzis/Beaufort_ROMS_2020_dvd_myroms_ice_scratch/Forcing_files/precip_tair_pair_forcing_file_kaktovik_shelf_era5_data_2017_2024_flip_002.nc'
dsf = xr.open_dataset(path)
Tair = dsf['Tair']#.compute() # dims: (time, lat, lon)

ocean_time_aligned, temp_time_aligned, Tair_daily = align_times(dsr, dsr_temp, dsf, Tair)

# Align the datasets to the overlapping time range
dsr = dsr.sel(ocean_time=ocean_time_aligned)
dsr_temp = dsr_temp.sel(temp_time=temp_time_aligned)

# Tair_daily is already aligned and converted to daily
# It has coords 'tair_time' matching the aligned period

# Now - open other files
# Open ROMS grid file
path = '/pscratch/sd/d/dylan617/roms/beaufort_inputs/whats_up_with_grid_files4dylan/KakAKgrd_shelf_big010_smooth006_thin_sponge.nc'
dsg = xr.open_dataset(path)

# Open CICE grid 
grid_file = 'new_cice.grid.nc'
ds_grid = xr.open_dataset(grid_file)

# Horizontal dimensions from the grid
eta_t_len = ds_grid.dims['eta_t']  # 206
xi_t_len  = ds_grid.dims['xi_t']   # 608

# Other CICE-specific dimensions
nkice_len = 7    # number of ice layers (example)
nksnow_len = 1   # number of snow layers (example)
ncat_len = 5     # categories (example)
ntime_len = len(dsr.ocean_time)    # start with one time step

edges = ['E','N','W']

bry_ds = xr.Dataset()

# --- 4D ice layer variables (Sinz, Tinz) ---
for var in ['Sinz', 'Tinz']:
    for edge in edges:
        dim = 'eta_t' if edge in ['E','W'] else 'xi_t'
        shape = (ntime_len, ncat_len, nkice_len, bry_ds.dims.get(dim, eta_t_len if dim=='eta_t' else xi_t_len))
        bry_ds[f'{var}_{edge}_bry'] = xr.DataArray(
            np.full(shape, np.nan, dtype=np.float64),
            dims=('Time','ncat','nkice',dim)
        )

# --- 4D snow layer variable (Tsnz) ---
for edge in edges:
    dim = 'eta_t' if edge in ['E','W'] else 'xi_t'
    shape = (ntime_len, ncat_len, nksnow_len, bry_ds.dims.get(dim, eta_t_len if dim=='eta_t' else xi_t_len))
    bry_ds[f'Tsnz_{edge}_bry'] = xr.DataArray(
        np.full(shape, np.nan, dtype=np.float64),
        dims=('Time','ncat','nksnow',dim)
    )

# --- 3D surface ice variables (Tsfc, aicen, alvln, apondn, vlvln, vicen, vsnon) ---
for var in ['Tsfc', 'aicen', 'alvln', 'apondn', 'vlvln', 'vicen', 'vsnon']:
    for edge in edges:
        dim = 'eta_t' if edge in ['E','W'] else 'xi_t'
        shape = (ntime_len, ncat_len, bry_ds.dims.get(dim, eta_t_len if dim=='eta_t' else xi_t_len))
        bry_ds[f'{var}_{edge}_bry'] = xr.DataArray(
            np.full(shape, 0.0, dtype=np.float64),
            dims=('Time','ncat',dim)
        )

# --- 2D ice age variable (iage) ---
for edge in edges:
    dim = 'eta_t' if edge in ['E','W'] else 'xi_t'
    shape = (ntime_len, bry_ds.dims.get(dim, eta_t_len if dim=='eta_t' else xi_t_len))
    bry_ds[f'iage_{edge}_bry'] = xr.DataArray(
        np.full(shape, 0.0, dtype=np.float64),
        dims=('Time',dim)
    )

# --- 2D melt pond variables (hpondn, ipondn) ---
for var in ['hpondn','ipondn']:
    for edge in edges:
        dim = 'eta_t' if edge in ['E','W'] else 'xi_t'
        shape = (ntime_len, ncat_len, bry_ds.dims.get(dim, eta_t_len if dim=='eta_t' else xi_t_len))
        bry_ds[f'{var}_{edge}_bry'] = xr.DataArray(
            np.full(shape, 0.0, dtype=np.float64),
            dims=('Time','ncat',dim)
        )

# --- 2D brine variables (hbrine, fbrine) ---
for var in ['hbrine','fbrine']:
    for edge in edges:
        dim = 'eta_t' if edge in ['E','W'] else 'xi_t'
        shape = (ntime_len, ncat_len, bry_ds.dims.get(dim, eta_t_len if dim=='eta_t' else xi_t_len))
        bry_ds[f'{var}_{edge}_bry'] = xr.DataArray(
            np.full(shape, 0.0, dtype=np.float64),
            dims=('Time','ncat',dim)
        )

# --- 2D ice velocities (uvel, vvel) with fill values ---
vel_attrs = dict(_FillValue=-32767., missing_value=-32767., units="m s-1",
                 standard_name="eastward_sea_ice_velocity", 
                 cell_methods="area: mean where sea_ice",
                 coordinates="lon lat", grid_mapping="projection_stere")

for var in ['uvel','vvel']:
    for edge in edges:
        dim = 'eta_t' if edge in ['E','W'] else 'xi_t'
        shape = (ntime_len, bry_ds.dims.get(dim, eta_t_len if dim=='eta_t' else xi_t_len))
        bry_ds[f'{var}_{edge}_bry'] = xr.DataArray(
            np.full(shape, -32767., dtype=np.float64),
            dims=('Time',dim),
            attrs=vel_attrs
        )

# Ice thickness category bounds, from Section 2 of 
# https://gmd.copernicus.org/articles/15/4373/2022/
cat_bounds = [0.00, 0.64, 1.39, 2.47, 4.57]
ncat = len(cat_bounds)

def assign_thickness_to_categories(h, ncat, cat_bounds):
    """
    Assign ice thickness array to CICE categories.
    Parameters
    ----------
    h : np.ndarray
        Ice thickness array (ntime, eta_t or xi_t)
    ncat : int
        Number of ice thickness categories
    cat_bounds : list
        Lower bounds of thickness categories
    Returns
    -------
    vicen : np.ndarray
        Array of shape (ntime, ncat, eta_t or xi_t) with ice in categories
    """
    vicen = np.zeros((h.shape[0], ncat, h.shape[1]))
    for i in range(ncat):
        h_low = cat_bounds[i]
        h_high = cat_bounds[i + 1] if i + 1 < ncat else np.inf
        mask = (h >= h_low) & (h < h_high)
        vicen[:, i, :] = mask.astype(float) * h
    return vicen

def assign_concentration_to_categories(aice, vicen_cat):
    """
    Distribute total ice concentration into thickness categories.
    
    Parameters
    ----------
    aice : np.ndarray
        Ice concentration array (ntime, eta_t or xi_t)
    vicen_cat : np.ndarray
        Ice thickness in categories (ntime, ncat, eta_t or xi_t)
    
    Returns
    -------
    aicen_cat : np.ndarray
        Ice concentration in categories (ntime, ncat, eta_t or xi_t)
    """
    # Total thickness per grid cell
    total_thick = vicen_cat.sum(axis=1)  # sum over categories
    # Avoid division by zero
    with np.errstate(invalid='ignore', divide='ignore'):
        aicen_cat = (vicen_cat / total_thick[:, np.newaxis, :]) * aice[:, np.newaxis, :]
        aicen_cat = np.nan_to_num(aicen_cat)  # set NaNs to 0 where total_thick=0
    return aicen_cat

# =====================================================
# CICE Ice thickness (vicen) and ice concentration (aicen)
# The concentration in each category is computed based 
# on thickness
# =====================================================

# --- WEST boundary ---
h_west = dsr['ice_thickness_west'].values
vicen_west = assign_thickness_to_categories(h_west, ncat, cat_bounds)
bry_ds['vicen_W_bry'][:, :, :] = vicen_west

aice_west = dsr['Aice_west'].values
bry_ds['aicen_W_bry'][:, :, :] = assign_concentration_to_categories(aice_west, vicen_west)

# --- EAST boundary ---
h_east = dsr['ice_thickness_east'].values
vicen_east = assign_thickness_to_categories(h_east, ncat, cat_bounds)
bry_ds['vicen_E_bry'][:, :, :] = vicen_east

aice_east = dsr['Aice_east'].values
bry_ds['aicen_E_bry'][:, :, :] = assign_concentration_to_categories(aice_east, vicen_east)

# --- NORTH boundary ---
h_north = dsr['ice_thickness_north'].values
vicen_north = assign_thickness_to_categories(h_north, ncat, cat_bounds)
bry_ds['vicen_N_bry'][:, :, :] = vicen_north

aice_north = dsr['Aice_north'].values
bry_ds['aicen_N_bry'][:, :, :] = assign_concentration_to_categories(aice_north, vicen_north)

# =====================================================
# CICE Ice Velocities (uvel, vvel)
# ROMS values must be interpolated to CICE grid
# =====================================================

# ---------------------------
# WEST boundary (eta_t direction)
# ---------------------------
u_west = dsr['Uice_west'].values
v_west = dsr['Vice_west'].values

v_west_cice = np.empty((v_west.shape[0], eta_t_len))
v_west_cice[:, 1:-1] = 0.5 * (v_west[:, :-1] + v_west[:, 1:])
v_west_cice[:, 0] = v_west[:, 0]
v_west_cice[:, -1] = v_west[:, -1]

bry_ds['uvel_W_bry'][:, :] = u_west
bry_ds['vvel_W_bry'][:, :] = v_west_cice

# ---------------------------
# EAST boundary (eta_t direction)
# ---------------------------
u_east = dsr['Uice_east'].values
v_east = dsr['Vice_east'].values

v_east_cice = np.empty((v_east.shape[0], eta_t_len))
v_east_cice[:, 1:-1] = 0.5 * (v_east[:, :-1] + v_east[:, 1:])
v_east_cice[:, 0] = v_east[:, 0]
v_east_cice[:, -1] = v_east[:, -1]

bry_ds['uvel_E_bry'][:, :] = u_east
bry_ds['vvel_E_bry'][:, :] = v_east_cice

# ---------------------------
# NORTH boundary (xi_t direction)
# ---------------------------
u_north = dsr['Uice_north'].values
v_north = dsr['Vice_north'].values

u_north_cice = np.empty((u_north.shape[0], xi_t_len))
u_north_cice[:, 1:-1] = 0.5 * (u_north[:, :-1] + u_north[:, 1:])
u_north_cice[:, 0] = u_north[:, 0]
u_north_cice[:, -1] = u_north[:, -1]

bry_ds['uvel_N_bry'][:, :] = u_north_cice
bry_ds['vvel_N_bry'][:, :] = v_north

# East boundary
lon_e = dsg.lon_rho[:, -1].values[:, np.newaxis]  # shape (206, 1)
lat_e = dsg.lat_rho[:, -1].values[:, np.newaxis]  # shape (206, 1)

grid_e = xr.Dataset(
    coords={
        'y': np.arange(lon_e.shape[0]),
        'x': np.arange(lon_e.shape[1]),
        'lon': (('y', 'x'), lon_e),
        'lat': (('y', 'x'), lat_e)
    }
)

regridder_e = xe.Regridder(Tair_daily, grid_e, 'nearest_s2d', reuse_weights=False)
Tair_e_line = regridder_e(Tair_daily)  # shape: (time, 206, 1)
Tair_e_line = Tair_e_line[:, :, 0]      # squeeze to (time, 206)
print("East shape",Tair_e_line.shape)

# West boundary (first column) ---
lon_w = dsg.lon_rho[:, 0].values[:, np.newaxis]  # shape (206, 1)
lat_w = dsg.lat_rho[:, 0].values[:, np.newaxis]  # shape (206, 1)

grid_w = xr.Dataset(
    coords={
        'y': np.arange(lon_w.shape[0]),
        'x': np.arange(lon_w.shape[1]),
        'lon': (('y', 'x'), lon_w),
        'lat': (('y', 'x'), lat_w)
    }
)

regridder_e = xe.Regridder(Tair_daily, grid_w, 'nearest_s2d', reuse_weights=False)
Tair_w_line = regridder_e(Tair_daily)  # shape: (time, 206, 1)
Tair_w_line = Tair_w_line[:, :, 0]      # squeeze to (time, 206)
print("West shape:", Tair_w_line.shape)

# --- North boundary (last row) ---
lon_n = dsg.lon_rho[-1, :].values[np.newaxis, :]  # shape (1, N)
lat_n = dsg.lat_rho[-1, :].values[np.newaxis, :]

grid_n = xr.Dataset(
    coords={
        'y': np.arange(lon_n.shape[0]),
        'x': np.arange(lon_n.shape[1]),
        'lon': (('y', 'x'), lon_n),
        'lat': (('y', 'x'), lat_n)
    }
)

regridder_n = xe.Regridder(Tair_daily, grid_n, 'nearest_s2d', reuse_weights=False)
Tair_n_line = regridder_n(Tair_daily)  # shape: (time, 1, N)
Tair_n_line = Tair_n_line[:, 0, :]     # squeeze to (time, N)
print("North shape:", Tair_n_line.shape)

# To double check that the boundaries are correct, you can run
# plt.plot(lon_n[0,:],lat_n[0,:])
# plt.plot(lon_w[:,0],lat_w[:,0])
# plt.plot(lon_e[:,0],lat_e[:,0])

# =====================================================
# Now fill in surface ice temperatures. If Tair >= 0C, 
# set to -0.00001C. Bullet point 3 of Sec. 2.2.2
# =====================================================
# East boundary (eta_t direction)
Tair_e_clip = Tair_e_line.where(Tair_e_line < 0, -0.00001)
for i in range(ncat_len):
    bry_ds['Tsfc_E_bry'][:, i, :] = Tair_e_clip.values

# West boundary (eta_t direction)
Tair_w_clip = Tair_w_line.where(Tair_w_line < 0, -0.00001)
for i in range(ncat_len):
    bry_ds['Tsfc_W_bry'][:, i, :] = Tair_w_clip.values

# North boundary (xi_t direction)
Tair_n_clip = Tair_n_line.where(Tair_n_line < 0, -0.00001)
for i in range(ncat_len):
    bry_ds['Tsfc_N_bry'][:, i, :] = Tair_n_clip.values


# =====================================================
# From cice_bry.py! 
# 
# Additionally, atmosphere (from relevant model at corresponding times) and ocean (from
# clm file) data is used to help specify data for the variablesTsfc, Tinz and Sinz. Tsfc
# is assumed to be equal to the atmosphere surface temp. Tinz is linearly interpolated
# from the ocean surface temp to the air temperature (if there is less than 1cm of snow
# (at the particular grid cell in question)) and to ocean_temp+0.2*|ocean_temp - atm_temp|
# (if there is more than 1cm of snow). Sinz is linearly interpolated from the ocean surface
# salinity to 0.2 of the ocean surface salinity.

# Here's the problem. The ROMS bry temperatures can be much warmer than freezing, ice cannot
# This does not make sense, but I've written the code to do it in case I've misunderstood below

# The paper also doesn't match this:
# Inner snow and ice temperatures were obtained by linearly interpolating between the surface temperature and the freezing water temperature. 
# The same temperature trend was assumed for snow and ice. 
# Therefore, when snow was present, its height was taken into account as the thickness of each ice layer.
# Inner ice salinities were calculated to match multiyear and first-year ice (MYI and FYI, respectively) profiles described in the literature (Gerland et al., 1999). 
# We assumed that when ice thickness was >1.5 m it was MYI, else it was FYI. In the case of MYI, we used the profiles described in older versions of CICE (Hunke et al., 2015, Eq. 76). 
# In the case of FYI, we assumed a “C”-shaped profile defined by Eq. (1) (e.g., Fig. 3 of Gerland et al., 1999):
# Si=19.539Z2i−19.93Zi+8.913,(1)
# where Si is the salinity and Zi is the fractional depth of layer i – zero at the ice top and 1 at the ice bottom.

# OG based on the BRY CODE - Do NOT USE???
# def generate_ice_bry_profiles(Tair_bry, Tocn_bry, Socn_bry=None, hsnow_bry=None, nkice=7):
#     """
#     Generate ice internal temperature (Tinz) and salinity (Sinz) profiles for CICE boundaries.

#     Parameters
#     ----------
#     Tair_bry : dict of ndarray
#         Atmospheric surface temperature at each boundary (Time, horizontal)
#     Tocn_bry : dict of ndarray
#         Ocean surface temperature at each boundary (Time, horizontal)
#     Socn_bry : dict of ndarray, optional
#         Ocean surface salinity at each boundary (Time, horizontal)
#     hsnow_bry : dict of ndarray, optional
#         Snow thickness at each boundary (Time, horizontal). If None, assumed zero.
#     nkice : int
#         Number of vertical ice levels

#     Returns
#     -------
#     Tinz_bry : dict of ndarray
#         Internal ice temperature (Time, ncat, nkice, horizontal)
#     Sinz_bry : dict of ndarray
#         Internal ice salinity (Time, ncat, nkice, horizontal)
#     """

#     Tinz_bry = {}
#     Sinz_bry = {} if Socn_bry is not None else None
#     z = np.linspace(0, 1, nkice)  # normalized depth (0=bottom,1=top)

#     for bndry in Tocn_bry.keys():
#         Tocn = np.array(Tocn_bry[bndry])
#         Tair = np.array(Tair_bry[bndry])
#         nt, nh = Tocn.shape
#         ncat = 1  # you can tile later for ncat

#         if hsnow_bry is None or bndry not in hsnow_bry:
#             hsnow = np.zeros_like(Tocn)
#         else:
#             hsnow = np.array(hsnow_bry[bndry])

#         if Socn_bry is not None and bndry in Socn_bry:
#             Socn = np.array(Socn_bry[bndry])
#         else:
#             Socn = None

#         Tinz = np.empty((nt, ncat, nkice, nh))
#         Sinz = np.empty((nt, ncat, nkice, nh)) if Socn is not None else None

#         # Loop over time and horizontal points
#         for t in range(nt):
#             for i in range(nh):
#                 if hsnow[t, i] < 0.01:  # less than 1 cm of snow
#                     T_top = Tair[t, i]
#                 else:                   # more than 1 cm of snow
#                     T_top = Tocn[t, i] + 0.2 * abs(Tocn[t, i] - Tair[t, i])

#                 # Linear interpolation from bottom (ocean) to top
#                 Tinz[t, 0, :, i] = Tocn[t, i] + (T_top - Tocn[t, i]) * z

#                 if Socn is not None:
#                     # Linear interpolation from ocean salinity to 0.2 * ocean salinity
#                     Sinz[t, 0, :, i] = Socn[t, i] * (1 - 0.8 * z)

#         Tinz_bry[bndry] = Tinz
#         if Sinz_bry is not None:
#             Sinz_bry[bndry] = Sinz

#     return Tinz_bry, Sinz_bry

# Here is a more physically consistent way based on the paper
def generate_ice_bry_profiles(Tair_bry, Socn_bry, hsnow_bry, nkice):
    """
    Generate internal ice temperature (Tinz) and salinity (Sinz) profiles
    for each boundary (E, W, N) based on the seawater freezing temperature.

    Parameters
    ----------
    Tair_bry : dict of ndarray
        Air temperature at each boundary. Each array: (Time, horizontal)
    Socn_bry : dict of ndarray
        Ocean surface salinity at each boundary. Each array: (Time, horizontal)
    hsnow_bry : dict of ndarray
        Snow thickness at each boundary. Each array: (Time, horizontal)
    nkice : int
        Number of vertical ice levels

    Returns
    -------
    Tinz_bry : dict
        Internal ice temperature for each boundary.
        Each array has shape (Time, nkice, horizontal)
    Sinz_bry : dict
        Internal ice salinity for each boundary.
        Each array has shape (Time, nkice, horizontal)
    """
    z = np.linspace(0, 1, nkice)  # normalized vertical coordinate (0=top, 1=bottom)

    Tinz_bry = {}
    Sinz_bry = {}

    for side in Tair_bry.keys():
        Tair = np.array(Tair_bry[side])     # (Time, horizontal)
        Socn = np.array(Socn_bry[side])     # (Time, horizontal)
        hsnow = np.array(hsnow_bry[side])   # (Time, horizontal)

        nt, nh = Tair.shape
        Tinz = np.empty((nt, nkice, nh))
        Sinz = np.empty((nt, nkice, nh))

        for t in range(nt):
            for i in range(nh):
                # Freezing temperature based on ocean salinity [°C]
                T_freeze = -0.054 * Socn[t, i]

                # Top = air temp, bottom = freezing temp
                T_top = Tair[t, i]

                # Linear interpolation (top to bottom)
                Tinz[t, :, i] = T_top - (T_top - T_freeze) * z  # z=0 top, z=1 bottom

                # Ice salinity profile for FYI (C-shaped, absolute PSU)
                Zi = z  # fractional depth (0 top, 1 bottom)
                Sinz[t, :, i] = 19.539*Zi**2 - 19.93*Zi + 8.913

                # Clip physically reasonable bounds
                # -40C
                Tinz[t, :, i] = np.clip(Tinz[t, :, i], -40, 0) 
                # do not exceed ocean salinity, 1e-5 to prevent CICE's advection scheme 
                # from possibly (idk if it is monotonic) creating false extrema
                Sinz[t, :, i] = np.clip(Sinz[t, :, i], 1e-5, Socn[t, i]) 


        Tinz_bry[side] = Tinz
        Sinz_bry[side] = Sinz

    return Tinz_bry, Sinz_bry

nkice_len = 7

Tair_bry = {
    'E': bry_ds['Tsfc_E_bry'][:, 0, :].values,  # (Time, eta_t)
    'W': bry_ds['Tsfc_W_bry'][:, 0, :].values,
    'N': bry_ds['Tsfc_N_bry'][:, 0, :].values,
}

Tocn_bry = {
    'E': dsr_temp.temp_east.values,  # (Time, eta_t)
    'W': dsr_temp.temp_west.values,
    'N': dsr_temp.temp_north.values,
}

Socn_bry = {
    'E': dsr_salt.salt_east.values,
    'W': dsr_salt.salt_west.values,
    'N': dsr_salt.salt_north.values,
}

hsnow_bry = {
    'E': np.zeros_like(dsr_temp.temp_east.values),
    'W': np.zeros_like(dsr_temp.temp_west.values),
    'N': np.zeros_like(dsr_temp.temp_north.values),
}

# If you run with the version in cice_bry.py... not recommended
# Tinz_bry, Sinz_bry = generate_ice_bry_profiles(
#     Tair_bry=Tair_bry,
#     Tocn_bry=Tocn_bry,
#     Socn_bry=Socn_bry,
#     hsnow_bry=hsnow_bry,
#     nkice=nkice_len
# )

# Run as described in the paper
nkice_len = 7
Tinz_bry, Sinz_bry = generate_ice_bry_profiles(
    Tair_bry=Tair_bry,
    Socn_bry=Socn_bry,
    hsnow_bry=hsnow_bry,
    nkice=nkice_len
)

for side, dim in zip(['E', 'W', 'N'], ['eta_t', 'eta_t', 'xi_t']):
    Tinz_full = Tinz_bry[side][:, None, :, :]   # shape (Time, ncat, nkice, nh)
    Sinz_full = Sinz_bry[side][:, None, :, :]

    # Repeat along category axis without flattening horizontal variations
    bry_ds[f'Tinz_{side}_bry'][:, :, :, :] = np.repeat(Tinz_full, ncat, axis=1)
    bry_ds[f'Sinz_{side}_bry'][:, :, :, :] = np.repeat(Sinz_full, ncat, axis=1)

# Last, but certaintly not least, we have snow temperature 
# Set equal to ice temperature based on the paper

# Number of snow layers
nksnow = 1  

Tsnz_bry = {}

for side in ['E', 'W', 'N']:
    # Tinz shape: (Time, nkice, horizontal)
    Tinz = Tinz_bry[side]  

    # Take the top layer (index 0) along the vertical
    Tsnz = Tinz[:, 0, :]  # shape: (Time, horizontal)

    # Expand to include categories and snow layers
    Tsnz_full = np.tile(Tsnz[:, None, None, :], (1, ncat, nksnow, 1))  # (Time, ncat, nksnow, horizontal)

    # Assign to dataset
    bry_ds[f'Tsnz_{side}_bry'][:, :, :, :] = Tsnz_full


# Now create attributes for each variable in bry_ds
# Base variables and theira placeholder attributes
base_attrs = {
    'aicen':  {'units': '',        'standard_name': 'sea_ice_area_fraction'},
    'vicen':  {'units': 'm',      'standard_name': 'sea_ice_thickness'},
    'vsnon':  {'units': 'm',      'standard_name': 'snow_thickness'},
    'Tsfc':   {'units': 'degC',   'standard_name': 'sea_ice_surface_temperature'},
    'Tinz':   {'units': 'degC',   'standard_name': 'sea_ice_internal_temperature'},
    'Sinz':   {'units': 'PSU',    'standard_name': 'sea_ice_internal_salinity'},
    'Tsnz':   {'units': 'degC',   'standard_name': 'snow_temperature'},
    'alvln':  {'units': '',        'standard_name': 'level_ice_area_fraction'},
    'vlvln':  {'units': 'm',      'standard_name': 'level_ice_thickness'},
    'apondn': {'units': '',        'standard_name': 'melt_pond_area_fraction'},
    'hpondn': {'units': 'm',      'standard_name': 'melt_pond_depth'},
    'ipondn': {'units': 'm',      'standard_name': 'melt_pond_ice_thickness'},
    'iage':   {'units': 'yr',     'standard_name': 'sea_ice_age'}
}

# Directions to loop over
directions = ['E_bry', 'W_bry', 'N_bry']

# Apply attributes to all directional variables
for base_var, attrs_dict in base_attrs.items():
    for dir_suffix in directions:
        var_name = f"{base_var}_{dir_suffix}"
        if var_name in bry_ds.variables:
            # Add fill values plus the placeholder units & standard_name
            bry_ds[var_name].attrs.update({
                '_FillValue': -32767.,
                'missing_value': -32767
            })
            bry_ds[var_name].attrs.update(attrs_dict)

global_attrs = {
    'roms_grid': "CICE_b_grid_KakAKgrd_shelf_big010_smooth006_thin_sponge.nc",
    'cice_grid': "KakAKgrd_shelf_big010_smooth006_thin_sponge.nc",
    'type': "CICE ",
    'history': "Created by Dylan Schlichting & Brianna Undzis",
    'Conventions': "CF",
    'Institution': "Los Alamos National Laboratory",
    'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
}

bry_ds.attrs.update(global_attrs)

bry_ds.to_netcdf('CICE_bry_beaufort_roms.nc')