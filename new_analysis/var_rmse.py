def get_FLNS():
    case_name = ['CNTL','T048','T060','OBS']
    dataset = {}
    
    ########### data 
    for c in case_name:
        if c == 'OBS':
            data_path = f'/lcrc/group/e3sm/diagnostics/observations/Atm/climatology/ceres_ebaf_surface_v4.1/ceres_ebaf_surface_v4.1_ANN_200101_201812_climo.nc'
            fid = xr.open_dataset(data_path)
            rlds = fid['rlds'][0,...]
            rlus = fid['rlus'][0,...]
            flns = rlus - rlds
            
        else:
            data_path = f'/home/ac.tzhang/large/tune_20231222/{c}/post/atm/180x360_aave/clim/10yr/20231213.v3.LR.piControl-PPE.tune.chrysalis_ANN_010101_011012_climo.nc'
            fid = xr.open_dataset(data_path)
            lat = fid['lat']
            lon = fid['lon']
            hyam = fid['hyam']
            hybm = fid['hybm']
            P0 = fid['P0']
            PS = fid['PS']
            flns = fid['FLNS'][0,...]
    
        dataset[c] = flns

    return dataset,lat,lon

def get_LWCF():
    case_name = ['CNTL','T048','T060','OBS']
    dataset = {}
    
    ########### data 
    for c in case_name:
        if c == 'OBS':
            data_path = f'/lcrc/group/e3sm/diagnostics/observations/Atm/climatology/ceres_ebaf_toa_v4.1/ceres_ebaf_toa_v4.1_ANN_200101_201812_climo.nc'
            fid = xr.open_dataset(data_path)
            data =fid['rlutcs'][0,...] - fid['rlut'][0,...]
            
        else:
            data_path = f'/home/ac.tzhang/large/tune_20231222/{c}/post/atm/180x360_aave/clim/10yr/20231213.v3.LR.piControl-PPE.tune.chrysalis_ANN_010101_011012_climo.nc'
            fid = xr.open_dataset(data_path)
            lat = fid['lat']
            lon = fid['lon']
            data = fid['LWCF']
    
            data = data[0,...]
    
        dataset[c] = data

    return dataset,lat,lon
    

def get_SWCF():
    case_name = ['CNTL','T048','T060','OBS']
    dataset = {}
    
    ########### data 
    for c in case_name:
        if c == 'OBS':
            data_path = f'/lcrc/group/e3sm/diagnostics/observations/Atm/climatology/ceres_ebaf_toa_v4.1/ceres_ebaf_toa_v4.1_ANN_200101_201812_climo.nc'
            fid = xr.open_dataset(data_path)
            data =fid['rsutcs'][0,...] - fid['rsut'][0,...]
            
        else:
            data_path = f'/home/ac.tzhang/large/tune_20231222/{c}/post/atm/180x360_aave/clim/10yr/20231213.v3.LR.piControl-PPE.tune.chrysalis_ANN_010101_011012_climo.nc'
            fid = xr.open_dataset(data_path)
            lat = fid['lat']
            lon = fid['lon']
            data = fid['SWCF']
    
            data = data[0,...]
    
        dataset[c] = data

    return dataset,lat,lon
    

def get_Z500():
    case_name = ['CNTL','T048','T060','OBS']
    dataset = {}
    
    ########### data 
    for c in case_name:
        if c == 'OBS':
            data_path = f'/lcrc/group/e3sm/diagnostics/observations/Atm/climatology/ERA5/ERA5_ANN_197901_201912_climo.nc'
            fid = xr.open_dataset(data_path)
            zg500 = fid['zg'][0,15,...].interp(lat=lat_mod,lon=lon_mod,method="nearest")
            lat_obs = fid['lat']
            lon_obs = fid['lon']
            
        else:
            data_path = f'/home/ac.tzhang/large/tune_20231222/{c}/post/atm/180x360_aave/clim/10yr/20231213.v3.LR.piControl-PPE.tune.chrysalis_ANN_010101_011012_climo.nc'
            fid = xr.open_dataset(data_path)
            lat_mod = fid['lat']
            lon_mod = fid['lon']
            hyam = fid['hyam']
            hybm = fid['hybm']
            P0 = fid['P0']
            PS = fid['PS']
            Z3 = fid['Z3']
    
            pressure = hyam * P0 + hybm * PS
    
            zg500 = logLinearInterpolation(Z3, pressure)[0,5,...]
            
        dataset[c] = zg500  

    return dataset, lat_mod,lon_mod

def get_T850():
    case_name = ['CNTL','T048','T060','OBS']
    dataset = {}
    
    ########### data 
    for c in case_name:
        if c == 'OBS':
            data_path = f'/lcrc/group/e3sm/diagnostics/observations/Atm/climatology/ERA5/ERA5_ANN_197901_201912_climo.nc'
            fid = xr.open_dataset(data_path)
            T850 = fid['ta'][0,6,...].interp(lat=lat_mod,lon=lon_mod,method="nearest")
            lat_obs = fid['lat']
            lon_obs = fid['lon']
            
        else:
            data_path = f'/home/ac.tzhang/large/tune_20231222/{c}/post/atm/180x360_aave/clim/10yr/20231213.v3.LR.piControl-PPE.tune.chrysalis_ANN_010101_011012_climo.nc'
            fid = xr.open_dataset(data_path)
            lat_mod = fid['lat']
            lon_mod = fid['lon']
            hyam = fid['hyam']
            hybm = fid['hybm']
            P0 = fid['P0']
            PS = fid['PS']
            T = fid['T']
    
            pressure = hyam * P0 + hybm * PS
            T850 = logLinearInterpolation(T, pressure)[0,2,...]
            T850 = T850.where(T850 > 100, drop=False)
    
    
        dataset[c] = T850        

    return dataset,lat_mod,lon_mod

def get_U850():
    case_name = ['CNTL','T048','T060','OBS']
    dataset = {}
    
    ########### data 
    for c in case_name:
        if c == 'OBS':
            data_path = f'/lcrc/group/e3sm/diagnostics/observations/Atm/climatology/ERA5/ERA5_ANN_197901_201912_climo.nc'
            fid = xr.open_dataset(data_path)
            data = fid['ua'][0,6,...].interp(lat=lat_mod,lon=lon_mod,method="nearest")
            lat_obs = fid['lat']
            lon_obs = fid['lon']
            
        else:
            data_path = f'/home/ac.tzhang/large/tune_20231222/{c}/post/atm/180x360_aave/clim/10yr/20231213.v3.LR.piControl-PPE.tune.chrysalis_ANN_010101_011012_climo.nc'
            fid = xr.open_dataset(data_path)
            lat_mod = fid['lat']
            lon_mod = fid['lon']
            hyam = fid['hyam']
            hybm = fid['hybm']
            P0 = fid['P0']
            PS = fid['PS']
            U = fid['U']
    
            pressure = hyam * P0 + hybm * PS
            data = logLinearInterpolation(U, pressure)[0,2,...]
    
    
        dataset[c] = data        

    return dataset,lat_mod,lon_mod


def get_PRECT():
    case_name = ['CNTL','T048','T060','OBS']
    dataset = {}
    
    ########### data 
    for c in case_name:
        if c == 'OBS':
            data_path = f'/lcrc/group/e3sm/diagnostics/observations/Atm/climatology/GPCP_v3.2/GPCP_v3.2_ANN_198301_202112_climo.nc'
            fid = xr.open_dataset(data_path)
            data = fid['sat_gauge_precip'][0,...].interp(lat=lat_mod,lon=lon_mod,method="nearest")
            
        else:
            data_path = f'/home/ac.tzhang/large/tune_20231222/{c}/post/atm/180x360_aave/clim/10yr/20231213.v3.LR.piControl-PPE.tune.chrysalis_ANN_010101_011012_climo.nc'
            fid = xr.open_dataset(data_path)
            lat_mod = fid['lat']
            lon_mod = fid['lon']
            data = (fid['PRECC'] + fid['PRECL']) * 86400 * 1000
    
            data = data[0,...]
    
        dataset[c] = data 

    return dataset,lat_mod,lon_mod
    

def area_RMSE(model, obs):
    '''
    calculate weighted area RMSE
    input data is xarray DataArray
    '''
    weights = np.cos(np.deg2rad(model.lat))
    weights.name = "weights"
    # available in xarray version 0.15 and later

    bias_2 = (model - obs) ** 2

    bias_2_weighted = bias_2.weighted(weights)
    bias_2_mean = bias_2_weighted.mean(("lat","lon"))
    
 
    return bias_2_mean ** 0.5



cluster_10_rmse = area_RMSE(dataset['T048'],dataset['OBS']).data