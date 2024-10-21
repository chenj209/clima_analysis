spcam_path    = "/temp_share/nncam-cases/nncam-diag_spcam/spcam.baseline/atm/hist/"
nncam_path    = "/temp_share/nncam-cases/nncam-couple/2021_11_15/atm/hist/"
nncamrh_path  = "/temp_share/stabilities.analysis/hist.plot/hist.nc-data/case1_6years/"
# nncamrh_path  = "/temp_share/nncam-cases/neuroGCM/baseline_nn_rh/atm/hist/"
cam5_path     = "/temp_share/nncam-cases/nncam-diag_cam5/2022_11_10/atm/hist/"
rb1_path = "/share3/chenj209/CESM_logs/replay_buffer_spinup5_seed1117_full/nc_files/"
rb2_path = "/share3/chenj209/CESM_logs/replay_buffer_spin5_seed1_full/nc_files/"
norb1_path = "/share3/chenj209/CESM_logs/noreplay_buffer_noprevQT_spinup5_seed1117_full/nc_files/"
rb1117_noprevQT_path = "/share3/chenj209/CESM/conv_mem_spinup5/run/"

NUM_CASES = 8
CASE_NAMES = ["spcam.baseline", "2021_11_15", "crash1_rh_rerun0612", "2022_11_10", "conv_mem_spinup5", \
        "conv_mem_spinup5", "conv_mem_spinup5", "conv_mem_spinup5"]
DATA_PATHS = [spcam_path, nncam_path, nncamrh_path, cam5_path, rb1_path, rb2_path, norb1_path, rb1117_noprevQT_path]
NUM_MONTHS = 12*6
START_YEAR = 1998
END_YEAR = 2003 # including
