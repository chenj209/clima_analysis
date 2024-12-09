load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"   
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"    
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"    
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
; load "$NCARG_ROOT/lib/ncarg/data/cdf/landsea.nc"

begin 
; Parameters **********************************
  season  = (/"ANN","JJA","DJF"/)
  
  startfile = 1
  endfile   = 12
  ntime = 12

  expname = (/"SPCAM","NNCAM","NNCAM(PhyC)","CAM5"/)
  ncases = dimsizes(expname)*dimsizes(season)

listyears = "{1999,2000,2001,2002,2003}"
listyears1 = "{0000,0001,0002,0003,}"
listmonths = (/"01","02","03","04","05","06","07","08","09","10","11","12"/)

field_name = "Precip"

cam5_name = "2022_11_10"
cam5_path = "/temp_share/nncam-cases/nncam-diag_cam5/2022_11_10/atm/hist/"
nncam_name = "2021_11_15"
nncam_path = "/temp_share/nncam-cases/nncam-couple/2021_11_15/atm/hist/"
; nncamrh_name = "crash1_rh_rerun0612"
; nncamrh_path = "/temp_share/stabilities.analysis/hist.plot/hist.nc-data/case1_6years/"
; nncamrh_name = "new50_nn_rh"
; nncamrh_path = "/temp_share/nncam-cases/neuroGCM/new50_nn_rh/atm/hist/"
nncamrh_name = "baseline_nn_rh"
nncamrh_path = "/temp_share/nncam-cases/neuroGCM/baseline_nn_rh/atm/hist_old/"
spcam_name = "spcam.baseline"
spcam_path = "/temp_share/nncam-cases/nncam-diag_spcam/spcam.baseline/atm/hist/"

  ; mon_name = (/"JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"/)

;***********************************************
  ; if (season.eq."ANN") then 
  ;   fili1 = systemfunc ("ls /home/zgj_stu/WORK2/cuizy/TRMM_3B42_v7/regrid_data_res2/3B42_v7_global_0610monthly.nc")
  ; end if 
  ; if (season.eq."JJA") then 
  ;   fili1 = systemfunc ("ls /home/zgj_stu/WORK2/cuizy/TRMM_3B42_v7/regrid_data_res2/3B42_v7_global_0610_JJA_rg.nc")
  ; end if 
  ; if (season.eq."DJF") then 
  ;   fili1 = systemfunc ("ls /home/zgj_stu/WORK2/cuizy/TRMM_3B42_v7/regrid_data_res2/3B42_v7_global_0610_DJF_rg.nc")
  ; end if 
  ; do i=1,12
  ;     filename = "/home/hyl/plots/amwg_plots/"
  ; end do


  ; timetrmm = f1->TIME
  ; latmm = f1->LAT
  ; lonmm = f1->LON 
  ; ptrmm0 = f1->PCP(:,{-50:50},:)
  ; printMinMax(latmm, False)

  ; ptrmm = 24*ptrmm0 
  ; copy_VarMeta(ptrmm0, ptrmm)

  ; ptrmm_mean = dim_avg_n_Wrap(ptrmm, 0)
  ; printVarSummary(ptrmm_mean)
  ; printMinMax(ptrmm, False)

  ; read data2 -- FHIST cases
  ; ===========================



  f1 = addfile(spcam_path+spcam_name+".cam.h0.1998-01.nc","r")
  hyam = f1->hyam
  hybm = f1->hybm
  lat = f1->lat
  lon = f1->lon
  levref = f1->lev
  P0mb =   f1->P0*0.01

  landfrac = f1->LANDFRAC(0,:,:)

  nlat =  dimsizes(lat)
  nlon =  dimsizes(lon)



  f2 = addfile("./TRMM.day.1999-2003.nc","r")
  lat_trmm = f2->lat
  lon_trmm = f2->lon

  landfrac_trmm = landfrac({lat_trmm(0):lat_trmm(51)},:)

  precip_trmm = f2->precip

  field = new((/ncases+3,nlat/), float, -9999)
  field!0 = "case"
  field!1 = "lat"
  field&lat = lat

  ; field_trmm = new((/3, dimsizes(lat_trmm)/), float, -9999)
  ; field_trmm!0 = "case"
  ; field_trmm!1 = "lat"
  ; field_trmm&lat = lat_trmm

  ; field_trmm_land = new((/3, dimsizes(lat_trmm)/), float, -9999)
  ; field_trmm_land!0 = "case"
  ; field_trmm_land!1 = "lat"
  ; field_trmm_land&lat = lat_trmm

  field_land = new((/ncases+3,nlat/), float, -9999)
  field_land!0 = "case"
  field_land!1 = "lat"
  field_land&lat = lat

  field_ocean = new((/ncases+3,nlat/), float, -9999)
  field_ocean!0 = "case"
  field_ocean!1 = "lat"
  field_ocean&lat = lat

  field_diff = new((/ncases,nlat/), float, -9999)
  field_diff!0 = "case"
  field_diff!1 = "lat"
  field_diff&lat = lat

  field_month = new((/12, nlat, nlon/), float, -9999)
  field_month!0 = "month"
  field_month!1 = "lat"
  field_month!2 = "lon"
  field_month&lat = lat
  field_month&lon = lon 

  field_month_land = new((/12, nlat, nlon/), float, -9999)
  field_month_land!0 = "month"
  field_month_land!1 = "lat"
  field_month_land!2 = "lon"
  field_month_land&lat = lat
  field_month_land&lon = lon 


  trmm_month_land = new((/12, dimsizes(lat_trmm), dimsizes(lon_trmm)/), float, -9999)
  trmm_month_land!0 = "month"
  trmm_month_land!1 = "lat"
  trmm_month_land!2 = "lon"
  trmm_month_land&lat = lat_trmm
  trmm_month_land&lon = lon_trmm

  field_month_ocean = new((/12, nlat, nlon/), float, -9999)
  field_month_ocean!0 = "month"
  field_month_ocean!1 = "lat"
  field_month_ocean!2 = "lon"
  field_month_ocean&lat = lat
  field_month_ocean&lon = lon 


  trmm_month_ocean = new((/12, dimsizes(lat_trmm), dimsizes(lon_trmm)/), float, -9999)
  trmm_month_ocean!0 = "month"
  trmm_month_ocean!1 = "lat"
  trmm_month_ocean!2 = "lon"
  trmm_month_ocean&lat = lat_trmm
  trmm_month_ocean&lon = lon_trmm


  opt = True
  opt@nval_crit = 12   ; require at least 12 values for the "avg" is calculated.

  trmm_month = calculate_monthly_values (precip_trmm, "avg", 0, opt)
  do i=0, 11
    trmm_month_land(i,:,:) = mask(trmm_month(i,:,:),landfrac_trmm.ge.1, True)
    trmm_month_ocean(i,:,:) = mask(trmm_month(i,:,:),landfrac_trmm.le.0, True)
  end do

  field(9,{lat_trmm(0):lat_trmm(51)}) =  dim_avg_n_Wrap(dim_avg_n_Wrap(trmm_month, 0), 1)
  field(10,{lat_trmm(0):lat_trmm(51)}) =  dim_avg_n_Wrap(dim_avg_n_Wrap(trmm_month(5:7,:,:), 0), 1)
  field(11,{lat_trmm(0):lat_trmm(51)}) =  dim_avg_n_Wrap(dim_avg_n_Wrap(trmm_month((/0,1,11/),:,:), 0), 1)

  field_land(9,{lat_trmm(0):lat_trmm(51)}) =  dim_avg_n_Wrap(dim_avg_n_Wrap(trmm_month_land, 0), 1)
  field_land(10,{lat_trmm(0):lat_trmm(51)}) =  dim_avg_n_Wrap(dim_avg_n_Wrap(trmm_month_land(5:7,:,:), 0), 1)
  field_land(11,{lat_trmm(0):lat_trmm(51)}) =  dim_avg_n_Wrap(dim_avg_n_Wrap(trmm_month_land((/0,1,11/),:,:), 0), 1)

  field_ocean(9,{lat_trmm(0):lat_trmm(51)}) =  dim_avg_n_Wrap(dim_avg_n_Wrap(trmm_month_ocean, 0), 1)
  field_ocean(10,{lat_trmm(0):lat_trmm(51)}) =  dim_avg_n_Wrap(dim_avg_n_Wrap(trmm_month_ocean(5:7,:,:), 0), 1)
  field_ocean(11,{lat_trmm(0):lat_trmm(51)}) =  dim_avg_n_Wrap(dim_avg_n_Wrap(trmm_month_ocean((/0,1,11/),:,:), 0), 1)


  do i=0, 11
    print(i+1)
    fili = systemfunc("ls "+spcam_path+spcam_name+".cam.h0."+listyears+"-"+listmonths(i)+".nc") 
    f = addfiles(fili(:), "r")
    ListSetType(f, "cat")
    varorg = f[:]->PRECC(:,:,:)
    ; printVarSummary(fsns)

    
    field_month(i,:,:) = dim_avg_n_Wrap(varorg,0)*24*3600*1000
    ; field_month_land(i,:,:) = dim_avg_n_Wrap(varland,0)*24*3600*1000
    field_month_land(i,:,:) = mask(field_month(i,:,:),landfrac.ge.1, True)
    field_month_ocean(i,:,:) = mask(field_month(i,:,:),landfrac.le.0, True)

  end do

  field(0,:) =  dim_avg_n_Wrap(dim_avg_n_Wrap(field_month, 0), 1)
  field(1,:) =  dim_avg_n_Wrap(dim_avg_n_Wrap(field_month(5:7,:,:), 0), 1)
  field(2,:) =  dim_avg_n_Wrap(dim_avg_n_Wrap(field_month((/0,1,11/),:,:), 0), 1)

  field_land(0,:) =  dim_avg_n_Wrap(dim_avg_n_Wrap(field_month_land, 0), 1)
  field_land(1,:) =  dim_avg_n_Wrap(dim_avg_n_Wrap(field_month_land(5:7,:,:), 0), 1)
  field_land(2,:) =  dim_avg_n_Wrap(dim_avg_n_Wrap(field_month_land((/0,1,11/),:,:), 0), 1)

  field_ocean(0,:) =  dim_avg_n_Wrap(dim_avg_n_Wrap(field_month_ocean, 0), 1)
  field_ocean(1,:) =  dim_avg_n_Wrap(dim_avg_n_Wrap(field_month_ocean(5:7,:,:), 0), 1)
  field_ocean(2,:) =  dim_avg_n_Wrap(dim_avg_n_Wrap(field_month_ocean((/0,1,11/),:,:), 0), 1)

  print("loading spcam data completed")


  do i=0, 11
    print(i+1)
    fili = systemfunc("ls "+nncam_path+nncam_name+".cam.h0."+listyears+"-"+listmonths(i)+".nc") 
    f = addfiles(fili(:), "r")
    ListSetType(f, "cat")
    ; varorg = f[:]->PRECC(:,:,:)
    varorg = f[:]->cp(:,:,:)
    ; printVarSummary(fsns)
    field_month(i,:,:) = dim_avg_n_Wrap(varorg,0)*24*3600*1000
    field_month_land(i,:,:) = mask(field_month(i,:,:),landfrac.ge.1, True)
    field_month_ocean(i,:,:) = mask(field_month(i,:,:),landfrac.le.0, True)
  end do

  field(3,:) = dim_avg_n_Wrap(dim_avg_n_Wrap(field_month, 0), 1)
  field(4,:) = dim_avg_n_Wrap(dim_avg_n_Wrap(field_month(5:7,:,:), 0), 1)
  field(5,:) = dim_avg_n_Wrap(dim_avg_n_Wrap(field_month((/0,1,11/),:,:), 0), 1)

  field_land(3,:) = dim_avg_n_Wrap(dim_avg_n_Wrap(field_month_land, 0), 1)
  field_land(4,:) = dim_avg_n_Wrap(dim_avg_n_Wrap(field_month_land(5:7,:,:), 0), 1)
  field_land(5,:) = dim_avg_n_Wrap(dim_avg_n_Wrap(field_month_land((/0,1,11/),:,:), 0), 1)

  field_ocean(3,:) = dim_avg_n_Wrap(dim_avg_n_Wrap(field_month_ocean, 0), 1)
  field_ocean(4,:) = dim_avg_n_Wrap(dim_avg_n_Wrap(field_month_ocean(5:7,:,:), 0), 1)
  field_ocean(5,:) = dim_avg_n_Wrap(dim_avg_n_Wrap(field_month_ocean((/0,1,11/),:,:), 0), 1)
  print("loading nncam data completed")


  do i=0, 11
    print(i+1)
    fili = systemfunc("ls "+nncamrh_path+nncamrh_name+".cam.h0."+listyears+"-"+listmonths(i)+".nc") 
    f = addfiles(fili(:), "r")
    ListSetType(f, "cat")
    ; varorg = f[:]->PRECC(:,:,:)
    varorg = f[:]->cp(:,:,:)
    ; printVarSummary(fsns)
    field_month(i,:,:) = dim_avg_n_Wrap(varorg,0)*24*3600*1000
    field_month_land(i,:,:) = mask(field_month(i,:,:),landfrac.ge.1, True)
    field_month_ocean(i,:,:) = mask(field_month(i,:,:),landfrac.le.0, True)
  end do

  field(6,:) =  dim_avg_n_Wrap(dim_avg_n_Wrap(field_month, 0), 1)
  field(7,:) =  dim_avg_n_Wrap(dim_avg_n_Wrap(field_month(5:7,:,:), 0), 1)
  field(8,:) =  dim_avg_n_Wrap(dim_avg_n_Wrap(field_month((/0,1,11/),:,:), 0), 1)

  field_land(6,:) =  dim_avg_n_Wrap(dim_avg_n_Wrap(field_month_land, 0), 1)
  field_land(7,:) =  dim_avg_n_Wrap(dim_avg_n_Wrap(field_month_land(5:7,:,:), 0), 1)
  field_land(8,:) =  dim_avg_n_Wrap(dim_avg_n_Wrap(field_month_land((/0,1,11/),:,:), 0), 1)

  field_ocean(6,:) =  dim_avg_n_Wrap(dim_avg_n_Wrap(field_month_ocean, 0), 1)
  field_ocean(7,:) =  dim_avg_n_Wrap(dim_avg_n_Wrap(field_month_ocean(5:7,:,:), 0), 1)
  field_ocean(8,:) =  dim_avg_n_Wrap(dim_avg_n_Wrap(field_month_ocean((/0,1,11/),:,:), 0), 1)

  do i=0, 11
    print(i+1)
    fili = systemfunc("ls "+cam5_path+cam5_name+".cam.h0."+listyears+"-"+listmonths(i)+".nc") 
    f = addfiles(fili(:), "r")
    ListSetType(f, "cat")
    varorg = f[:]->PRECT(:,:,:)
    ; printVarSummary(fsns)
    field_month(i,:,:) = dim_avg_n_Wrap(varorg,0)*24*3600*1000
    field_month_land(i,:,:) = mask(field_month(i,:,:),landfrac.ge.1, True)
    field_month_ocean(i,:,:) = mask(field_month(i,:,:),landfrac.le.0, True)
  end do

  field(9,:) =  dim_avg_n_Wrap(dim_avg_n_Wrap(field_month, 0), 1)
  field(10,:) =  dim_avg_n_Wrap(dim_avg_n_Wrap(field_month(5:7,:,:), 0), 1)
  field(11,:) =  dim_avg_n_Wrap(dim_avg_n_Wrap(field_month((/0,1,11/),:,:), 0), 1)

  field_land(9,:) =  dim_avg_n_Wrap(dim_avg_n_Wrap(field_month_land, 0), 1)
  field_land(10,:) =  dim_avg_n_Wrap(dim_avg_n_Wrap(field_month_land(5:7,:,:), 0), 1)
  field_land(11,:) =  dim_avg_n_Wrap(dim_avg_n_Wrap(field_month_land((/0,1,11/),:,:), 0), 1)

  field_ocean(9,:) =  dim_avg_n_Wrap(dim_avg_n_Wrap(field_month_ocean, 0), 1)
  field_ocean(10,:) =  dim_avg_n_Wrap(dim_avg_n_Wrap(field_month_ocean(5:7,:,:), 0), 1)
  field_ocean(11,:) =  dim_avg_n_Wrap(dim_avg_n_Wrap(field_month_ocean((/0,1,11/),:,:), 0), 1)

  print("loading cam5 data completed")
  
  cvar = field
  prob = field


  r1 = new((/ncases/), float, -9999)
  r1!0 = "case"

  rmse = new((/ncases-3/), float, -9999)
  rmse!0 = "case"

  field_diff(0:2,:) = field(3:5,:) - field(0:2,:)
  field_diff(3:5,:) = field(6:8,:) - field(0:2,:)
  field_diff(6:8,:) = field(9:11,:) - field(0:2,:)


  

 ; ; T-test for the difference  -----------
 ;  siglvl  = 0.05

 ;  vartrmm    = dim_variance_n(ptrmm,0) 
 ;  ; printVarSummary(ptrmm)
 ;  ; printVarSummary(vartrmm)
    
 ;  s1 = ptrmm_mean
 ;  s1 = 60
 ;  s2 = ptrmm_mean
 ;  s2 = 60    ; how many data in one grids which is used to calculate the variance.
 ;  ; printVarSummary(s1)

 ;  iflag   = True                                   ; population variance similar
 ;  tval_opt= False                                  ; p-value only
 ;  prob(0,:,:) = ttest(ptrmm_mean, vartrmm, s1, cpcp(0,:,:), cvar(0,:,:), s2, iflag, tval_opt)
 ;  prob(1,:,:) = ttest(cpcp(0,:,:), cvar(0,:,:), s1, cpcp(1,:,:), cvar(1,:,:), s2, iflag, tval_opt)
 ;  prob(2,:,:) = ttest(cpcp(0,:,:), cvar(0,:,:), s1, cpcp(2,:,:), cvar(2,:,:), s2, iflag, tval_opt)
 ;  ; prob(3,:,:) = ttest(cpcp(0,:,:), cvar(0,:,:), s1, cpcp(3,:,:), cvar(3,:,:), s2, iflag, tval_opt)
 ;  ; prob(4,:,:) = ttest(cpcp(0,:,:), cvar(0,:,:), s1, cpcp(4,:,:), cvar(4,:,:), s2, iflag, tval_opt)
 ;  ; prob(5,:,:) = ttest(cpcp(0,:,:), cvar(0,:,:), s1, cpcp(5,:,:), cvar(5,:,:), s2, iflag, tval_opt)
 ;  ; prob(6,:,:) = ttest(cpcp(0,:,:), cvar(0,:,:), s1, cpcp(6,:,:), cvar(6,:,:), s2, iflag, tval_opt)
 ;  ; prob(7,:,:) = ttest(cpcp(0,:,:), cvar(0,:,:), s1, cpcp(7,:,:), cvar(7,:,:), s2, iflag, tval_opt)
 ;  printVarSummary(prob)

;  printVarSummary(ptrmm_mean)
;  printVarSummary(cpcp)

  ; cdiff(0,:,:) = cpcp(0,:,:) - ptrmm_mean(:,:)
  ; cdiff(1,:,:) = cpcp(1,:,:) - cpcp(0,:,:)
  ; cdiff(2,:,:) = cpcp(2,:,:) - cpcp(0,:,:)
  ; ; cdiff(3,:,:) = cpcp(3,:,:) - cpcp(0,:,:)
  ; ; cdiff(4,:,:) = cpcp(4,:,:) - cpcp(0,:,:)
  ; ; cdiff(5,:,:) = cpcp(5,:,:) - cpcp(0,:,:)
  ; ; cdiff(6,:,:) = cpcp(6,:,:) - cpcp(0,:,:)
  ; ; cdiff(7,:,:) = cpcp(7,:,:) - cpcp(0,:,:)
  ; if (prob.lt.siglvl) then
  ;  . . .   ; difference is significant
  ; end if
  ; cdiff = where(prob.lt.siglvl, cdiff, 0.) ;;95% signigicant level 
  ; print(prob)

  ; ; Spatial-correlation for experiments with TRMM  -----------
  ; ; Method 1: accurate
  ; ; print(r1mean)
  ; ; Method 2: ttested
  ; do i = 0,ncases-1
  ;    r2(i) = pattern_cor(ptrmm_mean, cpcp(i,:,:), 1.0, 1)
  ;    rmse1(i,:) = dim_rmsd_n(ptrmm_mean, cpcp(i,:,:), 1)  ; accurate
  ; end do 
  ; printVarSummary(r2)
  ; print(r2)


; create plot(0-3)CAM
; ===========================
  wks   = gsn_open_wks ("pdf", "1d_"+field_name+"_"+nncamrh_name+"_vs_spcam0903")        ; send graphics to PNG file
  ; wks1   = gsn_open_wks ("png", field_name+"_"+nncam_name+"_vs_spcam")

  dum1a     = new( 6, "graphic")
  plot  = new(6, graphic)
  res                     = True

  res@xyLineThicknesses        = (/3.0,3.0,3.0,2.0, 2.5/)        
  res@xyLineColors             = (/"black","blue","green","red","darkgreen"/)  
  ; res@xyMonoDashPattern        = (/True,True,True,False/)              ; all solid 
  res@xyDashPatterns     = (/0,0,0,0,3/) 
  res@tiYAxisString            = "Precipitation mm/day"
  ; res@tiXAxisString            = "latitude" 


  res@tmXLMode                 = "Explicit"
  res@tmXLValues               = (/-90, -60, -30, 0, 30, 60, 90/)
  res@tmXLLabels               = (/"90S","60S","30S", "0", "30N", "60N", "90N"/)

  res@tmXBLabelFontHeightF     = 0.02
  res@tmYLLabelFontHeightF     = 0.02
  res@gsnLeftStringFontHeightF  = 0.03
  res@gsnRightStringFontHeightF = 0.03

  res@txFont = "helvetica-bold"
  res@lbTitleFont = "helvetica-bold"
  res@tiXAxisFont = "helvetica-bold"
  res@tiYAxisFont = "helvetica-bold"
  res@tmXBLabelFont = "helvetica-bold"
  res@tmYLLabelFont = "helvetica-bold"
  res@lbLabelFont = "helvetica-bold"


  resl = res

  resl@pmLegendDisplayMode    = "Always"            ; turn on legend
  resl@pmLegendSide           = "Top"               ; Change location of 
  resl@pmLegendParallelPosF   = .25                 ; move units right
  resl@pmLegendOrthogonalPosF = -0.4                ; move units down
  resl@pmLegendWidthF         = 0.125               ; Change width and
  resl@pmLegendHeightF        = 0.15                ; height of legend.
  resl@lgPerimOn              = True               ; turn off/on box around
  resl@lgLabelFontHeightF     = .015                ; label font height
  resl@xyExplicitLegendLabels = (/"SPCAM","NNCAM","NNCAM(PhyC)","CAM5","TRMM"/)  ; create explicit labels


  res1 = res

  ; res1@xyLineThicknesses        = (/2.0/)        
  ; res1@xyLineColors             = (/"grey"/)  
  ; res1@xyMonoDashPattern        = True              ; all solid 
  ; res@tiXAxisString            = "latitude" 


  resl@gsnLeftString  = "a) "+season(0)
  resl@gsnRightString = "global"
  plot(0) = gsn_csm_xy (wks, lat, field((/0,3,6,9/),:), resl) 
  dum1a(0) = gsn_add_polyline(wks,plot(0),(/0,0/),(/0,10/), resl)
  ; plot1(0) = gsn_add_polyline(wks,plot(0),lat_trmm, field_trmm(0,:), resl)

  res@gsnLeftString  = "b) "+season(1)
  res@gsnRightString = "global"
  plot(2) = gsn_csm_xy(wks, lat, field((/1,4,7,10/),:), res)
  dum1a(2) = gsn_add_polyline(wks,plot(2),(/0,0/),(/0,12/), res)
  ; plot1(2) = gsn_add_polyline(wks,plot(2),lat_trmm, field_trmm(1,:), res)

  res@gsnLeftString  = "c) "+season(2)
  plot(4) = gsn_csm_xy(wks, lat, field((/2,5,8,11/),:), res)
  dum1a(4) = gsn_add_polyline(wks,plot(4),(/0,0/),(/0,8/), res)
  ; plot1(4) = gsn_add_polyline(wks,plot(4),lat_trmm, field_trmm(2,:), res)

  res@gsnLeftString  = "d) "+season(0)
  res@gsnRightString = "global land"
  plot(1) = gsn_csm_xy (wks, lat, field_land((/0,3,6,9/),:), res) 
  dum1a(1) = gsn_add_polyline(wks,plot(1),(/0,0/),(/0,10/), res)
  ; plot1(1) = gsn_add_polyline(wks,plot(1),lat_trmm, field_trmm_land(0,:), res)

  res@gsnLeftString  = "e) "+season(1)
  plot(3) = gsn_csm_xy(wks, lat, field_land((/1,4,7,10/),:), res)
  dum1a(3) = gsn_add_polyline(wks,plot(3),(/0,0/),(/0,12/), res)
  ; plot1(3) = gsn_add_polyline(wks,plot(3),lat_trmm, field_trmm_land(1,:), res)

  res@gsnLeftString  = "f) "+season(2)
  plot(5) = gsn_csm_xy(wks, lat, field_land((/2,5,8,11/),:), res)
  dum1a(5) = gsn_add_polyline(wks,plot(5),(/0,0/),(/0,10/), res)
  ; plot1(5) = gsn_add_polyline(wks,plot(5),lat_trmm, field_trmm_land(2,:), res)

  ; res@gsnLeftString  = "h) "+season(0)
  ; res@gsnRightString = "global ocean"
  ; plot(2) = gsn_csm_xy (wks, lat, field_ocean((/0,3,6,9/),:), res) 
  ; dum1a(2) = gsn_add_polyline(wks,plot(2),(/0,0/),(/0,10/), res)
  ; ; plot1(1) = gsn_add_polyline(wks,plot(1),lat_trmm, field_trmm_land(0,:), res)

  ; res@gsnLeftString  = "i) "+season(1)
  ; plot(5) = gsn_csm_xy(wks, lat, field_ocean((/1,4,7,10/),:), res)
  ; dum1a(5) = gsn_add_polyline(wks,plot(5),(/0,0/),(/0,12/), res)
  ; ; plot1(3) = gsn_add_polyline(wks,plot(3),lat_trmm, field_trmm_land(1,:), res)

  ; res@gsnLeftString  = "j) "+season(2)
  ; plot(8) = gsn_csm_xy(wks, lat, field_ocean((/2,5,8,11/),:), res)
  ; dum1a(8) = gsn_add_polyline(wks,plot(8),(/0,0/),(/0,8/), res)
  ; plot1(5) = gsn_add_polyline(wks,plot(5),lat_trmm, field_trmm_land(2,:), res)

;************************************************
; create panel
;************************************************

  resP                  = True                   ; modify the panel plot
  resP@gsnFrame         = False                  ; don't advance panel plot
  resP@gsnMaximize      = True
  resP@txString         = " "     ; old way of setting main title  

  gsn_panel(wks, plot, (/3,2/),resP)               ; now draw as one plot

  ; frame(wks1)
  frame(wks)

  end