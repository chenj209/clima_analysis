load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"   
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"    
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"    
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

begin 
; Parameters **********************************
  season  = (/"ANN","JJA","DJF"/)
  
  firstyr = 1980
  lastyr  = 1982
  nyears  = 2
  startfile = 1
  endfile   = 12
  ntime = 12

  expname = (/"SPCAM","NNCAM","NNCAM(PhyC)","CAM5"/)
  ncases = dimsizes(expname)*dimsizes(season)

listyears = "{1999,2000,2001,2002,2003}"
; listyears = "{1998,1999,2000,2001,2002}"
listmonths = (/"01","02","03","04","05","06","07","08","09","10","11","12"/)

field_name = "Precip"

cam5_name = "2022_11_10"
cam5_path = "/temp_share/nncam-cases/nncam-diag_cam5/2022_11_10/atm/hist/"

nncam_name = "2021_11_15"
nncam_path = "/temp_share/nncam-cases/nncam-couple/2021_11_15/atm/hist/"

; nncamrh_name = "baseline_nn_rh"
; nncamrh_path = "/temp_share/nncam-cases/neuroGCM/baseline_nn_rh/atm/hist_old/"
nncamrh_name = "crash1_rh_rerun0612"
nncamrh_path = "/temp_share/stabilities.analysis/hist.plot/hist.nc-data/case1_6years/"

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
  ; lat = f1->lat
  ; lon = f1->lon
  ; Add TRMM file path
  trmm_path = "/cust_users/x-w19/PRECP-obs/TRMM/daily-regrid_res02/"
  trmm_file = "3B42_daily_1998_2010.nc"

  ; Read TRMM data
  f_trmm = addfile(trmm_path + trmm_file, "r")
  trmm_precip = f_trmm->PCP
  trmm_time = f_trmm->TIME
  lat = f_trmm->lat
  lon = f_trmm->lon

  levref = f1->lev
  P0mb =   f1->P0*0.01

  nlat =  dimsizes(lat)
  nlon =  dimsizes(lon)

  field = new((/ncases,nlat,nlon/), float, -9999)
  field!0 = "case"
  field!1 = "lat"
  field!2 = "lon"
  field&lat = lat
  field&lon = lon 
  field_diff = new((/ncases-3,nlat,nlon/), float, -9999)
  field_diff!0 = "case"
  field_diff!1 = "lat"
  field_diff!2 = "lon"
  field_diff&lat = lat
  field_diff&lon = lon

  field_month = new((/12, nlat, nlon/), float, -9999)
  field_month!0 = "month"
  field_month!1 = "lat"
  field_month!2 = "lon"
  field_month&lat = lat
  field_month&lon = lon 


  ; Convert time to datetime
  time_conv = cd_calendar(trmm_time, 0)
  year = time_conv(:,0)
  month = time_conv(:,1)
  day = time_conv(:,2)

  ; Initialize array for monthly means
  trmm_monthly = new((/12, dimsizes(lat), dimsizes(lon)/), float)
  trmm_monthly = 0.0  ; Initialize with zeros
  ; Calculate monthly means for each year in listyears
  do y = 1999, 2003  ; Adjust this range to match your TRMM data
  ; do y = 1998, 2002  ; Adjust this range to match your TRMM data
    do m = 1, 12
      
      indices = ind(year.eq.y .and. month.eq.m)
      
      monthly_avg = dim_avg_n_Wrap(trmm_precip(indices,:,:), 0)
      delete(indices)
      trmm_monthly(m-1,:,:) = trmm_monthly(m-1,:,:) + monthly_avg
    end do
  end do

  ; Calculate multi-year monthly means
  trmm_monthly = trmm_monthly / 5.0  ; Divide by number of years

  ; Calculate seasonal means
  field(0,:,:) = dim_avg_n(trmm_monthly, 0)  ; Annual mean
  field(1,:,:) = dim_avg_n(trmm_monthly(5:7,:,:), 0)  ; JJA mean
  field(2,:,:) = dim_avg_n(trmm_monthly((/11,0,1/),:,:), 0)  ; DJF mean

  print("loading TRMM data completed")

  do i=0, 11
    print(i+1)
    fili = systemfunc("ls "+nncam_path+nncam_name+".cam.h0."+listyears+"-"+listmonths(i)+".nc") 
    f = addfiles(fili(:), "r")
    ListSetType(f, "cat")
    fsns = f[:]->PRECC(:,:,:)
    ; Regrid to TRMM grid if necessary
    if (dimsizes(fsns&lat) .ne. dimsizes(lat) .or. dimsizes(fsns&lon) .ne. dimsizes(lon)) then
      fsns_regrid = linint2_Wrap(fsns&lon, fsns&lat, fsns, True, lon, lat, 0)
    else
      fsns_regrid = fsns
    end if
    field_month(i,:,:) = dim_avg_n_Wrap(fsns_regrid,0)*24*3600*1000
  end do

  field(3,:,:) = dim_avg_n_Wrap(field_month, 0)
  field(4,:,:) = dim_avg_n_Wrap(field_month(5:7,:,:), 0)
  field(5,:,:) = dim_avg_n_Wrap(field_month((/0,1,11/),:,:), 0)

  print("loading nncam data completed")

  do i=0, 11
    print(i+1)
    fili = systemfunc("ls "+nncamrh_path+nncamrh_name+".cam.h0."+listyears+"-"+listmonths(i)+".nc") 
    f = addfiles(fili(:), "r")
    ListSetType(f, "cat")
    fsns = f[:]->PRECC(:,:,:)
    ; Regrid to TRMM grid if necessary
    if (dimsizes(fsns&lat) .ne. dimsizes(lat) .or. dimsizes(fsns&lon) .ne. dimsizes(lon)) then
      fsns_regrid = linint2_Wrap(fsns&lon, fsns&lat, fsns, True, lon, lat, 0)
    else
      fsns_regrid = fsns
    end if
    field_month(i,:,:) = dim_avg_n_Wrap(fsns_regrid,0)*24*3600*1000
  end do

  field(6,:,:) = dim_avg_n_Wrap(field_month, 0)
  field(7,:,:) = dim_avg_n_Wrap(field_month(5:7,:,:), 0)
  field(8,:,:) = dim_avg_n_Wrap(field_month((/0,1,11/),:,:), 0)

  print("loading nncam(rh) data completed")

  do i=0, 11
    print(i+1)
    fili = systemfunc("ls "+cam5_path+cam5_name+".cam.h0."+listyears+"-"+listmonths(i)+".nc") 
    f = addfiles(fili(:), "r")
    ListSetType(f, "cat")
    fsns = f[:]->PRECT(:,:,:)
    ; Regrid to TRMM grid if necessary
    if (dimsizes(fsns&lat) .ne. dimsizes(lat) .or. dimsizes(fsns&lon) .ne. dimsizes(lon)) then
      fsns_regrid = linint2_Wrap(fsns&lon, fsns&lat, fsns, True, lon, lat, 0)
    else
      fsns_regrid = fsns
    end if
    field_month(i,:,:) = dim_avg_n_Wrap(fsns_regrid,0)*24*3600*1000
  end do

  field(9,:,:) = dim_avg_n_Wrap(field_month, 0)
  field(10,:,:) = dim_avg_n_Wrap(field_month(5:7,:,:), 0)
  field(11,:,:) = dim_avg_n_Wrap(field_month((/0,1,11/),:,:), 0)

  print("loading cam5 data completed")
  
  cvar = field
  prob = field


  r1 = new((/ncases/), float, -9999)
  r1!0 = "case"

  rmse = new((/ncases-3/), float, -9999)
  rmse!0 = "case"

  field_diff(0:2,:,:) = field(3:5,:,:) - field(0:2,:,:)
  field_diff(3:5,:,:) = field(6:8,:,:) - field(0:2,:,:)
  field_diff(6:8,:,:) = field(9:11,:,:) - field(0:2,:,:)

  rad    = 4.0*atan(1.0)/180.0
  clat = cos(lat*rad)

  precip_areaavg = wgt_areaave_Wrap(field, clat, 1.0, 1) 
  precip_diff_areaavg = wgt_areaave_Wrap(field_diff, clat, 1.0, 1) 

  rmse(0) = sqrt(wgt_areaave_Wrap(( field(3,:,:) - field(0,:,:) )^2 , clat, 1.0, 0))
  rmse(1) = sqrt(wgt_areaave_Wrap(( field(4,:,:) - field(1,:,:) )^2 , clat, 1.0, 0))
  rmse(2) = sqrt(wgt_areaave_Wrap(( field(5,:,:) - field(2,:,:) )^2 , clat, 1.0, 0))
  rmse(3) = sqrt(wgt_areaave_Wrap(( field(6,:,:) - field(0,:,:) )^2 , clat, 1.0, 0))
  rmse(4) = sqrt(wgt_areaave_Wrap(( field(7,:,:) - field(1,:,:) )^2 , clat, 1.0, 0))
  rmse(5) = sqrt(wgt_areaave_Wrap(( field(8,:,:) - field(2,:,:) )^2 , clat, 1.0, 0))
  rmse(6) = sqrt(wgt_areaave_Wrap(( field(9,:,:) - field(0,:,:) )^2 , clat, 1.0, 0))
  rmse(7) = sqrt(wgt_areaave_Wrap(( field(10,:,:) - field(1,:,:) )^2, clat, 1.0, 0))
  rmse(8) = sqrt(wgt_areaave_Wrap(( field(11,:,:) - field(2,:,:) )^2, clat, 1.0, 0))


  

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
  wks   = gsn_open_wks ("pdf", "2025-"+field_name+"_"+nncamrh_name+"_vs_trmm0903_2003_fix")        ; send graphics to PNG file
  ; wks1   = gsn_open_wks ("png", field_name+"_"+nncam_name+"_vs_spcam")

  plot  = new(12, graphic)
  res                     = True
  res@gsnDraw             = False
  res@gsnFrame            = False
  res@gsnAddCyclic        = True
  res@gsnMaximize         = False
  
  res@cnLineLabelsOn      = False
  res@cnLinesOn           = False
  res@cnFillOn            = True               ; turn on color
  res@lbLabelBarOn        = False
  ; res@lbOrientation       = "Vertical"
  ; res@pmLabelBarParallelPosF = -0.16
  res@pmLabelBarOrthogonalPosF = 0.21
  res@pmLabelBarWidthF    = 0.75


  res@mpMinLonF           = min(lon)
  res@mpMaxLonF           = max(lon)
  res@mpCenterLonF        = 180
  res@mpMinLatF           = min(lat)
  res@mpMaxLatF           = max(lat)

  res@mpDataBaseVersion   = "MediumRes"  
  res@mpDataSetName       = "Earth..4"
  ; res@mpOutlineSpecifiers = (/"China:provinces","India","Nepal","Bhutan","Bangladesh","Myanmar", \
  ;                             "Thailand","Cambodia","Vietnam"/) 
  res@mpNationalLineThicknessF    = 0.5
  res@mpGeophysicalLineThicknessF = 0.5      ; double the thickness of geophysical boundaries

  res@tiMainString        = " "
  res@gsnRightString      = " " ; "mm/day"
  res@gsnLeftStringFontHeightF  = 0.025
  res@gsnRightStringFontHeightF = 0.025
  res@tiXAxisFontHeightF        = 0.015
  res@tmYLLabelFontHeightF      = 0.02
  res@tmXBLabelFontHeightF      = 0.015

  res@txFont = "helvetica-bold"
  res@lbTitleFont = "helvetica-bold"
  res@tiXAxisFont = "helvetica-bold"
  res@tiYAxisFont = "helvetica-bold"
  res@tmXBLabelFont = "helvetica-bold"
  res@tmYLLabelFont = "helvetica-bold"
  res@lbLabelFont = "helvetica-bold"

  pres1 = res 
  pres1@cnFillPalette = "NCV_blu_red" 
  pres1@cnLevelSelectionMode = "ExplicitLevels"
  ; pres1@cnMinLevelValF  = 0.0
  ; pres1@cnMaxLevelValF  = 16
  ; pres1@cnLevelSpacingF = 1

  pres1@cnLevels = (/0.2,0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 17/)

  pres1@gsnLeftString  = "a) TRMM "+season(0)+" mean:"+sprintf("%.3f",precip_areaavg(0))
  plot(0) = gsn_csm_contour_map(wks, field(0,:,:), pres1)

  pres1@gsnLeftString  = "b) TRMM "+season(1)+" mean:"+sprintf("%.3f",precip_areaavg(1))
  plot(1) = gsn_csm_contour_map(wks, field(1,:,:), pres1)

  pres1@gsnLeftString  = "c) TRMM "+season(2)+" mean:"+sprintf("%.3f",precip_areaavg(2))
  plot(2) = gsn_csm_contour_map(wks, field(2,:,:), pres1)

  pres1@gsnLeftString  = "d) "+expname(1)+" "+season(0)+" mean:"+sprintf("%.3f",precip_areaavg(3)) ;+" rmse:"+sprintf("%.3f",rmse(0))
  plot(3) = gsn_csm_contour_map(wks, field(3,:,:), pres1)

  pres1@gsnLeftString  = "e) "+expname(1)+" "+season(1)+" mean:"+sprintf("%.3f",precip_areaavg(4)) ;+" rmse:"+sprintf("%.3f",rmse(1))
  plot(4) = gsn_csm_contour_map(wks, field(4,:,:), pres1)

  pres1@gsnLeftString  = "f) "+expname(1)+" "+season(2)+" mean:"+sprintf("%.3f",precip_areaavg(5)) ;+" rmse:"+sprintf("%.3f",rmse(2))
  plot(5) = gsn_csm_contour_map(wks, field(5,:,:), pres1)

  pres1@gsnLeftString  = "g) "+expname(2)+" "+season(0)+" mean:"+sprintf("%.3f",precip_areaavg(6)) ;+" rmse:0.708";+sprintf("%.3f",rmse(3))
  plot(6) = gsn_csm_contour_map(wks, field(6,:,:), pres1)

  pres1@gsnLeftString  = "h) "+expname(2)+" "+season(1)+" mean:"+sprintf("%.3f",precip_areaavg(7)) ;+" rmse:"+sprintf("%.3f",rmse(4))
  plot(7) = gsn_csm_contour_map(wks, field(7,:,:), pres1)

  pres1@gsnLeftString  = "i) "+expname(2)+" "+season(2)+" mean:"+sprintf("%.3f",precip_areaavg(8)) ;+" rmse:"+sprintf("%.3f",rmse(5))
  plot(8) = gsn_csm_contour_map(wks, field(8,:,:), pres1)

  pres1@gsnLeftString  = "j) "+expname(3)+" "+season(0)+" mean:"+sprintf("%.3f",precip_areaavg(9)) ;+" rmse:"+sprintf("%.3f",rmse(6))
  plot(9) = gsn_csm_contour_map(wks, field(9,:,:), pres1)

  pres1@gsnLeftString  = "k) "+expname(3)+" "+season(1)+" mean:"+sprintf("%.3f",precip_areaavg(10)) ;+" rmse:"+sprintf("%.3f",rmse(7))
  plot(10) = gsn_csm_contour_map(wks, field(10,:,:), pres1)

  pres1@gsnLeftString  = "l) "+expname(3)+" "+season(2)+" mean:"+sprintf("%.3f",precip_areaavg(11)) ;+" rmse:"+sprintf("%.3f",rmse(8))
  plot(11) = gsn_csm_contour_map(wks, field(11,:,:), pres1)


;************************************************
; create panel
;************************************************
  resP                  = True                   ; modify the panel plot
  resP@gsnFrame         = False                  ; don't advance panel plot

  resP@gsnPaperOrientation = "auto"
  ; resP@gsnPaperWidth = 9.5
  ; resP@gsnPaperHeight = 7
  resP@gsnMaximize      = True
  resP@txString         = " "     ; old way of setting main title  

  resP@gsnPanelLabelBar    = True                ; add common colorbar
  ; resP@lbTitleString = "mm/day"
  ; resP@lbTitlePosition = "Top"
  ; resP@lbTitleDirection = "Across"
  ; resP@lbTitleOffsetF = -0.03
  ; resP@lbTitleFontHeightF = 0.014
  ; resP@lbOrientation    = "Vertical"
  resP@pmLabelBarWidthF   = 0.65
  resP@pmLabelBarHeightF  = .04
  resP@lbLabelFontHeightF = .01

  ; res@lbLabelAngleF      = 45
  ; resP@gsnPanelRowSpec  = True
  ; resP@gsnPanelCenter   = False

  ; gsn_panel(wks1, plot, (/3,2/),resP)
  gsn_panel(wks, plot, (/4,3/),resP)               ; now draw as one plot

  ; frame(wks1)
  frame(wks)


  ; delete(wks1)
  delete(wks)
  delete(plot)
  delete(res)
  delete(pres1)
  delete(resP)






  wks   = gsn_open_wks ("pdf", "2024-diff_"+field_name+"_"+nncamrh_name+"_vs_trmm0903_2003_fix")        ; send graphics to PNG file

  plot  = new(9, graphic)
  res                     = True
  res@gsnDraw             = False
  res@gsnFrame            = False
  res@gsnAddCyclic        = True
  res@gsnMaximize         = False
  
  res@cnLineLabelsOn      = False
  res@cnLinesOn           = False
  res@cnFillOn            = True               ; turn on color
  res@lbLabelBarOn        = False
  ; res@lbOrientation       = "Vertical"
  ; res@pmLabelBarParallelPosF = -0.16
  res@pmLabelBarOrthogonalPosF = 0.21
  res@pmLabelBarWidthF    = 0.75


  res@mpMinLonF           = min(lon)
  res@mpMaxLonF           = max(lon)
  res@mpCenterLonF        = 180
  res@mpMinLatF           = min(lat)
  res@mpMaxLatF           = max(lat)

  res@mpDataBaseVersion   = "MediumRes"  
  res@mpDataSetName       = "Earth..4"
  ; res@mpOutlineSpecifiers = (/"China:provinces","India","Nepal","Bhutan","Bangladesh","Myanmar", \
  ;                             "Thailand","Cambodia","Vietnam"/) 
  res@mpNationalLineThicknessF    = 0.5
  res@mpGeophysicalLineThicknessF = 0.5      ; double the thickness of geophysical boundaries

  res@tiMainString        = " "
  res@gsnRightString      = " " ; "mm/day"
  res@gsnLeftStringFontHeightF  = 0.025
  res@gsnRightStringFontHeightF = 0.025
  res@tiXAxisFontHeightF = 0.015
  res@tmYLLabelFontHeightF      = 0.02
  res@tmXBLabelFontHeightF      = 0.015

  res@txFont = "helvetica-bold"
  res@lbTitleFont = "helvetica-bold"
  res@tiXAxisFont = "helvetica-bold"
  res@tiYAxisFont = "helvetica-bold"
  res@tmXBLabelFont = "helvetica-bold"
  res@tmYLLabelFont = "helvetica-bold"
  res@lbLabelFont = "helvetica-bold"

  pres1 = res 
  pres1@cnFillPalette = "NCV_blu_red" 
  ; pres1@cnLevelSelectionMode = "ExplicitLevels"
  ; pres1@cnMinLevelValF  = -5
  ; pres1@cnMaxLevelValF  = 5
  ; pres1@cnLevelSpacingF = 1
  pres1@cnLevelSelectionMode = "ExplicitLevels"

  pres1@cnLevels = (/-10,-9,-8,-7,-6,-5,-4, -3, -2,-1,0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10/)

  pres1@gsnLeftString  = "a) Diff "+expname(1)+" "+season(0)+" rmse:"+sprintf("%.3f",rmse(0));" Bias:"+sprintf("%.3f",precip_diff_areaavg(0))
  plot(0) = gsn_csm_contour_map(wks, field_diff(0,:,:), pres1)

  pres1@gsnLeftString  = "b) Diff "+expname(1)+" "+season(1)+" rmse:"+sprintf("%.3f",rmse(1));" Bias:"+sprintf("%.3f",precip_diff_areaavg(1))
  plot(1) = gsn_csm_contour_map(wks, field_diff(1,:,:), pres1)

  pres1@gsnLeftString  = "c) Diff "+expname(1)+" "+season(2)+" rmse:"+sprintf("%.3f",rmse(2));" Bias:"+sprintf("%.3f",precip_diff_areaavg(2))
  plot(2) = gsn_csm_contour_map(wks, field_diff(2,:,:), pres1)

  ;pres1@gsnLeftString  = "d) Diff "+expname(2)+" "+season(0)+" rmse:0.708";+sprintf("%.3f",rmse(3));" Bias:"+sprintf("%.3f",precip_diff_areaavg(3))
  pres1@gsnLeftString  = "d) Diff "+expname(2)+" "+season(0)+" rmse:"+sprintf("%.3f",rmse(3));" Bias:"+sprintf("%.3f",precip_diff_areaavg(3))
  plot(3) = gsn_csm_contour_map(wks, field_diff(3,:,:), pres1)  

  pres1@gsnLeftString  = "e) Diff "+expname(2)+" "+season(1)+" rmse:"+sprintf("%.3f",rmse(4));" Bias:"+sprintf("%.3f",precip_diff_areaavg(4))
  plot(4) = gsn_csm_contour_map(wks, field_diff(4,:,:), pres1)

  pres1@gsnLeftString  = "f) Diff "+expname(2)+" "+season(2)+" rmse:"+sprintf("%.3f",rmse(5));" Bias:"+sprintf("%.3f",precip_diff_areaavg(5))
  plot(5) = gsn_csm_contour_map(wks, field_diff(5,:,:), pres1)

  pres1@gsnLeftString  = "g) Diff "+expname(3)+" "+season(0)+" rmse:"+sprintf("%.3f",rmse(6));" Bias:"+sprintf("%.3f",precip_diff_areaavg(6))
  plot(6) = gsn_csm_contour_map(wks, field_diff(6,:,:), pres1)

  pres1@gsnLeftString  = "h) Diff "+expname(3)+" "+season(1)+" rmse:"+sprintf("%.3f",rmse(7));" Bias:"+sprintf("%.3f",precip_diff_areaavg(7))
  plot(7) = gsn_csm_contour_map(wks, field_diff(7,:,:), pres1)

  pres1@gsnLeftString  = "i) Diff "+expname(3)+" "+season(2)+" rmse:"+sprintf("%.3f",rmse(8));" Bias:"+sprintf("%.3f",precip_diff_areaavg(8))
  plot(8) = gsn_csm_contour_map(wks, field_diff(8,:,:), pres1)


;************************************************
; create panel
;************************************************
  resP                  = True                   ; modify the panel plot
  resP@gsnFrame         = False                  ; don't advance panel plot
  resP@gsnMaximize      = True
  resP@txString         = " "     ; old way of setting main title  

  resP@gsnPanelLabelBar    = True                ; add common colorbar
  ; resP@lbTitleString = "mm/day"
  ; resP@lbTitlePosition = "Top"
  ; resP@lbTitleDirection = "Across"
  ; resP@lbTitleOffsetF = -0.03
  ; resP@lbTitleFontHeightF = 0.014
  ; resP@lbOrientation    = "Vertical"
  resP@pmLabelBarWidthF   = 0.5
  resP@pmLabelBarHeightF  = .06
  resP@lbLabelFontHeightF = .014
  ; res@lbLabelAngleF      = 45
  ; resP@gsnPanelRowSpec  = True
  ; resP@gsnPanelCenter   = False

  gsn_panel(wks, plot, (/3,3/),resP)               ; now draw as one plot
  frame(wks)


  ; delete(fili)
  ; delete(f)
  ; delete(plot)
  ; delete(time)
  ; delete(PRECT)

  delete(wks)

end
