@echo off
setlocal

REM --- Basic parameters (edit if your box size changes) ---
set Lx=64
set Ly=64
set DX=1.0
set NBINS_R=40
set NCELLS_X=16
set NCELLS_Y=16

echo [RUN] All charges (q = +1 and -1)...
python defect_fermion_boson_test.py --defects_csv yee_coupled_defects_defects.csv --Lx %Lx% --Ly %Ly% --dx %DX% --n_bins_r %NBINS_R% --n_cells_x %NCELLS_X% --n_cells_y %NCELLS_Y% --charge all

echo.
echo [RUN] Positive defects only (q = +1)...
python defect_fermion_boson_test.py --defects_csv yee_coupled_defects_defects.csv --Lx %Lx% --Ly %Ly% --dx %DX% --n_bins_r %NBINS_R% --n_cells_x %NCELLS_X% --n_cells_y %NCELLS_Y% --charge plus

echo.
echo [RUN] Negative defects only (q = -1)...
python defect_fermion_boson_test.py --defects_csv yee_coupled_defects_defects.csv --Lx %Lx% --Ly %Ly% --dx %DX% --n_bins_r %NBINS_R% --n_cells_x %NCELLS_X% --n_cells_y %NCELLS_Y% --charge minus

echo.
echo [DONE] All three analyses finished.
endlocal
pause
