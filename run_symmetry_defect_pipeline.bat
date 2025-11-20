@echo off
cd /d C:\GitHub\substrate_demo

python defect_braid_analysis_gpu.py ^
  --input_dir fermion_output ^
  --prefix yee_sim ^
  --dx 1.0 ^
  --max_track_dist 2.5 ^
  --max_exchange_sep 5.0 ^
  --out_prefix fermion_defects

python defect_braid_analysis_gpu.py ^
  --input_dir boson_output ^
  --prefix yee_sim ^
  --dx 1.0 ^
  --max_track_dist 2.5 ^
  --max_exchange_sep 5.0 ^
  --out_prefix boson_defects

pause
