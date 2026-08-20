@echo off
call "C:\Users\hansu\anaconda3\Scripts\activate.bat" pcn
cd /d C:\Users\hansu\Desktop\inven
python compare_mopsoams_vs_pcn_pbestseed.py > compare_mopsoams_vs_pcn_pbestseed_output.log 2>&1
