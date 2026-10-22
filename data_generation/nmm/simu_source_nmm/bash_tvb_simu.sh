for R in {10..998..10}
do
python generate_tvb_data.py --folder nmm_simu_tvb --a_start $(($R-10)) --a_end $R
sleep 60
done
python generate_tvb_data.py --folder nmm_simu_tvb --a_start 990 --a_end 998

