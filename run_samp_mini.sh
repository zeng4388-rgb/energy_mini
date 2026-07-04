export OMP_NUM_THREADS=1

mkdir -p nest_out_mini/samp

for fgtype in ALG_NE2001 ALG_YMW16
do
    echo "Running nest_samp_mini.py with galaxy type: ${fgtype}"
    python3 nest_samp_mini.py --fits --config config_mini.json -g ${fgtype} -o ${fgtype}
done
