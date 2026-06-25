export UCX_TLS=tcp
export OMP_NUM_THREADS=1

fov1=0.55
fov2=30
galaxy_type=ALG_YMW16

mkdir -p nest_out/simu
mkdir -p simu

for phis in 1e3 1e4
do
    fout1=simdat_${phis}
    fout2=simdat_${phis}_upper
    fin1=./simu/simdat_${phis}_${fov1}.txt
    fin2=./simu/simdat_${phis}_${fov2}.txt

    # Check input files exist
    if [ ! -f "$fin1" ] || [ ! -f "$fin2" ]; then
        echo "ERROR: Missing simulation input files, run simudat.sh first"
        echo "  Missing: $fin1 or $fin2"
        exit 1
    fi

    echo "Running nest_simu.py for phi*=${phis}"
    mpiexec.hydra -n 4 ./nest_simu.py -f1 $fin1 -f2 $fin2 -o $fout1 -g ${galaxy_type}
    mpiexec.hydra -n 4 ./nest_simu.py -f1 $fin1 -f2 $fin2 -o $fout2 -g ${galaxy_type}
done
exit
