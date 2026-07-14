#!/bin/sh
# 模拟数据嵌套采样(单任务)

set -e
export OMP_NUM_THREADS=1

fov=200
galaxy_type=ALG_YMW16

mkdir -p nest_out_mini/simu
mkdir -p simu_mini

for phis in 1e3
do
    fout1=simdat_${phis}
    fin1=./simu_mini/simdat_${phis}_${fov}.txt

    # 检查输入文件是否存在
    if [ ! -f "$fin1" ]; then
        echo "ERROR: 缺少模拟输入文件，请先运行 simudat_mini.sh 生成数据"
        echo "  缺失: $fin1"
        exit 1
    fi

    echo "Running nest_simu_mini.py for phi*=${phis}"
    python3 nest_simu_mini.py --config config_mini.json -f1 $fin1 -o $fout1 -g ${galaxy_type}
done
exit
