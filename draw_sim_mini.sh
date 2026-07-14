#!/bin/sh
set -e

# 绘制模拟数据后验分布图
# 输出目录: plots_mini/simu/

mkdir -p plots_mini/simu

for phis in 1e3
do
    o1=./plots_mini/simu/simdat_${phis}.pdf
    in1=./nest_out_mini/simu/simdat_${phis}
    python3 pltpost.py -c config_mini.json -f $in1 -o $o1 -title "Mock data (mini) phi*=${phis}" -up -bo -truth ${in1}.truth.json
done
exit