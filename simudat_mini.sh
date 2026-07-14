#!/bin/sh
set -e

# FRB number (加大: 20 -> 300, 提升推断样本量)
Nfrb=300

# E_iso energy function parameters
alpha=-1.5
logEs=43.0
logE0=38.0
dnu=400

# Width parameters
mu=0.4
sigma=0.3

# Selection criteria
np=2
sn0=10

# CHIME 望远镜参数 (从 tel_svy.txt)
g=1.4
bw=400
Ts=50
fov=200

# Host galaxy categories
galaxy_type=ALG_YMW16

# 创建输出目录
mkdir -p simu_mini nest_out_mini/simu

# 缩减: 只跑 phis=1e3 一组
for phis in 1e3
do
    outputfile=./simu_mini/simdat_${phis}_${fov}.txt
    echo "python3 simufrb.py -ns $Nfrb -phis $phis -alpha $alpha -logEs $logEs -logE0 $logE0 -dnu $dnu -fgt $galaxy_type -mu ${mu} -sig ${sigma} -ga $g -npol $np -bw $bw -ts $Ts -sn0 $sn0 -fov $fov -out $outputfile --config config_mini.json"
    python3 simufrb.py -ns $Nfrb -phis $phis -alpha $alpha -logEs $logEs -logE0 $logE0 -dnu $dnu -fgt $galaxy_type -mu ${mu} -sig ${sigma} -ga $g -npol $np -bw $bw -ts $Ts -sn0 $sn0 -fov $fov -out $outputfile --config config_mini.json

    # 写入注入真值(原始物理单位),供 pltpost.py 画 truth 竖线
    # 路径与 pltpost.py 的 -f 参数(./nest_out_mini/simu/simdat_${phis})绑定
    truth_file=./nest_out_mini/simu/simdat_${phis}.truth.json
    cat > $truth_file <<EOF
{"phis": ${phis}, "alpha": ${alpha}, "logEs": ${logEs}, "logE0": ${logE0}, "mu_w": ${mu}, "sigma_w": ${sigma}}
EOF
    echo "[truth] 注入真值写入 $truth_file"
done
exit
