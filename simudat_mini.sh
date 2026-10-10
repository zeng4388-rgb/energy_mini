#!/bin/sh
# 造 mock: 按注入真值跑前向模型(完整探测选择函数), 供 run_simu_mini.sh 做注入-恢复
set -e

T_START=$(date +%s)
SUMMARY=""
fmt_time() {
    s=$1
    printf "%dm %ds" $((s / 60)) $((s % 60))
}
timed() {
    name=$1; shift
    t0=$(date +%s)
    if "$@"; then tag=ok; else tag=FAIL; fi
    SUMMARY="$SUMMARY  $name : $(fmt_time $(( $(date +%s) - t0 ))) [$tag]
"
    [ "$tag" = ok ]
}
trap 'printf "%s" "$SUMMARY"' EXIT

# FRB number (下调: 1000 -> 300)
Nfrb=300

# E_iso energy function parameters
# logEs/logE0 为 1 GHz 参考带宽口径（2026-10-09 起 dnu=1000）：
# 旧 400 MHz 口径的 43.0/38.0 各 +log10(2.5)=0.398
gamma=-1.5
logEs=43.398
logE0=38.398
dnu=1000

# Width parameters
mu=0.4
sigma=0.3

# 宿主 DM（ln 空间，Zhang 2020 基准）
mu_host=3.50
sigma_host=1.24

# Selection criteria
np=2
sn0=10

# CHIME 望远镜参数 (从 tel_svy.txt)
g=1.4
bw=400
Ts=50
fov=200

mkdir -p simu_mini nest_out_mini/simu

# 缩减: 只跑 phis=1e3 一组
for phis in 1e3
do
    outputfile=./simu_mini/simdat_${phis}_${fov}.txt
    # 实测成本: 约 5.9 s / 3373 事件 / 单核 (Nfrb=300 时远小于此)
    timed "simufrb phis=${phis} N=$Nfrb" python3 simufrb.py \
        -ns $Nfrb -phis $phis -gamma $gamma -logEs $logEs -logE0 $logE0 -dnu $dnu \
        -mu ${mu} -sig ${sigma} -mu_host ${mu_host} -sigma_host ${sigma_host} \
        -ga $g -npol $np -bw $bw -ts $Ts -sn0 $sn0 -fov $fov \
        -out $outputfile --config config_mini.json

    # 写入注入真值(原始物理单位)。绘图端(pltpost -truth / pltpz -truth)统一用
    # frb_util.truth_vector 读这个文件 —— shell 里不再重抄一遍参数。
    truth_file=./nest_out_mini/simu/simdat_${phis}.truth.json
    cat > $truth_file <<EOF
{"phis": ${phis}, "gamma": ${gamma}, "logEs": ${logEs}, "logE0": ${logE0}, "mu_w": ${mu}, "sigma_w": ${sigma}, "ln_mu_host": ${mu_host}, "ln_sigma_host": ${sigma_host}}
EOF
    echo "[truth] 注入真值写入 $truth_file"
done

echo "[simudat_mini.sh 完成, 耗时 $(fmt_time $(( $(date +%s) - T_START )))]"
exit
