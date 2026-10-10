#!/bin/sh
# 模拟数据出图: 8 维角分布(叠注入真值竖线) + 逐事件 P(z) 恢复检验
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

fov=200

# 输出目录: plots_mini/simu/
mkdir -p plots_mini/simu

for phis in 1e3
do
    in1=./nest_out_mini/simu/simdat_${phis}
    timed "pltpost simdat_${phis}" python3 pltpost.py -c config_mini.json \
        -f "$in1" -o ./plots_mini/simu/simdat_${phis}.pdf \
        -title "Mock data (mini) phi*=${phis}" -up -bo -truth "${in1}.truth.json"

    # 逐事件 P(z) 恢复检验: 参数取注入真值(读同一个 .truth.json, 不在 shell 里重抄),
    # 输出页 3 就是 z_med vs 注入 z + 标准化残差分布
    timed "pltpz 恢复 simdat_${phis}" python3 pltpz.py --config config_mini.json \
        -f1 ./simu_mini/simdat_${phis}_${fov}.txt -truth "${in1}.truth.json" \
        -o ./plots_mini/simu/simdat_${phis}_pz.pdf
done

echo "[draw_sim_mini.sh 完成, 耗时 $(fmt_time $(( $(date +%s) - T_START )))]"
exit
