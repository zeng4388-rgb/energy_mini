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

# 链目录: 默认读 run 脚本维护的 latest 符号链接（运行目录制, B3）；
# 手工指定某次提交: CC_SIMU_DIR=nest_out_mini/simu/<tag> sh draw_sim_mini.sh
SIMU_DIR=${CC_SIMU_DIR:-nest_out_mini/simu/latest}
if [ ! -d "$SIMU_DIR" ]; then
    echo "ERROR: 链目录 $SIMU_DIR 不存在（先跑 run_simu_mini.sh，或用 CC_SIMU_DIR=nest_out_mini/simu/<tag> 指定某次提交）" >&2
    exit 1
fi

for phis in 1e3
do
    in1=./$SIMU_DIR/simdat_${phis}
    # truth.json 是 simudat 写的【输入】, 留平铺路径; 链在 tag 目录（运行目录制, B3）
    timed "pltpost simdat_${phis}" python3 pltpost.py -c config_mini.json \
        -f "$in1" -o ./plots_mini/simu/simdat_${phis}.pdf \
        -title "Mock data (mini) phi*=${phis}" -up -bo \
        -truth ./nest_out_mini/simu/simdat_${phis}.truth.json

    # 逐事件 P(z) 恢复检验: 参数取注入真值(读同一个 .truth.json, 不在 shell 里重抄),
    # 输出页 3 就是 z_med vs 注入 z + 标准化残差分布。truth.json 是【输入】留平铺路径,
    # 链在 tag 目录里 ⇒ 这里显式指回平铺位置。
    timed "pltpz 恢复 simdat_${phis}" python3 pltpz.py --config config_mini.json \
        -f1 ./simu_mini/simdat_${phis}_${fov}.txt \
        -truth ./nest_out_mini/simu/simdat_${phis}.truth.json \
        -o ./plots_mini/simu/simdat_${phis}_pz.pdf
done

echo "[draw_sim_mini.sh 完成, 耗时 $(fmt_time $(( $(date +%s) - T_START )))]"
exit
