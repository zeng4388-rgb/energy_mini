#!/bin/sh
# 真实样本出图: 8 维角分布 × 3 条 MW 线 + 逐事件 P(z) × 3 条 MW 线
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

# 输出目录: plots_mini/samp/
mkdir -p plots_mini/samp

for fgtype in ALG_NE2001 ALG_YMW16 ALG_NE2025
do
    echo "Plotting posterior for ${fgtype}..."
    timed "pltpost ${fgtype}" python3 pltpost.py -c config_mini.json \
        -f ./nest_out_mini/samp/${fgtype} -o ./plots_mini/samp/${fgtype}.pdf \
        -title "Real Sample (mini) - ${fgtype}" -up -bo
done

# 逐事件红移后验 P(z): 三条线各出一版(参数取各自 MultiNest 后验中位数)。
# MW 模型对 P(z) 最敏感(扣多扣少直接平移 P(z)), 三线对比即系统误差可视化。
# 二元组: 输出名 : MW 模型(-g 只是标签, 宿主参数已是第 7/8 维自由参数)
for cfg in "ALG_NE2001:ne2001" "ALG_YMW16:ymw16" "ALG_NE2025:ne2025"
do
    out=$(echo $cfg | cut -d: -f1)
    mw=$(echo $cfg | cut -d: -f2)
    echo "Plotting P(z) for ${out}..."
    timed "pltpz   ${out}" python3 pltpz.py --config config_mini.json \
        -g ${out} -mw ${mw} -nest ./nest_out_mini/samp/${out} \
        -o ./plots_mini/samp/${out}_pz.pdf
done

echo "[draw_samp_mini.sh 完成, 耗时 $(fmt_time $(( $(date +%s) - T_START )))]"
exit
