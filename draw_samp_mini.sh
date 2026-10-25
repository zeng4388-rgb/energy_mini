#!/bin/sh
# 真实样本出图: 8 维角分布 × 3 条 MW 线 + 逐事件 P(z) × 3 条 MW 线
# 线名 = -mw 的大写(NE2001 / YMW16 / NE2025)，与 run_samp_mini.sh 的 -o 一致；
# 旧名 ALG_* 来自已删除的星系类型×MW 宿主标定表（2026-10-11 改）。
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
# 失败时摘要也要打出来, 所以挂在 EXIT trap 上
trap 'printf "%s" "$SUMMARY"' EXIT

# 输出目录: plots_mini/samp/
mkdir -p plots_mini/samp

for mw in ne2001 ymw16 ne2025
do
    out=$(echo "$mw" | tr 'a-z' 'A-Z')

    echo "Plotting posterior for ${out} (MW=${mw})..."
    timed "pltpost ${out}" python3 pltpost.py -c config_mini.json \
        -f "./nest_out_mini/samp/${out}" -o "./plots_mini/samp/${out}.pdf" \
        -title "Real Sample (mini) - MW ${out}" -up -bo

    # 逐事件红移后验 P(z): 参数取各自 MultiNest 后验中位数。
    # MW 模型对 P(z) 最敏感(扣多扣少直接平移 P(z)), 三线对比即系统误差可视化。
    echo "Plotting P(z) for ${out}..."
    timed "pltpz   ${out}" python3 pltpz.py --config config_mini.json \
        -mw "${mw}" -nest "./nest_out_mini/samp/${out}" \
        -o "./plots_mini/samp/${out}_pz.pdf"
done

echo "[draw_samp_mini.sh 完成, 耗时 $(fmt_time $(( $(date +%s) - T_START )))]"
exit 0
