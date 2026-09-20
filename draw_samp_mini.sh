#!/bin/sh
set -e

# 脚本级计时
T_START=$(date +%s)
fmt_time() {
    s=$1
    printf "%dm %ds" $((s / 60)) $((s % 60))
}

# 真实样本后验分布图: 2 条 MW 模型线 + NE2025 线 + 宿主 vpar 敏感性线
# 输出目录: plots_mini/samp/

mkdir -p plots_mini/samp

for fgtype in ALG_NE2001 ALG_YMW16 ALG_NE2025 ALG_NE2025_vparNE2001
do
    echo "Plotting posterior for ${fgtype}..."
    python3 pltpost.py -c config_mini.json -f ./nest_out_mini/samp/${fgtype} -o ./plots_mini/samp/${fgtype}.pdf -title "Real Sample (mini) - ${fgtype}" -up -bo
done

# 逐事件红移后验 P(z): 四条线各出一版(参数取各自 MultiNest 后验中位数)
# MW 模型对 P(z) 最敏感(扣多扣少直接平移 P(z)), 四线对比即系统误差可视化
# 三元组: 输出名 : 宿主 gtype(与 run_samp_mini.sh 的 -g 一致) : MW 模型
for cfg in "ALG_NE2001:ALG_NE2001:ne2001" "ALG_YMW16:ALG_YMW16:ymw16" \
           "ALG_NE2025:ALG_YMW16:ne2025" "ALG_NE2025_vparNE2001:ALG_NE2001:ne2025"
do
    out=$(echo $cfg | cut -d: -f1)
    gt=$(echo $cfg | cut -d: -f2)
    mw=$(echo $cfg | cut -d: -f3)
    echo "Plotting P(z) for ${out}..."
    python3 pltpz.py --config config_mini.json -g ${gt} -mw ${mw} \
        -nest ./nest_out_mini/samp/${out} -o ./plots_mini/samp/${out}_pz.pdf
done

echo "[draw_samp_mini.sh 完成, 耗时 $(fmt_time $(( $(date +%s) - T_START )))]"
