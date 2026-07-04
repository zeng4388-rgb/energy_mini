#!/bin/sh

# 绘制真实样本后验分布图
# 输出目录: plots_mini/samp/

mkdir -p plots_mini/samp

for fgtype in ALG_NE2001 ALG_YMW16
do
    echo "Plotting posterior for ${fgtype}..."
    python3 pltpost.py --config config_mini.json -f ./nest_out_mini/samp/${fgtype} -o ./plots_mini/samp/${fgtype}.eps -title "Real Sample (mini) - ${fgtype}" -up -bo
done
