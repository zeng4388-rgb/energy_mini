#!/bin/sh
# 模拟数据嵌套采样(单任务): 注入-恢复检验, 是宿主参数可辨识性的闸口

set -e
export OMP_NUM_THREADS=1

T_START=$(date +%s)
SUMMARY=""
fmt_time() {
    s=$1
    printf "%dm %ds" $((s / 60)) $((s % 60))
}
# timed <标签> <命令...>: 单任务计时并累积摘要; 失败标 FAIL 后仍返回非零
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

fov=200

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
    timed "nest_simu phis=${phis}" python3 nest_simu_mini.py \
        --config config_mini.json -f1 "$fin1" -o "$fout1"
done

echo "[run_simu_mini.sh 完成, 耗时 $(fmt_time $(( $(date +%s) - T_START )))]"
exit
