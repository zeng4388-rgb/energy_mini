#!/bin/sh
# 模拟数据嵌套采样(单任务): 注入-恢复检验, 是宿主参数可辨识性的闸口
#
# 运行目录制（B3, 2026-10-31）：链与 manifest 进 nest_out_mini/simu/$RUN_TAG/，
# latest 符号链接指向本次 tag；mock 与 truth.json 是【输入】，仍留平铺路径
# （链与 truth 的绑定由 manifest 的 input_md5 承担）。
# 日志只重定向不用 tee：dash 无 PIPESTATUS，tee 会吃掉 python 的退出码。

set -e
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

RUN_TAG=${CC_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
RUN_DIR="nest_out_mini/simu/$RUN_TAG"
mkdir -p "$RUN_DIR"
ln -sfn "$RUN_TAG" nest_out_mini/simu/latest

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

mkdir -p simu_mini

for phis in 1e3
do
    fout1=simdat_${phis}
    fin1=./simu_mini/simdat_${phis}_${fov}.txt
    log="$RUN_DIR/${fout1}.run.log"

    # 检查输入文件是否存在
    if [ ! -f "$fin1" ]; then
        echo "ERROR: 缺少模拟输入文件，请先运行 simudat_mini.sh 生成数据"
        echo "  缺失: $fin1"
        exit 1
    fi

    echo "Running nest_simu_mini.py for phi*=${phis}  (日志 ${log}, tag ${RUN_TAG})"
    if ! timed "nest_simu phis=${phis}" python3 nest_simu_mini.py \
        --config config_mini.json -f1 "$fin1" -o "$RUN_DIR/$fout1" >> "$log" 2>&1
    then
        echo "ERROR: nest_simu 失败, 日志尾部:" >&2
        tail -20 "$log" >&2
        exit 1
    fi
done

echo "[run_simu_mini.sh 完成, 耗时 $(fmt_time $(( $(date +%s) - T_START ))), tag ${RUN_TAG}]"
exit
