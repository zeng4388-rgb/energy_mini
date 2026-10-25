#!/bin/sh
# 真实数据嵌套采样(三条线并行: NE2001 / YMW16 / NE2025)
#
# 线名 = 银河系电子密度模型的大写(-mw 的值)。宿主 DM 已是第 7/8 维自由参数，三条线
# 共用同一个 8 维模型（宿主对数正态 + IGM 准高斯 + 晕均匀窗），唯一差别是 -mw 决定
# 读 filtered.fits 的哪一列 dm_exc ⇒ 线间差异是纯 MW 效应，ln Z 可直接横比。
# 旧的 ALG_* 线名来自已删除的"星系类型 × MW 模型"宿主标定表（vpar_alg_ymw16 那套），
# 2026-10-11 起统一改成 MW 模型名。
#
# 逐线耗时取自各线 stdout 末行 "[samp] MultiNest 总耗时 …s"（重定向到
# nest_out_mini/samp/<线>.run.log），并汇总写入 nest_out_mini/samp/run_samp_timing.txt
# （run_all_mini.sh 末尾会引用这个文件名）。
# python 必须是【直接】后台任务而不是包在子壳函数里: `python3 … &` 的 $! 才是 python
# 自己的 PID, kill 得动。反过来，父脚本(run_all_mini.sh) kill 到的是【本脚本这个子壳】,
# 所以这里必须注册 trap 把 SIGTERM/SIGINT 转发给三个 python —— 否则子壳一死, 三个
# MultiNest 进程被 init 收养, 继续满载跑几小时（孤儿进程泄漏, 2026-10-11 修）。

set -e
export OMP_NUM_THREADS=1

T_START=$(date +%s)
fmt_time() {
    s=$1
    printf "%dm %ds" $((s / 60)) $((s % 60))
}

mkdir -p nest_out_mini/samp
LOGDIR=nest_out_mini/samp
TIMING_FILE="$LOGDIR/run_samp_timing.txt"

PIDS=""
NAMES=""

cleanup_children() {
    for p in $PIDS; do kill "$p" 2>/dev/null || true; done
    # 收尸剩余子进程,避免僵尸（|| true: set -e 下 wait 返回非零会静默中止,
    # 连 ERROR 都打不出来 —— 这是本脚本原先的真实 bug）
    for p in $PIDS; do wait "$p" 2>/dev/null || true; done
}
trap 'cleanup_children; exit 130' INT TERM

for mw in ne2001 ymw16 ne2025
do
    out=$(echo "$mw" | tr 'a-z' 'A-Z')
    log="$LOGDIR/${out}.run.log"
    : > "$log"
    echo "Running ${out}: nest_samp_mini.py -mw ${mw}  (日志 ${log})"
    python3 nest_samp_mini.py --fits --config config_mini.json -mw "${mw}" -o "${out}" \
        > "$log" 2>&1 &
    PIDS="$PIDS $!"
    NAMES="$NAMES $out"
done

# 任一失败立即 kill 其余并退出。$! 是 python 本体, 所以 kill 有效。
fail=0
failed=""
for pid in $PIDS; do
    if ! wait $pid; then
        fail=1
        failed="$failed $pid"
        cleanup_children
        break
    fi
done
trap - INT TERM

echo "---------- 逐线耗时 ----------"
: > "$TIMING_FILE"
for out in $NAMES; do
    sec=$(sed -n 's/.*总耗时 \([0-9.]*\)s.*/\1/p' "$LOGDIR/${out}.run.log" | tail -1 | cut -d. -f1)
    if [ -n "$sec" ]; then
        msg="  $out : $(fmt_time "$sec") (见 ${out}.run.log)"
    else
        msg="  $out : 未完成/未报告(见 ${out}.run.log)"
    fi
    echo "$msg"
    echo "$msg" >> "$TIMING_FILE"
done

_e=$(( $(date +%s) - T_START ))
if [ $fail -eq 0 ]; then
    echo "[run_samp_mini.sh 完成, 3 条线并行(墙钟), 耗时 $(fmt_time $_e)]"
else
    echo "ERROR: run_samp_mini.sh 有任务失败 (pid:$failed),已中止其余任务; 墙钟 $(fmt_time $_e)" >&2
fi
exit $fail
