#!/bin/sh
# 真实数据嵌套采样(三条线并行: NE2001 / YMW16 / NE2025)
#
# 线 1/2: NE2001 / YMW16 —— 目录自带 dm_exc 列
# 线 3:   NE2025 —— -mw ne2025 选 mwprop 预计算的 dm_exc_ne2025 列
#
# -g 现在只是输出标签 + -mw 缺省推断用: ln_mu_host / ln_sigma_host 已升级为
# 第 7/8 维自由参数, 宿主 DM 不再按星系类型分组固定, 因此三条线之间的后验
# 差异是纯粹的"MW 模型"效应(不含宿主参数未重标定的混杂), 可直接对比 ln Z。
#
# 逐线耗时取自各线 stdout 末行 "[samp] MultiNest 总耗时 …s"（重定向到
# nest_out_mini/samp/<线>.run.log）。python 必须是【直接】后台任务而不是包在
# 子壳函数里: `python3 … &` 的 $! 才是 python 自己的 PID, kill 得动;
# 子 shell 里跑 python 的话 kill 只杀子壳, MultiNest 会变孤儿继续占核。

set -e
export OMP_NUM_THREADS=1

T_START=$(date +%s)
fmt_time() {
    s=$1
    printf "%dm %ds" $((s / 60)) $((s % 60))
}

mkdir -p nest_out_mini/samp
LOGDIR=nest_out_mini/samp

PIDS=""
NAMES=""
for cfg in "ALG_NE2001:" "ALG_YMW16:" "ALG_YMW16:ne2025:ALG_NE2025"
do
    g=$(echo $cfg | cut -d: -f1)
    mw=$(echo $cfg | cut -d: -f2)
    out=$(echo $cfg | cut -d: -f3)
    [ -z "$out" ] && out=$g
    log="$LOGDIR/${out}.run.log"
    : > "$log"
    if [ -n "$mw" ]; then
        echo "Running ${out}: nest_samp_mini.py -g ${g} -mw ${mw}  (日志 ${log})"
        python3 nest_samp_mini.py --fits --config config_mini.json -g "${g}" -mw "${mw}" -o "${out}" \
            > "$log" 2>&1 &
    else
        echo "Running ${out}: nest_samp_mini.py -g ${g}  (日志 ${log})"
        python3 nest_samp_mini.py --fits --config config_mini.json -g "${g}" -o "${out}" \
            > "$log" 2>&1 &
    fi
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
        for p in $PIDS; do
            if [ "$p" != "$pid" ]; then kill $p 2>/dev/null || true; fi
        done
        # 收尸剩余子进程,避免僵尸（|| true: set -e 下 wait 返回非零会静默中止,
        # 连 ERROR 都打不出来 —— 这是本脚本原先的真实 bug）
        for p in $PIDS; do
            wait $p 2>/dev/null || true
        done
        break
    fi
done

echo "---------- 逐线耗时 ----------"
for out in $NAMES; do
    sec=$(sed -n 's/.*总耗时 \([0-9.]*\)s.*/\1/p' "$LOGDIR/${out}.run.log" | tail -1 | cut -d. -f1)
    if [ -n "$sec" ]; then
        echo "  $out : $(fmt_time $sec) (见 ${out}.run.log)"
    else
        echo "  $out : 未完成/未报告(见 ${out}.run.log)"
    fi
done

_e=$(( $(date +%s) - T_START ))
if [ $fail -eq 0 ]; then
    echo "[run_samp_mini.sh 完成, 3 条线并行(墙钟), 耗时 $(fmt_time $_e)]"
else
    echo "ERROR: run_samp_mini.sh 有任务失败 (pid:$failed),已中止其余任务; 墙钟 $(fmt_time $_e)" >&2
fi
exit $fail
