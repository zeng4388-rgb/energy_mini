#!/bin/sh
# 真实数据嵌套采样(三条线并行: NE2001 / YMW16 / NE2025)
#
# 线名 = 银河系电子密度模型的大写(-mw 的值)。宿主 DM 已是第 7/8 维自由参数，三条线
# 共用同一个 8 维模型（宿主对数正态 + IGM 准高斯 + 晕均匀窗），唯一差别是 -mw 决定
# 读 filtered.fits 的哪一列 dm_exc ⇒ 线间差异是纯 MW 效应，ln Z 可直接横比。
# 旧的 ALG_* 线名来自已删除的"星系类型 × MW 模型"宿主标定表（vpar_alg_ymw16 那套），
# 2026-10-11 起统一改成 MW 模型名。
#
# 运行目录制（B1, 2026-10-31）：一次提交的全部产物进 nest_out_mini/samp/$RUN_TAG/，
# latest 符号链接指向本次 tag。日志用【追加】：同 tag 续跑(resume)不截断上一次输出；
# 换 config 重跑必须换 tag（采样器端 manifest 的 check_resume 也会拦混血续跑）。
# 逐线耗时取自各线 stdout 末行 "[samp] MultiNest 总耗时 …s"，汇总写入平铺的
# nest_out_mini/samp/run_samp_timing.txt（run_all_mini.sh 末尾引用这个文件名）。
# PYTHONUNBUFFERED（B2）：python stdout 进文件后默认块缓冲，被墙钟杀掉时
# [samp]/Single eval time/[llfail] 全丢；无缓冲后 tail -f 也能实时看进度。
# python 必须是【直接】后台任务而不是包在子壳函数里: `python3 … &` 的 $! 才是 python
# 自己的 PID, kill 得动。反过来，父脚本(run_all_mini.sh) kill 到的是【本脚本这个子壳】,
# 所以这里必须注册 trap 把 SIGTERM/SIGINT 转发给三个 python —— 否则子壳一死, 三个
# MultiNest 进程被 init 收养, 继续满载跑几小时（孤儿进程泄漏, 2026-10-11 修）。
# 不用 tee：POSIX sh 管道下 $! 是 tee 的 PID，kill-$! 逻辑就废了；看进度用 tail -f。

set -e
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

RUN_TAG=${CC_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
RUN_DIR="nest_out_mini/samp/$RUN_TAG"
mkdir -p "$RUN_DIR"
ln -sfn "$RUN_TAG" nest_out_mini/samp/latest

T_START=$(date +%s)
fmt_time() {
    s=$1
    printf "%dm %ds" $((s / 60)) $((s % 60))
}

LOGDIR="$RUN_DIR"
TIMING_FILE="nest_out_mini/samp/run_samp_timing.txt"

PIDS=""
NAMES=""

cleanup_children() {
    for p in $PIDS; do kill "$p" 2>/dev/null || true; done
    # 收尸剩余子进程,避免僵尸（|| true: set -e 下 wait 返回非零会静默中止,
    # 连 ERROR 都打不出来 —— 这是本脚本原先的真实 bug）
    for p in $PIDS; do wait "$p" 2>/dev/null || true; done
}
trap 'cleanup_children; echo "续跑: CC_RUN_TAG=$RUN_TAG sh run_samp_mini.sh" >&2; exit 130' INT TERM

for mw in ne2001 ymw16 ne2025
do
    out=$(echo "$mw" | tr 'a-z' 'A-Z')
    log="$LOGDIR/${out}.run.log"
    echo "Running ${out}: nest_samp_mini.py -mw ${mw}  (日志 ${log}, tag ${RUN_TAG})"
    # -o 传全路径：采样器对含 / 的 -o 直接用，链与 manifest 都落进 RUN_DIR
    python3 nest_samp_mini.py --fits --config config_mini.json -mw "${mw}" -o "$RUN_DIR/$out" \
        >> "$log" 2>&1 &
    PIDS="$PIDS $!"
    NAMES="$NAMES $out"
done

# 任一失败立即 kill 其余并退出。$! 是 python 本体, 所以 kill 有效。
fail=0
failed=""
_i=0
for pid in $PIDS; do
    _i=$((_i + 1))
    if ! wait $pid; then
        fail=1
        # 记线名而不是裸 PID（P3）：ERROR 行直接告诉你是哪条线挂了
        failed="$failed $(echo "$NAMES" | awk -v i=$_i '{print $i}')($pid)"
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
        msg="  $out : $(fmt_time "$sec") (见 ${RUN_TAG}/${out}.run.log)"
    else
        msg="  $out : 未完成/未报告(见 ${RUN_TAG}/${out}.run.log)"
    fi
    echo "$msg"
    echo "$msg" >> "$TIMING_FILE"
done

_e=$(( $(date +%s) - T_START ))
if [ $fail -eq 0 ]; then
    echo "[run_samp_mini.sh 完成, 3 条线并行(墙钟), 耗时 $(fmt_time $_e), tag ${RUN_TAG}]"
else
    echo "ERROR: run_samp_mini.sh 有任务失败 (pid:$failed),已中止其余任务; 墙钟 $(fmt_time $_e)" >&2
    echo "续跑: CC_RUN_TAG=$RUN_TAG sh run_samp_mini.sh" >&2
fi
exit $fail
