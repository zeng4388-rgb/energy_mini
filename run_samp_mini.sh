#!/bin/sh
# 真实数据嵌套采样(并行跑两个 galaxy type)

set -e
export OMP_NUM_THREADS=1

mkdir -p nest_out_mini/samp

# 两个 galaxy type 后台并行,各占 1 核;保存 PID 而非依赖 jobs -p
PIDS=""
for fgtype in ALG_NE2001 ALG_YMW16
do
    echo "Running nest_samp_mini.py with galaxy type: ${fgtype}"
    python3 nest_samp_mini.py --fits --config config_mini.json -g ${fgtype} -o ${fgtype} &
    PIDS="$PIDS $!"
done

# 任一失败立即 kill 另一个并退出
fail=0
for pid in $PIDS; do
    wait $pid || fail=1
    if [ $fail -ne 0 ]; then
        for p in $PIDS; do
            [ "$p" != "$pid" ] && kill $p 2>/dev/null
        done
        # 收尸剩余子进程,避免僵尸
        for p in $PIDS; do
            wait $p 2>/dev/null
        done
        echo "ERROR: run_samp_mini.sh 有任务失败 (pid=$pid),已中止其余任务" >&2
        exit 1
    fi
done
exit
