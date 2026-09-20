#!/bin/sh
# 真实数据嵌套采样(四条线并行, 加 run_simu_mini.sh 共 5 进程, 5 核内)
#
# 线 1/2: ALG_NE2001 / ALG_YMW16 —— 目录自带 dm_exc 列, 旧行为
# 线 3:   ALG_NE2025 —— -mw ne2025 选 mwprop 预计算的 dm_exc_ne2025 列;
#         宿主 vpar 借用 ALG_YMW16 组(mwprop 无 NE2025 标定版本)
# 线 4:   ALG_NE2025_vparNE2001 —— 敏感性检验: 同一 NE2025 列换用
#         ALG_NE2001 宿主 vpar, 用于量化"宿主 vpar 未随 MW 模型重标定"
#         的混杂效应(对比线 3/4 后验差异; 差异小则线 3 可近似视为
#         纯 MW 模型效应, 参照 Shin 2023 5.3.1 的不敏感结论表述)。
#         不需要时注释掉线 4 即可(并行任务数变 3)。
#
# 注意: 三线后验差异测的是"MW 模型 + 宿主 vpar 固定"的联合效应,
# 论文表述建议用"M 模型系统误差的量级估计(含宿主 vpar 未重标定混杂)"
# 的保守措辞, 线 4 用于分离该混杂。

set -e
export OMP_NUM_THREADS=1

# 脚本级计时
T_START=$(date +%s)
fmt_time() {
    s=$1
    printf "%dm %ds" $((s / 60)) $((s % 60))
}

mkdir -p nest_out_mini/samp

# 四条线后台并行,各占 1 核;保存 PID 而非依赖 jobs -p
PIDS=""
for cfg in "ALG_NE2001:" "ALG_YMW16:" "ALG_YMW16:ne2025:ALG_NE2025" "ALG_NE2001:ne2025:ALG_NE2025_vparNE2001"
do
    g=$(echo $cfg | cut -d: -f1)
    mw=$(echo $cfg | cut -d: -f2)
    out=$(echo $cfg | cut -d: -f3)
    [ -z "$out" ] && out=$g
    if [ -n "$mw" ]; then
        echo "Running nest_samp_mini.py: -g ${g} -mw ${mw} -o ${out}"
        python3 nest_samp_mini.py --fits --config config_mini.json -g ${g} -mw ${mw} -o ${out} &
    else
        echo "Running nest_samp_mini.py: -g ${g} -o ${out}"
        python3 nest_samp_mini.py --fits --config config_mini.json -g ${g} -o ${out} &
    fi
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

echo "[run_samp_mini.sh 完成, 4 条线并行, 耗时 $(fmt_time $(( $(date +%s) - T_START )))]"
exit
