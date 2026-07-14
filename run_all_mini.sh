#!/bin/sh
# ============================================================
# cc_energy_mini 一键运行脚本
# 按顺序执行完整管线：生成数据 → 采样(并行) → 绘图
# 嵌套采样阶段：3 个任务(模拟+2个galaxy type)后台并行跑
# 任一步骤失败即中止（set -e），并统计各子脚本耗时
# ============================================================

set -e
export OMP_NUM_THREADS=1

# 格式化秒数为 "Xm Ys"
fmt_time() {
    s=$1
    printf "%dm %ds" $((s / 60)) $((s % 60))
}

echo ""
echo "=========================================="
echo "  cc_energy_mini 完整管线开始运行"
echo "=========================================="
echo ""

T_TOTAL_START=$(date +%s)

# ====== 阶段 1：数据准备（串行，都很快） ======
echo "===== Step 0/5: 预处理 CHIME catalog ====="
T0_START=$(date +%s)
python3 preprocess_catalog.py
T0=$(( $(date +%s) - T0_START ))
echo ""

echo "===== Step 1/5: 生成模拟数据 (Nfrb=300, phis=1e3) ====="
T1_START=$(date +%s)
bash simudat_mini.sh
T1=$(( $(date +%s) - T1_START ))
echo ""

# ====== 阶段 2：嵌套采样（3 个任务并行，各占 1 核） ======
# run_simu_mini.sh(单任务) 与 run_samp_mini.sh(2 个 galaxy type 并行) 同时启动,
# 合计 3 个嵌套采样进程并行。失败检测由子脚本自己的 set -e / wait 检查负责。
echo "===== Step 2+3/5: 嵌套采样（并行跑 3 个任务） ====="
T23_START=$(date +%s)

bash run_simu_mini.sh &
PID_A=$!

bash run_samp_mini.sh &
PID_B=$!

# 任一失败立即 kill 另一个并退出,避免 A 失败仍阻塞等 B 跑完浪费时间
fail=0
wait $PID_A || fail=1
if [ $fail -ne 0 ]; then
    kill $PID_B 2>/dev/null
    wait $PID_B 2>/dev/null
    echo "ERROR: run_simu_mini.sh 失败,已中止 run_samp_mini.sh" >&2
    exit 1
fi
wait $PID_B || fail=1
if [ $fail -ne 0 ]; then
    echo "ERROR: run_samp_mini.sh 失败" >&2
    exit 1
fi

T23=$(( $(date +%s) - T23_START ))
echo ""

# ====== 阶段 3：绘图（串行，都很快） ======
echo "===== Step 4/5: 绘制后验分布图 ====="
T4A_START=$(date +%s)
bash draw_sim_mini.sh
T4A=$(( $(date +%s) - T4A_START ))

T4B_START=$(date +%s)
bash draw_samp_mini.sh
T4B=$(( $(date +%s) - T4B_START ))

T_TOTAL=$(( $(date +%s) - T_TOTAL_START ))

echo ""
echo "=========================================="
echo "  全部完成！"
echo "  结果输出目录: nest_out_mini/"
echo "  图片输出目录: plots_mini/"
echo "=========================================="
echo ""
echo "========== 各步骤耗时统计 =========="
printf "  Step 0  preprocess           : %s\n" "$(fmt_time $T0)"
printf "  Step 1  simudat              : %s\n" "$(fmt_time $T1)"
printf "  Step 2+3 嵌套采样 (3任务并行) : %s\n" "$(fmt_time $T23)"
printf "  Step 4a draw_sim             : %s\n" "$(fmt_time $T4A)"
printf "  Step 4b draw_samp            : %s\n" "$(fmt_time $T4B)"
echo "  -----------------------------------"
printf "  总计                          : %s\n" "$(fmt_time $T_TOTAL)"
echo "======================================"
