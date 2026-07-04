#!/bin/sh
# ============================================================
# cc_energy_mini 一键运行脚本
# 按顺序执行完整管线：生成数据 → 采样 → 绘图
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

# Step 1: 生成模拟数据
echo "===== Step 1/4: 生成模拟数据 (Nfrb=20, phis=1e3) ====="
T1_START=$(date +%s)
bash simudat_mini.sh
T1_END=$(date +%s)
T1=$((T1_END - T1_START))
echo ""

# Step 2: 模拟数据嵌套采样
echo "===== Step 2/4: 模拟数据嵌套采样 (n_live=200) ====="
T2_START=$(date +%s)
bash run_simu_mini.sh
T2_END=$(date +%s)
T2=$((T2_END - T2_START))
echo ""

# Step 3: 真实数据嵌套采样
echo "===== Step 3/4: 真实数据嵌套采样 (全部FRB, n_live=200) ====="
T3_START=$(date +%s)
bash run_samp_mini.sh
T3_END=$(date +%s)
T3=$((T3_END - T3_START))
echo ""

# Step 4: 绘制后验分布图
echo "===== Step 4/4: 绘制后验分布图 ====="
T4A_START=$(date +%s)
bash draw_sim_mini.sh
T4A_END=$(date +%s)
T4A=$((T4A_END - T4A_START))

T4B_START=$(date +%s)
bash draw_samp_mini.sh
T4B_END=$(date +%s)
T4B=$((T4B_END - T4B_START))
T4=$((T4A + T4B))
echo ""

T_TOTAL_END=$(date +%s)
T_TOTAL=$((T_TOTAL_END - T_TOTAL_START))

echo "=========================================="
echo "  全部完成！"
echo "  结果输出目录: nest_out_mini/"
echo "  图片输出目录: plots_mini/"
echo "=========================================="
echo ""
echo "========== 各步骤耗时统计 =========="
printf "  Step 1  simudat_mini.sh  : %ds (%s)\n" "$T1" "$(fmt_time $T1)"
printf "  Step 2  run_simu_mini.sh : %ds (%s)\n" "$T2" "$(fmt_time $T2)"
printf "  Step 3  run_samp_mini.sh : %ds (%s)\n" "$T3" "$(fmt_time $T3)"
printf "  Step 4a draw_sim_mini.sh : %ds (%s)\n" "$T4A" "$(fmt_time $T4A)"
printf "  Step 4b draw_samp_mini.sh: %ds (%s)\n" "$T4B" "$(fmt_time $T4B)"
printf "  Step 4  绘图合计         : %ds (%s)\n" "$T4" "$(fmt_time $T4)"
echo "  -----------------------------------"
printf "  总计                      : %ds (%s)\n" "$T_TOTAL" "$(fmt_time $T_TOTAL)"
echo "======================================"
