#!/bin/sh
# ============================================================
# cc_energy_mini 一键运行脚本
# 按顺序执行完整管线：数据准备 → 造 mock → 采样(并行) → 绘图 → PPC
#   Step 2+3: 4 个嵌套采样进程并行(模拟 1 条 + 真实样本 3 条 MW 线)，各占 1 核
#   Step 5  : PPC 逐线串行，每线内部 -j 个进程
# 任一步骤失败即中止（set -e），每个子脚本内部还会逐任务计时
# ============================================================

set -e
export OMP_NUM_THREADS=1

# PPC 抽样组数与并行度（可用环境变量覆盖: PPC_DRAWS=300 PPC_J=8 sh run_all_mini.sh）
# 成本实测: 单个 3373 事件 mock ≈ 5.9 s(单核) → draws/×5.9/j 秒每线
PPC_DRAWS=${PPC_DRAWS:-100}
PPC_J=${PPC_J:-4}

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
echo "===== Step 1/5: 预处理 CHIME catalog ====="
T0_START=$(date +%s)
python3 preprocess_catalog.py
T0=$(( $(date +%s) - T0_START ))
echo ""

echo "===== Step 1b/5: 数据分布图（dm_exc 三模型对比 + 样本概览） ====="
T0B_START=$(date +%s)
python3 pltdatadist.py --config config_mini.json
T0B=$(( $(date +%s) - T0B_START ))
echo ""

echo "===== Step 2/5: 生成模拟数据 (Nfrb=300, phis=1e3) ====="
T1_START=$(date +%s)
bash simudat_mini.sh
T1=$(( $(date +%s) - T1_START ))
echo ""

# ====== 阶段 2：嵌套采样（4 个进程并行，各占 1 核） ======
# run_simu_mini.sh(单任务) 与 run_samp_mini.sh(3 条线并行) 同时启动,
# 合计 4 个嵌套采样进程并行(4 核内)。失败检测由子脚本自己的 set -e / wait 检查负责。
echo "===== Step 3/5: 嵌套采样（并行跑 4 个任务: 模拟 + 3 条真实样本线） ====="
T23_START=$(date +%s)

bash run_simu_mini.sh &
PID_A=$!

bash run_samp_mini.sh &
PID_B=$!

# 任一失败立即 kill 另一个并退出,避免 A 失败仍阻塞等 B 跑完浪费时间
# kill/wait 都要 || true: set -e 下它们返回非零会让脚本在打印错误信息之前就静默退出
fail=0
wait $PID_A || fail=1
if [ $fail -ne 0 ]; then
    kill $PID_B 2>/dev/null || true
    wait $PID_B 2>/dev/null || true
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
echo "===== Step 4/5: 绘制后验分布图 + 逐事件 P(z) ====="
T4A_START=$(date +%s)
bash draw_sim_mini.sh
T4A=$(( $(date +%s) - T4A_START ))

T4B_START=$(date +%s)
bash draw_samp_mini.sh
T4B=$(( $(date +%s) - T4B_START ))
echo ""

# ====== 阶段 4：后验预测检验（逐线串行, 每线内部 -j 并行） ======
echo "===== Step 5/5: PPC (3 条线, 各 ${PPC_DRAWS} 组抽样 / ${PPC_J} 进程) ====="
mkdir -p plots_mini/samp
T5_START=$(date +%s)
T5SUM=""
for line in "ALG_NE2001:ne2001" "ALG_YMW16:ymw16" "ALG_NE2025:ne2025"
do
    out=$(echo $line | cut -d: -f1)
    mw=$(echo $line | cut -d: -f2)
    _t0=$(date +%s)
    if python3 ppcheck.py --config config_mini.json \
            --nest ./nest_out_mini/samp/${out} -mw ${mw} \
            --draws ${PPC_DRAWS} -j ${PPC_J} \
            -o ./plots_mini/samp/${out}_ppc.pdf
    then tag=ok; else tag=FAIL; fi
    _e=$(( $(date +%s) - _t0 ))
    T5SUM="${T5SUM}  ppc ${out} : $(fmt_time $_e) [${tag}]
"
    [ "$tag" = ok ]
done
printf "%s" "$T5SUM"
T5=$(( $(date +%s) - T5_START ))
echo ""

T_TOTAL=$(( $(date +%s) - T_TOTAL_START ))

echo "=========================================="
echo "  全部完成！"
echo "  结果输出目录: nest_out_mini/"
echo "  图片输出目录: plots_mini/"
echo "  逐线采样耗时: nest_out_mini/samp/run_samp_timing.txt"
echo "=========================================="
echo ""
echo "========== 各步骤耗时统计 =========="
printf "  Step 1   preprocess           : %s\n" "$(fmt_time $T0)"
printf "  Step 1b  datadist plots       : %s\n" "$(fmt_time $T0B)"
printf "  Step 2   simudat              : %s\n" "$(fmt_time $T1)"
printf "  Step 3   嵌套采样 (4任务并行)  : %s\n" "$(fmt_time $T23)"
printf "  Step 4a  draw_sim             : %s\n" "$(fmt_time $T4A)"
printf "  Step 4b  draw_samp            : %s\n" "$(fmt_time $T4B)"
printf "  Step 5   PPC (3线×%s抽样)    : %s\n" "${PPC_DRAWS}" "$(fmt_time $T5)"
echo "  -----------------------------------"
printf "  总计                          : %s\n" "$(fmt_time $T_TOTAL)"
echo "======================================"
echo "手工步骤(不在本脚本内, 按需单独跑):"
echo "  系统误差线: 同一条真实数据线加环境变量/改 config 重跑"
echo "    CC_SURVIVAL=0.4723 sh run_samp_mini.sh   # one-off 口径存活率线 (1688/3574, 缺省全部 burst 口径 1688/4840)"
echo "    config: analysis.sfr_curve=Yuksel2008    # SFR 曲线"
echo "    config: dm_model.igm.F=0.09 / 0.5        # IGM 涨落宽度括号线(Macquart 边际化区间)"
echo "    config: 删掉 prior_shape 段              # 平坦先验敏感度(ln Z 不可跨先验比)"
echo "  模型比较: 读 nest_out_mini/samp/*.log 的 ln Z ± err(仅同一先验内可比)"
exit
