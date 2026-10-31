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
# 一次 campaign 一个 tag：samp/simu 两个 run 目录共用，manifest 原样记录（B1-B3）
export CC_RUN_TAG=${CC_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
# 子脚本的 python stdout 进文件；不加则块缓冲，被墙钟杀掉时日志全丢（B2）
export PYTHONUNBUFFERED=1

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
# preprocess 直接覆盖 data/filtered.*；先按 tag 备份，误跑可回滚（B5）
for f in data/filtered.fits data/filtered.txt; do
    if [ -f "$f" ]; then cp -p "$f" "${f}.bak_$CC_RUN_TAG"; fi
done
python3 preprocess_catalog.py
T0=$(( $(date +%s) - T0_START ))
echo ""

echo "===== Step 1b/5: 数据分布图（dm_exc 三模型对比 + 样本概览） ====="
T0B_START=$(date +%s)
python3 pltdatadist.py --config config_mini.json
T0B=$(( $(date +%s) - T0B_START ))
echo ""

echo "===== Step 2/5: 生成模拟数据 (Nfrb=600, phis=1e3) ====="
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

# kill 必须打到【子进程】而不只是子壳：`bash xxx.sh &` 的 $! 是子壳 PID，子壳收到
# SIGTERM 后它启动的 python 会被 init 收养、继续满载跑几小时（孤儿进程泄漏）。
# run_samp_mini.sh 内部已自带 trap 转发，这里再补一层 pkill -P 双保险，覆盖
# run_simu_mini.sh 那条前台 python（它没有 trap）。pkill 不可用时退回只 kill 子壳。
kill_branch() {
    pkill -P "$1" 2>/dev/null || true
    kill "$1" 2>/dev/null || true
    wait "$1" 2>/dev/null || true
}

# 开局体检：B（真实样本三线）若在头 60 s 内就崩（缺 dm_exc_ne2025 列、filtered.fits
# 缺 NSAMPLE/NREF10 header 键之类的接线错误），不要傻等 A 跑完几小时才发现。
B_DONE=0
# 轮询代替开局固定等 60 s：每 5 s 探一次 PID_B，上限仍 60 s；开局正常时省 ~55 s（B5）
_i=0
while [ $_i -lt 12 ] && kill -0 $PID_B 2>/dev/null; do
    sleep 5
    _i=$((_i + 1))
done
if ! kill -0 $PID_B 2>/dev/null; then
    if wait $PID_B; then
        B_DONE=1        # 极少见：已经成功退出，记下免得下面重复 wait 拿到 127
    else
        kill_branch $PID_A
        echo "ERROR: run_samp_mini.sh 在开局 60 s 内失败,已中止 run_simu_mini.sh" >&2
        exit 1
    fi
fi

# 任一失败立即 kill 另一个并退出,避免 A 失败仍阻塞等 B 跑完浪费时间
# kill/wait 都要 || true: set -e 下它们返回非零会让脚本在打印错误信息之前就静默退出
fail=0
wait $PID_A || fail=1
if [ $fail -ne 0 ]; then
    kill_branch $PID_B
    echo "ERROR: run_simu_mini.sh 失败,已中止 run_samp_mini.sh" >&2
    exit 1
fi
if [ "$B_DONE" -eq 0 ]; then
    wait $PID_B || fail=1
    if [ $fail -ne 0 ]; then
        echo "ERROR: run_samp_mini.sh 失败" >&2
        exit 1
    fi
fi

T23=$(( $(date +%s) - T23_START ))
echo ""

# ====== 阶段 2b：链自动验收（碎后验/贴边链不许流进出图和 PPC, B4） ======
echo "===== Step 3b/5: 链自动验收 qa_chains (4 条链) ====="
qa_gate() {
    if ! python3 qa_chains.py --config config_mini.json "$@"; then
        echo "ERROR: qa FAIL ($*)。链位置见上；放行: CC_QA_STRICT=0 CC_RUN_TAG=$CC_RUN_TAG sh run_all_mini.sh（mock 按 tag 种子逐位复现、链走 resume，同 tag 重跑安全）" >&2
        exit 1
    fi
}
for mw in ne2001 ymw16 ne2025
do
    out=$(echo "$mw" | tr 'a-z' 'A-Z')
    qa_gate --nest "./nest_out_mini/samp/$CC_RUN_TAG/$out"
done
qa_gate --nest "./nest_out_mini/simu/$CC_RUN_TAG/simdat_1e3" \
        --truth ./nest_out_mini/simu/simdat_1e3.truth.json
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
for mw in ne2001 ymw16 ne2025
do
    out=$(echo "$mw" | tr 'a-z' 'A-Z')
    _t0=$(date +%s)
    if python3 ppcheck.py --config config_mini.json \
            --nest ./nest_out_mini/samp/$CC_RUN_TAG/${out} -mw ${mw} \
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
echo "    CC_SURVIVAL=0.4639 sh run_samp_mini.sh   # one-off 口径存活率线 (1563/3369, 缺省全部 burst 口径 1563/4334)"
echo "    config: analysis.sfr_curve=Yuksel2008    # SFR 曲线"
echo "    config: dm_model.igm.F=0.09 / 0.5        # IGM 涨落宽度括号线(Macquart 边际化区间)"
echo "    config: dm_model.halo.DMsmin/DMsmax=20/80 # 晕窗括号线(均值 50, 与主线 U[0,60] 对照)"
echo "    config: 删掉 prior_shape 段              # 平坦先验敏感度(ln Z 不可跨先验比)"
echo "  模型比较: 读 nest_out_mini/samp/*.log 的 ln Z ± err(仅同一先验内可比)"
exit
