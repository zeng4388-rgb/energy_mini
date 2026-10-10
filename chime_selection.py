"""CHIME/FRB 探测概率的平滑选择函数（模拟端判据 + 似然端 S 加权，2026-10-07 接通）。

STATUS（2026-10-09）：本模块仍是【形状探针】，不是可用的校准概率。
k 无标定依据（k=1 = τ 边际化面值）。似然端（IntE_S/log_IntBeam_S/Norm1D_E_S/
rate_2d_E_S）已接线并由 V23 用独立 brute-force 参考钉住数值正确性，
但【水平】结论仍须待服务器 smooth 重拟合 + PPC 之后才有意义，且报告时必须
挂 log10 k 半宽 0.45 dex 的括号。

数据来源（全部为【读取】而非拟合，故不引入新自由度）：
McGregor et al. 2026 (arXiv:2606.26334) 公开的 4 维 logistic 选择函数
`chimefrb-selection/chimefrb_selection/data/fits/4d_selection_function/
 fluence_scattering_time_width_dm/IRLS_output_..._order3_snr12_sl5.0.npz`
的 35 个系数（beta_est）。求值脚本：本仓库外部 `csel_dl/read_delta.py`。
【已独立复核 2026-10-05】用 read_delta 的 Sel() 重解各锚点，与抽表逐点一致
（偏差 <0.01%）。

为什么用 order-3 而不是论文正文自称 fiducial 的 order-4：order-4 那份拟合外推
是坏的（F=10^4 Jy ms 处 S→0.0000），order-3 正确饱和到 1.0。包自带的
SelectionFunction 默认也是 degree=3。

【τ 约定已核对 2026-10-05】注入库字段名 tau_1_ghz_ms（chimefrb-selection 包
utils.py:139/147 的读取处）——模型 τ 特征 = 1 GHz、单位 ms。论文 Table 2 的
内禀 τ 中位 7.49 ms@600MHz 按 τ∝ν^-4 换算 = 0.9707 ms@1GHz。

下表是【τ 边际化】选择函数：对论文 Table 2 的内禀 τ 分布
LN(median 0.9707 ms@1GHz, σ_ln=1.79) 做 64 个 ln 空间分位节点边际化
（τ 截断 100 ms@1GHz；30/100/300 ms 截断的 F50 差 <0.5%，尾部不敏感）。
mock 无 τ 列，这是 round 1 唯一自洽的 τ 处理；代价是把论文的总体 τ 假设
烤进了选择函数——第二轮 mock 自带 τ 列后改为逐事件 S(F, τ, w)，本表退役。

  τ 边际化 vs 旧冻结切片（τ=0.01 ms，2026-10-05 前的表）：
  F50(w=1) 8.7 → 160.8 Jy ms（18.5×，1.27 dex）；过渡带 2.06 → 2.36–2.67 dex。
  旧表系统性乐观 ~20×（P1），已整表重算（V19b/c 钉同步更新）。

已知残差（单 logistic 对边际形状）——【双侧误差，符号在暗端翻转】。w=1 ms 实测
S̄ 与同 (F50,W595) 的 logistic 之比：
        F[Jy ms]   S̄         S_logi     logi/S̄
        1.00     0.00026    0.00737    28.1 ×  高估
        2.00     0.00502    0.01429     2.85 ×  高估
        3.2      0.020      0.020      ~1.0     符号翻转点
        8.00     0.1219     0.0524      0.43 ×  低估 2.3×
        20.0     0.2752     0.1180      0.43 ×  低估 2.3×
        40.0     0.3694     0.2071      0.56 ×  低估 1.8×
  ⇒ 中段（S̄∈[0.05,0.37]）logistic 【低估】 1.8–2.5× ⇒ mock 相对欠 populate；
    极暗端（S̄≲0.02）转为【高估】，且发散（F=1 Jy ms 处 28×）⇒ mock 相对
    【过】populate。后半个方向正是 round-1 里"暗端灌满、比值反而恶化"的通道，
    所以这条残差不是小量，也不能只用"低估 ~2×"一句概括。
  τ 列落地（逐事件 S(F,τ,w,DM)）后此残差整体消失。

DM 仍冻结在 500 pc cm^-3。【修正】旧文档写的"DM 200→2000 使 F50 移动 0.17 dex"
是【冻结 τ 切片】下的产物，边际化后大幅缩小且非单调，不能作为对称括号引用：
  τ 边际化后实测（F50 于 DM=200/500/2000）：
    w=1 ms  → 158.3 / 160.8 / 151.1 Jy ms，全宽 −0.02 dex（500 处是【极大值】）
    w=10 ms → 340.6 / 405.1 / 438.9 Jy ms，全宽 +0.11 dex（单调）
  即冻结在 500 的偏差 ≤0.06 dex（半宽），且 w=1 处方向是【偏悲观】——500 给
  最大 F50，两端都更低。H1 的"两端同冻结"因此只值 0.06 dex，不是 0.17。

【k_offset 水平括号——由实测项相加，不再是拍脑袋的 0.3】
  τ 先验项（主导）：扰动比取论文 Table 2 的【内部矛盾】——同一 Collaboration 对
    Cat1/Cat2 的内禀 τ 中位给出 2.15 vs 7.49 ms@600MHz，比 3.48（0.54 dex），
    论文自己把这归因于两者对 τ 上限量的不同处理。按此扰动 τ 中位重算 F50：
      w=0.1/1/10/100 ms 的半宽 = 0.42 / 0.55 / 0.33 / 0.17 dex
      主表区(0.1–10 ms)平均半宽 = 0.43 dex
  DM 切片项：≤0.06 dex（见上）
  ⇒ 合成取 log10 k 的半宽 = 0.45 dex（0.43 ⊕ 0.06），比原定的 0.3 宽 1.5×。
  ⚠ 仍是【约定括号】而非推导值：它由"扰动多大"的选择决定（这里用论文自身的
    Cat1/Cat2 不一致作尺度），且 τ 项在 w>10 ms 快速缩小、在 w<1 ms 反而变大，
    单一常数括号是粗化。第三轮 τ 列落地后此项整体消失。

【H1 归一化配对规则，2026-10-05】第二轮似然升级时，分子（逐事件 S 因子）与
分母（C_sel、Λ_S）必须同用本 DM 切片——DM 口径混用会使 −N log C 不再是分子
密度的归一化常数，误差随 θ 变、以 N=1688 杠杆扭曲后验。两端同迁 DM(z) 是
第三轮的事（选择函数随之 θ 依赖，属模型前向结构、合法），绝不单端迁移。
【耦合提醒】第三轮加 τ 列时，τ 与 DM 是同一曲面的两个自变量，必须【同时】进
先验/同时冻结；只把 τ 变成逐事件而 DM 仍冻结，等于再次单端迁移，违反同一规则。
"""
import os

import numpy as np

# ┌─ STATUS（2026-10-05）─────────────────────────────────────────────────
# │ 本模块输出的是【形状探针】，不是可用的校准概率：
# │ · τ 约定已定：注入字段 tau_1_ghz_ms（1 GHz、ms，utils.py:139 核对）
# │ · F50(w)/W595(w) 已按论文内禀 τ 分布边际化（旧冻结表乐观 ~20×），
# │   延展到 300 ms；表外续接【窄端=保守约定(0.74 dex 偏悲观，非物理饱和)，
# │   宽端=幂律】，旧钳位已废
# │ · 支撑分两层：F50 到 300 ms 都在注入流量覆盖内；只有 w=300 的 W595
#   （需 F95=11332>1e4）含外推。>300 ms 仅承载 S→0 方向，无定量含义
# │ · 仍存的一条系统误差：未逐点过 KNN 掩码（17.6 MB 未下）
# │ · k_offset 无标定依据（k=1 = 边际化面值），取值域 (0,100] 只拦单位错误；
# │   系统误差括号半宽 = K_LOG10_HALF_WIDE = 0.45 dex（τ 先验 0.43 ⊕ DM 0.06，
#   非旧的 0.3；τ 扰动尺度取论文 Table 2 自身 Cat1/Cat2 不一致 3.48×）
# │ · 单 logistic 的暗端残差是【双侧】的：S̄∈[0.05,0.37] 低估 1.8–2.5×，而
# │   S̄≲0.02 转为【高估】(F=1 Jy ms 处 28×)——后半个方向就是 round-1 暗端灌满
# │   的通道。见 docstring 的实测残差表；τ 列落地后整体消失
# └───────────────────────────────────────────────────────────────────────

# 内禀（静止系）宽度 [ms] -> τ 边际化 50% 探测概率处的流量 [Jy ms]。
# 主表 0.03–30 ms 覆盖目录 width_fitb 与 fiducial 宽度支撑（目录 26.65 ms 是
#   【观测帧】，对应静止系更小——本表自变量是内禀宽度）。
# 延展区 50–300 ms（H2，2026-10-05）：先验角 (μ_w,σ_w)=(2,1) 的 μ±6σ 支撑
#   远超 30 ms，静默钳位会把 10 s 宽的暴当成 30 ms 处理、严重过估探测。
#   支撑性核对（论文 §2.2：注入流量 0.1–1e4 Jy ms、注入本征宽度上界 5 s）：
#     · 宽度维：全表 0.03–300 ms 都在 5 s 注入覆盖【之内】✓
#     · 流量维要分两层看，别混：
#         – 中心层：所有节点的 F50 ≤ 4153 Jy ms < 1e4 ⇒ 【F50 本身】到 300 ms
#           都在注入流量覆盖内，延展区锚点不是流量外推（D1 的纠正，成立）。
#         – 带宽层：W595 需要 F95 也被覆盖。w=200 ms 的 F95=9294 ✓，而
#           w=300 ms 的 F95=11332 越界 ~13% ⇒ 只有【300 ms 处的带宽 1.00 dex】
#           含外推成分，其 F50 仍可信。V21(a) 管中心、V21(e) 管带宽，各测一层。
#   ⚠ 原声明的两条"系统误差"经复核【均不成立】，已撤回：
#     (a) "τ 截断在 w≥50 ms 敏感、τ≤100 vs ≤30 差 +37~58%" —— 实为 +0.1~0.4%，
#         全 w 范围都不敏感；且 τ 边际化的贡献随 w【递减】：F50(边际化)/F50(单切片
#         τ=0.9707) 从 w=1 ms 的 2.29 降到 w≥50 ms 的 0.93~0.97（内禀宽度主导
#         有效宽度后，τ 展宽相对不再重要）。原结论方向相反。
#     (b) "未逐点过 KNN 掩码"（17.6 MB 未下载）—— 这条【仍然成立】，保留。
_W_INT_MS = np.array([0.03, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00,
                      1.50, 2.00, 3.00, 5.00, 8.00, 12.00, 20.00, 30.00,
                      50.00, 100.00, 200.00, 300.00])
# τ 边际化 F50（2026-10-05 整表重算，推导见模块 docstring）
# ⚠ 延展区 50–300 ms 原值 (6429.4/8396.9/11451.2/13965.0) 偏高 3.36–5.12×，
#   已用与主表【同一个函数】重算替换（脚本 csel_dl/rebuild_table.py）。主表
#   0.03–30 ms 的 16 个节点全部复现到比值 1.00，说明延展区原先走了另一条代码
#   路径 —— 这就是 V21(a)「表必须能被自己的 recipe 再生」这条钉的来源。
_F50_JYMS = np.array([84.9, 109.5, 137.1, 153.0, 157.0, 158.5, 159.4, 160.8,
                      167.4, 177.1, 201.7, 259.2, 348.0, 460.0, 660.4, 879.6,
                      1255.9, 2013.9, 3189.8, 4152.7])
# τ 边际化 5–95% 过渡全宽 [dex]（随宽度变化；旧标量 2.06 是冻结切片特定值，P2）。
# 峰值 2.67 在 w≈2 ms，之后【平滑单调】收窄到 1.00（300 ms）。原表声明的
# "宽端塌缩 2.48→0.82→0.61" 不存在：0.61 恰等于 w=300 处单 τ 切片的带宽(0.642)，
# 是求 F95 时撞到 bracket 上限的假信号（order-4 上踩过同一个坑）。
_W595_DEX = np.array([2.36, 2.42, 2.49, 2.56, 2.59, 2.62, 2.64, 2.65,
                      2.66, 2.67, 2.67, 2.66, 2.64, 2.61, 2.55, 2.48,
                      2.32, 1.59, 1.10, 1.00])

W_CLIP_LO, W_CLIP_HI = float(_W_INT_MS[0]), float(_W_INT_MS[-1])
W595_MED = float(np.median(_W595_DEX))     # 报告用

# k_offset 取值域上限：非 1 取值目前均无标定依据（见 detection_probability
# docstring），上限只为拦单位错误，不是物理括号。
K_MAX = 100.0

# k_offset 的【系统误差括号】半宽 [dex]，由实测项合成（推导见模块 docstring）：
#   τ 先验项 0.43（主导；扰动尺度取论文 Table 2 的 Cat1/Cat2 内禀 τ 中位不一致
#   2.15 vs 7.49 ms = 3.48× = 0.54 dex）⊕ DM 切片项 0.06。
# 用途：给 φ₀ 添误差棒的低优先项；step-vs-S 两条后验的参数差才是选择系统性的
# 主度量。仍是约定括号（扰动尺度是选出来的），不是推导值。
K_LOG10_HALF_WIDE = 0.45

# 注入实验的流量覆盖上界 [Jy ms]（McGregor 2026 §2.2：注入 0.1–1e4 Jy ms）。
# 这是选择函数【有数据支撑】的流量上限：F50 越过它，该宽度的探测概率就是在
# 没有注入样本的区间上评的，属于外推。V21(a) 用它当表的硬约束。
F_INJ_MAX_JYMS = 1.0e4

# 【带宽层】支撑上界：最后一个 F95 仍在注入流量覆盖之内的表节点 [ms]。
# 按 2026-10-05 重算的表：w=200 ms 的 F95=9294 ✓；w=300 ms 的 F95=11332 越界
# ~13%。注意这【不】否定 300 ms 的 F50（4153 < 1e4，中心层仍受支撑）——它只
# 说明 300 ms 处的 W595=1.00 dex 含外推成分。两层分开管：V21(a)=中心层，
# V21(e)=带宽层。
# ⚠ w > 300 ms 的幂律续接：宽度维仍在 5 s 注入覆盖内（D1 纠正），但 F50 迅速
#   冲出 1e4 流量覆盖（3000 ms 处已 ~1.4e5），所以续接段只有【方向】意义
#   （S→0），无定量意义 —— 三档声明的第三档。
W_SUPPORT_HI_MS = 200.0


def f50_of_width(w_int_ms):
    """50% 探测概率处的流量 [Jy ms]。

    表内对数-对数线性内插；表外【续接】而非钳位（H2，2026-10-05）：

    窄端（w < 0.03 ms，注入宽度下界之下）取平 —— 这是【保守约定，不是物理推论】。
      曾以为的依据"w≪τ 时饱和"实测后【不成立】：饱和确实发生，但饱和尺度跟着 τ
      走（单切片 τ=0.971 ms 要到 w≈0.1–0.3 才平；τ=1e-3 ms 在 w≈0.03 就平）。
      σ_ln=1.79 的 τ 先验一路下到 1e-4 ms，不存在能让【整个混合】去饱和的单一 τ
      尺度，故拟合自身的 F50 在 w<0.03 仍继续下降：
        w=0.003/0.01/0.03 ms → 15.4 / 39.1 / 84.9 Jy ms
      取平 = 把该区间判得比拟合自身【难探 5.5×（0.74 dex）】，方向明确为悲观。
      影响面：仅 μ_w≲−1 的先验角触发；V18 类参数点 mock 内禀宽度下界 0.036 ms，
      不触发。需要乐观侧时改用上面的拟合自身延续。

    宽端（w > 300 ms）按末段局部斜率（200→300 ms 实测 +0.65 dex/dex，延表修正
    前是 0.49）幂律延续 —— 超宽暴 F50 发散、S→0 方向正确。
    ⚠ 三档支撑声明（D1 定稿，"300 ms 是用外推锚点做外推"的旧说法不准确）：
      ① 主表 0.03–30 ms：F50 与 W595 都在注入覆盖内 —— 有支撑。
      ② 延展区 30–300 ms：宽度维在 5 s 内、中心层 F50≤4153<1e4 Jy ms 也在流量
         覆盖内 ⇒【F50 受支撑】；只有 w=300 的 W595 需 F95=11332 略越 1e4
         ⇒【带宽含外推】。另有一条已声明系统误差：未逐点过 KNN 掩码。
      ③ 续接区 >300 ms：宽度维仍在 5 s 内，但 F50 迅速冲出流量覆盖（3000 ms 处
         已 ~1.4e5 Jy ms）⇒ 只承载 S→0 的【方向】，无定量含义。
    """
    w = np.asarray(w_int_ms, dtype=float)
    if np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("f50_of_width 需要正的有限宽度，得到 %r" % (w,))
    lw = np.log10(w)
    lw_c = np.clip(lw, np.log10(W_CLIP_LO), np.log10(W_CLIP_HI))
    f_tab = np.power(10.0, np.interp(lw_c, np.log10(_W_INT_MS), np.log10(_F50_JYMS)))
    slope_wide = ((np.log10(_F50_JYMS[-1]) - np.log10(_F50_JYMS[-2]))
                  / (np.log10(_W_INT_MS[-1]) - np.log10(_W_INT_MS[-2])))
    # 宽端幂律续接；窄端 np.interp 的 clip 已给出平续
    return np.where(lw > np.log10(W_CLIP_HI),
                    f_tab * np.power(w / W_CLIP_HI, slope_wide), f_tab)


def w595_of_width(w_int_ms):
    """τ 边际化选择函数的 5–95% 过渡全宽 [dex]，随 log(宽度) 线性内插。

    旧实现是标量 W595_DEX=2.06——那是冻结切片特定值；边际化后主表
    2.36–2.67（P2 修正），延展区塌缩到 0.61–0.82。表外两端平续（np.interp
    clip 行为）：窄端回到主表值、宽端冻结在 0.61。
    """
    w = np.asarray(w_int_ms, dtype=float)
    if np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("w595_of_width 需要正的有限宽度，得到 %r" % (w,))
    lw = np.clip(np.log10(w), np.log10(W_CLIP_LO), np.log10(W_CLIP_HI))
    return np.interp(lw, np.log10(_W_INT_MS), _W595_DEX)


def detection_probability(flu_jyms, w_int_ms, k_offset=1.0):
    """P(detected | fluence, intrinsic width)——τ 边际化【形状探针】。

    k_offset：F_50(w) 的【乘性】水平因子（F50_eff = k·F50），k=1 = 边际化面值。
    恒等式 S_k(F) = S_1(F/k)（V19d 钉住）。⚠ k 不是已标定的自由参数，只是
    系统误差探针：它只平移水平、不改形状，且目前不存在有依据的非 1 取值——
    fluence 水平定标必须等似然端换 S 并重拟合。在未拟合参数点上用目录定标 k
    属于循环论证（2026-10-05 实测：V18 测试点面值 medF≈1.0 vs 目录≈3.0，是
    种群点失配，不是 S 面值被证伪）。
    k 取值域 (0, 100]：上限只拦单位错误。
    """
    f = np.asarray(flu_jyms, dtype=float)
    if np.any(~np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("detection_probability 需要正的有限 fluence")
    if not (0.0 < k_offset <= K_MAX):
        raise ValueError("k_offset 需在 (0, %g]（只拦单位错误，非 1 取值均无标定依据，"
                         "见 docstring），得到 %r" % (K_MAX, k_offset))
    u = np.log10(f) - (np.log10(f50_of_width(w_int_ms)) + np.log10(k_offset))
    a = 2.0 * np.log(19.0) / w595_of_width(w_int_ms)
    return 1.0 / (1.0 + np.exp(-np.clip(a * u, -500.0, 500.0)))


# 与 CC_SURVIVAL 同一套 env 覆盖约定
VALID_MODES = ('step', 'smooth')


def resolve_selection(cfg=None):
    """返回 (mode, k_offset)。config 缺省 = 'step'，即现有全部基线/回归钉不变。

    env CC_SEL_MODEL=smooth|step 和 CC_SEL_K=<0,100] 覆盖 config，优先级最高。
    """
    mode, k = 'step', 1.0
    if cfg is not None:
        sel = cfg.get('selection', {}) or {}
        unknown = set(sel) - {'model', 'k_offset'}
        if unknown:
            raise ValueError("config['selection'] 出现未知键 %s（可用: model, k_offset）"
                             % sorted(unknown))
        mode = sel.get('model', mode)
        k = float(sel.get('k_offset', k))
    env_mode = os.environ.get('CC_SEL_MODEL')
    if env_mode:
        mode = env_mode
    env_k = os.environ.get('CC_SEL_K')
    if env_k:
        try:
            k = float(env_k)
        except ValueError:
            raise ValueError("CC_SEL_K 需为浮点数，得到 %r" % env_k)
    if mode not in VALID_MODES:
        raise ValueError("selection model 需为 %s 之一，得到 %r" % (list(VALID_MODES), mode))
    if not (0.0 < k <= K_MAX):
        raise ValueError("selection k_offset 需在 (0, %g]（非 1 取值均无标定依据，"
                         "见 chime_selection.detection_probability docstring），得到 %r"
                         % (K_MAX, k))
    return mode, k
