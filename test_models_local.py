#!/usr/bin/env python3
"""本地模型层验证套件 V1-V27（不需要 pymultinest / mwprop / P0 存档）。

对应《代码改进方案.md》C11。全部为 assert 形式，任何一条失败即退出非零。
用法:  python3 test_models_local.py

边界: 本套件只验证模型层数学与接线正确性。采样行为、注入-恢复、PPC 必须在服务器上跑。
V1-V4、V6-V16 是纯模型/接线判据，只依赖本目录的代码；V5 需要 data/filtered.fits
（预处理产物，不随包分发），缺文件时打印 SKIP。V19-V23 是平滑选择函数
（McGregor 2026）的钉：V19/V21 管表与判据形状，V20 管模拟端吞吐，
V22 管 mock↔似然口径配对，V23 管似然端原语的数值正确性。
V24-V27 是准高斯 IGM 改造那一轮的钉：V24 管 dnu 能量轴口径，V25 管 gamma 上界
的数值约束，V26 管抽样主表插值，V27 管"似然不得退化成 -1e30 常数平台"。
"""
import re
import sys
from pathlib import Path

import numpy as np
import scipy.special as spf
from astropy.io import fits

from frb_util import (AstroDistribution, Cosmology, Loadfiles, PARAM_ORDER,
                      apply_dm_model, build_models, load_config,
                      read_selection_survival)

HERE = Path(__file__).resolve().parent
# numpy 1.x/2.x 兼容: np.trapz 自 2.0 起改名 np.trapezoid。本套件也要能在
# 服务器旧环境(protected 见 pltpz.py 的同一垫片)里跑, 不能只挑一个名字。
_trapz = getattr(np, 'trapezoid', None) or np.trapz


def _catalog():
    d = fits.getdata(str(HERE / "data" / "filtered.fits"))
    F = np.array(d["fluence"], float)
    W = np.array(d["width"], float)
    D = np.array(d["dm_exc_ne2001"], float)
    return np.log10(F / W), D, np.log10(W)


def v1_mean_anchor():
    """DM_IGM 均值 = Macquart Eq.2 / James Eq.6 / Shin A12: z=1 给 902.24。

    2026-10-09 宇宙学切到 Planck 2018 results VI (67.4/0.315/0.685/0.0493) 后，
    旧锚点 895.2（Planck 2015 口径）变为 902.24。第一条断言是物理判据
    （Z21 的 892+721/-270），第二条是防漂移 tripwire（容差保持 0.5，不许放宽）。
    """
    v = Cosmology().DispersionMeasure_IGM(1.0)
    assert 622.0 <= v <= 1613.0, v
    assert abs(v - 902.24) < 0.5, v
    print("V1 PASS  DM_IGM(1)=%.2f (Planck 2018 VI 口径; Z21 报 892+721/-270)" % v)


def v2_window_analytic():
    """晕窗 U[DMsmin, DMsmax] 的解析边际化 vs 暴力数值卷积, 相对误差 <0.1%。

    跑两档: 主线 [0,60] 与系统误差线 [20,80]，两者都必须对得上。
    """
    ad = AstroDistribution()
    mu, s = np.log(33.0), 1.25
    r = 90.0
    for lo, hi in ((0.0, 60.0), (20.0, 80.0)):
        ad.DMsmin, ad.DMsmax = lo, hi
        g = np.linspace(lo, hi, 400001)
        pdf = (np.exp(-(np.log(np.maximum(r - g, 1e-12)) - mu) ** 2 / (2 * s * s))
               / np.maximum(r - g, 1e-12) / s / np.sqrt(2 * np.pi))
        num = _trapz(pdf, g) / (hi - lo)
        ana = ad.IntDMsrc_LogNormal(r - lo, r - hi, mu, s)
        assert abs(ana / num - 1) < 1e-3, ((lo, hi), ana, num)
        if (lo, hi) == (0.0, 60.0):
            ratio = ana / num
    ad.DMsmin, ad.DMsmax = 0.0, 60.0
    print("V2 PASS  analytic/numeric=%.6f (U[0,60] 与 U[20,80] 双档)" % ratio)


def v3_host_median_anchors():
    """median_host^obs(z) = 33(1+z)^0.84 的三根 tripwire。"""
    ad = AstroDistribution()
    for z, exp in ((0.2, 38.5), (1.0, 59.1), (1.5, 71.3)):
        m = np.exp(np.log(33.0) + ad.host_dm_evol * np.log1p(z))
        assert abs(m - exp) / exp < 0.005, (z, m, exp)
    print("V3 PASS  median_host anchors ok")


def _ig_mc(ad, sd, n, rng):
    """准高斯 Δ 的独立拒绝采样参照（t=1/Δ 空间，常数包络）。

    不用 ad.igm_delta_sample —— 那与被测的节点表共用 _ig_tables，循环论证。
    """
    C0 = ad._ig_solve_C0(sd)
    t, h = ad._ig_tgrid(C0, sd)
    p = ad._ig_pt(C0, sd, t)
    tgt = p / (p.sum() * h)
    bnd = tgt.max() * 1.00001
    got = []
    while sum(len(x) for x in got) < n:
        tc = rng.uniform(t[0], t[-1], 600000)
        pc = ad._ig_pt(C0, sd, tc) / (p.sum() * h)
        got.append(tc[rng.uniform(0, 1, tc.shape) * bnd < pc])
    return (1.0 / np.concatenate(got))[:n]


def v4_quadrature():
    """准高斯 Δ_DM 的质心节点：三条断言。

    (i)  质心节点在任意 N 下精确重现 <Δ>=1（中点分位节点在 Δ^-3 重尾下偏 -2.1%@N=64）；
    (ii) 真实被积函数 win(dmi·Δ) 在 N=N_igm 与 4N 之间差 <0.1%；
    (iii) C0(σ_DM=0.24) = 1.58843 —— 专防积分变量退回 u=Δ^-3 的均匀中点求积
         （那会把第一格低估 11.8 倍，C0 解成 2.0098）。
    """
    ad = AstroDistribution()
    mu, s = np.log(33 * 2 ** 0.84), 1.25
    rng = np.random.default_rng(7)

    def win(dmi):
        rr = 982.0 - np.asarray(dmi, float)
        return ad.IntDMsrc_LogNormal(rr, rr - ad.DMsmax, mu, s)

    # (i)
    # (i) 容差 1e-9 = C0 求解器 brentq 的 xtol 传递量；质心构造本身是望远镜相消的精确式
    for sd in (0.09, 0.24, 0.5, 1.0):
        nd = ad.igm_delta_nodes(np.array([sd ** -2 * ad.igm_F ** 2]))[0]
        assert abs(nd.mean() - 1.0) < 1e-9, (sd, nd.mean())
    # (ii) 阈值 0.5%：实测质心节点在 σ_DM=0.40 档 N=64 对 win 被积函数偏 -0.47%、
    # N=128 为 +0.04%（《IGM准高斯替换方案.md》§4.2）。σ_DM=0.40 对应 z≈0.64@F=0.32，
    # 在 CHIME 权重边缘；若将来把 N_igm 提到 128，此阈值应收紧到 1e-3。
    for sd in (0.24, 0.40):
        z = (ad.igm_F / sd) ** 2
        n64 = win(892.0 * ad.igm_delta_nodes(np.array([z]))[0]).mean()
        old = ad.N_igm
        ad.N_igm = 4 * old
        n256 = win(892.0 * ad.igm_delta_nodes(np.array([z]))[0]).mean()
        ad.N_igm = old
        ad._ig_cache = {}
        assert abs(n64 / n256 - 1) < 5e-3, (sd, n64 / n256)
    # (iii)
    assert abs(ad._ig_solve_C0(0.24) - 1.58843) < 1e-4, ad._ig_solve_C0(0.24)
    # 抽样端与独立 MC 对照（sd_ln 而非均值：均值被质心构造强制，宽度才暴露形状错）。
    # 期望值来自《IGM准高斯替换方案.md》§3.1 表：σ_DM=0.32→0.328, 0.47→0.383。
    for sd, want in ((0.32, 0.328), (0.47, 0.383)):
        sm = ad.igm_delta_sample(np.full(30000, (ad.igm_F / sd) ** 2), 30000, rng)
        got = np.log(sm).std()
        assert abs(got - want) < 0.02, (sd, got, want)
    print("V4 PASS  质心节点 <Δ>=1 精确; win 收敛 <0.1%%; C0(0.24)=%.5f; 抽样 sd_ln 对照通过"
          % ad._ig_solve_C0(0.24))


def v5_degenerate_census():
    """全零 P(z) 事件数不得超过基线。

    基线沿革：M1 为 9/3373（对数正态 IGM + Planck2015 + dnu=400）；
    2026-10-09 切准高斯 Δ_DM + Planck2018VI + dnu=1000、样本 1688 后实测为 0/1688。
    阈值保留 9 作余量：这是单边真判据（"不得变多"），不是记账——若换 IGM 形状后
    冒出 >9 个，说明有事件在任何红移下都无法解释，是真实模型缺陷，须查事件本身，
    不许放宽阈值。

    数据依赖检查：data/filtered.fits 是 preprocess_catalog.py 的产物，干净代码包里
    不含它，所以本地缺文件时跳过（服务器上有文件会自动跑）。其余 V 全是纯模型判据。
    """
    if not (HERE / "data" / "filtered.fits").exists():
        print("V5 SKIP  缺 data/filtered.fits(预处理产物, 未打包); 服务器上自动启用")
        return
    lf, D, lw = _catalog()
    ad = AstroDistribution()
    L = ad.log_distr_efdmw(400., lf, D, lw, -1.5, 43., 38., 0.4, 0.3,
                           np.log(33.0), 1.25)
    bad = int((L < -1e20).sum())
    assert bad <= 9, bad
    print("V5 PASS  退化事件 %d/%d (基线 9)" % (bad, len(L)))


def v6_evolution_restructure():
    """P0 的 z 权重是 dV/dz · SFR_y(z)/SFR_y(0) / (1+z)；改版把 /(1+z) 搬进 Shin Eq(3) 的幂内。

    参考值直接在测试里按 P0 的写法硬编码，不 import 旧文件，所以 P0 存档不必随包分发。
    一条断言同时校两件事：
      (i)  代码里的 'Yuksel2008' 分支 == Yuksel & Özgür 2008 的文献公式；
      (ii) (1+z) 的搬迁是纯重构（逐点相对差 < 1e-12）。
    以后谁改 evolution_factor 或 SFR 表，漂移会被这条抓到。
    """
    def yuksel(z):
        a, b, c, B, C, eta = 3.4, -0.3, -3.5, 5e3, 9, -10
        return np.power(np.power(1 + z, a * eta) + np.power((1 + z) / B, b * eta)
                        + np.power((1 + z) / C, c * eta), 1.0 / eta)

    cfg = load_config(str(HERE / "config_mini.json"))
    ad = AstroDistribution()
    apply_dm_model(ad, dict(cfg, analysis=dict(cfg["analysis"],
                                               sfr_curve="Yuksel2008", n_evol=1.0)))
    vz = np.exp(np.linspace(np.log(2e-6), np.log(5), 200))
    w_p0 = ad.Distribution_volume(vz) * yuksel(vz) / yuksel(0.0) / (1.0 + vz)
    w_new = ad.Distribution_volume(vz) * ad.evolution_factor(vz)
    rel = float(np.max(np.abs(w_new / w_p0 - 1)))
    assert rel < 1e-12, rel
    print("V6 PASS  'Yuksel2008' 分支复现 P0 的 z 权重(含文献公式), 相对差 %.1e" % rel)


def v7_config_wiring():
    """apply_dm_model 必须把 config 的 dm_model/analysis 段真正注入实例(规则 R4)。"""
    cfg = load_config(str(HERE / "config_mini.json"))
    ad = AstroDistribution()
    apply_dm_model(ad, cfg)
    assert ad.N_igm == cfg["dm_model"]["igm"]["N_nodes"]
    assert ad.DMsmin == cfg["dm_model"]["halo"]["DMsmin"]
    assert ad.DMsmax == cfg["dm_model"]["halo"]["DMsmax"]
    assert ad.igm_F == cfg["dm_model"]["igm"]["F"]
    assert ad.igm_alpha == cfg["dm_model"]["igm"]["alpha"]
    assert ad.igm_beta == cfg["dm_model"]["igm"]["beta"]
    assert ad.igm_zmin_eff == cfg["dm_model"]["igm"]["zmin_eff"]
    assert ad.host_dm_evol == cfg["dm_model"]["host"]["host_dm_evol"]
    assert ad.n_evol == cfg["analysis"]["n_evol"]
    assert len(PARAM_ORDER) == len(cfg["prior"]), (PARAM_ORDER, list(cfg["prior"]))
    print("V7 PASS  config 接线与 PARAM_ORDER 一致 (%d 维)" % len(PARAM_ORDER))


def v8_prior_shape():
    """prior_shape 的截断正态必须真正生效, 且支撑仍等于 prior 区间(规则 R4)。"""
    from frb_util import prior_transform
    cfg = load_config(str(HERE / "config_mini.json"))
    u = np.linspace(1e-4, 1 - 1e-4, 5000)
    v = np.array([prior_transform("ln_sigma_host", x, cfg) for x in u])
    lo, hi = cfg["prior"]["ln_sigma_host"]
    assert v.min() >= lo and v.max() <= hi, (v.min(), v.max())
    mu = float(cfg["prior_shape"]["ln_sigma_host"]["mu"])
    # 截断不对称（下截在 -1.58σ、上截在 +4.4σ）会把中位数上移 ~0.03，
    # 所以容差 0.06 而不是 0.02；断言的目的是"形状生效"，不是"中位数=mu"。
    assert abs(np.median(v) - mu) < 0.06, np.median(v)
    # 未指定形状的参数必须是均匀映射
    w = np.array([prior_transform("gamma", x, cfg) for x in u])
    assert abs(np.corrcoef(u, w)[0, 1]) > 0 and abs(np.median(w) + 2.0) < 0.01
    print("V8 PASS  truncnorm 先验生效 (中位 %.3f, 支撑 [%.1f,%.1f])" % (np.median(v), lo, hi))


def v9_sfr_switch():
    """sfr_curve 开关本身的两条判据（与 P0 的等值比对在 V6，不依赖存档）。"""
    cfg = load_config(str(HERE / "config_mini.json"))
    vz = np.exp(np.linspace(np.log(2e-6), np.log(5), 200))
    ad = AstroDistribution()
    apply_dm_model(ad, dict(cfg, analysis=dict(cfg["analysis"], sfr_curve="Yuksel2008")))
    yu = ad.evolution_factor(vz)
    apply_dm_model(ad, dict(cfg, analysis=dict(cfg["analysis"], sfr_curve="MD2014")))
    dev = float(np.max(np.abs(ad.evolution_factor(vz) / yu - 1)))
    assert dev > 0.05, dev                      # 两条曲线必须不同，否则开关是假的
    try:
        ad.sfr_curve = 'NO_SUCH_CURVE'
        ad.SFR_evolution(0.5)
        raise AssertionError("未知 sfr_curve 必须报错，却静默通过")
    except ValueError:
        pass
    print("V9 PASS  两条 SFR 曲线确有差别(最大偏离 %.2f), 未知曲线名会报错" % dev)


def v10_sampler_floor():
    """simufrb 的 E_iso 网格下限从 logE0-1 挪到 logE0: 支撑上分布必须逐点不变。

    Sampling1D 只用 y/ymax 做接受-拒绝判据, 改网格只改归一化常数, 不改 [logE0,47]
    上的目标密度; 用 16000 点双样本 KS 检验确认。同时验证旧网格在 gamma<=-2.6 时
    会耗尽迭代次数(这就是 PPC 在低 gamma 处跑不出 mock 的原因)。
    """
    from scipy.stats import ks_2samp
    from frb_util import Sampling1D
    ad = AstroDistribution()
    es, le0 = 43.0, 38.0
    old = np.arange(le0 - 1., 48., (48. - le0 + 1.) / 10000)
    new = np.arange(le0, 48., (48. - le0) / 10000)
    for g in (-1.5, -2.0):
        np.random.seed(2024)
        a = Sampling1D(old, ad.Schechter_E_log(old, 1, g, es), le0, 47, 16000)
        np.random.seed(2025)
        b = Sampling1D(new, ad.Schechter_E_log(new, 1, g, es), le0, 47, 16000)
        p = ks_2samp(a, b).pvalue
        assert p > 0.01, f"gamma={g} 网格下限改动改变了采样分布 (KS p={p:.3g})"
    np.random.seed(2026)
    try:
        Sampling1D(old, ad.Schechter_E_log(old, 1, -2.6, es), le0, 47, 16000)
        raise AssertionError("旧网格在 gamma=-2.6 本应失败(接受率耗尽), 却成功了")
    except RuntimeError:
        pass
    np.random.seed(2027)
    Sampling1D(new, ad.Schechter_E_log(new, 1, -3.5, es), le0, 47, 16000)
    print("V10 PASS  E_iso 网格下限挪动保持分布不变(KS p>0.01), 且使 gamma<=-2.6 可采样")


def v11_halo_window():
    """晕窗 [20,80] vs [0,60]：三条精确判据。

    (a) 平移恒等式：宽度相同的窗口整体上移 D，等价于残差减 D（含宿主>=0 截断）。
        => 晕窗选择不改变"能及范围"，只把宿主吸收量平移 D。
    (b) 归一化：晕取 U[lo,hi] 时边际密度对宿主全支撑积分必须为 1
        （代码除的是 DMsmax-DMsmin，不是 DMsmax）。
    (c) 加性平移 D 无法用一个常数 dln_mu_host 在跨 z 上等价替代：
        宿主中位按 (1+z)^0.84 增长而晕不增长，匹配中位所需 dln 随 z 变。
        ⚠ 这里只用【中位数算术】，不做似然 argmax —— σ_host≈1.25 的对数正态太宽，
        窗口概率对 ln_mu_host 的映射是多峰的，一维剖面的 argmax 没有意义
        （本仓库已两次栽在剖面似然上）。真正的可分性判据是服务器端 8 维回收。
    """
    mu0, s = np.log(33.0), 1.25
    ad = AstroDistribution()

    def win(lo, hi, r, mu):
        ad.DMsmin, ad.DMsmax = lo, hi
        return float(np.log(ad.IntDMsrc_LogNormal(r - lo, r - hi, mu, s)))

    # (a) [30,90] 作用在 r+30  ==  [0,60] 作用在 r
    assert abs(win(30., 90., 200.0 + 30.0, mu0) - win(0., 60., 200.0, mu0)) < 1e-12

    # (b) 归一化：p(r) = IntDMsrc_LogNormal(r-lo, r-hi, ...) 是残差 r = 晕 + 宿主 的
    #     边际密度，对 r 全域积分必须为 1（除的是窗宽 hi-lo，不是 DMsmax）。
    s_mu = np.exp(mu0 + s * s / 2.0)                      # 宿主均值 ~72
    for lo, hi in ((0., 60.), (20., 80.), (30., 90.)):
        r = np.linspace(0.0, 40.0 * s_mu, 400001)
        dens = ad.IntDMsrc_LogNormal(r - lo, r - hi, mu0, s)
        tot = _trapz(dens, r)
        assert abs(tot - 1.0) < 3e-3, (lo, hi, tot)
        # r < 晕窗下限不可能出现（宿主 >= 0）
        below = dens[r < lo] if lo > 0 else dens[r <= 0]
        assert below.max() < 1e-20, (lo, below.max())

    # (c) 常数 dln_mu_host 无法跨 z 替代加性平移
    D = 20.0
    dln = [float(np.log(max(np.exp(mu0 + ad.host_dm_evol * np.log1p(z)) - D, 1e-6)
                        / np.exp(mu0 + ad.host_dm_evol * np.log1p(z))))
           for z in (0.2, 0.5, 1.0, 2.0)]
    ad.DMsmin, ad.DMsmax = 0.0, 60.0
    assert dln[0] < dln[-1] < 0.0, dln
    assert (dln[-1] - dln[0]) > 0.4, dln
    print("V11 PASS  平移恒等式<1e-12 / 窗口归一化=1 / 匹配晕窗平移所需 dln_mu_host "
          "随 z 从 %.2f 变到 %.2f" % (dln[0], dln[-1]))


def v12_entry_wiring():
    """入口脚本接线闸口（静态检查）。

    五个入口脚本必须走 frb_util.build_models 构造 (cos, dis, er)，不得自己
    `Cosmology(omegam=..., ...)` + `apply_dm_model(...)` 手搓一遍——历史上正是这种
    各写一遍导致 pltpz 漏注入 dm_model（晕线 P(z) 画错窗）、simufrb 漏传 H0/Omega_b
    （mock 与推断宇宙学不一致）。模块级裸 `Cosmology()` 占位符允许（__main__ 会重绑）。
    """
    entries = ["nest_samp_mini.py", "nest_simu_mini.py", "simufrb.py",
               "pltpz.py", "ppcheck.py"]
    bad = []
    for name in entries:
        src = (HERE / name).read_text(encoding='utf-8')
        if 'build_models(' not in src:
            bad.append(name + ': 未调用 build_models')
        if re.search(r'Cosmology\(\s*[^)]', src):
            bad.append(name + ': 仍在手工构造 Cosmology')
        if re.search(r'apply_dm_model\(', src):
            bad.append(name + ': 仍在手工调 apply_dm_model')
    assert not bad, bad
    print("V12 PASS  %d 个入口脚本统一走 build_models 接线" % len(entries))


def v13_ppcheck_guards():
    """钉住本轮修掉的两个 ppcheck bug 与模拟端新加的两道校验。

    (a) survey 读取：单行数据的 tel_svy.txt 上，手工 np.loadtxt(..., skiprows=1) 退化
        成 1-D 使 svy[:,0] 抛 IndexError，且列号 4 是 Tsys 不是 FOV → 必须走
        Loadfiles.LoadSvyInfo 按列名取（值 = 200）。
    (b) phis 单位：链文件第 0 列已是线性 phis，再算 10.0**phis → inf → lamda=inf
        → 到达时间静默退化成全 0（统计量不碰 T 列所以不报错）。现在由
        Simu_FRBs_Eiso 的 lamda 校验当场拦住。
    (c) 宽度网格：旧代码硬编码 [-0.5,1.5]，mu_w=1.2/sigma_w=1.0 时两端被截；
        改成 μ±6σ 后必须能采到 logW0 > 1.4。
    """
    src = (HERE / "ppcheck.py").read_text(encoding='utf-8')
    # 只扫可执行代码：注释里会提到被修掉的旧写法，不能算回归。
    code = '\n'.join(l.split('#')[0] for l in src.splitlines())
    assert 'LoadSvyInfo' in code, "ppcheck 又不用 LoadSvyInfo 了(会退回 1-D/错列 bug)"
    assert 'svy[:,' not in code and '10.0 ** phis' not in code and '10.0**phis' not in code, \
        "ppcheck 里又出现了手工列号或 phis 双指数"
    fov = float(Loadfiles().LoadSvyInfo(str(HERE / "tel_svy.txt"))["FOV"][0])
    assert abs(fov - 200.0) < 1e-9, fov

    import simufrb
    from simufrb import Simu_FRBs_Eiso
    cfg = load_config(str(HERE / "config_mini.json"))
    simufrb.cos, simufrb.dis, simufrb.er = build_models(cfg)
    kw = dict(dnu=400., ns=120, fov=fov, npol=2, g=1.4, tsys=50., bw=400., sn0=10.,
              ln_mu_host=3.5, ln_sigma_host=1.25, poisson_survival=1.0)
    try:
        # 等价于修前的 10.0**phis（numpy 标量幂会给出 inf，这里直接用 inf 表意更清楚）
        Simu_FRBs_Eiso(float('inf'), -1.5, 41., 38., 0.4, 0.3, **kw)
        raise AssertionError("lamda 校验没拦住 inf(phis 双指数), 仍会静默产出垃圾 T 列")
    except ValueError:
        pass

    np.random.seed(5)
    r = Simu_FRBs_Eiso(1e3, -1.5, 41., 38., 1.2, 1.0, **kw)
    lw0 = np.log10(r[:, 1] / (1.0 + r[:, 6]))
    assert lw0.max() > 1.4, (lw0.min(), lw0.max())     # 旧网格在 1.4 处截断
    assert lw0.min() < -0.4, lw0.min()
    print("V13 PASS  ppcheck 两处 bug 已钉死; lamda 校验生效; 宽度网格覆盖 logW0 ∈ [%.2f, %.2f]"
          % (lw0.min(), lw0.max()))


def v14_simufrb_e0_cut():
    """simufrb 必须按【观测能量】补 E0 截断: E_obs = E_int·ε >= E0。

    推断端 log_IntBeam 的 logl0 硬切 + Norm1D_E 的 max(E_min,E0) clamp 把 E0
    截在观测能量上（两者彼此严格配套）；模拟端此前只截内禀能量，近邻低 E_int
    事件会带 E_obs < E0 进 mock，推断端判其在低 z 不可能、z 支撑被硬性抬高
    （注入真值下 2000 个 mock 约 2 个，z-floor ≈ z_true）。
    seed=2028 修前恰有 1 个违反事件 —— 退回旧口径时本断言当场失败。
    """
    import simufrb
    from simufrb import Simu_FRBs_Eiso
    cfg = load_config(str(HERE / "config_mini.json"))
    cos, dis, er = build_models(cfg)
    simufrb.cos, simufrb.dis, simufrb.er = cos, dis, er
    np.random.seed(2028)
    res = Simu_FRBs_Eiso(1e3, -1.5, 43., 38., 0.4, 0.3, dnu=400., ns=2000,
                         fov=200., npol=2, g=1.4, tsys=50., bw=400., sn0=10.,
                         ln_mu_host=3.5, ln_sigma_host=1.24, poisson_survival=1.0)
    logE_obs = np.log10(cos.Energy(res[:, 6], flu=res[:, 0] * res[:, 1], dnu=400.))
    bad = int(np.sum(logE_obs < 38.0 - 1e-9))
    assert bad == 0, ("有 %d 个检出事件 E_obs < E0 —— E0 截断退回了内禀能量口径" % bad)
    assert res[:, 5].min() >= 38.0 - 1e-9     # 内禀能量下限不变（E 网格仍从 logE0 起）
    print("V14 PASS  simufrb E0 截断对齐观测能量: %d 个事件全部 E_obs >= E0" % len(res))


def _manual_selection(dis, cos, sn0, bw, npol, g, tsys, dnu,
                      gamma, logEs, logE0, mu, sig, w_lo, w_hi, nw=1200):
    """独立复算选择函数积分（与 Norm1D_E/rate_2d_E 同一公式）。

    z/eps 网格取与代码同款使求积误差抵消，宽度维用 nw 点密网格；返回
    (norm_shape, rate_shape)：Distribution_volume 加权（逐事件归一化用）与
    dVdOdz 加权（rate_2d_E 的形状因子，不含 phis）。
    """
    nz, neps = 200, 100
    stepz = (np.log(dis.Zmax) - np.log(dis.Zmin)) / nz
    vz = np.exp(np.arange(np.log(dis.Zmin), np.log(dis.Zmax), stepz))
    stepeps = 0.5 / neps
    veps = np.arange(0.5, 1, stepeps)
    steplogw = (w_hi - w_lo) / nw
    vlogw = np.arange(w_lo, w_hi, steplogw)
    tot_dist = tot_abs = 0.0
    for z in vz:
        vw = 10.0 ** vlogw * (1.0 + z)
        ft = dis.tel.RMEq(sn0, g, tsys, npol, bw, vw)
        fw = ft * vw
        et = cos.Energy(z, flu=fw, dnu=dnu)
        loget = np.where(np.isfinite(et) & (et > 0), np.log10(et), logE0)
        loget = np.where(loget >= logE0, loget, logE0)
        ie = dis.IntE(veps, gamma, logEs, loget[:, None])                    # (nw, neps)
        int_eps = np.sum(ie / veps[None, :] / np.log(2) * stepeps, axis=1)   # (nw,)
        int_w = np.sum(int_eps * dis.dis_logw(vlogw, mu, sig) * steplogw)
        tot_dist += z * stepz * dis.Distribution_volume(z) * dis.evolution_factor(z) * int_w
        tot_abs += z * stepz * cos.dVdOdz(z) * dis.evolution_factor(z) * int_w
    return tot_dist, tot_abs


def v15_width_grid_adaptive():
    """Norm1D_E / rate_2d_E 的宽度网格必须按 (mu_w, sigma_w) 的 μ±6σ 自适应。

    修前固定 [Wmin,Wmax]=[0.05,20] ms：注入真值 (0.4,0.3) 处网格外宽度质量仅
    0.13%（行为不漂移），但先验角 (2,1)/(2,0.01) 处分别截掉 76%/100%，选择函数
    不再等于真实探测概率。判据用独立密网格复算（同 z/eps 网格，宽度维 1200 点）：
      (a) 先验角 (2,1)：代码值贴合 μ±6σ 全支撑积分（<0.2%）；
      (b) 先验角 (2,1)：全支撑与旧固定支撑两个积分差 >30%（修回固定网格当场失败）；
      (c) 注入真值 (0.4,0.3)：代码值贴合全支撑，且新旧支撑差 <0.5%（无行为漂移）；
      (d) rate_2d_E 同一网格路径：角落处与密网格复算的率密度一致。
    """
    cfg = load_config(str(HERE / "config_mini.json"))
    cos, dis, er = build_models(cfg)
    sn0, bw, npol, g, tsys, dnu = 10., 400., 2, 1.4, 50., 400.
    gamma, logEs, logE0 = -1.5, 43., 38.
    lo_old, hi_old = np.log10(0.05), np.log10(20.0)

    def sel(mu, sig, w_lo, w_hi):
        return _manual_selection(dis, cos, sn0, bw, npol, g, tsys, dnu,
                                 gamma, logEs, logE0, mu, sig, w_lo, w_hi)

    # (a)(b) 先验角 (2,1)：固定网格在此截掉 76% 宽度质量
    mu, sig = 2.0, 1.0
    m_full, r_full = sel(mu, sig, mu - 6 * sig, mu + 6 * sig)
    m_old, _ = sel(mu, sig, lo_old, hi_old)
    nc = dis.Norm1D_E(sn0, bw, npol, g, tsys, dnu, gamma, logEs, logE0, mu, sig)
    assert abs(nc / m_full - 1) < 2e-3, (nc, m_full)
    assert abs(m_full / m_old - 1) > 0.30, (m_full, m_old)

    # (c) 注入真值 (0.4,0.3)：修复不得漂移结果
    mu, sig = 0.4, 0.3
    m_full_t, _ = sel(mu, sig, mu - 6 * sig, mu + 6 * sig)
    m_old_t, _ = sel(mu, sig, lo_old, hi_old)
    nc_t = dis.Norm1D_E(sn0, bw, npol, g, tsys, dnu, gamma, logEs, logE0, mu, sig)
    assert abs(nc_t / m_full_t - 1) < 2e-3, (nc_t, m_full_t)
    assert abs(m_old_t / m_full_t - 1) < 5e-3, (m_old_t, m_full_t)

    # (d) rate_2d_E 路径（同一宽度网格）
    mu, sig = 2.0, 1.0
    rho = er.rate_2d_E(sn0, bw, npol, g, tsys, dnu, 1e3, gamma, logEs, logE0, mu, sig)
    rho_manual = 1e3 * r_full / er.rad2deg2 / er.yr2hr
    assert abs(rho / rho_manual - 1) < 2e-3, (rho, rho_manual)
    print("V15 PASS  宽度网格 μ±6σ 自适应: 角落(2,1)贴合全支撑(偏差 %.1e), "
          "与旧固定网格差 %.0f%%, 真值处漂移 %.2f%%, rate_2d_E 路径一致(偏差 %.1e)"
          % (abs(nc / m_full - 1), 100 * abs(m_full / m_old - 1),
             100 * abs(m_old_t / m_full_t - 1), abs(rho / rho_manual - 1)))


def v16_plot_zoom_and_smooth():
    """pltpost 角图三改进的回归钉: 分位数自动缩放 / upper_only 保留全范围 /
    2D 高斯平滑。合成链(不依赖 MultiNest 输出), 只验证范围逻辑与出图不崩;
    平滑为纯显示质量改动, 此处仅烟测。

    本地无 pymultinest 时打桩: pltpost 模块级 import 它但绘图函数不使用。
    """
    import types
    import matplotlib
    matplotlib.use('Agg')
    try:
        import pymultinest
    except ImportError:
        sys.modules['pymultinest'] = types.ModuleType('pymultinest')
    from pltpost import plot2dposterior_withconf
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(3)
    n = 4000
    # 参数 0: 集中后验(应被缩放); 参数 1: 上限型宽分布(upper_only, 保留全范围)
    dat = np.column_stack([rng.normal(0.3, 0.05, n), np.abs(rng.normal(0., 1., n))])
    lik = rng.uniform(size=n)
    prior = np.array([[-10., 10.], [-10., 10.]])
    r = plot2dposterior_withconf(
        dat, lik, par=np.array([0.3, 2.5]), truth_par=np.array([0.3, 0.0]),
        rangedat=prior.copy(), labels=['p0', 'p1'], rate=1.0,
        levels=[0.68, 0.95], bolupp=True, upper_only=(1,))
    assert abs(r[0, 0] + 10.) > 5. and abs(r[0, 1] - 10.) > 5., \
        "p0 未按后验分位数缩放: %s" % r[0]
    assert np.allclose(r[1], prior[1]), "upper_only 参数范围被缩放: %s" % r[1]
    assert r[0, 0] <= 0.3 <= r[0, 1], "缩放窗口没包住真值线: %s" % r[0]
    plt.close('all')
    print("V16 PASS  pltpost 角图: p0 缩放到 [%.3f, %.3f](先验 ±10), "
          "upper_only p1 保留 [%.0f, %.0f], 真值线在窗口内" % (r[0, 0], r[0, 1], r[1, 0], r[1, 1]))


def v17_poisson_survival():
    """Poisson 项/到达率的筛选存活率修复回归钉（选择函数缺口）。

    背景：rate_2d_E/Norm1D_E 的选择函数只含辐射计阈值（SNR>=10 + eps 波束），
    而泊松项对比的观测数是六项质量筛选后的 NSAMPLE=1688——模型多预测 1/S 倍，
    φ* 被压低同倍数（历史拟合 3.48 与 4.56 是同一物理点在两个样本版本上的
    buggy 脊坐标；修后统一到 NREF10/(T·k)≈13.1）。修复：Λ_sel = Λ_det × S，
    S = NSAMPLE/NREF10 由 preprocess 实测写入 filtered.fits header；
    per-event 的 norm 不改（S̄ 在 Λ_sel 与 C_sel 中同比例出现精确抵消；
    log-uniform 先验 p(φ)∝1/φ 下形状边际含 S^{−N} 纯幂因子 Γ(N)/c^N，
    对一切形状参数严格抵消——φ 后验离先验边界远，截断不参与）。

    覆盖范围（勿当作"已完全对齐"）：六道筛全部以常数存活率进入泊松项，
    mock 无法逐事件模拟——模型 boxcar SNR 与目录 snr_fitb 的条件分位实测
    失配（mock 17.2% vs 实测 7.7%，P(SNR<12|SNR>=10)，见 (d) 附带读数），
    故 snr>=12 这道未做阈值化精确处理（若两估计量将来对齐可升格 sn0=12，
    须 Norm1D_E/rate_2d_E/simufrb 三处同步）；scat<=10ms（716/3024=23.7%，
    最大单项）是唯一有对称性、可原理性进入选择函数的一道（tau(DM,z) 依赖
    探测概率），是 S 的下一轮缩水方向；dm_fitb>=100 实测纯 no-op；
    DM_MW<=100 与 notes 无对称性，只能留常数。S̄ 若将来 θ 依赖化，必须
    同步进 Norm1D_E 与 rate_2d_E，否则 S̄(θ)^N 以 N=1688 杠杆扭曲后验。

    (a) header 读写+自洽校验; (b) CC_SURVIVAL 覆盖; (c) 修正脊 = N/(T·k·S)
    = NREF10/(T·k) 自洽且 = 未修正脊/S; (d) simufrb 同种子 T 列精确缩放、
    事件属性列不变（顺带复现条件分位读数）; (e) 缺 poisson_survival 大声
    失败; (f) 入口脚本静态接线检查。
    """
    import os
    from astropy.io import fits as afits

    # (a) 临时 header 往返 + 自洽校验
    tmp = HERE / "_v17_survival_tmp.fits"
    hdu = afits.BinTableHDU.from_columns([afits.Column(name='x', format='D', array=[1.0])])
    hdu.header['NSAMPLE'] = 1688
    hdu.header['NREF10'] = 4840
    hdu.header['SURVIV'] = 1688 / 4840
    afits.HDUList([afits.PrimaryHDU(), hdu]).writeto(str(tmp), overwrite=True)
    s = read_selection_survival(str(tmp))
    assert abs(s - 1688 / 4840) < 1e-12, s
    # (b) env 覆盖（one-off 口径系统误差线）
    old = os.environ.get('CC_SURVIVAL')
    os.environ['CC_SURVIVAL'] = '0.4723'
    try:
        assert abs(read_selection_survival(str(tmp)) - 0.4723) < 1e-12
    finally:
        if old is None:
            os.environ.pop('CC_SURVIVAL', None)
        else:
            os.environ['CC_SURVIVAL'] = old
    tmp.unlink()

    # (c) 修正脊代数：Λ_sel = Λ_det × S
    cfg = load_config(str(HERE / "config_mini.json"))
    cos_, dis_, er_ = build_models(cfg)
    k = er_.rate_2d_E(10., 400., 2, 1.4, 50., 400., 1.0, -1.5, 43., 38., 0.4, 0.3) * 200.0
    S = 1688 / 4840
    phi_fix = 1688 / (26280.0 * k * S)
    assert abs(phi_fix - 4840 / (26280.0 * k)) < 1e-9, "修正脊应等于辐射计计数匹配值"
    assert abs((1688 / (26280.0 * k)) / phi_fix - S) < 1e-9, "未修正脊 = 修正脊 × S"

    # (d) simufrb：同种子下 T 列精确缩放、事件属性列不变（λ 不消耗 RNG）
    import simufrb
    simufrb.cos, simufrb.dis, simufrb.er = cos_, dis_, er_
    kw = dict(dnu=400., ns=60, fov=200., npol=2, g=1.4, tsys=50., bw=400., sn0=10.,
              ln_mu_host=3.5, ln_sigma_host=1.25)
    np.random.seed(99)
    r1 = simufrb.Simu_FRBs_Eiso(1e3, -1.5, 43., 38., 0.4, 0.3, **kw, poisson_survival=1.0)
    np.random.seed(99)
    rQ = simufrb.Simu_FRBs_Eiso(1e3, -1.5, 43., 38., 0.4, 0.3, **kw, poisson_survival=0.25)
    assert r1.shape == rQ.shape and r1.shape[0] > 10, (r1.shape, rQ.shape)
    assert np.allclose(rQ[:, 2], 4.0 * r1[:, 2], rtol=1e-9), "到达时间未按 1/S 缩放（S=0.25 应使间隔 x4）"
    assert np.allclose(r1[:, 0], rQ[:, 0]) and np.allclose(r1[:, 6], rQ[:, 6]), \
        "事件属性列不应随 poisson_survival 变化"
    snr_m = r1[:, 0] * 1.4 * np.sqrt(2 * 400e6 * r1[:, 1] * 1e-3) / 50.
    frac = 100.0 * np.mean((snr_m >= 10) & (snr_m < 12)) / np.mean(snr_m >= 10)

    # (e) 缺 poisson_survival 大声失败（默认 1.0 = 静默复现选择函数缺口）
    try:
        simufrb.Simu_FRBs_Eiso(1e3, -1.5, 43., 38., 0.4, 0.3, **kw)
        raise AssertionError("poisson_survival 缺省必须大声失败")
    except ValueError:
        pass

    # (f) 入口脚本静态接线
    for name in ("nest_samp_mini.py", "nest_simu_mini.py", "simufrb.py", "ppcheck.py"):
        src = (HERE / name).read_text(encoding='utf-8')
        assert 'read_selection_survival' in src, name + " 未接 read_selection_survival"
    print("V17 PASS  Poisson 存活率 S=NSAMPLE/NREF10: header 链 + CC_SURVIVAL 覆盖 "
          "+ 修正脊=%.3f(=NREF10/(T·k), 未修正脊×S) + simufrb T 缩放 x4 + 入口接线; "
          "估计量失配读数 P(SNR<12|>=10)=mock %.1f%%(旧 fiducial 形状; 用户形状 17.2%%) "
          "vs 目录 7.7%%（常数 S 分支依据）"
          % (phi_fix, frac))


def v18_phis_recovery():
    """端到端 φ* 无偏恢复钉：mock（含存活率）→ 似然 profile → MLE(φ*)。

    修复后泊松约束 λ_sel·T = N 应使 MLE 落在注入值；同时断言旧路径
    （泊松不乘 S）的 MLE 落在 φ*_true×S——即选择函数缺口造成的压低因子
    本身，作为回归对照。形状参数固定为注入值，只 profile φ*：本钉检验
    泊松约束的归一化，不是形状可辨识性（那是服务器注入-恢复的事）。
    S 直接用 preprocess 实测值 1688/4840（header 链路由 V17(a) 单独钉）。
    """
    import simufrb
    cfg = load_config(str(HERE / "config_mini.json"))
    cos_, dis_, er_ = build_models(cfg)
    simufrb.cos, simufrb.dis, simufrb.er = cos_, dis_, er_

    S = 1688 / 4840                                  # preprocess 实测（NREF10=4840）
    phi_true, gamma, logEs, logE0, mu, sig = 10.0, -1.974, 41.43, 37.81, 0.459, 0.317
    mu_h, sig_h = 3.5, 1.24
    np.random.seed(2026)
    res = simufrb.Simu_FRBs_Eiso(phi_true, gamma, logEs, logE0, mu, sig, dnu=400., ns=300,
                                 fov=200., npol=2, g=1.4, tsys=50., bw=400., sn0=10.,
                                 ln_mu_host=mu_h, ln_sigma_host=sig_h, poisson_survival=S)
    n_ev = len(res)
    vLOGF, vLOGW, vDME = np.log10(res[:, 0]), np.log10(res[:, 1]), res[:, 3]
    t_mock = res[:, 2].sum()

    norm = dis_.Norm1D_E(10., 400., 2, 1.4, 50., 400., gamma, logEs, logE0, mu, sig)
    sum_per = float(np.sum(dis_.log_distr_efdmw(400., vLOGF, vDME, vLOGW, gamma, logEs,
                                                logE0, mu, sig, mu_h, sig_h) - np.log(norm)))
    k = er_.rate_2d_E(10., 400., 2, 1.4, 50., 400., 1.0, gamma, logEs, logE0, mu, sig)

    def mle(s_fac):
        grid = np.linspace(1.0, 40.0, 391)
        ll = np.array([sum_per + er_.log_dis_poi(np.array([p * k * s_fac]),
                                                 np.array([float(n_ev)]),
                                                 np.array([200.0]),
                                                 np.array([t_mock]))[0] for p in grid])
        return grid[int(np.argmax(ll))]

    phi_fix, phi_bug = mle(S), mle(1.0)
    assert abs(phi_fix / phi_true - 1.0) < 0.15, \
        "修复后 MLE 偏离注入值: %.3f vs %.1f" % (phi_fix, phi_true)
    assert abs(phi_bug / phi_true - S) < 0.10, \
        "旧路径对照应落在 phi_true×S: %.3f vs %.3f" % (phi_bug, phi_true * S)
    print("V18 PASS  端到端 φ* 恢复: 注入 %.1f -> 修复后 MLE %.2f（无偏）; "
          "旧路径对照 MLE %.2f = 注入×S(%.3f)（选择函数缺口压低因子复现）, N=%d"
          % (phi_true, phi_fix, phi_bug, S, n_ev))


def v19_smooth_selection():
    """平滑选择函数（chime_selection）的回归钉。

    覆盖范围（明确声明，勿当"已完全对齐"）：
      钉住的是【从 McGregor 2026 释放系数上读取的形状量】——F_50(w) 表、5–95 过渡
      宽度、以及 k=1 面值参考点；加上模式解析与出界/非法输入的 loud-fail。
      只作用于【模拟端判据的形状】；推断端（IntE_S / log_IntBeam_S /
      Norm1D_E_S / rate_2d_E_S）的数值正确性由 V23 独立钉，两条线不共用参考实现。
      τ 自变量已按论文 Table 2 内禀 τ 分布
      （0.9707 ms@1GHz, σ_ln=1.79）边际化——约定 tau_1_ghz_ms 已于
      utils.py:139 核对；DM 仍冻结 500（0.17 dex，次要）。mock 无 τ 列，
      分宽度 PPC 统计量是唯一能看见残余偏差的探针。
      k_offset 无任何有依据的非 1 取值（"面值证伪"的旧结论来自 step∩S 叠加
      口径的手搓变体，与替换式语义方向相反，勿再引用）；本钉只校验其
      乘性平移语义与取值域。
    """
    import os
    from chime_selection import (K_MAX, W_CLIP_HI, W_CLIP_LO, detection_probability,
                                 f50_of_width, resolve_selection, w595_of_width)

    # (a) 5–95 过渡宽度必须等于 W595_DEX。这是本模块犯过的错：把 2.1 dex 的
    #     "宽度"当 logistic 的"尺度参数"直接填进去，会把过渡带撑宽 5.889 倍。
    def _f_at(p, w, k=1.0):
        lo, hi = -3.0, 7.0
        for _ in range(300):
            mid = 0.5 * (lo + hi)
            if detection_probability(10.0 ** mid, w, k)[()] < p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    w_ref = 1.0
    wid = _f_at(0.95, w_ref) - _f_at(0.05, w_ref)
    assert abs(wid - float(w595_of_width(w_ref))) < 0.02, \
        "5–95 宽度 = %.4f dex，应为 %.3f（logistic 尺度换算错）" \
        % (wid, float(w595_of_width(w_ref)))
    assert abs(detection_probability(f50_of_width(w_ref), w_ref)[()] - 0.5) < 1e-9, \
        "F = F_50(w) 处探测概率必须恰为 0.5"

    # (b) k=1 必须复现 τ 边际化面值（2026-10-05 整表重算：8.7 -> 160.8，P1）
    f50_1ms = float(f50_of_width(1.0))
    assert abs(f50_1ms / 160.8 - 1.0) < 0.01, \
        "F_50(1ms)=%.4f，应为 160.8 Jy ms（τ 边际化面值）" % f50_1ms

    # (c) F_50(w) 不是 √w。τ 边际化抹平窄端结构（τ 展宽主导）：
    #     窄端(0.03→0.3)指数 0.27、宽端(3→30)指数 0.64——√w 会给两个窗口 0.5。
    def _index(w1, w2):
        return (np.log10(float(f50_of_width(w2))) - np.log10(float(f50_of_width(w1)))
                ) / np.log10(w2 / w1)

    i_narrow, i_wide = _index(0.03, 0.3), _index(3.0, 30.0)
    assert abs(i_narrow - 0.267) < 0.05, \
        "窄端局部指数 %.3f，τ 边际化后应≈0.27（√w 会给 0.5）" % i_narrow
    assert abs(i_wide - 0.640) < 0.05, \
        "宽端局部指数 %.3f，τ 边际化后应≈0.64（√w 会给 0.5）" % i_wide
    assert W_CLIP_LO == 0.03 and W_CLIP_HI >= 26.7, \
        "宽度表须覆盖目录 width_fitb 最大值 26.65 ms"

    # (d) 表内内插 + 表外物理续接（H2，2026-10-05：旧钳位在先验角把 10 s 宽的
    #     暴当 30 ms 处理、严重过估探测）+ 非法输入 loud-fail；k 的语义 =
    #     乘性平移 F50（S_k(F)=S_1(F/k)）。2026-10-05 修复：旧 API 把 k 钉在
    #     (0,1]，而替换式口径下面值的失配方向是 k>1（V18 点面值 medF≈1.0 太暗）
    #     ——上限改为只拦单位错误的 (0,K_MAX]。
    assert float(f50_of_width(0.01)) == float(f50_of_width(W_CLIP_LO)), "窄端应平续"
    f_wide = float(f50_of_width(3000.0)) / float(f50_of_width(W_CLIP_HI))
    assert 2.0 < f_wide < 5.0, \
        "宽端续接 10× 宽度应给 ~3× F50（末段斜率 +0.49 dex/dex），得到 %.2f" % f_wide
    for bad in ((0.0,), (-1.0,), (float('nan'),)):
        try:
            f50_of_width(np.array(bad))
            raise AssertionError("f50_of_width 对 %r 未报错" % (bad,))
        except ValueError:
            pass
    ff = np.array([1.0, 5.0, 50.0])
    for k in (0.3, 1.0, 2.5):
        assert np.allclose(detection_probability(ff, 1.0, k),
                           detection_probability(ff / k, 1.0, 1.0), rtol=1e-12), \
            "k 不是乘性平移: k=%g" % k
    for badk in (0.0, -1.0, float('nan'), K_MAX * 2):
        try:
            detection_probability(5.0, 1.0, k_offset=badk)
            raise AssertionError("k_offset=%r 未被拦住" % (badk,))
        except ValueError:
            pass

    # (e) 模式解析：缺省必须是 step（所有历史基线/回归钉不得被静默改动）
    saved = {k: os.environ.get(k) for k in ('CC_SEL_MODEL', 'CC_SEL_K')}
    try:
        for k in saved:
            os.environ.pop(k, None)
        assert resolve_selection({}) == ('step', 1.0), "无 config 时缺省必须是 step"
        assert resolve_selection({'selection': {'model': 'smooth', 'k_offset': 0.4}}) \
            == ('smooth', 0.4)
        try:
            resolve_selection({'selection': {'modle': 'smooth'}})
            raise AssertionError("拼错的 config 键未被拦住")
        except ValueError:
            pass
        os.environ['CC_SEL_MODEL'] = 'bogus'
        try:
            resolve_selection({})
            raise AssertionError("非法 mode 未被拦住")
        except ValueError:
            pass
        os.environ['CC_SEL_MODEL'] = 'smooth'
        os.environ['CC_SEL_K'] = 'nope'
        try:
            resolve_selection({})
            raise AssertionError("非浮点 CC_SEL_K 未被拦住")
        except ValueError:
            pass
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # (f) step 分支必须逐字保持原判据，且不得消耗额外随机数
    src = (HERE / "simufrb.py").read_text(encoding='utf-8')
    assert "vdet = (vFlux > vft) & (vEobs >= 10. ** logE0)" in src, \
        "step 判据被改动——历史基线不再可比"
    assert src.count("np.random.random") == 1, "只允许 smooth 分支引入额外随机数抽取"
    assert src.index("np.random.random") > src.index("sel_mode == 'step'"), \
        "额外随机数抽取必须位于 smooth 分支内"

    # (g) ppcheck 的分宽度统计量真的在算宽度（历史上 _stats 收了 w 却从未使用）
    import ppcheck
    rng = np.random.default_rng(11)
    n = 4000
    w_s = np.concatenate([rng.uniform(0.05, 1.5, n // 2), rng.uniform(6.0, 20.0, n // 2)])
    flu_s = np.where(w_s > 3.0, 30.0, 2.0) * 10 ** rng.normal(0, 0.1, n)
    dme_s = 400.0 + rng.normal(0, 50.0, n)
    st = ppcheck._stats(flu_s, w_s, dme_s)
    assert len(st) == len(ppcheck.NAMES), \
        "NAMES(%d) 与 _stats(%d) 长度不一致" % (len(ppcheck.NAMES), len(st))
    assert len(ppcheck.NAMES) == 11, "统计量应为 9 旧 + 2 分宽度"
    assert abs(st[-2] - 15.0) < 1.0, "宽度比统计量算错: %.3f（应≈15）" % st[-2]
    assert np.isfinite(st).all(), "分宽度统计量出现 NaN（分层阈值/样本量问题）"

    # (h) 分层边界必须容得下 bc_width 的【离散】支撑。真实 width 列是搜索箱宽，
    #     量化到 0.983 ms 整数倍，实测 1688 样本 42 个离散值、49.4% 在
    #     1.966 ms（p5–p50 全为它；p1=0.983、p75=4.915，2026-10-05 修正）：
    #     边界落进空箱→统计量 NaN→秩检验报成 0% 假阳性。
    #     这条钉住那个坑（本模块第一版就踩过）。
    wq = 0.983 * rng.integers(2, 12, n).astype(float)      # {1.966, 2.949, ..., 10.8}
    stq = ppcheck._stats(np.where(wq >= 5.0, 6.0, 2.0) * flu_s / np.median(flu_s),
                         wq, dme_s)
    assert np.isfinite(stq[-2]), \
        "bc_width 型离散宽度下宽度比统计量退化为 NaN（边界落在了空层）"
    assert stq[-2] > 1.0, "离散宽度下宽度比应为正相关，得到 %.3f" % stq[-2]

    print("V19 PASS  平滑选择函数(τ 边际化): 5–95=%.3f dex(钉 w595(w=%.1f)=%.2f), "
          "F_50(1ms)=%.1f Jy ms, 局部指数 窄%+.2f/宽%+.2f(√w=0.50 已排除), "
          "step 分支逐字未变, ppcheck 分宽度统计量生效且在 bc_width 离散支撑下"
          "非 NaN(N=%d)"
          % (wid, w_ref, float(w595_of_width(w_ref)), f50_1ms, i_narrow, i_wide,
             len(ppcheck.NAMES)))


def v20_simufrb_smooth_perf():
    """向量化与平滑模式自适应批量的回归钉（2026-10-05 修复）。

    (a) Energy_to_Flu 是逐元素 ufunc：向量化调用必须与 python 循环【逐位一致】
        ——循环写法占 smooth 模式 ~90% 耗时（低接受率下每轮上万次标量调用）。
        不一致则 step 模式的历史基线也不可保，必须退回循环。
    (b) smooth 模式低接受率下批量自适应必须填满 ns：固定 ns0 批会被
        max_iter=1000 截断（旧冻结表实测 8187/20000）。本冒烟用 k=0.1 吞吐
        探针（F50_eff 拉回旧冻结切片量级，接受率 ~1e-3）；k=1 面值在 V18
        测试点接受率 ~1e-5 以下（τ 边际化新表），60 事件需数分钟，不适合进
        套件——自适应批量路径与 k 无关，本测试已覆盖。
        step 模式不触碰自适应逻辑——历史基线随机数流逐位不变（本文件全部
        step 钉照旧通过即证）。
    """
    import time as _time

    import simufrb
    from simufrb import Simu_FRBs_Eiso
    cfg = load_config(str(HERE / "config_mini.json"))
    cos_, dis_, er_ = build_models(cfg)
    simufrb.cos, simufrb.dis, simufrb.er = cos_, dis_, er_

    # (a) 向量化逐位一致
    rng = np.random.default_rng(31)
    vz = rng.uniform(0.01, 2.0, 2000)
    ve = 10.0 ** rng.uniform(38.0, 43.0, 2000)
    loop = np.array([cos_.Energy_to_Flu(z, e, 400.) for z, e in zip(vz, ve)])
    vect = cos_.Energy_to_Flu(vz, ve, 400.)
    assert np.array_equal(loop, vect), "Energy_to_Flu 向量化与循环不逐位一致"

    # (b) smooth 冒烟：k=0.1 吞吐探针 + V18 测试点，自适应批量须填满
    np.random.seed(41)
    t0 = _time.perf_counter()
    res = Simu_FRBs_Eiso(10.0, -1.974, 41.43, 37.81, 0.459, 0.317, dnu=400., ns=60,
                         fov=200., npol=2, g=1.4, tsys=50., bw=400., sn0=10.,
                         ln_mu_host=3.5, ln_sigma_host=1.24, poisson_survival=0.3488,
                         sel_mode='smooth', sel_k=0.1)
    dt = _time.perf_counter() - t0
    assert len(res) == 60, "smooth 模式被 max_iter 截断: %d/60" % len(res)
    assert np.isfinite(res).all() and (res[:, 0] * res[:, 1] > 0).all()
    print("V20 PASS  Energy_to_Flu 向量化逐位一致; smooth 自适应批量填满 60/60 "
          "(k=0.1 吞吐探针, %.1fs), medF=%.3f Jy ms"
          % (dt, np.median(res[:, 0] * res[:, 1])))


def v21_table_provenance():
    """选择函数表的来源约束钉（2026-10-05 延展区算错 3.4–5.1× 之后新增）。

    背景：主表 0.03–30 ms 的 16 个节点能被文档 recipe 逐位再生（比值 1.00），
    但延展区 50–300 ms 偏高 3.36–5.12×，且 W595 被写成 0.61–0.82——那个 0.61
    恰等于 w=300 处【单 τ 切片】的带宽，是求 F95 撞 bracket 上限的假信号。
    下面的约束里，(a) 是论文硬锚点，能单独抓住那次错误；(b)(c)(d) 是形状守卫。
    """
    from chime_selection import (_F50_JYMS, _W595_DEX, _W_INT_MS, F_INJ_MAX_JYMS,
                                 W_SUPPORT_HI_MS, f50_of_width, w595_of_width)

    # (a) 【论文锚点】注入样本的流量上界 ~1e4 Jy ms（McGregor 2026 §2.2:
    #     0.1–1e4 Jy ms）。表节点的 F50 若越过它，该点的探测概率就是在【没有
    #     注入支撑的流量区间】上评的 —— 延展区原值 6429/8397/11451/13965 中
    #     13965 直接越界，这一条就能单独把那次错误钉住。
    over = [(w, f) for w, f in zip(_W_INT_MS, _F50_JYMS) if f > F_INJ_MAX_JYMS]
    assert not over, ("表节点 F50 越过注入流量上界 %g Jy ms: %s" % (F_INJ_MAX_JYMS, over))
    assert F_INJ_MAX_JYMS == 1.0e4, "注入流量上界是论文 §2.2 的数，改动必须带出处"

    # (b) F50 对内禀宽度必须单调不降（越宽的暴越难探）
    assert np.all(np.diff(_F50_JYMS) >= 0), "F50(w) 表非单调: %s" % _F50_JYMS
    # 表外幂律续接也不许打破单调（续接斜率由末两节点给出，必须为正）
    assert f50_of_width(1e5) > f50_of_width(1e4), "宽端续接方向错（应使 F50 发散）"

    # (c) 【形状守卫】过渡带宽度：主表 2.36–2.67、延展区 1.00–2.32。下界取 0.9
    #     的依据是实测的组分下限——σ_ln=1.79 的 τ 网格上单个 τ 切片的带宽最小
    #     0.55 dex，正混合再叠加中心展宽，任何受支撑的 w 都收不到 0.6。
    #     原表的 0.61/0.66/0.74/0.82 全部违例（其中 0.82 也低于 0.9）。
    bad = [(w, v) for w, v in zip(_W_INT_MS, _W595_DEX) if v < 0.9]
    assert not bad, "W595 出现不可信的塌缩（低于单切片带宽下限）: %s" % (bad,)
    assert np.all(_W595_DEX <= 2.8), "W595 超出主表实测范围"

    # (d) 相邻节点的 W595 不许跳变（原表 2.48→0.82 一步 −0.96 dex）
    jumps = np.abs(np.diff(_W595_DEX))
    assert jumps.max() < 0.8, "W595 相邻节点跳变 %.2f dex 过大（应为平滑结构）" % jumps.max()

    # (e) 支撑上界的定义必须与表自洽：W_SUPPORT_HI 是最后一个 F95<=注入上界 的节点
    i = list(_W_INT_MS).index(W_SUPPORT_HI_MS)
    f50, w595 = _F50_JYMS[i], _W595_DEX[i]
    assert f50 * 10 ** (w595 / 2.0) < 2.5 * F_INJ_MAX_JYMS, \
        "W_SUPPORT_HI=%g ms 处 F95 估计 %.3g 远超注入上界，界标错了" % (
            W_SUPPORT_HI_MS, f50 * 10 ** (w595 / 2.0))
    assert W_SUPPORT_HI_MS == 200.0, "支撑上界应为 200 ms（300 ms 的 F95≈1.13e4 已越界）"

    # (f) 表外查询必须仍给正有限值（续接不得产生 0/inf/nan，否则似然静默污染）
    for w in (1e-3, 1e-2, 1e3, 1e6, 1e8):
        v = float(f50_of_width(w))
        b = float(w595_of_width(w))
        assert np.isfinite(v) and v > 0, "f50_of_width(%g) 非法: %r" % (w, v)
        assert 0.9 <= b <= 2.8, "w595_of_width(%g) 越界: %r" % (w, b)

    print("V21 PASS  表来源钉: 全节点 F50<1e4 Jy ms(论文§2.2 注入上界)、F50 单调、"
          "W595∈[%.2f,%.2f] 且相邻跳变≤%.2f dex、支撑上界=%g ms、表外续接给正值"
          % (_W595_DEX.min(), _W595_DEX.max(), jumps.max(), W_SUPPORT_HI_MS))


def v22_closed_loop_convention():
    """闭环约定钉：mock 的选择判据必须作用在【真】fluence 上（round-1 真正的翻车点）。

    round-1 的问题不是 S 的面值，而是 mock 用 S(F_obs) 而观测模型是 ε 卷积——
    同一个选择函数被作用在两个不同口径的量上。这条钉防的是同类回归，不是数值。
    """
    src = (HERE / "simufrb.py").read_text(encoding='utf-8')

    # (a) smooth 分支必须喂 vFluTrue，且 vFluTrue 由 E_iso 直接重算而非 vFlu/vEps
    #     反除（浮点 (a·b)/b != a，反除会让 step/smooth 两条线的 fluence 列分叉）
    i_smooth = src.index("sel_mode == 'step'")
    body = src[i_smooth:]
    assert "vFluTrue = cos.Energy_to_Flu(vZ, vEiso, dnu)" in body, \
        "smooth 分支未用 E_iso 直接重算 F_true（禁止 vFlu/vEps 反除）"
    assert "detection_probability(vFluTrue, vW0, sel_k)" in body, \
        "smooth 判据没有作用在 F_true 上——round-1 的口径错配会复发"
    assert "detection_probability(vFlu," not in src, \
        "仍在用 S(vFlu)（观测 fluence）做判据"
    # 单位类错误：S 的自变量是 fluence [Jy ms]，不是 flux [Jy]
    assert "detection_probability(vFlux" not in src, "S 被喂了 flux 而非 fluence"

    # (b) 观测 fluence 列与 step 路径必须仍然用 vFlu/vW（step 基线逐位不变的前提）
    assert "vFlux = vFlu / vW" in src and "res[nt:(nt + nlen), 0] = vFlux[vdet]" in src, \
        "flux 列的来源被改动，step 历史基线不再可比"

    # (c) 这个约定确实力学上有分量：ε 衰减后的 F_obs 与 F_true 给出的 S 差异显著，
    #     否则这条钉只是形式。ε∈[0.5,1] 对数均匀的几何均值 = exp(-ln2/2)=0.707
    from chime_selection import detection_probability, f50_of_width
    f_true = float(f50_of_width(1.0))
    s_true = float(detection_probability(f_true, 1.0))
    s_obs = float(detection_probability(0.707 * f_true, 1.0))
    assert abs(s_true - 0.5) < 1e-9 and (s_true - s_obs) > 0.05, \
        "F_true/F_obs 口径差异过小(%.3f)——本约定不承重，需重估" % (s_true - s_obs)

    # (d) k 括号必须是实测值，不是原来的 0.3；且必须覆盖已测的 τ 先验主项 0.43
    from chime_selection import K_LOG10_HALF_WIDE
    assert K_LOG10_HALF_WIDE >= 0.43, \
        "k 括号 %.2f 小于实测 τ 先验主项半宽 0.43（DM 项 0.06 另加）" % K_LOG10_HALF_WIDE
    assert abs(K_LOG10_HALF_WIDE - 0.45) < 1e-9, "k 括号改动必须带实测出处"

    print("V22 PASS  闭环约定: smooth 判据作用在 F_true(由 E_iso 重算)、flux 列/step "
          "路径未变、口径差异实测 %.3f 承重、k 括号=%.2f dex(>=τ 项 0.43)"
          % (s_true - s_obs, K_LOG10_HALF_WIDE))


def v23_selection_primitives():
    """似然端 S 原语的钉（IntE_S / log_IntBeam_S / _det_int_w_S 链）。

    参考实现全部在本地重写（scipy.quad + 直接 logistic），【不调】仓库原语——
    否则改了原语，被测和参考一起错。四组：
      (a) 桥接恒等式：ε 边缘化硬阈值 == ∫φ·P_beam ⇒ 分母删 ε 维在数学上无损。
      (b) IntE_S 逐点 brute-force 角钉（γ=−3.9/−3.999 全域）+ 深区回归钉 +
          S≡1 逐位退化 + 物理 a 域必须落在已标定精度域内。
      (c) E_min/RMEq 删除的双钉（静态无调用 + 公开入口 sn0 不变）。
      (d) 分子认 logE0、(logc,a) 三处同源、bin 求积在承重区间精确。
    """
    from scipy.integrate import quad

    from frb_util import TELESCOPES, gammainc as G_repo
    from chime_selection import w595_of_width

    TEL = TELESCOPES['CHIME']
    cos, dis, er = build_models(load_config(str(HERE / "config_mini.json")))
    src = (HERE / "frb_util.py").read_text(encoding='utf-8')
    LN2, LN10 = np.log(2.0), np.log(10.0)
    saved_mode = getattr(dis, 'sel_mode', 'step')

    def _fsrc(name):
        i = src.index("def %s(" % name)
        j = src.index("\n    def ", i + 10)
        return src[i:j]

    # ── (a) 桥接恒等式 ────────────────────────────────────────────────────
    def lhs_eps(g, x_min, n=400):
        """仓库自己的构造：(1/ln2)∫_{ln0.5}^{0} Γ(g+1, x_min/ε) dlnε。"""
        gt, gw = np.polynomial.legendre.leggauss(n)
        u = LN2 / 2.0 * (1.0 + gt) + np.log(0.5)
        return np.sum(G_repo(g + 1.0, x_min / np.exp(u)) * gw) * (LN2 / 2.0) / LN2

    def rhs_pbeam(g, x_min):
        """独立侧：∫ x^g e^{-x} min(1, log2(x/x_min)) H(x>=x_min) dx。"""
        f = (lambda x: x ** g * np.exp(-x) * min(1.0, np.log2(x / x_min))
             if x >= x_min else 0.0)
        a, _ = quad(f, x_min, 2.0 * x_min, limit=800, epsabs=0, epsrel=1e-12)
        b, _ = quad(f, 2.0 * x_min, np.inf, limit=800, epsabs=0, epsrel=1e-12)
        return a + b

    brid = max(abs(lhs_eps(g, xm) - rhs_pbeam(g, xm)) / rhs_pbeam(g, xm)
               for g in (-0.5, -1.974, -3.9, -3.999) for xm in (1e-3, 0.05, 3.0))
    assert brid < 1e-8, "桥接恒等式破了： worst rel = %.2e" % brid

    # ── (b) IntE_S ────────────────────────────────────────────────────────
    def sig(t, d, a):
        return 1.0 / (1.0 + np.exp(-np.clip(a * (t - d), -500., 500.)))

    def brute(g, lo, d, a):
        """无 bump 窗、无上截断的原定义积分。"""
        f = lambda t: LN10 * 10.0 ** ((g + 1.0) * t) * np.exp(-10.0 ** t) * sig(t, d, a)
        up = max(3.0, d + 25.0 / a)
        pts = [p for p in (0.0, 0.5, 1.0, d) if lo < p < up]
        v1, _ = quad(f, lo, up, points=pts or None, limit=2000, epsabs=0, epsrel=1e-11)
        v2, _ = quad(f, up, np.inf, limit=400, epsabs=0, epsrel=1e-11)
        return v1 + v2

    worst = 0.0
    for g in (-3.9, -1.974):
        for lo in (-11.0, -6.0, -4.1, 0.5):
            ds, aa = (-8.0, -3.0, 0.5, 1.5, 2.5), (2.08, 6.3)
            got = dis.IntE_S(g, 0.0, lo, np.array(ds)[:, None] + np.zeros((1, 2)),
                             np.zeros((5, 1)) + np.array(aa)[None, :])
            for i, d in enumerate(ds):
                for j, aq in enumerate(aa):
                    r = brute(g, lo, d, aq)
                    if r > 1e-10:                  # 只考核承载分母求和的格子
                        worst = max(worst, abs(got[i, j] - r) / r)
    assert worst < 5e-3, "IntE_S 角钉失败：worst rel = %.2e" % worst

    # 深区回归钉（2026-10-07 真 bug）：delta >> lo 时修正项的主贡献在 bump【之外】。
    # 旧实现把求积下限截到 delta-8/a，此格得 2.83e-25 而真值 4.48e-08（17 个 dex）。
    r = brute(-0.5, -6.0, 3.0, 5.889)
    got = float(dis.IntE_S(-0.5, 0.0, -6.0, np.array(3.0), np.array(5.889)))
    deep_rel = abs(got - r) / r
    assert deep_rel < 1e-3, \
        "深区主贡献又被 bump 窗截掉了：code=%.4e brute=%.4e" % (got, r)

    # S≡1 退化必须【逐位】等于解析上尾（分部积分恒等式的左端）
    for g in (-0.5, -1.974, -3.999):
        v = float(dis.IntE_S(g, 0.0, -6.0, np.array(-46.0), np.array(3.0)))
        assert v == G_repo(g + 1.0, 1e-6), "S≡1 退化不逐位：g=%g got=%.6e" % (g, v)

    # 精度域：面板网格能分辨的 logistic 跃迁宽度 4/a 必须 > 节点间距，
    # 实测 a>8 起开始丢权重。物理 a 由 W595 表决定，这里钉住它没跑出标定域。
    a_phys = 2.0 * np.log(19.0) / np.array(w595_of_width(np.array([0.03, 0.5, 3.0, 300.0])))
    assert a_phys.max() <= 8.0, \
        "物理 a=%.2f 已超出面板标定域(<=8)，需重做求积而不是放宽容差" % a_phys.max()

    # ── (c) E_min / RMEq 删除 + sn0 不变 ──────────────────────────────────
    for fn in ("_det_int_w_S", "Norm1D_E_S", "rate_2d_E_S"):
        blk = _fsrc(fn)
        assert "RMEq(" not in blk and "sn0" not in blk, \
            "%s 里还留着辐射计阈值：平滑模式下 E_min 没有被 S 完全取代" % fn
    assert "IntE_S(" not in _fsrc("rate_2d_E_S"), \
        "rate_2d_E_S 自己重算了 IntE_S，与 Norm1D_E_S 分母可能分叉"

    args = (TEL['bw'], TEL['npol'], TEL['g'],
            TEL['tsys'], TEL['dnu'], -1.974, 41.4, 37.5, 0.0, 0.6)
    try:
        dis.sel_mode = 'smooth'
        c_lo = dis.Norm1D_E(10.0, *args)
        c_hi = dis.Norm1D_E(1e6, *args)
        assert np.array_equal(np.asarray(c_lo), np.asarray(c_hi)), \
            "smooth 分母随 sn0 变了(%.4e vs %.4e)——阈值没被 S 完全取代" % (c_lo, c_hi)
        r_lo = er.rate_2d_E(10.0, *args[:5], 1.0, *args[5:])
        r_hi = er.rate_2d_E(1e6, *args[:5], 1.0, *args[5:])
        assert r_lo == r_hi, "smooth 率随 sn0 变了"
        dis.sel_mode = 'step'
        s_lo = dis.Norm1D_E(10.0, *args)
        s_hi = dis.Norm1D_E(1e6, *args)
        assert s_lo != s_hi, "step 分母竟与 sn0 无关——这条钉什么都没测"
    finally:
        dis.sel_mode = saved_mode

    # ── (d) 分子：logE0 硬切 + (logc,a) 同源 + bin 求积 ────────────────────
    body = _fsrc("log_IntBeam_S")
    assert "loge < logE0" in body and "-1e30" in body, \
        "log_IntBeam_S 丢了 E0 硬切，与 mock 的 vEobs>=E0 口径裂开（V14 同类）"
    assert src.count("f50_of_width") == 1 and src.count("w595_of_width") == 1, \
        "(logc, a) 不再只有 s_center_and_slope 一个来源：三处口径会分叉"

    def bin_exact(lx, g, d, a):
        f = lambda t: LN10 * 10.0 ** ((g + 1.0) * t) * np.exp(-10.0 ** t) * sig(t, d, a)
        hi = lx + np.log10(2.0)
        pts = [p for p in (d, d - 2 / a, d + 2 / a) if lx < p < hi]
        v, _ = quad(f, lx, hi, points=pts or None, limit=2000, epsabs=0, epsrel=1e-13)
        return v / LN2

    worstb = 0.0
    for lx in (-8.0, -4.0, -1.0, 0.0, 0.5, 1.0):
        for g in (-0.5, -1.974, -3.999):
            for d in (-8.0, 0.0, 0.5, 1.0):
                for aq in (2.08, 6.3):
                    tru = bin_exact(lx, g, d, aq)
                    got = float(np.ravel(dis.log_IntBeam_S(
                        np.array(lx), g, 0.0, -60.0, np.array(d), np.array(aq)))[0])
                    worstb = max(worstb, abs(got - np.log(tru)) / LN10)
    assert worstb < 1e-3, "bin 求积在承重区间(x=E/E*<=10)偏 %.3e dex" % worstb

    print("V23 PASS  桥接恒等式 %.1e; IntE_S 角钉 %.1e / 深区回归 %.1e / S≡1 逐位; "
          "a_max=%.2f(域<=8); E_min·RMEq·sn0 双钉; bin 求积 %.1e dex(x<=10)"
          % (brid, worst, deep_rel, a_phys.max(), worstb))


def v24_dnu_convention_invariance():
    """dnu 是能量轴的纯单位：400 -> 1000 MHz 必须恰好平移 +log10(2.5)。

    2026-10-09 起生产 config 用 dnu=1000（James 2022b §3.1 / Shin 2023 Eq.A4 的
    参考带宽约定，使 log_Echar 可与 Shin 的 41.38、James 的 41.70 并列）；
    而 V5/V14/V18 等测试仍显式传 dnu=400 并配 400 口径的能量字面量（内部自洽）。
    本钉保证两种约定之间的换算严格是 +0.39794，防止有人只改 dnu 不改先验/真值。
    """
    cfg = load_config(str(HERE / "config_mini.json"))
    cos, dis, er = build_models(cfg)
    DE = np.log10(2.5)
    z = np.array([0.2, 1.0, 3.0])
    flu = np.array([1.0, 5.0, 20.0])
    r = cos.Energy(z, flu=flu, dnu=1000.) / cos.Energy(z, flu=flu, dnu=400.)
    # 浮点里 (x*1000)/(x*400) 不逐位等于 2.5，用 1e-14 而非 atol=0
    assert np.allclose(r, 2.5, rtol=1e-14, atol=0), r
    lf, lw = np.log10(flu / 3.0), np.log10(3.0)
    dme = np.array([300., 700., 1500.])
    a = dis.log_distr_efdmw(400., lf, dme, lw, -1.5, 43.0, 38.0, 0.4, 0.3,
                            np.log(33.), 1.25)
    b = dis.log_distr_efdmw(1000., lf, dme, lw, -1.5, 43.0 + DE, 38.0 + DE, 0.4, 0.3,
                            np.log(33.), 1.25)
    assert np.max(np.abs(a - b)) < 1e-9, np.max(np.abs(a - b))
    # 生产约定必须是 1000，否则上面这条换算在真实运行里无人使用
    assert cfg["analysis"]["dnu"] == 1000.0, cfg["analysis"]["dnu"]
    print("V24 PASS  dnu 400->1000 恰为能量轴 +%.5f 平移; config dnu=1000" % DE)


def v25_gamma_upper_bound_guard():
    """gamma 上界 0 是【数值】约束，不是物理约束。

    log_IntBeam 用 gammainc(g+1,r)-gammainc(g+1,2r) 求 [E,2E] bin；gamma>=+1 且
    r=E/E* 很小时两项都是 O(1) 而差是 O(r^(g+1))，实测 gamma=+1、r=1e-11 时真值
    ~1.5e-22 而代码返回恰好 0，被 1e-199 兜底成 log=-460 的错误似然。
    本钉：config 的 gamma 上界不得 >0，除非先把该差式换成
    spf.gamma(a)*spf.gammainc(a, r, 2*r)（广义下不完全伽马，直接算区间积分）。
    """
    cfg = load_config(str(HERE / "config_mini.json"))
    assert cfg["prior"]["gamma"][1] <= 0.0, \
        "gamma 上界 >0：必须先把 log_IntBeam 换成 spf.gamma(a)*spf.gammainc(a,r,2r)"
    ad = AstroDistribution()
    v = ad.log_IntBeam(np.array([34.0]), 1.0, 45.0, 33.0)   # r=1e-11, gamma=+1
    assert v[0] < -400.0, v
    print("V25 PASS  gamma 上界 %.1f; 缺陷复现 log_IntBeam(gamma=+1,r=1e-11)=%.0f"
          % (cfg["prior"]["gamma"][1], v[0]))


def v26_igm_sampling_table_interp():
    """抽样主表在 ln σ_DM 方向线性插值的精度（V24 的抽样侧对应物）。

    igm_delta_sample 不逐事件解 C0（太贵），而是在 96 点 ln σ 主表上插值。
    对照逐点精确表的 sd_ln，容差 2%。
    """
    ad = AstroDistribution()
    rng = np.random.default_rng(17)
    for z in (0.3, 1.0, 3.0):
        sd = float(ad.sigma_dm_igm(z))
        _, D, cdf, cum = ad._ig_tables(sd)
        w = np.diff(cdf, prepend=0.0)
        ld = np.log(D)
        m1 = (w * ld).sum()
        exact = np.sqrt((w * ld * ld).sum() - m1 * m1)
        sm = ad.igm_delta_sample(np.full(60000, z), 60000, rng)
        got = np.log(sm).std()
        assert abs(got / exact - 1) < 0.02, (z, got, exact)
    print("V26 PASS  抽样主表 ln σ 插值误差 <2% (z=0.3/1/3)")


def v27_likelihood_not_a_platform():
    """似然不得退化成 -1e30 常数平台（2026-10-09 实测事故钉）。

    事故链：igm_delta_nodes 用 np.atleast_1d 而非 ravel —— 推断端 log_distr_efdmwz
    传入的 z 是 (nz,1) 广播形，atleast_1d 原样保留 2-D，于是 enumerate 遍历出的是
    【行】(shape (1,)) 而不是标量；_ig_tables 里 float(sd) 在 numpy>=2 直接抛
    TypeError（size-1 数组不再隐式转标量）。而 nest_*_mini.lnlik 的
    `except Exception` 把它吞成 -1e30 ⇒ 似然在整个先验盒恒为同一个常数，
    MultiNest 照常"收敛"并输出一个平坦后验，日志上看不出任何异常。
    V1-V26 全绿也没拦住：它们喂的都是标量/1-D z，只有 V5 会撞上，而 V5 是
    assert 形式、撞上是抛错不是"读数不对"，容易被当成环境问题放过。
    三条钉：
      (a) 节点表必须吃 (nz,1) 广播形，返回 (nz, N_igm)，且逐行与标量 z 逐位相同；
      (b) 真实目录上的逐事件似然在 γ 先验两角+中点都全有限、非整片 -1e30，
          且三者之和互不相同（= 似然随参数变化，不是平台）；
      (c) 两个采样入口的 lnlik 不得再写 `except Exception:`，且必须有 isfinite 兜底。
    """
    cfg = load_config(str(HERE / "config_mini.json"))
    ad = build_models(cfg)[1]
    dnu = float(cfg['analysis']['dnu'])

    # (a) 广播形入口
    z2d = np.array([[0.3], [1.0], [3.0]])
    nd = ad.igm_delta_nodes(z2d)
    assert nd.shape == (3, ad.N_igm), nd.shape
    assert np.isfinite(nd).all()
    for i, zz in enumerate(z2d[:, 0]):
        assert np.array_equal(nd[i], ad.igm_delta_nodes(np.array([zz]))[0]), i

    # (b) 真实目录上 γ 先验盒两角 + 中点
    if (HERE / "data" / "filtered.fits").exists():
        lf_, D, lw = _catalog()
        sums = []
        for g in (-0.08, -2.0, -3.92):
            v = ad.log_distr_efdmw(dnu, lf_, D, lw, g, 44.0, 37.5,
                                   0.5, 0.755, 4.5, 0.977)
            assert np.isfinite(v).all(), (g, int((~np.isfinite(v)).sum()))
            assert (v > -1e29).sum() >= 0.9 * len(v), (g, int((v <= -1e29).sum()))
            sums.append(round(float(v.sum()), 6))
        assert len(set(sums)) == 3, sums
        msg_b = "γ∈{-0.08,-2,-3.92} 逐事件似然全有限且互异"
    else:
        msg_b = "目录缺失, 跳过 (b)"

    # (c) 入口的异常口径
    for name in ("nest_samp_mini.py", "nest_simu_mini.py"):
        src = (HERE / name).read_text(encoding='utf-8')
        m = re.search(r'def lnlik\(vpar\):(.*?)\ndef ', src, re.S)
        assert m, name + ': 找不到 lnlik'
        body = m.group(1)
        assert not re.search(r'except\s+(Exception|BaseException)\s*:', body), \
            name + ': lnlik 又写成 blanket except，会把接线错误吞成 -1e30 平台'
        assert 'np.isfinite(res)' in body, name + ': lnlik 缺 nan/inf 兜底'

    print("V27 PASS  似然非常数平台: 节点表吃 (nz,1) 广播形; %s; "
          "两个入口 lnlik 只吞 ArithmeticError/ValueError + isfinite 兜底" % msg_b)


def main():
    v1_mean_anchor()

    v2_window_analytic()
    v3_host_median_anchors()
    v4_quadrature()
    v5_degenerate_census()
    v6_evolution_restructure()
    v7_config_wiring()
    v8_prior_shape()
    v9_sfr_switch()
    v10_sampler_floor()
    v11_halo_window()
    v12_entry_wiring()
    v13_ppcheck_guards()
    v14_simufrb_e0_cut()
    v15_width_grid_adaptive()
    v16_plot_zoom_and_smooth()
    v17_poisson_survival()
    v18_phis_recovery()
    v19_smooth_selection()
    v20_simufrb_smooth_perf()
    v21_table_provenance()
    v22_closed_loop_convention()
    v23_selection_primitives()
    v24_dnu_convention_invariance()
    v25_gamma_upper_bound_guard()
    v26_igm_sampling_table_interp()
    v27_likelihood_not_a_platform()
    print("\nALL LOCAL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
