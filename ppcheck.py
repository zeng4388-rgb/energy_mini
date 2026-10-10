#!/usr/bin/env python3
"""后验预测检验 (PPC)。对应《代码改进方案.md》C10。

流程: 等权后验抽 N_draws 组参数 -> 每组用 simufrb 生成 ns = N_real 个 mock 事件
(走完整探测选择函数) -> 对比真实数据与 mock 的统计量分布 -> 报真实值在 mock 中的
分位(tail probability), 落在 2.5%/97.5% 之外标红; 另出 spaghetti 图。

三条硬规则(方案 C10):
  1. mock 事件数 = 真实样本数, 不是 300;
  2. 判定用统计量分位表, 不只看 spaghetti 图;
  3. 每个进程显式 default_rng(seed), 禁止依赖全局 np.random 种子。

用法:
  python3 ppcheck.py --nest ./nest_out_mini/samp/ALG_NE2025 -mw ne2025 \
      --draws 200 -j 4 -o plots_mini/ppc_ALG_NE2025.pdf
"""
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.stats import spearmanr

from frb_util import (TELESCOPES, PARAM_ORDER, Loadfiles, build_models, load_config,
                      read_selection_survival)
from chime_selection import resolve_selection
import simufrb
from simufrb import Simu_FRBs_Eiso

TEL = TELESCOPES['CHIME']


def _stats(flu, w, dme):
    """一组统计量: 抓不同类别的模型失配。"""
    ld = np.log10(np.maximum(dme, 1e-6))
    lf = np.log10(np.maximum(flu, 1e-12))
    q = np.percentile(dme, [10, 25, 50, 75, 90])
    m, s = ld.mean(), ld.std()
    skew = np.mean(((ld - m) / s) ** 3)
    xs = np.sort(lf)
    sel = xs <= np.percentile(xs, 30)
    slope = np.polyfit(xs[sel], np.arange(len(xs))[sel] / len(xs), 1)[0] if sel.sum() > 10 else np.nan
    rho = spearmanr(dme, lf).statistic
    # 分宽度层的流量比：平滑选择函数的 F_50(w) 只由宽度控制，这两个量是它唯一的
    # 直接检验。此前 _stats 收了 w 却从不用，而 round-1 最大的已知偏差（冻结 τ、
    # 以及对窄/宽尾的相对选错）恰好全在宽度维上，现有统计量对它完全失明。
    # 分层边界不是随手取的：目录 width 列是 bc_width（preprocess_catalog.py:102），
    # 它是【搜索箱宽】，量化到 0.983 ms 采样的整数倍 —— 实测 1688 样本只有 42 个
    # 离散值，49.4% 挤在 1.966 ms（p5–p50 全为它；p1=0.983、p75=4.915，2026-10-05
    # 复核修正，旧注释"p1..p75 全 1.966/75%"有误）。边界落在空箱（如 <0.983 ms）
    # 会选出空层并把本统计量变成 NaN（进而在秩检验里报成 0% 假阳性）。边界必须
    # 落在 bc_width 的离散支撑内：≤2.0 ms（{0.983,1.966} 两箱，实测 51.4%）
    # 与 ≥5.0 ms（5.898 ms 起的箱，实测 19.4%）。
    narrow = w <= 2.0
    wide = w >= 5.0
    fratio = (np.median(flu[wide]) / np.median(flu[narrow])
              if narrow.sum() > 10 and wide.sum() > 10 else np.nan)
    wtail = np.percentile(w, 90) / np.percentile(w, 50) if len(w) > 10 else np.nan
    return np.array([*q, skew, slope, rho, np.percentile(flu, 99), fratio, wtail])


NAMES = ["DM_EG p10", "p25", "p50", "p75", "p90",
         "skew(log DM_EG)", "fluence 低端斜率", "Spearman(DM_EG,logF)", "fluence p99",
         "medF(w>=5ms)/medF(w<=2ms)", "p90(w)/p50(w)"]


def _one_draw(task):
    seed, params, ns, fov, survival, sel_mode, sel_k = task
    rng = np.random.default_rng(seed)          # 规则 3: 显式种子
    np.random.seed(seed % (2 ** 32 - 1))       # simufrb 内部仍用全局接口, 逐进程隔离
    phis, gamma, logEs, logE0, mu_w, sig_w, mu_h, sig_h = params
    # post_equal_weights 第 0 列已是先验变换后的【线性】 phis（prior_transform 对
    # log_phi0 返回 10**(lo+u*(hi-lo))；pltpost 对它取 log10 也反证）。
    # 原来写 10.0**phis 会溢出成 inf → lamda=inf → 到达时间列被静默写成全 0。
    res = Simu_FRBs_Eiso(phis, gamma, logEs, logE0, mu_w, sig_w,
                         dnu=TEL['dnu'], ns=ns, fov=fov, npol=TEL['npol'],
                         g=TEL['g'], tsys=TEL['tsys'], bw=TEL['bw'], sn0=TEL['sn0'],
                         ln_mu_host=mu_h, ln_sigma_host=sig_h,
                         poisson_survival=survival,
                         sel_mode=sel_mode, sel_k=sel_k)
    if len(res) == 0:
        return None
    if len(res) < ns:
        # 规则 1 要求 mock 事件数 = 真实样本数; 被 max_iter 截断会让各统计量的
        # 有效样本量在不同 draw 之间漂移, 分位数就不严格可比了。
        print(f"[ppc][warn] seed={seed} 只生成 {len(res)}/{ns} 个事件(检测率过低被截断)")
    flu = res[:, 0] * res[:, 1]
    keep = rng.choice(len(res), size=min(2000, len(res)), replace=False)
    return _stats(flu, res[:, 1], res[:, 3]), res[keep, 3], flu[keep]


def _init_worker(config_path):
    """Pool worker 初始化：fork 下其实已继承，但 Windows/spawn 下子进程会重新
    import simufrb 拿到【默认】模型参数（晕窗/σ_ln/宇宙学全丢），
    所以必须显式在每个 worker 里按 config 重新接线。"""
    simufrb.cos, simufrb.dis, simufrb.er = build_models(load_config(config_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nest", required=True, help="MultiNest 输出前缀")
    ap.add_argument("-mw", default="ne2001", choices=["ne2001", "ymw16", "ne2025"])
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("-j", type=int, default=1)
    ap.add_argument("-o", required=True, help="输出 PDF")
    ap.add_argument("--config", default="config_mini.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # 整体替换 simufrb 的三个模块级实例：mock 与拟合必须共用同一宇宙学 + dm_model。
    # 只调 apply_dm_model 会漏掉宇宙学，且原先还有一行
    # apply 到临时实例上（死码）。
    simufrb.cos, simufrb.dis, simufrb.er = build_models(cfg)

    post = np.loadtxt(args.nest.rstrip("/") + "post_equal_weights.dat")
    params = post[:, :len(PARAM_ORDER)]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(params), size=min(args.draws, len(params)), replace=False)

    d = fits.getdata(cfg["data"]["catalog_path"])
    col = {"ne2001": "dm_exc_ne2001", "ymw16": "dm_exc_ymw16",
           "ne2025": "dm_exc_ne2025"}[args.mw]
    dme_real = np.array(d[col], float)
    flu_real = np.array(d["fluence"], float)
    w_real = np.array(d["width"], float)
    ns = len(dme_real)                          # 规则 1: 与真实样本同数
    # 走 Loadfiles.LoadSvyInfo（按列名取, 与推断/模拟端同一入口）。
    # 原先手工 np.loadtxt + skiprows=1 有两个错：tel_svy.txt 只有一行数据时退化
    # 成 1-D 数组使 svy[:,0] 抛 IndexError；且列号 4 是 Tsys 而不是 FOV。
    svy = Loadfiles().LoadSvyInfo(cfg["data"]["survey_info_path"])
    fov = float(svy["FOV"][np.where(np.array(svy["SURVEY"]) == "CHIME")[0][0]])
    print(f"[ppc] CHIME FOV = {fov} deg^2")
    # mock 与真实样本必须同一选择定义：模式只来自 config/env，不接受隐式默认
    sel_mode, sel_k = resolve_selection(cfg)
    print(f"[ppc] selection mode = {sel_mode}, k_offset = {sel_k}")
    # 存活率分母随模式走（step→NREF10，smooth→NSNR12），与 nest_samp 同一定义
    survival = read_selection_survival(cfg["data"]["catalog_path"], mode=sel_mode)

    real = _stats(flu_real, w_real, dme_real)
    tasks = [(1000 + i, params[i], ns, fov, survival, sel_mode, sel_k) for i in idx]
    if args.j > 1:
        with mp.Pool(args.j, initializer=_init_worker, initargs=(args.config,)) as pool:
            out = pool.map(_one_draw, tasks)
    else:
        out = [_one_draw(t) for t in tasks]
    good = [o for o in out if o is not None]
    # 全失败必须当场报错：否则下面 mock[:, k] 在一个空数组上崩成一堆难读的 IndexError，
    # 看上去像统计量写错而不是"mock 一个都没生成出来"。
    if not good:
        raise RuntimeError("所有 mock 组都失败了(0/%d)——检查参数范围/阈值是否让检测率为零"
                           % len(tasks))
    if len(good) < len(tasks):
        print(f"[ppc][warn] {len(tasks) - len(good)}/{len(tasks)} 组 mock 为空, 已剔除")
    mock = np.array([g[0] for g in good])
    mock_dm = [g[1] for g in good]
    mock_flu = [g[2] for g in good]
    print(f"[ppc] 有效 mock {len(mock)}/{len(tasks)} 组, ns={ns}")

    print("\n统计量            真实值        mock 中位     真实值分位   判定")
    n_bad = 0
    for k, name in enumerate(NAMES):
        col = mock[:, k]
        fin = np.isfinite(col) & np.isfinite(real[k])
        if not fin.any():
            # NaN 参与比较恒为 False，会把秩静默压成 0% 假阳性——全 NaN 必须
            # 显式标 NaN 并计失败（如分宽度统计量在全部 mock 组遇到空层）。
            print(f"{name:18s} {real[k]:11.3g} {'NaN':>12s} {'NaN':>10s}   **NaN**")
            n_bad += 1
            continue
        if fin.sum() < len(col):
            print(f"[ppc][warn] {name}: {len(col) - fin.sum()} 组 mock 为 NaN，"
                  f"秩只在有限值上计算")
        rank = 100.0 * np.mean(col[fin] <= real[k])
        bad = rank < 2.5 or rank > 97.5
        n_bad += bad
        print(f"{name:18s} {real[k]:11.3g} {np.median(col[fin]):12.3g} "
              f"{rank:10.1f}%   {'**FAIL**' if bad else 'ok'}")
    print(f"\n[ppc] 越界统计量 {n_bad}/{len(NAMES)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for k, (vals, mocks, lab) in enumerate([(dme_real, mock_dm, "DM_EG"),
                                            (flu_real, mock_flu, "fluence")]):
        ax = axes[k]
        for m in mocks[:100]:                       # spaghetti: 半透明灰线
            xs = np.sort(m)
            ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), color="0.6",
                    lw=0.5, alpha=0.25)
        xs = np.sort(vals)                          # 真实数据: 粗黑线
        ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), "k-", lw=2, label="real")
        ax.set_xscale("log")
        ax.set_xlabel(lab)
        ax.set_ylabel("CDF")
        ax.legend(loc="lower right")
    fig.suptitle(f"PPC: {args.nest}  (统计量分位表见 stdout; {n_bad} 项越界)")
    Path(args.o).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.o)
    print(f"[ppc] 图: {args.o}")


if __name__ == "__main__":
    main()
