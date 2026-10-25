#!/usr/bin/env python3
"""逐事件红移后验 P(z | DM, F, w) 可视化

z 在推断中是隐变量（对 z 边际化），本脚本把每个事件的 P(z) 单独画出来——
这是对 MW 模型选择最敏感的可视化（DM_MW 扣多扣少直接平移 P(z)）。

参数来源（三选一, 均按 PARAM_ORDER 给满 8 个值）:
    -par    "phis gamma log_Echar log_E0 mu_w sigma_w ln_mu_host ln_sigma_host"
    -truth  <simudat_mini.sh 写的 .truth.json>   注入真值(恢复检验用它, 不必重抄参数)
    -nest   <MultiNest 输出前缀>                 读 post_equal_weights.dat 取中位数

用法（真实样本，参数取某条线后验中位数；线名 = MW 模型大写）:
    python3 pltpz.py --config config_mini.json -mw ne2025 \
        -nest ./nest_out_mini/samp/NE2025 -o ./plots_mini/samp/NE2025_pz.pdf
用法（mock 参数恢复检验，叠加注入真值 z；mock 的 DMe 本身就是河外残差，与 -mw 无关）:
    python3 pltpz.py --config config_mini.json \
        -truth ./nest_out_mini/simu/simdat_1e3.truth.json \
        -f1 ./simu_mini/simdat_1e3_200.txt \
        -o ./plots_mini/simu/pz_recovery.pdf
"""

import argparse
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.backends.backend_pdf as pdfback
import matplotlib.pyplot as plt

from frb_util import load_config, build_models, Loadfiles, PARAM_ORDER, truth_vector

# numpy 1.x/2.x 兼容: np.trapz 自 2.0 起更名为 np.trapezoid（服务器旧环境保护）
_trapz = getattr(np, 'trapezoid', None) or np.trapz


def load_params(args):
    """取 len(PARAM_ORDER) 个参数: -par > -truth(注入真值 JSON) > -nest(后验中位数)

    第 0 个分量的单位约定：链文件与 -par 用【线性】 phis（与 post_equal_weights
    第 0 列、.truth.json 的 "phis" 键一致），-truth 走 truth_vector 返回 log10。
    phis 完全不进入 P(z) 计算（只有 gamma/log_Echar/log_E0/宽度/宿主 DM 参与），
    所以这里不做任何换算猜测，只在打印时把两种写法都列出来，避免误读。
    """
    if args.par:
        v = [float(x) for x in args.par.replace(',', ' ').split()]
        if len(v) != len(PARAM_ORDER):
            raise ValueError("-par 需要 %d 个值: %s, 得到 %d" % (len(PARAM_ORDER), " ".join(PARAM_ORDER), len(v)))
        return np.array(v), 'manual'
    if args.truth:
        return truth_vector(args.truth), args.truth + ' (注入真值)'
    if args.nest:
        f = args.nest.rstrip('/') + 'post_equal_weights.dat'
        dat = np.loadtxt(f)
        if dat.ndim == 1:
            dat = dat[None, :]
        return np.median(dat[:, :len(PARAM_ORDER)], axis=0), f + ' median'
    raise ValueError("必须给 -par / -truth / -nest 之一来定参数")


def pz_stats(logL, vz):
    """单个事件的 (z_med, z16, z84)；完全无约束返回 NaN"""
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        lw = logL + np.log(vz)
    if not np.any(np.isfinite(lw) & (lw > -1e29)):
        return np.nan, np.nan, np.nan
    lw = np.where(np.isfinite(lw), lw, -1e300)
    p = np.exp(lw - np.max(lw))
    norm = _trapz(p, vz)
    if not np.isfinite(norm) or norm <= 0:
        return np.nan, np.nan, np.nan
    p = p / norm
    cdf = np.concatenate([[0.0], np.cumsum((p[:-1] + p[1:]) / 2.0 * np.diff(vz))])
    cdf = cdf / cdf[-1]
    q = np.interp([0.16, 0.5, 0.84], cdf, vz)
    return q[1], q[0], q[2]


def main():
    parser = argparse.ArgumentParser(description='Per-event redshift posterior P(z)')
    parser.add_argument('--config', action='store', dest='config_path',
                        default='config_mini.json', help='Path to config file')
    parser.add_argument('-mw', action='store', dest='mwmodel', type=str, default=None,
                        choices=['ne2001', 'ymw16', 'ne2025'],
                        help='河外 DM 取哪一列 dm_exc_<模型>（真实样本模式必填；'
                             'mock 模式(-f1) 的 DMe 已是残差，本参数不参与）')
    parser.add_argument('-par', action='store', dest='par', type=str, default=None,
                        help='8 个参数值: "phis gamma log_Echar log_E0 mu_w sigma_w ln_mu_host ln_sigma_host"')
    parser.add_argument('-truth', action='store', dest='truth', type=str, default=None,
                        help='注入真值 JSON（simudat_mini.sh 写的 .truth.json），'
                             '恢复检验时用它而不是 -par，避免在 shell 里重抄一遍参数')
    parser.add_argument('-nest', action='store', dest='nest', type=str, default=None,
                        help='MultiNest 输出前缀（读 post_equal_weights.dat 中位数）')
    parser.add_argument('-f1', action='store', dest='simu1', type=str, default=None,
                        help='mock 数据文件（叠加注入真值 z, 恢复检验模式）')
    parser.add_argument('-o', action='store', dest='output', type=str, required=True,
                        help='输出 PDF 路径')
    parser.add_argument('--nevt', action='store', dest='nevt', type=int, default=20,
                        help='P(z) 曲线抽样个数')
    args = parser.parse_args()
    t_start = time.perf_counter()

    config = load_config(args.config_path)
    dnu = config.get('analysis', {}).get('dnu', 1000.0)
    label = args.mwmodel or 'mock'

    # 数据与河外 DM
    lf = Loadfiles()
    if args.simu1:
        cat = lf.LoadSimuData(args.simu1)
        vLOGF = np.log10(cat['S'])
        vLOGW = np.log10(cat['W'])
        vDME = cat['DMe']
        z_true = cat['Z']
    else:
        if args.mwmodel is None:
            raise ValueError("真实样本模式必须给 -mw（ne2001 / ymw16 / ne2025 三条线之一）")
        vF, vW, vDM_obs, vDM_ne2001, vDM_ymw16, vDM_ne2025, vSVY = lf.LoadFitsCatalog(
            config['data']['catalog_path'])
        # 与 nest_samp_mini 同式: flux = fluence / width [Jy]，宽度取 ms。
        # 这两行历史上漏掉过，导致真实样本模式一进来就 UnboundLocalError（V28 钉住）。
        vLOGF = np.log10(vF / vW)
        vLOGW = np.log10(vW)
        vDME = {'ne2001': vDM_ne2001, 'ymw16': vDM_ymw16, 'ne2025': vDM_ne2025}[args.mwmodel]
        if vDME is None:
            raise ValueError(f"dm_exc_{args.mwmodel} 列不可用（ne2025 需先跑 preprocess_catalog.py）")
        if not np.all(np.isfinite(vDME)):
            raise ValueError(f"dm_exc_{args.mwmodel} 含 {int(np.sum(~np.isfinite(vDME)))} 个非有限值")
        z_true = None
    print(f'[pltpz] N = {len(vLOGF)} (line={label}), params from ', end='')

    vpar, src = load_params(args)
    # 第 0 维口径不统一（见 load_params docstring）：truth_vector 返回的已是 log10(phis)，
    # -par / -nest 给的是线性 phis。旧写法一律再取一次 log10 ⇒ -truth 模式把 3.0 当线性值
    # 打成 log10(3)=0.477，log10(phi0)<=0 时还直接 nan。phis 本身不进 P(z)，只影响打印。
    phi0_col = float(vpar[0])
    if args.truth:
        log_phi0, phis_lin = phi0_col, 10.0 ** phi0_col
    else:
        phis_lin = phi0_col
        log_phi0 = np.log10(phis_lin) if phis_lin > 0 else float('nan')
    gamma, logEs, logE0, mu, sigma, ln_mu_host, ln_sigma_host = vpar[1:]
    print(f'{src}: phi0={phis_lin:.4g} (log10={log_phi0:.3f}) '
          f'gamma={gamma:.3f} log_Echar={logEs:.2f} '
          f'log_E0={logE0:.2f} mu_w={mu:.3f} sigma_w={sigma:.3f} '
          f'ln_mu_host={ln_mu_host:.3f} ln_sigma_host={ln_sigma_host:.3f}')

    # z 网格与逐事件似然网格 (nz, N)
    # 必须走 build_models：过去这里只用裸 AstroDistribution()，于是系统误差线
    # 画出来的 P(z) 仍按默认 dm_model/宇宙学，与同一条线的 MultiNest 后验不是同一个模型。
    _, dis, _ = build_models(config)
    nz = 400
    vz = np.exp(np.arange(np.log(dis.Zmin), np.log(dis.Zmax),
                          (np.log(dis.Zmax) - np.log(dis.Zmin)) / nz))
    grid = dis.log_distr_efdmw_grid(dnu, vLOGF, vDME, vLOGW, vz,
                                    gamma, logEs, logE0, mu, sigma,
                                    ln_mu_host, ln_sigma_host)

    z_med = np.full(grid.shape[1], np.nan)
    z_lo = np.full_like(z_med, np.nan)
    z_hi = np.full_like(z_med, np.nan)
    for i in range(grid.shape[1]):
        z_med[i], z_lo[i], z_hi[i] = pz_stats(grid[:, i], vz)
    n_bad = int(np.sum(~np.isfinite(z_med)))
    print(f'[pltpz] 约束有效 {grid.shape[1] - n_bad}/{grid.shape[1]} 个事件'
          + (f'（{n_bad} 个无约束, 已剔除）' if n_bad else ''))

    ok = np.isfinite(z_med)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with pdfback.PdfPages(args.output) as pp:
        # 页 1: 抽样 P(z) 曲线 —— 颜色按该事件 DM_EG 编码。连续色标在细线上
        # 相邻难分(实测 20 条 viridis 中段全挤在蓝绿), 改成 6 档离散色
        # (turbo 均匀取色, 蓝->红 = DM 小->大), 图例直接给每档 log10 DM 区间,
        # 颜色既有含义又肉眼可分。
        # 抽样展示用【确定性】选取: 按 DM_EG 排序, 掐头去尾各 5%, 中央 90% 内
        # 等距取 nevt 条。不用随机数(随机抽样让同一份数据每次渲染出不同曲线,
        # 无法对账); 掐极端是因为 DM 两端事件的 P(z) 常是尖锐单峰(低 DM)或
        # 多峰宽平台(高 DM), 会把 y 轴撑爆压扁其余曲线(实测 10x 峰)。
        ok_idx = np.flatnonzero(ok)
        order = np.argsort(vDME[ok])
        ok_sorted = ok_idx[order]
        lo_i, hi_i = int(0.05 * len(ok_sorted)), int(0.95 * len(ok_sorted))
        ok_core = ok_sorted[max(lo_i, 0):max(hi_i, lo_i + 1)]
        n_pick = min(args.nevt, len(ok_core))
        sel = np.unique(np.round(np.linspace(0, len(ok_core) - 1, n_pick)).astype(int))
        pick = ok_core[sel]
        fig, ax = plt.subplots(figsize=(9.0, 6))
        dlog_pick = np.log10(np.maximum(vDME[pick], 1e-6))
        lo_, hi_ = float(dlog_pick.min()), float(dlog_pick.max())
        if not hi_ > lo_:
            hi_ = lo_ + 1e-6
        edges_c = np.linspace(lo_, hi_, 7)
        cols6 = [plt.get_cmap('turbo')(x) for x in np.linspace(0.05, 0.92, 6)]
        for i in pick:
            k6 = int(np.clip(np.searchsorted(edges_c, np.log10(max(vDME[i], 1e-6)),
                                             side='right') - 1, 0, 5))
            lw = grid[:, i] + np.log(vz)
            lw = np.where(np.isfinite(lw), lw, -1e300)
            p = np.exp(lw - np.max(lw))
            p /= _trapz(p, vz)
            ax.plot(vz, p, lw=1.0, alpha=0.85, color=cols6[k6])
        handles6 = [plt.Line2D([], [], color=c, lw=3,
                               label='[%.2f, %.2f)' % (edges_c[k], edges_c[k + 1]))
                    for k, c in enumerate(cols6)]
        ax.legend(handles=handles6, title='DM$_{EG}$ [pc cm$^{-3}$] (log$_{10}$)',
                  loc='upper right', fontsize=9, title_fontsize=10, framealpha=0.9)
        ax.set_xlabel('redshift z')
        ax.set_ylabel('P(z | DM, F, w)')
        ax.set_title(f'Per-event P(z), N={len(pick)}/{len(ok)} '
                     f'evenly spaced in central 90% DM (line={label})')
        fig.tight_layout()
        pp.savefig(fig)
        plt.close(fig)

        # 页 2: z_med ± 68% vs DM_EG
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(vDME[ok], z_med[ok],
                    yerr=[z_med[ok] - z_lo[ok], z_hi[ok] - z_med[ok]],
                    fmt='.', ms=4, elinewidth=0.7, alpha=0.5, color='C0')
        ax.set_xlabel('DM$_{EG}$ [pc cm$^{-3}$]')
        ax.set_ylabel('z (median, 68% CI)')
        ax.set_title('Per-event redshift posterior vs extragalactic DM')
        fig.tight_layout()
        pp.savefig(fig)
        plt.close(fig)

        # 页 3（mock 模式）: 恢复检验
        if z_true is not None:
            fig, axes = plt.subplots(1, 2, figsize=(13, 6))
            ax = axes[0]
            lim = (0, max(1.05 * np.nanmax(z_true), np.nanmax(z_hi[ok])))
            ax.plot(lim, lim, 'k--', lw=1.0, label='y = x')
            ax.errorbar(z_true[ok], z_med[ok],
                        yerr=[z_med[ok] - z_lo[ok], z_hi[ok] - z_med[ok]],
                        fmt='.', ms=4, elinewidth=0.7, alpha=0.5, color='C2')
            # 注入 z 每 0.5 一箱: 黑方块连线 = 每箱 med(z_med), 文本框/stdout =
            # 每箱【逐事件配对误差】的中位数 med(z_med - z_true)。用配对定义而
            # 不是 med(z_med)-箱心: 低 z 端 z_med>=0 边界会把后者抬高(实测
            # +0.18 vs +0.06), 配对误差才是逐事件恢复偏置的本征量。
            zt = np.asarray(z_true)[ok]
            zm = np.asarray(z_med)[ok]
            edges = np.arange(0.0, float(np.nanmax(zt)) + 0.5, 0.5)
            if len(edges) >= 2:
                zc = 0.5 * (edges[:-1] + edges[1:])
                med = np.full(len(zc), np.nan)
                bias = np.full(len(zc), np.nan)
                cnt = np.zeros(len(zc), dtype=int)
                for k in range(len(zc)):
                    m_ = (zt >= edges[k]) & (zt < edges[k + 1])
                    cnt[k] = int(m_.sum())
                    if cnt[k]:
                        med[k] = float(np.median(zm[m_]))
                        bias[k] = float(np.median(zm[m_] - zt[m_]))
                ax.plot(zc, med, 'ks-', ms=5, lw=1.0, label='binned median ($\\Delta z$=0.5)')
                print('[pltpz] 注入 z 分箱恢复 (箱心 / n / med(z_med) / med(z_med-z_true)):')
                for k in range(len(zc)):
                    if cnt[k]:
                        print('  %.2f  %4d  %8.4f  %+8.4f'
                              % (zc[k], cnt[k], med[k], bias[k]))
                # 面板内文本框: 每箱配对误差中位数(带 N), 偏置走势图上直接可读
                box = ['median bias per z bin']
                for k in range(len(zc)):
                    if cnt[k]:
                        box.append('%.1f-%.1f: %+0.2f (N=%d)'
                                   % (edges[k], edges[k + 1], bias[k], cnt[k]))
                ax.text(0.98, 0.03, '\n'.join(box), transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=9, family='monospace',
                        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.6', alpha=0.9))
            ax.set_xlabel('injected z (truth)')
            ax.set_ylabel('z (median, 68% CI)')
            ax.set_title('Per-event z recovery (mock)')
            ax.legend(fontsize=9)
            ax = axes[1]
            resid = (z_med[ok] - z_true[ok]) / np.maximum(z_hi[ok] - z_lo[ok], 1e-6)
            ax.hist(resid, bins=40, color='C2', alpha=0.7)
            ax.axvline(0, color='k', ls='--', lw=1.0)
            ax.set_xlabel('(z_med - z_true) / 68% width')
            ax.set_ylabel('Count')
            ax.set_title(f'standardized residual (med={np.median(resid):+.2f})')
            fig.tight_layout()
            pp.savefig(fig)
            plt.close(fig)
    print(f'[pltpz] 输出: {args.output}')
    print(f"[pltpz.py 总耗时 {time.perf_counter()-t_start:.1f}s]")


if __name__ == '__main__':
    main()
