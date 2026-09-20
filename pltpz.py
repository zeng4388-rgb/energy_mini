#!/usr/bin/env python3
"""逐事件红移后验 P(z | DM, F, w) 可视化

z 在推断中是隐变量（对 z 边际化），本脚本把每个事件的 P(z) 单独画出来——
这是对 MW 模型选择最敏感的可视化（DM_MW 扣多扣少直接平移 P(z)）。

参数来源（二选一）:
    -par "phis alpha logEs logE0 mu sigma"   手动给定（mock 用注入真值）
    -nest <MultiNest 输出前缀>                读 post_equal_weights.dat 取中位数

用法（真实样本，参数取某条线后验中位数）:
    python3 pltpz.py --config config_mini.json -g ALG_YMW16 -mw ne2025 \
        -nest ./nest_out_mini/samp/ALG_NE2025 -o ./plots_mini/samp/ALG_NE2025_pz.pdf
用法（mock 参数恢复检验，叠加注入真值 z）:
    python3 pltpz.py --config config_mini.json -g ALG_YMW16 -mw ymw16 \
        -par "1e3 -1.5 43 38 0.4 0.3" -f1 ./simu_mini/simdat_1e3_200.txt \
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

from frb_util import load_config, AstroDistribution, Loadfiles

# numpy 1.x/2.x 兼容: np.trapz 自 2.0 起更名为 np.trapezoid（服务器旧环境保护）
_trapz = getattr(np, 'trapezoid', None) or np.trapz


def load_params(args):
    """取 6 参数: -par 优先, 否则 -nest 的 post_equal_weights.dat 中位数"""
    if args.par:
        v = [float(x) for x in args.par.replace(',', ' ').split()]
        if len(v) != 6:
            raise ValueError(f"-par 需要 6 个值 phis alpha logEs logE0 mu sigma, 得到 {len(v)}")
        return np.array(v), 'manual'
    if args.nest:
        f = args.nest.rstrip('/') + 'post_equal_weights.dat'
        dat = np.loadtxt(f)
        if dat.ndim == 1:
            dat = dat[None, :]
        return np.median(dat[:, :6], axis=0), f + ' median'
    raise ValueError("必须给 -par 或 -nest 之一来定参数")


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
    parser.add_argument('-g', action='store', dest='fgt', type=str, default='ALG_YMW16',
                        help='Host galaxy type (ETG/LTG_*/ALG_*)')
    parser.add_argument('-mw', action='store', dest='mwmodel', type=str, default=None,
                        help='MW model: ne2001 / ymw16 / ne2025 (default: 按 -g 后缀推断)')
    parser.add_argument('-par', action='store', dest='par', type=str, default=None,
                        help='"phis alpha logEs logE0 mu sigma"')
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
    dnu = config.get('analysis', {}).get('dnu', 400.0)
    fgt = args.fgt
    if fgt and fgt.find('ETG') >= 0:
        fgt = 'ETG'

    # 数据与河外 DM
    lf = Loadfiles()
    if args.simu1:
        cat = lf.LoadSimuData(args.simu1)
        vLOGF = np.log10(cat['S'])
        vLOGW = np.log10(cat['W'])
        vDME = cat['DMe']
        z_true = cat['Z']
    else:
        vF, vW, vDM_obs, vDM_ne2001, vDM_ymw16, vDM_ne2025, vSVY = lf.LoadFitsCatalog(
            config['data']['catalog_path'])
        mwmodel = args.mwmodel
        if mwmodel is None:
            mwmodel = 'ne2001' if (fgt and fgt.find('NE2001') >= 0) else 'ymw16'
        mwmodel = mwmodel.lower()
        vDME = {'ne2001': vDM_ne2001, 'ymw16': vDM_ymw16, 'ne2025': vDM_ne2025}[mwmodel]
        if vDME is None:
            raise ValueError(f"dm_exc_{mwmodel} 列不可用（ne2025 需先跑 preprocess_catalog.py）")
        z_true = None
    print(f'[pltpz] N = {len(vLOGF)}, params from ', end='')

    vpar, src = load_params(args)
    phis, alpha, logEs, logE0, mu, sigma = vpar
    print(f'{src}: phis={phis:.3g} alpha={alpha:.3f} logEs={logEs:.2f} '
          f'logE0={logE0:.2f} mu={mu:.3f} sigma={sigma:.3f}')

    # z 网格与逐事件似然网格 (nz, N)
    dis = AstroDistribution()
    nz = 400
    vz = np.exp(np.arange(np.log(dis.Zmin), np.log(dis.Zmax),
                          (np.log(dis.Zmax) - np.log(dis.Zmin)) / nz))
    grid = dis.log_distr_efdmw_grid(dnu, vLOGF, vDME, vLOGW, vz,
                                    alpha, logEs, logE0, mu, sigma, gtype=fgt)

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
        # 页 1: 抽样 P(z) 曲线
        rng = np.random.default_rng(42)
        ok_idx = np.flatnonzero(ok)
        pick = rng.choice(ok_idx, size=min(args.nevt, len(ok_idx)), replace=False)
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in pick:
            lw = grid[:, i] + np.log(vz)
            lw = np.where(np.isfinite(lw), lw, -1e300)
            p = np.exp(lw - np.max(lw))
            p /= _trapz(p, vz)
            ax.plot(vz, p, lw=0.8, alpha=0.6)
        ax.set_xlabel('redshift z')
        ax.set_ylabel('P(z | DM, F, w)')
        ax.set_title(f'Per-event P(z), N={len(pick)} sampled (host={args.fgt})')
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
