#!/usr/bin/env python3
"""数据空间分布图：读预处理后的 filtered.fits，出 MW 模型对比与样本概览

纯数据空间检查（零推断依赖），每次改筛选条件或换 MW 模型后必看：
  1. 三列 dm_exc (ne2001/ymw16/ne2025) 同图叠加 + 各列 median 与负值数
     —— ne2025 是否"扣过头"一眼可见;
  2. 反推 DM_MW 分布（= dm_obs - dm_exc），对照 Shin 2023 的
     disk 中位 ~50 pc/cm3 与固定总值 80 pc/cm3 参考线;
  3. dm_obs / fluence / width 样本分布;
  4. dm_exc 模型间散点（偏离 y=x 的即为模型分歧事件）。

用法:
    python3 pltdatadist.py --config config_mini.json
    python3 pltdatadist.py -i data/filtered.fits -o plots_mini/datadist/test.pdf
"""

import argparse
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from frb_util import load_config, Loadfiles


def main():
    parser = argparse.ArgumentParser(description='Preprocessed data distribution plots')
    parser.add_argument('--config', action='store', dest='config_path',
                        default='config_mini.json', help='Path to config file')
    parser.add_argument('-i', '--input', dest='input', default=None,
                        help='输入 filtered FITS（默认取 config 的 data.catalog_path）')
    parser.add_argument('-o', '--output', dest='output', default=None,
                        help='输出 PDF（默认 plots_mini/datadist/datadist.pdf）')
    args = parser.parse_args()
    t_start = time.perf_counter()

    config = load_config(args.config_path)
    fname = args.input or config['data']['catalog_path']
    output = args.output or str(Path(config['output']['plot_dir']) / 'datadist' / 'datadist.pdf')

    lf = Loadfiles()
    vF, vW, vDM_obs, vDM_ne2001, vDM_ymw16, vDM_ne2025, vSVY = lf.LoadFitsCatalog(fname)
    print(f'[datadist] 加载 {len(vF)} 个事件: {fname}')

    models = [('NE2001', vDM_ne2001, 'C0'), ('YMW16', vDM_ymw16, 'C1')]
    if vDM_ne2025 is not None:
        models.append(('NE2025', vDM_ne2025, 'C2'))
    else:
        print('[datadist] 无 dm_exc_ne2025 列，相关面板跳过（重跑 preprocess 生成）')

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))

    # (0,0) 三列 dm_exc 叠加
    ax = axes[0, 0]
    bins = np.linspace(0, 1500, 75)
    for name, v, c in models:
        ax.hist(v, bins=bins, histtype='step', lw=1.8, density=True, color=c,
                label=f'{name}: med={np.median(v):.0f}, neg={int(np.sum(v < 0))}')
    ax.set_xlabel('DM_exc [pc cm$^{-3}$]')
    ax.set_ylabel('PDF')
    ax.set_title('Extragalactic DM per MW model')
    ax.legend(fontsize=9)

    # (0,1) 反推 DM_MW，对照 Shin 2023 参考
    ax = axes[0, 1]
    bins = np.linspace(0, 150, 75)
    for name, v, c in models:
        dm_mw = vDM_obs - v
        ax.hist(dm_mw, bins=bins, histtype='step', lw=1.8, density=True, color=c,
                label=f'{name}: med={np.median(dm_mw):.0f}')
    ax.axvline(50, color='k', ls=':', lw=1.2)
    ax.axvline(80, color='k', ls='--', lw=1.2)
    ax.text(51, ax.get_ylim()[1] * 0.92, 'Shin disk med=50', fontsize=8, rotation=90)
    ax.text(81, ax.get_ylim()[1] * 0.92, 'Shin fixed total=80', fontsize=8, rotation=90)
    ax.set_xlabel('DM_MW = DM_obs - DM_exc [pc cm$^{-3}$]')
    ax.set_ylabel('PDF')
    ax.set_title('Inferred Milky Way DM')
    ax.legend(fontsize=9)

    # (0,2) dm_obs
    ax = axes[0, 2]
    ax.hist(vDM_obs, bins=np.linspace(0, 2500, 80), color='C3', alpha=0.7)
    ax.axvline(np.median(vDM_obs), color='k', ls='--', lw=1.2,
               label=f'med={np.median(vDM_obs):.0f}')
    ax.set_xlabel('DM_obs [pc cm$^{-3}$]')
    ax.set_ylabel('Count')
    ax.set_title(f'Observed DM (N={len(vDM_obs)})')
    ax.legend(fontsize=9)

    # (1,0) fluence
    ax = axes[1, 0]
    ax.hist(np.log10(vF), bins=60, color='C4', alpha=0.7)
    ax.axvline(np.log10(np.median(vF)), color='k', ls='--', lw=1.2,
               label=f'med={np.median(vF):.2f} Jy ms')
    ax.set_xlabel('log$_{10}$ Fluence [Jy ms]')
    ax.set_ylabel('Count')
    ax.set_title('Fluence')
    ax.legend(fontsize=9)

    # (1,1) width
    ax = axes[1, 1]
    ax.hist(np.log10(vW), bins=60, color='C5', alpha=0.7)
    ax.axvline(np.log10(np.median(vW)), color='k', ls='--', lw=1.2,
               label=f'med={np.median(vW):.2f} ms')
    ax.set_xlabel('log$_{10}$ Width [ms]')
    ax.set_ylabel('Count')
    ax.set_title('Pulse width')
    ax.legend(fontsize=9)

    # (1,2) 模型间 dm_exc 散点（有 ne2025 时比 ne2025 vs ymw16）
    ax = axes[1, 2]
    if vDM_ne2025 is not None:
        x, y, xl, yl = vDM_ymw16, vDM_ne2025, 'DM_exc YMW16', 'DM_exc NE2025'
    else:
        x, y, xl, yl = vDM_ne2001, vDM_ymw16, 'DM_exc NE2001', 'DM_exc YMW16'
    lim = (min(np.min(x), np.min(y)) - 20, np.percentile(np.maximum(x, y), 99.5))
    ax.plot(lim, lim, 'k--', lw=1.0, label='y = x')
    ax.scatter(x, y, s=6, alpha=0.35, color='C6')
    n_off = int(np.sum(np.abs(x - y) > 0.2 * np.maximum(np.abs(x), np.abs(y))))
    ax.set_xlabel(xl + ' [pc cm$^{-3}$]')
    ax.set_ylabel(yl + ' [pc cm$^{-3}$]')
    ax.set_title(f'Model spread (|Δ|>20%: {n_off})')
    ax.legend(fontsize=9)

    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(f'[datadist] 输出: {output}')
    print(f"[pltdatadist.py 总耗时 {time.perf_counter()-t_start:.1f}s]")


if __name__ == '__main__':
    main()
