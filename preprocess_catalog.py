#!/usr/bin/env python3
"""FRB Catalog 预处理脚本

读取原始 FITS catalog，按筛选条件过滤，计算 NE2025 河外 DM，输出干净数据。
分析代码统一读取预处理后的数据，不直接接触原始 catalog。

输出只保留下游计算所需列: name / fluence / width / dm_obs /
dm_exc_ne2001 / dm_exc_ymw16 / dm_exc_ne2025。
中间列(snr_fitb/scat_time/注释列/gl/gb)仅用于筛选与 NE2025 计算，用完即删。

dm_exc_ne2025 由官方 mwprop 包计算(Ocker & Cordes 2026, NE2025 模型)：
    mwprop 调用封装在 frb_util.py (ne2025_dm_mw / ne2025_dm_mw_batch /
    ne2025_verify_plateau，内部延迟 import，推断端不受影响)；
    积分距离从 config 的 analysis.mwprop_dist_kpc 读取(默认 100 kpc)。
    dm_exc_ne2025 = dm_obs - DM_MW,NE2025(gl, gb, dist_kpc)

用法:
    python preprocess_catalog.py
    python preprocess_catalog.py --input data/chimefrbcat2.fits --output data/filtered
    python preprocess_catalog.py --verify-dist      # 先做 50/100/150 kpc 平台验证
    python preprocess_catalog.py --skip-ne2025      # 未装 mwprop 时跳过 NE2025 列
"""

import numpy as np
import argparse
import time
from pathlib import Path
from astropy.io import fits

from frb_util import load_config, ne2025_dm_mw_batch, ne2025_verify_plateau


def load_raw_catalog(fname):
    """读取原始 FITS catalog"""
    with fits.open(fname) as hdul:
        data = hdul[1].data
        col_names = data.columns.names
        print(f"原始数据: {len(data)} 个 burst")
        print(f"列名: {col_names}")
        return data, col_names


def filter_minimum(data):
    """最小筛选：只排除重复暴和无fluence的burst

    筛选标准:
    1. repeater_name 为空 → one-off FRB
    2. fluence 不为 NaN → 有有效通量测量
    """
    n_raw = len(data)

    # 排除重复暴
    rep = np.array(data['repeater_name'])
    is_oneoff = np.array([str(r).strip() == '' for r in rep])
    n_after_rep = np.sum(is_oneoff)

    # 排除无fluence
    fluence = np.array(data['fluence'], dtype=float)
    has_fluence = ~np.isnan(fluence)
    n_after_fluence = np.sum(is_oneoff & has_fluence)

    # 组合筛选
    mask = is_oneoff & has_fluence
    filtered = data[mask]

    print(f"\n筛选结果:")
    print(f"  原始:        {n_raw}")
    print(f"  排除重复暴:  {n_raw} → {n_after_rep} (排除 {n_raw - n_after_rep})")
    print(f"  排除无fluence: {n_after_rep} → {n_after_fluence} (排除 {n_after_rep - n_after_fluence})")
    print(f"  最终样本:    {n_after_fluence} 个 one-off FRB")

    return filtered


def extract_columns(data):
    """提取分析所需列，统一单位

    最终落盘列（只保留下游计算所需）:
    - name: FRB 名称（行标识）
    - fluence: 通量×持续时间 [Jy·ms]
    - width: 脉冲宽度 [ms]（从秒转换）
    - dm_obs: 观测色散量 dm_fitb [pc·cm⁻³]
    - dm_exc_ne2001: DM - DM_MW(NE2001) [pc·cm⁻³]
    - dm_exc_ymw16: DM - DM_MW(YMW16) [pc·cm⁻³]
    - dm_exc_ne2025: DM - DM_MW(NE2025) [pc·cm⁻³]（mwprop 计算，步骤 [5/6]）

    中间列（用完即删，不写入输出文件）:
    - snr_fitb / scat_time: 质量筛选条件 1/3
    - ra_dec_notes/exp_notes/fluence_notes/notes_fitb: 质量筛选条件 6
    - gl/gb: 银河坐标 [deg]，dm_exc_ne2025 计算用
    """
    n = len(data)
    result = {}

    # 名称
    result['name'] = np.array(data['tns_name'])

    # fluence [Jy·ms]
    result['fluence'] = np.array(data['fluence'], dtype=float)

    # width: FITS TUNIT29 确认 bc_width 单位为秒，转为毫秒
    result['width'] = np.array(data['bc_width'], dtype=float) * 1000.0   # [ms]

    # DM
    result['dm_obs'] = np.array(data['dm_fitb'], dtype=float)
    result['dm_exc_ne2001'] = np.array(data['dm_exc_ne2001'], dtype=float)
    result['dm_exc_ymw16'] = np.array(data['dm_exc_ymw16'], dtype=float)

    # 质量筛选条件所需列
    result['snr_fitb'] = np.array(data['snr_fitb'], dtype=float)
    # FITS TUNIT30 确认 scat_time 单位为秒，转为毫秒（筛选阈值 10 ms）
    result['scat_time'] = np.array(data['scat_time'], dtype=float) * 1000.0   # [ms]
    for key in ['ra_dec_notes', 'exp_notes', 'fluence_notes', 'notes_fitb']:
        if key not in data.columns.names:
            raise KeyError(f"目录缺少注释列 {key}，可用列: {data.columns.names}")
        result[key] = np.array(data[key], dtype=object)

    # 银河坐标 [deg]（dm_exc_ne2025 计算用；算完即删，不写入输出）
    if 'gl' not in data.columns.names or 'gb' not in data.columns.names:
        raise KeyError(f"目录缺少 gl/gb 列（NE2025 计算需要），可用列: {data.columns.names}")
    result['gl'] = np.array(data['gl'], dtype=float)
    result['gb'] = np.array(data['gb'], dtype=float)

    return result


def _has_content(v):
    """注释列的值是否为非空内容（空白串与 'nan' 视为空）"""
    s = str(v).strip()
    return s != '' and s.lower() != 'nan'


def filter_quality(result):
    """数据质量筛选：六项新标准（在 extract_columns 之后对 result 字典操作）

    筛选标准（不满足任一条即剔除）:
    1. snr_fitb >= 12
    2. dm_fitb(=dm_obs) >= 100 pc·cm⁻³
    3. scat_time <= 10 ms（NaN = 未测到散射，保留）
    4. DM_MW,NE2001 <= 100 pc·cm⁻³，剔除低银纬、银河系 DM 预测高不确定区
    5. dm_fitb >= 1.5·max(DM_MW,NE2001, DM_MW,YMW16)，
       即河外 DM 至少为 MW 预测的 0.5 倍
    6. ra_dec_notes / exp_notes / fluence_notes / notes_fitb 四列全为空
       （任何一列有注释说明该暴的坐标/曝光/流量/拟合存在问题，剔除）

    说明:
    - 条件 4/5 的 DM_MW 无需调用电子密度模型计算：目录自带
      dm_exc_model = dm_fitb - DM_MW,model，反推 DM_MW = dm_fitb - dm_exc 即可。
      条件 5 数学上等价于 min(dm_exc_ne2001, dm_exc_ymw16) >= dm_obs/3。
    - NaN 策略: snr_fitb / dm_obs / dm_exc_* 为 NaN 直接剔除（判据无法评估，
      且 NaN 会击穿下游似然）; scat_time 为 NaN 保留。
    """
    n0 = len(result['dm_obs'])
    snr = result['snr_fitb']
    dm = result['dm_obs']                # 即 dm_fitb
    scat = result['scat_time']
    exc_ne = result['dm_exc_ne2001']
    exc_ym = result['dm_exc_ymw16']
    # 银河系 DM 反推（由目录 excess 列，无需外部模型）
    mw_ne2001 = dm - exc_ne
    mw_ymw16 = dm - exc_ym

    print(f"\n质量筛选结果:")

    # 0. 判据列 NaN 剔除（scat_time 不参与，见 docstring）
    ok = np.isfinite(snr) & np.isfinite(dm) & np.isfinite(exc_ne) & np.isfinite(exc_ym)
    print(f"  判据列 NaN:        {n0} → {np.sum(ok)} (排除 {n0 - np.sum(ok)})")

    # 1. 信噪比下限
    ok1 = ok & (snr >= 12)
    print(f"  snr_fitb >= 12:    {np.sum(ok)} → {np.sum(ok1)} (排除 {np.sum(ok) - np.sum(ok1)})")
    ok = ok1

    # 2. 观测 DM 下限
    ok1 = ok & (dm >= 100)
    print(f"  dm_fitb >= 100:    {np.sum(ok)} → {np.sum(ok1)} (排除 {np.sum(ok) - np.sum(ok1)})")
    ok = ok1

    # 3. 散射时标上限（NaN 视为无散射测量，保留）
    ok1 = ok & (np.isnan(scat) | (scat <= 10))
    print(f"  scat_time <= 10ms: {np.sum(ok)} → {np.sum(ok1)} (排除 {np.sum(ok) - np.sum(ok1)})")
    ok = ok1

    # 4. NE2001 银河系 DM 上限
    ok1 = ok & (mw_ne2001 <= 100)
    print(f"  DM_MW,ne2001 <= 100: {np.sum(ok)} → {np.sum(ok1)} (排除 {np.sum(ok) - np.sum(ok1)})")
    ok = ok1

    # 5. 观测 DM 不低于两个模型银河系 DM 较大者的 1.5 倍
    ok1 = ok & (dm >= 1.5 * np.maximum(mw_ne2001, mw_ymw16))
    print(f"  dm >= 1.5 * max(DM_MW):  {np.sum(ok)} → {np.sum(ok1)} (排除 {np.sum(ok) - np.sum(ok1)})")
    ok = ok1

    # 6. 四个注释列全为空
    has_note = np.zeros(n0, dtype=bool)
    for key in ['ra_dec_notes', 'exp_notes', 'fluence_notes', 'notes_fitb']:
        vals = np.asarray(result[key], dtype=object)
        has_note |= np.array([_has_content(v) for v in vals])
    ok1 = ok & (~has_note)
    print(f"  注释列全空:        {np.sum(ok)} → {np.sum(ok1)} (排除 {np.sum(ok) - np.sum(ok1)})")
    ok = ok1

    print(f"  质量筛选后样本:    {np.sum(ok)} 个 FRB")

    mask = ok
    for key in result:
        result[key] = result[key][mask]

    # 注释列仅用于筛选，不写入输出文件
    for key in ['ra_dec_notes', 'exp_notes', 'fluence_notes', 'notes_fitb']:
        del result[key]

    return result


def compute_dm_exc_ne2025(result, dist_kpc, nproc=1):
    """调用 frb_util 的 mwprop 封装计算 dm_exc_ne2025 列（薄编排层）

    dm_exc_ne2025 = dm_obs - DM_MW,NE2025(gl, gb, dist_kpc)
    重活在 frb_util.ne2025_dm_mw_batch（mwprop 调用/进度/容错/nproc 并行），
    这里只做目录列组装与三列 dm_exc 同样本对比（median / 负值数）。
    """
    try:
        dm_mw, n_fail = ne2025_dm_mw_batch(result['gl'], result['gb'], dist_kpc,
                                           nproc=nproc)
    except ImportError:
        raise ImportError("需要官方包 mwprop: pip install mwprop；"
                          "或加 --skip-ne2025 跳过此列")
    result['dm_exc_ne2025'] = result['dm_obs'] - dm_mw
    if n_fail > 0:
        bad = ~np.isfinite(dm_mw)
        names = ', '.join(str(s) for s in result['name'][bad][:10])
        print(f"[warn] NE2025 计算失败 {n_fail} 个事件（置 NaN）: {names}"
              + (" ..." if n_fail > 10 else ""))
    ok = np.isfinite(dm_mw)
    if ok.any():
        print(f"  DM_MW,NE2025 [pc/cm3]: min={np.min(dm_mw[ok]):.1f} "
              f"median={np.median(dm_mw[ok]):.1f} max={np.max(dm_mw[ok]):.1f}")
    for col in ['dm_exc_ne2001', 'dm_exc_ymw16', 'dm_exc_ne2025']:
        v = result[col]
        vok = v[np.isfinite(v)]
        med = np.median(vok) if len(vok) else float('nan')
        n_neg = int(np.sum(vok < 0)) if len(vok) else 0
        print(f"  {col:>16}: median={med:8.1f}   负值 {n_neg}/{len(vok)}")
    return result


def save_fits(result, fname, extra_header=None):
    """保存为 FITS 格式"""
    cols = []
    for key in result:
        if result[key].dtype.kind in ('U', 'S'):
            # 字符串列：使用最大长度避免截断
            max_len = max(len(s) for s in result[key])
            col = fits.Column(name=key, format=f'{max_len}A', array=result[key])
        else:
            col = fits.Column(name=key, format='D', array=result[key])
        cols.append(col)

    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.header['CATALOG'] = 'CHIME/FRB Catalog 2 (filtered)'
    hdu.header['NAXIS2'] = len(result['fluence'])
    hdu.header['WUNIT'] = 'ms'
    if extra_header:
        for k, v in extra_header.items():
            hdu.header[k] = v

    hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
    hdul.writeto(fname, overwrite=True)
    print(f"  FITS: {fname}")


def save_txt(result, fname):
    """保存为 TXT 格式（空格分隔，首行为列名）"""
    keys = list(result.keys())
    header = ' '.join(keys)

    # 分离字符串列和数值列
    str_keys = [k for k in keys if result[k].dtype.kind in ('U', 'S', 'O')]
    num_keys = [k for k in keys if result[k].dtype.kind not in ('U', 'S', 'O')]

    # 用混合格式写入
    with open(fname, 'w') as f:
        f.write(f'# {header}\n')
        for i in range(len(result[keys[0]])):
            parts = []
            for k in keys:
                val = result[k][i]
                if k in str_keys:
                    parts.append(str(val))
                else:
                    parts.append(f'{float(val):.6f}')
            f.write(' '.join(parts) + '\n')
    print(f"  TXT:  {fname}")


def print_summary(result):
    """打印数据摘要"""
    print(f"\n数据摘要:")
    for key in result:
        vals = result[key]
        if vals.dtype.kind in ('U', 'S', 'O'):
            print(f"  {key:>15}: {len(vals)} 条")
        else:
            print(f"  {key:>15}: min={np.nanmin(vals):.4f}  max={np.nanmax(vals):.4f}  median={np.nanmedian(vals):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FRB Catalog 预处理')
    parser.add_argument('-i', '--input', default='data/chimefrbcat2.fits',
                        help='输入 FITS 文件路径')
    parser.add_argument('-o', '--output', default='data/filtered',
                        help='输出文件路径（不含扩展名）')
    parser.add_argument('--skip-ne2025', action='store_true', dest='skip_ne2025',
                        help='跳过 dm_exc_ne2025 计算（环境未装 mwprop 时使用）')
    parser.add_argument('--verify-dist', action='store_true', dest='verify_dist',
                        help='先做 50/100/150 kpc 平台验证（检查积分距离取值）')
    parser.add_argument('--config', action='store', dest='config_path', type=str,
                        default='config_mini.json', help='Path to config file')
    parser.add_argument('--nproc', action='store', dest='nproc', type=int, default=4,
                        help='NE2025 并行进程数（mwprop 为纯 Python 标量接口, '
                             '多进程是唯一有效加速; 1=串行）')
    args = parser.parse_args()
    t_start = time.perf_counter()

    print("=" * 50)
    print("FRB Catalog 预处理")
    print("=" * 50)

    # 1. 加载原始数据
    print(f"\n[1/6] 加载原始数据: {args.input}")
    data, col_names = load_raw_catalog(args.input)

    # 2. 筛选
    print(f"\n[2/6] 最小筛选（排除重复暴 + 无fluence）")
    filtered = filter_minimum(data)

    # 3. 提取列并转换单位
    print(f"\n[3/6] 提取列并转换单位")
    result = extract_columns(filtered)
    print_summary(result)

    # 4. 数据质量筛选
    print(f"\n[4/6] 数据质量筛选（snr/dm/scat/DM_MW/注释列 六项标准）")
    result = filter_quality(result)

    # 5. NE2025 河外 DM（mwprop，需服务器环境）
    # 积分距离从 config 读取（analysis.mwprop_dist_kpc，默认 100 kpc）
    config = load_config(args.config_path)
    dist_kpc = config.get('analysis', {}).get('mwprop_dist_kpc', 100.)
    mwprop_ver = None
    if args.verify_dist:
        try:
            ne2025_verify_plateau(result['gl'], result['gb'], dist_kpc)
        except ImportError:
            print("[warn] 未安装 mwprop，跳过平台验证")
    if args.skip_ne2025:
        print(f"\n[5/6] 跳过 dm_exc_ne2025（--skip-ne2025）")
    else:
        print(f"\n[5/6] 计算 dm_exc_ne2025（mwprop/NE2025，积分距离 {dist_kpc} kpc，"
              f"{args.nproc} 进程）")
        result = compute_dm_exc_ne2025(result, dist_kpc, nproc=args.nproc)
        try:
            import mwprop
            mwprop_ver = getattr(mwprop, '__version__', 'unknown')
        except Exception:
            mwprop_ver = 'unknown'

    # 中间列用完即删：输出只保留下游计算所需列
    # （注释列已在 filter_quality 内删除）
    for key in ['gl', 'gb', 'snr_fitb', 'scat_time']:
        result.pop(key, None)

    # 6. 保存
    print(f"\n[6/6] 保存预处理数据")
    print("\n最终输出数据:")
    print_summary(result)
    fits_out = args.output + '.fits'
    txt_out = args.output + '.txt'
    Path(fits_out).parent.mkdir(parents=True, exist_ok=True)
    extra_header = {}
    if mwprop_ver is not None:
        extra_header = {'MWPVER': mwprop_ver, 'MWPDIST': dist_kpc}
    save_fits(result, fits_out, extra_header=extra_header)
    save_txt(result, txt_out)

    print(f"\n完成！分析代码使用预处理后的数据:")
    print(f'  config.json: "catalog_path": "{fits_out}"')
    print(f"[preprocess_catalog.py 总耗时 {time.perf_counter()-t_start:.1f}s]")