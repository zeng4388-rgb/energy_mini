#!/usr/bin/env python3

import numpy as np
import time
import pymultinest
import argparse

from frb_util import *

dis = AstroDistribution()
cos = Cosmology()
er = EventRate()
tel = Telescope()
lf = Loadfiles()

dnu = 1000.0  # 能量定义参考带宽，__main__ 中从 config 更新
SURVIVAL = 1.0  # Poisson 项样本存活率，__main__ 中从 filtered.fits header 读

def lnlik(vpar):
    """能量版似然函数（CHIME 单巡天）。

    vpar 顺序由 frb_util.PARAM_ORDER 定义（8 维全启用）。
    项目决定不再扩展多巡天：tel_svy.txt 只取 CHIME 一行，三个 per-survey
    循环（旧版 norm/loglik_fdm/rho 的 for i in range(len(vFOV))）已塌缩为标量。

    异常口径（2026-10-09 收紧）：只吞【数值】失败，绝不 `except Exception`。
    实测事故：igm_delta_nodes 形状 bug 抛 TypeError 被吞成 -1e30，似然在整个
    先验空间恒为常数平台，MultiNest 照常"收敛"并输出一个看上去正常的平坦后验。
    TypeError/AttributeError/KeyError 是接线错误，必须当场炸掉。
    """
    try:
        # 归一化因子（能量版）
        norm = dis.Norm1D_E(vSN0, vBW, vNpol, vG, vTs,
                            dnu, vpar[1], vpar[2], vpar[3], vpar[4], vpar[5])
        # 逐事件似然（能量版，对 z 边际化）
        loglik_norm = np.sum(
            dis.log_distr_efdmw(dnu, vLOGF, vDME, vLOGW,
                                vpar[1], vpar[2], vpar[3], vpar[4], vpar[5],
                                vpar[6], vpar[7]) - np.log(norm))
        # 泊松似然（能量版事件率）
        rho = er.rate_2d_E(vSN0, vBW, vNpol, vG, vTs,
                           dnu, vpar[0], vpar[1], vpar[2], vpar[3],
                           vpar[4], vpar[5])
        # Λ_sel = Λ_det × S：rate_2d_E 只含辐射计阈值，样本还过了六道筛
        # （覆盖范围说明见 frb_util.read_selection_survival）
        loglik_poi = np.sum(er.log_dis_poi(np.array([rho * SURVIVAL]),
                                           vN, vFOV, vTime))
        res = loglik_norm + loglik_poi
    except (ArithmeticError, ValueError):
        # 数值域失败（OverflowError、brentq 括号失配等）：给一个有限的极低似然，
        # 别让一次求值废掉整轮 10 小时的采样。
        import traceback
        print('Numerical error: @', vpar)
        traceback.print_exc()
        return -1e30
    # numpy 默认只 warn 不抛，nan/inf 会一路穿到这里；MultiNest 的 C 层拿 nan
    # 做比较是未定义行为，故显式兜底。
    if not np.isfinite(res):
        return -1e30
    return res

def myprior(cube, ndim, nparams):
    """先验映射，顺序与形状全部来自 frb_util（附录 A 规则 R3/R4）。

    cfg['prior'] 给支撑区间，cfg['prior_shape'] 可选地给区间内的分布形状
    （如 ln_sigma_host 的截断正态）。两个采样脚本共用 prior_transform，
    不再各自重复一遍映射表。
    """
    for i, key in enumerate(PARAM_ORDER[:ndim]):
        cube[i] = prior_transform(key, cube[i], config)

def myloglike(cube, ndim, nparams):
    cube2 = np.zeros(ndim)
    for i in range(0, ndim):
        cube2[i] = cube[i]
    res = lnlik(cube2)
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='E_iso Measurements Program for FRB sample')
    parser.add_argument('-fc', action='store', dest='fcat', type=str,
                        help='Input FRB catalog file (txt or fits)')
    parser.add_argument('-fs', action='store', dest='fsvy', type=str,
                        help='Input Survey Information file')
    parser.add_argument('-o', action='store', dest='fout', type=str,
                        help='Output file basename')
    parser.add_argument('-g', action='store', dest='fgt', type=str, default=None,
                        help='仅用于未给 -mw 时推断 MW 模型(后缀 NE2001/YMW16); 宿主模型已统一为对数正态')
    parser.add_argument('--fits', action='store_true', dest='use_fits',
                        help='Load catalog from FITS file instead of txt')
    parser.add_argument('-mw', action='store', dest='mwmodel', type=str, default=None,
                        help='Milky Way electron density model: ne2001 / ymw16 / ne2025'
                             ' (default: inferred from -g suffix, keeps old behavior)')
    parser.add_argument('--config', action='store', dest='config_path', type=str,
                        default='config_mini.json', help='Path to config.json')

    args = parser.parse_args()
    t_total0 = time.perf_counter()
    fcat = args.fcat
    fsvy = args.fsvy
    fout = args.fout
    fgt = args.fgt
    use_fits = args.use_fits
    mwmodel = args.mwmodel

    # 加载全局配置
    config = load_config(args.config_path)
    dnu = config.get('analysis', {}).get('dnu', TELESCOPES['CHIME']['dnu'])
    # 宇宙学 + dm_model 一次性注入，并让 er 与 dis 共享同一个 cos（重绑模块全局，
    # lnlik 用的就是这三个对象）
    cos, dis, er = build_models(config)

    # 如果未指定 fcat/fsvy，从 config 读取默认路径
    if fcat is None:
        fcat = config['data']['catalog_path']
        # 仅在用户未显式指定 --fits 时，根据文件扩展名自动检测
        if not use_fits:
            use_fits = fcat.endswith('.fits') or fcat.endswith('.fit')
    if fsvy is None:
        fsvy = config['data']['survey_info_path']

    # 加载数据（已预处理：排除重复暴 + 无 fluence 的 burst + 六项质量筛选）
    if use_fits:
        vF, vW, vDM_obs, vDM_ne2001, vDM_ymw16, vDM_ne2025, _ = lf.LoadFitsCatalog(fcat)
        vLOGF = np.log10(vF / vW)   # flux = fluence / width [Jy]
        vLOGW = np.log10(vW)
    else:
        raise ValueError(
            f"遗留 TXT 加载路径已移除（LoadCatalogue 的列约定与 preprocess 输出不兼容）。"
            f"请改用 FITS：--fits 或把 config 的 catalog_path 指向 .fits（{fcat}）")

    # MW 模型选择（与 -g 宿主星系类型解耦；未指定 -mw 时按 -g 后缀推断，保持旧行为）
    if mwmodel is None:
        mwmodel = 'ne2001' if (fgt and fgt.find('NE2001') >= 0) else 'ymw16'
    mwmodel = mwmodel.lower()
    if mwmodel == 'ne2001':
        vDME = vDM_ne2001   # 已经是 DM_obs - DM_MW(NE2001)
    elif mwmodel == 'ymw16':
        vDME = vDM_ymw16    # 已经是 DM_obs - DM_MW(YMW16)
    elif mwmodel == 'ne2025':
        if vDM_ne2025 is None:
            raise ValueError("filtered.fits 无 dm_exc_ne2025 列: "
                             "先在服务器跑 preprocess_catalog.py（勿加 --skip-ne2025）")
        vDME = vDM_ne2025   # DM_obs - DM_MW(NE2025, mwprop 预计算)
    else:
        raise ValueError(f"未知 MW 模型: {mwmodel}（可选 ne2001 / ymw16 / ne2025）")
    print(f'[samp] MW model = {mwmodel} (宿主模型: 观测帧对数正态, 8 维)')

    print(f'[samp] 加载 {len(vLOGF)} 个已筛查 FRB（one-off，含 fluence）')

    # 泊松项存活率：header 缺失当场报错，绝不默认 1.0（= 静默复现选择函数缺口）
    # mode 取 dis.sel_mode（build_models 按 config/env 注入）⇒ SURVIV 分母与
    # Norm1D_E/rate_2d_E 里用的 S 同一个模式，三处不可能分叉
    SURVIVAL = read_selection_survival(fcat, mode=dis.sel_mode)

    # 加载巡天信息：CHIME 单巡天（项目决定不再扩展），只取 CHIME 一行。
    # 旧版按巡天分组的 Counter(vSVY) / vLOGF_2d 等列表已删除。
    svy_info = lf.LoadSvyInfo(fsvy)
    names = list(svy_info['SURVEY'])
    if 'CHIME' not in names:
        raise ValueError(f"{fsvy} 中未找到 CHIME 行（本项目只支持单巡天）")
    i = names.index('CHIME')
    vSN0 = float(svy_info['SN0'][i])
    vTs = float(svy_info['Tsys'][i])
    vG = float(svy_info['Gain'][i])
    vBW = float(svy_info['BW'][i])
    vNpol = float(svy_info['Npol'][i])
    # log_dis_poi 内部对 lamda 做布尔索引赋值，0-d 数组会抛 IndexError，
    # 故 FOV/TIME/N 保持 1 元素数组。
    vFOV = np.array([float(svy_info['FOV'][i])])
    vTime = np.array([float(svy_info['TIME'][i])])
    vN = np.array([float(len(vLOGF))])

    # 先验范围（从 config.json 读取）
    cfg = config['prior']
    vpara = np.array([cfg[k][0] for k in PARAM_ORDER])
    vparb = np.array([cfg[k][1] for k in PARAM_ORDER])

    print('------------par range-----------')
    print('lower:', vpara)
    print('upper:', vparb)
    a1 = time.perf_counter()
    # sanity check：通过 myprior 变换后再求似然（vpara[0] 是 log_phi0，不能直送 lnlik）
    cube_test = np.zeros(len(vpara))
    myprior(cube_test, len(vpara), len(vpara))
    print(myloglike(cube_test, len(vpara), len(vpara)))
    a2 = time.perf_counter()
    print(f'Single eval time: {a2-a1:.3f}s')
    print("Running Nest Sampling ...")
    # run MultiNest
    import os
    output_dir = config['output']['nest_out_dir'] + 'samp/'
    os.makedirs(output_dir, exist_ok=True)
    pymultinest.run(myloglike, myprior, len(vpara),
                    importance_nested_sampling=False,
                    resume=False,
                    verbose=True,
                    sampling_efficiency='model',
                    n_live_points=1000,
                    outputfiles_basename=config['output']['nest_out_dir'] + 'samp/' + fout)
    print(f"[samp] MultiNest 总耗时 {time.perf_counter()-t_total0:.1f}s "
          f"({(time.perf_counter()-t_total0)/60:.1f} min)")
