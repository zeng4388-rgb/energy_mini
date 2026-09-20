#!/usr/bin/env python3

import numpy as np
from collections import Counter
import time
import pymultinest
import warnings
import sys
import argparse

from frb_util import *

dis = AstroDistribution()
cos = Cosmology()
er = EventRate()
tel = Telescope()
lf = Loadfiles()

dnu = 400.0  # 默认值，__main__ 中从 config 更新

def lnlik(vpar):
    """能量版似然函数

    vpar: [phis, alpha, log_Es, log_E0, mu_w, sigma_w]
    """
    try:
        # 归一化因子（能量版）
        norm = np.zeros(vFOV.shape)
        for i in range(len(norm)):
            norm[i] = dis.Norm1D_E(vSN0[i], vBW[i], vNpol[i], vG[i], vTs[i],
                                   dnu, vpar[1], vpar[2], vpar[3], vpar[4], vpar[5])
        # 逐事件似然（能量版，对 z 边际化）
        loglik_fdm = np.zeros(vFOV.shape)
        for i in range(len(vFOV)):
            loglik_fdm[i] = np.sum(
                dis.log_distr_efdmw(dnu, vLOGF_2d[i], vDME_2d[i], vLOGW_2d[i],
                                    vpar[1], vpar[2], vpar[3], vpar[4], vpar[5],
                                    gtype=fgt) - np.log(norm[i]))
        loglik_norm = np.sum(loglik_fdm)
        # 泊松似然（能量版事件率）
        rho = np.zeros(vFOV.shape)
        for i in range(len(rho)):
            rho[i] = er.rate_2d_E(vSN0[i], vBW[i], vNpol[i], vG[i], vTs[i],
                                  dnu, vpar[0], vpar[1], vpar[2], vpar[3],
                                  vpar[4], vpar[5])
        loglik_poi = np.sum(er.log_dis_poi(rho, vN, vFOV, vTime))
        res = loglik_norm + loglik_poi
        return res
    except Exception:
        import traceback
        print('Numerical error: @', vpar)
        traceback.print_exc()
        return -1e30

def myprior(cube, ndim, nparams):
    """从 config.json 读取先验范围（能量参数）

    cube[0] 输出 linear phis（下游 rate_2d_E 需要线性值）
    cube[2], cube[3] 输出 log10(E*), log10(E0)

    注意:参数顺序 [phis, alpha, log_Es, log_E0, mu_w, sigma_w] 在多处硬编码依赖:
    - pltpost.py 第 97/109 行的 vari != 3 把 log_E0 特殊处理(只画 95% 上限)
    - pltpost.py 第 245-252 行的 rangedat 顺序
    改顺序要同步改那些地方。
    """
    cfg = config['prior']
    cube[0] = 10.0 ** (cfg['log_phis'][0] + cube[0] * (cfg['log_phis'][1] - cfg['log_phis'][0]))
    cube[1] = cfg['alpha'][0] + cube[1] * (cfg['alpha'][1] - cfg['alpha'][0])
    cube[2] = cfg['log_Es'][0] + cube[2] * (cfg['log_Es'][1] - cfg['log_Es'][0])
    cube[3] = cfg['log_E0'][0] + cube[3] * (cfg['log_E0'][1] - cfg['log_E0'][0])
    cube[4] = cfg['mu_w'][0] + cube[4] * (cfg['mu_w'][1] - cfg['mu_w'][0])
    cube[5] = cfg['sigma_w'][0] + cube[5] * (cfg['sigma_w'][1] - cfg['sigma_w'][0])

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
    parser.add_argument('-g', action='store', dest='fgt', type=str,
                        help='Host galaxy type (ETG, LTG_NE2001, LTG_YMW16, ALG_NE2001, ALG_YMW16)')
    parser.add_argument('-halo', action='store_true', dest='bolhalo',
                        help='Bool option: removing the DM from dark halo')
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
    bolhalo = args.bolhalo
    use_fits = args.use_fits
    mwmodel = args.mwmodel

    # 加载全局配置
    config = load_config(args.config_path)
    dnu = config.get('analysis', {}).get('dnu', TELESCOPES['CHIME']['dnu'])
    halo_dm = config.get('analysis', {}).get('halo_dm', 0.0)
    # 用 config 中的宇宙学参数创建新实例，并同步到 dis / er
    cosmo_cfg = config.get('cosmology', {})
    cos = Cosmology(omegam=cosmo_cfg.get('Omega_m', 0.308),
                    omegal=cosmo_cfg.get('Omega_L', 0.692))
    dis.cos = cos
    er.cos = cos
    er.ad = dis

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
        vF, vW, vDM_obs, vDM_ne2001, vDM_ymw16, vDM_ne2025, vSVY = lf.LoadFitsCatalog(fcat)
        vLOGF = np.log10(vF / vW)   # flux = fluence / width [Jy]
        vLOGW = np.log10(vW)
    else:
        frb_cat = lf.LoadCatalogue(fcat)
        vDM_ne2001 = frb_cat['DM'] - frb_cat['DM_NE2001']
        vDM_ymw16 = frb_cat['DM'] - frb_cat['DM_YMW16']
        vDM_ne2025 = None   # 遗留 TXT 路径无 NE2025 列
        vLOGF = np.log10(frb_cat['S'])
        vLOGW = np.log10(frb_cat['W'])
        vSVY = frb_cat['SURVEY']

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
    print(f'[samp] MW model = {mwmodel}, host galaxy type = {fgt}')

    if bolhalo:
        print('[warn] -halo 已弃用: 银河系晕 DM 已由 U[0, DMsmax] 显式卷积建模'
              '（log_distr_efdmwz, DMsmax=60, 均值30=Dolag2015），'
              '再扣 halo_dm 会重复扣除。仅 DM_MW 固定值敏感性实验时临时使用。')
        vDME = vDME - halo_dm

    if fgt and fgt.find('ETG') >= 0:
        fgt = 'ETG'

    print(f'[samp] 加载 {len(vLOGF)} 个已筛查 FRB（one-off，含 fluence）')

    # 加载巡天信息
    svy_info = lf.LoadSvyInfo(fsvy)

    # 按巡天分组
    cts = Counter(vSVY)

    # 自动检测巡天列表（从 tel_svy.txt 读取）
    survey_names = svy_info['SURVEY']
    vN = np.array([cts.get(s, 0) for s in survey_names])
    vSN0 = svy_info['SN0']
    vTs = svy_info['Tsys']
    vG = svy_info['Gain']
    vBW = svy_info['BW']
    vNpol = svy_info['Npol']
    vFOV = svy_info['FOV']
    vTime = svy_info['TIME']

    # 按巡天名称分组数据
    vLOGF_2d = [vLOGF[vSVY == s] for s in survey_names]
    vDME_2d = [vDME[vSVY == s] for s in survey_names]
    vLOGW_2d = [vLOGW[vSVY == s] for s in survey_names]

    # 先验范围（从 config.json 读取）
    cfg = config['prior']
    vpara = np.array([cfg['log_phis'][0], cfg['alpha'][0], cfg['log_Es'][0],
                      cfg['log_E0'][0], cfg['mu_w'][0], cfg['sigma_w'][0]])
    vparb = np.array([cfg['log_phis'][1], cfg['alpha'][1], cfg['log_Es'][1],
                      cfg['log_E0'][1], cfg['mu_w'][1], cfg['sigma_w'][1]])

    print('------------par range-----------')
    print('lower:', vpara)
    print('upper:', vparb)
    a1 = time.perf_counter()
    # sanity check：通过 myprior 变换后再求似然（vpara[0] 是 log_phis，不能直送 lnlik）
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
