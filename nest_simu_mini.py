#!/usr/bin/env python3

import numpy as np
import time
import pymultinest
import warnings
import sys
import argparse

from frb_util import *
dis = AstroDistribution()
cos = Cosmology()
er = EventRate()
lf = Loadfiles()

# CHIME 望远镜参数（从 TELESCOPES 字典读取，dnu 在 __main__ 中从 config 更新）
tel = TELESCOPES['CHIME']
g = tel['g']
bw = tel['bw']
tsys = tel['tsys']
fov = tel['fov']
npol = tel['npol']
sn0 = tel['sn0']
dnu = tel['dnu']

def lnlik(vpar):
    """能量版似然函数（CHIME 单巡天）

    vpar: [phis, alpha, log_Es, log_E0, mu_w, sigma_w]
    """
    try:
        norm = dis.Norm1D_E(sn0, bw, npol, g, tsys, dnu,
                            vpar[1], vpar[2], vpar[3], vpar[4], vpar[5])
        loglik_fdm = np.sum(
            dis.log_distr_efdmw(dnu, vLOGFLUX, vDME, vLOGW,
                                vpar[1], vpar[2], vpar[3], vpar[4], vpar[5],
                                gtype=fgt) - np.log(norm))
        rho = er.rate_2d_E(sn0, bw, npol, g, tsys, dnu,
                           vpar[0], vpar[1], vpar[2], vpar[3], vpar[4], vpar[5])
        loglik_poi = np.sum(er.log_dis_poi(np.array([rho]), vN, vFOV, vT))
        res = loglik_fdm + loglik_poi
        return res
    except Exception:
        import traceback
        print('Numerical error: @', vpar)
        traceback.print_exc()
        return -1e30

def myprior(cube, ndim, nparams):
    """从 config 读取先验范围（能量参数）

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
    parser = argparse.ArgumentParser(description='E_iso Measurements Program for Simulated FRBs')
    parser.add_argument('-f1', action='store', dest='simu1', type=str, help='Input simulation data file')
    parser.add_argument('-fs', action='store', dest='fsvy', type=str,
                        help='Survey info file (default: from config)')
    parser.add_argument('-o', action='store', dest='fout', type=str, help='Output file basename')
    parser.add_argument('-g', action='store', dest='fgt', type=str, help='Host galaxy type')
    parser.add_argument('--config', action='store', dest='config_path', type=str,
                        default='config_mini.json', help='Path to config file')
    args = parser.parse_args()
    t_total0 = time.perf_counter()
    simu1 = args.simu1
    fout = args.fout
    fgt = args.fgt

    # 加载全局配置
    config = load_config(args.config_path)
    dnu = config.get('analysis', {}).get('dnu', TELESCOPES['CHIME']['dnu'])
    cosmo_cfg = config.get('cosmology', {})
    cos = Cosmology(omegam=cosmo_cfg.get('Omega_m', 0.308),
                    omegal=cosmo_cfg.get('Omega_L', 0.692))
    # 同步宇宙学实例到 dis / er
    dis.cos = cos
    er.cos = cos
    er.ad = dis

    # 从 tel_svy.txt 读取 CHIME 望远镜参数，覆盖模块级硬编码值
    # （保证与 nest_samp_mini.py / simufrb.py 用同一套参数源）
    fsvy = args.fsvy if args.fsvy else config['data']['survey_info_path']
    svy_info = lf.LoadSvyInfo(fsvy)
    survey_names = list(svy_info['SURVEY'])
    if 'CHIME' in survey_names:
        idx = survey_names.index('CHIME')
        g = float(svy_info['Gain'][idx])
        bw = float(svy_info['BW'][idx])
        tsys = float(svy_info['Tsys'][idx])
        fov = float(svy_info['FOV'][idx])
        npol = float(svy_info['Npol'][idx])
        sn0 = float(svy_info['SN0'][idx])
    else:
        print(f'[warn] tel_svy.txt 中未找到 CHIME，回退到 TELESCOPES 默认值')

    cat = lf.LoadSimuData(simu1)
    vLOGFLUX = np.log10(cat['S'])
    vLOGW = np.log10(cat['W'])
    vDME = cat['DMe']
    vdT = cat['T']
    vN = np.array([len(vLOGFLUX)])
    vFOV = np.array([fov])
    # 模拟数据的观测时长必须用 sum(vT)：模拟器固定生成 ns 个事件，
    # 到达时间 vT ~ exponential(1/λ)，实际总时长 = sum(vT) ≈ ns/λ。
    # header 中的 T_obs 是 tel_svy.txt 的巡天时长（如 CHIME 26280 hr），
    # 不是模拟数据的实际时长。若用 T_obs 会导致泊松似然把 phis 偏离真值
    # 约 T_obs/sum(vT) 倍（可达 3+ 个量级），truth 线飞出 posterior 范围。
    vT = np.array([np.sum(vdT)])

    # 先验范围（从 config 读取）
    cfg = config['prior']
    vpara = np.array([cfg['log_phis'][0], cfg['alpha'][0], cfg['log_Es'][0],
                      cfg['log_E0'][0], cfg['mu_w'][0], cfg['sigma_w'][0]])
    vparb = np.array([cfg['log_phis'][1], cfg['alpha'][1], cfg['log_Es'][1],
                      cfg['log_E0'][1], cfg['mu_w'][1], cfg['sigma_w'][1]])

    print('------------par range-----------')
    print('lower:', vpara)
    print('upper:', vparb)
    print(f'CHIME params: g={g}, bw={bw}, tsys={tsys}, fov={fov}, dnu={dnu}')
    a1 = time.perf_counter()
    # sanity check：通过 myprior 变换后再求似然（vpara[0] 是 log_phis，不能直送 lnlik）
    cube_test = np.zeros(len(vpara))
    myprior(cube_test, len(vpara), len(vpara))
    print(myloglike(cube_test, len(vpara), len(vpara)))
    a2 = time.perf_counter()
    print(f'Single eval time: {a2-a1:.3f}s')
    print("Running Nest Sampling ...")
    import os
    output_dir = config['output']['nest_out_dir'] + 'simu/'
    os.makedirs(output_dir, exist_ok=True)
    pymultinest.run(myloglike, myprior, len(vpara),
                    importance_nested_sampling=False,
                    resume=False,
                    verbose=True,
                    sampling_efficiency='model',
                    n_live_points=1000,
                    outputfiles_basename=config['output']['nest_out_dir'] + 'simu/' + fout)
    print(f"[simu] MultiNest 总耗时 {time.perf_counter()-t_total0:.1f}s "
          f"({(time.perf_counter()-t_total0)/60:.1f} min)")
