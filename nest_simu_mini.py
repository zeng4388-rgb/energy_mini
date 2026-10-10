#!/usr/bin/env python3

import numpy as np
import time
import pymultinest
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
SURVIVAL = 1.0  # Poisson 项样本存活率，__main__ 中从 filtered.fits header 读

def lnlik(vpar):
    """能量版似然函数（CHIME 单巡天）

    vpar 顺序由 frb_util.PARAM_ORDER 定义（8 维全启用）。
    异常口径与 nest_samp_mini.lnlik 一致（2026-10-09 收紧）：只吞数值失败，
    接线错误必须当场炸。注入-恢复跑的是同一条似然链，这里若把 TypeError
    吞成 -1e30，会得到一个"成功收敛"的平坦后验，恢复检验整轮作废。
    """
    try:
        norm = dis.Norm1D_E(sn0, bw, npol, g, tsys, dnu,
                            vpar[1], vpar[2], vpar[3], vpar[4], vpar[5])
        loglik_fdm = np.sum(
            dis.log_distr_efdmw(dnu, vLOGFLUX, vDME, vLOGW,
                                vpar[1], vpar[2], vpar[3], vpar[4], vpar[5],
                                vpar[6], vpar[7]) - np.log(norm))
        rho = er.rate_2d_E(sn0, bw, npol, g, tsys, dnu,
                           vpar[0], vpar[1], vpar[2], vpar[3], vpar[4], vpar[5])
        # Λ_sel = Λ_det × S（与 nest_samp_mini 同一定义，见 read_selection_survival）
        loglik_poi = np.sum(er.log_dis_poi(np.array([rho]) * SURVIVAL, vN, vFOV, vT))
        res = loglik_fdm + loglik_poi
    except (ArithmeticError, ValueError):
        import traceback
        print('Numerical error: @', vpar)
        traceback.print_exc()
        return -1e30
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
    parser = argparse.ArgumentParser(description='E_iso Measurements Program for Simulated FRBs')
    parser.add_argument('-f1', action='store', dest='simu1', type=str, help='Input simulation data file')
    parser.add_argument('-fs', action='store', dest='fsvy', type=str,
                        help='Survey info file (default: from config)')
    parser.add_argument('-o', action='store', dest='fout', type=str, help='Output file basename')
    parser.add_argument('--config', action='store', dest='config_path', type=str,
                        default='config_mini.json', help='Path to config file')
    args = parser.parse_args()
    t_total0 = time.perf_counter()
    simu1 = args.simu1
    fout = args.fout

    # 加载全局配置
    config = load_config(args.config_path)
    dnu = config.get('analysis', {}).get('dnu', TELESCOPES['CHIME']['dnu'])
    # 宇宙学 + dm_model 一次性注入，并让 er 与 dis 共享同一个 cos（重绑模块全局）
    cos, dis, er = build_models(config)
    # 与 nest_samp_mini 同一定义：Poisson 的观测数是筛选后样本，λ 需乘存活率。
    # mode 取 dis.sel_mode（build_models 已按 config/env 注入）⇒ 分母与似然同源
    SURVIVAL = read_selection_survival(config['data']['catalog_path'], mode=dis.sel_mode)

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
    vpara = np.array([cfg[k][0] for k in PARAM_ORDER])
    vparb = np.array([cfg[k][1] for k in PARAM_ORDER])

    print('------------par range-----------')
    print('lower:', vpara)
    print('upper:', vparb)
    print(f'CHIME params: g={g}, bw={bw}, tsys={tsys}, fov={fov}, dnu={dnu}')
    a1 = time.perf_counter()
    # sanity check：通过 myprior 变换后再求似然（vpara[0] 是 log_phi0，不能直送 lnlik）
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
