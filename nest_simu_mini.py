#!/usr/bin/env python3

import numpy as np
import time
import sys
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
LLGUARD = LlfailGuard()  # -1e30 失败计数 + 熔断（frb_util 单一实现, A1）

def lnlik(vpar):
    """能量版似然函数（CHIME 单巡天）

    vpar 顺序由 frb_util.PARAM_ORDER 定义（8 维全启用）。
    异常口径与 nest_samp_mini.lnlik 一致（2026-10-09 收紧）：只吞数值失败，
    接线错误必须当场炸。注入-恢复跑的是同一条似然链，这里若把 TypeError
    吞成 -1e30，会得到一个"成功收敛"的平坦后验，恢复检验整轮作废。
    """
    LLGUARD.note_eval()
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
    except (ArithmeticError, ValueError) as exc:
        return LLGUARD.fail(vpar, exc)
    # numpy 默认只 warn 不抛，nan/inf 会一路穿到这里；MultiNest 的 C 层拿 nan
    # 做比较是未定义行为，故显式兜底（同样计入熔断计数）。
    if not np.isfinite(res):
        return LLGUARD.fail(vpar)
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
    parser.add_argument('-f1', action='store', dest='simu1', type=str, required=True,
                        help='Input simulation data file (simufrb.py 的产物)')
    parser.add_argument('-fs', action='store', dest='fsvy', type=str,
                        help='Survey info file (default: from config)')
    parser.add_argument('-o', action='store', dest='fout', type=str, required=True,
                        help='Output file basename（MultiNest 直接在其后拼 post_equal_weights.dat）')
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
    # （保证与 nest_samp_mini.py / simufrb.py 用同一套参数源；缺 CHIME 当场报错，
    #  与 nest_samp_mini.py 一致——静默回退默认值会让模拟线与真实线用不同口径）
    fsvy = args.fsvy if args.fsvy else config['data']['survey_info_path']
    svy_info = lf.LoadSvyInfo(fsvy)
    survey_names = list(svy_info['SURVEY'])
    if 'CHIME' not in survey_names:
        raise ValueError(f"{fsvy} 中未找到 CHIME 行（本项目只支持单巡天）")
    idx = survey_names.index('CHIME')
    g = float(svy_info['Gain'][idx])
    bw = float(svy_info['BW'][idx])
    tsys = float(svy_info['Tsys'][idx])
    fov = float(svy_info['FOV'][idx])
    npol = float(svy_info['Npol'][idx])
    sn0 = float(svy_info['SN0'][idx])

    cat = lf.LoadSimuData(simu1)
    vLOGFLUX = np.log10(cat['S'])
    vLOGW = np.log10(cat['W'])
    vDME = cat['DMe']
    vdT = cat['T']
    # 与 nest_samp_mini 同一道护栏：mock 里的 NaN/Inf（simufrb 被 max_iter 截断留零行、
    # 或注入参数非法）会在 log_distr_efdmw 里被吞成 -1e30，注入-恢复整轮作废而日志
    # 看不出来（V27 钉的就是这类静默平台）。
    for _name, _arr in (('S', cat['S']), ('W', cat['W']), ('DMe', vDME), ('T', vdT)):
        _bad = int(np.sum(~np.isfinite(_arr)))
        if _bad:
            raise ValueError(f"{simu1} 的 {_name} 列含 {_bad} 个非有限值(NaN/Inf)")
    vN = np.array([len(vLOGFLUX)])
    vFOV = np.array([fov])
    # 模拟数据的观测时长必须用 sum(vT)：模拟器固定生成 ns 个事件，
    # 到达时间 vT ~ exponential(1/λ)，实际总时长 = sum(vT) ≈ ns/λ。
    # header 中的 T_obs 是 tel_svy.txt 的巡天时长（如 CHIME 26280 hr），
    # 不是模拟数据的实际时长。若用 T_obs 会导致泊松似然把 phis 偏离真值
    # 约 T_obs/sum(vT) 倍（可达 3+ 个量级），truth 线飞出 posterior 范围。
    vT = np.array([np.sum(vdT)])

    # 先验范围（从 config 读取）。名字刻意不叫 vpara/vparb：已删除的宿主标定参数组
    # 就叫 vpar_*，同根命名会让人以为那套标定还在（2026-10-11 与 nest_samp 同步改名）。
    cfg = config['prior']
    par_lo = np.array([cfg[k][0] for k in PARAM_ORDER])
    par_hi = np.array([cfg[k][1] for k in PARAM_ORDER])

    print('------------par range-----------')
    print('lower:', par_lo)
    print('upper:', par_hi)
    print(f'CHIME params: g={g}, bw={bw}, tsys={tsys}, fov={fov}, dnu={dnu}')
    a1 = time.perf_counter()
    # sanity check 取先验盒【中心】u=0.5（u=0 是角点 γ=-4/σ_w=0.01，实测每个事件都
    # 返回 -1e30，打印值不携带信息）。注入-恢复看的就是这个数字是否有限且可读。
    cube_test = np.full(len(par_lo), 0.5)
    myprior(cube_test, len(par_lo), len(par_lo))
    print('lnlik @ 先验盒中心 =', myloglike(cube_test, len(par_lo), len(par_lo)))
    a2 = time.perf_counter()
    print(f'Single eval time: {a2-a1:.3f}s')
    print("Running Nest Sampling ...")
    import os
    import scipy
    # -o 含 / 就是全路径（run 脚本传 $RUN_DIR/<名>），直接用；裸名才拼传统平铺
    # 前缀——手工单跑行为不变（A3 配套）。
    if '/' in fout or os.sep in fout:
        basename = fout
    else:
        basename = config['output']['nest_out_dir'] + 'simu/' + fout
    os.makedirs(os.path.dirname(os.path.abspath(basename)), exist_ok=True)
    skw = sampler_kwargs(config)
    doc = {
        'line': os.path.basename(fout),
        'run_tag': os.environ.get('CC_RUN_TAG', ''),
        'argv': sys.argv,
        'config': config,
        'config_md5': config_md5(config),
        'input_md5': {os.path.basename(simu1): file_md5(simu1)},
        'SURVIVAL': float(SURVIVAL),
        'sel_mode': dis.sel_mode,
        'sampler': skw,
        'versions': {'python': sys.version.split()[0], 'numpy': np.__version__,
                     'scipy': scipy.__version__},
        'start_utc': utc_now(),
        'pid': os.getpid(),
    }
    # resume 一致性闸：resume.dat 已存在而 config/数据/采样设置变了就拒绝续跑，
    # 否则 MultiNest 只认 ndim，会无声产出两轮混血链（A3）。
    if skw['resume']:
        check_resume(basename, doc)
    write_manifest(manifest_path(basename), doc)
    pymultinest.run(myloglike, myprior, len(par_lo),
                    importance_nested_sampling=False,
                    verbose=True,
                    sampling_efficiency='model',
                    outputfiles_basename=basename, **skw)
    wall = time.perf_counter() - t_total0
    print(f"[simu] MultiNest 总耗时 {wall:.1f}s ({wall/60:.1f} min)")
    doc.update(end_utc=utc_now(), wall_seconds=round(wall, 1),
               n_iter=count_lines(str(basename) + 'ev.dat'),
               n_post=count_lines(str(basename) + 'post_equal_weights.dat'),
               n_eval=LLGUARD.n_eval, n_llfail=LLGUARD.n_fail)
    write_manifest(manifest_path(basename), doc)
