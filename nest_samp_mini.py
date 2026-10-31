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

dnu = 1000.0  # 能量定义参考带宽，__main__ 中从 config 更新
SURVIVAL = 1.0  # Poisson 项样本存活率，__main__ 中从 filtered.fits header 读
LLGUARD = LlfailGuard()  # -1e30 失败计数 + 熔断（frb_util 单一实现, A1）

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
    LLGUARD.note_eval()
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
    except (ArithmeticError, ValueError) as exc:
        # 数值域失败（OverflowError、brentq 括号失配等）：给一个有限的极低似然，
        # 别让一次求值废掉整轮 10 小时的采样；但计入熔断计数（A1）。
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
    parser = argparse.ArgumentParser(description='E_iso Measurements Program for FRB sample')
    parser.add_argument('-fc', action='store', dest='fcat', type=str,
                        help='Input FRB catalog file (txt or fits)')
    parser.add_argument('-fs', action='store', dest='fsvy', type=str,
                        help='Input Survey Information file')
    parser.add_argument('-o', action='store', dest='fout', type=str, required=True,
                        help='Output file basename（MultiNest 直接在其后拼 post_equal_weights.dat，'
                             '无分隔符；必填 ⇒ 缺名当场报错，不会白跑完 sanity check）')
    parser.add_argument('--fits', action='store_true', dest='use_fits',
                        help='Load catalog from FITS file instead of txt')
    parser.add_argument('-mw', action='store', dest='mwmodel', type=str, required=True,
                        choices=['ne2001', 'ymw16', 'ne2025'],
                        help='三条推断线之一：河外 DM 取对应的 dm_exc_<模型> 列。'
                             '宿主(对数正态)/IGM(准高斯)/晕(均匀窗)模型三线完全共用')
    parser.add_argument('--config', action='store', dest='config_path', type=str,
                        default='config_mini.json', help='Path to config.json')

    args = parser.parse_args()
    t_total0 = time.perf_counter()
    fcat = args.fcat
    fsvy = args.fsvy
    fout = args.fout
    use_fits = args.use_fits

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

    # 三条推断线之间唯一的差别就是这一列选择（dm_exc = DM_obs - DM_MW(<模型>)，
    # ISM/厚盘由目录扣除，晕不在其中 → 仍走 U[DMsmin,DMsmax] 均匀边际化）。
    # 宿主 DM 已是第 7/8 维自由参数，不再按星系类型或 MW 模型分组标定，所以线间
    # 差异是纯 MW 效应，三条线的 ln Z 可以直接横比。
    vDME = {'ne2001': vDM_ne2001, 'ymw16': vDM_ymw16, 'ne2025': vDM_ne2025}[args.mwmodel]
    if vDME is None:
        raise ValueError("filtered.fits 无 dm_exc_ne2025 列: "
                         "先在服务器跑 preprocess_catalog.py（勿加 --skip-ne2025）")
    # NaN/Inf 一律当场炸：NaN 进 IntDMsrc_LogNormal 后 np.maximum 仍是 NaN，最后被
    # log_distr_efdmw 的 np.where(tot>0,...,-1e30) 吞成常数平台，采样器照样"收敛"
    # 出一个平坦后验（V27 钉的就是这类静默失败）。
    if not np.all(np.isfinite(vDME)):
        n_bad = int(np.sum(~np.isfinite(vDME)))
        raise ValueError(f"dm_exc_{args.mwmodel} 有 {n_bad} 个非有限值(NaN/Inf)，"
                         f"禁止带入似然；重跑 preprocess_catalog.py 并检查 NE2025 是否失败")
    print(f'[samp] MW model = {args.mwmodel} (宿主对数正态 + IGM 准高斯 + 晕均匀窗, 8 维)')

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

    # 先验范围（从 config.json 读取）。名字刻意不叫 vpara/vparb：已删除的宿主
    # 标定参数组就叫 vpar_*，同根命名会让人以为那套标定还在（2026-10-11 改名）。
    cfg = config['prior']
    par_lo = np.array([cfg[k][0] for k in PARAM_ORDER])
    par_hi = np.array([cfg[k][1] for k in PARAM_ORDER])

    print('------------par range-----------')
    print('lower:', par_lo)
    print('upper:', par_hi)
    a1 = time.perf_counter()
    # sanity check 取先验盒【中心】u=0.5：u=0 是盒子角点（γ=-4, σ_w=0.01,
    # ln_sigma_host=0.2），实测每个事件都返回 -1e30 ⇒ 打印出来的 -1.688e+33 不携带
    # 任何信息，白跑 10 s。中心点给的是一个可读的有限值，能真正当 smoke test 用。
    # （通过 myprior 变换后再求似然：par_lo[0] 是 log_phi0，不能直送 lnlik）
    cube_test = np.full(len(par_lo), 0.5)
    myprior(cube_test, len(par_lo), len(par_lo))
    print('lnlik @ 先验盒中心 =', myloglike(cube_test, len(par_lo), len(par_lo)))
    a2 = time.perf_counter()
    print(f'Single eval time: {a2-a1:.3f}s')
    print("Running Nest Sampling ...")
    # run MultiNest
    import os
    import scipy
    # -o 含 / 就是全路径（run 脚本传 $RUN_DIR/<名>），直接用；裸名才拼传统平铺
    # 前缀——手工单跑行为不变（A3 配套）。
    if '/' in fout or os.sep in fout:
        basename = fout
    else:
        basename = config['output']['nest_out_dir'] + 'samp/' + fout
    os.makedirs(os.path.dirname(os.path.abspath(basename)), exist_ok=True)
    skw = sampler_kwargs(config)
    doc = {
        'line': os.path.basename(fout),
        'mw': args.mwmodel,
        'run_tag': os.environ.get('CC_RUN_TAG', ''),
        'argv': sys.argv,
        'config': config,
        'config_md5': config_md5(config),
        'input_md5': {os.path.basename(fcat): file_md5(fcat),
                      os.path.basename(fsvy): file_md5(fsvy)},
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
    print(f"[samp] MultiNest 总耗时 {wall:.1f}s ({wall/60:.1f} min)")
    doc.update(end_utc=utc_now(), wall_seconds=round(wall, 1),
               n_iter=count_lines(str(basename) + 'ev.dat'),
               n_post=count_lines(str(basename) + 'post_equal_weights.dat'),
               n_eval=LLGUARD.n_eval, n_llfail=LLGUARD.n_fail)
    write_manifest(manifest_path(basename), doc)
