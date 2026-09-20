import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
import time
import warnings
import sys
import argparse

from frb_util import *

dis = AstroDistribution()
cos = Cosmology()
er = EventRate()
tel = Telescope()
lf = Loadfiles()

def Simu_FRBs_Eiso(phis, alpha, logEs, logE0, mu, sigma, dnu, ns, fov, npol, g, tsys, bw, sn0, fgt='ETG'):
    """基于能量 Schechter 函数的 FRB 模拟器

    采样流程：
    1. 从截断 Schechter 函数采样 E_iso
    2. 从共动体积分布采样 z
    3. 从 log-normal 分布采样 w_rest
    4. 计算 fluence: F = E_iso * (1+z)^(2+α) / (dnu * 4π * D_L² * Jyms2CGS)，谱指数 α = -1.39 (Shin 2023)
    5. 计算流量: S = F / w_obs
    """
    res = np.zeros((ns, 10))
    ns0 = ns
    nt = 0
    lamda = er.rate_2d_E(sn0, bw, npol, g, tsys, dnu, phis, alpha, logEs, logE0, mu, sigma) * fov
    max_iter = 1000   # 防止极端参数下检测率过低导致死循环
    iter_cnt = 0
    while ns > 0:
        iter_cnt += 1
        if iter_cnt > max_iter:
            print(f"[warn] 达到最大迭代次数 {max_iter}，已生成 {nt}/{ns0} 个 FRB；"
                  f"可能参数导致检测率过低，提前终止")
            res = res[:nt]   # 截断到实际生成数量，避免残留零行
            break
        # 采样 E_iso（截断 Schechter 能量函数）
        # 注意：当 alpha < -1 时，Schechter_log 在低能端 (l/ls)^(alpha+1) 发散，
        # Sampling1D 用 y/ymax 归一化会使采样密度强烈集中在低能端。
        # 这是 alpha<-1 Schechter 函数的固有数学性质，非代码 bug。
        # 后果：mock 样本的 E_iso 分布会偏低能端，恢复检验时需关注此偏差。
        # 若需抑制此效应，可将采样下限设为 logE0（而非 logE0-1）以避开发散区。
        vlogE = np.arange(logE0 - 1., 48., (48. - logE0 + 1.) / 10000)
        vlik = dis.Schechter_E_log(vlogE, 1, alpha, logEs)
        vlogE = Sampling1D(vlogE, vlik, logE0, 47, ns0)
        vEiso = np.power(10., vlogE)

        # 采样 ε（波束效率，0.5-1）
        vlnEps = np.random.uniform(-np.log(2), 0, ns0)
        vEps = np.exp(vlnEps)

        # 采样星系红移（加入 SFR 宇宙学演化权重 + 1/(1+z) 时间膨胀因子）
        # p(z) ∝ dV/dz · R(z) / (1+z)，与推断端 rate_2d_E 的 (dV/dz)/(1+z)·evolution 一致
        vZg = np.arange(0, 5.1, 5.1 / 10000)
        vlik = dis.Distribution_volume(vZg) * dis.evolution_factor(vZg) / (1.0 + vZg)
        vZ = Sampling1D(vZg, vlik, 0, 5.0, ns0)

        # 采样脉冲宽度（静止系）
        vlogW0 = np.arange(-0.5, 1.5, 2. / 10000)
        vlik = dis.dis_logw(vlogW0, mu, sigma)
        vlogW0 = Sampling1D(vlogW0, vlik, -0.4, 1.4, ns0)
        vW0 = np.power(10., vlogW0)       # 静止系宽度 [ms]
        vW = vW0 * (1 + vZ)               # 观测系宽度 [ms]

        # 采样宿主星系 DM
        vDMH0 = np.arange(0, 5001., 5001. / 10000)
        vlik = dis.Distribution_HostGalaxyDM(vDMH0, fgalaxy_type=fgt)
        vDMH0 = Sampling1D(vDMH0, vlik, 0, 5000, ns0)
        # 宿主 DM 随红移演化：DM_host(z) = DM_host(0)·sqrt(SFR(z)/SFR(0))
        # 统一用 SFR_evolution（Yuksel 2008），与推断端 kappa(z) 保持同一曲线
        vDMH = vDMH0 * np.sqrt(dis.SFR_evolution(vZ)) / np.sqrt(dis.SFR_evolution(0))

        # 采样银河系晕 DM: DM_MW,halo ~ U[0, DMsmax]，观测帧（z≈0）
        # DM_MW = DM_MW,ISM + DM_MW,halo；ISM 部分真实数据由目录 dm_exc 列扣除，
        # mock 的 DMe 与之对应(不含 ISM)。与推断端 log_distr_efdmwz 卷积窗口同源，
        # 改值只改 frb_util.DMsmax。
        vDMHalo = np.random.uniform(0, dis.DMsmax, ns0)

        # 计算 IGM DM
        vDMI = cos.DispersionMeasure_IGM(vZ)

        # 外星系 DM（DM_MW,halo 位于观测帧，直接相加，不除 (1+z)）
        vDME = vDMH / (1 + vZ) + vDMI + vDMHalo

        # 检测阈值
        vft = tel.RMEq(sn0, g, tsys, npol, bw, vW)

        # 从 E_iso 计算 fluence: F = E_iso * (1+z)^(2+α) / (dnu * 4π * D_L² * Jyms2CGS)，α = -1.39 (Shin 2023)
        # 使用 Energy_to_Flu 反推
        vFlu = np.array([cos.Energy_to_Flu(z, e, dnu) for z, e in zip(vZ, vEiso * vEps)])

        # 从 fluence 计算流量: S = F / w_obs
        vFlux = vFlu / vW   # S = F / w_obs [Jy]

        nlen = len(vFlux[vFlux > vft])

        # 空集早退：本轮无事件通过阈值，跳过避免 vT 计算异常
        if nlen == 0:
            print(f"[warn] 第 {iter_cnt} 轮采样无事件通过阈值，继续重试 (nt={nt}/{ns0})")
            continue

        # 采样事件到达时间（Poisson 过程，间隔服从指数分布）
        larray = np.repeat(lamda, nlen)
        vT = np.random.exponential(1.0 / larray)
        if nlen > ns:
            nlen = ns

        res[nt:(nt + nlen), 0] = vFlux[vFlux > vft][0:nlen]
        res[nt:(nt + nlen), 1] = vW[vFlux > vft][0:nlen]
        # vT 长度 = nlen（通过阈值的事件数），与 vFlux[vFlux>vft][0:nlen] 等对齐
        # vT 是独立同分布的指数到达时间，与具体哪个事件通过阈值无关
        res[nt:(nt + nlen), 2] = vT[0:nlen]
        res[nt:(nt + nlen), 3] = vDME[vFlux > vft][0:nlen]
        res[nt:(nt + nlen), 4] = vft[vFlux > vft][0:nlen]
        res[nt:(nt + nlen), 5] = vlogE[vFlux > vft][0:nlen]  # logE 替代 logL
        res[nt:(nt + nlen), 6] = vZ[vFlux > vft][0:nlen]
        res[nt:(nt + nlen), 7] = vDMI[vFlux > vft][0:nlen]
        res[nt:(nt + nlen), 8] = vDMH[vFlux > vft][0:nlen]
        res[nt:(nt + nlen), 9] = vDMHalo[vFlux > vft][0:nlen]
        nt = nt + nlen
        ns = ns - nlen
        pct = float(nt) / ns0 * 100
        print(f"{pct:0.1f}% mock FRBs have been simulated.")
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FRB sample simulator (E_iso energy function)')
    parser.add_argument('-ns', action='store', dest='Ns', type=int, help='FRB number')
    parser.add_argument('-phis', action='store', dest='phis', type=float,
                        help='Characteristic event rate density [Gpc^-3 yr^-1]')
    parser.add_argument('-alpha', action='store', dest='alpha', type=float,
                        help='Power-law index of energy function')
    parser.add_argument('-logEs', action='store', dest='logEs', type=float,
                        help='log10 of characteristic energy E* [erg]')
    parser.add_argument('-logE0', action='store', dest='logE0', type=float,
                        help='log10 of lower energy cutoff E0 [erg]')
    parser.add_argument('-dnu', action='store', dest='dnu', type=float,
                        help='观测带宽 Δν_obs [MHz] (CHIME: 400 MHz)')
    parser.add_argument('-mu', action='store', dest='mu', type=float,
                        help='Mean of logarithmic intrinsic width distribution')
    parser.add_argument('-sig', action='store', dest='sigma', type=float,
                        help='Std dev of logarithmic intrinsic width distribution')
    parser.add_argument('-fgt', action='store', dest='fgt', type=str,
                        help='Host galaxy type')
    parser.add_argument('-ga', action='store', dest='gain', type=float,
                        help='Telescope gain [K/Jy]')
    parser.add_argument('-npol', action='store', dest='npol', type=int,
                        help='Polarization channel number')
    parser.add_argument('-bw', action='store', dest='bw', type=float,
                        help='Bandwidth [MHz]')
    parser.add_argument('-ts', action='store', dest='tsys', type=float,
                        help='System temperature [K]')
    parser.add_argument('-sn0', action='store', dest='sn0', type=float,
                        help='Detection threshold SNR')
    parser.add_argument('-fov', action='store', dest='fov', type=float,
                        help='Field of view [deg^2]')
    parser.add_argument('-out', action='store', dest='output', help='Output file path')
    parser.add_argument('-Tobs', action='store', dest='Tobs', type=float,
                        help='Total observation time [hr] (written to output header for nest_simu_mini.py)')
    parser.add_argument('--config', action='store', dest='config_path', type=str,
                        default='config_mini.json', help='Path to config file')

    args = parser.parse_args()
    Ns = args.Ns
    phis = args.phis
    alpha = args.alpha
    logEs = args.logEs
    logE0 = args.logE0
    dnu = args.dnu
    mu = args.mu
    sigma = args.sigma
    fgt = args.fgt
    gain = args.gain
    npol = args.npol
    bw = args.bw
    Ts = args.tsys
    sn0 = args.sn0
    fov = args.fov

    output = args.output
    Tobs = args.Tobs

    # 加载配置并同步宇宙学参数到 dis / er（与 nest_samp/simu_mini.py 保持一致）
    config = load_config(args.config_path)
    cosmo_cfg = config.get('cosmology', {})
    cos = Cosmology(omegam=cosmo_cfg.get('Omega_m', 0.308),
                    omegal=cosmo_cfg.get('Omega_L', 0.692))
    dis.cos = cos
    er.cos = cos
    er.ad = dis

    # 如果未指定 Tobs，从 tel_svy.txt 读 CHIME 的 TIME
    if Tobs is None:
        fsvy = config['data']['survey_info_path']
        svy_info = lf.LoadSvyInfo(fsvy)
        survey_names = list(svy_info['SURVEY'])
        if 'CHIME' in survey_names:
            Tobs = float(svy_info['TIME'][survey_names.index('CHIME')])
            print(f"[info] T_obs 未指定，从 {fsvy} 读取 CHIME TIME = {Tobs} hr")
        else:
            raise ValueError(f"T_obs 未指定且 {fsvy} 中无 CHIME，请用 -Tobs 指定观测时长")
    print(f"[info] T_obs = {Tobs} hr, Ns = {Ns}")
    t_simu0 = time.perf_counter()

    res = Simu_FRBs_Eiso(phis, alpha, logEs, logE0, mu, sigma, dnu, Ns, fov, npol, gain, Ts, bw, sn0, fgt=fgt)
    header = f"T_obs {Tobs}\nS W T DMe thres logE Z DMi DMh DMmw"
    np.savetxt(output, res, delimiter=' ', header=header, comments="#")
    print(f"[simufrb] 生成 {Ns} 个 mock FRB, 总耗时 {time.perf_counter()-t_simu0:.1f}s")
