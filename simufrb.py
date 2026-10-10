import numpy as np
import time
import argparse

from frb_util import *
# 放在星号导入之后：chime_selection 的两个名字必须赢过任何同名导出
from chime_selection import detection_probability, resolve_selection

dis = AstroDistribution()
cos = Cosmology()
er = EventRate()
tel = Telescope()
lf = Loadfiles()

def Simu_FRBs_Eiso(phis, gamma, logEs, logE0, mu, sigma, dnu, ns, fov, npol, g, tsys, bw, sn0,
                   ln_mu_host=None, ln_sigma_host=None, poisson_survival=None,
                   sel_mode='step', sel_k=1.0):
    """基于能量 Schechter 函数的 FRB 模拟器

    采样流程：
    1. 从截断 Schechter 函数采样 E_iso
    2. 从共动体积分布采样 z
    3. 从 log-normal 分布采样 w_rest
    4. 计算 fluence: F = E_iso * (1+z)^(2+α) / (dnu * 4π * D_L² * Jyms2CGS)，谱指数 alpha_spec = -1.39 (Shin 2023)
    5. 计算流量: S = F / w_obs
    6. 检测: S > S_min(w_obs) 且 E_obs = E_iso·ε >= E0（观测能量口径，与推断端
       log_IntBeam / Norm1D_E 的 E0 截断一致，见循环内注释）

    sel_mode='step'  即上述 0/1 辐射计阈值判据（历史基线，所有回归钉按此标定）。
    sel_mode='smooth' 改用 McGregor 2026 实测的平滑探测概率 P(det|F, w_intrinsic)，
      见 chime_selection 模块。两种模式的 E0 截断与 ε 波束效率完全相同。
    """
    if sel_mode not in ('step', 'smooth'):
        raise ValueError("sel_mode 需为 'step' 或 'smooth'，得到 %r" % sel_mode)
    if poisson_survival is None:
        raise ValueError("Simu_FRBs_Eiso 需要显式 poisson_survival（filtered.fits header 的 "
                         "SURVIV，见 frb_util.read_selection_survival）：到达时间率必须是"
                         "筛选后事件率，缺省 1.0 会静默复现选择函数缺口")
    if not (0.0 < poisson_survival <= 1.0):
        raise ValueError("poisson_survival 需在 (0,1]，得到 %r" % poisson_survival)
    res = np.zeros((ns, 10))
    ns0 = ns            # 首批 = ns：step 模式的随机数流与历史基线逐位一致
    ns_orig = ns
    nt = 0
    n_cand = 0          # smooth 模式批量自适应的累计候选/接受计数
    n_acc = 0
    # 到达时间率 = 筛选后事件率（与 nest_*_mini 的泊松项同一定义）
    lamda = er.rate_2d_E(sn0, bw, npol, g, tsys, dnu, phis, gamma, logEs, logE0, mu, sigma) * fov \
        * poisson_survival
    # loud-fail: 单位/量纲搞错时宁可当场报错, 也不要产出一列垃圾到达时间。
    # （ppcheck 曾经把已经是线性值的 phis 再取 10** → inf → lamda=inf
    #  → exponential(1/lamda)=exponential(0) 全部退化成 0, 而统计量不碰 T 列,
    #   所以静默无误 —— 这类错误必须拦在这里。）
    if not np.isfinite(lamda) or lamda <= 0:
        raise ValueError(f"lamda={lamda} 非法（phis={phis}, fov={fov}, "
                         f"gamma={gamma}, logEs={logEs}）；检查参数单位是否为线性 phis")
    if sel_mode == 'smooth':
        # lamda 走 er.rate_2d_E → 按 ad.sel_mode 分派到 rate_2d_E_S，与逐事件接受
        # 判据 detection_probability 同一个 S（旧 P3 的"率按 step、接受按 S"已消除）。
        # 仍要提醒的是【水平】而非口径：k_offset 无标定依据、单 logistic 低估暗端
        # S 约 2×（chime_selection STATUS），所以 smooth 的 Λ/T 只保证自洽，不保证
        # 是可信的绝对率。
        print("[simufrb][warn] smooth: lamda/T 列与事件同用实测 S（自洽），但 S 的"
              "绝对水平未经 k_offset 标定——勿把 smooth mock 的 T 列当校准率用。")
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
        # gamma < -1 时 Schechter_E_log ∝ (E/E*)^(gamma+1)·exp(-E/E*) 单调递减，
        # 密度峰值落在低能端。若网格下限取 logE0-1 而 Sampling1D 的接受域是
        # [logE0, 47]，归一化 ymax 就被窗口外(logE0-1)那个更高的点主导：
        # gamma=-2.6 时 y(logE0)/ymax = 10^(gamma+1) = 2.5%，接受率耗尽 10000 次迭代。
        # 把下限挪到 logE0（= 接受域左端点）使 ymax 落在支撑内，分布不变
        # （[logE0,47] 上的网格取值完全相同，只改归一化常数），接受率提升 ~40 倍。
        vlogE = np.arange(logE0, 48., (48. - logE0) / 10000)
        # E 网格上界写死 47/48, 依赖"指数拐在上界之上有 100 倍以上余量"这一假设：
        # logEs=44 时 10^47 = 1000·E*, 尾部 exp(-1000) 可忽略。config 先验上界是 44,
        # 这条断言把假设写死, 将来抬高 log_Echar 先验时会当场报错而不是静默丢质量。
        if not (logE0 < 47. and logEs <= 45.5):
            raise ValueError(f"E 采样网格[logE0,47] 覆盖不足: logE0={logE0}, logEs={logEs}"
                             f"（要求 logE0<47 且 logEs<=45.5）")
        vlik = dis.Schechter_E_log(vlogE, 1, gamma, logEs)
        vlogE = Sampling1D(vlogE, vlik, logE0, 47, ns0)
        vEiso = np.power(10., vlogE)

        # 采样 ε（波束效率，0.5-1）
        vlnEps = np.random.uniform(-np.log(2), 0, ns0)
        vEps = np.exp(vlnEps)

        # 采样星系红移（加入 SFR 宇宙学演化权重 + 1/(1+z) 时间膨胀因子）
        # p(z) ∝ dV/dz · R(z) / (1+z)，与推断端 rate_2d_E 的 (dV/dz)/(1+z)·evolution 一致
        vZg = np.arange(0, 5.1, 5.1 / 10000)
        vlik = dis.Distribution_volume(vZg) * dis.evolution_factor(vZg)
        vZ = Sampling1D(vZg, vlik, 0, 5.0, ns0)

        # 采样脉冲宽度（静止系）
        # 网格与接受域必须由注入的 (mu, sigma) 推导，不能硬编码：旧写法固定用
        # [-0.5,1.5]/[-0.4,1.4]，对 mu=0.4,sigma=0.3 只截 0.4%，但先验允许
        # mu_w 到 2.0、sigma_w 到 1.0 —— sigma_w=1.0 时截掉 37%（两端各 ~20%），
        # mu_w=1.2 时截掉上端 25%，mock 宽度分布被【静默截断】，注入-恢复与 PPC
        # 会出现假偏差（且截断同时改变探测选择函数，污染的不只是宽度列）。
        # 对数正态取 μ±6σ 已覆盖 1-1e-9 的质量，等效无截断。
        w_lo, w_hi = mu - 6. * sigma, mu + 6. * sigma
        vlogW0 = np.arange(w_lo, w_hi, (w_hi - w_lo) / 10000)
        vlik = dis.dis_logw(vlogW0, mu, sigma)
        vlogW0 = Sampling1D(vlogW0, vlik, w_lo, w_hi, ns0)
        vW0 = np.power(10., vlogW0)       # 静止系宽度 [ms]
        vW = vW0 * (1 + vZ)               # 观测系宽度 [ms]

        # 采样宿主星系 DM：观测帧对数正态，中位数 = exp(ln_mu_host)·(1+z)^host_dm_evol
        # （Zhang 2020 Eq.5/6）。观测帧量，直接相加，不再除 (1+z)。
        if ln_mu_host is None or ln_sigma_host is None:
            raise ValueError("Simu_FRBs_Eiso 需要 ln_mu_host / ln_sigma_host")
        ln_mu_z = ln_mu_host + dis.host_dm_evol * np.log1p(vZ)
        vDMH = np.random.lognormal(ln_mu_z, ln_sigma_host, ns0)

        # 采样银河系晕 DM: DM_MW,halo ~ U[DMsmin, DMsmax]，观测帧（z≈0）
        # DM_MW = DM_MW,ISM + DM_MW,halo；ISM 部分真实数据由目录 dm_exc 列扣除，
        # mock 的 DMe 与之对应(不含 ISM)。与推断端 log_distr_efdmwz 卷积窗口同源，
        # 改值只改 config 的 dm_model.halo。
        vDMHalo = np.random.uniform(dis.DMsmin, dis.DMsmax, ns0)

        # 计算 IGM DM：均值(Macquart Eq.2 = James Eq.6 = Shin A12) × 准高斯涨落 Δ。
        # Δ 从与推断端同一张 CDF 表反抽样（igm_delta_sample），两端单一数据源；
        # 历史上"两端各自实现"是本仓库的 bug 类别（frb_util.apply_dm_model 注释）。
        vDMI = cos.DispersionMeasure_IGM(vZ) * dis.igm_delta_sample(vZ, ns0, None)

        # 外星系 DM：三项全部在观测帧，直接相加
        vDME = vDMH + vDMI + vDMHalo

        # 检测阈值
        vft = tel.RMEq(sn0, g, tsys, npol, bw, vW)

        # 从 E_iso 计算 fluence: F = E_iso * (1+z)^(2+α) / (dnu * 4π * D_L² * Jyms2CGS)，α = -1.39 (Shin 2023)
        # 使用 Energy_to_Flu 反推。Energy_to_Flu 是逐元素 ufunc，向量化与 python
        # 循环逐位一致（V20 钉住）；循环写法是 smooth 模式 ~90% 的耗时
        # （低接受率下每轮上万次标量调用）。
        vFlu = cos.Energy_to_Flu(vZ, vEiso * vEps, dnu)

        # 从 fluence 计算流量: S = F / w_obs
        vFlux = vFlu / vW   # S = F / w_obs [Jy]

        # E0 截断必须与推断端同口径：推断端 log_IntBeam 的 logl0 硬切 +
        # Norm1D_E 的 max(E_min, E0) clamp 把 E0 截在【观测能量】E_obs = E_int·ε 上
        # （对每个假设 z 成立），此前模拟端只截内禀能量（E_int >= E0），近邻低
        # E_int 事件会带着 E_obs < E0 进 mock，被推断端判为低 z 不可能，z 支撑被
        # 硬性抬高（注入真值下 2000 个 mock 约 2 个，z-floor ≈ z_true）。
        # 补上这一刀后接受集 = {E_obs > max(阈值, E0)}，与选择函数完全一致。
        vEobs = vEiso * vEps
        if sel_mode == 'step':
            vdet = (vFlux > vft) & (vEobs >= 10. ** logE0)
        else:
            # 平滑判据。自变量是【内禀】宽度 vW0（选择函数的 width 特征即注入用的
            # 静止系高斯宽度），不是观测系 vW=vW0(1+z)；vft/vFlux 这条硬阈值不再
            # 参与判据，res[:,4] 的 thres 列仍按辐射计公式写出以保持列契约。
            # 只在 smooth 分支消耗随机数：step 分支必须与历史基线逐位一致。
            # S 作用在【真】fluence 上——波束平均的探测概率是 F_true 的函数
            # （2026-10-05 修正：旧写法 S(vFlu) 把探测概率算在了波束衰减之后）。
            # vFluTrue 直接由 E_iso 重算而非 vFlu/vEps 反除（浮点 (a·b)/b ≠ a）；
            # vFlu（观测值，喂 flux 列）完全不动 → step 分支逐位不变。
            vFluTrue = cos.Energy_to_Flu(vZ, vEiso, dnu)
            vdet = (np.random.random(ns0) < detection_probability(vFluTrue, vW0, sel_k)) \
                & (vEobs >= 10. ** logE0)

        nlen = len(vFlux[vdet])
        n_cand += ns0
        n_acc += nlen

        # 空集早退：本轮无事件通过阈值，跳过避免 vT 计算异常
        # （分母用 ns_orig：smooth 模式下 ns0 已被自适应抬到批量上限，
        #   打 ns0 会输出 5/500000 这类误导数字——P4，2026-10-05 修正）
        if nlen == 0:
            print(f"[warn] 第 {iter_cnt} 轮采样无事件通过阈值，继续重试 (nt={nt}/{ns_orig})")
            continue

        # 采样事件到达时间（Poisson 过程，间隔服从指数分布）
        larray = np.repeat(lamda, nlen)
        vT = np.random.exponential(1.0 / larray)
        if nlen > ns:
            nlen = ns

        res[nt:(nt + nlen), 0] = vFlux[vdet][0:nlen]
        res[nt:(nt + nlen), 1] = vW[vdet][0:nlen]
        # vT 长度 = nlen（通过阈值的事件数），与 vFlux[vdet][0:nlen] 等对齐
        # vT 是独立同分布的指数到达时间，与具体哪个事件通过阈值无关
        res[nt:(nt + nlen), 2] = vT[0:nlen]
        res[nt:(nt + nlen), 3] = vDME[vdet][0:nlen]
        res[nt:(nt + nlen), 4] = vft[vdet][0:nlen]
        res[nt:(nt + nlen), 5] = vlogE[vdet][0:nlen]  # logE 替代 logL
        res[nt:(nt + nlen), 6] = vZ[vdet][0:nlen]
        res[nt:(nt + nlen), 7] = vDMI[vdet][0:nlen]
        res[nt:(nt + nlen), 8] = vDMH[vdet][0:nlen]
        res[nt:(nt + nlen), 9] = vDMHalo[vdet][0:nlen]
        nt = nt + nlen
        ns = ns - nlen
        # 平滑模式批量自适应：低接受率下固定 ns0 批会使迭代数 ~(ns/acc)/ns0
        # 爆炸并被 max_iter 截断（面值 S 在 V18 测试点接受率 ~4e-4，实测被截到
        # 8187/20000）。估计量带 1/(n_cand+1) 下限：早期全拒批次也能让批量
        # 几何增长收敛；上限 50 万候选约束内存（实测进程峰值 ~163MB，含
        # Energy_to_Flu/Luminosity_Distance 的 ufunc 临时数组——P5，
        # 2026-10-05 实测；服务器有 per-process 限制时调低此上限）。
        # step 模式不触碰——历史基线的随机数流逐位不变。
        if sel_mode == 'smooth':
            est = (n_acc / n_cand) if n_acc > 0 else 1.0 / (n_cand + 1.0)
            ns0 = int(np.clip(ns / est, ns_orig, 500_000))
        pct = float(nt) / ns_orig * 100
        print(f"{pct:0.1f}% mock FRBs have been simulated.")
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FRB sample simulator (E_iso energy function)')
    parser.add_argument('-ns', action='store', dest='Ns', type=int, help='FRB number')
    parser.add_argument('-phis', action='store', dest='phis', type=float,
                        help='Characteristic event rate density [Gpc^-3 yr^-1]')
    parser.add_argument('-gamma', action='store', dest='gamma', type=float,
                        help='Schechter differential power-law index (Shin 2023 alpha)')
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
    parser.add_argument('-mu_host', action='store', dest='mu_host', type=float,
                        help='ln-space median of observed-frame host DM at z=0')
    parser.add_argument('-sigma_host', action='store', dest='sigma_host', type=float,
                        help='ln-space width of host DM lognormal')
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
    gamma = args.gamma
    logEs = args.logEs
    logE0 = args.logE0
    dnu = args.dnu
    mu = args.mu
    sigma = args.sigma
    ln_mu_host = args.mu_host
    ln_sigma_host = args.sigma_host
    gain = args.gain
    npol = args.npol
    bw = args.bw
    Ts = args.tsys
    sn0 = args.sn0
    fov = args.fov

    output = args.output
    Tobs = args.Tobs

    # 加载配置并一次性注入宇宙学 + dm_model（重绑模块全局 dis/cos/er，
    # Simu_FRBs_Eiso 用的就是这三个对象；过去这里漏传 H0/Omega_b）
    config = load_config(args.config_path)
    cos, dis, er = build_models(config)

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

    # 选择函数模式：config['selection'] 或 env CC_SEL_MODEL / CC_SEL_K（缺省 step=历史基线）
    sel_mode, sel_k = resolve_selection(config)
    print(f"[simufrb] selection mode = {sel_mode}, k_offset = {sel_k}")
    # 到达时间率 = 筛选后事件率（与推断端泊松项同一定义）。
    # 模式先定，SURVIV 的分母才跟着变（smooth→NSNR12）；且 lamda 用的
    # er.rate_2d_E 会按 er.ad.sel_mode 自动分派到 rate_2d_E_S ⇒ 平滑模式下
    # T 列与事件数同口径（旧 P3 的"率按 step、接受按 S"不自洽由此消除）。
    poisson_survival = read_selection_survival(config['data']['catalog_path'], mode=sel_mode)
    res = Simu_FRBs_Eiso(phis, gamma, logEs, logE0, mu, sigma, dnu, Ns, fov, npol, gain, Ts, bw, sn0,
                         ln_mu_host=ln_mu_host, ln_sigma_host=ln_sigma_host,
                         poisson_survival=poisson_survival,
                         sel_mode=sel_mode, sel_k=sel_k)
    header = f"T_obs {Tobs}\nS W T DMe thres logE Z DMi DMh DMmw"
    np.savetxt(output, res, delimiter=' ', header=header, comments="#")
    print(f"[simufrb] 生成 {Ns} 个 mock FRB, 总耗时 {time.perf_counter()-t_simu0:.1f}s")
