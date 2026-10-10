import numpy as np
import chime_selection
from scipy import integrate
from scipy.interpolate import interp1d
import scipy.special as spf
import json
import os
from pathlib import Path

def load_config(config_path='config_mini.json'):
    """加载全局配置文件

    路径解析规则：
    - 绝对路径：直接使用
    - 相对路径：基于当前工作目录(CWD)解析，而非脚本所在目录
    默认文件名 config_mini.json 对应本项目实际使用的配置文件。
    """
    p = Path(config_path)
    if not p.is_absolute():
        p = Path.cwd() / p
    with open(p, encoding='utf-8') as f:
        return json.load(f)

# FRB 谱指数（k 修正用），参考 Shin 2023 的最佳拟合中心值。
# 注意 Shin 2023 中该量是【拟合参数】，α = -1.39 +0.86/-1.19；本仓库取中心值固定。
# 命名规则见《代码改进方案.md》附录 A：谱指数一律 alpha_spec，Schechter 指数一律 gamma。
alpha_spec = -1.39

# 参数向量的唯一权威顺序（附录 A 规则 R3）。
# 任何脚本的标签、绘图范围、n_params 都必须从这里取，禁止硬编码索引。
PARAM_ORDER = ['log_phi0', 'gamma', 'log_Echar', 'log_E0',
               'mu_w', 'sigma_w', 'ln_mu_host', 'ln_sigma_host']
# 只报 95% 上限的参数（替代 pltpost.py 里裸索引的 `vari != 3` 特判）
PARAM_UPPER_ONLY = {'log_E0'}
N_PARAMS = len(PARAM_ORDER)

# 望远镜参数库（硬编码，方便后续扩展）
# bw: 辐射计方程用的【实际观测】带宽 [MHz]（CHIME 400-800 MHz 带）
# dnu: 能量定义用的【参考】带宽 [MHz]。与 bw 是两个层面：bw 决定仪器响应与
#   选择效应，dnu 只是把观测 fluence 换算成"与望远镜无关、可跨实验比较"的能量。
#   取 1 GHz 与 James et al. 2022b §3.1 / Shin et al. 2023 Eq.(A4) 的约定一致
#   （两家均明写 "Δν taken as 1 GHz"），使 log_Echar 可直接与 Shin 的 41.38、
#   James 的 41.70 并列。改它只平移能量轴 +log10(dnu_new/dnu_old)，似然严格不变。
TELESCOPES = {
    'CHIME': {
        'g': 1.4,        # Gain [K/Jy]
        'bw': 400,       # Bandwidth [MHz] (观测帧, 辐射计方程用)
        'tsys': 50.0,    # System temperature [K]
        'fov': 200.0,    # Field of view [deg^2]
        'npol': 2,       # Polarization channels
        'sn0': 10.0,     # Detection threshold SNR
        'dnu': 1000.0,   # 能量定义参考带宽 Δν_ref [MHz]（非观测带宽，见上）
    },
    # 后续加新望远镜在这里添加：
    # 'Parkes': {'g': 0.6, 'bw': 300, 'tsys': 25, 'fov': 4.0, 'npol': 2, 'sn0': 10, 'dnu': 300},
}

class Cosmology:
    _cache = {}  # (omegam, omegal, omegab, h0) -> instance，避免重复积分

    def __init__(self, omegam=0.315, omegal=0.685, omegab=0.0493, h0=0.674):
        key = (omegam, omegal, omegab, h0)
        cached = Cosmology._cache.get(key)
        if cached is not None:
            self.__dict__.update(cached.__dict__)
            return
        self.Omega_m = omegam
        self.Omega_b = omegab
        self.Omega_L = omegal
        self.c = 2.9979245800e10
        self.pc2cm = 3.08567758e18
        self.km2cm = 1e5
        self.Mpc2cm = 3.08567758e24
        self.Gpc2cm = 3.08567758e27
        self.Jy2CGS = 1e-23
        self.MHz2Hz = 1e6
        self.Jyms2CGS = 1e-26
        # 默认值 = Planck 2018 results VI (A&A 641, A6)，即 Shin 2023 §1 所引的
        # "Planck Collaboration et al. 2020"；James 2022b §2.3 明写同一组
        # (H0=67.4, Omega_m=0.315, Omega_L=0.685) 且声明适用 0<=z<=5。
        # h0 可由 config 注入（cosmology.H0）。注意 Z21 用的是 67.74 —— 改 H0 是
        # 刻意的物理选择，且 DM_IGM ∝ Omega_b*h0、距离 ∝ 1/h0，会整体平移能量标度。
        self.h0 = h0
        self.H0 = self.h0 * 100 * self.km2cm / self.Mpc2cm
        self.Rhoc = 1.88 * self.h0 * self.h0 * 1e-29    #The critical density of universe in gram/cm^3
        self.Nc = self.Rhoc / 1.6726e-24       #The number density of universe in Hydrogen atom, in units of 1/cm^3
        # 弥散电离重子占宇宙重子的比例。沿用 Luo et al. 2018 公开代码
        # (frblf_erd/frb_util.py:24) 的取值，原代码与论文均未附出处。
        # 只有 Omega_b * f_IGM * yhe_corr 的乘积进入 <DM_cosmic>(z)，本仓库不做
        # 拆解；乘积由 V1 锚到 Z21 的 <DM_IGM>(z=1)=892 校验。
        self.f_IGM = 0.83

        self.vz = np.arange(-7, 6, 0.03)
        self.vz = np.power(10., self.vz)
        self.vz[0] = 0
        func = lambda z: self.c / self.H(z)
        self.vd = [ integrate.quad(func, 0, zv)[0] for zv in self.vz ]
        self.cd_interp = interp1d(self.vz, self.vd)
                
        func2 = lambda z: (1+z) * self.c / self.H(z) * self.Omega_b * self.Nc
        self.vdm = [ integrate.quad(func2, 0, zv)[0] for zv in self.vz ]
        self.dmigm = interp1d(self.vz, self.vdm)
        
        self.vld = self.Luminosity_Distance(self.vz)
        self.Ld2z = interp1d(self.vld, self.vz)
        self.Ld1 = self.Luminosity_Distance(1.0)
        self.Cd1 = self.Comoving_Distance(1.0)
        Cosmology._cache[key] = self
        
    def E(self, z):
        """
        logarithmic time derivative of scale factor
        """
        return np.sqrt(self.Omega_m * np.power(1 + z, 3.0) + self.Omega_L)
    
    def H(self, z):
        """
        calculate the Hubble ratio
        z: redshifts
        """
        return self.H0 * self.E(z)
    
    def Comoving_Distance(self, z):
        """
        calculate the comoving distance
        z: redshifts
        """
        return self.cd_interp(z)

    def dVdOdz(self,z):
        """
        calculate the diffrential comoving volume dV/dz/dOmega in units of Gpc^3
        z: redshifts
        """
        drdz = self.c / self.H(z)
        cd2 = self.Comoving_Distance(z)
        cd2 = cd2 * cd2
        dcv = cd2 * drdz / (self.Gpc2cm * self.Gpc2cm * self.Gpc2cm)
        return dcv
    
    def Luminosity_Distance(self, z):
        """
        calculate the luminosity distance in cm
        (注意: 返回单位是 cm, 不是 Mpc)
        z: redshifts
        """
        dl = (1 + z) * self.Comoving_Distance(z)
        return dl

    def Energy(self, z, flu=1.0, dnu=1000.):
        """计算各向同性能量 E_iso (Lin et al. 2024, ApJ 962, 73, Eq.7)

        公式: E = 4π D_L² · Δν_ref · F / (1+z)^(2+α)
        其中谱指数 α = -1.39 (Shin 2023 Table 1 最佳拟合)，即分母 (1+z)^0.61
        Δν_ref 为【能量定义参考带宽】(默认 1 GHz, 与 James 2022b §3.1 /
        Shin 2023 Eq.A4 一致)，不是 CHIME 的观测带宽(那个是 bw)。

        flu: fluence [Jy·ms]
        dnu: 参考带宽 Δν_ref [MHz]
        z: redshift
        """
        ld = self.Luminosity_Distance(z)
        ld2 = ld * ld
        ener = flu * self.Jyms2CGS * dnu * self.MHz2Hz * 4 * np.pi * ld2 / np.power(1+z, 2 + alpha_spec)
        return ener

    def DispersionMeasure_IGM(self, z, yhe_corr=7./8):
        """<DM_cosmic>(z)：理论积分 × 弥散电离重子份数 × 氦电子产额。

        yhe_corr = 1 - Y_He/2 = 0.875 (Y_He=0.25) 是单位重子质量提供的电子数
        (X + Y_He/2)，即 Macquart et al. 2020 Eq.2 链中的同一因子。形参原名 chi，
        易被误读成独立的电离度权重，故改名；取值沿用 Luo et al. 2018 代码。
        """
        return self.dmigm(z) * self.f_IGM * yhe_corr / self.Mpc2cm * 1e6

    def Energy_to_Flu(self, z, ener, dnu=1000.):
        """从内禀能量反推观测 fluence (Energy 的反函数)

        公式: F = E · (1+z)^(2+α) / (4π D_L² · Δν_ref)
        其中谱指数 α = -1.39 (Shin 2023)，即分子 (1+z)^0.61
        Δν_ref 为能量定义参考带宽（默认 1 GHz，见 Energy）。
        """
        ld = self.Luminosity_Distance(z)
        ld2 = ld*ld
        flu = ener * np.power(1+z, 2 + alpha_spec) / 4 / np.pi / ld2 / dnu / self.MHz2Hz / self.Jyms2CGS
        return flu

class Telescope:
    def __init__(self):
        self.MHz2Hz = 1e6
        self.ms2s = 1e-3

    def RMEq(self, snr, g, tsys, npol, bw, w):
        s = snr*tsys/g/np.sqrt(npol*bw*self.MHz2Hz*w*self.ms2s)
        return s

class AstroDistribution:
    def __init__(self):
        self.tel = Telescope()
        self.cos = Cosmology()
        # 宿主观测帧中位数演化指数（Zhang 2020 Eq.6 的 alpha，仅 ~1.4 sigma 显著，
        # 故主线固定 + 敏感性线 {0, 0.84, 1.84}）。由调用方从 config dm_model.host 覆盖。
        self.host_dm_evol = 0.84
        # Shin 2023 Eq(3) 的演化指数；网格扫描参数，不进参数向量（附录 A 规则 R5）
        self.n_evol = 1.0
        # 事件率演化所用的 SFR 曲线（MD2014=与 Shin 可比；Yuksel2008=改版前基线）
        self.sfr_curve = 'MD2014'
        # --- IGM 涨落：准高斯 Δ_DM ---
        # Macquart et al. 2020 (Nature 581, 391) 原始 = James et al. 2022b Eq.(8)
        # = Shin et al. 2023 Eq.(A16)，三处同式同参数（α=β=3 已逐字核对）。
        #   p(Δ) = κ Δ^(-β) exp[ -(Δ^(-α) - C0)^2 / (2 α^2 σ_DM^2) ],  Δ = DM_IGM/<DM_IGM>
        #   σ_DM(z) = igm_F · z^(-1/2)；C0 由 <Δ>=1 数值解；κ 归一化。
        # 实现要点（实测结论，见《IGM准高斯替换方案.md》§4）：
        #   - 积分变量用 t = 1/Δ（p_t ∝ t·exp(...)，t→0 端光滑）。在 u=Δ^(-α) 空间
        #     做均匀中点求积会把第一格低估 11.8 倍，使 C0 的解整体偏掉（2.0098 vs 1.58843）。
        #   - 节点用每层【质心】而非中点分位：质心在任意 N 下精确重现 <Δ>=1
        #     （实测 -0.000%），中点分位 N=64 偏 -2.14%（Δ^-3 重尾使最高分位点无界）。
        #   - <Δ>=1 是构造强制的，因此不再需要 m_ratio（已删除）。
        self.igm_F = 0.32          # Shin A16 下方 / James §2.3：fixed in all the fits
        self.igm_alpha = 3.0
        self.igm_beta = 3.0
        self.igm_zmin_eff = 0.01   # 对齐 James §2.5 网格下限；dV/dz ∝ z² 使更低处贡献 ~1e-6
        self.N_igm = 64            # 质心节点数；win 被积函数在 σ_DM<=0.40 处误差 <=0.5%
        self._IG_NT = 20000        # t 网格点数（C0 解对 NT 不敏感：2e4~2e6 五位小数不变）
        self._IG_N_SD = 96         # 抽样主表的 ln σ_DM 网格点数
        self._IG_N_U = 1024        # 抽样主表的分位点数
        self._ig_cache = {}        # 键 round(σ_DM,10) -> (C0, Δ升序, CDF, 累积一阶矩)
        self._ig_samp_tab = None   # (N_SD, N_U) 抽样主表，惰性构建
        self.Zmax = 5
        self.Zmin = 0.01           # 原 2e-6：σ_DM=F z^-1/2 在 z→0 发散；z<0.01 体积可忽略
        # MW 晕 DM 均匀窗 [DMsmin, DMsmax] [pc/cm^3]，宽度 = DMsmax - DMsmin。
        # NE2001 与 NE2025 均【不含晕组件】（NE2025 高银纬渐近 DM 仅由厚盘
        # n1*H1 ~ 24 pc/cm^3 给出，Ocker & Cordes 2026），故目录 dm_exc 残差中
        # 必然仍含完整银河系晕；对晕做均匀边际化，不是重复扣除。
        # 主线保持 U[0,60]（均值 30）以便与 P0 可比；系统误差线用 [20,80]（均值 50
        # = Z21 Fig4 取值；文献锚点 Dolag ~30 / Yamasaki 43(30-245) / P&Z 50-80）。
        # ⚠ 晕与 ln_mu_host 在【固定 z】处精确简并：窗口上移 Δ 等价于残差减 Δ,
        #   宿主少吸收 Δ；但晕不随 z 增长而宿主按 (1+z)^host_dm_evol 增长, 所以跨 z
        #   只是【近似】简并——补偿同一平移所需的 dln_mu_host 随 z 变化(Δ=20 时
        #   从 z=0.2 的 -0.73 到 z=2 的 -0.28, 见 V11)。数据理论上能用宿主项的
        #   z-形状分开两者, 但这份样本的 z 本身是从 DM 推出来的, 实际可分性待 8 维回收。
        #   结论不变：晕既不设为自由参数(一阶简并, 稀释采样), 也不取固定值
        #   (把不确定性从误差棒里抹掉), 只作为先验窗口的系统误差线报告。
        # Shin 2023 S5.3.1 扫 DMMW 50-100：仅 mu_host/sigma_host 移动且 < 自身 1 sigma。
        # 观测帧，不乘 (1+z)。
        self.DMsmin = 0.
        self.DMsmax = 60.
        # Wmin/Wmax 历史上是 Norm1D_E/rate_2d_E 的固定宽度网格边界；宽度网格改为
        # 按 (mu_w, sigma_w) 的 μ±6σ 自适应后（与模拟端同源），这两个值不再被
        # 选择函数使用，仅作历史参考保留。
        self.Wmax = 20
        self.Wmin = 0.05

    def sigma_dm_igm(self, z):
        """σ_DM(z) = igm_F · z^(-1/2)（James §2.3 / Shin A16 下方）。

        z 由 igm_zmin_eff 截底：σ_DM 在 z→0 发散，而 z<0.01 的共动体积贡献 ~1e-6。
        """
        z = np.maximum(np.asarray(z, dtype=float), self.igm_zmin_eff)
        return self.igm_F / np.sqrt(z)

    def _ig_tgrid(self, C0, sd):
        """t = 1/Δ 的均匀网格。t→0 端 p_t ∝ t 光滑，中点求积有效。

        高斯核在 u = t^3 = Δ^(-α) 空间的宽度是 α·σ_DM（指数分母 2α²σ²），
        不是 σ_DM —— 写成 σ 会让分布窄 3 倍（实测 sd_ln 0.12 vs 真值 0.33），
        而均值因质心构造仍被强制为 1，静默不报警。
        """
        sdu = self.igm_alpha * sd
        tmax = (max(C0, 0.0) + 40.0 * sdu) ** (1.0 / 3.0)
        h = tmax / self._IG_NT
        return (np.arange(self._IG_NT) + 0.5) * h, h

    def _ig_pt(self, C0, sd, t):
        """未归一 p_t(t) ∝ t·exp[-(t^3-C0)^2/(2 α² σ_DM²)]。

        指数平移因子只依赖 (C0, sd)，不依赖传入的 t 数组——否则网格矩与
        拒绝采样/抽样会各用一个任意尺度，静默偏 ~2%。
        """
        sdu = self.igm_alpha * sd
        tmax = (max(C0, 0.0) + 40.0 * sdu) ** (1.0 / 3.0)
        tmin = 0.5 * tmax / self._IG_NT
        tref = float(np.clip(np.cbrt(C0), tmin, tmax))   # np.cbrt 支持负 C0；C0**(1/3) 给复数
        ex = -(t ** 3 - C0) ** 2 / (2.0 * sdu ** 2) - (-(tref ** 3 - C0) ** 2 / (2.0 * sdu ** 2))
        return np.exp(ex) * t

    def _ig_solve_C0(self, sd):
        """解 C0 使 <Δ> = <1/t> = 1（James Eq.8 下方 / Shin A16 下方）。"""
        from scipy.optimize import brentq

        def f(C0):
            t, h = self._ig_tgrid(C0, sd)
            p = self._ig_pt(C0, sd, t)
            return (p / t).sum() / p.sum() - 1.0
        return brentq(f, -40.0 * sd - 5.0, 40.0 * sd + 20.0, xtol=1e-11, rtol=1e-13)

    def _ig_tables(self, sd):
        """缓存 (C0, Δ升序, CDF(Δ), 累积一阶矩 E[Δ·1{Δ<=x}])。键 round(sd,10)。"""
        key = round(float(sd), 10)
        hit = self._ig_cache.get(key)
        if hit is not None:
            return hit
        C0 = self._ig_solve_C0(sd)
        t, h = self._ig_tgrid(C0, sd)
        W = self._ig_pt(C0, sd, t) * h
        W = W / W.sum()
        D = 1.0 / t                                   # 随数组序递减
        cdf = np.clip(np.cumsum(W[::-1])[::-1], 0.0, 1.0)   # P(Δ <= D_i)
        # cumsum 舍入会让 cdf 末点变成 1+1e-14，使 np.interp 的 xp 非单调、
        # 末层质心落错区间（实测 mean 偏 -2e-9），故必须 clip 到 [0,1]。
        cum = np.cumsum((W * D)[::-1])[::-1]          # E[Δ·1{Δ <= D_i}]
        rev = slice(None, None, -1)                   # 转为 Δ 升序（cdf/cum 随之升序）
        out = (C0, D[rev], cdf[rev], cum[rev])
        self._ig_cache[key] = out
        return out

    def igm_delta_nodes(self, z):
        """(n, N_igm) 质心节点表：第 j 层质心 = N·[M(q_{j+1}) - M(q_j)]。

        等权平均精确重现 <Δ>=1（构造性），重尾不再丢质量。
        推断端 log_distr_efdmwz 用；与 igm_delta_sample 共用 _ig_tables。

        z 按【扁平列表】解释：任意形状都先 ravel，返回 (z.size, N_igm)。
        推断端传的 z 是 (nz,1) 广播形 —— 若用 atleast_1d 而非 ravel，enumerate
        会遍历出【行】(shape (1,)) 而不是标量，_ig_tables 里 float(sd) 在
        numpy>=2 直接抛 TypeError（size-1 数组不再隐式转标量），而 lnlik 的
        except 会把它吞成 -1e30 ⇒ 整个似然变成常数平台，MultiNest 无声跑废。
        """
        zs = np.ravel(np.asarray(z, dtype=float))
        out = np.empty((zs.size, self.N_igm))
        e = np.arange(self.N_igm + 1) / self.N_igm
        for i, zz in enumerate(zs):
            _, _, cdf, cum = self._ig_tables(float(self.sigma_dm_igm(zz)))
            # 补精确端点 (0,0) 与 (1,<Δ>)：否则 np.interp 在 e=0 处 clamp 到
            # cum[0]>0，首层质心偏小 ~2e-9。补上后各层质心望远镜相消，
            # mean(nodes) = cum[-1] - 0 = <Δ>，严格到求积精度。
            c_ext = np.concatenate(([0.0], cdf, [1.0]))
            m_ext = np.concatenate(([0.0], cum, [cum[-1]]))
            out[i] = self.N_igm * np.diff(np.interp(e, c_ext, m_ext))
        return out

    def _ig_sampling_table(self):
        """(N_SD, N_U) 的 Δ 分位主表，供 igm_delta_sample 用，惰性构建一次。

        逐事件精确求解 C0 太贵（ns 次 brentq × 2·NT 数组运算），而抽样只要 ~1%
        精度，故在 ln σ_DM 方向线性插值（V26 钉住插值误差）。σ_DM 只依赖 z，
        所以主表与 z 网格无关，构建一次全进程复用。
        """
        if self._ig_samp_tab is not None:
            return self._ig_samp_tab, self._ig_samp_lnsd, self._ig_samp_u
        lo = float(self.sigma_dm_igm(self.Zmax))
        hi = float(self.sigma_dm_igm(self.igm_zmin_eff))
        lnsd = np.linspace(np.log(lo), np.log(hi), self._IG_N_SD)
        u = (np.arange(self._IG_N_U) + 0.5) / self._IG_N_U
        tab = np.empty((self._IG_N_SD, self._IG_N_U))
        for i, s in enumerate(np.exp(lnsd)):
            _, D, cdf, _ = self._ig_tables(s)
            tab[i] = np.interp(u, cdf, D)
        self._ig_samp_tab, self._ig_samp_lnsd, self._ig_samp_u = tab, lnsd, u
        return tab, lnsd, u

    def igm_delta_sample(self, z, size, rng):
        """模拟端抽样：与 igm_delta_nodes 同一套 _ig_tables（两端单一数据源）。

        rng=None 时退回全局 np.random（simufrb 的历史约定）。
        """
        z = np.atleast_1d(np.asarray(z, dtype=float))
        if rng is None:
            u = np.random.uniform(0.0, 1.0, size)
        else:
            u = rng.uniform(0.0, 1.0, size)
        tab, lnsd, ug = self._ig_sampling_table()
        sd = np.clip(self.sigma_dm_igm(z[:size] if len(z) >= size else z),
                     np.exp(lnsd[0]), np.exp(lnsd[-1]))
        pos = (np.log(sd) - lnsd[0]) / (lnsd[1] - lnsd[0])
        i0 = np.clip(np.floor(pos).astype(int), 0, len(lnsd) - 2)
        fr = np.clip(pos - i0, 0.0, 1.0)
        out = np.empty(size)
        for i in range(size):
            a = np.interp(u[i], ug, tab[i0[i]])
            b = np.interp(u[i], ug, tab[i0[i] + 1])
            out[i] = a + fr[i] * (b - a)
        return out

    def log_IntBeam(self, logl, gamma, logls, logl0):
        """波束卷积核 ∫_{E}^{2E} x^γ e^{-x} dx / ln2，写成两个上不完全伽马之差。

        ⚠ gamma 上界为 0 是【数值】约束，不是物理约束：本式用
        gammainc(gamma+1, r) - gammainc(gamma+1, 2r) 求 [E,2E] 的 bin，r=E/E* 很小
        时两项都是 O(1) 而差是 O(r^(γ+1))，有效位数按 γ+1 个数量级损失。实测
        （独立参照 = 指数替换求积）：γ=-1 相对误差 2.6e-14，γ=0 为 1.1e-5，
        γ=+1 时真值 ~1.5e-22 而本式返回恰好 0（被 1e-199 兜底成 log=-460）。
        若要扩 gamma 上界到 >0，必须先把本式换成
        spf.gamma(a)*spf.gammainc(a, r, 2*r)（广义下不完全伽马，直接算区间积分）。
        smooth 选择路径 log_IntBeam_S 无此缺陷（bin 上直接 Gauss-Legendre）。
        """
        ratio = np.power(10., logl-logls)
        lik0 = gammainc(gamma+1, ratio) - gammainc(gamma+1, 2*ratio)
        lik = lik0/np.log(2)      # Beam efficiency from 50% to 100%
        lik = np.where(np.isfinite(lik) & (lik > 0), lik, 1e-199)
        loglik = np.log(lik)
        # 用 np.where 代替布尔索引赋值，同时支持标量和数组输入
        loglik = np.where(logl < logl0, -1e30, loglik)
        return loglik

    # --- E_iso 能量函数 ---
    def Schechter_E_log(self, loge, phis, gamma, logEs):
        """Schechter 能量函数 per dex"""
        e = np.power(10., loge)
        es = np.power(10., logEs)
        phi = np.log(10) * phis * np.power(e / es, (gamma + 1)) * np.exp(-e / es)
        return phi

    def IntE(self, eps, gamma, logEs, logE_min):
        """Schechter 能量累积积分"""
        with np.errstate(invalid='ignore'):
             ratio = np.power(10., logE_min-logEs)
        return gammainc(gamma+1, ratio/eps)

    # ---------- 平滑选择函数（McGregor 2026 实测 S）在似然端的原语 ----------
    # 约定与 IntE 严格一致：未归一化上尾，x = E/E*，S 写成 x 的 logistic
    #   S(x) = σ(a·(log10 x − delta)),  delta = log10(c) − logEs,  c = 50% 探测能量
    #
    # 【为什么必须分部积分，不能直接求积】被积函数 x^γ e^{-x} 在 γ→−4、x0→1e-11 时
    # 是近发散的陡幂律（x0^{γ+1} 量级达 1e31），任何对数网格求积都要上千节点才收敛。
    # 分部积分把陡尾交给【解析】的 gammainc，剩下的 S′ 项只是一段光滑指数：
    #   ∫_{x0}^∞ x^γ e^{-x} S dx = Γ(γ+1,x0)·S(x0) + a·∫_{log10 x0}^{∞} Γ(γ+1,10^t)·S(1−S) dt
    # （换元 t=log10 x 后 S′dx = a·S(1−S)dt，无额外 ln10 因子——这里曾多乘 (ln10)²
    #   导致 4–5 倍偏差，V23 的两个解析退化极限钉死了它。）
    # 修正项的积分下限【必须是 lo】，不能按 bump 宽度截：Γ(γ+1,10^t) 向小 t 方向按
    #   x^{γ+1} 增长，比 S(1−S) 的 e^{-a|Δt|} 快，delta ≫ lo 时 bump 之外的深区才是
    #   主贡献（V23 的逐格 brute-force 钉就是为此加的）。
    # 选择函数模式：build_models 按 config/env 覆盖（chime_selection.resolve_selection）
    sel_mode = 'step'
    sel_k = 1.0

    _S_LOGX_CAP = 2.5          # 修正项上截断：Γ(γ+1,x>316) < 1e-137，绝对可忽略
    _S_PANEL = 1.0             # 修正项面板宽 [dex]（面板内 e^{a·t} 最多变 e^6）
    _S_N = 8                   # 每面板 Gauss-Legendre 节点
    _S_N_BIN = 5               # 分子 bin 节点（5 点 Gauss 精确到 9 阶）

    def _s_sig(self, t, delta, a):
        return 1.0 / (1.0 + np.exp(-np.clip(a * (t - delta), -500., 500.)))

    def IntE_S(self, gamma, logEs, logE0, logc, a):
        """∫_{E0/E*}^∞ x^γ e^{-x} S(x) dx —— 取代 IntE 的硬阈值下限。

        logc/a 形状任意（分母 (nz,nlogw,1)），返回二者广播后的同形状数组。
        退化极限：S≡1 → gammainc(γ+1,x0)（逐位精确）。
        精度域（V23 钉）：与逐点 brute-force 比，显著格子相对误差 ≤1e-8，全域
        ≤4e-5，条件 a ≤ 8（物理 a = 2ln19/W595 ∈ [2.08, 6.30]）。a ≳ 10 时
        logistic 跃迁宽度 4/a 已小于面板内节点距，会开始丢权重——a→∞ 的硬阶跃
        【不是】本式的极限，不要用。
        要求 logEs/logE0 为标量（节点网格由 lo=logE0−logEs 决定，与逐格子的
        (logc, a) 无关 ⇒ gammainc 全程只算 n_pan·_S_N ≈ 112 个共享点）。
        """
        logc = np.asarray(logc, dtype=float)     # 允许标量/任意形状
        a = np.asarray(a, dtype=float)
        lo = float(logE0) - float(logEs)          # log10 x0，标量
        delta = logc - logEs
        g1 = gamma + 1.0
        # 节点：[lo, CAP] 等宽面板。求积下限必须是 lo —— 恒等式要求 ∫_lo^∞，
        # 早先按 bump 宽度截到 max(lo, delta−8/a) 在 delta ≫ lo 时丢掉了主贡献
        # （Γ(γ+1,10^t) 沿 t 向【下】按 x^{γ+1} 增长，比 S(1−S) 的 e^{-a|Δt|} 衰减
        # 快得多，bump 不是可截断的局域峰）。
        n_pan = max(1, int(np.ceil((self._S_LOGX_CAP - lo) / self._S_PANEL)))
        gt, gw = np.polynomial.legendre.leggauss(self._S_N)
        edges = np.minimum(lo + self._S_PANEL * np.arange(n_pan + 1), self._S_LOGX_CAP)
        corr = np.zeros(np.broadcast(delta, a).shape)
        for p in range(n_pan):
            hw = (edges[p + 1] - edges[p]) / 2.0
            if hw <= 0:
                break
            t = (edges[p] + edges[p + 1]) / 2.0 + gt * hw
            gwt = gammainc(g1, np.power(10.0, t)) * gw * hw          # 共享权重
            # S(1−S) 是 u 的偶函数 ⇒ 一次 exp，深区不溢出
            v = np.exp(-np.minimum(np.abs(a[..., np.newaxis]
                                          * (t - delta[..., np.newaxis])), 500.0))
            corr += a * ((v / (1.0 + v) ** 2) * gwt).sum(axis=-1)
        return gammainc(g1, np.power(10.0, lo)) * self._s_sig(lo, delta, a) + corr

    def log_IntBeam_S(self, loge, gamma, logEs, logE0, logc, a, n=None):
        """log_IntBeam 的 S 加权版：bin 核 ∫_{E}^{2E} x^γ e^{-x} S dx / ln2。

        5 点 Gauss–Legendre 即可：bin 只有 0.301 dex。逐 bin 与 brute-force 比，
        x=E/E* ≤ 10（承载似然权重的全部区间）最大偏差 2.2e-5 dex；x=18 → 1.4e-3,
        x=32 → 2.4e-2，而那时 bin 值本身已 < 1e-25（该事件已被 e^{-x} 罚掉 ≥25 dex），
        对后验无影响。端点矩形会高估 n/(n−1)，故用 Gauss。

        波束 ε∈[0.5,1] 的对数均匀卷积【就是】这个平铺核，与分母删掉的 ε 维是
        不同数学对象（阈值 vs 密度平滑），所以分子保留、分母删除。

        logE0 与 log_IntBeam 位置一致：E0 是能量函数物理下限，S 不替代它，
        必须继续硬切（否则分子不认 E0，与 mock 的 vEobs >= E0 配对裂开，V14 修的
        就是这对口径）。
        """
        n = self._S_N_BIN if n is None else n
        logc = np.asarray(logc, dtype=float)
        a = np.asarray(a, dtype=float)
        logx0 = np.where(np.isfinite(loge - logEs), loge - logEs, -300.0)
        gt, gw = np.polynomial.legendre.leggauss(n)
        hw = np.log10(2.0) / 2.0                       # bin 的半宽 [dex]
        # 区间是 [logx0, logx0+log10 2]，中心 = logx0+hw：写成 logx0 + gt*hw 会把
        # bin 平移半格，在 e^{-x} 陡降段是 10 个 dex 量级的错（V23 的 S=1 退化钉抓住）
        t = logx0[..., np.newaxis] + hw * (1.0 + gt)
        x = np.power(10.0, t)
        sg = self._s_sig(t, (logc - logEs)[..., np.newaxis], a[..., np.newaxis])
        integ = np.power(x, gamma) * np.exp(-x) * sg * x * np.log(10.0)
        out = np.sum(integ * gw, axis=-1) * hw / np.log(2.0)   # Gauss: ∫≈(b−a)/2·Σgw f = hw·Σgw f
        out = np.where(np.isfinite(out) & (out > 0), out, 1e-199)
        loglik = np.log(out)
        loglik = np.where(loge < logE0, -1e30, loglik)   # 与 log_IntBeam:250 同一刀
        return loglik

    def s_center_and_slope(self, z3d, wint_ms, dnu, k_offset):
        """(F_50(w)·k) → (log10 中心能量 c, logistic 斜率 a)。

        分母、率、分子三处共用这一个函数 ⇒ 口径不可能分叉（V23 有静态钉）。
        """
        f50 = chime_selection.f50_of_width(wint_ms) * k_offset
        logc = np.log10(self.cos.Energy(z3d, flu=f50, dnu=dnu))
        a = 2.0 * np.log(19.0) / chime_selection.w595_of_width(wint_ms)
        return logc, a

    def _det_int_w_S(self, dnu, gamma, logEs, logE0, mu, sigma, nz=200, nlogw=200):
        """∫_{E0}^∞ x^γ e^{-x} S(x) dx 对 (z, 内禀宽度) 网格求值，再对宽度边缘化。

        Norm1D_E_S 与 rate_2d_E_S 共用本函数 ⇒ 二者不可能用不同口径的 S。
        【没有 ε 维、没有 RMEq、没有 E_min】：阈值整个交给 S，只留 E0 作下限。
        返回 (vz, stepz, int_w)。
        """
        stepz = (np.log(self.Zmax) - np.log(self.Zmin)) / nz
        vz = np.exp(np.arange(np.log(self.Zmin), np.log(self.Zmax), stepz))       # (nz,)
        w_lo, w_hi = mu - 6. * sigma, mu + 6. * sigma
        steplogw = (w_hi - w_lo) / nlogw
        vlogw = np.arange(w_lo, w_hi, steplogw)                                    # (nlogw,)
        z3d = vz[:, np.newaxis, np.newaxis]                                        # (nz,1,1)
        wint = np.power(10.0, vlogw)[np.newaxis, :, np.newaxis]                    # (1,nlogw,1)
        logc, a = self.s_center_and_slope(z3d, wint, dnu, self.sel_k)               # (nz,nlogw,1)
        det = self.IntE_S(gamma, logEs, logE0, logc, a)[:, :, 0]                    # (nz,nlogw)
        int_w = np.sum(det * self.dis_logw(vlogw, mu, sigma)[np.newaxis, :] * steplogw, axis=1)
        return vz, stepz, int_w

    def Norm1D_E_S(self, dnu, gamma, logEs, logE0, mu, sigma, nz=200, nlogw=200):
        """Norm1D_E 的平滑选择函数版（逐事件似然的分母 C_sel）。"""
        vz, stepz, int_w = self._det_int_w_S(dnu, gamma, logEs, logE0, mu, sigma, nz, nlogw)
        fz = self.Distribution_volume(vz) * self.evolution_factor(vz)
        nf = float(np.sum(vz * stepz * fz * int_w))
        return 1e-199 if nf <= 0 else nf

    def log_IntBeam_E(self, loge, gamma, logEs, logE0):
        """波束卷积后的 Schechter 能量函数（与 log_IntBeam 数学形式相同）"""
        return self.log_IntBeam(loge, gamma, logEs, logE0)
    
    def SFR_evolution(self, z, curve=None):
        """恒星形成率历史。curve 取自 config 的 analysis.sfr_curve。

        'MD2014'     : Madau & Dickinson 2014，即 Shin 2023 Eq (A9)（默认，便于与 Shin 对比）
        'Yuksel2008' : 改版前基线所用的曲线
        归一化系数在 evolution_factor 的比值中消掉。两条曲线都可跑，其差作为
        「事件率演化曲线」的系统误差报告，而不是把 P0 与 P6 的对比混淆掉。
        """
        if curve is None:
            curve = self.sfr_curve
        zp1 = 1.0 + np.asarray(z, dtype=float)
        if curve == 'MD2014':
            return 0.015 * np.power(zp1, 2.7) / (1.0 + np.power(zp1 / 2.9, 5.6))
        if curve == 'Yuksel2008':
            aa, bb, cc, BB, CC, eta = 3.4, -0.3, -3.5, 5e3, 9, -10
            return np.power(np.power(zp1, aa * eta) + np.power(zp1 / BB, bb * eta)
                            + np.power(zp1 / CC, cc * eta), 1.0 / eta)
        raise ValueError("未知 sfr_curve: %r（可选 MD2014 / Yuksel2008）" % (curve,))

    def evolution_factor(self, z, n_evol=None):
        """FRB 事件率宇宙学演化因子。Shin 2023 Eq.(A8) / James 2022b Eq.(13) 的原式是

            Phi(z) = Phi0 * (1+z)^(-1) * [SFR(z)/SFR(0)]^n

        本实现把 (1+z) 折进幂内：Phi ∝ [(SFR/SFR(0))/(1+z)]^n。两者**仅在 n=1 时
        恒等**（V6 已证逐点相对差 <1e-12）；n≠1 时相差 (1+z)^(1-n)。
        当前 n_evol 固定为 1.0（Shin Table 1 最佳拟合 n=0.96 在 1σ 内与 1 不可分），
        故数值无差异。**若将来把 n 放开为自由参数，必须先改回原式**，否则 n=0 会
        连源帧→观测帧的时间膨胀一起丢掉（先验中位 z 偏 +28.6%）。
        n_evol=0 -> 不随 SFH 演化；n_evol=1 -> 完美追踪 SFH。
        调用方不得再额外乘/除 (1+z)。
        """
        if n_evol is None:
            n_evol = self.n_evol
        ratio = (self.SFR_evolution(z) / self.SFR_evolution(0.0)) / (1.0 + z)
        return np.power(ratio, n_evol)

    def Distribution_volume(self, z):
        """
        Differential comoving volume
        """
        r = self.cos.Comoving_Distance(z)/self.cos.Comoving_Distance(1)
        pv = r * r / self.cos.E(z)
        return pv
    
    def log_Distribution_volume(self, z):
        """
        Logarithm of differential comoving volume

        对 z<0 的元素返回 -1e30（物理上禁止）。用 np.where 统一处理标量/数组/0-d 数组，
        避免布尔索引赋值对 0-d 数组报 TypeError。
        """
        # 对 z<0 的位置用 1.0 占位计算（结果会被 np.where 丢弃），避免 log(负数) 产生 nan
        z_safe = np.where(np.asarray(z) < 0, 1.0, z)
        pv = np.log(self.Distribution_volume(z_safe))
        return np.where(np.asarray(z) < 0, -1e30, pv)

    def IntDMsrc_LogNormal(self, u1, u2, ln_mu_host, ln_sigma_host):
        """均匀晕 U[DMsmin, DMsmax] 与对数正态宿主卷积后的边际密度（观测帧）。

        P(u2 < DM_host^obs < u1) / (DMsmax - DMsmin)，其中调用方给
        u1 = 残差 - DMsmin, u2 = u1 - (DMsmax - DMsmin)。
        对数正态的 CDF 解析可积，故无需数值卷积；ln 空间，勿与 log10 混用。
        u2 <= 0 由 np.maximum 截断到 1e-12（erf -> -1），即宿主 DM >= 0 的自然下界。
        """
        s2 = ln_sigma_host * np.sqrt(2.0)
        cdf = lambda u: 0.5 * (1.0 + spf.erf((np.log(np.maximum(u, 1e-12)) - ln_mu_host) / s2))
        return np.maximum(cdf(u1) - cdf(u2), 1e-300) / (self.DMsmax - self.DMsmin)

    def dis_logw(self, logw0, mu, sigma):
        a = 1./np.sqrt(2.*np.pi*sigma*sigma)
        b = np.exp(-(logw0-mu)*(logw0-mu)/2/sigma/sigma)
        return a*b

    def log_dis_logw(self, logw0, mu, sigma):
        a = 2*np.pi*sigma*sigma
        b = -(logw0-mu)*(logw0-mu)/2/sigma/sigma
        return b-1/2.*np.log(a)

    def log_distr_efdmwz(self, dnu, logflux, dme, logw, z, gamma, logEs, logE0,
                         mu, sigma, ln_mu_host, ln_sigma_host):
        """能量版联合概率 p(E_iso, DM_host | z)

        将观测流量 logflux 和脉冲宽度 logw 耦合为 E_iso，再用 Schechter 能量函数评估。
        E_iso = S*w*Δν_ref*4π*D_L²/(1+z)^(2+alpha_spec)，alpha_spec = -1.39 (Shin 2023
        Table 1)，Δν_ref 为能量定义参考带宽（默认 1 GHz，见 TELESCOPES 注释）。

        DM 通道全程【观测帧】：残差 r = dme - DM_IGM(z) = DM_halo + DM_host^obs，
        宿主中位数取 Zhang 2020 Eq(6) 的观测帧量 A(1+z)^host_dm_evol。
        因此窗口边界不含任何 (1+z) 或 kappa 因子（雅可比恒为 1）。
        """
        flux = np.power(10., logflux)
        w_ms = np.power(10., logw)       # 脉冲宽度 [ms]
        fluence = flux * w_ms            # fluence = S * w [Jy·ms]
        loge = np.log10(self.cos.Energy(z, flu=fluence, dnu=dnu))
        logw0 = logw - np.log10(1+z)     # 内禀宽度（观测系 logw 扣 (1+z)）
        if getattr(self, 'sel_mode', 'step') == 'smooth':
            # S 作用在【真】fluence 对应的能量上；[E,2E] 平铺核的积分变量正是真能量
            # ⇒ 核内乘 S 自动就是 F_true 口径，与 mock smooth 判据同构（V22 钉 mock
            # 侧、V23 钉这里与分母同源）。logE0 照常硬切：E0 不被 S 替代。
            logc, a = self.s_center_and_slope(z, np.power(10.0, logw0), dnu, self.sel_k)
            logint1 = self.log_IntBeam_S(loge, gamma, logEs, logE0, logc, a)
        else:
            logint1 = self.log_IntBeam_E(loge, gamma, logEs, logE0)
        logfw = self.log_dis_logw(logw0, mu, sigma)
        # p(z) ∝ (dV/dz) · evolution(z) / (1+z)
        # /(1+z) 为源帧→观测帧时间膨胀因子,与 Norm1D_E / rate_2d_E / simufrb.py 采样端一致。
        # (1+z) 时间膨胀已在 evolution_factor（Shin Eq.3 幂内）中，此处不再重复除
        logfz = self.log_Distribution_volume(z)
        dmi_mean = self.cos.DispersionMeasure_IGM(z)
        ln_mu_z = ln_mu_host + self.host_dm_evol * np.log(1.0 + z)
        # 对 IGM 涨落做准高斯 Δ_DM 的质心节点平均（James Eq.8 = Shin A16）。
        # 只有窗口项依赖 DM_IGM，其余项（beam/gammainc、logfz、logfw、evolution）
        # 在循环外已算一次。节点表按 σ_DM(z) 缓存，z 网格不变时每次似然求值零重建。
        # 广播：节点列 (nz,1) × dmi_mean (nz,1) × dme (N,) -> (nz,N)。
        nodes = self.igm_delta_nodes(z)                  # (nz, N_igm)
        tot = 0.0
        for j in range(self.N_igm):
            r = dme - dmi_mean * nodes[:, j:j + 1]       # 观测帧残差 = 晕 + 宿主
            # 晕 U[DMsmin, DMsmax] -> 宿主窗口 [(r-DMsmax)+, (r-DMsmin)+]，宽度 DMsmax-DMsmin
            u1 = r - self.DMsmin
            u2 = u1 - (self.DMsmax - self.DMsmin)
            tot = tot + self.IntDMsrc_LogNormal(u1, u2, ln_mu_z, ln_sigma_host)
        # u2<0 由 IntDMsrc_LogNormal 内部截断到 0（窗口下界取 0），物理正常事件仍非零。
        # tot==0 才是物理不可能（所有节点的 DM_IGM 都超过观测 DM）。
        logint2 = np.where(tot > 0, np.log(tot / self.N_igm), -1e30)
        loglikv = logint1 + logfz + logfw + logint2 + np.log(self.evolution_factor(z))
        return loglikv

    def log_distr_efdmw_grid(self, dnu, logflux, dme, logw, vz, gamma, logEs, logE0,
                             mu, sigma, ln_mu_host, ln_sigma_host):
        """逐事件×逐红移对数似然网格 (nz, N)

        log_distr_efdmw 的 z 边际化与 pltpz.py 的逐事件 P(z) 可视化共用此网格。
        logflux/dme/logw 接受标量或 (N,) 数组, vz 为 (nz,) 红移网格;
        返回未边际化的 log p(z, 事件 | 参数)，z 积分测度 ∝ vz·dz 由调用方处理。
        """
        return self.log_distr_efdmwz(dnu, np.atleast_1d(logflux), np.atleast_1d(dme),
                                     np.atleast_1d(logw), np.atleast_1d(vz)[:, np.newaxis],
                                     gamma, logEs, logE0, mu, sigma,
                                     ln_mu_host, ln_sigma_host)

    def log_distr_efdmw(self, dnu, logflux, dme, logw, gamma, logEs, logE0,
                        mu, sigma, ln_mu_host, ln_sigma_host):
        """能量版联合概率，对红移 z 边际化（完全向量化）

        对于有红移的事件，可直接使用 log_distr_efdmwz；本函数用于仅有 DM 的事件。
        一次性传入整个 vz 数组，消除 Python z 循环。
        log_distr_efdmwz 内部全部支持广播: z(nz,) × event(N,) → (nz, N)
        """
        # z 网格 400 (向量化后 z 循环已消除, 加大 z 仅增 ~50ms, 性价比高)
        # Norm1D_E/rate_2d_E 的 z 网格保持 200 (gammainc 瓶颈), 但事件似然的
        # z 边际化可以更精细, 不受 gammainc 限制
        nz = 400
        stepz = (np.log(self.Zmax) - np.log(self.Zmin)) / nz
        vz = np.exp(np.arange(np.log(self.Zmin), np.log(self.Zmax), stepz))  # (nz,)

        # 统一转为至少 1D, 保证 likv 是 (nz, N)
        scalar_input = np.isscalar(logflux)
        logflux_a = np.atleast_1d(logflux)
        dme_a = np.atleast_1d(dme)
        logw_a = np.atleast_1d(logw)

        # 一次性广播: z 升维为 (nz,1), 事件参数 (N,) → likv (nz, N)
        # log_distr_efdmwz 内部全部支持广播: Energy(z(nz,1), flu(N,)) → (nz, N)
        likv = np.exp(self.log_distr_efdmw_grid(dnu, logflux_a, dme_a, logw_a, vz,
                                                gamma, logEs, logE0, mu, sigma,
                                                ln_mu_host, ln_sigma_host))
        # 对 z 求和: (nz, N) → (N,)
        lik = np.sum(vz[:, np.newaxis] * stepz * likv, axis=0)

        ind = lik > 0
        ind2 = lik <= 0
        loglik = lik.copy()
        loglik[ind] = np.log(lik[ind])
        loglik[ind2] = np.ones(loglik[ind2].shape) * -1e30
        if scalar_input:
            return loglik[0]
        return loglik

    def Norm1D_E(self, sn0, bw, npol, g, tsys, dnu, gamma, logEs, logE0, mu, sigma):
        """能量版归一化因子（完全向量化）

        检测阈值从 L_min 转换为 E_min：E_min = S_min * w_obs * dnu * 4π * D_L² / (1+z)^(2+α)，谱指数 α = -1.39 (Shin 2023)
        三维广播 (nz, nlogw, neps) 一次性计算，消除 Python z 循环。
        内存: 200×200×100×8B = 32MB

        sel_mode='smooth' 时分派到 Norm1D_E_S（阈值整个交给实测 S）。
        """
        # sel_mode 由 build_models 挂在实例上（er.ad 与本对象同一实例 ⇒ 三处同源）。
        # 缺省 'step'：下方旧路径一字未改，历史基线/回归钉按构造不受影响。
        # 分派保留 sn0/bw/npol/g/tsys 形参 —— V23 的 sn0 不变性钉要靠【公开入口】
        # 传不同 sn0 才有意义（若把形参删掉，那条钉就变成什么都没测）。
        if getattr(self, 'sel_mode', 'step') == 'smooth':
            return self.Norm1D_E_S(dnu, gamma, logEs, logE0, mu, sigma)
        # 网格: z=200, eps=100, logw=200 (4M 元素, gammainc 约 1.4s)
        # gammainc 是绝对瓶颈(0.34μs/元素), 向量化不改变总计算量;
        # eps 维度对 gammainc 影响最大(广播维), 保持 100;
        # z 保持 200, 精度提升靠 log_distr_efdmw 的 z 网格加大;
        # logw 网格与模拟端(simufrb μ±6σ 截断)同源自适应, 见下
        nz, neps, nlogw = 200, 100, 200
        stepz = (np.log(self.Zmax) - np.log(self.Zmin)) / nz
        vz = np.exp(np.arange(np.log(self.Zmin), np.log(self.Zmax), stepz))      # (nz,)
        stepeps = (1-0.5) / neps
        veps = np.arange(0.5, 1, stepeps)                                         # (neps,)
        # 宽度网格必须覆盖宽度分布的全部支撑, 与模拟端 simufrb 的 μ±6σ 截断同源。
        # 旧写法固定 [Wmin,Wmax]=[0.05,20] ms: 注入真值 (0.4,0.3) 处网格外质量仅
        # 0.13%(行为不漂移), 但先验角 (mu_w,sigma_w)=(2,1)/(2,0.01) 处分别截掉
        # 76%/100% —— 选择函数不再等于真实探测概率。200 点对 sigma_w∈[0.01,1] 的
        # 最宽 12 dex 6σ 窗仍光滑(高斯, 0.06 dex/点)。
        w_lo, w_hi = mu - 6. * sigma, mu + 6. * sigma
        steplogw = (w_hi - w_lo) / nlogw
        vlogw = np.arange(w_lo, w_hi, steplogw)                                   # (nlogw,)

        # 三维广播: z(nz,1,1) × vlogw(1,nlogw,1) × veps(neps,)
        z3d = vz[:, np.newaxis, np.newaxis]                                       # (nz,1,1)
        vw = np.power(10, vlogw)[np.newaxis, :, np.newaxis] * (1 + z3d)           # (nz,nlogw,1)
        ft = self.tel.RMEq(sn0, g, tsys, npol, bw, vw)                            # (nz,nlogw,1)
        fw = ft * vw                                                              # (nz,nlogw,1)
        et = self.cos.Energy(z3d, flu=fw, dnu=dnu)                                # (nz,nlogw,1)
        loget = np.log10(et)
        loget = np.where(np.isfinite(loget) & (loget >= logE0), loget, logE0)
        # IntE: veps(neps,) × loget(nz,nlogw,1) → (nz,nlogw,neps)
        int_eps_grid = self.IntE(veps, gamma, logEs, loget)                       # (nz,nlogw,neps)
        int_eps = np.sum(int_eps_grid / veps[np.newaxis, np.newaxis, :] / np.log(2) * stepeps, axis=2)  # (nz,nlogw)
        int_w = np.sum(int_eps * self.dis_logw(vlogw, mu, sigma)[np.newaxis, :] * steplogw, axis=1)     # (nz,)
        # /(1+z) 为源帧→观测帧时间膨胀因子，与 rate_2d_E / simufrb.py 采样端一致
        fz = self.Distribution_volume(vz) * self.evolution_factor(vz)  # (nz,)
        nf = np.sum(vz * stepz * fz * int_w)
        if nf <= 0:
            nf = 1e-199
        return nf

class EventRate:
    def __init__(self):
        # 注意：self.ad 是独立创建的 AstroDistribution 实例，默认宇宙学。
        # 调用方若修改了宇宙学参数（如从 config 读取），必须显式同步：
        #     er.cos = cos; er.ad = dis
        # 否则 rate_2d_E 等方法会用 self.ad 的默认宇宙学，与外部 dis 不一致。
        self.cos = Cosmology()
        self.ad = AstroDistribution()
        self.tel = Telescope()
        self.s2h = 1./3600
        self.yr2hr = 365*24.
        self.rad2deg2 = 3282.806350011744
        self.Gpc2Mpc = 1e3

    def rate_2d_E(self, sn0, bw, npol, g, tsys, dnu, phis, gamma, logEs, logE0, mu, sigma):
        """能量版事件率密度 rho_deg [deg^-2 hr^-1]（完全向量化）

        检测阈值从 L_min 转换为 E_min：E_min = S_min * w_obs * dnu * 4π * D_L² / (1+z)^(2+α)，谱指数 α = -1.39 (Shin 2023)
        三维广播 (nz, nlogw, neps) 一次性计算，消除 Python z 循环。
        与 Norm1D_E 结构同构，仅 fz 用 dVdOdz 且多乘 phis。

        sel_mode='smooth' 时分派到 rate_2d_E_S（与 Norm1D_E_S 共用 _det_int_w_S，
        保证分母与率用同一个 S）；形参同样保留，理由见 Norm1D_E 处注释。
        """
        if getattr(self.ad, 'sel_mode', 'step') == 'smooth':
            return self.rate_2d_E_S(dnu, phis, gamma, logEs, logE0, mu, sigma)
        # 网格: z=200, eps=100, logw=200 (与 Norm1D_E 一致)
        nz, neps, nlogw = 200, 100, 200
        stepz = (np.log(self.ad.Zmax) - np.log(self.ad.Zmin)) / nz
        vz = np.exp(np.arange(np.log(self.ad.Zmin), np.log(self.ad.Zmax), stepz))  # (nz,)
        stepeps = (1-0.5) / neps
        veps = np.arange(0.5, 1, stepeps)                                           # (neps,)
        # 宽度网格与 Norm1D_E 同一套 μ±6σ 自适应(注释见彼处): 覆盖宽度分布全支撑,
        # 与模拟端 simufrb 的 μ±6σ 截断同源, 消除固定 [0.05,20] ms 网格在先验角落
        # 截掉 76%-100% 宽度质量的问题。
        w_lo, w_hi = mu - 6. * sigma, mu + 6. * sigma
        steplogw = (w_hi - w_lo) / nlogw
        vlogw = np.arange(w_lo, w_hi, steplogw) # (nlogw,)

        # 三维广播: z(nz,1,1) × vlogw(1,nlogw,1) × veps(neps,)
        z3d = vz[:, np.newaxis, np.newaxis]                                         # (nz,1,1)
        vw = np.power(10, vlogw)[np.newaxis, :, np.newaxis] * (1 + z3d)             # (nz,nlogw,1)
        ft = self.tel.RMEq(sn0, g, tsys, npol, bw, vw)                              # (nz,nlogw,1)
        fw = ft * vw                                                                # (nz,nlogw,1)
        et = self.cos.Energy(z3d, flu=fw, dnu=dnu)                                  # (nz,nlogw,1)
        loget = np.log10(et)
        loget = np.where(np.isfinite(loget) & (loget >= logE0), loget, logE0)
        # IntE: veps(neps,) × loget(nz,nlogw,1) → (nz,nlogw,neps)
        int_eps_grid = phis * self.ad.IntE(veps, gamma, logEs, loget)               # (nz,nlogw,neps)
        int_eps = np.sum(int_eps_grid / veps[np.newaxis, np.newaxis, :] / np.log(2) * stepeps, axis=2)  # (nz,nlogw)
        int_w = np.sum(int_eps * self.ad.dis_logw(vlogw, mu, sigma)[np.newaxis, :] * steplogw, axis=1)   # (nz,)
        fz = self.cos.dVdOdz(vz) * self.ad.evolution_factor(vz)              # (nz,)
        rho = np.sum(vz * stepz * fz * int_w)
        rho_deg = rho/self.rad2deg2/self.yr2hr
        return rho_deg

    def rate_2d_E_S(self, dnu, phis, gamma, logEs, logE0, mu, sigma, nz=200, nlogw=200):
        """rate_2d_E 的平滑选择函数版 —— Λ_S = φ·Ω·T·k_S(θ)。

        S 在此是【模型的前向结构】（地位等同 IntE），不是外部实测常数存活率，
        故它随 θ 变化合法，不违反 V17 那条禁令（见 frb_util V17 注记旁的区分）。
        """
        vz, stepz, int_w = self.ad._det_int_w_S(dnu, gamma, logEs, logE0, mu, sigma, nz, nlogw)
        fz = self.cos.dVdOdz(vz) * self.ad.evolution_factor(vz)
        rho = phis * float(np.sum(vz * stepz * fz * int_w))
        return rho / self.rad2deg2 / self.yr2hr

    def log_dis_poi(self, rho, N, Omega, T):
        lamda = rho*Omega*T
        ind = lamda <= 0
        lamda[ind] = 1e-199
        loglik = N*np.log(lamda)-lamda-spf.gammaln(N+1)
        return loglik

class Loadfiles:
    #def __init__(self):

    def LoadSvyInfo(self, fname):
        # tel_svy.txt 带中文注释(UTF-8)，而 np.loadtxt 默认用 locale 编码
        # （Windows=GBK，HPC 常=ASCII）→ 必须显式指定，否则同一份包在两端行为不同。
        # 手工过滤注释行而不是 comments='#'：后者在 numpy>=1.23 会对注释行报
        # "contained no data" 弃用告警；也不传 encoding=：那要 numpy>=1.14。
        with open(fname, encoding='utf-8') as fh:
            rows = (l for l in fh if not l.lstrip().startswith('#'))
            cat = np.loadtxt(rows, dtype=str)
        row, col = cat.shape
        cat2 = {}
        for i in range(col):
            cat2[cat[0,i]] = cat[1:, i]
        cat2['SURVEY'] = np.array(cat2['SURVEY'])
        cat2['FOV'] = np.array(cat2['FOV'], dtype=float)
        cat2['TIME'] = np.array(cat2['TIME'], dtype=float)
        cat2['Gain'] = np.array(cat2['Gain'], dtype=float)
        cat2['Tsys'] = np.array(cat2['Tsys'], dtype=float)
        cat2['BW'] = np.array(cat2['BW'], dtype=float)
        cat2['Npol'] = np.array(cat2['Npol'], dtype=float)
        cat2['SN0'] = np.array(cat2['SN0'], dtype=float)
        return cat2

    def LoadSimuData(self, fname):
        """读取模拟 FRB 数据文件

        Header 格式（# 开头）：
          #T_obs 26280.0                          ← 可选的 key-value 元数据
          #S W T DMe thres logE Z DMi DMh DMs      ← 列名

        若 header 中包含 T_obs，则返回字典中会包含 'T_obs' 键。
        """
        col_names = None
        t_obs = None
        with open(fname) as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith('#'):
                    content = stripped[1:].strip()
                    if content.startswith('T_obs'):
                        parts = content.split()
                        if len(parts) >= 2:
                            t_obs = float(parts[1])
                    elif col_names is None:
                        col_names = content.split()
                else:
                    break
        if col_names is None:
            raise ValueError(f"无法从 {fname} 的 header 中解析列名")

        cat = np.loadtxt(fname, dtype=float, comments='#')
        cat2 = {}
        for i, name in enumerate(col_names):
            cat2[name] = cat[:, i]
        if t_obs is not None:
            cat2['T_obs'] = t_obs
        return cat2

    def LoadFitsCatalog(self, fname):
        """读取 CHIME/FRB Catalog 2 FITS 文件

        返回: vF(fluence Jy·ms), vW(width ms), vDM_obs, vDM_ne2001, vDM_ymw16,
              vDM_ne2025(可选列, 缺失时为 None), vSVY

        CHIME Catalog 2 关键列名:
          fluence, fluence_err — fluence 及误差 [Jy ms]
          bc_width — 校准后脉冲宽度 [ms]
          dm_fitb — 拟合 DM [pc cm^-3]
          dm_exc_ne2001 — DM - DM_MW(NE2001)，即河外 DM
          dm_exc_ymw16 — DM - DM_MW(YMW16)，即河外 DM
          红移：Catalog 2 中无红移列，需后续用 P(z|DM) 处理
        """
        try:
            from astropy.io import fits
        except ImportError:
            raise ImportError("需要安装 astropy: pip install astropy")

        with fits.open(fname) as hdul:
            data = hdul[1].data
            col_names = data.columns.names

            # === fluence ===
            fluence_key = next((k for k in ['fluence', 'Fluence'] if k in col_names), None)
            if fluence_key is None:
                raise KeyError(f"未找到 fluence 列，可用列: {col_names}")
            vF = np.array(data[fluence_key], dtype=float)

            # === width ===
            width_key = next((k for k in ['bc_width', 'width_fitb', 'width', 'Width'] if k in col_names), None)
            if width_key is None:
                raise KeyError(f"未找到 width 列，可用列: {col_names}")
            vW = np.array(data[width_key], dtype=float)
            # 优先读取 header 标记的单位，否则用中位数启发式判断
            hdr = hdul[1].header
            wunit = hdr.get('WUNIT', '').lower()
            if wunit == 'ms':
                pass  # 已经是毫秒，无需转换
            else:
                med_w = np.nanmedian(vW)
                if np.isfinite(med_w) and med_w < 1.0:
                    vW = vW * 1000.0

            # === DM_obs（观测 DM） ===
            dm_key = next((k for k in ['dm_obs', 'dm_fitb', 'bonsai_dm', 'dm', 'DM'] if k in col_names), None)
            if dm_key is None:
                raise KeyError(f"未找到 DM 列，可用列: {col_names}")
            vDM_obs = np.array(data[dm_key], dtype=float)

            # === DM_exc_ne2001: DM - DM_MW,NE2001（河外 DM） ===
            ne2001_key = next((k for k in ['dm_exc_ne2001', 'DM_NE2001'] if k in col_names), None)
            if ne2001_key is None:
                raise KeyError(f"FITS 缺少 NE2001 河外 DM 列，可用列: {col_names}")
            vDM_ne2001 = np.array(data[ne2001_key], dtype=float)

            # === DM_exc_ymw16: DM - DM_MW,YMW16（河外 DM） ===
            ymw16_key = next((k for k in ['dm_exc_ymw16', 'DM_YMW16'] if k in col_names), None)
            if ymw16_key is None:
                raise KeyError(f"FITS 缺少 YMW16 河外 DM 列，可用列: {col_names}")
            vDM_ymw16 = np.array(data[ymw16_key], dtype=float)

            # === DM_exc_ne2025: 可选列, 本管线用 mwprop/NE2025 预计算, 目录不自带 ===
            # 缺失时返回 None（回归保护: 旧 filtered.fits 不选 --mw ne2025 时行为不变）
            ne2025_key = next((k for k in ['dm_exc_ne2025', 'DM_NE2025']
                               if k in col_names), None)
            vDM_ne2025 = (np.array(data[ne2025_key], dtype=float)
                          if ne2025_key is not None else None)

            # === 巡天（Catalog 2 为单一 CHIME 巡天，此列可选） ===
            svy_key = next((k for k in ['survey', 'SURVEY', 'telescope'] if k in col_names), None)
            if svy_key is not None:
                vSVY = np.array(data[svy_key])
            else:
                vSVY = np.array(['CHIME'] * len(vDM_obs))

        return vF, vW, vDM_obs, vDM_ne2001, vDM_ymw16, vDM_ne2025, vSVY

# --- mwprop/NE2025 银河系电子密度模型封装（仅数据预处理端调用） ---
# NE2025: Ocker & Cordes 2026 (arXiv:2602.11838)，官方纯 Python 实现 mwprop
# (pip install mwprop，依赖 numpy/matplotlib/scipy/astropy/mpmath)。
# 用途: dm_exc_ne2025 = dm_obs - DM_MW,NE2025(gl, gb, dist_kpc)，
# 积分距离 dist_kpc 从 config 的 analysis.mwprop_dist_kpc 读取（默认 100 kpc，
# 覆盖全银河系含晕；ne2025_verify_plateau 可验证平台）。
# 推断端不依赖 mwprop：以下函数内部延迟 import，未被调用时无额外依赖。

def ne2025_dm_mw(gl, gb, dist_kpc=100.):
    """单方向调用 mwprop，返回银河系总 DM [pc/cm^3]

    调用: ne2025(ldeg=gl, bdeg=gb, dmd=dist_kpc, ndir=-1, classic=False)
    ndir<0 为距离→DM 模式，总 DM 在返回字典 Dv['DM']；
    classic=False 只关闭 Fortran 风格打印，不改数值。
    重定向 stdout 防止 mwprop 意外打印刷屏；异常正常向外抛出。
    """
    import io
    from contextlib import redirect_stdout
    from mwprop.nemod.NE2025 import ne2025
    with redirect_stdout(io.StringIO()):
        Dk, Dv, Du, Dd = ne2025(ldeg=float(gl), bdeg=float(gb),
                                dmd=float(dist_kpc), ndir=-1, classic=False)
    return float(Dv['DM'])


def _ne2025_worker(task):
    """多进程 worker: 计算单方向 DM_MW, 失败返回 NaN（顶层函数, 兼容 Windows spawn）"""
    gl, gb, dist_kpc = task
    try:
        return ne2025_dm_mw(gl, gb, dist_kpc)
    except Exception:
        return np.nan


def ne2025_dm_mw_batch(vgl, vgb, dist_kpc=100., progress=200, nproc=1):
    """批量计算银河系总 DM [pc/cm^3]，返回 (dm_mw, n_fail)

    mwprop 的 ne2025() 是标量视线积分接口, numpy 广播无效, 纯 Python 又比
    Fortran 慢 ~45 倍——nproc>1 时用 multiprocessing 并行是唯一有效加速
    （服务器 Linux fork 启动开销可忽略; 每进程独立延迟 import mwprop）。
    失败方向置 NaN; nproc=1 走串行路径（含进度与告警打印）。
    """
    vgl = np.atleast_1d(vgl)
    vgb = np.atleast_1d(vgb)
    n = len(vgl)
    if nproc > 1 and n > 1:
        import time
        import multiprocessing as mp
        tasks = [(float(vgl[i]), float(vgb[i]), float(dist_kpc)) for i in range(n)]
        t0 = time.perf_counter()
        with mp.Pool(nproc) as pool:
            dm_mw = np.array(pool.map(_ne2025_worker, tasks,
                                      chunksize=max(1, n // (nproc * 4))))
        dt = time.perf_counter() - t0
        n_fail = int(np.sum(~np.isfinite(dm_mw)))
        print(f"  NE2025 完成: {n} 个方向, {nproc} 进程并行, "
              f"{dt:.1f}s ({dt/max(n,1)*1000:.0f} ms/方向), 失败 {n_fail} 个")
        return dm_mw, n_fail

    dm_mw = np.full(n, np.nan)
    n_fail = 0
    import time
    t0 = time.perf_counter()
    for i in range(n):
        try:
            dm_mw[i] = ne2025_dm_mw(vgl[i], vgb[i], dist_kpc)
        except Exception as e:
            n_fail += 1
            if n_fail <= 5:
                print(f"[warn] (gl={vgl[i]:.2f}, gb={vgb[i]:.2f}) NE2025 计算失败: {e}")
        if (i + 1) % progress == 0:
            print(f"  NE2025 进度: {i+1}/{n} ({time.perf_counter()-t0:.1f}s)")
    dt = time.perf_counter() - t0
    print(f"  NE2025 完成: {n} 个方向, {dt:.1f}s ({dt/max(n,1)*1000:.0f} ms/方向), "
          f"失败 {n_fail} 个")
    return dm_mw, n_fail


def ne2025_verify_plateau(vgl, vgb, dist_kpc=100.):
    """平台验证：同一方向 dmd=50/100/150 kpc 三档，DM 应收敛（差异 <1%）

    验证 dist_kpc 是否已覆盖全银河系（含晕）。取 3 个代表方向：
    首个 + 银纬绝对值最高/最低的事件。
    """
    vgl = np.atleast_1d(vgl)
    vgb = np.atleast_1d(vgb)
    idx = sorted({0, int(np.argmax(np.abs(vgb))), int(np.argmin(np.abs(vgb)))})
    print(f"\n[平台验证] 距离→DM 应随积分距离收敛（50/100/150 kpc，"
          f"当前取值 {dist_kpc} kpc）:")
    for i in idx:
        dms = [ne2025_dm_mw(vgl[i], vgb[i], d) for d in (50, 100, 150)]
        spread = (max(dms) - min(dms)) / max(dms) * 100
        print(f"  (gl={vgl[i]:7.2f}, gb={vgb[i]:6.2f}): DM = "
              + " / ".join(f"{d:.2f}" for d in dms) + f"   波动 {spread:.2f}%")


def prior_transform(key, u, cfg):
    """把 MultiNest 单位超立方样本 u∈[0,1) 映射到物理先验。

    两个采样脚本共用，消除 myprior 的重复实现。cfg['prior'][key] 始终是支撑区间
    [lo, hi]（pltpost 的绘图范围与 MultiNest 的采样盒子都用它）；若 cfg['prior_shape']
    为该参数指定了形状，则在区间内做非线性变换。
    """
    lo, hi = cfg['prior'][key][0], cfg['prior'][key][1]
    if key == 'log_phi0':                      # 下游 rate_2d_E 需要线性 phis
        return 10.0 ** (lo + u * (hi - lo))
    shape = cfg.get('prior_shape', {}).get(key)
    if shape and shape.get('kind') == 'truncnorm':
        from scipy.stats import truncnorm
        m, s = float(shape['mu']), float(shape['sigma'])
        return truncnorm.ppf(u, (lo - m) / s, (hi - m) / s, loc=m, scale=s)
    return lo + u * (hi - lo)


def apply_dm_model(dis, cfg):
    """把 config 的 dm_model 段注入 AstroDistribution 实例（附录 A 规则 R4/R5）。

    推断端与模拟端都必须调用，保证两端单一来源（历史上"两端各自实现"是本仓库的 bug 类别）。
    """
    dm = cfg.get('dm_model', {})
    host = dm.get('host', {})
    halo = dm.get('halo', {})
    igm = dm.get('igm', {})
    dis.host_dm_evol = float(host.get('host_dm_evol', 0.84))
    dis.n_evol = float(cfg.get('analysis', {}).get('n_evol', 1.0))
    dis.sfr_curve = str(cfg.get('analysis', {}).get('sfr_curve', dis.sfr_curve))
    dis.DMsmin = float(halo.get('DMsmin', 0.0))
    dis.DMsmax = float(halo.get('DMsmax', 60.0))
    # 晕窗主线 U[0,60]（均值 30，与 Shin §2.3 的"盘 50 + 晕 30 = 80"分解一致）。
    # 不设副线开关：均匀窗被 IntDMsrc_LogNormal 解析积掉，不增加采样维度，
    # 与固定值跑的是同一条线、同一张图，只让 ln_mu_host 的误差棒更诚实。
    # CC_HALO 环境变量覆盖已于 2026-10-09 删除（多线产出数据混乱，且晕与
    # ln_mu_host 在固定 z 处精确简并，副线提供的信息量有限）。
    dis.igm_F = float(igm.get('F', dis.igm_F))
    dis.igm_alpha = float(igm.get('alpha', dis.igm_alpha))
    dis.igm_beta = float(igm.get('beta', dis.igm_beta))
    dis.igm_zmin_eff = float(igm.get('zmin_eff', dis.igm_zmin_eff))
    n = int(igm.get('N_nodes', dis.N_igm))
    if n != dis.N_igm or not hasattr(dis, '_ig_cache'):
        dis.N_igm = n
        dis._ig_cache = {}          # 节点数/参数变了必须清空缓存，否则用旧表
    return dis


def build_models(cfg):
    """按 config 组装 (cos, dis, er) —— 四个入口脚本唯一的合法构造入口。

    宇宙学与 dm_model 两段必须同时注入：过去每个脚本各写一遍这两步，结果漏了
    pltpz 的 dm_model（P(z) 图与采样用不同晕窗）和 simufrb 的 H0/Omega_b
    （改 config 的 H0 会让模拟端与推断端宇宙学不一致）。
    er 与 dis 必须共享同一个 cos：EventRate 自带独立的默认实例。
    """
    cc = cfg.get('cosmology', {})
    cos = Cosmology(omegam=cc.get('Omega_m', 0.308), omegal=cc.get('Omega_L', 0.692),
                    omegab=cc.get('Omega_b', 0.0484), h0=cc.get('H0', 67.81) / 100.0)
    dis = AstroDistribution()
    dis.cos = cos
    apply_dm_model(dis, cfg)
    # 模式挂在 dis 上；er.ad 指向同一实例 ⇒ Norm1D_E / rate_2d_E / log_distr_efdmwz
    # 三处必然同源（H1 的归一化配对由构造保证，不靠自觉）
    dis.sel_mode, dis.sel_k = chime_selection.resolve_selection(cfg)
    er = EventRate()
    er.cos = cos
    er.ad = dis
    return cos, dis, er


def read_selection_survival(catalog_path, mode='step'):
    """读 filtered.fits 的样本存活率（Poisson 项/到达率的选择修正）。

    背景（选择函数缺口）：rate_2d_E/Norm1D_E 的选择函数只含辐射计阈值
    （SNR>=sn0 + eps 波束），而样本还经过了"one-off + 有 fluence + 六项质量
    筛选"——Poisson 项对比的观测数是筛选后的 NSAMPLE，模型预测的却是筛选前
    的检出数 NREF10。修复：Λ_selected = Λ_det × SURVIV，
    SURVIV = NSAMPLE/NREF10，由 preprocess_catalog.py 从真实筛选链与 catalog
    的 snr_fitb 分布实测后写入 FITS header。header 缺失时当场报错——
    默认 S=1 等于静默复现本 bug，绝不允许。

    数学要点：per-event 的 norm 不需要乘 S——S̄ 在 Λ_sel 与 C_sel（检测区
    形状密度积分）中同比例出现，精确抵消；log-uniform 先验（p(φ)∝1/φ）下
    形状边际含 S^{−N} 纯幂因子 Γ(N)/c^N，对一切形状参数严格抵消，
    S 只移动 phis（口径选择不改 log_Echar/γ/宽度/宿主的边际）。

    mode 参数（2026-10-05）：决定分母用哪个参照计数（step→NREF10，smooth→NSNR12）。

    覆盖范围（勿当作"已完全对齐"）：六道筛全部以常数存活率进入泊松项，
    mock 无法逐事件模拟——模型 boxcar SNR 与目录 snr_fitb 的条件分位实测
    失配（mock 17.2% vs 实测 7.7%，P(SNR<12|SNR>=10)，见 test_models_local
    V17），故 snr>=12 这道未做阈值化精确处理（若两估计量将来对齐可升格
    sn0=12，须 Norm1D_E/rate_2d_E/simufrb 三处同步）；scat<=10ms
    （716/3024=23.7%，最大单项）是唯一有对称性、可原理性进入选择函数的
    一道（tau(DM,z) 依赖探测概率），是 S 的下一轮缩水方向；dm_fitb>=100
    实测纯 no-op；DM_MW<=100 与 notes 无对称性，只能留常数。S̄ 若将来
    θ 依赖化，必须同步进 Norm1D_E 与 rate_2d_E，否则 S̄(θ)^N 以 N=1688
    杠杆扭曲后验。（区分 2026-10-05：本条禁的是【外部实测常数存活率】被
    做成 θ 依赖——那会让一个测量值携带 S̄(θ)^N 杠杆；选择函数 S 作为模型
    前向结构进入 Λ_S 与 C_sel（地位同 rate_2d_E 里的 IntE）是合法的
    θ 依赖，两者不冲突。）

    env CC_SURVIVAL 覆盖（约定系统误差线）：0.4723 = 1688/3574 为 one-off
    口径（one-off ∧ snr>=10 参照，杠杆 3574/4840 → 1.355×）；缺省为
    全部 burst 口径（含重复暴 burst 计数）。
    """
    env = os.environ.get('CC_SURVIVAL')
    if env:
        s = float(env)
        if not (0.0 < s <= 1.0):
            raise ValueError("CC_SURVIVAL 需在 (0,1]，得到 %r" % env)
        print("[survival] CC_SURVIVAL 覆盖 -> S = %.4f" % s)
        return s
    try:
        from astropy.io import fits
        with fits.open(catalog_path) as hdul:
            hdr = hdul[1].header
        n_sam = int(hdr['NSAMPLE'])
        # 参照计数随选择函数口径切换：step 的阈值是 sn0=10 ⇒ 分母 NREF10；
        # smooth 的 S 本身就是 snr>12 的探测概率 ⇒ 阈值已由模型承担，分母换
        # NSNR12，否则 snr>=12 这道筛被计两遍（S 里一次、SURVIV 里一次）。
        ref_key = 'NREF10' if mode == 'step' else 'NSNR12'
        n_ref = int(hdr[ref_key])
        s = n_sam / n_ref
        if mode == 'step' and 'SURVIV' in hdr                 and abs(float(hdr['SURVIV']) * n_ref - n_sam) > 0.5:
            raise ValueError("header 不自洽：SURVIV=%r 与 NSAMPLE/NREF10=%.6f 不符"
                             % (hdr['SURVIV'], s))
    except FileNotFoundError:
        raise FileNotFoundError(
            "找不到 %s：Poisson 项需要样本存活率 SURVIV，先运行 "
            "preprocess_catalog.py 生成带 header 的 filtered.fits" % catalog_path)
    except KeyError as e:
        raise KeyError(
            "%s 缺少 header 键 %s（旧版预处理产物）。默认 S=1 等于静默复现"
            "选择函数缺口的 bug，故当场报错；重跑 preprocess_catalog.py 重新生成"
            % (catalog_path, e))
    if not (0.0 < s <= 1.0):
        raise ValueError("header SURVIV=%r 非法（需在 (0,1]）" % s)
    if n_ref < n_sam:
        raise ValueError("参照计数 %s=%d 小于 NSAMPLE=%d，不可能" % (ref_key, n_ref, n_sam))
    print("[survival] Poisson/到达率存活率 S = %.4f (NSAMPLE=%d / %s=%d, mode=%s, %s)"
          % (s, n_sam, ref_key, n_ref, mode, catalog_path))
    return s


_TRUTH_KEYS = {
    'log_phi0':      ('phis', 'phi_star', 'phi*'),
    'gamma':         ('gamma', 'alpha'),
    'log_Echar':     ('log_Echar', 'logEs', 'log_Es'),
    'log_E0':        ('log_E0', 'logE0'),
    'mu_w':          ('mu_w', 'mu'),
    'sigma_w':       ('sigma_w', 'sigma'),
    'ln_mu_host':    ('ln_mu_host', 'mu_host'),
    'ln_sigma_host': ('ln_sigma_host', 'sigma_host'),
}


def truth_vector(path):
    """注入真值 JSON -> PARAM_ORDER 顺序的向量（pltpost / pltpz 共用的唯一实现）。

    JSON 按原始物理单位写（phis 是线性值），这里对 log_phi0 取 log10 以与后验链对齐；
    其余键容忍多种历史写法。缺参数键直接报错，不静默补零。
    """
    missing = [k for k in PARAM_ORDER if k not in _TRUTH_KEYS]
    if missing:
        raise KeyError("PARAM_ORDER 新增了参数但没有对应真值键: %s" % missing)
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    out = []
    for key in PARAM_ORDER:
        for k in _TRUTH_KEYS[key]:
            if k in d:
                out.append(float(d[k]))
                break
        else:
            raise KeyError("%s 里找不到 %s 的参数键(试过的写法: %s)"
                           % (path, key, "/".join(_TRUTH_KEYS[key])))
    v = np.array(out)
    v[PARAM_ORDER.index('log_phi0')] = np.log10(v[PARAM_ORDER.index('log_phi0')])
    return v


def gammainc(alpha, x):
    if alpha==0:
        return -spf.expi(-x)

    elif (alpha<0):
        return (gammainc(alpha+1,x)-np.power(x, alpha)*np.exp(-x))/alpha

    else:
        return spf.gammaincc(alpha,x)*spf.gamma(alpha)

def Sampling1D(x, y, x1, x2, n):
    ymax = np.max(y)
    if ymax <= 0:
        raise ValueError("Sampling1D: 目标分布在采样范围内全为零，无法采样")
    fuc = interp1d(x, y / ymax, bounds_error=False, fill_value=0.0)
    nt = 0
    res = np.array([])
    max_iter = 10000
    iter_cnt = 0
    while (nt < n):
        iter_cnt += 1
        if iter_cnt > max_iter:
            raise RuntimeError(
                f"Sampling1D: 达到最大迭代次数 {max_iter}，仅采到 {nt}/{n} 个样本，"
                f"接受率过低，请检查分布形状或采样范围")
        vx = np.random.uniform(x1, x2, n - nt)
        vy = np.random.uniform(0, 1, n - nt)
        res = np.append(res, vx[vy <= fuc(vx)])
        nt = len(res)
    return res
