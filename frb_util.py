import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
import scipy.special as spf
import json
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
    with open(p) as f:
        return json.load(f)

# FRB 谱指数，参考 Shin 2023
ALPHA_SPEC = -1.39

# 望远镜参数库（硬编码，方便后续扩展）
# dnu: 观测帧带宽 Δν_obs [MHz]
TELESCOPES = {
    'CHIME': {
        'g': 1.4,        # Gain [K/Jy]
        'bw': 400,       # Bandwidth [MHz] (观测帧)
        'tsys': 50.0,    # System temperature [K]
        'fov': 200.0,    # Field of view [deg^2]
        'npol': 2,       # Polarization channels
        'sn0': 10.0,     # Detection threshold SNR
        'dnu': 400.0,    # 观测带宽 Δν_obs [MHz] (400-800 MHz)
    },
    # 后续加新望远镜在这里添加：
    # 'Parkes': {'g': 0.6, 'bw': 300, 'tsys': 25, 'fov': 4.0, 'npol': 2, 'sn0': 10, 'dnu': 300},
}

class Cosmology:
    _cache = {}  # (omegam, omegal, omegab) -> instance，避免重复积分

    def __init__(self, omegam=0.308, omegal=0.692, omegab=0.0484):
        key = (omegam, omegal, omegab)
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
        self.h0 = 0.6781
        self.H0 = self.h0 * 100 * self.km2cm / self.Mpc2cm
        self.Rhoc = 1.88 * self.h0 * self.h0 * 1e-29    #The critical density of universe in gram/cm^3
        self.Nc = self.Rhoc / 1.6726e-24       #The number density of universe in Hydrogen atom, in units of 1/cm^3
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

    def Energy(self, z, flu=1.0, dnu=400.):
        """计算各向同性能量 E_iso (Lin et al. 2024, ApJ 962, 73, Eq.7)

        公式: E = 4π D_L² · Δν_obs · F / (1+z)^(2+α)
        其中谱指数 α = -1.39 (Shin 2023)，即分母 (1+z)^0.61
        Δν_obs 为观测帧带宽（CHIME 400-800 MHz，Δν=400 MHz）

        flu: fluence [Jy·ms]
        dnu: 观测帧带宽 Δν_obs [MHz]
        z: redshift
        """
        ld = self.Luminosity_Distance(z)
        ld2 = ld * ld
        ener = flu * self.Jyms2CGS * dnu * self.MHz2Hz * 4 * np.pi * ld2 / np.power(1+z, 2 + ALPHA_SPEC)
        return ener

    def DispersionMeasure_IGM(self, z, chi=7./8):
        """
        calculate dispersion measure of intergalactic medium by integrating redshift
        """
        return self.dmigm(z) * self.f_IGM * chi / self.Mpc2cm * 1e6

    def Energy_to_Flu(self, z, ener, dnu=400.):
        """从内禀能量反推观测 fluence (Energy 的反函数)

        公式: F = E · (1+z)^(2+α) / (4π D_L² · Δν_obs)
        其中谱指数 α = -1.39 (Shin 2023)，即分子 (1+z)^0.61
        Δν_obs 为观测帧带宽（CHIME 400-800 MHz，Δν=400 MHz）
        """
        ld = self.Luminosity_Distance(z)
        ld2 = ld*ld
        flu = ener * np.power(1+z, 2 + ALPHA_SPEC) / 4 / np.pi / ld2 / dnu / self.MHz2Hz / self.Jyms2CGS
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
        self.vpar_etg = np.array([0.001713, 1.099, 0.2965, 0.01246, 1.055, 0.7262])
        self.vpar_ltg_ne2001 = np.array([0.01715, 1.062, 0.5202, 0.00416, 0.7227, 1.151])
        self.vpar_ltg_ymw16 = np.array([0.01561, 0.759, 0.3013, 0.01889, 1.042, 0.5791])
        self.vpar_alg_ne2001 = np.array([0.005485, 0.8665, 1.009, 0.01406, 1.069, 0.5069])
        self.vpar_alg_ymw16 = np.array([0.01199, 0.7597, 0.3082, 0.01735, 1.048, 0.6025])
        self.Zmax = 5
        self.Zmin = 2e-6
        # MW 晕 DM 均匀分布上限 [pc/cm^3]: U[0,60] 均值 30 (Dolag et al. 2015)。
        # 总 DM_MW = DM_MW,ISM(目录 dm_exc 逐事件, 中位 ~50) + 晕均值 30 ≈ 80
        # (Shin 2023, ApJ 944, 105, 附录 A.2: DM_MW 固定 80, 偏差由宿主吸收;
        #  本管线宿主为固定参数, 用均匀卷积代偿吸收)。观测帧, 不乘 (1+z)。
        self.DMsmax = 60.
        self.Wmax = 20
        self.Wmin = 0.05

    def log_IntBeam(self, logl, alpha, logls, logl0):
        ratio = np.power(10., logl-logls)
        lik0 = gammainc(alpha+1, ratio) - gammainc(alpha+1, 2*ratio)
        lik = lik0/np.log(2)      # Beam efficiency from 50% to 100%
        lik = np.where(np.isfinite(lik) & (lik > 0), lik, 1e-199)
        loglik = np.log(lik)
        # 用 np.where 代替布尔索引赋值，同时支持标量和数组输入
        loglik = np.where(logl < logl0, -1e30, loglik)
        return loglik

    # --- E_iso 能量函数 ---
    def Schechter_E_log(self, loge, phis, alpha, logEs):
        """Schechter 能量函数 per dex"""
        e = np.power(10., loge)
        es = np.power(10., logEs)
        phi = np.log(10) * phis * np.power(e / es, (alpha + 1)) * np.exp(-e / es)
        return phi

    def IntE(self, eps, alpha, logEs, logE_min):
        """Schechter 能量累积积分"""
        with np.errstate(invalid='ignore'):
             ratio = np.power(10., logE_min-logEs)
        return gammainc(alpha+1, ratio/eps)

    def log_IntBeam_E(self, loge, alpha, logEs, logE0):
        """波束卷积后的 Schechter 能量函数（与 log_IntBeam 数学形式相同）"""
        return self.log_IntBeam(loge, alpha, logEs, logE0)
    
    def Distribution_Local_galaxy_DM(self, dmv, vpar):
        """
        General form of DM distribution function of host galaxies, using double gaussian function in logarithmic DM
        vpar[i] is the i-th parameter of this distribution function
        """
        val = vpar[0] * np.exp(-np.power((dmv - vpar[1]) / vpar[2], 2.)) \
              + vpar[3] * np.exp(-np.power((dmv - vpar[4]) / vpar[5], 2.))
        return val

    def Distribution_HostGalaxyDM(self, dmv0, fgalaxy_type=None, vpar=np.array([0,50])):
        """
        DM distribution functions of different type of host galaxies
        ETG: early-tyep galaxies
        LTG: late-type galaxies
        ALG: all the galaxies
        NE2001: the referenced galaxy electron density using NE2001 model
        YMW16: the referenced galaxyt electron density using YMW16 model
        """
        # 注意:每个分支必须显式 return。早期版本把 None 分支写成赋值后落入
        # else,但 Python 的 if/elif/else 是互斥分支,命中 if 后不会再到 else,
        # 导致默认调用隐式返回 None。现已改为每个分支独立 return。
        if not fgalaxy_type:
            # 默认使用 log 空间双高斯（vpar_etg），与 log_IntDMsrc 默认分支同源，
            # 避免"默认分布用线性高斯、解析积分却假设 log 双高斯"的不一致（原 bug 4）
            dmv = dmv0.copy()
            dmv[dmv0<1e-9] = np.ones(dmv[dmv0<1e-9].shape)*1e-9
            res = self.Distribution_Local_galaxy_DM(np.log10(dmv), self.vpar_etg)
            res[dmv0<=0] = np.zeros(res[dmv0<=0].shape)
            return res
        if fgalaxy_type == 'ETG':
            dmv = dmv0.copy()
            dmv[dmv0<1e-9] = np.ones(dmv[dmv0<1e-9].shape)*1e-9
            res = self.Distribution_Local_galaxy_DM(np.log10(dmv), self.vpar_etg)
            res[dmv0<=0] = np.zeros(res[dmv0<=0].shape)
            return res
        if fgalaxy_type == 'LTG_NE2001':
            dmv = dmv0.copy()
            dmv[dmv0<1e-9] = np.ones(dmv[dmv0<1e-9].shape)*1e-9
            res = self.Distribution_Local_galaxy_DM(np.log10(dmv),
                    self.vpar_ltg_ne2001)
            res[dmv0<=0] = np.zeros(res[dmv0<=0].shape)
            return res
        if fgalaxy_type == 'LTG_YMW16':
            dmv = dmv0.copy()
            dmv[dmv0<1e-9] = np.ones(dmv[dmv0<1e-9].shape)*1e-9
            res = self.Distribution_Local_galaxy_DM(np.log10(dmv),
                    self.vpar_ltg_ymw16)
            res[dmv0<=0] = np.zeros(res[dmv0<=0].shape)
            return res
        if fgalaxy_type == 'ALG_NE2001':
            dmv = dmv0.copy()
            dmv[dmv0<1e-9] = np.ones(dmv[dmv0<1e-9].shape)*1e-9
            res = self.Distribution_Local_galaxy_DM(np.log10(dmv), self.vpar_alg_ne2001)
            res[dmv0<=0] = np.zeros(res[dmv0<=0].shape)
            return res
        if fgalaxy_type == 'ALG_YMW16':
            dmv = dmv0.copy()
            dmv[dmv0<1e-9] = np.ones(dmv[dmv0<1e-9].shape)*1e-9
            res = self.Distribution_Local_galaxy_DM(np.log10(dmv), self.vpar_alg_ymw16)
            res[dmv0<=0] = np.zeros(res[dmv0<=0].shape)
            return res
        # 兜底:既非预定义类型也非 None,允许用户传可调用对象
        if callable(fgalaxy_type):
            return fgalaxy_type(dmv0, vpar)
        raise ValueError(f"未识别的 fgalaxy_type: {fgalaxy_type!r}，"
                         f"可选: ETG / LTG_NE2001 / LTG_YMW16 / ALG_NE2001 / ALG_YMW16 或自定义可调用对象")

    def SFR(self, z):
        """
        Star-forming history, the values taken from Hopkins & Beacom (2016)
        """
        sfr = (0.017 + 0.13 * z)/(1 + np.power(z/3.3, 5.3))
        return sfr

    def SFR_evolution(self, z):
        """
        Star-formation rate history, adopted from Yuksel et al. (2008)
        用于 FRB 事件率密度的宇宙学演化
        """
        a = 3.4; b = -0.3; c = -3.5; B = 5e3; C = 9; eta = -10
        p1 = np.power(1+z, a*eta)
        p2 = np.power((1+z)/B, b*eta)
        p3 = np.power((1+z)/C, c*eta)
        p = p1+p2+p3
        return np.power(p, 1./eta)

    def evolution_factor(self, z):
        """FRB 事件率宇宙学演化因子，归一化到 z=0

        返回 SFR_evolution(z) / SFR_evolution(0)，使 phis 代表本地事件率密度
        """
        return self.SFR_evolution(z) / self.SFR_evolution(0.0)

    def kappa(self, z):
        """
        Normalized SFH from redshift of z to redshift of 0 (nearby universe)

        统一使用 SFR_evolution（Yuksel et al. 2008），与 evolution_factor / 事件率
        演化保持同一套 SFR 曲线，避免 mock 数据与推断端 DM-z 关系错位。
        """
        return np.sqrt(self.SFR_evolution(0)/self.SFR_evolution(z))

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

    def IntDMsrc(self, u1, u2, vpar):
        """
        Analytic integral when marginalizing the uniform DM_MW,halo distribution
        """
        a1 = vpar[0]
        b1 = vpar[1]
        c1 = vpar[2]
        a2 = vpar[3]
        b2 = vpar[4]
        c2 = vpar[5]
        k1 = np.power(10.,b1)*a1*c1*np.exp(c1*c1*np.log(10)*np.log(10)/4)
        k2 = np.power(10.,b2)*a2*c2*np.exp(c2*c2*np.log(10)*np.log(10)/4)
        q1 = (c1*c1*np.log(10)*np.log(10)+b1*np.log(100)-2*np.log(u1))/c1/np.log(100)
        q2 = (c1*c1*np.log(10)*np.log(10)+b1*np.log(100)-2*np.log(u2))/c1/np.log(100)
        q3 = (c2*c2*np.log(10)*np.log(10)+b2*np.log(100)-2*np.log(u1))/c2/np.log(100)
        q4 = (c2*c2*np.log(10)*np.log(10)+b2*np.log(100)-2*np.log(u2))/c2/np.log(100)
        int_h = np.log(10)*np.sqrt(np.pi)/2 * (k1*(spf.erf(q2)-spf.erf(q1))+k2*(spf.erf(q4)-spf.erf(q3)))
        int_hs = int_h/self.DMsmax   #   integral including uniform DM_MW,halo
        return int_hs

    def log_IntDMsrc(self, u1, u2, gtype=None):
        """
        Logarithmic integrals of above marginalization in different galaxy type cases
        """
        if not gtype:
            # 默认使用 vpar_etg（与 Distribution_HostGalaxyDM 默认分支同源）
            res = np.log(np.maximum(self.IntDMsrc(u1, u2, self.vpar_etg), 1e-300))
            return res
        elif gtype == 'ETG':
            res = np.log(np.maximum(self.IntDMsrc(u1, u2, self.vpar_etg), 1e-300))
            return res
        elif gtype == 'LTG_NE2001':
            res = np.log(np.maximum(self.IntDMsrc(u1, u2, self.vpar_ltg_ne2001), 1e-300))
            return res
        elif gtype == 'LTG_YMW16':
            res = np.log(np.maximum(self.IntDMsrc(u1, u2, self.vpar_ltg_ymw16), 1e-300))
            return res
        elif gtype == 'ALG_NE2001':
            res = np.log(np.maximum(self.IntDMsrc(u1, u2, self.vpar_alg_ne2001), 1e-300))
            return res
        elif gtype == 'ALG_YMW16':
            res = np.log(np.maximum(self.IntDMsrc(u1, u2, self.vpar_alg_ymw16), 1e-300))
            return res
        else:
            return -1e30
           
    def dis_logw(self, logw0, mu, sigma):
        a = 1./np.sqrt(2.*np.pi*sigma*sigma)
        b = np.exp(-(logw0-mu)*(logw0-mu)/2/sigma/sigma)
        return a*b

    def log_dis_logw(self, logw0, mu, sigma):
        a = 2*np.pi*sigma*sigma
        b = -(logw0-mu)*(logw0-mu)/2/sigma/sigma
        return b-1/2.*np.log(a)

    def log_distr_efdmwz(self, dnu, logflux, dme, logw, z, alpha, logEs, logE0, mu, sigma, gtype=None):
        """能量版联合概率 p(E_iso, DM_host | z)

        将观测流量 logflux 和脉冲宽度 logw 耦合为 E_iso，再用 Schechter 能量函数评估。
        物理关系：E_iso = S * w * dnu * 4π * D_L² / (1+z)^(2+α)，谱指数 α = -1.39 (Shin 2023)
        """
        flux = np.power(10., logflux)
        w_ms = np.power(10., logw)       # 脉冲宽度 [ms]
        fluence = flux * w_ms            # fluence = S * w [Jy·ms]
        loge = np.log10(self.cos.Energy(z, flu=fluence, dnu=dnu))
        logint1 = self.log_IntBeam_E(loge, alpha, logEs, logE0)
        logw0 = logw - np.log10(1+z)
        logfw = self.log_dis_logw(logw0, mu, sigma)
        # p(z) ∝ (dV/dz) · evolution(z) / (1+z)
        # /(1+z) 为源帧→观测帧时间膨胀因子,与 Norm1D_E / rate_2d_E / simufrb.py 采样端一致。
        logfz = self.log_Distribution_volume(z) - np.log(1+z)
        dmi = self.cos.DispersionMeasure_IGM(z)
        # DM_MW = DM_MW,ISM + DM_MW,halo。目录 dme(dm_exc) 已扣 ISM 部分,残差
        # r = dme - dmi = DM_MW,halo + DM_host/(1+z),其中 DM_MW,halo ~ U[0, DMsmax]
        # 位于观测帧(z≈0),不随 (1+z) 缩放。对 δ∈[0,DMsmax] 卷积后宿主 DM 窗口为
        # [(r-DMsmax)(1+z), r(1+z)]·κ,即 u2 先在观测帧扣 DMsmax 再乘 (1+z)。
        u1 = (dme-dmi)*(1+z)*self.kappa(z)
        u2 = (dme-dmi-self.DMsmax)*(1+z)*self.kappa(z)
        # u2 是 DM_host 卷积窗口下边界,物理上 host DM ≥ 0,所以 u2<0 时应截断到 0
        # (从 0 积到 u1),而不是把整条似然清零。早期版本用 ind = u2>0 把 u2<=0 的
        # 事件 logint2 置 -1e30,会把"残差 DM 较小、host DM 窗口被 0 截断"这类物理
        # 上正常的事件强制赋予零似然,系统性压低似然。
        # 真正物理上不可能的是 u1<=0(IGM DM 已超过观测 DM,即使 host DM=0 都无法解释)。
        u2_clip = np.maximum(u2, 1e-6)   # 截断到小正数,避免 IntDMsrc 内部 log(u2) 产生 nan
        # 用 np.where 代替布尔索引赋值，同时支持标量和数组输入
        # 对 u1<=0 的元素用 1e-6 占位计算(结果会被 np.where 丢弃),避免传 nan 给 erf
        u1_safe = np.where(u1 > 0, u1, 1e-6)
        logint2_raw = self.log_IntDMsrc(u1_safe, u2_clip, gtype=gtype)
        logint2 = np.where(u1 > 0, logint2_raw, -1e30)
        # DM_MW,halo 为观测帧项:对 δ∈[0,DMsmax] 卷积时,逐点密度雅可比 (1+z) 与
        # 卷积测度换元 dδ=-dX/(1+z) 精确抵消,净密度即 IntDMsrc(u1,u2)/DMsmax,
        # 不再额外补 log(1+z)(旧版把均匀项放在宿主帧才需要)。
        loglikv = logint1 + logfz + logfw + logint2 + np.log(self.evolution_factor(z))
        return loglikv

    def log_distr_efdmw_grid(self, dnu, logflux, dme, logw, vz, alpha, logEs, logE0, mu, sigma, gtype=None):
        """逐事件×逐红移对数似然网格 (nz, N)

        log_distr_efdmw 的 z 边际化与 pltpz.py 的逐事件 P(z) 可视化共用此网格。
        logflux/dme/logw 接受标量或 (N,) 数组, vz 为 (nz,) 红移网格;
        返回未边际化的 log p(z, 事件 | 参数)，z 积分测度 ∝ vz·dz 由调用方处理。
        """
        return self.log_distr_efdmwz(dnu, np.atleast_1d(logflux), np.atleast_1d(dme),
                                     np.atleast_1d(logw), np.atleast_1d(vz)[:, np.newaxis],
                                     alpha, logEs, logE0, mu, sigma, gtype=gtype)

    def log_distr_efdmw(self, dnu, logflux, dme, logw, alpha, logEs, logE0, mu, sigma, gtype=None):
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
                                                alpha, logEs, logE0, mu, sigma, gtype=gtype))
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

    def Norm1D_E(self, sn0, bw, npol, g, tsys, dnu, alpha, logEs, logE0, mu, sigma):
        """能量版归一化因子（完全向量化）

        检测阈值从 L_min 转换为 E_min：E_min = S_min * w_obs * dnu * 4π * D_L² / (1+z)^(2+α)，谱指数 α = -1.39 (Shin 2023)
        三维广播 (nz, nlogw, neps) 一次性计算，消除 Python z 循环。
        内存: 200×200×100×8B = 32MB
        """
        # 网格: z=200, eps=100, logw=200 (4M 元素, gammainc 约 1.4s)
        # gammainc 是绝对瓶颈(0.34μs/元素), 向量化不改变总计算量;
        # eps 维度对 gammainc 影响最大(广播维), 保持 100;
        # z/logw 保持 200, 精度提升靠 log_distr_efdmw 的 z 网格加大
        nz, neps, nlogw = 200, 100, 200
        stepz = (np.log(self.Zmax) - np.log(self.Zmin)) / nz
        vz = np.exp(np.arange(np.log(self.Zmin), np.log(self.Zmax), stepz))      # (nz,)
        stepeps = (1-0.5) / neps
        veps = np.arange(0.5, 1, stepeps)                                         # (neps,)
        steplogw = (np.log10(self.Wmax) - np.log10(self.Wmin)) / nlogw
        vlogw = np.arange(np.log10(self.Wmin), np.log10(self.Wmax), steplogw)     # (nlogw,)

        # 三维广播: z(nz,1,1) × vlogw(1,nlogw,1) × veps(neps,)
        z3d = vz[:, np.newaxis, np.newaxis]                                       # (nz,1,1)
        vw = np.power(10, vlogw)[np.newaxis, :, np.newaxis] * (1 + z3d)           # (nz,nlogw,1)
        ft = self.tel.RMEq(sn0, g, tsys, npol, bw, vw)                            # (nz,nlogw,1)
        fw = ft * vw                                                              # (nz,nlogw,1)
        et = self.cos.Energy(z3d, flu=fw, dnu=dnu)                                # (nz,nlogw,1)
        loget = np.log10(et)
        loget = np.where(np.isfinite(loget) & (loget >= logE0), loget, logE0)
        # IntE: veps(neps,) × loget(nz,nlogw,1) → (nz,nlogw,neps)
        int_eps_grid = self.IntE(veps, alpha, logEs, loget)                       # (nz,nlogw,neps)
        int_eps = np.sum(int_eps_grid / veps[np.newaxis, np.newaxis, :] / np.log(2) * stepeps, axis=2)  # (nz,nlogw)
        int_w = np.sum(int_eps * self.dis_logw(vlogw, mu, sigma)[np.newaxis, :] * steplogw, axis=1)     # (nz,)
        # /(1+z) 为源帧→观测帧时间膨胀因子，与 rate_2d_E / simufrb.py 采样端一致
        fz = self.Distribution_volume(vz) * self.evolution_factor(vz) / (1.0 + vz)  # (nz,)
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

    def rate_2d_E(self, sn0, bw, npol, g, tsys, dnu, phis, alpha, logEs, logE0, mu, sigma):
        """能量版事件率密度 rho_deg [deg^-2 hr^-1]（完全向量化）

        检测阈值从 L_min 转换为 E_min：E_min = S_min * w_obs * dnu * 4π * D_L² / (1+z)^(2+α)，谱指数 α = -1.39 (Shin 2023)
        三维广播 (nz, nlogw, neps) 一次性计算，消除 Python z 循环。
        与 Norm1D_E 结构同构，仅 fz 用 dVdOdz 且多乘 phis。
        """
        # 网格: z=200, eps=100, logw=200 (与 Norm1D_E 一致)
        nz, neps, nlogw = 200, 100, 200
        stepz = (np.log(self.ad.Zmax) - np.log(self.ad.Zmin)) / nz
        vz = np.exp(np.arange(np.log(self.ad.Zmin), np.log(self.ad.Zmax), stepz))  # (nz,)
        stepeps = (1-0.5) / neps
        veps = np.arange(0.5, 1, stepeps)                                           # (neps,)
        steplogw = (np.log10(self.ad.Wmax) - np.log10(self.ad.Wmin)) / nlogw
        vlogw = np.arange(np.log10(self.ad.Wmin), np.log10(self.ad.Wmax), steplogw) # (nlogw,)

        # 三维广播: z(nz,1,1) × vlogw(1,nlogw,1) × veps(neps,)
        z3d = vz[:, np.newaxis, np.newaxis]                                         # (nz,1,1)
        vw = np.power(10, vlogw)[np.newaxis, :, np.newaxis] * (1 + z3d)             # (nz,nlogw,1)
        ft = self.tel.RMEq(sn0, g, tsys, npol, bw, vw)                              # (nz,nlogw,1)
        fw = ft * vw                                                                # (nz,nlogw,1)
        et = self.cos.Energy(z3d, flu=fw, dnu=dnu)                                  # (nz,nlogw,1)
        loget = np.log10(et)
        loget = np.where(np.isfinite(loget) & (loget >= logE0), loget, logE0)
        # IntE: veps(neps,) × loget(nz,nlogw,1) → (nz,nlogw,neps)
        int_eps_grid = phis * self.ad.IntE(veps, alpha, logEs, loget)               # (nz,nlogw,neps)
        int_eps = np.sum(int_eps_grid / veps[np.newaxis, np.newaxis, :] / np.log(2) * stepeps, axis=2)  # (nz,nlogw)
        int_w = np.sum(int_eps * self.ad.dis_logw(vlogw, mu, sigma)[np.newaxis, :] * steplogw, axis=1)   # (nz,)
        fz = self.cos.dVdOdz(vz)/(1+vz) * self.ad.evolution_factor(vz)              # (nz,)
        rho = np.sum(vz * stepz * fz * int_w)
        rho_deg = rho/self.rad2deg2/self.yr2hr
        return rho_deg

    def log_dis_poi(self, rho, N, Omega, T):
        lamda = rho*Omega*T
        ind = lamda <= 0
        lamda[ind] = 1e-199
        loglik = N*np.log(lamda)-lamda-spf.gammaln(N+1)
        return loglik

class Loadfiles:
    #def __init__(self):

    def LoadCatalogue(self, fname):
        """[DEPRECATED] 遗留 TXT 加载器

        注意：此加载器期望的列名（S/Seu/Sel/W/.../SURVEY/Gain/...）与
        preprocess_catalog.py 输出的 TXT 格式（name/fluence/width/dm_obs/...）
        完全不兼容。mini 管线默认走 --fits 路径（LoadFitsCatalog）。
        若需使用 TXT 路径，需先手动整理为兼容格式或新增适配器。
        """
        cat = np.loadtxt(fname, dtype=str)
        row, col = cat.shape
        cat2 = {}
        for i in range(col):
            cat2[cat[0,i]] = cat[1:, i]

        cat2['S'] = np.array(cat2['S'], dtype=float)
        cat2['Seu'] = np.array(cat2['Seu'], dtype=float)
        cat2['Sel'] = np.array(cat2['Sel'], dtype=float)
        cat2['W'] = np.array(cat2['W'], dtype=float)
        cat2['Weu'] = np.array(cat2['Weu'], dtype=float)
        cat2['Wel'] = np.array(cat2['Wel'], dtype=float)
        cat2['F'] = np.array(cat2['F'], dtype=float)
        cat2['Feu'] = np.array(cat2['Feu'], dtype=float)
        cat2['Fel'] = np.array(cat2['Fel'], dtype=float)
        cat2['DM'] = np.array(cat2['DM'], dtype=float)
        cat2['DM_NE2001'] = np.array(cat2['DM_NE2001'], dtype=float)
        cat2['DM_YMW16'] = np.array(cat2['DM_YMW16'], dtype=float)
        cat2['SURVEY'] = np.array(cat2['SURVEY'])
        cat2['Gain'] = np.array(cat2['Gain'], dtype=float)
        cat2['Tsys'] = np.array(cat2['Tsys'], dtype=float)
        cat2['BW'] = np.array(cat2['BW'], dtype=float)
        cat2['Npol'] = np.array(cat2['Npol'], dtype=float)
        cat2['SN0'] = np.array(cat2['SN0'], dtype=float)
        return cat2

    def LoadSvyInfo(self, fname):
        cat = np.loadtxt(fname, dtype=str)
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


def gammainc(alpha, x):
    if alpha==0:
        return -spf.expi(-x)

    elif (alpha<0):
        return (gammainc(alpha+1,x)-np.power(x, alpha)*np.exp(-x))/alpha

    else:
        return spf.gammaincc(alpha,x)*spf.gamma(alpha)

def getargv(argv, key):
    for i in range(0, len(argv)):
        arg = argv[i]
        if (arg == key):
            return argv[i + 1]

def chkargv(argv, key):
    for i in range(0, len(argv)):
        arg = argv[i]
        if (arg == key):
            return True
    return False

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

def SamplingND(fuc, par_range, maxv_ori, n):
    """拒绝采样 N 维分布。

    与 Sampling1D 类似，maxv 会在发现更高峰值时上调并重置已采样本。
    加 max_iter 防止极端分布（尖峰）下反复重置导致死循环。
    """
    nt = 0
    maxv = maxv_ori
    res = np.array([])
    npar, m = par_range.shape
    res = res.reshape((0, npar))
    max_iter = 10000
    iter_cnt = 0
    while (nt < n):
        iter_cnt += 1
        if iter_cnt > max_iter:
            raise RuntimeError(
                f"SamplingND: 达到最大迭代次数 {max_iter}，仅采到 {nt}/{n} 个样本，"
                f"接受率过低，请检查分布形状或采样范围")
        vpar = np.random.uniform(0, 1, (n-nt, npar))
        for i in range(npar):
            lv = par_range[i,0]
            rv = par_range[i,1]
            vpar[:,i] = vpar[:,i] * (rv-lv) + lv

        vy = np.random.uniform(0, 1, n-nt)*maxv
        fv = fuc(vpar)
        if (np.max(fv) > maxv):
            maxv = np.max(fv)*1.5
            nt = 0
            res = np.array([])
            res = res.reshape((0, npar))
        else:
            res = np.vstack( (res, vpar[vy <= fv,:]))
            nt, m = res.shape
    return res
