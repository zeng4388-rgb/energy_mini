#!/usr/bin/env python3
"""FRB simulator based on Schechter energy function

Samples mock FRBs from the truncated Schechter energy function,
computes observables (fluence, flux, width, DM), and outputs
a catalog for injection-recovery tests.

Usage:
    python simufrb.py -ns 1000 -phis 1e4 -alpha -1.5 -logEs 42 -logE0 38
        -mu 0.0 -sig 0.5 -dnu 400 -fgt ETG -ga 0.7 -npol 2 -bw 300
        -ts 30 -sn0 10 -fov 30 -out simu_out.txt
"""

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

def Simu_FRBs_Eiso(phis, alpha, logEs, logE0, mu, sigma, dnu, ns, fov, npol, g, tsys, bw, sn0, fgt='ETG'):
    """FRB simulator based on the energy Schechter function

    Sampling flow:
    1. Sample E_iso from truncated Schechter function
    2. Sample z from comoving volume distribution
    3. Sample w_rest from log-normal distribution
    4. Compute fluence: F = E_iso * (1+z) / (dnu * 4pi * D_L^2 * Jyms2CGS)
    5. Compute flux: S = F / w_obs
    """
    res = np.zeros((ns, 10))
    ns0 = ns
    nt = 0
    lamda = er.rate_2d_E(sn0, bw, npol, g, tsys, dnu, phis, alpha, logEs, logE0, mu, sigma) * fov
    while ns > 0:
        # sample E_iso (truncated Schechter energy function)
        vlogE = np.arange(logE0 - 1., 48., (48. - logE0 + 1.) / 10000)
        vlik = dis.Schechter_E_log(vlogE, 1, alpha, logEs)
        vlogE = Sampling1D(vlogE, vlik, logE0, 47, ns0)
        vEiso = np.power(10., vlogE)

        # sample epsilon (beam efficiency, 0.5-1)
        vlnEps = np.random.uniform(-np.log(2), 0, ns0)
        vEps = np.exp(vlnEps)

        # sample galaxy redshift
        vZg = np.arange(0, 5.1, 5.1 / 10000)
        vlik = dis.Distribution_volume(vZg)
        vZ = Sampling1D(vZg, vlik, 0, 5.0, ns0)

        # sample pulse width (rest frame)
        vlogW0 = np.arange(-0.5, 1.5, 2. / 10000)
        vlik = dis.dis_logw(vlogW0, mu, sigma)
        vlogW0 = Sampling1D(vlogW0, vlik, -0.4, 1.4, ns0)
        vW0 = np.power(10., vlogW0)       # rest-frame width [ms]
        vW = vW0 * (1 + vZ)               # observed width [ms]

        # sample host galaxy DM
        vDMH0 = np.arange(0, 5001., 5001. / 10000)
        vlik = dis.Distribution_HostGalaxyDM(vDMH0, fgalaxy_type=fgt)
        vDMH0 = Sampling1D(vDMH0, vlik, 0, 5000, ns0)
        vDMH = vDMH0 * np.sqrt(dis.SFR(vZ)) / np.sqrt(dis.SFR(0))

        # sample in-source DM
        vDMS = np.random.uniform(0, 50, ns0)

        # compute IGM DM
        vDMI = cos.DispersionMeasure_IGM(vZ)

        # extragalactic DM
        vDME = (vDMH + vDMS) / (1 + vZ) + vDMI

        # detection threshold
        vft = tel.RMEq(sn0, g, tsys, npol, bw, vW)

        # compute fluence from E_iso: F = E_iso * (1+z) / (dnu * 4pi * D_L^2 * Jyms2CGS)
        vFlu = np.array([cos.Energy_to_Flu(z, e, dnu) for z, e in zip(vZ, vEiso * vEps)])

        # compute flux from fluence: S = F / w_obs
        vFlux = vFlu / vW   # S = F / w_obs [Jy]

        nlen = len(vFlux[vFlux > vft])

        # sample event arrival times (Poisson process, exponential intervals)
        larray = np.repeat(lamda, nlen)
        vT = np.random.exponential(1.0 / larray)
        if nlen > ns:
            nlen = ns

        res[nt:(nt + nlen), 0] = vFlux[vFlux > vft][0:nlen]
        res[nt:(nt + nlen), 1] = vW[vFlux > vft][0:nlen]
        res[nt:(nt + nlen), 2] = vT[0:nlen]
        res[nt:(nt + nlen), 3] = vDME[vFlux > vft][0:nlen]
        res[nt:(nt + nlen), 4] = vft[vFlux > vft][0:nlen]
        res[nt:(nt + nlen), 5] = vlogE[vFlux > vft][0:nlen]  # logE instead of logL
        res[nt:(nt + nlen), 6] = vZ[vFlux > vft][0:nlen]
        res[nt:(nt + nlen), 7] = vDMI[vFlux > vft][0:nlen]
        res[nt:(nt + nlen), 8] = vDMH[vFlux > vft][0:nlen]
        res[nt:(nt + nlen), 9] = vDMS[vFlux > vft][0:nlen]
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
                        help='Reference intrinsic spectral width [MHz]')
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

    res = Simu_FRBs_Eiso(phis, alpha, logEs, logE0, mu, sigma, dnu, Ns, fov, npol, gain, Ts, bw, sn0, fgt=fgt)
    np.savetxt(output, res, delimiter=' ', header="S W T DMe thres logE Z DMi DMh DMs", comments="#")
