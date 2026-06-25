#!/usr/bin/env python3

import numpy as np
from collections import Counter
from scipy import interpolate
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

dnu = 400.0

def lnlik(vpar):
    """Energy-band likelihood function."""
    try:
        norm = np.zeros(vFOV.shape)
        for i in range(len(norm)):
            norm[i] = dis.Norm1D_E(vSN0[i], vBW[i], vNpol[i], vG[i], vTs[i],
                                   dnu, vpar[1], vpar[2], vpar[3], vpar[4], vpar[5])
        loglik_fdm = np.zeros(vN.shape)
        for i in range(len(vN)):
            loglik_fdm[i] = np.sum(
                dis.log_distr_efdmw(dnu, vLOGF_2d[i], vDME_2d[i], vLOGW_2d[i],
                                    vpar[1], vpar[2], vpar[3], vpar[4], vpar[5],
                                    gtype=fgt) - np.log(norm[i]))
        loglik_norm = np.sum(loglik_fdm)
        rho = np.zeros(vFOV.shape)
        for i in range(len(rho)):
            rho[i] = er.rate_2d_E(vSN0[i], vBW[i], vNpol[i], vG[i], vTs[i],
                                  dnu, vpar[0], vpar[1], vpar[2], vpar[3],
                                  vpar[4], vpar[5])
        loglik_poi = np.sum(er.log_dis_poi(rho, vN, vFOV, vTime))
        res = loglik_norm + loglik_poi
        return res
    except Exception:
        print('Numerical error: @', vpar)
        return -1e99

def myprior(cube, ndim, nparams):
    """Transform unit cube to prior ranges for energy parameters."""
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
    parser.add_argument('--config', action='store', dest='config_path', type=str,
                        default='config.json', help='Path to config.json')

    args = parser.parse_args()
    fcat = args.fcat
    fsvy = args.fsvy
    fout = args.fout
    fgt = args.fgt
    bolhalo = args.bolhalo
    use_fits = args.use_fits

    config = load_config(args.config_path)
    dnu = config.get('analysis', {}).get('dnu', 400.0)
    halo_dm = config.get('analysis', {}).get('halo_dm', 0.0)
    cosmo_cfg = config.get('cosmology', {})
    cos = Cosmology(omegam=cosmo_cfg.get('Omega_m', 0.308),
                    omegal=cosmo_cfg.get('Omega_L', 0.692))

    if fcat is None:
        fcat = config['data']['catalog_path']
        if not use_fits:
            use_fits = fcat.endswith('.fits') or fcat.endswith('.fit')
    if fsvy is None:
        fsvy = config['data']['survey_info_path']

    if use_fits:
        vF, vW, vDM_obs, vDM_ne2001, vDM_ymw16, vSVY = lf.LoadFitsCatalog(fcat)
        if fgt and fgt.find('NE2001') >= 0:
            vDME = vDM_ne2001
        else:
            vDME = vDM_ymw16
        if bolhalo:
            vDME = vDME - halo_dm
        vLOGF = np.log10(vF / vW)
        vLOGW = np.log10(vW)
    else:
        frb_cat = lf.LoadCatalogue(fcat)
        if fgt and fgt.find('NE2001') >= 0:
            vDME = frb_cat['DM'] - frb_cat['DM_NE2001']
        else:
            vDME = frb_cat['DM'] - frb_cat['DM_YMW16']
        if bolhalo:
            vDME = vDME - halo_dm
        vLOGF = np.log10(frb_cat['S'])
        vLOGW = np.log10(frb_cat['W'])
        vSVY = frb_cat['SURVEY']

    if fgt and fgt.find('ETG') >= 0:
        fgt = 'ETG'

    svy_info = lf.LoadSvyInfo(fsvy)

    cts = Counter(vSVY)

    survey_names = svy_info['SURVEY']
    vN = np.array([cts.get(s, 0) for s in survey_names])
    vSN0 = svy_info['SN0']
    vTs = svy_info['Tsys']
    vG = svy_info['Gain']
    vBW = svy_info['BW']
    vNpol = svy_info['Npol']
    vFOV = svy_info['FOV']
    vTime = svy_info['TIME']

    vLOGF_2d = [vLOGF[vSVY == s] for s in survey_names]
    vDME_2d = [vDME[vSVY == s] for s in survey_names]
    vLOGW_2d = [vLOGW[vSVY == s] for s in survey_names]

    cfg = config['prior']
    vpara = np.array([cfg['log_phis'][0], cfg['alpha'][0], cfg['log_Es'][0],
                      cfg['log_E0'][0], cfg['mu_w'][0], cfg['sigma_w'][0]])
    vparb = np.array([cfg['log_phis'][1], cfg['alpha'][1], cfg['log_Es'][1],
                      cfg['log_E0'][1], cfg['mu_w'][1], cfg['sigma_w'][1]])

    vpar_range = np.dstack((vpara.transpose(), vparb.transpose()))[0, :, :]

    print('------------par range-----------')
    print(vpar_range)
    a1 = time.perf_counter()
    print(myloglike(vpara, len(vpara), len(vpara)))
    a2 = time.perf_counter()
    print(a1, a2)
    print("Running Nest Sampling ...")

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
