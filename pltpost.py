#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import json
import argparse
import pymultinest
from frb_util import load_config

plt.style.use("classic")
#mpl.rcParams['font.family'] ='Times New Roman'
mpl.rcParams['font.size']=24
mpl.rcParams['xtick.labelsize']=16
mpl.rcParams['ytick.labelsize']=16
mpl.rcParams['axes.labelsize']=22

def plot2dposterior_withconf(dat, likli, indx=[], labels=[], rate=0.5,
        par=np.array([]), truth_par=None,
                    parm=np.array([]), rangedat=np.array([]), levels=[0.68],
                    bolupp=False):
    '''Plot a 2D posterier with likelihood burning curve
    dat: a n x m numpy matrix, each data point is a row
    indx: a array of integer indicating wich column (parameter) to plot
    labels: the LaTex label of the given parameter
    rate: the percentage of data to plot, e.g. 0.3 means plot data from 70% to
    the end.
    par: 后验 MAP(黑色实线)
    truth_par: 注入真值数组(红色虚线),与 par 同长度,已在调用前完成 log10(phis) 变换
    bolupp: 是否使用 uniform prior 风格的标记（原为模块全局变量，现已参数化）
    '''
    row, col = dat.shape;
    if len(indx) == 0:
        indx = np.arange(0, col);

    if len(labels) == 0:
        labels = ['a'];
        labels = labels * (col);
    dat=dat.copy()
    dat = dat[np.ix_(np.arange(int(row - row * rate), row), indx)];
    likli = likli[np.ix_(np.arange(int(row - row * rate), row))];
    row, col = dat.shape;
    #max_yticks = 3
    max_yticks = 4
    if len(rangedat) == 0:
        rangedat = np.zeros((col, 2));
        for i in range(0, col):
            rangedat[i, 0] = np.min(dat[:, i]);
            rangedat[i, 1] = np.max(dat[:, i]);

    npar = col;

    #index of y
    for vari in range(0, npar):
        # plot the histogram
        #ax = plt.subplot2grid((npar, npar), (vari, vari))

        ax=plt.subplot(npar, npar, vari*npar+vari+1)
        ind = ((dat[:, vari] < rangedat[vari, 1]) & (dat[:, vari] > rangedat[vari, 0]))
        n, bins, patches = plt.hist(dat[ind, vari], 100, density=True, \
                                    histtype='stepfilled', range=(rangedat[vari, 0], rangedat[vari, 1]))
        plt.setp(patches, 'facecolor', 'lightblue', 'alpha', 0.6)
        yloc = plt.MaxNLocator(max_yticks)
        ax.xaxis.set_major_locator(yloc)
        xloc = plt.MaxNLocator(max_yticks)
        ax.yaxis.set_major_locator(xloc)
        plt.xlabel(labels[vari])
        if bolupp:
            if len(levels)>0:
                for ls in [0.68, 0.95]:
                    # 复用主直方图的 n/bins,避免重复 histogram 调用导致口径不一致
                    histc = n
                    bin_edges = bins
                    vom=(bin_edges[:-1]+bin_edges[1:])*0.5
                    ind=np.argsort(histc)
                    ind=ind[::-1]
                    v=histc.copy()
                    v[0]=histc[ind[0]]
                    for i in range(1, len(ind)):
                        v[i] = v[i-1]+histc[ind[i]]

                    v=v/float(np.sum(histc))
                    where_idx = np.where((v[:-1] < ls) & (v[1:]>=ls))[0]
                    if len(where_idx) == 0:
                        # 后验高度集中在少数 bin，累积和首轮即超过 ls，跳过此置信水平
                        continue
                    thre= histc[ind[where_idx][0]]

                    lv=np.min(vom[histc>=thre])
                    rv=np.max(vom[histc>=thre])
                    print(vari, '-th parameter', 'sigma=', ls, 'lv=', lv, 'rv=', rv, thre)

                    if vari != 3:
                        if ls == 0.68:
                            plt.plot([lv, lv], [0, max(n)], ls='dashed', color='k', linewidth=1)
                            plt.plot([rv, rv], [0, max(n)], ls='dashed', color='k', linewidth=1)
                        else:
                            plt.plot([lv, lv], [0, max(n)], ls='dotted', color='k', linewidth=1)
                            plt.plot([rv, rv], [0, max(n)], ls='dotted', color='k', linewidth=1)
                        print(rv-par[vari], lv-par[vari])
                    elif ls == 0.95:
                        print(rv)
                        plt.plot([rv, rv], [0, max(n)], ls='solid', color='k', linewidth=2)

            if par.size > 0 and vari != 3:
                plt.plot([par[vari], par[vari]], [0, max(n)], ls='solid',
                         color='k', linewidth=2, label='MAP')
        else:
            if len(levels)>0:
                for ls in [0.68, 0.95]:
                    # 复用主直方图的 n/bins,避免重复 histogram 调用导致口径不一致
                    histc = n
                    bin_edges = bins
                    vom=(bin_edges[:-1]+bin_edges[1:])*0.5
                    ind=np.argsort(histc)
                    ind=ind[::-1]
                    v=histc.copy()
                    v[0]=histc[ind[0]]
                    for i in range(1, len(ind)):
                        v[i] = v[i-1]+histc[ind[i]]

                    v=v/float(np.sum(histc))
                    where_idx = np.where((v[:-1] < ls) & (v[1:]>=ls))[0]
                    if len(where_idx) == 0:
                        continue
                    thre= histc[ind[where_idx][0]]

                    lv=np.min(vom[histc>=thre])
                    rv=np.max(vom[histc>=thre])
                    print(vari, '-th parameter', 'sigma=', ls, 'lv=', lv, 'rv=', rv, thre)
                    if ls == 0.68:
                        plt.plot([lv, lv], [0, max(n)], ls='dashed', color='k', linewidth=1)
                        plt.plot([rv, rv], [0, max(n)], ls='dashed', color='k', linewidth=1)
                    else:
                        plt.plot([lv, lv], [0, max(n)], ls='dotted', color='k', linewidth=1)
                        plt.plot([rv, rv], [0, max(n)], ls='dotted', color='k', linewidth=1)
                    print(rv-par[vari], lv-par[vari])
            if par.size > 0:
                plt.plot([par[vari], par[vari]], [0, max(n)], ls='solid',
                         color='k', linewidth=2, label='MAP')

        if truth_par is not None:
            plt.plot([truth_par[vari], truth_par[vari]], [0, max(n)], ls='dashed',
                     color='red', linewidth=2)

        if len(parm) > 0:
            for i in range(len(parm)):
                parm0=parm[i]
                plt.plot([parm0[vari], parm0[vari]], [0, max(n)], ls='dashed',
                     color='k', linewidth=2)
        
        plt.xlim(rangedat[vari, :])
        #index of x
        for varj in range(vari + 1, npar):

            #ax = plt.subplot2grid((npar, npar), (vari, varj))
            ax=plt.subplot(npar, npar, (vari)*npar+varj+1)
            x = dat[:, varj]
            y = dat[:, vari]

            ind = (
            (x < rangedat[varj, 1]) & (x > rangedat[varj, 0]) & (y < rangedat[vari, 1]) & (y > rangedat[vari, 0]))
            liklisub = likli[ind]
            x = dat[ind, varj]
            y = dat[ind, vari]
            ngridx = 60
            ngridy = 60
            #generate 2D histogram
            H, xedges, yedges = np.histogram2d(x, y, bins=(ngridx, ngridy),
                                               range=(rangedat[varj, :], rangedat[vari, :]))
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            #hisogram is row leading need transpose to plot with contourf
            H = H.transpose()
            #get the center of bin
            xedges = (xedges[:-1] + xedges[1:]) / 2
            yedges = (yedges[:-1] + yedges[1:]) / 2
            mxx, mxy = np.meshgrid(xedges, yedges)
            plt.contourf(mxx, mxy, H, 100, cmap='Blues');
            plt.title(f"{labels[vari]}-{labels[varj]}")
            #plt.title("%s - %s" % labels[vari] % labels[varj])
            #Prepare to count the confidence level
            indx, indy = np.meshgrid(np.arange(0, ngridx), np.arange(0, ngridy))
            vmx2 = np.squeeze(np.reshape(mxx, (-1, 1)));
            vmy2 = np.squeeze(np.reshape(mxy, (-1, 1)));
            vm = np.squeeze(np.reshape(H, (-1, 1)));

            vx = np.squeeze(np.reshape(indx, (-1, 1)));
            vy = np.squeeze(np.reshape(indy, (-1, 1)));
            #Sort according to likelihood
            vm2 = np.sort(vm)[::-1];
            #Get index and convert everything to array
            ix = np.argsort(vm, axis=0)[::-1];
            ix = np.ix_(ix);
            vx2 = vx[ix];
            vy2 = vy[ix];
            vmx2 = vmx2[ix];
            vmy2 = vmy2[ix];
            #get the cumulative of the hisogram
            vm2 = np.cumsum(vm2 / np.sum(vm2));
            cmxx2 = H;
            mxx2 = mxx;
            mxy2 = mxy;
            #Form 2D cumulative plot
            for ki in range(0, len(vm2)):
                mxx2[vy2[ki], vx2[ki]] = vmx2[ki];
                mxy2[vy2[ki], vx2[ki]] = vmy2[ki];
                cmxx2[vy2[ki], vx2[ki]] = vm2[ki];
            conls = plt.contour(mxx2, mxy2, cmxx2, levels, colors='k');
            plt.clabel(conls, inline=1, fontsize=10)
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)
            yloc = plt.MaxNLocator(max_yticks)
            ax.xaxis.set_major_locator(yloc)
            xloc = plt.MaxNLocator(max_yticks)
            ax.yaxis.set_major_locator(xloc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plotting posteriors of the Bayesian inference')
    parser.add_argument('-f', action='store', dest='fname', type=str, help='Input catalog file')
    parser.add_argument('-o', action='store', dest='fout', type=str, help='Save the output file')
    parser.add_argument('-title', action='store', dest='title', type=str, help='Name the title of this plot')
    parser.add_argument('-up', action='store_true', dest='bolupp', help='Bool option: Choose uniform prior or not')
    parser.add_argument('-bo', action='store_true', dest='bolout', help='Bool option: Save plot or not')
    parser.add_argument('-c', action='store', dest='config_path', type=str, default='config_mini.json',
                        help='Config file path (default: config_mini.json)')
    parser.add_argument('-truth', action='store', dest='truth_path', type=str, default=None,
                        help='注入真值 JSON 文件路径(原始物理单位): '
                             'phis(线性), alpha, logEs, log_E0, mu_w, sigma_w。'
                             '内部会对 phis 做 log10 变换以与后验链对齐。')
    args = parser.parse_args()
    fname = args.fname
    fout = args.fout
    tit = args.title
    bolupp = args.bolupp
    bolout = args.bolout

    # 从 config 读取先验范围作为默认绘图范围（路径解析与 frb_util.load_config 一致）
    cfg = load_config(args.config_path)
    pri = cfg['prior']
    rdat = np.array([
        pri['log_phis'],   # log10(phis) 绘图范围
        pri['alpha'],
        pri['log_Es'],
        pri['log_E0'],
        pri['mu_w'],
        pri['sigma_w']
    ])

    # 解析注入真值文件(若有),phis 内部做 log10 变换以与 mxchain[:,0] 对齐
    truth_par = None
    if args.truth_path:
        with open(args.truth_path) as tf:
            truth_dict = json.load(tf)
        # 接受多种 key 命名以兼容不同写法
        phis_val = truth_dict.get('phis', truth_dict.get('phi_star', truth_dict.get('phi*')))
        truth_par = np.array([
            np.log10(float(phis_val)),                  # log10(phis),与 mxchain[:,0] 一致
            float(truth_dict.get('alpha')),
            float(truth_dict.get('logEs', truth_dict.get('log_Es'))),
            float(truth_dict.get('logE0', truth_dict.get('log_E0'))),
            float(truth_dict.get('mu_w', truth_dict.get('mu'))),
            float(truth_dict.get('sigma_w', truth_dict.get('sigma')))
        ])
        print(f'[truth] 注入真值(已 log10 phis): {truth_par}')

    a = pymultinest.Analyzer(n_params = 6, outputfiles_basename=fname)
    b = a.get_equal_weighted_posterior()

    vlik=b[:,-1]
    allres=b[:,:-1]
    mxchain = allres
    mxchain[:,0] = np.log10(mxchain[:,0])
    vpar = mxchain[np.argmax(vlik),:]
    print(vpar)

    if bolout:
        plt.figure(figsize=(20, 16))
        plot2dposterior_withconf(mxchain, vlik, par=vpar, truth_par=truth_par, indx=[], rangedat=rdat, rate=1.0, levels=[0.68, 0.95],
            labels=[r'$\log\,\phi^*$', r'$\alpha$', r'$\log E^*$', r'$\log E_0$', r'$\mu_w$', r'$\sigma_w$'],
            bolupp=bolupp)
        plt.suptitle(tit)
        plt.savefig(fout)

    else:
        plot2dposterior_withconf(mxchain, vlik, par=vpar, truth_par=truth_par, indx=[], rangedat=rdat, rate=1.0, levels=[0.68, 0.95],
            labels=[r'$\log\,\phi^*$', r'$\alpha$', r'$\log E^*$', r'$\log E_0$', r'$\mu_w$', r'$\sigma_w$'],
            bolupp=bolupp)
        plt.suptitle(tit)
        plt.show()
