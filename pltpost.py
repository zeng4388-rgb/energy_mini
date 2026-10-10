#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import pymultinest
from scipy.ndimage import gaussian_filter
from frb_util import (load_config, truth_vector, PARAM_ORDER, PARAM_UPPER_ONLY)

plt.style.use("classic")
#mpl.rcParams['font.family'] ='Times New Roman'
mpl.rcParams['font.size']=24
mpl.rcParams['xtick.labelsize']=16
mpl.rcParams['ytick.labelsize']=16
mpl.rcParams['axes.labelsize']=22

def plot2dposterior_withconf(dat, likli, indx=[], labels=[], rate=0.5,
        par=np.array([]), truth_par=None,
                    parm=np.array([]), rangedat=np.array([]), levels=[0.68],
                    bolupp=False, upper_only=()):
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

    # --- 改进 1: 绘图范围分位数自动缩放 ---
    # 先验区间仍是外框（prior_rangedat 留作 1D 面板的先验边界参考线）；非
    # upper_only 参数改用后验 [0.5, 99.5]% 分位数缩放，直方图 100 个 bins 全部
    # 花在后验核上而不是大片空白先验区。MAP 与注入真值线包进窗口（截断在先验
    # 内）——恢复检验时真值线不会被裁掉，真值严重偏离时窗口随之拉大，本身就是
    # 可见的诊断。upper_only（如 log_E0 上限参数）保留全范围：贴边堆积本身就是
    # 结果，缩放会掩盖它。
    prior_rangedat = np.array(rangedat, dtype=float, copy=True)
    for i in range(0, col):
        if i in upper_only:
            continue
        plo, phi_ = np.percentile(dat[:, i], [0.5, 99.5])
        if not (np.isfinite(plo) and np.isfinite(phi_)):
            continue
        if phi_ - plo <= 0:
            pad = max(1e-6, 1e-6 * abs(plo))
            plo, phi_ = plo - pad, phi_ + pad
        pts = [plo, phi_]
        if par.size > 0:
            pts.append(par[i])
        if truth_par is not None:
            pts.append(truth_par[i])
        lo = max(prior_rangedat[i, 0], min(pts))
        hi = min(prior_rangedat[i, 1], max(pts))
        if hi > lo:
            rangedat[i, 0], rangedat[i, 1] = lo, hi

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

                    if vari not in upper_only:
                        if ls == 0.68:
                            plt.plot([lv, lv], [0, max(n)], ls='dashed', color='k', linewidth=1, label='68% CI')
                            plt.plot([rv, rv], [0, max(n)], ls='dashed', color='k', linewidth=1)
                        else:
                            plt.plot([lv, lv], [0, max(n)], ls='dotted', color='k', linewidth=1)
                            plt.plot([rv, rv], [0, max(n)], ls='dotted', color='k', linewidth=1, label='95% CI')
                        print(rv-par[vari], lv-par[vari])
                    elif ls == 0.95:
                        print(rv)
                        plt.plot([rv, rv], [0, max(n)], ls='solid',
                                 color='k', linewidth=2, label='95% upper')

            if par.size > 0 and vari not in upper_only:
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
                        plt.plot([lv, lv], [0, max(n)], ls='dashed', color='k', linewidth=1, label='68% CI')
                        plt.plot([rv, rv], [0, max(n)], ls='dashed', color='k', linewidth=1)
                    else:
                        plt.plot([lv, lv], [0, max(n)], ls='dotted', color='k', linewidth=1)
                        plt.plot([rv, rv], [0, max(n)], ls='dotted', color='k', linewidth=1, label='95% CI')
                    print(rv-par[vari], lv-par[vari])
            if par.size > 0:
                plt.plot([par[vari], par[vari]], [0, max(n)], ls='solid',
                         color='k', linewidth=2, label='MAP')

        if truth_par is not None:
            plt.plot([truth_par[vari], truth_par[vari]], [0, max(n)], ls='dashed',
                     color='red', linewidth=2, label='injected truth')

        if len(parm) > 0:
            for i in range(len(parm)):
                parm0=parm[i]
                plt.plot([parm0[vari], parm0[vari]], [0, max(n)], ls='dashed',
                     color='k', linewidth=2)

        # 缩放后若先验边界落在窗口内, 用灰点线标出(保留先验边界刻度)
        if prior_rangedat[vari, 0] > rangedat[vari, 0]:
            plt.plot([prior_rangedat[vari, 0]]*2, [0, max(n)], ls=':',
                     color='0.6', lw=0.8)
        if prior_rangedat[vari, 1] < rangedat[vari, 1]:
            plt.plot([prior_rangedat[vari, 1]]*2, [0, max(n)], ls=':',
                     color='0.6', lw=0.8)

        # --- 改进 3: 图例(只放第一个 1D 面板, 避免每个面板重复) ---
        if vari == 0:
            handles, lbls = plt.gca().get_legend_handles_labels()
            if lbls:
                plt.legend(loc='best', fontsize=14, frameon=False)

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
            # 改进 2: 高斯平滑(σ≈1 bin)后再画底色与置信度等高线。60×60 网格直接上
            # 100 级 contourf 散斑大; 平滑压掉采样噪声, 质量排序置信度计算在同一
            # 平滑场上进行, 底色与黑线自洽。
            H = gaussian_filter(H, 1.0)
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

    # 返回实际使用的绘图范围(缩放后), 供调用方/回归测试检查
    return rangedat

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
                             'phis(线性), gamma, log_Echar, log_E0, mu_w, sigma_w, ln_mu_host, ln_sigma_host。'
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
    n_show = len(PARAM_ORDER)
    keys = PARAM_ORDER[:n_show]
    rdat = np.array([pri[k] for k in keys])
    upper_only = tuple(i for i, k in enumerate(keys) if k in PARAM_UPPER_ONLY)

    # 解析注入真值文件(若有)。映射与 pltpz 共用 frb_util.truth_vector（单一实现）
    truth_par = None
    if args.truth_path:
        truth_par = truth_vector(args.truth_path)
        print(f'[truth] 注入真值(已 log10 phis): {truth_par}')

    a = pymultinest.Analyzer(n_params = n_show, outputfiles_basename=fname)
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
            labels=[r'$\log\,\phi_0$', r'$\gamma$', r'$\log E_{\rm char}$', r'$\log E_0$', r'$\mu_w$', r'$\sigma_w$', r'$\ln\mu_{\rm host}$', r'$\ln\sigma_{\rm host}$'],
            bolupp=bolupp, upper_only=upper_only)
        plt.suptitle(tit)
        plt.savefig(fout)

    else:
        plot2dposterior_withconf(mxchain, vlik, par=vpar, truth_par=truth_par, indx=[], rangedat=rdat, rate=1.0, levels=[0.68, 0.95],
            labels=[r'$\log\,\phi_0$', r'$\gamma$', r'$\log E_{\rm char}$', r'$\log E_0$', r'$\mu_w$', r'$\sigma_w$', r'$\ln\mu_{\rm host}$', r'$\ln\sigma_{\rm host}$'],
            bolupp=bolupp, upper_only=upper_only)
        plt.suptitle(tit)
        plt.show()
