#!/usr/bin/env python3
"""链自动验收（B4, 2026-10-30）：碎后验/贴边链/真值越界不再靠人工翻日志。

只依赖 numpy + frb_util（不 import pymultinest，本地无 MultiNest 也能跑）。
用法:
    python3 qa_chains.py --nest ./nest_out_mini/samp/<tag>/NE2001 [--config config_mini.json]
    python3 qa_chains.py --nest ./nest_out_mini/simu/<tag>/simdat_1e3 \
        --truth ./nest_out_mini/simu/simdat_1e3.truth.json

检查项与判据:
  等权样本 N            报告；<1000 提示 WARN（823/525 那类碎后验的第一信号）
  每参数贴边比例        先验边界 1% 缓冲内计贴边；>5% WARN、>20% FAIL；
                        upper_only 参数(log_E0)按设计可堆边界 → 封顶 WARN
  lnZ ± err             stats.dat 的 Global Log-Evidence 行
  末段 ΔlnZ             ev.dat 第 9 列(index 8)运行中 lnZ 的末段差；缺文件跳过
  接受率 / llfail / 边缘警告   run.log 解析；缺日志跳过这三项不报错
                        （llfail 优先取 manifest 的 n_llfail，更可靠）
  ρ(γ, log_Echar) + 最强三对 |ρ|   简并诊断（D 项"报告联合约束"的代码层落地）
  --truth               真值落在 99% 后验区间外的参数数 >1 即 FAIL

输出: 人读表格 + <basename>qa.json；退出码 0=PASS/WARN、1=FAIL。
CC_QA_STRICT=0 全局降级为只报不拦（放行口径写在 run_all 的失败提示里）。
"""

import argparse
import json
import os
import re
import sys

import numpy as np

from frb_util import (PARAM_ORDER, PARAM_UPPER_ONLY, load_config,
                      manifest_path, read_manifest, truth_vector)

EDGE_BUF = 0.01          # 先验宽度的 1% 以内算贴边
WARN_FRAC, FAIL_FRAC = 0.05, 0.20
MIN_POST_WARN = 1000


def parse_stats(path):
    """stats.dat 首行: 'Nested Sampling Global Log-Evidence : <val> +/- <err>'。"""
    if not os.path.exists(path):
        return None, None
    with open(path, encoding='utf-8', errors='replace') as f:
        head = f.read(2000)
    m = re.search(r'Global Log-Evidence\s*:\s*([-+0-9.eE]+)\s*\+/-\s*([-+0-9.eE]+)', head)
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def parse_ev_tail_dlogl(path, frac=0.05):
    """ev.dat 第 9 列(index 8) 末段差。

    注意该列是 dead point 的 logL 轨迹（单调不减、不收敛到 stats.dat 的 lnZ，
    实测末值 -15169.0 vs lnZ -15195.9），只作采样是否还在吃新似然的诊断，
    【不能】当 lnZ 收敛判据（P2, 2026-10-31）。真 lnZ 轨迹见 parse_runlog。
    """
    if not os.path.exists(path):
        return None
    try:
        d = np.loadtxt(path)
    except (ValueError, OSError):
        return None
    if d.ndim != 2 or d.shape[1] < 9:
        return None
    z = d[:, 8]
    k = max(1, int(len(z) * frac))
    return float(z[-1] - z[-1 - k])


def parse_runlog(path):
    """接受率末值 / [llfail] 计数 / MultiNest 边缘警告 / 真 lnZ 末段差。

    lnZ 轨迹只存在于 run.log 的 'Nested Sampling ln(Z):' 更新行（ev.dat 第 9 列
    不是 lnZ），取末两值之差作末段收敛诊断；缺日志全 None。
    """
    if not os.path.exists(path):
        return None, None, None, None
    acc, nll, nedge = None, 0, 0
    lnz_series = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            m = re.search(r'Acceptance Rate:\s*([0-9.]+)', line)
            if m:
                acc = float(m.group(1))
            m = re.search(r'Nested Sampling ln\(Z\):\s*([-+0-9.eE]+)', line)
            if m:
                lnz_series.append(float(m.group(1)))
            if line.startswith('[llfail] #'):
                nll += 1
            if 'converging towards the edge of the prior' in line:
                nedge += 1
    dlnz = float(lnz_series[-1] - lnz_series[-2]) if len(lnz_series) >= 2 else None
    return acc, nll, nedge, dlnz


def edge_fractions(chain, prior):
    """每参数贴边质量比例；col0 是线性 phis，先验是 log10 ⇒ 换轴再比。"""
    out = {}
    for i, key in enumerate(PARAM_ORDER):
        x = chain[:, i]
        if key == 'log_phi0':
            x = np.log10(np.where(x > 0, x, np.nan))
        lo, hi = prior[key]
        buf = EDGE_BUF * (hi - lo)
        out[key] = float(np.mean((x <= lo + buf) | (x >= hi - buf)))
    return out


def rho_report(chain):
    """8 维 Pearson 相关（col0 取 log10）；返回 (rho(gamma,logEchar) 或 None, 最强三对)。

    常数列（后验完全退化）会给 nan，JSON 里落成 NaN 会毒化下游解析 ⇒ 归 None（P3）。
    """
    x = chain.copy()
    x[:, 0] = np.log10(np.where(x[:, 0] > 0, x[:, 0], np.nan))
    m = np.corrcoef(x.T)
    pairs = []
    for i in range(len(PARAM_ORDER)):
        for j in range(i + 1, len(PARAM_ORDER)):
            if np.isfinite(m[i, j]):
                pairs.append((abs(m[i, j]), PARAM_ORDER[i], PARAM_ORDER[j], float(m[i, j])))
    pairs.sort(reverse=True)
    top3 = [(a, b, round(r, 3)) for _, a, b, r in pairs[:3]]
    rg = float(m[1, 2]) if np.isfinite(m[1, 2]) else None
    return rg, top3


def main():
    ap = argparse.ArgumentParser(description='MultiNest 链自动验收')
    ap.add_argument('--nest', required=True, help='MultiNest 输出前缀(basename)')
    ap.add_argument('--config', default='config_mini.json')
    ap.add_argument('--run-log', default=None, help='运行日志路径(缺省 <basename>.run.log)')
    ap.add_argument('--truth', default=None, help='注入真值 JSON(模拟线才给)')
    args = ap.parse_args()

    cfg = load_config(args.config)
    post_f = str(args.nest) + 'post_equal_weights.dat'
    if not os.path.exists(post_f):
        raise SystemExit('[qa] 缺等权后验文件: %s' % post_f)
    chain = np.loadtxt(post_f)
    if chain.ndim == 1:
        chain = chain[None, :]
    par = chain[:, :len(PARAM_ORDER)]

    # 贴边必须按【链跑的时候】的先验判：manifest 存了完整 config，优先用它；
    # 与当前 config 的先验不一致时明说（否则改了边界后旧链的贴边堆会隐身）。
    man = read_manifest(manifest_path(args.nest)) if os.path.exists(
        manifest_path(args.nest)) else None
    prior = cfg['prior']
    if man and isinstance(man.get('config'), dict) and 'prior' in man['config']:
        if man['config']['prior'] != prior:
            print('[qa] !! 链的先验与当前 config 不同，贴边按链侧先验判')
        prior = man['config']['prior']
    else:
        print('[qa] 无 manifest，按当前 config 判贴边（链的出处/先验未经校验）')

    strict = os.environ.get('CC_QA_STRICT', '1') != '0'
    warns, fails = [], []

    def level(kind, key, msg):
        """kind: 'warn'/'fail'；CC_QA_STRICT=0 时 fail 降级为 warn。"""
        if kind == 'fail' and not strict:
            kind = 'warn'
            msg += '  [CC_QA_STRICT=0 降级]'
        (warns if kind == 'warn' else fails).append('%s: %s' % (key, msg))
        return kind

    res = {'nest': str(args.nest), 'n_post': int(len(par))}
    print('[qa] %s  等权样本 N=%d' % (args.nest, len(par)))
    if len(par) < MIN_POST_WARN:
        level('warn', 'N', '等权样本 %d < %d（后验碎裂/采样不足的第一信号）'
              % (len(par), MIN_POST_WARN))

    # --- 贴边 ---
    ef = edge_fractions(par, prior)
    res['edge_frac'] = {k: round(v, 4) for k, v in ef.items()}
    for key, v in ef.items():
        cap = 'warn' if key in PARAM_UPPER_ONLY else 'fail'
        if v > FAIL_FRAC:
            level(cap, 'edge:' + key, '贴边质量 %.1f%% > 20%%%s'
                  % (100 * v, '（upper_only，按设计封顶 WARN）' if cap == 'warn' else ''))
        elif v > WARN_FRAC:
            level('warn', 'edge:' + key, '贴边质量 %.1f%% > 5%%' % (100 * v))
        print('    贴边 %-14s %5.1f%%%s' % (key, 100 * v,
                                            '  <- upper_only' if key in PARAM_UPPER_ONLY else ''))

    # --- lnZ / 末段 ΔlnZ / 接受率 / llfail ---
    lnz, lnzerr = parse_stats(str(args.nest) + 'stats.dat')
    res['lnZ'], res['lnZ_err'] = lnz, lnzerr
    print('    lnZ = %s +/- %s' % ('%.3f' % lnz if lnz is not None else 'n/a',
                                   '%.3f' % lnzerr if lnzerr is not None else 'n/a'))
    dlogl = parse_ev_tail_dlogl(str(args.nest) + 'ev.dat')
    res['tail_dlogL'] = None if dlogl is None else round(dlogl, 4)
    logfile = args.run_log or (str(args.nest) + '.run.log')
    acc, nll_log, nedge, dlnz = parse_runlog(logfile)
    res['tail_dlnZ'] = None if dlnz is None else round(dlnz, 4)
    if dlnz is not None:
        print('    末段 ΔlnZ = %+.4f  (run.log 的 Nested Sampling ln(Z) 末两值)' % dlnz)
    if dlogl is not None:
        print('    末段 ΔlogL = %+.4f  (ev.dat 第9列 dead-point 轨迹, 仅参考, 非 lnZ)' % dlogl)
    nll = int(man['n_llfail']) if man and 'n_llfail' in man else nll_log
    res.update(acceptance=acc, n_llfail=nll, mn_edge_warnings=nedge,
               manifest=bool(man), run_tag=(man or {}).get('run_tag', ''))
    if acc is not None:
        print('    接受率 = %.4f' % acc)
    if nll:
        level('warn', 'llfail', '似然失败 %d 次（见 [llfail] 行/manifest）' % nll)
    if nedge:
        level('warn', 'mn-edge', 'MultiNest 自身报先验边缘收敛 %d 次' % nedge)

    # --- 简并诊断 ---
    rg, top3 = rho_report(par)
    res['rho_gamma_logEchar'] = None if rg is None else round(rg, 3)
    res['rho_top3'] = top3
    print('    rho(gamma, log_Echar) = %s   最强三对: %s'
          % ('n/a' if rg is None else '%+.2f' % rg,
             ', '.join('%s-%s %+.2f' % t for t in top3) if top3 else 'n/a'))

    # --- 真值越界（模拟线） ---
    if args.truth:
        tv = truth_vector(args.truth)
        x = par.copy()
        x[:, 0] = np.log10(np.where(x[:, 0] > 0, x[:, 0], np.nan))
        out = []
        for i, key in enumerate(PARAM_ORDER):
            lo, hi = np.percentile(x[:, i], [0.5, 99.5])
            if not (lo <= tv[i] <= hi):
                out.append('%s(真值 %.3f 在 [%.3f, %.3f] 外)' % (key, tv[i], lo, hi))
        res['truth_outside_99'] = out
        print('    真值越界(99%%): %s' % ('; '.join(out) if out else '无'))
        if len(out) > 1:
            level('fail', 'truth', '%d 个参数真值在 99%% 后验区间外: %s'
                  % (len(out), '; '.join(out)))

    res.update(warns=warns, fails=fails,
               verdict='FAIL' if fails else ('WARN' if warns else 'PASS'))
    with open(str(args.nest) + 'qa.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write('\n')
    for w in warns:
        print('  WARN  ' + w)
    for fl in fails:
        print('  FAIL  ' + fl)
    print('[qa] verdict = %s  (%sqa.json)' % (res['verdict'], os.path.basename(args.nest)))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
