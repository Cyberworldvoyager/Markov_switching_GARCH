"""
在原始数据和新数据上同时运行: GARCH / 2-state(改进) / 3-state GARCH-DF
"""
import numpy as np, pandas as pd, time, warnings, math, os
from scipy.optimize import minimize
from scipy.special import gammaln
from numba import njit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
import sys
sys.stdout.reconfigure(line_buffering=True)

# Paths relative to project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
FIGS = os.path.join(ROOT, 'figures')
RESULTS = os.path.join(ROOT, 'results')

# =====================================================================
# Numba Kim filter
# =====================================================================
@njit
def _kim_core(y, mu_vec, gv, alpha, beta, nu_vec, P, rc, rl, n_exp, T):
    n_states = len(mu_vec)
    nu_sub = nu_vec - 2.0
    e = np.zeros((T, n_states))
    for t in range(T):
        for k in range(n_states):
            e[t, k] = y[t] - mu_vec[k]
    m_y = 0.0
    for t in range(T): m_y += y[t]
    m_y /= T
    var_y = 0.0
    for t in range(T): var_y += (y[t] - m_y) ** 2
    var_y /= T
    h = np.full((T, n_exp), var_y)
    uus = np.zeros((T, n_states))
    for k in range(n_states):
        uus[0, k] = var_y * nu_sub[k] / nu_vec[k]
    pi_stat = np.ones(n_states) / n_states
    pt = np.zeros((T, n_exp))
    for i in range(n_exp): pt[0, i] = pi_stat[rc[i]] * pi_stat[rl[i]]
    s = 0.0
    for i in range(n_exp): s += pt[0, i]
    for i in range(n_exp): pt[0, i] /= s
    lgamma_r = np.zeros(n_states)
    log_pi_t = np.zeros(n_states)
    for k in range(n_states):
        nu = nu_vec[k]
        lgamma_r[k] = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
        log_pi_t[k] = 0.5 * np.log(nu * np.pi)
    gv_arr = np.full(n_exp, gv)
    ll = 0.0
    for t in range(1, T):
        for i in range(n_exp):
            h[t, i] = gv_arr[i] + alpha * uus[t - 1, rl[i]] + beta * h[t - 1, i]
            if h[t, i] < 1e-30: h[t, i] = 1e-30
        fvec = np.zeros(n_exp)
        for i in range(n_exp):
            k = rc[i]
            nu_c = nu_vec[k]
            sc2 = h[t, i] * (nu_c - 2.0) / nu_c
            if sc2 < 1e-30: sc2 = 1e-30
            x = e[t, k]
            logf = lgamma_r[k] - log_pi_t[k] - 0.5 * np.log(sc2) \
                   - ((nu_c + 1.0) / 2.0) * np.log(1.0 + x * x / (sc2 * nu_c))
            if logf < -700: val = 1e-300
            elif logf > 700: val = 1e300
            else: val = np.exp(logf)
            if val < 1e-300: val = 1e-300
            fvec[i] = val
        pp = np.zeros(n_exp)
        for i in range(n_exp):
            for j in range(n_exp): pp[j] += pt[t - 1, i] * P[i, j]
        joint = np.zeros(n_exp)
        fpt = 0.0
        for i in range(n_exp):
            joint[i] = pp[i] * fvec[i]
            fpt += joint[i]
        if fpt < 1e-300: fpt = 1e-300
        for i in range(n_exp): pt[t, i] = joint[i] / fpt
        ll += np.log(fpt)
        ph_sum = np.zeros(n_states)
        pt_sum = np.zeros(n_states)
        for i in range(n_exp):
            k = rc[i]
            ph_sum[k] += pt[t, i] * h[t, i]
            pt_sum[k] += pt[t, i]
        for k in range(n_states):
            hk = ph_sum[k] / pt_sum[k] if pt_sum[k] > 1e-300 else gv
            uus[t, k] = e[t, k] ** 2 * nu_sub[k] / nu_vec[k]
            for i in range(n_exp):
                if rc[i] == k: h[t, i] = hk
    return ll, pt, e, h


def kim_filter(y, mu_vec, gv, alpha, beta, nu_vec, trans_mat, n_states):
    mu_vec = np.asarray(mu_vec, dtype=np.float64)
    nu_vec = np.asarray(nu_vec, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    T = len(y)
    n_exp = n_states * n_states
    rc = np.repeat(np.arange(n_states), n_states).astype(np.int64)
    rl = np.tile(np.arange(n_states), n_states).astype(np.int64)
    P = np.zeros((n_exp, n_exp))
    for i in range(n_exp):
        for j in range(n_exp):
            if rl[j] == rc[i]:
                P[i, j] = trans_mat[rc[i], rc[j]]
    return _kim_core(y, mu_vec, float(gv), float(alpha), float(beta),
                     nu_vec, P, rc, rl, n_exp, T)


def smooth_nv(pt, trans_mat, n_states):
    T = len(pt)
    n_exp = n_states * n_states
    rc = np.repeat(np.arange(n_states), n_states)
    rl = np.tile(np.arange(n_states), n_states)
    P = np.zeros((n_exp, n_exp))
    for i in range(n_exp):
        for j in range(n_exp):
            if rl[j] == rc[i]:
                P[i, j] = trans_mat[rc[i], rc[j]]
    psm = np.zeros((T, n_states))
    for k in range(n_states):
        psm[T - 1, k] = pt[T - 1, rc == k].sum()
    ps = pt[T - 1].copy()
    for t in range(T - 2, 0, -1):
        pp = np.maximum(pt[t] @ P, 1e-300)
        psn = pt[t] * (P @ (ps / pp))
        s = psn.sum()
        ps = psn / s if s > 0 else psn
        for k in range(n_states):
            psm[t, k] = ps[rc == k].sum()
    return psm


# =====================================================================
# Helpers
# =====================================================================
def vix_metrics(dates, vol_ann, vix, label):
    vol22 = pd.Series(vol_ann).rolling(22, min_periods=10).mean()
    vm = pd.Series(vol22.values, index=dates)
    cm = vm.index.intersection(vix.index)
    if len(cm) < 20: return None
    vx = vix.loc[cm, 'VIX']; vmc = vm.loc[cm]
    vx_n = vx / vx.mean(); vm_n = vmc / vmc.mean()
    mask = vm_n.notna() & vx_n.notna()
    corr = np.corrcoef(vm_n[mask], vx_n[mask])[0, 1]
    mse = np.mean((vm_n[mask] - vx_n[mask]) ** 2)
    mae = np.mean(np.abs(vm_n[mask] - vx_n[mask]))
    return {'label': label, 'n': int(mask.sum()),
            'corr': corr, 'mse': mse, 'mae': mae,
            'vx': vx, 'vm': vmc, 'vx_n': vx_n, 'vm_n': vm_n, 'cm': cm}


def fit_garch(y):
    def nll(p):
        mu, w, a, b, nu = p
        if w <= 0 or a < 0 or b < 0 or a + b >= 0.999 or nu < 2.5: return 1e10
        ee = y - mu; h = np.var(y); ll = 0.0
        for t in range(1, len(y)):
            h = w + a * ee[t - 1] ** 2 + b * h
            if h <= 0: return 1e10
            sc2 = max(h * (nu - 2) / nu, 1e-30)
            ll += gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) \
                  - 0.5 * np.log(sc2) - ((nu + 1) / 2) * np.log(1 + ee[t] ** 2 / (sc2 * nu))
        return -ll
    x0 = [np.mean(y), np.var(y) * 0.05, 0.04, 0.93, 8.0]
    res = minimize(nll, x0, method='Nelder-Mead',
                   options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-6})
    mu, w, a, b, nu = res.x
    ee = y - mu; h = np.zeros(len(y)); h[0] = np.var(y)
    for t in range(1, len(y)): h[t] = w + a * ee[t - 1] ** 2 + b * h[t - 1]
    return {'mu': mu, 'omega': w, 'alpha': a, 'beta': b, 'nu': nu, 'll': -res.fun, 'h': h}


def fit_ms(y, n_states, n_rand=6, seed=42):
    rng = np.random.RandomState(seed)
    n_exp = n_states * n_states

    if n_states == 2:
        def nll(p):
            try:
                mu0, mu1, gv, a, b, nu0, nu1, pr, qr = p
                if gv <= 0 or a < 0 or b < 0 or a + b >= 1.5 or nu0 < 2.1 or nu1 < 2.1: return 1e10
                if pr < 0.01 or pr > 0.99 or qr < 0.01 or qr > 0.99: return 1e10
                tm = np.array([[pr, 1 - pr], [1 - qr, qr]])
                ll, _, _, _ = kim_filter(y, [mu0, mu1], gv, a, b, [nu0, nu1], tm, 2)
                return -ll if np.isfinite(ll) else 1e10
            except: return 1e10
        bd = [(-3, 3), (-3, 3), (1e-8, 2), (1e-8, 0.5), (0.3, 0.999),
              (2.1, 50), (2.1, 50), (0.05, 0.95), (0.05, 0.95)]
        fixed = [
            [0.02, 0.08, 0.01, 0.04, 0.93, 5.0, 10.0, 0.15, 0.65],
            [0.05, 0.05, 0.02, 0.08, 0.90, 3.0, 15.0, 0.20, 0.80],
            [0.01, 0.10, 0.005, 0.10, 0.88, 4.0, 8.0, 0.10, 0.70],
            [0.00, 0.05, 0.01, 0.05, 0.94, 3.5, 12.0, 0.18, 0.60],
        ]
        n_params = 9
    else:  # 3-state
        def nll(p):
            try:
                mu0, mu1, mu2, gv, a, b, nu0, nu1, nu2 = p[:9]
                t00, t01, t11, t12, t22, t20 = p[9:15]
                if gv <= 0 or a < 0 or b < 0 or a + b >= 1.5: return 1e10
                if any(n < 2.1 for n in [nu0, nu1, nu2]): return 1e10
                tm = np.zeros((3, 3))
                tm[0, 0] = t00; tm[0, 1] = t01; tm[0, 2] = max(1 - t00 - t01, 0)
                tm[1, 1] = t11; tm[1, 0] = max(1 - t11 - t12, 0.01); tm[1, 2] = t12
                tm[2, 2] = t22; tm[2, 0] = t20; tm[2, 1] = max(1 - t22 - t20, 0)
                for ii in range(3):
                    s = tm[ii].sum()
                    tm[ii] = tm[ii] / s if s > 0 else np.ones(3) / 3
                ll, _, _, _ = kim_filter(y, [mu0, mu1, mu2], gv, a, b, [nu0, nu1, nu2], tm, 3)
                return -ll if np.isfinite(ll) else 1e10
            except: return 1e10
        bd = [
            (-3, 3), (-3, 3), (-3, 3),
            (1e-8, 2), (1e-8, 0.5), (0.3, 0.999),
            (2.1, 50), (2.1, 50), (2.1, 50),
            (0.05, 0.95), (0.01, 0.5),
            (0.05, 0.95), (0.01, 0.5),
            (0.05, 0.95), (0.01, 0.5),
        ]
        fixed = [
            [0.01, 0.05, 0.10, 0.01, 0.05, 0.92, 3.0, 8.0, 20.0, 0.80, 0.10, 0.70, 0.15, 0.75, 0.10],
            [0.02, 0.04, 0.08, 0.02, 0.08, 0.88, 4.0, 6.0, 15.0, 0.70, 0.15, 0.65, 0.20, 0.70, 0.15],
            [0.00, 0.05, 0.12, 0.005, 0.10, 0.85, 2.5, 10.0, 25.0, 0.85, 0.08, 0.60, 0.25, 0.80, 0.08],
        ]
        n_params = 15

    all_starts = [np.array(x) for x in fixed]
    for _ in range(n_rand):
        if n_states == 2:
            x0 = [rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0.001, 0.1),
                  rng.uniform(0.01, 0.2), rng.uniform(0.7, 0.98),
                  rng.uniform(2.5, 20), rng.uniform(2.5, 30),
                  rng.uniform(0.05, 0.5), rng.uniform(0.3, 0.95)]
        else:
            x0 = [rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1),
                  rng.uniform(0.001, 0.1), rng.uniform(0.01, 0.2), rng.uniform(0.7, 0.98),
                  rng.uniform(2.5, 10), rng.uniform(4, 15), rng.uniform(10, 30),
                  rng.uniform(0.5, 0.95), rng.uniform(0.02, 0.3),
                  rng.uniform(0.5, 0.95), rng.uniform(0.02, 0.3),
                  rng.uniform(0.5, 0.95), rng.uniform(0.02, 0.3)]
        all_starts.append(np.array(x0))

    best_ll = -1e10; best_x = None
    nm_maxiter = 1000 if n_states == 2 else 1500
    for i, x0 in enumerate(all_starts):
        if nll(x0) >= 1e9: continue
        t0 = time.time()
        try:
            r = minimize(nll, x0, method='Nelder-Mead',
                         options={'maxiter': nm_maxiter, 'xatol': 1e-5, 'fatol': 1e-5})
            dt = time.time() - t0
            if -r.fun > best_ll and np.isfinite(r.fun):
                best_ll = -r.fun; best_x = r.x.copy()
            print(f"    [{i+1}/{len(all_starts)}] LL={-r.fun:.2f} ({dt:.1f}s)", flush=True)
        except Exception as ex:
            print(f"    [{i+1}/{len(all_starts)}] FAILED: {ex}", flush=True)

    if best_x is not None:
        print("  Polishing...", flush=True)
        try:
            r2 = minimize(nll, best_x, method='L-BFGS-B', bounds=bd,
                          options={'maxiter': 500, 'ftol': 1e-14})
            if -r2.fun > best_ll:
                best_ll = -r2.fun; best_x = r2.x
                print(f"  -> LL={best_ll:.2f}", flush=True)
        except: pass

    print(f"  Best {n_states}-state LL: {best_ll:.2f}", flush=True)
    return best_x, best_ll, n_params


def get_transmat(x, n_states):
    if n_states == 2:
        pr, qr = x[7], x[8]
        return np.array([[pr, 1 - pr], [1 - qr, qr]])
    else:
        t00, t01, t11, t12, t22, t20 = x[9:15]
        tm = np.zeros((3, 3))
        tm[0, 0] = t00; tm[0, 1] = t01; tm[0, 2] = max(1 - t00 - t01, 0)
        tm[1, 1] = t11; tm[1, 0] = max(1 - t11 - t12, 0.01); tm[1, 2] = t12
        tm[2, 2] = t22; tm[2, 0] = t20; tm[2, 1] = max(1 - t22 - t20, 0)
        for i in range(3):
            s = tm[i].sum()
            if s > 0: tm[i] /= s
        return tm


def get_params(x, n_states):
    if n_states == 2:
        return x[:2], x[2], x[3], x[4], x[5:7]
    else:
        return x[:3], x[3], x[4], x[5], x[6:9]


# =====================================================================
# Main
# =====================================================================
def main():
    T0 = time.time()
    print("=" * 60, flush=True)
    print("Full Comparison: Original + New Data", flush=True)
    print("=" * 60, flush=True)

    # Load all data
    df_orig = pd.read_excel(os.path.join(DATA, 'dueker_jbes1997', 'spret_dueker.xls'), engine='xlrd')
    y_orig = df_orig['SPRET'].values * 100.0
    dates_orig = pd.bdate_range('1982-01-04', periods=len(y_orig))

    y_new = pd.read_csv(os.path.join(DATA, 'sp500_returns_new.csv'),
                        index_col=0, parse_dates=True).iloc[:, 0].values
    dates_new = pd.read_csv(os.path.join(DATA, 'sp500_returns_new.csv'),
                            index_col=0, parse_dates=True).index
    vix = pd.read_csv(os.path.join(DATA, 'vix_data.csv'),
                      index_col=0, parse_dates=True)

    print(f"Original: {len(y_orig)} ({dates_orig[0].date()}~{dates_orig[-1].date()})", flush=True)
    print(f"New:      {len(y_new)} ({dates_new[0].date()}~{dates_new[-1].date()})", flush=True)

    # JIT warmup
    print("JIT warmup...", flush=True)
    _ = kim_filter(y_orig[:100], [0.0, 0.05], 0.01, 0.05, 0.93, [5.0, 10.0],
                   np.array([[0.85, 0.15], [0.35, 0.65]]), 2)

    all_results = {}

    for dataset_name, y, dates in [('Original (1982-1991)', y_orig, dates_orig),
                                     ('New (2016-2026)', y_new, dates_new)]:
        print(f"\n{'='*60}", flush=True)
        print(f"  {dataset_name}", flush=True)
        print(f"{'='*60}", flush=True)

        # GARCH
        print(f"\n  Standard t-GARCH...", flush=True)
        gr = fit_garch(y)
        print(f"    LL={gr['ll']:.2f}  a={gr['alpha']:.4f}  b={gr['beta']:.4f}  "
              f"a+b={gr['alpha']+gr['beta']:.4f}  nu={gr['nu']:.1f}", flush=True)
        vol_g = np.sqrt(np.maximum(gr['h'], 0)) * np.sqrt(252)
        m_g = vix_metrics(dates, vol_g, vix, f'GARCH')

        # 2-state
        print(f"\n  2-state GARCH-DF...", flush=True)
        x2, ll2, np2 = fit_ms(y, 2, n_rand=6)
        m_2 = None; psm_2 = None; vol_2 = None
        if x2 is not None:
            mu_v, gv, a, b, nu_v = get_params(x2, 2)
            tm = get_transmat(x2, 2)
            ll_f, pt_f, _, h_f = kim_filter(y, mu_v, gv, a, b, nu_v, tm, 2)
            psm_2 = smooth_nv(pt_f, tm, 2)
            vol_2 = np.sqrt(np.maximum(h_f[:, 0], 0)) * np.sqrt(252)
            m_2 = vix_metrics(dates, vol_2, vix, f'GARCH-DF (2-state)')
            print(f"    mu=[{x2[0]:.4f}, {x2[1]:.4f}]  gv={x2[2]:.6f}  a={x2[3]:.4f}  b={x2[4]:.4f}  a+b={x2[3]+x2[4]:.4f}", flush=True)
            print(f"    nu=[{x2[5]:.1f}, {x2[6]:.1f}]  p={x2[7]:.4f}  q={x2[8]:.4f}", flush=True)
            if m_2: print(f"    VIX corr={m_2['corr']:.4f}  MSE={m_2['mse']:.6f}", flush=True)

        # 3-state
        print(f"\n  3-state GARCH-DF...", flush=True)
        x3, ll3, np3 = fit_ms(y, 3, n_rand=4)
        m_3 = None; psm_3 = None; vol_3 = None
        if x3 is not None:
            mu_v3, gv3, a3, b3, nu_v3 = get_params(x3, 3)
            tm3 = get_transmat(x3, 3)
            ll_3f, pt_3f, _, h_3f = kim_filter(y, mu_v3, gv3, a3, b3, nu_v3, tm3, 3)
            psm_3 = smooth_nv(pt_3f, tm3, 3)
            vol_3 = np.sqrt(np.maximum(h_3f[:, 0], 0)) * np.sqrt(252)
            m_3 = vix_metrics(dates, vol_3, vix, f'GARCH-DF (3-state)')
            print(f"    mu=[{x3[0]:.4f}, {x3[1]:.4f}, {x3[2]:.4f}]", flush=True)
            print(f"    gv={gv3:.6f}  a={a3:.4f}  b={b3:.4f}  a+b={a3+b3:.4f}", flush=True)
            print(f"    nu=[{x3[6]:.1f}, {x3[7]:.1f}, {x3[8]:.1f}]", flush=True)
            print(f"    Trans:", flush=True)
            for i in range(3): print(f"      [{tm3[i,0]:.3f} {tm3[i,1]:.3f} {tm3[i,2]:.3f}]", flush=True)
            if m_3: print(f"    VIX corr={m_3['corr']:.4f}  MSE={m_3['mse']:.6f}", flush=True)

        tag = 'orig' if 'Original' in dataset_name else 'new'
        all_results[tag] = {
            'garch': gr, 'x2': x2, 'll2': ll2, 'np2': np2, 'm_g': m_g, 'm_2': m_2,
            'x3': x3, 'll3': ll3, 'np3': np3, 'm_3': m_3,
            'psm_2': psm_2, 'vol_2': vol_2, 'psm_3': psm_3, 'vol_3': vol_3,
            'vol_g': vol_g, 'dates': dates, 'y': y,
        }

    # ===================== SUMMARY TABLE =====================
    print(f"\n{'='*70}", flush=True)
    print("FULL COMPARISON TABLE", flush=True)
    print(f"{'='*70}", flush=True)

    print(f"\n--- Log-Likelihood ---", flush=True)
    print(f"  {'Model':<25} {'LL(orig)':>12} {'LL(new)':>12} {'AIC(orig)':>12} {'AIC(new)':>12}", flush=True)
    print(f"  {'-'*75}", flush=True)

    ro, rn = all_results['orig'], all_results['new']
    g_ll_o, g_ll_n = ro['garch']['ll'], rn['garch']['ll']
    print(f"  {'GARCH(1,1)':<25} {g_ll_o:>12.2f} {g_ll_n:>12.2f} {-2*g_ll_o+10:>12.2f} {-2*g_ll_n+10:>12.2f}", flush=True)
    print(f"  {'GARCH-DF 2-state':<25} {ro['ll2']:>12.2f} {rn['ll2']:>12.2f} {-2*ro['ll2']+18:>12.2f} {-2*rn['ll2']+18:>12.2f}", flush=True)
    print(f"  {'GARCH-DF 3-state':<25} {ro['ll3']:>12.2f} {rn['ll3']:>12.2f} {-2*ro['ll3']+30:>12.2f} {-2*rn['ll3']+30:>12.2f}", flush=True)

    print(f"\n--- VIX Prediction ---", flush=True)
    print(f"  {'Model':<25} {'Period':>12} {'N':>6} {'Corr':>8} {'MSE':>10} {'MAE':>8}", flush=True)
    print(f"  {'-'*72}", flush=True)
    for tag, per in [('orig', '1990-1991'), ('new', '2016-2026')]:
        r = all_results[tag]
        for m in [r['m_g'], r['m_2'], r['m_3']]:
            if m: print(f"  {m['label']:<25} {per:>12} {m['n']:>6} {m['corr']:>8.4f} {m['mse']:>10.6f} {m['mae']:>8.4f}", flush=True)

    # ===================== PLOTS =====================
    print(f"\nGenerating plots...", flush=True)

    for tag, title_suffix in [('orig', 'Original (1982-1991)'), ('new', 'New (2016-2026)')]:
        r = all_results[tag]
        fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        dates = r['dates']; y = r['y']

        # Panel a: returns + regime
        axs[0].bar(dates, np.abs(y), width=1, alpha=0.3, color='gray')
        if r['psm_2'] is not None:
            axr = axs[0].twinx()
            axr.plot(dates, r['psm_2'][:, 1], 'r-', alpha=0.7, lw=0.5)
            axr.set_ylabel('P(High-vol)', color='r'); axr.set_ylim(0, 1)
        axs[0].set_ylabel('|Returns|')
        axs[0].set_title(f'(a) Returns & Regime Prob — {title_suffix}')

        # Panel b: volatility
        axs[1].plot(dates[2:], r['vol_g'][2:], 'b-', alpha=0.5, lw=0.5, label='GARCH')
        if r['vol_2'] is not None:
            axs[1].plot(dates[2:], r['vol_2'][2:], 'r-', alpha=0.5, lw=0.5, label='2-state')
        if r['vol_3'] is not None:
            axs[1].plot(dates[2:], r['vol_3'][2:], 'g-', alpha=0.5, lw=0.5, label='3-state')
        axs[1].set_ylabel('Ann. Vol (%)'); axs[1].legend()
        axs[1].set_title('(b) Conditional Volatility')

        # Panel c: VIX
        for m, c in [(r['m_g'], 'b'), (r['m_2'], 'r'), (r['m_3'], 'g')]:
            if m:
                axs[2].plot(m['cm'], m['vm'], c+'-', alpha=0.6, lw=0.7,
                           label=f'{m["label"]} (corr={m["corr"]:.3f})')
        if r['m_g']:
            axs[2].plot(r['m_g']['cm'], r['m_g']['vx'], 'k-', lw=0.8, label='VIX')
        axs[2].legend(fontsize=8); axs[2].set_ylabel('Volatility (%)')
        axs[2].set_title(f'(c) Model vs VIX')
        plt.tight_layout()
        fig.savefig(os.path.join(FIGS, f'fig_all_{tag}.png'), dpi=150, bbox_inches='tight')
        print(f"  fig_all_{tag}.png", flush=True)

        plt.close(fig)

    # Normalized comparison
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (tag, title) in zip(axes2, [('orig', 'Original (1990-1991)'), ('new', 'New (2016-2026)')]):
        r = all_results[tag]
        if r['m_g']:
            ax.plot(r['m_g']['cm'], r['m_g']['vx_n'], 'k-', lw=0.8, label='VIX')
            ax.plot(r['m_g']['cm'], r['m_g']['vm_n'], 'b-', alpha=0.5, lw=0.6,
                   label=f'GARCH ({r["m_g"]["corr"]:.3f})')
        if r['m_2']:
            ax.plot(r['m_2']['cm'], r['m_2']['vm_n'], 'r-', alpha=0.5, lw=0.6,
                   label=f'2-state ({r["m_2"]["corr"]:.3f})')
        if r['m_3']:
            ax.plot(r['m_3']['cm'], r['m_3']['vm_n'], 'g-', alpha=0.5, lw=0.6,
                   label=f'3-state ({r["m_3"]["corr"]:.3f})')
        ax.legend(fontsize=8); ax.set_title(title)
    fig2.suptitle('Normalized Volatility vs VIX', y=1.02)
    plt.tight_layout()
    fig2.savefig(os.path.join(FIGS, 'fig_all_normalized.png'), dpi=150, bbox_inches='tight')
    print("  fig_all_normalized.png", flush=True)
    plt.close(fig2)

    # Scatter
    fig3, axes3 = plt.subplots(2, 3, figsize=(15, 10))
    for row, (tag, tname) in enumerate([('orig', 'Original'), ('new', 'New')]):
        r = all_results[tag]
        for col, (m, color, mname) in enumerate([
            (r['m_g'], 'blue', 'GARCH'),
            (r['m_2'], 'red', '2-state'),
            (r['m_3'], 'green', '3-state')
        ]):
            ax = axes3[row, col]
            if m:
                mask = m['vm_n'].notna() & m['vx_n'].notna()
                ax.scatter(m['vm_n'][mask], m['vx_n'][mask], s=2, alpha=0.15, c=color)
                lims = [0, max(m['vm_n'][mask].quantile(0.99), m['vx_n'][mask].quantile(0.99)) * 1.1]
                ax.plot(lims, lims, 'k--', lw=0.5)
                ax.set_title(f'{tname}: {mname}\ncorr={m["corr"]:.3f}')
            ax.set_xlabel('Model (norm)'); ax.set_ylabel('VIX (norm)')
    plt.tight_layout()
    fig3.savefig(os.path.join(FIGS, 'fig_all_scatter.png'), dpi=150, bbox_inches='tight')
    print("  fig_all_scatter.png", flush=True)
    plt.close(fig3)

    # CSV
    rows = []
    for tag, per in [('orig', '1990-1991'), ('new', '2016-2026')]:
        r = all_results[tag]
        for m in [r['m_g'], r['m_2'], r['m_3']]:
            if m: rows.append({'Model': m['label'], 'Period': per, 'N': m['n'],
                               'Correlation': round(m['corr'], 4),
                               'MSE': round(m['mse'], 6), 'MAE': round(m['mae'], 4)})
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'vix_comparison_full.csv'), index=False)
    print("  CSV saved", flush=True)

    print(f"\nTotal time: {time.time() - T0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
