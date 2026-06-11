"""
Dueker (1997) 完整分析: GARCH vs GARCH-DF 预测VIX对比
生成报告所需的所有数据、表格和图表
"""
import numpy as np, pandas as pd, time, warnings, os
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import chi2 as chi2d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size'] = 11
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
FIGS = os.path.join(ROOT, 'figures')
RESULTS = os.path.join(ROOT, 'results')

# =====================================================================
# Kim Filter - GARCH-DF
# =====================================================================
def kim_df(y, mu, gv, a, b, nu_vec, p_vec):
    nu_vec=np.asarray(nu_vec,dtype=float); p_vec=np.asarray(p_vec,dtype=float)
    T=len(y); rc=np.array([0,0,1,1]); rl=np.array([0,1,0,1])
    P=np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            if rl[j]==rc[i]: P[i,j]=p_vec[rc[i]] if rc[j]==rc[i] else 1-p_vec[rc[i]]
    e=np.column_stack([y-mu[0],y-mu[1]]); h=np.full((T,4),np.var(y))
    uus=np.zeros((T,2))
    for k in range(2): uus[0,k]=np.var(y)*(nu_vec[k]-2)/nu_vec[k]
    pt=np.zeros((T,4)); pi0=(1-p_vec[1])/(2-p_vec[0]-p_vec[1])
    for i in range(4): pt[0,i]=(pi0 if rc[i]==0 else 1-pi0)*(pi0 if rl[i]==0 else 1-pi0)
    pt[0]/=pt[0].sum(); ll=0.0
    for t in range(1,T):
        for i in range(4): h[t,i]=gv+a*uus[t-1,rl[i]]+b*h[t-1,i]
        h[t]=np.maximum(h[t],1e-30)
        fvec=np.zeros(4)
        for i in range(4):
            k=rc[i]; nu_c=nu_vec[k]
            sc2=max(h[t,i]*(nu_c-2)/nu_c,1e-30); x=e[t,k]
            logf=gammaln((nu_c+1)/2)-gammaln(nu_c/2)-0.5*np.log(nu_c*np.pi)-0.5*np.log(sc2)-((nu_c+1)/2)*np.log(1+x**2/(sc2*nu_c))
            fvec[i]=max(np.exp(np.clip(logf,-700,700)),1e-300)
        pp=pt[t-1]@P; joint=pp*fvec; fpt=max(joint.sum(),1e-300)
        pt[t]=joint/fpt; ll+=np.log(fpt)
        ph=pt[t]*h[t]
        for k in range(2):
            m=rc==k; pdiv=pt[t,m].sum()
            hk=ph[m].sum()/pdiv if pdiv>1e-300 else gv
            uus[t,k]=e[t,k]**2*(nu_vec[k]-2)/nu_vec[k]; h[t,m]=hk
    return ll,pt,e,h

def smth(pt,p_vec):
    p_vec=np.asarray(p_vec,dtype=float); T=len(pt)
    rc=np.array([0,0,1,1]); rl=np.array([0,1,0,1])
    P=np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            if rl[j]==rc[i]: P[i,j]=p_vec[rc[i]] if rc[j]==rc[i] else 1-p_vec[rc[i]]
    psm=np.zeros((T,2)); psm[T-1,0]=pt[T-1,rc==0].sum(); psm[T-1,1]=1-psm[T-1,0]
    ps=pt[T-1].copy()
    for t in range(T-2,0,-1):
        pp=np.maximum(pt[t]@P,1e-300); psn=pt[t]*(P@(ps/pp)); s=psn.sum()
        ps=psn/s if s>0 else psn; psm[t,0]=ps[rc==0].sum(); psm[t,1]=1-psm[t,0]
    return psm

# =====================================================================
# Standard t-GARCH(1,1)
# =====================================================================
def fit_garch(y):
    """Fit standard t-GARCH(1,1), return params, LL, h series"""
    def nll(p):
        mu,w,a,b,nu=p
        if w<=0 or a<0 or b<0 or a+b>=0.999 or nu<2.5: return 1e10
        ee=y-mu; h=np.var(y); ll=0.0
        for t in range(1,len(y)):
            h=w+a*ee[t-1]**2+b*h
            if h<=0: return 1e10
            sc2=max(h*(nu-2)/nu,1e-30)
            ll+=gammaln((nu+1)/2)-gammaln(nu/2)-0.5*np.log(nu*np.pi)-0.5*np.log(sc2)-((nu+1)/2)*np.log(1+ee[t]**2/(sc2*nu))
        return -ll
    x0=[np.mean(y), np.var(y)*0.05, 0.04, 0.93, 8.0]
    res=minimize(nll,x0,method='Nelder-Mead',options={'maxiter':3000,'xatol':1e-6,'fatol':1e-6})
    mu,w,a,b,nu=res.x
    # Reconstruct h
    ee=y-mu; h=np.zeros(len(y)); h[0]=np.var(y)
    for t in range(1,len(y)): h[t]=w+a*ee[t-1]**2+b*h[t-1]
    return {'mu':mu,'omega':w,'alpha':a,'beta':b,'nu':nu,'ll':-res.fun,'h':h}

def garch_chi2(y, h, nu):
    """Chi-squared GOF for standard GARCH"""
    z=(y-nu*0)/np.sqrt(np.maximum(h,1e-10))  # simplified: z = y/sigma
    obs,_=np.histogram(z[2:],bins=np.quantile(z[2:],np.linspace(0,1,11)))
    exp=len(z[2:])/10; chi2=np.sum((obs-exp)**2/exp)
    return chi2, 1-chi2d.cdf(chi2,9)

# =====================================================================
# Volatility forecast & VIX comparison
# =====================================================================
def vix_metrics(dates, vol_ann, vix, label):
    """Compare annualized volatility with VIX"""
    vol22=pd.Series(vol_ann).rolling(22,min_periods=10).mean()
    vm=pd.Series(vol22.values,index=dates)
    cm=vm.index.intersection(vix.index)
    if len(cm)<20: return None
    vx=vix.loc[cm,'VIX']; vmc=vm.loc[cm]
    vx_n=vx/vx.mean(); vm_n=vmc/vmc.mean()
    mask=vm_n.notna()&vx_n.notna()
    corr=np.corrcoef(vm_n[mask],vx_n[mask])[0,1]
    mse=np.mean((vm_n[mask]-vx_n[mask])**2)
    mae=np.mean(np.abs(vm_n[mask]-vx_n[mask]))
    return {'label':label,'n':int(mask.sum()),
            'vix_mean':vx.mean(),'model_mean':vmc.mean(),
            'corr':corr,'mse':mse,'mae':mae,
            'vx':vx,'vm':vmc,'vx_n':vx_n,'vm_n':vm_n,'cm':cm}

# =====================================================================
# Main
# =====================================================================
def main():
    T0=time.time()
    print("="*60)
    print("Complete Analysis: GARCH vs GARCH-DF VIX Prediction")
    print("="*60)

    # --- Load data ---
    df_orig=pd.read_excel(os.path.join(DATA, 'dueker_jbes1997', 'spret_dueker.xls'), engine='xlrd')
    y_orig=df_orig['SPRET'].values*100.0
    dates_orig=pd.bdate_range('1982-01-04',periods=len(y_orig))

    df_new=pd.read_csv(os.path.join(DATA, 'sp500_returns_new.csv'), index_col=0, parse_dates=True)
    y_new=df_new.iloc[:,0].values; dates_new=df_new.index

    vix=pd.read_csv(os.path.join(DATA, 'vix_data.csv'), index_col=0, parse_dates=True)

    print(f"Original: {len(y_orig)} ({dates_orig[0].date()}~{dates_orig[-1].date()})")
    print(f"New:      {len(y_new)} ({dates_new[0].date()}~{dates_new[-1].date()})")
    print(f"VIX:      {len(vix)} ({vix.index[0].date()}~{vix.index[-1].date()})")

    # ===================== ORIGINAL DATA =====================
    print(f"\n{'='*50}")
    print("ORIGINAL DATA (1982-1991)")
    print(f"{'='*50}")

    # --- Standard GARCH ---
    print("\nFitting standard t-GARCH...")
    gr=fit_garch(y_orig)
    print(f"  LL={gr['ll']:.2f}  mu={gr['mu']:.4f}  w={gr['omega']:.6f}  "
          f"a={gr['alpha']:.4f}  b={gr['beta']:.4f}  nu={gr['nu']:.1f}  a+b={gr['alpha']+gr['beta']:.4f}")
    chi2_g,p_g=garch_chi2(y_orig,gr['h'],gr['nu'])
    print(f"  Chi2 GOF: {chi2_g:.2f}, p={p_g:.4f}")

    # --- GARCH-DF ---
    print("\nGARCH-DF (cached best params)...")
    p_df=np.array([0.02055,0.08677,0.0212,0.06218,0.93635,2.58,19.33,0.18,0.619])
    ll_df,pt_df,_,h_df=kim_df(y_orig,p_df[:2],p_df[2],p_df[3],p_df[4],p_df[5:7],p_df[7:9])
    psm_df=smth(pt_df,p_df[7:9])
    mu_t=psm_df[:,0]*p_df[0]+psm_df[:,1]*p_df[1]
    z=(y_orig-mu_t)/np.sqrt(np.maximum(h_df[:,0],1e-10))
    obs,_=np.histogram(z[2:],bins=np.quantile(z[2:],np.linspace(0,1,11)))
    exp=len(z[2:])/10; chi2_df=np.sum((obs-exp)**2/exp)
    p_df_chi2=1-chi2d.cdf(chi2_df,9)
    print(f"  LL={ll_df:.2f}  a={p_df[3]:.4f}  b={p_df[4]:.4f}  a+b={p_df[3]+p_df[4]:.4f}")
    print(f"  nu=[{p_df[5]:.1f},{p_df[6]:.1f}]  p={p_df[7]:.4f}  q={p_df[8]:.4f}")
    print(f"  Chi2 GOF: {chi2_df:.2f}, p={p_df_chi2:.4f}")

    # --- VIX comparison (original) ---
    vol_g_orig=np.sqrt(np.maximum(gr['h'],0))*np.sqrt(252)
    vol_df_orig=np.sqrt(np.maximum(h_df[:,0],0))*np.sqrt(252)
    m_g_orig=vix_metrics(dates_orig,vol_g_orig,vix,'GARCH (orig)')
    m_df_orig=vix_metrics(dates_orig,vol_df_orig,vix,'GARCH-DF (orig)')

    print(f"\nVIX Prediction (1990-1991):")
    print(f"  {'Model':<16} {'N':>6} {'Corr':>8} {'MSE':>10} {'MAE':>8}")
    for m in [m_g_orig,m_df_orig]:
        if m: print(f"  {m['label']:<16} {m['n']:>6} {m['corr']:>8.4f} {m['mse']:>10.6f} {m['mae']:>8.4f}")

    # ===================== NEW DATA =====================
    print(f"\n{'='*50}")
    print("NEW DATA (2016-2026)")
    print(f"{'='*50}")

    # --- Standard GARCH on new data ---
    print("\nFitting standard t-GARCH on new data...")
    gr_n=fit_garch(y_new)
    print(f"  LL={gr_n['ll']:.2f}  a={gr_n['alpha']:.4f}  b={gr_n['beta']:.4f}  "
          f"nu={gr_n['nu']:.1f}  a+b={gr_n['alpha']+gr_n['beta']:.4f}")
    chi2_gn,p_gn=garch_chi2(y_new,gr_n['h'],gr_n['nu'])
    print(f"  Chi2 GOF: {chi2_gn:.2f}, p={p_gn:.4f}")

    # --- GARCH-DF on new data ---
    print("\nFitting GARCH-DF on new data...")
    def nll_df(pp):
        try:
            ll,_,_,_=kim_df(y_new,pp[:2],pp[2],pp[3],pp[4],pp[5:7],pp[7:9])
            return -ll if np.isfinite(ll) else 1e10
        except: return 1e10
    res_df_n=minimize(nll_df,p_df,method='Nelder-Mead',options={'maxiter':150,'xatol':1e-4,'fatol':1e-4})
    pn=res_df_n.x
    print(f"  LL={-res_df_n.fun:.2f}  a={pn[3]:.4f}  b={pn[4]:.4f}  a+b={pn[3]+pn[4]:.4f}")
    print(f"  nu=[{pn[5]:.1f},{pn[6]:.1f}]  p={pn[7]:.4f}  q={pn[8]:.4f}")

    lln,ptn,_,hn=kim_df(y_new,pn[:2],pn[2],pn[3],pn[4],pn[5:7],pn[7:9])
    psmn=smth(ptn,pn[7:9])

    # --- VIX comparison (new) ---
    vol_g_new=np.sqrt(np.maximum(gr_n['h'],0))*np.sqrt(252)
    vol_df_new=np.sqrt(np.maximum(hn[:,0],0))*np.sqrt(252)
    m_g_new=vix_metrics(dates_new,vol_g_new,vix,'GARCH (new)')
    m_df_new=vix_metrics(dates_new,vol_df_new,vix,'GARCH-DF (new)')

    print(f"\nVIX Prediction (2016-2026):")
    print(f"  {'Model':<16} {'N':>6} {'Corr':>8} {'MSE':>10} {'MAE':>8}")
    for m in [m_g_new,m_df_new]:
        if m: print(f"  {m['label']:<16} {m['n']:>6} {m['corr']:>8.4f} {m['mse']:>10.6f} {m['mae']:>8.4f}")

    # ===================== PLOTS =====================
    print("\nGenerating figures...")

    # --- Figure 1: Original data overview ---
    fig1,axs1=plt.subplots(3,1,figsize=(14,10),sharex=True)
    axs1[0].bar(dates_orig,np.abs(y_orig),width=1,alpha=0.3,color='gray')
    ax0r=axs1[0].twinx(); ax0r.plot(dates_orig,psm_df[:,1],'r-',alpha=0.7,lw=0.5)
    ax0r.set_ylabel('P(High-vol)',color='r'); ax0r.set_ylim(0,1)
    axs1[0].set_ylabel('|Returns|'); axs1[0].set_title('(a) S&P 500 Daily Returns & Regime Probabilities (GARCH-DF)')
    axs1[1].plot(dates_orig[2:],vol_g_orig[2:],'b-',alpha=0.5,lw=0.5,label='GARCH')
    axs1[1].plot(dates_orig[2:],vol_df_orig[2:],'r-',alpha=0.5,lw=0.5,label='GARCH-DF')
    axs1[1].set_ylabel('Ann. Vol (%)'); axs1[1].legend(); axs1[1].set_title('(b) Conditional Volatility Comparison')
    if m_g_orig and m_df_orig:
        axs1[2].plot(m_g_orig['cm'],m_g_orig['vx'],'k-',lw=1,label='VIX')
        axs1[2].plot(m_g_orig['cm'],m_g_orig['vm'],'b-',alpha=0.7,lw=0.8,label='GARCH')
        axs1[2].plot(m_df_orig['cm'],m_df_orig['vm'],'r-',alpha=0.7,lw=0.8,label='GARCH-DF')
    axs1[2].legend(); axs1[2].set_ylabel('Volatility (%)'); axs1[2].set_title('(c) Model vs VIX (1990-1991)')
    plt.tight_layout()
    fig1.savefig(os.path.join(FIGS, 'fig1_original.png'), dpi=150, bbox_inches='tight')
    print("  fig1_original.png")

    # --- Figure 2: New data overview ---
    fig2,axs2=plt.subplots(3,1,figsize=(14,10),sharex=True)
    axs2[0].bar(dates_new,np.abs(y_new),width=1,alpha=0.3,color='gray')
    ax0r2=axs2[0].twinx(); ax0r2.plot(dates_new,psmn[:,1],'r-',alpha=0.7,lw=0.5)
    ax0r2.set_ylabel('P(High-vol)',color='r'); ax0r2.set_ylim(0,1)
    axs2[0].set_ylabel('|Returns|'); axs2[0].set_title('(a) S&P 500 Returns & Regime Prob (GARCH-DF, 2016-2026)')
    axs2[1].plot(dates_new[2:],vol_g_new[2:],'b-',alpha=0.5,lw=0.5,label='GARCH')
    axs2[1].plot(dates_new[2:],vol_df_new[2:],'r-',alpha=0.5,lw=0.5,label='GARCH-DF')
    axs2[1].set_ylabel('Ann. Vol (%)'); axs2[1].legend(); axs2[1].set_title('(b) Conditional Volatility')
    if m_g_new and m_df_new:
        axs2[2].plot(m_g_new['cm'],m_g_new['vx'],'k-',lw=0.8,label='VIX')
        axs2[2].plot(m_g_new['cm'],m_g_new['vm'],'b-',alpha=0.6,lw=0.7,label='GARCH')
        axs2[2].plot(m_df_new['cm'],m_df_new['vm'],'r-',alpha=0.6,lw=0.7,label='GARCH-DF')
    axs2[2].legend(); axs2[2].set_ylabel('Volatility (%)'); axs2[2].set_title('(c) Model vs VIX (2016-2026)')
    plt.tight_layout()
    fig2.savefig(os.path.join(FIGS, 'fig2_new_data.png'), dpi=150, bbox_inches='tight')
    print("  fig2_new_data.png")

    # --- Figure 3: Normalized comparison ---
    fig3,axs3=plt.subplots(1,2,figsize=(14,5))
    if m_g_orig and m_df_orig:
        axs3[0].plot(m_g_orig['cm'],m_g_orig['vx_n'],'k-',lw=0.8,label='VIX')
        axs3[0].plot(m_g_orig['cm'],m_g_orig['vm_n'],'b-',alpha=0.7,lw=0.7,label=f'GARCH (corr={m_g_orig["corr"]:.3f})')
        axs3[0].plot(m_df_orig['cm'],m_df_orig['vm_n'],'r-',alpha=0.7,lw=0.7,label=f'GARCH-DF (corr={m_df_orig["corr"]:.3f})')
        axs3[0].legend(fontsize=9); axs3[0].set_title('Original Data (1990-1991)')
        axs3[0].set_ylabel('Normalized Volatility')
    if m_g_new and m_df_new:
        axs3[1].plot(m_g_new['cm'],m_g_new['vx_n'],'k-',lw=0.8,label='VIX')
        axs3[1].plot(m_g_new['cm'],m_g_new['vm_n'],'b-',alpha=0.6,lw=0.6,label=f'GARCH (corr={m_g_new["corr"]:.3f})')
        axs3[1].plot(m_df_new['cm'],m_df_new['vm_n'],'r-',alpha=0.6,lw=0.6,label=f'GARCH-DF (corr={m_df_new["corr"]:.3f})')
        axs3[1].legend(fontsize=9); axs3[1].set_title('New Data (2016-2026)')
    fig3.suptitle('Normalized Model-Implied Volatility vs VIX',y=1.02)
    plt.tight_layout()
    fig3.savefig(os.path.join(FIGS, 'fig3_normalized.png'), dpi=150, bbox_inches='tight')
    print("  fig3_normalized.png")

    # --- Figure 4: Scatter plot ---
    fig4,axs4=plt.subplots(1,2,figsize=(12,5))
    if m_g_orig and m_df_orig:
        mask=m_g_orig['vm_n'].notna()&m_g_orig['vx_n'].notna()
        axs4[0].scatter(m_g_orig['vm_n'][mask],m_g_orig['vx_n'][mask],s=3,alpha=0.3,c='blue')
        mask2=m_df_orig['vm_n'].notna()&m_df_orig['vx_n'].notna()
        axs4[0].scatter(m_df_orig['vm_n'][mask2],m_df_orig['vx_n'][mask2],s=3,alpha=0.3,c='red')
        axs4[0].plot([0,3],[0,3],'k--',lw=0.5)
        axs4[0].set_xlabel('Model Vol (norm)'); axs4[0].set_ylabel('VIX (norm)')
        axs4[0].set_title('Original Data'); axs4[0].legend(['45°','GARCH','GARCH-DF'],fontsize=8)
    if m_g_new and m_df_new:
        mask=m_g_new['vm_n'].notna()&m_g_new['vx_n'].notna()
        axs4[1].scatter(m_g_new['vm_n'][mask],m_g_new['vx_n'][mask],s=2,alpha=0.15,c='blue')
        mask2=m_df_new['vm_n'].notna()&m_df_new['vx_n'].notna()
        axs4[1].scatter(m_df_new['vm_n'][mask2],m_df_new['vx_n'][mask2],s=2,alpha=0.15,c='red')
        axs4[1].plot([0,3],[0,3],'k--',lw=0.5)
        axs4[1].set_xlabel('Model Vol (norm)'); axs4[1].set_ylabel('VIX (norm)')
        axs4[1].set_title('New Data'); axs4[1].legend(['45°','GARCH','GARCH-DF'],fontsize=8)
    fig4.suptitle('Model vs VIX Scatter Plot')
    plt.tight_layout()
    fig4.savefig(os.path.join(FIGS, 'fig4_scatter.png'), dpi=150, bbox_inches='tight')
    print("  fig4_scatter.png")

    # ===================== SAVE CSV =====================
    # Original
    pd.DataFrame({
        'Date':dates_orig,'Return':y_orig,
        'GARCH_vol':vol_g_orig,'GARCH_DF_vol':vol_df_orig,
        'P_High':psm_df[:,1],'Cond_Var':h_df[:,0]
    }).to_csv(os.path.join(RESULTS, 'results_original_full.csv'), index=False)
    # New
    pd.DataFrame({
        'Date':dates_new,'Return':y_new,
        'GARCH_vol':vol_g_new,'GARCH_DF_vol':vol_df_new,
        'P_High':psmn[:,1],'Cond_Var':hn[:,0]
    }).to_csv(os.path.join(RESULTS, 'results_new_full.csv'), index=False)
    # VIX comparison
    rows=[]
    for m in [m_g_orig,m_df_orig,m_g_new,m_df_new]:
        if m: rows.append({'Model':m['label'],'N':m['n'],'VIX_mean':round(m['vix_mean'],2),
                          'Model_mean':round(m['model_mean'],2),'Correlation':round(m['corr'],4),
                          'MSE':round(m['mse'],6),'MAE':round(m['mae'],4)})
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'vix_comparison_table.csv'), index=False)
    print("  CSV files saved")

    # ===================== PRINT SUMMARY TABLE =====================
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"\n--- Parameter Estimates ---")
    print(f"{'':>20} {'GARCH':>14} {'GARCH-DF(orig)':>16} {'GARCH-DF(new)':>16}")
    print(f"{'Log-likelihood':>20} {gr['ll']:>14.2f} {ll_df:>16.2f} {lln:>16.2f}")
    print(f"{'mu(S=0)':>20} {gr['mu']:>14.4f} {p_df[0]:>16.4f} {pn[0]:>16.4f}")
    print(f"{'mu(S=1)':>20} {'-':>14} {p_df[1]:>16.4f} {pn[1]:>16.4f}")
    print(f"{'alpha':>20} {gr['alpha']:>14.4f} {p_df[3]:>16.4f} {pn[3]:>16.4f}")
    print(f"{'beta':>20} {gr['beta']:>14.4f} {p_df[4]:>16.4f} {pn[4]:>16.4f}")
    print(f"{'alpha+beta':>20} {gr['alpha']+gr['beta']:>14.4f} {p_df[3]+p_df[4]:>16.4f} {pn[3]+pn[4]:>16.4f}")
    print(f"{'nu':>20} {gr['nu']:>14.2f} {p_df[5]:>7.1f}/{p_df[6]:<7.1f} {pn[5]:>7.1f}/{pn[6]:<7.1f}")

    print(f"\n--- VIX Prediction ---")
    print(f"{'Model':<28} {'Period':>12} {'N':>6} {'Corr':>8} {'MSE':>10} {'MAE':>8}")
    print("-"*74)
    for m in [m_g_orig,m_df_orig,m_g_new,m_df_new]:
        if m:
            per='1990-1991' if 'orig' in m['label'] else '2016-2026'
            print(f"{m['label']:<28} {per:>12} {m['n']:>6} {m['corr']:>8.4f} {m['mse']:>10.6f} {m['mae']:>8.4f}")

    print(f"\nTotal time: {time.time()-T0:.0f}s")

    return {
        'garch_orig':gr, 'df_orig':p_df, 'll_df':ll_df,
        'garch_new':gr_n, 'df_new':pn, 'll_df_new':lln,
        'm_g_orig':m_g_orig, 'm_df_orig':m_df_orig,
        'm_g_new':m_g_new, 'm_df_new':m_df_new,
        'psm_orig':psm_df, 'psm_new':psmn,
        'vol_g_orig':vol_g_orig, 'vol_df_orig':vol_df_orig,
        'vol_g_new':vol_g_new, 'vol_df_new':vol_df_new,
    }

if __name__=='__main__':
    main()
