"""Positive additive weight models; fit log residuals and keep smearing separate."""
import numpy as np
from scipy.optimize import least_squares, nnls
from scipy.special import logsumexp


def raw_values(df,terms):
    values={'L':df['길이'],'B':df['폭'],'D':df['깊이'],'LB':df['길이']*df['폭'],
            'LBD':df['길이']*df['폭']*df['깊이'],'P':df['소요마력'],'PE':df['소요전력']}
    return np.column_stack([np.ones(len(df))]+[values[t].to_numpy() for t in terms])


def solve(df,base,gamma=None):
    terms=base['terms'];classes=base['classes'];raw=raw_values(df,terms)
    scales=np.median(raw,axis=0);x=raw/scales;p=x.shape[1]
    dummies=np.column_stack([(df['함분류']==c).to_numpy(float) for c in classes[1:]]) if len(classes)>1 else np.empty((len(df),0))
    centered=(df['취역연도']-df['함분류'].map(base['mean_years'])).to_numpy()/10
    years=np.column_stack([(df['함분류']==c).to_numpy(float)*centered for c in classes])
    z=np.column_stack([dummies,years]) if gamma is None else dummies
    offset=np.zeros(len(df)) if gamma is None else years@np.array([gamma[c] for c in classes])
    y=np.log(df[base['target']].to_numpy());logx=np.log(x)
    initial=np.log(np.maximum(nnls(x,np.exp(y-offset))[0],np.exp(np.median(y))*1e-10))
    seeds=[np.r_[initial,np.zeros(z.shape[1])]]
    if base.get('additive_coeff'):
        core=np.log([base['additive_coeff'][t]*scales[j] for j,t in enumerate(['constant']+terms)])
        dt=np.array(base['coef'][1+len(terms):])
        seeds.insert(0,np.r_[core,dt,([base['gamma'][c] for c in classes] if gamma is None else [])])
    def residual(theta):return logsumexp(logx+theta[:p],axis=1)+z@theta[p:]+offset-y
    def jac(theta):
        v=logx+theta[:p];shares=np.exp(v-logsumexp(v,axis=1)[:,None])
        return np.column_stack([shares,z])
    fits=[least_squares(residual,np.clip(seed,-39,39),jac=jac,bounds=(-40,40),max_nfev=1500,ftol=1e-10,xtol=1e-10,gtol=1e-10) for seed in seeds]
    good=[r for r in fits if r.success and np.isfinite(r.fun).all()]
    if not good:raise ValueError('덧셈식의 수치 최적화가 수렴하지 않았습니다.')
    result=min(good,key=lambda r:float(r.fun@r.fun))
    return result,scales,p


def apply_result(base,result,scales,p,gamma):
    m={**base,'kind':'additive','size_by_type':False,'gamma':gamma,
       'additive_coeff':{t:float(np.exp(result.x[j])/scales[j]) for j,t in enumerate(['constant']+base['terms'])},
       'coef':[0.0]*(1+len(base['terms']))+result.x[p:p+len(base['classes'])-1].tolist(),
       'smearing':float(np.exp(-result.fun).mean())}
    return m


def fit_baseline(df,target,terms):
    from model import BASE,SHIP_TYPES
    counts=df['함분류'].value_counts()
    classes=sorted(counts.index.tolist(),key=lambda c:(-int(counts[c]),SHIP_TYPES.index(c)))
    means=df.groupby('함분류')['취역연도'].mean().to_dict()
    base={'target':target,'terms':terms,'classes':classes,'mean_years':means,'year_by_type':True,
          'gamma':{c:0.0 for c in classes},'coef':[0.0]*(len(terms)+len(classes)),
          'class_counts':{c:int(counts[c]) for c in classes},'n':len(df),
          'ranges':{c:[float(df[c].min()),float(df[c].max())] for c in BASE[1:]},
          'diagnostics':[{'순번':str(row.get('순번',int(idx)+1)),'엑셀 행':int(idx)+2,'함종':row['함분류'],
                          '함유형':str(row.get('함유형','')),'취역연도':float(row['취역연도']),
                          '함종평균연도':means[row['함분류']],'실제중량':float(row[target])}
                         for idx,row in df.iterrows()]}

    result,scales,p=solve(df,base,None)
    full_rank=len(df)>len(result.x) and np.linalg.matrix_rank(result.jac)==len(result.x)
    if not full_rank:
        gamma={c:0.0 for c in base['classes']};result,scales,p=solve(df,base,gamma)
        if len(df)<=len(result.x) or np.linalg.matrix_rank(result.jac)<len(result.x):
            raise ValueError('덧셈식의 계수를 식별할 자료가 부족하거나 변수가 중복됩니다.')
    else:gamma=dict(zip(base['classes'],result.x[p+len(base['classes'])-1:].tolist()))
    m=apply_result(base,result,scales,p,gamma)
    m.update(year_estimated=full_rank,year_unavailable_reason='' if full_rank else '덧셈식 연도 효과 식별 불가 · 0%만 계산합니다.',parameters=len(result.x),condition_number=float(np.linalg.cond(result.jac)))
    return m


def refit(df,base,q,diagnostics=True):
    from model import predict_one
    gamma={c:g*q for c,g in base['gamma'].items()}
    result,scales,p=solve(df,base,gamma)
    m=apply_result(base,result,scales,p,gamma)
    m.update(gamma_base=dict(base['gamma']),year_scale=q)
    pred=predict_one(df,m);actual=df[base['target']].to_numpy()
    if not np.isfinite(pred).all() or (pred<=0).any():raise ValueError('덧셈식 예측값이 유효하지 않습니다.')
    den=float(((actual-actual.mean())**2).sum())
    m.update(wape_percent=float(np.sum(abs(pred-actual))/np.sum(actual)*100),mape_percent=float(np.mean(abs((pred-actual)/actual))*100),rmse_tonne=float(np.sqrt(np.mean((pred-actual)**2))),r2=None if den==0 else float(1-((actual-pred)**2).sum()/den))
    if diagnostics:
        records=[];dts={c:1.0 if j==0 else float(np.exp(m['coef'][1+len(m['terms'])+j-1])) for j,c in enumerate(m['classes'])}
        for j,(_,row) in enumerate(df.iterrows()):
            cls=row['함분류'];d=dict(base['diagnostics'][j])
            d.update({'γt':gamma[cls],'연도보정값':float(np.exp(gamma[cls]*(row['취역연도']-m['mean_years'][cls])/10)),'Dt':dts[cls],
                      '예측중량':float(pred[j]),'잔차':float(actual[j]-pred[j]),'오차율(%)':float((pred[j]-actual[j])/actual[j]*100)})
            records.append(d)
        m['diagnostics']=records
    return m


def predict(df,m):
    classes=m['classes'];unknown=set(df['함분류'])-set(classes)
    if unknown:raise ValueError('학습되지 않은 함종: '+', '.join(sorted(unknown)))
    x=raw_values(df,m['terms']);weights=np.array([m['additive_coeff'][t] for t in ['constant']+m['terms']])
    logs=logsumexp(np.log(x)+np.log(weights),axis=1)
    dt={c:0.0 if j==0 else m['coef'][1+len(m['terms'])+j-1] for j,c in enumerate(classes)}
    year=df['함분류'].map(m['gamma']).to_numpy()*(df['취역연도']-df['함분류'].map(m['mean_years'])).to_numpy()/10
    with np.errstate(over='raise',invalid='raise'):return np.exp(logs+df['함분류'].map(dt).to_numpy()+year)*m['smearing']


def verify_engine(path):
    """Frozen-build smoke test: exercise SciPy optimization and prediction."""
    import pandas as pd
    from pathlib import Path
    rng=np.random.default_rng(42);n=30
    df=pd.DataFrame({'함분류':['호위함']*n,'취역연도':rng.integers(1980,2020,n),'길이':rng.uniform(80,150,n),
                     '폭':rng.uniform(10,20,n),'깊이':rng.uniform(6,12,n),'소요마력':rng.uniform(10000,50000,n),
                     '소요전력':rng.uniform(1000,9000,n),'순항속력':rng.uniform(12,20,n)})
    df['SWBS200']=200+.02*df['소요마력']
    base=fit_baseline(df,'SWBS200',['P']);m=refit(df,base,0)
    np.testing.assert_allclose(predict(df,m),df['SWBS200'],rtol=1e-6)
    Path(path).write_text('additive regression engine verified',encoding='utf-8')
