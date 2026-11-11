"""Separate patrol-vessel calibration without removing other ship types' year effects."""
import numpy as np
PATROL='초계함'


def fit(df,target,terms,hull=False,additive=False):
    from model import fit_one
    other=df.loc[df['함분류']!=PATROL]
    if other.empty:raise ValueError('초계함 추세를 공유할 다른 함종 실적이 필요합니다.')
    if additive:
        from additive import fit_baseline
        core=fit_baseline(other,target,terms)
    else:core=fit_one(other,target,terms,hull_by_type=hull)
    return attach(df,core,core,1.0)


def attach(df,core,baseline,q,diagnostics=True):
    from model import predict_one
    rows=df.loc[df['함분류']==PATROL];n=len(rows)
    mean=float(rows['취역연도'].mean());centered=(rows['취역연도'].to_numpy()-mean)/10
    donor='호위함' if '호위함' in core['classes'] else core['classes'][0]
    proxy=rows.copy();proxy['함분류']=donor;proxy['취역연도']=core['mean_years'][donor]
    # predict_one includes C: the patrol Dt is calibrated after C, not before it.
    offset=np.log(predict_one(proxy,core));y=np.log(rows[core['target']].to_numpy())-offset
    is_hull=core.get('size_by_type',False)
    z=np.log((rows['길이']*rows['폭']*rows['깊이']).to_numpy()/core['volume_reference']) if is_hull else np.zeros(n)
    independent_size=is_hull and n>=2 and np.ptp(z)>1e-12
    sizecols=[np.ones(n)]+([z] if independent_size else [])
    size=np.column_stack(sizecols)
    full=np.column_stack([size,centered])
    can_year=n>=2 and np.ptp(centered)>0 and np.linalg.matrix_rank(full,tol=np.linalg.norm(full,2)*1e-10)==full.shape[1] and n>=full.shape[1]
    # At baseline estimate the patrol coefficients from its own rows when identifiable.
    if baseline.get('_patrol_core') is None:
        if can_year:
            free=np.linalg.lstsq(full,y,rcond=None)[0];base_gamma=float(free[-1])
        else:base_gamma=0.0
    else:
        base_gamma=baseline['patrol_info']['base_gamma'];can_year=baseline['patrol_info']['year_independent']
    gamma=base_gamma*q
    beta=np.linalg.lstsq(size,y-gamma*centered,rcond=None)[0]
    own_alpha=core['size_alpha'][donor]+float(beta[1]) if independent_size else (core['size_alpha'][donor] if is_hull else None)
    notes=[f'초계함 {n}척: 전용 Dt를 실적 중량 기준으로 추정(C 포함 보정).']
    if is_hull:notes.append('크기 지수: 초계함 독립 추정.' if independent_size else f'크기 지수: {donor} 공유(독립 추정 자료 부족).')
    else:notes.append('크기·출력 계수: 다른 함종과 공통 추세 공유.')
    notes.append('연도계수: 초계함 독립 추정 후 선택 연도 비율 적용.' if can_year else '연도계수: 0 고정(한 척 또는 크기·연도 분리 불가).')
    notes.append('포화 적합은 신규 함정의 예측 정확도를 검증하지 않습니다.')
    info={'count':n,'donor':donor,'size_independent':bool(independent_size),'year_independent':bool(can_year),
          'base_gamma':base_gamma,'note':' '.join(notes)}
    donor_log_dt=0.0 if donor==core['classes'][0] else core['coef'][1+len(core['terms'])+core['classes'][1:].index(donor)]
    m={**core,'classes':core['classes']+[PATROL],'coef':list(core['coef'])+[float(beta[0])+donor_log_dt],
       'mean_years':{**core['mean_years'],PATROL:mean},'gamma':{**core['gamma'],PATROL:gamma},
       'class_counts':{**core['class_counts'],PATROL:n},'n':len(df),
       'parameters':core['parameters']+1+int(independent_size)+int(can_year),
       'patrol_info':info,'_patrol_core':baseline.get('_patrol_core',baseline),
       'year_estimated':core.get('year_estimated',True) or can_year,
       'year_unavailable_reason':core.get('year_unavailable_reason','').replace('0%만 계산합니다.','해당 비초계함 연도계수는 0으로 적용합니다.'),
       'ranges':{c:[float(df[c].min()),float(df[c].max())] for c in core['ranges']}}
    if is_hull:m['size_alpha']={**core['size_alpha'],PATROL:own_alpha}
    if 'gamma_base' in core:m['gamma_base']={**core['gamma_base'],PATROL:base_gamma}
    prediction=predict_one(df,m);actual=df[m['target']].to_numpy()
    den=float(((actual-actual.mean())**2).sum())
    m.update(mape_percent=float(np.mean(abs((prediction-actual)/actual))*100),wape_percent=float(np.sum(abs(prediction-actual))/np.sum(actual)*100),
             rmse_tonne=float(np.sqrt(np.mean((prediction-actual)**2))),r2=None if den==0 else float(1-((actual-prediction)**2).sum()/den))
    if not diagnostics:return m
    previous={r['엑셀 행']:r for r in core['diagnostics']};records=[]
    for j,(idx,row) in enumerate(df.iterrows()):
        cls=row['함분류'];record=dict(previous.get(int(idx)+2,{}))
        dt=1.0 if cls==m['classes'][0] else float(np.exp(m['coef'][1+len(m['terms'])+m['classes'][1:].index(cls)]))
        record.update({'순번':str(row.get('순번',int(idx)+1)),'엑셀 행':int(idx)+2,'함종':cls,'함유형':str(row.get('함유형','')),
                       '취역연도':float(row['취역연도']),'함종평균연도':m['mean_years'][cls],'γt':m['gamma'][cls],
                       '연도보정값':float(np.exp(m['gamma'][cls]*(row['취역연도']-m['mean_years'][cls])/10)),'Dt':dt,
                       '실제중량':float(actual[j]),'예측중량':float(prediction[j]),'잔차':float(actual[j]-prediction[j]),'오차율(%)':float((prediction[j]-actual[j])/actual[j]*100)})
        if is_hull:record.update({'αt':m['size_alpha'][cls],'기준LBD':m['volume_reference']})
        records.append(record)
    m['diagnostics']=records
    return m


def refit(df,base,q,diagnostics=True):
    from year_search import refit_scale
    other=df.loc[df['함분류']!=PATROL]
    core=refit_scale(other,base['_patrol_core'],q,diagnostics)
    m=attach(df,core,base,q,diagnostics)
    m.update(year_scale=q,gamma_base={**core['gamma_base'],PATROL:base['patrol_info']['base_gamma']})
    return m
