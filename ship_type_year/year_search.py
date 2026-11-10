"""Full-fleet training-MAPE selection of year-effect scales."""
import numpy as np
MAX_PERCENT=200
SCALES=tuple(i/100 for i in range(MAX_PERCENT+1))

def metrics(actual,predicted):
    if not np.isfinite(predicted).all() or (predicted<=0).any():raise ValueError('유효하지 않은 예측값')
    return {'mape':float(np.mean(abs((predicted-actual)/actual))*100),'rmse':float(np.sqrt(np.mean((predicted-actual)**2)))}

def refit_scale(df,base,q,diagnostics=True):
    if base.get('kind')=='additive':
        from additive import refit
        return refit(df,base,q,diagnostics)
    from model import design,predict_one
    x=design(df,base['terms'],base['classes'],base['mean_years'],False)
    if base.get('size_by_type'):
        z=np.log((df['길이']*df['폭']*df['깊이']).to_numpy()/base['volume_reference'])
        x=np.column_stack([np.ones(len(df))]+[(df['함분류']==c).to_numpy(dtype=float) for c in base['classes'][1:]]+[(df['함분류']==c).to_numpy(dtype=float)*z for c in base['classes']])
    gamma={c:g*q for c,g in base['gamma'].items()}
    offset=df['함분류'].map(gamma).to_numpy()*(df['취역연도']-df['함분류'].map(base['mean_years'])).to_numpy()/10
    y=np.log(df[base['target']].to_numpy())
    coef=np.linalg.lstsq(x,y-offset,rcond=None)[0];residual=y-x@coef-offset
    with np.errstate(over='raise',invalid='raise'):
        m={**base,'coef':coef.tolist(),'gamma':gamma,'gamma_base':dict(base['gamma']),'year_scale':q,'smearing':float(np.exp(residual).mean())}
        if base.get('size_by_type'):
            ntypes=len(base['classes'])
            m['size_alpha']={c:float(coef[ntypes+j]) for j,c in enumerate(base['classes'])}
            m['coef']=[float(coef[0]),0.0]+coef[1:ntypes].tolist()
        pred=predict_one(df,m)
    actual=df[base['target']].to_numpy();scores=metrics(actual,pred);den=float(((actual-actual.mean())**2).sum())
    m.update(mape_percent=scores['mape'],rmse_tonne=scores['rmse'],r2=None if den==0 else float(1-((actual-pred)**2).sum()/den))
    if diagnostics:
        records=[]
        for j,(_,row) in enumerate(df.iterrows()):
            cls=row['함분류'];record=dict(base['diagnostics'][j])
            with np.errstate(over='raise',invalid='raise'):
                factor=float(np.exp(gamma[cls]*(row['취역연도']-m['mean_years'][cls])/10))
                dt=1.0 if cls==m['classes'][0] else float(np.exp(m['coef'][1+len(m['terms'])+m['classes'][1:].index(cls)]))
            record.update({'γt':gamma[cls],'연도보정값':factor,'Dt':dt,'예측중량':float(pred[j]),'잔차':float(actual[j]-pred[j]),'오차율(%)':float((pred[j]-actual[j])/actual[j]*100)})
            if m.get('size_by_type'):record['αt']=m['size_alpha'][cls]
            records.append(record)
        m['diagnostics']=records
    return m

def select_year_scale(df,baseline,progress=None):
    if not baseline['year_by_type']:
        return {**baseline,'year_scale':None,'gamma_base':{},'year_search':[],
                'selection':{'status':'연도 항 없음','basis':'train','train_n':len(df),'reason':''}}
    rows=[]
    for i,q in enumerate(SCALES):
        if progress:progress(i+1,len(SCALES),int(round(q*100)))
        row={'percent':int(round(q*100)),'train_mape':None,'train_rmse':None,'train_n':len(df),'selected':False,'error':''}
        if q>0 and not baseline.get('year_estimated',True):
            row['error']=baseline.get('year_unavailable_reason','연도 효과 식별 불가');rows.append(row);continue
        try:
            candidate=refit_scale(df,baseline,q,False)
            row.update(train_mape=candidate['mape_percent'],train_rmse=candidate['rmse_tonne'])
        except (ValueError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as e:row['error']=str(e)
        rows.append(row)
    valid=[r for r in rows if r['train_mape'] is not None and not r['error']]
    if not valid:raise ValueError('모든 연도 효과 후보의 실적함 적합 계산이 실패했습니다.')
    best=min(valid,key=lambda r:(round(r['train_mape'],8),round(r['train_rmse'],8),r['percent']))
    q=best['percent']/100
    result=refit_scale(df,baseline,q,True)
    for r in rows:r['selected']=r['percent']==best['percent']
    result.update(year_search=rows,selection={'status':'실적함 Train MAPE 최소 선택' if baseline.get('year_estimated',True) else '연도 효과 식별 불가 · 0% 적용','basis':'train','train_n':len(df),'total_n':len(df),'reason':baseline.get('year_unavailable_reason',''),
                                            'train_mape':best['train_mape'],'train_rmse':best['train_rmse']})
    return result
