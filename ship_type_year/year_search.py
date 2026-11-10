"""Full-fleet training-MAPE selection of year-effect scales."""
import numpy as np
SCALES=tuple(i/100 for i in range(0,101))

def metrics(actual,predicted):
    if not np.isfinite(predicted).all() or (predicted<=0).any():raise ValueError('유효하지 않은 예측값')
    return {'mape':float(np.mean(abs((predicted-actual)/actual))*100),'rmse':float(np.sqrt(np.mean((predicted-actual)**2)))}

def refit_scale(df,base,q,diagnostics=True):
    from model import design,predict_one
    x=design(df,base['terms'],base['classes'],base['mean_years'],False)
    gamma={c:g*q for c,g in base['gamma'].items()}
    offset=df['함분류'].map(gamma).to_numpy()*(df['취역연도']-df['함분류'].map(base['mean_years'])).to_numpy()/10
    y=np.log(df[base['target']].to_numpy())
    coef=np.linalg.lstsq(x,y-offset,rcond=None)[0];residual=y-x@coef-offset
    with np.errstate(over='raise',invalid='raise'):
        m={**base,'coef':coef.tolist(),'gamma':gamma,'gamma_base':dict(base['gamma']),'year_scale':q,'smearing':float(np.exp(residual).mean())}
        pred=predict_one(df,m)
    actual=df[base['target']].to_numpy();scores=metrics(actual,pred);den=float(((actual-actual.mean())**2).sum())
    m.update(mape_percent=scores['mape'],rmse_tonne=scores['rmse'],r2=None if den==0 else float(1-((actual-pred)**2).sum()/den))
    if diagnostics:
        records=[]
        for j,(_,row) in enumerate(df.iterrows()):
            cls=row['함분류'];record=dict(base['diagnostics'][j])
            with np.errstate(over='raise',invalid='raise'):
                factor=float(np.exp(gamma[cls]*(row['취역연도']-m['mean_years'][cls])/10))
                dt=1.0 if cls==m['classes'][0] else float(np.exp(coef[1+len(m['terms'])+m['classes'][1:].index(cls)]))
            record.update({'γt':gamma[cls],'연도보정값':factor,'Dt':dt,'예측중량':float(pred[j]),'잔차':float(actual[j]-pred[j]),'오차율(%)':float((pred[j]-actual[j])/actual[j]*100)})
            records.append(record)
        m['diagnostics']=records
    return m

def select_year_scale(df,baseline):
    if not baseline['year_by_type']:
        return {**baseline,'year_scale':None,'gamma_base':{},'year_search':[],
                'selection':{'status':'연도 항 없음','basis':'train','train_n':len(df),'reason':''}}
    rows=[]
    for q in SCALES:
        row={'percent':int(round(q*100)),'train_mape':None,'train_rmse':None,'train_n':len(df),'selected':False,'error':''}
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
    result.update(year_search=rows,selection={'status':'실적함 Train MAPE 최소 선택','basis':'train','train_n':len(df),'total_n':len(df),'reason':'',
                                            'train_mape':best['train_mape'],'train_rmse':best['train_rmse']})
    return result
