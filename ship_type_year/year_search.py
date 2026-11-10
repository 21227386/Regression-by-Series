"""Fold-local baseline estimation and constrained year-effect scale search."""
import numpy as np
SCALES=tuple(i/100 for i in range(0,11))

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

def validation_splits(df,mode='ship'):
    rng=np.random.default_rng(42);tests=[]
    if mode=='group':
        if '함급' not in df or df['함급'].isna().any() or df['함급'].astype(str).str.strip().eq('').any():return [],'함급별 검증에는 함급 열이 필요합니다. DDG/FFG 함유형은 사용하지 않습니다.'
        groups=df['함급'].astype(str).str.strip().to_numpy();labels=np.unique(groups);rng.shuffle(labels)
        if len(labels)<2:return [],'함급이 2개 이상 필요합니다.'
        tests=[np.flatnonzero(np.isin(groups,part)) for part in np.array_split(labels,min(5,len(labels)))]
    elif mode=='ship':
        parts=[[] for _ in range(5)]
        for cls in sorted(df['함분류'].unique()):
            positions=np.flatnonzero(df['함분류'].to_numpy()==cls)
            if len(positions)<3:continue
            rng.shuffle(positions)
            for k,part in enumerate(np.array_split(positions,min(5,len(positions)))):parts[k].extend(part.tolist())
        tests=[np.array(sorted(p),dtype=int) for p in parts if p]
    else:raise ValueError('검증 방식이 잘못되었습니다.')
    return [(np.setdiff1d(np.arange(len(df)),t),t) for t in tests],''

def select_year_scale(df,baseline,validation_mode='ship'):
    from model import fit_one,predict_one
    if not baseline['year_by_type']:return {**baseline,'year_scale':None,'gamma_base':{},'year_search':[],'selection':{'status':'연도 항 없음','basis':validation_mode,'validation_n':0,'folds':0,'reason':''},'cv_predictions':[]}
    splits,reason=validation_splits(df,validation_mode);folds=[];failures=[]
    for k,(tr,te) in enumerate(splits):
        training,testing=df.iloc[tr],df.iloc[te]
        try:
            b=fit_one(training,baseline['target'],baseline['terms'])
            if not set(testing['함분류'])<=set(b['classes']):raise ValueError('검증 함종의 학습 자료 없음')
            folds.append((k+1,training,testing,b))
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as e:failures.append(f'{k+1}차: {e}')
    n=sum(len(te) for _,_,te,_ in folds);usable=len(folds)>=2 and n>=3
    reason=reason or ' / '.join(failures[:3])
    if n<len(df):reason=(reason+' / ' if reason else '')+f'자료 부족 또는 분할 실패로 {len(df)-n}척은 검증하지 못했습니다.'
    if not usable:reason=reason or '유효 검증이 2회 이상·3척 이상이어야 합니다. 0% 임시 적용.'
    rows=[];candidates={};cv={}
    for q in SCALES:
        r={'percent':int(round(q*100)),'train_mape':None,'train_rmse':None,'validation_mape':None,'validation_rmse':None,'validation_n':n,'folds':len(folds),'selected':False,'error':''}
        try:
            m=refit_scale(df,baseline,q);candidates[q]=m;r.update(train_mape=m['mape_percent'],train_rmse=m['rmse_tonne'])
            if usable:
                actual=[];predicted=[];details=[]
                for fold,training,testing,b in folds:
                    trial=refit_scale(training,b,q,False);prediction=predict_one(testing,trial)
                    actual.extend(testing[b['target']].to_numpy());predicted.extend(prediction)
                    for j,(idx,row) in enumerate(testing.iterrows()):details.append({'fold':fold,'순번':str(row.get('순번',int(idx)+1)),'엑셀 행':int(idx)+2,'함종':row['함분류'],'함급':str(row.get('함급','')),'실제중량':float(row[b['target']]),'예측중량':float(prediction[j])})
                scores=metrics(np.asarray(actual),np.asarray(predicted));r.update(validation_mape=scores['mape'],validation_rmse=scores['rmse']);cv[q]=details
        except (ValueError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as e:r['error']=str(e)
        rows.append(r)
    valid=[r for r in rows if r['validation_mape'] is not None and not r['error']]
    if valid:
        best=min(valid,key=lambda r:(round(r['validation_mape'],8),round(r['validation_rmse'],8),r['percent']));q=best['percent']/100
        status='검증 선택' if n==len(df) else '부분 검증 선택'
    else:q=0.0;status='검증 불가 · 0% 임시 적용';reason=reason or '모든 후보의 검증 계산 실패'
    result=candidates.get(q) or refit_scale(df,baseline,q)
    for r in rows:r['selected']=r['percent']==int(round(q*100))
    best=next(r for r in rows if r['selected'])
    result.update(year_search=rows,selection={'status':status,'basis':validation_mode,'validation_n':n,'total_n':len(df),'folds':len(folds),'attempted_folds':len(splits),'reason':reason,'validation_mape':best['validation_mape'],'validation_rmse':best['validation_rmse']},cv_predictions=cv.get(q,[]))
    return result
