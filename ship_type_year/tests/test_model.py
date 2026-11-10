import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from model import TERMS, YEAR_TARGETS, WEIGHTS, design, train, predict, read_excel, save_model_excel, load_model_excel, coefficients, template, TOTAL_MODEL, TOTAL_TARGET, TOTAL_SUM_COLUMN

def fleet():
    rng=np.random.default_rng(67);n=100
    df=pd.DataFrame({'순번':[f'{i+1:03d}' for i in range(n)],'함분류':['구축함','호위함']*50,'함유형':['DDG','FFG']*50,
        '취역연도':np.array([1980+i%20 if i%2==0 else 2000+i%20 for i in range(n)]),
        '길이':rng.uniform(70,180,n),'폭':rng.uniform(10,25,n),'깊이':rng.uniform(6,15,n),
        '소요마력':rng.uniform(10000,80000,n),'소요전력':rng.uniform(1000,12000,n),'순항속력':rng.uniform(12,22,n)})
    means=df.groupby('함분류')['취역연도'].mean().to_dict()
    for i,terms in TERMS.items():
        flag=f'SWBS{i}' in YEAR_TARGETS
        x=design(df,terms,['구축함','호위함'],means,flag)
        coef=[1.2]+[.6]*len(terms)+[.3]+([.12,-.2] if flag else [])
        df[f'SWBS{i}']=np.exp(x@np.array(coef))
    x=design(df,['LBD','P'],['구축함','호위함'],means,True)
    df['SWBS700']=np.linspace(100,500,n)
    df['총중량']=np.exp(x@np.array([1.4,.7,.1,.3,.08,-.11]))+df['SWBS700']
    return df

class ModelTests(unittest.TestCase):
    def test_recover_class_slopes_and_means(self):
        df=fleet();b=train(df)
        for name,m in b['models'].items():
            if m['year_by_type']:
                expected=[.08,-.11] if name==TOTAL_MODEL else [.12,-.2]
                np.testing.assert_allclose(list(m['gamma'].values()),expected,atol=1e-10)
            self.assertEqual(m['mean_years'],df.groupby('함분류')['취역연도'].mean().to_dict())
            self.assertNotIn('mean_year',m)
        out=predict(df,b)
        for c in WEIGHTS:np.testing.assert_allclose(out[c+'_예측_tonne'],df[c],rtol=1e-9)
        np.testing.assert_allclose(out[TOTAL_MODEL+'_예측_tonne'],df['총중량']-df['SWBS700'],rtol=1e-9)
        np.testing.assert_allclose(out[TOTAL_SUM_COLUMN],out[[c+'_예측_tonne' for c in WEIGHTS]].sum(axis=1))
    def test_class_specific_ten_year_change_and_centering(self):
        df=fleet();b=train(df)
        for cls in ['구축함','호위함']:
            row=df.iloc[[0]].copy();row['함분류']=cls
            row['취역연도']=b['models']['SWBS200']['mean_years'][cls]
            later=row.copy();later['취역연도']+=10
            first=predict(row,b).iloc[0];second=predict(later,b).iloc[0]
            for name,m in b['models'].items():
                ratio=np.exp(m['gamma'][cls]) if m['year_by_type'] else 1
                self.assertAlmostEqual(second[name+'_예측_tonne']/first[name+'_예측_tonne'],ratio)
        diagnostics=b['models']['SWBS200']['diagnostics']
        for row in diagnostics:
            cls=row['함종'];m=b['models']['SWBS200']
            self.assertAlmostEqual(row['연도보정값'],np.exp(m['gamma'][cls]*(row['취역연도']-m['mean_years'][cls])/10))
    def test_700_not_fitted_and_total_subtracted_exactly_once(self):
        df=fleet();bundle=train(df);out=predict(df,bundle)
        self.assertNotIn('SWBS700',bundle['models'])
        self.assertNotIn('SWBS700_예측_tonne',out)
        self.assertEqual(len(bundle['models']),7)
        actual=np.array([r['실제중량'] for r in bundle['models'][TOTAL_MODEL]['diagnostics']])
        np.testing.assert_allclose(actual,df['총중량']-df['SWBS700'])
        changed=df.copy();changed['SWBS700']+=200;changed['총중량']+=200
        # Net-weight targets are unchanged when full weight and actual 700 rise equally.
        revised=train(changed)
        np.testing.assert_allclose(bundle['models'][TOTAL_MODEL]['coef'],revised['models'][TOTAL_MODEL]['coef'],rtol=1e-10)
        self.assertTrue(out['중량 범위'].eq('SWBS700 제외 중량').all())
    def test_zero_700_and_nonpositive_net_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'zero.xlsx';df=fleet();df.loc[0,'SWBS700']=0;df.loc[1,'SWBS700']=df.loc[1,'총중량']
            df.to_excel(p,index=False);valid=read_excel(p)
            self.assertIn(0,valid.index);self.assertNotIn(1,valid.index)
            self.assertEqual(valid.attrs['excluded_rows'][0]['엑셀 행'],3)
    def test_template_preserves_actual_700_for_deduction(self):
        from openpyxl import load_workbook
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'template.xlsx';template(p)
            wb=load_workbook(p);headers=[c.value for c in wb['데이터'][1]]
            self.assertIn('SWBS700(tonne)',headers);self.assertEqual(headers[-1],'총중량(tonne)')
            comment=wb['데이터'].cell(1,headers.index('SWBS700(tonne)')+1).comment.text
            self.assertIn('공제용',comment)
            self.assertIn('원래 경하중량',wb['데이터'].cell(1,len(headers)).comment.text)
            wb.close()
    def test_scale_refits_other_coefficients_and_C(self):
        from model import fit_one, predict_one
        from year_search import refit_scale
        df=fleet();base=fit_one(df,'SWBS200',TERMS[200]);half=refit_scale(df,base,.5)
        for cls in base['classes']:self.assertAlmostEqual(half['gamma'][cls],base['gamma'][cls]*.5)
        self.assertGreater(np.linalg.norm(np.array(base['coef'])-half['coef']),1e-7)
        # Smearing is recomputed from the newly fitted residuals.
        prediction=predict_one(df,half)/half['smearing']
        residual=np.log(df['SWBS200'].to_numpy())-np.log(prediction)
        self.assertAlmostEqual(half['smearing'],float(np.exp(residual).mean()))
    def test_candidate_selection_and_fold_local_fitting(self):
        from model import fit_one
        from year_search import select_year_scale
        from unittest.mock import patch
        df=fleet();rng=np.random.default_rng(18)
        df['SWBS200']*=np.exp(rng.normal(0,.22,len(df)))
        base=fit_one(df,'SWBS200',TERMS[200]);seen=[]
        original=fit_one
        def capture(training,*args):
            m=original(training,*args);seen.append((set(training.index),m));return m
        with patch('model.fit_one',side_effect=capture):chosen=select_year_scale(df,base,'ship')
        self.assertEqual(len(chosen['year_search']),10)
        self.assertEqual([r['percent'] for r in chosen['year_search']],list(range(100,0,-10)))
        self.assertEqual(chosen['selection']['validation_n'],100)
        self.assertTrue(all(len(indices)<100 for indices,m in seen))
        for indices,m in seen:self.assertEqual(m['mean_years'],df.loc[list(indices)].groupby('함분류')['취역연도'].mean().to_dict())
        rows=chosen['year_search'];winner=min(rows,key=lambda r:(round(r['validation_mape'],8),round(r['validation_rmse'],8),-r['percent']))
        self.assertEqual(chosen['year_scale'],winner['percent']/100)
        self.assertEqual(len(chosen['cv_predictions']),100)
        for r in chosen['cv_predictions']:self.assertNotIn(int(r['엑셀 행'])-2,seen[int(r['fold'])-1][0])
    def test_group_integrity_and_unavailable_validation(self):
        from model import fit_one
        from year_search import select_year_scale,validation_splits
        df=fleet();base=fit_one(df,'SWBS200',TERMS[200])
        failed=select_year_scale(df,base,'group')
        self.assertEqual(failed['year_scale'],1)
        self.assertIn('검증 불가',failed['selection']['status'])
        self.assertTrue(all(r['validation_mape'] is None for r in failed['year_search']))
        df['함급']=[f'class-{i//10}' for i in range(100)]
        splits,_=validation_splits(df,'group')
        for tr,te in splits:self.assertFalse(set(df.iloc[tr]['함급']) & set(df.iloc[te]['함급']))
    def test_type_specific_hull_slopes_and_equal_weight_pair(self):
        from model import fit_one, predict_one
        df=fleet()
        volume=df['길이']*df['폭']*df['깊이']
        ref=float(np.exp(np.log(volume).mean()))
        expected={'구축함':0.0,'호위함':.9}
        df['SWBS100']=[1318.8*(v/ref)**expected[c] for v,c in zip(volume,df['함분류'])]
        m=fit_one(df,'SWBS100',TERMS[100])
        self.assertTrue(m['size_by_type']);self.assertEqual(m['parameters'],4)
        for cls,alpha in expected.items():self.assertAlmostEqual(m['size_alpha'][cls],alpha)
        self.assertAlmostEqual(m['volume_reference'],ref)
        pair=df.iloc[[0,0]].copy()
        pair['길이']=[114,103];pair['폭']=[17,15];pair['깊이']=[8.5,7.5]
        weights=predict_one(pair,m)
        np.testing.assert_allclose(weights,[1318.8,1318.8],rtol=1e-9)
        # Unlike the old common exponent, changing size can have a different effect for each type.
        for cls,alpha in expected.items():
            one=df.iloc[[0]].copy();one['함분류']=cls
            larger=one.copy();larger['길이']*=2
            self.assertAlmostEqual(predict_one(larger,m)[0]/predict_one(one,m)[0],2**alpha)
    def test_hull_size_variation_required(self):
        df=fleet();mask=df['함분류']=='호위함'
        df.loc[mask,['길이','폭','깊이']]=[100,15,8]
        with self.assertRaisesRegex(ValueError,'호위함'):train(df)
    def test_missing_year_variation_is_explicit(self):
        df=fleet();df.loc[df['함분류']=='호위함','취역연도']=2000
        with self.assertRaisesRegex(ValueError,'호위함'):train(df)
    def test_model_roundtrip_and_diagnostics(self):
        df=fleet();b=train(df)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model.xlsx';save_model_excel(p,b);loaded=load_model_excel(p)
            for name,m in b['models'].items():
                np.testing.assert_allclose(predict(df,b)[name+'_예측_tonne'],predict(df,loaded)[name+'_예측_tonne'],rtol=1e-12)
                self.assertEqual(m['mean_years'],loaded['models'][name]['mean_years'])
            self.assertEqual(loaded['models']['SWBS200']['diagnostics'][0]['순번'],'001')
            self.assertIn('γt',coefficients(b))
            self.assertEqual(loaded['models']['SWBS100']['size_alpha'],b['models']['SWBS100']['size_alpha'])
            self.assertAlmostEqual(loaded['models']['SWBS100']['volume_reference'],b['models']['SWBS100']['volume_reference'])
            self.assertIn('αt',coefficients(b))
            self.assertEqual(len(loaded['models']['SWBS200']['year_search']),10)
            self.assertEqual(loaded['models']['SWBS200']['year_scale'],b['models']['SWBS200']['year_scale'])
    def test_patrol_pooling_and_input_exclusions(self):
        df=fleet();df.loc[df['함분류']=='호위함','함분류']='초계정'
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'fleet.xlsx';df.to_excel(p,index=False);data=read_excel(p)
            b=train(data);self.assertIn('호위함',b['models']['SWBS200']['gamma'])
            self.assertNotIn('초계정',b['models']['SWBS200']['gamma'])
