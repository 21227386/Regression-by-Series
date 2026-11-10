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
    def test_patrol_pooling_and_input_exclusions(self):
        df=fleet();df.loc[df['함분류']=='호위함','함분류']='초계정'
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'fleet.xlsx';df.to_excel(p,index=False);data=read_excel(p)
            b=train(data);self.assertIn('호위함',b['models']['SWBS200']['gamma'])
            self.assertNotIn('초계정',b['models']['SWBS200']['gamma'])
