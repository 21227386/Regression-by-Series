import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from model import train, design, RECOMMENDED_TERMS, TOTAL_MODEL, TOTAL_TARGET, predict, save_model_excel, load_model_excel
from final_equations import final_equation,render_png
from test_model import fleet

class ComparisonTests(unittest.TestCase):
    def recommended_fleet(self):
        df=fleet();means=df.groupby('함분류')['취역연도'].mean().to_dict()
        for name,terms in RECOMMENDED_TERMS.items():
            x=design(df,terms,['구축함','호위함'],means,True)
            beta=[1.2]+[.35,1.1,.08][:len(terms)]+[.25,.15,-.1]
            target=np.exp(x@np.array(beta))
            if name==TOTAL_MODEL:df['총중량']=target+df['SWBS700']
            else:df[name]=target
        return df
    def test_recommended_selection_predictions_and_excel(self):
        df=self.recommended_fleet();calls=[];b=train(df,progress=lambda *x:calls.append(x))
        self.assertEqual(calls[-1][0],2121)
        self.assertTrue(all(a[0]<b[0] for a,b in zip(calls,calls[1:])))
        self.assertEqual(len(b['formula_comparison']),21)
        self.assertGreaterEqual(len(b['formula_searches']),1414)
        out=predict(df,b)
        for name,m in b['models'].items():
            self.assertEqual(m['candidate_id'],'recommended',name)
            self.assertLess(m['mape_percent'],1e-8)
            expected=df['총중량']-df['SWBS700'] if name==TOTAL_MODEL else df[name]
            np.testing.assert_allclose(out[name+'_예측_tonne'],expected,rtol=1e-9)
            rows=[r for r in b['formula_comparison'] if r['계열']==name]
            self.assertEqual(sum(r['선택'] for r in rows),1)
            self.assertAlmostEqual(m['mape_percent'],min(r['MAPE(%)'] for r in rows if r['MAPE(%)'] is not None))
            self.assertTrue(render_png(final_equation(name,m,'구축함',df.iloc[0].to_dict())).startswith(b'\x89PNG'))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model.xlsx';save_model_excel(p,b);loaded=load_model_excel(p)
            self.assertEqual(len(loaded['formula_comparison']),0)
            self.assertEqual(len(loaded['formula_searches']),0)
            for name in b['models']:
                self.assertEqual(loaded['models'][name]['candidate_id'],'recommended')
                np.testing.assert_allclose(predict(df,loaded)[name+'_예측_tonne'],out[name+'_예측_tonne'],rtol=1e-12)
    def test_current_wins_when_current_fits_better(self):
        df=fleet();v=df['길이']*df['폭']*df['깊이']
        df['SWBS100']=[1000*(vv/20000)**({'구축함':.2,'호위함':1.2}[c]) for vv,c in zip(v,df['함분류'])]
        b=train(df)
        self.assertEqual(b['models']['SWBS100']['candidate_id'],'current')
        self.assertLess(b['models']['SWBS100']['mape_percent'],1e-8)
    def test_one_unidentifiable_candidate_does_not_block_other(self):
        df=self.recommended_fleet();mask=df['함분류']=='호위함'
        df.loc[mask,'깊이']=15000/(df.loc[mask,'길이']*df.loc[mask,'폭'])
        b=train(df);rows=[r for r in b['formula_comparison'] if r['계열']=='SWBS100']
        self.assertIsNone(rows[0]['MAPE(%)']);self.assertTrue(rows[0]['사유'])
        self.assertTrue(rows[1]['선택']);self.assertIn(b['models']['SWBS100']['candidate_id'],['recommended','additional'])

    def test_additive_selection_years_prediction_and_roundtrip(self):
        from model import ADDITIONAL_TERMS,fit_one,predict_one
        from additive import fit_baseline,refit,raw_values
        from year_search import select_year_scale
        df=fleet();means=df.groupby('함분류')['취역연도'].mean().to_dict()
        for name in ['SWBS200','SWBS300','SWBS500','SWBS600',TOTAL_MODEL]:
            terms=ADDITIONAL_TERMS[name];raw=raw_values(df,terms);scales=raw.mean(axis=0)
            weight=(raw/scales)@np.arange(100,100+raw.shape[1]*50,50)
            year=np.array([.12 if c=='구축함' else -.08 for c in df['함분류']])*(df['취역연도']-df['함분류'].map(means)).to_numpy()/10
            weight*=np.exp(year)*np.where(df['함분류']=='구축함',1,1.3)
            target=TOTAL_TARGET if name==TOTAL_MODEL else name
            df[target]=weight
            base=fit_baseline(df,target,terms);m=select_year_scale(df,base)
            self.assertEqual(m['year_scale'],1);self.assertLess(m['mape_percent'],1e-5)
            np.testing.assert_allclose(predict_one(df,m),weight,rtol=1e-6)
            half=refit(df,base,.5);self.assertGreater(half['mape_percent'],m['mape_percent'])
            self.assertTrue(all(v>0 for v in m['additive_coeff'].values()))
        df['총중량']=df[TOTAL_TARGET]+df['SWBS700']
        b=train(df)
        for name in ['SWBS200','SWBS300','SWBS500','SWBS600',TOTAL_MODEL]:
            self.assertEqual(b['models'][name]['candidate_id'],'additional',name)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'add.xlsx';save_model_excel(p,b);loaded=load_model_excel(p)
            for name,m in b['models'].items():
                np.testing.assert_allclose(predict(df,b)[name+'_예측_tonne'],predict(df,loaded)[name+'_예측_tonne'],rtol=1e-12)
                if m.get('kind')=='additive':
                    eq=final_equation(name,m,'구축함',df.iloc[0].to_dict())
                    self.assertIn('+',eq);self.assertTrue(render_png(eq).startswith(b'\x89PNG'))

    def test_additive_raw_features_can_be_identified_when_logs_are_collinear(self):
        from additive import fit_baseline,refit
        df=fleet();df['소요전력']=df['길이']**2
        df['SWBS300']=100+.01*df['소요전력']+2*df['길이']
        m=refit(df,fit_baseline(df,'SWBS300',['PE','L']),0)
        self.assertLess(m['mape_percent'],1e-5)
