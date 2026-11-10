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
        self.assertEqual([x[0] for x in calls],list(range(1,1415)))
        self.assertEqual(len(b['formula_comparison']),14)
        self.assertEqual(len(b['formula_searches']),1414)
        out=predict(df,b)
        for name,m in b['models'].items():
            self.assertEqual(m['candidate_id'],'recommended',name)
            self.assertLess(m['mape_percent'],1e-8)
            expected=df['총중량']-df['SWBS700'] if name==TOTAL_MODEL else df[name]
            np.testing.assert_allclose(out[name+'_예측_tonne'],expected,rtol=1e-9)
            rows=[r for r in b['formula_comparison'] if r['계열']==name]
            self.assertEqual(sum(r['선택'] for r in rows),1)
            self.assertAlmostEqual(m['mape_percent'],min(r['MAPE(%)'] for r in rows))
            self.assertTrue(render_png(final_equation(name,m,'구축함',df.iloc[0].to_dict())).startswith(b'\x89PNG'))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model.xlsx';save_model_excel(p,b);loaded=load_model_excel(p)
            self.assertEqual(len(loaded['formula_comparison']),14)
            self.assertEqual(len(loaded['formula_searches']),1414)
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
        self.assertTrue(rows[1]['선택']);self.assertEqual(b['models']['SWBS100']['candidate_id'],'recommended')
