import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from test_model import fleet
from model import fit_one,TERMS,predict_one,train,predict,save_model_excel,load_model_excel,TOTAL_MODEL,TOTAL_SUM_COLUMN,read_excel
from year_search import refit_scale


def with_patrol(n):
    df=fleet();p=df.iloc[:n].copy();p['함분류']='초계함';p['순번']=['P'+str(j+1) for j in range(n)]
    p.index=range(len(df),len(df)+n)
    return pd.concat([df,p])

class PatrolTests(unittest.TestCase):
    def test_single_patrol_exact_calibration_preserves_other_year_slopes(self):
        df=with_patrol(1);df['SWBS200']*=np.exp(np.random.default_rng(4).normal(0,.2,len(df)))
        base=fit_one(df,'SWBS200',TERMS[200]);other=fit_one(df.iloc[:-1],'SWBS200',TERMS[200])
        self.assertIn('초계함',base['classes']);self.assertEqual(base['gamma']['초계함'],0)
        self.assertTrue(base['year_estimated'])
        for c in other['classes']:self.assertAlmostEqual(base['gamma'][c],other['gamma'][c])
        self.assertGreater(base['smearing'],1)
        for q in [0,1,2]:
            m=refit_scale(df,base,q)
            self.assertAlmostEqual(predict_one(df.iloc[[-1]],m)[0],df.iloc[-1]['SWBS200'],places=8)
            np.testing.assert_allclose(predict_one(df.iloc[:-1],m),predict_one(df.iloc[:-1],refit_scale(df.iloc[:-1],other,q)),rtol=1e-12)
    def test_hull_one_two_three_and_collinear_years(self):
        for n in [1,2,5]:
            df=with_patrol(n);mask=df['함분류']=='초계함'
            df.loc[mask,'취역연도']=np.array([1991,2007,1999,2020,1985][:n])
            volume=(df.loc[mask,'길이']*df.loc[mask,'폭']*df.loc[mask,'깊이']).to_numpy()
            years=df.loc[mask,'취역연도'].to_numpy()
            df.loc[mask,'SWBS100']=1200*(volume/20000)**.45*np.exp(.2*(years-years.mean())/10)
            m=fit_one(df,'SWBS100',TERMS[100]);info=m['patrol_info']
            self.assertEqual(info['count'],n)
            self.assertEqual(info['size_independent'],n>=2)
            self.assertEqual(info['year_independent'],n>=3)
            if n==1:self.assertAlmostEqual(m['size_alpha']['초계함'],m['size_alpha']['호위함'])
            if n>=3:
                self.assertAlmostEqual(m['size_alpha']['초계함'],.45)
                self.assertAlmostEqual(m['gamma']['초계함'],.2)
        df=with_patrol(3);mask=df['함분류']=='초계함'
        v=(df.loc[mask,'길이']*df.loc[mask,'폭']*df.loc[mask,'깊이']).to_numpy()
        df['취역연도']=df['취역연도'].astype(float)
        df.loc[mask,'취역연도']=2000+np.log(v)*10
        m=fit_one(df,'SWBS100',TERMS[100])
        self.assertFalse(m['patrol_info']['year_independent']);self.assertNotEqual(m['gamma']['호위함'],0)
    def test_two_ship_year_coefficient_for_common_power_and_additive(self):
        from additive import fit_baseline
        df=with_patrol(2);mask=df['함분류']=='초계함';df.loc[mask,'취역연도']=[1990,2010]
        for additive in [False,True]:
            core=fit_baseline(df.loc[~mask],'SWBS200',['P']) if additive else fit_one(df.loc[~mask],'SWBS200',['P'])
            proxy=df.loc[mask].copy();proxy['함분류']='호위함';proxy['취역연도']=core['mean_years']['호위함']
            df.loc[mask,'SWBS200']=predict_one(proxy,core)*1.4*np.exp(.3*np.array([-1,1]))
            m=fit_baseline(df,'SWBS200',['P']) if additive else fit_one(df,'SWBS200',['P'])
            self.assertTrue(m['patrol_info']['year_independent'])
            self.assertAlmostEqual(m['gamma']['초계함'],.3)
            doubled=refit_scale(df,m,2)
            self.assertAlmostEqual(doubled['gamma']['초계함'],.6)
    def test_all_candidate_selection_excel_and_input_keep_patrol_separate(self):
        df=with_patrol(1);b=train(df)
        for m in b['models'].values():
            self.assertEqual(m['class_counts']['초계함'],1)
            self.assertEqual(m['class_counts']['호위함'],50)
            self.assertAlmostEqual(m['diagnostics'][-1]['오차율(%)'],0,places=8)
            self.assertEqual(m['gamma']['초계함'],0)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'fleet.xlsx';df.to_excel(p,index=False)
            self.assertEqual(read_excel(p).iloc[-1]['함분류'],'초계함')
            from additive import fit_baseline
            from year_search import select_year_scale
            b['models']['SWBS200']=select_year_scale(df,fit_baseline(df,'SWBS200',['P']))
            p=Path(tmp)/'model.xlsx';save_model_excel(p,b);loaded=load_model_excel(p)
            for name,m in b['models'].items():
                self.assertEqual(m['patrol_info'],loaded['models'][name]['patrol_info'])
            np.testing.assert_allclose(predict(df,b)[TOTAL_SUM_COLUMN],predict(df,loaded)[TOTAL_SUM_COLUMN],rtol=1e-12)
