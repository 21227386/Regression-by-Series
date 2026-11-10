import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from model import BASE, TERMS, WEIGHTS, design, train, predict, template, read_excel, save_model_excel, load_model_excel

def fleet():
    rng=np.random.default_rng(73);n=80
    df=pd.DataFrame({'순번':[f'{i+1:03d}' for i in range(n)],'함분류':['구축함','호위함']*40,'함유형':['DDG','FFG']*40,
        '길이':rng.uniform(70,180,n),'폭':rng.uniform(10,25,n),'깊이':rng.uniform(6,15,n),
        '소요마력':rng.uniform(10000,80000,n),'소요전력':rng.uniform(1000,10000,n),'순항속력':rng.uniform(12,22,n)})
    for i,terms in TERMS.items():
        x=design(df,terms,['구축함','호위함']);b=np.array([1.5]+[.6]*len(terms)+[.3])
        df[f'SWBS{i}']=np.exp(x@b)
    df['총중량']=df[WEIGHTS].sum(axis=1)
    return df

class ModelTests(unittest.TestCase):
    def test_exact_recovery_and_total_sum(self):
        df=fleet();b=train(df);out=predict(df,b)
        for c in WEIGHTS:np.testing.assert_allclose(out[c+'_예측_tonne'],df[c],rtol=1e-9)
        np.testing.assert_allclose(out['총중량_SWBS합계_tonne'],df['총중량'],rtol=1e-9)
        self.assertIn('총중량_독립회귀_예측_tonne',out)
    def test_year_never_changes_fit_or_prediction(self):
        df=fleet();a=train(df)
        df['취역연도']=np.arange(len(df))+1950;b=train(df)
        df['취역연도']='?';c=train(df)
        for key,m in a['models'].items():
            self.assertNotIn('year',m['terms']);self.assertNotIn('mean_year',m)
            np.testing.assert_array_equal(m['coef'],b['models'][key]['coef'])
            np.testing.assert_array_equal(m['coef'],c['models'][key]['coef'])
        one=df.iloc[[0]].copy();two=one.copy();one['취역연도']=1980;two['취역연도']=2100
        for target in WEIGHTS+['총중량_독립회귀']:
            np.testing.assert_array_equal(predict(one,a)[target+'_예측_tonne'],predict(two,a)[target+'_예측_tonne'])
    def test_year_optional_in_excel_and_bad_data_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'fleet.xlsx';df=fleet().astype(object);df['취역연도']='?';df.loc[0,'길이']='?';df.to_excel(p,index=False)
            loaded=read_excel(p);self.assertEqual(len(loaded),79)
            self.assertEqual(loaded.attrs['excluded_rows'][0]['엑셀 행'],2)
    def test_model_roundtrip_including_empty_term_models(self):
        df=fleet();bundle=train(df)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model.xlsx';save_model_excel(p,bundle);restored=load_model_excel(p)
            self.assertEqual(restored['models']['SWBS400']['terms'],[])
            for target in WEIGHTS+['총중량_독립회귀']:
                np.testing.assert_allclose(predict(df,bundle)[target+'_예측_tonne'],predict(df,restored)[target+'_예측_tonne'],rtol=1e-12)
    def test_template_without_year(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'template.xlsx';template(p)
            headers=pd.read_excel(p).columns
            self.assertNotIn('취역연도(year)',headers);self.assertEqual(headers[0],'순번')
    def test_400_700_same_type_same_weight(self):
        df=fleet();bundle=train(df);out=predict(df,bundle)
        for c in ['SWBS400','SWBS700']:
            self.assertEqual(out.loc[df['함분류']=='호위함',c+'_예측_tonne'].nunique(),1)
    def test_power_and_unseen_type(self):
        df=fleet();b=train(df);a=df.iloc[[0]].copy();higher=a.copy();higher['소요마력']*=2
        self.assertGreater(predict(higher,b)['SWBS200_예측_tonne'].iloc[0],predict(a,b)['SWBS200_예측_tonne'].iloc[0])
        a['함분류']='지원함'
        with self.assertRaises(ValueError):predict(a,b)
