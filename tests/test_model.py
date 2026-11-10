import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from model import BASE, WEIGHTS, TERMS, design, train, predict, read_excel, export

class ModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42); n = 120
        self.df = pd.DataFrame({'함분류':['A','B']*(n//2),'취역연도':rng.integers(1980,2025,n),
            '길이':rng.uniform(70,180,n),'폭':rng.uniform(10,25,n),'깊이':rng.uniform(6,15,n),
            '소요마력':rng.uniform(10000,80000,n),'소요전력':rng.uniform(1000,12000,n),'순항속력':rng.uniform(12,22,n)})
        for i,terms in TERMS.items():
            x=design(self.df,terms,['A','B'],self.df['취역연도'].mean())
            b=np.array([1.2]+[.15 if t=='year' else .65 for t in terms]+[.3])
            self.df[f'SWBS{i}']=np.exp(x@b)
        self.df['총중량']=self.df[WEIGHTS].sum(axis=1)
    def test_recover_and_roundtrip(self):
        b=train(self.df); out=predict(self.df,b)
        for c in WEIGHTS:
            np.testing.assert_allclose(out[c+'_예측_tonne'],self.df[c],rtol=1e-9)
        np.testing.assert_allclose(out['총중량_SWBS합계_tonne'],self.df['총중량'],rtol=1e-9)
        self.assertAlmostEqual(b['models']['SWBS100']['coef'][-1],.3)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'input.xlsx'; self.df.to_excel(p,index=False)
            loaded=read_excel(p); self.assertEqual(len(loaded),120)
            report=Path(tmp)/'report.xlsx'; export(report,loaded,b)
            with pd.ExcelFile(report) as workbook:
                self.assertIn('회귀계수',workbook.sheet_names)
    def test_new_data_and_saved_model(self):
        import json
        bundle=json.loads(json.dumps(train(self.df.iloc[:100])))
        out=predict(self.df.iloc[100:][BASE],bundle)
        for c in WEIGHTS:
            np.testing.assert_allclose(out[c+'_예측_tonne'],self.df.iloc[100:][c],rtol=1e-9)
    def test_unseen_class(self):
        b=train(self.df); data=self.df.iloc[:1].copy(); data['함분류']='NEW'
        with self.assertRaisesRegex(ValueError,'학습되지'): predict(data,b)
    def test_rank_deficient(self):
        self.df['취역연도']=2000
        with self.assertRaisesRegex(ValueError,'식별'): train(self.df)
    def test_zero_weight_rejected(self):
        self.df.loc[0,'SWBS100']=0
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.xlsx'; self.df.to_excel(p,index=False)
            with self.assertRaisesRegex(ValueError,'SWBS100'): read_excel(p)
    def test_insufficient(self):
        with self.assertRaisesRegex(ValueError,'부족'): train(self.df.iloc[:3])

if __name__=='__main__': unittest.main()
