import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from year_search import metrics,relaxed_wape_limit,constrained_candidate,NoFeasibleWAPE
from model import train,save_model_excel,load_model_excel,predict,TOTAL_SUM_COLUMN
from test_model import fleet

class WAPETests(unittest.TestCase):
    def test_absolute_weighted_error_not_cancelling_or_mape(self):
        m=metrics(np.array([100.,1000.]),np.array([150.,950.]))
        self.assertAlmostEqual(m['wape'],100*100/1100)
        self.assertAlmostEqual(m['mape'],27.5)
        from model import diagnostic_metrics
        d=diagnostic_metrics([{'실제중량':100,'예측중량':150},{'실제중량':1000,'예측중량':950}])
        self.assertAlmostEqual(d['wape_percent'],m['wape'])
    def test_relax_minimum_steps_across_all_candidates(self):
        self.assertEqual(relaxed_wape_limit([{'train_wape':10.2},{'train_wape':30}],10),11)
        self.assertEqual(relaxed_wape_limit([{'train_wape':10.0}],10),10)
        self.assertEqual(relaxed_wape_limit([{'train_wape':30},{'train_wape':9}],10),10)
        self.assertEqual(relaxed_wape_limit([{'train_wape':31.1}],10),32)
        with self.assertRaises(NoFeasibleWAPE):relaxed_wape_limit([{'train_wape':10.2}],10,False)
        with self.assertRaises(NoFeasibleWAPE):relaxed_wape_limit([{'train_wape':None,'error':'failed'}],10)
    def test_constraint_is_applied_to_each_year_candidate(self):
        rows=[{'percent':0,'train_mape':1.,'train_wape':12.,'train_rmse':1.,'error':''},
              {'percent':100,'train_mape':3.,'train_wape':9.,'train_rmse':3.,'error':''},
              {'percent':200,'train_mape':2.,'train_wape':10.,'train_rmse':2.,'error':''}]
        m={'year_search':rows,'selection':{},'candidate_id':'current','candidate_label':'existing','formula':'test'}
        with patch('year_search.refit_scale',return_value={'year_scale':2.}) as refit:
            chosen=constrained_candidate(None,{},m,10,10)
            self.assertEqual(refit.call_args.args[2],2.)
            self.assertEqual([r['percent'] for r in chosen['year_search'] if r['selected']],[200])
            self.assertFalse(chosen['year_search'][0]['wape_eligible'])
        self.assertIsNone(constrained_candidate(None,{},m,5,5))
    def test_real_training_relaxes_only_when_needed_and_excel_roundtrip(self):
        df=fleet();df['SWBS200']*=np.exp(np.random.default_rng(91).normal(0,.45,len(df)))
        b=train(df,wape_limit_percent=10)
        for name,m in b['models'].items():
            info=m['selection'];self.assertEqual(info['wape_requested_percent'],10)
            self.assertLessEqual(m['wape_percent'],info['wape_limit_percent']+1e-9)
            rows=[r for r in b['formula_searches'] if r['계열']==name]
            used=relaxed_wape_limit(rows,10)
            self.assertEqual(info['wape_limit_percent'],used)
            valid=[r for r in rows if r.get('train_wape') is not None and not r.get('error') and r['train_wape']<=used+1e-9]
            self.assertAlmostEqual(m['mape_percent'],min(r['train_mape'] for r in valid),places=7)
        self.assertTrue(b['models']['SWBS200']['selection']['wape_relaxed'])
        self.assertFalse(b['models']['SWBS100']['selection']['wape_relaxed'])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'wape.xlsx';save_model_excel(p,b);loaded=load_model_excel(p)
            for name,m in b['models'].items():
                self.assertAlmostEqual(loaded['models'][name]['wape_percent'],m['wape_percent'])
                self.assertEqual(loaded['models'][name]['selection']['wape_limit_percent'],m['selection']['wape_limit_percent'])
            np.testing.assert_allclose(predict(df,b)[TOTAL_SUM_COLUMN],predict(df,loaded)[TOTAL_SUM_COLUMN],rtol=1e-12)
