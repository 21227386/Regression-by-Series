import unittest
import tempfile
from pathlib import Path
import numpy as np
from openpyxl import load_workbook
from model import train,predict_one,diagnostic_metrics,diagnostic_rows
from additive import fit_baseline
from year_search import select_year_scale
from results_export import result_tables,save_regression_results,numerical_expression,SHEETS
from test_patrol import with_patrol

class ResultsExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df=with_patrol(1);cls.df['SWBS200']=200+.02*cls.df['소요마력']
        cls.bundle=train(cls.df,compare=False,wape_limit_percent=None)
        cls.bundle['models']['SWBS200']=select_year_scale(cls.df,fit_baseline(cls.df,'SWBS200',['P']))
    def test_compact_tables_match_graphs_all_and_per_type(self):
        tables=result_tables(self.bundle)
        self.assertEqual(list(tables),SHEETS)
        actual=tables['중량 비교']
        self.assertEqual(len(actual),len(self.df)*7)
        self.assertEqual(set(actual['함종']),set(self.df['함분류']))
        for name,m in self.bundle['models'].items():
            for d in m['diagnostics']:
                label='경하중량 직접회귀(700 제외)' if name.startswith('경하중량') else name
                row=actual[(actual['계열']==label)&(actual['순번']==d['순번'])].iloc[0]
                self.assertEqual(row['실측중량(tonne)'],d['실제중량'])
                self.assertEqual(row['추정중량(tonne)'],d['예측중량'])
                self.assertAlmostEqual(row['잔차(실측-추정, tonne)'],d['잔차'])
                self.assertEqual(row['오차율(추정-실측, %)'],d['오차율(%)'])
            for cls in [None]+m['classes']:
                label='경하중량 직접회귀(700 제외)' if name.startswith('경하중량') else name
                row=tables['적합도 요약'].query('계열 == @label')
                row=row[row['함종']==('전체' if cls is None else cls)].iloc[0]
                expected=diagnostic_metrics(diagnostic_rows(m,cls))
                self.assertEqual(row['척수'],expected['n'])
                self.assertAlmostEqual(row['MAPE(%)'],expected['mape_percent'])
    def test_numerical_equations_reproduce_final_predictions(self):
        for name,m in self.bundle['models'].items():
            for cls in m['classes']:
                frame=self.df[self.df['함분류']==cls].iloc[[0]];r=frame.iloc[0]
                expression=numerical_expression(m,cls).replace('^','**')
                env={'EXP':np.exp,'L':r['길이'],'B':r['폭'],'D':r['깊이'],'P':r['소요마력'],'PE':r['소요전력'],'Y':r['취역연도']}
                # Evaluate our generated expression only, never workbook or user-provided code.
                value=eval(expression,{'__builtins__':{}},env)
                self.assertAlmostEqual(value/predict_one(frame,m)[0],1,places=12)
    def test_workbook_contains_only_results_and_numeric_coefficients(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'results.xlsx';save_regression_results(p,self.bundle);wb=load_workbook(p)
            self.assertEqual(wb.sheetnames,SHEETS)
            for ws in wb:
                self.assertTrue(ws.auto_filter.ref);self.assertEqual(ws.freeze_panes,'A2')
            self.assertEqual(wb['중량 비교'].cell(2,2).value,'001')
            self.assertTrue(all(isinstance(wb['계수'].cell(i,4).value,(int,float)) for i in range(2,wb['계수'].max_row+1)))
            self.assertFalse(any(c.data_type=='f' for ws in wb for row in ws for c in row))
            wb.close()
