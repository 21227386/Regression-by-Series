import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from model import BASE, WEIGHTS, TERMS, design, train, predict, read_excel, export

class ModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42); n = 120
        self.df = pd.DataFrame({'함분류':['구축함','호위함']*(n//2),'취역연도':rng.integers(1980,2025,n),
            '길이':rng.uniform(70,180,n),'폭':rng.uniform(10,25,n),'깊이':rng.uniform(6,15,n),
            '소요마력':rng.uniform(10000,80000,n),'소요전력':rng.uniform(1000,12000,n),'순항속력':rng.uniform(12,22,n)})
        for i,terms in TERMS.items():
            x=design(self.df,terms,['구축함','호위함'],self.df['취역연도'].mean())
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
    def test_total_power_coefficient_and_prediction(self):
        from model import coefficients
        data=self.df.copy()
        x=design(data,['LBD','P','year'],['구축함','호위함'],data['취역연도'].mean())
        expected=np.array([1.1,.6,.35,.08,.2])
        data['총중량']=np.exp(x@expected)
        bundle=train(data)
        m=bundle['models']['총중량_독립회귀']
        self.assertEqual(m['terms'],['LBD','P','year'])
        np.testing.assert_allclose(m['coef'],expected,rtol=1e-9)
        base=data.iloc[[0]][BASE].copy(); base['함분류']='호위함'
        faster=base.copy(); faster['소요마력']*=2
        a=predict(base,bundle); b=predict(faster,bundle)
        self.assertAlmostEqual(b['총중량_독립회귀_예측_tonne'].iloc[0]/a['총중량_독립회귀_예측_tonne'].iloc[0],2**.35)
        rows=coefficients(bundle)
        formula=rows.loc[rows['계열']=='총중량_독립회귀','식'].iloc[0]
        self.assertIn('P^β',formula)
    def test_unseen_class(self):
        b=train(self.df); data=self.df.iloc[:1].copy(); data['함분류']='지원함'
        with self.assertRaisesRegex(ValueError,'학습되지'): predict(data,b)
    def test_patrol_predictions_equal_frigate(self):
        bundle=train(self.df)
        data=pd.concat([self.df.iloc[[0]]]*3,ignore_index=True)
        data['함분류']=['호위함','초계함','초계정']
        out=predict(data,bundle)
        self.assertEqual(out['함분류'].tolist(),['호위함']*3)
        self.assertEqual(out['함분류_입력'].tolist(),['호위함','초계함','초계정'])
        for c in WEIGHTS:
            np.testing.assert_allclose(out[c+'_예측_tonne'],out[c+'_예측_tonne'].iloc[0])
    def test_patrol_training_pooled(self):
        expected=train(self.df)
        data=self.df.copy()
        data.loc[data.index % 4 == 1,'함분류']='초계함'
        data.loc[data.index % 4 == 3,'함분류']='초계정'
        actual=train(data)
        for c in WEIGHTS:
            self.assertEqual(actual['models'][c]['classes'],['구축함','호위함'])
            np.testing.assert_allclose(actual['models'][c]['coef'],expected['models'][c]['coef'])
    def test_invalid_type(self):
        self.df.loc[0,'함분류']='미지정함'
        with self.assertRaisesRegex(ValueError,'지원하지 않는 함종'): train(self.df)
    def test_six_types(self):
        from model import SHIP_TYPES
        data=self.df.copy(); data['함분류']=SHIP_TYPES*20
        bundle=train(data)
        self.assertEqual(set(bundle['models']['SWBS100']['classes']),set(SHIP_TYPES))
    def test_old_alias_model_rejected(self):
        bundle=train(self.df)
        bundle['models']['SWBS100']['classes']=['구축함','초계함']
        with self.assertRaisesRegex(ValueError,'다시 학습'): predict(self.df,bundle)
    def test_rank_deficient(self):
        self.df['취역연도']=2000
        with self.assertRaisesRegex(ValueError,'식별'): train(self.df)
    def test_zero_weight_rejected(self):
        self.df.loc[0,'SWBS100']=0
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.xlsx'; self.df.to_excel(p,index=False)
            loaded=read_excel(p)
            self.assertEqual(len(loaded),119)
            self.assertIn('SWBS100',loaded.attrs['excluded_rows'][0]['제외 사유'])
    def test_unit_headers_width_and_exclusions(self):
        from model import HEADERS, INPUT_COLUMNS, template
        from openpyxl import load_workbook
        data=self.df.copy()
        data.loc[2,'길이']=np.nan
        data.loc[4,'총중량']=np.nan
        data.loc[7,'함분류']='   '
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'units.xlsx'
            template(p)
            wb=load_workbook(p)
            self.assertEqual([c.value for c in wb['데이터'][1]],[HEADERS[c] for c in INPUT_COLUMNS])
            for ws in wb:
                for dim in ws.column_dimensions.values():self.assertEqual(dim.width,10)
                for cell in ws[1]:
                    self.assertEqual(cell.font.sz,10)
                    self.assertEqual(cell.alignment.horizontal,'center')
                    self.assertEqual(cell.alignment.vertical,'center')
            wb.close()
            data.rename(columns=HEADERS).to_excel(p,index=False)
            loaded=read_excel(p)
            self.assertEqual(len(loaded),117)
            self.assertEqual([r['엑셀 행'] for r in loaded.attrs['excluded_rows']],[4,6,9])
            bundle=train(loaded)
            self.assertEqual(bundle['models']['SWBS100']['n'],117)
            report=Path(tmp)/'out.xlsx';export(report,loaded,bundle)
            with pd.ExcelFile(report) as book:self.assertIn('제외 행',book.sheet_names)
    def test_empty_fleet_rejected_and_batch_does_not_require_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'empty_values.xlsx'
            data=self.df.copy();data['총중량']=np.nan;data.to_excel(p,index=False)
            with self.assertRaisesRegex(ValueError,'전체 120척'):read_excel(p)
            data=data[BASE];data.loc[0,'폭']=np.nan;data.to_excel(p,index=False)
            loaded=read_excel(p,training=False)
            self.assertEqual(len(loaded),119)
    def test_malformed_rows_excluded_with_reasons(self):
        data=self.df.astype(object)
        data.loc[0,'길이']='?'
        data.loc[1,'소요마력']='abc'
        data.loc[2,'함분류']='?'
        data.loc[3,'취역연도']=2020.5
        data.loc[4,'SWBS200']=-1
        data.loc[5,'소요전력']=float('inf')
        data.loc[6,'폭']=True
        data.loc[7,'SWBS100']=' '
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.xlsx';data.to_excel(p,index=False)
            loaded=read_excel(p)
            self.assertEqual(len(loaded),112)
            self.assertEqual([r['엑셀 행'] for r in loaded.attrs['excluded_rows']],list(range(2,10)))
            bundle=train(loaded)
            self.assertEqual(bundle['models']['SWBS100']['n'],112)
            b=read_excel(p,training=False)
            self.assertEqual(len(b),114) # Missing/invalid actual weights are irrelevant for prediction.
    def test_most_frequent_reference_and_reparameterization(self):
        data=self.df.drop(index=list(range(0,20,2))) # 50 destroyers, 60 frigates
        bundle=train(data)
        m=bundle['models']['SWBS100']
        self.assertEqual(m['classes'][0],'호위함')
        self.assertEqual(m['class_counts'],{'호위함':60,'구축함':50})
        self.assertAlmostEqual(m['coef'][0],1.5)
        self.assertAlmostEqual(m['coef'][-1],-.3)
        np.testing.assert_allclose(predict(data,bundle)['SWBS100_예측_tonne'],data['SWBS100'],rtol=1e-9)
        # Patrol ships count toward the frigate reference, including during Excel ingestion.
        data=self.df.copy();data.loc[data.index%3==0,'함분류']='초계정'
        pooled=train(data)
        self.assertEqual(pooled['models']['SWBS100']['classes'][0],'호위함')
    def test_excel_model_roundtrip_and_visible_parameters(self):
        from model import save_model_excel, load_model_excel
        from openpyxl import load_workbook
        bundle=train(self.df)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model.xlsx';save_model_excel(p,bundle)
            loaded=load_model_excel(p)
            a=predict(self.df,bundle);b=predict(self.df,loaded)
            for c in WEIGHTS:
                np.testing.assert_allclose(a[c+'_예측_tonne'],b[c+'_예측_tonne'],rtol=1e-12)
            self.assertEqual(len(loaded['models']['SWBS100']['diagnostics']),120)
            self.assertEqual(loaded['models']['SWBS100']['class_counts'],{'구축함':60,'호위함':60})
            wb=load_workbook(p);ws=wb['Model Parameters'];ws['B2']=ws['B2'].value*2;wb.save(p);wb.close()
            edited=load_model_excel(p)
            np.testing.assert_allclose(predict(self.df,edited)['SWBS100_예측_tonne'],a['SWBS100_예측_tonne']*2,rtol=1e-12)
            wb=load_workbook(p);wb['Ship Types']['D2']=0;wb.save(p);wb.close()
            with self.assertRaisesRegex(ValueError,'Dt'):load_model_excel(p)
    def test_identifiers_and_group_diagnostics(self):
        from model import save_model_excel, load_model_excel, diagnostic_rows, diagnostic_metrics
        data=self.df.copy();data.insert(0,'순번',[f'{i+1:03d}' for i in range(len(data))])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'ids.xlsx';data.to_excel(p,index=False)
            loaded=read_excel(p)
            self.assertEqual(loaded['순번'].iloc[0],'001')
            bundle=train(loaded);model=bundle['models']['SWBS100']
            self.assertEqual(model['diagnostics'][0]['순번'],'001')
            group=diagnostic_rows(model,'호위함')
            self.assertEqual(len(group),60)
            self.assertTrue(all(r['함종']=='호위함' for r in group))
            self.assertAlmostEqual(diagnostic_metrics(group)['r2'],1)
            self.assertIsNone(diagnostic_metrics(group[:1])['r2'])
            self.assertEqual(diagnostic_metrics(diagnostic_rows(model,'지원함'))['n'],0)
            save_model_excel(p,bundle)
            restored=load_model_excel(p)
            self.assertEqual(restored['models']['SWBS100']['diagnostics'][0]['순번'],'001')
            data.loc[0,'순번']='002' # both ambiguous rows excluded
            data.loc[2,'순번']='?'
            data.loc[3,'순번']=' '
            data.to_excel(p,index=False)
            checked=read_excel(p)
            self.assertEqual(len(checked),116)
            self.assertEqual(checked['순번'].iloc[0],'005')
            self.assertEqual([r['엑셀 행'] for r in checked.attrs['excluded_rows']],[2,3,4,5])
    def test_new_subtype_and_year_header(self):
        from model import HEADERS, save_model_excel, load_model_excel
        data=self.df.copy();data.insert(1,'함유형','DDG')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'subtype.xlsx';data.rename(columns=HEADERS).to_excel(p,index=False)
            loaded=read_excel(p)
            self.assertEqual(loaded['함유형'].iloc[0],'DDG')
            model=train(loaded)
            self.assertEqual(model['models']['SWBS100']['diagnostics'][0]['함유형'],'DDG')
            save_model_excel(p,model)
            self.assertEqual(load_model_excel(p)['models']['SWBS100']['diagnostics'][0]['함유형'],'DDG')
            np.testing.assert_allclose(model['models']['SWBS100']['coef'],train(self.df)['models']['SWBS100']['coef'])
            data.loc[0,'함유형']='?';data.loc[1,'함유형']=' '
            data.rename(columns=HEADERS).to_excel(p,index=False)
            self.assertEqual(len(read_excel(p)),118)
    def test_expected_year_applies_to_series_and_both_totals(self):
        bundle=train(self.df)
        earlier=self.df.iloc[[0]][BASE].copy();earlier['취역연도']=2030
        later=earlier.copy();later['취역연도']=2040
        first=predict(earlier,bundle).iloc[0];second=predict(later,bundle).iloc[0]
        for name,m in bundle['models'].items():
            ratio=np.exp(m['coef'][1+m['terms'].index('year')]) if 'year' in m['terms'] else 1
            self.assertAlmostEqual(second[name+'_예측_tonne']/first[name+'_예측_tonne'],ratio)
        self.assertAlmostEqual(second['총중량_SWBS합계_tonne'],sum(second[c+'_예측_tonne'] for c in WEIGHTS))
        self.assertIn('총중량_독립회귀_예측_tonne',second)
    def test_insufficient(self):
        with self.assertRaisesRegex(ValueError,'부족'): train(self.df.iloc[:3])

if __name__=='__main__': unittest.main()
