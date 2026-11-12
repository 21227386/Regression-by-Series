import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from openpyxl import load_workbook
import model
from additive import fit_baseline
from test_model import fleet


class ShipTypeNameTests(unittest.TestCase):
    def test_old_new_and_mixed_input_share_one_class(self):
        df=fleet();df.loc[df['함분류']=='구축함','함분류']='고속함'
        df.loc[0,'함분류']='고속함 및 고속정'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'legacy.xlsx';df.to_excel(path,index=False)
            actual=model.read_excel(path)
        self.assertEqual(actual['함분류'].value_counts().to_dict(),{'고속함 및 고속정':50,'호위함':50})
        self.assertEqual(actual.loc[2,'함분류_입력'],'고속함')
        self.assertEqual(actual.attrs['excluded_rows'],[])
        renamed=df.replace({'함분류':{'고속함':'고속함 및 고속정'}})
        a=model.train(actual,compare=False,wape_limit_percent=None)
        b=model.train(renamed,compare=False,wape_limit_percent=None)
        for name,m in a['models'].items():
            np.testing.assert_allclose(m['coef'],b['models'][name]['coef'],rtol=1e-12)
            self.assertNotIn('고속함',m['classes'])
        np.testing.assert_allclose(model.predict(df,a)[model.TOTAL_SUM_COLUMN],model.predict(renamed,b)[model.TOTAL_SUM_COLUMN],rtol=1e-12)

    def test_template_display_order_keeps_legacy_training_classes(self):
        expected=['구축함','호위함','초계함','고속함 및 고속정','상륙함','지원함','기뢰전함']
        self.assertEqual(model.display_types(model.SHIP_TYPES),expected)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'template.xlsx';model.template(p);wb=load_workbook(p)
            try:
                validation=list(wb['데이터'].data_validations.dataValidation)[0]
                self.assertEqual(validation.formula1,'"'+','.join(expected+['초계정'])+'"')
            finally:wb.close()

    def test_presentation_changes_cannot_change_reference_or_coefficients(self):
        # Equal counts: high-speed ships historically precede landing ships.
        df=fleet();df['함분류']=df['함분류'].replace({'구축함':'고속함 및 고속정','호위함':'상륙함'})
        for fitter in [lambda d:model.fit_one(d,'SWBS100',model.TERMS[100]),
                       lambda d:model.fit_one(d,'SWBS200',model.TERMS[200]),
                       lambda d:fit_baseline(d,'SWBS200',['P'])]:
            before=fitter(df)
            with patch.object(model,'SHIP_TYPE_DISPLAY_ORDER',list(reversed(model.SHIP_TYPE_DISPLAY_ORDER))):
                after=fitter(df)
            self.assertEqual(before['classes'][0],'고속함 및 고속정')
            self.assertEqual(before['classes'],after['classes'])
            np.testing.assert_allclose(before['coef'],after['coef'],rtol=1e-12)

    def test_legacy_archive_and_results_use_current_name(self):
        from results_export import save_regression_results
        df=fleet();df['함분류']=df['함분류'].replace({'구축함':'고속함'})
        bundle=model.train(df,compare=False,wape_limit_percent=None)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'archive.xlsx';model.save_model_excel(path,bundle)
            wb=load_workbook(path)
            for ws in wb:
                for row in ws:
                    for cell in row:
                        if cell.value=='고속함 및 고속정':cell.value='고속함'
            wb.save(path);wb.close()
            loaded=model.load_model_excel(path)
            np.testing.assert_allclose(model.predict(df,bundle)[model.TOTAL_SUM_COLUMN],model.predict(df,loaded)[model.TOTAL_SUM_COLUMN],rtol=1e-12)
            for m in loaded['models'].values():
                self.assertIn('고속함 및 고속정',m['classes'])
                self.assertNotIn('고속함',m['classes'])
            path=Path(tmp)/'results.xlsx';save_regression_results(path,loaded)
            wb=load_workbook(path)
            try:
                values=[cell.value for ws in wb for row in ws for cell in row]
                self.assertIn('고속함 및 고속정',values)
                self.assertNotIn('고속함',values)
            finally:wb.close()
