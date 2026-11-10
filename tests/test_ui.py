import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import tkinter as tk
from app import App
from model import BASE
import test_model

@unittest.skipUnless(sys.platform=='win32' or os.environ.get('DISPLAY'), 'Desktop display required')
class UITests(unittest.TestCase):
    def test_upload_estimate_export_and_template(self):
        fixture=test_model.ModelTests(); fixture.setUp()
        fixture.df.insert(0,'순번',[f'{i+1:03d}' for i in range(len(fixture.df))])
        fixture.df.insert(2,'함유형','DDG')
        root=tk.Tk(); app=App(root); root.geometry('1380x900+0+0'); root.update()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                source=Path(tmp)/'fleet.xlsx'; fixture.df.to_excel(source,index=False)
                with patch.object(app,'choose',return_value=str(source)):
                    app.fit()
                deadline=time.monotonic()+30
                while app.busy and time.monotonic()<deadline:
                    root.update();time.sleep(.02)
                self.assertFalse(app.busy);self.assertIsNotNone(app.bundle)
                row=fixture.df.iloc[0]
                for c in app.inputs:app.inputs[c].set(str(row[c]))
                app.estimate();app.notebook.select(app.design_tab);root.update()
                self.assertIn('tonne',app.sum_value.get())
                self.assertEqual(len(app.result_tree.get_children()),7)
                last=app.result_tree.bbox(app.result_tree.get_children()[-1])
                self.assertTrue(last, 'SWBS700 must be visible')
                self.assertLessEqual(last[1]+last[3], app.result_tree.winfo_height())
                output=Path(tmp)/'result.xlsx'
                with patch.object(app,'choose',return_value=str(output)),patch('app.messagebox.showinfo'):
                    app.report()
                self.assertTrue(output.exists())
                form=Path(tmp)/'template.xlsx'
                with patch.object(app,'choose',return_value=str(form)),patch('app.messagebox.showinfo'):
                    app.make_template()
                self.assertTrue(form.exists())
                model_path=Path(tmp)/'model.xlsx'
                with patch.object(app,'choose',return_value=str(model_path)):
                    app.save()
                    self.assertTrue(model_path.exists())
                    app.load()
                self.assertEqual(len(app.ship_tree.get_children()),120)
                self.assertEqual(len(app.coeff_tree.get_children()),16)
                self.assertEqual(len(app.metric_tree.get_children()),8)
                app.open_ship_types();root.update()
                self.assertEqual(len(app.ship_tree.get_children()),60)
                self.assertEqual(len(app.coeff_tree.get_children()),8)
                self.assertEqual(app.metric_tree.item(app.metric_tree.get_children()[0])['values'][1],60)
                self.assertEqual(len(app.reg_graph.find_withtag('ship_number')),120)
                self.assertTrue(app.hover_label.winfo_ismapped())
                self.assertGreater(app.hover_label.winfo_height(),10)
                app.select_graph_ship('001','Selected 001');root.update()
                self.assertTrue(app.ship_tree.selection())
                self.assertEqual(str(app.ship_tree.item(app.ship_tree.selection()[0])['values'][0]).zfill(3),'001')
                app.open_regression();root.update()
                self.assertEqual(len(app.ship_tree.get_children()),120)
                self.assertEqual(len(app.coeff_tree.get_children()),16)
                for c in app.inputs:app.inputs[c].set(str(row[c]))
                app.inputs['취역연도'].set('')
                app.open_estimation();app.estimate();root.update()
                self.assertAlmostEqual(app.data.iloc[0]['취역연도'],app.bundle['models']['SWBS100']['mean_year'])
                self.assertEqual(app.data.iloc[0]['연도 적용'],'학습 평균연도')
                self.assertNotIn('순항속력',app.inputs)
                self.assertEqual(len(app.equation_images),11)
                self.assertEqual(app.reg_tabs.tab(1,'text'),'Equations & Guide')
                self.assertTrue(all(image.width()>400 for image in app.equation_images))
                def descendants(widget):
                    for child in widget.winfo_children():
                        yield child
                        yield from descendants(child)
                button_texts=[w.cget('text') for w in descendants(root) if isinstance(w,tk.Button)]
                self.assertNotIn('Load Regression Model',button_texts)
                self.assertNotIn('Equations & Coefficient Guide',button_texts)
                app.inputs['취역연도'].set('2035')
                app.estimate();root.update()
                self.assertEqual(app.data.iloc[0]['취역연도'],2035)
                self.assertIn('2035',app.prediction_note.get())
                self.assertIn('입력 취역 예상연도',app.prediction_note.get())
                from model import predict
                output_row=predict(app.data,app.bundle).iloc[0]
                self.assertEqual(app.sum_value.get(),f"{output_row['총중량_SWBS합계_tonne']:,.2f} tonne")
                self.assertEqual(app.total_value.get(),f"{output_row['총중량_독립회귀_예측_tonne']:,.2f} tonne")
                # Screenshot uses synthetic test data only, prominently identified.
                app.status.set('DEMONSTRATION  |  SYNTHETIC TEST DATA / 실제 함정 데이터가 아닙니다.')
                app.notebook.select(app.design_tab);root.update()
                if os.environ.get('UI_SCREENSHOT'):
                    from PIL import ImageGrab
                    root.update();time.sleep(1)
                    box=(root.winfo_rootx(),root.winfo_rooty(),root.winfo_rootx()+root.winfo_width(),root.winfo_rooty()+root.winfo_height())
                    ImageGrab.grab(bbox=box).save(os.environ['UI_SCREENSHOT'])
                    app.open_regression();root.update();time.sleep(.4)
                    ImageGrab.grab(bbox=box).save('regression-preview.png')
                    app.open_ship_types();root.update();time.sleep(.4)
                    ImageGrab.grab(bbox=box).save('ship-types-preview.png')
                    app.open_guide();root.update();time.sleep(.4)
                    ImageGrab.grab(bbox=box).save('equations-preview.png')
        finally:root.destroy()
