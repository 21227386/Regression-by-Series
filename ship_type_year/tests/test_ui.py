import os
import sys
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import tkinter as tk
from app import App
from test_model import fleet

@unittest.skipUnless(sys.platform=='win32' or os.environ.get('DISPLAY'),'Desktop display required')
class UITests(unittest.TestCase):
    def test_ship_type_year_gui(self):
        root=tk.Tk();app=App(root);root.geometry('1380x900+0+0');root.update()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                df=fleet();p=Path(tmp)/'fleet.xlsx';df.to_excel(p,index=False)
                with patch.object(app,'choose',return_value=str(p)):app.fit()
                deadline=time.monotonic()+30
                while app.busy and time.monotonic()<deadline:root.update();time.sleep(.02)
                self.assertFalse(app.busy);self.assertIsNotNone(app.bundle)
                self.assertEqual(len(app.search_tree.get_children()),10)
                self.assertEqual(app.bundle['models']['SWBS200']['selection']['basis'],'ship')
                for c,var in app.inputs.items():var.set(str(df.iloc[0][c]))
                app.inputs['함분류'].set('호위함');app.inputs['취역연도'].set('')
                app.open_estimation();app.estimate();root.update()
                self.assertEqual(app.data.iloc[0]['취역연도'],app.bundle['models']['SWBS100']['mean_years']['호위함'])
                self.assertIn('호위함',app.estimation_assumption.get())
                self.assertIn('tonne',app.sum_value.get());self.assertIn('tonne',app.total_value.get())
                self.assertEqual(len(app.result_tree.get_children()),6)
                app.inputs['취역연도'].set('2030');app.estimate();root.update()
                self.assertIn('2030',app.prediction_note.get())
                self.assertIn('700계열 제외',app.prediction_note.get())
                self.assertNotIn('SWBS700',app.bundle['models'])
                self.assertEqual(len(app.metric_tree.get_children()),7)
                p=Path(tmp)/'result.xlsx'
                with patch.object(app,'choose',return_value=str(p)),patch('app.messagebox.showinfo'):app.report()
                self.assertTrue(p.exists())
                if os.environ.get('UI_SCREENSHOT'):
                    from PIL import ImageGrab
                    app.status.set('DEMONSTRATION | SYNTHETIC TEST DATA / 실제 함정 데이터가 아닙니다.');root.update()
                    box=(root.winfo_rootx(),root.winfo_rooty(),root.winfo_rootx()+root.winfo_width(),root.winfo_rooty()+root.winfo_height())
                    ImageGrab.grab(bbox=box).save('ship-type-year-preview.png')
                    app.open_guide();root.update();time.sleep(.3)
                    ImageGrab.grab(bbox=box).save('ship-type-year-equations.png')
                    app.reg_target.set('SWBS200');app.update_diagnostics();app.reg_tabs.select(app.search_tab);root.update();time.sleep(.3)
                    ImageGrab.grab(bbox=box).save('year-search-preview.png')
        finally:root.destroy()
