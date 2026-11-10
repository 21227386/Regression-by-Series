import os
import sys
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import tkinter as tk
from app import App, TITLE
from test_model import fleet

@unittest.skipUnless(sys.platform=='win32' or os.environ.get('DISPLAY'),'Desktop display required')
class UITests(unittest.TestCase):
    def test_without_year_gui(self):
        root=tk.Tk();app=App(root);root.geometry('1380x900+0+0');root.update()
        try:
            self.assertEqual(TITLE,'without year');self.assertNotIn('취역연도',app.inputs)
            self.assertEqual(len(app.equation_images),10)
            with tempfile.TemporaryDirectory() as tmp:
                df=fleet();p=Path(tmp)/'fleet.xlsx';df.to_excel(p,index=False)
                with patch.object(app,'choose',return_value=str(p)):app.fit()
                deadline=time.monotonic()+30
                while app.busy and time.monotonic()<deadline:root.update();time.sleep(.02)
                self.assertFalse(app.busy);self.assertIsNotNone(app.bundle)
                for c,var in app.inputs.items():var.set(str(df.iloc[0][c]))
                app.open_estimation();app.estimate();root.update()
                self.assertEqual(len(app.result_tree.get_children()),7)
                self.assertIn('tonne',app.sum_value.get());self.assertIn('tonne',app.total_value.get())
                p=Path(tmp)/'report.xlsx'
                with patch.object(app,'choose',return_value=str(p)),patch('app.messagebox.showinfo'):app.report()
                self.assertTrue(p.exists())
                if os.environ.get('UI_SCREENSHOT'):
                    from PIL import ImageGrab
                    app.status.set('DEMONSTRATION | SYNTHETIC TEST DATA / 실제 함정 데이터가 아닙니다.');root.update()
                    box=(root.winfo_rootx(),root.winfo_rooty(),root.winfo_rootx()+root.winfo_width(),root.winfo_rooty()+root.winfo_height())
                    ImageGrab.grab(bbox=box).save('without-year-preview.png')
                    app.open_guide();root.update();time.sleep(.3)
                    ImageGrab.grab(bbox=box).save('without-year-equations.png')
        finally:root.destroy()
