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
        root=tk.Tk(); app=App(root); root.update()
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
                for c in BASE:app.inputs[c].set(str(row[c]))
                app.estimate();root.update()
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
                # Screenshot uses synthetic test data only, prominently identified.
                app.status.set('DEMONSTRATION  |  SYNTHETIC TEST DATA / 실제 함정 데이터가 아닙니다.')
                app.notebook.select(app.design_tab);root.update()
                if os.environ.get('UI_SCREENSHOT'):
                    from PIL import ImageGrab
                    root.update();time.sleep(1)
                    box=(root.winfo_rootx(),root.winfo_rooty(),root.winfo_rootx()+root.winfo_width(),root.winfo_rooty()+root.winfo_height())
                    ImageGrab.grab(bbox=box).save(os.environ['UI_SCREENSHOT'])
        finally:root.destroy()
