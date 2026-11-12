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
    def test_display_order_preserves_coefficient_indices(self):
        import math
        from model import train, display_types
        root=tk.Tk();app=App(root)
        try:
            df=fleet()
            types=['상륙함','고속함','호위함','구축함','지원함','초계함','기뢰전함']
            df['함분류']=[types[i%len(types)] for i in range(len(df))]
            app.bundle=train(df,compare=False,wape_limit_percent=None)
            app.refresh_model('legacy names');root.update()
            classes=app.bundle['models']['SWBS100']['classes']
            expected=display_types(classes)
            self.assertEqual(list(app.type_selector['values']),expected)
            self.assertEqual(list(app.formula_selector['values']),expected)
            rows=[app.coeff_tree.item(i)['values'] for i in app.coeff_tree.get_children()]
            for target,m in app.bundle['models'].items():
                target_rows=[r for r in rows if r[0]==target]
                self.assertEqual([r[1] for r in target_rows],display_types(m['classes']))
                for row in target_rows:
                    j=m['classes'].index(row[1])
                    dt=1.0 if j==0 else math.exp(m['coef'][1+len(m['terms'])+j-1])
                    self.assertAlmostEqual(float(row[4]),dt,delta=max(1e-7,abs(dt)*1e-7))
            self.assertNotIn('고속함',classes)
            self.assertIn('고속함 및 고속정',classes)
            app.open_ship_types()
            self.assertEqual(list(app.type_selector['values']),expected)
        finally:root.destroy()

    def test_ship_type_year_gui(self):
        root=tk.Tk();app=App(root);root.geometry('1380x940+0+0');root.update()
        self.assertLessEqual(app.header_description.winfo_rooty()+app.header_description.winfo_height(),app.header_frame.winfo_rooty()+app.header_frame.winfo_height()-4)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                from test_patrol import with_patrol
                df=with_patrol(1);p=Path(tmp)/'fleet.xlsx';df.to_excel(p,index=False)
                with patch.object(app,'choose',return_value=str(p)):app.fit()
                self.assertTrue(app.analysis_dialog.winfo_exists())
                deadline=time.monotonic()+120
                ticks=[];tick_times=[];last_tick=-1
                while app.busy and time.monotonic()<deadline:
                    root.update()
                    if app.busy:
                        ticks.append(app.analysis_tick)
                        if app.analysis_tick!=last_tick:
                            tick_times.append(time.monotonic());last_tick=app.analysis_tick
                    time.sleep(.02)
                self.assertGreater(max(ticks,default=0),5)
                self.assertLess(max((b-a for a,b in zip(tick_times,tick_times[1:])),default=0),1.0)
                self.assertFalse(app.busy);self.assertIsNotNone(app.bundle)
                self.assertFalse(app.analysis_dialog.winfo_exists())
                self.assertEqual(len(app.search_tree.get_children()),201)
                self.assertFalse(hasattr(app,'comparison_tree'))
                self.assertEqual(list(app.ship_type_entry['values']),['구축함','호위함','초계함','고속함 및 고속정','상륙함','지원함'])
                from model import display_types
                self.assertEqual(list(app.type_selector['values']),display_types(app.bundle['models']['SWBS100']['classes']))
                self.assertEqual(list(app.formula_selector['values']),display_types(app.bundle['models']['SWBS100']['classes']))
                self.assertNotIn('초계정',app.ship_type_entry['values'])
                self.assertNotIn('기뢰전함',app.ship_type_entry['values'])
                self.assertIn('Commissioning Year Adjustment by Ship Type',root.title())
                self.assertEqual(list(app.ship_type_entry['values']).count('초계함'),1)
                self.assertEqual(app.bundle['models']['SWBS100']['class_counts']['초계함'],1)
                self.assertEqual(app.reg_target.get(),'SWBS100')
                self.assertEqual(app.wape_initial.get(),'10')
                self.assertIn('WAPE',app.reg_stats.get())
                self.assertTrue(all(m['wape_percent']<=m['selection']['wape_limit_percent']+1e-9 for m in app.bundle['models'].values()))
                self.assertEqual(app.target_description.get(),'SWBS100 (Hull Structure)')
                app.reg_target.set('SWBS600');app.update_diagnostics()
                self.assertEqual(app.target_description.get(),'SWBS600 (Outfit & Furnishings)')
                app.reg_target.set('SWBS100');app.update_diagnostics()
                self.assertEqual(app.reg_tabs.index(app.reg_tabs.select()),0)
                self.assertFalse(hasattr(app,'validation_mode'))
                self.assertTrue(all('CV' not in c for c in app.search_tree['columns']))
                self.assertTrue(app.bundle['models']['SWBS100']['size_by_type'])
                self.assertIn('기준LBD(m³)',app.coeff_tree['columns'])
                self.assertEqual(app.bundle['models']['SWBS200']['selection']['basis'],'train')
                self.assertLessEqual(app.bundle['models']['SWBS200']['year_scale'],2)
                self.assertEqual(app.search_tree.item(app.search_tree.get_children()[0])['values'][0],0)
                self.assertEqual(app.search_tree.item(app.search_tree.get_children()[-1])['values'][0],200)
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
                p=Path(tmp)/'Regression Results.xlsx'
                app.ship_filter.set('초계함');app.update_diagnostics()
                with patch.object(app,'choose',return_value=str(p)):app.save()
                from openpyxl import load_workbook
                from results_export import SHEETS
                wb=load_workbook(p)
                self.assertEqual(wb.sheetnames,SHEETS)
                self.assertEqual(wb['중량 비교'].max_row,1+len(df)*7)
                wb.close()
                app.ship_filter.set('All Ship Types');app.update_diagnostics()
                app.open_guide();root.update()
                self.assertEqual(len(app.final_formula_images),7)
                captions=[w.cget('text') for w in app.final_body.winfo_children() if isinstance(w,tk.Label)]
                self.assertFalse(any(' / 기존 식' in t or ' / 추천 식' in t or ' / 추가 식' in t for t in captions))
                self.assertTrue(app.use_design_values.get())
                self.assertFalse(hasattr(app,'equation_images'))
                self.assertFalse(any(app.reg_tabs.tab(t,'text')=='Formula Comparison' for t in app.reg_tabs.tabs()))
                if os.environ.get('UI_SCREENSHOT'):
                    root.attributes('-topmost',True);root.lift();root.update()
                    from PIL import ImageGrab
                    app.status.set('DEMONSTRATION | SYNTHETIC TEST DATA / 실제 함정 데이터가 아닙니다.');app.open_estimation();root.update()
                    box=(root.winfo_rootx(),root.winfo_rooty(),root.winfo_rootx()+root.winfo_width(),root.winfo_rooty()+root.winfo_height())
                    ImageGrab.grab(bbox=box).save('ship-type-year-preview.png')
                    app.open_guide();root.update();time.sleep(.3)
                    ImageGrab.grab(bbox=box).save('ship-type-year-equations.png')
                    app.reg_target.set('SWBS200');app.update_diagnostics();app.reg_tabs.select(app.search_tab);root.update();time.sleep(.3)
                    ImageGrab.grab(bbox=box).save('year-search-preview.png')
                    app.reg_target.set('SWBS100');app.update_diagnostics();app.reg_tabs.select(3);root.update();time.sleep(.3)
                    ImageGrab.grab(bbox=box).save('type-size-coefficients.png')

        finally:root.destroy()
