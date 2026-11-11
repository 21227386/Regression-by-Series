"""Naval Ship Weight Estimator — native Windows desktop interface."""
import multiprocessing as mp
import time
import queue
import math
import base64
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from year_search import SCALES,MAX_PERCENT
from training_job import start_regression
from results_export import save_regression_results
from model import (BASE, SHIP_TYPES, read_excel, train, predict, export, template,
                   coefficients, validate_model_types, normalize_types, load_model_excel, SHIP_ALIASES, TOTAL_MODEL, TOTAL_SUM_COLUMN, diagnostic_rows, diagnostic_metrics, formula_text)

TARGET_NAMES={'SWBS100':'Hull Structure','SWBS200':'Propulsion Plant','SWBS300':'Electric Plant',
              'SWBS400':'Command & Surveillance','SWBS500':'Auxiliary Systems','SWBS600':'Outfit & Furnishings',
              TOTAL_MODEL:'Lightship (Excluding SWBS700)'}
TITLE = 'Naval Ship Weight Estimator - Commissioning Year Adjustment by Ship Type'
NAVY, INK, TEAL, GOLD = '#102638', '#19384a', '#087f87', '#c69c55'
BG, WHITE, MUTED = '#edf2f5', '#ffffff', '#607887'
EN, KO = 'Times New Roman', '맑은 고딕'
SERIES = ['Hull Structure', 'Propulsion', 'Electrical', 'Command & Surveillance',
          'Auxiliary Systems', 'Outfit & Furnishings']

class App:
    def __init__(self, root):
        self.root, self.bundle, self.data = root, None, None
        self.busy = False
        self.final_formula_key=None
        root.title(TITLE); root.geometry('1380x940'); root.minsize(1180, 800)
        root.configure(bg=BG)
        root.option_add('*Font', (KO, 10))
        style = ttk.Style(root); style.theme_use('clam')
        style.configure('TNotebook', background=BG, borderwidth=0)
        style.configure('TNotebook.Tab', font=(EN, 11, 'bold'), padding=(14, 10), background='#dce6ec')
        style.map('TNotebook.Tab', background=[('selected', WHITE)], foreground=[('selected', TEAL)])
        style.configure('Treeview', font=(KO, 10), rowheight=23, background=WHITE, fieldbackground=WHITE)
        style.configure('Treeview.Heading', font=(EN, 11, 'bold'), background='#dce7ed', foreground=INK)
        style.configure('TCombobox', padding=6)
        self.header()
        footer = tk.Frame(root, bg=NAVY, height=30); footer.pack(side='bottom', fill='x')
        self.status = tk.StringVar(value='READY  |  실적함 데이터를 업로드하여 시작하세요.')
        tk.Label(footer, textvariable=self.status, bg=NAVY, fg='#c6d7df', font=(KO, 9)).pack(side='left', padx=14, pady=5)
        tk.Label(footer, text='Copyright by Naval Ship Design Technology Division', font=(EN, 10), bg=NAVY, fg='#d7bd88').pack(side='right', padx=16)
        body = tk.Frame(root, bg=BG); body.pack(fill='both', expand=True)
        sidebar = tk.Frame(body, bg=NAVY, width=290); sidebar.pack(side='left', fill='y'); sidebar.pack_propagate(False)
        self.sidebar(sidebar)
        content = tk.Frame(body, bg=BG, padx=20, pady=15); content.pack(side='left', fill='both', expand=True)
        self.notebook = ttk.Notebook(content); self.notebook.pack(fill='both', expand=True)
        self.design_tab = tk.Frame(self.notebook, bg=WHITE)
        self.model_tab = tk.Frame(self.notebook, bg=WHITE)
        self.notebook.add(self.design_tab, text='Weight Estimation')
        self.notebook.add(self.model_tab, text='Regression Results')
        self.design_panel(); self.model_panel()
        self.inputs['함분류'].trace_add('write',lambda *args:self.update_year_reference())

    def label(self, parent, text, size=11, color=INK, bg=WHITE, bold=False, english=False):
        return tk.Label(parent, text=text, font=(EN if english else KO, size, 'bold' if bold else 'normal'), fg=color, bg=bg, anchor='w')

    def button(self, parent, text, action, primary=False):
        button = tk.Button(parent, text=text, command=lambda: self.safe(action), font=(EN, 11, 'bold'),
                           bg=TEAL if primary else '#e4edf2', fg=WHITE if primary else INK,
                           activebackground='#116d76' if primary else '#cfdee7', activeforeground=WHITE if primary else INK,
                           relief='raised', bd=3, cursor='hand2', padx=12, pady=5)
        return button

    def header(self):
        h = tk.Frame(self.root, bg=NAVY, height=166); h.pack(fill='x'); h.pack_propagate(False)
        words = tk.Frame(h, bg=NAVY); words.pack(side='left', padx=28, pady=10)
        self.label(words, 'NAVAL SHIP DESIGN  /  ENGINEERING ANALYTICS', 10, GOLD, NAVY, True, True).pack(anchor='w')
        self.label(words, 'Naval Ship Weight Estimator', 28, WHITE, NAVY, True, True).pack(anchor='w', pady=(8, 0))
        self.label(words, 'Total & Series / Commissioning Year Adjustment by Ship Type', 18, '#a9d6da', NAVY, False, True).pack(anchor='w')
        self.header_description=self.label(words, '함정 중량 추정 및 SWBS 계열별 중량 분석', 11, '#dfedf5', NAVY, True)
        self.header_description.pack(anchor='w', pady=(5, 0))
        self.header_frame=h
        self.root.after_idle(lambda:h.configure(height=max(156,words.winfo_reqheight()+20)))
        c = tk.Canvas(h, width=440, height=142, bg=NAVY, highlightthickness=0); c.pack(side='right', padx=25)
        self.ship_canvas = c; self.draw_ship(c)

    def draw_ship(self, c):
        for x in range(20, 441, 24): c.create_line(x, 12, x, 133, fill='#1b3a4e')
        for y in range(12, 140, 20): c.create_line(15, y, 430, y, fill='#1b3a4e')
        c.create_polygon(33,82,392,82,365,101,70,101,fill='#34586b',outline='#b2dbe1',width=2)
        c.create_polygon(109,81,127,61,227,61,251,81,fill='#426579',outline='#a9d5dd')
        c.create_polygon(157,60,162,43,198,43,211,60,fill='#426579',outline='#a9d5dd')
        c.create_line(181,43,181,17,fill='#b2dbe1',width=2)
        c.create_line(164,30,202,30,fill='#b2dbe1',width=2)
        c.create_line(181,18,204,38,fill='#8cb7c6')
        for x in [133,146,162,176,190,205]: c.create_rectangle(x,65,x+7,70,fill='#8be0e5',outline='')
        c.create_polygon(289,81,294,70,317,70,324,81,fill='#557b8b',outline='#a9d5dd')
        c.create_line(307,71,339,60,fill='#b2dbe1',width=3)
        for x in [89,99,109]: c.create_line(x,78,x+5,67,fill='#a9d5dd',width=3)
        c.create_line(30,108,403,108,fill=TEAL,width=2)
        c.create_line(70,117,365,117,fill=GOLD,arrow='both')
        c.create_text(217,129,text='L  ×  B  ×  D   →   WEIGHT / tonne',font=(EN,10),fill='#d7bd88')
        for x in [57,83,327,355]: c.create_line(x,102,x,111,fill='#92b6c4')

    def sidebar(self, p):
        inner = tk.Frame(p, bg=NAVY, padx=18, pady=14); inner.pack(fill='both', expand=True)
        self.label(inner, '1  /  Reference Fleet', 13, GOLD, NAVY, True, True).pack(anchor='w')
        self.label(inner, '실적함 데이터 · 자동 회귀 학습', 10, '#d2e1e8', NAVY).pack(anchor='w', pady=(6,8))
        self.button(inner, 'Download Input File Template', self.make_template).pack(fill='x', pady=3)
        self.button(inner, 'Input File Upload', self.fit, True).pack(fill='x', pady=3)
        settings=tk.Frame(inner,bg=NAVY);settings.pack(fill='x',pady=(5,0))
        tk.Label(settings,text='초기 WAPE 기준 (%)',font=(KO,10),bg=NAVY,fg='#d2e1e8').pack(side='left')
        self.wape_initial=tk.StringVar(value='10')
        ttk.Entry(settings,textvariable=self.wape_initial,width=6).pack(side='right')
        tk.Label(inner,text='후보 없으면 해당 계열만 1%p 자동 완화',font=(KO,9),bg=NAVY,fg='#b8cdd8').pack(anchor='w',pady=(2,3))
        self.fleet_info = tk.StringVar(value='아직 학습된 모델이 없습니다.')
        tk.Label(inner, textvariable=self.fleet_info, wraplength=245, justify='left', bg=NAVY, fg='#b8cdd8', font=(KO,10)).pack(anchor='w', pady=5)
        self.label(inner, '2  /  Regression', 13, GOLD, NAVY, True, True).pack(anchor='w', pady=(10,5))
        self.button(inner, 'View All Results', self.open_regression).pack(fill='x', pady=3)
        self.button(inner, 'View Ship Types Results', self.open_ship_types).pack(fill='x', pady=3)
        self.button(inner, 'Save Regression Model', self.save).pack(fill='x', pady=3)
        self.label(inner, '3  /  Weight Estimation', 13, GOLD, NAVY, True, True).pack(anchor='w', pady=(10,5))
        self.button(inner, 'Open Weight Estimation', self.open_estimation, True).pack(fill='x', pady=3)
        self.label(inner, '4  /  Reports', 13, GOLD, NAVY, True, True).pack(anchor='w', pady=(10,5))
        self.button(inner, 'Export Results to Excel', self.report).pack(fill='x', pady=3)
        tk.Label(inner, text='중량 tonne / 치수 m\n마력 hp / 전력 kW / 속력 kn', bg=NAVY, fg='#b8cdd8', font=(KO,9), justify='left').pack(anchor='w', pady=5)

    def design_panel(self):
        panel = tk.Frame(self.design_tab, bg=WHITE, padx=20, pady=15); panel.pack(fill='both', expand=True)
        self.label(panel, 'Weight Estimation', 21, bold=True, english=True).pack(anchor='w')
        self.label(panel, '함종·길이·폭·깊이·소요마력·소요전력으로 SWBS100~600 및 700계열 제외 경하중량을 추정합니다.', 10, MUTED).pack(anchor='w', pady=(3,12))
        form = tk.Frame(panel, bg=WHITE); form.pack(fill='x')
        self.inputs = {}
        fields = [('함분류','Ship Type / 함종'), ('길이','Length L / 길이 (m)'),
                  ('폭','Beam B / 폭 (m)'), ('깊이','Depth D / 깊이 (m)'),
                  ('소요마력','Propulsion Power / 소요마력 (hp)'), ('소요전력','Electrical Power / 소요전력 (kW)'),
                  ('취역연도','Expected Year / 취역 예상연도 (year)')]
        for i, (key, caption) in enumerate(fields):
            cell = tk.Frame(form, bg=WHITE); cell.grid(row=i//4,column=i%4,sticky='ew',padx=(0,12),pady=6)
            form.columnconfigure(i%4, weight=1)
            self.label(cell, caption, 10, MUTED, english=True).pack(anchor='w')
            var=tk.StringVar(); self.inputs[key]=var
            if key=='함분류':
                e=ttk.Combobox(cell,textvariable=var,values=[t for t in SHIP_TYPES if t!='기뢰전함'],state='readonly',width=15,font=(KO,11)); var.set('호위함');self.ship_type_entry=e
            else:
                e=tk.Entry(cell,textvariable=var,font=(EN,14),bg='#f3f7fa',fg=INK,relief='sunken',bd=2,width=15)
            e.pack(fill='x',ipady=4,pady=(4,0))
        row = tk.Frame(panel,bg=WHITE); row.pack(fill='x',pady=(10,4))
        self.estimate_button=self.button(row,'Estimate Ship Weight',self.estimate,True); self.estimate_button.pack(side='left')
        self.estimation_assumption=tk.StringVar(value='취역 예상연도 미입력 시 선택 함종의 평균연도를 적용합니다. 순항속력은 현재 식에 사용하지 않습니다.')
        tk.Label(row,textvariable=self.estimation_assumption,font=(KO,9),fg=MUTED,bg=WHITE,wraplength=600,justify='left').pack(side='left',padx=14)
        cards=tk.Frame(panel,bg=WHITE); cards.pack(fill='x',pady=12)
        self.sum_value=tk.StringVar(value='—'); self.total_value=tk.StringVar(value='—')
        for title,var,color in [('SWBS100–600 합계 / 700계열 제외',self.sum_value,TEAL),('경하중량 직접 회귀 / 700계열 제외',self.total_value,INK)]:
            card=tk.Frame(cards,bg='#edf4f6',highlightbackground='#d3e1e7',highlightthickness=1,padx=14,pady=10); card.pack(side='left',fill='both',expand=True,padx=(0,10))
            self.label(card,title,11,color,'#edf4f6',True,True).pack(anchor='w')
            tk.Label(card,textvariable=var,font=(EN,25,'bold'),fg=color,bg='#edf4f6').pack(anchor='w')
        lower=tk.Frame(panel,bg=WHITE); lower.pack(fill='both',expand=True)
        self.result_tree=self.table(lower,['SWBS','System / 계열','Weight (tonne)'],[85,230,130]); self.result_tree.pack(side='left',fill='both',expand=True)
        self.chart=tk.Canvas(lower,bg='#f5f8fa',width=290,height=180,highlightthickness=0); self.chart.pack(side='right',fill='both',expand=True,padx=(12,0))
        self.chart.bind('<Configure>',lambda e:self.draw_chart())
        self.last_weights=None
        self.prediction_note=tk.StringVar(value='모델 학습 또는 불러오기 후 예측할 수 있습니다. 학습 적합도는 신규 함정 예측 정확도를 보장하지 않습니다.')
        tk.Label(panel,textvariable=self.prediction_note,bg=WHITE,fg=MUTED,font=(KO,9),wraplength=880,justify='left',anchor='w').pack(fill='x',pady=(10,0))

    def table(self,parent,cols,widths):
        tree=ttk.Treeview(parent,columns=cols,show='headings',height=7)
        for c,w in zip(cols,widths): tree.heading(c,text=c); tree.column(c,width=w,minwidth=60,anchor='w' if c==cols[1] else 'center')
        return tree

    def open_estimation(self):
        self.notebook.select(self.design_tab)

    def open_guide(self):
        self.notebook.select(self.model_tab);self.reg_tabs.select(self.guide_tab)

    def open_regression(self):
        self.ship_filter.set('All Ship Types');self.type_selector.configure(state='disabled')
        self.update_diagnostics();self.notebook.select(self.model_tab);self.reg_tabs.select(0)

    def open_ship_types(self):
        self.require()
        classes=self.bundle['models']['SWBS100']['classes']
        if self.ship_filter.get() not in classes:self.ship_filter.set(classes[0])
        self.type_selector.configure(values=classes,state='readonly')
        self.update_diagnostics();self.notebook.select(self.model_tab);self.reg_tabs.select(0)

    def scroll_table(self, parent, columns, widths, height=8):
        frame=tk.Frame(parent,bg=WHITE); frame.pack(fill='both',expand=True)
        tree=self.table(frame,columns,widths);tree.configure(height=height)
        sy=ttk.Scrollbar(frame,orient='vertical',command=tree.yview)
        sx=ttk.Scrollbar(frame,orient='horizontal',command=tree.xview)
        tree.configure(yscrollcommand=sy.set,xscrollcommand=sx.set)
        sy.pack(side='right',fill='y');sx.pack(side='bottom',fill='x');tree.pack(fill='both',expand=True)
        return tree

    def model_panel(self):
        p=tk.Frame(self.model_tab,bg=WHITE,padx=18,pady=15);p.pack(fill='both',expand=True)
        self.label(p,'Regression Results',21,bold=True,english=True).pack(anchor='w')
        self.regression_summary=tk.StringVar(value='전체 실적함 파일을 업로드하면 회귀 결과가 표시됩니다.')
        tk.Label(p,textvariable=self.regression_summary,bg=WHITE,fg=MUTED,font=(KO,10),anchor='w',justify='left',wraplength=940).pack(fill='x',pady=5)
        select=tk.Frame(p,bg=WHITE);select.pack(fill='x',pady=5)
        self.label(select,'Analysis Target',12,bold=True,english=True).pack(side='left',padx=(0,12))
        self.reg_target=tk.StringVar(value='SWBS100')
        self.reg_selector=ttk.Combobox(select,textvariable=self.reg_target,state='readonly',width=23,values=[])
        self.reg_selector.pack(side='left');self.reg_selector.bind('<<ComboboxSelected>>',lambda e:self.update_diagnostics())
        self.target_description=tk.StringVar(value='SWBS100 (Hull Structure)')
        tk.Label(select,textvariable=self.target_description,bg=WHITE,fg=TEAL,font=(EN,12,'bold')).pack(side='left',padx=12)
        self.reg_stats=tk.StringVar(value='R² / RMSE / MAPE —')
        tk.Label(p,textvariable=self.reg_stats,bg=WHITE,fg=TEAL,font=(EN,11,'bold')).pack(anchor='w',pady=(0,5))
        filters=tk.Frame(p,bg=WHITE);filters.pack(fill='x',pady=(0,3))
        self.label(filters,'Ship Type',11,bold=True,english=True).pack(side='left',padx=(0,12))
        self.ship_filter=tk.StringVar(value='All Ship Types')
        self.type_selector=ttk.Combobox(filters,textvariable=self.ship_filter,state='disabled',width=18)
        self.type_selector.pack(side='left');self.type_selector.bind('<<ComboboxSelected>>',lambda e:self.update_diagnostics())
        self.show_ship_ids=tk.BooleanVar(value=True)
        tk.Checkbutton(filters,text='Show Ship Numbers',variable=self.show_ship_ids,command=self.draw_regression_graphs,
                       bg=WHITE,fg=INK,font=(EN,11),activebackground=WHITE).pack(side='left',padx=14)
        self.label(filters,'전체 모형의 함종별 결과 조회 · 별도 재학습 없음',9,MUTED).pack(side='left')
        tabs=ttk.Notebook(p);self.reg_tabs=tabs;tabs.pack(fill='both',expand=True,pady=(8,0))
        overview=tk.Frame(tabs,bg=WHITE);coefs=tk.Frame(tabs,bg=WHITE);ships=tk.Frame(tabs,bg=WHITE)
        stats=tk.Frame(tabs,bg=WHITE);notes=tk.Frame(tabs,bg=WHITE)
        for frame,title in [(overview,'Graphs'),(coefs,'Coefficients'),(ships,'Ship Diagnostics'),(stats,'Fit Statistics'),(notes,'Notes')]:tabs.add(frame,text=title)
        self.guide_tab=tk.Frame(tabs,bg=WHITE);tabs.insert(1,self.guide_tab,text='Equations & Guide')
        self.build_guide(self.guide_tab)
        tabs.bind('<<NotebookTabChanged>>',lambda e:self.root.after_idle(self.refresh_final_equations),add='+')
        self.search_tab=tk.Frame(tabs,bg=WHITE);tabs.insert(2,self.search_tab,text='Year Effect Search')
        self.search_note=tk.StringVar(value='연도 효과 0%~200% (1% 간격) 후보를 비교합니다.')
        tk.Label(self.search_tab,textvariable=self.search_note,bg=WHITE,fg=INK,font=(KO,10),wraplength=930,justify='left',anchor='w').pack(fill='x',padx=10,pady=8)
        self.search_graph=tk.Canvas(self.search_tab,bg='#f3f7fa',height=115,highlightthickness=0)
        self.search_graph.pack(side='bottom',fill='x',padx=10,pady=5)
        self.search_graph.bind('<Configure>',lambda e:self.draw_search_graph())
        self.search_tree=self.scroll_table(self.search_tab,['Year %','Train MAPE (%)','Train WAPE (%)','WAPE OK','Train RMSE (t)','Ships','Selected','Error'],[80,135,135,95,140,70,85,350])
        self.search_tree.tag_configure('chosen',background='#d8eeeb')
        self.reg_graph=tk.Canvas(overview,bg='#f3f7fa',height=320,highlightthickness=0)
        # Pack the graph after its information footer so hover details remain visible.
        self.reg_graph.bind('<Configure>',lambda e:self.draw_regression_graphs())
        self.graph_hover=tk.StringVar(value='실적함 데이터를 학습하거나 모델 엑셀을 불러오세요.')
        self.hover_label=tk.Label(overview,textvariable=self.graph_hover,bg=WHITE,fg=INK,font=(KO,9),anchor='w',wraplength=930)
        self.hover_label.pack(side='bottom',fill='x',pady=4)
        self.label(overview,'점의 순번으로 함정을 식별합니다. 마우스: 상세 정보 / 클릭: 진단표 이동. 적합도는 학습자료 기준입니다.',9,MUTED).pack(side='bottom',anchor='w',pady=3)
        self.label(overview,'왼쪽: 실제·예측 중량(tonne) / 오른쪽: 예측 중량 대비 오차율(%)',9,MUTED).pack(side='bottom',anchor='w',pady=3)
        self.reg_graph.pack(fill='both',expand=True,pady=8)
        cols=['계열','함종','학습척수','기준함종','Dt','K','C','LBD','LB','L','P','PE','D','B','k0','가산계수','연도효과%','기본γt','적용γt','함종평균연도','기준LBD(m³)']
        self.coeff_tree=self.scroll_table(coefs,cols,[160,85,75,85,100,120,100,100,100,100,100,100,100,100,110,350,100,100,100,100,130])
        self.coeff_tree.heading('LBD',text='LBD 지수 α/αt');self.coeff_tree.column('LBD',width=125)
        self.ship_diagnostics_tab=ships
        self.ship_tree=self.scroll_table(ships,['Ship No.','Excel Row','Ship Type','Subtype','Dt','F(Y)','Actual (t)','Predicted (t)','Residual (t)','Error (%)'],[85,80,90,100,85,95,120,120,120,100])
        self.ship_tree.tag_configure('large_error',foreground='#ae4437')
        self.metric_tree=self.scroll_table(stats,['Target','Ships','Train RMSE (t)','Train MAPE (%)','Train WAPE (%)','WAPE Limit (%)','Train R²','Parameters','Condition','Year %'],[180,65,135,110,110,120,100,90,120,90])
        self.log=tk.Text(notes,wrap='word',font=(EN,11),fg=INK,bg='#f3f7fa',relief='flat',padx=12,pady=12)
        sy=ttk.Scrollbar(notes,orient='vertical',command=self.log.yview);self.log.configure(yscrollcommand=sy.set)
        sy.pack(side='right',fill='y');self.log.pack(fill='both',expand=True)
        self.show('실적함 업로드 후 자동 학습합니다.\n기준 함종은 오류 행 제외 및 초계정의 호위함 통합(초계함 별도) 후 척수가 가장 많은 비초계함 함종입니다.\n동률이면 구축함·호위함·고속함·상륙함·기뢰전함·지원함 순입니다.\nK와 역변환 보정계수 C는 분리하여 표시합니다.')

    def build_guide(self,parent):
        canvas=tk.Canvas(parent,bg='#edf2f5',highlightthickness=0)
        scrollbar=ttk.Scrollbar(parent,orient='vertical',command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right',fill='y');canvas.pack(fill='both',expand=True)
        content=tk.Frame(canvas,bg='#edf2f5',padx=12,pady=12)
        window=canvas.create_window((0,0),window=content,anchor='nw')
        content.bind('<Configure>',lambda e:canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',lambda e:canvas.itemconfigure(window,width=e.width))
        self.label(content,'Regression Equations & Coefficients',20,TEAL,'#edf2f5',True,True).pack(anchor='w',pady=(0,6))
        tk.Label(content,text='실적함에 가장 잘 맞는 최종 선택식과 변수 설명입니다. 모든 중량은 tonne입니다.',
                 font=(KO,10),fg=MUTED,bg='#edf2f5',anchor='w').pack(fill='x',pady=(0,12))
        self.final_panel=tk.Frame(content,bg=WHITE,padx=12,pady=12);self.final_panel.pack(fill='x',pady=(0,14))
        self.label(self.final_panel,'Final Fitted Equations / 최종 추출식',16,TEAL,WHITE,True,True).pack(anchor='w')
        choices=tk.Frame(self.final_panel,bg=WHITE);choices.pack(fill='x',pady=8)
        self.formula_ship=tk.StringVar()
        self.formula_selector=ttk.Combobox(choices,textvariable=self.formula_ship,state='readonly',width=15)
        self.formula_selector.pack(side='left');self.formula_selector.bind('<<ComboboxSelected>>',lambda e:self.refresh_final_equations(force=True))
        self.use_design_values=tk.BooleanVar(value=False)
        tk.Checkbutton(choices,text='설계 제원도 숫자로 대입',variable=self.use_design_values,command=lambda:self.refresh_final_equations(force=True),bg=WHITE,font=(KO,10)).pack(side='left',padx=12)
        self.final_body=tk.Frame(self.final_panel,bg=WHITE);self.final_body.pack(fill='x')
        self.label(self.final_body,'실적함 학습 후 선택한 함종의 최종 계수가 수식에 표시됩니다.',10,MUTED).pack(anchor='w')
        self.final_formula_images=[]
        def scroll(event):
            if self.notebook.select()==str(self.model_tab) and self.reg_tabs.select()==str(self.guide_tab):
                steps=(-1 if getattr(event,'num',None)==4 else 1) if getattr(event,'num',None) in (4,5) else (-1 if event.delta>0 else 1)
                canvas.yview_scroll(steps*3,'units');return 'break'
        self.root.bind_all('<MouseWheel>',scroll,add='+')
        self.root.bind_all('<Button-4>',scroll,add='+');self.root.bind_all('<Button-5>',scroll,add='+')
        self.guide_canvas=canvas

    def refresh_final_equations(self,force=False):
        if not self.bundle or self.busy:return
        if self.reg_tabs.select()!=str(self.guide_tab) and not force:return
        from final_equations import final_equation,render_png,variable_description
        cls=self.formula_ship.get()
        if cls not in self.bundle['models']['SWBS100']['classes']:return
        design=None
        if self.use_design_values.get() and self.data is not None and len(self.data)==1 and '연도 적용' in self.data:
            design=self.data.iloc[0].to_dict()
            if design['함분류']!=cls:design=None
        width=min(900,max(550,self.guide_canvas.winfo_width()-70))
        key=(id(self.bundle),cls,self.use_design_values.get(),None if design is None else tuple((k,str(v)) for k,v in design.items()),width)
        if not force and key==self.final_formula_key:return
        self.root.configure(cursor='watch');self.root.update_idletasks()
        try:
            for child in self.final_body.winfo_children():child.destroy()
            self.final_formula_images=[]
            self.label(self.final_body,'계수는 표시용으로 7자리 유효숫자로 반올림합니다. 실제 계산은 원래 정밀도를 사용합니다.',9,MUTED).pack(anchor='w',pady=4)
            if self.use_design_values.get() and design is None:
                self.label(self.final_body,'Weight Estimation에서 같은 함종의 설계 제원을 입력해 예측하면 제원 숫자도 대입됩니다.',9,MUTED).pack(anchor='w')
            for name,m in self.bundle['models'].items():
                self.label(self.final_body,name+' / '+cls,11,TEAL,WHITE,True).pack(anchor='w',pady=(10,0))
                formula=final_equation(name,m,cls,design)
                photo=tk.PhotoImage(data=base64.b64encode(render_png(formula,width))).subsample(2)
                self.final_formula_images.append(photo)
                tk.Label(self.final_body,image=photo,bg=WHITE,anchor='w').pack(anchor='w')
                percent=m.get('year_scale')
                info=variable_description(m,cls)+'\n선택 연도 효과 '+('—' if percent is None else f'{percent*100:.0f}%')+' / '+m.get('selection',{}).get('status','')
                info+='\n'+self.wape_rule(m)
                if m.get('year_unavailable_reason'):info+=' / '+m['year_unavailable_reason']
                tk.Label(self.final_body,text=info,bg=WHITE,fg=MUTED,font=(KO,9),wraplength=width,justify='left').pack(anchor='w')
            self.label(self.final_body,'합산 중량(700계열 제외) = 위 SWBS100~600의 예측 중량 합계',11,INK,WHITE,True).pack(anchor='w',pady=8)
            self.final_formula_key=key
        finally:self.root.configure(cursor='')

    def show_analysis(self):
        self.analysis_dialog=tk.Toplevel(self.root);self.analysis_dialog.title('해석 중')
        self.analysis_dialog.transient(self.root);self.analysis_dialog.resizable(False,False)
        x=self.root.winfo_rootx()+max(0,(self.root.winfo_width()-460)//2)
        y=self.root.winfo_rooty()+max(0,(self.root.winfo_height()-210)//2)
        self.analysis_dialog.geometry(f'460x210+{x}+{y}');self.analysis_dialog.configure(bg=WHITE)
        self.analysis_dialog.protocol('WM_DELETE_WINDOW',lambda:None)
        self.analysis_dialog.grab_set()
        tk.Label(self.analysis_dialog,text='실적함 해석 중...',font=(KO,15,'bold'),fg=TEAL,bg=WHITE).pack(pady=(12,5))
        canvas=tk.Canvas(self.analysis_dialog,width=410,height=50,bg=WHITE,highlightthickness=0);canvas.pack()
        self.analysis_message=tk.StringVar(value='기초 회귀계수를 추정하고 있습니다.')
        tk.Label(self.analysis_dialog,textvariable=self.analysis_message,font=(KO,10),fg=INK,bg=WHITE).pack(pady=4)
        self.analysis_progress=ttk.Progressbar(self.analysis_dialog,maximum=21*len(SCALES),length=410);self.analysis_progress.pack(pady=8)
        self.analysis_tick=0
        self.analysis_started=time.monotonic()
        canvas.create_line(45,33,365,33,fill='#d5e3ea',width=2)
        lens=canvas.create_oval(192,8,218,34,outline=TEAL,width=3)
        handle=canvas.create_line(215,30,227,42,fill=TEAL,width=5)
        def animate():
            if not self.busy or not self.analysis_dialog.winfo_exists():return
            self.analysis_tick+=1
            cx=205+130*math.sin((time.monotonic()-self.analysis_started)*2.5)
            canvas.coords(lens,cx-13,8,cx+13,34)
            canvas.coords(handle,cx+10,30,cx+22,42)
            self.root.after(40,animate)
        animate()

    def close_analysis(self):
        if hasattr(self,'analysis_dialog') and self.analysis_dialog.winfo_exists():
            self.analysis_dialog.grab_release();self.analysis_dialog.destroy()

    def wape_rule(self,m):
        info=m.get('selection',{});initial=info.get('wape_requested_percent');used=info.get('wape_limit_percent')
        if used is None:return 'WAPE 기준: 미적용(이전 모델)'
        actual=m.get('wape_percent',0)
        return f"WAPE {actual:.3f}% ≤ 적용 기준 {used:g}% / 초기 {initial:g}% / "+('1%p 단위 자동 완화' if info.get('wape_relaxed') else '완화 없음')

    def selected_type(self):
        choice=self.ship_filter.get()
        return None if choice=='All Ship Types' else choice

    def update_diagnostics(self):
        target=self.reg_target.get()
        self.target_description.set(target+' ('+TARGET_NAMES.get(target,'Weight Model')+')')
        if not self.bundle:return
        import math
        cls=self.selected_type()
        base=self.bundle['models']['SWBS100']
        view='전체 함정' if cls is None else cls
        self.regression_summary.set(f"조회: {view} / 전체 학습 {base['n']}척 / 기준 함종: {base['classes'][0]} (Dt=1) / 제외 {len(self.bundle.get('excluded_rows',[]))}척\n함종별 적합도는 전체 모형의 해당 함종 자료에서 계산합니다. 순번은 그래프·진단표·엑셀에 보존됩니다.")
        model=self.bundle['models'].get(self.reg_target.get())
        if not model:return
        rows=diagnostic_rows(model,cls);metric=diagnostic_metrics(rows)
        def fmt(v,places=3):return '—' if v is None else f'{v:,.{places}f}'
        self.reg_stats.set(f"Train N {metric['n']}   R² {fmt(metric['r2'],4)}   RMSE {fmt(metric['rmse_tonne'],2)} t   MAPE {fmt(metric['mape_percent'],2)}%   WAPE {fmt(metric['wape_percent'],2)}%")
        self.ship_tree.delete(*self.ship_tree.get_children());self.ship_row_ids={}
        for d in sorted(rows,key=lambda r:abs(r['오차율(%)']),reverse=True):
            item=self.ship_tree.insert('','end',values=(d.get('순번',str(int(d['엑셀 행'])-1)),d['엑셀 행'],d['함종'],d.get('함유형',''),f"{d['Dt']:.5g}",f"{d.get('연도보정값',1):.5g}",f"{d['실제중량']:,.3f}",f"{d['예측중량']:,.3f}",f"{d['잔차']:,.3f}",f"{d['오차율(%)']:.2f}"),tags=('large_error',) if abs(d['오차율(%)'])>10 else ())
            self.ship_row_ids[str(d.get('순번',str(int(d['엑셀 행'])-1)))]=item
        self.metric_tree.delete(*self.metric_tree.get_children());self.coeff_tree.delete(*self.coeff_tree.get_children())
        for name,x in self.bundle['models'].items():
            m=diagnostic_metrics(diagnostic_rows(x,cls))
            self.metric_tree.insert('','end',values=(name,m['n'],fmt(m['rmse_tonne']),fmt(m['mape_percent']),fmt(m['wape_percent']),fmt(x.get('selection',{}).get('wape_limit_percent')),fmt(m['r2'],4),x['parameters'],f"{x['condition_number']:.2f}",'—' if x.get('year_scale') is None else f"{x['year_scale']*100:.0f}"))
            terms=dict(zip(x['terms'],x['coef'][1:1+len(x['terms'])]));offset=1+len(x['terms'])
            for j,ship_type in enumerate(x['classes']):
                if cls is not None and ship_type!=cls:continue
                values=[name,ship_type,x.get('class_counts',{}).get(ship_type,'—'),x['classes'][0],
                        f"{1.0 if j==0 else math.exp(x['coef'][offset+j-1]):.8g}",'—' if x.get('kind')=='additive' else f"{math.exp(x['coef'][0]):.8g}",f"{x['smearing']:.8g}"]
                values += [f"{x['size_alpha'][ship_type]:.8g}" if t=='LBD' and x.get('size_by_type') else f"{terms[t]:.8g}" if t in terms and x.get('kind')!='additive' else '—' for t in ['LBD','LB','L','P','PE','D','B']]
                self.coeff_tree.insert('','end',values=values+[f"{x['additive_coeff']['constant']:.8g}" if x.get('kind')=='additive' else '—',', '.join(t+'='+f'{v:.8g}' for t,v in x.get('additive_coeff',{}).items() if t!='constant') or '—']+[f"{x['year_scale']*100:.0f}" if x.get('year_scale') is not None else '—',f"{x.get('gamma_base',{}).get(ship_type,x['gamma'].get(ship_type,0)):.8g}" if x['year_by_type'] else '—',f"{x['gamma'][ship_type]:.8g}" if x['year_by_type'] else '—', f"{x['mean_years'][ship_type]:.4f}",f"{x['volume_reference']:,.3f}" if x.get('size_by_type') else '—'])
        self.graph_hover.set('함정별 표는 절대 오차율이 큰 순서입니다. |오차율| 10% 초과은 붉은색으로 표시합니다.')
        self.update_year_search()
        self.draw_regression_graphs()

    def update_year_search(self):
        self.search_tree.delete(*self.search_tree.get_children())
        if not self.bundle:return
        m=self.bundle['models'].get(self.reg_target.get(),{})
        if not m.get('year_by_type'):
            self.search_note.set('선택 계열은 연도 항이 없어 비율 탐색 대상이 아닙니다. 100계열은 함종별 규모 지수 αt를 사용합니다.');self.draw_search_graph();return
        info=m.get('selection',{});q=m.get('year_scale',1)
        self.search_note.set(f"{m.get('candidate_label','기존 식')} / 선택 {q*100:.0f}% / WAPE 기준 충족 후 Train MAPE 최소 / 유효 실적함 {m['n']}척 전체 사용.\n탐색 범위 0~200% · 1% 간격. 각 후보에서 나머지 계수와 C를 재추정합니다. 함종 필터와 관계없이 계열 전체에서 선택합니다.")
        self.search_note.set(self.search_note.get()+'\n'+self.wape_rule(m))
        def fmt(v):return '—' if v is None else f'{float(v):,.3f}'
        chosen_item=None
        for r in m.get('year_search',[]):
            item=self.search_tree.insert('','end',values=(r['percent'],fmt(r.get('train_mape')),fmt(r.get('train_wape')),'YES' if r.get('wape_eligible') else 'NO' if r.get('train_wape') is not None else '—',fmt(r.get('train_rmse')),r.get('train_n',m['n']),'YES' if r.get('selected') else '',r.get('error') or ''),tags=('chosen',) if r.get('selected') else ())
            if r.get('selected'):chosen_item=item
        if chosen_item:self.search_tree.see(chosen_item)
        self.draw_search_graph()

    def draw_search_graph(self):
        c=self.search_graph;c.delete('all')
        if not self.bundle:return
        rows=self.bundle['models'].get(self.reg_target.get(),{}).get('year_search',[])
        if not rows:return
        w=max(c.winfo_width(),600);left,right,top,bottom=62,w-25,25,82
        finite=[r[k] for r in rows for k in ['train_mape'] if r.get(k) is not None]
        ceiling=max(finite+[1])*1.1
        c.create_text(left,10,anchor='w',text='Train MAPE (%) / teal: WAPE eligible · gray: excluded',font=(EN,10),fill=INK)
        c.create_line(left,top,left,bottom,right,bottom,fill=MUTED)
        c.create_text(left-7,top,anchor='e',text=f'{ceiling:.2g}',font=(EN,9),fill=MUTED)
        c.create_text(left-7,bottom,anchor='e',text='0',font=(EN,9),fill=MUTED)
        limit=self.bundle['models'].get(self.reg_target.get(),{}).get('selection',{}).get('wape_limit_percent')
        for key,color in [('train_mape',TEAL)]:
            points=[]
            for r in sorted(rows,key=lambda v:v['percent']):
                x=left+r['percent']/MAX_PERCENT*(right-left)
                if r['percent']%20==0:c.create_text(x,bottom+14,text=str(r['percent'])+'%',font=(EN,9),fill=MUTED)
                if r.get(key) is None:continue
                y=bottom-r[key]/ceiling*(bottom-top);points.extend([x,y]);c.create_oval(x-3,y-3,x+3,y+3,fill=color if limit is None or r.get('wape_eligible') else '#9aabb5',outline='')
            if len(points)>=4:c.create_line(*points,fill=color,width=2)

    def select_graph_ship(self, number, message):
        self.graph_hover.set(message)
        item=self.ship_row_ids.get(str(number))
        if item:
            self.reg_tabs.select(self.ship_diagnostics_tab)
            self.ship_tree.selection_set(item);self.ship_tree.focus(item);self.ship_tree.see(item)

    def draw_regression_graphs(self):
        c=self.reg_graph;c.delete('all')
        w,h=max(c.winfo_width(),700),max(c.winfo_height(),220)
        if not self.bundle:return
        rows=diagnostic_rows(self.bundle['models'].get(self.reg_target.get(),{}),self.selected_type())
        if not rows:
            c.create_text(w/2,h/2,text='No training diagnostics available',font=(EN,15),fill=MUTED);return
        actual=[float(r['실제중량']) for r in rows];predicted=[float(r['예측중량']) for r in rows]
        errors=[float(r['오차율(%)']) for r in rows]
        max_weight=max(actual+predicted)*1.05 or 1
        max_error=max(max(abs(e) for e in errors)*1.15,1)
        for side in range(2):
            left=80+side*w/2;right=(side+1)*w/2-25;top=46;bottom=h-48
            c.create_text((left+right)/2,18,text='ACTUAL vs PREDICTED' if side==0 else 'RELATIVE ERROR vs PREDICTED',font=(EN,12,'bold'),fill=INK)
            c.create_line(left,top,left,bottom,right,bottom,fill=MUTED)
            for j in range(5):
                f=j/4;y=bottom-f*(bottom-top);x=left+f*(right-left)
                c.create_line(left,y,right,y,fill='#dce6ec')
                c.create_text(left-7,y,anchor='e',text=f'{max_weight*f:.3g}' if side==0 else f'{(2*f-1)*max_error:.2g}',font=(EN,9),fill=MUTED)
                c.create_text(x,bottom+12,text=f'{max_weight*f:.3g}',font=(EN,9),fill=MUTED)
            c.create_text((left+right)/2,h-12,text='Actual (tonne)' if side==0 else 'Predicted (tonne)',font=(EN,11),fill=INK)
            c.create_text(left-68,(top+bottom)/2,text='Predicted (t)' if side==0 else 'Error (%)',angle=90,font=(EN,10),fill=INK)
            if side==0:c.create_line(left,bottom,right,top,fill=GOLD,width=2,dash=(5,3))
            else:c.create_line(left,(top+bottom)/2,right,(top+bottom)/2,fill=GOLD,width=2,dash=(5,3))
            for i,d in enumerate(rows):
                x=left+(actual[i] if side==0 else predicted[i])/max_weight*(right-left)
                fraction=predicted[i]/max_weight if side==0 else (errors[i]+max_error)/(2*max_error)
                y=bottom-fraction*(bottom-top)
                color='#ae4437' if abs(d['오차율(%)'])>10 else TEAL
                point=c.create_oval(x-3,y-3,x+3,y+3,fill=color,outline=WHITE)
                number=str(d.get('순번',str(int(d['엑셀 행'])-1)))
                items=[point]
                if self.show_ship_ids.get():
                    items.append(c.create_text(x+4,y-6,text=number,anchor='sw',font=(EN,8),fill=color,tags=('ship_number',)))
                message=f"순번 {number} / Excel {d['엑셀 행']}행 / {d['함종']} / {d.get('함유형','')} / Dt {d['Dt']:.5g} / F(Y) {d.get('연도보정값',1):.4g} / 실제 {d['실제중량']:,.2f} t / 예측 {d['예측중량']:,.2f} t / 오차 {d['오차율(%)']:.2f}%"
                for item in items:
                    c.tag_bind(item,'<Enter>',lambda e,t=message:self.graph_hover.set(t))
                    c.tag_bind(item,'<Button-1>',lambda e,n=number,t=message:self.select_graph_ship(n,t))

    def draw_chart(self):
        c=self.chart; c.delete('all'); width=max(c.winfo_width(),240)
        c.create_text(12,16,anchor='w',text='SWBS100–600 / EXCLUDING 700',font=(EN,11,'bold'),fill=INK)
        weights=getattr(self,'last_weights',None)
        if weights is None:
            c.create_text(width/2,100,text='Awaiting design estimate',font=(EN,13,'italic'),fill=MUTED); return
        mx=max(weights) or 1; step=max(23,min(33,(c.winfo_height()-35)/6))
        for i,w in enumerate(weights):
            y=38+i*step
            c.create_text(12,y+7,anchor='w',text=str((i+1)*100),font=(EN,10),fill=INK)
            c.create_rectangle(49,y,width-72,y+14,fill='#e0eaef',outline='')
            c.create_rectangle(49,y,49+(width-121)*w/mx,y+14,fill=TEAL if i%2==0 else '#406d83',outline='')
            c.create_text(width-7,y+7,anchor='e',text=f'{w:,.0f}',font=(EN,10),fill=INK)

    def show(self,text):
        self.log.configure(state='normal'); self.log.delete('1.0','end'); self.log.insert('end',text); self.log.configure(state='disabled')
    def safe(self,action):
        if self.busy: messagebox.showinfo('학습 중','현재 학습이 끝난 후 진행하세요.'); return
        try: action()
        except Exception as e: messagebox.showerror('처리 실패',str(e))
    def choose(self,save=False,ext='.xlsx',initialfile=None):
        fn=filedialog.asksaveasfilename if save else filedialog.askopenfilename
        options = dict(defaultextension=ext,filetypes=[('Excel Workbook','*.xlsx')])
        if initialfile is not None: options['initialfile'] = initialfile
        return fn(**options)
    def require(self):
        if self.bundle is None: raise ValueError('실적함 데이터를 업로드하거나 저장된 모델을 불러오세요.')
    def make_template(self):
        p=self.choose(True,initialfile='Input File Template.xlsx')
        if p: template(p); self.status.set('TEMPLATE SAVED  |  '+str(p)); messagebox.showinfo('양식 저장 완료','첫 번째 시트에 실적함 자료를 입력한 후 Input File Upload를 누르세요.')
    def fit(self):
        try:initial_limit=float(self.wape_initial.get())
        except ValueError:raise ValueError('WAPE 기준을 숫자로 입력하세요.')
        if not math.isfinite(initial_limit) or initial_limit<0:raise ValueError('WAPE 기준은 0 이상의 유한한 수여야 합니다.')
        p=self.choose()
        if not p:return
        self.busy=True; self.status.set('TRAINING  |  0%~200% (1% 간격) 후보 재학습·실적함 적합도 비교 중…')
        state={};self.show_analysis()
        try:
            process,updates=start_regression(p,initial_limit)
            self.training_process=process
        except Exception:
            self.busy=False;self.close_analysis();raise
        exit_seen=[None]
        def finish():
            latest=None
            # Bound the work per UI tick: progress cannot monopolize the event loop.
            for _ in range(16):
                try:msg=updates.get_nowait()
                except queue.Empty:break
                except (EOFError,OSError) as e:
                    state['error']=f'해석 프로세스 통신 오류: {e}';break
                if msg[0]=='progress':latest=msg
                elif msg[0]=='result':state['result']=(msg[1],msg[2]);break
                elif msg[0]=='error':state['error']=msg[1];break
            if latest:
                _,done,total,message=latest
                self.analysis_progress.configure(maximum=total,value=done);self.analysis_message.set(message)
            if not state:
                if process.is_alive():self.root.after(50,finish);return
                if exit_seen[0] is None:exit_seen[0]=time.monotonic()
                if time.monotonic()-exit_seen[0]<1:
                    self.root.after(50,finish);return
                state['error']=f'해석 프로세스가 결과 없이 종료되었습니다. (종료 코드: {process.exitcode})'
            process.join(timeout=0)
            updates.close()
            self.busy=False;self.close_analysis()
            if 'error' in state:
                self.status.set('UPLOAD FAILED  |  기존 모델은 유지됩니다.');messagebox.showerror('학습 실패',state['error']);return
            self.data,self.bundle=state['result'];self.clear_prediction()
            self.refresh_model(Path(p).name)
            skipped = self.data.attrs.get('excluded_rows', [])
            if skipped: messagebox.showinfo('입력 오류 함정 데이터 제외', f'{len(skipped)}척 제외, {len(self.data)}척으로 학습했습니다. 회귀 탭과 결과 엑셀의 제외 행 시트에서 확인하세요.')
            relaxed=sum(bool(m.get('selection',{}).get('wape_relaxed')) for m in self.bundle['models'].values())
            self.status.set(f'MODEL READY  |  WAPE 기준 자동 완화 {relaxed}개 모형. 적용 기준은 Year Effect Search / Fit Statistics에서 확인하세요.' if relaxed else 'MODEL READY  |  WAPE 초기 기준 충족. 신규 설계 제원을 입력하세요.')
            self.reg_target.set('SWBS100');self.update_diagnostics()
            self.notebook.select(self.model_tab);self.reg_tabs.select(0)
        self.root.after(100,finish)
    def clear_prediction(self):
        self.sum_value.set('—'); self.total_value.set('—'); self.last_weights=None
        self.result_tree.delete(*self.result_tree.get_children());self.draw_chart()
        self.prediction_note.set('새 모델이 적용되었습니다. 설계 제원을 입력하고 Estimate Ship Weight를 누르세요.')
    def refresh_model(self,source):
        import math
        m=self.bundle['models']['SWBS100'];ref=m['classes'][0]
        self.update_year_reference()
        self.fleet_info.set(f"{source}\n유효 {m['n']}척 / {len(m['classes'])}개 함종\n기준: {ref} (Dt=1)")
        self.regression_summary.set(f"학습 {m['n']}척 / 기준 함종: {ref} (Dt=1) / 제외 {len(self.bundle.get('excluded_rows', []))}척\n기준 함종은 유효 학습 척수가 가장 많은 비초계함 함종입니다. C는 K와 별도 적용합니다.")
        self.final_formula_key=None
        self.formula_selector.configure(values=m['classes'])
        if self.formula_ship.get() not in m['classes']:self.formula_ship.set(m['classes'][0])
        self.reg_selector.configure(values=list(self.bundle['models']))
        if self.reg_target.get() not in self.bundle['models']:self.reg_target.set('SWBS100')
        classes=m['classes']
        if self.ship_filter.get() not in ['All Ship Types']+classes:self.ship_filter.set('All Ship Types')
        self.type_selector.configure(values=classes,state='disabled' if self.ship_filter.get()=='All Ship Types' else 'readonly')
        self.update_diagnostics()
        self.show('\n'.join(['기준 함종 / Reference: '+ref,
                   '유효 학습 척수 / Valid ships: '+str(m['n']),
                   'Dt는 기준 함종을 1로 한 함종별 상대 계수입니다.',
                   '계수 표의 LBD/LB/L/P/PE 열은 해당 변수의 회귀지수이며 γt는 해당 함종의 10년당 연도 효과이며 함종평균연도를 기준으로 합니다.',
                   '100계열의 LBD 열은 함종별 αt이며, 기준LBD로 나눈 규모 비율에 적용합니다. 함정별 Dt는 Ship Diagnostics에서 확인합니다. 척별 독립 계수를 따로 학습하지 않습니다.',
                   '잔차 = 실제−예측, 오차율 = (예측−실제)/실제 × 100.',
                   'R²·RMSE·MAPE 및 그래프는 학습자료 기준이며 독립 검증 성능이 아닙니다.',
                   'Save Regression Model은 중량 비교·최종 수식·계수·적합도 요약을 결과용 엑셀로 저장합니다.']+self.bundle.get('warnings',[])))
    def save(self):
        self.require();p=self.choose(True,initialfile='Regression Results.xlsx')
        if p:save_regression_results(p,self.bundle);self.status.set('REGRESSION RESULTS SAVED  |  '+p)
    def load(self):
        p=self.choose()
        if not p:return
        bundle=load_model_excel(p)
        self.bundle,self.data=bundle,None;self.clear_prediction();self.refresh_model(Path(p).name)
        self.notebook.select(self.model_tab);self.status.set('EXCEL MODEL LOADED  |  신규 설계함 예측 가능')
    def update_year_reference(self):
        if not self.bundle:return
        cls=SHIP_ALIASES.get(self.inputs['함분류'].get(),self.inputs['함분류'].get())
        means=self.bundle['models']['SWBS100']['mean_years']
        if cls in means:self.estimation_assumption.set(f'함종별 연도 보정: 미입력 시 {cls} 평균 {means[cls]:.2f}년 적용. γt는 모형·함종마다 다릅니다.')
        else:self.estimation_assumption.set('선택 함종은 학습되지 않았습니다. 해당 함종의 실적함 자료가 필요합니다.')

    def design_data(self):
        row={}
        for key,var in self.inputs.items():
            if key=='함분류':row[key]=var.get();continue
            if key=='취역연도' and not var.get().strip():
                self.require()
                cls=SHIP_ALIASES.get(row['함분류'],row['함분류'])
                means=self.bundle['models']['SWBS100']['mean_years']
                if cls not in means:raise ValueError('학습되지 않은 함종: '+cls)
                row[key]=float(means[cls])
                row['연도 적용']='함종별 학습 평균연도'
                continue
            try:value=float(var.get().replace(',','').strip())
            except ValueError:raise ValueError(f'{key}: 숫자를 입력하세요.')
            import math
            if not math.isfinite(value) or (key!='취역연도' and value<=0):raise ValueError(f'{key}: 유효한 양수를 입력하세요.')
            if key=='취역연도' and (not 1800<=value<=2200 or value%1):raise ValueError('취역 예상연도는 1800~2200의 정수여야 합니다.')
            row[key]=value
        if '연도 적용' not in row:row['연도 적용']='입력 취역 예상연도'
        return normalize_types(pd.DataFrame([row]))
    def estimate(self):
        self.require();data=self.design_data();result=predict(data,self.bundle)
        self.data=data;self.formula_ship.set(data.iloc[0]['함분류']);self.use_design_values.set(True);self.final_formula_key=None;self.render_result(result.iloc[0]);self.status.set('ESTIMATION COMPLETE  |  Export Results to Excel로 저장하세요.')
    def render_result(self,row):
        self.sum_value.set(f"{row[TOTAL_SUM_COLUMN]:,.2f} tonne")
        key=TOTAL_MODEL+'_예측_tonne'
        self.total_value.set(f"{row[key]:,.2f} tonne" if key in row else 'Not trained / 미학습')
        self.last_weights=[float(row[f'SWBS{i}_예측_tonne']) for i in range(100,700,100)]
        self.result_tree.delete(*self.result_tree.get_children())
        for i,(name,w) in enumerate(zip(SERIES,self.last_weights)):self.result_tree.insert('','end',values=(str((i+1)*100),name,f'{w:,.2f}'))
        note=f"함종: {row['함분류']} / 700계열 제외 / 취역 예상연도: {row['취역연도']:.2f} ({row.get('연도 적용','입력연도')})  |  "+(row['주의'] or '학습자료의 입력 범위 내')
        if key not in row:note+='  |  원래 경하중량과 공제용 실적 SWBS700을 포함한 자료를 학습하면 직접 회귀를 사용할 수 있습니다.'
        self.prediction_note.set(note);self.draw_chart()
    def report(self):
        self.require()
        if self.data is None:raise ValueError('학습 또는 예측을 먼저 실행하세요.')
        p=self.choose(True)
        if p:export(p,self.data,self.bundle);self.status.set('REPORT SAVED  |  '+p);messagebox.showinfo('저장 완료',p)

if __name__=='__main__':
    import sys
    mp.freeze_support()
    if len(sys.argv)==3 and sys.argv[1]=='--verify-equation-renderer':
        from final_equations import render_png
        Path(sys.argv[2]).write_bytes(render_png(r'\widehat W_{100}=1318.8\left(\frac{LBD}{15000}\right)^{0.8}\times1.2\exp\left[0.03\frac{Y-2005}{10}\right]\times1.01'))
    elif len(sys.argv)==3 and sys.argv[1]=='--verify-regression-engine':
        from additive import verify_engine
        verify_engine(sys.argv[2])
    elif len(sys.argv)==3 and sys.argv[1]=='--verify-training-worker':
        from training_job import verify_worker
        verify_worker(sys.argv[2])
    else:
        root=tk.Tk();App(root);root.mainloop()
