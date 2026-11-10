"""Naval Ship Weight Estimator — native Windows desktop interface."""
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from equation_guide import GUIDE_ROWS, TABLE_EXPLANATION
from model import (BASE, SHIP_TYPES, read_excel, train, predict, export, template,
                   coefficients, validate_model_types, normalize_types, save_model_excel, load_model_excel, diagnostic_rows, diagnostic_metrics)

TITLE = 'Naval Ship Weight Estimator(Total & Series)'
NAVY, INK, TEAL, GOLD = '#102638', '#19384a', '#087f87', '#c69c55'
BG, WHITE, MUTED = '#edf2f5', '#ffffff', '#607887'
EN, KO = 'Times New Roman', '맑은 고딕'
SERIES = ['Hull Structure', 'Propulsion', 'Electrical', 'Command & Surveillance',
          'Auxiliary Systems', 'Outfit & Furnishings', 'Armament']

class App:
    def __init__(self, root):
        self.root, self.bundle, self.data = root, None, None
        self.busy = False
        root.title(TITLE); root.geometry('1380x900'); root.minsize(1180, 800)
        root.configure(bg=BG)
        root.option_add('*Font', (KO, 10))
        style = ttk.Style(root); style.theme_use('clam')
        style.configure('TNotebook', background=BG, borderwidth=0)
        style.configure('TNotebook.Tab', font=(EN, 12, 'bold'), padding=(22, 10), background='#dce6ec')
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

    def label(self, parent, text, size=11, color=INK, bg=WHITE, bold=False, english=False):
        return tk.Label(parent, text=text, font=(EN if english else KO, size, 'bold' if bold else 'normal'), fg=color, bg=bg, anchor='w')

    def button(self, parent, text, action, primary=False):
        button = tk.Button(parent, text=text, command=lambda: self.safe(action), font=(EN, 11, 'bold'),
                           bg=TEAL if primary else '#e4edf2', fg=WHITE if primary else INK,
                           activebackground='#116d76' if primary else '#cfdee7', activeforeground=WHITE if primary else INK,
                           relief='raised', bd=3, cursor='hand2', padx=12, pady=5)
        return button

    def header(self):
        h = tk.Frame(self.root, bg=NAVY, height=146); h.pack(fill='x'); h.pack_propagate(False)
        words = tk.Frame(h, bg=NAVY); words.pack(side='left', padx=28, pady=10)
        self.label(words, 'NAVAL SHIP DESIGN  /  ENGINEERING ANALYTICS', 10, GOLD, NAVY, True, True).pack(anchor='w')
        self.label(words, 'Naval Ship Weight Estimator', 28, WHITE, NAVY, True, True).pack(anchor='w', pady=(8, 0))
        self.label(words, '(Total & Series)', 18, '#a9d6da', NAVY, False, True).pack(anchor='w')
        self.label(words, '함정 중량 추정 및 SWBS 계열별 중량 분석', 10, '#b8cdd8', NAVY).pack(anchor='w', pady=(5, 0))
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
        tk.Label(inner, text='기존 + 추가 자료를 합쳐 재업로드\n초계함·초계정 → 호위함 계수\n중량 tonne / 치수 m\n마력 hp / 전력 kW / 속력 kn', bg=NAVY, fg='#b8cdd8', font=(KO,9), justify='left').pack(anchor='w', pady=5)

    def design_panel(self):
        panel = tk.Frame(self.design_tab, bg=WHITE, padx=20, pady=15); panel.pack(fill='both', expand=True)
        self.label(panel, 'Weight Estimation', 21, bold=True, english=True).pack(anchor='w')
        self.label(panel, '함종·길이·폭·깊이·소요마력·소요전력으로 SWBS100~700 및 경하중량을 추정합니다.', 10, MUTED).pack(anchor='w', pady=(3,12))
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
                e=ttk.Combobox(cell,textvariable=var,values=SHIP_TYPES+['초계함','초계정'],state='readonly',width=15,font=(KO,11)); var.set('호위함')
            else:
                e=tk.Entry(cell,textvariable=var,font=(EN,14),bg='#f3f7fa',fg=INK,relief='sunken',bd=2,width=15)
            e.pack(fill='x',ipady=4,pady=(4,0))
        row = tk.Frame(panel,bg=WHITE); row.pack(fill='x',pady=(10,4))
        self.estimate_button=self.button(row,'Estimate Ship Weight',self.estimate,True); self.estimate_button.pack(side='left')
        self.estimation_assumption=tk.StringVar(value='취역 예상연도 미입력 시 학습 평균연도를 적용합니다. 순항속력은 현재 식에 사용하지 않습니다.')
        tk.Label(row,textvariable=self.estimation_assumption,font=(KO,9),fg=MUTED,bg=WHITE,wraplength=600,justify='left').pack(side='left',padx=14)
        cards=tk.Frame(panel,bg=WHITE); cards.pack(fill='x',pady=12)
        self.sum_value=tk.StringVar(value='—'); self.total_value=tk.StringVar(value='—')
        for title,var,color in [('SWBS100–700 Sum / 계열별 합산 중량',self.sum_value,TEAL),('Lightship Regression / 전체 경하중량 직접 추정',self.total_value,INK)]:
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
        self.reg_stats=tk.StringVar(value='R² / RMSE / MAPE —')
        tk.Label(select,textvariable=self.reg_stats,bg=WHITE,fg=TEAL,font=(EN,11,'bold')).pack(side='left',padx=12)
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
        self.reg_graph=tk.Canvas(overview,bg='#f3f7fa',height=320,highlightthickness=0)
        # Pack the graph after its information footer so hover details remain visible.
        self.reg_graph.bind('<Configure>',lambda e:self.draw_regression_graphs())
        self.graph_hover=tk.StringVar(value='실적함 데이터를 학습하거나 모델 엑셀을 불러오세요.')
        self.hover_label=tk.Label(overview,textvariable=self.graph_hover,bg=WHITE,fg=INK,font=(KO,9),anchor='w',wraplength=930)
        self.hover_label.pack(side='bottom',fill='x',pady=4)
        self.label(overview,'점의 순번으로 함정을 식별합니다. 마우스: 상세 정보 / 클릭: 진단표 이동. 적합도는 학습자료 기준입니다.',9,MUTED).pack(side='bottom',anchor='w',pady=3)
        self.label(overview,'왼쪽: 실제·예측 중량(tonne) / 오른쪽: 예측 중량 대비 오차율(%)',9,MUTED).pack(side='bottom',anchor='w',pady=3)
        self.reg_graph.pack(fill='both',expand=True,pady=8)
        cols=['계열','함종','학습척수','기준함종','Dt','K','C','LBD','LB','L','P','PE','γ','기준연도']
        self.coeff_tree=self.scroll_table(coefs,cols,[160,85,75,85,100,120,100,100,100,100,100,100,100,100])
        self.ship_diagnostics_tab=ships
        self.ship_tree=self.scroll_table(ships,['Ship No.','Excel Row','Ship Type','Subtype','Dt','Actual (t)','Predicted (t)','Residual (t)','Error (%)'],[85,80,90,100,85,120,120,120,100])
        self.ship_tree.tag_configure('large_error',foreground='#ae4437')
        self.metric_tree=self.scroll_table(stats,['Target','Ships','RMSE (tonne)','MAPE (%)','R²','Parameters','Condition'],[180,65,135,100,100,90,120])
        self.log=tk.Text(notes,wrap='word',font=(EN,11),fg=INK,bg='#f3f7fa',relief='flat',padx=12,pady=12)
        sy=ttk.Scrollbar(notes,orient='vertical',command=self.log.yview);self.log.configure(yscrollcommand=sy.set)
        sy.pack(side='right',fill='y');self.log.pack(fill='both',expand=True)
        self.show('실적함 업로드 후 자동 학습합니다.\n기준 함종은 오류 행 제외 및 초계함·초계정의 호위함 통합 후 척수가 가장 많은 함종입니다.\n동률이면 구축함·호위함·고속함·상륙함·기뢰전함·지원함 순입니다.\nK와 역변환 보정계수 C는 분리하여 표시합니다.')

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
        tk.Label(content,text='왼쪽은 수식 이미지, 오른쪽은 해당 변수와 계수의 의미입니다. 모든 중량은 tonne입니다.',
                 font=(KO,10),fg=MUTED,bg='#edf2f5',anchor='w').pack(fill='x',pady=(0,12))
        self.equation_images=[];self.equation_description_labels=[]
        assets=Path(__file__).resolve().parent/'assets'/'equations'
        for key,title,equation,description in GUIDE_ROWS:
            card=tk.Frame(content,bg=WHITE,highlightbackground='#d0dfe7',highlightthickness=1,padx=12,pady=10)
            card.pack(fill='x',pady=(0,10));card.columnconfigure(1,weight=1)
            self.label(card,title,13,TEAL,WHITE,True,True).grid(row=0,column=0,columnspan=2,sticky='w',pady=(0,7))
            photo=tk.PhotoImage(file=str(assets/(key+'.png'))).subsample(2)
            self.equation_images.append(photo)
            tk.Label(card,image=photo,bg=WHITE,anchor='w').grid(row=1,column=0,sticky='nw',padx=(0,12))
            note=tk.Label(card,text=description,font=(KO,10),fg=INK,bg=WHITE,justify='left',anchor='nw',wraplength=340)
            note.grid(row=1,column=1,sticky='nw')
            self.equation_description_labels.append(note)
            card.bind('<Configure>',lambda e,label=note:label.configure(wraplength=max(180,e.width-555)))
        for title,description in TABLE_EXPLANATION:
            card=tk.Frame(content,bg=WHITE,padx=14,pady=12);card.pack(fill='x',pady=(0,10))
            self.label(card,title,12,TEAL,WHITE,True).pack(anchor='w',pady=(0,5))
            note=tk.Label(card,text=description,font=(KO,10),fg=INK,bg=WHITE,justify='left',anchor='w',wraplength=900)
            note.pack(fill='x');card.bind('<Configure>',lambda e,label=note:label.configure(wraplength=max(200,e.width-32)))
        def scroll(event):
            if self.notebook.select()==str(self.model_tab) and self.reg_tabs.select()==str(self.guide_tab):
                steps=(-1 if getattr(event,'num',None)==4 else 1) if getattr(event,'num',None) in (4,5) else (-1 if event.delta>0 else 1)
                canvas.yview_scroll(steps*3,'units');return 'break'
        self.root.bind_all('<MouseWheel>',scroll,add='+')
        self.root.bind_all('<Button-4>',scroll,add='+');self.root.bind_all('<Button-5>',scroll,add='+')
        self.guide_canvas=canvas

    def selected_type(self):
        choice=self.ship_filter.get()
        return None if choice=='All Ship Types' else choice

    def update_diagnostics(self):
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
        self.reg_stats.set(f"N {metric['n']}   R² {fmt(metric['r2'],4)}   RMSE {fmt(metric['rmse_tonne'],2)} t   MAPE {fmt(metric['mape_percent'],2)}%")
        self.ship_tree.delete(*self.ship_tree.get_children());self.ship_row_ids={}
        for d in sorted(rows,key=lambda r:abs(r['오차율(%)']),reverse=True):
            item=self.ship_tree.insert('','end',values=(d.get('순번',str(int(d['엑셀 행'])-1)),d['엑셀 행'],d['함종'],d.get('함유형',''),f"{d['Dt']:.5g}",f"{d['실제중량']:,.3f}",f"{d['예측중량']:,.3f}",f"{d['잔차']:,.3f}",f"{d['오차율(%)']:.2f}"),tags=('large_error',) if abs(d['오차율(%)'])>=20 else ())
            self.ship_row_ids[str(d.get('순번',str(int(d['엑셀 행'])-1)))]=item
        self.metric_tree.delete(*self.metric_tree.get_children());self.coeff_tree.delete(*self.coeff_tree.get_children())
        for name,x in self.bundle['models'].items():
            m=diagnostic_metrics(diagnostic_rows(x,cls))
            self.metric_tree.insert('','end',values=(name,m['n'],fmt(m['rmse_tonne']),fmt(m['mape_percent']),fmt(m['r2'],4),x['parameters'],f"{x['condition_number']:.2f}"))
            terms=dict(zip(x['terms'],x['coef'][1:1+len(x['terms'])]));offset=1+len(x['terms'])
            for j,ship_type in enumerate(x['classes']):
                if cls is not None and ship_type!=cls:continue
                values=[name,ship_type,x.get('class_counts',{}).get(ship_type,'—'),x['classes'][0],
                        f"{1.0 if j==0 else math.exp(x['coef'][offset+j-1]):.8g}",f"{math.exp(x['coef'][0]):.8g}",f"{x['smearing']:.8g}"]
                values += [f"{terms[t]:.8g}" if t in terms else '—' for t in ['LBD','LB','L','P','PE','year']]
                self.coeff_tree.insert('','end',values=values+[f"{x['mean_year']:.4f}"])
        self.graph_hover.set('함정별 표는 절대 오차율이 큰 순서입니다. |오차율| 20% 이상은 붉은색으로 표시합니다.')
        self.draw_regression_graphs()

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
                color='#ae4437' if abs(d['오차율(%)'])>=20 else TEAL
                point=c.create_oval(x-3,y-3,x+3,y+3,fill=color,outline=WHITE)
                number=str(d.get('순번',str(int(d['엑셀 행'])-1)))
                items=[point]
                if self.show_ship_ids.get():
                    items.append(c.create_text(x+4,y-6,text=number,anchor='sw',font=(EN,8),fill=color,tags=('ship_number',)))
                message=f"순번 {number} / Excel {d['엑셀 행']}행 / {d['함종']} / {d.get('함유형','')} / Dt {d['Dt']:.5g} / 실제 {d['실제중량']:,.2f} t / 예측 {d['예측중량']:,.2f} t / 오차 {d['오차율(%)']:.2f}%"
                for item in items:
                    c.tag_bind(item,'<Enter>',lambda e,t=message:self.graph_hover.set(t))
                    c.tag_bind(item,'<Button-1>',lambda e,n=number,t=message:self.select_graph_ship(n,t))

    def draw_chart(self):
        c=self.chart; c.delete('all'); width=max(c.winfo_width(),240)
        c.create_text(12,16,anchor='w',text='SWBS WEIGHT DISTRIBUTION',font=(EN,11,'bold'),fill=INK)
        weights=getattr(self,'last_weights',None)
        if weights is None:
            c.create_text(width/2,100,text='Awaiting design estimate',font=(EN,13,'italic'),fill=MUTED); return
        mx=max(weights) or 1; step=max(23,min(33,(c.winfo_height()-35)/7))
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
        p=self.choose()
        if not p:return
        self.busy=True; self.status.set('TRAINING  |  입력 검증 및 회귀계수 추정 중…')
        state={}
        def worker():
            try:
                data=read_excel(p); state['result']=(data,train(data))
            except Exception as e:state['error']=str(e)
        thread=threading.Thread(target=worker,daemon=True);thread.start()
        def finish():
            if thread.is_alive(): self.root.after(100,finish);return
            self.busy=False
            if 'error' in state:
                self.status.set('UPLOAD FAILED  |  기존 모델은 유지됩니다.');messagebox.showerror('학습 실패',state['error']);return
            self.data,self.bundle=state['result'];self.clear_prediction()
            self.refresh_model(Path(p).name)
            skipped = self.data.attrs.get('excluded_rows', [])
            if skipped: messagebox.showinfo('입력 오류 함정 데이터 제외', f'{len(skipped)}척 제외, {len(self.data)}척으로 학습했습니다. 회귀 탭과 결과 엑셀의 제외 행 시트에서 확인하세요.')
            self.status.set('MODEL READY  |  계수 추정 완료. 신규 설계 제원을 입력하세요.')
            self.notebook.select(self.model_tab)
        self.root.after(100,finish)
    def clear_prediction(self):
        self.sum_value.set('—'); self.total_value.set('—'); self.last_weights=None
        self.result_tree.delete(*self.result_tree.get_children());self.draw_chart()
        self.prediction_note.set('새 모델이 적용되었습니다. 설계 제원을 입력하고 Estimate Ship Weight를 누르세요.')
    def refresh_model(self,source):
        import math
        m=self.bundle['models']['SWBS100'];ref=m['classes'][0]
        self.estimation_assumption.set(f"취역 예상연도 미입력 시 학습 평균 {m['mean_year']:.2f}년을 적용합니다. 순항속력은 식에 사용하지 않습니다.")
        self.fleet_info.set(f"{source}\n유효 {m['n']}척 / {len(m['classes'])}개 함종\n기준: {ref} (Dt=1)")
        self.regression_summary.set(f"학습 {m['n']}척 / 기준 함종: {ref} (Dt=1) / 제외 {len(self.bundle.get('excluded_rows', []))}척\n기준 함종은 유효 학습 척수가 가장 많은 함종입니다. C는 K와 별도 적용합니다.")
        self.reg_selector.configure(values=list(self.bundle['models']))
        if self.reg_target.get() not in self.bundle['models']:self.reg_target.set('SWBS100')
        classes=m['classes']
        if self.ship_filter.get() not in ['All Ship Types']+classes:self.ship_filter.set('All Ship Types')
        self.type_selector.configure(values=classes,state='disabled' if self.ship_filter.get()=='All Ship Types' else 'readonly')
        self.update_diagnostics()
        self.show('\n'.join(['기준 함종 / Reference: '+ref,
                   '유효 학습 척수 / Valid ships: '+str(m['n']),
                   'Dt는 기준 함종을 1로 한 함종별 상대 계수입니다.',
                   '계수 표의 LBD/LB/L/P/PE 열은 해당 변수의 회귀지수이며 γ는 10년당 연도 효과입니다.',
                   '함정별 Dt는 Ship Diagnostics에서 확인합니다. 척별 독립 계수를 따로 학습하지 않습니다.',
                   '잔차 = 실제−예측, 오차율 = (예측−실제)/실제 × 100.',
                   'R²·RMSE·MAPE 및 그래프는 학습자료 기준이며 독립 검증 성능이 아닙니다.',
                   '엑셀 모델에는 전체 정밀도의 계수·학습 범위·진단 결과가 저장됩니다.']+self.bundle.get('warnings',[])))
    def save(self):
        self.require();p=self.choose(True,initialfile='Regression Model.xlsx')
        if p:save_model_excel(p,self.bundle);self.status.set('MODEL SAVED  |  '+p)
    def load(self):
        p=self.choose()
        if not p:return
        bundle=load_model_excel(p)
        self.bundle,self.data=bundle,None;self.clear_prediction();self.refresh_model(Path(p).name)
        self.notebook.select(self.model_tab);self.status.set('EXCEL MODEL LOADED  |  신규 설계함 예측 가능')
    def design_data(self):
        row={}
        for key,var in self.inputs.items():
            if key=='함분류':row[key]=var.get();continue
            if key=='취역연도' and not var.get().strip():
                self.require()
                row[key]=float(self.bundle['models']['SWBS100']['mean_year'])
                row['연도 적용']='학습 평균연도'
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
        self.data=data;self.render_result(result.iloc[0]);self.status.set('ESTIMATION COMPLETE  |  Export Results to Excel로 저장하세요.')
    def render_result(self,row):
        self.sum_value.set(f"{row['총중량_SWBS합계_tonne']:,.2f} tonne")
        key='총중량_독립회귀_예측_tonne'
        self.total_value.set(f"{row[key]:,.2f} tonne" if key in row else 'Not trained / 미학습')
        self.last_weights=[float(row[f'SWBS{i}_예측_tonne']) for i in range(100,800,100)]
        self.result_tree.delete(*self.result_tree.get_children())
        for i,(name,w) in enumerate(zip(SERIES,self.last_weights)):self.result_tree.insert('','end',values=(str((i+1)*100),name,f'{w:,.2f}'))
        note=f"함종: {row['함분류']} / 취역 예상연도: {row['취역연도']:.2f} ({row.get('연도 적용','입력연도')})  |  "+(row['주의'] or '학습자료의 입력 범위 내')
        if key not in row:note+='  |  실제 총중량을 포함한 실적함 자료를 학습하면 직접 회귀를 사용할 수 있습니다.'
        self.prediction_note.set(note);self.draw_chart()
    def report(self):
        self.require()
        if self.data is None:raise ValueError('학습 또는 예측을 먼저 실행하세요.')
        p=self.choose(True)
        if p:export(p,self.data,self.bundle);self.status.set('REPORT SAVED  |  '+p);messagebox.showinfo('저장 완료',p)

if __name__=='__main__':
    root=tk.Tk();App(root);root.mainloop()
