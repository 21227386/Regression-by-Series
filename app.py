"""Naval Ship Weight Estimator — native Windows desktop interface."""
import json
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from model import (BASE, SHIP_TYPES, read_excel, train, predict, export, template,
                   coefficients, validate_model_types, normalize_types)

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
        style.configure('Treeview', font=(KO, 10), rowheight=29, background=WHITE, fieldbackground=WHITE)
        style.configure('Treeview.Heading', font=(EN, 11, 'bold'), background='#dce7ed', foreground=INK)
        style.configure('TCombobox', padding=6)
        self.header()
        footer = tk.Frame(root, bg=NAVY, height=30); footer.pack(side='bottom', fill='x')
        self.status = tk.StringVar(value='READY  |  실적함 데이터를 업로드하여 시작하세요.')
        tk.Label(footer, textvariable=self.status, bg=NAVY, fg='#c6d7df', font=(KO, 9)).pack(side='left', padx=14, pady=7)
        tk.Label(footer, text='Copyright by Naval Ship Design Technology Division', font=(EN, 10), bg=NAVY, fg='#d7bd88').pack(side='right', padx=16)
        body = tk.Frame(root, bg=BG); body.pack(fill='both', expand=True)
        sidebar = tk.Frame(body, bg=NAVY, width=290); sidebar.pack(side='left', fill='y'); sidebar.pack_propagate(False)
        self.sidebar(sidebar)
        content = tk.Frame(body, bg=BG, padx=20, pady=15); content.pack(side='left', fill='both', expand=True)
        self.notebook = ttk.Notebook(content); self.notebook.pack(fill='both', expand=True)
        self.design_tab = tk.Frame(self.notebook, bg=WHITE)
        self.model_tab = tk.Frame(self.notebook, bg=WHITE)
        self.notebook.add(self.design_tab, text='Design & Weight Estimation')
        self.notebook.add(self.model_tab, text='Regression Model & Coefficients')
        self.design_panel(); self.model_panel()

    def label(self, parent, text, size=11, color=INK, bg=WHITE, bold=False, english=False):
        return tk.Label(parent, text=text, font=(EN if english else KO, size, 'bold' if bold else 'normal'), fg=color, bg=bg, anchor='w')

    def button(self, parent, text, action, primary=False):
        button = tk.Button(parent, text=text, command=lambda: self.safe(action), font=(EN, 11, 'bold'),
                           bg=TEAL if primary else '#e4edf2', fg=WHITE if primary else INK,
                           activebackground='#116d76' if primary else '#cfdee7', activeforeground=WHITE if primary else INK,
                           relief='raised', bd=3, cursor='hand2', padx=12, pady=9)
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
        inner = tk.Frame(p, bg=NAVY, padx=18, pady=20); inner.pack(fill='both', expand=True)
        self.label(inner, '01  /  REFERENCE FLEET', 13, GOLD, NAVY, True, True).pack(anchor='w')
        self.label(inner, '실적함 데이터 · 자동 회귀 학습', 10, '#d2e1e8', NAVY).pack(anchor='w', pady=(6,12))
        self.button(inner, 'Download Input File Template', self.make_template).pack(fill='x', pady=5)
        self.button(inner, 'Input File Upload', self.fit, True).pack(fill='x', pady=5)
        self.fleet_info = tk.StringVar(value='아직 학습된 모델이 없습니다.')
        tk.Label(inner, textvariable=self.fleet_info, wraplength=245, justify='left', bg=NAVY, fg='#b8cdd8', font=(KO,10)).pack(anchor='w', pady=12)
        self.label(inner, '02  /  MODEL LIBRARY', 13, GOLD, NAVY, True, True).pack(anchor='w', pady=(18,8))
        self.button(inner, 'Save Regression Model', self.save).pack(fill='x', pady=5)
        self.button(inner, 'Load Regression Model', self.load).pack(fill='x', pady=5)
        self.label(inner, '03  /  REPORTS', 13, GOLD, NAVY, True, True).pack(anchor='w', pady=(22,8))
        self.button(inner, 'Export Results to Excel', self.report).pack(fill='x', pady=5)
        self.button(inner, 'Batch Design File Upload', self.forecast).pack(fill='x', pady=5)
        tk.Label(inner, text='업데이트 학습 안내\n기존 실적함 + 추가 실적함을 합친\n전체 엑셀을 다시 업로드하세요.\n\n초계함·초계정 → 호위함 계수 적용\n중량: tonne / 치수: m\n마력: hp / 전력: kW / 속력: kn', bg=NAVY, fg='#b8cdd8', font=(KO,9), justify='left').pack(anchor='w', pady=20)

    def design_panel(self):
        panel = tk.Frame(self.design_tab, bg=WHITE, padx=20, pady=15); panel.pack(fill='both', expand=True)
        self.label(panel, 'New Ship Design', 21, bold=True, english=True).pack(anchor='w')
        self.label(panel, '설계 제원을 입력하면 학습된 계수로 경하중량과 계열별 중량을 추정합니다.', 10, MUTED).pack(anchor='w', pady=(3,12))
        form = tk.Frame(panel, bg=WHITE); form.pack(fill='x')
        self.inputs = {}
        fields = [('함분류','Ship Type / 함종'), ('취역연도','Commissioning Year / 취역연도'),
                  ('길이','Length L / 길이 (m)'), ('폭','Beam B / 폭 (m)'), ('깊이','Depth D / 깊이 (m)'),
                  ('소요마력','Propulsion Power / 소요마력 (hp)'), ('소요전력','Electrical Power / 소요전력 (kW)'), ('순항속력','Cruising Speed / 순항속력 (kn)')]
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
        self.label(row,'속력은 기록용 입력이며 현재 회귀식에는 사용되지 않습니다.',9,MUTED).pack(side='left',padx=14)
        cards=tk.Frame(panel,bg=WHITE); cards.pack(fill='x',pady=12)
        self.sum_value=tk.StringVar(value='—'); self.total_value=tk.StringVar(value='—')
        for title,var,color in [('Series Sum / 계열 합산 중량',self.sum_value,TEAL),('Direct Regression / 경하중량 직접 회귀',self.total_value,INK)]:
            card=tk.Frame(cards,bg='#edf4f6',highlightbackground='#d3e1e7',highlightthickness=1,padx=14,pady=10); card.pack(side='left',fill='both',expand=True,padx=(0,10))
            self.label(card,title,11,color,'#edf4f6',True,True).pack(anchor='w')
            tk.Label(card,textvariable=var,font=(EN,25,'bold'),fg=color,bg='#edf4f6').pack(anchor='w')
        lower=tk.Frame(panel,bg=WHITE); lower.pack(fill='both',expand=True)
        self.result_tree=self.table(lower,['SWBS','System / 계열','Weight (tonne)'],[85,230,130]); self.result_tree.pack(side='left',fill='both',expand=True)
        self.chart=tk.Canvas(lower,bg='#f5f8fa',width=290,height=205,highlightthickness=0); self.chart.pack(side='right',fill='both',expand=True,padx=(12,0))
        self.chart.bind('<Configure>',lambda e:self.draw_chart())
        self.last_weights=None
        self.prediction_note=tk.StringVar(value='모델 학습 또는 불러오기 후 예측할 수 있습니다. 학습 적합도는 신규 함정 예측 정확도를 보장하지 않습니다.')
        tk.Label(panel,textvariable=self.prediction_note,bg=WHITE,fg=MUTED,font=(KO,9),wraplength=880,justify='left',anchor='w').pack(fill='x',pady=(10,0))

    def table(self,parent,cols,widths):
        tree=ttk.Treeview(parent,columns=cols,show='headings',height=7)
        for c,w in zip(cols,widths): tree.heading(c,text=c); tree.column(c,width=w,minwidth=60,anchor='w' if c==cols[1] else 'center')
        return tree

    def model_panel(self):
        p=tk.Frame(self.model_tab,bg=WHITE,padx=20,pady=18); p.pack(fill='both',expand=True)
        self.label(p,'Regression Model & Coefficients',21,bold=True,english=True).pack(anchor='w')
        self.label(p,'C(역변환 편향 보정)는 K와 구분하여 적용합니다. 아래 적합도는 학습자료 기준입니다.',10,MUTED).pack(anchor='w',pady=8)
        self.metric_tree=self.table(p,['Target','Ships','RMSE (tonne)','MAPE (%)','R²'],[180,70,150,130,120]); self.metric_tree.pack(fill='x',pady=8)
        box=tk.Frame(p,bg=WHITE); box.pack(fill='both',expand=True)
        self.log=tk.Text(box,wrap='none',font=(EN,11),fg=INK,bg='#f3f7fa',relief='flat',padx=12,pady=12)
        sy=ttk.Scrollbar(box,orient='vertical',command=self.log.yview); sx=ttk.Scrollbar(box,orient='horizontal',command=self.log.xview)
        self.log.configure(yscrollcommand=sy.set,xscrollcommand=sx.set)
        sy.pack(side='right',fill='y'); sx.pack(side='bottom',fill='x'); self.log.pack(fill='both',expand=True)
        self.show('Input File Upload로 전체 실적함 엑셀을 선택하면 자동으로 회귀계수를 추정합니다.\n\nWtotal = K (LBD)^α P^β Dt exp[γ(Y−Ȳ)/10] C\n\n실제 함정 데이터로 학습하기 전에는 예측값을 표시하지 않습니다.')

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
    def choose(self,save=False,ext='.xlsx'):
        fn=filedialog.asksaveasfilename if save else filedialog.askopenfilename
        return fn(defaultextension=ext,filetypes=[('Excel' if ext=='.xlsx' else 'Model JSON','*'+ext)])
    def require(self):
        if self.bundle is None: raise ValueError('실적함 데이터를 업로드하거나 저장된 모델을 불러오세요.')
    def make_template(self):
        p=self.choose(True)
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
            self.status.set('MODEL READY  |  계수 추정 완료. 신규 설계 제원을 입력하세요.')
            self.notebook.select(self.model_tab)
        self.root.after(100,finish)
    def clear_prediction(self):
        self.sum_value.set('—'); self.total_value.set('—'); self.last_weights=None
        self.result_tree.delete(*self.result_tree.get_children());self.draw_chart()
        self.prediction_note.set('새 모델이 적용되었습니다. 설계 제원을 입력하고 Estimate Ship Weight를 누르세요.')
    def refresh_model(self,source):
        m=self.bundle['models']['SWBS100']
        self.fleet_info.set(f"{source}\n학습자료 {m['n']}척 / {len(m['classes'])}개 함종\n"+', '.join(m['classes']))
        self.metric_tree.delete(*self.metric_tree.get_children())
        for n,x in self.bundle['models'].items():
            r='—' if x['r2'] is None else f"{x['r2']:.4f}"
            self.metric_tree.insert('','end',values=(n,x['n'],f"{x['rmse_tonne']:,.2f}",f"{x['mape_percent']:.2f}",r))
        text=['학습 적합도는 신규 함급 검증 성능이 아닙니다.','K / 회귀지수 / 함종별 Dt / 기준연도 / 역변환 보정계수 C를 아래에 표시합니다.']+self.bundle['warnings']
        for n,m in self.bundle['models'].items():
            coef=m['coef'];terms=m['terms']; p=1+len(terms)
            import math
            text.extend(['',f'[{n}]',f"K = {math.exp(coef[0]):.8g}    C = {m['smearing']:.8g}    기준연도 = {m['mean_year']:.4f}",
                         '   '.join(f'{t} 계수 = {v:.8g}' for t,v in zip(terms,coef[1:p])),
                         '   '.join(f'Dt({c}) = {1.0 if j==0 else math.exp(coef[p+j-1]):.8g}' for j,c in enumerate(m['classes'])),
                         f"조건수 = {m['condition_number']:.2f}"])
            if n=='총중량_독립회귀' and 'P' not in terms:text.append('이전 모델: 소요마력 미포함. 새 식을 적용하려면 다시 학습하세요.')
        self.show('\n'.join(text))
    def save(self):
        self.require();p=self.choose(True,'.json')
        if p:
            with open(p,'w',encoding='utf-8') as f:json.dump(self.bundle,f,ensure_ascii=False,indent=2,allow_nan=False)
            self.status.set('MODEL SAVED  |  '+p)
    def load(self):
        p=self.choose(False,'.json')
        if not p:return
        with open(p,encoding='utf-8') as f:b=json.load(f)
        if b.get('version') not in (1,2) or not all('SWBS'+str(i) in b.get('models',{}) for i in range(100,800,100)):raise ValueError('지원하지 않는 모델 파일입니다.')
        validate_model_types(b);self.bundle,self.data=b,None;self.clear_prediction();self.refresh_model(Path(p).name)
        self.status.set('MODEL LOADED  |  신규 설계함 예측 가능')
    def design_data(self):
        row={}
        for key,var in self.inputs.items():
            if key=='함분류':row[key]=var.get();continue
            try:value=float(var.get().replace(',','').strip())
            except ValueError:raise ValueError(f'{key}: 숫자를 입력하세요.')
            import math
            if not math.isfinite(value) or (key!='취역연도' and value<=0):raise ValueError(f'{key}: 유효한 양수를 입력하세요.')
            if key=='취역연도' and (not 1800<=value<=2200 or value%1):raise ValueError('취역연도는 1800~2200의 정수여야 합니다.')
            row[key]=value
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
        note=f"적용 함종: {row['함분류']}  |  "+(row['주의'] or '학습자료의 입력 범위 내')
        if key not in row:note+='  |  실제 총중량을 포함한 실적함 자료를 학습하면 직접 회귀를 사용할 수 있습니다.'
        self.prediction_note.set(note);self.draw_chart()
    def forecast(self):
        self.require();p=self.choose()
        if not p:return
        data=read_excel(p,training=False);result=predict(data,self.bundle);self.data=data
        self.render_result(result.iloc[0]);self.notebook.select(self.design_tab)
        self.status.set(f'BATCH READY  |  {len(data)}척 예측 완료. 화면은 첫 함정, 엑셀은 전체 함정 결과입니다.')
    def report(self):
        self.require()
        if self.data is None:raise ValueError('학습 또는 예측을 먼저 실행하세요.')
        p=self.choose(True)
        if p:export(p,self.data,self.bundle);self.status.set('REPORT SAVED  |  '+p);messagebox.showinfo('저장 완료',p)

if __name__=='__main__':
    root=tk.Tk();App(root);root.mainloop()
