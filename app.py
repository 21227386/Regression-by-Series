import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from model import read_excel, train, predict, export, template, coefficients, validate_model_types

class App:
    def __init__(self, root):
        self.root = root
        self.bundle = None
        self.data = None
        root.title('함정 SWBS 중량 회귀분석')
        root.geometry('1000x650')
        bar = ttk.Frame(root, padding=10); bar.pack(fill='x')
        for label, action in [('입력 양식 저장', self.make_template), ('엑셀 학습', self.fit),
                              ('모델 저장', self.save), ('모델 불러오기', self.load),
                              ('신규 함정 예측', self.forecast), ('결과 엑셀 저장', self.report)]:
            ttk.Button(bar, text=label, command=lambda f=action: self.safe(f)).pack(side='left', padx=3)
        ttk.Label(root, text='중량 tonne · 길이/폭/깊이 m · 추진출력 hp · 전력 kW · 순항속력 kn', padding=10).pack(anchor='w')
        self.log = tk.Text(root, wrap='word', padx=15, pady=15)
        self.log.pack(fill='both', expand=True)
        self.show('1. 입력 양식을 저장해 첫 번째 시트에 함정별 데이터를 입력하세요.\n'
                  '2. 엑셀 학습으로 SWBS별 계수를 추정하세요. 모든 SWBS 중량은 양수여야 합니다.\n'
                  '3. 신규 함정 예측에서는 함분류·취역연도·치수·출력·속력만 입력하면 됩니다.\n'
                  '4. 결과 엑셀과 모델 JSON을 저장하세요.\n\n'
                  '함종: 구축함·호위함·고속함·상륙함·기뢰전함·지원함\n'
                  '초계함·초계정은 호위함으로 통합하여 같은 Dt를 적용합니다.\n'
                  '총중량 기본값은 SWBS100~700 예측 합계입니다.\n'
                  '선택 총중량 독립회귀는 K(LBD)^α Dt exp[γ(Y−평균연도)/10] 모형입니다.\n'
                  '순항속력은 제공식에 없어 회귀에서 제외됩니다.\n'
                  '학습 적합도는 새 함정에 대한 정확도를 보장하지 않습니다.\n'
                  '신규 함급의 독립 검증자료로 예측 성능을 확인하세요.')
    def show(self, text):
        self.log.delete('1.0', 'end'); self.log.insert('end', text)
    def safe(self, action):
        try: action()
        except Exception as e: messagebox.showerror('처리 실패', str(e))
    def choose(self, save=False, ext='.xlsx'):
        fn = filedialog.asksaveasfilename if save else filedialog.askopenfilename
        return fn(defaultextension=ext, filetypes=[('Excel' if ext=='.xlsx' else '모델 JSON', '*'+ext)])
    def require(self):
        if self.bundle is None: raise ValueError('먼저 학습하거나 모델을 불러오세요.')
    def make_template(self):
        p = self.choose(True)
        if p: template(p); messagebox.showinfo('저장 완료', p)
    def fit(self):
        p = self.choose()
        if not p: return
        data = read_excel(p)
        bundle = train(data)
        self.bundle, self.data = bundle, data
        lines = ['학습 완료 — 아래 수치는 학습자료 적합도입니다.\n']
        for n,m in bundle['models'].items():
            lines.append(f"{n}: {m['n']}척, RMSE {m['rmse_tonne']:.3f} tonne, MAPE {m['mape_percent']:.2f}%, R² {m['r2']} (조건수 {m['condition_number']:.1f})")
        lines += ['\n' + w for w in bundle['warnings']]
        lines += ['\n계수 및 함분류별 Dt:\n', coefficients(bundle).to_string(index=False)]
        self.show('\n'.join(lines))
    def save(self):
        self.require(); p = self.choose(True, '.json')
        if p:
            with open(p,'w',encoding='utf-8') as f: json.dump(self.bundle,f,ensure_ascii=False,indent=2,allow_nan=False)
    def load(self):
        p = self.choose(False, '.json')
        if not p: return
        with open(p,encoding='utf-8') as f: b = json.load(f)
        if b.get('version') not in (1, 2) or not all('SWBS'+str(i) in b.get('models',{}) for i in range(100,800,100)):
            raise ValueError('지원하지 않는 모델 파일입니다.')
        validate_model_types(b)
        self.bundle, self.data = b, None
        self.show('모델을 불러왔습니다. 신규 함정 예측을 실행하세요.\n\n'+coefficients(b).to_string(index=False))
    def forecast(self):
        self.require(); p = self.choose()
        if not p: return
        data = read_excel(p, training=False)
        result = predict(data,self.bundle)
        self.data = data
        self.show(result.to_string(index=False)+'\n\n총중량_SWBS합계는 각 계열 예측값의 합계입니다. 단위: tonne.\n결과 엑셀 저장 버튼으로 저장하세요.')
    def report(self):
        self.require()
        if self.data is None: raise ValueError('학습 또는 신규 예측을 먼저 실행하세요.')
        p = self.choose(True)
        if p: export(p,self.data,self.bundle); messagebox.showinfo('저장 완료',p)

if __name__ == '__main__':
    root = tk.Tk(); App(root); root.mainloop()
