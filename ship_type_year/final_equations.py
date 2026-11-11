"""Numerical, typeset equations for the final fitted model (display only)."""
import io
import os
import tempfile
from pathlib import Path
import numpy as np


def number(value):
    s=f'{float(value):.7g}'
    if 'e' in s:
        a,b=s.split('e');return a+r'\times10^{'+str(int(b))+'}'
    return s

def final_equation(name,model,ship_type,design=None):
    from model import TOTAL_MODEL,predict_one
    if ship_type not in model['classes']:raise ValueError('학습되지 않은 함종입니다.')
    terms=dict(zip(model['terms'],model['coef'][1:1+len(model['terms'])]))
    j=model['classes'].index(ship_type);offset=1+len(model['terms'])
    dt=1.0 if j==0 else float(np.exp(model['coef'][offset+j-1]))
    factors=[number(np.exp(model['coef'][0]))]
    if model.get('kind')=='additive':
        add=model['additive_coeff'];parts=[number(add['constant'])]
        for term in model['terms']:
            value={'LBD':'LBD','LB':'LB','L':'L','B':'B','D':'D','P':'P','PE':r'P_E'}[term]
            if design is not None:
                value=number({'LBD':design['길이']*design['폭']*design['깊이'],'LB':design['길이']*design['폭'],'L':design['길이'],'B':design['폭'],'D':design['깊이'],'P':design['소요마력'],'PE':design['소요전력']}[term])
            parts.append(number(add[term])+r'\times '+value)
        factors=[r'\left('+ '+'.join(parts)+r'\right)']
    elif model.get('size_by_type'):
        value='LBD' if design is None else number(design['길이']*design['폭']*design['깊이'])
        factors.append(r'\left(\frac{'+value+'}{'+number(model['volume_reference'])+r'}\right)^{'+number(model['size_alpha'][ship_type])+'}')
    else:
        for term,beta in terms.items():
            symbolic={'LBD':'LBD','LB':'LB','L':'L','P':'P','PE':r'P_E','D':'D','B':'B'}[term]
            if design is not None:
                symbolic=number({'LBD':design['길이']*design['폭']*design['깊이'],'LB':design['길이']*design['폭'],'L':design['길이'],'P':design['소요마력'],'PE':design['소요전력'],'D':design['깊이'],'B':design['폭']}[term])
            factors.append(r'\left('+symbolic+r'\right)^{'+number(beta)+'}')
    factors.append(number(dt))
    if model['year_by_type']:
        year='Y' if design is None else number(design['취역연도'])
        factors.append(r'\exp\!\left['+number(model['gamma'][ship_type])+r'\frac{'+year+'-'+number(model['mean_years'][ship_type])+r'}{10}\right]')
    factors.append(number(model['smearing']))
    lhs=r'\widehat W_{\mathrm{total}-700}' if name==TOTAL_MODEL else r'\widehat W_{'+name[4:]+'}'
    equation=lhs+'='+r'\,\times\,'.join(factors)
    if design is not None:
        import pandas as pd
        row=dict(design);row['함분류']=ship_type
        result=float(predict_one(pd.DataFrame([row]),model)[0])
        equation+='='+number(result)+r'\ \mathrm{tonne}'
    return equation


def render_png(equation,width=850):
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'nswe-math-cache'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    with matplotlib.rc_context({'mathtext.fontset':'stix','font.family':'STIXGeneral'}):
        fig=Figure(figsize=(width/100,.85),dpi=200)
        canvas=FigureCanvasAgg(fig)
        label=fig.text(.015,.5,'$'+equation+'$',ha='left',va='center',fontsize=20,color='#19384a')
        canvas.draw();extent=label.get_window_extent(canvas.get_renderer()).width
        if extent>width*2-45:label.set_fontsize(20*(width*2-45)/extent)
        stream=io.BytesIO();fig.savefig(stream,format='png',transparent=True,dpi=200)
        return stream.getvalue()


def variable_description(model,ship_type):
    meanings={'LBD':'LBD: 길이×폭×깊이(m³)','LB':'LB: 길이×폭(m²)','L':'L: 길이(m)','B':'B: 폭(m)','D':'D: 깊이(m)','P':'P: 소요마력(hp)','PE':'PE: 소요전력(kW)'}
    rows=[meanings[t] for t in model['terms']]
    if model.get('kind')=='additive':
        rows.append('k0: 기본 중량(tonne). k1…: 제원별 가산계수(tonne/해당 제원 단위). 괄호 안 기여 중량을 더합니다.')
        rows.append('계수: '+', '.join(('k0' if t=='constant' else t+' 가산계수')+'='+f'{v:.7g}' for t,v in model['additive_coeff'].items()))
    else:
        rows.append('K: 기본 중량계수(단위는 식에 따라 결정). 각 거듭제곱의 숫자: 해당 제원의 회귀지수(무차원).')
    if model.get('size_by_type'):rows.append('V0: 학습 LBD 기하평균(m³). 크기 지수는 선택한 함종의 값입니다.')
    rows.append('Dt: 기준 함종 대비 함종계수(무차원). Y: 취역(예상)연도(year). Ȳt: 선택 함종의 학습 평균연도.')
    rows.append('γt: 선택 비율을 이미 반영한 함종별 연도계수(10년 기준). C: 별도 역변환 편향 보정계수(마지막 배수).')
    if ship_type=='초계함' and model.get('patrol_info'):rows.append(model['patrol_info']['note'])
    return '\n'.join(rows)
