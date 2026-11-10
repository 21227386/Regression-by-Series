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
    if model.get('size_by_type'):
        value='LBD' if design is None else number(design['길이']*design['폭']*design['깊이'])
        factors.append(r'\left(\frac{'+value+'}{'+number(model['volume_reference'])+r'}\right)^{'+number(model['size_alpha'][ship_type])+'}')
    else:
        for term,beta in terms.items():
            symbolic={'LBD':'LBD','LB':'LB','L':'L','P':'P','PE':r'P_E'}[term]
            if design is not None:
                symbolic=number({'LBD':design['길이']*design['폭']*design['깊이'],'LB':design['길이']*design['폭'],'L':design['길이'],'P':design['소요마력'],'PE':design['소요전력']}[term])
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
