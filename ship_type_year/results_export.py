"""Compact numerical export of the Graphs and final Equations & Guide results."""
import math
import unicodedata
import numpy as np
import pandas as pd
from model import TOTAL_MODEL,diagnostic_rows,diagnostic_metrics
from final_equations import variable_description

SHEETS=['중량 비교','최종 수식','계수','적합도 요약']
SYMBOL={'LBD':'(L*B*D)','LB':'(L*B)','L':'L','B':'B','D':'D','P':'P','PE':'PE'}
MEANING={'LBD':'길이×폭×깊이(m³)','LB':'길이×폭(m²)','L':'길이(m)','B':'폭(m)','D':'깊이(m)','P':'소요마력(hp)','PE':'소요전력(kW)'}


def dt_value(m,cls):
    j=m['classes'].index(cls)
    return 1.0 if j==0 else float(np.exp(m['coef'][len(m['terms'])+j]))


def numerical_expression(m,cls):
    """Text equation with full-precision coefficients; no Excel names are required."""
    def num(v):return format(float(v),'.17g')
    if m.get('kind')=='additive':
        add=m['additive_coeff']
        core='('+num(add['constant'])+''.join(' + '+num(add[t])+'*'+SYMBOL[t] for t in m['terms'])+')'
    elif m.get('size_by_type'):
        core=num(np.exp(m['coef'][0]))+'*((L*B*D)/'+num(m['volume_reference'])+')^('+num(m['size_alpha'][cls])+')'
    else:
        core=num(np.exp(m['coef'][0]))+''.join('*'+SYMBOL[t]+'^('+num(v)+')' for t,v in zip(m['terms'],m['coef'][1:]))
    core+='*'+num(dt_value(m,cls))
    if m['year_by_type']:core+='*EXP(('+num(m['gamma'][cls])+')*(Y-('+num(m['mean_years'][cls])+'))/10)'
    return core+'*'+num(m['smearing'])


def result_tables(bundle):
    comparisons=[];equations=[];values=[];summary=[]
    for target,m in bundle['models'].items():
        label='경하중량 직접회귀(700 제외)' if target==TOTAL_MODEL else target
        for d in m.get('diagnostics',[]):
            comparisons.append({'계열':label,'순번':str(d['순번']),'함종':d['함종'],'함유형':d.get('함유형',''),
                '취역연도(year)':d['취역연도'],'실측중량(tonne)':d['실제중량'],'추정중량(tonne)':d['예측중량'],
                '잔차(실측-추정, tonne)':d['실제중량']-d['예측중량'],
                '오차율(추정-실측, %)':d['오차율(%)']})
        for cls in m['classes']:
            equations.append({'계열':label,'함종':cls,'최종 추정식':('W_light_minus700' if target==TOTAL_MODEL else 'W'+target[4:])+' = '+numerical_expression(m,cls),
                              '변수 및 계수 설명':'모든 중량은 tonne. EXP는 자연지수함수입니다.\n'+variable_description(m,cls)})
            def add(name,value,meaning):values.append({'계열':label,'함종':cls,'계수':name,'값':float(value),'단위 및 의미':meaning})
            if m.get('kind')=='additive':
                add('k0',m['additive_coeff']['constant'],'기본 중량(tonne)')
                for t in m['terms']:add('k_'+t,m['additive_coeff'][t],MEANING[t]+'의 가산계수. 단위: tonne/해당 제원 단위')
            else:
                add('K',np.exp(m['coef'][0]),'기본 중량계수. 단위는 식에 따라 결정')
                for t,v in zip(m['terms'],m['coef'][1:]):
                    add('αt' if m.get('size_by_type') else t+' 지수',m['size_alpha'][cls] if m.get('size_by_type') else v,MEANING[t]+'의 회귀지수(무차원)')
            add('Dt',dt_value(m,cls),'기준 함종 '+m['classes'][0]+'의 Dt=1에 대한 상대 계수')
            add('C',m['smearing'],'별도 역변환 편향 보정계수(무차원)')
            if m.get('size_by_type'):add('V0',m['volume_reference'],'학습 때 저장한 기준 LBD(m³)')
            if m['year_by_type']:
                add('γt',m['gamma'][cls],'선택 연도 비율이 이미 반영된 10년 기준 연도계수. q를 다시 곱하지 않습니다.')
                add('Ȳt',m['mean_years'][cls],'해당 함종의 평균 취역연도(year)')
            if m.get('year_scale') is not None:add('연도 효과(%)',m['year_scale']*100,'기본 연도계수 적용 비율')
        for cls in [None]+m['classes']:
            metric=diagnostic_metrics(diagnostic_rows(m,cls));selection=m.get('selection',{})
            summary.append({'조회 구분':'전체' if cls is None else '함종별','함종':'전체' if cls is None else cls,'계열':label,
                '척수':metric['n'],'R²':metric['r2'],'RMSE(tonne)':metric['rmse_tonne'],
                'MAPE(%)':metric['mape_percent'],'WAPE(%)':metric['wape_percent'],
                '초기 WAPE 기준(%)':selection.get('wape_requested_percent'),'WAPE 적용기준(%, 전체)':selection.get('wape_limit_percent'),
                'WAPE 기준 완화':'예' if selection.get('wape_relaxed') else '아니오'})
    for cls in bundle['models']['SWBS100']['classes']:
        equations.append({'계열':'SWBS100~600 합계(700 제외)','함종':cls,'최종 추정식':'W_sum_minus700 = W100+W200+W300+W400+W500+W600',
                          '변수 및 계수 설명':'각 계열의 최종 예측값을 합합니다. 합계 자체에 별도의 K·Dt·C를 곱하지 않습니다.'})
    return dict(zip(SHEETS,[pd.DataFrame(comparisons),pd.DataFrame(equations),pd.DataFrame(values),pd.DataFrame(summary)]))


def save_regression_results(path,bundle):
    from openpyxl import load_workbook
    from openpyxl.styles import Font,PatternFill,Alignment
    tables=result_tables(bundle)
    with pd.ExcelWriter(path,engine='openpyxl') as writer:
        for name,table in tables.items():table.to_excel(writer,sheet_name=name,index=False)
    wb=load_workbook(path)
    widths={'중량 비교':[28,12,12,14,18,22,22,28,28],'최종 수식':[28,12,95,85],
            '계수':[28,12,22,25,65],'적합도 요약':[13,12,28,10,15,20,17,17,20,28,18]}
    for ws in wb:
        ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
        ws.sheet_properties.tabColor='087F87'
        for j,width in enumerate(widths[ws.title],1):ws.column_dimensions[ws.cell(1,j).column_letter].width=width
        for row in ws:
            for cell in row:
                cell.font=Font(name='Times New Roman',size=10)
                cell.alignment=Alignment(vertical='top',wrap_text=True)
                if isinstance(cell.value,str):cell.data_type='s' # Identifiers and equation text are literal, never Excel formulas.
            if row[0].row>1:
                lines=max(sum(max(1,math.ceil(sum(2 if unicodedata.east_asian_width(ch) in 'WF' else 1 for ch in part)/widths[ws.title][j])) for part in str(cell.value or '').split('\n')) for j,cell in enumerate(row))
                ws.row_dimensions[row[0].row].height=min(409,max(22,lines*15+8))
        for cell in ws[1]:
            cell.font=Font(name='Times New Roman',size=10,bold=True,color='FFFFFF')
            cell.fill=PatternFill('solid',fgColor='102638');cell.alignment=Alignment(horizontal='center',vertical='center',wrap_text=True)
        ws.row_dimensions[1].height=36
        if ws.title=='중량 비교':
            for row in ws.iter_rows(min_row=2):
                row[1].number_format='@'
                for cell in row[5:]:cell.number_format='0.0000'
                if abs(row[-1].value)>10:
                    for cell in row:cell.font=Font(name='Times New Roman',size=10,color='AE4437')
        if ws.title=='적합도 요약':
            for row in ws.iter_rows(min_row=2):
                for cell in row[4:10]:cell.number_format='0.0000'
    wb.save(path)
