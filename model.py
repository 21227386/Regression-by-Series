"""SWBS log-linear regression; weights tonne, dimensions m, power hp/kW."""
import json
import re
import numpy as np
import pandas as pd

BASE = ['함분류', '취역연도', '길이', '폭', '깊이', '소요마력', '소요전력', '순항속력']
WEIGHTS = [f'SWBS{i}' for i in range(100, 800, 100)]
HEADERS = dict(zip(BASE + WEIGHTS + ['총중량'], ['함종', '취역연도 (년)', '길이(m)', '폭(m)', '깊이(m)', '소요마력(hp)', '소요전력(kW)', '순항속력(kn)'] + [f'{c}(tonne)' for c in WEIGHTS] + ['총중량(tonne)']))
# The supplied equations determine these terms; cruising speed is deliberately unused.
TERMS = {100: ['LBD'], 200: ['P', 'year'], 300: ['PE', 'L', 'year'],
         400: ['year'], 500: ['LBD', 'PE', 'year'], 600: ['LB'], 700: ['year']}
ALIASES = {'총 중량': '총중량', '충중량': '총중량', '취취역연도': '취역연도',
           '함정분류': '함분류', '함종': '함분류'}

SHIP_TYPES = ['구축함', '호위함', '고속함', '상륙함', '기뢰전함', '지원함']
SHIP_ALIASES = {'초계함': '호위함', '초계정': '호위함'}

def normalize_types(df):
    out = df.copy()
    if out['함분류'].isna().any():
        raise ValueError('함분류에 빈 값이 있습니다.')
    original = out['함분류'].astype(str).str.strip()
    normalized = original.replace(SHIP_ALIASES)
    invalid = sorted(set(normalized) - set(SHIP_TYPES))
    if invalid:
        raise ValueError('지원하지 않는 함종: ' + ', '.join(invalid) +
                         '. 허용 함종: ' + ', '.join(SHIP_TYPES) + ' (초계함·초계정은 호위함 적용)')
    if '함분류_입력' not in out:
        out['함분류_입력'] = original
    out['함분류'] = normalized
    return out

def validate_model_types(bundle):
    for model in bundle['models'].values():
        if any(c not in SHIP_TYPES for c in model['classes']):
            raise ValueError('이전 모델의 함종 체계가 다릅니다. 새 프로그램에서 원본 엑셀로 다시 학습하세요.')

def read_excel(path, training=True):
    df = pd.read_excel(path, engine='openpyxl')
    names = [re.sub(r'\((?:년|m|hp|kw|kn|tonne|ton)\)$', '', re.sub(r'\s+', '', str(c)), flags=re.IGNORECASE) for c in df.columns]
    df.columns = [ALIASES.get(c, c) for c in names]
    if df.columns.duplicated().any():
        raise ValueError('중복된 열 이름이 있습니다.')
    required = BASE + (WEIGHTS + ['총중량'] if training else [])
    missing = [c for c in required if c not in df]
    if missing:
        raise ValueError('필수 열 누락: ' + ', '.join(missing))
    if df.empty:
        raise ValueError('데이터가 없습니다.')
    # Only inspect program input fields; blank optional user notes are irrelevant.
    blanks = df[required].isna() | df[required].apply(lambda col: col.map(lambda v: isinstance(v, str) and not v.strip()))
    excluded = []
    for idx in df.index[blanks.any(axis=1)]:
        excluded.append({'엑셀 행': int(idx) + 2, '함종': '' if pd.isna(df.at[idx, '함분류']) else str(df.at[idx, '함분류']),
                         '제외 사유': '빈 항목: ' + ', '.join(HEADERS[c] for c in required if blanks.at[idx, c])})
    input_count = len(df)
    df = df.loc[~blanks.any(axis=1)].copy()
    if df.empty:
        raise ValueError(f'전체 {input_count}척이 빈 입력 항목으로 제외되어 분석할 자료가 없습니다. 모든 필수 항목을 채워 주세요.')
    df = normalize_types(df)
    df.attrs['excluded_rows'] = excluded
    df.attrs['input_count'] = input_count
    numeric = BASE[1:] + (WEIGHTS + ['총중량'] if training else [])
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        bad = ~np.isfinite(df[c])
        if c != '취역연도':
            bad |= df[c] <= 0
        else:
            bad |= (df[c] < 1800) | (df[c] > 2200) | (df[c] % 1 != 0)
        if bad.any():
            rows = ', '.join(str(int(i) + 2) for i in df.index[bad][:10])
            raise ValueError(f'{c}: 엑셀 {rows}행 값 확인 필요. 수치·양수(연도는 1800~2200 정수)여야 합니다. 0중량은 로그모형에서 처리할 수 없습니다.')
    return df

def design(df, terms, classes, mean_year):
    unknown = sorted(set(df['함분류']) - set(classes))
    if unknown:
        raise ValueError('학습되지 않은 함분류: ' + ', '.join(unknown))
    values = {'LBD': np.log(df['길이'] * df['폭'] * df['깊이']),
              'LB': np.log(df['길이'] * df['폭']), 'L': np.log(df['길이']),
              'P': np.log(df['소요마력']), 'PE': np.log(df['소요전력']),
              'year': (df['취역연도'] - mean_year) / 10}
    cols = [np.ones(len(df))] + [np.asarray(values[t]) for t in terms]
    cols += [(df['함분류'] == c).astype(float).to_numpy() for c in classes[1:]]
    return np.column_stack(cols)

def fit_one(df, target, terms):
    classes = sorted(df['함분류'].unique().tolist())
    mean_year = float(df['취역연도'].mean())
    x = design(df, terms, classes, mean_year)
    if len(df) <= x.shape[1]:
        raise ValueError(f'{target}: {x.shape[1]}개 계수에 비해 자료 {len(df)}척이 부족합니다. 최소 {x.shape[1]+1}척 필요(안정적 추정에는 더 많은 자료 권장).')
    if np.linalg.matrix_rank(x) < x.shape[1]:
        raise ValueError(f'{target}: 설명변수가 중복되거나 일정하여 계수를 식별할 수 없습니다. 연도·규모·함분류의 다양성을 확인하세요.')
    y = np.log(df[target].to_numpy())
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ coef
    smear = float(np.exp(residual).mean())
    predicted = np.exp(x @ coef) * smear
    actual = df[target].to_numpy()
    denom = float(((actual - actual.mean()) ** 2).sum())
    return {'target': target, 'terms': terms, 'classes': classes, 'mean_year': mean_year,
            'coef': coef.tolist(), 'smearing': smear,
            'ranges': {c: [float(df[c].min()), float(df[c].max())] for c in BASE[1:]},
            'n': len(df), 'parameters': x.shape[1], 'condition_number': float(np.linalg.cond(x)),
            'rmse_tonne': float(np.sqrt(np.mean((actual-predicted)**2))),
            'mape_percent': float(np.mean(abs((actual-predicted)/actual))*100),
            'r2': None if denom == 0 else float(1-((actual-predicted)**2).sum()/denom)}

def predict_one(df, model):
    x = design(df, model['terms'], model['classes'], model['mean_year'])
    with np.errstate(over='raise', invalid='raise'):
        return np.exp(x @ np.asarray(model['coef'])) * model['smearing']

def train(df):
    df = normalize_types(df)
    models = {f'SWBS{i}': fit_one(df, f'SWBS{i}', terms) for i, terms in TERMS.items()}
    excluded = df.attrs.get('excluded_rows', [])
    warnings = ([f'입력 {df.attrs.get("input_count", len(df))}척 중 빈 항목이 있는 {len(excluded)}척 제외, {len(df)}척 학습. 제외 행: ' + ', '.join(str(r['엑셀 행']) for r in excluded)] if excluded else [])
    if '총중량' in df:
        gap = np.abs(df['총중량'] - df[WEIGHTS].sum(axis=1)) / df['총중량']
        if (gap > .01).any():
            warnings.append(f'입력 총중량과 SWBS 합계가 1% 이상 다른 자료 {(gap > .01).sum()}척. 총중량 정의를 확인하세요.')
        # Independent baseline, explicitly distinct from the seven supplied formulas.
        try:
            models['총중량_독립회귀'] = fit_one(df, '총중량', ['LBD', 'P', 'year'])
        except ValueError as e:
            warnings.append('선택 총중량 독립회귀 미실행: ' + str(e))
    return {'version': 2, 'models': models, 'warnings': warnings, 'excluded_rows': excluded}

def predict(df, bundle):
    validate_model_types(bundle)
    df = normalize_types(df)
    out = df.copy()
    for name, model in bundle['models'].items():
        out[name + '_예측_tonne'] = predict_one(df, model)
    out['총중량_SWBS합계_tonne'] = sum(out[c + '_예측_tonne'] for c in WEIGHTS)
    warnings = []
    model = bundle['models']['SWBS100']
    for _, row in df.iterrows():
        outside = [c for c, (lo, hi) in model['ranges'].items()
                   if c != '순항속력' and not lo <= row[c] <= hi]
        warnings.append('학습범위 밖: ' + ', '.join(outside) if outside else '')
    out['주의'] = warnings
    return out

def coefficients(bundle):
    rows = []
    for name, m in bundle['models'].items():
        b = m['coef']; terms = m['terms']; p = 1+len(terms)
        row = {'계열': name, 'K': float(np.exp(b[0])), '역변환보정': m['smearing'],
               '기준함분류': m['classes'][0], '기준취역연도': m['mean_year'],
               '식': 'W = K × ' + ' × '.join({'LBD':'(LBD)^α', 'LB':'(LB)^α', 'L':'L^β',
                  'P':'P^β' if 'LBD' in terms else 'P^α', 'PE':'PE^β' if 'LBD' in terms else 'PE^α',
                  'year':'exp[γ(Y−기준연도)/10]'}[t] for t in terms) + ' × Dt × 역변환보정'}
        row.update({t + '_회귀계수': float(v) for t, v in zip(terms, b[1:p])})
        for j, cls in enumerate(m['classes']):
            rows.append({**row, '함분류': cls, 'Dt': 1.0 if j == 0 else float(np.exp(b[p+j-1]))})
    return pd.DataFrame(rows)

def export(path, df, bundle):
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        predict(df, bundle).to_excel(writer, sheet_name='예측결과', index=False)
        coefficients(bundle).to_excel(writer, sheet_name='회귀계수', index=False)
        audit = [{**r, '구분': '학습'} for r in bundle.get('excluded_rows', [])]
        audit += [{**r, '구분': '현재 입력'} for r in df.attrs.get('excluded_rows', [])]
        if audit: pd.DataFrame(audit).drop_duplicates().to_excel(writer, sheet_name='제외 행', index=False)
        metrics = [{k: m[k] for k in ['target','n','parameters','rmse_tonne','mape_percent','r2','condition_number']} for m in bundle['models'].values()]
        pd.DataFrame(metrics).to_excel(writer, sheet_name='학습적합도', index=False)
        pd.DataFrame({'주의': bundle['warnings'] + ['적합도는 학습자료 기준이며 외부 검증 성능이 아닙니다.', '총중량_SWBS합계는 100~700 합계입니다. 독립회귀 총중량과 구분하세요.', '순항속력은 입력에 보존하지만 제공된 식에 없어 회귀에 사용하지 않습니다.']}).to_excel(writer, sheet_name='안내', index=False)

def template(path):
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        pd.DataFrame(columns=[HEADERS[c] for c in BASE + WEIGHTS + ['총중량']]).to_excel(writer, sheet_name='데이터', index=False)
        pd.DataFrame({'함종': SHIP_TYPES + ['초계함', '초계정'],
                      '적용 함종': SHIP_TYPES + ['호위함', '호위함']}).to_excel(writer, sheet_name='함종 안내', index=False)
        pd.DataFrame({'열': [HEADERS[c] for c in BASE + WEIGHTS + ['총중량']],
                      '단위': ['문자','년','m','m','m','hp','kW','kn'] + ['tonne'] * 8}).to_excel(writer, sheet_name='단위', index=False)

    # Format the workbook without adding fabricated reference-ship values.
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.comments import Comment
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = load_workbook(path)
    ws = wb['데이터']; ws.freeze_panes = 'A2'
    ws.auto_filter.ref = ws.dimensions
    units = ['함종', '년', 'm', 'm', 'm', 'hp', 'kW', 'kn'] + ['tonne'] * 8
    for cell, unit in zip(ws[1], units):
        cell.comment = Comment('단위: ' + unit + (' / 경하중량' if cell.value == HEADERS['총중량'] else ''), 'Naval Ship Design')
    validation = DataValidation(type='list', formula1='"' + ','.join(SHIP_TYPES + ['초계함','초계정']) + '"')
    validation.errorTitle = '함종 확인'; validation.error = '목록에서 함종을 선택하세요.'
    validation.showErrorMessage = True; ws.add_data_validation(validation); validation.add('A2:A10000')
    guide = wb.create_sheet('사용 안내')
    for text in ['Naval Ship Weight Estimator(Total & Series)',
                 '첫 번째 데이터 시트에 실적함 1척당 1행으로 입력하세요.',
                 '기존 자료와 추가 자료를 합친 전체 파일을 업로드하면 계수를 다시 추정합니다.',
                 'SWBS100~700: 각 계열 총중량(tonne), 총중량: 실제 경하중량(tonne).',
                 '학습 입력 항목이 하나라도 비어 있는 함정 행은 모든 회귀분석에서 제외합니다. 총중량도 필수입니다.',
                 '길이/폭/깊이 m, 소요마력 hp, 소요전력 kW, 순항속력 kn.',
                 '속력은 기록용이며 현재 회귀식에는 사용하지 않습니다.',
                 '0 또는 결측 SWBS 중량은 학습할 수 없습니다.',
                 '초계함·초계정은 호위함과 통합 학습합니다.',
                 '신규 함정은 GUI에서 직접 입력하거나 기본 제원 8개 열의 엑셀을 일괄 업로드하세요.']:
        guide.append([text])
    for sheet in wb:
        sheet.sheet_properties.tabColor = '087F87'
        for cell in sheet[1]:
            cell.fill = PatternFill('solid', fgColor='102638')
            cell.font = Font(name='Times New Roman', color='FFFFFF', bold=True, size=12)
            cell.alignment = Alignment(vertical='center', wrap_text=True)
        sheet.row_dimensions[1].height = 60
        for col in sheet.columns:
            sheet.column_dimensions[col[0].column_letter].width = 10
    for row in guide.iter_rows(min_row=2):
        row[0].alignment = Alignment(wrap_text=True, vertical='top')
        guide.row_dimensions[row[0].row].height = 110
    wb.save(path)
