"""SWBS log-linear regression; weights tonne, dimensions m, power hp/kW."""
import json
import re
import numpy as np
import pandas as pd

BASE = ['함분류', '취역연도', '길이', '폭', '깊이', '소요마력', '소요전력', '순항속력']
WEIGHTS = [f'SWBS{i}' for i in range(100, 800, 100)]
HEADERS = dict(zip(BASE + WEIGHTS + ['총중량'], ['함종', '취역연도(year)', '길이(m)', '폭(m)', '깊이(m)', '소요마력(hp)', '소요전력(kW)', '순항속력(kn)'] + [f'{c}(tonne)' for c in WEIGHTS] + ['총중량(tonne)']))
HEADERS['순번'] = '순번'
HEADERS['함유형'] = '함유형'
INPUT_COLUMNS = ['순번', '함분류', '함유형'] + BASE[1:] + WEIGHTS + ['총중량']
# The supplied equations determine these terms; cruising speed is deliberately unused.
TERMS = {100: ['LBD'], 200: ['P', 'year'], 300: ['PE', 'L', 'year'],
         400: ['year'], 500: ['LBD', 'PE', 'year'], 600: ['LB'], 700: ['year']}
ALIASES = {'총 중량': '총중량', '충중량': '총중량', '취취역연도': '취역연도',
           '함정분류': '함분류', '함종': '함분류', '함정번호': '순번', '번호': '순번'}

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
    df = pd.read_excel(path, engine='openpyxl', dtype=object)
    names = [re.sub(r'\((?:년|year|m|hp|kw|kn|tonne|ton)\)$', '', re.sub(r'\s+', '', str(c)), flags=re.IGNORECASE) for c in df.columns]
    df.columns = [ALIASES.get(c, c) for c in names]
    if df.columns.duplicated().any():
        raise ValueError('중복된 열 이름이 있습니다.')
    serial_auto = '순번' not in df
    if serial_auto: df.insert(0, '순번', [str(i+1) for i in range(len(df))])
    required = ['순번'] + BASE + (WEIGHTS + ['총중량'] if training else [])
    if '함유형' in df: required.append('함유형')
    missing = [c for c in required if c not in df]
    if missing:
        raise ValueError('필수 열 누락: ' + ', '.join(missing))
    if df.empty:
        raise ValueError('데이터가 없습니다.')
    # Reject malformed rows as a whole, keeping a traceable per-row audit.
    reasons = {idx: [] for idx in df.index}
    for c in required:
        blank = df[c].isna() | df[c].map(lambda v: isinstance(v, str) and not v.strip())
        for idx in df.index[blank]: reasons[idx].append(HEADERS[c] + ': 빈 값')
        if c == '순번':
            def identifier(v):
                if pd.isna(v): return ''
                if isinstance(v, (bool, np.bool_)): return str(v)
                if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v) and float(v).is_integer(): return str(int(v))
                return str(v).strip()
            ids = df[c].map(identifier)
            invalid_ids = ~blank & (ids.str.contains('?', regex=False) | ids.isin(['inf','-inf','nan','True','False']))
            duplicate = ~blank & ids.duplicated(keep=False)
            for idx in df.index[invalid_ids]: reasons[idx].append('순번: 유효한 식별번호 필요')
            for idx in df.index[duplicate]: reasons[idx].append('순번: 중복된 식별번호')
            df[c] = ids
        elif c == '함유형':
            values = df[c].astype(str).str.strip()
            for idx in df.index[~blank & values.str.contains('?', regex=False)]: reasons[idx].append('함유형: 유효한 분류명 필요')
            df[c] = values
        elif c == '함분류':
            types = df[c].astype(str).str.strip().replace(SHIP_ALIASES)
            bad = ~blank & ~types.isin(SHIP_TYPES)
            for idx in df.index[bad]: reasons[idx].append('함종: 허용되지 않은 값')
        else:
            values = pd.to_numeric(df[c], errors='coerce')
            invalid = ~np.isfinite(values) | df[c].map(lambda v: isinstance(v, (bool, np.bool_)))
            for idx in df.index[~blank & invalid]: reasons[idx].append(HEADERS[c] + ': 숫자 형식 오류 또는 무한값')
            if c == '취역연도':
                bounds = (values < 1800) | (values > 2200) | (values % 1 != 0)
                detail = ': 1800~2200의 정수 필요'
            else:
                bounds = values <= 0
                detail = ': 0보다 큰 값 필요'
            for idx in df.index[~blank & ~invalid & bounds]: reasons[idx].append(HEADERS[c] + detail)
            df[c] = values
    excluded = [{'순번': str(df.at[idx, '순번']), '엑셀 행': int(idx) + 2,
                 '함종': '' if pd.isna(df.at[idx, '함분류']) else str(df.at[idx, '함분류']),
                 '제외 사유': '; '.join(items)} for idx, items in reasons.items() if items]
    input_count = len(df)
    df = df.loc[[idx for idx, items in reasons.items() if not items]].copy()
    if df.empty:
        examples = ' / '.join(f"{r['엑셀 행']}행: {r['제외 사유']}" for r in excluded[:5])
        raise ValueError(f'전체 {input_count}척이 빈 값 또는 잘못된 입력으로 제외되어 분석할 자료가 없습니다. ' + examples)
    df = normalize_types(df)
    df.attrs['excluded_rows'] = excluded
    df.attrs['input_count'] = input_count
    df.attrs['serial_auto'] = serial_auto
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
    counts = df['함분류'].value_counts()
    classes = sorted(counts.index.tolist(), key=lambda c: (-int(counts[c]), SHIP_TYPES.index(c)))
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
            'class_counts': {c: int(counts[c]) for c in classes},
            'diagnostics': [{'순번': str(df.at[idx, '순번']) if '순번' in df else str(int(idx)+1), '엑셀 행': int(idx) + 2, '함종': str(df.at[idx, '함분류']), '함유형': str(df.at[idx, '함유형']) if '함유형' in df else '',
                             '실제중량': float(actual[j]), '예측중량': float(predicted[j]),
                             '잔차': float(actual[j]-predicted[j]),
                             '오차율(%)': float((predicted[j]-actual[j])/actual[j]*100),
                             'Dt': 1.0 if df.at[idx, '함분류']==classes[0] else float(np.exp(coef[1+len(terms)+classes[1:].index(df.at[idx, '함분류'])]))}
                            for j, idx in enumerate(df.index)],
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
    warnings = ([f'입력 {df.attrs.get("input_count", len(df))}척 중 빈 값 또는 오류가 있는 {len(excluded)}척 제외, {len(df)}척 학습. 제외 행: ' + ', '.join(str(r['엑셀 행']) for r in excluded)] if excluded else [])
    if df.attrs.get('serial_auto'):
        warnings.append('순번 없는 이전 양식: 원본 데이터 순서에 따라 1부터 식별번호를 자동 부여했습니다.')
    if '총중량' in df:
        gap = np.abs(df['총중량'] - df[WEIGHTS].sum(axis=1)) / df['총중량']
        if (gap > .01).any():
            warnings.append(f'입력 총중량과 SWBS 합계가 1% 이상 다른 자료 {(gap > .01).sum()}척. 총중량 정의를 확인하세요.')
        # Independent baseline, explicitly distinct from the seven supplied formulas.
        try:
            models['총중량_독립회귀'] = fit_one(df, '총중량', ['LBD', 'P', 'year'])
        except ValueError as e:
            warnings.append('선택 총중량 독립회귀 미실행: ' + str(e))
    return {'version': 3, 'models': models, 'warnings': warnings, 'excluded_rows': excluded}

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
               '기준함종': m['classes'][0], '기준취역연도': m['mean_year'],
               '식': 'W = K × ' + ' × '.join({'LBD':'(LBD)^α', 'LB':'(LB)^α', 'L':'L^β',
                  'P':'P^β' if 'LBD' in terms else 'P^α', 'PE':'PE^β' if 'LBD' in terms else 'PE^α',
                  'year':'exp[γ(Y−기준연도)/10]'}[t] for t in terms) + ' × Dt × 역변환보정'}
        row.update({t + '_회귀계수': float(v) for t, v in zip(terms, b[1:p])})
        for j, cls in enumerate(m['classes']):
            rows.append({**row, '함종': cls, '학습척수': m.get('class_counts', {}).get(cls), 'Dt': 1.0 if j == 0 else float(np.exp(b[p+j-1]))})
    return pd.DataFrame(rows)

def export(path, df, bundle):
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        predict(df, bundle).to_excel(writer, sheet_name='예측결과', index=False)
        coefficients(bundle).to_excel(writer, sheet_name='회귀계수', index=False)
        diagnostic_data = [{'계열': name, **r} for name, m in bundle['models'].items() for r in m.get('diagnostics', [])]
        if diagnostic_data: pd.DataFrame(diagnostic_data).to_excel(writer, sheet_name='함정별 회귀진단', index=False)
        audit = [{**r, '구분': '학습'} for r in bundle.get('excluded_rows', [])]
        audit += [{**r, '구분': '현재 입력'} for r in df.attrs.get('excluded_rows', [])]
        if audit: pd.DataFrame(audit).drop_duplicates().to_excel(writer, sheet_name='제외 행', index=False)
        metrics = [{k: m[k] for k in ['target','n','parameters','rmse_tonne','mape_percent','r2','condition_number']} for m in bundle['models'].values()]
        pd.DataFrame(metrics).to_excel(writer, sheet_name='학습적합도', index=False)
        pd.DataFrame({'주의': bundle['warnings'] + ['적합도는 학습자료 기준이며 외부 검증 성능이 아닙니다.', '총중량_SWBS합계는 100~700 합계입니다. 독립회귀 총중량과 구분하세요.', '순항속력은 입력에 보존하지만 제공된 식에 없어 회귀에 사용하지 않습니다.']}).to_excel(writer, sheet_name='안내', index=False)

def template(path):
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        pd.DataFrame(columns=[HEADERS[c] for c in INPUT_COLUMNS]).to_excel(writer, sheet_name='데이터', index=False)
        pd.DataFrame({'함종': SHIP_TYPES + ['초계함', '초계정'],
                      '적용 함종': SHIP_TYPES + ['호위함', '호위함']}).to_excel(writer, sheet_name='함종 안내', index=False)
        pd.DataFrame({'열': [HEADERS[c] for c in INPUT_COLUMNS],
                      '단위': ['식별번호','문자','문자','year','m','m','m','hp','kW','kn'] + ['tonne'] * 8}).to_excel(writer, sheet_name='단위', index=False)

    # Format the workbook without adding fabricated reference-ship values.
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.comments import Comment
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = load_workbook(path)
    ws = wb['데이터']; ws.freeze_panes = 'A2'
    ws.auto_filter.ref = ws.dimensions
    units = ['식별번호', '함종', '함유형', 'year', 'm', 'm', 'm', 'hp', 'kW', 'kn'] + ['tonne'] * 8
    for cell, unit in zip(ws[1], units):
        cell.comment = Comment('단위: ' + unit + (' / 경하중량' if cell.value == HEADERS['총중량'] else ''), 'Naval Ship Design')
    validation = DataValidation(type='list', formula1='"' + ','.join(SHIP_TYPES + ['초계함','초계정']) + '"')
    validation.errorTitle = '함종 확인'; validation.error = '목록에서 함종을 선택하세요.'
    validation.showErrorMessage = True; ws.add_data_validation(validation); validation.add('B2:B10000')
    ws.column_dimensions['A'].number_format = '@'
    guide = wb.create_sheet('사용 안내')
    for text in ['Naval Ship Weight Estimator(Total & Series)',
                 '첫 번째 시트의 순번에 함정 고유 번호를 입력하세요. 중복·빈 번호는 제외됩니다. 001처럼 앞의 0을 유지할 수 있습니다.',
                 '함유형은 별도 기록 항목이며 회귀계수에는 사용하지 않습니다. 입력한 경우 빈 값·?가 있는 행은 제외합니다.',
                 '기존 자료와 추가 자료를 합친 전체 파일을 업로드하면 계수를 다시 추정합니다.',
                 'SWBS100~700: 각 계열 총중량(tonne), 총중량: 실제 경하중량(tonne).',
                 '빈 값, ?, 수치 오류, 0·음수, 잘못된 함종·연도가 있는 함정 행은 모든 회귀분석에서 제외합니다. 총중량도 필수입니다.',
                 '길이/폭/깊이 m, 소요마력 hp, 소요전력 kW, 순항속력 kn.',
                 '속력은 기록용이며 현재 회귀식에는 사용하지 않습니다.',
                 '0 또는 결측 SWBS 중량은 학습할 수 없습니다.',
                 '초계함·초계정은 호위함과 통합 학습합니다.',
                 '신규 함정은 GUI의 설계 제원 입력란에서 직접 입력하세요.']:
        guide.append([text])
    for sheet in wb:
        sheet.sheet_properties.tabColor = '087F87'
        for cell in sheet[1]:
            cell.fill = PatternFill('solid', fgColor='102638')
            cell.font = Font(name='Times New Roman', color='FFFFFF', bold=True, size=10)
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        sheet.row_dimensions[1].height = 60
        for col in sheet.columns:
            sheet.column_dimensions[col[0].column_letter].width = 10
    for row in guide.iter_rows(min_row=2):
        row[0].alignment = Alignment(wrap_text=True, vertical='top')
        guide.row_dimensions[row[0].row].height = 110
    wb.save(path)


MODEL_TERMS = ['LBD', 'LB', 'L', 'P', 'PE', 'year']

def save_model_excel(path, bundle):
    """Write a human-readable model; these tables are the actual loadable data."""
    parameters, types, ranges, diagnostics = [], [], [], []
    for name, m in bundle['models'].items():
        nterms = len(m['terms'])
        parameters.append({'Target': name, 'K': float(np.exp(m['coef'][0])), 'C': m['smearing'],
                           'Terms': ','.join(m['terms']), 'MeanYear': m['mean_year'],
                           'ReferenceType': m['classes'][0], 'N': m['n'], 'Parameters': m['parameters'],
                           'RMSE_tonne': m['rmse_tonne'], 'MAPE_percent': m['mape_percent'],
                           'R2': m['r2'], 'ConditionNumber': m['condition_number'],
                           **{t: float(v) for t, v in zip(m['terms'], m['coef'][1:1+nterms])}})
        for j, c in enumerate(m['classes']):
            types.append({'Target': name, 'ShipType': c, 'Count': m.get('class_counts', {}).get(c),
                          'Dt': 1.0 if j==0 else float(np.exp(m['coef'][1+nterms+j-1]))})
        for c, bounds in m['ranges'].items():
            ranges.append({'Target': name, 'Variable': c, 'Min': bounds[0], 'Max': bounds[1]})
        diagnostics += [{'Target': name, **row} for row in m.get('diagnostics', [])]
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        pd.DataFrame(parameters).to_excel(writer, sheet_name='Model Parameters', index=False)
        pd.DataFrame(types).to_excel(writer, sheet_name='Ship Types', index=False)
        pd.DataFrame(ranges).to_excel(writer, sheet_name='Ranges', index=False)
        coefficients(bundle).to_excel(writer, sheet_name='회귀계수 보기', index=False)
        pd.DataFrame(diagnostics, columns=['Target','순번','엑셀 행','함종','함유형','실제중량','예측중량','잔차','오차율(%)','Dt']).to_excel(writer, sheet_name='Diagnostics', index=False)
        pd.DataFrame(bundle.get('excluded_rows', []), columns=['순번','엑셀 행','함종','제외 사유']).to_excel(writer, sheet_name='Excluded Rows', index=False)
        pd.DataFrame({'Note': ['Naval Ship Weight Estimator / Excel Model v1',
                              'Model Parameters, Ship Types, Ranges are the data used when loading this model.',
                              'K and C are separate. The reference ship type has Dt=1.',
                              'R2, RMSE and diagnostics describe the original training fit; editing coefficients does not recompute them.',
                              '기준 함종은 유효 학습 척수가 가장 많은 함종입니다. 동률이면 구축함·호위함·고속함·상륙함·기뢰전함·지원함 순입니다.',
                              '회귀계수 보기 시트는 열람용입니다. 모델을 변경하려면 재학습 후 저장을 권장합니다.'] + bundle.get('warnings', [])}).to_excel(writer, sheet_name='Notes', index=False)
        pd.DataFrame([{'Format': 'NSWE Excel Model', 'Version': 1}]).to_excel(writer, sheet_name='Format', index=False)
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill
    wb = load_workbook(path)
    for ws in wb:
        ws.freeze_panes = 'A2'; ws.auto_filter.ref = ws.dimensions
        for col in ws.columns: ws.column_dimensions[col[0].column_letter].width = 20
        for cell in ws[1]:
            cell.font = Font(name='Times New Roman', size=10, bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='102638')
    wb.save(path)

def load_model_excel(path):
    """Load only explicit spreadsheet tables, never executable serialization."""
    try:
        with pd.ExcelFile(path, engine='openpyxl') as book:
            meta = pd.read_excel(book, sheet_name='Format')
            if len(meta)!=1 or meta.iloc[0]['Format']!='NSWE Excel Model' or meta.iloc[0]['Version']!=1:
                raise ValueError('지원하지 않는 모델 형식입니다.')
            params = pd.read_excel(book, sheet_name='Model Parameters')
            types = pd.read_excel(book, sheet_name='Ship Types')
            ranges = pd.read_excel(book, sheet_name='Ranges')
            diagnostics = pd.read_excel(book, sheet_name='Diagnostics', dtype={'순번': str})
            if '순번' not in diagnostics:
                diagnostics['순번'] = (diagnostics['엑셀 행']-1).astype(str)
            if '함유형' in diagnostics: diagnostics['함유형'] = diagnostics['함유형'].fillna('')
            excluded = pd.read_excel(book, sheet_name='Excluded Rows', dtype={'순번':str}).fillna('').to_dict('records')
            notes = pd.read_excel(book, sheet_name='Notes')['Note'].dropna().astype(str).tolist()[6:]
        expected = set(WEIGHTS + ['총중량_독립회귀'])
        if params.empty or params['Target'].duplicated().any() or not set(WEIGHTS)<=set(params['Target']) or not set(params['Target'])<=expected:
            raise ValueError('계열별 모델 목록이 잘못되었습니다.')
        def finite(value, name, positive=False):
            v = float(value)
            if not np.isfinite(v) or (positive and v<=0): raise ValueError(name+' 값이 유효하지 않습니다.')
            return v
        def integer(value, name):
            v=finite(value,name,True)
            if v%1: raise ValueError(name+'은 양의 정수여야 합니다.')
            return int(v)
        models = {}
        for _, row in params.iterrows():
            name=row['Target']; terms=str(row['Terms']).split(',')
            wanted = TERMS[int(name[4:])] if name in WEIGHTS else ['LBD','P','year']
            if terms!=wanted: raise ValueError(name+' 설명변수 목록이 다릅니다.')
            group=types[types['Target']==name]
            if group.empty or group['ShipType'].duplicated().any() or not set(group['ShipType'])<=set(SHIP_TYPES): raise ValueError('함종 목록이 잘못되었습니다.')
            ref=row['ReferenceType']
            if ref not in set(group['ShipType']): raise ValueError('기준 함종이 없습니다.')
            classes=[ref]+[c for c in group['ShipType'] if c!=ref]
            dt={r['ShipType']: finite(r['Dt'],'Dt',True) for _,r in group.iterrows()}
            if not np.isclose(dt[ref],1): raise ValueError('기준 함종의 Dt는 1이어야 합니다.')
            coef=[float(np.log(finite(row['K'],'K',True)))]+[finite(row[t],t) for t in terms]+[float(np.log(dt[c])) for c in classes[1:]]
            bounds={}
            rgroup=ranges[ranges['Target']==name]
            if rgroup['Variable'].duplicated().any():raise ValueError('학습 범위가 중복되었습니다.')
            for _,r in rgroup.iterrows():
                lo,hi=finite(r['Min'],'Min'),finite(r['Max'],'Max')
                if lo>hi:raise ValueError('학습 범위의 최소·최대값이 잘못되었습니다.')
                bounds[r['Variable']]=[lo,hi]
            if set(bounds)!=set(BASE[1:]):raise ValueError('학습 범위가 누락되었습니다.')
            counts={r['ShipType']: integer(r['Count'],'Count') for _,r in group.iterrows() if pd.notna(r['Count'])}
            models[name]={'target': '총중량' if name=='총중량_독립회귀' else name,
                          'terms':terms, 'classes':classes, 'coef':coef, 'smearing':finite(row['C'],'C',True),
                          'mean_year':finite(row['MeanYear'],'MeanYear'), 'ranges':bounds,
                          'n':integer(row['N'],'N'), 'parameters':integer(row['Parameters'],'Parameters'),
                          'rmse_tonne':finite(row['RMSE_tonne'],'RMSE'), 'mape_percent':finite(row['MAPE_percent'],'MAPE'),
                          'r2':None if pd.isna(row['R2']) else finite(row['R2'],'R2'),
                          'condition_number':finite(row['ConditionNumber'],'ConditionNumber',True),
                          'class_counts':counts,
                          'diagnostics':diagnostics[diagnostics['Target']==name].drop(columns='Target').to_dict('records')}
        bundle={'version':3,'models':models,'warnings':notes,'excluded_rows':excluded}
        validate_model_types(bundle)
        return bundle
    except (KeyError, TypeError, IndexError) as e:
        raise ValueError('모델 엑셀의 필수 시트 또는 열이 잘못되었습니다. Save Regression Model로 저장한 파일을 선택하세요.') from e


def diagnostic_rows(model, ship_type=None):
    """Filter the fitted full-fleet model's diagnostics; do not fit a new model."""
    return [r for r in model.get('diagnostics', []) if ship_type is None or r['함종'] == ship_type]

def diagnostic_metrics(rows):
    if not rows: return {'n': 0, 'rmse_tonne': None, 'mape_percent': None, 'r2': None}
    actual=np.array([r['실제중량'] for r in rows], dtype=float)
    predicted=np.array([r['예측중량'] for r in rows], dtype=float)
    denominator=float(((actual-actual.mean())**2).sum())
    return {'n': len(rows), 'rmse_tonne': float(np.sqrt(np.mean((actual-predicted)**2))),
            'mape_percent': float(np.mean(np.abs((predicted-actual)/actual))*100),
            'r2': None if denominator==0 else float(1-((actual-predicted)**2).sum()/denominator)}
