# without year

연도 보정을 제거한 별도 Windows 프로그램입니다. 기존 main 브랜치의 Naval Ship Weight Estimator와 기존 EXE는 변경하지 않습니다.

실적함 파일 업로드 시 연도 없는 식으로 모든 계수를 새로 추정합니다. 이전 연도 포함 모델의 계수를 그대로 사용하지 않습니다. 기존 실적함 엑셀도 읽을 수 있으며 취역연도 열은 기록으로만 보존합니다. 이 버전 양식과 설계 화면에서는 연도 입력을 제거했습니다. 순항속력도 기존과 같이 추정식에는 사용하지 않습니다.

## 추정식

- 100: K(LBD)^α Dt C
- 200: KP^α Dt C
- 300: K PE^α L^β Dt C
- 400: K Dt C
- 500: K(LBD)^α PE^β Dt C
- 600: K(LB)^α Dt C
- 700: K Dt C
- 전체 경하중량 직접 회귀: K(LBD)^α P^β Dt C

계열 합산 총중량과 전체 경하중량 직접 회귀값을 모두 표시합니다. C는 K와 별도인 역변환 편향 보정계수입니다. 기준 함종은 유효 학습 척수가 가장 많은 함종이며 초계함·초계정은 호위함으로 합칩니다. 순번, 함유형, 오류 행 제외, 함종별 결과, 그래프 및 수식 이미지 가이드는 유지합니다.

400·700은 함종만 사용하므로 동일 함종의 예측 중량이 같습니다. 연도 보정 제거가 실제 정확도 개선을 보장하지는 않으므로 같은 실적함 데이터로 기존 버전과 비교하세요. 기존 연도 포함 모델과 모델 엑셀 형식을 구분하여 혼용을 거부합니다.

## 사용

1. Download Input File Template로 양식 저장.
2. 기존+추가 자료를 합친 전체 엑셀을 Input File Upload로 업로드하여 재학습.
3. Regression에서 적합도·계수·함정별 오차 확인. Save Regression Model로 without year 전용 엑셀 모델 저장.
4. Weight Estimation에서 함종·길이·폭·깊이·소요마력·전력 입력.
5. Reports에서 예측 결과 엑셀 저장.

## 개발 및 Windows 빌드

이 폴더를 작업 디렉터리로 사용합니다. `python -m pip install -r requirements.txt` 후 `python -m unittest discover -s tests -v`로 검사합니다. 수식 PNG는 assets/equations에 포함됩니다. build_windows.bat을 Windows Python 3.12 환경에서 실행하면 dist/without year.exe를 생성합니다. 최종 실행 PC에는 Python이 필요하지 않습니다.
