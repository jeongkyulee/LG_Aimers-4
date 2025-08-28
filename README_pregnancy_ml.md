
# IVF/DI 임신 성공 예측 — 데이터 전처리 & 모델링 리포트

본 문서는 업로드된 두 개의 노트북(`데이터_전처리_0220.ipynb`, `LG_aimers_catboost.ipynb`)을 통합해 **전처리 → 모델링 → 제출 파일 생성**까지의 전체 파이프라인을 상세히 정리한 README입니다.

> **예측 목표(Target)**: `임신 성공 여부` (이진 분류, 0/1)

---

## 1. 데이터 및 환경

- **입력 파일**: `train.csv`, `test.csv`, `submission.csv` (각 노트북에서 경로는 비워져 있으므로 실제 경로를 지정해야 합니다)
- **개발 환경**: Google Colab/로컬 Jupyter (두 노트북 모두 Colab 의존 코드 포함)
- **주요 라이브러리**
  - 공통: `pandas`, `scikit-learn`
  - 모델링(1): `autogluon.tabular` (AutoGluon TabularPredictor)
  - 모델링(2): `h2o` (H2O AutoML)
  - 모델링(3): `catboost` (CatBoostClassifier)

---

## 2. 전처리 파이프라인 (`데이터_전처리_0220.ipynb`)

### 2.1 데이터 로드 & 사본 생성
```python
train = pd.read_csv('...')
test  = pd.read_csv('...')
submission = pd.read_csv('...')

train_copy = train.copy()
test_copy  = test.copy()
```

### 2.2 컬럼 삭제(Drop) — 정보 누수/중복/유용성 낮은 특성 제거
노트북에서는 아래 **유형의 변수들**을 다수 제거합니다. (일부는 노트북 상 표기가 생략(…)되어 카테고리 단위로 정리함)

- **유전 검사/진단 관련 플래그**: `착상 전 유전 검사 사용 여부`, `PGD 시술 여부`, `PGS 시술 여부`, `착상 전 유전 진단 사용 여부` 등  
- **불임 원인 상세 분류(다단계 플래그)**: `여성/남성/부부 주/부 불임 원인`, `불임 원인 - 여성 요인/자궁경부 문제/.../정자 운동성/정자 형태/정자 면역학적 요인` 등
- **기증/대리모 관련 변수**: `기증 배아 사용 여부`, `대리모 여부`, `정자/난자 기증자 나이` 등
- **시계열 경과일/ID 등**: `난자 해동/채취/혼합/배아 해동 경과일`, `ID` 등
- **그 외**: 필요에 따라 테스트셋에서만 추가 제거(예: 집계/누수 가능 변수)

실제 코드 예:
```python
drop_cols_final = [
    # 유전검사/진단, 불임 원인 다단계 플래그, 기증/대리모/기증자 나이,
    # 경과일(채취/혼합/해동) 및 보조 식별자 등
    # (노트북 내 일부는 '...'로 축약되어 있음)
]
train_copy = train_copy.drop(drop_cols_final, axis=1)
test_copy  = test_copy.drop([...], axis=1)  # 테스트 전용 일부 추가 제거
```

### 2.3 이상치(Outlier) IQR 클리핑
- **대상 열**: 배아/난자/정자 수 등 수치형 컬럼 집합 `numeric_cols`
- **규칙**: (Q1−1.5·IQR) 미만 → 하한으로, (Q3+1.5·IQR) 초과 → 상한으로 **클리핑**
- **주의**: `-1`은 결측/특수 표기 가정, IQR 계산에서 제외

```python
def out_iqr(s):
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - 1.5*IQR, Q3 + 1.5*IQR

for col in numeric_cols:
    lower, upper = out_iqr(train_copy[train_copy[col] != -1][col])
    train_copy.loc[train_copy[col] <  lower, col] = lower
    train_copy.loc[train_copy[col] >= upper, col] = upper
    test_copy.loc [test_copy[col]  <  lower, col] = lower
    test_copy.loc [test_copy[col]  >= upper, col] = upper
```

### 2.4 범주형 정제(카테고리 정규화)
- `정자 출처`, `난자 출처` 등에서 **'알 수 없음'**/미할당 값을 **의미 있는 카테고리**로 치환
  - 예: `정자 출처 == "미할당"` → `"기증 제공"`  
       `난자 출처 == "본인 제공" & 난자 기증자 나이 == "알 수 없음"` → `"난자 본인 제공"`
- **목적**: 누락/애매한 값을 다운스트림 모델이 다루기 쉽게 **명시적 클래스**로 변환

### 2.5 다중선택 문자열 특성 → 멀티핫 인코딩 + 집계
- **특정 시술 유형**: `"IUI/IVF/ICSI/.../FER"` 형태의 **슬래시 구분 문자열**
  - 각 키워드별 **이진 플래그(0/1)** 생성  
  - **복합 시술 수** = `문자열 내 "/" 개수 + 1` (없으면 1, 결측은 0으로 대체)
  - 원본 열 제거
```python
ops = ['IUI','ICI','Generic DI','IVI','GIFT','ICSI','IVF','BLASTOCYST','AH','FER']
for op in ops:
    train_copy[op] = train_copy['특정 시술 유형'].str.contains(op, na=False).astype(int).astype(str)
    test_copy [op] =  test_copy['특정 시술 유형'].str.contains(op, na=False).astype(int).astype(str)

train_copy['복합 시술 수'] = train_copy['특정 시술 유형'].str.count('/') + 1
test_copy ['복합 시술 수'] =  test_copy['특정 시술 유형'].str.count('/') + 1
train_copy['복합 시술 수'] = train_copy['복합 시술 수'].fillna(0)
test_copy ['복합 시술 수'] =  test_copy['복합 시술 수'].fillna(0)
train_copy.drop('특정 시술 유형', axis=1, inplace=True)
test_copy .drop('특정 시술 유형', axis=1, inplace=True)
```

- **배아 생성 주요 이유**: `"현재 시술용/배아 저장용/기증용"` 포함 여부를 **개별 플래그**로 변환
  - 생성 열 예: `배아 생성 이유_현재 시술용`, `배아 생성 이유_배아 저장용`, `배아 생성 이유_기증용`
  - 원본 열 제거

### 2.6 저장(중간 산출물)
```python
train_copy.to_csv('train_processed.csv', index=False, encoding='utf-8-sig')
test_copy .to_csv('test_processed.csv',  index=False, encoding='utf-8-sig')
```

---

## 3. 모델링 A — AutoGluon Tabular (`LG_aimers_catboost.ipynb`)

### 3.1 데이터 분할
```python
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split

target = '임신 성공 여부'
train_data, val_data = train_test_split(train, test_size=0.2, random_state=42, stratify=train[target])
```

### 3.2 학습 & 검증 (개념 요약)
- `TabularPredictor`로 **AutoML 탐색** (다양한 기반 모델/스택 자동 시도)
- 내부적으로 **교차검증/블렌딩**을 수행하며, 검증셋으로 성능 점검
- 대표 평가 지표: `Accuracy`, `F1`, `ROC-AUC` (노트북에서 `sklearn.metrics` 로드)

### 3.3 예측 & 임계값 적용
```python
y_test_pred_proba = predictor.predict_proba(test)        # 양성 확률
y_test_pred       = (y_test_pred_proba[:, 1] >= 0.5).astype(int)  # 임계값 0.5
```

### 3.4 제출 파일 생성
```python
submission = pd.read_csv('submission.csv')
submission['임신 성공 여부'] = y_test_pred
submission.to_csv('submission_autogluon.csv', index=False)
```

---

## 4. 모델링 B — H2O AutoML (`LG_aimers_catboost.ipynb`)

### 4.1 초기화 & 학습 (개념 요약)
```python
import h2o
from h2o.automl import H2OAutoML

h2o.init()
# (데이터프레임을 H2OFrame으로 변환 후)
# aml = H2OAutoML(max_models=..., seed=..., balance_classes=True, nfolds=...)
# aml.train(x=특성목록, y=target, training_frame=...)
```

- **여러 모델(GBM/DRF/GLM/DeepLearning/StackedEnsemble 등)**을 자동 탐색
- 내부 **교차검증/클래스 불균형 보정** 옵션 제공

### 4.2 예측 & 제출 파일 생성
```python
# preds = aml.leader.predict(test_h2o)   # 확률 예측
submission = pd.read_csv('submission.csv')
submission['임신 성공 여부'] = y_test_pred  # or preds.as_data_frame()['p1']
submission.to_csv('submission_h2o_automl.csv', index=False)

# 컬럼명을 'probability'로 변경하는 변형 버전도 존재
submission.rename(columns={'임신 성공 여부':'probability'}, inplace=True)
submission.to_csv('submission_h2o_automl_final.csv', index=False)
h2o.shutdown(prompt=False)
```

---

## 5. 모델링 C — CatBoost (DI 서브셋, `데이터_전처리_0220.ipynb`)

### 5.1 DI/IVF 서브셋 분리
```python
train_di  = train[train['시술 유형'] == 'DI']
train_ivf = train[train['시술 유형'] == 'IVF']
```

### 5.2 CatBoostClassifier 학습 (DI 서브셋 예시)
```python
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

df = train_di.copy()
target_col = '임신 성공 여부'
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

X = df.drop(columns=[target_col])
y = df[target_col]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

best_params = dict(learning_rate=0.1, iterations=3000, depth=6)

model = CatBoostClassifier(
    **best_params,
    auto_class_weights='Balanced',  # 클래스 불균형 자동 보정
    cat_features=categorical_cols,  # 범주형 자동 인코딩
    verbose=500,
    random_state=42,
)
model.fit(X_tr, y_tr)
# (성능 평가는 accuracy/f1/roc_auc 등으로 별도 계산 가능)
```

**특징**
- **범주형 직접 처리**(원핫 불필요) → 고카디널리티에 유리
- `auto_class_weights='Balanced'`로 **클래스 불균형** 자동 보정
- 충분한 `iterations`(3,000)으로 **학습 용량 확보**

---

## 6. 제출물/산출물 정리

- 전처리 결과: `train_processed.csv`, `test_processed.csv`
- 제출 파일(예시):
  - AutoGluon: `submission_autogluon.csv`
  - H2O AutoML: `submission_h2o_automl.csv`, `submission_h2o_automl_final.csv`
- (선택) CatBoost 예측 결과를 동일 포맷으로 저장하여 **블렌딩/스태킹**에 활용 가능

---

## 7. 재현 방법 (로컬/Colab)

1) **환경 준비**
```bash
pip install pandas scikit-learn catboost
pip install -U autogluon.tabular
pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o
```

2) **데이터 경로 지정** (두 노트북의 빈 경로 `''` 부분을 실제 파일 경로로 수정)

3) **전처리 노트북 실행** → `train_processed.csv`, `test_processed.csv` 생성

4) **모델링 노트북 실행**
   - AutoGluon/H2O: 전체 데이터 대상으로 AutoML 학습 → 제출 파일 생성
   - CatBoost: `시술 유형 == 'DI'` 서브셋 등 부분 데이터 대상으로 별도 학습/평가

5) **(선택) 앙상블**
   - 서로 다른 제출 파일을 평균/가중 평균으로 블렌딩해 최종 제출 생성

---

## 8. 설계 의도 & 권장 개선점

- **설계 의도**
  - 정보 누수/중복 가능성이 높은 특성 제거로 **모델 일반화** 강화
  - 다중 선택 문자열을 **명시적 이진 피처 + 집계(복합 시술 수)**로 변환해 **해석성/성능** 확보
  - AutoML(Autogluon/H2O)로 **빠른 베이스라인 확보**, CatBoost로 **범주형 강점** 극대화

- **개선 제안**
  1. **평가 파이프라인 명시화**: `accuracy`, `f1`, `roc_auc` 지표를 공통 함수로 정의하여 AutoML/CatBoost 모두 동일 기준으로 비교
  2. **교차검증**: Stratified K-Fold 적용으로 분할 편차 축소
  3. **클래스 불균형 처리**: `class_weight`, `SMOTE`, `threshold 튜닝(0.5→Youden J 최적)` 실험
  4. **피처 중요도**: Permutation/SHAP으로 중요 피처 해석 및 전처리/선택 개선
  5. **서브셋별 모델**: `DI`, `IVF` 등 **시술 유형별 전용 모델** 학습 후 앙상블
  6. **하이퍼파라미터 탐색**: CatBoost `depth/l2_leaf_reg/learning_rate/od_type` 등 그리드/베이지안 최적화

---

## 9. 참고 체크리스트

- [ ] `train/test/submission` 실제 경로 채움
- [ ] 전처리 동일 규칙이 **train/test 모두**에 적용되었는지 확인
- [ ] 수치형 `-1` 처리 로직 유지/수정 여부 결정
- [ ] 제출 컬럼명(`임신 성공 여부`/`probability`) 대회 규격과 일치 확인
- [ ] 랜덤시드/분할 방식 고정(재현성)
