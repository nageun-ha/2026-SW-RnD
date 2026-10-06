# CIFAR-10 CNN 구조별 성능 비교 — 1주차: 데이터 파이프라인

모델만 바꿔가며 공정하게 비교할 수 있도록 **데이터 분할·전처리를 고정**한 코드입니다.

## 폴더 구조

```
cifar10_cnn/
├── config.py          # 모든 설정값 (시드, 분할 크기, 배치 크기, 증강 on/off, 경로)
├── data.py            # 로딩 · 분할 · transform · DataLoader + 검증 체크
├── eda.py             # 탐색적 데이터 분석 및 시각화
├── requirements.txt
├── splits/            # 분할 인덱스 + 정규화 통계 (★ 팀원과 공유)
├── figures/           # EDA 결과 그림/표
└── data/              # CIFAR-10 원본 (자동 다운로드, 공유 불필요)
```

## 실행 방법

```bash
conda activate swRnD
pip install -r requirements.txt   # 처음 한 번 (GPU 없으면 CPU 버전 torch로 충분)

python data.py   # 데이터 다운로드 + 분할 생성 + 검증 체크
python eda.py    # EDA 그림/표를 figures/에 저장
```

- 처음 실행하면 CIFAR-10(약 170MB)이 `data/`에 다운로드됩니다.
- `splits/` 폴더가 이미 있으면 그 분할을 그대로 사용합니다.
  **팀원은 반드시 같은 `splits/` 파일을 받아서 사용**해 주세요.

## 이후 주차에서 사용하는 법

```python
from data import set_seed, get_dataloaders, get_test_loader

set_seed()                                        # 맨 처음 한 번
train_loader, val_loader = get_dataloaders()      # 증강 여부는 config.USE_AUGMENTATION
# train_loader, val_loader = get_dataloaders(augment=False)  # baseline 비교용

# ... 학습 및 val로 모델 선택 ...

test_loader = get_test_loader(confirm_final_eval=True)  # 최종 평가 때만!
```

## 설계 결정 요약

| 항목 | 결정 | 이유 |
|---|---|---|
| 분할 | train 45,000 / val 5,000 (클래스당 500장, stratified), 시드 42 | 클래스 비율 유지 + 재현성 |
| 분할 공유 | `splits/split_seed42.json`에 인덱스 저장 | 라이브러리 버전이 달라도 팀원 모두 같은 분할 사용 |
| 테스트셋 | `get_test_loader(confirm_final_eval=True)`로만 접근 가능 | 튜닝 과정에서 실수로 쓰는 것 방지 |
| 정규화 통계 | train 45,000장으로만 계산, `splits/norm_stats_seed42.json`에 저장 | val/test 정보 누수 방지 |
| transform 분리 | train용·평가용 CIFAR10 객체를 따로 만들고 같은 인덱스로 `Subset` | `random_split`은 transform을 공유해서 val에도 증강이 들어가는 문제 |
| 증강 | 패딩 4 + 랜덤 크롭 32, 좌우반전 p=0.5 (train만) | CIFAR-10 표준 증강, `USE_AUGMENTATION`으로 on/off |
| 재현성 | random·numpy·torch 시드 + DataLoader generator 고정, `NUM_WORKERS=0` | Windows/CPU 환경 안정성 |

## 검증 체크 (`python data.py`)

1. train/val 인덱스 중복 없음, 합치면 50,000장 전체
2. 같은 시드로 재분할하면 저장된 파일과 동일
3. 전체·train·val의 클래스 비율 차이 ≤ 0.1%p
4. val 로더 배치가 반복해서 꺼내도 동일하고, 원본 이미지에 평가용 transform만 적용한 결과와 일치 (증강 없음)
5. 배치 shape / 값 범위 / 채널별 평균·표준편차 출력

## EDA 결과물 (`figures/`)

- `class_distribution.png`, `.csv` — split별 클래스 개수
- `sample_grid.png` — 클래스별 샘플 이미지
- `normalization_stats.csv` — 정규화 전후 채널별 평균/표준편차
- `augmentation_compare.png` — 원본 vs 증강 결과

## 주의 사항

- `config.py`의 `SEED`, `VAL_SIZE`를 바꾸면 새로운 분할 파일이 생성됩니다. 바꿀 땐 팀원과 먼저 합의해 주세요.
- GPU 환경에서도 `cudnn.deterministic=True`로 고정했지만, GPU 종류에 따라 학습 결과가 아주 미세하게 다를 수 있습니다.
