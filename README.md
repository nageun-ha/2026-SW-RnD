# CIFAR-10 CNN 구조별 성능 비교

- 1주차: 데이터 분할·전처리 고정 (모델만 바꿔가며 공정하게 비교하기 위함)
- 2주차: Baseline CNN + 학습/검증/평가 파이프라인 + 파이프라인 검증
- 3주차: 구조 변형 5종 (bn, dropout, deep, wide, gap) — Baseline과 한 가지씩만 다름 + 비교 실험 일괄 실행

## 폴더 구조

```
cifar10_cnn/
├── config.py          # 모든 설정값 (시드, 분할, 장치 선택, 학습 기본 하이퍼파라미터)
├── data.py            # 로딩 · 분할 · transform · DataLoader + 데이터 검증 체크
├── eda.py             # 탐색적 데이터 분석 및 시각화
├── models/            # 모델 레지스트리 (파일 하나 추가 = 모델 하나 추가)
│   ├── __init__.py
│   ├── baseline.py    # BaselineCNN (Conv-ReLU-Pool × 3 + FC × 2), 수정 금지
│   ├── builder.py     # 3주차 공통 빌더 ConfigurableCNN (기본값 = Baseline)
│   └── bn.py, dropout.py, deep.py, wide.py, gap.py   # 3주차 변형
├── model_summary.py   # 빌더=Baseline 동일성 검사 + 모델별 파라미터 표
├── engine.py          # train_one_epoch / evaluate / confusion_matrix
├── train.py           # 학습 실행 (커맨드라인 인자로 하이퍼파라미터 변경)
├── evaluate.py        # 클래스별 정확도 + confusion matrix
├── sanity_check.py    # 파이프라인 검증 4종 (overfit은 eval 모드 판정)
├── run_experiments.py # 3주차 비교 실험 일괄 실행 (모델 × 시드, 완료분 건너뛰기)
├── compare_results.py # 3주차 결과 취합 → 요약 CSV + 비교 그림 4종
├── requirements.txt
├── splits/            # 분할 인덱스 + 정규화 통계 (★ 팀원과 공유)
├── figures/
│   ├── *.png, *.csv   # 1주차 EDA 그림/표
│   └── week3/         # 3주차 비교 그림 4종 (학습 곡선, loss, 정확도 막대, 파라미터 vs 정확도)
├── results/
│   ├── <실험이름>/     # 실험별 args.json, history.csv, val_*.csv/png (best.pt는 git 제외)
│   ├── model_summary.csv           # 모델별 파라미터 수
│   ├── week3_summary.csv           # 3주차 모델별 요약 (시드 평균 ± 표준편차)
│   └── week3_per_class_val_acc.csv # 3주차 클래스별 val 정확도
└── data/              # CIFAR-10 원본 (자동 다운로드, 공유 불필요)
```

---

# 3주차: 1차 모델 구조 비교

Baseline과 **딱 한 가지씩만** 다른 변형 5종을 추가하고, 6개 모델 × 시드 3개를 같은 조건으로 학습해 비교한다.
변형은 공통 빌더 `models/builder.py`의 `ConfigurableCNN`을 상속해 옵션 하나만 바꿔 넘긴다.
(`baseline.py`, `train.py`, `engine.py`는 수정하지 않음)

## 실행 순서와 정상 출력

```bash
python model_summary.py
```
정상이면 `[통과]` 층 구성 동일 / 파라미터 수 동일 / 가중치 이름·shape 동일 / 출력 동일, 이어서 모델별 파라미터 표와
`[통과] 모든 모델 출력 shape = (2, 10)`. 표는 `results/model_summary.csv`로 저장.

```bash
python sanity_check.py --model bn        # bn, dropout, deep, wide, gap 각각 실행
```
정상이면 4개 모두 `[통과]`. 실측값(RTX 5060, seed 42):

| 모델 | 초기 loss (이론값 2.3026) | overfit 도달 step (eval 모드 acc ≥ 0.99) |
|---|---|---|
| baseline | 2.3057 | 26 |
| bn | 2.3042 | 26 |
| dropout | 2.3057 | 32 |
| deep | 2.3028 | 31 |
| wide | 2.3032 | 21 |
| gap | 2.3115 | 74 |

```bash
python run_experiments.py --dry-run      # 실행할 명령과 예상 시간만 출력
python run_experiments.py                # 실제 실행
```
- 시작: `실험 18개 = 모델 6개 × 시드 3개 (20 epoch) | 완료 1개, 실행 예정 17개` (baseline seed 42는 2주차 결과 재사용)
- 실험마다: `[n/17] <실험이름> | 경과 H:MM:SS | 예상 남은 시간 H:MM:SS` 뒤에 train.py 학습 로그
- 끝: `전체 종료 | 총 소요 ... | 성공 17개 / 실패 0개`
- 예상 소요: 약 **1시간 50분** (RTX 5060, 모델별 1 epoch 실측 기반). 중간에 꺼져도 다시 실행하면 이어서 진행
- 여러 명이 동시에 돌릴 때는 시드를 나눈다: `python run_experiments.py --seeds 1`

```bash
python compare_results.py                # 결과 취합 → results/week3_*.csv + figures/week3/*.png
```
- 18개 실험의 `history.csv`가 `results/` 아래에 모두 있어야 한다 (팀원 결과는 `git pull`로 받기)
- 클래스별 정확도는 `best.pt`가 있는 실험만 `evaluate.py`로 계산된다 (`--no-eval`이면 기존 CSV만 사용)

## 비교한 변형

| 이름 | Baseline 대비 변경점 | 파라미터 | Baseline 대비 | 1 epoch 실측 | 확인하려는 것 |
|---|---|---|---|---|---|
| baseline | (기준) Conv-ReLU-Pool × 3 + FC × 2 | 620,362 | 1.00x | 14.5초 | — |
| bn | 각 Conv 뒤 BatchNorm2d (Conv bias 제거) | 620,586 | +224 | 15.2초 | 정규화 효과 |
| dropout | FC 사이 Dropout(0.5) | 620,362 | 0 | 15.1초 | 규제 효과 |
| deep | 블록마다 Conv3x3 2개 (VGG식) | 814,122 | 1.31x | 18.8초 | 깊이 효과 |
| wide | 채널 2배 (64, 128, 256) | 1,422,218 | 2.29x | 17.6초 | 너비 효과 |
| gap | GAP + FC 1개 (Flatten+FC 2개 대신) | 94,538 | 0.15x | 16.8초 | 분류부 구조 효과 |

학습 조건은 전부 동일: SGD(momentum 0.9), lr 0.01, weight decay 5e-4, batch 128, 20 epoch, 증강 ON, 시드 42 / 1 / 2.

**결과 해석 시 주의**
- wide: 마지막 채널이 2배라 첫 FC 입력도 2048 → 4096으로 커진다. 파라미터 증가분의 대부분이 이 FC.
- gap: 파라미터가 약 1/6로 줄어 "분류부 구조" 효과와 "모델 크기 감소" 효과가 섞여 있다.
- 학습 시간은 파라미터 수에 비례하지 않는다 (CPU 데이터 로딩·증강이 병목, `NUM_WORKERS=0`).

## 결과 요약

6개 모델 × 시드 3개, validation 기준 (테스트셋 미사용). 전체 표: `results/week3_summary.csv`

| 순위 | 모델 | 파라미터 | best val acc (%) | 마지막 5 epoch 평균 (%) | Baseline 대비 (%p) | 차이 > 시드 표준편차? |
|---|---|---|---|---|---|---|
| 1 | bn | 620,586 | **82.73 ± 0.73** | 81.41 ± 0.37 | **+2.23** | 예 |
| 2 | wide | 1,422,218 | 82.64 ± 0.58 | 81.47 ± 0.22 | +2.14 | 예 |
| 3 | deep | 814,122 | 81.92 ± 1.60 | 80.74 ± 1.13 | +1.42 | 아니오 |
| 4 | baseline | 620,362 | 80.50 ± 0.64 | 79.80 ± 0.84 | — | — |
| 5 | dropout | 620,362 | 79.93 ± 0.26 | 78.89 ± 0.24 | −0.57 | 아니오 |
| 6 | gap | 94,538 | 72.10 ± 0.38 | 70.47 ± 0.14 | **−8.40** | 예 |

- 표준편차는 시드 3개의 표본 표준편차. "차이 > 시드 표준편차?"가 "아니오"인 deep·dropout은 시드 운과 구분되지 않는다.
- **bn**: 파라미터 +224개만으로 가장 큰 향상, 초반 수렴도 가장 빠름 (1 epoch val acc 약 56% vs baseline 약 45%).
- **wide**: bn과 비슷한 향상이지만 파라미터 2.3배. **deep**: 초반이 느리고(1 epoch 약 24%) 시드 간 편차가 큼.
- **dropout**: 20 epoch에서는 오히려 소폭 하락. train-val 격차가 거의 없는 상태라 규제가 도움이 되지 않음.
- **gap**: 3블록 후 4×4 특징을 평균으로 뭉개고 FC 은닉층을 없애 크게 하락 (모델 크기 1/6 효과와 섞여 있음).

**클래스별 val 정확도** (상위 3개 + baseline, 시드 평균, `results/week3_per_class_val_acc.csv`)
- 모든 모델에서 **cat이 가장 낮음**: baseline 56.1% → bn 63.7% / deep 65.8% / wide 68.5%. 구조 변경으로 가장 많이 오른 클래스도 cat (wide +12.4%p).
- baseline의 cat은 시드 간 편차가 매우 큼 (표준편차 13.5%p) → 클래스 단위 비교는 시드 3개로도 불안정.

**주의**: baseline seed 42는 2주차 결과(RTX 5060, epoch당 약 17초)를 재사용했고, 나머지 17개 실험은 다른 장치(epoch당 약 7초)에서 학습됨.
그래서 baseline의 epoch 시간 표준편차가 크게 나온다(10.37 ± 5.98초, 중앙값 6.92초). 정확도 비교에는 장치 간 미세한 수치 차이만 영향을 준다.

## 그림

모두 `figures/week3/` 폴더에 있다 (`python compare_results.py`로 생성).

| 그림 | 경로 | 내용 |
|---|---|---|
| 학습 곡선 (val acc) | `figures/week3/1_val_acc_curves.png` | epoch별 val acc, 시드 평균선 + 최소~최대 띠, 8~20 epoch 확대 |
| 학습 곡선 (loss) | `figures/week3/2_loss_curves.png` | epoch별 train/val loss |
| best val acc 막대 | `figures/week3/3_best_val_acc_bar.png` | 모델별 평균(막대) ± 표준편차, 시드별 점, baseline 기준선 |
| 파라미터 수 vs 정확도 | `figures/week3/4_params_vs_acc.png` | 모델 크기 대비 성능 |

## 3주차 설계 결정

| 항목 | 결정 | 이유 |
|---|---|---|
| 변형 구현 | 공통 빌더 + 옵션 하나만 다른 작은 클래스 | "Baseline과 한 가지만 다름"을 코드 구조로 보장, 중복 코드 제거 |
| 빌더 동일성 검사 | Baseline 가중치를 `strict=True`로 불러오고 출력까지 비교 | 파라미터 수만 같아서는 구조가 같다는 보장이 없음 (층 번호까지 일치 확인) |
| BN 사용 시 Conv bias | `bias=False` | BN이 배치 평균을 빼며 bias를 상쇄, BN의 β가 같은 역할 → 표현력 동일 |
| dropout=0 | Dropout 층 자체를 넣지 않음 | 층 번호가 밀리면 Baseline과 가중치 이름이 달라짐 |
| overfit 판정 | 매 step 업데이트 후 eval 모드 정확도로 판정 | 실측: dropout은 train 모드 acc가 도달 후에도 31번 99% 미만으로 흔들림, bn은 train 14 / eval 26 step으로 차이 |
| 반복 실험 | 시드 3개(42, 1, 2), 분할은 `splits/` 고정 | 시드 운에 따른 차이와 구조 효과를 구분하기 위함 |
| 실행 순서 | 시드 바깥, 모델 안쪽 | 중간에 멈춰도 "모든 모델 × 시드 1개" 비교가 먼저 가능 |
| 이어서 실행 | `history.csv`가 목표 epoch까지 있으면 건너뜀 | 중단 후 같은 명령으로 재개, 팀원 결과도 pull하면 인식 |
| 팀 분담 | 시드별로 나눠 실행 | 한 시드의 6개 모델이 같은 장치에서 학습 → 장치 차이 없이 모델 비교 |
| `NUM_WORKERS` | 0 유지 | 바꾸면 증강 난수 흐름이 달라져 2주차 baseline 결과를 재사용할 수 없음 |
| 비교 지표 | best val acc + 마지막 5 epoch 평균 함께 보고 | best는 최댓값이라 낙관적, 마지막 5 epoch 평균이 더 안정적 |
| 차이 판단 | 차이가 시드 표준편차보다 큰지 표시 | 시드 3개로는 작은 차이를 구조 효과라고 단정할 수 없음 |

---

# 2주차: Baseline CNN과 학습 파이프라인

## 실행 순서와 정상 출력

```bash
python sanity_check.py
```
정상이면 4개 모두 `[통과]`이고 마지막 줄이 `요약: shape 통과, init_loss 통과, overfit 통과, overlap 통과`.

| 검증 | 정상 출력 예시 |
|---|---|
| shape | 입력 `(256, 3, 32, 32)` → 출력 `(256, 10)`, 전체 파라미터 `620,362` |
| init_loss | `초기 loss: 2.30xx` (이론값 ln 10 = 2.3026, ±0.3 이내) |
| overfit | step 50 안팎에서 `acc 1.0000`, loss가 0에 가까워짐 |
| overlap | `train 45,000 / val 5,000 / 겹치는 인덱스 0개` |

```bash
python train.py                    # config 기본값: SGD, lr 0.01, batch 128, 20 epoch, 증강 ON
```
- epoch마다 `train loss/acc | val loss/acc | 소요시간` 한 줄 출력, val acc가 갱신되면 `*best` 표시
- 1 epoch 후 val acc 약 0.44, 2 epoch 후 약 0.53. loss는 2.30에서 시작해 점점 감소
- 20 epoch 후 best val acc **약 0.80** (RTX 5060 실측, epoch당 약 17초 / CPU는 약 21초)
- 결과: `results/baseline_sgd_lr0.01_bs128_ep20_aug_seed42/` 에 `args.json`, `history.csv`, `best.pt`

```bash
python evaluate.py --run baseline_sgd_lr0.01_bs128_ep20_aug_seed42
```
- 클래스별 정확도(낮은 순)와 confusion matrix 출력, CSV·PNG 저장
- CIFAR-10에서는 보통 cat·bird·dog가 낮고 automobile·truck·ship이 높게 나옴

## 자주 쓰는 옵션

```bash
python train.py --epochs 5 --lr 0.001 --optimizer adam
python train.py --no-augment                 # 증강 없는 baseline
python train.py --seed 1                     # 다른 시드 (train/val 분할은 그대로)
python train.py --run-name my_exp --overwrite
python sanity_check.py --check overfit       # 검증 하나만
```

## 새 모델 추가 (3~4주차)

1. `models/새모델.py` 파일을 만들고 클래스 위에 `@register_model("새모델")`을 붙인다.
2. `python sanity_check.py --model 새모델` 로 검증한다.
3. `python train.py --model 새모델` 로 학습한다.

다른 파일은 고칠 필요가 없다.

## 2주차 설계 결정

| 항목 | 결정 | 이유 |
|---|---|---|
| 장치 | cuda → mps → cpu 자동 선택, cuda는 실제 연산 테스트까지 | GPU가 있어도 PyTorch 빌드가 안 맞으면 자동으로 다음 장치 사용 |
| 학습 루프 | `engine.py` 하나를 train·evaluate·sanity_check가 공유 | 검증한 코드 = 학습에 쓰는 코드 |
| best 기준 | val 정확도 | 테스트셋은 6주차 `evaluate.py --test` 전까지 사용 금지 |
| 시드 | `--seed`는 초기화·셔플·증강에만 영향, 분할은 `splits/` 고정 | 시드를 바꿔 반복 실험해도 같은 val로 비교 가능 |
| 기록 | 매 epoch `history.csv` 덮어쓰기, `args.json`에 장치·버전 기록 | 중단돼도 기록 보존, 팀원 간 결과 차이 추적 |
| Baseline | BN·Dropout 없음 | 이후 구조 개선 효과를 비교할 기준점 |

> 재현성: 같은 장치에서는 같은 시드로 결과가 완전히 동일하다 (CPU에서 확인).
> cuda / mps / cpu 사이에서는 연산 방식 차이로 소수점 아래 결과가 조금 다를 수 있다.

---

# 1주차: 데이터 파이프라인

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
