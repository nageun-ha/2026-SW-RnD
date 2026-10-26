# CIFAR-10 CNN 구조별 성능 비교

- 1주차: 데이터 분할·전처리 고정 (모델만 바꿔가며 공정하게 비교하기 위함)
- 2주차: Baseline CNN + 학습/검증/평가 파이프라인 + 파이프라인 검증
- 3주차: 구조 변형 5종 (bn, dropout, deep, wide, gap) — Baseline과 한 가지씩만 다름

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
├── sanity_check.py    # 파이프라인 검증 4종
├── requirements.txt
├── splits/            # 분할 인덱스 + 정규화 통계 (★ 팀원과 공유)
├── figures/           # EDA 결과 그림/표
├── results/           # 실험별 결과 폴더 (체크포인트 .pt는 git 제외)
└── data/              # CIFAR-10 원본 (자동 다운로드, 공유 불필요)
```

---

# 3주차: 1차 구조 비교용 모델 변형

모든 변형은 Baseline과 **딱 한 가지만** 다르다. 공통 빌더 `models/builder.py`의
`ConfigurableCNN`을 상속하고 옵션 하나만 바꿔 넘긴다. (`baseline.py`, `train.py`, `engine.py`는 그대로)

| 이름 | Baseline 대비 변경점 | 파라미터 | Baseline 대비 | 확인하려는 것 |
|---|---|---|---|---|
| baseline | (기준) Conv-ReLU-Pool × 3 + FC × 2 | 620,362 | 1.00x | — |
| bn | 각 Conv 뒤 BatchNorm2d (Conv bias 제거) | 620,586 | +224 | 정규화 효과 |
| dropout | FC 사이 Dropout(0.5) | 620,362 | 0 | 규제 효과 |
| deep | 블록마다 Conv3x3 2개 (VGG식) | 814,122 | 1.31x | 깊이 효과 |
| wide | 채널 2배 (64, 128, 256) | 1,422,218 | 2.29x | 너비 효과 |
| gap | GAP + FC 1개 (Flatten+FC 2개 대신) | 94,538 | 0.15x | 분류부 구조 효과 |

```bash
python model_summary.py              # 빌더 기본값 = Baseline 확인 + 위 표 출력
python sanity_check.py --model bn    # 새 모델 파이프라인 검증
python train.py --model bn           # 학습 (나머지 하이퍼파라미터는 Baseline과 동일하게)
```

**결과 해석 시 주의**
- wide: 마지막 채널이 2배라 첫 FC 입력도 2048 → 4096으로 커진다. 파라미터 증가분의 대부분이 이 FC.
- gap: 파라미터가 약 1/6로 줄어 "분류부 구조" 효과와 "모델 크기 감소" 효과가 섞여 있다.
- bn: Conv bias를 뺀 이유 — BN이 배치 평균을 빼면서 bias가 상쇄되고, BN의 β가 같은 역할을 하기 때문 (표현력 동일).

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
