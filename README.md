# CIFAR-10 CNN 구조별 성능 비교

- 1주차: 데이터 분할·전처리 고정 (모델만 바꿔가며 공정하게 비교하기 위함)
- 2주차: Baseline CNN + 학습/검증/평가 파이프라인 + 파이프라인 검증
- 3주차: 구조 변형 5종 (bn, dropout, deep, wide, gap) — Baseline과 한 가지씩만 다름 + 비교 실험 일괄 실행
- 4주차: BN 고정 Deep×Wide 조합 + 큰 모델에서 Dropout·GAP 재확인, 5주차 후보 선정
- 5주차: 후보 2개(bn_deep_wide, bn_deep)의 하이퍼파라미터 튜닝(lr·스케줄러·weight decay) + Grad-CAM 검증 (진행 중)

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
│   ├── bn.py, dropout.py, deep.py, wide.py, gap.py   # 3주차 변형
│   └── bn_deep.py, bn_wide.py, bn_deep_wide.py, bn_deep_wide_dropout.py, bn_deep_wide_gap.py   # 4주차 변형
├── model_summary.py   # 빌더=Baseline 동일성 검사 + 모델별 파라미터 표
├── engine.py          # train_one_epoch / evaluate / confusion_matrix
├── train.py           # 학습 실행 (커맨드라인 인자로 하이퍼파라미터 변경)
├── evaluate.py        # 클래스별 정확도 + confusion matrix
├── sanity_check.py    # 파이프라인 검증 4종 (overfit은 eval 모드 판정)
├── run_experiments.py # 비교 실험 일괄 실행 (--week 3 / 4, --week 5 --stage A~D, 완료분 건너뛰기)
├── compare_results.py # 주차별 결과 취합 → 요약 CSV + 비교 그림 (--week 4 추가 분석, --week 5 단계별 표, --verify)
├── week5_selection.py # 5주차 탐색 범위와 승자 선택 규칙 (run_experiments·compare_results가 같이 사용)
├── gradcam.py         # Grad-CAM 클래스 (직접 구현)
├── visualize_gradcam.py # Grad-CAM 그림 생성 + 구현 검증 (학습 없음)
├── requirements.txt
├── splits/            # 분할 인덱스 + 정규화 통계 (★ 팀원과 공유)
├── figures/
│   ├── *.png, *.csv   # 1주차 EDA 그림/표
│   ├── week3/         # 3주차 비교 그림 4종 (학습 곡선, loss, 정확도 막대, 파라미터 vs 정확도)
│   ├── week4/         # 4주차 비교 그림 4종 (같은 구성)
│   ├── week5/         # 5주차 그림 3종 (A단계 곡선, B단계 wd 막대, 기본 설정 대비 최종 설정)
│   └── gradcam/       # Grad-CAM 그림, 혼동행렬, 선택한 이미지 인덱스 CSV
├── results/
│   ├── <실험이름>/     # 실험별 args.json, history.csv, val_*.csv/png (best.pt 포함)
│   ├── model_summary.csv           # 모델별 파라미터 수
│   ├── week3_summary.csv           # 3주차 모델별 요약 (시드 평균 ± 표준편차)
│   ├── week3_per_class_val_acc.csv # 3주차 클래스별 val 정확도
│   ├── week4_*.csv                 # 4주차 요약, 2×2 효과표, 변형 비교, cat/dog, 20 vs 40 epoch, 클래스별
│   ├── week5_*.csv                 # 5주차 단계 A·B 표, 모델별 최종 설정
│   └── week5_selection.json        # 5주차에 실제로 쓴 승자 기록 (직접 지정한 경우 포함)
└── data/              # CIFAR-10 원본 (자동 다운로드, 공유 불필요)
```

---

# Grad-CAM 시각화

저장된 체크포인트(`results/<실험이름>/best.pt`)로 모델이 이미지의 어느 위치를 보고 판단했는지 그린다. 학습은 하지 않는다.
외부 라이브러리 없이 `gradcam.py`에 직접 구현했다.

> 기본 데이터는 validation이다. `--test`는 6주차 최종 평가 전용이며, 실행하면 경고가 출력된다.
> 체크포인트(`best.pt`)가 있는 실험만 실행할 수 있다. 저장소에는 4주차 21개 실험과 3주차 baseline seed 42의 `best.pt`가 있고,
> 나머지 3주차 실험은 `best.pt`가 없다.

## 실행

```bash
python visualize_gradcam.py --mode check
python visualize_gradcam.py --mode single --runs baseline_sgd_lr0.01_bs128_ep20_aug_seed42
python visualize_gradcam.py --mode compare --runs <run1> <run2> --per-class 1
python visualize_gradcam.py --mode misclassified --runs <run> --pairs cat,dog automobile,truck --max-per-pair 8
```

| 모드 | 그림 |
|---|---|
| `single` | 이미지마다 [원본 / 히트맵 / 오버레이] |
| `compare` | 행=이미지, 열=모델 (첫 열은 원본), 칸마다 예측과 confidence |
| `misclassified` | 헷갈린 샘플의 [원본 / 예측 클래스 CAM / 정답 클래스 CAM] + 혼동행렬 |
| `check` | 구현 검증 7개 (PASS/FAIL, 실패 시 종료 코드 1). 체크포인트가 다른 실험이면 `--runs`로 지정 |

## 주요 옵션

| 옵션 | 설명 |
|---|---|
| `--per-class k` | 클래스별 k장 선택 (기본 1, `config.SEED`로 고정) |
| `--indices 12 57` 또는 `--indices <파일>_indices.csv` | 이미지를 직접 지정. CSV의 split이 현재 실행과 다르면 멈춘다 |
| `--layer N` | `features[N]` 층으로 계산 (생략하면 마지막 Conv 다음 ReLU, 8×8) |
| `--pairs cat,dog` | `misclassified`에서 볼 클래스 쌍 |
| `--max-per-pair N` | 쌍마다 그릴 샘플 수 (기본 8, 양방향에 반씩) |

`--layer` 번호는 구조마다 다른 층을 가리킬 수 있다. 실행할 때 출력되는 층 종류와 CAM 원본 크기를 확인한다.

## 결과 위치

`figures/gradcam/<모드>_<val|test>_<모델들>.png`와 같은 이름의 `_indices.csv`
(`misclassified`는 `_confusion.png`, `_confusion.csv` 추가, `--layer`를 지정하면 이름에 `_layerN` 추가)

6주차에 같은 이미지를 쓰려면 저장된 `_indices.csv`를 `--indices`에 넘긴다. 실험 이름이 길면 파일 이름이 줄어들고, 전체 목록은 CSV의 `runs` 열에 남는다.

## 설계 결정

| 항목 | 결정 | 이유 |
|---|---|---|
| inplace ReLU | 메모리에서만 끄고 전후 logits 일치를 assert로 확인 | 저장한 특징맵이 덮어써지거나 역전파 에러가 나는 것을 방지 |
| 모드 | 항상 `model.eval()` | Dropout과 BN 때문에 결과가 흔들리거나 배치의 다른 이미지에 영향받는 것을 방지 |
| 미분 대상 | softmax 전 logit | softmax를 거치면 다른 클래스 점수가 섞여 해석이 흐려짐 |
| target layer | 마지막 Conv 다음 ReLU를 자동 탐색 | 층 번호를 고정하지 않아 모델 구조가 달라도 동작 |
| 히트맵 색 | `inferno`, 0~1 고정 | 밝기가 한 방향으로 변해 없는 경계가 생기지 않고, 칸끼리 색을 비교할 수 있음 |

## 해석 시 주의

- CAM 원본은 8×8이라 32×32 그림에서도 세밀한 위치까지 가리키지는 않는다.
- 이미지별로 0~1 정규화하므로 이미지끼리 밝기를 비교해 중요도를 판단하면 안 된다.
- 학습되지 않은 모델의 CAM도 그럴듯해 보일 수 있으므로, 예측이 맞았는지와 `--mode check` 결과를 함께 본다.

---

# 5주차: 하이퍼파라미터 튜닝

4주차에서 고른 후보 2개(bn_deep_wide, bn_deep)의 학습률·스케줄러·weight decay를 조정해, 6주차 최종 모델 확정에 쓸 **모델별 최적 설정**을 정한다.
두 모델 중 어느 것을 최종 모델로 할지는 5주차에 정하지 않고 6주차에 정한다. (상태: 튜닝 실험 A~D, Grad-CAM 검증 완료)

## 5주차 선정 기준 (실험 전 작성)

> **결과를 본 뒤에 기준을 바꾸지 않는다.** 기준에서 문제를 발견해도 이번 선정은 이 기준대로 하고, 문제점은 결과와 함께 따로 기록한다.

**고정하는 것** (모든 실험 동일)
- 데이터 분할(`splits/`), 정규화 통계, 증강(패딩 4 + 랜덤 크롭, 좌우반전): 1주차 그대로
- 모델 구조: 4주차의 `bn_deep_wide`, `bn_deep` 파일 그대로
- batch size 128: batch size를 바꾸면 같은 lr이라도 한 번의 업데이트 크기와 epoch당 업데이트 횟수가 달라져 lr의 의미가 함께 바뀐다. 두 값을 같이 튜닝하지 않기 위해 고정한다.
- SGD, momentum 0.9, 40 epoch, `NUM_WORKERS=0`
- validation만 사용한다. 테스트셋은 6주차 최종 평가 전까지 쓰지 않는다.

**지표** (4주차와 같은 정의)
- 주 지표: `last5_val_acc`, 36~40 epoch val acc의 평균. 아래의 모든 판정에 쓴다.
- 보조 지표: `best_val_acc`, train−val 차이(마지막 epoch). 함께 보고하되 판정에는 쓰지 않는다.

**차이 판정**
- 기준값은 **0.45%p로 고정**한다. val 5,000장에서 정확도가 89% 근처일 때의 표본 오차 √(0.89 × 0.11 / 5000) ≈ 0.44%p에서 왔다. 결과를 본 뒤 다시 계산하지 않는다.
  - 4주차 시드 표준편차(0.27~0.36%p)로 계산한 "시드 1개짜리 두 실험의 차이"가 흔들리는 폭(약 0.4~0.5%p)과도 비슷하다. 통계적 유의성 검정은 아니다.
- 1위와의 주 지표 차이가 0.45%p 이하인 설정은 모두 "1위와 차이 없음" 그룹에 넣는다 (1위 포함).
- 그룹에 설정이 2개 이상이면 **기본값(lr 0.01, scheduler none, wd 5e-4)에 가장 가까운 설정**을 고른다. "가깝다"는 아래 순서로 판정한다.
  1. 기본값과 다른 항목 수가 적은 것
  2. lr이 0.01에 가까운 것 (배수 기준: 0.05(5배)가 0.1(10배)보다 가깝다)
  3. wd가 5e-4에 가까운 것 (배수 기준: 1e-3(2배) → 1e-4(5배) → 5e-3(10배) 순으로 가깝다)
  4. scheduler가 none인 것
  - 이유: 차이가 확인되지 않은 변경은 받아들이지 않는다. 바꾼 값이 적을수록 4주차 결과와 비교하기 쉽다.

**탐색 순서** (한 번에 한 가지만 바꾸고, 앞 단계의 승자를 다음 단계의 기준으로 쓴다. 두 모델은 각각 따로 진행한다)

| 단계 | 바꾸는 것 | 고정하는 것 | 시드 | 모델당 새 실험 |
|---|---|---|---|---|
| A | lr {0.1, 0.05, 0.01} × scheduler {none, cosine} = 6조합 | wd 5e-4 | 42 | 5회 (lr 0.01 + none은 4주차 seed 42 결과를 다시 씀) |
| B | wd {1e-4, 1e-3, 5e-3} | A 승자의 lr, scheduler | 42 | 3회 (wd 5e-4는 A 결과를 다시 씀) |
| C | 시드 1, 2 추가 | B 승자 = 모델별 최종 설정 | 1, 2 | 2회 (최종 설정이 기본값이면 4주차 결과를 다시 써서 0회) |
| D (선택) | epoch 60 | 최종 설정 | 42 | 1회 |

- **A, B 승자**: 위의 차이 판정으로 고른다.
- **C (최종 설정 확인)**: 최종 설정의 시드 3개 평균 ± 표본 표준편차를 같은 모델의 4주차 기본 설정(시드 3개)과 비교한다.
  평균 차이가 0.45%p보다 크고, 두 설정의 시드 표준편차 중 큰 값보다도 클 때만 "개선"으로 본다.
  개선이 아니면 그 모델의 최종 설정은 기본값(4주차 설정)으로 둔다.
- **D (수렴 확인, 참고용)**: C의 시드 3개 중 2개 이상에서 best epoch이 36~40이면(4주차 "아직 수렴 안 됨"과 같은 판정) 최종 설정으로 60 epoch, seed 42를 1회 돌린다.
  선정에는 쓰지 않는다. cosine이면 T_max도 60으로 바뀌므로 "같은 설정을 더 오래 돌린 것"이 아님을 결과에 함께 적는다.
- **발산한 실험**: loss가 NaN이 되거나 val acc가 10% 근처에 머무는 실험(예: lr 0.1)도 설정을 바꿔 다시 돌리지 않고, 결과 그대로 표에 남긴다.

**실행 환경과 분담**
- A·B는 seed 42 실험 한 개끼리 비교하고, A의 lr 0.01 + none은 4주차 결과(RTX 5060 PC에서 학습)를 다시 쓴다.
  장치가 다르면 같은 설정·시드라도 학습 경로가 조금씩 달라지므로 **A·B는 4주차를 학습한 PC에서 실행한다.**
- C는 시드별로 팀원이 나눠 돌려도 된다. 이때는 `args.json`의 장치·PyTorch 버전을 결과에 함께 적는다.
- 같은 모델 안에서는 A → B → C 순서를 지켜야 하고, 두 모델은 서로 기다리지 않는다.
- 예상 규모: 새 실험 약 20회(A 10회 + B 6회 + C 4회) × 약 13~15분 ≈ 4.5~5시간 (RTX 5060, 4주차 epoch 시간 기준)

**한계 (실험 전에 미리 적어 둠)**
- A·B는 시드 1개로 승자를 고른다. 시드 운으로 승자가 바뀔 수 있고, C는 승자 하나만 다시 확인하므로 A·B에서 떨어진 설정이 실제로는 더 좋았을 가능성이 남는다.
- val로 고르고 val로 보고하므로 결과가 낙관적이다. 6주차 테스트셋 결과는 이보다 낮게 나올 수 있다.
- 한 번에 하나씩 바꾸므로 lr과 wd의 상호작용을 다 보지 못한다. SGD에서 wd의 실제 효과는 lr에 비례해 커지므로, A에서 lr이 바뀌면 알맞은 wd도 달라질 수 있다. B를 A 승자 위에서 하는 것으로 일부만 반영한다.
- `last5`는 lr을 끝까지 줄이는 cosine에 유리하다. 끝부분 lr이 거의 0이라 epoch마다의 흔들림(4주차 약 1%p)이 사라지기 때문이다. 흔들림이 줄어드는 것도 실제로 얻는 이득이라 지표는 그대로 쓴다.
- 40 epoch 고정이라 수렴이 느린 설정에 불리하다. D는 참고용이다.

## 실행과 취합

단계마다 `--dry-run`으로 실행 목록과 예상 시간을 먼저 확인한다.

```bash
python run_experiments.py --week 5 --stage A --dry-run   # 모델당 5회 (lr 0.01 + none은 4주차 결과 재사용), 약 2시간 20분
python run_experiments.py --week 5 --stage A
python compare_results.py --week 5                        # A 표와 승자 확인 → 팀 확인 후 B로
python run_experiments.py --week 5 --stage B              # A 결과에서 승자를 자동으로 골라 wd 3개 (모델당 3회)
python run_experiments.py --week 5 --stage C              # B 승자의 시드 1, 2 (팀원과 나눌 때: --seeds 1 / --seeds 2)
python run_experiments.py --week 5 --stage D              # (선택) 조건이 맞는 모델만 60 epoch 1회
```
- B·C·D는 앞 단계 결과로 승자를 자동으로 고른다. 앞 단계가 덜 끝났으면 그 모델은 계획하지 않고 이유를 출력한다.
- 자동으로 고를 수 없을 때만 직접 지정한다: `--models bn_deep --lr 0.05 --scheduler cosine`(A 승자), `--weight-decay 0.001`(B 승자).
  실제로 실행하면 쓴 승자가 `results/week5_selection.json`에 기록되고, `compare_results.py --week 5`도 같은 설정을 쓴다.
- 4주차 결과를 다시 쓰는 조합은 결과가 없어도 새로 학습하지 않고 "학습 안 함"으로 알린다 (`git pull`로 받기).
- `compare_results.py --week 5`는 단계가 덜 끝났어도 끝난 부분까지 표를 채운다.
  - `results/week5_stageA.csv`, `week5_stageB.csv`: 설정별 last5, best, train−val 차이, 1위와의 차이, 차이 없음 그룹, 선택된 설정
  - `results/week5_final.csv`: 튜닝한 설정과 4주차 기본 설정의 시드 3개 비교, 개선 판정, 최종 설정, D 필요 여부
  - `figures/week5/`: `1_stageA_val_acc_curves.png`, `2_stageB_wd_last5.png`, `3_final_vs_default.png`
- 3·4주차 취합은 기본 하이퍼파라미터 실험만 읽으므로 5주차 결과가 생겨도 바뀌지 않는다. `--week 3 --verify`, `--week 4 --verify`로 확인할 수 있다.

## 결과 요약

새 실험 22회(A 10 + B 6 + C 4 + D 2), 모두 성공. RTX 5060 한 대에서 총 5시간 30분(A 2:25, B 1:26, C 0:57, D 0:43).
validation 기준이며 테스트셋은 쓰지 않았다. 판정은 위 사전 기준을 그대로 적용했다 (`compare_results.py --week 5`).

**모델별 최종 설정** (`results/week5_final.csv`, 시드 3개 평균 ± 표본 표준편차)

| 모델 | 최종 설정 | last5 val acc (%, 주 지표) | 4주차 기본 설정 | 차이 (%p) | 판정 | best val acc (%) | train−val 차이 (%p) | 시드별 best epoch |
|---|---|---|---|---|---|---|---|---|
| bn_deep_wide | **lr 0.05 + cosine + wd 1e-3** | **92.71 ± 0.27** | 88.91 ± 0.27 | **+3.81** | 개선 | 92.83 ± 0.25 (기본 89.82) | 6.49 ± 0.23 (기본 7.17) | 39, 38, 39 |
| bn_deep | **lr 0.05 + cosine + wd 5e-4** | **90.55 ± 0.19** | 87.55 ± 0.36 | **+3.00** | 개선 | 90.74 ± 0.12 (기본 88.25) | 6.21 ± 0.14 (기본 5.02) | 39, 38, 40 |

- 개선 판정: 두 모델 모두 차이가 0.45%p와 두 설정의 시드 표준편차 중 큰 값(0.27, 0.36)보다 크다.
- 두 모델 중 최종 모델은 6주차에 정한다. 이 표는 모델끼리의 비교가 아니다.

**A: lr × scheduler** (`results/week5_stageA.csv`, seed 42, wd 5e-4, last5 %)

| 설정 | bn_deep_wide | 1위와 차이 | bn_deep | 1위와 차이 |
|---|---|---|---|---|
| lr 0.1 + cosine | **92.36** | 1위 | 90.66 | −0.02 (차이 없음) |
| lr 0.05 + cosine | 92.27 | −0.08 (차이 없음) | **90.68** | 1위 |
| lr 0.01 + cosine | 91.73 | −0.63 | 89.58 | −1.10 |
| lr 0.01 + none (4주차 재사용) | 88.62 | −3.74 | 87.14 | −3.54 |
| lr 0.05 + none | 84.70 | −7.66 | 85.12 | −5.56 |
| lr 0.1 + none | 81.22 | −11.13 | 78.58 | −12.10 |

- 두 모델 모두 "차이 없음" 그룹이 lr 0.1 + cosine과 lr 0.05 + cosine이었고, 기본값에 더 가까운 **lr 0.05 + cosine**이 승자다.
- 효과의 대부분은 스케줄러다. lr 0.01에서 cosine만 켜도 +3.1%p(bn_deep_wide), +2.4%p(bn_deep).
- lr을 키우고 끝까지 고정(none)하면 val acc가 마지막까지 크게 흔들려 기본 설정보다 낮다 (`figures/week5/1_stageA_val_acc_curves.png`).

**B: weight decay** (`results/week5_stageB.csv`, seed 42, lr 0.05 + cosine, last5 %)

| wd | bn_deep_wide | train−val (%p) | bn_deep | train−val (%p) |
|---|---|---|---|---|
| 1e-4 | 91.78 (−1.09) | 7.49 | 89.46 (−1.46) | 6.78 |
| 5e-4 (A 재사용) | 92.27 (−0.60) | 7.21 | 90.68 (−0.24, 차이 없음) | 6.17 |
| 1e-3 | **92.87** (1위) | 6.31 | 90.92 (1위) | 5.43 |
| 5e-3 | 90.91 (−1.96) | 4.60 | 89.04 (−1.88) | 3.40 |

- bn_deep_wide: 2위와 0.60%p 차이라 **wd 1e-3** 단독 승자.
- bn_deep: wd 1e-3이 1위지만 5e-4와 0.24%p 차이라 "차이 없음" → 기본값에 가까운 **wd 5e-4**가 승자.
- wd를 키울수록 train−val 차이는 줄지만, 5e-3에서는 정확도도 함께 떨어진다.

**D: 60 epoch 수렴 확인** (참고용, 선정에 쓰지 않음, seed 42)

| 모델 | last5 40 → 60 epoch | best 40 → 60 epoch (best epoch) | train−val 40 → 60 |
|---|---|---|---|
| bn_deep_wide | 92.87 → 93.02 (+0.15) | 93.00 (39) → 93.16 (60) | 6.31 → 6.54 |
| bn_deep | 90.68 → 91.24 (+0.56) | 90.84 (39) → 91.30 (57) | 6.17 → 7.08 |

- 두 모델 모두 C에서 "아직 수렴 안 됨"(시드 2개 이상에서 best epoch 36~40)이라 D를 실행했다.
- bn_deep_wide는 차이가 0.45%p 안이고, bn_deep은 +0.56%p로 기준보다 조금 크다. 시드 1개라 단정하지 않는다.
- 60 epoch에서는 cosine 일정(T_max)도 60으로 바뀌므로 "같은 설정을 더 오래 학습한 것"이 아니다. 6주차 epoch 수를 정할 때 참고만 한다.

**val loss** (seed 42): 4주차 기본 설정의 bn_deep_wide는 val loss가 32 epoch에 가장 낮고(0.330) 끝에서 0.377로 올랐지만,
최종 설정은 36 epoch 0.263 → 끝 0.266으로 다시 오르지 않았다. lr이 끝에서 거의 0이 되어 후반 흔들림이 줄어든 것으로 보이며, 시드 1개 관찰이다.

## 한계 (결과를 본 뒤 발견한 것 포함, 기준은 바꾸지 않음)

- **C는 lr·scheduler·wd를 한꺼번에 바꾼 비교다.** 각 요소의 기여는 시드 1개인 A·B 표로만 추정할 수 있다.
- **"아직 수렴 안 됨" 표시는 cosine에서 의미가 약하다.** 끝으로 갈수록 lr이 0에 가까워져 정확도가 마지막에 조금씩 오르므로 best epoch이 끝에 몰리는 것이 자연스럽다. 4주차 고정 lr에서 정한 판정을 그대로 쓴 결과 두 모델 모두 D 대상이 됐다.
- **bn_deep의 wd 선택:** seed 42에서는 wd 1e-3이 0.24%p 높았지만 "차이 없음" 규칙에 따라 5e-4를 골랐다. 시드 1개 차이라 어느 쪽이 더 좋은지는 알 수 없다.
- **train−val 차이:** bn_deep은 최종 설정에서 5.02 → 6.21%p로 늘었다. 정확도 개선이 과적합 감소를 뜻하지는 않는다.
- 미리 적은 한계도 그대로다: A·B는 시드 1개로 골랐고, val로 고르고 val로 보고해 결과가 낙관적이며, `last5`는 cosine에 유리하다 (다만 best 기준으로도 같은 순서다).
- epoch 시간: A의 첫 실험만 epoch당 28.8초로 느렸고 나머지는 20~23초였다. 정확도 비교에는 영향이 없다.

## 그림

모두 `figures/week5/` 폴더에 있다 (`python compare_results.py --week 5`로 생성).

| 그림 | 경로 | 내용 |
|---|---|---|
| A단계 학습 곡선 | `figures/week5/1_stageA_val_acc_curves.png` | 모델별 val acc, 색 = lr, 선 모양 = scheduler, 굵은 선 = 선택된 설정 |
| B단계 wd 비교 | `figures/week5/2_stageB_wd_last5.png` | wd별 last5 막대, 1위에서 0.45%p 이내 구간 표시 |
| 기본 설정 대비 최종 설정 | `figures/week5/3_final_vs_default.png` | 시드 3개 평균 ± 표준편차와 시드별 점, 개선 판정 |

## Grad-CAM 검증 (새 학습 없음, validation만 사용)

기존 Grad-CAM 도구가 후보 모델 가중치로 정상 동작하는지만 확인했다. 본격적인 분석은 6주차에 한다.

```bash
python visualize_gradcam.py --mode check --runs <실험 이름>     # 실험마다 따로 실행 (check는 실험 1개만 받음)
python visualize_gradcam.py --mode compare --per-class 1 \
    --runs baseline_sgd_lr0.01_bs128_ep40_aug_seed42 bn_deep_sgd_lr0.01_bs128_ep40_aug_seed42 bn_deep_wide_sgd_lr0.01_bs128_ep40_aug_seed42
```

**`--mode check` 결과** (검증 7개, 체크포인트마다 실행)

| 체크포인트 (seed 42) | 설정 | 결과 | val acc 재현 (체크포인트 = 재계산) |
|---|---|---|---|
| baseline (4주차) | lr 0.01 + none + wd 5e-4 | **7/7 PASS** | 83.86% |
| bn_deep (4주차) | lr 0.01 + none + wd 5e-4 | **7/7 PASS** | 88.36% |
| bn_deep_wide (4주차) | lr 0.01 + none + wd 5e-4 | **7/7 PASS** | 89.90% |
| bn_deep_wide (5주차 최종 설정) | lr 0.05 + cosine + wd 1e-3 | **7/7 PASS** | 93.00% |
| bn_deep (5주차 최종 설정) | lr 0.05 + cosine + wd 5e-4 | **7/7 PASS** | 90.84% |

- 모두 CAM 전부 0인 이미지 0/5,000장, target class를 바꾸면 CAM이 달라지고(평균 절대 차이 0.17~0.22), 학습 모델과 랜덤 모델의 CAM이 다르다(0.24~0.27).
- 자동 선택된 층은 baseline `features[7]`, bn_deep·bn_deep_wide `features[19]`이고 모두 CAM 원본 8×8이다.

**`--mode compare` 그림과 6주차에 다시 쓸 이미지 목록**
- 그림: `figures/gradcam/compare_val_baseline_sgd_lr0.01_bs128_ep40_aug_seed42_and_2_more_67b5a34f.png`
- 이미지 목록: `figures/gradcam/compare_val_baseline_sgd_lr0.01_bs128_ep40_aug_seed42_and_2_more_67b5a34f_indices.csv`
  (val 위치 487, 3890, 3137, 2138, 2064, 4202, 390, 3628, 1023, 490 — 클래스별 1장, `config.SEED`로 고정)
- 6주차에 같은 이미지로 그리기: `--indices figures/gradcam/compare_val_baseline_sgd_lr0.01_bs128_ep40_aug_seed42_and_2_more_67b5a34f_indices.csv`

**6주차 확인할 것** (그림에서 눈에 띈 점, 해석하지 않고 기록만 함. 클래스당 1장이라 일반화할 수 없음)
- airplane(#487): baseline은 비행기 몸체보다 오른쪽 아래 하늘 쪽이 밝고, bn_deep·bn_deep_wide는 몸체 쪽이 밝다.
- ship(#1023): baseline은 배 아래 물 쪽이 밝고, 두 모델은 배 위쪽이 밝다.
- bird(#3137): 세 모델 모두 frog로 틀렸고, 히트맵이 새보다 주변 줄무늬 배경 쪽에 퍼져 있다.
- cat(#2138)·dog(#4202): 모델마다 예측이 갈린다 (cat: bn_deep만 dog로 틀림, dog: baseline만 cat으로 틀림).
- deer(#2064): 세 모델 모두 사슴 몸보다 아래쪽 가운데가 밝다.
- 위 그림은 4주차 기본 설정 체크포인트다. 5주차 최종 설정 체크포인트로도 같은 이미지(`--indices`)로 그려 비교할 것.

## 5주차 설계 결정

| 항목 | 결정 | 이유 |
|---|---|---|
| 스케줄러 옵션 | `train.py --scheduler {none, cosine, step}`, 기본 none | 기본값이면 4주차까지와 같은 동작. bn_deep seed 42의 epoch 1 결과가 4주차와 소수점 끝까지 같음을 확인 |
| 스케줄러 갱신 시점 | epoch이 끝날 때마다 `step()` 한 번 | `engine.py`(학습 루프)를 고치지 않고 `train.py`에서만 처리. cosine은 T_max = 전체 epoch |
| step 일정 | 전체 epoch의 50%, 75% 지점에서 ×0.1 (`config.py`) | 40 epoch → 21~30 epoch은 0.1배, 31~40 epoch은 0.01배. epoch 수를 바꿔도 비율이 유지됨. 이번 탐색에서는 쓰지 않음 (초안 범위가 none·cosine) |
| 폴더 이름 | 기본값과 다른 scheduler·wd·momentum만 `_seed` 앞에 붙임 | 기존 39개 폴더 이름이 그대로 유지됨을 확인. 예전 이름에는 wd가 없어 wd만 바꾼 실험이 4주차 폴더를 "완료"로 착각하는 문제가 있었음. 5주차 설정 144개의 이름 충돌 없음 확인 |
| history.csv `lr` 열 | 그 epoch에 쓴 lr을 맨 끝 열에 기록 | 기존 열 순서를 읽는 코드가 깨지지 않게 함. 4주차 이전 기록에는 이 열이 없음 |
| 튜닝 범위 | lr·scheduler·wd만. batch size·momentum은 고정 | batch size는 lr의 의미를 바꾸고, momentum까지 바꾸면 실험 수가 크게 늘어남 |
| 탐색 방식 | 한 번에 하나씩 (A → B → C) | 모든 조합(모델당 6 × 4 = 24회)보다 적은 실험으로, 각 변경의 효과를 따로 설명할 수 있음 |
| weight decay 적용 범위 | 모든 파라미터 (BN 포함), 4주차와 같음 | 적용 범위까지 바꾸면 wd 값 외에 다른 변수가 생김 |
| 차이 기준 | 0.45%p로 고정 | 시드 1개 실험끼리 비교하므로 4주차의 "시드 표준편차" 기준을 쓸 수 없음. val 표본 오차로 미리 정함 |
| 승자 규칙 위치 | `week5_selection.py` 한 곳 | `run_experiments.py`(다음 단계 계획)와 `compare_results.py`(결과 표)가 서로 다른 승자를 고르는 일을 막음. torch를 쓰지 않아 취합이 가벼움 |
| 실험 찾기 | 폴더 이름이 아니라 `args.json` 값으로 찾음 | 이름 규칙이 바뀌거나 `--run-name`을 써도 같은 설정을 정확히 찾음. scheduler가 없는 예전 기록은 none으로 봄 |
| 재사용 실험 | 결과가 없어도 새로 학습하지 않음 | 다른 PC에서 4주차 결과 없이 돌리면 lr 0.01 + none을 그 PC에서 새로 학습해 A의 기준점이 바뀌는 것을 막음 |
| 직접 지정 기록 | `results/week5_selection.json` | 자동 선택이 안 돼 직접 정한 승자를 취합 때도 같게 쓰기 위함. 자동 선택 기록은 나중에 결과가 바뀌었는지 확인하는 데 씀 |
| 3·4주차 취합 | 기본 하이퍼파라미터 실험만 읽음 | 5주차 실험도 모델·epoch(40)가 같아 그대로 두면 4주차 표에 "같은 (모델, 시드) 중복"으로 섞임 |
| `--verify` | 임시 폴더에 다시 만들어 기존 표·그림과 비교 | 코드를 고친 뒤에도 3·4주차 결과가 같은지 파일을 덮어쓰지 않고 확인 (3단계 확인: 4주차 표 6개·그림 4개 모두 같음) |

---

# 4주차: 2차 모델 구조 비교

BN을 고정한 상태에서 Deep과 Wide의 조합 효과(2×2)를 보고, Dropout과 GAP을 큰 모델에서 다시 확인한다.
비교 대상은 baseline, bn, bn_deep, bn_wide, bn_deep_wide, bn_deep_wide_dropout, bn_deep_wide_gap 7개이고,
40 epoch × 시드 3개(42, 1, 2), 총 21개 실험이다. 나머지 하이퍼파라미터는 `config.py` 기본값을 쓴다.

## 4주차 후보 선정 기준 (실험 전 작성)

5주차 하이퍼파라미터 튜닝으로 넘길 후보 2~3개를 아래 기준으로 고른다.

> **결과를 본 뒤에 기준을 바꾸지 않는다.** 기준에서 문제를 발견해도 이번 선정은 이 기준대로 하고, 문제점은 결과와 함께 따로 기록한다.

**선정 전 조건**
- 21개 실험이 모두 40 epoch까지 끝나고 학습 설정이 같아야 한다.
- validation 결과만 사용한다. 테스트셋은 선정에 사용하지 않는다.

**지표** (모두 시드별로 구한 뒤 시드 3개로 평균·표본 표준편차를 계산)
- 주 지표: `last5_val_acc`, 36~40 epoch val acc의 평균. 아래의 차이 판정에도 이 지표의 평균과 표준편차를 쓴다.
- 보조 지표: `best_val_acc`. 최댓값이라 낙관적이므로 함께 보고하되 선정 판정에는 쓰지 않는다.

**선정 절차**
1. `baseline`은 기준점이므로 후보에서 제외하고, 나머지 6개 모델을 주 지표 평균이 높은 순으로 정렬한다.
2. 1위 모델과의 평균 차이가 두 모델의 시드 표준편차 중 큰 값 이하이면 "1위와 차이 없음" 그룹에 넣는다. 1위 모델도 이 그룹에 포함한다.
3. 그룹을 특징 추출부 구조별로 나누고, 구조마다 모델을 하나만 남긴다. 파라미터 수가 적은 쪽을 남기고, 파라미터 수가 같으면 epoch 시간이 짧은 쪽을 남긴다.
   - 구조 구분: `bn` / `bn_deep` / `bn_wide` / `bn_deep_wide`, `bn_deep_wide_dropout`, `bn_deep_wide_gap`(마지막 3개는 특징 추출부가 같으므로 한 구조)
4. 남은 모델이 4개이면 파라미터 수가 적은 순으로 3개를 후보로 정한다. 2~3개이면 모두 후보로 정한다.
5. 남은 모델이 1개이면, 그룹 밖의 다른 구조 모델 중 주 지표 평균이 가장 높은 모델 1개를 더해 후보를 2개로 만든다.

**"아직 수렴 안 됨" 표시**
- 시드 3개 중 2개 이상에서 best epoch(최고 val acc가 처음 나온 epoch)이 36~40이면 해당 모델에 "아직 수렴 안 됨"으로 표시한다.
- 이 표시는 후보 제외 사유가 아니다. 5주차 epoch 수를 정할 때 참고하고, 표시된 모델의 주 지표는 낮게 평가됐을 수 있음을 결과에 함께 적는다.

## 실행과 취합

```bash
python run_experiments.py --week 4 --dry-run   # 실행할 명령과 예상 시간만 출력
python run_experiments.py --week 4             # 21개 실험 실행 (RTX 5060에서 총 4시간 38분)
python compare_results.py --week 4             # 결과 취합 → results/week4_*.csv + figures/week4/*.png
python compare_results.py --week 4 --verify    # 다시 계산해 기존 출력과 같은지만 확인 (파일을 쓰지 않음)
```
- `compare_results.py --week 4`는 epochs=40이고 4주차 모델 7개인 실험만 읽는다. 같은 `results/`에 있는 3주차 ep20 폴더는 읽지 않고 목록만 출력한다.
- 폴더 이름과 `args.json`의 epoch가 다르거나, 끝나지 않은 실험, 같은 (모델, 시드) 중복이 있으면 에러로 멈춘다.
- 클래스별 정확도를 위해 21개 실험에 `evaluate.py`를 val로 실행한다 (`--test` 없음).

## 결과 요약

7개 모델 × 시드 3개, 40 epoch, validation 기준 (테스트셋 미사용). 전체 표: `results/week4_summary.csv`

| 순위 | 모델 | 파라미터 | 마지막 5 epoch 평균 (%, 주 지표) | best val acc (%) | 시드별 best epoch | train−val 차이 (%p) | 수렴 |
|---|---|---|---|---|---|---|---|
| 1 | bn_deep_wide | 2,197,706 | **88.91 ± 0.27** | 89.82 ± 0.21 | 32, 32, 39 | 7.17 ± 0.74 | |
| 2 | bn_deep_wide_dropout | 2,197,706 | 88.13 ± 0.27 | 89.65 ± 0.19 | 39, 31, 38 | 5.33 ± 0.17 | 아직 수렴 안 됨 |
| 3 | bn_deep | 814,570 | 87.55 ± 0.36 | 88.25 ± 0.22 | 40, 38, 38 | 5.02 ± 0.49 | 아직 수렴 안 됨 |
| 4 | bn_deep_wide_gap | 1,148,874 | 87.11 ± 0.19 | 88.59 ± 0.41 | 35, 28, 39 | 8.00 ± 1.13 | |
| 5 | bn_wide | 1,422,666 | 86.00 ± 0.18 | 86.85 ± 0.32 | 39, 40, 40 | 4.02 ± 1.00 | 아직 수렴 안 됨 |
| 6 | bn | 620,586 | 84.38 ± 0.09 | 85.62 ± 0.16 | 37, 40, 38 | 3.46 ± 1.52 | 아직 수렴 안 됨 |
| 기준 | baseline | 620,362 | 83.27 ± 0.36 | 83.86 ± 0.26 | 39, 40, 40 | 3.90 ± 1.18 | 아직 수렴 안 됨 |

- train−val 차이는 마지막 epoch의 (train acc − val acc)다. train acc는 증강이 켜지고 학습 모드(Dropout/BN)인 상태로 잰 값이라 근사치다.

**BN 고정 2×2** (`results/week4_factorial.csv`, last5 기준, 단위 %p)

| | Wide 없음 | Wide 있음 |
|---|---|---|
| **Deep 없음** | bn 84.38 ± 0.09 | bn_wide 86.00 ± 0.18 |
| **Deep 있음** | bn_deep 87.55 ± 0.36 | bn_deep_wide 88.91 ± 0.27 |

- Deep 효과 **+3.17 ± 0.37**, Wide 효과 **+1.61 ± 0.21**: 둘 다 시드 편차보다 크고, Deep이 Wide의 약 2배다.
- 상호작용 −0.26 ± 0.49 (best 기준 +0.34 ± 0.47): 0과 구분되지 않는다. 두 효과가 거의 그대로 더해진다 (단순 합 예상 89.16%, 실제 88.91%).
- Deep은 파라미터를 1.31배, Wide는 2.29배로 늘리므로 Deep이 파라미터 대비 효율도 높다.
- 3주차(BN 없음, 20 epoch)에서는 Wide가 Deep보다 컸다. BN과 epoch 수가 함께 바뀌어 어느 쪽 때문인지는 알 수 없다.

**큰 모델에서 Dropout·GAP** (`results/week4_variants.csv`, bn_deep_wide 대비)
- 모델이 커질수록 train−val 차이가 커진다 (bn 3.46 → bn_deep_wide 7.17%p).
  bn_deep_wide는 val loss가 20~32 epoch에 가장 낮고 이후 다시 오른다 → 과적합 경향 (`figures/week4/2_loss_curves.png`).
- **Dropout**: train−val 차이는 −1.84%p로 줄었지만, last5 −0.78%p(시드 편차보다 큼), best −0.17%p(편차 이하)로 정확도 이득은 없다.
  train acc를 Dropout이 켜진 상태로 재므로 차이 감소의 일부는 측정 방식 때문이다.
- **GAP**: last5 −1.79%p, 파라미터 약 절반. 3주차(FC 대비 −9.33%p)보다 격차가 크게 줄었지만, BN·구조·epoch가 함께 달라 원인은 단정할 수 없다.
  파라미터가 더 적은 bn_deep(87.55%)보다 낮아 파라미터 대비 효율은 좋지 않다.

**클래스별 val 정확도** (`results/week4_per_class_val_acc.csv`, `results/week4_cat_dog.csv`)
- 구조를 바꾼 6개 모델 모두 baseline 대비 cat의 향상 폭이 가장 크다 (bn_deep_wide: 65.93 → 84.87%, +18.9%p).
- 그래도 cat·dog가 가장 낮은 두 클래스이고, bn_deep_wide_dropout은 cat/dog 시드 표준편차가 6.8~8.3%p로 크다.

**20 vs 40 epoch** (`results/week4_epoch_compare.csv`)
- baseline last5 +3.47%p, bn +2.97%p로 epoch를 늘린 이득이 시드 편차보다 크다.
- 20 epoch 결과는 다른 PC·PyTorch 버전에서 학습되어 epoch 시간 차이는 비교하지 않는다.

## 5주차 후보 (선정 기준 그대로 적용)

1. baseline 제외 last5 순위: bn_deep_wide 88.91 > bn_deep_wide_dropout 88.13 > bn_deep 87.55 > bn_deep_wide_gap 87.11 > bn_wide 86.00 > bn 84.38
2. 2위와 1위의 차이 0.78%p가 두 표준편차 중 큰 값(0.27)보다 커서, "1위와 차이 없음" 그룹은 bn_deep_wide 하나다.
3. 남은 모델이 1개이므로 그룹 밖 다른 구조 중 1위를 더한다. dropout·gap은 bn_deep_wide와 특징 추출부가 같은 구조라 제외되고, bn_deep이 들어간다.

| 후보 | 파라미터 | 참고 |
|---|---|---|
| **bn_deep_wide** | 2,197,706 | 주 지표 1위. 과적합 경향이 있어 규제·학습률 조정 여지가 있다 |
| **bn_deep** | 814,570 | "아직 수렴 안 됨": 주 지표가 낮게 평가됐을 수 있다 |

**한계 (기준은 바꾸지 않고 기록만 함)**
- 시드 3개라 표준편차 추정이 불안정하고, "시드 편차보다 큼"은 유의성 검정이 아니다. val 5,000장 자체의 오차(약 ±0.45%p)는 시드 편차에 들어 있지 않다.
- 모든 모델이 같은 val로 평가되고 선정도 val로 했으므로 성능이 약간 낙관적이다. 한 가지 하이퍼파라미터 설정에서만 비교했다.
- 7개 중 5개 모델이 "아직 수렴 안 됨"이라 epoch를 늘리면 순위가 바뀔 수 있다.
- last5는 늦게 수렴하는 모델(Dropout)에 불리하고, best epoch는 epoch마다 약 1%p 흔들려 수렴 판정이 불안정하다.
- Dropout은 best 기준으로 1위와 차이가 없지만, 같은 구조 규칙 때문에 후보 추가 단계에서 검토되지 않았다.

## 그림

모두 `figures/week4/` 폴더에 있다 (`python compare_results.py --week 4`로 생성).

| 그림 | 경로 | 내용 |
|---|---|---|
| 학습 곡선 (val acc) | `figures/week4/1_val_acc_curves.png` | epoch별 val acc, 시드 평균선 + 최소~최대 띠, 16~40 epoch 확대 |
| 학습 곡선 (loss) | `figures/week4/2_loss_curves.png` | epoch별 train/val loss (큰 모델의 후반 val loss 상승) |
| best val acc 막대 | `figures/week4/3_best_val_acc_bar.png` | 모델별 평균 ± 표준편차, 시드별 점, baseline 기준선 |
| 파라미터 수 vs 정확도 | `figures/week4/4_params_vs_acc.png` | 같은 파라미터 수의 모델은 옆으로 살짝 밀어 표시 |

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
python compare_results.py --overwrite-week3   # 결과 취합 → results/week3_*.csv + figures/week3/*.png
python compare_results.py --verify-week3      # 다시 계산해 기존 week3_summary.csv와 같은지만 확인 (파일을 쓰지 않음)
```
- 3주차 출력은 확정본이라, 이미 있으면 `--overwrite-week3` 없이는 덮어쓰지 않고 멈춘다.
- `--verify-week3` 결과(4주차 시점): 정확도·epoch·파라미터 열과 그림 4종은 모두 같고, 시간 열만 최대 3.8초 다르다.
  `week3_summary.csv`를 만든 뒤 같은 시드로 다시 학습한 ep20 기록이 올라왔기 때문이다 (정확도는 같고 시간만 다름).
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

모두 `figures/week3/` 폴더에 있다 (`python compare_results.py --overwrite-week3`로 생성).

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
