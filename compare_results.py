"""
구조 비교 실험(results/) 결과를 주차별로 취합해 표와 그림으로 정리한다. (3주차: ep20 × 6모델, 4주차: ep40 × 7모델)

실행 예:
    python compare_results.py                  # 3주차 (기본): 표 + 그림 4종 + 상위 3개 모델 클래스별 정확도
    python compare_results.py --week 4         # 4주차: 같은 표/그림 + 추가 분석 4종 (아래 참고)
    python compare_results.py --week 4 --no-eval      # evaluate.py를 호출하지 않고 이미 있는 CSV만 사용
    python compare_results.py --week 4 --force-eval   # 클래스별 정확도 CSV가 있어도 evaluate.py를 다시 실행
    python compare_results.py --verify-week3   # 3주차를 다시 계산해 기존 week3_summary.csv와 같은지 확인 (파일을 쓰지 않음)
    python compare_results.py --overwrite-week3       # 3주차 출력을 정말로 다시 만들 때만 (기본은 덮어쓰기 거부)
    python compare_results.py --week 4 --verify       # 4주차를 임시 폴더에 다시 만들어 기존 출력과 같은지 확인 (파일을 쓰지 않음)
    python compare_results.py --week 5                # 5주차: 단계별 표 + 모델별 최종 설정 + 그림 3종

5주차 (규칙은 week5_selection.py, 기준은 README "5주차 선정 기준")
    읽는 것: bn_deep_wide, bn_deep 중 batch 128·SGD·momentum 0.9·증강인 실험 (4주차 기본 설정 결과 포함)
    쓰는 것: results/week5_stageA.csv   A: lr × scheduler 6조합 (seed 42), 1위와의 차이, 차이 없음 그룹, 승자
             results/week5_stageB.csv   B: A 승자 위에서 wd 4개 (seed 42)
             results/week5_final.csv    C: 튜닝한 설정 vs 4주차 기본 설정 (시드 3개 평균 ± 표준편차), 최종 설정, D 필요 여부
             figures/week5/             1_stageA_val_acc_curves, 2_stageB_wd_last5, 3_final_vs_default
    단계가 덜 끝났으면 끝난 부분까지만 표에 채우고, 다음 단계는 "앞 단계 미완료"로 남긴다 (에러로 멈추지 않음).
    3·4주차 취합은 기본 하이퍼파라미터(lr 0.01, wd 5e-4, scheduler none, momentum 0.9) 실험만 읽으므로
    5주차 결과가 results/에 생겨도 3·4주차 표는 바뀌지 않는다.

3주차 출력 보호:
    3주차 결과(results/week3_*.csv, figures/week3/)는 보고서에 쓴 확정본이다. --week 기본값이 3이라
    옵션 없이 실행하면 실수로 덮어쓸 수 있으므로, week3_summary.csv가 이미 있으면 --overwrite-week3 없이는 멈춘다.
    (지금 results/의 ep20 history.csv는 week3_summary.csv를 만든 뒤 다시 올라온 기록이라 정확도는 같지만
     epoch_time_sec가 달라, 덮어쓰면 시간 열이 바뀐다. 차이는 --verify-week3로 확인할 수 있다.)

주차별 입력과 출력 (서로 섞이지 않는다):
    3주차  읽는 것: epochs=20 이고 모델이 baseline, bn, dropout, deep, wide, gap 인 실험
           쓰는 것: results/week3_*.csv, figures/week3/
    4주차  읽는 것: epochs=40 이고 모델이 baseline, bn, bn_deep, bn_wide, bn_deep_wide,
                    bn_deep_wide_dropout, bn_deep_wide_gap 인 실험
           쓰는 것: results/week4_*.csv, figures/week4/
    results/ 에는 3주차 ep20 폴더와 4주차 ep40 폴더가 함께 있으므로(baseline, bn은 양쪽에 있음)
    위 조건에 맞는 폴더만 읽고, 나머지는 읽지 않은 실험으로 목록만 출력한다.
    아래의 경우에는 에러로 멈춘다 (다른 주차 결과가 섞여 비교가 틀어지는 것을 막기 위함):
      - 폴더 이름의 ep 값과 args.json의 epochs가 다름 (폴더를 복사해 이름만 바꾼 경우 등)
      - history.csv의 epoch가 1~N이 아님 (끝나지 않은 실험, 다른 epoch 설정의 기록)
      - 같은 (모델, 시드)가 두 번 있음
      - 조건에 맞는 실험이 하나도 없음

3주차 결과 (이 스크립트가 만드는 것):
    results/week3_summary.csv            모델별 요약표 (시드 평균 ± 표준편차, best val acc 내림차순)
    results/week3_per_class_val_acc.csv  상위 모델 + Baseline의 val 클래스별 정확도 (시드 평균 ± 표준편차)
    figures/week3/                       그림 4종
4주차 결과:
    results/week4_summary.csv            3주차와 같은 요약표 (기준 모델 = baseline)
    results/week4_per_class_val_acc.csv  7개 모델 전체의 val 클래스별 정확도
    results/week4_factorial.csv          [분석 1] BN 고정 2x2 효과표
    results/week4_variants.csv           [분석 2] bn_deep_wide 대비 dropout/gap 변형의 차이 + overfit_gap
    results/week4_cat_dog.csv            [분석 3] cat, dog 클래스 정확도 모델별 비교
    results/week4_epoch_compare.csv      [분석 4] baseline, bn의 20 epoch(3주차) vs 40 epoch(4주차)
    figures/week4/                       그림 4종 (1_val_acc_curves, 2_loss_curves, 3_best_val_acc_bar, 4_params_vs_acc)

※ 테스트셋은 사용하지 않는다.
   - history.csv(val 기준)와 val_per_class_acc.csv만 읽는다. test_* 파일은 읽지 않는다.
   - evaluate.py는 항상 --test 없이(= val 평가로) 호출한다.

표의 정의 (시드 표준편차는 표본 표준편차, ddof=1, 시드 3개 기준이라 값 자체가 불안정함에 주의)
    best_val_acc_pct      : 전체 epoch 중 최고 val acc (%)  <- 최댓값이라 약간 낙관적인 지표
    last5_val_acc_pct     : 마지막 5 epoch의 val acc 평균 (%)  <- best보다 안정적인 지표 (4주차는 36~40 epoch)
    best_epoch            : best val acc가 처음 나온 epoch
    overfit_gap_pp        : 마지막 epoch의 (train acc - val acc), %p
                            주의: train acc는 증강이 켜진 상태 + 학습 모드(Dropout/BN)에서 잰 값이라
                            "과적합 정도"의 근사치일 뿐이며, 음수가 나올 수 있다(특히 Dropout이 있는 모델).
    epoch_time_sec        : epoch당 평균 소요 시간(초, train+val 포함)
    diff_vs_baseline_pp   : (모델 평균 best val acc) - (Baseline 평균 best val acc), %p
                            표준편차는 두 모델 시드 표준편차를 독립으로 가정해 sqrt(s1^2 + s2^2)로 합성
    diff_vs_seed_std      : |차이|가 두 모델 시드 표준편차 중 큰 값보다 크면 "larger", 아니면 "smaller"
                            "smaller"이면 의미 있는 차이라고 보기 어렵다 (통계적 유의성 검정은 아님)

[분석 1] BN 고정 2x2 효과표 (week4_factorial.csv)
    BN을 켠 채로 Deep(블록당 Conv 2개)과 Wide(채널 2배)를 각각/함께 적용한 4개 모델로 계산한다.
        Deep 효과 = bn_deep - bn
        Wide 효과 = bn_wide - bn
        상호작용  = (bn_deep_wide - bn_wide) - (bn_deep - bn)
                  = bn_deep_wide - bn_deep - bn_wide + bn
    지표는 best val acc와 last5 val acc 두 가지로 각각 계산한다 (단위 %p, 시드 평균 기준).
    해석:
        - 상호작용이 0에 가까우면: Deep과 Wide의 효과가 더해진다 (함께 써도 각자의 이득이 그대로 유지됨).
          bn_deep_wide의 기대값은 bn + Deep 효과 + Wide 효과이고, 표에 additive_expected_pct로 함께 적었다.
        - 음수이면: 효과가 겹친다 (함께 쓰면 각자의 이득의 합보다 작다. 같은 문제를 둘 다 해결하고 있었다는 뜻).
        - 양수이면: 함께 쓸 때 합보다 더 좋아진다 (시너지).
    "0에 가깝다"의 기준: 표준편차(합성) 안이면 0과 구분되지 않는다고 본다(*_vs_std 열이 "smaller").
        효과의 표준편차는 관련된 모델들의 시드 표준편차를 독립으로 가정해 sqrt(sum s^2)로 합성한다
        (효과 = 모델 2개 → 2개, 상호작용 = 모델 4개 → 4개). 그래서 상호작용은 효과보다 잡음이 크다.
        이 판정은 요약표의 diff_vs_seed_std(두 표준편차 중 큰 값)보다 엄격한 기준이다.

[분석 2] bn_deep_wide 대비 dropout/gap 변형 (week4_variants.csv)
    bn_deep_wide(기준)와 bn_deep_wide_dropout, bn_deep_wide_gap의 best/last5 val acc 차이와 overfit_gap 차이.
    표준편차 판정은 요약표와 같은 규칙(두 모델 시드 표준편차 중 큰 값과 비교).
    주의: bn_deep_wide_dropout은 train acc를 Dropout이 켜진 상태에서 재므로 overfit_gap이 작게(음수로) 나올 수 있다.
          overfit_gap이 줄어든 것을 곧바로 과적합이 줄었다고 읽지 말고 val acc와 함께 본다.
    bn_deep_wide_gap은 FC 층이 바뀌어 파라미터 수가 다르다(num_params 열). 구조 효과와 크기 효과가 섞인다.

[분석 3] cat, dog 클래스별 비교 (week4_cat_dog.csv)
    7개 모델 전체의 클래스별 val 정확도(시드 평균 ± 표준편차)에서 cat, dog만 뽑아 Baseline 대비 차이를 적는다.
    클래스당 val 500장 x 시드 3개라 값이 많이 흔들린다. 차이 판정은 같은 규칙(max 표준편차).

[분석 4] 20 epoch vs 40 epoch (week4_epoch_compare.csv)
    baseline과 bn의 3주차(ep20) 결과와 4주차(ep40) 결과를 나란히 놓고, 40-20 차이를 적는다.
    val_acc_ep20_pct는 20번째 epoch의 val acc다 (40 epoch 실험이면 학습 중간 값).
    두 결과는 train/val 분할과 하이퍼파라미터(epochs 제외)가 같아야 한다. 다르면 경고한다.
    last5는 20 epoch는 16~20, 40 epoch는 36~40 epoch 평균이다.
"""

import argparse
import contextlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 창 없이 파일로만 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import week5_selection as w5  # torch를 쓰지 않는 규칙 모듈 (run_experiments.py와 같은 승자 규칙)

# ---------------------------------------------------------------------
# 설정 (config.py는 torch를 import하고 장치 검사까지 하므로, 이 스크립트는 일부러 import하지 않는다)
# ---------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

# 주차별 입력 조건. 이 조건에 맞는 실험만 읽는다 (run_experiments.py의 WEEK_PRESETS와 같은 모델·epoch).
WEEK_CONFIGS = {
    3: {"models": ["baseline", "bn", "dropout", "deep", "wide", "gap"], "epochs": 20, "default_top_k": 3},
    4: {"models": ["baseline", "bn", "bn_deep", "bn_wide", "bn_deep_wide",
                   "bn_deep_wide_dropout", "bn_deep_wide_gap"], "epochs": 40, "default_top_k": None},  # None = 전체
    5: {"models": w5.MODELS, "epochs": w5.EPOCHS, "default_top_k": None},  # 5주차는 run_week5가 따로 처리
}

# 3·4주차 실험의 하이퍼파라미터 (config.py 기본값). 5주차에 이 값을 바꾼 실험이 results/에 함께 생기므로,
# 3·4주차 취합은 이 값과 같은 실험만 읽는다. scheduler가 없는 args.json(4주차 이전)은 none으로 본다.
DEFAULT_HPARAMS = {"lr": w5.DEFAULT_LR, "weight_decay": w5.DEFAULT_WEIGHT_DECAY,
                   "momentum": w5.FIXED_SETTINGS["momentum"], "scheduler": w5.DEFAULT_SCHEDULER}

# --verify로 다시 만들어 비교할 파일. 3주차 클래스별 표는 체크포인트가 없어 다시 계산할 수 없으므로 뺀다.
VERIFY_TABLES = {3: ["summary"], 4: ["summary", "factorial", "variants", "epoch_compare", "per_class", "cat_dog"]}


def week_paths(week, results_dir=RESULTS_DIR, figures_dir=FIGURES_DIR):
    """
    주차별 출력 경로. 모든 파일 이름이 week{N}_ 로 시작하므로 다른 주차 출력을 덮어쓰지 않는다.
    results_dir, figures_dir를 바꾸면 다른 곳(--verify의 임시 폴더)에 같은 이름으로 저장한다.
    """
    results_dir, figures_dir = Path(results_dir), Path(figures_dir)
    return {
        "summary": results_dir / f"week{week}_summary.csv",
        "per_class": results_dir / f"week{week}_per_class_val_acc.csv",
        "factorial": results_dir / f"week{week}_factorial.csv",
        "variants": results_dir / f"week{week}_variants.csv",
        "cat_dog": results_dir / f"week{week}_cat_dog.csv",
        "epoch_compare": results_dir / f"week{week}_epoch_compare.csv",
        "stage_a": results_dir / f"week{week}_stageA.csv",
        "stage_b": results_dir / f"week{week}_stageB.csv",
        "final": results_dir / f"week{week}_final.csv",
        "fig_dir": figures_dir / f"week{week}",
    }


BASELINE = "baseline"
FIG_DPI = 150                      # config.FIG_DPI와 동일
LAST_N_EPOCHS = 5                  # "마지막 N epoch 평균 val acc"의 N
FOCUS_CLASSES = ["cat", "dog"]     # 4주차 분석 3에서 따로 뽑아 비교하는 클래스
FACTORIAL_MODELS = {"base": "bn", "deep": "bn_deep", "wide": "bn_wide", "both": "bn_deep_wide"}
VARIANT_REF = "bn_deep_wide"
VARIANTS = ["bn_deep_wide_dropout", "bn_deep_wide_gap"]
EPOCH_COMPARE_MODELS = ["baseline", "bn"]  # 20 epoch(3주차)와 40 epoch(4주차)를 비교할 모델
EPOCH_TIME_OUTLIER_RATIO = 1.5     # 같은 모델의 시드 중앙값보다 이 배수 이상 느린 실험은 경고
PER_CLASS_MISMATCH_PP = 0.2        # 클래스별 평균 vs history의 best val acc 차이 허용치(%p)
VAL_PER_CLASS_TOTAL = 500          # val은 클래스당 500장 (test는 1000장이라 구분 가능)

# 실험 간에 같아야 하는 설정들 (다르면 비교가 공정하지 않으므로 경고)
SAME_SETTING_KEYS = ["epochs", "batch_size", "optimizer", "lr", "momentum", "weight_decay", "augment"]

# 모델별 고정 색 (Okabe-Ito 색약 친화 팔레트). 모든 그림에서 같은 모델은 같은 색.
MODEL_COLORS = {
    "baseline": "#000000",
    "bn": "#0072B2",
    "dropout": "#E69F00",
    "deep": "#009E73",
    "wide": "#CC79A7",
    "gap": "#D55E00",
    # 4주차: 3주차와 같은 요소는 같은 계열 색을 쓴다
    "bn_deep": "#009E73",
    "bn_wide": "#CC79A7",
    "bn_deep_wide": "#D55E00",
    "bn_deep_wide_dropout": "#E69F00",
    "bn_deep_wide_gap": "#56B4E9",
}
FALLBACK_COLORS = ["#F0E442", "#999999", "#882255", "#44AA99"]


def color_of(model, _fallback={}):
    """모델 이름 → 색. 목록에 없는 새 모델은 예비 색을 순서대로 배정한다."""
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]
    if model not in _fallback:
        _fallback[model] = FALLBACK_COLORS[len(_fallback) % len(FALLBACK_COLORS)]
    return _fallback[model]


def parse_args():
    """커맨드라인 인자를 읽는다."""
    parser = argparse.ArgumentParser(description="비교 결과 취합 (3주차 / 4주차 / 5주차)")
    parser.add_argument("--week", type=int, default=3, choices=sorted(WEEK_CONFIGS),
                        help="취합할 주차 (기본 3). 3 = ep20 × 6모델, 4 = ep40 × 7모델, 5 = 하이퍼파라미터 튜닝")
    parser.add_argument("--verify", action="store_true",
                        help="3·4주차: 임시 폴더에 다시 만들어 기존 출력과 같은지 확인 (아무 파일도 쓰지 않음)")
    parser.add_argument("--top-k", type=int, default=None,
                        help="클래스별 정확도를 비교할 상위 모델 수 (Baseline 제외). "
                             "기본: 3주차 3개, 4주차 전체")
    parser.add_argument("--verify-week3", action="store_true",
                        help="3주차를 다시 계산해 기존 week3_summary.csv·그림과 같은지 확인 (아무 파일도 쓰지 않음)")
    parser.add_argument("--overwrite-week3", action="store_true",
                        help="이미 있는 3주차 출력(week3_*.csv, figures/week3/)을 덮어쓰도록 허용")
    parser.add_argument("--no-eval", action="store_true",
                        help="evaluate.py를 호출하지 않고 이미 있는 val_per_class_acc.csv만 사용")
    parser.add_argument("--force-eval", action="store_true",
                        help="val_per_class_acc.csv가 있어도 evaluate.py를 다시 실행")
    return parser.parse_args()


# =====================================================================
# 1. 실험 결과 읽기 + 일관성 점검
# =====================================================================
class ResultsError(Exception):
    """집계에 쓸 수 없는 결과가 섞여 있을 때 (다른 주차 결과 혼입, 끝나지 않은 실험 등)."""


def is_default_hparams(args):
    """3·4주차와 같은 하이퍼파라미터(DEFAULT_HPARAMS)로 학습한 실험인지."""
    for key, expected in DEFAULT_HPARAMS.items():
        actual = args.get(key, w5.DEFAULT_SCHEDULER if key == "scheduler" else None)
        if isinstance(expected, float):
            if actual is None or not w5.same(actual, expected):
                return False
        elif actual != expected:
            return False
    return True


def load_runs(models, epochs):
    """
    results/ 아래에서 (모델 in models) 이고 (args.json의 epochs == epochs) 이고
    (하이퍼파라미터가 기본값)인 실험만 읽는다. 기본값 조건은 5주차 튜닝 실험이 섞이지 않게 하기 위함이다.
    반환: (읽은 실험 목록, 읽지 않은 실험 목록)

    다른 주차 결과가 섞이면 ResultsError로 멈춘다:
      - 폴더 이름의 ep 값 ≠ args.json의 epochs (복사해서 이름만 바꾼 경우 등)
      - history.csv의 epoch가 1~epochs가 아님 (끝나지 않은 실험, 다른 epoch 설정의 기록)
      - 같은 (모델, 시드)가 중복
      - 조건에 맞는 실험이 하나도 없음
    """
    selected, ignored = [], []
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        args_path, history_path = run_dir / "args.json", run_dir / "history.csv"
        if not (args_path.exists() and history_path.exists()):
            print(f"[경고] args.json 또는 history.csv가 없어 건너뜁니다: {run_dir.name}")
            continue
        args = json.loads(args_path.read_text(encoding="utf-8"))
        history = pd.read_csv(history_path)
        run = {"name": run_dir.name, "model": args["model"], "seed": args["seed"],
               "args": args, "history": history, "tuned": not is_default_hparams(args)}
        wanted = args["model"] in models and args.get("epochs") == epochs and not run["tuned"]
        (selected if wanted else ignored).append(run)

    tag = f"_ep{epochs}_"
    problems, seen = [], {}
    for run in selected:
        if tag not in run["name"]:
            problems.append(f"{run['name']}: args.json의 epochs는 {epochs}인데 폴더 이름에 '{tag}'가 없습니다 "
                            "(다른 주차 결과를 복사해 이름을 바꿨을 가능성)")
        if list(run["history"]["epoch"]) != list(range(1, epochs + 1)):
            problems.append(f"{run['name']}: history.csv의 epoch가 1~{epochs}이 아닙니다 "
                            f"({len(run['history'])}행) — 끝나지 않은 실험이거나 다른 epoch 설정의 기록입니다")
        key = (run["model"], run["seed"])
        if key in seen:
            problems.append(f"{run['name']}: {seen[key]}와 (모델, 시드)가 같습니다 (중복)")
        seen[key] = run["name"]
    for run in ignored:  # 반대 방향: 이름은 목표 ep인데 args.json의 epochs가 다른 경우
        # (epochs가 같은데 튜닝 설정이라 빠진 5주차 실험은 섞인 것이 아니므로 제외)
        if run["model"] in models and tag in run["name"] and run["args"].get("epochs") != epochs:
            problems.append(f"{run['name']}: 폴더 이름은 ep{epochs}인데 args.json의 epochs는 "
                            f"{run['args'].get('epochs')}입니다 — 다른 주차 결과가 섞였을 수 있습니다")
    if problems:
        raise ResultsError("집계할 수 없는 결과가 섞여 있습니다:\n  - " + "\n  - ".join(problems))
    if not selected:
        found = ", ".join(f"{m} ep{e} x{n}" for (m, e), n in sorted(Counter(
            (r["model"], r["args"].get("epochs")) for r in ignored).items())) or "없음"
        raise ResultsError(f"results/에 조건에 맞는 실험이 없습니다 (모델 {len(models)}개: {', '.join(models)} / epochs={epochs}).\n"
                           f"  발견된 실험: {found}\n"
                           f"  해당 주차의 결과 폴더(…{tag}…)를 results/ 아래에 두었는지 확인하세요.")
    return selected, ignored


def check_consistency(runs):
    """
    비교가 공정한지 점검하고, 문제가 있으면 경고를 출력한다. (중단하지는 않음)
      1) device(cuda/mps/cpu)가 서로 다른 실험이 섞여 있는지  ← 요청 사항
      2) 학습 설정(epochs, lr 등)이 서로 다른 실험이 섞여 있는지
         (끝나지 않은 실험·epoch가 다른 폴더는 load_runs에서 에러로 막는다)
      3) 시드가 2개 미만인 모델이 있는지 (표준편차를 못 구함)
      4) 같은 모델 안에서 epoch 시간이 유독 긴 실험이 있는지 (시간 비교가 왜곡될 수 있음)
    """
    n_warnings = 0

    # 1) 장치
    by_device = {}
    for run in runs:
        by_device.setdefault(run["args"].get("device", "unknown"), []).append(run["name"])
    if len(by_device) > 1:
        n_warnings += 1
        print("[경고] args.json의 device가 서로 다른 실험이 섞여 있습니다. "
              "장치마다 연산 방식이 달라 정확도·시간 비교에 영향을 줄 수 있습니다.")
        for device, names in sorted(by_device.items()):
            print(f"       - {device}: {len(names)}개 ({', '.join(names)})")

    # 2) 학습 설정
    for key in SAME_SETTING_KEYS:
        values = {}
        for run in runs:
            values.setdefault(json.dumps(run["args"].get(key)), []).append(run["name"])
        if len(values) > 1:
            n_warnings += 1
            print(f"[경고] 실험마다 '{key}' 값이 다릅니다: "
                  + ", ".join(f"{k} ({len(v)}개)" for k, v in values.items()))

    # 3) 시드 수
    seed_counts = pd.Series([r["model"] for r in runs]).value_counts()
    for model, count in seed_counts.items():
        if count < 2:
            n_warnings += 1
            print(f"[경고] {model}: 시드가 {count}개라 표준편차를 계산할 수 없습니다.")

    # 4) epoch 시간 이상치
    for model in seed_counts.index:
        times = {r["name"]: r["history"]["epoch_time_sec"].mean() for r in runs if r["model"] == model}
        if len(times) < 3:
            continue
        median = float(np.median(list(times.values())))
        for name, value in times.items():
            if value > EPOCH_TIME_OUTLIER_RATIO * median:
                n_warnings += 1
                print(f"[경고] {name}: epoch당 {value:.1f}초로 같은 모델 시드 중앙값({median:.1f}초)의 "
                      f"{value / median:.1f}배입니다. 실행 당시 환경(다른 프로그램 동시 실행 등)이 달랐을 수 있어 "
                      "소요 시간 비교에 주의하세요. (표에는 중앙값 열도 함께 저장합니다)")

    if n_warnings == 0:
        print("[점검] device·학습 설정·epoch 수·시간이 모두 일관됩니다.")
    return n_warnings


# =====================================================================
# 2. 요약표
# =====================================================================
def per_run_metrics(runs):
    """실험(모델 × 시드) 하나당 한 행의 지표 표를 만든다."""
    rows = []
    for run in runs:
        h = run["history"]
        best_idx = h["val_acc"].idxmax()  # 동점이면 처음 나온 epoch (train.py의 best 기준과 동일)
        rows.append({
            "run": run["name"], "model": run["model"], "seed": run["seed"],
            "num_params": run["args"]["num_params"],
            "best_val_acc_pct": h["val_acc"].max() * 100,
            "last5_val_acc_pct": h["val_acc"].tail(LAST_N_EPOCHS).mean() * 100,
            "best_epoch": int(h.loc[best_idx, "epoch"]),
            "overfit_gap_pp": (h["train_acc"].iloc[-1] - h["val_acc"].iloc[-1]) * 100,
            "epoch_time_sec": h["epoch_time_sec"].mean(),
            # 분석 4(20 vs 40 epoch)용: 20번째 epoch의 val acc. 요약표(summarize)에는 쓰지 않는다.
            "val_acc_ep20_pct": (h.loc[h["epoch"] == 20, "val_acc"].iloc[0] * 100) if (h["epoch"] == 20).any() else np.nan,
        })
    return pd.DataFrame(rows)


METRIC_COLS = ["best_val_acc_pct", "last5_val_acc_pct", "best_epoch", "overfit_gap_pp", "epoch_time_sec"]


def summarize(per_run):
    """모델별로 시드 평균·표준편차를 구하고 best val acc 내림차순으로 정렬한다."""
    spec = {"n_seeds": ("seed", "count"), "num_params": ("num_params", "first")}
    for col in METRIC_COLS:
        spec[f"{col}_mean"] = (col, "mean")
        spec[f"{col}_std"] = (col, "std")  # ddof=1 (표본 표준편차)
    spec["epoch_time_sec_median"] = ("epoch_time_sec", "median")
    summary = per_run.groupby("model").agg(**spec).reset_index()
    summary = summary.sort_values("best_val_acc_pct_mean", ascending=False).reset_index(drop=True)

    # Baseline 대비 차이 (%p)
    base = summary[summary["model"] == BASELINE]
    if base.empty:
        print(f"[경고] '{BASELINE}' 결과가 없어 Baseline 대비 차이를 계산하지 못했습니다.")
        summary["diff_vs_baseline_pp_mean"] = np.nan
        summary["diff_vs_baseline_pp_std"] = np.nan
        summary["diff_vs_seed_std"] = "-"
        return summary

    base_mean = base["best_val_acc_pct_mean"].iloc[0]
    base_std = base["best_val_acc_pct_std"].iloc[0]
    std = summary["best_val_acc_pct_std"]
    summary["diff_vs_baseline_pp_mean"] = summary["best_val_acc_pct_mean"] - base_mean
    summary["diff_vs_baseline_pp_std"] = np.sqrt(std ** 2 + base_std ** 2)
    larger = summary["diff_vs_baseline_pp_mean"].abs() > np.maximum(std, base_std)
    summary["diff_vs_seed_std"] = np.where(larger, "larger", "smaller")
    is_base = summary["model"] == BASELINE
    summary.loc[is_base, "diff_vs_baseline_pp_std"] = np.nan
    summary.loc[is_base, "diff_vs_seed_std"] = "-"
    return summary


def pm(mean, std, digits=2):
    """'평균 ± 표준편차' 문자열 (표준편차가 없으면 평균만)."""
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def print_summary(summary):
    """요약표를 읽기 좋은 형태로 콘솔에 출력한다."""
    table = pd.DataFrame({
        "model": summary["model"],
        "seeds": summary["n_seeds"],
        "params": summary["num_params"].map("{:,}".format),
        "best val acc %": [pm(m, s) for m, s in zip(summary.best_val_acc_pct_mean, summary.best_val_acc_pct_std)],
        f"last{LAST_N_EPOCHS} val acc %": [pm(m, s) for m, s in zip(summary.last5_val_acc_pct_mean, summary.last5_val_acc_pct_std)],
        "best epoch": [pm(m, s, 1) for m, s in zip(summary.best_epoch_mean, summary.best_epoch_std)],
        "train-val gap %p": [pm(m, s) for m, s in zip(summary.overfit_gap_pp_mean, summary.overfit_gap_pp_std)],
        "sec/epoch": [pm(m, s, 1) for m, s in zip(summary.epoch_time_sec_mean, summary.epoch_time_sec_std)],
        "vs baseline %p": [pm(m, s) for m, s in zip(summary.diff_vs_baseline_pp_mean, summary.diff_vs_baseline_pp_std)],
        "> seed std?": summary["diff_vs_seed_std"],
    })
    print("\n[모델별 요약: 시드 평균 ± 표준편차, best val acc 내림차순]")
    print(table.to_string(index=False))


# =====================================================================
# 3. 그림
# =====================================================================
def stack_curves(runs, model, column):
    """한 모델의 시드별 곡선을 (epoch × 시드) 표로 쌓는다."""
    curves = {r["seed"]: r["history"].set_index("epoch")[column] for r in runs if r["model"] == model}
    return pd.DataFrame(curves).sort_index()


def plot_val_acc_curves(runs, order, path, epochs):
    """그림 1: 모델별 val acc 학습 곡선 (시드 평균 선 + 최소~최대 음영)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), layout="constrained")
    zoom_low = None
    for model in order:
        acc = stack_curves(runs, model, "val_acc") * 100
        mean, lo, hi = acc.mean(axis=1), acc.min(axis=1), acc.max(axis=1)
        for ax in axes:
            ax.plot(mean.index, mean, color=color_of(model), lw=2,
                    ls="--" if model == BASELINE else "-", label=model)
            ax.fill_between(mean.index, lo, hi, color=color_of(model), alpha=0.15, linewidth=0)
    final_best = max(stack_curves(runs, m, "val_acc").iloc[-1].mean() * 100 for m in order)
    zoom_low = final_best - 8  # 오른쪽 그림: 최고 모델에서 8%p 이내만 확대

    axes[0].set_title("val acc per epoch (line = mean over seeds, band = min~max)")
    zoom_start = int(epochs * 0.4)  # 20 epoch면 8, 40 epoch면 16부터 확대
    axes[1].set_title(f"zoom: epochs {zoom_start}~{epochs}, near the top (lower models are cut off)")
    axes[1].set_xlim(zoom_start, None)
    axes[1].set_ylim(zoom_low, final_best + 2)
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_ylabel("val acc (%)")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower right")
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def plot_loss_curves(runs, order, path):
    """그림 2: 모델별 train/val loss 곡선 (모델마다 칸을 나눠 과적합을 비교)."""
    n_cols = 3
    n_rows = math.ceil(len(order) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.8 * n_rows),
                             sharex=True, sharey=True, layout="constrained", squeeze=False)
    train_color, val_color = "#0072B2", "#D55E00"
    for ax, model in zip(axes.ravel(), order):
        for column, color, label, style in [("train_loss", train_color, "train loss", "-"),
                                            ("val_loss", val_color, "val loss", "--")]:
            loss = stack_curves(runs, model, column)
            ax.plot(loss.index, loss.mean(axis=1), color=color, ls=style, lw=2, label=label)
            ax.fill_between(loss.index, loss.min(axis=1), loss.max(axis=1), color=color, alpha=0.15, linewidth=0)
        ax.set_title(model)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(order):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("loss")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("train vs val loss (mean over seeds, band = min~max)\n"
                 "note: train loss is measured with augmentation and in train mode (Dropout/BN active), "
                 "so it is not directly comparable to val loss", fontsize=10)
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def plot_best_acc_bar(summary, per_run, path):
    """그림 3: best val acc 막대그래프 (오차막대 = 시드 표준편차, Baseline 기준선, 시드별 점)."""
    fig, ax = plt.subplots(figsize=(9 if len(summary) <= 6 else 11, 5.5), layout="constrained")
    x = np.arange(len(summary))
    means = summary["best_val_acc_pct_mean"].to_numpy()
    stds = summary["best_val_acc_pct_std"].fillna(0).to_numpy()
    ax.bar(x, means, yerr=stds, capsize=6, width=0.6,
           color=[color_of(m) for m in summary["model"]], alpha=0.85, error_kw={"lw": 1.5})
    for i, model in enumerate(summary["model"]):  # 시드별 값을 점으로 (3개뿐이라 분포를 그대로 보여줌)
        seed_values = per_run.loc[per_run["model"] == model, "best_val_acc_pct"]
        ax.scatter(np.full(len(seed_values), i), seed_values, color="white", edgecolor="black", s=28, zorder=3)
    for i, model in enumerate(summary["model"]):  # 값 라벨은 오차막대·시드 점보다 위에 둔다
        top = max(means[i] + stds[i], per_run.loc[per_run["model"] == model, "best_val_acc_pct"].max())
        ax.text(i, top + 0.4, f"{means[i]:.2f}", ha="center", va="bottom", fontsize=9)

    base = summary[summary["model"] == BASELINE]
    if not base.empty:
        base_mean = base["best_val_acc_pct_mean"].iloc[0]
        ax.axhline(base_mean, color="black", ls="--", lw=1, label=f"baseline mean ({base_mean:.2f}%)")
        ax.legend(loc="lower left")

    low = (means - stds).min()
    bottom = max(0, math.floor((low - 5) / 10) * 10)  # 10단위로 내린 값에서 시작 (차이가 보이도록)
    ax.set_ylim(bottom, 100 if means.max() > 90 else means.max() + stds.max() + 3)
    ax.set_xticks(x, [m.replace("_", "\n") for m in summary["model"]])
    ax.set_ylabel(f"best val acc (%)   [y-axis starts at {bottom}]")
    ax.set_title("best val acc by model (bar = mean, error bar = std over seeds, dots = each seed)")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def _plot_params_vs_acc_legacy(summary, path):
    """그림 4 (3주차 배치): 점 옆에 모델 이름을 적는다. 3주차 그림을 그대로 재현하기 위해 유지."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5), layout="constrained")
    for _, row in summary.iterrows():
        ax.errorbar(row["num_params"], row["best_val_acc_pct_mean"], yerr=row["best_val_acc_pct_std"],
                    fmt="o", ms=9, capsize=4, color=color_of(row["model"]))
        # x가 거의 같은 baseline/dropout/bn 라벨이 겹치지 않도록 baseline·dropout은 왼쪽에 쓴다
        left = row["model"] in (BASELINE, "dropout")
        ax.annotate(row["model"], (row["num_params"], row["best_val_acc_pct_mean"]),
                    xytext=(-10 if left else 10, 0), textcoords="offset points",
                    ha="right" if left else "left", va="center", fontsize=10,
                    bbox={"fc": "white", "ec": "none", "alpha": 0.8, "pad": 1})
    base = summary[summary["model"] == BASELINE]
    if not base.empty:
        ax.axhline(base["best_val_acc_pct_mean"].iloc[0], color="black", ls="--", lw=1, label="baseline mean")
        ax.legend(loc="lower right")
    ax.set_xscale("log")  # 9.5만 ~ 142만으로 범위가 넓어 로그 축
    ax.margins(x=0.12)    # 가장자리 라벨이 잘리지 않게 여백
    ax.set_xlabel("number of parameters (log scale)")
    ax.set_ylabel("best val acc (%)")
    ax.set_title("parameters vs best val acc (error bar = std over seeds)\n"
                 "baseline / dropout / bn share almost the same x (620k)")
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def spread_x(summary, tol=1.02, step=1.10):
    """
    파라미터 수가 거의 같은(비율 tol 미만) 모델끼리 묶어, 그림에서 겹치지 않게 x를 step배씩 옆으로 민다.
    반환: {모델: 그릴 때 쓸 x}. 실제 파라미터 수는 바뀌지 않는다 (그림에서만 이동).
    """
    ordered = summary.sort_values("num_params")
    xs, cluster = {}, []

    def flush():
        for i, row in enumerate(cluster):
            xs[row["model"]] = row["num_params"] * step ** (i - (len(cluster) - 1) / 2)

    for _, row in ordered.iterrows():
        if cluster and row["num_params"] / cluster[-1]["num_params"] < tol:
            cluster.append(row)
        else:
            flush()
            cluster = [row]
    flush()
    return xs


def plot_params_vs_acc(summary, path, legacy_layout=True):
    """
    그림 4: 파라미터 수 vs best val acc 산점도 (오차막대 = 시드 표준편차).
    legacy_layout=True(3주차): 점 옆에 이름. False(4주차): 같은 크기 모델이 많아 이름이 겹치므로
    점을 옆으로 살짝 밀고 범례로 구분한다.
    """
    if legacy_layout:
        return _plot_params_vs_acc_legacy(summary, path)
    fig, ax = plt.subplots(figsize=(9.5, 5.8), layout="constrained")
    xs = spread_x(summary)
    for _, row in summary.iterrows():
        ax.errorbar(xs[row["model"]], row["best_val_acc_pct_mean"], yerr=row["best_val_acc_pct_std"],
                    fmt="o", ms=9, capsize=4, color=color_of(row["model"]), label=row["model"])
    base = summary[summary["model"] == BASELINE]
    if not base.empty:
        ax.axhline(base["best_val_acc_pct_mean"].iloc[0], color="black", ls="--", lw=1, label="baseline mean")
    shifted = any(abs(xs[m] / p - 1) > 1e-9 for m, p in zip(summary["model"], summary["num_params"]))
    ax.set_xscale("log")
    ax.margins(x=0.12)
    ax.set_xlabel("number of parameters (log scale)")
    ax.set_ylabel("best val acc (%)")
    ax.set_title("parameters vs best val acc (error bar = std over seeds)"
                 + ("\nmodels with (almost) the same parameter count are shifted sideways for visibility" if shifted else ""))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


# =====================================================================
# 4. 상위 모델 클래스별 정확도 (evaluate.py --run 호출)
# =====================================================================
def run_evaluate(run_name):
    """
    evaluate.py --run <실험>을 호출한다. (--test는 절대 붙이지 않음 → val 평가)
    성공하면 True. 실패하면 이유를 출력하고 False.
    """
    command = [sys.executable, "evaluate.py", "--run", run_name]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}  # Windows 콘솔 인코딩과 무관하게 한글 출력 해석
    result = subprocess.run(command, cwd=PROJECT_DIR, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", env=env)
    if result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout).strip().splitlines()[-3:])
        print(f"[경고] evaluate.py 실행 실패 ({run_name}):\n       " + tail.replace("\n", "\n       "))
        return False
    return True


def load_val_per_class(run_name):
    """results/<실험>/val_per_class_acc.csv를 읽고, val 결과가 맞는지 확인한다."""
    path = RESULTS_DIR / run_name / "val_per_class_acc.csv"
    table = pd.read_csv(path)
    if not (table["total"] == VAL_PER_CLASS_TOTAL).all():  # val은 클래스당 500장
        raise ValueError(f"{path}: 클래스당 {VAL_PER_CLASS_TOTAL}장이 아닙니다 (val 결과가 아닐 수 있음)")
    return table.set_index("class")["accuracy"] * 100


def collect_per_class(runs, summary, top_k, no_eval, force_eval):
    """
    상위 top_k개 모델(+Baseline, top_k=None이면 전체 모델)의 모든 시드에 대해 val 클래스별 정확도를 모은다.
    CSV가 없거나, 체크포인트(best.pt)보다 오래됐거나, --force-eval이면 evaluate.py를 호출한다.
    반환: (실험 × 클래스) 정확도 표 (%), 비교 대상 모델 순서
    """
    ranked = [m for m in summary["model"] if m != BASELINE]
    targets = ([BASELINE] if BASELINE in set(summary["model"]) else []) + ranked[:top_k]
    scope = "전체 모델" if top_k is None else f"best val acc 상위 {top_k}개 + {BASELINE}"
    print(f"\n[클래스별 정확도 비교 대상] {', '.join(targets)}  ({scope})")

    columns, owners = {}, {}
    for run in runs:
        if run["model"] not in targets:
            continue
        run_dir = RESULTS_DIR / run["name"]
        csv_path, ckpt_path = run_dir / "val_per_class_acc.csv", run_dir / "best.pt"
        stale = csv_path.exists() and ckpt_path.exists() and csv_path.stat().st_mtime < ckpt_path.stat().st_mtime
        need_eval = force_eval or stale or not csv_path.exists()
        if need_eval and not no_eval:
            if not ckpt_path.exists():
                print(f"[경고] {run['name']}: best.pt가 없어 evaluate.py를 실행할 수 없습니다.")
            else:
                print(f"  evaluate.py --run {run['name']}")
                run_evaluate(run["name"])
        if not csv_path.exists():
            print(f"[경고] {run['name']}: val_per_class_acc.csv가 없어 클래스별 비교에서 제외합니다.")
            continue
        columns[run["name"]] = load_val_per_class(run["name"])
        owners[run["name"]] = run

    if not columns:
        return None, targets
    per_class = pd.DataFrame(columns).T  # 행 = 실험, 열 = 클래스

    # 검증: 클래스 정확도 평균(=전체 val acc, 클래스당 500장 동일)이 history의 best val acc와 같아야 한다.
    for name, run in owners.items():
        history_best = run["history"]["val_acc"].max() * 100
        if abs(per_class.loc[name].mean() - history_best) > PER_CLASS_MISMATCH_PP:
            print(f"[경고] {name}: 클래스별 평균 {per_class.loc[name].mean():.2f}% ≠ history best "
                  f"{history_best:.2f}% (체크포인트/장치가 다를 수 있음)")
    return per_class.assign(model=[owners[n]["model"] for n in per_class.index]), targets


def build_per_class_table(per_class, targets):
    """실험별 표 → 모델별 (클래스 × [평균, 표준편차, Baseline 대비 차이]) 하나의 표."""
    class_cols = [c for c in per_class.columns if c != "model"]
    n_runs = per_class.groupby("model").size()
    out = pd.DataFrame(index=class_cols)
    models = [m for m in targets if m in n_runs.index]
    for model in models:
        values = per_class[per_class["model"] == model][class_cols]
        out[f"{model}_mean_pct"] = values.mean()
        out[f"{model}_std_pct"] = values.std()  # 시드가 1개면 NaN
    if BASELINE in models:
        for model in models:
            if model != BASELINE:
                out[f"{model}_diff_vs_baseline_pp"] = out[f"{model}_mean_pct"] - out[f"{BASELINE}_mean_pct"]
    # 클래스당 500장으로 같으므로 클래스 평균 = 전체 val acc
    macro = {col: out[col].mean() if col.endswith("_mean_pct") or col.endswith("_pp") else np.nan
             for col in out.columns}
    out.loc["macro_avg"] = pd.Series(macro)
    out.index.name = "class"
    print("\n[val 클래스별 정확도 (%): 시드 평균 ± 표준편차]  시드 수: "
          + ", ".join(f"{m}={n_runs[m]}" for m in models))
    show = pd.DataFrame({m: [pm(out.loc[c, f"{m}_mean_pct"], out.loc[c, f"{m}_std_pct"]) for c in out.index]
                         for m in models}, index=out.index)
    print(show.to_string())
    return out


# =====================================================================
# 5. 4주차 추가 분석
# =====================================================================
def _flag(diff, *stds):
    """|차이|가 표준편차(들) 중 큰 값보다 크면 'larger', 아니면 'smaller'. 표준편차를 모르면 '-'."""
    stds = [x for x in stds if not pd.isna(x)]
    if pd.isna(diff) or not stds:
        return "-"
    return "larger" if abs(diff) > max(stds) else "smaller"


def _quad(*stds):
    """독립이라고 가정하고 표준편차를 합성: sqrt(sum s^2). 하나라도 NaN이면 NaN."""
    if any(pd.isna(x) for x in stds):
        return np.nan
    return float(np.sqrt(sum(x ** 2 for x in stds)))


def factorial_table(per_run):
    """
    [분석 1] BN 고정 2x2 효과표. (정의와 해석은 파일 맨 위 docstring 참고)
    Deep 효과 = bn_deep - bn, Wide 효과 = bn_wide - bn,
    상호작용 = (bn_deep_wide - bn_wide) - (bn_deep - bn).  단위 %p.
    """
    present = set(per_run["model"])
    missing = [m for m in FACTORIAL_MODELS.values() if m not in present]
    if missing:
        print(f"[경고] 2x2 효과표에 필요한 모델이 없어 건너뜁니다: {', '.join(missing)}")
        return None
    rows = []
    for metric in ("best_val_acc_pct", "last5_val_acc_pct"):
        grouped = per_run.groupby("model")[metric]
        mean, std = grouped.mean(), grouped.std()
        m = {k: mean[v] for k, v in FACTORIAL_MODELS.items()}
        s = {k: std[v] for k, v in FACTORIAL_MODELS.items()}
        deep = m["deep"] - m["base"]
        wide = m["wide"] - m["base"]
        inter = (m["both"] - m["wide"]) - (m["deep"] - m["base"])
        deep_std, wide_std = _quad(s["deep"], s["base"]), _quad(s["wide"], s["base"])
        inter_std = _quad(s["both"], s["wide"], s["deep"], s["base"])
        inter_flag = _flag(inter, inter_std) if not pd.isna(inter_std) else "-"
        if inter_flag == "-":
            reading = "표준편차를 알 수 없음(시드 1개)"
        elif inter_flag == "smaller":
            reading = "표준편차 안: 0과 구분되지 않음 -> 두 효과가 더해진 것으로 볼 수 있음"
        elif inter < 0:
            reading = "음수: 효과가 겹침 (함께 쓰면 각자의 이득의 합보다 작음)"
        else:
            reading = "양수: 함께 쓰면 각자의 이득의 합보다 큼 (시너지)"
        rows.append({
            "metric": metric,
            "bn_mean": m["base"], "bn_std": s["base"],
            "bn_deep_mean": m["deep"], "bn_deep_std": s["deep"],
            "bn_wide_mean": m["wide"], "bn_wide_std": s["wide"],
            "bn_deep_wide_mean": m["both"], "bn_deep_wide_std": s["both"],
            "deep_effect_pp": deep, "deep_effect_std_pp": deep_std,
            "deep_effect_vs_std": _flag(deep, deep_std) if not pd.isna(deep_std) else "-",
            "wide_effect_pp": wide, "wide_effect_std_pp": wide_std,
            "wide_effect_vs_std": _flag(wide, wide_std) if not pd.isna(wide_std) else "-",
            "interaction_pp": inter, "interaction_std_pp": inter_std, "interaction_vs_std": inter_flag,
            "additive_expected_pct": m["base"] + deep + wide,  # 상호작용이 0일 때 bn_deep_wide의 기대값
            "interaction_reading": reading,
        })
    return pd.DataFrame(rows)


def variants_table(summary):
    """[분석 2] bn_deep_wide 대비 dropout/gap 변형의 val acc 차이와 overfit_gap 차이."""
    by_model = summary.set_index("model")
    if VARIANT_REF not in by_model.index:
        print(f"[경고] '{VARIANT_REF}' 결과가 없어 변형 비교를 건너뜁니다.")
        return None
    ref = by_model.loc[VARIANT_REF]
    rows = []
    for model in [VARIANT_REF] + VARIANTS:
        if model not in by_model.index:
            print(f"[경고] '{model}' 결과가 없어 변형 비교에서 제외합니다.")
            continue
        r = by_model.loc[model]
        row = {"model": model, "n_seeds": int(r["n_seeds"]), "num_params": int(r["num_params"])}
        for key, label in (("best_val_acc_pct", "best_val_acc"), ("last5_val_acc_pct", "last5_val_acc"),
                           ("overfit_gap_pp", "overfit_gap")):
            row[f"{label}_mean"], row[f"{label}_std"] = r[f"{key}_mean"], r[f"{key}_std"]
            if model == VARIANT_REF:
                row[f"{label}_diff_vs_ref_pp"], row[f"{label}_diff_std_pp"], row[f"{label}_vs_seed_std"] = np.nan, np.nan, "-"
            else:
                diff = r[f"{key}_mean"] - ref[f"{key}_mean"]
                row[f"{label}_diff_vs_ref_pp"] = diff
                row[f"{label}_diff_std_pp"] = _quad(r[f"{key}_std"], ref[f"{key}_std"])
                row[f"{label}_vs_seed_std"] = _flag(diff, r[f"{key}_std"], ref[f"{key}_std"])
        rows.append(row)
    return pd.DataFrame(rows)


def cat_dog_table(per_class_table, model_order):
    """[분석 3] 클래스별 정확도 표에서 cat, dog만 뽑아 모델별로 비교한다 (Baseline 대비 차이 포함)."""
    models = [m for m in model_order if f"{m}_mean_pct" in per_class_table.columns]
    rows = []
    for model in models:
        row = {"model": model}
        for cls in FOCUS_CLASSES:
            mean, std = per_class_table.loc[cls, f"{model}_mean_pct"], per_class_table.loc[cls, f"{model}_std_pct"]
            row[f"{cls}_mean_pct"], row[f"{cls}_std_pct"] = mean, std
            diff_col = f"{model}_diff_vs_baseline_pp"
            if diff_col in per_class_table.columns:
                diff = per_class_table.loc[cls, diff_col]
                row[f"{cls}_diff_vs_baseline_pp"] = diff
                row[f"{cls}_vs_seed_std"] = _flag(diff, std, per_class_table.loc[cls, f"{BASELINE}_std_pct"])
            else:  # baseline 자신
                row[f"{cls}_diff_vs_baseline_pp"], row[f"{cls}_vs_seed_std"] = np.nan, "-"
        row["cat_dog_mean_pct"] = np.mean([row[f"{c}_mean_pct"] for c in FOCUS_CLASSES])
        rows.append(row)
    return pd.DataFrame(rows)


EPOCH_COMPARE_COLS = ["best_val_acc_pct", "last5_val_acc_pct", "val_acc_ep20_pct",
                      "best_epoch", "overfit_gap_pp", "epoch_time_sec"]


def epoch_compare_table(runs40, runs20):
    """
    [분석 4] baseline, bn의 20 epoch(3주차) vs 40 epoch(4주차). 모델마다 20, 40, delta_40_minus_20(= 40 - 20) 세 줄.
    40-20의 표준편차는 두 조건의 시드 표준편차를 독립으로 합성한다. 판정은 요약표와 같은 규칙(max 표준편차).
    """
    pr40, pr20 = per_run_metrics(runs40), per_run_metrics(runs20)
    rows = []
    for model in EPOCH_COMPARE_MODELS:
        a, b = pr20[pr20["model"] == model], pr40[pr40["model"] == model]
        if a.empty or b.empty:
            print(f"[경고] {model}: 20 epoch 또는 40 epoch 결과가 없어 epoch 비교에서 제외합니다.")
            continue
        r20 = {"model": model, "epochs": "20", "n_seeds": len(a)}
        r40 = {"model": model, "epochs": "40", "n_seeds": len(b)}
        delta = {"model": model, "epochs": "delta_40_minus_20", "n_seeds": min(len(a), len(b))}
        for col in EPOCH_COMPARE_COLS:
            m20, s20, m40, s40 = a[col].mean(), a[col].std(), b[col].mean(), b[col].std()
            r20[f"{col}_mean"], r20[f"{col}_std"] = m20, s20
            r40[f"{col}_mean"], r40[f"{col}_std"] = m40, s40
            delta[f"{col}_mean"], delta[f"{col}_std"] = m40 - m20, _quad(s20, s40)
            if col in ("best_val_acc_pct", "last5_val_acc_pct"):
                delta[f"{col}_vs_seed_std"] = _flag(m40 - m20, s20, s40)
        rows.extend([r20, r40, delta])
    return pd.DataFrame(rows) if rows else None


def warn_epoch_compare_settings(runs40, runs20):
    """20 epoch와 40 epoch 실험이 epochs 외의 설정(device 포함)에서 다르면 경고한다 (공정한 비교가 아님)."""
    for key in [k for k in SAME_SETTING_KEYS if k != "epochs"] + ["device"]:
        v20 = {json.dumps(r["args"].get(key)) for r in runs20}
        v40 = {json.dumps(r["args"].get(key)) for r in runs40}
        if v20 != v40 or len(v20) > 1:
            print(f"[경고] 20 epoch와 40 epoch 실험의 '{key}'가 다릅니다 (20: {sorted(v20)}, 40: {sorted(v40)}). "
                  "epoch 외 요인이 섞여 있어 비교에 주의하세요.")


def print_table(title, df):
    """분석 표를 소수 2자리로 콘솔에 출력한다."""
    print(f"\n{title}")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


def save_table(df, path):
    df.round(4).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"표 저장: {path}")


# =====================================================================
# 6. 3·4주차 취합 (저장 경로를 받아서 저장 → --verify가 임시 폴더로 같은 코드를 실행할 수 있음)
# =====================================================================
def describe_runs(runs):
    return ", ".join(f"{m}x{n}" for m, n in pd.Series([r["model"] for r in runs]).value_counts().sort_index().items())


def describe_ignored(ignored):
    """읽지 않은 실험을 (모델, epoch, 튜닝 여부)별로 센다. 튜닝 설정(5주차)이면 표시를 붙인다."""
    counts = Counter((r["model"], r["args"].get("epochs"), r.get("tuned", False)) for r in ignored)
    return ", ".join(f"{m} ep{e}{'(튜닝 설정)' if tuned else ''}x{n}" for (m, e, tuned), n in sorted(counts.items()))


def run_week(week, paths, top_k, no_eval, force_eval):
    """
    3·4주차 취합 본체. 표와 그림을 paths에 저장한다. 결과가 섞여 있으면 ResultsError.
    (예전 main()의 내용을 그대로 옮기고, 저장 위치만 paths로 받게 했다)
    """
    cfg = WEEK_CONFIGS[week]
    runs, ignored = load_runs(cfg["models"], cfg["epochs"])
    print(f"[{week}주차] epochs={cfg['epochs']} 실험 {len(runs)}개 읽음: {describe_runs(runs)}")
    if ignored:
        print("  읽지 않은 실험(다른 주차/모델/튜닝 설정): " + describe_ignored(ignored))
    expected = len(cfg["models"]) * 3
    if len(runs) != expected:
        print(f"[경고] 실험이 {len(runs)}개입니다 (모델 {len(cfg['models'])}개 x 시드 3개 = {expected}개가 기대값). "
              "일부 모델/시드가 없거나 아직 학습 중일 수 있습니다.")
    check_consistency(runs)

    per_run = per_run_metrics(runs)
    summary = summarize(per_run)
    save_table(summary, paths["summary"])
    print_summary(summary)

    paths["fig_dir"].mkdir(parents=True, exist_ok=True)
    order = list(summary["model"])
    plot_val_acc_curves(runs, order, paths["fig_dir"] / "1_val_acc_curves.png", cfg["epochs"])
    plot_loss_curves(runs, order, paths["fig_dir"] / "2_loss_curves.png")
    plot_best_acc_bar(summary, per_run, paths["fig_dir"] / "3_best_val_acc_bar.png")
    plot_params_vs_acc(summary, paths["fig_dir"] / "4_params_vs_acc.png", legacy_layout=(week == 3))
    print(f"그림 저장: {paths['fig_dir']}")

    # ---- 4주차 추가 분석 1, 2, 4 (클래스별 평가가 필요 없는 것)
    if week == 4:
        factorial = factorial_table(per_run)
        if factorial is not None:
            print_table("[분석 1] BN 고정 2x2 효과표 (%p)", factorial.drop(columns=["interaction_reading"]))
            for _, r in factorial.iterrows():
                print(f"  - {r['metric']}: 상호작용 {r['interaction_pp']:+.2f} ± {r['interaction_std_pp']:.2f} %p -> {r['interaction_reading']}")
            save_table(factorial, paths["factorial"])
        variants = variants_table(summary)
        if variants is not None:
            print_table("[분석 2] bn_deep_wide 대비 dropout/gap 변형", variants)
            save_table(variants, paths["variants"])
        try:
            runs20, _ = load_runs(EPOCH_COMPARE_MODELS, WEEK_CONFIGS[3]["epochs"])
        except ResultsError as error:
            print(f"[경고] 3주차(ep20) 결과를 읽지 못해 20 vs 40 epoch 비교를 건너뜁니다:\n{error}")
            runs20 = None
        if runs20:
            warn_epoch_compare_settings([r for r in runs if r["model"] in EPOCH_COMPARE_MODELS], runs20)
            compare = epoch_compare_table([r for r in runs if r["model"] in EPOCH_COMPARE_MODELS], runs20)
            if compare is not None:
                print_table("[분석 4] baseline, bn: 20 epoch(3주차) vs 40 epoch(4주차)", compare)
                save_table(compare, paths["epoch_compare"])

    # ---- 클래스별 정확도 (evaluate.py 호출)
    per_class, targets = collect_per_class(runs, summary, top_k, no_eval, force_eval)
    if per_class is None:
        print("[경고] 클래스별 정확도 결과가 하나도 없어 표를 만들지 못했습니다.")
        return
    table = build_per_class_table(per_class, targets)
    table.round(4).to_csv(paths["per_class"], encoding="utf-8-sig")
    print(f"\n표 저장: {paths['per_class']}")

    if week == 4:  # 분석 3
        missing = [c for c in FOCUS_CLASSES if c not in table.index]
        if missing:
            print(f"[경고] 클래스 {missing}가 표에 없어 cat/dog 비교를 건너뜁니다.")
        else:
            cat_dog = cat_dog_table(table, order)
            print_table("[분석 3] cat, dog 클래스 정확도 (val, %)", cat_dog)
            save_table(cat_dog, paths["cat_dog"])


# =====================================================================
# 7. 3·4주차 재현 확인 (--verify)
# =====================================================================
def compare_csv(old_path, new_path):
    """
    두 CSV를 열 단위로 비교한다.
    반환: ({열: 최대 차이}, 다른 열 목록, 문제 설명). 열 이름·행 수부터 다르면 문제 설명만 채운다.
    숫자 열은 1e-9보다 큰 차이를 다르다고 보고, 둘 다 NaN인 칸은 같다고 본다.
    """
    old = pd.read_csv(old_path, encoding="utf-8-sig")
    new = pd.read_csv(new_path, encoding="utf-8-sig")
    if list(old.columns) != list(new.columns):
        return None, None, f"열이 다릅니다 (기존 {list(old.columns)} / 재계산 {list(new.columns)})"
    if len(old) != len(new):
        return None, None, f"행 수가 다릅니다 (기존 {len(old)} / 재계산 {len(new)})"
    worst, bad = {}, []
    for col in old.columns:
        if pd.api.types.is_numeric_dtype(old[col]) and pd.api.types.is_numeric_dtype(new[col]):
            a, b = old[col].to_numpy(float), new[col].to_numpy(float)
            if (np.isnan(a) != np.isnan(b)).any():
                worst[col] = math.inf
            else:
                diff = np.where(np.isnan(a), 0.0, np.abs(a - b))
                worst[col] = float(diff.max()) if len(diff) else 0.0
        else:
            worst[col] = 0.0 if (old[col].astype(str) == new[col].astype(str)).all() else math.inf
        if worst[col] > 1e-9:
            bad.append(col)
    return worst, bad, None


def verify_week(week):
    """
    3·4주차 출력을 임시 폴더에 다시 만들어 기존 파일과 비교한다. results/·figures/에는 아무것도 쓰지 않는다.
    evaluate.py는 호출하지 않는다 (--no-eval과 같음, 이미 있는 val_per_class_acc.csv만 사용).
    시간 열(epoch_time_*)은 실행 환경에 따라 달라지는 값이라 학습 결과 열과 나눠서 보고한다.
    종료 코드: 0 = 모두 같음, 1 = 학습 결과 열이 다르거나 비교할 수 없음, 2 = 소요 시간 열만 다름.
    """
    import matplotlib.image as mpimg
    real = week_paths(week)
    print(f"[{week}주차 재현 확인] 임시 폴더에 다시 만들어 기존 출력과 비교합니다. 파일은 쓰지 않습니다.")
    ok, time_only = True, False
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "results").mkdir()
        new = week_paths(week, tmp / "results", tmp / "figures")
        log = io.StringIO()
        try:
            with contextlib.redirect_stdout(log):  # 취합 과정의 긴 출력은 숨기고 비교 결과만 보여 준다
                run_week(week, new, WEEK_CONFIGS[week]["default_top_k"], no_eval=True, force_eval=False)
        except ResultsError as error:
            print(f"[오류] {error}")
            return 1
        for key in VERIFY_TABLES[week]:
            name = real[key].name
            if not real[key].exists() or not new[key].exists():
                print(f"[불일치] {name}: {'기존 파일이 없습니다' if not real[key].exists() else '다시 만들지 못했습니다'}")
                ok = False
                continue
            worst, bad, problem = compare_csv(real[key], new[key])
            if problem:
                print(f"[불일치] {name}: {problem}")
                ok = False
                continue
            time_bad = [c for c in bad if "epoch_time" in c]
            other_bad = [c for c in bad if c not in time_bad]
            n_other = len([c for c in worst if "epoch_time" not in c])
            n_rows = len(pd.read_csv(real[key], encoding="utf-8-sig"))
            if other_bad:
                print(f"[불일치] {name}: 학습 결과 열이 다릅니다 {other_bad}")
                ok = False
            else:
                print(f"[일치] {name}: 시간 열을 뺀 {n_other}개 열 × {n_rows}행 모두 같습니다.")
            if time_bad:
                print(f"[시간만 다름] {name}: {time_bad} 최대 {max(worst[c] for c in time_bad):.2f}초 차이 "
                      "(정확도·손실 값과는 무관)")
                time_only = True
        # 그림: 픽셀 단위 비교 (matplotlib 버전·폰트가 다르면 조금 다를 수 있어 정보로만 보고)
        for png in sorted(new["fig_dir"].glob("*.png")):
            old_png = real["fig_dir"] / png.name
            if not old_png.exists():
                print(f"[정보] 기존 그림이 없어 비교하지 않음: {png.name}")
                continue
            a, b = mpimg.imread(png), mpimg.imread(old_png)
            if a.shape != b.shape:
                print(f"[정보] {png.name}: 크기가 다릅니다 {a.shape} vs {b.shape}")
            else:
                d = float(np.abs(a.astype(float) - b.astype(float)).max())
                print(f"[{'일치' if d == 0 else '정보'}] {png.name}: 픽셀 최대 차이 {d:.4g}"
                      + ("" if d == 0 else " (matplotlib 버전·폰트 차이일 수 있음)"))
    if not ok:
        print(f"[결과] {week}주차 출력의 학습 결과 열이 기존 파일과 다릅니다.")
        return 1
    if time_only:
        print(f"[결과] {week}주차 학습 결과 열은 모두 같고, 소요 시간 열만 다릅니다.")
        return 2
    print(f"[결과] {week}주차 출력이 기존 파일과 모두 같습니다.")
    return 0


# =====================================================================
# 8. 5주차 취합 (규칙은 week5_selection.py)
# =====================================================================
STAGE_COLUMNS = ["model", "stage", "lr", "scheduler", "weight_decay", "seed", "status", "source",
                 "last5_val_acc_pct", "best_val_acc_pct", "best_epoch", "overfit_gap_pp", "epoch_time_sec",
                 "diverged", "diff_vs_top_pp", "same_as_top", "winner", "selected", "selected_by", "run", "device"]
# lr별 색 (Okabe-Ito), scheduler별 선 모양 → 색과 선 모양 두 가지로 6조합을 구분
LR_COLORS = {0.1: "#D55E00", 0.05: "#E69F00", 0.01: "#0072B2"}
SCHEDULER_STYLES = {"none": "-", "cosine": "--", "step": ":"}
STAGE_A_YLIM_FROM_EPOCH = 5   # 그림 1의 y축 범위는 이 epoch 이후 값으로 정한다 (초반 낮은 값 때문에 차이가 안 보이는 것 방지)


def mark_selected(rows, cfg, source):
    """실제로 다음 단계에 쓴 설정(자동이든 직접 지정이든)을 표에 표시한다."""
    for row in rows or []:
        row["selected"] = cfg is not None and w5.cfg_equal((row["lr"], row["scheduler"], row["weight_decay"]), cfg)
        row["selected_by"] = source if row["selected"] else ""


def stage_frame(rows):
    """행 목록 → 저장용 표. '_'로 시작하는 내부 값(그림용 history)은 뺀다."""
    frame = pd.DataFrame([{k: v for k, v in row.items() if not k.startswith("_")} for row in rows])
    for col in STAGE_COLUMNS:
        if col not in frame:
            frame[col] = np.nan
    return frame[STAGE_COLUMNS]


def warn_week5_devices(model, stage, rows):
    """A·B 안에서 장치·PyTorch 버전이 섞였는지 확인한다 (README: A·B는 4주차를 학습한 PC에서 실행)."""
    devices = {(r["device"]) for r in rows or [] if r["status"] == "완료"}
    if len(devices) > 1:
        print(f"[경고] {model} {stage}: 서로 다른 장치의 결과가 섞여 있습니다 {sorted(devices)}. "
              "시드 1개 비교라 장치 차이가 결과에 섞일 수 있습니다.")


def warn_saved_auto(saved, model, winners):
    """run_experiments.py가 자동으로 고른 기록과 지금 결과로 고른 승자가 다르면 경고한다."""
    for stage in ("A", "B"):
        entry = saved.get(model, {}).get(stage)
        current = winners[stage]
        if entry and entry.get("source") == "자동" and current is not None \
                and not w5.cfg_equal((float(entry["lr"]), entry["scheduler"], float(entry["weight_decay"])), current):
            print(f"[경고] {model} {stage}: 실행 당시 자동 승자({w5.cfg_label((entry['lr'], entry['scheduler'], entry['weight_decay']))})와 "
                  f"지금 결과로 고른 승자({w5.cfg_label(current)})가 다릅니다. 실행 뒤 결과가 바뀌었는지 확인하세요.")


def final_row(model, winners, judge, index):
    """모델별 최종 설정 요약 한 줄."""
    row = {"model": model,
           "A_winner": w5.cfg_label(winners["A"]) if winners["A"] else "", "A_source": winners["A_source"] or "",
           "B_winner": w5.cfg_label(winners["B"]) if winners["B"] else "", "B_source": winners["B_source"] or "",
           "judgment": winners["problem"] if judge is None else judge["judgment"]}
    if judge is None:
        return row
    for side in ("tuned", "default"):
        s = judge[side]
        row[f"{side}_n_seeds"] = s["n_seeds"]
        for col in ("last5_val_acc_pct", "best_val_acc_pct", "overfit_gap_pp"):
            row[f"{side}_{col}_mean"], row[f"{side}_{col}_std"] = s[f"{col}_mean"], s[f"{col}_std"]
        row[f"{side}_best_epochs"] = ",".join(str(e) for e in s["best_epochs"])
    row.update(diff_pp=judge["diff_pp"], larger_seed_std_pp=judge["larger_seed_std_pp"],
               threshold_pp=w5.DIFF_THRESHOLD_PP, improved=judge["improved"])
    if judge["chosen"] is not None:
        chosen = judge["chosen"]
        row.update(final_lr=chosen[0], final_scheduler=chosen[1], final_weight_decay=chosen[2],
                   not_converged=judge["not_converged"], d_needed=judge["d_needed"])
        record = w5.find_run(index, model, chosen, w5.SEARCH_SEED, w5.D_EPOCHS)
        if record is None:
            row["d_status"] = "없음"
        elif not record["complete"]:
            row["d_status"] = "미완료"
        else:
            metrics = w5.run_metrics(record)
            row.update(d_status="완료", d_last5_val_acc_pct=metrics["last5_val_acc_pct"], d_best_epoch=metrics["best_epoch"])
    return row


def print_stage(title, frame):
    """단계 표를 읽기 좋게 출력한다."""
    if frame.empty:
        print(f"\n{title}: 아직 결과 없음")
        return
    show = frame[["model", "lr", "scheduler", "weight_decay", "status", "source", "last5_val_acc_pct",
                  "best_val_acc_pct", "overfit_gap_pp", "diff_vs_top_pp", "same_as_top", "selected"]].copy()
    show["weight_decay"] = show["weight_decay"].map("{:g}".format)
    print(f"\n{title}")
    print(show.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


def plot_stage_a(rows, path):
    """그림 1: A단계 val acc 곡선. 모델마다 칸을 나누고, 색 = lr, 선 모양 = scheduler, 굵은 선 = 선택된 설정."""
    done = [r for r in rows if r["status"] == "완료"]
    if not done:
        print("[정보] A단계 완료 결과가 없어 그림 1을 그리지 않습니다.")
        return
    models = [m for m in w5.MODELS if any(r["model"] == m for r in done)]
    fig, axes = plt.subplots(1, len(models), figsize=(7.5 * len(models), 5.2), layout="constrained", squeeze=False)
    for ax, model in zip(axes[0], models):
        values = []
        for row in [r for r in done if r["model"] == model]:
            h = row["_history"]
            acc = h["val_acc"] * 100
            selected = row.get("selected", False)
            ax.plot(h["epoch"], acc, color=LR_COLORS.get(row["lr"], "#999999"),
                    ls=SCHEDULER_STYLES.get(row["scheduler"], "-"), lw=2.8 if selected else 1.4,
                    label=f"lr {row['lr']:g}, {row['scheduler']}" + (" (selected)" if selected else ""))
            if not row.get("diverged"):
                values.extend(acc[h["epoch"] >= STAGE_A_YLIM_FROM_EPOCH].tolist())
        if values:
            ax.set_ylim(min(values) - 1, max(values) + 1)
        ax.set_title(f"{model} (seed {w5.SEARCH_SEED})")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val acc (%)")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Stage A: val acc per epoch (color = lr, line style = scheduler, thick = selected)\n"
                 f"y-axis range is set from epoch {STAGE_A_YLIM_FROM_EPOCH} on; diverged runs may be cut off",
                 fontsize=10)
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"그림 저장: {path}")


def plot_stage_b(rows, path):
    """그림 2: B단계 wd별 last5 막대. 1위에서 0.45%p 이내 구간을 색 띠로 표시하고 선택된 설정은 진하게."""
    done = [r for r in rows if r["status"] == "완료"]
    if not done:
        print("[정보] B단계 완료 결과가 없어 그림 2를 그리지 않습니다.")
        return
    models = [m for m in w5.MODELS if any(r["model"] == m for r in rows)]
    fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 5), layout="constrained", squeeze=False)
    for ax, model in zip(axes[0], models):
        model_rows = [r for r in rows if r["model"] == model]
        labels = [f"{r['weight_decay']:g}" + ("\n(from A)" if r["source"] == "A 결과 재사용" else "") for r in model_rows]
        heights = [r.get("last5_val_acc_pct", np.nan) if r["status"] == "완료" else np.nan for r in model_rows]
        colors = [MODEL_COLORS.get(model, "#0072B2") if r.get("selected") else "#BBBBBB" for r in model_rows]
        x = np.arange(len(model_rows))
        ax.bar(x, np.nan_to_num(heights), color=colors, width=0.6)
        valid = [h for h in heights if not pd.isna(h)]
        if not valid:  # 이 모델은 B 결과가 아직 하나도 없음
            ax.set_title(f"{model}: no stage B results yet")
            ax.axis("off")
            continue
        top = max(valid)
        ax.axhspan(top - w5.DIFF_THRESHOLD_PP, top, color="#009E73", alpha=0.15,
                   label=f"within {w5.DIFF_THRESHOLD_PP}%p of best")
        for i, h in enumerate(heights):
            ax.text(i, (h if not pd.isna(h) else min(valid)) + 0.05, "n/a" if pd.isna(h) else f"{h:.2f}",
                    ha="center", va="bottom", fontsize=9)
        base = model_rows[0]
        ax.set_xticks(x, labels)
        ax.set_ylim(min(valid) - 1.5, top + 1)
        ax.set_xlabel("weight decay")
        ax.set_ylabel("last5 val acc (%)")
        ax.set_title(f"{model}: lr {base['lr']:g} + {base['scheduler']} (seed {w5.SEARCH_SEED})")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Stage B: last5 val acc by weight decay (dark bar = selected, y-axis does not start at 0)", fontsize=10)
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"그림 저장: {path}")


def plot_final(judges, path):
    """그림 3: 모델별 4주차 기본 설정 vs 튜닝한 설정 (시드 3개 평균 ± 표준편차, 점 = 시드)."""
    ready = [j for j in judges if j is not None and j["default"]["n_seeds"] > 0 and j["tuned"]["n_seeds"] > 0]
    if not ready:
        print("[정보] C단계 결과가 없어 그림 3을 그리지 않습니다.")
        return
    # 모델이 1개뿐이어도 제목이 잘리지 않도록 최소 폭을 둔다
    fig, axes = plt.subplots(1, len(ready), figsize=(max(7.0, 5.5 * len(ready)), 5.3), layout="constrained", squeeze=False)
    for ax, judge in zip(axes[0], ready):
        same_cfg = w5.cfg_equal(judge["tuned_cfg"], w5.DEFAULT_CFG)
        sides = [("default", "week4 default")] + ([] if same_cfg else [("tuned", "tuned")])
        all_values = []
        for i, (side, label) in enumerate(sides):
            s = judge[side]
            values = [v["last5_val_acc_pct"] for v in s["per_seed"].values()]
            all_values += values
            ax.bar(i, s["last5_val_acc_pct_mean"], yerr=0 if pd.isna(s["last5_val_acc_pct_std"]) else s["last5_val_acc_pct_std"],
                   capsize=6, width=0.55, color=MODEL_COLORS.get(judge["model"], "#0072B2") if side == "tuned" else "#999999")
            ax.scatter(np.full(len(values), i), values, color="white", edgecolor="black", s=28, zorder=3)
            ax.text(i, max(values) + 0.15, f"{s['last5_val_acc_pct_mean']:.2f}\n(n={s['n_seeds']})", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(sides)), [label for _, label in sides])
        ax.set_ylim(min(all_values) - 1.5, max(all_values) + 1.2)
        if same_cfg:
            verdict = "stage B winner = default setting"
        elif judge["improved"] is None:
            verdict = "stage C not finished"
        else:
            verdict = (f"diff {judge['diff_pp']:+.2f}%p -> "
                       + ("improved" if judge["improved"] else "not improved, keep default"))
        tuned_label = w5.cfg_label(judge["tuned_cfg"])
        ax.set_title(f"{judge['model']}\ntuned: {tuned_label}\n{verdict}", fontsize=9)
        ax.set_ylabel("last5 val acc (%)")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Stage C: tuned vs week 4 default (bar = mean ± std over seeds, dots = each seed)\n"
                 f"improved = diff > {w5.DIFF_THRESHOLD_PP}%p and > the larger seed std; y-axis does not start at 0",
                 fontsize=10)
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"그림 저장: {path}")


def run_week5(paths, results_dir=RESULTS_DIR):
    """
    5주차 취합. 단계가 덜 끝났어도 끝난 부분까지 표를 채우고, 다음 단계는 '앞 단계 미완료'로 남긴다.
    승자는 week5_selection.resolve_winners로 고른다 (run_experiments.py와 같은 규칙, 직접 지정 기록도 반영).
    """
    index = w5.load_index(results_dir)
    saved = w5.load_selection(results_dir)
    print(f"[5주차] {', '.join(w5.MODELS)} 실험 {len(index)}개 읽음 "
          f"(완료 {sum(r['complete'] for r in index)}개, 4주차 기본 설정 결과 포함)")
    unfinished = [r["name"] for r in index if not r["complete"]]
    if unfinished:
        print("[경고] 끝나지 않은 실험 (표에서 '미완료'): " + ", ".join(unfinished))

    rows_a, rows_b, finals, judges = [], [], [], []
    for model in w5.MODELS:
        winners = w5.resolve_winners(index, model, saved)
        mark_selected(winners["A_rows"], winners["A"], winners["A_source"])
        mark_selected(winners["B_rows"], winners["B"], winners["B_source"])
        rows_a += winners["A_rows"]
        rows_b += winners["B_rows"] or []
        warn_week5_devices(model, "A", winners["A_rows"])
        warn_week5_devices(model, "B", winners["B_rows"])
        warn_saved_auto(saved, model, winners)
        judge = w5.judge_final(index, model, winners["B"]) if winners["B"] is not None else None
        judges.append(judge)
        finals.append(final_row(model, winners, judge, index))

    # 단계가 덜 끝났어도 항상 저장한다: 예전 실행의 파일이 남아 최신 결과처럼 보이는 것을 막기 위함
    frame_a, frame_b = stage_frame(rows_a), stage_frame(rows_b)
    save_table(frame_a, paths["stage_a"])
    save_table(frame_b, paths["stage_b"])
    final = pd.DataFrame(finals)
    save_table(final, paths["final"])
    print_stage(f"[A] lr × scheduler (seed {w5.SEARCH_SEED}, 차이 기준 {w5.DIFF_THRESHOLD_PP}%p)", frame_a)
    print_stage(f"[B] weight decay (A 승자 위, seed {w5.SEARCH_SEED})", frame_b)
    print("\n[C] 모델별 최종 설정")
    for row in finals:
        print(f"  - {row['model']}: A 승자 {row['A_winner'] or '-'} ({row['A_source'] or '-'}) | "
              f"B 승자 {row['B_winner'] or '-'} ({row['B_source'] or '-'})\n      → {row['judgment']}")
        if "final_lr" in row:
            print(f"      최종 설정: {w5.cfg_label((row['final_lr'], row['final_scheduler'], row['final_weight_decay']))} | "
                  f"아직 수렴 안 됨: {row['not_converged']} | D 필요: {row['d_needed']} (D 결과: {row.get('d_status', '-')})")

    paths["fig_dir"].mkdir(parents=True, exist_ok=True)
    plot_stage_a(rows_a, paths["fig_dir"] / "1_stageA_val_acc_curves.png")
    plot_stage_b(rows_b, paths["fig_dir"] / "2_stageB_wd_last5.png")
    plot_final(judges, paths["fig_dir"] / "3_final_vs_default.png")


# =====================================================================
# main
# =====================================================================
def main():
    args = parse_args()
    if args.verify_week3:  # 예전 옵션 이름도 그대로 쓸 수 있게 둔다
        args.week, args.verify = 3, True
    if args.verify:
        if args.week not in VERIFY_TABLES:
            print("[오류] --verify는 --week 3, 4에서만 씁니다.")
            sys.exit(1)
        sys.exit(verify_week(args.week))

    week, cfg, paths = args.week, WEEK_CONFIGS[args.week], week_paths(args.week)
    if week == 3 and paths["summary"].exists() and not args.overwrite_week3:
        # 확정된 3주차 출력을 실수로 덮어쓰지 않도록 막는다 (파일 맨 위 '3주차 출력 보호' 참고)
        print(f"[중단] 3주차 출력이 이미 있습니다: {paths['summary']}\n"
              "  4주차를 취합하려면:        python compare_results.py --week 4\n"
              "  3주차 값만 확인하려면:     python compare_results.py --verify-week3\n"
              "  정말 3주차를 다시 만들려면: python compare_results.py --overwrite-week3")
        sys.exit(1)
    if week == 5:
        run_week5(paths)
        return
    top_k = args.top_k if args.top_k is not None else cfg["default_top_k"]
    try:
        run_week(week, paths, top_k, args.no_eval, args.force_eval)
    except ResultsError as error:
        print(f"[오류] {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
