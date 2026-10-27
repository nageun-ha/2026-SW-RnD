"""
3주차 구조 비교 실험(results/) 결과를 취합해 표와 그림으로 정리한다.

실행 예:
    python compare_results.py                 # 표 + 그림 4종 + (상위 모델) 클래스별 정확도
    python compare_results.py --top-k 2       # 클래스별 정확도 비교 대상을 상위 2개 모델로
    python compare_results.py --no-eval       # evaluate.py를 호출하지 않고 이미 있는 CSV만 사용
    python compare_results.py --force-eval    # 클래스별 정확도 CSV가 있어도 evaluate.py를 다시 실행

결과:
    results/week3_summary.csv            모델별 요약표 (시드 평균 ± 표준편차, best val acc 내림차순)
    results/week3_per_class_val_acc.csv  상위 모델 + Baseline의 val 클래스별 정확도 (시드 평균 ± 표준편차)
    figures/week3/                       그림 4종 (아래 4개 파일)
        1_val_acc_curves.png, 2_loss_curves.png, 3_best_val_acc_bar.png, 4_params_vs_acc.png

※ 테스트셋은 사용하지 않는다.
   - history.csv(val 기준)와 val_per_class_acc.csv만 읽는다. test_* 파일은 읽지 않는다.
   - evaluate.py는 항상 --test 없이(= val 평가로) 호출한다.

표의 정의 (시드 표준편차는 표본 표준편차, ddof=1, 시드 3개 기준이라 값 자체가 불안정함에 주의)
    best_val_acc_pct      : 20 epoch 중 최고 val acc (%)  ← 최댓값이라 약간 낙관적인 지표
    last5_val_acc_pct     : 마지막 5 epoch의 val acc 평균 (%)  ← best보다 안정적인 지표
    best_epoch            : best val acc가 처음 나온 epoch
    overfit_gap_pp        : 마지막 epoch의 (train acc − val acc), %p
                            주의: train acc는 증강이 켜진 상태 + 학습 모드(Dropout/BN)에서 잰 값이라
                            "과적합 정도"의 근사치일 뿐이며, 음수가 나올 수 있다(특히 dropout).
    epoch_time_sec        : epoch당 평균 소요 시간(초, train+val 포함)
    diff_vs_baseline_pp   : (모델 평균 best val acc) − (Baseline 평균 best val acc), %p
                            표준편차는 두 모델 시드 표준편차를 독립으로 가정해 sqrt(s1² + s2²)로 합성
    diff_vs_seed_std      : |차이|가 두 모델 시드 표준편차 중 큰 값보다 크면 "larger", 아니면 "smaller"
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 창 없이 파일로만 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# 설정 (config.py는 torch를 import하고 장치 검사까지 하므로, 이 스크립트는 일부러 import하지 않는다)
# ---------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
FIG_DIR = PROJECT_DIR / "figures" / "week3"
SUMMARY_CSV = RESULTS_DIR / "week3_summary.csv"
PER_CLASS_CSV = RESULTS_DIR / "week3_per_class_val_acc.csv"

BASELINE = "baseline"
FIG_DPI = 150                      # config.FIG_DPI와 동일
DEFAULT_TOP_K = 3                  # 클래스별 정확도를 비교할 상위 모델 수 (Baseline은 항상 추가)
LAST_N_EPOCHS = 5                  # "마지막 N epoch 평균 val acc"의 N
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
}
FALLBACK_COLORS = ["#56B4E9", "#F0E442", "#999999", "#882255"]


def color_of(model, _fallback={}):
    """모델 이름 → 색. 목록에 없는 새 모델은 예비 색을 순서대로 배정한다."""
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]
    if model not in _fallback:
        _fallback[model] = FALLBACK_COLORS[len(_fallback) % len(FALLBACK_COLORS)]
    return _fallback[model]


def parse_args():
    """커맨드라인 인자를 읽는다."""
    parser = argparse.ArgumentParser(description="3주차 구조 비교 결과 취합")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help="클래스별 정확도를 비교할 상위 모델 수 (Baseline 제외, 기본 3)")
    parser.add_argument("--no-eval", action="store_true",
                        help="evaluate.py를 호출하지 않고 이미 있는 val_per_class_acc.csv만 사용")
    parser.add_argument("--force-eval", action="store_true",
                        help="val_per_class_acc.csv가 있어도 evaluate.py를 다시 실행")
    return parser.parse_args()


# =====================================================================
# 1. 실험 결과 읽기 + 일관성 점검
# =====================================================================
def load_runs():
    """results/ 아래의 실험 폴더(args.json + history.csv가 있는 것)를 모두 읽는다."""
    runs = []
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        args_path, history_path = run_dir / "args.json", run_dir / "history.csv"
        if not (args_path.exists() and history_path.exists()):
            print(f"[경고] args.json 또는 history.csv가 없어 건너뜁니다: {run_dir.name}")
            continue
        args = json.loads(args_path.read_text(encoding="utf-8"))
        history = pd.read_csv(history_path)
        runs.append({"name": run_dir.name, "model": args["model"], "seed": args["seed"],
                     "args": args, "history": history})
    if not runs:
        raise SystemExit(f"읽을 수 있는 실험 결과가 없습니다: {RESULTS_DIR}")
    return runs


def check_consistency(runs):
    """
    비교가 공정한지 점검하고, 문제가 있으면 경고를 출력한다. (중단하지는 않음)
      1) device(cuda/mps/cpu)가 서로 다른 실험이 섞여 있는지  ← 요청 사항
      2) 학습 설정(epochs, lr 등)이 서로 다른 실험이 섞여 있는지
      3) epoch 수가 설정보다 적은(도중에 끊긴) 실험이 있는지
      4) 시드가 2개 미만인 모델이 있는지 (표준편차를 못 구함)
      5) 같은 모델 안에서 epoch 시간이 유독 긴 실험이 있는지 (시간 비교가 왜곡될 수 있음)
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

    # 3) 끊긴 실험
    for run in runs:
        expected = run["args"].get("epochs")
        if expected is not None and len(run["history"]) != expected:
            n_warnings += 1
            print(f"[경고] {run['name']}: history가 {len(run['history'])} epoch뿐입니다 (설정 {expected}).")

    # 4) 시드 수
    seed_counts = pd.Series([r["model"] for r in runs]).value_counts()
    for model, count in seed_counts.items():
        if count < 2:
            n_warnings += 1
            print(f"[경고] {model}: 시드가 {count}개라 표준편차를 계산할 수 없습니다.")

    # 5) epoch 시간 이상치
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


def plot_val_acc_curves(runs, order, path):
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
    axes[1].set_title("zoom: epochs 8~20, near the top (lower models are cut off)")
    axes[1].set_xlim(8, None)
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
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
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
    ax.set_xticks(x, summary["model"])
    ax.set_ylabel(f"best val acc (%)   [y-axis starts at {bottom}]")
    ax.set_title("best val acc by model (bar = mean, error bar = std over seeds, dots = each seed)")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)


def plot_params_vs_acc(summary, path):
    """그림 4: 파라미터 수 vs best val acc 산점도 (오차막대 = 시드 표준편차)."""
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
    상위 top_k개 모델(+Baseline)의 모든 시드에 대해 val 클래스별 정확도를 모은다.
    CSV가 없거나, 체크포인트(best.pt)보다 오래됐거나, --force-eval이면 evaluate.py를 호출한다.
    반환: (실험 × 클래스) 정확도 표 (%), 비교 대상 모델 순서
    """
    ranked = [m for m in summary["model"] if m != BASELINE]
    targets = ([BASELINE] if BASELINE in set(summary["model"]) else []) + ranked[:top_k]
    print(f"\n[클래스별 정확도 비교 대상] {', '.join(targets)}  (best val acc 상위 {top_k}개 + {BASELINE})")

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
# main
# =====================================================================
def main():
    args = parse_args()
    runs = load_runs()
    print(f"실험 {len(runs)}개 읽음: "
          + ", ".join(f"{m}×{n}" for m, n in pd.Series([r['model'] for r in runs]).value_counts().sort_index().items()))
    check_consistency(runs)

    per_run = per_run_metrics(runs)
    summary = summarize(per_run)
    summary.round(4).to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")  # utf-8-sig: 엑셀에서 안 깨지게
    print_summary(summary)
    print(f"\n표 저장: {SUMMARY_CSV}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    order = list(summary["model"])
    plot_val_acc_curves(runs, order, FIG_DIR / "1_val_acc_curves.png")
    plot_loss_curves(runs, order, FIG_DIR / "2_loss_curves.png")
    plot_best_acc_bar(summary, per_run, FIG_DIR / "3_best_val_acc_bar.png")
    plot_params_vs_acc(summary, FIG_DIR / "4_params_vs_acc.png")
    print(f"그림 저장: {FIG_DIR}")

    per_class, targets = collect_per_class(runs, summary, args.top_k, args.no_eval, args.force_eval)
    if per_class is None:
        print("[경고] 클래스별 정확도 결과가 하나도 없어 표를 만들지 못했습니다.")
        return
    table = build_per_class_table(per_class, targets)
    table.round(4).to_csv(PER_CLASS_CSV, encoding="utf-8-sig")
    print(f"\n표 저장: {PER_CLASS_CSV}")


if __name__ == "__main__":
    main()
