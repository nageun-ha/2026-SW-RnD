"""
구조 비교 실험 일괄 실행 스크립트 (3주차, 4주차).

train.py를 subprocess로 호출하므로 학습 로직은 train.py 그대로다.
하이퍼파라미터는 --model, --seed, --epochs 외에는 넘기지 않는다 → 나머지는 모두 train.py 기본값(config.py).

실행 예:
    python run_experiments.py --dry-run                 # 실행할 명령과 예상 시간만 출력
    python run_experiments.py                           # 3주차: 6개 모델 × 시드 3개 = 18개 실험 (20 epoch)
    python run_experiments.py --week 4                  # 4주차: 7개 모델 × 시드 3개 = 21개 실험 (40 epoch)
    python run_experiments.py --week 4 --seeds 1        # 4주차 중 시드 1만 (팀원과 나눠 돌릴 때)
    python run_experiments.py --models bn gap --seeds 42
    python run_experiments.py --epochs 5                # 빠른 시험용 (결과 폴더 이름이 ep5로 따로 생김)

이어서 실행
    results/<실험이름>/history.csv 가 이미 목표 epoch까지 채워져 있으면 건너뛴다.
    → 중간에 꺼져도 같은 명령을 다시 실행하면 남은 실험만 돈다.
    (실험 '도중'에 끊긴 것은 처음부터 다시 학습한다. train.py에 이어서 학습하는 기능은 없음)
"""

import argparse
import subprocess
import sys
import time
from types import SimpleNamespace

import pandas as pd

import config
from models import list_models
from train import make_run_name

DEFAULT_SEEDS = [42, 1, 2]

# 주차별 실험 조건. config.EPOCHS를 40으로 바꾸지 않는 이유:
# 그 값은 train.py의 기본값이기도 해서, 바꾸면 3주차 명령이 ep40 폴더를 만들어 3주차 재현이 깨진다.
WEEK_PRESETS = {
    3: {"models": ["baseline", "bn", "dropout", "deep", "wide", "gap"],
        "epochs": config.EPOCHS},
    4: {"models": ["baseline", "bn", "bn_deep", "bn_wide", "bn_deep_wide",
                   "bn_deep_wide_dropout", "bn_deep_wide_gap"],
        "epochs": 40},
}

# 새 모델의 epoch 시간 추정용: 어떤 3주차 변경을 합친 모델인지 적어 둔다.
# 여기 없는 모델은 자기 이름 하나로 계산한다 (3주차 모델이면 그대로 실측값을 쓰게 됨).
TIME_PARTS = {
    "baseline": [],
    "bn_deep": ["bn", "deep"],
    "bn_wide": ["bn", "wide"],
    "bn_deep_wide": ["bn", "deep", "wide"],
    "bn_deep_wide_dropout": ["bn", "deep", "wide", "dropout"],
    "bn_deep_wide_gap": ["bn", "deep", "wide", "gap"],
}


def parse_args():
    """
    커맨드라인 인자를 읽는다.
    --models, --epochs를 직접 주면 프리셋 값보다 우선한다. 그래서 --week 4 --seeds 1처럼 일부만 바꿀 수 있다.
    """
    parser = argparse.ArgumentParser(description="구조 비교 실험 일괄 실행")
    parser.add_argument("--week", type=int, default=3, choices=sorted(WEEK_PRESETS),
                        help="실험 프리셋 (기본 3)")
    parser.add_argument("--models", nargs="+", default=None, choices=list_models())
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="실행하지 않고 명령과 예상 시간만 출력")
    args = parser.parse_args()

    preset = WEEK_PRESETS[args.week]
    if args.models is None:
        args.models = preset["models"]
    if args.epochs is None:
        args.epochs = preset["epochs"]
    return args


def run_name_for(model, seed, epochs):
    """
    train.py가 만들 결과 폴더 이름을 그대로 계산한다.
    train.py의 make_run_name을 재사용 → 이름 규칙이 어긋나 '끝난 실험을 못 알아보는' 일이 없음.
    """
    return make_run_name(SimpleNamespace(
        model=model, seed=seed, epochs=epochs,
        optimizer=config.OPTIMIZER, lr=config.LR,
        batch_size=config.BATCH_SIZE, augment=config.USE_AUGMENTATION,
    ))


def read_history(run_dir):
    """history.csv를 읽는다. 없거나 깨져 있으면 None."""
    path = run_dir / "history.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


def is_complete(history, epochs):
    """목표 epoch까지 기록이 다 있으면 완료로 본다."""
    return history is not None and len(history) > 0 and history["epoch"].max() >= epochs


def learn_epoch_times(models, seeds, epochs):
    """
    이미 끝난 실험들의 history.csv에서 모델별 'epoch당 평균 초'를 모은다.
    → 예상 남은 시간을 실제 측정값으로 계산하기 위함.
    """
    times = {}
    for model in models:
        values = []
        for seed in seeds:
            history = read_history(config.RESULTS_DIR / run_name_for(model, seed, epochs))
            if history is not None and "epoch_time_sec" in history:
                values.extend(history["epoch_time_sec"].tolist())
        if values:
            times[model] = sum(values) / len(values)
    return times


def load_week3_times():
    """
    3주차 요약표에서 모델별 epoch 시간 중앙값을 읽는다. (읽기만 하고 파일은 바꾸지 않는다)
    평균이 아니라 중앙값을 쓰는 이유: 3주차 baseline은 장치가 섞여 평균이 10.37초로 왜곡됐다 (중앙값 6.92초).
    """
    path = config.RESULTS_DIR / "week3_summary.csv"
    if not path.exists():
        return {}
    table = pd.read_csv(path, encoding="utf-8-sig")
    return dict(zip(table["model"], table["epoch_time_sec_median"]))


def relative_time(model, week3_times):
    """
    baseline을 1로 봤을 때 모델의 epoch 시간 비율을 추정한다.

    파라미터 수로 추정하지 않는 이유: 3주차에 시간이 파라미터 수에 비례하지 않았다.
    (wide는 파라미터 2.3배인데 시간 1.16배, gap은 파라미터 0.15배인데 시간 거의 같음)
    조합 모델은 각 변경의 비율을 곱한다. 대략적인 값이며 3주차 자료가 없으면 1로 둔다.
    """
    base = week3_times.get("baseline")
    ratio = 1.0
    for part in TIME_PARTS.get(model, [model]):
        if base and part in week3_times:
            ratio *= week3_times[part] / base
    return ratio


def estimate_seconds(model, epochs, epoch_times, week3_times):
    """
    실험 하나의 예상 소요 시간(초).
    1) 그 모델의 실측값이 있으면 그대로 쓴다.
    2) 없으면 'baseline 기준 시간 × 상대 비율'로 추정한다. 기준 시간은 이번 실행 계획에서 이미 끝난
       실험으로 환산하고, 끝난 실험이 없으면 config 기본값을 쓴다.
       → PC 속도가 달라도 실험이 하나 끝나는 순간 그 PC 기준으로 추정이 맞춰진다.
    """
    if model in epoch_times:
        per_epoch = epoch_times[model]
    else:
        if epoch_times:
            scales = [t / relative_time(m, week3_times) for m, t in epoch_times.items()]
            scale = sum(scales) / len(scales)
        else:
            scale = config.DEFAULT_EPOCH_SEC
        per_epoch = scale * relative_time(model, week3_times)
    return per_epoch * epochs + config.RUN_STARTUP_SEC


def format_duration(seconds):
    """초 → 'H:MM:SS' 문자열."""
    seconds = int(round(seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def build_plan(args):
    """
    실행 계획(실험 목록)을 만든다.
    시드를 바깥, 모델을 안쪽 반복으로 → seed 42의 모든 모델이 먼저 끝남
    (중간에 멈춰도 '모든 모델 × 시드 1개' 비교는 먼저 가능)
    """
    plan = []
    for seed in args.seeds:
        for model in args.models:
            name = run_name_for(model, seed, args.epochs)
            run_dir = config.RESULTS_DIR / name
            history = read_history(run_dir)
            if is_complete(history, args.epochs):
                status = "done"
            elif history is not None:
                status = "partial"   # 기록이 일부만 있음 → --overwrite로 처음부터 다시
            else:
                status = "todo"
            cmd = [sys.executable, "train.py", "--model", model,
                   "--seed", str(seed), "--epochs", str(args.epochs)]
            if status == "partial":
                cmd.append("--overwrite")  # train.py는 기존 기록이 있으면 멈추므로 필요
            plan.append({"model": model, "seed": seed, "name": name,
                         "status": status, "cmd": cmd})
    return plan


def main():
    """계획 출력 → (dry-run이 아니면) 순서대로 실행 → 요약 출력."""
    # 출력을 파일로 저장할 때(> log.txt)도 진행 상황과 train.py 로그 순서가 섞이지 않도록 줄마다 즉시 출력
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    plan = build_plan(args)
    epoch_times = learn_epoch_times(args.models, args.seeds, args.epochs)
    week3_times = load_week3_times()
    pending = [p for p in plan if p["status"] != "done"]

    print(f"실험 {len(plan)}개 = 모델 {len(args.models)}개 × 시드 {len(args.seeds)}개 "
          f"({args.epochs} epoch) | 완료 {len(plan) - len(pending)}개, 실행 예정 {len(pending)}개")
    print(f"파이썬: {sys.executable}\n")

    total_estimate = 0.0
    for i, p in enumerate(plan, 1):
        if p["status"] == "done":
            print(f"[{i:2d}/{len(plan)}] 건너뜀(완료)  {p['name']}")
            continue
        estimate = estimate_seconds(p["model"], args.epochs, epoch_times, week3_times)
        total_estimate += estimate
        note = "  ※ 미완료 기록 있음 → 처음부터 다시" if p["status"] == "partial" else ""
        print(f"[{i:2d}/{len(plan)}] 실행 예정    {' '.join(p['cmd'][1:])}"
              f"   (예상 {format_duration(estimate)}){note}")
    print(f"\n예상 총 소요 시간: {format_duration(total_estimate)}")
    print("  (이미 끝난 실험은 history.csv 실측값 기준. 없는 모델은 3주차 시간 비율로 추정하며, "
          "끝난 실험이 하나도 없으면 RTX 5060 기준값을 씀)")

    if args.dry_run or not pending:
        return

    failures = []
    session_start = time.perf_counter()
    for done_count, p in enumerate(pending, 1):
        remaining = sum(estimate_seconds(q["model"], args.epochs, epoch_times, week3_times)
                        for q in pending[done_count - 1:])
        print(f"\n{'=' * 70}\n[{done_count}/{len(pending)}] {p['name']} | "
              f"경과 {format_duration(time.perf_counter() - session_start)} | "
              f"예상 남은 시간 {format_duration(remaining)}\n{'=' * 70}")

        try:
            # 학습 로그가 그대로 화면에 보이도록 출력을 가로채지 않는다
            result = subprocess.run(p["cmd"], cwd=config.PROJECT_DIR)
        except KeyboardInterrupt:
            # Ctrl+C: 지금까지 결과는 남아 있으므로 다시 실행하면 이어서 진행됨
            print("\n중단되었습니다. 같은 명령을 다시 실행하면 남은 실험부터 이어서 진행합니다.")
            failures.append((p["name"], "중단(Ctrl+C)"))
            break

        if result.returncode != 0:
            # 하나가 실패해도 나머지는 계속 진행
            failures.append((p["name"], f"종료 코드 {result.returncode}"))
            print(f"[실패] {p['name']} (종료 코드 {result.returncode}) → 다음 실험으로 진행")
        else:
            # 방금 끝난 실험의 실측 epoch 시간으로 남은 시간 추정을 갱신
            epoch_times = learn_epoch_times(args.models, args.seeds, args.epochs)

    print(f"\n{'=' * 70}\n전체 종료 | 총 소요 {format_duration(time.perf_counter() - session_start)} | "
          f"성공 {len(pending) - len(failures)}개 / 실패 {len(failures)}개")
    if failures:
        print("실패 목록:")
        for name, reason in failures:
            print(f"  - {name}: {reason}")
        sys.exit(1)


if __name__ == "__main__":
    main()
