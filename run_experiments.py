"""
비교 실험 일괄 실행 스크립트 (3주차, 4주차 구조 비교 / 5주차 하이퍼파라미터 튜닝).

train.py를 subprocess로 호출하므로 학습 로직은 train.py 그대로다.
3·4주차: 하이퍼파라미터는 --model, --seed, --epochs 외에는 넘기지 않는다 → 나머지는 train.py 기본값(config.py).
5주차  : --lr, --scheduler, --weight-decay를 추가로 넘긴다. 단계와 승자 규칙은 week5_selection.py에 있다.

실행 예:
    python run_experiments.py --dry-run                 # 실행할 명령과 예상 시간만 출력
    python run_experiments.py                           # 3주차: 6개 모델 × 시드 3개 = 18개 실험 (20 epoch)
    python run_experiments.py --week 4                  # 4주차: 7개 모델 × 시드 3개 = 21개 실험 (40 epoch)
    python run_experiments.py --week 4 --seeds 1        # 4주차 중 시드 1만 (팀원과 나눠 돌릴 때)
    python run_experiments.py --models bn gap --seeds 42
    python run_experiments.py --epochs 5                # 빠른 시험용 (결과 폴더 이름이 ep5로 따로 생김)

5주차 (README "5주차 선정 기준" 참고, 단계마다 --dry-run으로 먼저 확인)
    python run_experiments.py --week 5 --stage A --dry-run     # lr × scheduler (모델당 5회, 4주차 결과 1개 재사용)
    python run_experiments.py --week 5 --stage B               # wd (A 결과에서 승자를 자동으로 고름)
    python run_experiments.py --week 5 --stage C --seeds 1     # 최종 설정의 시드 1 (팀원과 시드별로 나눠 돌릴 때)
    python run_experiments.py --week 5 --stage D               # (선택) 60 epoch 수렴 확인. 조건이 맞을 때만 계획됨
    python run_experiments.py --week 5 --stage B --models bn_deep --lr 0.05 --scheduler cosine
                                                               # A 승자를 자동으로 못 고를 때 직접 지정
    직접 지정한 승자는 results/week5_selection.json에 기록되어 compare_results.py --week 5도 같은 설정을 쓴다.
    4주차 결과를 다시 쓰는 조합(lr 0.01 + none + wd 5e-4)은 결과가 없어도 학습하지 않고 알려 준다.

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
import week5_selection as week5
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
    # 5주차: 모델·epoch·탐색 범위는 week5_selection.py에 있다 (compare_results.py와 같은 값을 쓰기 위함)
    5: {"models": week5.MODELS, "epochs": week5.EPOCHS},
}
WEEK5_STAGES = ["A", "B", "C", "D"]

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
    parser = argparse.ArgumentParser(description="비교 실험 일괄 실행")
    parser.add_argument("--week", type=int, default=3, choices=sorted(WEEK_PRESETS),
                        help="실험 프리셋 (기본 3)")
    parser.add_argument("--stage", choices=WEEK5_STAGES, default=None,
                        help="5주차 단계: A(lr×scheduler), B(wd), C(시드 추가), D(60 epoch 수렴 확인)")
    parser.add_argument("--models", nargs="+", default=None, choices=list_models())
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="기본: 3·4주차 42 1 2 / 5주차 A·B·D 42, C 1 2")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None,
                        help="5주차 B·C·D: A 승자를 직접 지정할 때의 lr (--scheduler와 함께)")
    parser.add_argument("--scheduler", choices=["none", "cosine", "step"], default=None,
                        help="5주차 B·C·D: A 승자를 직접 지정할 때의 scheduler (--lr와 함께)")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="5주차 C·D: B 승자를 직접 지정할 때의 wd")
    parser.add_argument("--dry-run", action="store_true", help="실행하지 않고 명령과 예상 시간만 출력")
    args = parser.parse_args()

    if args.week == 5:
        # 5주차는 README에 미리 정한 범위대로만 돌린다. 기준 밖의 조합을 실수로 만들지 않게 막는다.
        if args.stage is None:
            parser.error("--week 5에는 --stage A, B, C, D 중 하나가 필요합니다.")
        if args.epochs is not None:
            parser.error("5주차는 epoch이 기준에 고정되어 있습니다 (A·B·C 40, D 60). --epochs를 빼 주세요.")
        if (args.lr is None) != (args.scheduler is None):
            parser.error("A 승자를 직접 지정하려면 --lr와 --scheduler를 함께 주세요.")
        if args.stage == "A" and (args.lr is not None or args.weight_decay is not None):
            parser.error("A는 lr·scheduler를 탐색하는 단계라 --lr, --scheduler, --weight-decay를 받지 않습니다.")
        if args.stage == "B" and args.weight_decay is not None:
            parser.error("B는 wd를 탐색하는 단계라 --weight-decay를 받지 않습니다.")
        args.epochs = week5.D_EPOCHS if args.stage == "D" else week5.EPOCHS
        if args.seeds is None:
            args.seeds = week5.C_SEEDS if args.stage == "C" else [week5.SEARCH_SEED]
    else:
        if args.stage or args.lr is not None or args.scheduler or args.weight_decay is not None:
            parser.error("--stage, --lr, --scheduler, --weight-decay는 --week 5에서만 씁니다.")
        if args.seeds is None:
            args.seeds = DEFAULT_SEEDS

    preset = WEEK_PRESETS[args.week]
    if args.models is None:
        args.models = preset["models"]
    if args.epochs is None:
        args.epochs = preset["epochs"]
    return args


def run_name_for(model, seed, epochs, lr=config.LR, scheduler=config.SCHEDULER,
                 weight_decay=config.WEIGHT_DECAY):
    """
    train.py가 만들 결과 폴더 이름을 그대로 계산한다.
    train.py의 make_run_name을 재사용 → 이름 규칙이 어긋나 '끝난 실험을 못 알아보는' 일이 없음.
    lr·scheduler·wd를 생략하면 config 기본값 → 3·4주차 이름은 예전과 같다.
    """
    return make_run_name(SimpleNamespace(
        model=model, seed=seed, epochs=epochs,
        optimizer=config.OPTIMIZER, lr=lr, scheduler=scheduler, weight_decay=weight_decay,
        momentum=config.MOMENTUM, batch_size=config.BATCH_SIZE, augment=config.USE_AUGMENTATION,
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
            plan.append({"model": model, "seed": seed, "name": name, "epochs": args.epochs,
                         "status": status, "cmd": cmd, "reuse": None})
    return plan


def week5_entry(model, seed, epochs, cfg, stage):
    """
    5주차 실험 하나의 계획. 상태는 3·4주차와 같고(done/partial/todo), 다른 단계의 결과를 다시 쓰는
    실험인데 결과가 없으면 'missing_reuse'로 표시해 학습하지 않는다.
    (예: 다른 PC에서 4주차 결과 없이 A를 돌리면, lr 0.01 + none을 그 PC에서 새로 학습해 버리는 일을 막음)
    """
    lr, scheduler, weight_decay = cfg
    name = run_name_for(model, seed, epochs, lr, scheduler, weight_decay)
    history = read_history(config.RESULTS_DIR / name)
    reuse = week5.reuse_source(stage, cfg, seed, epochs)
    if is_complete(history, epochs):
        status = "done"
    elif reuse is not None:
        status = "missing_reuse"
    elif history is not None:
        status = "partial"
    else:
        status = "todo"
    # 기본값이어도 --lr 등을 모두 적는다: 로그만 봐도 어떤 설정인지 알 수 있게 하기 위함 (폴더 이름은 같음)
    cmd = [sys.executable, "train.py", "--model", model, "--seed", str(seed), "--epochs", str(epochs),
           "--lr", str(lr), "--scheduler", scheduler, "--weight-decay", str(weight_decay)]
    if status == "partial":
        cmd.append("--overwrite")
    return {"model": model, "seed": seed, "name": name, "epochs": epochs, "status": status,
            "cmd": cmd, "reuse": reuse, "cfg": cfg}


def build_week5_plan(args):
    """
    5주차 단계별 실행 계획을 만든다. 모델이 바깥 반복이다
    (한 모델의 A가 먼저 끝나면 그 모델의 B를 바로 시작할 수 있게 하기 위함).
    반환: (계획, 모델별 안내 문구, 기록 파일에 남길 승자)
    """
    index = week5.load_index(config.RESULTS_DIR)
    saved = week5.load_selection(config.RESULTS_DIR)
    override_a = (args.lr, args.scheduler) if args.lr is not None else None
    plan, notes, selections = [], [], {}

    for model in args.models:
        if args.stage == "A":
            configs = week5.stage_configs("A")
            notes.append(f"{model}: lr {week5.STAGE_A_LRS} × scheduler {week5.STAGE_A_SCHEDULERS}, wd 기본값")
        else:
            winners = week5.resolve_winners(index, model, saved, override_a, args.weight_decay)
            if winners["A"] is None:
                notes.append(f"{model}: A 승자를 고를 수 없어 건너뜀 — {winners['problem']}\n"
                             f"      A를 먼저 끝내거나 --models {model} --lr <값> --scheduler <값>으로 직접 지정하세요.")
                continue
            selections[model] = {"A": {**week5.cfg_dict(winners["A"]), "source": winners["A_source"]}}
            a_text = f"A 승자 = {week5.cfg_label(winners['A'])} ({winners['A_source']})"
            if args.stage == "B":
                configs = week5.stage_configs("B", winners["A"])
                notes.append(f"{model}: {a_text} → wd {week5.STAGE_B_WEIGHT_DECAYS}")
            else:
                if winners["B"] is None:
                    notes.append(f"{model}: {a_text}, B 승자를 고를 수 없어 건너뜀 — {winners['problem']}\n"
                                 f"      B를 먼저 끝내거나 --models {model} --weight-decay <값>으로 직접 지정하세요.")
                    continue
                selections[model]["B"] = {**week5.cfg_dict(winners["B"]), "source": winners["B_source"]}
                b_text = f"B 승자 = {week5.cfg_label(winners['B'])} ({winners['B_source']})"
                if args.stage == "C":
                    configs = [winners["B"]]
                    notes.append(f"{model}: {a_text}\n      {b_text} → 시드 {args.seeds}")
                else:  # D: README 조건(최종 설정의 시드 2개 이상에서 best epoch 36~40)이 맞을 때만 계획
                    judge = week5.judge_final(index, model, winners["B"])
                    if judge["chosen"] is None:
                        notes.append(f"{model}: D 판정 불가 — {judge['judgment']}")
                        continue
                    if not judge["d_needed"]:
                        notes.append(f"{model}: D 조건 미충족 (최종 설정이 40 epoch 안에 수렴) → 건너뜀")
                        continue
                    configs = [judge["chosen"]]
                    notes.append(f"{model}: 최종 설정 = {week5.cfg_label(judge['chosen'])} ({judge['judgment']}) "
                                 f"→ {week5.D_EPOCHS} epoch, 시드 {args.seeds}")
        for cfg in configs:
            for seed in args.seeds:
                plan.append(week5_entry(model, seed, args.epochs, cfg, args.stage))
    return plan, notes, selections


def save_week5_selection(selections):
    """
    이번 실행에서 쓴 승자를 results/week5_selection.json에 기록한다 (dry-run에서는 쓰지 않음).
    '직접 지정'은 compare_results.py가 같은 설정을 쓰게 하려는 것이고, '자동'은 나중에 결과가 바뀌었는지
    확인하기 위한 기록이다. 이번에 다루지 않은 모델·단계의 기록은 그대로 둔다.
    """
    if not selections:
        return
    saved = week5.load_selection(config.RESULTS_DIR)
    for model, stages in selections.items():
        saved.setdefault(model, {}).update(stages)
    path = week5.save_selection(saved, config.RESULTS_DIR)
    print(f"승자 기록: {path}")


def main():
    """계획 출력 → (dry-run이 아니면) 순서대로 실행 → 요약 출력."""
    # 출력을 파일로 저장할 때(> log.txt)도 진행 상황과 train.py 로그 순서가 섞이지 않도록 줄마다 즉시 출력
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    selections = {}
    if args.week == 5:
        mismatches = week5.check_defaults_match(config)
        if mismatches:  # 기본값이 어긋나면 "4주차 결과 재사용"이 틀린 폴더를 가리키게 되므로 멈춘다
            sys.exit("week5_selection.py와 config.py의 기본값이 다릅니다:\n  " + "\n  ".join(mismatches))
        plan, notes, selections = build_week5_plan(args)
        # 예상 시간은 4주차(같은 PC·같은 모델)의 실측 epoch 시간으로 계산한다. lr 등은 시간에 거의 영향이 없다.
        time_seeds, time_epochs = DEFAULT_SEEDS, week5.EPOCHS
    else:
        plan, notes = build_plan(args), []
        time_seeds, time_epochs = args.seeds, args.epochs
    epoch_times = learn_epoch_times(args.models, time_seeds, time_epochs)
    week3_times = load_week3_times()
    pending = [p for p in plan if p["status"] in ("todo", "partial")]
    n_done = sum(p["status"] == "done" for p in plan)
    missing_reuse = [p for p in plan if p["status"] == "missing_reuse"]

    if args.week == 5:
        print(f"[5주차 단계 {args.stage}] 실험 {len(plan)}개 ({args.epochs} epoch) | "
              f"완료·재사용 {n_done}개, 실행 예정 {len(pending)}개"
              + (f", 재사용 결과 없음 {len(missing_reuse)}개" if missing_reuse else ""))
        for note in notes:
            print(f"  - {note}")
    else:
        print(f"실험 {len(plan)}개 = 모델 {len(args.models)}개 × 시드 {len(args.seeds)}개 "
              f"({args.epochs} epoch) | 완료 {n_done}개, 실행 예정 {len(pending)}개")
    print(f"파이썬: {sys.executable}\n")

    total_estimate = 0.0
    for i, p in enumerate(plan, 1):
        if p["status"] == "done":
            label = f"건너뜀({p['reuse']})" if p["reuse"] else "건너뜀(완료)"
            print(f"[{i:2d}/{len(plan)}] {label}  {p['name']}")
            continue
        if p["status"] == "missing_reuse":
            print(f"[{i:2d}/{len(plan)}] 학습 안 함   {p['name']}  ※ {p['reuse']} 대상인데 결과가 없음 "
                  "(git pull로 받거나 앞 단계를 먼저 끝내세요)")
            continue
        estimate = estimate_seconds(p["model"], p["epochs"], epoch_times, week3_times)
        total_estimate += estimate
        note = "  ※ 미완료 기록 있음 → 처음부터 다시" if p["status"] == "partial" else ""
        print(f"[{i:2d}/{len(plan)}] 실행 예정    {' '.join(p['cmd'][1:])}"
              f"   (예상 {format_duration(estimate)}){note}")
    print(f"\n예상 총 소요 시간: {format_duration(total_estimate)}")
    print("  (이미 끝난 실험은 history.csv 실측값 기준. 없는 모델은 3주차 시간 비율로 추정하며, "
          "끝난 실험이 하나도 없으면 RTX 5060 기준값을 씀)")

    if args.dry_run or not pending:
        return

    save_week5_selection(selections)  # 실제로 실행할 때만 기록 (dry-run은 파일을 쓰지 않음)
    failures = []
    session_start = time.perf_counter()
    for done_count, p in enumerate(pending, 1):
        remaining = sum(estimate_seconds(q["model"], q["epochs"], epoch_times, week3_times)
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
            epoch_times = learn_epoch_times(args.models, time_seeds, time_epochs)

    print(f"\n{'=' * 70}\n전체 종료 | 총 소요 {format_duration(time.perf_counter() - session_start)} | "
          f"성공 {len(pending) - len(failures)}개 / 실패 {len(failures)}개")
    if failures:
        print("실패 목록:")
        for name, reason in failures:
            print(f"  - {name}: {reason}")
        sys.exit(1)


if __name__ == "__main__":
    main()
