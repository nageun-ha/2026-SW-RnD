"""
6주차 최종 모델 선택 규칙, 테스트 평가 대상, Grad-CAM·오분류 분석용 클래스 쌍 선택 규칙.

README "6주차 최종 선정 기준 (테스트 전 작성)"을 코드로 옮긴 것이다.
compare_results.py --week 6이 이 파일로 최종 모델과 테스트 평가 대상 9개를 정해 표로 남긴다.
  → 테스트셋을 쓰기 전에 무엇을 고르고 무엇을 평가할지 파일로 고정해 두기 위함이다.

테스트셋을 읽지 않는다.
  - 최종 모델 선택은 5주차 val 결과(results/week5_final.csv)와 각 실험의 history.csv(val)만 쓴다.
  - 클래스 쌍 선택(select_pairs)은 혼동행렬을 인자로 받기만 하고 파일을 읽지 않는다.
    테스트 혼동행렬은 README 기준을 커밋한 뒤 6주차 테스트 평가 단계에서 넘긴다.

torch와 config.py를 import하지 않는다 (week5_selection.py와 같은 이유: 장치 검사 없이 가볍게 취합하기 위함).
판정에 쓰는 값(0.45%p, 시드, epoch, 기본 설정)은 week5_selection.py의 값을 그대로 가져온다.
  → 5주차와 6주차가 서로 다른 기준값을 쓰는 일을 막는다.
"""

import json
from pathlib import Path

import pandas as pd

import week5_selection as w5

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
WEEK5_FINAL_FILE_NAME = "week5_final.csv"     # 5주차 모델별 최종 설정 (C 판정 결과)
PARAMS_FILE_NAME = "model_summary.csv"        # 모델별 파라미터 수 (model_summary.py 출력)

# ---------------------------------------------------------------------
# 최종 모델 선택 (README "최종 모델 선택 규칙")
# ---------------------------------------------------------------------
CANDIDATES = w5.MODELS               # 5주차 후보 2개: bn_deep_wide, bn_deep
SEEDS = w5.ALL_SEEDS                 # 42, 1, 2
EPOCHS = w5.EPOCHS                   # 40 epoch (시드 3개가 있는 것은 40 epoch뿐)
REFERENCE_EPOCHS = w5.D_EPOCHS       # 60 epoch: 참고로만 적고 선택에는 쓰지 않는다
DIFF_THRESHOLD_PP = w5.DIFF_THRESHOLD_PP  # 0.45%p (5주차에 실험 전에 고정한 값)
WEEK5_CSV_TOL = 1e-4                 # week5_final.csv는 소수 4자리로 반올림해 저장되므로 이만큼의 차이는 같다고 본다

# ---------------------------------------------------------------------
# 테스트 평가 대상 (README "테스트 평가 대상", 이 9개 외에는 테스트하지 않음)
# ---------------------------------------------------------------------
BASELINE_MODEL = "baseline"
BASELINE_CFG = w5.DEFAULT_CFG        # 4주차 baseline ep40: lr 0.01 + none + wd 5e-4
ROLE_FINAL = "최종 모델"
ROLE_BASELINE = "비교 기준 (baseline)"
ROLE_OTHER = "다른 후보 (참고용, 선택에 안 씀)"
N_TEST_TARGETS = 9                   # 역할 3개 × 시드 3개

# ---------------------------------------------------------------------
# Grad-CAM·오분류 분석용 클래스 쌍 (README "클래스 쌍 선택 규칙")
# ---------------------------------------------------------------------
# torchvision CIFAR10.classes와 같은 순서 (data.get_class_names()). 이 파일은 torchvision을 import하지 않으므로
# 혼동행렬을 넘길 때 class_names를 함께 넘기면 순서가 같은지 확인한다.
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
TOP_PAIRS = 3                        # 양방향 오분류 수 상위 몇 쌍을 고를지
ALWAYS_PAIRS = [("cat", "dog"), ("automobile", "truck")]  # 순위와 상관없이 항상 포함 (1~5주차에서 계속 본 쌍)


class SelectionError(Exception):
    """선택에 필요한 결과가 없거나 서로 맞지 않을 때 (조용히 다른 값으로 고르지 않고 멈추기 위함)."""


# =====================================================================
# 결과 읽기
# =====================================================================
def load_week5_final(results_dir=RESULTS_DIR):
    """
    5주차 모델별 최종 설정 표를 읽는다. 후보 모델마다 최종 설정과 시드 3개 지표가 있어야 한다.
    반환: 모델 이름을 index로 하는 DataFrame
    """
    path = Path(results_dir) / WEEK5_FINAL_FILE_NAME
    if not path.exists():
        raise SelectionError(f"{path.name}이 없습니다. 먼저 python compare_results.py --week 5를 실행하세요.")
    table = pd.read_csv(path, encoding="utf-8-sig").set_index("model")
    missing = [m for m in CANDIDATES if m not in table.index]
    if missing:
        raise SelectionError(f"{path.name}에 후보 모델이 없습니다: {missing}")
    for model in CANDIDATES:
        row = table.loc[model]
        if pd.isna(row.get("final_lr")) or int(row["tuned_n_seeds"]) != len(SEEDS):
            raise SelectionError(f"{model}: 5주차 최종 설정 또는 시드 {len(SEEDS)}개 결과가 없습니다 ({row['judgment']})")
    return table


def load_params(results_dir=RESULTS_DIR):
    """모델별 파라미터 수 (results/model_summary.csv)."""
    path = Path(results_dir) / PARAMS_FILE_NAME
    if not path.exists():
        raise SelectionError(f"{path.name}이 없습니다. 먼저 python model_summary.py를 실행하세요.")
    table = pd.read_csv(path, encoding="utf-8-sig")
    return dict(zip(table["model"], table["params"].astype(int)))


def load_index(results_dir=RESULTS_DIR):
    """
    후보 모델 + baseline의 실험 기록을 읽는다.
    week5_selection.load_index와 같은 방식(args.json 값으로 설정을 찾음, FIXED_SETTINGS가 다른 실험은 제외)이고,
    읽는 모델에 baseline만 더했다. 각 기록에 best.pt가 있는지도 함께 적는다.
    """
    models = CANDIDATES + [BASELINE_MODEL]
    records = []
    for run_dir in sorted(Path(results_dir).iterdir()):
        args_path, history_path = run_dir / "args.json", run_dir / "history.csv"
        if not (run_dir.is_dir() and args_path.exists() and history_path.exists()):
            continue
        args = json.loads(args_path.read_text(encoding="utf-8"))
        if args.get("model") not in models:
            continue
        if any(args.get(key) != value for key, value in w5.FIXED_SETTINGS.items()):
            continue
        history = pd.read_csv(history_path)
        epochs = int(args["epochs"])
        records.append({
            "name": run_dir.name, "model": args["model"], "seed": int(args["seed"]), "epochs": epochs,
            "cfg": (float(args["lr"]), args.get("scheduler", w5.DEFAULT_SCHEDULER), float(args["weight_decay"])),
            "complete": list(history["epoch"]) == list(range(1, epochs + 1)),
            "has_checkpoint": (run_dir / "best.pt").exists(),
            "device": args.get("device"), "torch_version": args.get("torch_version"),
            "history": history,
        })
    return records


def final_cfg(week5_row):
    """week5_final.csv 한 행 → 그 모델의 최종 설정 (lr, scheduler, wd)."""
    return (float(week5_row["final_lr"]), week5_row["final_scheduler"], float(week5_row["final_weight_decay"]))


# =====================================================================
# 최종 모델 선택
# =====================================================================
def candidate_rows(index, week5_final, params):
    """
    후보 모델마다 선택에 쓰는 값을 모은다.
      - 주 지표(last5) 평균·표준편차: week5_final.csv 값 (README에 적은 숫자의 출처)
      - 같은 값을 history.csv에서 다시 계산해 week5_final.csv와 같은지 확인한다 (recomputed_*, matches_week5_final)
      - 60 epoch(seed 42)는 참고용 열(ref_*)로만 남긴다
    """
    rows = []
    for model in CANDIDATES:
        w5_row = week5_final.loc[model]
        cfg = final_cfg(w5_row)
        summary = w5.seed_summary(index, model, cfg, SEEDS, EPOCHS)
        if summary["n_seeds"] != len(SEEDS):
            raise SelectionError(f"{model} {w5.cfg_label(cfg)}: 40 epoch 시드 {summary['missing_seeds']}의 결과가 없습니다.")
        epoch_times = [summary["per_seed"][s]["epoch_time_sec"] for s in SEEDS]
        recomputed_mean, recomputed_std = summary["last5_val_acc_pct_mean"], summary["last5_val_acc_pct_std"]
        mean, std = float(w5_row["tuned_last5_val_acc_pct_mean"]), float(w5_row["tuned_last5_val_acc_pct_std"])

        # 참고: 같은 설정의 60 epoch (seed 42만 있음). 40 epoch seed 42와 비교한다
        ref60 = [w5.find_run(index, model, cfg, s, REFERENCE_EPOCHS) for s in SEEDS]
        ref60_done = {s: r for s, r in zip(SEEDS, ref60) if r is not None and r["complete"]}
        ref_seed = w5.SEARCH_SEED
        ref40_last5 = summary["per_seed"][ref_seed]["last5_val_acc_pct"]
        ref60_last5 = w5.run_metrics(ref60_done[ref_seed])["last5_val_acc_pct"] if ref_seed in ref60_done else float("nan")

        rows.append({
            "model": model, "params": params[model],
            "final_lr": cfg[0], "final_scheduler": cfg[1], "final_weight_decay": cfg[2], "n_seeds": summary["n_seeds"],
            "last5_val_acc_pct_mean": mean, "last5_val_acc_pct_std": std,
            "recomputed_last5_mean": recomputed_mean, "recomputed_last5_std": recomputed_std,
            "matches_week5_final": (abs(recomputed_mean - mean) <= WEEK5_CSV_TOL
                                    and abs(recomputed_std - std) <= WEEK5_CSV_TOL),
            "best_val_acc_pct_mean": summary["best_val_acc_pct_mean"],
            "best_val_acc_pct_std": summary["best_val_acc_pct_std"],
            "epoch_time_sec_mean": sum(epoch_times) / len(epoch_times),
            "ref_ep60_n_seeds": len(ref60_done),
            "ref_ep40_seed42_last5": ref40_last5, "ref_ep60_seed42_last5": ref60_last5,
            "ref_ep60_gain_pp": ref60_last5 - ref40_last5,
        })
    return rows


def select_final_model(rows):
    """
    최종 모델 선택 (README 규칙, 4주차·5주차 C와 같은 차이 판정).
      1) 후보 2개를 last5 평균이 높은 순으로 놓는다.
      2) 평균 차이가 (두 모델의 시드 표준편차 중 큰 값)보다 크고 0.45%p보다도 크면 "차이 있음" → 평균이 높은 쪽
      3) 아니면 "차이 없음" → 파라미터 수가 적은 쪽 (같으면 epoch 시간이 짧은 쪽, 4주차 규칙과 같음)
    rows의 각 행에 판정 열을 채우고, 판정 요약 dict를 돌려준다.
    """
    if len(rows) != 2:
        raise SelectionError(f"후보는 2개여야 합니다: {[r['model'] for r in rows]}")
    top, second = sorted(rows, key=lambda r: r["last5_val_acc_pct_mean"], reverse=True)
    diff = top["last5_val_acc_pct_mean"] - second["last5_val_acc_pct_mean"]
    larger_std = max(top["last5_val_acc_pct_std"], second["last5_val_acc_pct_std"])
    differs = diff > DIFF_THRESHOLD_PP and diff > larger_std
    if differs:
        chosen, rule = top, "차이 있음 → last5 평균이 높은 쪽"
    else:
        chosen = min(rows, key=lambda r: (r["params"], r["epoch_time_sec_mean"]))
        rule = "차이 없음 → 파라미터 수가 적은 쪽"
    for rank, row in enumerate((top, second), start=1):
        row.update(rank=rank, diff_vs_top_pp=row["last5_val_acc_pct_mean"] - top["last5_val_acc_pct_mean"],
                   larger_seed_std_pp=larger_std, threshold_pp=DIFF_THRESHOLD_PP, differs=differs,
                   selected=row is chosen, rule=rule)
    other = second if chosen is top else top
    return {"chosen": chosen["model"], "other": other["model"], "diff_pp": diff,
            "larger_seed_std_pp": larger_std, "differs": differs, "rule": rule}


# =====================================================================
# 테스트 평가 대상 9개
# =====================================================================
def test_targets(index, week5_final, selection):
    """
    테스트할 실험 9개 (역할 3개 × 시드 3개). 하나라도 없거나 best.pt가 없으면 멈춘다.
    best_val_acc_pct는 best.pt를 저장한 epoch의 val acc다 (train.py가 val acc 최고일 때 저장).
      → 6주차 "val 대비 test 차이"는 같은 체크포인트의 이 값과 test acc를 비교한다.
    """
    roles = [(ROLE_FINAL, selection["chosen"], final_cfg(week5_final.loc[selection["chosen"]])),
             (ROLE_BASELINE, BASELINE_MODEL, BASELINE_CFG),
             (ROLE_OTHER, selection["other"], final_cfg(week5_final.loc[selection["other"]]))]
    rows, problems = [], []
    for role, model, cfg in roles:
        for seed in SEEDS:
            record = w5.find_run(index, model, cfg, seed, EPOCHS)
            if record is None or not record["complete"] or not record["has_checkpoint"]:
                state = "없음" if record is None else ("미완료" if not record["complete"] else "best.pt 없음")
                problems.append(f"{model} {w5.cfg_label(cfg)} seed {seed}: {state}")
                continue
            metrics = w5.run_metrics(record)
            rows.append({"role": role, "model": model, "lr": cfg[0], "scheduler": cfg[1], "weight_decay": cfg[2],
                         "epochs": EPOCHS, "seed": seed, "run": record["name"],
                         "best_epoch": metrics["best_epoch"], "best_val_acc_pct": metrics["best_val_acc_pct"],
                         "last5_val_acc_pct": metrics["last5_val_acc_pct"], "device": record["device"]})
    if problems:
        raise SelectionError("테스트 평가 대상 중 쓸 수 없는 실험이 있습니다:\n  " + "\n  ".join(problems))
    if len(rows) != N_TEST_TARGETS or len({r["run"] for r in rows}) != N_TEST_TARGETS:
        raise SelectionError(f"테스트 평가 대상은 서로 다른 실험 {N_TEST_TARGETS}개여야 합니다: {[r['run'] for r in rows]}")
    return rows


def build_selection(results_dir=RESULTS_DIR):
    """
    최종 모델 선택 표와 테스트 평가 대상 표를 만든다 (compare_results.py --week 6이 저장).
    반환: (선택 표 DataFrame, 대상 표 DataFrame, 판정 요약 dict)
    """
    week5_final = load_week5_final(results_dir)
    params = load_params(results_dir)
    index = load_index(results_dir)
    rows = candidate_rows(index, week5_final, params)
    selection = select_final_model(rows)
    targets = test_targets(index, week5_final, selection)
    selection_frame = pd.DataFrame(sorted(rows, key=lambda r: r["rank"]))
    return selection_frame, pd.DataFrame(targets), selection


# =====================================================================
# Grad-CAM·오분류 분석용 클래스 쌍
# =====================================================================
def select_pairs(cm, class_names=None, top_k=TOP_PAIRS, always=ALWAYS_PAIRS):
    """
    혼동행렬에서 분석할 클래스 쌍을 고른다 (README "클래스 쌍 선택 규칙").
      cm[i][j] = 정답 i를 j로 예측한 개수 (engine.confusion_matrix와 같은 방향). 시드 3개를 더한 행렬을 넘긴다.
      1) 순서 없는 쌍 (A, B)마다 양방향 오분류 수 = cm[A][B] + cm[B][A]
      2) 많은 순으로 top_k쌍. 동점이면 클래스 번호(CLASS_NAMES 순서)가 앞선 쌍을 먼저 둔다
      3) always의 쌍은 순위와 상관없이 포함한다 (top_k 안에 이미 있으면 중복하지 않음)
    반환: 고른 쌍의 dict 목록 (상위 쌍 → 항상 포함 쌍 순서). 각 dict에 전체 순위(rank)와 포함 이유를 적는다.
    """
    names = CLASS_NAMES if class_names is None else list(class_names)
    if names != CLASS_NAMES:
        raise SelectionError(f"클래스 순서가 다릅니다: {names}")
    matrix = pd.DataFrame(cm).to_numpy()
    n = len(names)
    if matrix.shape != (n, n):
        raise SelectionError(f"혼동행렬 크기가 {matrix.shape}입니다 ({n}×{n}이어야 함)")

    pairs = [{"class_a": names[i], "class_b": names[j], "a_to_b": int(matrix[i, j]), "b_to_a": int(matrix[j, i]),
              "total": int(matrix[i, j] + matrix[j, i]), "_order": (i, j)}
             for i in range(n) for j in range(i + 1, n)]
    pairs.sort(key=lambda p: (-p["total"], p["_order"]))
    for rank, pair in enumerate(pairs, start=1):
        pair["rank"] = rank

    always_keys = {frozenset(p) for p in always}
    unknown = [p for p in always if not set(p) <= set(names)]
    if unknown:
        raise SelectionError(f"항상 포함할 쌍에 없는 클래스가 있습니다: {unknown}")
    chosen = []
    for pair in pairs[:top_k]:
        key = frozenset((pair["class_a"], pair["class_b"]))
        pair["reason"] = f"상위 {top_k}쌍" + (" + 항상 포함" if key in always_keys else "")
        chosen.append(pair)
    chosen_keys = {frozenset((p["class_a"], p["class_b"])) for p in chosen}
    for a, b in always:
        if frozenset((a, b)) in chosen_keys:
            continue
        pair = next(p for p in pairs if frozenset((p["class_a"], p["class_b"])) == frozenset((a, b)))
        pair["reason"] = "항상 포함"
        chosen.append(pair)
        chosen_keys.add(frozenset((a, b)))
    return [{k: v for k, v in p.items() if not k.startswith("_")} for p in chosen]
